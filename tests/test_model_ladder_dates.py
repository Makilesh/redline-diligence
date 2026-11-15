"""
The ladder against what the provider has announced: retirements, ids served by
another model, and the gemini-3.8-flash rung.

Facts are from https://aistudio.google.com/docs/models and /docs/deprecations,
read 2026-11-15, and from ListModels on the four live keys the same day.
"""

import asyncio
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

import src.llm.budget_tracker as bt
from src.llm import model_registry as reg
from src.llm.model_registry import (
    AGENT_LADDER,
    MODEL_LIMITS,
    RETIREMENTS,
    SERVED_BY,
    SYNTHESIS_LADDER,
    is_local,
    is_retired,
    same_backend,
)

K0, K1, K2 = "A" * 40, "B" * 40, "C" * 40
UTC = timezone.utc


def _tracker(monkeypatch, keys=(K0, K1)):
    t = bt.BudgetTracker.__new__(bt.BudgetTracker)
    t._api_keys = list(keys)
    t._rate_limiters = {}
    t._is_mock = True
    t._mock_budgets = {}
    monkeypatch.setattr(bt.BudgetTracker, "_instance", t)
    monkeypatch.delenv("SYNTHESIS_MODEL_PIN", raising=False)
    return t


# ─── Retirements ──────────────────────────────────────────────────────────────


class TestRetirements:
    def test_no_ladder_keeps_a_rung_past_its_retirement_date(self):
        """
        Fails on the retirement day until the rung is taken off the ladder.

        The selector already stops offering it that day (see below); this makes
        the stale entry visible instead of leaving a rung that silently never
        serves.
        """
        today = datetime.now(UTC).date()
        for name, ladder in (("synthesis", SYNTHESIS_LADDER), ("agent", AGENT_LADDER)):
            stale = [m for m in ladder if m in RETIREMENTS and today >= RETIREMENTS[m].on]
            assert not stale, (
                f"{name} ladder still lists {stale}, retired on "
                f"{[str(RETIREMENTS[m].on) for m in stale]} — remove the rung (and its "
                f"MODEL_LIMITS entry once no key needs it)"
            )

    def test_gemini_2_5_flash_retires_on_the_announced_date(self):
        r = RETIREMENTS["gemini/gemini-2.5-flash"]
        assert r.on == date(2026, 12, 1)
        assert r.becomes == "gemini/gemini-3.5-flash-lite"

    def test_the_boundary_is_midnight_utc(self):
        model = "gemini/gemini-2.5-flash"
        assert not is_retired(model, datetime(2026, 11, 30, 23, 59, tzinfo=UTC))
        assert is_retired(model, datetime(2026, 12, 1, 0, 0, tzinfo=UTC))
        pacific_evening = datetime(2026, 11, 30, 16, 0, tzinfo=timezone(timedelta(hours=-8)))
        assert is_retired(model, pacific_evening), "already 1 December in UTC"
        assert not is_retired("gemini/gemini-3.6-flash", datetime(2030, 1, 1, tzinfo=UTC))

    def test_every_retirement_names_a_known_model(self):
        for model, r in RETIREMENTS.items():
            assert model in MODEL_LIMITS
            assert r.becomes is None or r.becomes in MODEL_LIMITS
            assert r.source.startswith("https://")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("now, expected", [
        (datetime(2026, 11, 30, 12, 0, tzinfo=UTC), "gemini/gemini-2.5-flash"),
        (datetime(2026, 12, 1, 0, 0, tzinfo=UTC), "gemini/gemini-3.5-flash-lite"),
    ])
    async def test_selector_stops_offering_the_rung_on_the_day(self, monkeypatch, now, expected):
        t = _tracker(monkeypatch)
        monkeypatch.setattr(bt, "is_retired", lambda m: reg.is_retired(m, now))
        above = SYNTHESIS_LADDER[:SYNTHESIS_LADDER.index("gemini/gemini-2.5-flash")]
        for model in above:
            t.report_model_unavailable(model)
            for key_index in range(2):
                t.mark_slot_unavailable(key_index, model)

        choice = await asyncio.create_task(t.get_model_for_synthesis())
        assert choice.model == expected

    @pytest.mark.asyncio
    async def test_a_pinned_run_on_a_retired_model_fails_loudly(self, monkeypatch):
        t = _tracker(monkeypatch)
        monkeypatch.setenv("SYNTHESIS_MODEL_PIN", "gemini/gemini-2.5-flash")
        monkeypatch.setattr(bt, "is_retired", lambda m: m == "gemini/gemini-2.5-flash")
        with pytest.raises(RuntimeError, match="retirement"):
            await t.get_model_for_synthesis()


# ─── Ids served by another model ─────────────────────────────────────────────


class TestServedBy:
    def test_the_announced_routes(self):
        assert SERVED_BY["gemini/gemini-3.7-flash"] == "gemini/gemini-3.8-flash"
        assert SERVED_BY["gemini/gemini-3.5-flash"] == "gemini/gemini-3.6-flash"
        assert set(same_backend("gemini/gemini-3.8-flash")) == {
            "gemini/gemini-3.8-flash", "gemini/gemini-3.7-flash"}
        assert same_backend("gemini/gemini-3-flash-preview") == ("gemini/gemini-3-flash-preview",)
        assert same_backend("gemini/not-registered") == ("gemini/not-registered",)

    def test_targets_are_registered_and_ranked_above_their_aliases(self):
        for alias, target in SERVED_BY.items():
            assert target in MODEL_LIMITS and alias in MODEL_LIMITS
            for ladder in (SYNTHESIS_LADDER, AGENT_LADDER):
                if alias in ladder and target in ladder:
                    assert ladder.index(target) < ladder.index(alias), (
                        f"{alias} is served by {target}; ask for the real model first"
                    )

    @pytest.mark.parametrize("failed, also", [
        ("gemini/gemini-3.7-flash", "gemini/gemini-3.8-flash"),
        ("gemini/gemini-3.8-flash", "gemini/gemini-3.7-flash"),
        ("gemini/gemini-3.6-flash", "gemini/gemini-3.5-flash"),
    ])
    def test_an_outage_cools_every_id_of_the_model(self, monkeypatch, failed, also):
        t = _tracker(monkeypatch)
        t.report_model_unavailable(failed)
        cooled = t._models_in_cooldown()
        assert {failed, also} <= cooled
        assert "gemini/gemini-3-flash-preview" not in cooled

    def test_the_streak_is_shared(self, monkeypatch):
        """3.8 then 3.7 failing is two failures of one model: the second backoff doubles."""
        t = _tracker(monkeypatch)
        t.report_model_unavailable("gemini/gemini-3.8-flash")
        t.report_model_unavailable("gemini/gemini-3.7-flash")
        assert t._model_failures["gemini/gemini-3.8-flash"] == 2
        assert t._model_failures["gemini/gemini-3.7-flash"] == 2
        remaining = t._model_cooldowns["gemini/gemini-3.8-flash"] - bt.time.monotonic()
        assert round(remaining) == round(2 * bt.SERVICE_COOLDOWN_SECONDS)

    def test_one_good_answer_clears_the_whole_family(self, monkeypatch):
        t = _tracker(monkeypatch)
        t.report_model_unavailable("gemini/gemini-3.8-flash")
        t.note_model_healthy("gemini/gemini-3.7-flash")
        assert t._models_in_cooldown() == set()
        assert not t._model_failures


# ─── gemini-3.8-flash ─────────────────────────────────────────────────────────


class TestGemini38FlashRung:
    MODEL = "gemini/gemini-3.8-flash"

    def test_registered_on_the_reasoning_tier(self):
        limits = MODEL_LIMITS[self.MODEL]
        assert limits.reasoning is True
        assert (limits.rpm, limits.tpm, limits.rpd) == (5, 250_000, 20), (
            "assumed equal to the other Flash models until the console says otherwise"
        )

    def test_leads_synthesis_and_stays_off_the_agent_ladder(self):
        """At 20 RPD it would be drained by agent traffic in five queries."""
        assert SYNTHESIS_LADDER[0] == self.MODEL
        assert self.MODEL not in AGENT_LADDER

    def test_every_reasoning_rung_ranks_above_every_lite_rung(self):
        cloud = [m for m in SYNTHESIS_LADDER if not is_local(m)]
        tiers = [MODEL_LIMITS[m].reasoning for m in cloud]
        assert tiers == sorted(tiers, reverse=True), "a Lite rung sits above a reasoning rung"


# ─── Dead keys ────────────────────────────────────────────────────────────────


class TestDeadKeys:
    @staticmethod
    def _transport(behaviour):
        def handler(request: httpx.Request) -> httpx.Response:
            verdict = behaviour[request.headers["x-goog-api-key"]]
            if verdict == "down":
                raise httpx.ConnectError("no route", request=request)
            if verdict == 400:
                return httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT",
                                      "details": [{"reason": "API_KEY_INVALID"}]}})
            return httpx.Response(verdict, json={"models": []})
        return httpx.MockTransport(handler)

    @pytest.mark.asyncio
    async def test_startup_check_retires_only_rejected_keys(self, monkeypatch):
        t = _tracker(monkeypatch, keys=(K0, K1, K2, "D" * 40))
        transport = self._transport({K0: 200, K1: 401, K2: "down", "D" * 40: 400})

        retired = await t.check_keys(transport=transport)

        assert retired == [1, 3]
        assert t._api_keys[0] == K0 and t._api_keys[2] == K2, "no network is not a dead key"
        assert t._api_keys[1] == "" and t._api_keys[3] == ""

    @pytest.mark.asyncio
    async def test_a_rejected_key_is_never_selected_again_for_any_model(self, monkeypatch):
        t = _tracker(monkeypatch)
        t.mark_key_unusable(0)
        for _ in range(3):
            assert (await t.get_model_for_synthesis()).key_index == 1
            assert (await t.get_model_for_agent()).key_index == 1

    def test_the_check_is_opt_in(self, monkeypatch):
        monkeypatch.delenv("GEMINI_KEY_CHECK_AT_STARTUP", raising=False)
        assert not bt._env_flag("GEMINI_KEY_CHECK_AT_STARTUP")
        monkeypatch.setenv("GEMINI_KEY_CHECK_AT_STARTUP", "1")
        assert bt._env_flag("GEMINI_KEY_CHECK_AT_STARTUP")
