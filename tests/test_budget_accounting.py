"""
Quota accounting — the local counter must equal what the provider served.

Every test here is a way the counter used to drift: a refusal that kept its
unit, a retry that never paid for one, a per-minute 429 booked as a spent day,
a window slot held by a request that was never sent.
"""

import asyncio

import pytest

from src.llm.budget_tracker import BudgetTracker
from src.llm.model_registry import AGENT_LADDER, LOCAL_MODEL, rpm_for, usable_rpd

K0, K1 = "A" * 40, "B" * 40
TOP = AGENT_LADDER[0]
SECOND = AGENT_LADDER[1]


def _tracker(keys=(K0, K1)) -> BudgetTracker:
    t = BudgetTracker.__new__(BudgetTracker)
    t._api_keys = list(keys)
    t._rate_limiters = {}
    t._is_mock = True
    t._mock_budgets = {}
    return t


def _used(t: BudgetTracker, key_index: int, model: str) -> int:
    return t._mock_budgets.get(t._slot(key_index, model), {}).get("used_today", 0)


class TestRefund:
    @pytest.mark.asyncio
    async def test_refund_returns_the_unit_selection_took(self):
        t = _tracker()
        choice = await t.get_model_for_agent()
        assert _used(t, choice.key_index, choice.model) == 1

        await t.refund_slot(choice.key_index, choice.model)
        assert _used(t, choice.key_index, choice.model) == 0

    @pytest.mark.asyncio
    async def test_refund_never_goes_below_zero(self):
        t = _tracker()
        await t.refund_slot(0, TOP)
        await t.refund_slot(0, TOP)
        assert _used(t, 0, TOP) == 0

    @pytest.mark.asyncio
    async def test_refund_does_not_reopen_a_slot_the_provider_closed(self):
        """A concurrent request's refund must not undo a per-day 429."""
        t = _tracker()
        await t.get_model_for_agent()
        await t.mark_slot_exhausted(0, TOP)
        assert _used(t, 0, TOP) == usable_rpd(TOP)

        await t.refund_slot(0, TOP)
        assert _used(t, 0, TOP) == usable_rpd(TOP)
        assert (await t.get_model_for_agent()).key_index == 1

    @pytest.mark.asyncio
    async def test_refund_ignores_the_local_model(self):
        t = _tracker()
        await t.refund_slot(-1, LOCAL_MODEL)
        assert t._mock_budgets == {}

    def test_key_index_lookup(self):
        t = _tracker()
        assert t.key_index_of(K1) == 1
        assert t.key_index_of("not-a-configured-key") == -1
        assert t.key_index_of(None) == -1
        t.mark_key_unusable(1)
        assert t.key_index_of(K1) == -1, "a retired key no longer owns a slot"


class TestExtraRequests:
    @pytest.mark.asyncio
    async def test_a_same_slot_retry_costs_a_unit_and_a_window_entry(self):
        t = _tracker()
        choice = await t.get_model_for_agent()
        limiter = t._get_rate_limiter(t._slot(choice.key_index, choice.model))

        assert await t.charge_extra_request(choice.key_index, choice.model)
        assert _used(t, choice.key_index, choice.model) == 2
        assert limiter.current_usage == 2

    @pytest.mark.asyncio
    async def test_no_retry_when_the_daily_allowance_is_gone(self):
        t = _tracker()
        slot = t._slot(0, TOP)
        for _ in range(usable_rpd(TOP)):
            await t._try_consume(slot)

        assert not await t.charge_extra_request(0, TOP)
        assert _used(t, 0, TOP) == usable_rpd(TOP)
        assert t._get_rate_limiter(slot).current_usage == 0, "window entry was handed back"

    @pytest.mark.asyncio
    async def test_no_retry_when_the_minute_window_is_full(self):
        t = _tracker()
        slot = t._slot(0, TOP)
        limiter = t._get_rate_limiter(slot)
        for _ in range(rpm_for(TOP)):
            assert await limiter.try_acquire()

        assert not await t.charge_extra_request(0, TOP)
        assert _used(t, 0, TOP) == 0, "never blocks, never debits"

    @pytest.mark.asyncio
    async def test_local_model_is_free(self):
        t = _tracker()
        assert await t.charge_extra_request(-1, LOCAL_MODEL)
        assert t._mock_budgets == {}

    @pytest.mark.asyncio
    async def test_a_retired_key_cannot_be_charged(self):
        t = _tracker()
        t.mark_key_unusable(0)
        assert not await t.charge_extra_request(0, TOP)


class TestPerMinuteRefusal:
    @pytest.mark.asyncio
    async def test_per_minute_429_rests_the_slot_without_spending_the_day(self):
        t = _tracker()
        await t.get_model_for_agent()                      # 1 unit on (key 0, TOP)
        t.note_quota_refusal(0, TOP, per_minute=True, retry_after=31.0)
        await t.mark_slot_exhausted(0, TOP)

        assert _used(t, 0, TOP) == 1, "a per-minute refusal must not burn 475 units"
        nxt = await t.get_model_for_agent()
        assert (nxt.model, nxt.key_index) == (TOP, 1), "rotate keys, stay on the model"

    @pytest.mark.asyncio
    async def test_the_slot_returns_when_the_delay_has_passed(self):
        t = _tracker(keys=(K0,))
        t.note_quota_refusal(0, TOP, per_minute=True, retry_after=5.0)
        await t.mark_slot_exhausted(0, TOP)
        assert (await t.get_model_for_agent()).model == SECOND

        t._slot_cooldowns[t._slot(0, TOP)] -= 6.0          # jump past the delay
        assert (await t.get_model_for_agent()).model == TOP

    @pytest.mark.asyncio
    async def test_provider_delay_is_bounded(self):
        import src.llm.budget_tracker as bt

        t = _tracker()
        assert t._cool_slot("s", 0.0) == bt.MIN_RATE_LIMIT_COOLDOWN_SECONDS
        assert t._cool_slot("s", 86_400.0) == bt.MAX_RATE_LIMIT_COOLDOWN_SECONDS
        assert t._cool_slot("s", None) == bt.RATE_LIMIT_COOLDOWN_SECONDS

    @pytest.mark.asyncio
    async def test_per_day_429_still_spends_the_slot(self):
        t = _tracker()
        t.note_quota_refusal(0, TOP, per_minute=False)
        await t.mark_slot_exhausted(0, TOP)
        assert _used(t, 0, TOP) == usable_rpd(TOP)

    @pytest.mark.asyncio
    async def test_without_a_note_a_429_is_daily(self):
        """Callers that know nothing about the quota id keep the old behaviour."""
        t = _tracker()
        await t.mark_slot_exhausted(0, TOP)
        assert _used(t, 0, TOP) == usable_rpd(TOP)

    @pytest.mark.asyncio
    async def test_a_note_is_used_once(self):
        t = _tracker()
        t.note_quota_refusal(0, TOP, per_minute=True)
        await t.mark_slot_exhausted(0, TOP)                # consumes the note
        await t.mark_slot_exhausted(0, TOP)                # a later, unexplained 429
        assert _used(t, 0, TOP) == usable_rpd(TOP)

    @pytest.mark.asyncio
    async def test_a_stale_note_is_ignored(self, monkeypatch):
        import src.llm.budget_tracker as bt

        t = _tracker()
        t.note_quota_refusal(0, TOP, per_minute=True)
        per_minute, retry_after, noted_at = t._quota_hints[t._slot(0, TOP)]
        t._quota_hints[t._slot(0, TOP)] = (
            per_minute, retry_after, noted_at - bt.QUOTA_HINT_TTL_SECONDS - 1
        )
        await t.mark_slot_exhausted(0, TOP)
        assert _used(t, 0, TOP) == usable_rpd(TOP)


class TestWindowLeak:
    @pytest.mark.asyncio
    async def test_losing_the_race_for_the_last_unit_returns_the_window_slot(self):
        """
        Two requests see budget on the top slot; one unit is left. The loser is
        routed elsewhere and must not leave an RPM entry behind for a request
        that was never sent to that slot.
        """
        t = _tracker(keys=(K0,))
        slot = t._slot(0, TOP)
        for _ in range(usable_rpd(TOP) - 1):
            await t._try_consume(slot)

        a, b = await asyncio.gather(t.get_model_for_agent(), t.get_model_for_agent())

        assert sorted([a.model, b.model]) == sorted([TOP, SECOND])
        assert _used(t, 0, TOP) == usable_rpd(TOP), "the last unit was sold exactly once"
        assert t._get_rate_limiter(slot).current_usage == 1


class TestRequestScopedExclusion:
    @pytest.mark.asyncio
    async def test_excluded_model_is_skipped_for_this_request_only(self):
        t = _tracker()
        assert (await t.get_model_for_agent(exclude=frozenset({TOP}))).model == SECOND
        assert (await t.get_model_for_agent()).model == TOP, "no effect on other requests"
        assert t._models_in_cooldown() == set(), "exclusion is not a health signal"

    @pytest.mark.asyncio
    async def test_excluding_every_cloud_rung_leaves_the_local_model(self):
        t = _tracker()
        cloud = frozenset(m for m in AGENT_LADDER if m != LOCAL_MODEL)
        assert (await t.get_model_for_agent(exclude=cloud)).model == LOCAL_MODEL
