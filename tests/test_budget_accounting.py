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
        slot, per_minute, retry_after, noted_at = bt._quota_hint.get()
        bt._quota_hint.set(
            (slot, per_minute, retry_after, noted_at - bt.QUOTA_HINT_TTL_SECONDS - 1)
        )
        await t.mark_slot_exhausted(0, TOP)
        assert _used(t, 0, TOP) == usable_rpd(TOP)

    @pytest.mark.asyncio
    async def test_concurrent_refusals_do_not_steal_each_others_note(self):
        """
        Two requests hit the same per-minute 429. With the note stored on the
        shared tracker the first to finish consumed it and the second, finding
        none, booked the slot as spent for the day.
        """
        t = _tracker()

        async def refused_request():
            t.note_quota_refusal(0, TOP, per_minute=True, retry_after=20.0)
            await asyncio.sleep(0)                         # the other request runs here
            await t.mark_slot_exhausted(0, TOP)

        await asyncio.gather(refused_request(), refused_request())
        assert _used(t, 0, TOP) == 0
        assert t._slot_in_cooldown(t._slot(0, TOP))

    @pytest.mark.asyncio
    async def test_a_note_for_another_slot_is_not_applied(self):
        t = _tracker()
        t.note_quota_refusal(1, TOP, per_minute=True)
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


class TestTokensPerMinute:
    """TPM is an input-token limit per (key, model) slot over a rolling minute."""

    def test_reservations_accumulate_up_to_the_allowance(self):
        from src.llm.failures import TokenWindowFull
        from src.llm.model_registry import usable_tpm

        t = _tracker()
        limit = usable_tpm(TOP)
        t.reserve_input_tokens(0, TOP, limit - 1000)
        t.reserve_input_tokens(0, TOP, 1000)               # exactly full
        with pytest.raises(TokenWindowFull) as full:
            t.reserve_input_tokens(0, TOP, 1)
        assert 0 < full.value.retry_after <= 60

    # Monotonic clock readings (seconds since boot) in the minute below 2**16,
    # 2**17 and 2**18 seconds, where a stamp plus 60 crosses a power of two.
    @pytest.mark.parametrize("clock", [65508.146, 131025.45, 262121.621])
    def test_the_wait_never_exceeds_the_window_whatever_the_clock_reads(self, clock, monkeypatch):
        # Windows' monotonic clock ticks every ~16 ms, so a refused request often
        # reads the same instant as the booking it waits for, and the wait is the
        # whole minute. Computed as stamp + 60 - now, it came out an ulp over 60 s
        # whenever the stamp lay in the minute below a power of two: at ~36.4 h of
        # uptime the test above failed with 60.00000000001455.
        import src.llm.budget_tracker as budget_tracker
        from src.llm.failures import TokenWindowFull
        from src.llm.model_registry import usable_tpm

        monkeypatch.setattr(budget_tracker.time, "monotonic", lambda: clock)
        t = _tracker()
        limit = usable_tpm(TOP)
        t.reserve_input_tokens(0, TOP, limit)
        with pytest.raises(TokenWindowFull) as full:
            t.reserve_input_tokens(0, TOP, 1)
        assert full.value.retry_after == 60.0

    def test_each_slot_has_its_own_window(self):
        from src.llm.model_registry import usable_tpm

        t = _tracker()
        t.reserve_input_tokens(0, TOP, usable_tpm(TOP))
        assert t.reserve_input_tokens(1, TOP, 5000) is not None       # other key
        assert t.reserve_input_tokens(0, SECOND, 5000) is not None    # other model

    def test_tokens_roll_off_after_a_minute(self):
        from src.llm.failures import TokenWindowFull
        from src.llm.model_registry import usable_tpm

        t = _tracker()
        t.reserve_input_tokens(0, TOP, usable_tpm(TOP))
        with pytest.raises(TokenWindowFull):
            t.reserve_input_tokens(0, TOP, 10)

        t._token_windows[t._slot(0, TOP)][0][0] -= 61      # a minute later
        assert t.reserve_input_tokens(0, TOP, 10) is not None

    def test_wait_is_until_enough_has_rolled_off(self):
        from src.llm.failures import TokenWindowFull
        from src.llm.model_registry import usable_tpm

        t = _tracker()
        limit = usable_tpm(TOP)
        t.reserve_input_tokens(0, TOP, limit // 2)
        t.reserve_input_tokens(0, TOP, limit // 2)
        window = t._token_windows[t._slot(0, TOP)]
        window[0][0] -= 50                                 # first entry frees in ~10s
        window[1][0] -= 20                                 # second in ~40s

        with pytest.raises(TokenWindowFull) as small:
            t.reserve_input_tokens(0, TOP, limit // 4)     # the first entry is enough
        with pytest.raises(TokenWindowFull) as large:
            t.reserve_input_tokens(0, TOP, limit - 10)     # needs both gone
        assert 8 < small.value.retry_after < 12
        assert 38 < large.value.retry_after < 42

    def test_estimate_is_corrected_to_the_providers_count(self):
        from src.llm.model_registry import usable_tpm

        t = _tracker()
        limit = usable_tpm(TOP)
        reservation = t.reserve_input_tokens(0, TOP, limit)           # over-estimate
        reservation.adjust(limit // 10)                               # provider's count
        assert t.reserve_input_tokens(0, TOP, limit // 2) is not None
        reservation.adjust(None)                                      # no usage: keep it
        reservation.adjust(0)

    def test_a_refused_request_gives_its_tokens_back(self):
        from src.llm.model_registry import usable_tpm

        t = _tracker()
        reservation = t.reserve_input_tokens(0, TOP, usable_tpm(TOP))
        reservation.release()
        reservation.release()                                         # idempotent
        assert t.reserve_input_tokens(0, TOP, usable_tpm(TOP)) is not None

    def test_a_request_larger_than_the_allowance_is_left_to_the_provider(self):
        from src.llm.model_registry import usable_tpm

        t = _tracker()
        assert t.reserve_input_tokens(0, TOP, usable_tpm(TOP) * 2) is not None

    def test_local_and_unknown_models_are_not_metered(self):
        t = _tracker()
        assert t.reserve_input_tokens(-1, LOCAL_MODEL, 10**9) is None
        assert t.reserve_input_tokens(0, "gemini/not-in-the-registry", 10**9) is None


class TestQuotaDay:
    """The daily counters reset when the PROVIDER's day changes: midnight Pacific."""

    def test_quota_day_is_the_pacific_calendar_date(self):
        from datetime import date, datetime, timezone

        from src.llm.model_registry import quota_day

        # 03:00 UTC on 15 Nov is still 14 Nov in Los Angeles (19:00 PST).
        assert quota_day(datetime(2026, 11, 15, 3, 0, tzinfo=timezone.utc)) == date(2026, 11, 14)
        # 08:00 UTC is midnight PST: the provider's day has just turned.
        assert quota_day(datetime(2026, 11, 15, 8, 0, tzinfo=timezone.utc)) == date(2026, 11, 15)
        # In summer the reset is at 07:00 UTC (PDT).
        assert quota_day(datetime(2026, 7, 1, 6, 59, tzinfo=timezone.utc)) == date(2026, 6, 30)
        assert quota_day(datetime(2026, 7, 1, 7, 0, tzinfo=timezone.utc)) == date(2026, 7, 1)

    def test_fallback_without_tzdata_never_resets_early(self, monkeypatch):
        """
        No IANA database: fixed UTC-8. In summer that is an hour LATE, which
        only withholds a fresh slot for an hour; an hour early would offer a
        slot the provider still considers spent.
        """
        from datetime import date, datetime, timezone

        import src.llm.model_registry as registry

        def missing(_name):
            raise registry.ZoneInfoNotFoundError("no tzdata")

        monkeypatch.setattr(registry, "ZoneInfo", missing)
        assert registry.quota_day(datetime(2026, 7, 1, 7, 30, tzinfo=timezone.utc)) == date(2026, 6, 30)
        assert registry.quota_day(datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)) == date(2026, 7, 1)

    @pytest.mark.asyncio
    async def test_utc_midnight_does_not_reset_the_counters(self, monkeypatch):
        """
        The audit scenario. A slot is spent; UTC midnight passes; the provider
        has NOT reset. The slot must stay closed until the Pacific day turns,
        and then open — it used to open 8 hours early, 429, and be stamped
        spent for the whole next day.
        """
        from datetime import date

        import src.llm.budget_tracker as bt

        day = {"today": date(2026, 11, 14)}
        monkeypatch.setattr(bt, "quota_day", lambda: day["today"])

        t = _tracker(keys=(K0,))
        await t.mark_slot_exhausted(0, TOP)                 # provider: spent for 14 Nov
        assert (await t.get_model_for_agent()).model == SECOND

        # 00:30 UTC on the 15th is still the 14th in Pacific: quota_day unchanged.
        assert (await t.get_model_for_agent()).model == SECOND

        day["today"] = date(2026, 11, 15)                   # midnight Pacific
        assert (await t.get_model_for_agent()).model == TOP
        assert _used(t, 0, TOP) == 1

    @pytest.mark.asyncio
    async def test_a_counter_dated_ahead_of_the_quota_day_is_reset(self, monkeypatch):
        """
        Rows written when the day was counted in UTC can be dated tomorrow.
        The availability check and the debit must agree about them.
        """
        from datetime import date

        import src.llm.budget_tracker as bt

        monkeypatch.setattr(bt, "quota_day", lambda: date(2026, 11, 14))
        t = _tracker(keys=(K0,))
        slot = t._slot(0, TOP)
        t._mock_budgets[slot] = {"used_today": usable_rpd(TOP), "reset_date": "2026-11-15"}

        assert await t._budget_available(slot)
        assert await t._try_consume(slot), "offered by the check, so the debit must succeed"
        assert t._mock_budgets[slot] == {"used_today": 1, "reset_date": "2026-11-14"}

    @pytest.mark.asyncio
    async def test_sql_reset_matches_any_other_day(self):
        """Postgres path: the reset must fire for a stored date on either side."""
        calls = []

        class _Conn:
            async def execute(self, query, *args):
                calls.append(query)
                return "UPDATE 1"

        class _Acquire:
            async def __aenter__(self):
                return _Conn()

            async def __aexit__(self, *exc):
                return False

        t = _tracker(keys=(K0,))
        t._is_mock = False
        t._pool = type("P", (), {"acquire": lambda self: _Acquire()})()

        assert await t._try_consume(t._slot(0, TOP))
        reset = next(q for q in calls if "used_today = 0" in q)
        assert "reset_date <> $1::date" in reset


class TestSharedProjectWarning:
    """Limits are per Google Cloud project; keys from one project add nothing."""

    @staticmethod
    def _warnings(caplog):
        return [r for r in caplog.records if "per Google Cloud PROJECT" in r.getMessage()]

    @pytest.mark.asyncio
    async def test_warns_once_when_two_keys_are_refused_with_headroom(self, caplog):
        import logging

        t = _tracker()
        with caplog.at_level(logging.WARNING, logger="src.llm.budget_tracker"):
            await t.mark_slot_exhausted(0, TOP)
            assert not self._warnings(caplog), "one refusal proves nothing"
            await t.mark_slot_exhausted(1, TOP)             # we had counted 0 of 475
            await t.mark_slot_exhausted(1, SECOND)
            await t.mark_slot_exhausted(0, SECOND)
        assert len(self._warnings(caplog)) == 1

    @pytest.mark.asyncio
    async def test_silent_when_our_own_count_explains_the_refusal(self, caplog):
        import logging

        t = _tracker()
        slot = t._slot(1, TOP)
        for _ in range(usable_rpd(TOP) - 3):
            await t._try_consume(slot)                      # we really did spend it

        with caplog.at_level(logging.WARNING, logger="src.llm.budget_tracker"):
            await t.mark_slot_exhausted(0, TOP)
            await t.mark_slot_exhausted(1, TOP)
        assert not self._warnings(caplog)
