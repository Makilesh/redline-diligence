"""
The call layer end to end: a scripted provider, the real BudgetTracker (in its
in-memory mode) and the real wrapper functions.

Each scenario states what the provider did and asserts three things — where the
request ended up, how many provider requests it cost, and that the tracker's
count matches what the provider actually served.
"""

import asyncio
import json

import litellm
import pytest

import src.llm.budget_tracker as bt
import src.llm.litellm_wrapper as wrapper
from src.llm.failures import ProseAgentFailed, StructuredOutputError
from src.llm.model_registry import AGENT_LADDER, LOCAL_MODEL, SYNTHESIS_LADDER, usable_rpd

K0, K1 = "A" * 40, "B" * 40
TOP, SECOND, THIRD = AGENT_LADDER[0], AGENT_LADDER[1], AGENT_LADDER[2]
SYN_TOP, SYN_SECOND = SYNTHESIS_LADDER[0], SYNTHESIS_LADDER[1]

PER_DAY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"


def _err(kind: str) -> Exception:
    if kind in ("429-day", "429-minute"):
        quota_id = PER_DAY if kind == "429-day" else PER_MINUTE
        return litellm.RateLimitError(
            message=f'GeminiException - {{"error": {{"code": 429, "details": [{{"quotaId": '
                    f'"{quota_id}"}}, {{"retryDelay": "20s"}}]}}}}',
            llm_provider="gemini", model="m")
    return {
        "503": lambda: litellm.ServiceUnavailableError(
            message="503 high demand", llm_provider="gemini", model="m"),
        "timeout": lambda: litellm.Timeout(
            message="Timeout passed=60.0", model="m", llm_provider="gemini"),
        "404": lambda: litellm.NotFoundError(
            message="no longer available to new users", model="m", llm_provider="gemini"),
        "auth": lambda: litellm.BadRequestError(
            message="API key not valid. Please pass a valid API key.",
            model="m", llm_provider="gemini"),
        "500": lambda: litellm.InternalServerError(
            message="An internal error has occurred", llm_provider="gemini", model="m"),
        "conn": lambda: litellm.APIConnectionError(
            message="Connection reset by peer", llm_provider="gemini", model="m"),
        "bad-request": lambda: litellm.BadRequestError(
            message="Request contains an invalid argument", model="m", llm_provider="gemini"),
        "unclassified": lambda: RuntimeError("stream could not be opened"),
    }[kind]()


class _Resp:
    def __init__(self, content, finish_reason="stop"):
        message = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": message, "finish_reason": finish_reason})()]
        self.usage = None


class Provider:
    """
    Scripted litellm.acompletion.

    `rule(model, key_index, n, kwargs)` returns an error-kind string to raise,
    a _Resp to return, or None for the default good reply.
    """

    def __init__(self, rule, good='{"ok": true}'):
        self.rule = rule
        self.good = good
        self.calls: list[tuple[str, int]] = []
        self.served: list[tuple[str, int]] = []
        self.requests: list[dict] = []

    async def __call__(self, **kwargs):
        key_index = {K0: 0, K1: 1}.get(kwargs.get("api_key"), -1)
        n = len(self.calls)
        self.calls.append((kwargs["model"], key_index))
        self.requests.append(kwargs)
        verdict = self.rule(kwargs["model"], key_index, n, kwargs)
        if isinstance(verdict, str):
            raise _err(verdict)
        self.served.append((kwargs["model"], key_index))
        return verdict if verdict is not None else _Resp(self.good)


@pytest.fixture
def tracker(monkeypatch):
    t = bt.BudgetTracker.__new__(bt.BudgetTracker)
    t._api_keys = [K0, K1]
    t._rate_limiters = {}
    t._is_mock = True
    t._mock_budgets = {}
    monkeypatch.setattr(bt.BudgetTracker, "_instance", t)
    monkeypatch.setenv("VERIFICATION_BACKEND", "cloud")
    monkeypatch.setattr(wrapper, "RETRY_BACKOFF_SECONDS", 0.0)
    return t


def _install(monkeypatch, rule, **kw) -> Provider:
    provider = Provider(rule, **kw)
    monkeypatch.setattr(wrapper.litellm, "acompletion", provider)
    return provider


def _units(t) -> dict[tuple[str, int], int]:
    out = {}
    for slot, b in t._mock_budgets.items():
        if b["used_today"]:
            key, _, model = slot.partition("::")
            out[(model, int(key))] = b["used_today"]
    return out


async def _agent_call(t, served=None):
    """What Agents 1, 5 and 6 do: select once, then call."""
    choice = await t.get_model_for_agent()
    return await wrapper.call_structured_agent(
        system_prompt="s", user_prompt="u", model=choice.model, api_key=choice.api_key,
        served=served,
    )


def _first_slot_fails(kind):
    return lambda model, key, n, kw: kind if (model == TOP and key == 0) else None


# ─── Structured agents: one failure on the selected slot ──────────────────────


class TestStructuredAgentDescends:
    """Agents 1/5/6 select once; the descent has to happen inside the call."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind, lands_on", [
        ("429-day", (TOP, 1)),        # this key is spent: next key, same model
        ("429-minute", (TOP, 1)),
        ("404", (TOP, 1)),            # this key cannot use the model
        ("auth", (TOP, 1)),           # this key is dead
        ("503", (SECOND, 0)),         # the model is down for every key
        ("timeout", (SECOND, 0)),
        ("500", (SECOND, 0)),
        ("conn", (SECOND, 0)),
    ])
    async def test_one_failure_costs_one_extra_request(self, tracker, monkeypatch, kind, lands_on):
        provider = _install(monkeypatch, _first_slot_fails(kind))
        served = wrapper.ServedModel(TOP)

        result = await _agent_call(tracker, served)

        assert result == {"ok": True}
        assert provider.calls == [(TOP, 0), lands_on], "no retry on the slot that refused"
        assert served.model == lands_on[0]

    @pytest.mark.asyncio
    async def test_never_sleeps_on_a_refused_key(self, tracker, monkeypatch):
        """It used to back off 20s + 40s against the key that had said 429."""
        slept = []

        async def no_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr(wrapper.asyncio, "sleep", no_sleep)
        _install(monkeypatch, _first_slot_fails("429-day"))
        await _agent_call(tracker)
        assert slept == []

    @pytest.mark.asyncio
    async def test_malformed_request_does_not_walk_the_ladder(self, tracker, monkeypatch):
        """A 400 fails the same way on every rung; trying them only spends quota."""
        provider = _install(monkeypatch, lambda m, k, n, kw: None if m == LOCAL_MODEL else "bad-request")
        # Local model is tried once as the last resort, as before.
        result = await _agent_call(tracker)
        assert result == {"ok": True}
        assert provider.calls == [(TOP, 0), (LOCAL_MODEL, -1)]

    @pytest.mark.asyncio
    async def test_everything_down_reports_the_cloud_failure(self, tracker, monkeypatch):
        """No local model listening is not the interesting error."""
        _install(monkeypatch, lambda m, k, n, kw: "conn" if m == LOCAL_MODEL else "503")
        with pytest.raises(litellm.ServiceUnavailableError):
            await _agent_call(tracker)

    @pytest.mark.asyncio
    async def test_provider_outage_tries_each_model_once(self, tracker, monkeypatch):
        """
        Every cloud model 503s. One request per model, then the local model —
        not a second lap of the ladder because all of them are cooling.
        """
        provider = _install(monkeypatch, lambda m, k, n, kw: None if m == LOCAL_MODEL else "503")
        served = wrapper.ServedModel(TOP)

        await _agent_call(tracker, served)

        assert provider.calls == [(TOP, 0), (SECOND, 0), (THIRD, 0), (LOCAL_MODEL, -1)]
        assert served.model == LOCAL_MODEL
        assert _units(tracker) == {}, "nothing was served, nothing is recorded"

    @pytest.mark.asyncio
    async def test_hops_scale_with_the_number_of_keys(self, monkeypatch):
        """
        Five keys, top model spent on all of them (stale counter). A fixed
        bound of six used five hops rotating keys and one on the next model.
        """
        keys = [c * 40 for c in "ABCDE"]
        t = bt.BudgetTracker.__new__(bt.BudgetTracker)
        t._api_keys = list(keys)
        t._rate_limiters = {}
        t._is_mock = True
        t._mock_budgets = {}
        monkeypatch.setattr(bt.BudgetTracker, "_instance", t)

        calls = []

        async def provider(**kwargs):
            calls.append((kwargs["model"], keys.index(kwargs["api_key"])))
            if kwargs["model"] == TOP or (kwargs["model"] == SECOND and kwargs["api_key"] in keys[:3]):
                raise _err("429-day")
            return _Resp('{"ok": true}')

        monkeypatch.setattr(wrapper.litellm, "acompletion", provider)
        result = await _agent_call(t)

        assert result == {"ok": True}
        assert calls[-1] == (SECOND, 3), "reached a live slot instead of the local model"


# ─── Accounting ───────────────────────────────────────────────────────────────


class TestCountMatchesWhatWasServed:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["429-minute", "404", "auth", "503", "conn"])
    async def test_a_refused_request_is_not_counted(self, tracker, monkeypatch, kind):
        provider = _install(monkeypatch, _first_slot_fails(kind))
        await _agent_call(tracker)

        served_slot = provider.served[0]
        assert _units(tracker) == {served_slot: 1}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["timeout", "500"])
    async def test_an_admitted_request_stays_counted(self, tracker, monkeypatch, kind):
        """The provider may have counted it; under-counting is the costly error."""
        _install(monkeypatch, _first_slot_fails(kind))
        await _agent_call(tracker)
        assert _units(tracker) == {(TOP, 0): 1, (SECOND, 0): 1}

    @pytest.mark.asyncio
    async def test_per_minute_429_does_not_spend_the_day(self, tracker, monkeypatch):
        _install(monkeypatch, _first_slot_fails("429-minute"))
        await _agent_call(tracker)
        assert _units(tracker) == {(TOP, 1): 1}, "was 475 units for one unserved request"
        assert tracker._slot_in_cooldown(tracker._slot(0, TOP))

    @pytest.mark.asyncio
    async def test_per_day_429_spends_the_slot(self, tracker, monkeypatch):
        _install(monkeypatch, _first_slot_fails("429-day"))
        await _agent_call(tracker)
        assert _units(tracker) == {(TOP, 0): usable_rpd(TOP), (TOP, 1): 1}

    @pytest.mark.asyncio
    async def test_every_provider_request_is_paid_for(self, tracker, monkeypatch):
        """A repair request is a request: one unit and one window entry each."""
        provider = _install(
            monkeypatch,
            lambda m, k, n, kw: _Resp("not json") if n == 0 else None,
        )
        await _agent_call(tracker)

        assert provider.calls == [(TOP, 0), (TOP, 0)]
        assert _units(tracker) == {(TOP, 0): 2}
        assert tracker._get_rate_limiter(tracker._slot(0, TOP)).current_usage == 2


class TestTokensPerMinuteAtTheCallBoundary:
    """A request that would overrun TPM is not sent; it rotates like a 429."""

    @staticmethod
    def _fill(tracker, key_index, model):
        from src.llm.model_registry import usable_tpm

        tracker.reserve_input_tokens(key_index, model, usable_tpm(model))

    @pytest.mark.asyncio
    async def test_structured_call_rotates_to_the_next_key_without_sending(
        self, tracker, monkeypatch
    ):
        self._fill(tracker, 0, TOP)
        provider = _install(monkeypatch, lambda m, k, n, kw: None)

        assert await _agent_call(tracker) == {"ok": True}
        assert provider.calls == [(TOP, 1)], "nothing was sent on the full slot"
        assert _units(tracker) == {(TOP, 1): 1}, "the unit reserved for it came back"
        assert tracker._slot_in_cooldown(tracker._slot(0, TOP))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_prose_call_hands_the_refusal_to_the_ladder(
        self, tracker, monkeypatch, streaming
    ):
        from src.llm.failures import TokenWindowFull

        self._fill(tracker, 0, SYN_TOP)
        provider = _install(monkeypatch, lambda m, k, n, kw: None, good="answer")
        choice = await tracker.get_model_for_synthesis()

        with pytest.raises(TokenWindowFull) as full:
            if streaming:
                await wrapper.stream_prose_agent(
                    "s", "u", model=choice.model, on_token=lambda _t: None,
                    api_key=choice.api_key)
            else:
                await wrapper.call_prose_agent(
                    "s", "u", model=choice.model, api_key=choice.api_key)

        assert provider.calls == []
        assert wrapper.is_quota_error(full.value), "the synthesizer's ladder rotates on this"
        # What the synthesizer does next: mark the slot, select again.
        await tracker.mark_slot_exhausted(choice.key_index, choice.model)
        assert _units(tracker) == {}, "rested for the minute, not spent for the day"
        nxt = await tracker.get_model_for_synthesis()
        assert (nxt.model, nxt.key_index) == (SYN_TOP, 1)

    @pytest.mark.asyncio
    async def test_window_holds_the_providers_count_not_the_estimate(self, tracker, monkeypatch):
        reply = _Resp('{"ok": true}')
        reply.usage = type("U", (), {"prompt_tokens": 1234})()
        _install(monkeypatch, lambda m, k, n, kw: reply)

        choice = await tracker.get_model_for_agent()
        await wrapper.call_structured_agent(
            "s" * 3000, "u" * 3000, model=choice.model, api_key=choice.api_key)

        window = tracker._token_windows[tracker._slot(0, TOP)]
        assert [entry[1] for entry in window] == [1234]

    @pytest.mark.asyncio
    async def test_estimate_stands_when_the_provider_reports_no_usage(self, tracker, monkeypatch):
        _install(monkeypatch, lambda m, k, n, kw: None)
        choice = await tracker.get_model_for_agent()
        await wrapper.call_structured_agent(
            "s" * 3000, "u" * 3000, model=choice.model, api_key=choice.api_key)

        (entry,) = tracker._token_windows[tracker._slot(0, TOP)]
        assert entry[1] == 2001, "6000 characters at 3 per token, rounded up"

    @pytest.mark.asyncio
    async def test_a_refused_request_does_not_occupy_the_window(self, tracker, monkeypatch):
        _install(monkeypatch, _first_slot_fails("503"))
        await _agent_call(tracker)
        assert not tracker._token_windows[tracker._slot(0, TOP)]
        assert len(tracker._token_windows[tracker._slot(0, SECOND)]) == 1


# ─── Unusable structured output ───────────────────────────────────────────────


class TestUnusableOutput:
    @pytest.mark.asyncio
    async def test_repair_request_shows_the_model_its_own_reply(self, tracker, monkeypatch):
        provider = _install(
            monkeypatch,
            lambda m, k, n, kw: _Resp("Sure! {query_type: financial") if n == 0 else None,
        )
        assert await _agent_call(tracker) == {"ok": True}

        repair = provider.requests[1]["messages"]
        assert [m["role"] for m in repair] == ["system", "user", "assistant", "user"]
        assert "Sure! {query_type: financial" in repair[2]["content"]
        assert "JSON object" in repair[3]["content"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reply, finish", [(None, "stop"), ("", "length"),
                                               ('{"a": "cut of', "length")])
    async def test_cut_off_or_empty_reply_gets_more_room(self, tracker, monkeypatch, reply, finish):
        """With a thinking model an empty reply at the limit is a budget problem."""
        provider = _install(
            monkeypatch,
            lambda m, k, n, kw: _Resp(reply, finish) if n == 0 else None,
        )
        choice = await tracker.get_model_for_agent()
        await wrapper.call_structured_agent(
            "s", "u", model=choice.model, api_key=choice.api_key, max_tokens=600)

        first, second = provider.requests
        assert second["max_tokens"] == 2 * first["max_tokens"] == 1200
        assert len(second["messages"]) == 2, "same question, not a re-ask"

    @pytest.mark.asyncio
    async def test_two_bad_replies_move_to_another_model(self, tracker, monkeypatch):
        provider = _install(
            monkeypatch, lambda m, k, n, kw: _Resp("not json") if m == TOP else None)
        served = wrapper.ServedModel(TOP)

        assert await _agent_call(tracker, served) == {"ok": True}
        assert provider.calls == [(TOP, 0), (TOP, 0), (SECOND, 0)]
        assert served.model == SECOND
        assert tracker._models_in_cooldown() == set(), "bad output is not an outage"
        assert (await tracker.get_model_for_agent()).model == TOP

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reply", ['["financial", "legal"]', '"just a string"', "42", "null"])
    async def test_json_that_is_not_an_object_is_rejected(self, tracker, monkeypatch, reply):
        """It used to be returned as-is and crash the agent on its first .get."""
        _install(monkeypatch, lambda m, k, n, kw: _Resp(reply))
        with pytest.raises(StructuredOutputError):
            await wrapper.call_structured_agent("s", "u", model=LOCAL_MODEL)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reply", [
        '```json\n{"ok": true}\n```',
        '```\n{"ok": true}\n```',
        'Here is the JSON you asked for: {"ok": true} Let me know if it helps.',
    ])
    async def test_wrapped_objects_are_salvaged_without_a_second_request(
        self, tracker, monkeypatch, reply
    ):
        provider = _install(monkeypatch, lambda m, k, n, kw: _Resp(reply))
        assert await _agent_call(tracker) == {"ok": True}
        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_no_repair_when_the_slot_cannot_pay_for_it(self, tracker, monkeypatch):
        """One daily unit left: the bad reply used it, so move on instead."""
        slot = tracker._slot(0, TOP)
        for _ in range(usable_rpd(TOP) - 1):
            await tracker._try_consume(slot)
        provider = _install(
            monkeypatch, lambda m, k, n, kw: _Resp("not json") if (m, k) == (TOP, 0) else None)

        assert await _agent_call(tracker) == {"ok": True}
        assert provider.calls.count((TOP, 0)) == 1
        assert tracker._mock_budgets[slot]["used_today"] == usable_rpd(TOP)

    @pytest.mark.asyncio
    async def test_unmetered_call_still_raises_value_error(self, monkeypatch):
        """Direct callers with no tracker keep the documented contract."""
        monkeypatch.setattr(bt.BudgetTracker, "_instance", None)
        provider = _install(monkeypatch, lambda m, k, n, kw: _Resp("not json"))
        with pytest.raises(ValueError):
            await wrapper.call_structured_agent("s", "u", model="gemini/x", api_key="k")
        assert len(provider.calls) == 2, "one request and one repair, not three identical ones"


# ─── Verification ladder ──────────────────────────────────────────────────────


class TestVerificationLadder:
    @pytest.mark.asyncio
    async def test_shares_the_structured_descent(self, tracker, monkeypatch):
        provider = _install(monkeypatch, _first_slot_fails("503"))

        async def run():
            result, model = await wrapper.call_verification_agent_with_model("s", "u")
            return result, model, wrapper.active_verification_model()

        result, model, active = await asyncio.create_task(run())
        assert (result, model, active) == ({"ok": True}, SECOND, SECOND)
        assert provider.calls == [(TOP, 0), (SECOND, 0)]

    @pytest.mark.asyncio
    async def test_a_good_answer_ends_the_failure_streak(self, tracker, monkeypatch):
        """Only the synthesizer used to report health; agent models never recovered."""
        tracker.report_model_unavailable(TOP)
        tracker._model_cooldowns[TOP] = 0.0          # cooldown over, streak still recorded
        assert tracker._model_failures[TOP] == 1

        _install(monkeypatch, lambda m, k, n, kw: None)
        await wrapper.call_verification_agent("s", "u")
        assert TOP not in tracker._model_failures

    @pytest.mark.asyncio
    async def test_local_backend_never_touches_the_cloud(self, tracker, monkeypatch):
        monkeypatch.setenv("VERIFICATION_BACKEND", "local")
        provider = _install(monkeypatch, lambda m, k, n, kw: None)
        _result, model = await wrapper.call_verification_agent_with_model("s", "u")
        assert model == LOCAL_MODEL and provider.calls == [(LOCAL_MODEL, -1)]
        assert _units(tracker) == {}


# ─── Prose and streaming ──────────────────────────────────────────────────────


async def _synthesis_call(t, **kw):
    choice = await t.get_model_for_synthesis()
    return choice, await wrapper.call_prose_agent(
        system_prompt="s", user_prompt="u", model=choice.model, api_key=choice.api_key, **kw)


class TestProseAgent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["429-day", "429-minute", "404", "auth", "503", "timeout"])
    async def test_slot_level_refusals_are_raised_on_first_sight(self, tracker, monkeypatch, kind):
        """429/404/auth used to be retried three times on the key that refused."""
        provider = _install(monkeypatch, lambda m, k, n, kw: kind)
        with pytest.raises(Exception) as raised:
            await _synthesis_call(tracker)
        assert len(provider.calls) == 1
        assert not isinstance(raised.value, ProseAgentFailed), "the ladder needs the real class"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_a_429_costs_the_same_on_the_blocking_and_streaming_paths(
        self, tracker, monkeypatch, streaming
    ):
        """
        Blocking /query slept 2s + 4s on the key that had said 429 before the
        ladder could rotate (36s with every rung refusing); the streaming path
        raised at once. A 429 is a property of the slot — rotate, don't sleep.
        """
        slept = []

        async def no_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr(wrapper.asyncio, "sleep", no_sleep)
        provider = _install(monkeypatch, lambda m, k, n, kw: "429-day")
        choice = await tracker.get_model_for_synthesis()

        with pytest.raises(litellm.RateLimitError):
            if streaming:
                await wrapper.stream_prose_agent(
                    "s", "u", model=choice.model, on_token=lambda _t: None,
                    api_key=choice.api_key)
            else:
                await wrapper.call_prose_agent(
                    "s", "u", model=choice.model, api_key=choice.api_key)

        assert len(provider.calls) == 1 and slept == []

    @pytest.mark.asyncio
    async def test_in_place_retries_are_charged(self, tracker, monkeypatch):
        provider = _install(monkeypatch, lambda m, k, n, kw: _Resp(""))
        with pytest.raises(ProseAgentFailed):
            await _synthesis_call(tracker)
        assert len(provider.calls) == wrapper.MAX_RETRIES
        assert _units(tracker) == {(SYN_TOP, 0): wrapper.MAX_RETRIES}

    @pytest.mark.asyncio
    async def test_a_500_goes_to_the_ladder_instead_of_spending_three_units(
        self, tracker, monkeypatch
    ):
        """An admitted request keeps its unit; retrying in place tripled the cost."""
        provider = _install(monkeypatch, lambda m, k, n, kw: "500")
        with pytest.raises(litellm.InternalServerError) as raised:
            await _synthesis_call(tracker)
        assert len(provider.calls) == 1
        assert _units(tracker) == {(SYN_TOP, 0): 1}
        assert wrapper.is_service_unavailable(raised.value), "the synthesizer moves on this"

    @pytest.mark.asyncio
    async def test_retries_stop_when_the_slot_cannot_pay(self, tracker, monkeypatch):
        slot = tracker._slot(0, SYN_TOP)
        for _ in range(usable_rpd(SYN_TOP) - 2):
            await tracker._try_consume(slot)                 # two units left
        provider = _install(monkeypatch, lambda m, k, n, kw: _Resp(""))

        with pytest.raises(ProseAgentFailed):
            await _synthesis_call(tracker)
        assert len(provider.calls) == 2
        assert tracker._mock_budgets[slot]["used_today"] == usable_rpd(SYN_TOP)

    @pytest.mark.asyncio
    async def test_exhausted_retries_send_the_ladder_on(self, tracker, monkeypatch):
        """Three empty completions used to end synthesis in a refusal."""
        _install(monkeypatch, lambda m, k, n, kw: _Resp(None))
        with pytest.raises(ProseAgentFailed) as raised:
            await _synthesis_call(tracker)
        assert wrapper.is_service_unavailable(raised.value)

    @pytest.mark.asyncio
    async def test_connection_failures_cost_nothing(self, tracker, monkeypatch):
        _install(monkeypatch, lambda m, k, n, kw: "conn")
        with pytest.raises(ProseAgentFailed):
            await _synthesis_call(tracker)
        assert _units(tracker) == {}


class _Chunk:
    def __init__(self, text):
        delta = type("D", (), {"content": text})()
        self.choices = [type("C", (), {"delta": delta})()]
        self.usage = None


def _stream(parts, fail_after=None):
    async def gen():
        for i, part in enumerate(parts):
            if fail_after is not None and i == fail_after:
                raise ConnectionError("stream reset")
            yield _Chunk(part)
    return gen()


class TestStreaming:
    @pytest.mark.asyncio
    async def test_failure_before_the_first_token_falls_back_and_is_charged(
        self, tracker, monkeypatch
    ):
        def rule(model, key, n, kw):
            if kw.get("stream"):
                return "unclassified"
            return _Resp("the whole answer")

        provider = _install(monkeypatch, rule)
        tokens = []
        choice = await tracker.get_model_for_synthesis()
        text = await wrapper.stream_prose_agent(
            "s", "u", model=choice.model, on_token=tokens.append, api_key=choice.api_key)

        assert text == "the whole answer" and tokens == ["the whole answer"]
        assert len(provider.calls) == 2
        assert _units(tracker) == {(SYN_TOP, 0): 2}

    @pytest.mark.asyncio
    async def test_mid_stream_failure_is_surfaced_and_reported(self, tracker, monkeypatch):
        async def provider(**kwargs):
            return _stream(["Revenue ", "was ", "$452.8M"], fail_after=2)

        monkeypatch.setattr(wrapper.litellm, "acompletion", provider)
        tokens = []
        choice = await tracker.get_model_for_synthesis()

        with pytest.raises(wrapper.StreamInterrupted) as raised:
            await wrapper.stream_prose_agent(
                "s", "u", model=choice.model, on_token=tokens.append, api_key=choice.api_key)

        assert tokens == ["Revenue ", "was "] and raised.value.emitted_chars == 12
        assert _units(tracker) == {(SYN_TOP, 0): 1}, "tokens were served; the unit stays"
        assert SYN_TOP in tracker._models_in_cooldown()

    @pytest.mark.asyncio
    async def test_empty_stream_sends_the_ladder_on(self, tracker, monkeypatch):
        async def provider(**kwargs):
            return _stream([])

        monkeypatch.setattr(wrapper.litellm, "acompletion", provider)
        choice = await tracker.get_model_for_synthesis()
        with pytest.raises(ProseAgentFailed):
            await wrapper.stream_prose_agent(
                "s", "u", model=choice.model, on_token=lambda _t: None, api_key=choice.api_key)


# ─── Request scope vs provider health ─────────────────────────────────────────


class TestSkipIsRequestScoped:
    """skip_model_for_request used to write the process-wide cooldown."""

    @pytest.mark.asyncio
    async def test_a_skip_affects_only_the_request_that_made_it(self, tracker):
        async def request_a():
            tracker.skip_model_for_request(SYN_TOP)
            return (await tracker.get_model_for_synthesis()).model

        async def request_b():
            return (await tracker.get_model_for_synthesis()).model

        assert await asyncio.create_task(request_a()) == SYN_SECOND
        assert await asyncio.create_task(request_b()) == SYN_TOP, "another user is unaffected"
        assert tracker._models_in_cooldown() == set()
        assert not getattr(tracker, "_model_failures", {})

    @pytest.mark.asyncio
    async def test_a_rejected_answer_from_every_model_cools_none_of_them(self, tracker):
        """
        The audit scenario: every synthesis model returns an answer the caller
        rejects. One query must not leave the ladder degraded for the next.
        """
        async def one_query():
            tried = []
            for _ in range(6):
                choice = await tracker.get_model_for_synthesis()
                tried.append(choice.model)
                tracker.skip_model_for_request(choice.model)      # "unusable answer"
            return tried

        tried = await asyncio.create_task(one_query())
        assert tried == SYNTHESIS_LADDER[:6], "each model once, in order"
        assert tracker._models_in_cooldown() == set()

        nxt = await asyncio.create_task(tracker.get_model_for_synthesis())
        assert nxt.model == SYN_TOP

    @pytest.mark.asyncio
    async def test_skips_are_honoured_when_everything_else_is_cooling(self, tracker):
        """The blocking pass ignores cooldowns; it must not ignore a skip."""
        for model in AGENT_LADDER:
            tracker.report_model_unavailable(model)

        async def request():
            tracker.skip_model_for_request(TOP)
            return (await tracker.get_model_for_agent()).model

        assert await asyncio.create_task(request()) == SECOND

    @pytest.mark.asyncio
    async def test_a_pinned_run_is_never_substituted(self, tracker, monkeypatch):
        monkeypatch.setenv("SYNTHESIS_MODEL_PIN", SYN_TOP)

        async def request():
            tracker.skip_model_for_request(SYN_TOP)
            return (await tracker.get_model_for_synthesis()).model

        assert await asyncio.create_task(request()) == SYN_TOP

    @pytest.mark.asyncio
    async def test_a_new_query_starts_with_no_skips(self, tracker):
        async def two_queries_in_one_task():
            wrapper.set_trace_context(deal_id="d1")
            tracker.skip_model_for_request(SYN_TOP)
            first = (await tracker.get_model_for_synthesis()).model
            wrapper.set_trace_context(deal_id="d2")
            second = (await tracker.get_model_for_synthesis()).model
            return first, second

        assert await asyncio.create_task(two_queries_in_one_task()) == (SYN_SECOND, SYN_TOP)

    @pytest.mark.asyncio
    async def test_a_stale_skip_expires(self, tracker):
        async def request():
            tracker.skip_model_for_request(SYN_TOP)
            bt._request_skips.get()[SYN_TOP] -= bt.REQUEST_SKIP_TTL_SECONDS + 1
            return (await tracker.get_model_for_synthesis()).model

        assert await asyncio.create_task(request()) == SYN_TOP

    def test_outside_a_request_there_is_nothing_to_scope_to(self, tracker):
        tracker.skip_model_for_request(SYN_TOP)
        assert bt._models_skipped_for_request() == frozenset()


class TestProviderHealthIsReportedByTheWrapper:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind, cooled", [
        ("503", True), ("timeout", True), ("500", True),
        ("429-day", False), ("404", False), ("auth", False), ("conn", False),
    ])
    async def test_only_provider_faults_cool_the_model(self, tracker, monkeypatch, kind, cooled):
        import openai

        _install(monkeypatch, lambda m, k, n, kw: kind)
        with pytest.raises((openai.OpenAIError, ProseAgentFailed)):
            await _synthesis_call(tracker)
        assert (SYN_TOP in tracker._models_in_cooldown()) is cooled

    @pytest.mark.asyncio
    async def test_one_failed_call_is_one_step_of_backoff(self, tracker, monkeypatch):
        """However the call ended, it must not escalate the cooldown twice."""
        _install(monkeypatch, lambda m, k, n, kw: "500")
        with pytest.raises(litellm.InternalServerError):
            await _synthesis_call(tracker)
        assert tracker._model_failures[SYN_TOP] == 1

    @pytest.mark.asyncio
    async def test_structured_503_cools_the_model_for_everyone(self, tracker, monkeypatch):
        _install(monkeypatch, _first_slot_fails("503"))
        await asyncio.create_task(_agent_call(tracker))
        assert TOP in tracker._models_in_cooldown()
        assert (await asyncio.create_task(tracker.get_model_for_agent())).model == SECOND

    @pytest.mark.asyncio
    async def test_empty_completions_are_not_an_outage(self, tracker, monkeypatch):
        _install(monkeypatch, lambda m, k, n, kw: _Resp(""))
        with pytest.raises(ProseAgentFailed):
            await _synthesis_call(tracker)
        assert tracker._models_in_cooldown() == set()


def test_scripted_provider_reply_is_valid_json():
    """Guards the fixture itself: the default good reply must parse."""
    assert json.loads(Provider(lambda *a: None).good) == {"ok": True}
