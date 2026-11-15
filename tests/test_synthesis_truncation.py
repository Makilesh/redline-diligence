"""
A synthesis reply cut off at max_tokens is never presented as a finished answer.

Gemini 3.x counts thinking against max_tokens. In the 2026-11-15 evening run 29
of 54 synthesis calls stopped at 2,996 of 3,000 tokens and shipped mid-sentence,
and mh_06 was lost when three rungs in a row were cut off before their first
citation. The provider double below models that budget exactly: a model thinks
`think` tokens, then writes `answer` tokens, both counted against the request's
max_tokens; a request too small for both ends with finish_reason "length" and
whatever text fitted — nothing at all if it was still thinking. Profiles come
from live probes of the same day (gemini-3.6-flash, budget 16,384): mh_05
needed 3,992 + 1,818 tokens and started answering after 37.9s.
"""

import asyncio
import logging

import pytest

import src.agents.answer_synthesizer as synth
import src.llm.budget_tracker as bt
import src.llm.litellm_wrapper as wrapper
from src.llm.failures import CompletionTruncated, FailureKind, classify_failure
from src.utils import metrics
from tests.fixtures.virtual_time import run_virtual

K0, K1 = "A" * 40, "B" * 40
M38, M37 = "gemini/gemini-3.8-flash", "gemini/gemini-3.7-flash"
M36 = "gemini/gemini-3.6-flash"

SENTENCE = "Revenue was $452.8 million [📄 fin.txt | p.1]. "
CHARS_PER_TOKEN = 3  # measured 2.8-3.2 on live synthesis answers


def _text(tokens: int) -> str:
    """The first `tokens` tokens of a long, cited answer."""
    chars = tokens * CHARS_PER_TOKEN
    return (SENTENCE * (chars // len(SENTENCE) + 1))[:chars]


class Model:
    """A thinking model: `think` tokens of thought, then `answer` tokens of text."""

    def __init__(self, think=2000, answer=1500, rate=250.0, overhead=1.0):
        self.think, self.answer, self.rate, self.overhead = think, answer, rate, overhead

    def reply(self, budget: int):
        thought = min(self.think, budget)
        visible = max(0, min(self.answer, budget - self.think))
        finish = "stop" if self.think + self.answer <= budget else "length"
        usage = {
            "prompt_tokens": 5000,
            "completion_tokens": thought + visible,
            "completion_tokens_details": {"reasoning_tokens": thought},
        }
        return _text(visible), finish, usage, self.overhead + thought / self.rate


LOOPS = Model(think=500, answer=10**9)  # writes until it is stopped


def _choice(**fields):
    return type("C", (), fields)()


class Provider:
    """Scripted litellm.acompletion for Model profiles, streamed or not."""

    def __init__(self, models=None, default=None):
        self.models = models or {}
        self.default = default or Model()
        self.requests: list[dict] = []

    @property
    def budgets(self):
        return [(r["model"], r["max_tokens"]) for r in self.requests]

    async def __call__(self, **kwargs):
        self.requests.append(dict(kwargs))
        m = self.models.get(kwargs["model"], self.default)
        text, finish, usage, ttft = m.reply(kwargs["max_tokens"])
        if kwargs.get("stream"):
            return self._stream(text, finish, usage, ttft)
        await asyncio.sleep(ttft + len(text) / (m.rate * CHARS_PER_TOKEN))
        response = type("R", (), {})()
        message = _choice(content=text)
        response.choices = [_choice(message=message, finish_reason=finish)]
        response.usage = usage
        return response

    async def _stream(self, text, finish, usage, ttft):
        await asyncio.sleep(ttft)
        for i in range(0, len(text), 200):
            chunk = type("Chunk", (), {})()
            chunk.choices = [_choice(delta=_choice(content=text[i:i + 200]), finish_reason=None)]
            chunk.usage = None
            yield chunk
            await asyncio.sleep(0.05)
        last = type("Chunk", (), {})()
        last.choices = [_choice(delta=_choice(content=None), finish_reason=finish)]
        last.usage = None
        yield last
        tail = type("Chunk", (), {})()
        tail.choices = []
        tail.usage = usage
        yield tail


@pytest.fixture
def tracker(monkeypatch):
    t = bt.BudgetTracker.__new__(bt.BudgetTracker)
    t._api_keys = [K0, K1]
    t._rate_limiters = {}
    t._is_mock = True
    t._mock_budgets = {}
    monkeypatch.setattr(bt.BudgetTracker, "_instance", t)
    monkeypatch.setattr(wrapper, "RETRY_BACKOFF_SECONDS", 0.0)
    for name in ("PROSE_FIRST_TOKEN_TIMEOUT_SECONDS", "SYNTHESIS_MODEL_PIN", "SYNTHESIS_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    return t


def _install(monkeypatch, provider):
    monkeypatch.setattr(wrapper.litellm, "acompletion", provider)
    return provider


def _units(t):
    out = {}
    for slot, b in t._mock_budgets.items():
        if b["used_today"]:
            key, _, model = slot.partition("::")
            out[(model, int(key))] = b["used_today"]
    return out


async def _captured(coro):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        return e


async def _blocking(t, streamed, max_tokens=3000, outcome=None):
    """One /query-style synthesis request on (M36, key 0), selection paid for."""
    await t._try_consume(t._slot(0, M36))
    return await wrapper.call_prose_agent(
        "s", "u", model=M36, api_key=K0, max_tokens=max_tokens,
        stream_with_deadline=streamed, outcome=outcome)


async def _sse(t, tokens, max_tokens=3000, outcome=None):
    """One /query/stream-style request on (M36, key 0), selection paid for."""
    await t._try_consume(t._slot(0, M36))
    return await wrapper.stream_prose_agent(
        "s", "u", model=M36, on_token=tokens.append, api_key=K0,
        max_tokens=max_tokens, outcome=outcome)


# ─── /query: the blocking call and the internal stream behind it ─────────────


@pytest.mark.parametrize("streamed", [False, True], ids=["blocking", "internal-stream"])
class TestQueryPath:
    def test_cut_off_reply_gets_one_retry_with_twice_the_room(self, tracker, monkeypatch, streamed):
        """mh_05's profile at the old budget: cut at 3,000, complete at 6,000."""
        provider = _install(monkeypatch, Provider(default=Model(think=3992, answer=1818)))
        outcome = wrapper.ProseOutcome()

        text, _ = run_virtual(_blocking(tracker, streamed, outcome=outcome))

        assert text == _text(1818), "the complete answer, not the cut one"
        assert provider.budgets == [(M36, 3000), (M36, 6000)]
        assert (outcome.finish_reason, outcome.max_tokens, outcome.length_retries) == ("stop", 6000, 1)
        assert outcome.reasoning_tokens == 3992 and not outcome.truncated
        assert _units(tracker) == {(M36, 0): 2}, "the retry is a request, and is paid for"
        assert M36 not in tracker._models_in_cooldown(), "a verbose model is not a sick one"

    def test_cut_off_twice_is_a_model_failure_without_cooling(self, tracker, monkeypatch, streamed):
        provider = _install(monkeypatch, Provider(default=LOOPS))

        raised, _ = run_virtual(_captured(_blocking(tracker, streamed)))

        assert isinstance(raised, CompletionTruncated)
        failure = classify_failure(raised)
        assert failure.kind is FailureKind.MODEL_FAILED and failure.another_model_helps
        assert not failure.refundable, "both requests were served"
        assert provider.budgets == [(M36, 3000), (M36, 6000)], "one retry, never a third request"
        assert _units(tracker) == {(M36, 0): 2}
        assert tracker._models_in_cooldown() == set()

    def test_cut_off_while_thinking_gets_room_not_a_same_size_retry(self, tracker, monkeypatch, streamed):
        """An empty reply at the limit is a budget problem: it used to be re-asked at 3,000, twice."""
        provider = _install(monkeypatch, Provider(default=Model(think=3500, answer=800)))

        text, _ = run_virtual(_blocking(tracker, streamed))

        assert text == _text(800)
        assert provider.budgets == [(M36, 3000), (M36, 6000)]

    def test_no_retry_when_the_slot_cannot_pay(self, tracker, monkeypatch, streamed):
        provider = _install(monkeypatch, Provider(default=Model(think=2500, answer=1500)))

        async def cannot_pay(model, api_key):
            return False

        monkeypatch.setattr(wrapper, "_charge_retry", cannot_pay)
        raised, _ = run_virtual(_captured(_blocking(tracker, streamed)))

        assert isinstance(raised, CompletionTruncated)
        assert provider.budgets == [(M36, 3000)]

    def test_a_reply_that_fits_is_untouched(self, tracker, monkeypatch, streamed):
        provider = _install(monkeypatch, Provider(default=Model(think=1200, answer=900)))
        outcome = wrapper.ProseOutcome()

        text, _ = run_virtual(_blocking(tracker, streamed, outcome=outcome))

        assert text == _text(900)
        assert provider.budgets == [(M36, 3000)]
        assert (outcome.finish_reason, outcome.length_retries) == ("stop", 0)
        assert _units(tracker) == {(M36, 0): 1}

    def test_finish_reasons_reach_the_usage_log_and_the_node_perf(self, tracker, monkeypatch, streamed):
        _install(monkeypatch, Provider(default=Model(think=2500, answer=1500)))
        metrics.install_llm_usage_tap()
        usage = []

        class Grab(logging.Filter):
            def filter(self, record):
                if record.msg == metrics.LLM_USAGE_MESSAGE:
                    usage.append(record)
                return True

        grab = Grab()
        logging.getLogger(metrics.LLM_USAGE_LOGGER).addFilter(grab)

        async def in_node():
            with metrics.node_scope("answer_synthesizer") as scope:
                await _blocking(tracker, streamed)
            return scope

        try:
            scope, _ = run_virtual(in_node())
        finally:
            logging.getLogger(metrics.LLM_USAGE_LOGGER).removeFilter(grab)

        assert [(u.finish_reason, u.max_tokens, u.completion_tokens, u.reasoning_tokens)
                for u in usage] == [("length", 3000, 3000, 2500), ("stop", 6000, 4000, 2500)]
        assert scope["finish_reasons"] == ["length", "stop"]


def test_perf_block_carries_the_finish_reasons():
    from src.workflow.orchestrator import _attach_perf

    with metrics.node_scope("answer_synthesizer") as scope:
        metrics.record_llm_call(M36, "answer_synthesizer", 5000, 2996, 12000.0, finish_reason="length")
        metrics.record_llm_call(M36, "answer_synthesizer", 5000, 4810, 20000.0, finish_reason="stop")
    delta = _attach_perf({"agent_trace": [{"agent": "answer_synthesizer"}]}, scope, "answer_synthesizer")
    perf = delta["agent_trace"][-1]["perf"]
    assert perf["finish_reasons"] == ["length", "stop"]
    assert perf["llm_calls"] == 2 and perf["completion_tokens"] == 2996 + 4810


# ─── /query/stream: the tokens are already on the client's screen ────────────


class TestStreamPath:
    def test_cut_after_text_is_returned_and_flagged(self, tracker, monkeypatch):
        provider = _install(monkeypatch, Provider(default=Model(think=2000, answer=2000)))
        tokens, outcome = [], wrapper.ProseOutcome()

        text, _ = run_virtual(_sse(tracker, tokens, outcome=outcome))

        assert text == _text(1000) == "".join(tokens), "what the client saw is what is returned"
        assert outcome.truncated and outcome.finish_reason == "length"
        assert outcome.length_retries == 0, "no silent retry once text has been sent"
        assert provider.budgets == [(M36, 3000)]
        assert _units(tracker) == {(M36, 0): 1}

    def test_cut_before_any_text_is_retried_invisibly(self, tracker, monkeypatch):
        """L6 live: 2,880 of 2,996 tokens thinking, nothing worth showing."""
        provider = _install(monkeypatch, Provider(default=Model(think=3200, answer=1600)))
        tokens, outcome = [], wrapper.ProseOutcome()

        text, _ = run_virtual(_sse(tracker, tokens, outcome=outcome))

        assert text == _text(1600) == "".join(tokens)
        assert provider.budgets == [(M36, 3000), (M36, 6000)]
        assert not outcome.truncated and outcome.length_retries == 1
        assert _units(tracker) == {(M36, 0): 2}

    def test_cut_before_any_text_twice_raises(self, tracker, monkeypatch):
        provider = _install(monkeypatch, Provider(default=Model(think=10**6, answer=10)))
        raised, _ = run_virtual(_captured(_sse(tracker, [])))
        assert isinstance(raised, CompletionTruncated)
        assert provider.budgets == [(M36, 3000), (M36, 6000)]
        assert tracker._models_in_cooldown() == set()

    def test_clean_stream_reports_stop(self, tracker, monkeypatch):
        _install(monkeypatch, Provider(default=Model(think=1000, answer=500)))
        outcome = wrapper.ProseOutcome()
        run_virtual(_sse(tracker, [], outcome=outcome))
        assert outcome.finish_reason == "stop" and not outcome.truncated


# ─── The synthesizer ──────────────────────────────────────────────────────────


def _state():
    return {
        "original_query": "What was FY2023 revenue?",
        "current_query": "What was FY2023 revenue?",
        "query_type": "financial",
        "expanded_context": [{
            "chunk_id": "c1", "source_file": "fin.txt", "page_number": 1,
            "text": "Total revenue for FY2023 was $452.8 million.",
        }],
        "numerical_registry": {},
        "inconsistencies": [],
    }


class TestBudget:
    def test_default_and_override(self, monkeypatch):
        monkeypatch.delenv("SYNTHESIS_MAX_TOKENS", raising=False)
        assert synth.synthesis_max_tokens() == 8192
        monkeypatch.setenv("SYNTHESIS_MAX_TOKENS", "12000")
        assert synth.synthesis_max_tokens() == 12000
        monkeypatch.setenv("SYNTHESIS_MAX_TOKENS", "lots")
        assert synth.synthesis_max_tokens() == 8192
        monkeypatch.setenv("SYNTHESIS_MAX_TOKENS", "50")
        assert synth.synthesis_max_tokens() == 1000
        monkeypatch.setenv("SYNTHESIS_MAX_TOKENS", "10000000")
        assert synth.synthesis_max_tokens() == 65_536

    def test_default_holds_the_largest_measured_need(self):
        """mh_05 live: 3,992 thinking + 1,818 answer."""
        assert synth.DEFAULT_SYNTHESIS_MAX_TOKENS >= 1.4 * (3992 + 1818)

    def test_first_token_deadline_at_the_new_budget(self):
        assert wrapper.first_token_deadline(8192) == pytest.approx(2 + 8192 / 150)
        # The healthy live call that took 37.9s to start fits; at 3,000 it would not have.
        assert wrapper.first_token_deadline(8192) > 37.9 > wrapper.first_token_deadline(3000)
        # The slow top rung that took 71.2s to start is still abandoned.
        assert wrapper.first_token_deadline(8192) < 71.2
        # The length retry's own deadline follows its budget, under the total.
        assert wrapper.first_token_deadline(16384) == pytest.approx(2 + 16384 / 150)
        assert wrapper.first_token_deadline(16384) < wrapper.PROSE_TIMEOUT_SECONDS


@pytest.mark.parametrize("streamed", [False, True], ids=["query", "stream"])
def test_slow_thinker_that_needs_the_budget_is_not_abandoned(tracker, monkeypatch, streamed):
    """mh_05 live on gemini-3.6-flash: 3,992 thinking tokens, first text at 37.9s."""
    provider = _install(monkeypatch, Provider({M38: Model(think=3992, answer=1818, rate=110.0)}))
    if streamed:
        monkeypatch.setattr(synth, "_token_sink", lambda: (lambda _event: None))

    result, wall = run_virtual(synth.synthesize_answer(_state()))

    assert result["agent_trace"][0]["model"] == M38
    assert result["generated_answer"] == _text(1818)
    assert provider.budgets == [(M38, 8192)]
    assert 37.0 < wall < 45.0


def test_query_path_retries_with_more_room_and_traces_it(tracker, monkeypatch):
    provider = _install(monkeypatch, Provider({M38: Model(think=7000, answer=1800)}))

    result, _ = run_virtual(synth.synthesize_answer(_state()))

    trace = result["agent_trace"][0]
    assert result["generated_answer"] == _text(1800)
    assert result["answer_truncated"] is False
    assert provider.budgets == [(M38, 8192), (M38, 16384)]
    assert (trace["finish_reason"], trace["max_tokens"], trace["length_retries"]) == ("stop", 16384, 1)
    assert trace["answer_truncated"] is False
    assert _units(tracker) == {(M38, 0): 2}


def test_query_path_cut_twice_tries_the_next_rung_for_this_request_only(tracker, monkeypatch):
    provider = _install(monkeypatch, Provider({M38: LOOPS, M37: Model(think=1500, answer=1200)}))

    result, _ = run_virtual(synth.synthesize_answer(_state()))

    assert result["agent_trace"][0]["model"] == M37
    assert result["generated_answer"] == _text(1200)
    assert provider.budgets == [(M38, 8192), (M38, 16384), (M37, 8192)]
    assert _units(tracker) == {(M38, 0): 2, (M37, 0): 1}
    assert tracker._models_in_cooldown() == set(), "nothing is cooled for being verbose"


def test_stream_path_flags_a_cut_answer(tracker, monkeypatch):
    provider = _install(monkeypatch, Provider({M38: Model(think=6000, answer=4000)}))
    events = []
    monkeypatch.setattr(synth, "_token_sink", lambda: events.append)

    result, _ = run_virtual(synth.synthesize_answer(_state()))

    shown = "".join(e["text"] for e in events if e["type"] == "token")
    assert result["generated_answer"] == shown == _text(8192 - 6000)
    assert result["answer_truncated"] is True
    trace = result["agent_trace"][0]
    assert (trace["finish_reason"], trace["answer_truncated"], trace["max_tokens"]) == ("length", True, 8192)
    assert provider.budgets == [(M38, 8192)]
    assert not any(e["type"] == "answer_reset" for e in events)


def test_failed_synthesis_and_refusal_are_never_truncated(tracker, monkeypatch):
    refused = run_virtual(synth.synthesize_answer({**_state(), "force_refusal": True}))[0]
    assert refused["answer_truncated"] is False


# ─── Verification and the response ───────────────────────────────────────────


class TestVerdict:
    def test_cut_answer_is_never_passed(self):
        from src.agents.hallucination_validator import TRUNCATED_ANSWER_FLAG, _mark_truncation

        cut = {"answer_truncated": True}
        assert _mark_truncation(cut, "passed", []) == ("warning", [TRUNCATED_ANSWER_FLAG])
        assert _mark_truncation(cut, "failed", ["x"]) == ("failed", ["x", TRUNCATED_ANSWER_FLAG])
        assert _mark_truncation({}, "passed", ["x"]) == ("passed", ["x"])

    @pytest.mark.asyncio
    async def test_flag_reaches_state_stage_and_response_on_the_stream(self, monkeypatch):
        from api.routes.query import _build_response
        from src.workflow.orchestrator import stream_query
        from tests.test_verification_flow import GOOD_ANSWER, QUESTION, _patch_pipeline

        cut = GOOD_ANSWER[:-40]

        async def cut_stream(**kwargs):
            kwargs["on_token"](cut)
            kwargs["outcome"].finish_reason = "length"
            return cut

        app, _ = _patch_pipeline(monkeypatch, [GOOD_ANSWER], stream=cut_stream)
        events = [e async for e in stream_query(app, QUESTION, deal_id="d1", session_id="s-cut")]

        name, final = events[-1]
        assert name == "result"
        assert final["generated_answer"] == cut
        assert final["answer_truncated"] is True
        assert final["validation_status"] == "warning", "the same answer uncut is 'passed'"
        assert final["hallucination_flags"][-1].startswith("The answer was cut off")
        stage = next(p for n, p in events if n == "stage" and p["agent"] == "answer_synthesizer")
        assert stage["detail"]["answer_truncated"] is True
        assert "cut off" in stage["summary"]

        response = _build_response(final, "s-cut")
        assert response.answer_truncated is True
        assert response.validation_status == "warning"

    @pytest.mark.asyncio
    async def test_uncut_answer_is_unchanged(self, monkeypatch):
        from api.routes.query import _build_response
        from src.workflow.orchestrator import run_query
        from tests.test_verification_flow import GOOD_ANSWER, QUESTION, _patch_pipeline

        app, _ = _patch_pipeline(monkeypatch, [GOOD_ANSWER])
        result = await run_query(app, QUESTION, deal_id="d1", session_id="s-uncut")

        assert result["answer_truncated"] is False
        assert result["validation_status"] == "passed"
        assert _build_response(result, "s-uncut").answer_truncated is False
