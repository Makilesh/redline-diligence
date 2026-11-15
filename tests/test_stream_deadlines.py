"""
What a hung, overloaded or slow model costs, in virtual time.

The provider double models LiteLLM 1.89.3's real behaviour on each path, which
is what made a hung model expensive in the first place:

  * blocking request — the HTTP read timeout is the request's `timeout`, so a
    model that never answers raises litellm.Timeout after exactly that long;
  * streamed request — LiteLLM's Gemini streaming call ignores `timeout` and
    uses its module-level client (read timeout litellm.request_timeout, 6000s
    by default); a thinking model sends no
    bytes until it has finished thinking (live probe, 2026-11-15: first chunk
    at 138.7s, every answer token within the next 0.8s).

Healthy timings come from the 2026-11-15 live run: synthesis started answering
4.3-5.7s after the request and finished in 5-14s; a 503 took 3-8s to arrive.
"""

import asyncio
from datetime import datetime, timezone

import litellm
import pytest

import src.agents.answer_synthesizer as synth
import src.llm.budget_tracker as bt
import src.llm.litellm_wrapper as wrapper
from src.llm.failures import FailureKind, ResponseDeadlineExceeded, classify_failure
from src.llm.model_registry import AGENT_LADDER, serving_model
from tests.fixtures.virtual_time import run_virtual

K0, K1 = "A" * 40, "B" * 40
M38, M37 = "gemini/gemini-3.8-flash", "gemini/gemini-3.7-flash"
M36, M35 = "gemini/gemini-3.6-flash", "gemini/gemini-3.5-flash"
PREVIEW = "gemini/gemini-3-flash-preview"
LITE, LITE2 = AGENT_LADDER[0], AGENT_LADDER[1]
LOCAL = "ollama/qwen2.5:14b"

ANSWER = "Total revenue for FY2023 was $452.8 million [fin.txt | p.1]. " * 8


class Backend:
    """How one served model behaves."""

    def __init__(self, kind="healthy", ttft=5.0, chunks=40, gap=0.05, fail_after=8.0,
                 text=ANSWER, prompt_tokens=1234):
        self.kind, self.ttft, self.chunks, self.gap = kind, ttft, chunks, gap
        self.fail_after, self.text, self.prompt_tokens = fail_after, text, prompt_tokens

    def pieces(self):
        n = max(1, self.chunks)
        size = max(1, -(-len(self.text) // n))
        return [self.text[i:i + size] for i in range(0, len(self.text), size)]


def _chunk(text=None, usage=None):
    delta = type("D", (), {"content": text})()
    chunk = type("Chunk", (), {})()
    chunk.choices = [type("C", (), {"delta": delta})()] if text is not None else []
    chunk.usage = usage
    return chunk


class Provider:
    """Scripted litellm.acompletion with LiteLLM's timeout semantics (see module doc)."""

    LITELLM_STREAM_READ_TIMEOUT = 6000.0

    def __init__(self, backends=None, default=None):
        self.backends = backends or {}
        self.default = default or Backend()
        self.calls: list[tuple[str, int, bool]] = []
        self.requests: list[dict] = []

    def backend(self, model):
        return self.backends.get(model) or self.backends.get(serving_model(model)) or self.default

    async def __call__(self, **kwargs):
        model = kwargs["model"]
        key = {K0: 0, K1: 1}.get(kwargs.get("api_key"), -1)
        self.calls.append((model, key, bool(kwargs.get("stream"))))
        self.requests.append(kwargs)
        b = self.backend(model)
        if kwargs.get("stream"):
            return self._stream(b)
        if b.kind == "hung":
            await asyncio.sleep(kwargs["timeout"])
            raise litellm.Timeout(message=f"Timeout passed={kwargs['timeout']}",
                                  model=model, llm_provider="gemini")
        if b.kind == "503":
            await asyncio.sleep(b.fail_after)
            raise litellm.ServiceUnavailableError(message="503 high demand",
                                                  llm_provider="gemini", model=model)
        await asyncio.sleep(b.ttft + b.chunks * b.gap)
        message = type("M", (), {"content": b.text})()
        response = type("R", (), {})()
        response.choices = [type("C", (), {"message": message, "finish_reason": "stop"})()]
        response.usage = {"prompt_tokens": b.prompt_tokens}
        return response

    async def _stream(self, b):
        if b.kind == "hung":
            await asyncio.sleep(self.LITELLM_STREAM_READ_TIMEOUT)
            raise litellm.Timeout(message="Timeout passed=6000.0", model="m",
                                  llm_provider="gemini")
        if b.kind == "503":
            await asyncio.sleep(b.fail_after)
            raise litellm.ServiceUnavailableError(message="503 high demand",
                                                  llm_provider="gemini", model="m")
        await asyncio.sleep(b.ttft)
        pieces = b.pieces()
        for i, piece in enumerate(pieces):
            yield _chunk(piece)
            if b.kind == "stalls" and i == 2:
                await asyncio.sleep(10_000)
            await asyncio.sleep(b.gap)
        yield _chunk(usage={"prompt_tokens": b.prompt_tokens})


@pytest.fixture
def tracker(monkeypatch):
    t = bt.BudgetTracker.__new__(bt.BudgetTracker)
    t._api_keys = [K0, K1]
    t._rate_limiters = {}
    t._is_mock = True
    t._mock_budgets = {}
    monkeypatch.setattr(bt.BudgetTracker, "_instance", t)
    monkeypatch.setattr(wrapper, "RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.delenv("PROSE_FIRST_TOKEN_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SYNTHESIS_MODEL_PIN", raising=False)
    monkeypatch.delenv("SYNTHESIS_MAX_TOKENS", raising=False)
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
    """Returns the coroutine's result, or the exception it raised."""
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        return e


async def _prose(t, model, streamed, tokens=None, max_tokens=3000):
    """One synthesis request on (model, key 0), on either path."""
    slot = t._slot(0, model)
    await t._try_consume(slot)          # what selection debits
    if streamed:
        return await wrapper.stream_prose_agent(
            "s", "u", model=model, on_token=(tokens.append if tokens is not None else (lambda _t: None)),
            api_key=K0, max_tokens=max_tokens)
    return await wrapper.call_prose_agent(
        "s", "u", model=model, api_key=K0, max_tokens=max_tokens, stream_with_deadline=True)


# ─── The deadline itself ──────────────────────────────────────────────────────


class TestFirstTokenDeadline:
    def test_derived_from_the_thinking_budget(self):
        assert wrapper.first_token_deadline(3000) == pytest.approx(22.0)
        # More room to think means more time before the first token.
        assert wrapper.first_token_deadline(6000) == pytest.approx(42.0)
        assert wrapper.first_token_deadline(100) == wrapper.FIRST_TOKEN_FLOOR_SECONDS
        assert wrapper.first_token_deadline(10**6) == wrapper.PROSE_TIMEOUT_SECONDS

    def test_covers_a_healthy_model_spending_its_whole_budget_thinking(self):
        """Slowest healthy synthesis rate in the live run: 214 tokens/s."""
        assert wrapper.first_token_deadline(3000) > 3000 / 214.0 + 2.0

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("PROSE_FIRST_TOKEN_TIMEOUT_SECONDS", "15")
        assert wrapper.first_token_deadline(3000) == 15.0
        monkeypatch.setenv("PROSE_FIRST_TOKEN_TIMEOUT_SECONDS", "nonsense")
        assert wrapper.first_token_deadline(3000) == pytest.approx(22.0)

    def test_a_missed_deadline_is_a_timeout(self):
        failure = classify_failure(ResponseDeadlineExceeded("m", "first_token", 22.0))
        assert failure.kind is FailureKind.TIMEOUT
        assert failure.another_model_helps
        assert not failure.refundable, "the provider admitted the request"


# ─── One request, each kind of model ──────────────────────────────────────────


@pytest.mark.parametrize("streamed", [False, True], ids=["blocking", "streamed"])
class TestOneRequest:
    def test_healthy_fast_answer_is_untouched(self, tracker, monkeypatch, streamed):
        provider = _install(monkeypatch, Provider(default=Backend(ttft=5.0, chunks=40)))
        tokens = []
        text, wall = run_virtual(_prose(tracker, M36, streamed, tokens))
        assert text == ANSWER
        assert wall == pytest.approx(5.0 + 40 * 0.05, abs=0.2)
        assert len(provider.calls) == 1 and provider.calls[0][2] is True
        assert _units(tracker) == {(M36, 0): 1}
        if streamed:
            assert "".join(tokens) == ANSWER

    def test_slow_but_streaming_answer_is_not_cut(self, tracker, monkeypatch, streamed):
        """18s of thinking, then 90s of answer: 108s, inside every deadline."""
        long_answer = ("Revenue was $452.8M [fin.txt | p.1]. " * 30)[:900]
        provider = _install(monkeypatch, Provider(
            default=Backend(ttft=18.0, chunks=900, gap=0.1, text=long_answer)))
        text, wall = run_virtual(_prose(tracker, M36, streamed))
        assert text == long_answer
        assert wall == pytest.approx(108.0, abs=0.5)
        assert len(provider.calls) == 1

    def test_hung_model_costs_the_first_token_deadline(self, tracker, monkeypatch, streamed):
        provider = _install(monkeypatch, Provider({M38: Backend("hung")}))
        raised, wall = run_virtual(_captured(_prose(tracker, M38, streamed)))
        assert isinstance(raised, ResponseDeadlineExceeded) and raised.phase == "first_token"
        assert wall == pytest.approx(wrapper.first_token_deadline(3000), abs=0.1)
        assert len(provider.calls) == 1, "a timeout is never retried on the same model"
        assert _units(tracker) == {(M38, 0): 1}, "admitted: the unit is kept"
        cooled = tracker._models_in_cooldown()
        assert M38 in cooled and M37 in cooled, "3.7-flash is served by the same model"

    def test_a_503_is_unchanged(self, tracker, monkeypatch, streamed):
        _install(monkeypatch, Provider({M36: Backend("503", fail_after=8.0)}))
        raised, wall = run_virtual(_captured(_prose(tracker, M36, streamed)))
        assert isinstance(raised, litellm.ServiceUnavailableError)
        assert wall == pytest.approx(8.0, abs=0.1)
        assert _units(tracker) == {}, "a 503 is refunded"
        assert {M36, M35} <= tracker._models_in_cooldown()

    def test_stall_after_text_is_an_interruption(self, tracker, monkeypatch, streamed):
        _install(monkeypatch, Provider(default=Backend("stalls", ttft=5.0, chunks=40)))
        raised, wall = run_virtual(_captured(_prose(tracker, M36, streamed)))
        assert isinstance(raised, wrapper.StreamInterrupted)
        assert isinstance(raised.cause, ResponseDeadlineExceeded) and raised.cause.phase == "stall"
        assert wall == pytest.approx(5.0 + 2 * 0.05 + wrapper.STREAM_STALL_SECONDS, abs=0.2)
        assert M36 in tracker._models_in_cooldown()

    def test_total_budget_is_enforced_on_the_stream(self, tracker, monkeypatch, streamed):
        """LiteLLM does not apply `timeout` to a Gemini stream; the wrapper must."""
        _install(monkeypatch, Provider(default=Backend(ttft=5.0, chunks=100, gap=15.0)))
        raised, wall = run_virtual(_captured(_prose(tracker, M36, streamed)))
        assert isinstance(raised, wrapper.StreamInterrupted)
        assert raised.cause.phase == "total"
        assert wall == pytest.approx(wrapper.PROSE_TIMEOUT_SECONDS, abs=0.1)

    def test_usage_is_requested_and_corrects_the_token_booking(self, tracker, monkeypatch, streamed):
        provider = _install(monkeypatch, Provider(default=Backend(prompt_tokens=1234)))
        run_virtual(_prose(tracker, M36, streamed))
        assert provider.requests[0]["stream_options"] == {"include_usage": True}
        window = tracker._token_windows[tracker._slot(0, M36)]
        assert [entry[1] for entry in window] == [1234]

    def test_local_model_has_no_first_token_deadline(self, tracker, monkeypatch, streamed):
        """No rung below it: a deadline there could only lose the answer."""
        _install(monkeypatch, Provider(default=Backend(ttft=40.0)))
        text, wall = run_virtual(_prose(tracker, LOCAL, streamed))
        assert text == ANSWER and wall > 40.0


def test_blocking_call_without_the_option_is_unchanged(tracker, monkeypatch):
    """Direct callers of call_prose_agent keep the blocking request and its timeout."""
    provider = _install(monkeypatch, Provider({M36: Backend("hung")}))
    raised, wall = run_virtual(_captured(
        wrapper.call_prose_agent("s", "u", model=M36, api_key=K0)))
    assert isinstance(raised, litellm.Timeout)
    assert wall == pytest.approx(wrapper.PROSE_TIMEOUT_SECONDS)
    assert provider.calls == [(M36, 0, False)]


# ─── The synthesizer, on the sequence the live run met ────────────────────────


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


def _live_sequence():
    """
    2026-11-15, first affected question: the 3.7/3.8 model hangs, the 3.6/3.5
    model answers 503 after ~8s, gemini-3-flash-preview answers in 8.45s.
    """
    return Provider({
        M38: Backend("hung"),
        M36: Backend("503", fail_after=8.0),
        PREVIEW: Backend(ttft=6.0, chunks=49, gap=0.05),
    })


@pytest.mark.parametrize("streamed", [False, True], ids=["blocking", "streamed"])
def test_first_affected_question_costs_seconds_not_minutes(tracker, monkeypatch, streamed):
    """
    Live: 144.6s in the synthesizer (120.2s of it one timeout), four calls.
    Now: one first-token deadline, one 503, the answer — and neither alias is
    asked, because its model has just failed.
    """
    provider = _install(monkeypatch, _live_sequence())
    sink = []
    if streamed:
        monkeypatch.setattr(synth, "_token_sink", lambda: sink.append)

    result, wall = run_virtual(synth.synthesize_answer(_state()))

    assert result["agent_trace"][0]["model"] == PREVIEW
    assert result["generated_answer"] == ANSWER
    assert [c[0] for c in provider.calls] == [M38, M36, PREVIEW]
    deadline = wrapper.first_token_deadline(synth.synthesis_max_tokens())
    assert wall == pytest.approx(deadline + 8.0 + 6.0 + 49 * 0.05, abs=0.5)
    # 56.6s of it is the first-token deadline at the 8,192-token budget (22s at
    # the old 3,000 — Decision 66 says why the deadline follows the budget).
    assert wall < 75.0
    assert _units(tracker) == {(M38, 0): 1, (PREVIEW, 0): 1}


def test_the_next_question_skips_the_sick_models_entirely(tracker, monkeypatch):
    provider = _install(monkeypatch, _live_sequence())
    run_virtual(synth.synthesize_answer(_state()))
    provider.calls.clear()

    result, wall = run_virtual(synth.synthesize_answer(_state()))

    assert [c[0] for c in provider.calls] == [PREVIEW]
    assert wall == pytest.approx(6.0 + 49 * 0.05, abs=0.2)
    assert result["agent_trace"][0]["model"] == PREVIEW


# ─── The structured (agent and judge) path ────────────────────────────────────


def test_a_hung_lite_model_costs_the_structured_timeout(tracker, monkeypatch):
    """Was 60s per hung structured call; ~3 such calls per query."""
    provider = _install(monkeypatch, Provider(
        {LITE: Backend("hung")},
        default=Backend(ttft=1.0, chunks=4, gap=0.05, text='{"ok": true}'),
    ))

    async def agent_call():
        choice = await tracker.get_model_for_agent()
        return await wrapper.call_structured_agent(
            system_prompt="s", user_prompt="u", model=choice.model, api_key=choice.api_key)

    result, wall = run_virtual(agent_call())
    assert result == {"ok": True}
    assert [c[0] for c in provider.calls] == [LITE, LITE2]
    assert wall == pytest.approx(wrapper.STRUCTURED_TIMEOUT_SECONDS + 1.2, abs=0.1)
    assert provider.requests[0]["timeout"] == wrapper.STRUCTURED_TIMEOUT_SECONDS


def test_the_local_model_keeps_its_longer_structured_timeout(tracker, monkeypatch):
    provider = _install(monkeypatch, Provider(default=Backend(text='{"ok": true}')))
    run_virtual(wrapper.call_structured_agent(system_prompt="s", user_prompt="u", model=LOCAL))
    assert provider.requests[0]["timeout"] == wrapper.LOCAL_STRUCTURED_TIMEOUT_SECONDS


def test_virtual_clock_is_not_wall_clock():
    """Guards the fixture: a two-minute sleep must not take two minutes."""
    started = datetime.now(timezone.utc)
    _, wall = run_virtual(asyncio.sleep(120))
    assert wall == pytest.approx(120.0)
    assert (datetime.now(timezone.utc) - started).total_seconds() < 5
