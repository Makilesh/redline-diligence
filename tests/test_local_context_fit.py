"""
A synthesis context too long for the local model's window is cut to fit and answered,
not failed (Decision 113).

Decision 106 made Ollama refuse a prompt longer than the window (counted with the
model's own tokenizer) instead of cutting its instructions off; the run then ended as
`synthesis_failed`. A full context (MAX_CONTEXT_CHARS, ~16.7k tokens) never fits the
default 16,384 window. The refusal states the prompt's real size, so the context is cut
by what did not fit, measured on that prompt, leaving room for the reply, and the same
model is asked again — best-ranked evidence first, as everywhere.
"""
from __future__ import annotations

import asyncio
import re

import litellm
import pytest

import src.agents.answer_synthesizer as synth
import src.llm.litellm_wrapper as wrapper
from src.llm.failures import LocalContextExceeded, context_refusal_sizes
from src.llm.model_registry import LOCAL_MODEL
import src.llm.budget_tracker as bt
from tests.test_local_context_window import REFUSAL, _Resp

CITED = "Revenue was $452.8 million [📄 fin_00.txt | p.1]."


def refusal(prompt_tokens: int, window: int = 16_384):
    message = (f'OllamaException - {{"error":"request ({prompt_tokens} tokens) exceeds the available '
               f'context size ({window} tokens), try increasing it","type":"exceed_context_size_error"}}')
    return litellm.APIConnectionError(message=message, llm_provider="ollama", model="qwen2.5:14b")


class Script:
    """litellm.acompletion that answers each request with the next scripted reply."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    async def __call__(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Resp(reply)


@pytest.fixture
def no_keys(monkeypatch):
    """No cloud key: the ladder's only rung is the local model."""
    t = bt.BudgetTracker.__new__(bt.BudgetTracker)
    t._api_keys = []
    t._rate_limiters = {}
    t._is_mock = True
    t._mock_budgets = {}
    monkeypatch.setattr(bt.BudgetTracker, "_instance", t)
    for name in ("PROSE_FIRST_TOKEN_TIMEOUT_SECONDS", "SYNTHESIS_MODEL_PIN", "SYNTHESIS_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    return t


@pytest.fixture
def ollama(monkeypatch):
    def install(*replies):
        script = Script(*replies)
        monkeypatch.setattr(wrapper.litellm, "acompletion", script)
        monkeypatch.setattr(wrapper, "RETRY_BACKOFF_SECONDS", 0.0)
        monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
        return script
    return install


# 40 passages of 2,000 characters: 80,000, over MAX_CONTEXT_CHARS, as a large deal's are.
CHUNKS = [
    {"chunk_id": f"c{i:02d}", "source_file": f"fin_{i:02d}.txt", "page_number": 1,
     "text": f"Passage {i:02d}. " + "Revenue for FY2023 was $452.8 million and EBITDA $92.8 million. " * 31}
    for i in range(40)
]


def _state():
    return {"original_query": "What was FY2023 revenue?", "current_query": "What was FY2023 revenue?",
            "query_type": "financial", "expanded_context": CHUNKS, "numerical_registry": {},
            "inconsistencies": []}


def _prompt(request) -> str:
    return "".join(m["content"] for m in request["messages"])


def _passages(request) -> list[str]:
    return re.findall(r"Passage (\d\d)\.", _prompt(request))


def test_the_refusal_counts_are_read():
    assert context_refusal_sizes(Exception(REFUSAL)) == (5507, 4096)
    assert context_refusal_sizes(refusal(20_000)) == (20_000, 16_384)
    assert context_refusal_sizes(Exception('{"n_prompt_tokens": 9000, "n_ctx": 8192}')) == (9000, 8192)
    assert context_refusal_sizes(Exception("connection refused")) is None


def test_the_budget_leaves_room_for_the_reply():
    # 20,000 tokens for 80,000 characters (4 a token) in a 16,384 window with an
    # 8,192 budget: a quarter of the window, 4,096, is kept for the reply, so
    # 7,712 tokens go — 30,848 characters, plus a tenth.
    assert synth.fitted_context_chars(80_000, 20_000, 16_384, 64_000, 8_192) == 64_000 - 33_932 - 1
    # The output budget, when smaller than a quarter of the window: 18,000 tokens
    # against a 15,384 target, 2,616 over at 4 characters a token, plus a tenth.
    assert synth.fitted_context_chars(72_000, 18_000, 16_384, 64_000, 1_000) == 64_000 - 11_510 - 1


def test_a_context_too_long_for_the_window_is_cut_and_answered(ollama, no_keys):
    script = ollama(refusal(20_000), CITED)
    result = asyncio.run(synth.synthesize_answer(_state()))

    assert result["generated_answer"] == CITED
    first, second = script.requests
    assert second["model"] == first["model"] == LOCAL_MODEL
    # The best-ranked passages are the ones kept, in order.
    kept, shown = _passages(second), _passages(first)
    assert 0 < len(kept) < len(shown) and kept == shown[: len(kept)]
    [trace] = result["agent_trace"]
    [fit] = trace["context_fits"]
    assert (fit["window"], fit["prompt_tokens"]) == (16_384, 20_000)
    assert fit["fitted_chars"] == synth.fitted_context_chars(
        len(synth.ANSWER_SYNTHESIZER_SYSTEM_PROMPT) + len(first["messages"][-1]["content"]),
        20_000, 16_384, fit["context_chars"], synth.synthesis_max_tokens())
    # What was cut is what the reply needs and the refusal said did not fit.
    assert len(_prompt(second)) <= len(_prompt(first)) - (20_000 - 12_288) * len(_prompt(first)) / 20_000


def test_the_reader_is_told_what_a_fitted_context_left_out(ollama, no_keys):
    # Decision 118: the counts reach the result, not only the admin trace.
    script = ollama(refusal(20_000), CITED)
    result = asyncio.run(synth.synthesize_answer(_state()))
    first, second = script.requests
    cut = result["evidence_cut"]
    assert cut["passages_used"] == len(_passages(second)) < cut["passages_available"] == len(_passages(first))
    assert 0 < cut["chars_used"] < cut["chars_available"] <= synth.MAX_CONTEXT_CHARS
    assert cut == synth.evidence_cut(CHUNKS, result["agent_trace"][0]["context_fits"][0]["fitted_chars"])


def test_an_answer_from_the_full_context_reports_no_cut(ollama, no_keys):
    ollama(CITED)
    assert asyncio.run(synth.synthesize_answer(_state()))["evidence_cut"] is None
    ollama(refusal(200_000))  # failed: nothing was answered, so nothing was left out of an answer
    assert asyncio.run(synth.synthesize_answer(_state()))["evidence_cut"] is None


def test_a_second_refusal_cuts_again_and_a_third_fails_with_the_count(ollama, no_keys):
    script = ollama(refusal(20_000), refusal(13_000), CITED)
    result = asyncio.run(synth.synthesize_answer(_state()))
    assert result["generated_answer"] == CITED
    assert len(script.requests) == 3
    assert len(result["agent_trace"][0]["context_fits"]) == 2
    lengths = [len(_prompt(r)) for r in script.requests]
    assert lengths[0] > lengths[1] > lengths[2]

    script = ollama(refusal(20_000), refusal(13_000), refusal(12_500))
    result = asyncio.run(synth.synthesize_answer(_state()))
    assert result["generated_answer"] == synth.SYNTHESIS_FAILED_ANSWER
    assert len(script.requests) == 3
    [trace] = result["agent_trace"]
    assert "12500 tokens" in trace["error"] and len(trace["context_fits"]) == 2


def test_a_window_too_small_for_any_evidence_still_fails_at_once(ollama, no_keys):
    # 200,000 tokens: what would be left is under MIN_FITTED_CONTEXT_CHARS.
    script = ollama(refusal(200_000))
    result = asyncio.run(synth.synthesize_answer(_state()))
    assert result["generated_answer"] == synth.SYNTHESIS_FAILED_ANSWER
    assert len(script.requests) == 1
    assert "context_fits" not in result["agent_trace"][0]


def test_the_streamed_path_raises_the_same_error_with_its_counts(ollama):
    ollama(refusal(20_000))
    with pytest.raises(LocalContextExceeded) as caught:
        asyncio.run(wrapper.stream_prose_agent(
            system_prompt="Cite.", user_prompt="Q?", model=LOCAL_MODEL, on_token=lambda _t: None,
            max_tokens=500, agent="answer_synthesizer"))
    assert (caught.value.prompt_tokens, caught.value.window) == (20_000, 16_384)


def test_streamed_synthesis_is_fitted_too(ollama, no_keys, monkeypatch):
    tokens: list[str] = []
    monkeypatch.setattr(synth, "_token_sink", lambda: lambda event: tokens.append(event))

    class _Stream:
        def __init__(self, text):
            self.text = text

        def __aiter__(self):
            async def gen():
                delta = type("D", (), {"content": self.text})()
                choice = type("C", (), {"delta": delta, "finish_reason": "stop"})()
                yield type("K", (), {"choices": [choice], "usage": None})()
            return gen()

    calls: list[dict] = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise refusal(20_000)
        return _Stream(CITED)

    monkeypatch.setattr(wrapper.litellm, "acompletion", acompletion)
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    result = asyncio.run(synth.synthesize_answer(_state()))
    assert result["generated_answer"] == CITED
    assert len(calls) == 2 and len(_prompt(calls[1])) < len(_prompt(calls[0]))
