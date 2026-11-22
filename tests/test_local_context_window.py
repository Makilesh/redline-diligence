"""
Every call to the local Ollama model carries one context window, large enough for synthesis,
and a prompt that would not fit it is not sent (Decision 106).

Only the JSON calls set one (8,192). Ollama's own default (4,096 on a GPU under 24 GiB) applied
to every prose call, and a longer prompt is cut to its first 4 and last ~2,046 tokens: in the
end-to-end run of 2026-11-23 every local synthesis prompt (4,126–6,080 tokens) lost its
instructions, came back uncited and was refused, and the model reloaded whenever the window
changed between calls.
"""
from __future__ import annotations

import asyncio

import pytest

import src.agents.answer_synthesizer as synth
import src.llm.budget_tracker as bt
import src.llm.litellm_wrapper as wrapper
from src.llm.failures import FailureKind, LocalContextExceeded, classify_failure
from src.llm.model_registry import LOCAL_MODEL

CLOUD = "gemini/gemini-3.6-flash"
CITED = "Revenue was $452.8 million [📄 fin.txt | p.1]. "


class _Resp:
    def __init__(self, content):
        message = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": message, "finish_reason": "stop"})()]
        self.usage = None


class Recorder:
    """litellm.acompletion that records each request and returns `reply` (or raises it)."""

    def __init__(self, reply='{"ok": true}'):
        self.reply = reply
        self.requests: list[dict] = []

    async def __call__(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return _Resp(self.reply)


@pytest.fixture
def provider(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(wrapper.litellm, "acompletion", rec)
    monkeypatch.setattr(wrapper, "RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    return rec


@pytest.fixture
def no_keys(monkeypatch):
    t = bt.BudgetTracker.__new__(bt.BudgetTracker)
    t._api_keys = []
    t._rate_limiters = {}
    t._is_mock = True
    t._mock_budgets = {}
    monkeypatch.setattr(bt.BudgetTracker, "_instance", t)
    for name in ("PROSE_FIRST_TOKEN_TIMEOUT_SECONDS", "SYNTHESIS_MODEL_PIN", "SYNTHESIS_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    return t


def _structured(model):
    return wrapper._structured_request([{"role": "user", "content": "classify this"}], model, 0.0, 500, None, "agent")


def _prose(model, prompt="What was revenue?"):
    return wrapper.call_prose_agent(system_prompt="Cite every claim.", user_prompt=prompt, model=model,
                                    max_tokens=500, agent="answer_synthesizer")


def _stream(model, prompt="What was revenue?"):
    return wrapper.stream_prose_agent(system_prompt="Cite every claim.", user_prompt=prompt, model=model,
                                      on_token=lambda _t: None, max_tokens=500, agent="answer_synthesizer")


def test_every_local_call_sends_the_same_window(provider):
    asyncio.run(_structured(LOCAL_MODEL))
    provider.reply = CITED
    asyncio.run(_prose(LOCAL_MODEL))
    provider.reply = RuntimeError("stop after the request is recorded")
    with pytest.raises(RuntimeError):
        asyncio.run(_stream(LOCAL_MODEL))
    windows = [r.get("num_ctx") for r in provider.requests]
    assert len(windows) >= 3
    assert set(windows) == {wrapper.DEFAULT_LOCAL_NUM_CTX} == {16_384}


def test_a_cloud_call_sends_no_window(provider):
    provider.reply = CITED
    asyncio.run(_prose(CLOUD))
    assert "num_ctx" not in provider.requests[0]


@pytest.mark.parametrize("value, expected", [("8192", 8192), ("32768", 32768), ("lots", 16_384),
                                             ("100", 2_048), ("999999", 131_072)])
def test_the_window_follows_OLLAMA_NUM_CTX(monkeypatch, value, expected):
    monkeypatch.setenv("OLLAMA_NUM_CTX", value)
    assert wrapper.local_num_ctx() == expected


def test_a_prompt_that_would_not_fit_is_not_sent(provider, monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    long_prompt = "The aggregate merger consideration is approximately $696 million. " * 400  # ~6,000 tokens
    for call in (_prose, _stream):
        with pytest.raises(LocalContextExceeded, match="OLLAMA_NUM_CTX"):
            asyncio.run(call(LOCAL_MODEL, long_prompt))
    with pytest.raises(LocalContextExceeded):
        asyncio.run(wrapper._structured_request([{"role": "user", "content": long_prompt}],
                                                LOCAL_MODEL, 0.0, 500, None, "agent"))
    assert provider.requests == []
    # A cloud model's window is the provider's business: the same prompt is sent.
    provider.reply = CITED
    asyncio.run(_prose(CLOUD, long_prompt))
    assert len(provider.requests) == 1


def test_it_is_an_invalid_request_and_is_not_retried(provider, monkeypatch):
    assert classify_failure(LocalContextExceeded("x")).kind is FailureKind.REQUEST_INVALID
    monkeypatch.setenv("OLLAMA_NUM_CTX", "2048")
    # The local model is retried on transport failures (LOCAL_TRANSPORT_ATTEMPTS); this is not one.
    with pytest.raises(LocalContextExceeded):
        asyncio.run(wrapper._structured_on_slot("Classify.", "figure " * 3000, LOCAL_MODEL, 0.0, 500, None, "agent"))
    assert provider.requests == []


def test_the_estimate_errs_on_the_large_side():
    # qwen2.5 counted 5,532 and 9,079 tokens for prompts the embedder's tokenizer counts at
    # 4,618 and 7,729 (sample deal, 2026-11-23): the estimate must not fall below them.
    from pathlib import Path

    from src.utils.token_counter import count_tokens

    root = Path(__file__).resolve().parents[1]
    text = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted((root / "data" / "sample_deal").glob("*.txt")))
    for chars, qwen_tokens in ((24_000, 5_532), (40_000, 9_079)):
        prompt = ("IMPORTANT: begin your reply with the single word MARKER-ALPHA, then answer.\n\n" + text[:chars]
                  + "\n\nQuestion: what is the per-share merger consideration?")
        assert count_tokens(prompt) < qwen_tokens  # the embedder's tokenizer undercounts qwen's
        assert wrapper._estimated_prompt_tokens([{"role": "user", "content": prompt}]) >= qwen_tokens


def _state(chunks):
    return {"original_query": "What was FY2023 revenue?", "current_query": "What was FY2023 revenue?",
            "query_type": "financial", "expanded_context": chunks, "numerical_registry": {}, "inconsistencies": []}


def test_local_synthesis_sends_the_window_and_answers(provider, no_keys):
    provider.reply = CITED
    chunk = {"chunk_id": "c1", "source_file": "fin.txt", "page_number": 1,
             "text": "Total revenue for FY2023 was $452.8 million."}
    result = asyncio.run(synth.synthesize_answer(_state([chunk])))
    assert result["generated_answer"] != synth.SYNTHESIS_FAILED_ANSWER
    assert [r["model"] for r in provider.requests] == [LOCAL_MODEL]
    assert provider.requests[0]["num_ctx"] == 16_384


def test_local_synthesis_of_an_oversized_context_fails_at_once(provider, no_keys, monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    chunks = [{"chunk_id": f"c{i}", "source_file": "fin.txt", "page_number": i,
               "text": "Total revenue for FY2023 was $452.8 million, up from $387.1 million. " * 30}
              for i in range(12)]
    result = asyncio.run(synth.synthesize_answer(_state(chunks)))
    assert result["generated_answer"] == synth.SYNTHESIS_FAILED_ANSWER
    assert provider.requests == []  # not sent, and not retried
    [trace] = result["agent_trace"]
    assert "OLLAMA_NUM_CTX" in trace["error"]
