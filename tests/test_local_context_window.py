"""
Every call to the local Ollama model carries one context window, and Ollama is asked to refuse
what does not fit it instead of truncating silently (Decision 106).

Only the JSON calls set a window (8,192). Ollama's own default (4,096 on a GPU under 24 GiB)
applied to every prose call, and a longer prompt is cut to its first 4 and last ~2,046 tokens:
in the end-to-end run of 2026-11-23 every local synthesis prompt (4,126–6,080 tokens) lost its
instructions, came back uncited and was refused, and the model reloaded whenever the window
changed. A reply running past the window was cut the same way (context shift). Ollama refuses
instead when asked — truncate=false, counted with the model's own tokenizer — and stops a reply
at the window with shift=false; LiteLLM must send both as top-level fields.
"""
from __future__ import annotations

import asyncio

import litellm
import pytest
from litellm.llms.ollama.completion.transformation import OllamaConfig

import src.agents.answer_synthesizer as synth
import src.llm.budget_tracker as bt
import src.llm.litellm_wrapper as wrapper
from src.llm.failures import FailureKind, LocalContextExceeded, classify_failure
from src.llm.model_registry import LOCAL_MODEL

CLOUD = "gemini/gemini-3.6-flash"
CITED = "Revenue was $452.8 million [📄 fin.txt | p.1]. "
# What LiteLLM raised for Ollama's refusal in the live check (2026-11-23).
REFUSAL = ('OllamaException - {"error":"{\\"error\\":{\\"code\\":400,\\"message\\":\\"request (5507 tokens) '
           'exceeds the available context size (4096 tokens), try increasing it\\",\\"type\\":'
           '\\"exceed_context_size_error\\",\\"n_prompt_tokens\\":5507,\\"n_ctx\\":4096}}"}')


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


def _refusal():
    return litellm.APIConnectionError(message=REFUSAL, llm_provider="ollama", model="qwen2.5:14b")


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


def _prose(model):
    return wrapper.call_prose_agent(system_prompt="Cite every claim.", user_prompt="What was revenue?", model=model,
                                    max_tokens=500, agent="answer_synthesizer")


def _stream(model):
    return wrapper.stream_prose_agent(system_prompt="Cite every claim.", user_prompt="What was revenue?", model=model,
                                      on_token=lambda _t: None, max_tokens=500, agent="answer_synthesizer")


def test_every_local_call_sends_one_window_and_asks_ollama_to_refuse(provider):
    asyncio.run(_structured(LOCAL_MODEL))
    provider.reply = CITED
    asyncio.run(_prose(LOCAL_MODEL))
    provider.reply = RuntimeError("stop after the request is recorded")
    with pytest.raises(RuntimeError):
        asyncio.run(_stream(LOCAL_MODEL))
    assert len(provider.requests) >= 3
    for request in provider.requests:
        assert (request["num_ctx"], request["truncate"], request["shift"]) == (16_384, False, False)


def test_a_cloud_call_sends_none_of_it(provider):
    provider.reply = CITED
    asyncio.run(_prose(CLOUD))
    assert not {"num_ctx", "truncate", "shift"} & set(provider.requests[0])


@pytest.mark.parametrize("value, expected", [("8192", 8192), ("32768", 32768), ("lots", 16_384),
                                             ("100", 2_048), ("999999", 131_072)])
def test_the_window_follows_OLLAMA_NUM_CTX(monkeypatch, value, expected):
    monkeypatch.setenv("OLLAMA_NUM_CTX", value)
    assert wrapper.local_num_ctx() == expected


def test_litellm_sends_the_flags_where_ollama_reads_them():
    # Inside `options` Ollama ignores them (measured: the prompt was truncated anyway).
    data = OllamaConfig().transform_request(
        "qwen2.5:14b", [{"role": "user", "content": "hello"}],
        {"num_ctx": 16_384, "truncate": False, "shift": False, "num_predict": 50}, {}, {})
    assert data["truncate"] is False and data["shift"] is False
    assert "truncate" not in data["options"] and "shift" not in data["options"]
    assert data["options"]["num_ctx"] == 16_384


def test_a_context_refusal_is_an_invalid_request_and_is_not_retried(provider):
    assert classify_failure(_refusal()).kind is FailureKind.REQUEST_INVALID  # not "connection"
    provider.reply = _refusal()
    with pytest.raises(LocalContextExceeded, match="OLLAMA_NUM_CTX"):
        asyncio.run(wrapper._structured_on_slot("Classify.", "figures", LOCAL_MODEL, 0.0, 500, None, "agent"))
    assert len(provider.requests) == 1  # local transport retries do not apply to it
    provider.requests.clear()
    with pytest.raises(LocalContextExceeded):
        asyncio.run(_prose(LOCAL_MODEL))
    assert len(provider.requests) == 1


def _state(chunks):
    return {"original_query": "What was FY2023 revenue?", "current_query": "What was FY2023 revenue?",
            "query_type": "financial", "expanded_context": chunks, "numerical_registry": {}, "inconsistencies": []}


CHUNK = {"chunk_id": "c1", "source_file": "fin.txt", "page_number": 1, "text": "Total revenue for FY2023 was $452.8 million."}


def test_local_synthesis_sends_the_window_and_answers(provider, no_keys):
    provider.reply = CITED
    result = asyncio.run(synth.synthesize_answer(_state([CHUNK])))
    assert result["generated_answer"] != synth.SYNTHESIS_FAILED_ANSWER
    [request] = provider.requests
    assert request["model"] == LOCAL_MODEL and request["num_ctx"] == 16_384 and request["truncate"] is False


def test_local_synthesis_that_does_not_fit_fails_at_once_with_the_reason(provider, no_keys):
    provider.reply = _refusal()
    result = asyncio.run(synth.synthesize_answer(_state([CHUNK])))
    assert result["generated_answer"] == synth.SYNTHESIS_FAILED_ANSWER
    assert len(provider.requests) == 1  # refused once, not retried
    [trace] = result["agent_trace"]
    assert "context window" in trace["error"] and "5507 tokens" in trace["error"]
