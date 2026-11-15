"""
A request that gives up on a model also gives up on the ids the provider
serves with it (model_registry.SERVED_BY, Decision 69).

gemini-3.7-flash is served by gemini-3.8-flash and gemini-3.5-flash by
gemini-3.6-flash. The process-wide cooldown already treated each pair as one
backend (Decision 61), but the request-scoped skip did not: after 3.8-flash
was cut off twice, or returned an uncited draft, the same request asked
3.7-flash — the same model — next.
"""

import asyncio

import pytest

import src.agents.answer_synthesizer as synth
import src.llm.budget_tracker as bt
import src.llm.litellm_wrapper as wrapper
from src.llm.model_registry import SYNTHESIS_LADDER
from tests.fixtures.virtual_time import run_virtual

K0, K1 = "A" * 40, "B" * 40
M38, M37 = "gemini/gemini-3.8-flash", "gemini/gemini-3.7-flash"
M36, M35 = "gemini/gemini-3.6-flash", "gemini/gemini-3.5-flash"
M3P = "gemini/gemini-3-flash-preview"

CITED = "Total revenue for FY2023 was $452.8 million [📄 fin.txt | p.1]."
UNCITED = "Total revenue for FY2023 was $452.8 million."


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


def test_the_ladder_still_has_the_aliases_this_file_is_about():
    assert SYNTHESIS_LADDER[:4] == [M38, M37, M36, M35]


# ─── The tracker ──────────────────────────────────────────────────────────────


class TestSkipCoversTheBackend:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("skipped, next_model", [
        (M38, M36),   # the target: its alias goes with it
        (M37, M36),   # the alias: the target goes with it
        (M36, M38),   # a lower backend: the top rung is untouched
    ])
    async def test_a_skip_takes_every_id_of_the_model(self, tracker, skipped, next_model):
        async def request():
            tracker.skip_model_for_request(skipped)
            return (await tracker.get_model_for_synthesis()).model

        assert await asyncio.create_task(request()) == next_model

    @pytest.mark.asyncio
    async def test_a_model_without_aliases_is_skipped_alone(self, tracker):
        async def request():
            tracker.skip_model_for_request(M3P)
            return bt._models_skipped_for_request()

        assert await asyncio.create_task(request()) == frozenset({M3P})

    @pytest.mark.asyncio
    async def test_other_requests_still_get_the_top_rung(self, tracker):
        async def request_a():
            tracker.skip_model_for_request(M38)
            return (await tracker.get_model_for_synthesis()).model

        async def request_b():
            return (await tracker.get_model_for_synthesis()).model

        assert await asyncio.create_task(request_a()) == M36
        assert await asyncio.create_task(request_b()) == M38
        assert tracker._models_in_cooldown() == set(), "a skip is not a cooldown"

    @pytest.mark.asyncio
    async def test_the_skip_outlives_a_cooldown_another_request_cleared(self, tracker):
        """
        The wrapper cools 3.8 and 3.7 together on a 503, but one success by any
        other request clears that. The failing request must still not ask 3.7.
        """
        async def request():
            tracker.report_model_unavailable(M38)       # the wrapper saw the 503
            tracker.skip_model_for_request(M38)         # the ladder moves on
            tracker.note_model_healthy(M38)             # someone else's call succeeded
            return (await tracker.get_model_for_synthesis()).model

        assert await asyncio.create_task(request()) == M36


# ─── The synthesizer's ladder, end to end ────────────────────────────────────


def _chunk(content, finish=None, usage=None):
    c = type("Chunk", (), {})()
    delta = type("D", (), {"content": content})()
    c.choices = [type("C", (), {"delta": delta, "finish_reason": finish})()]
    c.usage = usage
    return c


class Provider:
    """Scripted streaming litellm.acompletion: per model, a reply kind."""

    def __init__(self, kinds):
        self.kinds = kinds
        self.models: list[str] = []

    async def __call__(self, **kwargs):
        model = kwargs["model"]
        self.models.append(model)
        kind = self.kinds.get(model, "good")
        assert kwargs.get("stream"), "synthesis streams on both routes"
        return self._stream(kind, kwargs["max_tokens"])

    async def _stream(self, kind, budget):
        await asyncio.sleep(1.0)
        if kind == "loops":            # writes until it is stopped, at any budget
            yield _chunk("Revenue was " * 50)
            yield _chunk(None, "length")
        elif kind == "uncited":
            yield _chunk(UNCITED)
            yield _chunk(None, "stop")
        elif kind == "drops":          # the connection dies after some text
            yield _chunk("Total revenue ")
            raise ConnectionResetError("stream reset by peer")
        else:
            yield _chunk(CITED)
            yield _chunk(None, "stop")
        yield _chunk(None, None, {"prompt_tokens": 5000, "completion_tokens": 100})


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


def test_cut_off_twice_skips_the_alias_too(tracker, monkeypatch):
    """
    /query. Before: 3.8, 3.8 with twice the room, then 3.7 — the same model,
    cut off again for two more units. (On /query/stream a cut after text is
    returned flagged instead, so no rung is skipped there.)
    """
    provider = Provider({M38: "loops", M37: "good"})
    monkeypatch.setattr(wrapper.litellm, "acompletion", provider)

    result, _ = run_virtual(synth.synthesize_answer(_state()))

    assert provider.models == [M38, M38, M36]
    assert result["agent_trace"][0]["model"] == M36
    assert result["generated_answer"] == CITED


def test_an_uncited_draft_skips_the_alias_too(tracker, monkeypatch):
    """Before: 3.8 uncited, then 3.7 — a second draft from the same model."""
    provider = Provider({M38: "uncited", M37: "uncited"})
    monkeypatch.setattr(wrapper.litellm, "acompletion", provider)

    result, _ = run_virtual(synth.synthesize_answer(_state()))

    assert provider.models == [M38, M36]
    assert result["agent_trace"][0]["model"] == M36
    assert result["generated_answer"] == CITED


def test_a_dropped_stream_skips_the_alias_too(tracker, monkeypatch):
    provider = Provider({M38: "drops"})
    monkeypatch.setattr(wrapper.litellm, "acompletion", provider)
    events = []
    monkeypatch.setattr(synth, "_token_sink", lambda: events.append)

    result, _ = run_virtual(synth.synthesize_answer(_state()))

    assert provider.models == [M38, M36]
    assert result["generated_answer"] == CITED
    assert any(e["type"] == "answer_reset" for e in events)
