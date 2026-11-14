"""
Retrieval model work is batched, and batching changes nothing but the cost.

Three properties, none of which needs model weights:

- the cross-encoder scores pairs longest-first (so a batch is padded to its own
  longest member) and hands the scores back in the caller's order;
- half precision is a CUDA-only default with an off switch, and the activation
  runs in float32 either way;
- Agent 3 embeds every query of a decomposed question in one call, reranks every
  pass in one call, scores each candidate against the query that fetched it, and
  does not go back to Qdrant for payloads the search already returned.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import src.agents.retrieval_executor as rx
import src.vector_db.reranker as reranker
from tests.conftest import make_scored_point


class _FakeCrossEncoder:
    """Scores a pair by its passage length; records what predict was given."""

    def __init__(self, dtype=torch.float32):
        self.default_activation_function = torch.nn.Sigmoid()
        self.dtype = dtype
        self.calls: list[dict] = []
        self.activation_input_dtypes: list[torch.dtype] = []

    def predict(self, pairs, batch_size=32, show_progress_bar=None, activation_fct=None):
        self.calls.append({"pairs": [tuple(p) for p in pairs], "batch_size": batch_size})
        logits = torch.tensor([[len(p[1]) / 10.0 - 2.0] for p in pairs], dtype=self.dtype)
        out = activation_fct(logits)
        self.activation_input_dtypes.append(out.dtype)
        return np.asarray([row[0].item() for row in out], dtype=out.numpy().dtype)


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeCrossEncoder()
    monkeypatch.setattr(reranker, "_reranker_model", model)
    return model


class TestScorePairs:
    def test_scores_come_back_in_input_order(self, fake_model):
        pairs = [("q", "short"), ("q", "a much longer passage " * 5), ("q", "medium length")]

        scores = reranker._score_pairs(pairs)

        expected = [1 / (1 + np.exp(-(len(p) / 10.0 - 2.0))) for _, p in pairs]
        assert scores.dtype == np.float32
        assert scores.tolist() == pytest.approx(expected, abs=1e-6)

    def test_model_sees_pairs_longest_first(self, fake_model):
        """Padding is per batch, so length order is what bounds the wasted tokens."""
        pairs = [("q", "x" * n) for n in (5, 400, 50, 120, 9)]

        reranker._score_pairs(pairs)

        seen = [len(p[1]) for p in fake_model.calls[0]["pairs"]]
        assert seen == sorted(seen, reverse=True)
        assert fake_model.calls[0]["batch_size"] == reranker.RERANKER_BATCH_SIZE

    def test_query_length_counts_towards_the_order(self, fake_model):
        """Pairs of different queries are batched together; the pair is what is padded."""
        pairs = [("short q", "p" * 20), ("a very long sub-question " * 4, "p" * 20)]

        reranker._score_pairs(pairs)

        assert fake_model.calls[0]["pairs"][0][0].startswith("a very long")

    def test_activation_runs_in_float32_on_half_precision_logits(self, monkeypatch):
        """
        fp16 resolves ~0.0005 near 1.0; a sigmoid computed there ties the best
        chunks. The logits may be half precision, the scores may not.
        """
        model = _FakeCrossEncoder(dtype=torch.float16)
        monkeypatch.setattr(reranker, "_reranker_model", model)

        scores = reranker._score_pairs([("q", "x" * 61), ("q", "x" * 62)])

        assert model.activation_input_dtypes == [torch.float32]
        assert scores.dtype == np.float32


class TestAsyncWrappers:
    @pytest.mark.asyncio
    async def test_rerank_async_keeps_its_contract(self, fake_model):
        """One query, N passages in, N scores out in passage order."""
        passages = ["tiny", "x" * 300, "x" * 30]

        scores = await reranker.rerank_async("q", passages)

        assert len(scores) == 3
        assert int(np.argmax(scores)) == 1
        assert all(0.0 <= s <= 1.0 for s in scores)

    @pytest.mark.asyncio
    async def test_pairs_of_different_queries_are_one_model_call(self, fake_model):
        pairs = [("q1", "a" * 10), ("q2", "b" * 200), ("q1", "c" * 50), ("q3", "d" * 90)]

        scores = await reranker.rerank_pairs_async(pairs)

        assert len(fake_model.calls) == 1
        assert sorted(fake_model.calls[0]["pairs"]) == sorted(pairs)
        assert np.argsort(-scores).tolist() == [1, 3, 2, 0]

    @pytest.mark.asyncio
    async def test_empty_input_is_an_error(self, fake_model):
        with pytest.raises(ValueError):
            await reranker.rerank_pairs_async([])
        with pytest.raises(ValueError):
            await reranker.rerank_async("q", [])


class TestHalfPrecision:
    def test_never_on_cpu(self, monkeypatch):
        monkeypatch.setattr(reranker, "RERANKER_FP16", "1")
        assert reranker._use_half_precision("cpu") is False

    @pytest.mark.parametrize("setting,expected", [
        ("auto", True), ("1", True), ("0", False), ("false", False), ("off", False),
    ])
    def test_cuda_default_and_off_switch(self, monkeypatch, setting, expected):
        monkeypatch.setattr(reranker, "RERANKER_FP16", setting)
        assert reranker._use_half_precision("cuda") is expected

    @pytest.mark.parametrize("cuda,halved", [(True, True), (False, False)])
    def test_weights_are_converted_at_load(self, monkeypatch, cuda, halved):
        monkeypatch.setattr(reranker, "RERANKER_FP16", "auto")
        monkeypatch.setattr(reranker, "_reranker_model", None)
        monkeypatch.setattr(reranker.torch.cuda, "is_available", lambda: cuda)

        with patch.object(reranker, "CrossEncoder", return_value=MagicMock()) as ctor:
            model = reranker._get_reranker_model()

        assert ctor.call_args.kwargs["device"] == ("cuda" if cuda else "cpu")
        assert model.model.half.called is halved


# ==============================================================================
# Agent 3
# ==============================================================================


def _point(chunk_id: str, text: str):
    point = make_scored_point(chunk_id, 0.5)
    point.payload.update({"text": text, "source_file": "doc.txt", "section_heading": "S"})
    return point


class _Recorder:
    """Stubs every external call of the retrieval node and records them."""

    def __init__(self, monkeypatch, results_by_query: dict[str, tuple[list, list]]):
        self.embed_calls: list[list[str]] = []
        self.rerank_calls: list[list[tuple[str, str]]] = []
        self.search_calls: list[dict] = []
        self.results_by_query = results_by_query

        async def embed(texts):
            self.embed_calls.append(list(texts))
            return np.zeros((len(texts), 4), dtype=np.float32)

        async def search(**kwargs):
            self.search_calls.append(kwargs)
            return self.results_by_query.get(kwargs["query_text"], ([], []))

        async def rerank(pairs):
            self.rerank_calls.append(list(pairs))
            # Score = how many of the query's words the passage contains.
            return np.asarray(
                [sum(w in p for w in q.split()) / 10.0 for q, p in pairs], dtype=np.float32)

        async def expand(**kwargs):
            return list(kwargs["chunks"])

        monkeypatch.setattr(rx, "embed_texts_async", embed)
        monkeypatch.setattr(rx, "compute_sparse_bm25", lambda text: f"sparse:{text}")
        monkeypatch.setattr(rx, "hybrid_search", search)
        monkeypatch.setattr(rx, "rerank_pairs_async", rerank)
        monkeypatch.setattr(rx, "expand_context", expand)


def _state(**overrides) -> dict:
    state = {
        "current_query": "alpha beta", "query_type": "multi_hop", "parsed_intent": {},
        "deal_id": "deal-1", "include_pii": False, "rewrite_iteration": 0,
        "retrieval_config": {}, "extracted_filters": {"document_category": "financial"},
        "sub_questions": [],
    }
    state.update(overrides)
    return state


class TestBatchedPasses:
    @pytest.mark.asyncio
    async def test_decomposed_question_is_one_embed_and_one_rerank(self, monkeypatch):
        rec = _Recorder(monkeypatch, {
            "alpha beta": ([_point("c1", "alpha"), _point("c2", "alpha beta")], [_point("c3", "x")]),
            "gamma": ([_point("c4", "gamma")], [_point("c1", "alpha")]),
            "delta": ([], [_point("c5", "delta delta")]),
        })

        out = await rx.retrieval_executor_node(_state(sub_questions=["gamma", "delta", "  "]))

        assert rec.embed_calls == [["alpha beta", "gamma", "delta"]]
        assert len(rec.rerank_calls) == 1
        assert len(rec.search_calls) == 3
        assert [p["candidates"] for p in out["retrieval_passes"]] == [3, 2, 1]

    @pytest.mark.asyncio
    async def test_each_candidate_is_scored_against_the_query_that_fetched_it(self, monkeypatch):
        rec = _Recorder(monkeypatch, {
            "alpha beta": ([_point("c1", "alpha")], []),
            "gamma": ([_point("c1", "alpha"), _point("c4", "gamma")], []),
        })

        await rx.retrieval_executor_node(_state(sub_questions=["gamma"]))

        queries_by_chunk: dict[str, set] = {}
        for query, passage in rec.rerank_calls[0]:
            queries_by_chunk.setdefault(passage.rsplit(" | ", 1)[-1], set()).add(query)
        assert queries_by_chunk == {"alpha": {"alpha beta", "gamma"}, "gamma": {"gamma"}}
        assert all(p.startswith("Document: doc.txt | Section: S | ") for _, p in rec.rerank_calls[0])

    @pytest.mark.asyncio
    async def test_each_pass_is_sorted_by_its_own_scores(self, monkeypatch):
        _Recorder(monkeypatch, {
            "alpha beta": (
                [_point("c1", "nothing"), _point("c2", "alpha"), _point("c3", "alpha beta")], []),
        })

        out = await rx.retrieval_executor_node(_state())

        assert [c["chunk_id"] for c in out["reranked_results"]] == ["c3", "c2", "c1"]
        assert [c["reranker_score"] for c in out["reranked_results"]] == pytest.approx([0.2, 0.1, 0.0])

    @pytest.mark.asyncio
    async def test_payloads_come_from_the_search_not_a_second_fetch(self, monkeypatch):
        import src.vector_db.hybrid_search as hs

        async def no_scroll(*args, **kwargs):
            raise AssertionError("retrieval re-fetched payloads it already had")

        monkeypatch.setattr(hs, "fetch_chunks_by_ids", no_scroll)
        _Recorder(monkeypatch, {
            "alpha beta": ([_point("c1", "alpha")], [_point("c2", "beta"), _point("c1", "alpha")]),
        })

        out = await rx.retrieval_executor_node(_state())

        assert {c["chunk_id"] for c in out["reranked_results"]} == {"c1", "c2"}
        assert all(c["text"] for c in out["reranked_results"])

    @pytest.mark.asyncio
    async def test_no_candidates_means_no_model_call(self, monkeypatch):
        rec = _Recorder(monkeypatch, {})

        out = await rx.retrieval_executor_node(_state(sub_questions=["gamma"]))

        assert rec.rerank_calls == []
        assert out["reranked_results"] == []
        assert [p["candidates"] for p in out["retrieval_passes"]] == [0, 0]

    @pytest.mark.asyncio
    async def test_sub_questions_still_drop_the_parents_category(self, monkeypatch):
        """Batching must not blur which filters belong to which pass."""
        rec = _Recorder(monkeypatch, {})

        await rx.retrieval_executor_node(_state(sub_questions=["gamma"]))
        await rx.retrieval_executor_node(_state())

        decomposed, single = rec.search_calls[:2], rec.search_calls[2]
        assert all("document_category" not in c["metadata_filters"] for c in decomposed)
        assert single["metadata_filters"]["document_category"] == "financial"
        assert all(c["deal_id"] == "deal-1" for c in rec.search_calls)
        assert all(c["metadata_filters"]["include_pii"] is False for c in rec.search_calls)
