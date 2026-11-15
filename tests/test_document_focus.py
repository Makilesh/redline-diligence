"""
Document focus (Decision 67): after fusion, run a dense search inside the
leading document and add what it finds to the rerank pool.

Offline: Qdrant, the embedder and the cross-encoder are stubbed.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.agents.retrieval_executor as rx
from src.agents.retrieval_strategy import RETRIEVAL_CONFIGS, clamp_retrieval_config
from src.vector_db.hybrid_search import _build_filter
from src.vector_db.rrf_fusion import leading_document
from tests.conftest import make_scored_point


def _point(chunk_id: str, doc: str, text: str = ""):
    point = make_scored_point(chunk_id, 0.5)
    point.payload.update({"doc_id": doc, "source_file": f"{doc}.txt",
                          "text": text or f"text of {chunk_id}"})
    return point


class TestLeadingDocument:
    def test_summed_fused_score_picks_the_document(self):
        fused = [("a1", 0.030), ("b1", 0.029), ("b2", 0.020), ("a2", 0.010)]
        doc, share = leading_document(fused, {"a1": "A", "a2": "A", "b1": "B", "b2": "B"})
        assert doc == "B"
        assert share == pytest.approx(0.049 / 0.089)

    def test_one_strong_chunk_loses_to_a_document_that_keeps_coming_up(self):
        fused = [("x", 0.016), ("y1", 0.012), ("y2", 0.011)]
        assert leading_document(fused, {"x": "X", "y1": "Y", "y2": "Y"})[0] == "Y"

    def test_no_documents(self):
        assert leading_document([], {}) == (None, 0.0)
        assert leading_document([("a", 0.1)], {}) == (None, 0.0)


class TestTrustedDocumentScope:
    def test_doc_id_becomes_a_filter_condition(self):
        keys = [c.key for c in _build_filter("deal", {}, doc_id="D1").must]
        assert "doc_id" in keys and "deal_id" in keys and "contains_pii" in keys

    def test_doc_id_is_never_taken_from_llm_filters(self):
        keys = [c.key for c in _build_filter("deal", {"doc_id": "D1"}).must]
        assert "doc_id" not in keys

    def test_focus_depth_is_clamped_when_an_llm_rewrites_the_config(self):
        assert clamp_retrieval_config({"document_focus_k": 500})["document_focus_k"] == 30
        assert clamp_retrieval_config({"document_focus_k": -3})["document_focus_k"] == 0


class _Stubs:
    def __init__(self, monkeypatch, corpus_wide, in_document):
        self.search_calls: list[dict] = []
        self.focus_calls: list[dict] = []
        self.reranked: list[str] = []

        async def embed(texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

        async def search(**kwargs):
            self.search_calls.append(kwargs)
            return corpus_wide

        async def dense(**kwargs):
            self.focus_calls.append(kwargs)
            return in_document.get(kwargs["doc_id"], [])[: kwargs["top_k"]]

        async def rerank(pairs):
            self.reranked = [p for _, p in pairs]
            # The clause that answers scores highest wherever it came from.
            return np.asarray([0.9 if "answer" in p else 0.1 for _, p in pairs], dtype=np.float32)

        async def expand(**kwargs):
            return list(kwargs["chunks"])

        monkeypatch.setattr(rx, "embed_texts_async", embed)
        monkeypatch.setattr(rx, "compute_sparse_bm25", lambda text: "sparse")
        monkeypatch.setattr(rx, "hybrid_search", search)
        monkeypatch.setattr(rx, "dense_search", dense)
        monkeypatch.setattr(rx, "rerank_pairs_async", rerank)
        monkeypatch.setattr(rx, "expand_context", expand)


def _state(config: dict) -> dict:
    return {"current_query": "closing condition", "query_type": "legal", "parsed_intent": {},
            "deal_id": "deal-1", "include_pii": False, "rewrite_iteration": 0,
            "retrieval_config": config, "extracted_filters": {}, "sub_questions": []}


# The corpus-wide lists agree on document A (its preamble names the parties)
# but hold another agreement's version of the clause; A's own clause is only
# found by searching inside A.
CORPUS_WIDE = (
    [_point("a_preamble", "A"), _point("b_clause", "B", "B's clause"), _point("a_def", "A")],
    [_point("a_preamble", "A"), _point("a_def", "A"), _point("a_sig", "A")],
)
IN_A = [_point("a_clause", "A", "the answer"), _point("a_def", "A"), _point("a_sig", "A")]


@pytest.mark.asyncio
async def test_focus_adds_the_leading_documents_own_passages(monkeypatch):
    stubs = _Stubs(monkeypatch, CORPUS_WIDE, {"A": IN_A})
    out = await rx.retrieval_executor_node(_state({**RETRIEVAL_CONFIGS["legal"], "document_focus_k": 5}))

    assert len(stubs.focus_calls) == 1 and stubs.focus_calls[0]["doc_id"] == "A"
    assert stubs.focus_calls[0]["top_k"] == 5
    assert out["reranked_results"][0]["chunk_id"] == "a_clause"
    # Added, de-duplicated, nothing removed: B's clause is still scored.
    ids = [c["chunk_id"] for c in out["reranked_results"]]
    assert sorted(ids) == sorted({"a_preamble", "b_clause", "a_def", "a_sig", "a_clause"})


@pytest.mark.asyncio
async def test_focus_off_is_the_old_pipeline(monkeypatch):
    stubs = _Stubs(monkeypatch, CORPUS_WIDE, {"A": IN_A})
    out = await rx.retrieval_executor_node(_state({**RETRIEVAL_CONFIGS["legal"], "document_focus_k": 0}))

    assert stubs.focus_calls == []
    assert "a_clause" not in [c["chunk_id"] for c in out["reranked_results"]]


@pytest.mark.asyncio
async def test_focus_keeps_the_callers_filters(monkeypatch):
    stubs = _Stubs(monkeypatch, CORPUS_WIDE, {"A": IN_A})
    state = _state({**RETRIEVAL_CONFIGS["legal"], "document_focus_k": 5})
    state["include_pii"] = False
    await rx.retrieval_executor_node(state)

    focus = stubs.focus_calls[0]
    assert focus["deal_id"] == "deal-1"
    assert focus["metadata_filters"]["include_pii"] is False


@pytest.mark.asyncio
async def test_no_candidates_means_no_focus_search(monkeypatch):
    stubs = _Stubs(monkeypatch, ([], []), {})
    out = await rx.retrieval_executor_node(_state({**RETRIEVAL_CONFIGS["legal"], "document_focus_k": 5}))

    assert out["reranked_results"] == []
    assert stubs.focus_calls == []
