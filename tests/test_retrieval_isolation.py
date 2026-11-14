"""
Deal, version and PII conditions are enforced by Qdrant on both search channels.

hybrid_search issues a dense and a sparse query. Each must carry the same
server-side filter: a chunk of another deal, a superseded version or a
PII-flagged chunk must be absent from BOTH lists, whatever the query matches and
whatever keys the (LLM-authored) metadata_filters dict contains. Any change to
how candidates are fetched — batching, a server-side fusion query — has to keep
this test passing.

In-memory Qdrant, hand-made vectors, no models.
"""

from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, VECTOR_SIZE
from src.vector_db.hybrid_search import hybrid_search

DEAL = "deal-a"
UNIT = [1.0] + [0.0] * (VECTOR_SIZE - 1)
TERM = SparseVector(indices=[7], values=[1.0])


def _payload(chunk_id: str, **overrides) -> dict:
    payload = {"chunk_id": chunk_id, "deal_id": DEAL, "text": chunk_id, "is_current_version": 1,
               "contains_pii": 0, "document_category": "legal", "doc_id": "doc-1"}
    payload.update(overrides)
    return payload


async def _client() -> AsyncQdrantClient:
    """Five chunks with identical vectors, so only the filter can tell them apart."""
    client = AsyncQdrantClient(location=":memory:")
    await setup_collections(client)
    payloads = [
        _payload("visible"),
        _payload("visible-financial", document_category="financial"),
        _payload("other-deal", deal_id="deal-b"),
        _payload("superseded", is_current_version=0),
        _payload("pii", contains_pii=1),
    ]
    await client.upsert(COLLECTION_NAME, points=[
        PointStruct(id=i, payload=p, vector={"dense": UNIT, "sparse": TERM})
        for i, p in enumerate(payloads)
    ])
    return client


async def _search(client, filters: dict, **kwargs) -> tuple[set[str], set[str]]:
    dense, sparse = await hybrid_search(
        query_text="q", query_vector=UNIT, query_sparse=TERM, deal_id=DEAL,
        metadata_filters=filters, client=client, **kwargs)
    return ({p.payload["chunk_id"] for p in dense}, {p.payload["chunk_id"] for p in sparse})


class TestBothChannelsAreFiltered:
    @pytest.mark.asyncio
    async def test_default_search_sees_only_current_non_pii_chunks_of_the_deal(self):
        dense, sparse = await _search(await _client(), {"include_pii": False})

        assert dense == sparse == {"visible", "visible-financial"}

    @pytest.mark.asyncio
    async def test_authorized_reader_gets_pii_but_nothing_else(self):
        dense, sparse = await _search(await _client(), {"include_pii": True})

        assert dense == sparse == {"visible", "visible-financial", "pii"}

    @pytest.mark.asyncio
    async def test_superseded_versions_need_the_trusted_argument(self):
        dense, sparse = await _search(await _client(), {"include_pii": False}, include_superseded=True)

        assert dense == sparse == {"visible", "visible-financial", "superseded"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_filters", [
        {"deal_id": "deal-b"},
        {"is_current_version": 0},
        {"contains_pii": 1},
        {"deal_id": ["deal-a", "deal-b"], "is_current_version": None, "contains_pii": None},
    ])
    async def test_model_authored_keys_cannot_widen_the_search(self, llm_filters):
        dense, sparse = await _search(await _client(), {**llm_filters, "include_pii": False})

        assert dense == sparse == {"visible", "visible-financial"}

    @pytest.mark.asyncio
    async def test_a_category_filter_narrows_both_channels(self):
        dense, sparse = await _search(
            await _client(), {"document_category": "financial", "include_pii": False})

        assert dense == sparse == {"visible-financial"}


class TestModelOutputCannotNarrowToNothing:
    """
    The filters dict is model output. Only a known document_category may become
    a hard filter: any other key would confine every search of the request to
    whatever the model echoed, and the rewrite loop cannot take it off again.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_filters", [
        {"doc_id": "some-other-doc"},
        {"is_table": 1},
        {"content_type": "table_markdown"},
        {"document_category": "not-a-category"},
        {"document_category": ["legal", {"nested": "object"}]},
        {"fiscal_year": "FY2023", "currency": "USD"},
    ])
    async def test_unknown_or_invalid_keys_are_ignored(self, llm_filters):
        dense, sparse = await _search(await _client(), {**llm_filters, "include_pii": False})

        assert dense == sparse == {"visible", "visible-financial"}

    @pytest.mark.parametrize("not_a_dict", [None, [], ["document_category"], "legal"])
    def test_a_filters_value_of_the_wrong_type_is_treated_as_no_filters(self, not_a_dict):
        from src.vector_db.hybrid_search import _build_filter

        keys = [c.key for c in _build_filter(DEAL, not_a_dict).must]

        assert keys == ["deal_id", "is_current_version", "contains_pii"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("extracted", [None, [], "legal"])
    async def test_agent_3_survives_malformed_filters_in_state(self, monkeypatch, extracted):
        import numpy as np

        import src.agents.retrieval_executor as rx

        seen = {}

        async def embed(texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

        async def search(**kwargs):
            seen.update(kwargs["metadata_filters"])
            return [], []

        async def expand(**kwargs):
            return []

        monkeypatch.setattr(rx, "embed_texts_async", embed)
        monkeypatch.setattr(rx, "compute_sparse_bm25", lambda text: None)
        monkeypatch.setattr(rx, "hybrid_search", search)
        monkeypatch.setattr(rx, "expand_context", expand)

        await rx.retrieval_executor_node({
            "current_query": "q", "query_type": "legal", "parsed_intent": {},
            "deal_id": DEAL, "include_pii": False, "rewrite_iteration": 0,
            "retrieval_config": {}, "extracted_filters": extracted,
        })

        assert seen == {"include_pii": False}
