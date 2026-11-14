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
