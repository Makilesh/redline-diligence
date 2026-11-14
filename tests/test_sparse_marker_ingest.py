"""
Every chunk the default ingest path writes records which BM25 text encoding
produced its sparse vector, so an index written under an older encoding can be
counted (collection_manager.sparse_index_status) instead of silently missed by
keyword search. A caller-supplied sparse function is not stamped: its encoding
is unknown.

In-memory Qdrant, stub embeddings, no model weights.
"""

from __future__ import annotations

import hashlib
import zlib

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SparseVector

import src.vector_db.hybrid_search as hs
from src.vector_db.collection_manager import setup_collections, sparse_index_status
from src.vector_db.constants import COLLECTION_NAME, VECTOR_SIZE
from tests.fixtures import ingestion_docs as fx


async def _stub_embed(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        vec = np.random.default_rng(seed).standard_normal(VECTOR_SIZE)
        rows.append(vec / np.linalg.norm(vec))
    return np.asarray(rows, dtype=np.float32)


def _stub_sparse(texts: list[str]) -> list[SparseVector]:
    out = []
    for text in texts:
        ids = sorted({zlib.crc32(w.lower().encode()) % 100_000 for w in text.split()})
        out.append(SparseVector(indices=ids, values=[1.0] * len(ids)))
    return out


class TestSparseEncodingMarker:
    @pytest.mark.asyncio
    async def test_default_ingest_stamps_every_point(self, monkeypatch, tmp_path):
        from src.data_processing.ingest_pipeline import index_document

        monkeypatch.setattr(hs, "compute_sparse_bm25_batch", _stub_sparse)
        client = AsyncQdrantClient(location=":memory:")
        await setup_collections(client)
        path = fx.write_txt(tmp_path)

        await index_document(str(path), path.name, "deal-s", client=client, embed_fn=_stub_embed)

        points, _ = await client.scroll(COLLECTION_NAME, limit=1000, with_payload=True)
        assert points
        assert {p.payload.get("sparse_encoding") for p in points} == {hs.SPARSE_ENCODING_VERSION}
        status = await sparse_index_status(client)
        assert status == {
            "current_encoding": hs.SPARSE_ENCODING_VERSION,
            "total_points": len(points),
            "stale_points": 0,
        }

    @pytest.mark.asyncio
    async def test_a_custom_sparse_function_is_not_stamped(self, tmp_path):
        """Its encoding is unknown, so claiming the current one would be a guess."""
        from src.data_processing.ingest_pipeline import index_document

        client = AsyncQdrantClient(location=":memory:")
        await setup_collections(client)
        path = fx.write_txt(tmp_path)

        await index_document(str(path), path.name, "deal-s", client=client,
                             embed_fn=_stub_embed, sparse_fn=_stub_sparse)

        points, _ = await client.scroll(COLLECTION_NAME, limit=1000, with_payload=True)
        assert all("sparse_encoding" not in p.payload for p in points)
