"""
Helpers for the re-ingestion tests (Decision 97): an in-memory index with
stub vectors, a document long enough to span several upsert batches, an
"old code" that cuts it differently, and clients that fail part-way.

"Old code" is the shipped pipeline with smaller chunks and parents, standing
in for an extraction change: it cuts the same file differently, as Decisions
89 and 93 do for PDFs, and its points carry no ingest_version, as an index
written before Decision 97 holds them.
"""

from __future__ import annotations

import hashlib
import zlib
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, SparseVector

import src.data_processing.ingest_pipeline as ip
import src.vector_db.hybrid_search as hs
from src.data_processing.semantic_chunker import SemanticChunker
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME, VECTOR_SIZE

DEAL = "deal-re"


async def stub_embed(texts: list[str]) -> np.ndarray:
    """Deterministic unit vectors seeded by text content."""
    rows = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        vec = np.random.default_rng(seed).standard_normal(VECTOR_SIZE)
        rows.append(vec / np.linalg.norm(vec))
    return np.asarray(rows, dtype=np.float32)


def stub_sparse(texts: list[str]) -> list[SparseVector]:
    """Bag-of-words sparse vectors (hashed term ids, unit weights)."""
    out = []
    for text in texts:
        ids = sorted({zlib.crc32(w.lower().encode()) % 100_000 for w in text.split()})
        out.append(SparseVector(indices=ids, values=[1.0] * len(ids)))
    return out


@pytest.fixture
def default_bm25_is_a_stub(monkeypatch):
    """index_document stamps sparse_encoding only for its default BM25 function;
    a stub in its place keeps the stamp without loading FastEmbed."""
    monkeypatch.setattr(hs, "compute_sparse_bm25_batch", stub_sparse)
    monkeypatch.setenv("LAYA_ENABLED", "0")


def supply_agreement() -> str:
    """Twelve articles of twenty clauses: 13 children and 5 parents with the shipped chunker."""
    parts = ["MASTER SUPPLY AGREEMENT\n\nThis Master Supply Agreement is made between Atlas Corp. "
             "and Borealis Ltd.\n"]
    for n in range(1, 13):
        body = " ".join(f"Clause {n}.{k}: the supplier shall deliver lot {n * 10 + k} within {k + 3} days of "
                        f"the purchase order, at the price set out in Schedule {n}." for k in range(1, 21))
        parts.append(f"ARTICLE {n} — OBLIGATIONS {n}\n\n{body}\n")
    return "\n".join(parts)


def write_supply_agreement(directory: Path, name: str = "supply.txt") -> Path:
    path = directory / name
    path.write_text(supply_agreement(), encoding="utf-8")
    return path


class OldChunker(SemanticChunker):
    """Smaller chunks than the shipped chunker: other boundaries, more of them."""

    def __init__(self, *args, **kwargs):
        super().__init__(target_tokens=160, min_tokens=60, max_tokens=240)


async def new_client() -> AsyncQdrantClient:
    client = AsyncQdrantClient(location=":memory:")
    await setup_collections(client)
    return client


async def index(client, path: Path, name: str | None = None, deal: str = DEAL, **kwargs) -> dict:
    """index_document as POST /ingest calls it, with stub dense vectors."""
    return await ip.index_document(str(path), name or path.name, deal, kwargs.pop("category", None),
                                   client=client, embed_fn=stub_embed, **kwargs)


async def points(client, collection: str = COLLECTION_NAME) -> dict:
    """{point id: (payload, vector)} for every point of a collection."""
    out, offset = {}, None
    while True:
        page, offset = await client.scroll(collection, limit=1000, offset=offset, with_payload=True,
                                           with_vectors=True)
        out.update({p.id: (p.payload, p.vector) for p in page})
        if offset is None:
            return out


def same(got: dict, expected: dict) -> bool:
    """
    The same points: ids, payloads and sparse vectors exactly, dense vectors to
    float32 rounding (local Qdrant normalises a vector on upsert, and one that
    overwrites an existing point can come out an ulp apart).
    """
    if got.keys() != expected.keys():
        return False
    for point_id, (payload, vector) in got.items():
        other_payload, other_vector = expected[point_id]
        if payload != other_payload or vector.get("sparse") != other_vector.get("sparse"):
            return False
        if not np.allclose(vector["dense"], other_vector["dense"], rtol=0, atol=1e-6):
            return False
    return True


async def strip_stamp(client, deal: str = DEAL) -> None:
    """Makes a deal's points look as an index written before Decision 97 holds them."""
    for collection in (COLLECTION_NAME, PARENT_COLLECTION_NAME):
        await client.delete_payload(collection, keys=["ingest_version"],
                                    points=Filter(must=[FieldCondition(key="deal_id", match=MatchValue(value=deal))]))


async def old_index(monkeypatch, client, path: Path, name: str | None = None, **kwargs) -> dict:
    """Indexes `path` as the old code would have: cut differently, unstamped."""
    with monkeypatch.context() as m:
        m.setattr(ip, "SemanticChunker", OldChunker)
        m.setattr(ip, "DEFAULT_PARENT_TARGET_TOKENS", 1024)
        result = await index(client, path, name, **kwargs)
    await strip_stamp(client)
    return result


async def fresh(path: Path, name: str | None = None, **kwargs) -> tuple[dict, dict]:
    """The current code's ingest of `path` into an empty store: (children, parents)."""
    client = await new_client()
    await index(client, path, name, **kwargs)
    return await points(client), await points(client, PARENT_COLLECTION_NAME)


def fail_child_batch(client, k: int) -> None:
    """The k-th child upsert of the client's life raises."""
    real = client.upsert
    calls = {"n": 0}

    async def upsert(collection_name, points, **kwargs):
        if collection_name == COLLECTION_NAME:
            calls["n"] += 1
            if calls["n"] == k:
                raise ConnectionError(f"child upsert batch {k} failed")
        return await real(collection_name=collection_name, points=points, **kwargs)
    client.upsert = upsert


def fail_parents(client) -> None:
    """Every parent upsert raises."""
    real = client.upsert

    async def upsert(collection_name, points, **kwargs):
        if collection_name == PARENT_COLLECTION_NAME:
            raise ConnectionError("parent upsert failed")
        return await real(collection_name=collection_name, points=points, **kwargs)
    client.upsert = upsert


def fail_delete(client) -> None:
    """Every delete raises: the process died between the upserts and the stale delete."""
    async def delete(*args, **kwargs):
        raise ConnectionError("the process died before the stale delete")
    client.delete = delete


def heal(client) -> None:
    """Removes whatever failure was installed on the client."""
    for name in ("upsert", "delete"):
        client.__dict__.pop(name, None)
