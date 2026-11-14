"""
The keyword (BM25) channel: what a term is, how it is weighted, and how an
index written under older rules is noticed.

FastEmbed 0.3 never lower-cases, keeps sentence-final periods on words and
treats "$452.8M" or "EV/EBITDA" as single terms, so a question and the passage
that answers it often shared no term at all. normalize_for_bm25 runs before the
encoder for documents and queries alike; these tests pin its rules, that both
sides go through it, that Qdrant applies IDF, and that points indexed before it
are counted rather than silently missed.

Only the `models`-marked test loads the real BM25 encoder.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

import src.vector_db.hybrid_search as hs
from src.vector_db.collection_manager import (
    ensure_sparse_idf,
    setup_collections,
    sparse_index_status,
)
from src.vector_db.constants import COLLECTION_NAME, VECTOR_SIZE


class TestNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        # Case: "Revenue" and "revenue" were different terms.
        ("What is the Revenue and EBITDA?", "what is the revenue and ebitda"),
        # A period that ends a sentence, not the text, used to stay on the word.
        ("Revenue grew in FY2023. Margins held.", "revenue grew in fy2023 margins held"),
        # Figures: the number is a term on its own; decimals and separators survive.
        ("Total revenue was $452.8M (17.0% growth).", "total revenue was 452.8 m 17.0 growth"),
        ("an escrow of $9,065,750", "an escrow of 9065750"),
        ("7.5x EV/EBITDA", "7.5 x ev ebitda"),
        # Defined terms and references.
        ("change-of-control under Section 280G", "change of control under section 280 g"),
        ("see Section 8.1(a).", "see section 8.1 a"),
        ("the Company's 10-K", "the company s 10 k"),
        # Markdown table cells and emphasis carry no terms.
        ("| **Revenue** | 452.8 |", "revenue 452.8"),
        # Full-width digits and non-breaking spaces (NFKC).
        ("FY２０２３ revenue", "fy2023 revenue"),
    ])
    def test_rules(self, raw, expected):
        assert hs.normalize_for_bm25(raw) == expected

    @pytest.mark.parametrize("text", [
        "Total revenue was $452.8M in FY2023. Adjusted EBITDA was $85 million.",
        "Wu's patents (U.S. Patent No. 10,123,456); Non-Compete: 24 months.",
        "",
        "--- *** ---",
    ])
    def test_idempotent(self, text):
        """FastEmbed >= 0.4 lower-cases itself; normalising twice must change nothing."""
        once = hs.normalize_for_bm25(text)
        assert hs.normalize_for_bm25(once) == once

    def test_question_and_passage_now_share_their_terms(self):
        question = set(hs.normalize_for_bm25("What is the Revenue and EBITDA?").split())
        passage = set(hs.normalize_for_bm25("revenue and ebitda were strong.").split())
        assert {"revenue", "ebitda"} <= question & passage


class _RecordingModel:
    """Stands in for the FastEmbed model; records the text it is asked to encode."""

    def __init__(self):
        self.seen: list[str] = []

    def embed(self, texts, batch_size=64):
        for text in texts:
            self.seen.append(text)
            ids = sorted({zlib.crc32(w.encode()) % 100_000 for w in text.split()})
            yield type("E", (), {"indices": np.asarray(ids), "values": np.ones(len(ids))})()


class TestBothSidesAreNormalised:
    def test_query_and_document_paths_encode_the_same_text(self, monkeypatch):
        model = _RecordingModel()
        monkeypatch.setattr(hs, "_bm25_model", model)

        query = hs.compute_sparse_bm25("EBITDA in FY2023?")
        documents = hs.compute_sparse_bm25_batch(["Adjusted EBITDA rose in FY2023.", "Nothing."])

        assert model.seen == ["ebitda in fy2023", "adjusted ebitda rose in fy2023", "nothing"]
        assert set(query.indices) <= set(documents[0].indices)
        assert not set(query.indices) & set(documents[1].indices)

    def test_empty_input_is_still_rejected(self):
        with pytest.raises(ValueError):
            hs.compute_sparse_bm25("")
        with pytest.raises(ValueError):
            hs.compute_sparse_bm25_batch(["ok", ""])

    @pytest.mark.models
    def test_real_encoder_matches_across_case_and_sentence_ends(self):
        query = set(hs.compute_sparse_bm25("What is the Revenue and EBITDA in FY2023?").indices)
        passage = set(hs.compute_sparse_bm25(
            "revenue and ebitda both rose in fy2023. Margins were flat.").indices)

        assert len(query & passage) == 3  # revenue, ebitda, fy2023; stopwords dropped
        assert len(query) == 3


# ==============================================================================
# IDF
# ==============================================================================


async def _legacy_collection() -> AsyncQdrantClient:
    """A child collection as created before the IDF modifier was set."""
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    return client


async def _modifier(client) -> Modifier | None:
    info = await client.get_collection(COLLECTION_NAME)
    return info.config.params.sparse_vectors["sparse"].modifier


async def _sparse_ranking(client) -> list[str]:
    """Three documents; the query has one term in every document and one in a single one."""
    unit = [1.0] + [0.0] * (VECTOR_SIZE - 1)
    docs = {
        "common-only-strong": SparseVector(indices=[1], values=[1.4]),
        "rare-and-common": SparseVector(indices=[1, 2], values=[0.6, 0.6]),
        "common-only-weak": SparseVector(indices=[1], values=[1.0]),
    }
    await client.upsert(COLLECTION_NAME, points=[
        PointStruct(id=i, payload={"name": name}, vector={"dense": unit, "sparse": vec})
        for i, (name, vec) in enumerate(docs.items())
    ])
    hits = await client.query_points(
        COLLECTION_NAME, query=SparseVector(indices=[1, 2], values=[1.0, 1.0]),
        using="sparse", limit=3, with_payload=True)
    return [p.payload["name"] for p in hits.points]


class TestIdfModifier:
    @pytest.mark.asyncio
    async def test_new_collections_are_created_with_idf(self):
        client = AsyncQdrantClient(location=":memory:")
        await setup_collections(client)

        assert await _modifier(client) == Modifier.IDF

    @pytest.mark.asyncio
    async def test_existing_collection_is_upgraded_in_place(self):
        client = await _legacy_collection()
        assert await _modifier(client) is None

        await setup_collections(client)

        assert await _modifier(client) == Modifier.IDF
        assert await ensure_sparse_idf(client) is True  # idempotent

    @pytest.mark.asyncio
    async def test_rare_term_outweighs_a_term_every_document_has(self):
        """Without IDF the strongest match on the ubiquitous term wins."""
        legacy = await _legacy_collection()
        assert (await _sparse_ranking(legacy))[0] == "common-only-strong"

        current = AsyncQdrantClient(location=":memory:")
        await setup_collections(current)
        assert (await _sparse_ranking(current))[0] == "rare-and-common"

    @pytest.mark.asyncio
    async def test_a_server_that_refuses_the_update_does_not_stop_startup(self):
        class Refusing:
            def __init__(self, inner):
                self.inner = inner

            async def get_collection(self, name):
                return await self.inner.get_collection(name)

            async def update_collection(self, **kwargs):
                raise RuntimeError("400: modifier cannot be changed")

        client = Refusing(await _legacy_collection())

        assert await ensure_sparse_idf(client) is False


# ==============================================================================
# Stale-index detection
# ==============================================================================


class TestStaleIndexDetection:
    @pytest.mark.asyncio
    async def test_points_from_an_older_encoding_are_counted(self):
        client = AsyncQdrantClient(location=":memory:")
        await setup_collections(client)
        unit = [1.0] + [0.0] * (VECTOR_SIZE - 1)
        payloads = [
            {"chunk_id": "current", "sparse_encoding": hs.SPARSE_ENCODING_VERSION},
            {"chunk_id": "previous-version", "sparse_encoding": hs.SPARSE_ENCODING_VERSION - 1},
            {"chunk_id": "before-the-marker-existed"},
        ]
        await client.upsert(COLLECTION_NAME, points=[
            PointStruct(id=i, payload=p, vector={"dense": unit}) for i, p in enumerate(payloads)
        ])

        status = await sparse_index_status(client)

        assert status["total_points"] == 3
        assert status["stale_points"] == 2
