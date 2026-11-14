"""
Version state and failure handling of index_document / mark_superseded
(audit I7, I8, I9), against an in-memory Qdrant with stub vectors.
"""

from __future__ import annotations

import pytest

from src.data_processing.ingest_pipeline import mark_superseded
from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME
from tests.fixtures import ingestion_docs as fx
from tests.test_ingestion_retrieval import DEAL, _client, _index, _payloads

AMENDED = fx.SAMPLE_TXT + "\nSection 8.3 — Amendment\nThe cap is reduced to $150 million.\n"


async def _supersede(client, tmp_path):
    old = await _index(client, fx.write_txt(tmp_path, "spa_v1.txt"))
    new = await _index(client, fx.write_txt(tmp_path, "spa_v2.txt", AMENDED),
                       supersedes_doc_id=old["doc_id"])
    outcome = await mark_superseded(DEAL, old["doc_id"], new["doc_id"], client=client)
    return old, new, outcome


def _state(payloads, doc_id) -> set[tuple[int, str]]:
    return {(p["is_current_version"], p.get("superseded_by", "")) for p in payloads
            if p["doc_id"] == doc_id}


class TestRetiredVersionStaysRetired:
    @pytest.mark.asyncio
    async def test_reindexing_a_superseded_document_does_not_make_it_current(self, tmp_path):
        client = await _client()
        old, new, outcome = await _supersede(client, tmp_path)
        assert outcome["ok"] and outcome["chunks_retired"] == old["chunks_created"]

        # The old file is ingested again, as a folder re-index would.
        again = await _index(client, tmp_path / "spa_v1.txt")

        assert again["doc_id"] == old["doc_id"]
        assert again["superseded_by"] == new["doc_id"]
        assert any("superseded by" in w for w in again["warnings"])
        for collection in (COLLECTION_NAME, PARENT_COLLECTION_NAME):
            payloads = await _payloads(client, collection)
            assert _state(payloads, old["doc_id"]) == {(0, new["doc_id"])}
            assert {p["is_current_version"] for p in payloads if p["doc_id"] == new["doc_id"]} == {1}

    @pytest.mark.asyncio
    async def test_reindexing_a_current_document_keeps_it_current(self, tmp_path):
        client = await _client()
        path = fx.write_txt(tmp_path)
        await _index(client, path)
        again = await _index(client, path)
        assert again["superseded_by"] == ""
        assert {p["is_current_version"] for p in await _payloads(client)} == {1}


class TestSupersessionIsReported:
    @pytest.mark.asyncio
    async def test_unknown_document_is_a_failure_not_a_silent_no_op(self, tmp_path):
        client = await _client()
        new = await _index(client, fx.write_txt(tmp_path))
        outcome = await mark_superseded(DEAL, "no-such-doc", new["doc_id"], client=client)
        assert outcome == {"ok": False, "chunks_retired": 0,
                           "error": "the superseded document is not indexed in this deal"}
        assert {p["is_current_version"] for p in await _payloads(client)} == {1}

    @pytest.mark.asyncio
    async def test_other_deals_document_is_not_retired(self, tmp_path):
        client = await _client()
        other = await _index(client, fx.write_txt(tmp_path, "a.txt"), deal_id="other-deal")
        new = await _index(client, fx.write_txt(tmp_path, "b.txt", AMENDED))
        outcome = await mark_superseded(DEAL, other["doc_id"], new["doc_id"], client=client)
        assert not outcome["ok"]
        assert {p["is_current_version"] for p in await _payloads(client)} == {1}

    @pytest.mark.asyncio
    async def test_store_failure_is_reported(self, tmp_path):
        client = await _client()
        old = await _index(client, fx.write_txt(tmp_path, "v1.txt"))
        new = await _index(client, fx.write_txt(tmp_path, "v2.txt", AMENDED))

        async def broken(**kwargs):
            raise ConnectionError("qdrant went away")

        client.set_payload = broken
        outcome = await mark_superseded(DEAL, old["doc_id"], new["doc_id"], client=client)
        assert outcome["ok"] is False
        assert "qdrant" not in (outcome["error"] or "")  # no internals in a client-facing reason

    @pytest.mark.asyncio
    async def test_self_supersession_is_refused(self, tmp_path):
        client = await _client()
        doc = await _index(client, fx.write_txt(tmp_path))
        outcome = await mark_superseded(DEAL, doc["doc_id"], doc["doc_id"], client=client)
        assert outcome["ok"] is False
        assert {p["is_current_version"] for p in await _payloads(client)} == {1}


class TestFailedReindexKeepsTheGoodCopy:
    @pytest.mark.asyncio
    async def test_failed_reindex_does_not_delete_the_serving_copy(self, tmp_path):
        client = await _client()
        path = fx.write_txt(tmp_path)
        first = await _index(client, path)
        before = sorted(p["chunk_id"] for p in await _payloads(client))
        parents_before = len(await _payloads(client, PARENT_COLLECTION_NAME))
        real_upsert = client.upsert

        async def failing_upsert(collection_name, points, **kwargs):
            if collection_name == PARENT_COLLECTION_NAME:
                raise ConnectionError("qdrant went away")
            return await real_upsert(collection_name=collection_name, points=points, **kwargs)

        client.upsert = failing_upsert
        with pytest.raises(ConnectionError):
            await _index(client, path)

        assert sorted(p["chunk_id"] for p in await _payloads(client)) == before
        assert len(await _payloads(client, PARENT_COLLECTION_NAME)) == parents_before
        assert first["chunks_created"] == len(before)

    @pytest.mark.asyncio
    async def test_failed_first_ingest_still_leaves_nothing(self, tmp_path):
        client = await _client()
        real_upsert = client.upsert

        async def failing_upsert(collection_name, points, **kwargs):
            if collection_name == PARENT_COLLECTION_NAME:
                raise ConnectionError("qdrant went away")
            return await real_upsert(collection_name=collection_name, points=points, **kwargs)

        client.upsert = failing_upsert
        with pytest.raises(ConnectionError):
            await _index(client, fx.write_txt(tmp_path))
        assert await _payloads(client) == []
