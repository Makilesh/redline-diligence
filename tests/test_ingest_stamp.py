"""
Every point index_document stores says which extraction and chunking code
wrote it (ingest_version, Decision 97), and the re-index path when a run is
cut short after an extraction change.

Point ids come from chunk positions, so a re-index whose upsert fails part-way
after a chunking change leaves chunks of both runs under one document, and
new children beside parents the old run wrote. Nothing is deleted then (the
copy that was serving must not be lost), so what tells the two runs apart is
the stamp; indexing the same file again completes the document.

In-memory Qdrant, stub vectors, no model weights.
"""

from __future__ import annotations

import pytest

import src.data_processing.ingest_pipeline as ip
from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME
from tests.fixtures import ingestion_docs as fx
from tests.fixtures import reingest_docs as rd
from tests.fixtures.reingest_docs import default_bm25_is_a_stub  # noqa: F401  (fixture)

pytestmark = pytest.mark.usefixtures("default_bm25_is_a_stub")

FAILURES = {
    "child upsert batch 3": lambda client: rd.fail_child_batch(client, 3),
    "parent upsert": rd.fail_parents,
    "stale delete": rd.fail_delete,
}


class TestStamp:
    @pytest.mark.asyncio
    async def test_every_child_and_parent_carries_the_current_version(self, tmp_path):
        client = await rd.new_client()
        await rd.index(client, fx.write_txt(tmp_path))
        children, parents = await rd.points(client), await rd.points(client, PARENT_COLLECTION_NAME)
        assert children and parents
        assert {p["ingest_version"] for p, _ in children.values()} == {ip.INGEST_VERSION}
        assert {p["ingest_version"] for p, _ in parents.values()} == {ip.INGEST_VERSION}

    @pytest.mark.asyncio
    async def test_a_custom_sparse_function_is_still_stamped_with_the_ingest_version(self, tmp_path):
        """The stamp is about the chunks; only sparse_encoding depends on the BM25 function."""
        client = await rd.new_client()
        await rd.index(client, fx.write_txt(tmp_path), sparse_fn=rd.stub_sparse)
        children = await rd.points(client)
        assert all(p["ingest_version"] == ip.INGEST_VERSION and "sparse_encoding" not in p
                   for p, _ in children.values())

    def test_the_stamp_is_written_by_index_document_not_by_extraction(self, tmp_path):
        doc = ip.extract_and_chunk(str(fx.write_txt(tmp_path)), "aurora_sample.txt", rd.DEAL)
        assert not any("ingest_version" in p for p in doc.chunks + doc.parents)


class TestInterruptedReindex:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", sorted(FAILURES))
    async def test_after_a_chunking_change_two_runs_share_the_document_until_it_is_indexed_again(
            self, failure, monkeypatch, tmp_path):
        monkeypatch.setattr(ip, "QDRANT_BATCH_SIZE", 5)
        path = rd.write_supply_agreement(tmp_path)
        client = await rd.new_client()
        old = await rd.old_index(monkeypatch, client, path)
        fresh_children, fresh_parents = await rd.fresh(path)
        assert old["chunks_created"] > len(fresh_children)  # the old code cut it finer

        FAILURES[failure](client)
        with pytest.raises(ConnectionError):
            await rd.index(client, path)
        children = await rd.points(client)
        assert {p.get("ingest_version") for p, _ in children.values()} == {None, ip.INGEST_VERSION}

        rd.heal(client)
        again = await rd.index(client, path)
        assert again["previously_indexed"] is True
        assert rd.same(await rd.points(client), fresh_children)
        assert rd.same(await rd.points(client, PARENT_COLLECTION_NAME), fresh_parents)

    @pytest.mark.asyncio
    async def test_with_the_chunking_unchanged_a_reindex_cut_short_leaves_the_same_content(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(ip, "QDRANT_BATCH_SIZE", 5)
        path = rd.write_supply_agreement(tmp_path)
        client = await rd.new_client()
        await rd.index(client, path)
        before = await rd.points(client)
        rd.fail_child_batch(client, 2)
        with pytest.raises(ConnectionError):
            await rd.index(client, path)
        assert rd.same(await rd.points(client), before)

    @pytest.mark.asyncio
    async def test_parents_a_failed_rollback_left_are_removed_by_the_next_upload(self, monkeypatch, tmp_path):
        """A first ingest whose parent upsert failed, then whose rollback could not delete the parents."""
        monkeypatch.setattr(ip, "QDRANT_BATCH_SIZE", 2)
        path = rd.write_supply_agreement(tmp_path)
        client = await rd.new_client()
        real_upsert, real_delete = client.upsert, client.delete
        parent_calls = {"n": 0}

        async def upsert(collection_name, points, **kwargs):
            if collection_name == PARENT_COLLECTION_NAME:
                parent_calls["n"] += 1
                if parent_calls["n"] == 4:
                    raise ConnectionError("parent upsert batch 4 failed")
            return await real_upsert(collection_name=collection_name, points=points, **kwargs)

        async def delete(collection_name, points_selector, **kwargs):
            if collection_name == PARENT_COLLECTION_NAME:
                raise ConnectionError("parent delete failed")
            return await real_delete(collection_name=collection_name, points_selector=points_selector, **kwargs)

        client.upsert, client.delete = upsert, delete
        with monkeypatch.context() as m:
            m.setattr(ip, "SemanticChunker", rd.OldChunker)
            m.setattr(ip, "DEFAULT_PARENT_TARGET_TOKENS", 1024)
            with pytest.raises(ConnectionError):
                await rd.index(client, path)
        rd.heal(client)
        assert not await rd.points(client, COLLECTION_NAME)      # children rolled back
        leftover = await rd.points(client, PARENT_COLLECTION_NAME)
        assert len(leftover) == 6                                # three parent batches stayed

        first = await rd.index(client, path)
        fresh_children, fresh_parents = await rd.fresh(path)
        assert first["previously_indexed"] is False              # no child was there: a first ingest
        assert len(fresh_parents) < len(leftover)
        assert rd.same(await rd.points(client), fresh_children)
        assert rd.same(await rd.points(client, PARENT_COLLECTION_NAME), fresh_parents)
