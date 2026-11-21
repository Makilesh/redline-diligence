"""
A first-time ingest is staged and committed (index_document), so an upload cut
short where no handler runs — a crash, an OOM kill, a cancelled request —
leaves nothing any reader sees. Before, the child points were written with
their deal_id batch by batch, and a run killed between batches left a partial
document that searches returned and the document list showed as indexed.

In-memory Qdrant, stub vectors; a crash is a BaseException that the ingest's
`except Exception` rollback does not catch, as with a killed process.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

import src.data_processing.ingest_pipeline as ip
import src.vector_db.qdrant_client as qdrant_module
from api.routes import deals
from src.vector_db import reingest_report
from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME, STAGED_DEAL_FIELD
from src.vector_db.hybrid_search import deal_read_filter
from tests.fixtures import reingest_docs as rd
from tests.fixtures.reingest_docs import default_bm25_is_a_stub  # noqa: F401  (fixture)

COLLECTIONS = (COLLECTION_NAME, PARENT_COLLECTION_NAME)


class Crash(BaseException):
    """The process dying mid-call: not an Exception, so no rollback runs."""


def crash_child_batch(client, k: int, error: type[BaseException] = Crash) -> None:
    """The k-th child upsert of the client's life dies."""
    real = client.upsert
    calls = {"n": 0}

    async def upsert(collection_name, points, **kwargs):
        if collection_name == COLLECTION_NAME:
            calls["n"] += 1
            if calls["n"] == k:
                raise error()
        return await real(collection_name=collection_name, points=points, **kwargs)
    client.upsert = upsert


def crash_commit_of(client, collection: str, error: type[BaseException] = Crash) -> None:
    """The commit (set_payload) of one collection dies; the other's goes through."""
    real = client.set_payload

    async def set_payload(collection_name, **kwargs):
        if collection_name == collection:
            raise error()
        return await real(collection_name=collection_name, **kwargs)
    client.set_payload = set_payload


async def readable(client, collection: str = COLLECTION_NAME, deal: str = rd.DEAL) -> int:
    """Points of the deal any reader can see (every reader filters on deal_id)."""
    flt = Filter(must=[FieldCondition(key="deal_id", match=MatchValue(value=deal))])
    return (await client.count(collection, count_filter=flt, exact=True)).count


async def searchable(client, deal: str = rd.DEAL) -> int:
    """Points of the deal a question may read (the search and tie-out filter)."""
    return (await client.count(COLLECTION_NAME, count_filter=deal_read_filter(deal), exact=True)).count


async def staged_left(client) -> dict:
    return {name: sum(1 for p, _ in (await rd.points(client, name)).values() if STAGED_DEAL_FIELD in p)
            for name in COLLECTIONS}


@pytest.fixture
def small_batches(monkeypatch):
    monkeypatch.setattr(ip, "QDRANT_BATCH_SIZE", 2)


# ==============================================================================
# A completed ingest stores what it stored before staging
# ==============================================================================


class TestCommittedDocument:
    @pytest.mark.asyncio
    async def test_first_ingest_commits_every_point_and_leaves_no_marker(self, tmp_path, small_batches):
        client = await rd.new_client()
        result = await rd.index(client, rd.write_supply_agreement(tmp_path))

        assert await readable(client) == result["chunks_created"]
        assert await readable(client, PARENT_COLLECTION_NAME) == result["parent_chunks_created"]
        assert await staged_left(client) == {COLLECTION_NAME: 0, PARENT_COLLECTION_NAME: 0}

    @pytest.mark.asyncio
    async def test_committed_points_equal_a_direct_write(self, tmp_path, small_batches):
        """A re-index writes in place, unstaged: the two must store the same points."""
        client = await rd.new_client()
        path = rd.write_supply_agreement(tmp_path)
        await rd.index(client, path)
        staged = {name: await rd.points(client, name) for name in COLLECTIONS}

        again = await rd.index(client, path)
        assert again["previously_indexed"] is True
        for name in COLLECTIONS:
            assert rd.same(await rd.points(client, name), staged[name])

    @pytest.mark.asyncio
    async def test_a_reindex_never_hides_the_serving_copy(self, tmp_path, small_batches):
        """Audit I9: during a re-index every written point carries its deal_id."""
        client = await rd.new_client()
        path = rd.write_supply_agreement(tmp_path)
        await rd.index(client, path)
        real = client.upsert
        written: list[dict] = []

        async def upsert(collection_name, points, **kwargs):
            written.extend(p.payload for p in points)
            return await real(collection_name=collection_name, points=points, **kwargs)
        client.upsert = upsert

        await rd.index(client, path)
        assert written and all(p.get("deal_id") == rd.DEAL and STAGED_DEAL_FIELD not in p for p in written)


# ==============================================================================
# A run cut short leaves nothing readable
# ==============================================================================


class TestRunCutShort:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [Crash, asyncio.CancelledError], ids=["killed", "cancelled"])
    async def test_mid_child_upsert(self, tmp_path, small_batches, error):
        client = await rd.new_client()
        crash_child_batch(client, 3, error)
        with pytest.raises(error):
            await rd.index(client, rd.write_supply_agreement(tmp_path))

        # Four children were written before the run died. Before staging they
        # were searchable, with no registry entry and no audit line.
        assert len(await rd.points(client)) == 4
        assert await readable(client) == 0 and await searchable(client) == 0
        assert await readable(client, PARENT_COLLECTION_NAME) == 0

    @pytest.mark.asyncio
    async def test_mid_parent_upsert(self, tmp_path, small_batches):
        client = await rd.new_client()
        real = client.upsert

        async def upsert(collection_name, points, **kwargs):
            if collection_name == PARENT_COLLECTION_NAME:
                raise Crash()
            return await real(collection_name=collection_name, points=points, **kwargs)
        client.upsert = upsert
        with pytest.raises(Crash):
            await rd.index(client, rd.write_supply_agreement(tmp_path))

        assert len(await rd.points(client)) > 0               # every child was written
        assert await readable(client) == 0 and await searchable(client) == 0

    @pytest.mark.asyncio
    async def test_between_the_two_commits_no_child_is_readable(self, tmp_path, small_batches):
        """Parents commit first: a child is never readable without its parent section."""
        client = await rd.new_client()
        crash_commit_of(client, COLLECTION_NAME)
        with pytest.raises(Crash):
            await rd.index(client, rd.write_supply_agreement(tmp_path))

        assert await readable(client, PARENT_COLLECTION_NAME) > 0
        assert await readable(client) == 0 and await searchable(client) == 0


# ==============================================================================
# Recovery: the same upload again, or a purge
# ==============================================================================


class TestRecovery:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cut", ["child upsert", "child commit"])
    async def test_uploading_the_same_file_again_commits_it_whole(self, tmp_path, small_batches, cut):
        client = await rd.new_client()
        path = rd.write_supply_agreement(tmp_path)
        if cut == "child upsert":
            crash_child_batch(client, 3)
        else:
            crash_commit_of(client, COLLECTION_NAME)
        with pytest.raises(Crash):
            await rd.index(client, path)
        for name in ("upsert", "set_payload"):
            client.__dict__.pop(name, None)

        result = await rd.index(client, path)
        fresh_children, fresh_parents = await rd.fresh(path)
        assert result["chunks_created"] == len(fresh_children)
        assert rd.same(await rd.points(client), fresh_children)
        assert rd.same(await rd.points(client, PARENT_COLLECTION_NAME), fresh_parents)

    @pytest.mark.asyncio
    async def test_leftovers_of_a_different_chunking_are_deleted_not_committed(
            self, monkeypatch, tmp_path, small_batches):
        client = await rd.new_client()
        path = rd.write_supply_agreement(tmp_path)
        with monkeypatch.context() as m:                        # the old code, cut short at its commit
            m.setattr(ip, "SemanticChunker", rd.OldChunker)
            m.setattr(ip, "DEFAULT_PARENT_TARGET_TOKENS", 1024)
            crash_commit_of(client, PARENT_COLLECTION_NAME)
            with pytest.raises(Crash):
                await rd.index(client, path)
        client.__dict__.pop("set_payload", None)
        old_ids = set(await rd.points(client)) | set(await rd.points(client, PARENT_COLLECTION_NAME))

        await rd.index(client, path)
        fresh_children, fresh_parents = await rd.fresh(path)
        assert old_ids - set(fresh_children) - set(fresh_parents), "the old chunking wrote other ids"
        assert rd.same(await rd.points(client), fresh_children)
        assert rd.same(await rd.points(client, PARENT_COLLECTION_NAME), fresh_parents)

    @pytest.mark.asyncio
    async def test_leftovers_stay_unreadable_when_their_delete_fails(self, monkeypatch, tmp_path, small_batches):
        """The cleanup is optional (Decision 88): the upload succeeds, the leftovers stay staged."""
        client = await rd.new_client()
        path = rd.write_supply_agreement(tmp_path)
        with monkeypatch.context() as m:
            m.setattr(ip, "SemanticChunker", rd.OldChunker)
            m.setattr(ip, "DEFAULT_PARENT_TARGET_TOKENS", 1024)
            crash_commit_of(client, PARENT_COLLECTION_NAME)
            with pytest.raises(Crash):
                await rd.index(client, path)
        client.__dict__.pop("set_payload", None)
        rd.fail_delete(client)
        monkeypatch.setenv("STRICT_OPTIONAL_STEPS", "0")

        result = await rd.index(client, path)
        fresh_children, fresh_parents = await rd.fresh(path)
        assert await readable(client) == result["chunks_created"] == len(fresh_children)
        assert await readable(client, PARENT_COLLECTION_NAME) == len(fresh_parents)
        assert sum((await staged_left(client)).values()) > 0   # the old run's other ids, unread

    @pytest.mark.asyncio
    async def test_a_failed_commit_rolls_everything_back(self, tmp_path, small_batches):
        client = await rd.new_client()
        crash_commit_of(client, COLLECTION_NAME, error=ConnectionError)
        with pytest.raises(ConnectionError):
            await rd.index(client, rd.write_supply_agreement(tmp_path))
        for name in COLLECTIONS:
            assert await rd.points(client, name) == {}

    @pytest.mark.asyncio
    async def test_purging_the_deal_deletes_staged_points(self, monkeypatch, tmp_path, small_batches):
        client = await rd.new_client()
        await rd.index(client, rd.write_supply_agreement(tmp_path, "kept.txt"), deal="other-deal")
        crash_commit_of(client, COLLECTION_NAME)
        with pytest.raises(Crash):
            await rd.index(client, rd.write_supply_agreement(tmp_path))
        client.__dict__.pop("set_payload", None)
        monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: client)

        await deals._delete_deal_vectors(rd.DEAL)
        for name in COLLECTIONS:
            assert all(p.get("deal_id") == "other-deal" for p, _ in (await rd.points(client, name)).values())
        assert await readable(client, deal="other-deal") > 0

    @pytest.mark.asyncio
    async def test_the_sweeper_finds_a_sandbox_holding_only_staged_points(self, monkeypatch, tmp_path,
                                                                          small_batches):
        client = await rd.new_client()
        sandbox = deals.new_sandbox_id(datetime(2020, 1, 1, tzinfo=timezone.utc))
        crash_child_batch(client, 3)
        with pytest.raises(Crash):
            await rd.index(client, rd.write_supply_agreement(tmp_path), deal=sandbox)
        monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: client)

        assert sandbox in await deals._facet_deal_ids(include_parents=True)
        assert sandbox not in await deals._facet_deal_ids()     # the deal list does not show it
        assert await deals._sweep_expired_sandboxes() == 1
        for name in COLLECTIONS:
            assert await rd.points(client, name) == {}


# ==============================================================================
# The re-ingest report lists them apart
# ==============================================================================


class TestReport:
    @pytest.mark.asyncio
    async def test_staged_documents_are_listed_by_their_deal_and_not_as_documents(self, tmp_path,
                                                                                  small_batches):
        client = await rd.new_client()
        kept = await rd.index(client, rd.write_supply_agreement(tmp_path, "kept.txt"))
        crash_child_batch(client, 3)
        with pytest.raises(Crash):
            await rd.index(client, _other_document(tmp_path))

        report = await reingest_report.build_report(client)
        assert set(report["deals"]) == {rd.DEAL}
        assert [r["doc_id"] for r in report["deals"][rd.DEAL]["documents"]] == [kept["doc_id"]]
        [row] = report["staged"]
        assert row["deal_id"] == rd.DEAL and row["doc_id"] != kept["doc_id"]
        assert row["children"] == 4 and row["parents"] == 0 and row["source_file"] == "other.txt"
        assert "staged, never committed: 1 documents" in reingest_report.format_report(report)

        assert (await reingest_report.build_report(client, deal_id="another-deal"))["staged"] == []
        assert len((await reingest_report.build_report(client, deal_id=rd.DEAL))["staged"]) == 1


# ==============================================================================
# Concurrent uploads of the same bytes (the same point ids)
# ==============================================================================


class TestConcurrentUploads:
    @pytest.mark.asyncio
    async def test_the_second_waits_and_reindexes_in_place(self, tmp_path, small_batches):
        """Staged side by side, the later upload's upserts would hide the earlier one's commit."""
        client = await rd.new_client()
        path = rd.write_supply_agreement(tmp_path)
        real = client.upsert
        hidden: list[str] = []

        async def upsert(collection_name, points, **kwargs):
            await asyncio.sleep(0.01)                           # let the other upload run
            result = await real(collection_name=collection_name, points=points, **kwargs)
            if await readable(client) and any(STAGED_DEAL_FIELD in p.payload for p in points):
                hidden.append(collection_name)                  # staged over a committed copy
            return result
        client.upsert = upsert

        first, second = await asyncio.gather(rd.index(client, path), rd.index(client, path))
        assert sorted([first["previously_indexed"], second["previously_indexed"]]) == [False, True]
        assert hidden == []
        client.__dict__.pop("upsert", None)
        fresh_children, fresh_parents = await rd.fresh(path)
        assert rd.same(await rd.points(client), fresh_children)
        assert rd.same(await rd.points(client, PARENT_COLLECTION_NAME), fresh_parents)
        assert ip._DOCUMENT_LOCKS == {}

    @pytest.mark.asyncio
    async def test_different_documents_do_not_wait_for_each_other(self, tmp_path):
        client = await rd.new_client()
        a, b = await asyncio.gather(rd.index(client, rd.write_supply_agreement(tmp_path)),
                                    rd.index(client, _other_document(tmp_path)))
        assert a["doc_id"] != b["doc_id"] and not a["previously_indexed"] and not b["previously_indexed"]

    @pytest.mark.asyncio
    async def test_cleanup_leaves_the_marker_on_a_point_staged_again_elsewhere(self, tmp_path, small_batches):
        """Another process (outside this one's lock) re-staged a point after the commit."""
        client = await rd.new_client()
        result = await rd.index(client, rd.write_supply_agreement(tmp_path))
        children = await rd.points(client)
        point_id, (payload, vector) = next(iter(children.items()))
        from qdrant_client.models import PointStruct
        await client.upsert(COLLECTION_NAME, points=[PointStruct(
            id=point_id, vector=vector, payload=ip._staged_payload(payload, rd.DEAL))])

        class Written:
            def __init__(self, ids):
                self.id = ids
        await ip._drop_staging_leftovers(client, rd.DEAL, result["doc_id"],
                                         [(COLLECTION_NAME, [Written(i) for i in children])])
        payload_after = (await rd.points(client))[point_id][0]
        assert payload_after.get(STAGED_DEAL_FIELD) == rd.DEAL and "deal_id" not in payload_after


class TestBackfill:
    @pytest.mark.asyncio
    async def test_identity_backfill_ignores_staged_points(self, tmp_path, small_batches):
        from src.vector_db.identity_backfill import backfill_document_identity

        client = await rd.new_client()
        await rd.index(client, rd.write_supply_agreement(tmp_path, "kept.txt"))
        crash_child_batch(client, 3)
        with pytest.raises(Crash):
            await rd.index(client, _other_document(tmp_path))
        before = await rd.points(client)

        stats = await backfill_document_identity(client)
        assert stats["documents"] == 1
        assert {i: p for i, (p, _) in (await rd.points(client)).items() if STAGED_DEAL_FIELD in p} ==             {i: p for i, (p, _) in before.items() if STAGED_DEAL_FIELD in p}


def _other_document(directory):
    path = directory / "other.txt"
    path.write_text(rd.supply_agreement().replace("Supply", "Distribution"), encoding="utf-8")
    return path
