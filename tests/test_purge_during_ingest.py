"""
A purge that runs while a file is being indexed is not undone by that ingest.

The purge (tab-close beacon, admin DELETE, sandbox sweeper) deleted the vectors
written so far and forgot the deal; the ingest then wrote the rest, committed,
and registered the document again, so the "deleted" upload stayed readable.
Now the ingest's ticket is marked, and the upload is deleted and refused (409).
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import api.routes.deals as deals
import api.routes.ingest as ingest_routes
import api.security as security
import platform_stubs
from platform_stubs import ADMIN_KEY, DEMO_DEAL
from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME
from tests.fixtures import reingest_docs as rd

platform_environment = platform_stubs.platform_environment
default_bm25_is_a_stub = rd.default_bm25_is_a_stub
ADMIN = {security.ADMIN_HEADER: ADMIN_KEY}


# ==============================================================================
# The ticket
# ==============================================================================


def test_a_purge_marks_only_the_open_ingests_of_its_deal(monkeypatch):
    async def nothing(*a, **k):
        return 0

    monkeypatch.setattr(deals, "_delete_deal_vectors", nothing)
    import src.workflow.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "purge_deal_checkpoints", nothing)
    with deals.watch_for_purge("deal-a") as a, deals.watch_for_purge("deal-b") as b:
        asyncio.run(deals.purge_deal("deal-a"))
        assert a.purged and not b.purged
    assert "deal-a" not in deals._open_ingests and "deal-b" not in deals._open_ingests


def test_an_ingest_that_starts_after_a_purge_is_not_marked(monkeypatch):
    async def nothing(*a, **k):
        return 0

    monkeypatch.setattr(deals, "_delete_deal_vectors", nothing)
    import src.workflow.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "purge_deal_checkpoints", nothing)
    asyncio.run(deals.purge_deal("deal-a"))
    with deals.watch_for_purge("deal-a") as ticket:
        assert not ticket.purged


# ==============================================================================
# The route
# ==============================================================================


@pytest.fixture
def route(monkeypatch, tmp_path):
    """The ingest route with index_document stubbed; `purge_midway` purges the deal inside it."""
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "audit"))
    state = {"purge_midway": False, "deleted": []}

    async def nothing(*a, **k):
        return 0

    monkeypatch.setattr(deals, "_delete_deal_vectors", nothing)
    import src.workflow.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "purge_deal_checkpoints", nothing)

    async def fake_index_document(path, filename, deal_id, category, **kwargs):
        if state["purge_midway"]:
            await deals.purge_deal(deal_id)
        return {
            "doc_id": "doc-1", "document_category": "legal", "chunks_created": 9,
            "parent_chunks_created": 2, "previously_indexed": False, "warnings": [],
            "risk_signals": [], "suspected_injection": [],
        }

    async def fake_delete_document(deal_id, doc_id, unless=None, **kwargs):
        if state.get("delete_fails"):
            raise RuntimeError("qdrant unreachable")
        if unless is not None and unless():
            return False
        state["deleted"].append((deal_id, doc_id))
        return True

    monkeypatch.setattr(ingest_routes, "index_document", fake_index_document)
    monkeypatch.setattr(ingest_routes, "delete_document", fake_delete_document)
    monkeypatch.setattr(deals, "_documents", {})
    state["audit"] = tmp_path / "audit"
    return state


def _upload():
    return TestClient(api_main.app).post(
        "/api/v1/ingest",
        headers=ADMIN,
        data={"deal_id": DEMO_DEAL},
        files={"file": ("memo.txt", b"Revenue was $452.8 million.", "text/plain")},
    )


def test_an_upload_whose_deal_is_purged_meanwhile_is_deleted_and_refused(route):
    route["purge_midway"] = True

    response = _upload()

    assert response.status_code == 409
    assert "discarded" in response.json()["detail"]
    assert "requested" in response.json()["detail"]  # true also when the purge itself failed
    assert route["deleted"] == [(DEMO_DEAL, "doc-1")]
    assert DEMO_DEAL not in deals._documents
    assert not deals._open_ingests
    [audit_file] = route["audit"].iterdir()
    events = [json.loads(line) for line in audit_file.read_text().splitlines()]
    assert [e["event"] for e in events] == ["document_discarded"]
    assert events[0]["reason"] == "deal_purged_during_ingest"
    assert events[0]["points_deleted"] is True


def test_an_upload_with_no_purge_is_registered_as_before(route):
    response = _upload()

    assert response.status_code == 200, response.text
    assert route["deleted"] == []
    assert [r["doc_id"] for r in deals._documents[DEMO_DEAL]] == ["doc-1"]
    assert not deals._open_ingests


# ==============================================================================
# delete_document against a real (in-memory) Qdrant
# ==============================================================================


@pytest.mark.usefixtures("default_bm25_is_a_stub")
def test_delete_document_removes_committed_and_staged_points(tmp_path, monkeypatch):
    import src.data_processing.ingest_pipeline as ip

    async def run():
        client = await rd.new_client()
        kept = await rd.index(client, rd.write_supply_agreement(tmp_path, "other.txt"), name="other.txt")
        path = rd.write_supply_agreement(tmp_path)
        path.write_text(path.read_text() + "\nA second, different document.")
        gone = await rd.index(client, path)
        assert kept["doc_id"] != gone["doc_id"]

        # Its parents staged (marker, no deal_id), as an upload cut short leaves
        # them; its children committed.
        await client.set_payload(
            collection_name=PARENT_COLLECTION_NAME, payload={ip.STAGED_DEAL_FIELD: rd.DEAL},
            points=ip._doc_filter(rd.DEAL, gone["doc_id"]), wait=True,
        )
        await client.delete_payload(
            collection_name=PARENT_COLLECTION_NAME, keys=["deal_id"],
            points=ip._doc_filter(rd.DEAL, gone["doc_id"]),
            wait=True,
        )
        staged = [pl for pl, _ in (await rd.points(client, PARENT_COLLECTION_NAME)).values()
                  if pl.get(ip.STAGED_DEAL_FIELD) and "deal_id" not in pl]
        assert staged and {pl["doc_id"] for pl in staged} == {gone["doc_id"]}
        await ip.delete_document(rd.DEAL, gone["doc_id"], client=client)

        for collection in (COLLECTION_NAME, PARENT_COLLECTION_NAME):
            left = {pl["doc_id"] for pl, _ in (await rd.points(client, collection)).values()}
            assert left == {kept["doc_id"]}, collection
        await client.close()

    asyncio.run(run())


def test_a_discard_whose_delete_fails_is_reported_as_a_failure(route):
    route["purge_midway"] = True
    route["delete_fails"] = True

    response = _upload()

    assert response.status_code == 500
    assert response.headers[security.REQUEST_ID_HEADER] in response.json()["detail"]
    assert DEMO_DEAL not in deals._documents
    [audit_file] = route["audit"].iterdir()
    events = [json.loads(line) for line in audit_file.read_text().splitlines()]
    assert [e["event"] for e in events] == ["document_discard_failed"]
    text = TestClient(api_main.app).get("/metrics", headers=ADMIN).text
    assert 'redline_ingest_total{outcome="error"} 1' in text
    assert 'outcome="rejected"' not in text


# ==============================================================================
# A purge running when an ingest starts; the same file uploaded again since
# ==============================================================================


def test_an_ingest_that_starts_while_a_purge_runs_is_marked(monkeypatch):
    import src.workflow.orchestrator as orchestrator

    seen = {}

    async def slow_delete(deal_id):
        with deals.watch_for_purge(deal_id) as ticket:  # an upload arriving mid-purge
            seen["marked"] = ticket.purged
        await asyncio.sleep(0)

    async def nothing(*a, **k):
        return 0

    monkeypatch.setattr(deals, "_delete_deal_vectors", slow_delete)
    monkeypatch.setattr(orchestrator, "purge_deal_checkpoints", nothing)
    asyncio.run(deals.purge_deal("deal-a"))
    assert seen["marked"] is True
    assert "deal-a" not in deals._purges_running and "deal-a" not in deals._purges_idle
    with deals.watch_for_purge("deal-a") as after:
        assert not after.purged


def test_a_failed_purge_still_ends(monkeypatch):
    async def broken(deal_id):
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(deals, "_delete_deal_vectors", broken)
    with pytest.raises(RuntimeError):
        asyncio.run(deals.purge_deal("deal-a"))
    assert "deal-a" not in deals._purges_running and "deal-a" not in deals._purges_idle
    asyncio.run(deals.wait_for_purges("deal-a"))  # returns at once


@pytest.mark.usefixtures("default_bm25_is_a_stub")
@pytest.mark.parametrize("b_state", ["in_flight", "registered"])
def test_a_discard_keeps_the_same_file_uploaded_again_after_the_purge(tmp_path, monkeypatch, b_state):
    """The reviewer's sequence, with the real pipeline on an in-memory Qdrant:
    upload A is marked by a purge; upload B of the same bytes starts after it
    (not marked) and writes first. A's discard must not delete B's points."""
    import src.data_processing.ingest_pipeline as ip
    import src.workflow.orchestrator as orchestrator

    async def nothing(*a, **k):
        return 0

    monkeypatch.setattr(deals, "_delete_deal_vectors", nothing)
    monkeypatch.setattr(orchestrator, "purge_deal_checkpoints", nothing)
    monkeypatch.setattr(deals, "_documents", {})

    async def run():
        client = await rd.new_client()
        path = rd.write_supply_agreement(tmp_path)
        doc_id = ip.compute_doc_id(rd.DEAL, ip.compute_content_hash(str(path)))
        with deals.watch_for_purge(rd.DEAL) as a:
            a.doc_id = doc_id
            await rd.index(client, path)
            await deals.purge_deal(rd.DEAL)
            assert a.purged
            with deals.watch_for_purge(rd.DEAL) as b:
                b.doc_id = doc_id
                assert not b.purged
                result = await rd.index(client, path)
                assert result["previously_indexed"]
                if b_state == "registered":
                    deals.register_document(
                        deal_id=rd.DEAL, doc_id=doc_id, filename="supply.txt",
                        document_category="legal", chunks_created=result["chunks_created"],
                        is_current_version=True, supersedes_doc_id=None,
                    )
                deleted = await ip.delete_document(
                    rd.DEAL, doc_id, client=client,
                    unless=lambda: deals.uploaded_again_since_purge(rd.DEAL, doc_id, a),
                )
        assert deleted is False
        assert len(await rd.points(client)) == result["chunks_created"]
        assert len(await rd.points(client, PARENT_COLLECTION_NAME)) == result["parent_chunks_created"]
        await client.close()

    asyncio.run(run())


@pytest.mark.usefixtures("default_bm25_is_a_stub")
def test_a_discard_with_no_later_upload_deletes(tmp_path, monkeypatch):
    import src.data_processing.ingest_pipeline as ip

    monkeypatch.setattr(deals, "_documents", {})

    async def run():
        client = await rd.new_client()
        path = rd.write_supply_agreement(tmp_path)
        doc_id = ip.compute_doc_id(rd.DEAL, ip.compute_content_hash(str(path)))
        with deals.watch_for_purge(rd.DEAL) as a:
            a.doc_id = doc_id
            a.purged = True
            await rd.index(client, path)
            with deals.watch_for_purge(rd.DEAL) as other:  # a different file in flight
                other.doc_id = "another-document"
                deleted = await ip.delete_document(
                    rd.DEAL, doc_id, client=client,
                    unless=lambda: deals.uploaded_again_since_purge(rd.DEAL, doc_id, a),
                )
        assert deleted is True
        assert not await rd.points(client) and not await rd.points(client, PARENT_COLLECTION_NAME)
        await client.close()

    asyncio.run(run())


@pytest.mark.usefixtures("default_bm25_is_a_stub")
@pytest.mark.parametrize("b_ends", ["fails_before_writing", "registers"])
def test_a_discard_waits_for_an_in_flight_reupload_and_decides_again(tmp_path, monkeypatch, b_ends):
    """A is marked by a purge; B, the same bytes started after it, is still in
    flight when A discards. A must not decide from B's ticket alone: if B then
    ends without registering (cancelled or failed before it writes), A's points
    would be left with no record. A waits for B and decides again."""
    import functools

    from fastapi import HTTPException

    import src.data_processing.ingest_pipeline as ip
    import src.workflow.orchestrator as orchestrator

    async def nothing(*a, **k):
        return 0

    monkeypatch.setattr(deals, "_delete_deal_vectors", nothing)
    monkeypatch.setattr(orchestrator, "purge_deal_checkpoints", nothing)
    monkeypatch.setattr(deals, "_documents", {})
    monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "audit"))

    async def run():
        client = await rd.new_client()
        monkeypatch.setattr(ingest_routes, "delete_document",
                            functools.partial(ip.delete_document, client=client))
        path = rd.write_supply_agreement(tmp_path)
        doc_id = ip.compute_doc_id(rd.DEAL, ip.compute_content_hash(str(path)))
        with deals.watch_for_purge(rd.DEAL) as a:
            a.doc_id = doc_id
            result = await rd.index(client, path)
            await deals.purge_deal(rd.DEAL)
            b_scope = deals.watch_for_purge(rd.DEAL)
            b = b_scope.__enter__()
            b.doc_id = doc_id
            discard = asyncio.create_task(
                ingest_routes._discard_purged_upload(rd.DEAL, doc_id, "supply.txt", "req-a", a)
            )
            await asyncio.sleep(0.05)
            assert not discard.done()  # waiting for B
            assert len(await rd.points(client)) == result["chunks_created"]
            if b_ends == "registers":
                deals.register_document(
                    deal_id=rd.DEAL, doc_id=doc_id, filename="supply.txt",
                    document_category="legal", chunks_created=result["chunks_created"],
                    is_current_version=True, supersedes_doc_id=None,
                )
            b_scope.__exit__(None, None, None)
            with pytest.raises(HTTPException) as refused:
                await asyncio.wait_for(discard, 10)
            assert refused.value.status_code == 409
        left = len(await rd.points(client)) + len(await rd.points(client, PARENT_COLLECTION_NAME))
        await client.close()
        return left, result

    left, result = asyncio.run(run())
    if b_ends == "registers":
        assert left == result["chunks_created"] + result["parent_chunks_created"]
    else:
        assert left == 0
    [audit_file] = (tmp_path / "audit").iterdir()
    [event] = [json.loads(line) for line in audit_file.read_text().splitlines()]
    assert event["event"] == "document_discarded"
    assert event["points_deleted"] is (b_ends != "registers")


def test_the_ticket_carries_the_doc_id_index_document_derives(monkeypatch, tmp_path):
    """The route hashes the upload before indexing; the id must be the one the
    pipeline derives, or a discard could not recognise a re-upload in flight."""
    import src.data_processing.ingest_pipeline as ip

    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "audit"))
    monkeypatch.setattr(deals, "_documents", {})
    seen = {}

    async def fake_index_document(path, filename, deal_id, category, **kwargs):
        [ticket] = deals._open_ingests[deal_id]
        doc = ip.extract_and_chunk(path, filename, deal_id, category)
        seen.update(ticket=ticket.doc_id, pipeline=doc.doc_id)
        return {
            "doc_id": doc.doc_id, "document_category": "legal", "chunks_created": 1,
            "parent_chunks_created": 1, "previously_indexed": False, "warnings": [],
            "risk_signals": [], "suspected_injection": [],
        }

    monkeypatch.setattr(ingest_routes, "index_document", fake_index_document)
    assert _upload().status_code == 200
    assert seen["ticket"] and seen["ticket"] == seen["pipeline"]


@pytest.mark.usefixtures("default_bm25_is_a_stub")
def test_a_discard_during_the_purge_waits_for_it_before_deciding(tmp_path, monkeypatch):
    """A re-uploads a document registered before the purge. Its discard starts
    while the purge is still running, before the purge drops the registry: the
    old record must not be taken for an upload made since, or the delete is
    skipped and the purge undone."""
    import functools

    from fastapi import HTTPException

    import src.data_processing.ingest_pipeline as ip
    import src.workflow.orchestrator as orchestrator

    monkeypatch.setattr(deals, "_documents", {})
    monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "audit"))

    async def run():
        client = await rd.new_client()
        monkeypatch.setattr(ingest_routes, "delete_document",
                            functools.partial(ip.delete_document, client=client))
        release = asyncio.Event()

        async def slow_vector_delete(deal_id):
            await release.wait()

        async def nothing(*a, **k):
            return 0

        monkeypatch.setattr(deals, "_delete_deal_vectors", slow_vector_delete)
        monkeypatch.setattr(orchestrator, "purge_deal_checkpoints", nothing)

        path = rd.write_supply_agreement(tmp_path)
        doc_id = ip.compute_doc_id(rd.DEAL, ip.compute_content_hash(str(path)))
        first = await rd.index(client, path)
        deals.register_document(
            deal_id=rd.DEAL, doc_id=doc_id, filename="supply.txt", document_category="legal",
            chunks_created=first["chunks_created"], is_current_version=True, supersedes_doc_id=None,
        )
        with deals.watch_for_purge(rd.DEAL) as a:
            a.doc_id = doc_id
            await rd.index(client, path)  # A: the same file again
            purge = asyncio.create_task(deals.purge_deal(rd.DEAL))
            await asyncio.sleep(0)
            assert a.purged and rd.DEAL in deals._documents  # purge running, registry not yet dropped
            discard = asyncio.create_task(
                ingest_routes._discard_purged_upload(rd.DEAL, doc_id, "supply.txt", "req-a", a)
            )
            await asyncio.sleep(0.05)
            assert not discard.done()
            release.set()
            await purge
            with pytest.raises(HTTPException) as refused:
                await asyncio.wait_for(discard, 10)
            assert refused.value.status_code == 409
        left = len(await rd.points(client)) + len(await rd.points(client, PARENT_COLLECTION_NAME))
        await client.close()
        return left

    assert asyncio.run(run()) == 0
