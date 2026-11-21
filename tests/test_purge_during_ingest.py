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

    async def fake_delete_document(deal_id, doc_id, **kwargs):
        state["deleted"].append((deal_id, doc_id))

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
    assert route["deleted"] == [(DEMO_DEAL, "doc-1")]
    assert DEMO_DEAL not in deals._documents
    assert not deals._open_ingests
    [audit_file] = route["audit"].iterdir()
    events = [json.loads(line) for line in audit_file.read_text().splitlines()]
    assert [e["event"] for e in events] == ["document_discarded"]
    assert events[0]["reason"] == "deal_purged_during_ingest"


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
