"""
The document list after a restart, once a new upload has been registered.

`_documents` holds only the uploads this process made. It used to be returned
alone whenever it was non-empty, so after a restart the first upload to a deal
hid every document indexed before it from the version browser and the risk
dashboard (both read `_deal_documents`), although retrieval still used them.
"""
from __future__ import annotations

import asyncio

import pytest

import api.routes.deals as deals

DEAL = "deal-restart"


def _payload(doc_id: str, filename: str, **extra) -> dict:
    return {"doc_id": doc_id, "source_file": filename, "document_category": "legal",
            "is_current_version": 1, **extra}


@pytest.fixture
def store(monkeypatch):
    """The vector store's chunk payloads for DEAL, and an empty registry (a fresh process)."""
    rows: list[dict] = []

    async def scroll(deal_id, fields):
        return [dict(r) for r in rows] if deal_id == DEAL else []

    monkeypatch.setattr(deals, "_scroll_deal_payloads", scroll)
    monkeypatch.setattr(deals, "_documents", {})
    monkeypatch.setattr(deals, "_deals", {})
    return rows


def _register(doc_id: str, filename: str, supersedes: str | None = None, signals=None) -> None:
    deals.register_document(
        deal_id=DEAL, doc_id=doc_id, filename=filename, document_category="legal",
        chunks_created=4, is_current_version=True, supersedes_doc_id=supersedes,
        risk_signals=signals,
    )


def _listed() -> dict[str, dict]:
    return {r["doc_id"]: r for r in asyncio.run(deals._deal_documents(DEAL))}


def test_an_upload_after_a_restart_keeps_the_documents_indexed_before_it(store):
    store += [_payload("old-1", "spa.pdf"), _payload("old-1", "spa.pdf"), _payload("old-2", "nda.pdf")]
    assert set(_listed()) == {"old-1", "old-2"}

    _register("new-1", "lease.pdf")

    listed = _listed()
    assert set(listed) == {"old-1", "old-2", "new-1"}
    assert listed["old-1"]["chunks_created"] == 2


def test_the_registry_record_wins_for_a_document_it_knows(store):
    store += [_payload("doc-1", "spa.pdf", risk_signals=["change_of_control"])]
    full = [{"signal_type": "change_of_control", "match_count": 3, "sample_matches": ["upon a Change of Control"]}]
    _register("doc-1", "spa.pdf", signals=full)

    [record] = _listed().values()
    assert record["upload_date"]  # a rebuild cannot recover it
    assert record["risk_signals"] == full
    assert record["chunks_created"] == 4


def test_a_rebuilt_document_superseded_by_a_new_upload_is_shown_superseded(store):
    store += [_payload("v1", "spa_v1.pdf")]
    _register("v2", "spa_v2.pdf", supersedes="v1")

    listed = _listed()
    assert listed["v1"]["is_current_version"] is False
    assert listed["v1"]["superseded_by"] == "v2"
    assert listed["v2"]["is_current_version"] is True and listed["v2"]["supersedes_doc_id"] == "v1"


def test_the_merge_does_not_change_the_registry(store):
    store += [_payload("v1", "spa_v1.pdf")]
    _register("v2", "spa_v2.pdf", supersedes="v1")
    _listed()
    assert [r["doc_id"] for r in deals._documents[DEAL]] == ["v2"]


def test_an_unreadable_store_still_lists_the_registry(store, monkeypatch):
    _register("new-1", "lease.pdf")

    async def unreadable(deal_id, fields):
        return []  # what _scroll_deal_payloads returns on any failure

    monkeypatch.setattr(deals, "_scroll_deal_payloads", unreadable)
    assert set(_listed()) == {"new-1"}


def test_the_risk_dashboard_counts_the_documents_from_before_the_restart(store, monkeypatch):
    store += [_payload("old-1", "spa.pdf", risk_signals=["change_of_control"])]
    _register("new-1", "lease.pdf", signals=[{"signal_type": "change_of_control", "match_count": 1, "sample_matches": []}])

    from fastapi.testclient import TestClient

    import api.main as api_main
    import api.security as security

    monkeypatch.setenv("ADMIN_API_KEY", "registry-test-key")
    body = TestClient(api_main.app).get(
        f"/api/v1/deals/{DEAL}/risk-signals", headers={security.ADMIN_HEADER: "registry-test-key"}
    )
    assert body.status_code == 200, body.text
    assert sorted(s["source_file"] for s in body.json()) == ["lease.pdf", "spa.pdf"]
