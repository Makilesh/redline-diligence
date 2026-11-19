"""
The startup check for documents the identity backfill would update
(Decision 88; the backfill is Decision 86's).

An index written before Decision 86 carries no doc_title / doc_parties, and
named-document focus then silently keeps the fused-score choice. Startup now
samples the index for such documents — reading their openings back and
running the extractor, since a spreadsheet lacks both fields legitimately —
reports them with the command that fixes it, and never blocks or fails.
In-memory Qdrant, stub vectors.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from src.utils import runtime_status
from src.vector_db import qdrant_client as qdrant_module
from src.vector_db.collection_manager import identity_index_status
from src.vector_db.constants import COLLECTION_NAME
from tests.fixtures import ingestion_docs as fx
from tests.test_fail_open import CONTRACT
from tests.test_ingestion_retrieval import _client, _index, _payloads

MINUTES = "minutes of a meeting, nothing more. " * 40


@pytest.fixture(autouse=True)
def _clean_status():
    runtime_status.reset_status()
    yield
    runtime_status.reset_status()


async def _room(tmp_path, *, backfilled: bool) -> tuple:
    """Two contracts and a document without identity; one contract's identity stripped unless backfilled."""
    client = await _client()
    old = await _index(client, fx.write_txt(tmp_path, "old_nda.txt", CONTRACT))
    await _index(client, fx.write_txt(tmp_path, "new_nda.txt", CONTRACT.replace("Gamma", "Delta")))
    await _index(client, fx.write_txt(tmp_path, "minutes.txt", MINUTES))
    if not backfilled:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        await client.delete_payload(
            collection_name=COLLECTION_NAME, keys=["doc_title", "doc_parties"],
            points=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=old["doc_id"]))]),
            wait=True,
        )
    return client, old


@pytest.mark.asyncio
async def test_an_index_that_predates_the_identity_fields(tmp_path):
    client, _ = await _room(tmp_path, backfilled=False)
    before = await _payloads(client)

    status = await identity_index_status(client)

    assert status == {"sampled_documents": 3, "without_identity": 2, "checked": 2,
                      "would_gain_identity": 1, "command": "python -m src.vector_db.identity_backfill"}
    # Read-only.
    assert await _payloads(client) == before


@pytest.mark.asyncio
async def test_a_document_without_identity_by_nature_is_not_reported(tmp_path):
    client, _ = await _room(tmp_path, backfilled=True)
    status = await identity_index_status(client)
    # The minutes name no agreement: no identity, and the backfill would add none.
    assert status["without_identity"] == 1 and status["would_gain_identity"] == 0


@pytest.mark.asyncio
async def test_the_check_is_sampled(tmp_path):
    client, _ = await _room(tmp_path, backfilled=False)
    status = await identity_index_status(client, sample_points=1, sample_documents=1)
    assert status["sampled_documents"] == 1 and status["checked"] <= 1


@pytest.mark.asyncio
async def test_an_empty_index():
    status = await identity_index_status(await _client())
    assert status["sampled_documents"] == 0 and status["would_gain_identity"] == 0


def test_the_result_is_a_degraded_mode_until_backfilled():
    runtime_status.set_identity_index({"checked": 4, "would_gain_identity": 3,
                                       "command": "python -m src.vector_db.identity_backfill"})
    detail = runtime_status.describe()["degraded_detail"]["identity_backfill_needed"]
    assert "3 of 4" in detail and "python -m src.vector_db.identity_backfill" in detail
    runtime_status.set_identity_index({"checked": 4, "would_gain_identity": 0, "command": "x"})
    assert "identity_backfill_needed" not in runtime_status.describe()["degraded"]
    runtime_status.set_identity_index(None)
    assert runtime_status.identity_index() is None


@pytest.mark.asyncio
async def test_startup_records_it_and_ready_reports_it(tmp_path, monkeypatch):
    client, _ = await _room(tmp_path, backfilled=False)
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: client)

    await api_main._check_identity_index()

    assert runtime_status.identity_index()["would_gain_identity"] == 1
    assert "identity_backfill_needed" in runtime_status.describe()["degraded"]
    monkeypatch.setenv("METRICS_PUBLIC", "1")
    monkeypatch.setattr(api_main, "_refresh_sparse_index_status", _noop)
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: _Reachable())
    body = TestClient(api_main.app).get("/ready").json()
    assert body["document_identity"]["would_gain_identity"] == 1
    assert "identity_backfill_needed" in body["degraded"]


@pytest.mark.asyncio
async def test_startup_survives_a_store_that_fails(monkeypatch):
    class Broken:
        async def scroll(self, **kwargs):
            raise ConnectionError("qdrant down")

    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: Broken())
    await api_main._check_identity_index()           # does not raise
    assert runtime_status.identity_index() is None
    assert "identity_backfill_needed" not in runtime_status.describe()["degraded"]


@pytest.mark.asyncio
async def test_startup_does_not_wait_on_a_slow_store(monkeypatch):
    async def slow(client=None, **kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr("src.vector_db.collection_manager.identity_index_status", slow)
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: None)
    monkeypatch.setattr(api_main, "_IDENTITY_CHECK_TIMEOUT_SECONDS", 0.05)
    await asyncio.wait_for(api_main._check_identity_index(), timeout=5)
    assert runtime_status.identity_index() is None


async def _noop(*args, **kwargs):
    return None


class _Reachable:
    async def get_collections(self):
        return []
