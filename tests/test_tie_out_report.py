"""
The deal-level figure tie-out report (Decision 92): read from a vector store
as a question would read the deal, bounded by a chunk cap and a time budget,
cached while the deal is unchanged, and served by GET /deals/{id}/tie-out
under the same read rules as every other deal read.

Offline: an in-memory Qdrant, hand-made payloads, no models.
"""

from __future__ import annotations

import asyncio
import itertools
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

import api.main as api_main
import api.routes.deals as deals
import platform_stubs
import src.vector_db.qdrant_client as qdrant_module
from api import security
from platform_stubs import ADMIN_KEY, DEMO_DEAL
from src.reports import tie_out
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, VECTOR_SIZE

platform_environment = platform_stubs.platform_environment

ADMIN = {security.ADMIN_HEADER: ADMIN_KEY}
PRIVATE_DEAL = "project_falcon_2025"


def statement(source: str, revenue: str, ebitda: str = "$92.8") -> dict:
    return {
        "source_file": source,
        "text": (
            "(in millions of USD)\n"
            "                     FY2023      FY2022\n"
            f"Revenue              {revenue}      $387.1\n"
            f"EBITDA               {ebitda}       $74.0\n"
        ),
    }


# ==============================================================================
# The report from a vector store
# ==============================================================================

UNIT = [1.0] + [0.0] * (VECTOR_SIZE - 1)


def _payload(deal: str, source: str, text: str, index: int = 0, **extra) -> dict:
    return {
        "deal_id": deal, "doc_id": f"doc-{source}", "source_file": source, "text": text,
        "chunk_index": index, "is_current_version": 1, "contains_pii": 0, "content_type": "table_text",
        "page_number": 3, "section_heading": "Income Statement", **extra,
    }


def _store(payloads: list[dict]) -> AsyncQdrantClient:
    async def build():
        client = AsyncQdrantClient(location=":memory:")
        await setup_collections(client)
        await client.upsert(COLLECTION_NAME, points=[
            PointStruct(id=i, payload=p, vector={"dense": UNIT}) for i, p in enumerate(payloads)
        ])
        return client

    return asyncio.run(build())


def _sample_room(deal: str = DEMO_DEAL) -> list[dict]:
    return [
        _payload(deal, "financials.txt", statement("x", "$452.8")["text"]),
        _payload(deal, "qoe.pdf", statement("x", "$452.8", ebitda="$99.0")["text"]),
        # What the report must never read.
        _payload(deal, "draft_financials.txt", statement("x", "$999.0")["text"], is_current_version=0),
        _payload(deal, "payroll.txt", statement("x", "$888.0")["text"], contains_pii=1),
        _payload("other_deal", "other.txt", statement("x", "$777.0")["text"]),
        _payload(deal, "qoe.pdf", "Revenue: FY2023: 1,000", content_type="table_narrative"),
    ]


class TestReport:
    def test_reads_what_a_question_may_read_and_nothing_else(self):
        report = asyncio.run(tie_out.build_tie_out(DEMO_DEAL, client=_store(_sample_room())))
        assert report.documents_total == report.documents_read == 2
        assert report.truncated is False and report.truncation_reason == ""
        assert [(f.metric, f.period) for f in report.conflicts] == [("EBITDA", "FY2023")]
        assert {(f.metric, f.period) for f in report.agreements} == {
            ("Revenue", "FY2023"), ("Revenue", "FY2022"), ("EBITDA", "FY2022")}
        values = report.conflicts[0].values
        assert {v.source_file for v in values} == {"financials.txt", "qoe.pdf"}
        assert all(v.page_number == 3 and v.section_heading == "Income Statement" for v in values)

    def test_payloads_only_never_vectors(self):
        client = _store(_sample_room())
        seen = []
        real_scroll = client.scroll

        async def scroll(**kwargs):
            seen.append(kwargs)
            return await real_scroll(**kwargs)

        client.scroll = scroll
        asyncio.run(tie_out.build_tie_out(DEMO_DEAL, client=client))
        assert seen and all(k["with_vectors"] is False for k in seen)
        assert all(set(k["with_payload"]) == set(tie_out.PAYLOAD_FIELDS) for k in seen)
        assert "contains_pii" not in tie_out.PAYLOAD_FIELDS

    def test_the_filter_is_the_one_questions_read_through(self):
        from src.vector_db.hybrid_search import _build_filter, deal_read_filter

        assert deal_read_filter(DEMO_DEAL).must == _build_filter(DEMO_DEAL, {}).must
        assert tie_out._deal_filter(DEMO_DEAL).must == deal_read_filter(DEMO_DEAL).must

    def test_a_deal_with_more_documents_than_one_listing_is_cut_short(self, monkeypatch):
        monkeypatch.setattr(tie_out, "MAX_DEAL_DOCUMENTS", 1)
        report = asyncio.run(tie_out.build_tie_out(DEMO_DEAL, client=_store(_sample_room())))
        assert report.truncated and report.truncation_reason == "chunk_cap"
        assert report.documents_total == report.documents_read == 1

    def test_the_chunk_cap_reads_whole_documents_and_names_the_rest(self):
        report = asyncio.run(tie_out.build_tie_out(DEMO_DEAL, client=_store(_sample_room()), chunk_cap=1))
        assert report.truncated and report.truncation_reason == "chunk_cap"
        assert report.documents_read == 1 and report.documents_not_read == ["qoe.pdf"]
        assert report.conflicts == [] and report.agreements == []

    def test_the_time_budget_stops_the_read(self):
        ticks = iter(range(1000))
        report = asyncio.run(tie_out.build_tie_out(
            DEMO_DEAL, client=_store(_sample_room()), budget_s=0.5, clock=lambda: next(ticks)))
        assert report.truncated and report.truncation_reason == "time_budget"
        assert report.documents_read < report.documents_total

    def test_a_budget_cut_keeps_the_documents_read_whole(self, monkeypatch):
        # One document per request; the clock passes the read's share of the
        # budget after the first, and stays inside the whole budget.
        monkeypatch.setattr(tie_out, "SCROLL_PAGE", 1)
        times = itertools.chain([0.0, 0.0], itertools.repeat(7.0))
        report = asyncio.run(tie_out.build_tie_out(
            DEMO_DEAL, client=_store(_sample_room()), budget_s=10, clock=lambda: next(times)))
        assert report.truncated and report.truncation_reason == "time_budget"
        assert report.documents_read == 1 and report.documents_not_read == ["qoe.pdf"]
        assert report.chunks_read == 1

    def test_a_read_that_ran_past_the_budget_is_still_compared(self, monkeypatch):
        # The first page took the whole budget and more; the comparison still
        # gets its share rather than discarding what was read.
        monkeypatch.setattr(tie_out, "SCROLL_PAGE", 1)
        times = itertools.chain([0.0, 0.0], itertools.repeat(50.0))
        report = asyncio.run(tie_out.build_tie_out(
            DEMO_DEAL, client=_store(_sample_room()), budget_s=10, clock=lambda: next(times)))
        assert report.truncation_reason == "time_budget"
        assert report.documents_read == 1 and report.chunks_read == 1

    def test_a_read_the_budget_cut_says_so_even_past_the_cap(self):
        ticks = iter(range(1000))
        report = asyncio.run(tie_out.build_tie_out(
            DEMO_DEAL, client=_store(_sample_room()), chunk_cap=1, budget_s=0.5, clock=lambda: next(ticks)))
        assert report.truncation_reason == "time_budget" and report.documents_read == 0

    def test_documents_are_read_a_page_at_a_time(self):
        documents = [("a", 3), ("b", 4), ("c", 9), ("d", 1), ("e", 2)]
        assert tie_out._batches(documents, 8) == [[("a", 3), ("b", 4)], [("c", 9)], [("d", 1), ("e", 2)]]

    def test_a_store_that_fails_is_an_error_not_an_empty_report(self):
        class Broken:
            async def facet(self, **kwargs):
                raise RuntimeError("connection refused at http://10.0.0.7:6333")

        with pytest.raises(RuntimeError):
            asyncio.run(tie_out.build_tie_out(DEMO_DEAL, client=Broken()))

    def test_labels_are_shown_without_table_furniture(self):
        assert tie_out._display_label("General and administrative.......................") == (
            "General and administrative")
        assert tie_out._display_label("Total assets  $") == "Total assets"
        assert tie_out._display_label("Principal payments  $  �  $") == "Principal payments"
        assert tie_out._display_label("Operating revenues—Other") == "Operating revenues—Other"

    def test_the_response_is_bounded(self, monkeypatch):
        monkeypatch.setattr(tie_out, "MAX_LISTED_AGREEMENTS", 2)
        # A converted table's row label can be any length the document prints.
        labels = ["A revenue line from " + "very " * 60 + "long contracts"] + [f"Segment revenue {i}" for i in range(4)]

        def workbook(source: str) -> dict:
            return {"source_file": source, "text": "", "structured_rows": [
                {"line_item": label, "values": {"FY2023": {
                    "raw": f"{i}.0", "raw_value": float(i), "normalized_value": float(i), "period": "FY2023"}}}
                for i, label in enumerate(labels)
            ]}

        report, read = tie_out.report_from_chunks(DEMO_DEAL, [workbook(f"doc{n}.xlsx") for n in range(12)])
        assert read == 12
        assert len(report.agreements) == 2 and report.agreements_total == 5
        assert report.agreements[0].metric.startswith("A revenue line")
        assert len(report.agreements[0].metric) <= tie_out.MAX_LABEL_CHARS
        assert all(len(f.values) == tie_out.MAX_VALUES_PER_FIGURE for f in report.agreements)
        assert all(f.more_documents == 2 for f in report.agreements)


class TestCache:
    def setup_method(self):
        tie_out.clear_cache()

    def test_an_unchanged_deal_is_answered_from_the_cache(self, monkeypatch):
        client = _store(_sample_room())
        builds = []
        real = tie_out.build_tie_out

        async def counting(deal_id, client=None):
            builds.append(deal_id)
            return await real(deal_id, client=client)

        monkeypatch.setattr(tie_out, "build_tie_out", counting)
        first = asyncio.run(tie_out.get_tie_out(DEMO_DEAL, client=client))
        second = asyncio.run(tie_out.get_tie_out(DEMO_DEAL, client=client))
        assert builds == [DEMO_DEAL] and first is second

        # An upload changes what the deal holds, and so the answer.
        asyncio.run(client.upsert(COLLECTION_NAME, points=[PointStruct(
            id=99, payload=_payload(DEMO_DEAL, "cim.pdf", statement("x", "$450.0")["text"]), vector={"dense": UNIT})]))
        third = asyncio.run(tie_out.get_tie_out(DEMO_DEAL, client=client))
        assert builds == [DEMO_DEAL, DEMO_DEAL] and third.documents_read == 3

        tie_out.forget_deal(DEMO_DEAL)
        assert not tie_out._cache

    def test_recording_a_document_drops_the_deal_s_cached_report(self, monkeypatch):
        monkeypatch.setattr(deals, "_documents", {})
        monkeypatch.setattr(deals, "_deals", {})
        tie_out._cache[(DEMO_DEAL, 5)] = (float("inf"), tie_out.TieOutReport(deal_id=DEMO_DEAL))
        tie_out._cache[(PRIVATE_DEAL, 5)] = (float("inf"), tie_out.TieOutReport(deal_id=PRIVATE_DEAL))
        deals.register_document(DEMO_DEAL, "doc-new", "new.txt", "financial", 3, True, None)
        assert list(tie_out._cache) == [(PRIVATE_DEAL, 5)]

    def test_a_build_that_never_frees_the_slot_is_a_busy_error(self):
        client = _store(_sample_room())

        async def run():
            lock = tie_out._build_lock()
            await lock.acquire()
            try:
                await tie_out.get_tie_out(DEMO_DEAL, client=client, wait_s=0.05)
            finally:
                lock.release()

        with pytest.raises(tie_out.TieOutBusy):
            asyncio.run(run())


# ==============================================================================
# The route
# ==============================================================================


@pytest.fixture
def store(monkeypatch):
    tie_out.clear_cache()
    payloads = _sample_room() + [
        _payload(PRIVATE_DEAL, "private.txt", statement("x", "$1.0")["text"]),
    ]
    client = _store(payloads)
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: client)
    yield client
    tie_out.clear_cache()


@pytest.fixture
def client():
    return TestClient(api_main.app)


def test_public_callers_read_the_public_deal_and_sandboxes_only(client, store, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    response = client.get(f"/api/v1/deals/{DEMO_DEAL}/tie-out")
    assert response.status_code == 200
    body = response.json()
    assert [c["metric"] for c in body["conflicts"]] == ["EBITDA"]
    assert body["conflicts"][0]["values"][0]["page_number"] == 3
    assert body["documents_read"] == 2 and body["truncated"] is False

    assert client.get(f"/api/v1/deals/{PRIVATE_DEAL}/tie-out").status_code == 404
    admin = client.get(f"/api/v1/deals/{PRIVATE_DEAL}/tie-out", headers=ADMIN)
    assert admin.status_code == 200 and admin.json()["documents_read"] == 1

    sandbox = deals.new_sandbox_id(datetime.now(timezone.utc))
    empty = client.get(f"/api/v1/deals/{sandbox}/tie-out")
    assert empty.status_code == 200 and empty.json()["documents_total"] == 0


def test_the_route_shares_the_read_rate_limit(client, store, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_READ_PER_MINUTE", "2")
    statuses = [
        client.get("/api/v1/deals").status_code,
        client.get(f"/api/v1/deals/{DEMO_DEAL}/tie-out").status_code,
        client.get(f"/api/v1/deals/{DEMO_DEAL}/tie-out").status_code,
    ]
    assert statuses == [200, 200, 429]


def test_a_malformed_deal_id_is_refused(client, store):
    assert client.get("/api/v1/deals/bad%20id/tie-out").status_code == 422


def test_a_failing_store_is_a_503_that_leaks_nothing(client, monkeypatch):
    tie_out.clear_cache()

    class Broken:
        async def count(self, **kwargs):
            raise RuntimeError("connection refused at http://10.0.0.7:6333 api_key=sk-live-123")

    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: Broken())
    response = client.get(f"/api/v1/deals/{DEMO_DEAL}/tie-out")
    assert response.status_code == 503
    assert "10.0.0.7" not in response.text and "sk-live" not in response.text
    assert "request id" in response.json()["detail"]


def test_a_busy_build_slot_is_a_503_with_retry_after(client, store, monkeypatch):
    async def busy(deal_id):
        raise tie_out.TieOutBusy(deal_id)

    monkeypatch.setattr(tie_out, "get_tie_out", busy)
    response = client.get(f"/api/v1/deals/{DEMO_DEAL}/tie-out")
    assert response.status_code == 503 and response.headers["Retry-After"] == "15"


def test_purging_a_deal_forgets_its_cached_report(store):
    asyncio.run(tie_out.get_tie_out(DEMO_DEAL, client=store))
    assert any(key[0] == DEMO_DEAL for key in tie_out._cache)
    tie_out.forget_deal(DEMO_DEAL)
    assert not any(key[0] == DEMO_DEAL for key in tie_out._cache)
