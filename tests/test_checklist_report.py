"""
The checklist review grid (Decision 95): read from a vector store as a question
would read the deal, one row per contract and one column per checklist item,
each cell a verbatim window and a status; bounded by a document cap and a time
budget, cached while the deal is unchanged, and served by
GET /deals/{id}/checklist under the same read rules as every other deal read.

Offline: an in-memory Qdrant, hand-made payloads, a stand-in reranker and
question encoder, no models.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

import api.main as api_main
import api.routes.deals as deals
import platform_stubs
import src.vector_db.qdrant_client as qdrant_module
import src.vector_db.reranker as reranker_module
from api import security
from platform_stubs import ADMIN_KEY, DEMO_DEAL
from src.reports import checklist
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, VECTOR_SIZE

platform_environment = platform_stubs.platform_environment

ADMIN = {security.ADMIN_HEADER: ADMIN_KEY}
PRIVATE_DEAL = "project_falcon_2025"
UNIT = [1.0] + [0.0] * (VECTOR_SIZE - 1)
SPARSE = SparseVector(indices=[7], values=[1.0])

ASSIGNMENT = ("Neither party may assign this Agreement without the prior written consent of the other "
              "party. Any assignment in breach of this Section is void.")
GOVERNING_LAW = "This Agreement is governed by the laws of the State of Delaware."
FILLER = "The Supplier shall deliver the Products to the Customer's warehouse each month."


# ==============================================================================
# Windows
# ==============================================================================


class TestWindows:
    def test_windows_are_verbatim_bounded_and_cover_the_text(self):
        text = " ".join(f"Sentence number {i} says something about the contract." for i in range(60))
        windows = checklist.sentence_windows(text, max_chars=200, max_windows=100)
        assert windows and all(0 < end - start <= 200 for start, end in windows)
        assert all(not text[s].isspace() and not text[e - 1].isspace() for s, e in windows)
        covered = set()
        for start, end in windows:
            covered.update(range(start, end))
        assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())

    def test_consecutive_windows_overlap_so_a_clause_is_whole_in_one(self):
        text = " ".join(f"Clause {i} is a sentence of about forty characters." for i in range(30))
        windows = checklist.sentence_windows(text, max_chars=300, max_windows=100)
        assert all(b[0] < a[1] for a, b in zip(windows, windows[1:]))
        assert all(b[0] > a[0] for a, b in zip(windows, windows[1:]))

    def test_a_sentence_longer_than_a_window_is_cut_at_spaces(self):
        text = "word " * 400
        windows = checklist.sentence_windows(text.strip(), max_chars=100, max_windows=100)
        assert all(end - start <= 100 for start, end in windows)
        assert all(text[s:e].split() == ["word"] * len(text[s:e].split()) for s, e in windows)

    def test_a_run_without_spaces_is_cut_hard(self):
        windows = checklist.sentence_windows("x" * 250, max_chars=100, max_windows=100)
        assert [e - s for s, e in windows] == [100, 100, 50]

    def test_blank_text_has_no_window(self):
        assert checklist.sentence_windows("") == []
        assert checklist.sentence_windows("   \n  ") == []

    def test_windows_stop_at_the_cap(self):
        text = " ".join(f"Sentence {i}." for i in range(500))
        assert len(checklist.sentence_windows(text, max_chars=60, max_windows=5)) == 5


# ==============================================================================
# The grid from a vector store
# ==============================================================================


def _payload(deal: str, source: str, text: str, index: int = 0, **extra) -> dict:
    return {
        "deal_id": deal, "doc_id": f"doc-{source}", "chunk_id": f"{deal}_{source}_{index:04d}",
        "source_file": source, "text": text, "chunk_index": index, "is_current_version": 1,
        "contains_pii": 0, "content_type": "text", "page_number": 4, "section_heading": "Assignment",
        "document_category": "legal", **extra,
    }


def _store(payloads: list[dict]) -> AsyncQdrantClient:
    async def build():
        client = AsyncQdrantClient(location=":memory:")
        await setup_collections(client)
        await client.upsert(COLLECTION_NAME, points=[
            PointStruct(id=i, payload=p, vector={"dense": UNIT, "sparse": SPARSE}) for i, p in enumerate(payloads)
        ])
        return client

    return asyncio.run(build())


def _room(deal: str = DEMO_DEAL) -> list[dict]:
    return [
        _payload(deal, "supply_agreement.txt", f"{FILLER} {ASSIGNMENT}", 0),
        _payload(deal, "supply_agreement.txt", GOVERNING_LAW, 1, section_heading="Governing Law"),
        _payload(deal, "licence_agreement.pdf", FILLER, 0, page_number=2),
        # Filed as financial by the filename rules: no row, but named — parties
        # read from an opening are no contract signal (a statement names its
        # company and its auditor).
        _payload(deal, "Acme_10-Q_EX-10.1_Distributor Agreement.txt", ASSIGNMENT, 0,
                 document_category="financial", doc_parties=["Acme Inc.", "Beta LLC"]),
        # Not a contract: no row, listed as excluded.
        _payload(deal, "financials_fy2023.txt", "Revenue grew 12% in FY2023.", 0, document_category="financial",
                 doc_parties=["AURORA TECHNOLOGIES, INC.", "Deloitte & Touche, LLP"]),
        # What the grid must never read or show.
        _payload(deal, "draft_supply_agreement.txt", ASSIGNMENT, 0, is_current_version=0),
        _payload(deal, "payroll.txt", ASSIGNMENT, 0, contains_pii=1),
        _payload("other_deal", "other.txt", ASSIGNMENT, 0),
        _payload(deal, "licence_agreement.pdf", f"Narrative: {ASSIGNMENT}", 1, content_type="table_narrative"),
    ]


def _score(question: str, passage: str) -> float:
    """A stand-in cross-encoder: the clause of the question's item scores high."""
    if "assigned" in question and "prior written consent" in passage:
        return 0.93
    if "governs" in question and "governed by the laws" in passage:
        return 0.88
    if "Narrative:" in passage:
        return 0.99  # would win if the grid ever read a generated representation
    return 0.004


@pytest.fixture
def models(monkeypatch):
    """The question encoder, the reranker and the thresholds, with no model loaded."""
    checklist.clear_cache()
    seen: list[list[tuple[str, str]]] = []

    async def encode(questions):
        return [(UNIT, SPARSE) for _ in questions]

    async def rerank(pairs):
        seen.append(list(pairs))
        return np.asarray([_score(q, p) for q, p in pairs], dtype=np.float32)

    monkeypatch.setattr(checklist, "_encode_questions", encode)
    monkeypatch.setattr(reranker_module, "rerank_pairs_async", rerank)
    # The identity calibration: a cell's probabilities are its score.
    monkeypatch.setitem(checklist.STATUS_MODELS, "bge-reranker-v2-m3",
                        checklist.StatusModel(IDENTITY, IDENTITY, absent_below=0.01, found_from=0.5))
    monkeypatch.setattr(checklist, "ANSWERABILITY_MODELS", {})
    yield seen
    checklist.clear_cache()


IDENTITY = checklist.Calibration(intercept=0.0, score_weight=1.0, item_offsets={})


ITEMS = ["anti_assignment", "governing_law"]


def _cells(report) -> dict[tuple[str, str], checklist.ChecklistCell]:
    return {(row.source_file, cell.item_id): cell for row in report.rows for cell in row.cells}


class TestGrid:
    def test_rows_are_the_contracts_a_question_may_read(self, models):
        report = asyncio.run(checklist.build_checklist(DEMO_DEAL, client=_store(_room()), item_ids=ITEMS))
        assert [row.source_file for row in report.rows] == ["licence_agreement.pdf", "supply_agreement.txt"]
        assert [(d.source_file, d.document_category) for d in report.documents_excluded] == [
            ("Acme_10-Q_EX-10.1_Distributor Agreement.txt", "financial"), ("financials_fy2023.txt", "financial")]
        assert report.documents_total == 4 and report.documents_checked == 2 and report.excluded_total == 2
        assert report.truncated is False and report.truncation_reason == ""
        shown = " ".join(c.passage.text for c in _cells(report).values() if c.passage)
        assert "Narrative:" not in shown

    def test_documents_the_caller_names_are_rows_whatever_their_category(self, models):
        report = asyncio.run(checklist.build_checklist(
            DEMO_DEAL, client=_store(_room()), item_ids=ITEMS,
            doc_ids=["doc-Acme_10-Q_EX-10.1_Distributor Agreement.txt", "doc-supply_agreement.txt"]))
        assert [row.source_file for row in report.rows] == [
            "Acme_10-Q_EX-10.1_Distributor Agreement.txt", "supply_agreement.txt"]
        assert report.documents_excluded == [] and report.documents_total == 2
        assert _cells(report)[("Acme_10-Q_EX-10.1_Distributor Agreement.txt", "anti_assignment")].status == "found"

    def test_each_cell_is_a_verbatim_window_with_its_place_and_status(self, models):
        report = asyncio.run(checklist.build_checklist(DEMO_DEAL, client=_store(_room()), item_ids=ITEMS))
        cells = _cells(report)
        assignment = cells[("supply_agreement.txt", "anti_assignment")]
        assert assignment.status == "found" and assignment.score == pytest.approx(0.93)
        # The chunk fits one window: the passage is the chunk, verbatim.
        assert assignment.passage.text == f"{FILLER} {ASSIGNMENT}"
        assert assignment.passage.section_heading == "Assignment" and assignment.passage.page_number == 4
        assert cells[("supply_agreement.txt", "governing_law")].status == "found"
        assert cells[("licence_agreement.pdf", "anti_assignment")].status == "not_found"
        assert cells[("licence_agreement.pdf", "anti_assignment")].passage.page_number == 2
        assert [(c.id, c.found, c.unclear, c.not_found) for c in report.items] == [
            ("anti_assignment", 1, 0, 1), ("governing_law", 1, 0, 1)]
        assert report.reranker == "bge-reranker-v2-m3" and report.answerability is False

    def test_the_status_bands(self):
        bands = checklist.StatusModel(IDENTITY, IDENTITY, absent_below=0.1, found_from=0.6)
        assert [checklist.cell_status("x", s, bands) for s in (0.05, 0.1, 0.59, 0.6, None)] == [
            "not_found", "unclear", "unclear", "found", "unclear"]
        never = checklist.StatusModel(IDENTITY, IDENTITY, absent_below=0.1, found_from=None)
        assert checklist.cell_status("x", 0.99, never) == "unclear"

    def test_an_item_s_offset_moves_its_scale(self):
        # Scores are not on one scale across questions: an item whose clauses
        # score low needs a lower score to count as found.
        low = checklist.Calibration(intercept=0.0, score_weight=1.0, item_offsets={"governing_law": 4.0})
        model = checklist.StatusModel(low, low, absent_below=0.1, found_from=0.6)
        assert checklist.cell_status("governing_law", 0.05, model) == "found"
        assert checklist.cell_status("change_of_control", 0.05, model) == "not_found"

    def test_every_reranker_profile_has_a_status_model_for_every_item(self):
        from src.decisions.reranker_calibration import PROFILES

        ids = {i.id for i in checklist.CHECKLIST}
        assert len(ids) == len(checklist.CHECKLIST) == 15
        assert {p.name for p in PROFILES} <= set(checklist.STATUS_MODELS)
        for model in [*checklist.STATUS_MODELS.values(), *checklist.ANSWERABILITY_MODELS.values()]:
            assert set(model.present.item_offsets) == ids and set(model.correct.item_offsets) == ids
            assert set(model.absent_below) <= ids and all(0.0 < v < 1.0 for v in model.absent_below.values())
            # Too few such clauses among the dev contracts to know what their absence looks like.
            assert {"most_favoured_nation", "no_solicit_of_customers", "liquidated_damages"}.isdisjoint(
                model.absent_below)
            assert {"governing_law", "anti_assignment", "cap_on_liability"} <= set(model.absent_below)
        assert all(m.reads_answerability for m in checklist.ANSWERABILITY_MODELS.values())

    def test_found_is_withdrawn_where_the_unseen_contracts_said_so(self):
        # Decision 95's pre-registered consequence: bge's found cells were right
        # for 24 of 31 unseen-contract cells (with Laya), under the 85% fixed in
        # advance; MiniLM's for 6 of 6.
        assert checklist.STATUS_MODELS["bge-reranker-v2-m3"].found_from is None
        assert checklist.ANSWERABILITY_MODELS["bge-reranker-v2-m3"].found_from is None
        assert checklist.STATUS_MODELS["ms-marco-MiniLM-L-6-v2"].found_from is not None

    def test_an_item_without_a_bar_is_never_not_found(self):
        model = checklist.StatusModel(IDENTITY, IDENTITY, absent_below={"governing_law": 0.1}, found_from=0.6)
        assert checklist.cell_status("governing_law", 0.01, model) == "not_found"
        assert checklist.cell_status("most_favoured_nation", 0.0001, model) == "unclear"
        assert checklist.cell_status("most_favoured_nation", 0.9, model) == "found"

    def test_the_rerank_reads_every_cell_s_windows_in_one_call_per_batch(self, models):
        asyncio.run(checklist.build_checklist(DEMO_DEAL, client=_store(_room()), item_ids=ITEMS))
        assert len(models) == 1
        assert all(p.startswith("Document: ") for _, p in models[0])

    def test_the_filter_is_the_one_questions_read_through(self):
        from src.vector_db.hybrid_search import deal_read_filter

        scoped = checklist._passage_filter(DEMO_DEAL, ["doc-a"])
        assert scoped.must[: len(deal_read_filter(DEMO_DEAL).must)] == deal_read_filter(DEMO_DEAL).must

    def test_items_are_a_subset_in_checklist_order_and_unknown_ones_are_refused(self, models):
        assert [i.id for i in checklist.resolve_items(["governing_law", "anti_assignment"])] == [
            "anti_assignment", "governing_law"]
        assert checklist.resolve_items(None) == checklist.CHECKLIST
        with pytest.raises(checklist.UnknownChecklistItem):
            checklist.resolve_items(["anti_assignment", "drop table"])

    def test_the_document_cap_checks_whole_rows_and_names_the_rest(self, models):
        report = asyncio.run(checklist.build_checklist(
            DEMO_DEAL, client=_store(_room()), item_ids=ITEMS, document_cap=1))
        assert report.truncated and report.truncation_reason == "document_cap"
        assert report.documents_checked == 1
        assert report.documents_not_checked == ["supply_agreement.txt"]

    def test_the_time_budget_stops_between_batches(self, models, monkeypatch):
        # One row a batch; the budget is spent once the first batch is reranked.
        monkeypatch.setattr(checklist, "ROWS_PER_BATCH", 1)
        report = asyncio.run(checklist.build_checklist(
            DEMO_DEAL, client=_store(_room()), item_ids=ITEMS, budget_s=10,
            clock=lambda: 100.0 if models else 0.0))
        assert report.truncated and report.truncation_reason == "time_budget"
        assert report.documents_checked == 1 and len(report.rows[0].cells) == 2
        assert report.documents_not_checked == ["supply_agreement.txt"]

    def test_a_store_that_fails_is_an_error_not_an_empty_grid(self, models):
        class Broken:
            async def facet(self, **kwargs):
                raise RuntimeError("connection refused at http://10.0.0.7:6333")

        with pytest.raises(RuntimeError):
            asyncio.run(checklist.build_checklist(DEMO_DEAL, client=Broken(), item_ids=ITEMS))

    def test_a_reranker_that_fails_is_an_error(self, models, monkeypatch):
        async def broken(pairs):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(reranker_module, "rerank_pairs_async", broken)
        with pytest.raises(RuntimeError):
            asyncio.run(checklist.build_checklist(DEMO_DEAL, client=_store(_room()), item_ids=ITEMS))

    def test_the_response_is_bounded(self, models):
        long_name = "Master Services Agreement " * 20 + ".txt"
        long_text = " ".join(["Neither party may assign this Agreement without the prior written consent"] * 40)
        room = [_payload(DEMO_DEAL, long_name, long_text, 0, section_heading="Heading " * 40)]
        report = asyncio.run(checklist.build_checklist(DEMO_DEAL, client=_store(room), item_ids=ITEMS))
        (row,) = report.rows
        assert len(row.source_file) <= checklist.MAX_SOURCE_CHARS
        for cell in row.cells:
            assert len(cell.passage.text) <= checklist.MAX_PASSAGE_CHARS
            assert len(cell.passage.section_heading) <= checklist.MAX_SECTION_CHARS
            assert len(cell.alternates) <= checklist.ALTERNATES
            assert all(len(a.text) <= checklist.MAX_ALTERNATE_CHARS for a in cell.alternates)
        found, absent = row.cells  # anti_assignment, governing_law
        assert found.status == "found" and len(found.alternates) == checklist.ALTERNATES
        # A not-found cell keeps only the opening of its closest passage.
        assert absent.status == "not_found" and absent.alternates == []
        assert len(absent.passage.text) <= checklist.MAX_ALTERNATE_CHARS

    def test_alternates_do_not_repeat_the_evidence(self):
        payload = {"chunk_id": "c1", "text": "x" * 100}
        other = {"chunk_id": "c2", "text": "y" * 100}
        windows = [checklist.ScoredWindow(payload, 0, 60, score=0.9),
                   checklist.ScoredWindow(payload, 30, 90, score=0.8),
                   checklist.ScoredWindow(other, 0, 60, score=0.7),
                   checklist.ScoredWindow(payload, 60, 100, score=0.6)]
        chosen = checklist._alternates(windows)
        assert [(w.payload["chunk_id"], w.start) for w in chosen] == [("c2", 0), ("c1", 60)]

    def test_a_document_with_nothing_retrieved_is_unclear_not_absent(self, models, monkeypatch):
        async def nothing(*args, **kwargs):
            return {}

        monkeypatch.setattr(checklist, "search_candidates", nothing)
        report = asyncio.run(checklist.build_checklist(DEMO_DEAL, client=_store(_room()), item_ids=ITEMS))
        assert {c.status for c in _cells(report).values()} == {"unclear"}
        assert all(c.passage is None for c in _cells(report).values())


WITH_LAYA = checklist.Calibration(intercept=0.0, score_weight=1.0, item_offsets={}, answerability_weight=1.0)


class TestAnswerability:
    def _blend(self, monkeypatch):
        model = checklist.StatusModel(WITH_LAYA, WITH_LAYA, absent_below=0.2, found_from=0.9,
                                      reads_answerability=True)
        monkeypatch.setitem(checklist.ANSWERABILITY_MODELS, "bge-reranker-v2-m3", model)
        monkeypatch.setenv("CHECKLIST_ANSWERABILITY", "1")
        monkeypatch.setenv("LAYA_ENABLED", "1")

    def test_laya_s_answerability_sets_the_status_when_it_is_read(self, models, monkeypatch):
        self._blend(monkeypatch)
        read = []

        async def laya(pairs):
            read.append(pairs)
            return [0.95 for _ in pairs]

        monkeypatch.setattr(checklist, "read_answerability", laya)
        report = asyncio.run(checklist.build_checklist(DEMO_DEAL, client=_store(_room()), item_ids=ITEMS))
        assert report.answerability is True and len(read) == 1
        cell = _cells(report)[("supply_agreement.txt", "anti_assignment")]
        assert cell.answerability == pytest.approx(0.95) and cell.status == "found"

    def test_laya_unavailable_falls_back_to_the_reranker_s_thresholds(self, models, monkeypatch):
        from src.decisions.laya_client import LayaUnavailable

        self._blend(monkeypatch)

        async def unavailable(pairs):
            raise LayaUnavailable("LAYA_ENABLED=0")

        monkeypatch.setattr(checklist, "read_answerability", unavailable)
        report = asyncio.run(checklist.build_checklist(DEMO_DEAL, client=_store(_room()), item_ids=ITEMS))
        assert report.answerability is False
        assert _cells(report)[("supply_agreement.txt", "anti_assignment")].status == "found"
        assert all(c.answerability is None for c in _cells(report).values())

    def test_the_cpu_host_does_not_read_laya_by_default(self, monkeypatch):
        import src.decisions.laya_client as laya_client

        blend = checklist.StatusModel(WITH_LAYA, WITH_LAYA, 0.2, 0.9, reads_answerability=True)
        monkeypatch.delenv("CHECKLIST_ANSWERABILITY", raising=False)
        monkeypatch.setenv("LAYA_ENABLED", "1")
        monkeypatch.setattr(laya_client, "laya_device", lambda: "cpu")
        assert checklist._use_answerability(blend) is False
        monkeypatch.setattr(laya_client, "laya_device", lambda: "cuda")
        assert checklist._use_answerability(blend) is True
        assert checklist._use_answerability(None) is False
        monkeypatch.setenv("CHECKLIST_ANSWERABILITY", "0")
        assert checklist._use_answerability(blend) is False

    def test_the_calibration_is_monotone_in_both_scores(self):
        blend = checklist.Calibration(intercept=-1.0, score_weight=2.0, item_offsets={"x": 0.5},
                                      answerability_weight=1.0)
        low, high = blend.probability("x", 0.1, 0.5), blend.probability("x", 0.9, 0.5)
        assert 0.0 <= low < high <= 1.0
        assert blend.probability("x", 0.5, 0.2) < blend.probability("x", 0.5, 0.8)
        # Without Laya's reading the answerability term is left out, not taken as 0.
        assert blend.probability("x", 0.5) == pytest.approx(1 / (1 + math.exp(-(-1.0 + 0.5))))


class TestCache:
    def test_an_unchanged_deal_is_answered_from_the_cache(self, models, monkeypatch):
        client = _store(_room())
        builds = []
        real = checklist.build_checklist

        async def counting(deal_id, client=None, item_ids=None):
            builds.append((deal_id, tuple(item_ids or ())))
            return await real(deal_id, client=client, item_ids=item_ids)

        monkeypatch.setattr(checklist, "build_checklist", counting)
        first = asyncio.run(checklist.get_checklist(DEMO_DEAL, ITEMS, client=client))
        second = asyncio.run(checklist.get_checklist(DEMO_DEAL, ITEMS, client=client))
        assert len(builds) == 1 and first is second
        # Another item set is another grid.
        asyncio.run(checklist.get_checklist(DEMO_DEAL, ["governing_law"], client=client))
        assert len(builds) == 2

        # An upload changes what the deal holds, and so the answer.
        asyncio.run(client.upsert(COLLECTION_NAME, points=[PointStruct(
            id=99, payload=_payload(DEMO_DEAL, "nda.txt", ASSIGNMENT), vector={"dense": UNIT, "sparse": SPARSE})]))
        third = asyncio.run(checklist.get_checklist(DEMO_DEAL, ITEMS, client=client))
        assert len(builds) == 3 and third.documents_checked == 3

        checklist.forget_deal(DEMO_DEAL)
        assert not checklist._cache

    def test_recording_a_document_drops_the_deal_s_cached_grids(self, monkeypatch):
        monkeypatch.setattr(deals, "_documents", {})
        monkeypatch.setattr(deals, "_deals", {})
        checklist._cache[(DEMO_DEAL, 5, ("governing_law",))] = (float("inf"), checklist.ChecklistReport(DEMO_DEAL))
        checklist._cache[(PRIVATE_DEAL, 5, ("governing_law",))] = (float("inf"), checklist.ChecklistReport(PRIVATE_DEAL))
        deals.register_document(DEMO_DEAL, "doc-new", "new.txt", "legal", 3, True, None)
        assert list(checklist._cache) == [(PRIVATE_DEAL, 5, ("governing_law",))]
        checklist.clear_cache()

    def test_a_build_that_never_frees_the_slot_is_a_busy_error(self, models):
        client = _store(_room())

        async def run():
            lock = checklist._build_lock()
            await lock.acquire()
            try:
                await checklist.get_checklist(DEMO_DEAL, ITEMS, client=client, wait_s=0.05)
            finally:
                lock.release()

        with pytest.raises(checklist.ChecklistBusy):
            asyncio.run(run())


# ==============================================================================
# The route
# ==============================================================================


@pytest.fixture
def store(monkeypatch, models):
    payloads = _room() + [_payload(PRIVATE_DEAL, "private_agreement.txt", ASSIGNMENT)]
    client = _store(payloads)
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: client)
    yield client


@pytest.fixture
def client():
    return TestClient(api_main.app)


def test_public_callers_read_the_public_deal_and_sandboxes_only(client, store, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    response = client.get(f"/api/v1/deals/{DEMO_DEAL}/checklist", params={"items": ",".join(ITEMS)})
    assert response.status_code == 200
    body = response.json()
    assert [c["id"] for c in body["items"]] == ITEMS
    assert body["documents_checked"] == 2 and body["truncated"] is False
    supply = next(r for r in body["rows"] if r["source_file"] == "supply_agreement.txt")
    assert supply["cells"][0]["status"] == "found"
    assert ASSIGNMENT in supply["cells"][0]["passage"]["text"]

    assert client.get(f"/api/v1/deals/{PRIVATE_DEAL}/checklist").status_code == 404
    admin = client.get(f"/api/v1/deals/{PRIVATE_DEAL}/checklist", params={"items": "anti_assignment"},
                       headers=ADMIN)
    assert admin.status_code == 200 and admin.json()["documents_checked"] == 1

    sandbox = deals.new_sandbox_id(datetime.now(timezone.utc))
    empty = client.get(f"/api/v1/deals/{sandbox}/checklist")
    assert empty.status_code == 200
    assert empty.json()["documents_total"] == 0 and empty.json()["rows"] == []
    assert len(empty.json()["items"]) == 15


def test_the_route_shares_the_read_rate_limit(client, store, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_READ_PER_MINUTE", "2")
    statuses = [
        client.get("/api/v1/deals").status_code,
        client.get(f"/api/v1/deals/{DEMO_DEAL}/checklist", params={"items": "governing_law"}).status_code,
        client.get(f"/api/v1/deals/{DEMO_DEAL}/checklist", params={"items": "governing_law"}).status_code,
    ]
    assert statuses == [200, 200, 429]


def test_a_malformed_deal_id_or_an_unknown_item_is_refused(client, store):
    assert client.get("/api/v1/deals/bad%20id/checklist").status_code == 422
    response = client.get(f"/api/v1/deals/{DEMO_DEAL}/checklist", params={"items": "governing_law,<script>"})
    assert response.status_code == 422
    assert "<script>" not in response.text and "governing_law" in response.json()["detail"]
    assert client.get(f"/api/v1/deals/{DEMO_DEAL}/checklist", params={"items": "x" * 700}).status_code == 422


def test_a_failing_store_is_a_503_that_leaks_nothing(client, monkeypatch):
    checklist.clear_cache()

    class Broken:
        async def count(self, **kwargs):
            raise RuntimeError("connection refused at http://10.0.0.7:6333 api_key=sk-live-123")

    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: Broken())
    response = client.get(f"/api/v1/deals/{DEMO_DEAL}/checklist")
    assert response.status_code == 503
    assert "10.0.0.7" not in response.text and "sk-live" not in response.text
    assert "request id" in response.json()["detail"]
    assert re.search(r"Checklist grid failed", response.json()["detail"])


def test_a_busy_build_slot_is_a_503_with_retry_after(client, store, monkeypatch):
    async def busy(deal_id, item_ids=None):
        raise checklist.ChecklistBusy(deal_id)

    monkeypatch.setattr(checklist, "get_checklist", busy)
    response = client.get(f"/api/v1/deals/{DEMO_DEAL}/checklist")
    assert response.status_code == 503 and response.headers["Retry-After"] == "30"


def test_purging_a_deal_forgets_its_cached_grids(client, store, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    assert client.get(f"/api/v1/deals/{DEMO_DEAL}/checklist", params={"items": "governing_law"}).status_code == 200
    assert any(key[0] == DEMO_DEAL for key in checklist._cache)

    async def no_checkpoints(graph, deal_id):
        return 0

    import src.workflow.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "purge_deal_checkpoints", no_checkpoints)
    monkeypatch.setattr(api_main, "get_graph", lambda: None)
    assert client.delete(f"/api/v1/deals/{DEMO_DEAL}", headers=ADMIN).status_code == 200
    assert not any(key[0] == DEMO_DEAL for key in checklist._cache)
