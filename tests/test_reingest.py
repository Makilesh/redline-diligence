"""
Re-ingesting an existing index (Decision 97): the read-only report that says
which documents need what, and the driver that re-ingests them from their
original files.

In-memory Qdrant (a local store for the command lines), stub vectors, no
model weights. The "old code" and the failing clients are in
tests/fixtures/reingest_docs.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, SparseVector

import src.data_processing.ingest_pipeline as ip
from src.vector_db import reingest_report, reingest_run
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME, VECTOR_SIZE
from tests.fixtures import ingestion_docs as fx
from tests.fixtures import reingest_docs as rd
from tests.fixtures.reingest_docs import DEAL
from tests.fixtures.reingest_docs import default_bm25_is_a_stub  # noqa: F401  (fixture)

pytestmark = pytest.mark.usefixtures("default_bm25_is_a_stub")

FAILURES = {
    "child upsert batch 3": lambda client: rd.fail_child_batch(client, 3),
    "parent upsert": rd.fail_parents,
    "stale delete": rd.fail_delete,
}


async def _row(client, doc_id: str, files: Path | None = None) -> dict:
    report = await reingest_report.build_report(client, files=files)
    return next(r for r in report["deals"][DEAL]["documents"] if r["doc_id"] == doc_id)


async def _run(client, files: Path, **kwargs) -> dict:
    return await reingest_run.run(client, files, embed_fn=rd.stub_embed, backoff_s=0.0, **kwargs)


# ==============================================================================
# A re-index cut short, as the report sees it
# ==============================================================================


class TestInterruptedReindex:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", sorted(FAILURES))
    async def test_the_report_flags_it_until_the_file_is_indexed_again(self, failure, monkeypatch, tmp_path):
        monkeypatch.setattr(ip, "QDRANT_BATCH_SIZE", 5)
        path = rd.write_supply_agreement(tmp_path)
        client = await rd.new_client()
        old = await rd.old_index(monkeypatch, client, path)

        FAILURES[failure](client)
        with pytest.raises(ConnectionError):
            await rd.index(client, path)
        row = await _row(client, old["doc_id"])
        assert row["action"] == "reingest"
        assert "mixed_stamps" in row["problems"]
        assert row["ingest_version"] == [0, ip.INGEST_VERSION]

        rd.heal(client)
        await rd.index(client, path)
        row = await _row(client, old["doc_id"])
        assert (row["action"], row["problems"]) == ("nothing", [])

    @pytest.mark.asyncio
    async def test_new_children_meet_parents_the_old_run_wrote(self, monkeypatch, tmp_path):
        """Why the mixed state matters: a parent no longer holds the child that names it."""
        monkeypatch.setattr(ip, "QDRANT_BATCH_SIZE", 5)
        path = rd.write_supply_agreement(tmp_path)
        client = await rd.new_client()
        old = await rd.old_index(monkeypatch, client, path)
        rd.fail_parents(client)
        with pytest.raises(ConnectionError):
            await rd.index(client, path)
        row = await _row(client, old["doc_id"])
        assert {"child_not_held_by_parent", "parent_holds_other_children"} & set(row["problems"])

    @pytest.mark.asyncio
    async def test_a_reupload_under_another_name_cut_short_is_left_for_an_operator(self, monkeypatch, tmp_path):
        """Review S3: same bytes and stamp, new name and category; one doc_id, two names. Not "nothing"."""
        monkeypatch.setattr(ip, "QDRANT_BATCH_SIZE", 5)
        path = rd.write_supply_agreement(tmp_path)
        client = await rd.new_client()
        first = await rd.index(client, path, "Supply v1.txt")
        rd.fail_child_batch(client, 2)
        with pytest.raises(ConnectionError):
            await rd.index(client, path, "Master Supply Agreement (executed).txt", category="financial")
        rd.heal(client)
        row = await _row(client, first["doc_id"])
        assert row["action"] == "operator" and "mixed_document_fields" in row["problems"]
        assert row["conflicts"]["source_file"] == ["Master Supply Agreement (executed).txt", "Supply v1.txt"]

    @pytest.mark.asyncio
    async def test_with_the_chunking_unchanged_a_reindex_cut_short_needs_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ip, "QDRANT_BATCH_SIZE", 5)
        path = rd.write_supply_agreement(tmp_path)
        client = await rd.new_client()
        first = await rd.index(client, path)
        rd.fail_child_batch(client, 2)
        with pytest.raises(ConnectionError):
            await rd.index(client, path)
        row = await _row(client, first["doc_id"])
        assert (row["action"], row["problems"]) == ("nothing", [])


# ==============================================================================
# Consistency
# ==============================================================================


def _state(chunk_id_suffixes: list[str], deal: str = "d", doc: str = "x", **extra) -> list[tuple]:
    rows = []
    for i, suffix in enumerate(chunk_id_suffixes):
        chunk_id = f"{deal}_{doc}_{suffix}"
        rows.append((ip.point_id_for(chunk_id), {"chunk_id": chunk_id, "chunk_index": i, **extra}))
    return rows


class TestConsistency:
    def _doc(self, children, parents) -> reingest_report.DocumentPoints:
        return reingest_report.DocumentPoints("d", "x", children, parents)

    def test_a_complete_document_has_no_problem(self):
        children = _state(["0000", "0001", "0002"], parent_chunk_id="d_x_p0000", ingest_version=1)
        children[2][1].pop("parent_chunk_id")  # a table chunk: no parent
        parents = [(ip.point_id_for("d_x_p0000"), {"chunk_id": "d_x_p0000", "ingest_version": 1,
                                                   "child_chunk_ids": ["d_x_0000", "d_x_0001"]})]
        assert reingest_report.document_problems(self._doc(children, parents)) == []

    @pytest.mark.parametrize(("mutate", "problem"), [
        (lambda c, p: c[0][1].update(ingest_version=None), "mixed_stamps"),
        (lambda c, p: p[0][1].update(is_current_version=0), "mixed_version_state"),
        (lambda c, p: c[1][1].update(parent_chunk_id="d_x_p0007"), "missing_parent"),
        (lambda c, p: p[0][1]["child_chunk_ids"].remove("d_x_0001"), "child_not_held_by_parent"),
        (lambda c, p: p[0][1]["child_chunk_ids"].append("d_x_0009"), "parent_holds_other_children"),
        (lambda c, p: p.append((ip.point_id_for("d_x_p0001"), {"chunk_id": "d_x_p0001", "ingest_version": 1,
                                                               "child_chunk_ids": []})), "orphan_parent"),
        (lambda c, p: c.append((ip.point_id_for("d_x_0001b"), {**c[1][1]})), "duplicate_chunk_id"),
        (lambda c, p: c[1][1].update(chunk_index=5), "foreign_point_id"),
        (lambda c, p: c[1][1].update(source_file="renamed.txt"), "mixed_document_fields"),
        (lambda c, p: c[1][1].update(document_category="legal"), "mixed_document_fields"),
        (lambda c, p: c[1][1].update(supersedes_doc_id="d-older"), "mixed_document_fields"),
        (lambda c, p: c.clear(), "orphan_parents_only"),
    ])
    def test_each_broken_invariant_is_named(self, mutate, problem):
        children = _state(["0000", "0001"], parent_chunk_id="d_x_p0000", ingest_version=1, is_current_version=1)
        parents = [(ip.point_id_for("d_x_p0000"), {"chunk_id": "d_x_p0000", "ingest_version": 1,
                                                   "is_current_version": 1,
                                                   "child_chunk_ids": ["d_x_0000", "d_x_0001"]})]
        mutate(children, parents)
        assert problem in reingest_report.document_problems(self._doc(children, parents))

    def test_a_current_parent_without_superseded_by_is_not_a_version_mismatch(self):
        """Ingestion writes superseded_by="" on children and nothing on parents."""
        children = _state(["0000"], parent_chunk_id="d_x_p0000", is_current_version=1, superseded_by="")
        parents = [(ip.point_id_for("d_x_p0000"), {"chunk_id": "d_x_p0000", "is_current_version": 1,
                                                   "child_chunk_ids": ["d_x_0000"]})]
        assert reingest_report.document_problems(self._doc(children, parents)) == []


# ==============================================================================
# The report
# ==============================================================================


class TestReport:
    @pytest.mark.asyncio
    async def test_what_each_document_needs(self, tmp_path):
        client = await rd.new_client()
        pdf = await rd.index(client, fx.write_pdf(tmp_path))
        txt = await rd.index(client, fx.write_txt(tmp_path))
        await rd.strip_stamp(client)  # both indexed before Decision 97
        current_pdf = await rd.index(client, fx.write_pdf(tmp_path, "second.pdf"), deal="other-deal")
        stale = await rd.index(client, fx.write_txt(tmp_path, "bm25.txt", fx.SAMPLE_TXT + "\nA note.\n"))
        await client.set_payload(COLLECTION_NAME, payload={"sparse_encoding": 1},
                                 points=Filter(must=[FieldCondition(key="doc_id",
                                                                    match=MatchValue(value=stale["doc_id"]))]))

        report = await reingest_report.build_report(client)
        rows = {r["doc_id"]: r for deal in report["deals"].values() for r in deal["documents"]}
        assert rows[pdf["doc_id"]]["action"] == "reingest"
        assert rows[pdf["doc_id"]]["ingest_version"] == 0
        assert "pdf extraction changed" in rows[pdf["doc_id"]]["reasons"][0]
        assert rows[txt["doc_id"]]["action"] == "nothing"           # .txt: unchanged since before the stamp
        assert rows[current_pdf["doc_id"]]["action"] == "nothing"   # stamped by the current code
        assert rows[stale["doc_id"]]["action"] == "reingest"
        assert "BM25 encoding" in rows[stale["doc_id"]]["reasons"][0]
        assert report["deals"][DEAL]["totals"]["by_action"] == {"reingest": 2, "operator": 0, "identity_failed": 0,
                                                                "backfill_identity": 0, "nothing": 1}
        scoped = await reingest_report.build_report(client, deal_id="other-deal")
        assert list(scoped["deals"]) == ["other-deal"]

    @pytest.mark.asyncio
    async def test_an_identity_the_backfill_would_change_is_its_action(self, tmp_path):
        client = await rd.new_client()
        doc = await rd.index(client, rd.write_supply_agreement(tmp_path))
        await client.set_payload(COLLECTION_NAME, payload={"doc_title": "OLD TITLE"},
                                 points=Filter(must=[FieldCondition(key="deal_id", match=MatchValue(value=DEAL))]))
        row = await _row(client, doc["doc_id"])
        assert row["action"] == "backfill_identity" and row["identity"]["backfill_changes"] is True

    @pytest.mark.asyncio
    async def test_an_identity_the_extractor_cannot_read_is_counted_failed(self, monkeypatch, tmp_path):
        """Not "nothing": the backfill would count it failed and leave it."""
        import src.vector_db.identity_backfill as bf

        monkeypatch.setenv("STRICT_OPTIONAL_STEPS", "0")
        client = await rd.new_client()
        doc = await rd.index(client, rd.write_supply_agreement(tmp_path))

        def crash(payloads):
            raise KeyError("spa")

        monkeypatch.setattr(bf, "extract_document_identity", crash)
        report = await reingest_report.build_report(client)
        (row,) = report["deals"][DEAL]["documents"]
        assert row["doc_id"] == doc["doc_id"] and row["action"] == "identity_failed"
        assert row["identity"]["failed"] and report["deals"][DEAL]["totals"]["identity_failed"] == 1

    @pytest.mark.asyncio
    async def test_the_report_writes_nothing(self, tmp_path):
        client = await rd.new_client()
        await rd.index(client, fx.write_pdf(tmp_path))
        await rd.strip_stamp(client)
        before = (await rd.points(client), await rd.points(client, PARENT_COLLECTION_NAME))
        await reingest_report.build_report(client, files=tmp_path)
        assert (await rd.points(client), await rd.points(client, PARENT_COLLECTION_NAME)) == before

    @pytest.mark.asyncio
    async def test_files_are_matched_by_their_bytes_not_their_names(self, tmp_path):
        client = await rd.new_client()
        indexed = await rd.index(client, fx.write_pdf(tmp_path), name="financial_report.pdf")
        originals = tmp_path / "originals"
        originals.mkdir()
        (originals / "renamed on disk.pdf").write_bytes((tmp_path / "financial_report.pdf").read_bytes())
        (originals / "financial_report.pdf").write_bytes(b"%PDF-1.4 another file of the same name")
        report = await reingest_report.build_report(client, files=originals)
        row = report["deals"][DEAL]["documents"][0]
        assert row["doc_id"] == indexed["doc_id"]
        assert Path(row["file"]).name == "renamed on disk.pdf"
        assert report["files"]["matched"] == 1
        assert [Path(p).name for p in report["files"]["not_indexed"]] == ["financial_report.pdf"]

    @pytest.mark.asyncio
    async def test_the_same_bytes_in_another_deal_are_another_document(self, tmp_path):
        client = await rd.new_client()
        path = fx.write_txt(tmp_path)
        a = await rd.index(client, path)
        b = await rd.index(client, path, deal="other-deal")
        matched, unmatched = reingest_report.match_files(tmp_path, {(DEAL, a["doc_id"]), ("other-deal", b["doc_id"])})
        assert matched == {(DEAL, a["doc_id"]): path, ("other-deal", b["doc_id"]): path} and unmatched == []

    def test_the_command_line_on_a_local_store(self, tmp_path, capsys):
        store = tmp_path / "qdrant"
        originals = tmp_path / "originals"  # apart from the store, whose lock file is not readable
        originals.mkdir()

        async def build():
            client = AsyncQdrantClient(path=str(store))
            await setup_collections(client)
            await rd.index(client, fx.write_pdf(originals))
            await rd.strip_stamp(client)
            await client.close()

        asyncio.run(build())
        out = tmp_path / "report.json"
        assert reingest_report.main(["--path", str(store), "--json", str(out), "--files", str(originals)]) == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["deals"][DEAL]["totals"]["by_action"]["reingest"] == 1
        assert report["deals"][DEAL]["documents"][0]["file"].endswith("financial_report.pdf")
        assert "reingest 1" in capsys.readouterr().out


def test_old_payloads_without_the_new_fields_still_read():
    """A point written before ingest_version (and before sparse_encoding) is reported, not refused."""
    async def go():
        client = await rd.new_client()
        await client.upsert(COLLECTION_NAME, points=[PointStruct(
            id=ip.point_id_for("d_x_0000"),
            vector={"dense": [1.0] + [0.0] * (VECTOR_SIZE - 1), "sparse": SparseVector(indices=[1], values=[1.0])},
            payload={"chunk_id": "d_x_0000", "deal_id": "d", "doc_id": "x", "chunk_index": 0, "text": "Revenue grew.",
                     "file_type": "pdf", "source_file": "x.pdf", "contains_pii": 0})])
        return await reingest_report.build_report(client)

    (row,) = asyncio.run(go())["deals"]["d"]["documents"]
    assert row["action"] == "reingest" and row["ingest_version"] == 0 and row["sparse_encoding"] is None


# ==============================================================================
# The driver
# ==============================================================================


class TestDriver:
    @pytest.mark.asyncio
    async def test_reingests_what_the_report_marks_under_the_indexed_name(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})  # pretend .txt extraction changed
        path = rd.write_supply_agreement(tmp_path)
        client = await rd.new_client()
        old = await rd.old_index(monkeypatch, client, path, name="Supply Agreement (signed).txt")
        originals = tmp_path / "originals"
        originals.mkdir()
        (originals / "scan_0042.txt").write_bytes(path.read_bytes())

        dry = await _run(client, originals, dry_run=True)
        assert [d["doc_id"] for d in dry["documents"]] == [old["doc_id"]]
        assert {p.get("ingest_version") for p, _ in (await rd.points(client)).values()} == {None}

        summary = await _run(client, originals)
        assert summary["done"] == 1 and summary["failed"] == []
        fresh_children, fresh_parents = await rd.fresh(path, name="Supply Agreement (signed).txt")
        assert rd.same(await rd.points(client), fresh_children)
        assert rd.same(await rd.points(client, PARENT_COLLECTION_NAME), fresh_parents)
        assert (await _row(client, old["doc_id"]))["action"] == "nothing"

    @pytest.mark.asyncio
    async def test_the_indexed_category_and_version_state_are_kept(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})
        auto = ip.extract_and_chunk(str(fx.write_txt(tmp_path, "a.txt")), "a.txt", DEAL).document_category
        override = "legal" if auto != "legal" else "financial"
        client = await rd.new_client()
        overridden = await rd.index(client, tmp_path / "a.txt", category=override)
        not_current = await rd.index(client, fx.write_txt(tmp_path, "b.txt", fx.SAMPLE_TXT + "\nB.\n"),
                                     is_current_version=False)
        newer = await rd.index(client, fx.write_txt(tmp_path, "c.txt", fx.SAMPLE_TXT + "\nC.\n"),
                               supersedes_doc_id=overridden["doc_id"])
        await rd.strip_stamp(client)

        summary = await _run(client, tmp_path)
        assert summary["done"] == 3
        rows = {r["doc_id"]: r for r in (await reingest_report.build_report(client))["deals"][DEAL]["documents"]}
        assert rows[overridden["doc_id"]]["document_category"] == override
        assert rows[not_current["doc_id"]]["is_current_version"] == 0
        assert rows[newer["doc_id"]]["supersedes_doc_id"] == overridden["doc_id"]
        assert all(r["action"] == "nothing" for r in rows.values())

        reclassified = await rd.index(client, fx.write_txt(tmp_path, "d.txt", fx.SAMPLE_TXT + "\nD.\n"),
                                      category=override)
        await rd.strip_stamp(client)
        await _run(client, tmp_path, reclassify=True)
        row = await _row(client, reclassified["doc_id"])
        assert row["document_category"] == auto

    @pytest.mark.asyncio
    async def test_a_failed_attempt_is_retried_until_the_document_is_whole(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})
        monkeypatch.setattr(ip, "QDRANT_BATCH_SIZE", 5)
        monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "audit"))
        path = rd.write_supply_agreement(tmp_path)
        client = await rd.new_client()
        old = await rd.old_index(monkeypatch, client, path)
        rd.fail_child_batch(client, 2)  # the first attempt only: the counter keeps counting
        log = tmp_path / "run.jsonl"
        with open(log, "w", encoding="utf-8") as handle:
            summary = await _run(client, tmp_path, log=handle)

        (outcome,) = summary["documents"]
        assert outcome["ok"] and outcome["attempts"] == 2
        attempts = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert [a["ok"] for a in attempts] == [False, True]
        assert "mixed_stamps" in attempts[0]["problems"]
        fresh_children, _ = await rd.fresh(path)
        assert rd.same(await rd.points(client), fresh_children)
        (event,) = [json.loads(line) for f in (tmp_path / "audit").glob("audit_*.jsonl")
                    for line in f.read_text(encoding="utf-8").splitlines()]
        assert event["event"] == "document_reingested" and event["doc_id"] == old["doc_id"] and event["ok"]

    @pytest.mark.asyncio
    async def test_a_document_that_keeps_failing_is_reported_not_hidden(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})
        monkeypatch.setattr(ip, "QDRANT_BATCH_SIZE", 5)
        path = rd.write_supply_agreement(tmp_path)
        client = await rd.new_client()
        await rd.old_index(monkeypatch, client, path)
        rd.fail_parents(client)  # on every attempt
        summary = await _run(client, tmp_path, retries=1)
        (failed,) = summary["failed"]
        assert failed["attempts"] == 2 and failed["problems"] and "ConnectionError" in failed["error"]

    @pytest.mark.asyncio
    async def test_a_file_the_pipeline_refuses_is_not_retried(self, monkeypatch, tmp_path):
        from src.data_processing.errors import DocumentExtractionError

        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})
        client = await rd.new_client()
        await rd.index(client, fx.write_txt(tmp_path))
        await rd.strip_stamp(client)
        before = await rd.points(client)

        def refuse(*args, **kwargs):
            raise DocumentExtractionError("The .txt file could not be parsed")

        monkeypatch.setattr(ip, "extract_and_chunk", refuse)
        summary = await _run(client, tmp_path)
        (failed,) = summary["failed"]
        assert failed["attempts"] == 1 and "DocumentExtractionError" in failed["error"]
        assert await rd.points(client) == before

    @pytest.mark.asyncio
    async def test_a_read_back_that_fails_is_an_attempt_to_repeat_not_a_crash(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})
        client = await rd.new_client()
        doc = await rd.index(client, fx.write_txt(tmp_path))
        await rd.strip_stamp(client)
        real, calls = reingest_run.read_document, {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("qdrant went away")
            return await real(*args, **kwargs)

        monkeypatch.setattr(reingest_run, "read_document", flaky)
        summary = await _run(client, tmp_path)
        (outcome,) = summary["documents"]
        assert outcome["ok"] and outcome["attempts"] == 2 and outcome["doc_id"] == doc["doc_id"]

    @pytest.mark.asyncio
    async def test_a_document_left_with_parents_only_is_left_for_an_operator(self, monkeypatch, tmp_path):
        """A failed first upload's parents: nothing is searchable, and the driver must not resurrect it."""
        path = rd.write_supply_agreement(tmp_path)
        client = await rd.new_client()
        doc = await rd.old_index(monkeypatch, client, path)
        await client.delete(COLLECTION_NAME, points_selector=Filter(must=[
            FieldCondition(key="doc_id", match=MatchValue(value=doc["doc_id"]))]))
        row = await _row(client, doc["doc_id"])
        assert row["action"] == "operator" and "orphan_parents_only" in row["problems"]
        before = await rd.points(client, PARENT_COLLECTION_NAME)

        summary = await _run(client, tmp_path)
        assert summary["planned"] == 0 and [d["doc_id"] for d in summary["for_operator"]] == [doc["doc_id"]]
        assert await rd.points(client) == {}
        assert await rd.points(client, PARENT_COLLECTION_NAME) == before

    @pytest.mark.asyncio
    async def test_a_file_edited_after_the_report_is_refused_not_indexed_beside_it(self, monkeypatch, tmp_path):
        """Review S5: its bytes are another document now; nothing is written and nothing retried."""
        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})
        path = fx.write_txt(tmp_path, "memo.txt")
        client = await rd.new_client()
        doc = await rd.index(client, path)
        await rd.strip_stamp(client)
        before = await rd.points(client)
        real = reingest_run.build_report

        async def report_then_edit(*args, **kwargs):
            report = await real(*args, **kwargs)
            path.write_text(fx.SAMPLE_TXT + "\nEdited after the report.\n", encoding="utf-8")
            return report

        monkeypatch.setattr(reingest_run, "build_report", report_then_edit)
        summary = await _run(client, tmp_path)
        (failed,) = summary["failed"]
        assert failed["doc_id"] == doc["doc_id"] and failed["attempts"] == 0
        assert failed["error"] == reingest_run.FILE_CHANGED
        assert await rd.points(client) == before  # no second document, no write

    @pytest.mark.asyncio
    async def test_a_file_removed_after_the_report_fails_alone_and_the_run_goes_on(self, monkeypatch, tmp_path):
        """Follow-up review S6: the copy step must not end the run for every other document."""
        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})
        client = await rd.new_client()
        gone = fx.write_txt(tmp_path, "a.txt")
        kept = fx.write_txt(tmp_path, "b.txt", fx.SAMPLE_TXT + "\nAnother document.\n")
        doc_gone, doc_kept = await rd.index(client, gone), await rd.index(client, kept)
        await rd.strip_stamp(client)
        real = reingest_run.build_report

        async def report_then_remove(*args, **kwargs):
            report = await real(*args, **kwargs)
            gone.unlink()
            return report

        monkeypatch.setattr(reingest_run, "build_report", report_then_remove)
        summary = await _run(client, tmp_path)
        (failed,) = summary["failed"]
        assert failed["doc_id"] == doc_gone["doc_id"] and failed["attempts"] == 0
        assert failed["error"].startswith(reingest_run.FILE_UNREADABLE)
        assert summary["done"] == 1
        rows = {r["doc_id"]: r for r in (await reingest_report.build_report(client))["deals"][DEAL]["documents"]}
        assert rows[doc_kept["doc_id"]]["action"] == "nothing"

    @pytest.mark.asyncio
    async def test_a_retired_document_stays_retired(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})
        client = await rd.new_client()
        old = await rd.index(client, fx.write_txt(tmp_path, "a.txt"))
        new = await rd.index(client, fx.write_txt(tmp_path, "b.txt", fx.SAMPLE_TXT + "\nAmended.\n"),
                             supersedes_doc_id=old["doc_id"])
        assert (await ip.mark_superseded(DEAL, old["doc_id"], new["doc_id"], client=client))["ok"]
        await rd.strip_stamp(client)

        summary = await _run(client, tmp_path)
        assert summary["done"] == 2 and summary["for_operator"] == []
        rows = {r["doc_id"]: r for r in (await reingest_report.build_report(client))["deals"][DEAL]["documents"]}
        assert (rows[old["doc_id"]]["is_current_version"], rows[old["doc_id"]]["superseded_by"]) == (0, new["doc_id"])
        assert (rows[new["doc_id"]]["is_current_version"], rows[new["doc_id"]]["supersedes_doc_id"]) == \
            (1, old["doc_id"])
        for collection in (COLLECTION_NAME, PARENT_COLLECTION_NAME):
            states = {(p["is_current_version"], p.get("superseded_by") or "")
                      for p, _ in (await rd.points(client, collection)).values() if p["doc_id"] == old["doc_id"]}
            assert states == {(0, new["doc_id"])}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("lowest_id_retired", [False, True])
    async def test_a_mixed_version_state_is_left_for_an_operator(self, monkeypatch, tmp_path, lowest_id_retired):
        """Review S4: some children retired, some current; the outcome must not hang on which id is lowest."""
        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})
        client = await rd.new_client()
        old = await rd.index(client, fx.write_txt(tmp_path, "a.txt"))
        new = await rd.index(client, fx.write_txt(tmp_path, "b.txt", fx.SAMPLE_TXT + "\nAmended.\n"),
                             supersedes_doc_id=old["doc_id"])
        await ip.mark_superseded(DEAL, old["doc_id"], new["doc_id"], client=client)
        ids = sorted(str(i) for i, (p, _) in (await rd.points(client)).items() if p["doc_id"] == old["doc_id"])
        current = ids[1:] if lowest_id_retired else ids[: len(ids) // 2 + 1]
        await client.set_payload(COLLECTION_NAME, payload={"is_current_version": 1, "superseded_by": ""},
                                 points=current)
        before = await rd.points(client)

        row = await _row(client, old["doc_id"])
        assert row["action"] == "operator" and "mixed_version_state" in row["problems"]
        assert row["conflicts"]["superseded_by"] == ["", new["doc_id"]]
        summary = await _run(client, tmp_path)
        assert [d["doc_id"] for d in summary["for_operator"]] == [old["doc_id"]]
        assert old["doc_id"] not in {d["doc_id"] for d in summary["documents"]}
        after = await rd.points(client)
        assert {i: v for i, v in after.items() if v[0]["doc_id"] == old["doc_id"]} == \
            {i: v for i, v in before.items() if v[0]["doc_id"] == old["doc_id"]}

    @pytest.mark.asyncio
    async def test_documents_without_a_file_are_listed_and_left_alone(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})
        client = await rd.new_client()
        doc = await rd.index(client, fx.write_txt(tmp_path))
        await rd.strip_stamp(client)
        before = await rd.points(client)
        empty = tmp_path / "none"
        empty.mkdir()
        summary = await _run(client, empty)
        assert summary["planned"] == 0
        assert [d["doc_id"] for d in summary["without_file"]] == [doc["doc_id"]]
        assert await rd.points(client) == before

    def test_the_command_line_exit_code(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(ip, "CURRENT_SINCE", {"txt": 1})
        store = tmp_path / "qdrant"
        originals = tmp_path / "originals"
        originals.mkdir()
        path = fx.write_txt(originals)

        async def build():
            client = AsyncQdrantClient(path=str(store))
            await setup_collections(client)
            await rd.index(client, path)
            await rd.strip_stamp(client)
            await client.close()

        asyncio.run(build())
        monkeypatch.setattr("src.vector_db.reranker.embed_texts_async", rd.stub_embed)
        assert reingest_run.main(["--path", str(store), "--files", str(originals), "--dry-run"]) == 0
        assert "would re-ingest aurora_sample.txt" in capsys.readouterr().out
        assert reingest_run.main(["--path", str(store), "--files", str(originals)]) == 0
        assert '"planned": 1' in capsys.readouterr().out
        assert reingest_run.main(["--path", str(store), "--files", str(originals)]) == 0
        assert '"planned": 0' in capsys.readouterr().out


# ==============================================================================
# Tables stored without their column order (Decision 108)
# ==============================================================================


def _table_document(rows: list[dict]) -> reingest_report.DocumentPoints:
    from src.vector_db.hybrid_search import SPARSE_ENCODING_VERSION

    children = _state(["0000", "0001"], ingest_version=1, file_type="xlsx",
                      sparse_encoding=SPARSE_ENCODING_VERSION, is_current_version=1)
    children[1][1]["structured_rows"] = rows
    return reingest_report.DocumentPoints("d", "x", children, [])


@pytest.mark.parametrize("rows, action", [
    ([{"label": "Revenue", "values": {"FY2023": "452.8", "FY2022": "387.1"}}], "reingest"),
    ([{"label": "Revenue", "values": {"FY2023": "452.8", "FY2022": "387.1"},
       "columns": ["FY2023", "FY2022"]}], "nothing"),
    ([{"label": "Revenue", "values": {"Column 2": "452.8", "Column 3": "387.1"}}], "nothing"),
    ([{"label": "Revenue", "values": {"FY2023": "452.8"}}], "nothing"),
    ([], "nothing"),
], ids=["header-names", "with-columns", "placeholders", "one-cell", "no-rows"])
def test_tables_stored_without_their_column_order_are_reingested(rows, action):
    row = reingest_report.plan_document(_table_document(rows), None, None)
    assert row["action"] == action
    assert row["unordered_table_chunks"] == (1 if action == "reingest" else 0)
    if action == "reingest":
        assert "without their column order" in row["reasons"][0]


def test_the_report_reads_the_rows_it_checks():
    assert "structured_rows" in reingest_report.CHILD_FIELDS


def test_collect_keeps_a_flag_not_the_rows():
    rows = [{"label": "Revenue", "values": {"FY2023": "452.8", "FY2022": "387.1"}}]
    assert reingest_report.rows_unordered(rows) is True
    assert reingest_report.rows_unordered(None) is False
    doc = _table_document([])
    doc.children[1][1].pop("structured_rows")
    doc.children[1][1][reingest_report.ROWS_UNORDERED] = True
    assert reingest_report.plan_document(doc, None, None)["unordered_table_chunks"] == 1
