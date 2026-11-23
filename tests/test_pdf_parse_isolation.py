"""
Ingest parses PDFs in a child process, and the API's process sees exactly what an
in-process parse returned (Decision 110).

On a thread of the single-worker API the parse held the GIL and stalled every other
request (measured: a question's retrieval took 33.3 s instead of 0.3 s during a 10-K's
parse). These tests pin what moving it must not change — the sections, the warnings, the
errors a client is shown — and what it adds: a parser that dies fails that upload, and
the next upload parses normally.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

import src.data_processing.parse_isolation as iso
from src.data_processing.errors import DocumentExtractionError, NoExtractableContentError
from src.data_processing.pdf_processor import PDFProcessor
from tests.fixtures import stress_docs as sx


def _in_process(path: Path, legal_mode: bool = False):
    processor = PDFProcessor(legal_mode=legal_mode)
    return processor.process(str(path), "doc-1"), list(processor.warnings)


def _crash(pdf_path, *_args):
    """Stands in for a parser that dies on a file (a segfault, the OOM killer); counts its runs."""
    with open(pdf_path + ".runs", "a", encoding="ascii") as f:
        f.write("run\n")
    os._exit(3)


def _hang(*_args):
    """Stands in for a parser stuck in a loop."""
    time.sleep(600)


def _crash_on_hostile(pdf_path, doc_id, legal_mode):
    if Path(pdf_path).name == "hostile.pdf":
        os._exit(3)
    return iso._parse(pdf_path, doc_id, legal_mode)  # the child's own, unpatched module


def test_the_parse_runs_in_another_process():
    assert iso.warm_up() != os.getpid()


@pytest.mark.parametrize("writer, legal_mode", [
    (sx.write_ruled_multipage_pdf, False),
    (sx.write_borderless_statement_pdf, False),
    (sx.write_partly_scanned_pdf, False),
    (sx.write_legal_pdf, True),
], ids=["ruled-multipage", "borderless-statement", "partly-scanned", "legal"])
def test_sections_and_warnings_equal_an_in_process_parse(tmp_path, writer, legal_mode):
    path = writer(tmp_path)
    assert iso.parse_pdf(str(path), "doc-1", legal_mode=legal_mode) == _in_process(path, legal_mode)


@pytest.mark.parametrize("writer, error", [
    (sx.write_encrypted_pdf, DocumentExtractionError),
    (sx.write_corrupt_pdf, DocumentExtractionError),
    (sx.write_scanned_pdf, NoExtractableContentError),
], ids=["encrypted", "corrupt", "scanned"])
def test_a_clients_error_crosses_unchanged(tmp_path, writer, error):
    path = writer(tmp_path)
    with pytest.raises(error) as in_process:
        _in_process(path)
    with pytest.raises(error) as isolated:
        iso.parse_pdf(str(path), "doc-1")
    assert str(isolated.value) == str(in_process.value)


def test_a_missing_file_is_still_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        iso.parse_pdf(str(tmp_path / "absent.pdf"), "doc-1")


def test_a_parser_that_dies_fails_its_upload_once_and_the_next_one_parses(tmp_path, monkeypatch):
    hostile = sx.write_ruled_multipage_pdf(tmp_path)
    monkeypatch.setattr(iso, "_parse", _crash)
    with pytest.raises(DocumentExtractionError, match="parser stopped"):
        iso.parse_pdf(str(hostile), "doc-1")
    assert Path(str(hostile) + ".runs").read_text().count("run") == 1  # not retried
    monkeypatch.undo()
    path = sx.write_borderless_statement_pdf(tmp_path)
    assert iso.parse_pdf(str(path), "doc-2") == _in_process(path)


def test_a_parse_that_never_ends_times_out_and_the_next_one_parses(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_PARSE_TIMEOUT_SECONDS", "3")
    monkeypatch.setattr(iso, "_parse", _hang)
    started = time.monotonic()
    with pytest.raises(DocumentExtractionError, match="longer than 3 seconds"):
        iso.parse_pdf(str(sx.write_ruled_multipage_pdf(tmp_path)), "doc-1")
    assert time.monotonic() - started < 60
    monkeypatch.undo()
    path = sx.write_borderless_statement_pdf(tmp_path)
    assert iso.parse_pdf(str(path), "doc-2") == _in_process(path)


@pytest.mark.parametrize("value, expected", [("", 900.0), ("120", 120.0), ("x", 900.0), ("-5", 900.0)])
def test_the_timeout_follows_PDF_PARSE_TIMEOUT_SECONDS(monkeypatch, value, expected):
    monkeypatch.setenv("PDF_PARSE_TIMEOUT_SECONDS", value)
    assert iso.timeout_seconds() == expected


def test_another_uploads_hostile_file_does_not_fail_this_one(tmp_path, monkeypatch):
    monkeypatch.setattr(iso, "_parse", _crash_on_hostile)
    hostile = sx.write_ruled_multipage_pdf(tmp_path, name="hostile.pdf")
    good = sx.write_borderless_statement_pdf(tmp_path)
    results: dict = {}

    def upload(name, path):
        try:
            results[name] = iso.parse_pdf(str(path), name)
        except Exception as e:  # noqa: BLE001 - the outcome is what is tested
            results[name] = e

    threads = [threading.Thread(target=upload, args=("hostile", hostile)),
               threading.Thread(target=upload, args=("good", good))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert isinstance(results["hostile"], DocumentExtractionError)
    good_sections, good_warnings = results["good"]
    assert [s.text for s in good_sections] == [s.text for s in _in_process(good)[0]]


def test_optional_steps_that_fail_in_the_child_are_counted_in_the_api(monkeypatch):
    import src.data_processing.pdf_processor as pp
    from src.utils import metrics
    from src.utils.fail_open import fail_open

    def process(self, pdf_path, doc_id):
        fail_open("pdf_table_title", lambda: 1 / 0, default=None)
        return []

    monkeypatch.setattr(pp.PDFProcessor, "process", process)
    monkeypatch.delenv("STRICT_OPTIONAL_STEPS", raising=False)
    original = metrics.record_optional_step_failure
    # The child's side, in process: the step is reported back, and the hook is restored.
    assert iso._parse("any.pdf", "doc-1", False)[2] == ["pdf_table_title"]
    assert metrics.record_optional_step_failure is original
    # The API's side: what the child reports is counted.
    counted = []
    monkeypatch.setattr(iso, "_run", lambda *a, **k: ([], [], ["pdf_table_title"]))
    monkeypatch.setattr(metrics, "record_optional_step_failure", counted.append)
    assert iso.parse_pdf("any.pdf", "doc-1") == ([], [])
    assert counted == ["pdf_table_title"]


def test_shutdown_stops_a_parse_in_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(iso, "_parse", _hang)
    outcome: dict = {}

    def upload():
        try:
            iso.parse_pdf(str(sx.write_ruled_multipage_pdf(tmp_path)), "doc-1")
        except Exception as e:  # noqa: BLE001
            outcome["error"] = e

    t = threading.Thread(target=upload)
    t.start()
    time.sleep(5)  # the child is started and parsing
    started = time.monotonic()
    iso.shutdown()
    t.join(timeout=60)
    assert not t.is_alive() and time.monotonic() - started < 60
    assert isinstance(outcome.get("error"), DocumentExtractionError)


def test_ingest_uses_the_child_process(tmp_path, monkeypatch):
    from src.data_processing.ingest_pipeline import extract_and_chunk

    calls = []
    real = iso.parse_pdf
    monkeypatch.setattr(iso, "parse_pdf", lambda *a, **k: calls.append(a) or real(*a, **k))
    path = sx.write_borderless_statement_pdf(tmp_path)
    doc = extract_and_chunk(str(path), path.name, "deal-iso", "financial")
    assert len(calls) == 1 and doc.chunks


def _oom_score_adj() -> str:
    with open("/proc/self/oom_score_adj", encoding="ascii") as f:
        return f.read().strip()


@pytest.mark.skipif(not Path("/proc/self/oom_score_adj").exists(), reason="Linux only")
def test_the_child_is_the_one_the_oom_killer_takes():
    # Under the container's memory ceiling, the kernel kills the highest score: the
    # parser, not the API holding the models.
    assert iso._run(_oom_score_adj) == "1000"
    assert _oom_score_adj() != "1000"
