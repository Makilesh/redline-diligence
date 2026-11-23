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
from pathlib import Path

import pytest

import src.data_processing.parse_isolation as iso
from src.data_processing.errors import DocumentExtractionError, NoExtractableContentError
from src.data_processing.pdf_processor import PDFProcessor
from tests.fixtures import stress_docs as sx


def _in_process(path: Path, legal_mode: bool = False):
    processor = PDFProcessor(legal_mode=legal_mode)
    return processor.process(str(path), "doc-1"), list(processor.warnings)


def _crash(*_args):
    """Stands in for a parser that dies on a file (a segfault, the OOM killer)."""
    os._exit(3)


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


def test_a_parser_that_dies_fails_the_upload_and_the_next_one_parses(tmp_path, monkeypatch):
    monkeypatch.setattr(iso, "_parse", _crash)
    with pytest.raises(DocumentExtractionError, match="parser stopped"):
        iso.parse_pdf(str(sx.write_ruled_multipage_pdf(tmp_path)), "doc-1")
    monkeypatch.undo()
    path = sx.write_borderless_statement_pdf(tmp_path)
    assert iso.parse_pdf(str(path), "doc-2") == _in_process(path)


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
    assert iso._submit(_oom_score_adj) == "1000"
    assert _oom_score_adj() != "1000"
