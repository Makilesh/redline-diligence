"""
PDF parsing runs in a child process, not on a thread of the API's process (Decision 110).

Ingest used to parse on asyncio.to_thread. The parse is CPU-bound Python (pdfplumber's
pdfminer layout analysis and the table stitcher) and holds the GIL, so the thread did not
keep the API responsive: the API runs one worker, and while a PDF parsed no other request
ran. Measured on 2026-11-23 with a 4.4 MB 10-K: a question asked 15 s into the upload
spent 33.3 s in retrieval (0.3 s idle) — 19 s of it embedding three short texts — and
finished within a second of the parse; the event loop stalled for most of the parse.

In a child process the parse has its own interpreter and its own GIL. It also contains a
parser crash or a memory blow-up on a hostile file: the upload fails and the API keeps
running. The child is not reused (max_tasks_per_child=1), so whatever one document left
behind in the parser's memory goes with the process, and a fresh child starts for the next
one. Nothing falls back to parsing in the API's process: a file that kills the parser
would kill the API too.

Only the PDF parse moves. The rest of extract_and_chunk (classification with the Laya
model, chunking, PII and risk scans) needs models the child would have to load again, and
measured, the parse was the stall.
"""
from __future__ import annotations

import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from src.data_processing.errors import DocumentExtractionError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# One parse at a time. Two threads parsing in the API's process shared one GIL, so this
# is the throughput there was; it also bounds the memory parsing can take.
MAX_WORKERS = 1

PARSER_STOPPED = (
    "The PDF could not be parsed: the parser stopped while reading it "
    "(the file may be malformed or too large)"
)

_lock = threading.Lock()
_pool: ProcessPoolExecutor | None = None


def _parse(pdf_path: str, doc_id: str, legal_mode: bool):
    """Runs in the child: PDFProcessor.process and the warnings it collected."""
    from src.data_processing.pdf_processor import PDFProcessor

    processor = PDFProcessor(legal_mode=legal_mode)
    sections = processor.process(pdf_path, doc_id)
    return sections, list(processor.warnings)


def _volunteer_for_the_oom_killer() -> None:
    """
    Runs first in each child: marks it the process to kill when memory runs out.

    The child shares the API container's memory ceiling (mem_limit), and the
    kernel's OOM killer picks the process with the highest score there — by size,
    the API's, which holds the models. A hostile PDF that exhausts memory should
    cost its own upload, not the API. Raising one's own oom_score_adj needs no
    privilege; it exists on Linux only, and a failure changes nothing else.
    """
    try:
        with open("/proc/self/oom_score_adj", "w", encoding="ascii") as f:
            f.write("1000")
    except OSError:
        pass


def _child_pid() -> int:
    return os.getpid()


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            # spawn on every platform: forking a process that has torch and threads
            # running is not safe, and spawn is what Windows and macOS use anyway.
            _pool = ProcessPoolExecutor(
                max_workers=MAX_WORKERS,
                mp_context=multiprocessing.get_context("spawn"),
                max_tasks_per_child=1,
                initializer=_volunteer_for_the_oom_killer,
            )
        return _pool


def _discard(pool: ProcessPoolExecutor) -> None:
    global _pool
    with _lock:
        if _pool is pool:
            _pool = None
    pool.shutdown(wait=False, cancel_futures=True)


def _submit(fn, *args):
    """Runs fn in the child. A broken pool is replaced and the call made once more."""
    for attempt in (1, 2):
        pool = _get_pool()
        try:
            return pool.submit(fn, *args).result()
        except BrokenProcessPool:
            _discard(pool)
            # The first break may be another upload's: every queued call fails with
            # the pool. A second break on a fresh pool is this file's.
            logger.warning("PDF parser process stopped", extra={"attempt": attempt})
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


def parse_pdf(pdf_path: str, doc_id: str, legal_mode: bool = False):
    """
    PDFProcessor(legal_mode).process(pdf_path, doc_id), run in a child process.

    Args:
        pdf_path: Absolute path to the PDF.
        doc_id: Document identifier (logging and section metadata).
        legal_mode: Clause segmentation instead of font-based headings.

    Returns:
        (sections, warnings): the PDFSection list and the processor's warnings.

    Raises:
        The processor's own exceptions (FileNotFoundError, DocumentExtractionError,
        NoExtractableContentError), unchanged.
        DocumentExtractionError: The parser process stopped on this file twice.
    """
    try:
        return _submit(_parse, pdf_path, doc_id, legal_mode)
    except BrokenProcessPool as e:
        raise DocumentExtractionError(PARSER_STOPPED) from e


def warm_up() -> int:
    """Starts a child once (the API's start-up and measurements); returns its pid."""
    return _submit(_child_pid)
