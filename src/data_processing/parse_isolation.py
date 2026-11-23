"""
PDF parsing runs in a child process, not on a thread of the API's process (Decision 110).

Ingest used to parse on asyncio.to_thread. The parse is CPU-bound Python (pdfplumber's
pdfminer layout analysis and the table stitcher) and holds the GIL, so the thread did not
keep the API responsive: the API runs one worker, and while a PDF parsed no other request
ran. Measured on 2026-11-23 with a 4.4 MB 10-K: a question asked 15 s into the upload
spent 33.3 s in retrieval (0.3 s idle) — 19 s of it embedding three short texts — and
finished within a second of the parse; the event loop stalled for most of the parse.

In a child process the parse has its own interpreter and its own GIL. It also contains a
parser that crashes, exhausts memory or never finishes on a hostile file: that upload
fails and the API keeps running. Parses run one at a time (the lock below), so a child
that dies or times out was parsing the caller's own file — no other upload is failed with
it, and nothing is retried. The child is not reused (max_tasks_per_child=1): whatever one
document left in the parser's memory goes with the process. Nothing falls back to parsing
in the API's process, where a file that kills the parser would kill the API.

Only the PDF parse moves. The rest of extract_and_chunk (classification with the Laya
model, chunking, PII and risk scans) needs models the child would have to load again, and
measured, the parse was the stall.
"""
from __future__ import annotations

import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool

from src.data_processing.errors import DocumentExtractionError
from src.utils import metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# A 257-page 10-K parses in 43-55 s on the development machine; two ARM cores are slower.
# Long enough for any real filing, short enough that a parser stuck in a loop does not
# hold every later upload behind it for good.
DEFAULT_TIMEOUT_SECONDS = 900.0

PARSER_STOPPED = (
    "The PDF could not be parsed: the parser stopped while reading it "
    "(the file may be malformed or too large)"
)
PARSER_TIMED_OUT = (
    "The PDF could not be parsed: reading it took longer than {seconds:.0f} seconds "
    "(the file may be malformed or too large)"
)

# Held for the whole of one parse: one child at a time, and a failure is that parse's own.
_parse_lock = threading.Lock()
_pool_lock = threading.Lock()
_pool: ProcessPoolExecutor | None = None


def timeout_seconds() -> float:
    """PDF_PARSE_TIMEOUT_SECONDS, or the default for a missing or unusable value."""
    try:
        value = float(os.getenv("PDF_PARSE_TIMEOUT_SECONDS", ""))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_TIMEOUT_SECONDS


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


def _parse(pdf_path: str, doc_id: str, legal_mode: bool):
    """
    Runs in the child: PDFProcessor.process, its warnings, and the optional steps that
    failed — fail_open counts those in this process's metrics, which nobody scrapes, so
    their names go back to be counted in the API's.
    """
    from src.data_processing.pdf_processor import PDFProcessor

    failed: list[str] = []
    count = metrics.record_optional_step_failure

    def record(step: str) -> None:
        failed.append(step)
        count(step)

    metrics.record_optional_step_failure = record
    try:
        processor = PDFProcessor(legal_mode=legal_mode)
        sections = processor.process(pdf_path, doc_id)
    finally:
        metrics.record_optional_step_failure = count
    return sections, list(processor.warnings), failed


def _child_pid() -> int:
    return os.getpid()


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            # spawn on every platform: forking a process that has torch and threads
            # running is not safe, and spawn is what Windows and macOS use anyway.
            _pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                max_tasks_per_child=1,
                initializer=_volunteer_for_the_oom_killer,
            )
        return _pool


def _discard(pool: ProcessPoolExecutor, *, kill: bool) -> None:
    """Drops the pool; with kill, terminates its child first (a parse that will not end)."""
    global _pool
    with _pool_lock:
        if _pool is pool:
            _pool = None
    if kill:
        # ProcessPoolExecutor has no public way to stop a running task before 3.14.
        for process in list(getattr(pool, "_processes", {}).values()):
            process.terminate()
    pool.shutdown(wait=False, cancel_futures=True)


def _run(fn, *args, timeout: float | None = None):
    """Runs fn in a child, one call at a time."""
    with _parse_lock:
        pool = _get_pool()
        future = pool.submit(fn, *args)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            _discard(pool, kill=True)
            raise
        except BrokenProcessPool:
            _discard(pool, kill=False)
            raise


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
        DocumentExtractionError: The parser process died on this file, or did not
            finish within PDF_PARSE_TIMEOUT_SECONDS.
    """
    seconds = timeout_seconds()
    try:
        sections, warnings, failed = _run(_parse, pdf_path, doc_id, legal_mode, timeout=seconds)
    except BrokenProcessPool as e:
        logger.warning("PDF parser process stopped", extra={"doc_id": doc_id})
        raise DocumentExtractionError(PARSER_STOPPED) from e
    except FutureTimeout as e:
        logger.warning("PDF parse timed out", extra={"doc_id": doc_id, "seconds": seconds})
        raise DocumentExtractionError(PARSER_TIMED_OUT.format(seconds=seconds)) from e
    for step in failed:
        metrics.record_optional_step_failure(step)
    return sections, warnings


def warm_up() -> int:
    """Starts a child once and returns its pid (measurements and tests)."""
    return _run(_child_pid)


def shutdown() -> None:
    """At the API's shutdown: a parse still running is stopped, not waited for."""
    with _pool_lock:
        pool = _pool
    if pool is not None:
        _discard(pool, kill=True)
