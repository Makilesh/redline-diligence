"""
Optional steps fail open; nothing else does (Decision 88).

Round 8's integration review found a crash in the query path:
named_document.name_document raised KeyError: 'spa' for any question that
wrote "SPA" while a retrieved document was titled "Share Purchase Agreement"
("spa" is also a stopword, as the legal form S.p.A., so it had no weight).
Nothing between that ranking heuristic and the request caught anything, so
one optional step failed the whole query. LegalBench-RAG never phrases a
question that way: thousands of evaluation queries never reached the line.

Every step on the query and ingest paths is one of three kinds (the table is
in DECISIONS_LOG Decision 88):

  required  the request cannot be served without it (embedding, search,
            fusion, rerank, synthesis; parsing, chunking, upsert). A failure
            fails the request.
  safety    PII detection and exclusion, deal isolation, the current-version
            filter, injection labelling. A failure fails closed, by raising:
            never a result that shows more than it should.
  optional  a quality heuristic with a documented default, which is what the
            step does when it is switched off or finds nothing: the named
            document's is the fused choice, the period preference's the lists
            as retrieved, identity extraction's no identity fields.

An optional step runs through fail_open() / fail_open_async(). An Exception
(only that: cancellation, KeyboardInterrupt and SystemExit pass through) is
logged once, as one ERROR record carrying the step name and the traceback,
counted in redline_optional_step_failures_total{step}, named in the running
graph node's trace (perf.fallbacks), and the default is returned. The default
must never widen what a caller may see — no step whose failure could do that
is optional.

STRICT_OPTIONAL_STEPS=1 re-raises after logging and counting, so a guard
cannot hide a bug from a test run: tests/conftest.py sets it for the suite.
"""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable, TypeVar

from src.utils import metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

T = TypeVar("T")

STRICT_ENV = "STRICT_OPTIONAL_STEPS"


def strict_optional_steps() -> bool:
    """STRICT_OPTIONAL_STEPS=1: a failing optional step raises instead of falling back."""
    return os.getenv(STRICT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _record(step: str, error: Exception) -> None:
    logger.error(
        "Optional step failed; using its default",
        extra={"step": step, "error_type": type(error).__name__},
        exc_info=True,
    )
    metrics.record_optional_step_failure(step)


def fail_open(step: str, fn: Callable[..., T], *args: Any, default: T, **kwargs: Any) -> T:
    """
    Runs one optional step; on an Exception, its documented default.

    Args:
        step: Name for the log, the metric label and the trace.
        fn: The step.
        *args: Its positional arguments.
        default: What the request continues with if the step raises.
        **kwargs: Its keyword arguments.

    Returns:
        fn's result, or `default`.

    Raises:
        Exception: The step's own, under STRICT_OPTIONAL_STEPS=1.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as error:
        _record(step, error)
        if strict_optional_steps():
            raise
        return default


async def fail_open_async(
    step: str, fn: Callable[..., Awaitable[T]], *args: Any, default: T, **kwargs: Any
) -> T:
    """fail_open for a coroutine function: awaits the step, or returns `default`."""
    try:
        return await fn(*args, **kwargs)
    except Exception as error:
        _record(step, error)
        if strict_optional_steps():
            raise
        return default
