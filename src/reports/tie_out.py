"""
Deal-level figure tie-out (Decision 92).

Tying out figures is a standard step of diligence: the revenue, EBITDA, debt
or headcount one document states is checked against every other document in
the data room that states it. The financial verifier already compares figures,
but only inside one question's retrieved context; this reads the whole deal.

Deterministic and LLM-free. Every labelled figure (metric + period) that two or
more documents state is listed with each document's value, page and section,
and whether they agree after unit and scale normalisation and the printed
rounding (src/utils/numerical_registry.py). Conflicts come first, largest gap
first, then agreements, then the counts of what was not compared: figures one
document states, labels that name nothing on their own, documents that state
one label twice with different values, figures with no readable period.

The deal is read as a question would read it: the same deal, current-version
and no-PII filter (hybrid_search.deal_read_filter), so the report can never show a
superseded draft or a PII-flagged passage that retrieval hides. Points are
scrolled with their payload only, never their vectors, one document at a time,
and the read stops at a chunk cap and a time budget; a report cut short says so
(`truncated`) and names the documents it did not read.

Failure classes (Decision 88): reading the store and building the registry are
the route's required steps — a failure is an error, never an empty report, since
"0 conflicts" from a report that read nothing would be a false statement. The
consoles treat the report as optional for a page load: when it cannot be
fetched the rest of the deal view renders and the panel says it could not load.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from src.utils.logger import setup_logger
from src.utils.numerical_registry import FigureTieOut, NumericalEntry, NumericalRegistry

logger = setup_logger(__name__)

# Bounds on one report. A FinanceBench 10-K is 300–2,200 child chunks, so the
# default cap reads a data room of 20–40 such filings whole.
DEFAULT_MAX_CHUNKS = 20_000
DEFAULT_TIME_BUDGET_S = 20.0
# Reading stops at this share of the budget; the rest is the comparison's.
READ_SHARE = 0.6
# Documents are read a page's worth at a time. Large pages keep the requests
# few: a local-mode store filters every point it holds on each request.
SCROLL_PAGE = 2048
# Documents one report lists. A deal with more is reported as cut short
# ("chunk_cap": larger than one report reads), never as read whole.
MAX_DEAL_DOCUMENTS = 5_000

# Bounds on the response: document text is untrusted and unbounded, the
# report is rendered as text by both consoles, and a large deal agrees on
# thousands of figures.
MAX_LISTED_CONFLICTS = 200
MAX_LISTED_AGREEMENTS = 200
MAX_VALUES_PER_FIGURE = 10
MAX_DOCUMENT_NAMES = 50
MAX_LABEL_CHARS = 120
MAX_PRINTED_CHARS = 40
MAX_SOURCE_CHARS = 200
MAX_SECTION_CHARS = 120
MAX_PERIOD_CHARS = 40
_DOT_LEADER = re.compile(r"\s*\.{3,}\s*")

# What the registry reads, and nothing else: no vectors, no PII or risk fields.
PAYLOAD_FIELDS = [
    "text",
    "source_file",
    "doc_id",
    "page_number",
    "section_heading",
    "chunk_index",
    "table_id",
    "is_table",
    "structured_rows",
    "scale_factor",
    "content_type",
    "filing_form",
    # The last day a filing reports on: a schedule after it (a debt maturity)
    # is not tied out. The route did not ask for these, so the registry never
    # saw them here and compared such schedules (Decision 93).
    "filing_period_end",
    "filing_fiscal_years",
    # Which rows a table chunk holds: a table is read once, from its structured
    # rows, not again from its markdown grid.
    "table_row_range",
    "metric_name",
    "normalized_value",
    "raw_value",
    "currency",
    "fiscal_year",
]
# A converted table is stored four ways under one table_id. The row-by-row
# representation (its structured_rows) and the markdown grid carry every row;
# the narrative repeats them, and the metrics summary is computed, not stated.
SKIPPED_CONTENT_TYPES = ["table_narrative", "computed_metric"]


def max_chunks() -> int:
    """Chunk cap for one report (env TIE_OUT_MAX_CHUNKS, default 20,000)."""
    try:
        return max(1, int(os.getenv("TIE_OUT_MAX_CHUNKS", str(DEFAULT_MAX_CHUNKS))))
    except ValueError:
        return DEFAULT_MAX_CHUNKS


def time_budget_s() -> float:
    """Time budget for one report in seconds (env TIE_OUT_TIME_BUDGET_S, default 20)."""
    try:
        return max(0.1, float(os.getenv("TIE_OUT_TIME_BUDGET_S", str(DEFAULT_TIME_BUDGET_S))))
    except ValueError:
        return DEFAULT_TIME_BUDGET_S


@dataclass
class TieOutValue:
    """One document's statement of a figure."""

    source_file: str
    page_number: int | None
    section_heading: str
    as_printed: str
    value: float
    scale_known: bool


@dataclass
class TieOutFigure:
    """A labelled figure two or more documents state."""

    metric: str
    period: str
    kind: str  # "amount" | "percent" | "multiple"
    status: str  # "conflict" | "agree"
    max_deviation_pct: float
    values: list[TieOutValue] = field(default_factory=list)
    more_documents: int = 0  # documents stating it beyond those listed


@dataclass
class TieOutReport:
    """The deal-level tie-out. Conflicts first, then agreements, then counts."""

    deal_id: str
    conflicts: list[TieOutFigure] = field(default_factory=list)
    agreements: list[TieOutFigure] = field(default_factory=list)
    conflicts_total: int = 0
    agreements_total: int = 0
    single_source_figures: int = 0
    ambiguous_figures: int = 0
    generic_figures: int = 0
    figures_without_period: int = 0
    documents_total: int = 0
    documents_read: int = 0
    documents_not_read: list[str] = field(default_factory=list)
    chunks_read: int = 0
    truncated: bool = False
    truncation_reason: str = ""  # "" | "chunk_cap" | "time_budget"
    elapsed_ms: float = 0.0


def _clip(text: str, limit: int) -> str:
    """`text` with its whitespace collapsed, cut to `limit` characters."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _display_label(label: str) -> str:
    """A row label as printed, without the table furniture around it (dot leaders, "$", nil dashes)."""
    return _clip(_DOT_LEADER.sub(" ", label).strip(" |:$\t—–-�"), MAX_LABEL_CHARS)


def _value(entry: NumericalEntry) -> TieOutValue:
    page = entry.page_number if isinstance(entry.page_number, int) and entry.page_number > 0 else None
    return TieOutValue(
        source_file=_clip(entry.source_file, MAX_SOURCE_CHARS),
        page_number=page,
        section_heading=_clip(entry.section_heading, MAX_SECTION_CHARS),
        as_printed=_clip(entry.as_printed, MAX_PRINTED_CHARS),
        value=entry.normalized_value,
        scale_known=entry.scale_known,
    )


def _figure(figure: FigureTieOut) -> TieOutFigure:
    listed = figure.entries[:MAX_VALUES_PER_FIGURE]
    return TieOutFigure(
        metric=_display_label(figure.metric_name),
        period=_clip(figure.fiscal_year, MAX_PERIOD_CHARS),
        kind=figure.kind,
        status=figure.status,
        max_deviation_pct=figure.max_deviation_pct,
        values=[_value(e) for e in listed],
        more_documents=len(figure.entries) - len(listed),
    )


def report_from_chunks(
    deal_id: str,
    chunks: list[dict],
    should_stop: Callable[[], bool] | None = None,
) -> tuple[TieOutReport, int]:
    """
    Builds the report from a deal's chunk payloads (no I/O).

    Args:
        deal_id: Deal the chunks belong to.
        chunks: Child chunk payloads, grouped by document in the order the
            documents should be read.
        should_stop: Polled between documents; True stops the build.

    Returns:
        (report, documents_read): how many documents, in first-seen order, were
        read before `should_stop` ended the build. The report's document and
        timing fields are left for the caller, which knows what it fetched.
    """
    registry, documents_read = NumericalRegistry.from_documents(chunks, should_stop=should_stop)
    result = registry.tie_out()
    conflicts, agreements = result.conflicts, result.agreements
    report = TieOutReport(
        deal_id=deal_id,
        conflicts=[_figure(f) for f in conflicts[:MAX_LISTED_CONFLICTS]],
        agreements=[_figure(f) for f in agreements[:MAX_LISTED_AGREEMENTS]],
        conflicts_total=len(conflicts),
        agreements_total=len(agreements),
        single_source_figures=result.single_source,
        ambiguous_figures=result.ambiguous,
        generic_figures=result.generic,
        figures_without_period=result.without_period,
        chunks_read=len(chunks),
    )
    return report, documents_read


def _deal_filter(deal_id: str):
    """What a question may read in this deal, minus the repeated table forms."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    from src.vector_db.hybrid_search import deal_read_filter

    base = deal_read_filter(deal_id)
    return Filter(
        must=base.must,
        must_not=[FieldCondition(key="content_type", match=MatchAny(any=SKIPPED_CONTENT_TYPES))],
    )


async def _documents(client, collection: str, deal_filter) -> tuple[list[tuple[str, int]], bool]:
    """
    (document name, readable chunks) for the deal's documents, by name, and
    whether that is all of them: a facet lists at most MAX_DEAL_DOCUMENTS
    (the largest first), and the caller reports a deal with more as cut short.
    """
    facet = await client.facet(
        collection_name=collection, key="source_file", facet_filter=deal_filter,
        limit=MAX_DEAL_DOCUMENTS + 1, exact=True,
    )
    hits = facet.hits[:MAX_DEAL_DOCUMENTS]
    listed_all = len(facet.hits) <= MAX_DEAL_DOCUMENTS
    return sorted((str(hit.value), int(hit.count)) for hit in hits), listed_all


def _batches(documents: list[tuple[str, int]], page: int) -> list[list[tuple[str, int]]]:
    """
    Consecutive documents grouped so each group's chunks fit one scroll page.

    A group is read in one request, so a deal needs about as many requests as
    pages; a document larger than a page is a group of its own.
    """
    batches: list[list[tuple[str, int]]] = []
    size = 0
    for name, count in documents:
        if batches and size + count <= page:
            batches[-1].append((name, count))
            size += count
        else:
            batches.append([(name, count)])
            size = count
    return batches


async def _scroll(client, collection: str, scroll_filter, deadline: float,
                  clock: Callable[[], float]) -> tuple[list[dict], bool]:
    """
    Every chunk the filter admits, payload only, until the deadline.

    Returns:
        (payloads, complete): complete is False when the deadline passed
        before the last page.
    """
    payloads: list[dict] = []
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            with_payload=PAYLOAD_FIELDS,
            with_vectors=False,
            limit=SCROLL_PAGE,
            offset=offset,
        )
        payloads.extend(p.payload or {} for p in points)
        if offset is None:
            return payloads, True
        if clock() > deadline:
            return payloads, False


async def build_tie_out(
    deal_id: str,
    client=None,
    chunk_cap: int | None = None,
    budget_s: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> TieOutReport:
    """
    Reads a deal from the vector store and builds its tie-out report.

    The documents are listed with their chunk counts first (a facet), so the
    chunk cap is applied before anything is fetched: documents are taken in
    name order while they fit, and a document is read whole or not at all —
    every figure listed is compared against everything its document states.
    The chosen documents are then scrolled a page's worth of documents at a
    time, payload only, until READ_SHARE of the time budget is spent, so a deal
    the budget cuts short still yields whole documents and time to compare
    them; documents the deadline cut off mid-read are dropped. The registry is
    built in a worker thread, so a large deal does not stall the event loop,
    and stops at the budget's end, or once the rest of the budget has passed
    since the read ended when the read ran over. Whatever was not read is named
    in the report and `truncated` says why.

    Args:
        deal_id: Deal to report on.
        client: AsyncQdrantClient; the application's singleton by default.
        chunk_cap: Overrides max_chunks().
        budget_s: Overrides time_budget_s().
        clock: Monotonic clock (tests pass a fake one).

    Returns:
        TieOutReport.

    Raises:
        Exception: Whatever the store raises. A report that could not read the
            deal must not be shown as one with nothing to tie out.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    from src.vector_db.constants import COLLECTION_NAME

    if client is None:
        from src.vector_db.qdrant_client import get_qdrant_client

        client = get_qdrant_client()
    cap = chunk_cap if chunk_cap is not None else max_chunks()
    budget = budget_s if budget_s is not None else time_budget_s()
    started = clock()
    deadline = started + budget
    read_deadline = started + budget * READ_SHARE

    deal_filter = _deal_filter(deal_id)
    documents, listed_all = await _documents(client, COLLECTION_NAME, deal_filter)
    chosen: list[tuple[str, int]] = []
    planned = 0
    capped = not listed_all
    for name, count in documents:
        if planned + count > cap:
            capped = True
            break
        chosen.append((name, count))
        planned += count

    # Whole documents only, in name order: a document the deadline cut off
    # mid-scroll is left out rather than compared on half its figures.
    by_name: dict[str, list[dict]] = {}
    whole: list[str] = []
    cut = False
    for batch in _batches(chosen, SCROLL_PAGE):
        if clock() > read_deadline:
            cut = True
            break
        names = [name for name, _ in batch]
        scoped = Filter(
            must=[*deal_filter.must, FieldCondition(key="source_file", match=MatchAny(any=names))],
            must_not=deal_filter.must_not,
        )
        payloads, complete = await _scroll(client, COLLECTION_NAME, scoped, read_deadline, clock)
        if not complete:
            cut = True
            break
        found: dict[str, list[dict]] = {name: [] for name in names}
        for payload in payloads:
            found.setdefault(str(payload.get("source_file") or ""), []).append(payload)
        for name, count in batch:
            # Fewer chunks than the facet counted: the document changed under
            # the read, and half a document is not compared.
            if len(found[name]) < count:
                cut = True
                continue
            by_name[name] = found[name]
            whole.append(name)
    ordered = [chunk for name in whole for chunk in by_name[name]]

    # A page is read whole once started, so the read can end past its share,
    # even past the budget (a local-mode store is slow). The comparison still
    # gets its share from there: the budget is then exceeded by at most one
    # page's read, and what was read is never thrown away.
    build_deadline = max(deadline, clock() + budget * (1 - READ_SHARE))
    report, built = await asyncio.to_thread(
        report_from_chunks, deal_id, ordered, lambda: clock() > build_deadline
    )
    cut = cut or built < len(whole)
    # When both bound the read, the budget is named: it is what stopped this
    # one (a deal past the cap still read less than the cap allows).
    reason = "time_budget" if cut else "chunk_cap" if capped else ""
    read = set(whole[:built])
    unread = [name for name, _ in documents if name not in read]
    report.documents_total = len(documents)
    report.documents_read = len(read)
    report.documents_not_read = [_clip(name, MAX_SOURCE_CHARS) for name in unread[:MAX_DOCUMENT_NAMES]]
    report.chunks_read = sum(len(by_name[name]) for name in whole[:built])
    report.truncated = bool(reason)
    report.truncation_reason = reason
    report.elapsed_ms = round((clock() - started) * 1000, 1)
    logger.info(
        "Tie-out report built",
        extra={
            "deal_id": deal_id,
            "documents_read": report.documents_read,
            "documents_total": report.documents_total,
            "chunks_read": report.chunks_read,
            "conflicts": report.conflicts_total,
            "agreements": report.agreements_total,
            "truncated": report.truncation_reason or "no",
            "elapsed_ms": report.elapsed_ms,
        },
    )
    return report


# ==============================================================================
# One build at a time, and a short memory of the last ones
# ==============================================================================
# A report reads a whole deal and parses every table in it — seconds of CPU on
# a large one, on a host with two cores — and the route sits under the read
# limit the consoles share (120 a minute), so builds run one at a time and a
# result is kept for a few minutes. Recording a document (deals.register_document)
# and purging a deal drop its cached report. The cache is per API process; the
# key carries the deal's readable chunk count, so another process's upload is
# not answered from an older report either, unless it left the count unchanged
# (a new version exactly as long as the old): then for at most CACHE_TTL_S.

CACHE_TTL_S = 300.0
CACHE_MAX_ENTRIES = 16
WAIT_FOR_BUILD_S = 30.0

_cache: dict[tuple[str, int], tuple[float, TieOutReport]] = {}
_locks: dict[int, asyncio.Lock] = {}


class TieOutBusy(Exception):
    """Another report was still being built when this one's wait ran out."""


def _build_lock() -> asyncio.Lock:
    """The build lock of the running event loop (tests run several loops)."""
    loop_id = id(asyncio.get_running_loop())
    lock = _locks.get(loop_id)
    if lock is None:
        _locks.clear()
        lock = _locks[loop_id] = asyncio.Lock()
    return lock


def forget_deal(deal_id: str) -> None:
    """Drops every cached report of a deal: a purged sandbox leaves no copy here."""
    for key in [k for k in _cache if k[0] == deal_id]:
        _cache.pop(key, None)


def clear_cache() -> None:
    """Empties the report cache. For tests."""
    _cache.clear()


def _cached(key: tuple[str, int], now: float) -> TieOutReport | None:
    hit = _cache.get(key)
    if hit is None:
        return None
    if hit[0] <= now:
        _cache.pop(key, None)
        return None
    return hit[1]


def _remember(key: tuple[str, int], report: TieOutReport, now: float) -> None:
    for stale in [k for k, (expires, _) in _cache.items() if expires <= now]:
        _cache.pop(stale, None)
    forget_deal(key[0])
    while len(_cache) >= CACHE_MAX_ENTRIES:
        _cache.pop(min(_cache, key=lambda k: _cache[k][0]), None)
    _cache[key] = (now + CACHE_TTL_S, report)


async def get_tie_out(deal_id: str, client=None, wait_s: float = WAIT_FOR_BUILD_S) -> TieOutReport:
    """
    The deal's tie-out report: from the cache while the deal is unchanged,
    otherwise built — one build at a time across the process.

    Args:
        deal_id: Deal to report on.
        client: AsyncQdrantClient; the application's singleton by default.
        wait_s: How long to wait for another build to finish.

    Returns:
        TieOutReport.

    Raises:
        TieOutBusy: When another build held the lock for all of `wait_s`.
        Exception: Whatever the store raises (see build_tie_out).
    """
    from src.vector_db.constants import COLLECTION_NAME

    if client is None:
        from src.vector_db.qdrant_client import get_qdrant_client

        client = get_qdrant_client()
    counted = await client.count(
        collection_name=COLLECTION_NAME, count_filter=_deal_filter(deal_id), exact=True
    )
    key = (deal_id, int(counted.count))
    report = _cached(key, time.monotonic())
    if report is not None:
        return report

    lock = _build_lock()
    try:
        await asyncio.wait_for(lock.acquire(), timeout=wait_s)
    except asyncio.TimeoutError:
        raise TieOutBusy(deal_id) from None
    try:
        # Another request may have built it while this one waited.
        report = _cached(key, time.monotonic())
        if report is None:
            report = await build_tie_out(deal_id, client=client)
            _remember(key, report, time.monotonic())
        return report
    finally:
        lock.release()
