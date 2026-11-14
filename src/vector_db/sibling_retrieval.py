# src/vector_db/sibling_retrieval.py
# Table_id-based sibling fetch for 4-representation tables
# Implemented in Phase 3, BUILD ORDER step 12

"""
Table sibling retrieval for the 4-representation table design.

When a table chunk is retrieved (is_table=1), the retrieval pipeline may
have only found one representation (e.g., narrative). The answer synthesizer
often needs the markdown representation for exact number verification.

This module fetches all sibling representations sharing the same table_id,
using the payload index for efficient filtering. Scroll is acceptable here
because this is a deterministic lookup by table_id, not relevance-based
retrieval.
"""

from __future__ import annotations

import asyncio
import time

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from src.vector_db.constants import (
    COLLECTION_NAME,
    QDRANT_MAX_RETRIES,
    QDRANT_BASE_DELAY_S,
    QDRANT_MAX_DELAY_S,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Siblings added per table when its exact grid is small: the 4 representations,
# plus headroom for a representation the semantic chunker had to split.
MAX_SIBLINGS_PER_TABLE = 8

# The exact grid (markdown, or the verbatim text of a table that was not
# converted) is attached whole up to this many chunks, even past the cap above:
# it is the one representation the synthesizer cannot do without. A 90-row,
# four-period schedule is 6 markdown chunks of ~512 tokens, so this covers
# tables of roughly 180 rows; a larger grid is attached around the hit.
MAX_GRID_CHUNKS_PER_TABLE = 12

# Hard bound on points read for one table_id, and the scroll page size.
MAX_TABLE_POINTS = 512
_SCROLL_PAGE_SIZE = 64

# Order in which the non-grid representations fill the remaining slots: the
# computed metrics carry information the grid does not state; row_by_row and
# narrative restate the grid in words.
_FILL_ORDER = ("metrics_summary", "row_by_row", "narrative")
_GRID_REPRESENTATION = "markdown"


def _representation(payload: dict) -> str | None:
    """A chunk's table representation; None for a verbatim (unconverted) table."""
    return payload.get("table_representation") or None


def _in_table_order(payloads: list[dict]) -> list[dict]:
    """Chunks in document order (chunk_index), ties broken by id for determinism."""
    return sorted(payloads, key=lambda p: (p.get("chunk_index") or 0, str(p.get("chunk_id"))))


def _grid_window(grid: list[dict], targets: list[int], limit: int) -> list[dict]:
    """
    The part of an oversized grid to attach: its first chunk and the chunks
    nearest the hits.

    Only the first markdown chunk carries the header row and the unit note, so it
    is always kept; without it the figures in a later chunk have no column labels.

    Args:
        grid: Every grid chunk of the table, in table order.
        targets: Positions in `grid` the hits correspond to.
        limit: Maximum chunks to return.

    Returns:
        At most `limit` grid chunks, in table order.
    """
    wanted = min(limit, len(grid))
    chosen = {0} if wanted else set()
    distance = 0
    while len(chosen) < wanted:
        for target in targets or [0]:
            for index in (target - distance, target + distance):
                if 0 <= index < len(grid) and len(chosen) < wanted:
                    chosen.add(index)
        distance += 1
    return [grid[i] for i in sorted(chosen)]


def select_siblings(
    table_points: list[dict],
    hits: list[dict],
    required_representations: list[str],
    max_grid_chunks: int = MAX_GRID_CHUNKS_PER_TABLE,
    max_siblings: int = MAX_SIBLINGS_PER_TABLE,
) -> tuple[list[dict], int]:
    """
    Chooses which of a table's chunks to add next to the retrieved ones.

    The exact grid comes first and in table order; the other representations
    fill what is left of `max_siblings`. When the grid is larger than
    `max_grid_chunks`, the chunks kept are the header chunk and those at the same
    relative position in the table as each hit (a hit on the 5th of 8 narrative
    chunks describes rows about five-eighths of the way down the grid).

    Args:
        table_points: Every payload sharing the table_id, hits included.
        hits: The retrieved chunks of this table.
        required_representations: Representations that may be added. A verbatim
            table has none and is always eligible.
        max_grid_chunks: Cap on grid chunks added.
        max_siblings: Cap on chunks added when the grid leaves room.

    Returns:
        (chunks to add, in the order they should follow the hit; number of grid
        chunks left out because the grid exceeded max_grid_chunks).
    """
    hit_ids = {h.get("chunk_id") for h in hits}
    wanted = set(required_representations)
    by_representation: dict[str | None, list[dict]] = {}
    for payload in _in_table_order(table_points):
        by_representation.setdefault(_representation(payload), []).append(payload)

    grid_key = _GRID_REPRESENTATION if _GRID_REPRESENTATION in by_representation else None
    grid = by_representation.get(grid_key, []) if (grid_key is None or grid_key in wanted) else []

    omitted = 0
    if len(grid) > max_grid_chunks:
        targets = []
        for hit in hits:
            peers = by_representation.get(_representation(hit), [])
            position = next(
                (i for i, p in enumerate(peers) if p.get("chunk_id") == hit.get("chunk_id")), 0)
            share = position / (len(peers) - 1) if len(peers) > 1 else 0.0
            targets.append(round(share * (len(grid) - 1)))
        kept = _grid_window(grid, targets, max_grid_chunks)
        omitted = len(grid) - len(kept)
        grid = kept

    selected = [p for p in grid if p.get("chunk_id") not in hit_ids]

    fill_order = [r for r in _FILL_ORDER if r in by_representation] + sorted(
        r for r in by_representation
        if r is not None and r != grid_key and r not in _FILL_ORDER
    )
    for representation in fill_order:
        if representation not in wanted:
            continue
        for payload in by_representation[representation]:
            if len(selected) >= max_siblings:
                break
            if payload.get("chunk_id") not in hit_ids:
                selected.append(payload)
    return selected, omitted


async def fetch_table_siblings(
    chunks: list[dict],
    required_representations: list[str],
    client: AsyncQdrantClient,
    include_pii: bool = False,
    deal_id: str | None = None,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """
    For any chunk with is_table=1, adds the sibling representations sharing its
    table_id. This is required because the 4-representation design indexes
    narrative/row_by_row/metrics_summary/markdown separately.

    Without this step, retrieval might return only the narrative chunk
    while the answer synthesizer needs the markdown for exact number verification.
    Uses the table_id payload index for efficient filtering.

    Every point of the table is read (paginated), so which siblings are added
    depends on the table and the hit, not on the order Qdrant happens to return
    points in. A single scroll capped at 8 used to return an arbitrary subset of
    a large table — a 90-row schedule is 20 points — and a different subset after
    a re-index. See select_siblings for what is kept.

    Siblings are placed directly after the first retrieved chunk of their table,
    grid first. The synthesizer fills its context budget in list order, so
    siblings appended at the end were the first thing it dropped.

    A table whose lookup fails after retries is left without siblings rather
    than failing the query, but not silently: its first retrieved chunk is marked
    `table_siblings_missing`, which Agent 3 reports in its trace.

    Args:
        chunks: Reranked chunks from retrieval (may include non-table chunks).
        required_representations: Representations that may be added.
        client: AsyncQdrantClient.
        include_pii: Compliance authorization override. If False, filters out PII content.
        deal_id: Deal scope for the sibling lookup. table_ids already embed the
                 deal, but the filter is applied anyway (defence in depth); when
                 None it is taken from each chunk's own deal_id payload.
        collection_name: Collection to scroll (default: main child collection).
    Returns:
        The input chunks, each table's siblings inserted after its first hit.
        Marks set on a table's first hit: `table_siblings_missing` (lookup
        failed) and `table_grid_chunks_omitted` (grid larger than the cap).
    """
    start = time.monotonic()
    logger.info(
        "fetch_table_siblings started",
        extra={
            "input_chunks": len(chunks),
            "required_representations": required_representations,
            "include_pii": include_pii,
        },
    )

    # table_id -> its retrieved chunks, in rank order
    hits_by_table: dict[str, list[dict]] = {}
    for c in chunks:
        if c.get("is_table") == 1 and c.get("table_id"):
            hits_by_table.setdefault(c["table_id"], []).append(c)
    if not hits_by_table:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "fetch_table_siblings completed (no table chunks)",
            extra={"elapsed_ms": round(elapsed_ms, 2)},
        )
        return chunks

    logger.info(
        "fetch_table_siblings found table_ids to expand",
        extra={"table_id_count": len(hits_by_table)},
    )

    # NOTE: scroll is acceptable here because we are doing deterministic lookup
    # by table_id (fetching known siblings), not relevance-based retrieval.
    # The "no scroll for retrieval" rule applies to the main search pipeline only.
    async def _scroll_for_table(tid: str, table_deal_id: str | None) -> list[dict]:
        """Every payload of one table_id, with exponential backoff retry."""
        if not table_deal_id:
            # Without a deal scope the lookup could cross deals; skip it.
            logger.warning("Skipping sibling lookup without deal_id", extra={"table_id": tid})
            return []
        must_conditions = [
            FieldCondition(key="deal_id", match=MatchValue(value=table_deal_id)),
            FieldCondition(key="table_id", match=MatchValue(value=tid)),
            FieldCondition(key="is_current_version", match=MatchValue(value=1)),
        ]
        if not include_pii:
            must_conditions.append(
                FieldCondition(key="contains_pii", match=MatchValue(value=0))
            )

        for attempt in range(QDRANT_MAX_RETRIES):
            try:
                op_start = time.monotonic()
                payloads: list[dict] = []
                offset = None
                while True:
                    page, offset = await client.scroll(
                        collection_name=collection_name,
                        scroll_filter=Filter(must=must_conditions),
                        limit=_SCROLL_PAGE_SIZE,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    payloads.extend(point.payload for point in page)
                    if offset is None or len(payloads) >= MAX_TABLE_POINTS:
                        break
                op_elapsed_ms = (time.monotonic() - op_start) * 1000
                logger.info(
                    "scroll for table siblings completed",
                    extra={
                        "table_id": tid,
                        "siblings_found": len(payloads),
                        "elapsed_ms": round(op_elapsed_ms, 2),
                        "attempt": attempt + 1,
                    },
                )
                return payloads
            except Exception as e:
                if attempt == QDRANT_MAX_RETRIES - 1:
                    logger.error(
                        "scroll for table siblings failed after max retries",
                        extra={
                            "table_id": tid,
                            "error": str(e),
                            "attempts": QDRANT_MAX_RETRIES,
                        },
                    )
                    raise
                delay = min(
                    QDRANT_BASE_DELAY_S * (2 ** attempt),
                    QDRANT_MAX_DELAY_S,
                )
                logger.warning(
                    f"scroll for table siblings attempt {attempt + 1} failed, retrying",
                    extra={
                        "table_id": tid,
                        "error": str(e),
                        "delay_s": delay,
                    },
                )
                await asyncio.sleep(delay)
        return []  # unreachable, but satisfies type checker

    # Launch all sibling scroll tasks concurrently
    table_ids = list(hits_by_table)
    sibling_results = await asyncio.gather(
        *[
            _scroll_for_table(tid, deal_id or hits_by_table[tid][0].get("deal_id"))
            for tid in table_ids
        ],
        return_exceptions=True,
    )

    # Choose each table's siblings, deduplicating by chunk_id across the context
    existing_ids = {c["chunk_id"] for c in chunks}
    siblings_after: dict[int, list[dict]] = {}
    siblings_added = 0
    tables_failed = 0
    for tid, table_points in zip(table_ids, sibling_results):
        hits = hits_by_table[tid]
        if isinstance(table_points, BaseException):
            tables_failed += 1
            logger.error(
                "sibling scroll task raised exception",
                extra={"table_id": tid, "error": str(table_points)},
            )
            hits[0]["table_siblings_missing"] = True
            continue
        selected, omitted = select_siblings(table_points, hits, required_representations)
        fresh = [p for p in selected if p["chunk_id"] not in existing_ids]
        existing_ids.update(p["chunk_id"] for p in fresh)
        siblings_after[id(hits[0])] = fresh
        siblings_added += len(fresh)
        if omitted:
            hits[0]["table_grid_chunks_omitted"] = omitted
            logger.warning(
                "table grid larger than the sibling budget; attached the header and "
                "the part nearest the hit",
                extra={"table_id": tid, "grid_chunks_omitted": omitted},
            )

    expanded: list[dict] = []
    for chunk in chunks:
        expanded.append(chunk)
        expanded.extend(siblings_after.get(id(chunk), []))

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "fetch_table_siblings completed",
        extra={
            "table_ids_expanded": len(table_ids),
            "tables_failed": tables_failed,
            "siblings_added": siblings_added,
            "total_chunks": len(expanded),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )
    return expanded
