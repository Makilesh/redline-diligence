"""
One table, one place in the reranker's candidate pool.

A converted table is indexed four ways — narrative, row_by_row (possibly
several chunks), markdown and metrics_summary, all sharing a table_id
(Decision 57) — so that a question phrased in prose, in row labels or in
figures finds it. The cost shows up before the cross-encoder: the reranker
scores only the top `reranker_top_k` fused candidates (20 for financial
questions), and one table can take three or four of them with near-duplicates.
On FinanceBench a statement page that was a single text chunk ranked first;
parsed as a table, its representations were out-ranked by other tables'
duplicates and the page left the top 10 (Decision 65, "representation
crowding").

Capping each table_id at a few slots in the pool frees the rest for other
evidence. Nothing is lost for the answer: once any representation of a table
is retrieved, sibling expansion (sibling_retrieval.py) adds its grid and the
other representations to the context as before.

Only converter tables are capped (chunks with `table_representation`). A
verbatim table (`table_text`) has a single representation; when the chunker
splits one into several chunks they hold different rows, not duplicates.
"""

from __future__ import annotations

import os


# Chosen on the FinanceBench dev half (Decision 68): one slot per table did at
# least as well as two on every metric.
DEFAULT_TABLE_REPRESENTATION_CAP = 1


def table_representation_cap() -> int:
    """
    TABLE_REPRESENTATION_CAP: candidates one table may hold in the rerank pool
    (0 = no cap; default 1).
    """
    try:
        return max(0, int(os.getenv("TABLE_REPRESENTATION_CAP", str(DEFAULT_TABLE_REPRESENTATION_CAP))))
    except ValueError:
        return DEFAULT_TABLE_REPRESENTATION_CAP


def cap_table_representations(ranked_ids: list[str], payloads: dict[str, dict],
                              per_table: int | None = None) -> list[str]:
    """
    Drops a table's candidates beyond its cap, keeping fused order.

    Args:
        ranked_ids: Chunk ids in fused rank order.
        payloads: chunk_id -> payload.
        per_table: Cap per table_id; None reads TABLE_REPRESENTATION_CAP, 0 disables.

    Returns:
        The ids to offer the reranker, in the same order (callers cut to
        reranker_top_k afterwards, so the freed slots go to the next candidates).
    """
    cap = table_representation_cap() if per_table is None else per_table
    if cap <= 0:
        return ranked_ids
    taken: dict[str, int] = {}
    kept = []
    for chunk_id in ranked_ids:
        payload = payloads.get(chunk_id) or {}
        table_id = payload.get("table_id") if payload.get("table_representation") else None
        if table_id:
            if taken.get(table_id, 0) >= cap:
                continue
            taken[table_id] = taken.get(table_id, 0) + 1
        kept.append(chunk_id)
    return kept
