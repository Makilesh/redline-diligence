"""
Answerability gate settings, re-derived offline from one Laya pass.

The gate has three settings that interact: which passages it reads (top-k of
the reranked list), what it shows Laya of each (the passage form), and where it
vetoes (the threshold). Each combination used to need its own run. Here Laya
scores every one of the top MAX_K passages in every form once, per question,
and every (form, k, threshold) verdict is then recomputed from that matrix with
no model loaded — the gate's own aggregation (max over facets and passages,
veto below the threshold) is exactly reproducible from it.

Passage forms (CHUNK_CONTEXT_HEADERS writes a chunk's `context_header`):

    header   "<context_header>\\n\\n<text>"   — what top_passages() sends today
    title    "<first header line>\\n\\n<text>" — "Document: ...", no section path
    bare     "<text>"

Without a header the three are the same string and are scored once.

Used by `python -m eval.run_answerability_eval --pair-scores` (dev set) and
eval/run_gold_removed_eval.py (LegalBench-RAG with the gold document removed).
"""

from __future__ import annotations

from typing import Iterable

FORMS = ("header", "title", "bare")
MAX_K = 5
THRESHOLDS = tuple(round(0.25 + 0.025 * i, 3) for i in range(11))  # 0.25 .. 0.50
KS = (1, 2, 3, 4, 5)

# Pre-registered (notes, before any result): the lowest dev answerable must sit
# at least this far above the threshold, as today's 0.35 sits 0.059 below 0.409.
MARGIN = 0.04
# Tie-break preference: least change from today's code path first.
FORM_PREFERENCE = {"header": 0, "title": 1, "bare": 2}


def passage_text(chunk: dict, form: str) -> str:
    """One chunk as Laya would read it in `form`."""
    text = chunk.get("text") or ""
    header = (chunk.get("context_header") or "").strip()
    if not header or form == "bare":
        return text
    if form == "title":
        return f"{header.splitlines()[0]}\n\n{text}"
    if form == "header":
        return f"{header}\n\n{text}"
    raise ValueError(f"unknown passage form {form!r}")


def ranked_chunks(chunks: list[dict], k: int = MAX_K) -> list[dict]:
    """The k best-reranked chunks with text, in the order top_passages() uses."""
    return sorted(
        (c for c in chunks if (c.get("text") or "").strip()),
        key=lambda c: float(c.get("reranker_score", 0.0)),
        reverse=True,
    )[:k]


async def pair_scores(facets: list[str], chunks: list[dict], instruction: str,
                      k: int = MAX_K, forms: Iterable[str] = FORMS) -> dict:
    """
    Laya P(stated) for each (form, facet, passage rank), one batched call.

    Returns:
        {form: {facet: [P for rank 1..n]}}, n <= k. A string that is identical
        across forms (no header) is scored once and shared.
    """
    from src.decisions.laya_client import adecide_batch, noul

    top = ranked_chunks(chunks, k)
    spec = {"stated": {"type": "noul", "instructions": instruction}}
    keys: dict[tuple[str, str], int] = {}
    requests = []
    for form in forms:
        for facet in facets:
            for chunk in top:
                pair = (facet, passage_text(chunk, form))
                if pair not in keys:
                    keys[pair] = len(requests)
                    requests.append(({"question": facet, "passage": pair[1]}, spec))
    answers = await adecide_batch(requests)
    scores = [noul(a.get("stated", {})) for a in answers]
    return {form: {facet: [round(scores[keys[(facet, passage_text(c, form))]], 4) for c in top]
                   for facet in facets}
            for form in forms}


def score_at(matrix: dict, form: str, k: int) -> float | None:
    """The gate's question score: max over facets of max over the top k passages."""
    per_facet = [max(ps[:k]) for ps in matrix[form].values() if ps[:k]]
    return max(per_facet) if per_facet else None


def row_score(row: dict, form: str, k: int) -> float | None:
    """
    The gate's score for a harness row, or None when the node would not consult
    Laya. That is a heuristic force_refusal (`consulted` False) — not every
    "refused" route: a context the router sends back for low precision is still
    scored, and counts as caught when vetoed, as in run_answerability_eval.
    """
    consulted = row.get("consulted", row["heuristic"] != "refused")
    return score_at(row["matrix"], form, k) if consulted else None


def laya_decision(heuristic: str, score: float | None, threshold: float) -> str:
    """
    combine_with_answerability's outcome, from the heuristic decision and score.

    Without a score (Laya not consulted) the heuristic stands; otherwise a veto
    refuses and a non-veto leaves the heuristic's decision as it was.
    """
    if score is None:
        return heuristic
    return "refused" if score < threshold else heuristic


def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def evaluate(rows: list[dict], form: str, k: int, threshold: float) -> dict:
    """
    Dev metrics of one setting. Rows carry `label`, `heuristic`, `matrix` and
    optionally `consulted`; only Laya-consulted rows have a score.
    """
    def score(r):
        return row_score(r, form, k)

    ans = [r for r in rows if r["label"] == "answerable"]
    una = [r for r in rows if r["label"] == "unanswerable"]
    ans_scores = [s for s in (score(r) for r in ans) if s is not None]
    una_scores = [s for s in (score(r) for r in una) if s is not None]
    return {
        "form": form, "k": k, "threshold": threshold,
        "false_vetoes": sum(s < threshold for s in ans_scores),
        "answerable": len(ans),
        "caught": sum(s < threshold for s in una_scores),
        "unanswerable": len(una),
        "refused_unanswerable": sum(
            laya_decision(r["heuristic"], score(r), threshold) == "refused" for r in una),
        "lowest_answerable": min(ans_scores) if ans_scores else None,
        "auc": _auc(ans_scores, una_scores),
    }


def grid(rows: list[dict], forms: Iterable[str] = FORMS, ks: Iterable[int] = KS,
         thresholds: Iterable[float] = THRESHOLDS) -> list[dict]:
    """evaluate() for every (form, k, threshold)."""
    return [evaluate(rows, f, k, t) for f in forms for k in ks for t in thresholds]


def admissible(cell: dict, margin: float = MARGIN) -> bool:
    """No false veto, and the threshold at least `margin` below the lowest answerable."""
    low = cell["lowest_answerable"]
    return (cell["false_vetoes"] == 0 and low is not None
            and cell["threshold"] <= round(low - margin, 6))


def select(cells: list[dict], margin: float = MARGIN) -> dict | None:
    """
    The pre-registered choice: most unanswerables caught among admissible
    settings; ties to higher AUC, then k=3, then header > title > bare, then the
    lower threshold.
    """
    ok = [c for c in cells if admissible(c, margin)]
    if not ok:
        return None
    return min(ok, key=lambda c: (-c["caught"], -(c["auc"] or 0.0), c["k"] != 3,
                                  FORM_PREFERENCE[c["form"]], c["threshold"]))
