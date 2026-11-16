"""
Prefer the filing for the period a question names.

A data room for a public target holds several years of one company's filings.
They repeat each other — every 10-K restates the prior years as comparatives,
and the boilerplate barely changes — so "What was FY2018 capex?" scores the
FY2019 10-K about as high as the FY2018 one. Measured on FinanceBench, all 84
filings in one store (Decision 65): the top reranked chunk came from the wrong
filing on 43 of 150 questions, 35 of them another filing of the same company.

Each filing carries, from ingestion, the fiscal years a question may call it
by and the identities of its registrant (src/data_processing/filing_metadata.py,
read from its cover page, never from the file name). When a question names a
fiscal year and the first search returns chunks of a filing that does not
cover it, one more filtered search looks for a filing of the *same registrant*
that does. If there is one, the off-period filing's chunks are set aside and
the on-period filing's best chunks take their place in the candidate lists;
fusion and reranking then run exactly as before.

What it deliberately does not do:
  - filter on the year alone. "Did Pfizer's net income jump in 2019?" asked of
    a room whose Pfizer filings are 2021 and 2023 must keep Pfizer, not swap in
    another company's 2019 10-K. Only a sibling filing can displace a filing;
  - act when the question names no year, or names one that no filing of that
    registrant covers (a question about a period absent from the room keeps
    the context it would have had, so the answerability gate sees the same
    evidence and can refuse);
  - read the company from the question. The registrant is whoever the
    retrieved chunks belong to, so tickers, nicknames ("JnJ") and former names
    need no alias table;
  - touch documents without filing metadata (contracts, CIMs, the sample deal):
    for them this is a no-op by construction.

A named quarter narrows the period (Decision 70). "Q2 2023" is answered by
the second-quarter 10-Q (or that quarter's release), not by the 2023 10-K,
which covers the year too; "Q4 2023" is answered by the annual report or the
fourth-quarter release, never by a 10-Q. So when the question names a quarter,
the first sibling search asks for a filing of that quarter, and only if the
registrant has none does the step fall back to the year alone — a quarter
absent from the room keeps the year's filings, as before. A filing whose
quarter is not stated (an 8-K, a 10-Q whose statements were not on its
opening pages) is never set aside for its quarter and never preferred for it.
FILING_QUARTER_PREFERENCE=0 reads the year only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from qdrant_client.models import FieldCondition, MatchAny, MatchValue, ScoredPoint

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# A two-digit year is not the start of a form name or a figure ("Q3 10-Q", "Q2 10%").
_YY = r"'?((?:19|20)\d\d(?!\d)|\d\d(?![\d\-%]))"
_FY = r"(?:(?:FY|fiscal(?:\s+year)?)\s?)?"
_ORDINAL = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
# Years a question qualifies as fiscal periods, with the quarter when one is
# named: (pattern, year group, quarter group or 0).
#   FY2018, FY 2022, FY22, fiscal 2023, fiscal year 2021;
#   Q2 2023, Q2'2023, Q22023, Q2 of FY2024, Q1 FY23; 2021 Q1, FY2023Q1; 2Q22;
#   second quarter of 2023, second fiscal quarter of 2023, first-quarter 2023,
#   fourth quarter of fiscal 2022; 2023 second quarter.
_QUALIFIED = (
    (re.compile(r"\bFY\s?" + _YY, re.IGNORECASE), 1, 0),
    (re.compile(r"\bfiscal\s+(?:year\s+)?((?:19|20)\d\d)(?!\d)", re.IGNORECASE), 1, 0),
    (re.compile(r"\bQ([1-4])\s?(?:of\s+)?" + _FY + _YY, re.IGNORECASE), 2, 1),
    (re.compile(r"(?<!\d)((?:19|20)\d\d)\s?Q([1-4])\b", re.IGNORECASE), 1, 2),
    (re.compile(r"\b([1-4])Q\s?" + _YY, re.IGNORECASE), 2, 1),
    (re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)[\s-]+(?:fiscal\s+)?quarter\s+(?:of\s+)?"
                + _FY + r"((?:19|20)\d\d)(?!\d)", re.IGNORECASE), 2, 1),
    (re.compile(r"(?<!\d)((?:19|20)\d\d)\s+(first|second|third|fourth)[\s-]+(?:fiscal\s+)?quarter\b",
                re.IGNORECASE), 1, 2),
)
# Any other year: "in 2022", "as of May 26, 2023". Not a figure ("$2,023",
# "2023.5") and not part of a longer number.
_BARE_YEAR = re.compile(r"(?<![\d$.,])((?:19|20)\d\d)(?![\d%]|\.\d)")


def _switch(name: str) -> bool:
    return os.getenv(name, "1").strip().lower() not in {"0", "false", "off", "no"}


def period_preference_enabled() -> bool:
    """FILING_PERIOD_PREFERENCE=0 turns the preference off (it is on by default)."""
    return _switch("FILING_PERIOD_PREFERENCE")


def quarter_preference_enabled() -> bool:
    """FILING_QUARTER_PREFERENCE=0 makes the preference read the year only (on by default)."""
    return _switch("FILING_QUARTER_PREFERENCE")


def _year(token: str) -> int:
    value = int(token)
    return 2000 + value if value < 100 else value


def _quarter(token: str) -> int:
    return int(token) if token.isdigit() else _ORDINAL[token.lower()]


def question_fiscal_period(question: str) -> tuple[int | None, int | None]:
    """
    The fiscal year a question is about, and the quarter of it when one is named.

    The latest year qualified as a fiscal period wins ("the FY2017 - FY2019
    average" is answered from the FY2019 filing, "change between FY2021 and
    FY2022" from FY2022, "between FY 2023 and Q2 of FY2024" from Q2 FY2024).
    Bare years count only when no qualified year is present, so "As of FY 2021,
    how much will Verizon pay retirees in 2024?" is about FY2021. The quarter is
    the latest one named for that year.

    Args:
        question: The query or sub-question.

    Returns:
        (four-digit year or None, quarter 1-4 or None).
    """
    found = [(_year(m.group(y)), _quarter(m.group(q)) if q else None)
             for p, y, q in _QUALIFIED for m in p.finditer(question)]
    if found:
        year = max(y for y, _ in found)
        quarters = [q for y, q in found if y == year and q]
        return year, (max(quarters) if quarters else None)
    bare = [int(m.group(1)) for m in _BARE_YEAR.finditer(question)]
    return (max(bare) if bare else None), None


def question_fiscal_year(question: str) -> int | None:
    """The fiscal year a question is about, or None (see question_fiscal_period)."""
    return question_fiscal_period(question)[0]


def _keys(payload: dict) -> set[str]:
    return set(payload.get("filing_entity_keys") or [])


def _off_period(payload: dict, year: int) -> bool:
    """A chunk of a filing whose stated fiscal years do not include `year`."""
    years = payload.get("filing_fiscal_years")
    return bool(years) and bool(_keys(payload)) and year not in years


def _off_quarter(payload: dict, year: int, quarter: int) -> bool:
    """
    A chunk of a filing of the named year but known to report another quarter:
    the annual report for Q1-Q3, any 10-Q for Q4 (a 10-Q is never the fourth
    quarter), a release or 10-Q of another stated quarter.
    """
    if not _keys(payload) or year not in (payload.get("filing_fiscal_years") or []):
        return False
    if str(payload.get("filing_form", "")).startswith("10-Q") and quarter == 4:
        return True
    stated = payload.get("filing_fiscal_quarter")
    return stated is not None and stated != quarter


def _merge(first: list[ScoredPoint], on_period: list[ScoredPoint],
           set_aside: set[str], limit: int) -> list[ScoredPoint]:
    """
    One candidate list, as if the set-aside chunks had never been in the index.

    Scores of the two searches are comparable — the same query vector against
    the same collection — so the union is simply re-sorted by score.
    """
    seen: set[str] = set()
    merged = []
    for point in sorted((*[p for p in first if p.payload["chunk_id"] not in set_aside], *on_period),
                        key=lambda p: p.score, reverse=True):
        chunk_id = point.payload["chunk_id"]
        if chunk_id not in seen:
            seen.add(chunk_id)
            merged.append(point)
    return merged[:limit]


@dataclass
class PeriodPreference:
    """What the preference did for one query (for logs and the trace)."""
    year: int | None = None
    quarter: int | None = None
    set_aside: list[str] = field(default_factory=list)     # source files set aside
    preferred: list[str] = field(default_factory=list)     # on-period source files found

    @property
    def applied(self) -> bool:
        return bool(self.set_aside)


async def prefer_named_period(
    query: str,
    dense_results: list[ScoredPoint],
    sparse_results: list[ScoredPoint],
    *,
    query_vector: list[float],
    query_sparse,
    deal_id: str,
    metadata_filters: dict,
    top_k_dense: int,
    top_k_sparse: int,
) -> tuple[list[ScoredPoint], list[ScoredPoint], PeriodPreference]:
    """
    Replaces chunks of off-period filings by the same registrant's on-period filing.

    Args:
        query: The query or sub-question whose year is read.
        dense_results: First-pass dense hits.
        sparse_results: First-pass sparse hits.
        query_vector: Dense embedding of the query.
        query_sparse: BM25 vector of the query.
        deal_id: Deal scope.
        metadata_filters: The filters of the first pass (applied again).
        top_k_dense: Dense list length to keep.
        top_k_sparse: Sparse list length to keep.

    Returns:
        (dense, sparse, what was done). The lists are returned unchanged when
        the question names no year, no retrieved filing is off-period, or no
        filing of the same registrant covers the period.
    """
    from src.vector_db.hybrid_search import hybrid_search

    outcome = PeriodPreference()
    if not period_preference_enabled():
        return dense_results, sparse_results, outcome
    year, quarter = question_fiscal_period(query)
    if not quarter_preference_enabled():
        quarter = None
    outcome.year, outcome.quarter = year, quarter
    if year is None:
        return dense_results, sparse_results, outcome

    first = [p.payload for p in (*dense_results, *sparse_results)]

    async def sibling(off: list[dict], conditions: list) -> tuple[set[str], list, list, set[str]]:
        """One filtered search for the same registrants' on-period filing."""
        entity_keys = sorted(set().union(*(_keys(p) for p in off))) if off else []
        if not entity_keys:
            return set(), [], [], set()
        on_dense, on_sparse = await hybrid_search(
            query_text=query,
            query_vector=query_vector,
            query_sparse=query_sparse,
            deal_id=deal_id,
            metadata_filters=metadata_filters,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            extra_conditions=[
                FieldCondition(key="filing_entity_keys", match=MatchAny(any=entity_keys)),
                FieldCondition(key="filing_fiscal_years", match=MatchAny(any=[year])),
                *conditions,
            ],
        )
        on_period = [p.payload for p in (*on_dense, *on_sparse)]
        confirmed = set().union(*(_keys(p) for p in on_period)) if on_period else set()
        return {p["chunk_id"] for p in off if _keys(p) & confirmed}, on_dense, on_sparse, confirmed

    set_aside: set[str] = set()
    on_dense: list[ScoredPoint] = []
    on_sparse: list[ScoredPoint] = []
    confirmed: set[str] = set()
    if quarter is not None:
        # The named quarter first: another year's filing or another quarter's
        # filing of this year gives way to the registrant's filing of that quarter.
        off = [p for p in first if _off_period(p, year) or _off_quarter(p, year, quarter)]
        set_aside, on_dense, on_sparse, confirmed = await sibling(
            off, [FieldCondition(key="filing_fiscal_quarter", match=MatchValue(value=quarter))])
    # The year alone, for registrants that have no filing of the named quarter.
    off = [p for p in first if _off_period(p, year) and not _keys(p) & confirmed]
    year_aside, year_dense, year_sparse, _ = await sibling(off, [])
    set_aside |= year_aside
    on_dense, on_sparse = [*on_dense, *year_dense], [*on_sparse, *year_sparse]
    if not set_aside:
        return dense_results, sparse_results, outcome

    outcome.set_aside = sorted({p.get("source_file", "") for p in first if p["chunk_id"] in set_aside})
    outcome.preferred = sorted({p.payload.get("source_file", "") for p in (*on_dense, *on_sparse)})
    logger.info(
        "Preferred the filing for the period the question names",
        extra={"fiscal_year": year, "fiscal_quarter": quarter, "set_aside": outcome.set_aside,
               "preferred": outcome.preferred, "chunks_set_aside": len(set_aside)},
    )
    return (
        _merge(dense_results, on_dense, set_aside, top_k_dense),
        _merge(sparse_results, on_sparse, set_aside, top_k_sparse),
        outcome,
    )
