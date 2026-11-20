"""
Deal-level checklist review grid (Decision 95).

A diligence reviewer reads every material contract in the data room for the
same short list of clauses: what the deal itself triggers (change of control,
assignment), what would bind the buyer after closing (non-compete,
exclusivity, most favoured nation, rights of first refusal), how the contract
can end, how liability is allocated, who owns the IP. This builds that grid —
one row per contract, one column per checklist item — and fills each cell with
the passage of that contract that best answers the item's question, verbatim,
with its page or section, and a status: found, not found, or unclear.

Deterministic and LLM-free: retrieval, the cross-encoder and (on a GPU) the
local answerability check (Laya). No text is generated; a cell shows only what
the document says.

How a cell is filled. Each item is one question (adapted from the CUAD
category it was measured against). Per item there is one deal-wide dense
search and one BM25 search, each grouped by document (`query_points_groups`,
GROUP_SIZE hits a document), so every row gets its own candidates however many
documents the deal holds. Within a document the two lists are fused by rank
(the legal weights of retrieval_strategy) and the top RERANK_DEPTH chunks are
cut into windows of whole sentences of at most WINDOW_CHARS characters; every
(question, window) pair of a batch of rows goes to the cross-encoder in one
call. The best window is the cell's evidence, the next two its alternates.

How a cell gets its status. A cross-encoder's score is not on one scale across
questions — on CUAD's tuning contracts an exact governing-law clause scores
0.013 for its question, while contracts with no change-of-control clause score
0.22 for theirs — so one threshold cannot serve fifteen items. The evidence's
score (with Laya's P(stated) on a GPU) is turned into two probabilities by
logistic models with one slope and an offset per item, fit per reranker on
CUAD contracts: that the contract has the clause, and that it has it and the
shown window is it. A cell is not found when the first is below the item's own
bar (at most 2% of that item's clauses on the dev contracts fall below it — an
item with fewer than 50 there has no bar and is never "not found"); found when
the second is high, with the CPU host's MiniLM only — bge's found bar failed on
contracts held out until the end and is withdrawn (Decision 95); and unclear
otherwise — most cells, since a reviewer is told "not found" only when that is
safe.

Why windows: a child chunk is ~512 tokens (2,158 characters at the median on
CUAD's dev contracts) while the clause a reviewer needs is a sentence or two
(261 at the median, 95% within 800). A cell that showed its whole chunk would
bury the clause, and a grid of 50 contracts would carry megabytes of text.

The deal is read as a question would read it: hybrid_search.deal_read_filter
(this deal, current versions, no PII-flagged passage), and only text a
document prints — not the generated table narrative or computed-metric
summaries, nor a redline's tracked changes. Rows are the deal's contracts —
the documents ingestion filed as "legal" (see is_contract); the others are
listed, with their category, as not in the grid.

Failure classes (Decision 88): reading the store, embedding the questions and
reranking are the build's required steps — a failure is an error (the route's
503), never an empty grid, since "not found" from a grid that read nothing
would be a false statement of absence. The deal filter is a safety step: it is
never relaxed. The answerability check is optional: when Laya is unavailable
it fails open to the reranker's own status model (the CPU host's mode), and the
report says which was used.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from src.utils.fail_open import fail_open_async
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ==============================================================================
# The checklist
# ==============================================================================


@dataclass(frozen=True)
class ChecklistItem:
    """
    One column of the grid.

    Attributes:
        id: Stable identifier (the API's `items` values).
        label: Column heading.
        question: What the cell answers. It is the search query, the question
            the cross-encoder scores passages against and the question Laya
            reads; adapted from the description of the CUAD category the item
            was measured against.
        cuad_category: That CUAD v1 category.
    """

    id: str
    label: str
    question: str
    cuad_category: str


# Chosen for M&A diligence (Decision 95): what the deal triggers, what binds
# the buyer after closing, exit risk, liability exposure, IP ownership,
# governing law and audit rights. The order groups them so.
CHECKLIST: tuple[ChecklistItem, ...] = (
    ChecklistItem(
        "change_of_control", "Change of control",
        "Does a party have the right to terminate, or is its consent or notice required, if the other "
        "party undergoes a change of control, such as a merger, a sale of its stock or of all or "
        "substantially all of its assets, or an assignment by operation of law?",
        "Change Of Control",
    ),
    ChecklistItem(
        "anti_assignment", "Anti-assignment",
        "Is a party's consent or notice required before the contract can be assigned or transferred "
        "to a third party?",
        "Anti-Assignment",
    ),
    ChecklistItem(
        "non_compete", "Non-compete",
        "Is a party prohibited from competing with the other party — from engaging in, developing or "
        "selling competing businesses, products or services, or from operating in a certain territory "
        "or field — during the contract or after it ends?",
        "Non-Compete",
    ),
    ChecklistItem(
        "exclusivity", "Exclusivity",
        "Does the contract grant a party an exclusive right or appointment, or commit a party to deal "
        "only with the other party — to buy all its requirements from it, or not to sell, license or "
        "work with third parties?",
        "Exclusivity",
    ),
    ChecklistItem(
        "no_solicit_of_customers", "No-solicit of customers",
        "Is a party prohibited from soliciting, diverting or doing business with the other party's "
        "customers, clients or partners, during the contract or after it ends?",
        "No-Solicit Of Customers",
    ),
    ChecklistItem(
        "most_favoured_nation", "Most favoured nation",
        "If a third party gets better terms for the technology, goods or services under the contract, "
        "is the buyer entitled to those better terms?",
        "Most Favored Nation",
    ),
    ChecklistItem(
        "right_of_first_refusal", "Right of first refusal",
        "Does a party have a right of first refusal, first offer or first negotiation — a right to be "
        "offered, or to match an offer for, equity, assets, technology, products or distribution "
        "rights before they go to a third party?",
        "Rofr/Rofo/Rofn",
    ),
    ChecklistItem(
        "termination_for_convenience", "Termination for convenience",
        "Can a party terminate the contract without cause, solely by giving notice and letting a "
        "notice period pass?",
        "Termination For Convenience",
    ),
    ChecklistItem(
        "cap_on_liability", "Cap on liability",
        "Does the contract limit a party's liability for breach, such as a maximum amount it can be "
        "liable for or a time limit for bringing claims?",
        "Cap On Liability",
    ),
    ChecklistItem(
        "uncapped_liability", "Uncapped liability",
        "Is a party's liability uncapped for a breach of its obligations, for example because the "
        "limitation of liability does not apply to indemnification, confidentiality or infringement "
        "of intellectual property?",
        "Uncapped Liability",
    ),
    ChecklistItem(
        "liquidated_damages", "Liquidated damages",
        "Does the contract set liquidated damages, a penalty, or a fixed fee or payment due on breach "
        "or on early termination?",
        "Liquidated Damages",
    ),
    ChecklistItem(
        "insurance", "Insurance",
        "Is a party required to maintain insurance for the benefit of the other party?",
        "Insurance",
    ),
    ChecklistItem(
        "ip_ownership_assignment", "IP ownership assignment",
        "Does a party assign, or agree to assign, to the other party its right, title and interest in "
        "intellectual property, inventions, work product or developments, or does such intellectual "
        "property vest in or belong to the other party?",
        "Ip Ownership Assignment",
    ),
    ChecklistItem(
        "governing_law", "Governing law",
        "Which state's or country's law governs the interpretation of the contract?",
        "Governing Law",
    ),
    ChecklistItem(
        "audit_rights", "Audit rights",
        "Does a party have the right to audit the books, records or physical locations of the other "
        "party to ensure compliance with the contract?",
        "Audit Rights",
    ),
)
ITEMS_BY_ID: dict[str, ChecklistItem] = {item.id: item for item in CHECKLIST}


class UnknownChecklistItem(ValueError):
    """An `items` value that names no checklist item."""


def resolve_items(item_ids: list[str] | None) -> tuple[ChecklistItem, ...]:
    """
    The checklist items to build, in the checklist's own order.

    Args:
        item_ids: Item ids, or None / empty for the whole checklist.

    Returns:
        The items, each once.

    Raises:
        UnknownChecklistItem: An id that is not in CHECKLIST.
    """
    if not item_ids:
        return CHECKLIST
    unknown = sorted({i for i in item_ids if i not in ITEMS_BY_ID})
    if unknown:
        raise UnknownChecklistItem(", ".join(unknown))
    wanted = set(item_ids)
    return tuple(item for item in CHECKLIST if item.id in wanted)


# ==============================================================================
# Bounds and settings
# ==============================================================================

# Retrieval. GROUP_SIZE hits a document from each channel are fused; the top
# RERANK_DEPTH chunks have their windows reranked. Chosen per reranker on
# CUAD's tuning contracts (Decision 95): the smallest depth within a point of
# the best evidence hit rate.
GROUP_SIZE = 10
RERANK_DEPTH = 2
RERANK_DEPTHS: dict[str, int] = {"bge-reranker-v2-m3": 2, "ms-marco-MiniLM-L-6-v2": 1}
# "document": the reranker reads "Document: <file> | Section: <heading> | <window>",
# the retrieval executor's form; "section" leaves the file name out.
PASSAGE_FORM = "document"
PASSAGE_FORMS: dict[str, str] = {"bge-reranker-v2-m3": "document", "ms-marco-MiniLM-L-6-v2": "document"}
# Windows of whole sentences, at most this long (95% of the CUAD dev gold spans
# of these categories fit in one), consecutive ones overlapping by about half.
WINDOW_CHARS = 800
# A chunk longer than this many windows' worth is scored on its first ones only.
MAX_WINDOWS_PER_CHUNK = 16
ALTERNATES = 2

# Work. Rows are reranked a batch at a time and the budget is checked between
# batches, so a grid the budget cuts short still has whole rows.
ROWS_PER_BATCH = 8
DEFAULT_MAX_DOCUMENTS = 100
DEFAULT_TIME_BUDGET_S = 60.0
# Documents one facet lists; a deal with more is reported as cut short.
MAX_DEAL_DOCUMENTS = 5_000
METADATA_PAGE = 1024

# Response bounds: document text is untrusted and unbounded, and both consoles
# render it as text.
MAX_PASSAGE_CHARS = WINDOW_CHARS
MAX_ALTERNATE_CHARS = 240
MAX_SOURCE_CHARS = 200
MAX_SECTION_CHARS = 120
MAX_DOCUMENT_NAMES = 50

# What a cell may show: text the document prints. A converted table's
# narrative is generated prose, its metrics summary is computed, its
# row-by-row form is reformatted, and a redline is a tracked-change draft.
SKIPPED_CONTENT_TYPES = ["table_narrative", "computed_metric", "table_row_by_row", "redline"]
PASSAGE_FIELDS = [
    "chunk_id", "doc_id", "source_file", "text", "page_number", "section_heading",
    "chunk_index", "content_type",
]
METADATA_FIELDS = ["doc_id", "source_file", "document_category"]

FOUND, NOT_FOUND, UNCLEAR = "found", "not_found", "unclear"


@dataclass(frozen=True)
class Calibration:
    """
    A probability from a cell's evidence scores:
    sigmoid(intercept + score_weight * logit(score)
            + answerability_weight * logit(answerability) + item_offsets[item]).

    One slope for every item and an offset per item, because a cross-encoder's
    scores are not on one scale across questions: on CUAD's tuning contracts an
    exact governing-law clause scores 0.013 for its question while contracts
    without a change-of-control clause score 0.22 for theirs (Decision 95).
    """

    intercept: float
    score_weight: float
    item_offsets: dict[str, float]
    answerability_weight: float = 0.0

    def probability(self, item_id: str, score: float, answerability: float | None = None) -> float:
        z = self.intercept + self.score_weight * _logit(score) + self.item_offsets.get(item_id, 0.0)
        if self.answerability_weight and answerability is not None:
            z += self.answerability_weight * _logit(answerability)
        return 1 / (1 + math.exp(-max(min(z, 60.0), -60.0)))


@dataclass(frozen=True)
class StatusModel:
    """
    How a cell's scores become its status.

    Attributes:
        present: P(the contract has the clause).
        correct: P(it has the clause and the shown passage is it).
        absent_below: per item, not_found when `present` is below it; an item
            without a bar is never not_found. (One float: the same bar for
            every item.) Per item, because a pooled bar lets a rare item's
            base rate speak for it: on CUAD's dev contracts it marked 4 of 11
            most-favoured-nation clauses "not found" while the pooled rate was
            2%. An item with fewer than 50 clauses among those contracts has
            no bar — 2% of them is less than one clause.
        found_from: found when `correct` is at or above it (None: never).
        reads_answerability: whether the model takes Laya's P(stated).
    """

    present: Calibration
    correct: Calibration
    absent_below: float | dict[str, float]
    found_from: float | None
    reads_answerability: bool = False

    def absent_bar(self, item_id: str) -> float | None:
        """The item's not-found bar, or None when it has none."""
        if isinstance(self.absent_below, dict):
            return self.absent_below.get(item_id)
        return self.absent_below


# Per reranker profile (reranker_calibration.PROFILES), fit by
# eval/run_checklist_eval.py analyse on CUAD's 232 dev contracts (tuning and
# confirmation halves, 3,480 cells; refit after the confirmation look, and
# checked on 48 contracts no round had used — Decision 95). absent_below puts
# at most 2% of an item's clauses below its bar; found_from keeps >= 90% of
# found cells right, passage included.
STATUS_MODELS: dict[str, StatusModel] = {
    "bge-reranker-v2-m3": StatusModel(
        present=Calibration(
            intercept=0.99037, score_weight=0.66413,
            item_offsets={
                "change_of_control": -2.17843,
                "anti_assignment": 1.24921,
                "non_compete": 0.04387,
                "exclusivity": 0.60965,
                "no_solicit_of_customers": -1.57697,
                "most_favoured_nation": -0.65466,
                "right_of_first_refusal": -0.39478,
                "termination_for_convenience": -0.11043,
                "cap_on_liability": 1.32444,
                "uncapped_liability": -0.86579,
                "liquidated_damages": -0.99006,
                "insurance": 0.75722,
                "ip_ownership_assignment": -1.56998,
                "governing_law": 4.41083,
                "audit_rights": 0.79777,
            },
        ),
        correct=Calibration(
            intercept=0.17795, score_weight=0.76551,
            item_offsets={
                "change_of_control": -2.77835,
                "anti_assignment": 1.26436,
                "non_compete": -0.02680,
                "exclusivity": -0.29663,
                "no_solicit_of_customers": -1.16316,
                "most_favoured_nation": -1.18714,
                "right_of_first_refusal": -0.24241,
                "termination_for_convenience": -0.19107,
                "cap_on_liability": 1.60377,
                "uncapped_liability": -0.47321,
                "liquidated_damages": -0.73766,
                "insurance": 0.89379,
                "ip_ownership_assignment": -1.77152,
                "governing_law": 4.29917,
                "audit_rights": 0.91040,
            },
        ),
        # Items without a bar have fewer than 50 clauses among the dev contracts.
        absent_below={
            "change_of_control": 0.057,
            "anti_assignment": 0.062,
            "exclusivity": 0.062,
            "termination_for_convenience": 0.062,
            "cap_on_liability": 0.062,
            "uncapped_liability": 0.062,
            "insurance": 0.049,
            "ip_ownership_assignment": 0.062,
            "governing_law": 0.062,
            "audit_rights": 0.062,
        },
        # Withdrawn (the pre-registered consequence, Decision 95): fit at 0.730,
        # it called 11 of 16 found cells right on the 48 unseen contracts, under
        # the 85% fixed in advance. The `correct` model is kept for the record.
        found_from=None,
    ),
    "ms-marco-MiniLM-L-6-v2": StatusModel(
        present=Calibration(
            intercept=1.04979, score_weight=0.51957,
            item_offsets={
                "change_of_control": -0.93320,
                "anti_assignment": 1.56609,
                "non_compete": -0.21589,
                "exclusivity": 0.35470,
                "no_solicit_of_customers": -1.50088,
                "most_favoured_nation": -1.32429,
                "right_of_first_refusal": -0.31957,
                "termination_for_convenience": -0.76925,
                "cap_on_liability": 1.42036,
                "uncapped_liability": -0.50936,
                "liquidated_damages": -0.61033,
                "insurance": 0.55222,
                "ip_ownership_assignment": -1.73299,
                "governing_law": 3.98904,
                "audit_rights": 0.96312,
            },
        ),
        correct=Calibration(
            intercept=0.31917, score_weight=0.65306,
            item_offsets={
                "change_of_control": -0.92899,
                "anti_assignment": 1.24822,
                "non_compete": -0.77248,
                "exclusivity": -0.20002,
                "no_solicit_of_customers": -1.32434,
                "most_favoured_nation": -1.90205,
                "right_of_first_refusal": -0.07072,
                "termination_for_convenience": -1.41846,
                "cap_on_liability": 1.72525,
                "uncapped_liability": -0.56330,
                "liquidated_damages": -0.09261,
                "insurance": 1.12513,
                "ip_ownership_assignment": -2.00395,
                "governing_law": 3.99523,
                "audit_rights": 1.37251,
            },
        ),
        # Items without a bar have fewer than 50 clauses among the dev contracts.
        absent_below={
            "change_of_control": 0.065,
            "anti_assignment": 0.065,
            "exclusivity": 0.065,
            "termination_for_convenience": 0.064,
            "cap_on_liability": 0.065,
            "uncapped_liability": 0.065,
            "insurance": 0.065,
            "ip_ownership_assignment": 0.054,
            "governing_law": 0.065,
            "audit_rights": 0.065,
        },
        found_from=0.805,
    ),
}
# The same with Laya's answerability read (GPU hosts).
ANSWERABILITY_MODELS: dict[str, StatusModel] = {
    "bge-reranker-v2-m3": StatusModel(
        present=Calibration(
            intercept=0.72429, score_weight=0.57089,
            answerability_weight=1.22155,
            item_offsets={
                "change_of_control": -2.13572,
                "anti_assignment": 1.09536,
                "non_compete": 0.26881,
                "exclusivity": 0.24955,
                "no_solicit_of_customers": -1.49895,
                "most_favoured_nation": -0.77427,
                "right_of_first_refusal": -0.39837,
                "termination_for_convenience": -0.03046,
                "cap_on_liability": 1.43541,
                "uncapped_liability": -0.73086,
                "liquidated_damages": -0.90942,
                "insurance": 0.84510,
                "ip_ownership_assignment": -1.77896,
                "governing_law": 4.22666,
                "audit_rights": 0.69368,
            },
        ),
        correct=Calibration(
            intercept=-0.56121, score_weight=0.63233,
            answerability_weight=2.73805,
            item_offsets={
                "change_of_control": -2.70627,
                "anti_assignment": 1.01235,
                "non_compete": 0.44899,
                "exclusivity": -1.12407,
                "no_solicit_of_customers": -1.00833,
                "most_favoured_nation": -1.26393,
                "right_of_first_refusal": -0.29368,
                "termination_for_convenience": 0.01734,
                "cap_on_liability": 1.99212,
                "uncapped_liability": 0.01174,
                "liquidated_damages": -0.62739,
                "insurance": 1.06516,
                "ip_ownership_assignment": -2.34192,
                "governing_law": 3.88992,
                "audit_rights": 0.63223,
            },
        ),
        # Items without a bar have fewer than 50 clauses among the dev contracts.
        absent_below={
            "change_of_control": 0.037,
            "anti_assignment": 0.048,
            "exclusivity": 0.048,
            "termination_for_convenience": 0.015,
            "cap_on_liability": 0.048,
            "uncapped_liability": 0.048,
            "insurance": 0.048,
            "ip_ownership_assignment": 0.048,
            "governing_law": 0.048,
            "audit_rights": 0.048,
        },
        # Withdrawn (the pre-registered consequence, Decision 95): fit at 0.705,
        # it called 24 of 31 found cells right on the 48 unseen contracts (77%),
        # under the 85% fixed in advance — a short contract that has no clause
        # still carries the item's dev prevalence in its offset.
        found_from=None,
        reads_answerability=True,
    ),
}


def max_documents() -> int:
    """Rows one grid checks (env CHECKLIST_MAX_DOCUMENTS, default 100)."""
    try:
        return max(1, int(os.getenv("CHECKLIST_MAX_DOCUMENTS", str(DEFAULT_MAX_DOCUMENTS))))
    except ValueError:
        return DEFAULT_MAX_DOCUMENTS


def time_budget_s() -> float:
    """Time budget for one grid in seconds (env CHECKLIST_TIME_BUDGET_S, default 60)."""
    try:
        return max(0.1, float(os.getenv("CHECKLIST_TIME_BUDGET_S", str(DEFAULT_TIME_BUDGET_S))))
    except ValueError:
        return DEFAULT_TIME_BUDGET_S


def answerability_mode() -> str:
    """
    Whether the grid reads Laya (env CHECKLIST_ANSWERABILITY): "auto" (default)
    on a GPU only, "1" wherever Laya loads, "0" never.
    """
    value = os.getenv("CHECKLIST_ANSWERABILITY", "auto").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return "on"
    if value in ("0", "false", "no", "off"):
        return "off"
    return "auto"


# ==============================================================================
# Windows
# ==============================================================================

# A sentence ends at ". ; ! ?" (with any closing quote or bracket) followed by
# whitespace, or at a line break. Abbreviations ("Inc.", "Sec.") also end one;
# windows pack several sentences, so a short piece costs nothing.
_SENTENCE_END = re.compile(r"[.;!?][\"'”’)\]]*\s+|\n\s*")


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentences(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Non-empty sentence spans of `text`, none longer than max_chars."""
    cuts = [0] + [m.end() for m in _SENTENCE_END.finditer(text) if 0 < m.end() < len(text)] + [len(text)]
    spans: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        a, b = _trimmed(text, a, b)
        while b - a > max_chars:
            # A run with no sentence end: cut at the last space that fits.
            space = max(text.rfind(" ", a + 1, a + max_chars + 1), text.rfind("\n", a + 1, a + max_chars + 1))
            cut = space if space > a else a + max_chars
            piece = _trimmed(text, a, cut)
            if piece[0] < piece[1]:
                spans.append(piece)
            a, b = _trimmed(text, cut, b)
        if a < b:
            spans.append((a, b))
    return spans


def sentence_windows(
    text: str, max_chars: int = WINDOW_CHARS, max_windows: int = MAX_WINDOWS_PER_CHUNK
) -> list[tuple[int, int]]:
    """
    Windows of whole sentences over `text`, as (start, end) offsets.

    Each window is a run of consecutive sentences at most `max_chars` long
    (a longer sentence is cut at spaces first). The next window starts at the
    first sentence of this one that begins in its second half — or at its last
    sentence — so consecutive windows overlap by about half and a clause that
    straddles two windows is whole in one of them. A window starts and ends on
    a non-space character, and `text[start:end]` is the window, verbatim.

    Args:
        text: Chunk text.
        max_chars: Longest window.
        max_windows: At most this many windows, from the start of the text.

    Returns:
        Windows in order; empty for blank text.
    """
    if not text or max_chars < 1:
        return []
    units = _sentences(text, max_chars)
    windows: list[tuple[int, int]] = []
    i = 0
    while i < len(units) and len(windows) < max_windows:
        start = units[i][0]
        j = i
        while j + 1 < len(units) and units[j + 1][1] - start <= max_chars:
            j += 1
        end = units[j][1]
        windows.append((start, end))
        if j == len(units) - 1:
            break
        middle = start + (end - start) // 2
        k = i + 1
        while k < j and units[k][0] < middle:
            k += 1
        i = k if j > i else j + 1
    return windows


# ==============================================================================
# Rows
# ==============================================================================


@dataclass
class DealDocument:
    """A readable document of the deal."""

    doc_id: str
    source_file: str
    document_category: str
    chunks: int


def is_contract(document: DealDocument) -> bool:
    """
    Whether a document gets a row: ingestion filed it as legal.

    The parties read from a document's opening were tried as a second signal
    and are not one: on the sample data room they were read from every
    document — the financial statements gave "Operating Income", "EBIT", the
    company and its auditor — so they would put statements, decks and memos in
    the grid. The category is not perfect either: with Laya off, ingestion's
    filename rules filed 627 of LegalBench-RAG's 714 agreements as legal (CUAD
    exhibits named after the 10-Q they were filed with went to "financial").
    So the documents left out are listed by name and category, never dropped
    silently.
    """
    return document.document_category == "legal"


def _clip(text: str, limit: int) -> str:
    """`text` with its whitespace collapsed, cut to `limit` characters."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _deal_filter(deal_id: str):
    """What a question may read in this deal (hybrid_search.deal_read_filter)."""
    from src.vector_db.hybrid_search import deal_read_filter

    return deal_read_filter(deal_id)


def _passage_filter(deal_id: str, doc_ids: list[str]):
    """The read filter, scoped to the rows, minus text a document does not print."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    base = _deal_filter(deal_id)
    return Filter(
        must=[*base.must, FieldCondition(key="doc_id", match=MatchAny(any=list(doc_ids)))],
        must_not=[FieldCondition(key="content_type", match=MatchAny(any=SKIPPED_CONTENT_TYPES))],
    )


async def deal_documents(client, collection: str, deal_id: str,
                         doc_ids: list[str] | None = None) -> tuple[list[DealDocument], bool]:
    """
    The deal's readable documents with their category, by name, and whether
    that is all of them.

    A facet lists the documents (at most MAX_DEAL_DOCUMENTS); each document's
    first readable chunk carries the document-level fields.

    Args:
        client: AsyncQdrantClient.
        collection: Child collection.
        deal_id: The deal.
        doc_ids: Only these documents (the eval harness's rows).

    Returns:
        (documents, listed_all).
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    base = _deal_filter(deal_id)
    must = list(base.must)
    if doc_ids is not None:
        must.append(FieldCondition(key="doc_id", match=MatchAny(any=list(doc_ids))))
    facet = await client.facet(
        collection_name=collection, key="doc_id", facet_filter=Filter(must=must),
        limit=MAX_DEAL_DOCUMENTS + 1, exact=True,
    )
    counts = {str(hit.value): int(hit.count) for hit in facet.hits[:MAX_DEAL_DOCUMENTS]}
    listed_all = len(facet.hits) <= MAX_DEAL_DOCUMENTS

    meta: dict[str, dict] = {}
    first = Filter(must=[*must, FieldCondition(key="chunk_index", match=MatchValue(value=0))])
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection, scroll_filter=first, with_payload=METADATA_FIELDS,
            with_vectors=False, limit=METADATA_PAGE, offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            meta.setdefault(str(payload.get("doc_id") or ""), payload)
        if offset is None:
            break
    # A document whose first chunk is not readable (PII-flagged) is described
    # by another of its chunks.
    for doc_id in [d for d in counts if d not in meta]:
        points, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[*base.must, FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            with_payload=METADATA_FIELDS, with_vectors=False, limit=1,
        )
        if points:
            meta[doc_id] = points[0].payload or {}

    documents = [
        DealDocument(
            doc_id=doc_id,
            source_file=str(meta.get(doc_id, {}).get("source_file") or doc_id),
            document_category=str(meta.get(doc_id, {}).get("document_category") or "other"),
            chunks=count,
        )
        for doc_id, count in counts.items()
    ]
    documents.sort(key=lambda d: (d.source_file.lower(), d.doc_id))
    return documents, listed_all


# ==============================================================================
# Candidates and scores
# ==============================================================================


@dataclass
class ScoredWindow:
    """One window of a candidate chunk, with its reranker score."""

    payload: dict
    start: int
    end: int
    raw: float = 0.0
    score: float = 0.0  # on the common scale
    chunk_rank: int = 0  # the chunk's place in the document's fused list

    @property
    def text(self) -> str:
        return str(self.payload.get("text") or "")[self.start:self.end]


def fuse_in_document(dense: list, sparse: list, depth: int) -> list[dict]:
    """
    A document's top chunks for one question: the dense and BM25 hits fused by
    rank, as reciprocal_rank_fusion does (k 60, the legal weights).

    Args:
        dense: The document's dense hits, best first (ScoredPoints).
        sparse: Its BM25 hits, best first.
        depth: Chunks to keep.

    Returns:
        Up to `depth` payloads, best first; ties in chunk_id order.
    """
    from src.agents.retrieval_strategy import RETRIEVAL_CONFIGS

    weights = RETRIEVAL_CONFIGS["legal"]
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for hits, weight in ((dense, weights["dense_weight"]), (sparse, weights["sparse_weight"])):
        for rank, point in enumerate(hits, start=1):
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id") or point.id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (60 + rank)
            payloads.setdefault(chunk_id, payload)
    order = sorted(scores, key=lambda c: (-scores[c], c))
    return [payloads[c] for c in order[:depth]]


def windows_of(chunks: list[dict]) -> list[ScoredWindow]:
    """Every window of the candidate chunks, unscored."""
    out: list[ScoredWindow] = []
    for rank, payload in enumerate(chunks):
        for start, end in sentence_windows(str(payload.get("text") or "")):
            out.append(ScoredWindow(payload=payload, start=start, end=end, chunk_rank=rank))
    return out


def passage_for_reranking(window: ScoredWindow, form: str = PASSAGE_FORM) -> str:
    """The text the cross-encoder scores (retrieval_executor._passage_for_reranking's form)."""
    parts = []
    if form == "document" and window.payload.get("source_file"):
        parts.append(f"Document: {window.payload['source_file']}")
    if window.payload.get("section_heading"):
        parts.append(f"Section: {window.payload['section_heading']}")
    parts.append(window.text)
    return " | ".join(parts)


_query_cache: dict[str, tuple[list[float], object]] = {}


async def _encode_questions(questions: list[str]) -> list[tuple[list[float], object]]:
    """Dense and BM25 vectors of the questions; kept for the process (there are 15)."""
    from src.vector_db.hybrid_search import compute_sparse_bm25
    from src.vector_db.reranker import embed_texts_async, get_embed_executor

    todo = [q for q in questions if q not in _query_cache]
    if todo:
        dense = await embed_texts_async(todo)
        loop = asyncio.get_running_loop()
        sparse = await loop.run_in_executor(get_embed_executor(), lambda: [compute_sparse_bm25(q) for q in todo])
        for question, vector, sparse_vector in zip(todo, dense, sparse):
            _query_cache[question] = (vector.tolist(), sparse_vector)
    return [_query_cache[q] for q in questions]


async def search_candidates(
    client, collection: str, deal_id: str, items: tuple[ChecklistItem, ...],
    doc_ids: list[str], depth: int = RERANK_DEPTH,
) -> dict[tuple[str, str], list[dict]]:
    """
    Per (document, item), the document's top `depth` chunks for the item's question.

    One dense and one BM25 query per item over all the rows, grouped by
    document, so each row gets its own candidates — a deal-wide top-k would
    give the contracts that match best everything and the others nothing.

    Args:
        client: AsyncQdrantClient.
        collection: Child collection.
        deal_id: The deal.
        items: Checklist items.
        doc_ids: The rows.
        depth: Chunks kept per cell.

    Returns:
        {(doc_id, item_id): payloads, best first}; a cell whose document gave
        no hit has no entry.

    Raises:
        Exception: Whatever the store raises.
    """
    from src.vector_db.constants import QUANTIZED_SEARCH_PARAMS

    if not items or not doc_ids:
        return {}
    vectors = await _encode_questions([item.question for item in items])
    scope = _passage_filter(deal_id, doc_ids)

    async def grouped(query, using: str, **extra):
        response = await client.query_points_groups(
            collection_name=collection, query=query, using=using, group_by="doc_id",
            limit=len(doc_ids), group_size=GROUP_SIZE, query_filter=scope,
            with_payload=PASSAGE_FIELDS, **extra,
        )
        return {str(group.id): list(group.hits) for group in response.groups}

    searches = []
    for dense_vector, sparse_vector in vectors:
        searches.append(grouped(dense_vector, "dense", search_params=QUANTIZED_SEARCH_PARAMS))
        searches.append(grouped(sparse_vector, "sparse"))
    results = await asyncio.gather(*searches)

    candidates: dict[tuple[str, str], list[dict]] = {}
    for n, item in enumerate(items):
        dense, sparse = results[2 * n], results[2 * n + 1]
        for doc_id in doc_ids:
            fused = fuse_in_document(dense.get(doc_id, []), sparse.get(doc_id, []), depth)
            if fused:
                candidates[(doc_id, item.id)] = fused
    return candidates


async def score_windows(
    cells: dict[tuple[str, str], list[dict]], items: tuple[ChecklistItem, ...],
    form: str = PASSAGE_FORM,
) -> dict[tuple[str, str], list[ScoredWindow]]:
    """
    Every window of every cell's candidate chunks scored against the cell's
    question, in one cross-encoder call.

    Args:
        cells: {(doc_id, item_id): candidate chunk payloads}.
        items: The checklist items (for their questions).
        form: Passage form (PASSAGE_FORM).

    Returns:
        {(doc_id, item_id): windows, best first}.

    Raises:
        Exception: Whatever the reranker raises.
    """
    from src.decisions.reranker_calibration import active_profile
    from src.vector_db.reranker import rerank_pairs_async

    question = {item.id: item.question for item in items}
    keyed = [(key, window) for key, chunks in cells.items() for window in windows_of(chunks)]
    if not keyed:
        return {key: [] for key in cells}
    raw = await rerank_pairs_async(
        [(question[key[1]], passage_for_reranking(window, form)) for key, window in keyed]
    )
    profile = active_profile()
    scored: dict[tuple[str, str], list[ScoredWindow]] = {key: [] for key in cells}
    for (key, window), value in zip(keyed, raw):
        window.raw = float(value)
        window.score = profile.calibrate(window.raw)
        scored[key].append(window)
    for windows in scored.values():
        windows.sort(key=lambda w: (-w.score, w.chunk_rank, w.start))
    return scored


async def read_answerability(pairs: list[tuple[str, str]]) -> list[float]:
    """
    Laya's P(stated) for (question, passage) pairs, in one batch.

    Raises:
        LayaUnavailable: Laya is disabled or cannot be loaded.
    """
    from src.decisions.answerability import ANSWERABILITY_INSTRUCTION
    from src.decisions.laya_client import adecide_batch, noul

    spec = {"stated": {"type": "noul", "instructions": ANSWERABILITY_INSTRUCTION}}
    answers = await adecide_batch([({"question": q, "passage": p}, spec) for q, p in pairs])
    return [noul(answer.get("stated", {})) for answer in answers]


async def _answerability_or_none(pairs: list[tuple[str, str]]) -> list[float] | None:
    """read_answerability, or None when Laya is unavailable — the quality gate's fallback too."""
    from src.decisions.laya_client import LayaUnavailable

    try:
        return await read_answerability(pairs)
    except LayaUnavailable as e:
        logger.warning(f"Checklist: Laya answerability unavailable, reranker thresholds only: {e}")
        return None


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def cell_status(item_id: str, score: float | None, model: StatusModel,
                answerability: float | None = None) -> str:
    """
    found / not_found / unclear for a cell's evidence (score None: no candidate).

    Found takes precedence: a cell is found when the passage is probably the
    clause, not found only when the clause is probably not in the contract.
    """
    if score is None:
        return UNCLEAR
    if model.found_from is not None and model.correct.probability(item_id, score, answerability) >= model.found_from:
        return FOUND
    bar = model.absent_bar(item_id)
    if bar is not None and model.present.probability(item_id, score, answerability) < bar:
        return NOT_FOUND
    return UNCLEAR


# ==============================================================================
# The report
# ==============================================================================


@dataclass
class ChecklistPassage:
    """A window of a document, verbatim, with where it is."""

    text: str
    page_number: int | None
    section_heading: str
    chunk_id: str
    score: float


@dataclass
class ChecklistCell:
    item_id: str
    status: str  # found | not_found | unclear
    score: float | None = None  # the evidence's reranker score, common scale
    answerability: float | None = None  # Laya's P(stated) for it, when read
    passage: ChecklistPassage | None = None
    alternates: list[ChecklistPassage] = field(default_factory=list)


@dataclass
class ChecklistRow:
    doc_id: str
    source_file: str
    document_category: str
    cells: list[ChecklistCell] = field(default_factory=list)


@dataclass
class ChecklistColumn:
    id: str
    label: str
    question: str
    found: int = 0
    unclear: int = 0
    not_found: int = 0


@dataclass
class ExcludedDocument:
    source_file: str
    document_category: str


@dataclass
class ChecklistReport:
    """The grid: one row per contract checked, one column per item."""

    deal_id: str
    items: list[ChecklistColumn] = field(default_factory=list)
    rows: list[ChecklistRow] = field(default_factory=list)
    documents_total: int = 0
    documents_checked: int = 0
    documents_not_checked: list[str] = field(default_factory=list)
    documents_excluded: list[ExcludedDocument] = field(default_factory=list)
    excluded_total: int = 0
    truncated: bool = False
    truncation_reason: str = ""  # "" | "document_cap" | "time_budget"
    reranker: str = ""
    answerability: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0


def _passage(window: ScoredWindow, limit: int) -> ChecklistPassage:
    payload = window.payload
    page = payload.get("page_number")
    return ChecklistPassage(
        text=_clip(window.text, limit),
        page_number=page if isinstance(page, int) and page > 0 else None,
        section_heading=_clip(payload.get("section_heading") or "", MAX_SECTION_CHARS),
        chunk_id=str(payload.get("chunk_id") or ""),
        score=round(window.score, 4),
    )


def _alternates(windows: list[ScoredWindow]) -> list[ScoredWindow]:
    """The next best windows that do not overlap the evidence or each other."""
    chosen: list[ScoredWindow] = []
    for window in windows[1:]:
        if len(chosen) == ALTERNATES:
            break
        if any(w.payload is window.payload and window.start < w.end and w.start < window.end
               for w in [windows[0], *chosen]):
            continue
        chosen.append(window)
    return chosen


def _status_models() -> tuple[str, StatusModel, StatusModel | None]:
    """The active reranker's profile name, its status model and its model with Laya (if any)."""
    from src.decisions.reranker_calibration import active_profile

    name = active_profile().name
    return name, STATUS_MODELS[name], ANSWERABILITY_MODELS.get(name)


def _use_answerability(model: StatusModel | None) -> bool:
    from src.decisions.laya_client import laya_device, laya_enabled

    mode = answerability_mode()
    if model is None or mode == "off" or not laya_enabled():
        return False
    return mode == "on" or laya_device() == "cuda"


async def build_checklist(
    deal_id: str,
    client=None,
    item_ids: list[str] | None = None,
    doc_ids: list[str] | None = None,
    document_cap: int | None = None,
    budget_s: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ChecklistReport:
    """
    Reads a deal and builds its checklist grid.

    Rows are the deal's contracts by name, at most `document_cap`; every item
    is searched once over all of them, then rows are reranked ROWS_PER_BATCH
    at a time while the time budget lasts (a batch started is finished). Rows
    the cap or the budget left out are named in the report, and `truncated`
    says why.

    Args:
        deal_id: Deal to read.
        client: AsyncQdrantClient; the application's singleton by default.
        item_ids: Checklist items (None: all).
        doc_ids: Only these documents, each a row whatever its category (the
            eval harness's rows).
        document_cap: Overrides max_documents().
        budget_s: Overrides time_budget_s().
        clock: Monotonic clock (tests pass a fake one).

    Returns:
        ChecklistReport.

    Raises:
        UnknownChecklistItem: An unknown item id.
        Exception: Whatever the store or a model raises. A grid that could not
            read the deal must not be shown as one where nothing was found.
    """
    from src.vector_db.constants import COLLECTION_NAME

    items = resolve_items(item_ids)
    if client is None:
        from src.vector_db.qdrant_client import get_qdrant_client

        client = get_qdrant_client()
    cap = document_cap if document_cap is not None else max_documents()
    budget = budget_s if budget_s is not None else time_budget_s()
    started = clock()
    deadline = started + budget
    timings: dict[str, float] = {}

    documents, listed_all = await deal_documents(client, COLLECTION_NAME, deal_id, doc_ids)
    # Documents named by the caller (the harness) are rows whatever their category.
    contracts = [d for d in documents if doc_ids is not None or is_contract(d)]
    excluded = [d for d in documents if doc_ids is None and not is_contract(d)]
    rows = contracts[:cap]
    capped = len(contracts) > cap or not listed_all
    timings["rows"] = round((clock() - started) * 1000, 1)

    profile_name, model, with_laya = _status_models()
    use_laya = _use_answerability(with_laya)
    depth = RERANK_DEPTHS.get(profile_name, RERANK_DEPTH)
    form = PASSAGE_FORMS.get(profile_name, PASSAGE_FORM)

    mark = clock()
    candidates = await search_candidates(client, COLLECTION_NAME, deal_id, items, [d.doc_id for d in rows],
                                         depth=depth)
    timings["search"] = round((clock() - mark) * 1000, 1)

    report_rows: list[ChecklistRow] = []
    cut = False
    rerank_ms = laya_ms = 0.0
    for b in range(0, len(rows), ROWS_PER_BATCH):
        if clock() > deadline:
            cut = True
            break
        batch = rows[b:b + ROWS_PER_BATCH]
        cells = {(d.doc_id, item.id): candidates[(d.doc_id, item.id)]
                 for d in batch for item in items if (d.doc_id, item.id) in candidates}
        mark = clock()
        scored = await score_windows(cells, items, form)
        rerank_ms += (clock() - mark) * 1000

        answerability: dict[tuple[str, str], float] = {}
        if use_laya:
            mark = clock()
            keys = [key for key, windows in scored.items() if windows]
            question = {item.id: item.question for item in items}
            values = await fail_open_async(
                "checklist_answerability", _answerability_or_none,
                [(question[k[1]], scored[k][0].text) for k in keys], default=None,
            )
            laya_ms += (clock() - mark) * 1000
            if values is None:
                use_laya = False  # the reranker's own thresholds, for the rest of the grid too
            else:
                answerability = dict(zip(keys, values))

        for document in batch:
            row = ChecklistRow(
                doc_id=document.doc_id,
                source_file=_clip(document.source_file, MAX_SOURCE_CHARS),
                document_category=document.document_category,
            )
            for item in items:
                windows = scored.get((document.doc_id, item.id)) or []
                if not windows:
                    row.cells.append(ChecklistCell(item_id=item.id, status=UNCLEAR))
                    continue
                best = windows[0]
                p = answerability.get((document.doc_id, item.id))
                if use_laya and p is not None and with_laya is not None:
                    status = cell_status(item.id, best.score, with_laya, p)
                else:
                    status = cell_status(item.id, best.score, model)
                # A not-found cell's passage is only the closest the document
                # came: its opening is enough to judge that, and it has no
                # alternates — which keeps a large grid's response small.
                absent = status == NOT_FOUND
                row.cells.append(ChecklistCell(
                    item_id=item.id,
                    status=status,
                    score=round(best.score, 4),
                    answerability=round(p, 4) if p is not None else None,
                    passage=_passage(best, MAX_ALTERNATE_CHARS if absent else MAX_PASSAGE_CHARS),
                    alternates=[] if absent else [_passage(w, MAX_ALTERNATE_CHARS) for w in _alternates(windows)],
                ))
            report_rows.append(row)
    timings["rerank"] = round(rerank_ms, 1)
    if laya_ms:
        timings["answerability"] = round(laya_ms, 1)

    checked = {row.doc_id for row in report_rows}
    not_checked = [d.source_file for d in contracts if d.doc_id not in checked]
    columns = []
    for n, item in enumerate(items):
        statuses = [row.cells[n].status for row in report_rows]
        columns.append(ChecklistColumn(
            id=item.id, label=item.label, question=item.question,
            found=statuses.count(FOUND), unclear=statuses.count(UNCLEAR), not_found=statuses.count(NOT_FOUND),
        ))
    reason = "time_budget" if cut else "document_cap" if capped else ""
    report = ChecklistReport(
        deal_id=deal_id,
        items=columns,
        rows=report_rows,
        documents_total=len(documents),
        documents_checked=len(report_rows),
        documents_not_checked=[_clip(n, MAX_SOURCE_CHARS) for n in not_checked[:MAX_DOCUMENT_NAMES]],
        documents_excluded=[
            ExcludedDocument(_clip(d.source_file, MAX_SOURCE_CHARS), d.document_category)
            for d in excluded[:MAX_DOCUMENT_NAMES]
        ],
        excluded_total=len(excluded),
        truncated=bool(reason),
        truncation_reason=reason,
        reranker=profile_name,
        answerability=use_laya,
        timings_ms=timings,
        elapsed_ms=round((clock() - started) * 1000, 1),
    )
    logger.info(
        "Checklist grid built",
        extra={
            "deal_id": deal_id,
            "documents_checked": report.documents_checked,
            "documents_total": report.documents_total,
            "items": len(items),
            "truncated": report.truncation_reason or "no",
            "reranker": profile_name,
            "answerability": use_laya,
            "elapsed_ms": report.elapsed_ms,
        },
    )
    return report


# ==============================================================================
# One build at a time, and a short memory of the last ones
# ==============================================================================
# A grid embeds nothing new but reranks every window of every cell — seconds of
# GPU, much more on the CPU host — and the route sits under the shared read
# limit, so builds run one at a time and a grid is kept for a few minutes.
# Recording a document (deals.register_document) and purging a deal drop its
# cached grids. The cache is per API process; its key carries the deal's
# readable chunk count, so another process's upload is not answered from an
# older grid either, unless it left the count unchanged (then for at most
# CACHE_TTL_S).

CACHE_TTL_S = 600.0
CACHE_MAX_ENTRIES = 16
WAIT_FOR_BUILD_S = 30.0

_cache: dict[tuple, tuple[float, ChecklistReport]] = {}
_locks: dict[int, asyncio.Lock] = {}


class ChecklistBusy(Exception):
    """Another grid was still being built when this one's wait ran out."""


def _build_lock() -> asyncio.Lock:
    """The build lock of the running event loop (tests run several loops)."""
    loop_id = id(asyncio.get_running_loop())
    lock = _locks.get(loop_id)
    if lock is None:
        _locks.clear()
        lock = _locks[loop_id] = asyncio.Lock()
    return lock


def forget_deal(deal_id: str) -> None:
    """Drops every cached grid of a deal: a purged sandbox leaves no copy here."""
    for key in [k for k in _cache if k[0] == deal_id]:
        _cache.pop(key, None)


def clear_cache() -> None:
    """Empties the grid cache and the question vectors. For tests."""
    _cache.clear()
    _query_cache.clear()


def _cached(key: tuple, now: float) -> ChecklistReport | None:
    hit = _cache.get(key)
    if hit is None:
        return None
    if hit[0] <= now:
        _cache.pop(key, None)
        return None
    return hit[1]


def _remember(key: tuple, report: ChecklistReport, now: float) -> None:
    for stale in [k for k, (expires, _) in _cache.items() if expires <= now]:
        _cache.pop(stale, None)
    # Grids of the deal at another chunk count describe an older deal.
    for older in [k for k in _cache if k[0] == key[0] and k[1] != key[1]]:
        _cache.pop(older, None)
    while len(_cache) >= CACHE_MAX_ENTRIES:
        _cache.pop(min(_cache, key=lambda k: _cache[k][0]), None)
    _cache[key] = (now + CACHE_TTL_S, report)


async def get_checklist(deal_id: str, item_ids: list[str] | None = None, client=None,
                        wait_s: float = WAIT_FOR_BUILD_S) -> ChecklistReport:
    """
    The deal's checklist grid: from the cache while the deal is unchanged,
    otherwise built — one build at a time across the process.

    Args:
        deal_id: Deal to read.
        item_ids: Checklist items (None: all).
        client: AsyncQdrantClient; the application's singleton by default.
        wait_s: How long to wait for another build to finish.

    Returns:
        ChecklistReport.

    Raises:
        UnknownChecklistItem: An unknown item id.
        ChecklistBusy: Another build held the lock for all of `wait_s`.
        Exception: Whatever the store or a model raises (see build_checklist).
    """
    from src.vector_db.constants import COLLECTION_NAME

    items = resolve_items(item_ids)
    if client is None:
        from src.vector_db.qdrant_client import get_qdrant_client

        client = get_qdrant_client()
    counted = await client.count(
        collection_name=COLLECTION_NAME, count_filter=_deal_filter(deal_id), exact=True
    )
    key = (deal_id, int(counted.count), tuple(item.id for item in items))
    report = _cached(key, time.monotonic())
    if report is not None:
        return report

    lock = _build_lock()
    try:
        await asyncio.wait_for(lock.acquire(), timeout=wait_s)
    except asyncio.TimeoutError:
        raise ChecklistBusy(deal_id) from None
    try:
        report = _cached(key, time.monotonic())
        if report is None:
            report = await build_checklist(deal_id, client=client, item_ids=[i.id for i in items])
            _remember(key, report, time.monotonic())
        return report
    finally:
        lock.release()
