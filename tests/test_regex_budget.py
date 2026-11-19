"""
Time budget of the regex-heavy text functions on adversarial input at their
realistic maximum size (Decision 88).

Every document and question passes through these functions, and a regex that
backtracks catastrophically turns one crafted upload, or one degenerate model
answer, into seconds or minutes of CPU per request. Two were found quadratic:
name_tokens on a long run of letters or digits (1.8 s for one legal question's
retrieved text, through name_document) and split_claims on a repeated
abbreviation ("Inc. Inc. ..."). The rest measured linear and far under the
250 ms budget on the development laptop (numbers in the decision).

The budgets here are generous on purpose — whole seconds against milliseconds
measured — and the two found cases are tested at sizes where quadratic time
would take far longer than the budget, so a slower CI runner cannot flake them
and a regression cannot pass them.
"""

from __future__ import annotations

import time

import pytest

from src.data_processing import document_identity as di
from src.data_processing.filing_metadata import extract_filing_metadata
from src.data_processing.injection_screen import instruction_spans, screen_text
from src.data_processing.legal_clause_segmenter import LegalClauseSegmenter
from src.data_processing.pii_detector import PIIDetector
from src.data_processing.text_processor import parse_text_sections
from src.utils.numerical_registry import NumericalRegistry
from src.vector_db.filing_period import question_fiscal_period
from src.vector_db.hybrid_search import normalize_for_bm25
from src.vector_db.named_document import name_document
from src.verification.claims import split_claims
from src.verification.numeric_grounding import extract_numbers
from src.verification.prompt_safety import escape_document_text

BUDGET_S = 2.0


def _fill(unit: str, size: int) -> str:
    return (unit * (size // len(unit) + 1))[:size]


def _seconds(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


# Families that stressed the patterns hardest in the timing study.
ADVERSARIAL = ["7", "a", "ABC ", "\n", "between A. ", "THIS ABC, AND ", "Inc. ", "(a) ", "1.1.1.1 ",
               "[a.pdf | p. 3] ", "[[x|", "< "]


class TestFoundQuadratic:
    def test_name_tokens_on_a_long_run(self):
        # Was ~4x slower per doubling: 87 ms at 4,000 characters, so minutes here.
        for unit in ("7", "a"):
            assert _seconds(di.name_tokens, _fill(unit, 200_000)) < BUDGET_S

    def test_naming_a_document_among_retrieved_long_runs(self):
        # 80 retrieved chunks of 4,000 characters: ~7 s before, as extrapolated
        # from 1.8 s at 2,000 characters.
        payloads = [{"chunk_id": f"c{i}", "doc_id": f"d{i % 20}", "source_file": f"d{i % 20}.pdf",
                     "text": _fill("a", 4_000), "doc_title": "SHARE PURCHASE AGREEMENT",
                     "doc_parties": ["Atlas Holdings Inc."]} for i in range(80)]
        assert _seconds(name_document, "What is the fee in the SPA with Atlas?", payloads) < BUDGET_S

    def test_split_claims_on_a_repeated_abbreviation(self):
        # Was 0.2 s at 12,000 characters, ~4x per doubling: ~6 s at 64,000.
        assert _seconds(split_claims, _fill("Atlas Inc. ", 64_000)) < BUDGET_S


@pytest.mark.parametrize("unit", ADVERSARIAL)
class TestRealisticMaxima:
    def test_identity_from_a_4000_character_opening(self, unit):
        payloads = [{"chunk_index": 0, "text": _fill(unit, di.OPENING_CHARS), "contains_pii": 0}]
        assert _seconds(di.extract_document_identity, payloads) < BUDGET_S

    def test_screens_on_a_parent_section(self, unit):
        text = _fill(unit, 12_000)
        for fn in (screen_text, instruction_spans, PIIDetector().detect, normalize_for_bm25,
                   escape_document_text, lambda t: extract_numbers(t, claims_only=False),
                   lambda t: extract_filing_metadata([{"text": t, "page_number": 1}])):
            assert _seconds(fn, text) < BUDGET_S

    def test_a_question(self, unit):
        assert _seconds(question_fiscal_period, _fill(unit, 2_000)) < BUDGET_S

    def test_a_whole_document(self, unit):
        text = _fill(unit, 300_000)
        assert _seconds(LegalClauseSegmenter().segment_document, [(1, text)]) < BUDGET_S
        assert _seconds(parse_text_sections, text) < BUDGET_S

    def test_the_figure_registry_over_a_context(self, unit):
        chunks = [{"source_file": f"f{i}.pdf", "text": _fill(unit, 4_000)} for i in range(10)]
        assert _seconds(NumericalRegistry.from_chunks, chunks) < BUDGET_S
