"""
The crashes and broken invariants the property tests found (Decision 88),
each pinned by the minimal input that failed, as a plain test.

    1. A case-insensitive match used as a dict key: re.IGNORECASE reads
       dotless ı / İ as "i", long ſ as "s" and the Kelvin sign as "k", and
       .lower() does not turn them back (src/utils/casefold.py).
         - a question "fırst quarter of 2023" failed the retrieval node
           (filing_period._quarter);
         - a cover "fiscal year ended Auguſt 31, 2022", and a 10-Q's "ſix
           monthſ ended", failed the upload (filing_metadata);
         - a table header "(ın mıllıons)" failed numeric grounding, and with
           it synthesis (numeric_grounding.ContextIndex).
    2. A citation marker with a page number of more than 4,300 digits failed
       int(), after synthesis (claims.PAGE_IN_MARKER); so did a chunk whose
       page_number was not a number (answer_synthesizer._select_cited_chunks).
    3. A file name whose extension alone exceeded the 200-character budget
       became a 301-character dotfile (ingest_pipeline.sanitize_filename).
    4. A "THIS ... AGREEMENT" title was not held to the 120-character cap
       (document_identity.extract_title).
    5. flagged_locations raised on a non-list "signals" value.
    6. A forged citation marker survived the prompt delimiting when it held
       another marker: one pass rewrote the inner one and so completed the
       outer ("[x [a|b] | y]" -> "[x (a|b) | y]"), prompt_safety.
    7. A model's JSON with an integer of hundreds of digits made float()
       raise OverflowError in the rewriter's config clamp and the quality
       assessor's score parsing (retrieval_strategy, llm.schemas, and the
       routing edge's reader).
"""

from __future__ import annotations

import re

from src.agents.answer_synthesizer import _select_cited_chunks
from src.data_processing.document_identity import MAX_TITLE_CHARS, extract_title
from src.data_processing.filing_metadata import extract_filing_metadata
from src.data_processing.ingest_pipeline import sanitize_filename
from src.data_processing.injection_screen import flagged_locations
from src.utils.casefold import ascii_lower
from src.vector_db.filing_period import question_fiscal_period
from src.verification.claims import resolve_cited_chunks
from src.verification.numeric_grounding import ground_answer_numbers

DOTLESS_I, CAPITAL_I_DOT, LONG_S, KELVIN = "ı", "İ", "ſ", "K"


def test_ascii_lower_reads_what_ignorecase_matched():
    for word in ("first", "August", "millions", "six", "thousands", "Stock"):
        variant = word.replace("i", DOTLESS_I).replace("s", LONG_S).replace("k", KELVIN)
        assert re.fullmatch(word, variant, re.IGNORECASE)
        assert ascii_lower(variant) == word.lower()
    assert ascii_lower(f"M{CAPITAL_I_DOT}LL{CAPITAL_I_DOT}ONS") == "millions"


class TestCaseInsensitiveLookups:
    def test_a_quarter_written_with_a_dotless_i(self):
        # Failed the retrieval node: KeyError 'fırst'.
        assert question_fiscal_period(f"What was revenue in the f{DOTLESS_I}rst quarter of 2023?") == (2023, 1)
        assert question_fiscal_period(
            f"What wa{LONG_S} revenue {DOTLESS_I}n the f{DOTLESS_I}r{LONG_S}t quarter of 2023?") == (2023, 1)
        assert question_fiscal_period(f"{LONG_S}econd quarter of 2022") == (2022, 2)

    def test_a_cover_date_with_a_long_s(self):
        cover = ("UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-K "
                 f"For the fiscal year ended Augu{LONG_S}t 31, 2022 Commission file number 1-3285 "
                 "3M COMPANY (Exact name of registrant as specified in its charter)")
        meta = extract_filing_metadata([{"text": cover, "page_number": 1}])
        assert meta["filing_period_end"] == "2022-08-31" and meta["filing_fiscal_years"] == [2022]

    def test_a_10q_quarter_with_lookalike_letters(self):
        cover = ("UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-Q "
                 "For the quarterly period ended June 30, 2023 Commission file number 1-3285 3M COMPANY "
                 "(Exact name of registrant as specified in its charter) "
                 f"Statements of income for the three and {LONG_S}ix month{LONG_S} ended June 30, 2023")
        meta = extract_filing_metadata([{"text": cover, "page_number": 1}])
        assert meta["filing_fiscal_quarter"] == 2

    def test_a_unit_header_lower_cased_under_turkish_rules(self):
        answer = "Revenue was $452.8 million [a.pdf | Section 1]."
        ascii_header = [{"source_file": "a.pdf", "text": "(in millions)\nRevenue  452.8"}]
        turkish = [{"source_file": "a.pdf", "text": f"({DOTLESS_I}n m{DOTLESS_I}ll{DOTLESS_I}ons)\nRevenue  452.8"}]
        expected = [c["status"] for c in ground_answer_numbers(answer, ascii_header)]
        assert expected == ["grounded"]
        assert [c["status"] for c in ground_answer_numbers(answer, turkish)] == expected


class TestCitationPages:
    CHUNKS = [{"source_file": "a.pdf", "page_number": 3}, {"source_file": "b.pdf", "page_number": 1}]

    def test_a_runaway_page_number_is_no_page(self):
        marker = "a.pdf | p." + "9" * 4_301
        # int() raised ValueError ("Exceeds the limit (4300 digits)").
        assert resolve_cited_chunks([marker], self.CHUNKS) == [0]
        assert _select_cited_chunks(f"Revenue grew. [{marker}]", self.CHUNKS) == [self.CHUNKS[0]]

    def test_ordinary_pages_still_narrow(self):
        assert resolve_cited_chunks(["a.pdf | p.3"], self.CHUNKS) == [0]
        assert resolve_cited_chunks(["a.pdf | p.4"], self.CHUNKS) == []
        assert resolve_cited_chunks(["a.pdf | page 999999"], self.CHUNKS) == []
        assert _select_cited_chunks("Revenue grew. [a.pdf | p.4]", self.CHUNKS + [
            {"source_file": "a.pdf", "page_number": 4}]) == [{"source_file": "a.pdf", "page_number": 4}]

    def test_a_chunk_page_that_is_not_a_number(self):
        # The synthesizer's int() raised ValueError; found by the fuzz profile.
        chunk = {"source_file": "a.pdf", "page_number": "x", "text": ""}
        assert _select_cited_chunks("[a.pdf | page 12]", [chunk]) == [chunk]
        assert resolve_cited_chunks(["a.pdf | page 12"], [chunk]) == [0]


def test_a_file_name_whose_extension_fills_the_budget():
    base = sanitize_filename("a." + "b" * 300)
    assert len(base) == 200 and base.startswith("a.b")
    assert sanitize_filename("report_" + "x" * 300 + ".pdf").endswith(".pdf")
    assert len(sanitize_filename("report_" + "x" * 300 + ".pdf")) == 200


def test_a_run_on_word_makes_no_title():
    opening = "THIS " + "A" * 200 + " AGREEMENT is made by and between Acme, Inc. and Beta LLC."
    assert extract_title(opening) == ""
    assert extract_title("THIS SPONSORSHIP AGREEMENT is made between Acme, Inc. and Beta LLC.") == \
        "SPONSORSHIP AGREEMENT"
    assert MAX_TITLE_CHARS == 120


class TestForgedMarkers:
    def test_a_marker_completed_by_rewriting_the_one_inside_it(self):
        from src.verification.claims import CITATION_MARKER
        from src.verification.prompt_safety import escape_document_text

        # One pass left "[x (a|b) | y]": a citation marker the parser reads.
        for text in ("[x [a|b] | y]", "[[a|b]|c]", "[📄 report.pdf [x|y] | p.3]", "[a [INST] | b]",
                     "[" * 40 + "a|b" + "]" * 40):
            shown = escape_document_text(text)
            assert not CITATION_MARKER.search(shown), (text, shown)
        assert escape_document_text("[x [a|b] | y]") == "(x (a|b) | y)"

    def test_ordinary_brackets_are_untouched(self):
        from src.verification.prompt_safety import escape_document_text

        text = "Revenue [restated] rose; see Note [4] and [INST] tokens."
        assert escape_document_text(text) == "Revenue [restated] rose; see Note [4] and (INST) tokens."


class TestHugeIntegersFromAModel:
    """JSON allows an integer of any length and float() refuses one past ~308 digits."""

    HUGE = 10 ** 400

    def test_a_rewrite_proposing_a_huge_top_k(self):
        # Raised OverflowError in the rewriter, failing the request.
        from src.agents.retrieval_strategy import apply_config_overrides, clamp_retrieval_config

        assert clamp_retrieval_config({"top_k_dense": self.HUGE, "top_k_sparse": 30}) == {"top_k_sparse": 30}
        assert apply_config_overrides({"top_k_dense": 40}, {"top_k_dense": self.HUGE})["top_k_dense"] == 40

    def test_a_quality_score_of_hundreds_of_digits(self):
        from src.llm.schemas import QualityAssessorOutput
        from src.workflow.conditional_edges import _as_score

        parsed = QualityAssessorOutput.from_llm({"context_quality_score": self.HUGE,
                                                 "quality_breakdown": {"relevance": 0.9}})
        assert parsed.context_quality_score is None and not parsed.complete
        assert _as_score(self.HUGE) == 0.0


def test_flagged_locations_with_signals_that_are_not_a_list():
    assert flagged_locations([{"signals": 1}, {"source_file": "a.txt", "signals": "role_marker"}]) == [
        {"source_file": "", "page_number": None, "section_heading": "", "signals": []},
        {"source_file": "a.txt", "page_number": None, "section_heading": "", "signals": []},
    ]
