"""
Document identity on text as the PDF path delivers it (Decision 89).

A PDF's opening differs from a text file's in ways the identity extractor
(Decision 86) was not written for: words separated by no-break spaces (pages
printed from HTML), a line break wherever the page wrapped, long titles
centred over two or three lines, and clause sections headed by the number the
segmenter read off them. Fixture openings are written from those layouts; no
benchmark document is copied.
"""

from __future__ import annotations

from src.data_processing.document_identity import (
    document_opening,
    extract_document_identity,
    extract_parties,
    extract_title,
    name_tokens,
)

NBSP = "\u00a0"


def _tokens(names: list[str]) -> set[str]:
    return set().union(*(name_tokens(n) for n in names)) if names else set()


def test_no_break_spaces_between_words_do_not_hide_title_or_parties():
    opening = (
        f"JOINT{NBSP}VENTURE{NBSP}AGREEMENT\n"
        f"This{NBSP}Agreement{NBSP}is{NBSP}made{NBSP}by{NBSP}and{NBSP}between{NBSP}Quillfeather{NBSP}Labs,{NBSP}Inc.,"
        f"{NBSP}a{NBSP}Nevada{NBSP}corporation{NBSP}(\u201cQuillfeather\u201d),{NBSP}and{NBSP}Marrowstone"
        f"{NBSP}Freight{NBSP}Corporation,{NBSP}a{NBSP}Delaware{NBSP}corporation{NBSP}(\u201cMarrowstone\u201d).\n"
    )
    assert extract_title(opening) == "JOINT VENTURE AGREEMENT"
    parties = extract_parties(opening)
    assert "Quillfeather Labs, Inc." in parties
    assert "Marrowstone Freight Corporation" in parties
    assert not any(NBSP in p for p in parties)


def test_a_line_wrapped_after_the_article_is_still_a_description():
    opening = (
        "SUPPLY AGREEMENT\n"
        "This Supply Agreement is entered into by and between Tamarack Resins, LLC, a\n"
        "Delaware limited liability company (\u201cTamarack\u201d) and Pellucid Coatings, LLC, an\n"
        "Oklahoma limited liability company (\u201cPellucid\u201d).\n"
    )
    parties = extract_parties(opening)
    assert "Tamarack Resins, LLC" in parties and "Pellucid Coatings, LLC" in parties
    # Not the state read as an organisation with the legal form "limited".
    assert not {"delaware", "oklahoma"} & _tokens(parties)


def test_exhibit_a_above_an_organisation_still_names_it():
    # Only a lower-case article ends a line as a description does; "EXHIBIT A" does not.
    opening = "EXHIBIT A\nHALVORSEN MARINE CORPORATION\nSERVICES AGREEMENT\n"
    assert {"halvorsen", "marine"} <= _tokens(extract_parties(opening))


def test_a_title_centred_over_two_lines_is_read_whole():
    opening = (
        "JOINT DEVELOPMENT AND MARKETING\n"
        "AGREEMENT\n"
        "BETWEEN\n"
        "KESTREL CELLS SA\n"
    )
    assert extract_title(opening) == "JOINT DEVELOPMENT AND MARKETING AGREEMENT"


def test_a_title_over_three_lines_skips_the_exhibit_number_above_it():
    opening = (
        "Exhibit 10.10\n"
        "NORTHERN RING CABLE SYSTEM 3\n"
        "CONSTRUCTION AND MAINTENANCE\n"
        "AGREEMENT\n"
        "Reykjavik\n"
    )
    assert extract_title(opening) == "NORTHERN RING CABLE SYSTEM 3 CONSTRUCTION AND MAINTENANCE AGREEMENT"


def test_a_title_continued_by_a_lower_case_line_takes_the_line_above():
    opening = "Amendment n\u00b0 02\nto the Fleet Support Agreement\nref. FS-118 issue 3\n"
    assert extract_title(opening) == "Amendment n\u00b0 02 to the Fleet Support Agreement"


def test_a_version_stamp_or_a_sentence_above_a_bare_title_is_not_joined():
    assert extract_title(
        "Execution Copy\nCOLLABORATIVE RESEARCH AND COMMERCIALIZATION\nAGREEMENT\n"
    ) == "COLLABORATIVE RESEARCH AND COMMERCIALIZATION AGREEMENT"
    assert extract_title("The parties agree as follows.\nAGREEMENT\n") == "AGREEMENT"


def test_a_complete_title_on_one_line_is_unchanged():
    opening = "NORTHWIND TRADERS LLC\nDISTRIBUTOR AGREEMENT\nThis Distributor Agreement is made ...\n"
    assert extract_title(opening) == "DISTRIBUTOR AGREEMENT"


def test_a_table_of_contents_line_is_not_a_title():
    opening = (
        "TABLE OF CONTENTS\n"
        "Section 2.10 No Duplication of Terms 16\n"
        "Section 2.11 Notices 17\n"
        "Article IV Licence Terms 20\n"
    )
    assert extract_title(opening) == ""


def test_clause_numbers_are_not_put_back_as_headings():
    payloads = [
        {"chunk_index": 0, "text": "SERVICES AGREEMENT\nThis Agreement is made by and between Orrin Data, Inc. "
                                    "and Velde Systems GmbH.", "section_heading": "PREAMBLE", "clause_id": "PREAMBLE"},
        {"chunk_index": 1, "text": "(a) \u201cServices\u201d means the services in Schedule 1.",
         "section_heading": "1(a)", "clause_id": "1(a)"},
        {"chunk_index": 2, "text": "Payment is due within 30 days.", "section_heading": "FEES", "clause_id": None},
    ]
    opening = document_opening(payloads)
    assert opening.splitlines()[0] == "SERVICES AGREEMENT"
    assert "PREAMBLE" not in opening and "1(a)" not in opening
    # A heading the document printed is still put back.
    assert "FEES\nPayment is due within 30 days." in opening
    assert extract_document_identity(payloads)["doc_title"] == "SERVICES AGREEMENT"
