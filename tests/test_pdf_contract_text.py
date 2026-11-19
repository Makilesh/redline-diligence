"""
A contract's text as the PDF path reads it (Decision 89).

1. A box ruled around the text is not a table. Exhibits printed from HTML,
   and word-processor files with page or paragraph borders, rule a box around
   their text; pdfplumber reports the box as a one-column table whose cells
   hold whole paragraphs with their line breaks collapsed. Taken as a table,
   every row inside it was removed from the page text: on 35 of 232 CUAD
   contracts more than half the text was indexed as flattened table cells,
   with no clause boundaries, no parent chunks, out of reading order, and no
   title line or party clause for the document's identity.
2. Text is read in canonical form. A font's ToUnicode map can name a code
   point canonically equal to the plain one: in 210 of 232 CUAD contracts
   every semicolon came out as U+037E GREEK QUESTION MARK.

Fixtures are generated; no benchmark document is copied.
"""

from __future__ import annotations

import pytest

from src.data_processing.ingest_pipeline import extract_and_chunk
from src.data_processing.multi_page_table_stitcher import MultiPageTableStitcher
from src.data_processing.pdf_processor import PDFProcessor

BODY = [
    "This Agreement is made on 2 March 2020 by and between Ostrava Pumps s.r.o., a Czech",
    'company ("Ostrava"), and Lindqvist Supply AB, a Swedish company ("Lindqvist").',
    "1. Appointment. Ostrava appoints Lindqvist as its exclusive distributor of the Products in",
    "the Territory, and Lindqvist accepts the appointment on the terms of this Agreement.",
    "2. Term. This Agreement runs for three years from the date above and renews for one-year",
    "periods unless either party gives ninety days' written notice before a renewal date.",
    "3. Prices. Lindqvist buys the Products at the prices in Schedule 1, which Ostrava may",
    "change once a year on sixty days' notice.",
]


def write_framed_contract(directory):
    """One page: a title cell and a body cell inside a ruled box, as an exhibit printed from HTML."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    left, right, top, divider = 60, 552, 70, 110
    page.insert_text((220, 95), "DISTRIBUTOR AGREEMENT", fontsize=14)
    y = 135
    for line in BODY:
        page.insert_text((72, y), line, fontsize=10)
        y += 16
    bottom = y
    for yy in (top, divider, bottom):
        page.draw_line((left, yy), (right, yy))
    for xx in (left, right):
        page.draw_line((xx, top), (xx, bottom))
    path = directory / "framed_distributor_agreement.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="module")
def framed(tmp_path_factory):
    return write_framed_contract(tmp_path_factory.mktemp("frame"))


def test_the_fixture_is_a_box_that_pdfplumber_reports_as_a_table(framed):
    tables = MultiPageTableStitcher().stitch_document_tables(str(framed))
    assert len(tables) == 1
    assert PDFProcessor._has_row_labels(tables[0])
    assert PDFProcessor._is_text_frame(tables[0])


def test_the_framed_text_is_read_as_clauses_with_parents(framed):
    doc = extract_and_chunk(str(framed), framed.name, "deal-frames", "legal")
    assert not any(c["is_table"] for c in doc.chunks)
    clause_ids = {c["clause_id"] for c in doc.chunks}
    assert {"1", "2", "3"} <= clause_ids
    assert all(c.get("parent_chunk_id") for c in doc.chunks)
    text = " ".join(" ".join(c["text"] for c in doc.chunks).split())
    for line in BODY:
        assert " ".join(line.split()) in text


def test_the_framed_contract_keeps_its_identity(framed):
    doc = extract_and_chunk(str(framed), framed.name, "deal-frames", "legal")
    first = doc.chunks[0]
    assert first["doc_title"] == "DISTRIBUTOR AGREEMENT"
    parties = " ".join(first["doc_parties"])
    assert "Ostrava Pumps" in parties and "Lindqvist Supply AB" in parties


class _Page:
    """What PyMuPDF's page.get_text("dict") gives for single-span lines (a built-in font cannot carry U+037E)."""

    def __init__(self, lines: list[tuple[float, str]]):
        self.lines = lines

    def get_text(self, kind, sort=False):
        return {"blocks": [{"lines": [{"spans": [{"text": text, "size": 10.0}], "bbox": (72.0, y, 400.0, y + 10.0)}
                                      for y, text in self.lines]}]}


def test_canonically_equal_characters_are_read_in_their_plain_form():
    rows = PDFProcessor._page_rows(_Page([
        (100.0, "by and between Ostrava Pumps s.r.o.\u037e and"),
        (112.0, "Cafe\u0301 Lindqvist AB\u037e"),
    ]))
    assert [r.text for r in rows] == ["by and between Ostrava Pumps s.r.o.; and", "Caf\u00e9 Lindqvist AB;"]


def test_compatibility_characters_are_left_as_written():
    # NFC only: a no-break space or a curly quote is the document's own text.
    rows = PDFProcessor._page_rows(_Page([(100.0, "(the \u201cAgreement\u201d)\u00a0dated")]))
    assert rows[0].text == "(the \u201cAgreement\u201d)\u00a0dated"


def test_a_grid_with_one_non_empty_column_is_a_frame():
    frame = {"headers": [], "rows": [["DISTRIBUTOR AGREEMENT"], ["This Agreement is made ..."]]}
    margin = {"headers": [], "rows": [["This Agreement is made ...", ""], ["1. Term. ...", ""]]}
    fees = {"headers": ["Service", "Fee"], "rows": [["Set-up", "$2,500"], ["Monthly support", "$400"]]}
    signatures = {"headers": [], "rows": [["ACME INC.", "BETA LLC"], ["By: ________", "By: ________"]]}
    assert PDFProcessor._is_text_frame(frame)
    assert PDFProcessor._is_text_frame(margin)
    assert not PDFProcessor._is_text_frame(fees)
    assert not PDFProcessor._is_text_frame(signatures)
