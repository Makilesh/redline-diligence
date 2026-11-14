"""
Realistic stress documents for the ingestion tests, generated in code.

tests/fixtures/ingestion_docs.py proves each format parses at all. These model
what a real data room contains and what breaks parsers there: borderless
financial statements with multi-row headers and parenthesised negatives,
workbooks with numeric year headers, percentage and per-share rows, merged
cells, formulas and hidden sheets, contracts with content controls, nested
tables and tracked changes inside cells, and the files that must be refused
loudly — scanned, encrypted, corrupt, mis-encoded.

Every figure a test asserts on is a module-level constant, so the expected
value is stated once, next to the document that carries it.
"""

from __future__ import annotations

import copy
import zipfile
from pathlib import Path

# ── Financial statement used by the PDF and XLSX fixtures ────────────────────
# (in thousands, except per share data). Years are deliberately newest-first,
# the order every 10-K uses, and one period is restated.
STATEMENT_TITLE = "Consolidated Statements of Operations"
STATEMENT_CAPTION = "(in thousands, except per share data)"
STATEMENT_YEARS = ["2023", "2022", "2021"]
STATEMENT_ROWS: list[tuple[str, list[str]]] = [
    ("Total revenues", ["452,800", "387,100", "341,900"]),
    ("Cost of revenues", ["(181,100)", "(158,300)", "(143,600)"]),
    ("Gross profit", ["271,700", "228,800", "198,300"]),
    ("Research and development", ["(96,400)", "(84,700)", "(79,900)"]),
    ("Sales and marketing", ["(81,500)", "(72,600)", "(66,100)"]),
    ("General and administrative", ["(38,200)", "(33,900)", "(31,500)"]),
    ("Restructuring charges", ["(4,300)", "—", "(12,800)"]),
    ("Operating income", ["51,300", "37,600", "8,000"]),
    ("Interest expense, net", ["(8,200)", "(9,100)", "(9,600)"]),
    ("Income (loss) before income taxes", ["43,100", "28,500", "(1,600)"]),
    ("Provision for income taxes", ["(9,900)", "(6,600)", "(400)"]),
    ("Net income (loss)", ["33,200", "21,900", "(2,000)"]),
    ("Diluted earnings (loss) per share", ["1.38", "0.92", "(0.09)"]),
    ("Weighted-average diluted shares", ["24,058", "23,804", "22,222"]),
]
STATEMENT_FOOTNOTE = (
    "(1) Restructuring charges relate to the consolidation of the Austin facility."
)


def _statement_page(page, rows, *, top: float, with_title: bool, continued: bool = False):
    """Draws one borderless statement page: label column, right-aligned amounts."""
    import fitz

    y = top
    if with_title:
        page.insert_text((72, y), "AURORA TECHNOLOGIES INC.", fontsize=13)
        y += 20
        title = STATEMENT_TITLE + (" (continued)" if continued else "")
        page.insert_text((72, y), title, fontsize=13)
        y += 16
        page.insert_text((72, y), STATEMENT_CAPTION, fontsize=9)
        y += 22
    # Two-row header: a spanning caption, then the years.
    page.insert_text((330, y), "Year Ended December 31,", fontsize=9)
    y += 13
    col_right = [400, 470, 540]
    for right, year in zip(col_right, STATEMENT_YEARS):
        width = fitz.get_text_length(year, fontsize=9)
        page.insert_text((right - width, y), year, fontsize=9)
    y += 16
    for label, values in rows:
        page.insert_text((72, y), label, fontsize=9)
        for right, value in zip(col_right, values):
            width = fitz.get_text_length(value, fontsize=9)
            page.insert_text((right - width, y), value, fontsize=9)
        y += 14
    return y


def write_borderless_statement_pdf(
    directory: Path, name: str = "aurora_10k_statements.pdf"
) -> Path:
    """
    Two-page income statement with no ruling lines — the layout of an actual
    10-K. pdfplumber's default lattice strategy finds no table here, so the
    numbers reach the index only if the text path keeps each row intact.
    """
    import fitz

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 60), "Item 8. Financial Statements and Supplementary Data", fontsize=16)
    page1.insert_text(
        (72, 84),
        "The following statements should be read with the accompanying notes.",
        fontsize=9,
    )
    _statement_page(page1, STATEMENT_ROWS[:8], top=120, with_title=True)
    page1.insert_text((72, 760), "Aurora Technologies Inc. | Form 10-K | Page 41", fontsize=7)

    page2 = doc.new_page()
    y = _statement_page(page2, STATEMENT_ROWS[8:], top=72, with_title=True, continued=True)
    page2.insert_text((72, y + 14), STATEMENT_FOOTNOTE, fontsize=8)
    page2.insert_text((72, 760), "Aurora Technologies Inc. | Form 10-K | Page 42", fontsize=7)

    path = directory / name
    doc.save(str(path))
    doc.close()
    return path


RULED_HEADER = ["Segment", "FY2022", "FY2023"]
RULED_ROWS = [
    ["Platform subscriptions", "301.2", "362.9"],
    ["Professional services", "55.4", "58.1"],
    ["Hardware resale", "30.5", "31.8"],
    ["Analytics add-ons", "12.9", "19.6"],
    ["Marketplace fees", "8.4", "11.3"],
    ["Training", "3.1", "3.9"],
    ["Support renewals", "41.7", "44.2"],
    ["Other", "(2.6)", "(1.9)"],
]


def _ruled_table(page, rows, *, left=72, top=120, col_w=150, row_h=24):
    """Draws a fully ruled grid pdfplumber's lattice strategy detects."""
    n_rows, n_cols = len(rows), len(rows[0])
    for r in range(n_rows + 1):
        yy = top + r * row_h
        page.draw_line((left, yy), (left + n_cols * col_w, yy))
    for c in range(n_cols + 1):
        xx = left + c * col_w
        page.draw_line((xx, top), (xx, top + n_rows * row_h))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((left + c * col_w + 6, top + r * row_h + 16), cell, fontsize=11)


def write_ruled_multipage_pdf(
    directory: Path, name: str = "segment_revenue_schedule.pdf", repeat_header: bool = True
) -> Path:
    """
    A ruled table split across two pages under a "(USD in millions)" caption
    that sits OUTSIDE the grid. With repeat_header the second page repeats the
    column header (what Word and Excel do when printing); without it the
    continuation is headerless.
    """
    import fitz

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 70), "Revenue by Segment", fontsize=18)
    page1.insert_text((72, 100), "(USD in millions)", fontsize=10)
    _ruled_table(page1, [RULED_HEADER] + RULED_ROWS[:4])
    page2 = doc.new_page()
    second = RULED_ROWS[4:]
    _ruled_table(page2, ([RULED_HEADER] + second) if repeat_header else second, top=72)

    path = directory / name
    doc.save(str(path))
    doc.close()
    return path


def write_scanned_pdf(directory: Path, name: str = "signed_side_letter_scan.pdf") -> Path:
    """Three pages that are images only — no text layer, as a scanner produces."""
    import fitz

    doc = fitz.open()
    pixmap = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 200, 260), False)
    pixmap.clear_with(235)
    for _ in range(3):
        page = doc.new_page()
        page.insert_image(fitz.Rect(36, 36, 559, 806), pixmap=pixmap)
    path = directory / name
    doc.save(str(path))
    doc.close()
    return path


def write_partly_scanned_pdf(directory: Path, name: str = "lease_with_scanned_exhibit.pdf") -> Path:
    """One text page followed by two image-only pages (a scanned exhibit)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Office Lease Summary", fontsize=18)
    page.insert_text(
        (72, 120),
        "The premises at 500 Congress Avenue are leased through March 31, 2029.",
        fontsize=11,
    )
    pixmap = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 200, 260), False)
    pixmap.clear_with(235)
    for _ in range(2):
        scanned = doc.new_page()
        scanned.insert_image(fitz.Rect(36, 36, 559, 806), pixmap=pixmap)
    path = directory / name
    doc.save(str(path))
    doc.close()
    return path


def write_encrypted_pdf(directory: Path, name: str = "board_minutes_protected.pdf") -> Path:
    """A PDF that needs a password to open."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Board minutes: the offer of $58.00 per share was approved.")
    path = directory / name
    doc.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    doc.close()
    return path


def write_corrupt_pdf(directory: Path, name: str = "truncated_upload.pdf") -> Path:
    """A PDF header followed by garbage — an interrupted upload."""
    path = directory / name
    path.write_bytes(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\n" + b"\x00\xff" * 64)
    return path


def write_legal_pdf(directory: Path, name: str = "share_purchase_agreement.pdf") -> Path:
    """
    Two-page contract in body text with numbered clauses, a running footer on
    each page, and one clause whose sentence continues across the page break.
    """
    import fitz

    doc = fitz.open()
    lines_page1 = [
        "SHARE PURCHASE AGREEMENT",
        "",
        "ARTICLE 8 INDEMNIFICATION",
        "8.1 Survival. The representations and warranties of the Seller shall",
        "survive the Closing for a period of eighteen (18) months.",
        "8.2 Cap. The aggregate liability of the Seller under this Article 8",
        "shall not exceed USD 174,000,000, being twenty-five percent (25%) of",
        "the Purchase Price, provided that:",
        "(a) the cap shall not apply to claims arising from fraud; and",
        "(b) the cap shall not apply to breaches of the Fundamental",
    ]
    lines_page2 = [
        "Representations, which shall be capped at the Purchase Price.",
        "8.3 Basket. The Seller shall not be liable until aggregate Losses",
        "exceed USD 3,500,000, after which the Seller is liable from the first",
        "dollar.",
    ]
    for number, lines in enumerate((lines_page1, lines_page2), start=1):
        page = doc.new_page()
        y = 80
        for line in lines:
            if line:
                page.insert_text((72, y), line, fontsize=11)
            y += 16
        page.insert_text(
            (72, 780), f"CONFIDENTIAL - Project Aurora - Page {number} of 2", fontsize=8
        )
    path = directory / name
    doc.save(str(path))
    doc.close()
    return path


# ── XLSX ─────────────────────────────────────────────────────────────────────

XLSX_YEARS = [2021, 2022, 2023]


def write_stress_xlsx(directory: Path, name: str = "aurora_operating_model.xlsx") -> Path:
    """
    A model workbook the way an analyst builds one:

    - "P&L": title rows, a "(USD in thousands, except per share data)" caption,
      NUMERIC year headers, percentage-formatted margin rows, a per-share row
      and a headcount row that must not be scaled by the table's thousands.
    - "Budget vs Actual": a two-row header whose top row is merged cells.
    - "Formulas": formula cells with no cached value (never opened in Excel).
    - "Assumptions": text only.
    - "Hidden Adjustments": a hidden sheet with a number diligence cares about.
    - "Empty": nothing.
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "P&L"
    ws.append(["Aurora Technologies Inc."])
    ws.append(["Consolidated Income Statement"])
    ws.append(["(USD in thousands, except per share data)"])
    ws.append([None])
    ws.append([None, *XLSX_YEARS])
    ws.append(["Total revenues", 341900, 387100, 452800])
    ws.append(["Cost of revenues", -143600, -158300, -181100])
    ws.append(["Gross profit", 198300, 228800, 271700])
    ws.append(["Gross margin %", 0.58, 0.5911, 0.6])
    ws.append(["Operating income", 8000, 37600, 51300])
    ws.append(["Net income (loss)", -2000, 21900, 33200])
    ws.append(["Diluted EPS", -0.09, 0.92, 1.38])
    ws.append(["Headcount (period end)", 1210, 1304, 1412])
    for cell in ws[9][1:]:
        cell.number_format = "0.0%"

    bva = wb.create_sheet("Budget vs Actual")
    bva.append(["USD in millions"])
    bva.append(["Line Item", "FY2023", None, "FY2024", None])
    bva.append([None, "Actual", "Budget", "Actual", "Budget"])
    bva.append(["Revenue", 452.8, 440.0, 498.1, 505.0])
    bva.append(["EBITDA", 77.0, 74.5, 88.2, 91.0])
    bva.merge_cells("B2:C2")
    bva.merge_cells("D2:E2")

    formulas = wb.create_sheet("Formulas")
    formulas.append(["Line Item", "FY2022", "FY2023"])
    formulas.append(["Revenue", 387.1, 452.8])
    formulas.append(["Cost of Revenue", 158.3, 181.1])
    formulas.append(["Gross Profit", "=B2-B3", "=C2-C3"])

    notes = wb.create_sheet("Assumptions")
    notes.append(["Assumption", "Detail"])
    notes.append(["Tax rate", "Statutory rate of twenty-one percent"])
    notes.append(["FX", "EUR translated at the period average"])

    hidden = wb.create_sheet("Hidden Adjustments")
    hidden.append(["USD in thousands"])
    hidden.append(["Adjustment", "FY2022", "FY2023"])
    hidden.append(["Unrecorded litigation reserve", 0, 4200])
    hidden.sheet_state = "hidden"

    wb.create_sheet("Empty")

    path = directory / name
    wb.save(str(path))
    return path


def write_encrypted_office_file(directory: Path, name: str = "cap_table_protected.xlsx") -> Path:
    """
    A password-protected Office file. Encrypted OOXML is an OLE compound file,
    not a zip, so it starts with the OLE magic number; the same is true of a
    legacy .doc/.xls renamed to the new extension.
    """
    path = directory / name
    path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 1024)
    return path


# ── DOCX ─────────────────────────────────────────────────────────────────────


def write_stress_docx(directory: Path, name: str = "spa_negotiation_draft.docx") -> Path:
    """
    A negotiated agreement draft with the constructs python-docx hides:

    - a clause inside a content control (<w:sdt>), as contract templates use;
    - a table with a tracked insertion and deletion INSIDE a cell;
    - a table nested in a table cell;
    - headings that are plain bold paragraphs ("ARTICLE 9 ..."), not styles.
    """
    import docx
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    document = docx.Document()
    document.add_heading("Share Purchase Agreement", level=0)
    document.add_heading("Article 8 Indemnification", level=1)
    document.add_paragraph(
        "The Seller shall indemnify the Buyer for Losses arising from any breach."
    )

    # A paragraph wrapped in a block-level content control.
    sdt = OxmlElement("w:sdt")
    sdt_content = OxmlElement("w:sdtContent")
    controlled = document.add_paragraph(
        "The escrow amount shall be USD 17,400,000 and shall be released after eighteen months."
    )
    controlled._p.getparent().remove(controlled._p)
    sdt_content.append(controlled._p)
    sdt.append(sdt_content)
    document.element.body.insert(len(document.element.body) - 1, sdt)

    document.add_heading("Schedule 1 Earn-out", level=2)
    table = document.add_table(rows=3, cols=3)
    for r, row in enumerate([
        ["Milestone", "FY2024", "FY2025"],
        ["Revenue target", "498.0", "560.0"],
        ["Earn-out payment", "", "20.0"],
    ]):
        for c, value in enumerate(row):
            table.cell(r, c).text = value

    # Tracked change inside the cell (2, 1): "12.5" deleted, "15.0" inserted.
    para = table.cell(2, 1).paragraphs[0]
    deletion = OxmlElement("w:del")
    deletion.set(qn("w:id"), "11")
    deletion.set(qn("w:author"), "Seller Counsel")
    del_run = OxmlElement("w:r")
    del_text = OxmlElement("w:delText")
    del_text.text = "12.5"
    del_run.append(del_text)
    deletion.append(del_run)
    para._p.append(deletion)
    ins = OxmlElement("w:ins")
    ins.set(qn("w:id"), "12")
    ins.set(qn("w:author"), "Seller Counsel")
    ins_run = OxmlElement("w:r")
    ins_text = OxmlElement("w:t")
    ins_text.text = "15.0"
    ins_run.append(ins_text)
    ins.append(ins_run)
    para._p.append(ins)

    # A paragraph styled as body text but functioning as a heading.
    bold = document.add_paragraph()
    bold.add_run("ARTICLE 9 TERMINATION").bold = True
    document.add_paragraph(
        "This Agreement may be terminated by either party if Closing has not occurred "
        "by the Long Stop Date of December 31, 2024."
    )

    # Nested table: an outer one-cell-wide layout table holding a fee table.
    outer = document.add_table(rows=1, cols=2)
    outer.cell(0, 0).text = "Termination fees"
    inner = outer.cell(0, 1).add_table(rows=2, cols=2)
    inner.cell(0, 0).text = "Company termination fee"
    inner.cell(0, 1).text = "USD 20,900,000"
    inner.cell(1, 0).text = "Reverse termination fee"
    inner.cell(1, 1).text = "USD 41,800,000"

    path = directory / name
    document.save(str(path))
    return path


def write_docx_without_body(directory: Path, name: str = "broken_package.docx") -> Path:
    """A zip with the .docx extension but no word/document.xml."""
    path = directory / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("readme.txt", "not a word document")
    return path


# ── PPTX ─────────────────────────────────────────────────────────────────────


def write_stress_pptx(directory: Path, name: str = "management_presentation.pptx") -> Path:
    """
    Deck with text inside a GROUP shape (python-pptx's slide.shapes does not
    descend into groups) and a slide with no title placeholder.
    """
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Transaction Overview"
    group = slide.shapes.add_group_shape()
    box = group.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(8), Inches(1))
    box.text_frame.text = "Net debt at closing is estimated at USD 212.4 million."

    blank = prs.slides.add_slide(prs.slide_layouts[6])
    note = blank.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(8), Inches(1))
    note.text_frame.text = "Customer churn fell to 6.1% in FY2023."

    path = directory / name
    prs.save(str(path))
    return path


# ── TXT ──────────────────────────────────────────────────────────────────────

ENCODED_TXT = (
    "DISCLOSURE SCHEDULE\n\n"
    "Section 4.2 — Indebtedness\n"
    "The Company’s facility of £25.0 million (€29.1 million) is governed by §12 "
    "of the Facility Agreement.\n"
)


def write_txt_encoded(directory: Path, encoding: str, name: str | None = None) -> Path:
    """The same disclosure schedule saved in a non-UTF-8 encoding."""
    path = directory / (name or f"disclosure_schedule_{encoding.replace('-', '')}.txt")
    path.write_bytes(ENCODED_TXT.encode(encoding))
    return path


def write_binary_as_txt(directory: Path, name: str = "export.txt") -> Path:
    """Binary data with a .txt extension."""
    path = directory / name
    path.write_bytes(bytes(range(256)) * 40)
    return path


def docx_tail_run(para):
    """Deep-copies a paragraph's first run (used to add text after tracked changes)."""
    return copy.deepcopy(para.runs[0]._r)
