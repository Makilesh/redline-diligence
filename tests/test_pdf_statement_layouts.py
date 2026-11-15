"""
Statement layouts found in real 10-K PDFs (FinanceBench) that the generated
fixtures did not cover. Each fixture here is generated; none is a filing.

1. Rules under the amount columns only. Filings rendered from EDGAR HTML often
   draw a rule under each figure, so pdfplumber finds a "table" that is a strip
   of numbers with every line item outside it. Accepting it removed the region
   from the page text, labels and all: on NIKE's FY2019 10-K, 32 of 34 ruled
   tables were such strips and the balance sheet page kept 48% of its tokens.
2. A bare-year header, "(Dollars in millions)  2019  2018" — the commonest
   statement header — parsed as the first value row (years look like amounts),
   so no header was found and the columns were named "Column 2" / "Column 3".
3. Section labels ("ASSETS", "Current assets:") between the period header and
   the first figures, and two in a row inside the statement.

Not covered here: a "$" printed as its own cell, apart from its figure. Joining
it to the amount makes more statements parse as tables, but on the FinanceBench
sample it cost retrieval (the extra table representations crowd the reranker's
candidates), so that change is held back; see the realdocs report.
"""

from __future__ import annotations

import pytest

from src.data_processing.ingest_pipeline import extract_and_chunk
from src.data_processing.pdf_processor import PDFProcessor

YEAR_X1 = (420.0, 520.0)   # right edges of the two amount columns
GRID_X0 = 346.0            # left edge of the rules under the amounts
ROWS = [
    ("ASSETS", None),
    ("Current assets:", None),
    ("Cash and equivalents", ("$4,466", "$4,249")),
    ("Short-term investments", ("197", "996")),
    ("Inventories", ("5,622", "5,261")),
    ("Total current assets", ("16,525", "15,134")),
    ("LIABILITIES AND SHAREHOLDERS' EQUITY", None),
    ("Current liabilities:", None),
    ("Accounts payable", ("$2,612", "$2,279")),
    ("Accrued liabilities", ("5,010", "3,269")),
    ("Total current liabilities", ("7,866", "6,040")),
]


def _right(page, x1: float, y: float, text: str, size: float = 9) -> None:
    import fitz

    page.insert_text((x1 - fitz.get_text_length(text, fontsize=size), y), text, fontsize=size)


def write_balance_sheet_pdf(directory, *, rule_amounts: bool = True):
    """One balance-sheet page; with rule_amounts, a grid over the amount columns only."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), "ACME CORP.", fontsize=16)
    page.insert_text((72, 84), "CONSOLIDATED BALANCE SHEETS", fontsize=13)
    page.insert_text((72, 120), "(Dollars in millions)", fontsize=9)
    _right(page, YEAR_X1[0], 120, "2019")
    _right(page, YEAR_X1[1], 120, "2018")
    y = 140
    first_y = y
    for label, values in ROWS:
        page.insert_text((72, y), label, fontsize=9)
        if values:
            for column in (0, 1):
                _right(page, YEAR_X1[column], y, values[column])
        y += 16
    if rule_amounts:
        left, right = GRID_X0, YEAR_X1[1] + 4
        for k in range(len(ROWS) + 1):
            yy = first_y - 12 + 16 * k
            page.draw_line((left, yy), (right, yy))
        for xx in (left, (YEAR_X1[0] + YEAR_X1[1]) / 2 - 40, right):
            page.draw_line((xx, first_y - 12), (xx, first_y - 12 + 16 * len(ROWS)))
    path = directory / "acme_balance_sheet.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="module", params=[True, False], ids=["amount-rules", "no-rules"])
def balance_sheet(request, tmp_path_factory):
    path = write_balance_sheet_pdf(tmp_path_factory.mktemp("pdf"), rule_amounts=request.param)
    return extract_and_chunk(str(path), path.name, "deal-layouts", "financial")


def _markdown(doc) -> list[str]:
    return [c["text"] for c in doc.chunks if c.get("table_representation") == "markdown"]


def test_every_line_item_keeps_its_label_and_figures(balance_sheet):
    text = "\n".join(c["text"] for c in balance_sheet.chunks)
    for label, values in ROWS:
        assert label in text
        if values:
            assert values[0] in text and values[1] in text


def test_the_statement_is_one_table_with_its_year_columns(balance_sheet):
    tables = _markdown(balance_sheet)
    assert len(tables) == 1
    grid = tables[0]
    # The bare-year row is the header, not the first line of figures.
    header = next(line for line in grid.splitlines() if line.startswith("|"))
    assert header.endswith("| 2019 | 2018 |")
    assert "Column 2" not in grid
    assert "| Cash and equivalents | $4,466 | $4,249 |" in grid
    assert "| Inventories | 5,622 | 5,261 |" in grid
    assert "| Total current liabilities | 7,866 | 6,040 |" in grid
    # The section labels stay in the table, in order.
    assert grid.index("Current assets:") < grid.index("Cash and equivalents")
    assert grid.index("Current liabilities:") < grid.index("Accounts payable")


def test_figures_are_attributed_to_the_right_year(balance_sheet):
    rows = {r["line_item"]: r for c in balance_sheet.chunks
            if c.get("table_representation") == "row_by_row" for r in c["structured_rows"]}
    cash = rows["Cash and equivalents"]["values"]
    assert cash["2019"]["raw_value"] == 4466 and cash["2018"]["raw_value"] == 4249
    assert cash["2019"]["normalized_value"] == pytest.approx(4_466_000_000)


def test_no_table_is_a_strip_of_unlabelled_numbers(balance_sheet):
    for c in balance_sheet.chunks:
        if c.get("is_table"):
            assert any(label in c["text"] for label, _ in ROWS)


def test_a_ruled_table_without_row_labels_is_left_to_the_text_path():
    strip = {"headers": [], "rows": [["$", "39,117", ""], ["21,643", "", ""], ["17,474", "", ""]]}
    labelled = {"headers": ["Segment", "FY2022", "FY2023"],
                "rows": [["Platform subscriptions", "301.2", "362.9"], ["Other", "(2.6)", "(1.9)"]]}
    assert not PDFProcessor._has_row_labels(strip)
    assert PDFProcessor._has_row_labels(labelled)
    assert PDFProcessor._has_row_labels({"headers": ["A"], "rows": []})
