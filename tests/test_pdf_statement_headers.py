"""
Statements rebuilt from PDF rows keep their period headers (Decision 93).

On FinanceBench's dev filings, 66 of the 85 converter tables on evidence pages
were headed "Column 2", "Column 3" (Decision 90): their figures had no period,
so retrieval, numeric grounding, the figure veto and the tie-out could not say
which year a figure was. The layouts below are the ones that lost the header,
each written as the rows PyMuPDF gives (text, left and right edge, and the
words of a line that glued two cells); none is a filing's text.
"""

from __future__ import annotations

import pytest

from src.data_processing.pdf_processor import PDFProcessor, _Cell, _Row

# Right edges of three period columns, 60 points apart, and of their amounts.
YEARS = (420.0, 480.0, 540.0)


def cell(text: str, x1: float, width: float | None = None, words=None) -> _Cell:
    return _Cell(text, x1 - (width if width is not None else 5.0 * len(text)), x1, words)


def glued(*parts: tuple[str, float, float]) -> _Cell:
    """One PyMuPDF line holding several cells, with each word's own edges."""
    words = tuple(parts)
    return _Cell("  ".join(p[0] for p in parts), parts[0][1], parts[-1][2], words)


def row(*cells: _Cell, top: float = 0.0, size: float = 9.0) -> _Row:
    return _Row(list(cells), top, top + size, size)


def label(text: str, x1: float | None = None) -> _Cell:
    return _Cell(text, 50.0, x1 if x1 is not None else 50.0 + 4.5 * len(text))


def header(*years: str, caption: str = "(Dollars in millions)", at=YEARS) -> _Row:
    cells = [label(caption)] if caption else []
    return row(*cells, *(cell(y, x, 16.0) for y, x in zip(years, at)))


def values(name: str, *figures: str, at=YEARS) -> _Row:
    return row(label(name), *(cell(f, x) for f, x in zip(figures, at)))


def tables(rows: list[_Row]):
    return PDFProcessor()._borderless_tables(rows)


def by_label(table) -> dict[str, list[str]]:
    return {r[0]: r[1:] for r in table.rows}


class TestCurrencyCells:
    def test_a_dollar_sign_cell_joins_its_amount(self):
        rows = [
            header("2023", "2022", "2021"),
            row(label("Cash and equivalents"), cell("$", 392, 7), cell("2,306,072", 420),
                cell("$", 452, 7), cell("1,011,315", 480), cell("$", 512, 7), cell("989,000", 540)),
            values("Short-term investments", "3,513,702", "3,749,985", "3,100,000"),
            values("Total assets", "5,819,774", "4,761,300", "4,089,000"),
        ]
        [table] = tables(rows)
        assert table.header == ["(Dollars in millions)", "2023", "2022", "2021"]
        assert by_label(table)["Cash and equivalents"] == ["$ 2,306,072", "$ 1,011,315", "$ 989,000"]

    def test_the_next_columns_sign_glued_to_an_amount(self):
        # "Net income | $ | 5,070 $ | 6,046 $ | 5,727": each column's "$" set
        # flush left, so PyMuPDF glues it to the amount before it.
        rows = [
            header("2023", "2022", "2021"),
            row(label("Net income"), cell("$", 392, 7),
                glued(("5,070", 398, 420), ("$", 452, 457)), glued(("6,046", 458, 480), ("$", 512, 517)),
                cell("5,727", 540)),
            values("Depreciation", "703", "717", "744"),
            row(label("Cash and equivalents, end of year"), cell("$", 392, 7),
                glued(("7,441", 398, 420), ("$", 452, 457)), glued(("8,574", 458, 480), ("$", 512, 517)),
                cell("9,889", 540)),
        ]
        [table] = tables(rows)
        assert table.header[1:] == ["2023", "2022", "2021"]
        assert by_label(table)["Net income"] == ["$ 5,070", "$ 6,046", "$ 5,727"]
        assert by_label(table)["Cash and equivalents, end of year"] == ["$ 7,441", "$ 8,574", "$ 9,889"]

    def test_a_signed_nil_is_a_value(self):
        rows = [
            header("2023", "2022", "2021"),
            row(label("Loss from discontinued operations"), cell("$", 392, 7), cell("—", 420),
                cell("$", 452, 7), cell("(0.01)", 480), cell("$", 512, 7), cell("—", 540)),
            values("Net income", "1.40", "3.06", "2.10"),
        ]
        [table] = tables(rows)
        assert by_label(table)["Loss from discontinued operations"] == ["$ —", "$ (0.01)", "$ —"]

    def test_two_amounts_glued_into_one_line(self):
        rows = [
            header("2023", "2022", at=YEARS[1:]),
            values("Receivables, net", "199,422", "233,682", at=YEARS[1:]),
            row(label("Merchandise inventories, net"), glued(("1,603,451", 440, 480), ("1,499,218", 500, 540))),
            values("Prepaid expenses", "130,246", "110,814", at=YEARS[1:]),
        ]
        [table] = tables(rows)
        assert by_label(table)["Merchandise inventories, net"] == ["1,603,451", "1,499,218"]

    def test_a_sign_with_nothing_after_it_is_no_row(self):
        assert PDFProcessor()._value_cells(row(label("Total"), cell("1,234", 420), cell("$", 452, 7))) is None


class TestLabels:
    def test_a_wrapped_label_does_not_end_the_statement(self):
        wrapped = "Adjustments to reconcile net income to net cash provided by operating activities:"
        assert len(wrapped) >= 80
        rows = [
            header("2023", "2022", "2021"),
            values("Net income", "4,756", "4,822", "5,260"),
            row(label(wrapped, x1=351.0)),
            values("Depreciation, amortization and accretion", "856", "788", "757"),
            values("Stock-based compensation", "1,440", "1,069", "909"),
        ]
        [table] = tables(rows)
        assert table.header[1:] == ["2023", "2022", "2021"]
        assert [r[0] for r in table.rows] == [
            "Net income", wrapped, "Depreciation, amortization and accretion", "Stock-based compensation"]

    def test_a_line_of_prose_still_ends_it(self):
        prose = ("The accompanying notes are an integral part of these statements and should be read with "
                 "them in full.")
        rows = [
            header("2023", "2022", "2021"),
            values("Net income", "4,756", "4,822", "5,260"),
            values("Depreciation", "856", "788", "757"),
            row(label(prose, x1=560.0)),
            values("Interest paid", "103", "100", "88"),
            values("Taxes paid", "778", "843", "469"),
            values("Dividends paid", "1,100", "1,050", "990"),
        ]
        found = tables(rows)
        assert found[0].header[1:] == ["2023", "2022", "2021"]
        assert [r[0] for r in found[0].rows] == ["Net income", "Depreciation"]
        # The figures after the paragraph are not given the statement's years.
        assert found[1].header[1:] == ["Column 2", "Column 3", "Column 4"]

    def test_a_label_over_figures_set_lower_is_one_row(self):
        rows = [
            header("2017", "2016", at=YEARS[1:]),
            values("Accrued income taxes", "410", "307", at=YEARS[1:]),
            row(label("TOTAL CURRENT LIABILITIES")),
            row(cell("27,194", 480), cell("26,532", 540)),
            values("LONG-TERM DEBT", "31,182", "29,684", at=YEARS[1:]),
        ]
        [table] = tables(rows)
        assert by_label(table)["TOTAL CURRENT LIABILITIES"] == ["27,194", "26,532"]
        assert "LONG-TERM DEBT" in by_label(table)


class TestHeaders:
    def test_balance_dates_glued_into_one_line(self):
        # A balance sheet printed under two dates PyMuPDF reads as one line.
        dates = _Cell("December 28, 2019  December 29, 2018", 420, 540, (
            ("December", 420, 455), ("28,", 457, 466), ("2019", 468, 480),
            ("December", 484, 515), ("29,", 517, 526), ("2018", 528, 540)))
        rows = [
            row(dates),
            row(label("ASSETS")),
            values("Cash and cash equivalents", "2,279", "1,130", at=YEARS[1:]),
            values("Trade receivables", "1,973", "2,129", at=YEARS[1:]),
        ]
        [table] = tables(rows)
        assert table.header[1:] == ["December 28, 2019", "December 29, 2018"]

    def test_numeric_dates_under_a_span(self):
        rows = [
            row(cell("12 Weeks Ended", 515, 60)),
            row(cell("3/25/2023", 480, 37), cell("3/19/2022", 540, 37)),
            row(label("Net Revenue"), cell("$", 452, 7), cell("17,846", 480), cell("$", 512, 7), cell("16,200", 540)),
            values("Cost of sales", "7,988", "7,433", at=YEARS[1:]),
        ]
        [table] = tables(rows)
        assert table.header[1:] == ["12 Weeks Ended 3/25/2023", "12 Weeks Ended 3/19/2022"]

    def test_spans_dealt_out_over_their_columns(self):
        four = (360.0, 420.0, 480.0, 540.0)
        rows = [
            row(cell("Three months ended June 30,", 425, 110), cell("Six months ended June 30,", 545, 100)),
            header("2022", "2021", "2022", "2021", caption="(in millions, except per share data)", at=four),
            row(label("Revenue")),
            values("Investment banking fees", "1,586", "3,470", "3,594", "6,440", at=four),
            values("Principal transactions", "4,990", "4,076", "10,095", "10,576", at=four),
        ]
        [table] = tables(rows)
        assert table.header[1:] == [
            "Three months ended June 30, 2022", "Three months ended June 30, 2021",
            "Six months ended June 30, 2022", "Six months ended June 30, 2021",
        ]

    def test_a_date_over_each_column_and_a_span_over_all(self):
        rows = [
            row(cell("Years Ended", 500, 44)),
            row(cell("December 2,", 425, 42), cell("December 3,", 485, 42), cell("November 27,", 545, 48)),
            header("2022", "2021", "2020", caption=""),
            values("Net income", "4,756", "4,822", "5,260"),
            values("Depreciation", "856", "788", "757"),
        ]
        [table] = tables(rows)
        assert table.header[1:] == [
            "Years Ended December 2, 2022", "Years Ended December 3, 2021", "Years Ended November 27, 2020"]

    def test_one_line_over_two_columns_is_cut_at_its_gap(self):
        line = glued(("January", 450, 478), ("28,", 480, 490), ("January", 505, 533), ("29,", 535, 545))
        rows = [
            row(line),
            header("2023", "2022", caption="", at=YEARS[1:]),
            values("Receivables, net", "199,422", "233,682", at=YEARS[1:]),
            values("Prepaid expenses", "130,246", "110,814", at=YEARS[1:]),
        ]
        [table] = tables(rows)
        assert table.header[1:] == ["January 28, 2023", "January 29, 2022"]

    def test_a_title_over_the_columns_is_not_a_period(self):
        rows = [
            row(cell("CONSOLIDATED STATEMENTS OF INCOME", 480, 166)),
            header("2017", "2016", "2015", caption="Year Ended December 31,"),
            values("Net operating revenues", "35,410", "41,863", "44,294"),
            values("Cost of goods sold", "13,256", "16,465", "17,482"),
        ]
        [table] = tables(rows)
        # The span printed in the label column heads every column; the title
        # over the page is no part of a column's name.
        assert table.header == ["", "Year Ended December 31, 2017", "Year Ended December 31, 2016",
                                "Year Ended December 31, 2015"]
        assert table.title == "CONSOLIDATED STATEMENTS OF INCOME"

    @pytest.mark.parametrize("label_cell, names", [
        ("Fiscal Years Ended", ["Fiscal Years Ended January 28, 2017", "Fiscal Years Ended January 30, 2016"]),
        ("(In millions) Three months ended March 31,", ["Three months ended March 31, 2021",
                                                        "Three months ended March 31, 2020"]),
        ("At December 31,", ["At December 31, 2021", "At December 31, 2020"]),
    ])
    def test_a_span_in_the_label_column_heads_every_column(self, label_cell, names):
        dates = label_cell == "Fiscal Years Ended"
        periods = ("January 28, 2017", "January 30, 2016") if dates else ("2021", "2020")
        rows = [
            header(*periods, caption=label_cell, at=YEARS[1:]),
            values("Receivables", "(185)", "123", at=YEARS[1:]),
            values("Merchandise inventories", "193", "86", at=YEARS[1:]),
        ]
        [table] = tables(rows)
        assert table.header[1:] == names
        assert table.header[0] == ("(In millions)" if label_cell.startswith("(") else "")

    @pytest.mark.parametrize("label_cell", ["(Dollars in millions)", "December 31,"])
    def test_a_label_cell_that_adds_nothing_stays_in_the_label_column(self, label_cell):
        rows = [
            header("December 31, 2021", "December 31, 2020", caption=label_cell, at=YEARS[1:]),
            values("Total assets", "14,535", "12,697", at=YEARS[1:]),
            values("Total liabilities", "6,075", "5,272", at=YEARS[1:]),
        ]
        [table] = tables(rows)
        assert table.header == [label_cell, "December 31, 2021", "December 31, 2020"]

    def test_a_unit_line_under_the_years_is_the_caption(self):
        rows = [
            header("2022", "2021", caption="", at=YEARS[1:]),
            row(label("(In millions, except par value amounts)", x1=400.0)),
            row(label("ASSETS")),
            values("Cash and cash equivalents", "4,835", "2,535", at=YEARS[1:]),
            values("Short-term investments", "1,020", "1,073", at=YEARS[1:]),
        ]
        [table] = tables(rows)
        assert table.caption == "(In millions, except par value amounts)"
        assert [r[0] for r in table.rows] == ["ASSETS", "Cash and cash equivalents", "Short-term investments"]

    def test_two_spans_in_one_unreadable_line_name_nothing(self):
        four = (360.0, 420.0, 480.0, 540.0)
        rows = [
            row(cell("Three Months Ended Six Months Ended", 545, 210)),
            header("2022", "2021", "2022", "2021", at=four),
            values("Revenue", "1", "2", "3", "4", at=four),
            values("Expenses", "5", "6", "7", "8", at=four),
            values("Taxes", "9", "10", "11", "12", at=four),
        ]
        # The two spans cannot be dealt out, so a quarter and a half-year would
        # both be "2022": the header is not read at all.
        [table] = tables(rows)
        assert table.header[1:] == ["Column 2", "Column 3", "Column 4", "Column 5"]
        assert table.start == 2

    def test_a_label_column_line_is_passed_over(self):
        eight = (200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0)
        rows = [
            row(*(cell(q, x, 60) for q, x in zip(("First Quarter", "Second Quarter", "Third Quarter",
                                                   "Fourth Quarter"), eight[1::2]))),
            row(*(cell("Fiscal Year", x, 50) for x in eight[1::2])),
            row(label("In Millions, Except Per", x1=137.0)),
            header("2020", "2019", "2020", "2019", "2020", "2019", "2020", "2019", caption="Share Amounts", at=eight),
            values("Net sales", *("4,002.5", "4,094.0", "4,420.8", "4,411.2", "4,180.3", "4,198.3", "5,023.0",
                                  "4,161.7"), at=eight),
            values("Gross margin", *("1,389.5", "1,342.8", "1,569.1", "1,509.7", "1,403.2", "1,443.0", "1,768.1",
                                     "1,461.3"), at=eight),
        ]
        [table] = tables(rows)
        assert table.header[1:3] == ["First Quarter Fiscal Year 2020", "First Quarter Fiscal Year 2019"]
        assert table.header[-1] == "Fourth Quarter Fiscal Year 2019"
        assert table.caption == "In Millions, Except Per"

    def test_month_and_day_lines_over_the_years(self):
        six = (310.0, 360.0, 400.0, 450.0, 495.0, 540.0)
        rows = [
            row(cell("13 Weeks Ended", 380, 65), cell("52 Weeks Ended", 517, 65)),
            row(*(cell("January", x, 30) for x in six)),
            row(*(cell(d, x - 3, 12) for d, x in zip(("28,", "29,", "30,", "28,", "29,", "30,"), six))),
            header("2023", "2022", "2021", "2023", "2022", "2021", at=six),
            values("Net sales", "3,226.8", "2,729.4", "2,198.7", "10,208.6", "8,630.9", "6,152.0", at=six),
            values("Gross profit", "1,212.5", "1,027.3", "855.0", "4,044.5", "3,368.6", "2,104.6", at=six),
        ]
        [table] = tables(rows)
        assert table.header[1] == "13 Weeks Ended January 28, 2023"
        assert table.header[4] == "52 Weeks Ended January 28, 2023"

    def test_qualifiers_over_blocks_of_columns(self):
        six = (300.0, 345.0, 390.0, 435.0, 500.0, 545.0)
        rows = [
            row(cell("Pension", 395, 60), cell("Retiree Medical", 545, 70)),
            header("2021", "2020", "2021", "2020", "2021", "2020", at=six),
            values("Obligation at beginning of year", "16,753", "15,230", "4,430", "3,753", "1,006", "988", at=six),
            values("Service cost", "374", "345", "77", "73", "34", "35", at=six),
            values("Interest cost", "381", "435", "104", "96", "21", "27", at=six),
        ]
        # Two pieces over six columns make blocks of three, which repeat a
        # year under "Pension": the table stays unnamed.
        [table] = tables(rows)
        assert table.header[1] == "Column 2"
        rows[0] = row(cell("Pension", 345, 60), cell("Retiree Medical", 435, 70), cell("Other", 545, 30))
        [table] = tables(rows)
        assert table.header[1:] == ["Pension 2021", "Pension 2020", "Retiree Medical 2021",
                                    "Retiree Medical 2020", "Other 2021", "Other 2020"]

    def test_a_caption_set_over_the_columns(self):
        rows = [
            row(cell("Health Care and Life", 540, 120)),
            header("2021", "2020", "2019", caption="At December 31,"),
            values("Discount rate", "2.8", "2.4", "3.2"),
            values("Trend rate", "6.2", "6.2", "6.3"),
        ]
        [table] = tables(rows)
        assert table.header[1:] == ["At December 31, Health Care and Life 2021",
                                    "At December 31, Health Care and Life 2020",
                                    "At December 31, Health Care and Life 2019"]

    def test_a_caption_over_one_block_or_a_paragraph_line_is_not(self):
        six = (300.0, 345.0, 390.0, 435.0, 500.0, 545.0)
        base = [
            row(cell("Multifamily", 345, 40), cell("Other Commercial", 435, 60), cell("by real estate", 545, 50)),
            header("2022", "2021", "2022", "2021", "2022", "2021", at=six),
            values("Retained loans secured by real estate", "79,139", "73,801", "47,593", "45,034", "126,732",
                   "118,835", at=six),
            values("Criticized", "1,916", "1,671", "1,992", "2,300", "3,908", "3,971", at=six),
        ]
        for line in ("Total retained loans secured", "modifications that are not considered concessions."):
            # Over the last block only, or a line of the paragraph beside the table.
            above = row(cell(line, 545, 90 if line.startswith("Total") else 250))
            [table] = tables([above, *base])
            assert table.header[1:3] == ["Multifamily 2022", "Multifamily 2021"]


class TestInheritedHeaders:
    def test_a_row_the_columns_cannot_hold_stays_as_text(self):
        unplaceable = row(label("Basic net income per share"), cell("0.29", 420), cell("3", 450, 4),
                          cell("1.51", 480), cell("1.69", 540))
        rows = [
            header("2023", "2022", "2021"),
            values("Net sales", "35,410", "41,863", "44,294"),
            values("Cost of sales", "13,256", "16,465", "17,482"),
            # A footnote marker read as a fourth figure: the row cannot be placed.
            unplaceable,
            values("Diluted net income per share", "0.29", "1.49", "1.67"),
            values("Average shares outstanding", "4,272", "4,317", "4,352"),
        ]
        [table] = tables(rows)
        assert table.header[1:] == ["2023", "2022", "2021"]
        assert by_label(table)[unplaceable.text] == ["", "", ""]
        assert by_label(table)["Average shares outstanding"] == ["4,272", "4,317", "4,352"]

    def test_a_statement_broken_by_section_labels_keeps_its_header(self):
        rows = [
            header("2023", "2022", "2021"),
            values("Net sales", "35,410", "41,863", "44,294"),
            values("Cost of sales", "13,256", "16,465", "17,482"),
            row(label("Supplemental information:")),
            row(label("Cash paid during the year for:")),
            row(label("Interest and income taxes,")),
            row(label("net of refunds:")),
            values("Interest", "347", "290", "293"),
            values("Income taxes", "1,517", "1,231", "1,177"),
        ]
        first, second = tables(rows)
        assert first.header[1:] == second.header[1:] == ["2023", "2022", "2021"]
        assert [r[0] for r in second.rows] == ["Interest", "Income taxes"]

    def test_a_header_that_names_no_period_is_not_overruled(self):
        rows = [
            header("2023", "2022", at=YEARS[1:]),
            values("Net sales", "35,410", "41,863", at=YEARS[1:]),
            values("Cost of sales", "13,256", "16,465", at=YEARS[1:]),
            row(cell("Domestic", 480, 40), cell("International", 540, 60)),
            values("Stores", "1,100", "300", at=YEARS[1:]),
            values("Employees", "71,000", "9,000", at=YEARS[1:]),
            values("Square feet", "40", "6", at=YEARS[1:]),
        ]
        first, second = tables(rows)
        assert first.header[1:] == ["2023", "2022"]
        assert second.header[1:] == ["Column 2", "Column 3"]

    def test_a_column_count_change_stops_the_search(self):
        rows = [
            header("2023", "2022", "2021"),
            values("Net sales", "35,410", "41,863", "44,294"),
            values("Cost of sales", "13,256", "16,465", "17,482"),
            values("Stores", "1,100", "300", at=YEARS[1:]),
            row(label("Square feet (millions)")),
            row(label("Average")),
            row(label("Year end")),
            row(label("Other")),
            values("Owned", "40", "6", "7"),
            values("Leased", "41", "7", "8"),
            values("Total", "81", "13", "15"),
        ]
        found = tables(rows)
        assert found[-1].header[1:] == ["Column 2", "Column 3", "Column 4"]


def _page(rows: list[_Row]) -> list[_Row]:
    for n, r in enumerate(rows):
        r.top, r.bottom = 100.0 + 14 * n, 109.0 + 14 * n
    return rows


class TestContinuedStatements:
    def _pages(self, tail_of_first_page: list[_Row]):
        first = _page([
            row(label("CONSOLIDATED BALANCE SHEETS")),
            header("2022", "2021", at=YEARS[1:]),
            values("Cash and cash equivalents", "2,605", "2,921", at=YEARS[1:]),
            values("Inventories", "2,388", "3,055", at=YEARS[1:]),
            *tail_of_first_page,
        ])
        second = _page([
            row(label("CONSOLIDATED BALANCE SHEETS (continued)")),
            row(label("Liabilities and Equity")),
            values("Accounts payable", "23,977", "24,833", at=YEARS[1:]),
            values("Long-term debt", "140,676", "143,425", at=YEARS[1:]),
            values("Total liabilities", "237,046", "236,236", at=YEARS[1:]),
        ])
        return PDFProcessor()._process_layout([first, second], [])

    def test_a_statement_continued_on_the_next_page_keeps_its_header_and_its_page(self):
        sections = self._pages([row(label("See notes to the financial statements."))])
        first, second = [s for s in sections if s.is_table]
        assert (first.page_number, first.page_range) == (1, [1, 1])
        assert (second.page_number, second.page_range) == (2, [2, 2])
        assert second.table_rows[0][1:] == ["2022", "2021"]
        assert second.table_rows[-1] == ["Total liabilities", "237,046", "236,236"]
        assert first.caption.startswith("CONSOLIDATED BALANCE SHEETS")
        assert second.caption.startswith("CONSOLIDATED BALANCE SHEETS (continued)")

    def test_a_continued_part_under_its_header_again_takes_the_statement_title(self):
        first = _page([
            row(label("CONSOLIDATED BALANCE SHEETS")),
            header("2022", "2021", at=YEARS[1:]),
            values("Cash and cash equivalents", "2,605", "2,921", at=YEARS[1:]),
            values("Inventories", "2,388", "3,055", at=YEARS[1:]),
        ])
        second = _page([
            header("2022", "2021", at=YEARS[1:]),
            values("Accounts payable", "23,977", "24,833", at=YEARS[1:]),
            values("Long-term debt", "140,676", "143,425", at=YEARS[1:]),
        ])
        tables_found = [s for s in PDFProcessor()._process_layout([first, second], []) if s.is_table]
        assert [s.page_number for s in tables_found] == [1, 2]
        assert tables_found[1].caption.startswith("CONSOLIDATED BALANCE SHEETS")

    def test_a_new_statement_under_the_same_years_is_not_joined_or_retitled(self):
        first = _page([
            row(label("CONSOLIDATED STATEMENTS OF EARNINGS")),
            header("2022", "2021", at=YEARS[1:]),
            values("Revenue", "51,761", "47,262", at=YEARS[1:]),
            values("Net earnings", "2,454", "1,798", at=YEARS[1:]),
        ])
        second = _page([
            row(label("CONSOLIDATED STATEMENTS OF CASH FLOWS")),
            header("2022", "2021", at=YEARS[1:]),
            values("Receivables", "(185)", "123", at=YEARS[1:]),
            values("Capital expenditures", "(582)", "(649)", at=YEARS[1:]),
        ])
        earnings, cash = [s for s in PDFProcessor()._process_layout([first, second], []) if s.is_table]
        assert (earnings.page_range, cash.page_range) == ([1, 1], [2, 2])
        assert "CASH FLOWS" in cash.caption and "EARNINGS" not in cash.caption

    def test_not_after_a_paragraph(self):
        prose = row(label("The weighted-average remaining lease term and discount rate were as follows, "
                          "for each of the periods presented.", x1=560.0))
        sections = self._pages([prose])
        found = [s for s in sections if s.is_table]
        assert [s.page_range for s in found] == [[1, 1], [2, 2]]
        assert found[1].table_rows[0][1:] == ["Column 2", "Column 3"]

    def test_not_past_a_ruled_table(self):
        processor = PDFProcessor()
        first = _page([header("2022", "2021", at=YEARS[1:]),
                       values("Cash", "2,605", "2,921", at=YEARS[1:]),
                       values("Inventories", "2,388", "3,055", at=YEARS[1:])])
        carried = processor._carried(first, processor._borderless_tables(first), [])
        assert carried is not None
        second = _page([values("Accounts payable", "23,977", "24,833", at=YEARS[1:]),
                        values("Long-term debt", "140,676", "143,425", at=YEARS[1:]),
                        values("Total liabilities", "237,046", "236,236", at=YEARS[1:])])
        assert processor._borderless_tables(second, carried, (50.0,))[0].header[1:] == ["Column 2", "Column 3"]
        assert processor._borderless_tables(second, carried, (900.0,))[0].header[1:] == ["2022", "2021"]


class TestTitles:
    def test_the_title_over_a_statement_is_its_caption(self):
        sections = PDFProcessor()._process_layout([_page([
            row(label("ADOBE SYSTEMS INCORPORATED")),
            row(label("CONSOLIDATED STATEMENTS OF CASH FLOWS")),
            row(label("(In thousands)")),
            header("2017", "2016", "2015", caption=""),
            values("Net income", "1,693,954", "1,168,782", "629,551"),
            values("Depreciation, amortization and accretion", "325,997", "331,535", "339,473"),
        ])], [])
        [table] = [s for s in sections if s.is_table]
        assert table.caption == "ADOBE SYSTEMS INCORPORATED CONSOLIDATED STATEMENTS OF CASH FLOWS (In thousands)"
        assert table.text.startswith(table.caption)

    def test_header_lines_the_columns_did_not_take_are_passed_over(self):
        rows = _page([
            row(label("CONSOLIDATED STATEMENTS OF CASH FLOWS")),
            row(label("(In thousands)")),
            row(cell("Years Ended", 480, 45)),
            row(cell("November 27,", 420, 48), cell("November 28,", 480, 48), cell("November 29,", 540, 48)),
            header("2015", "2014", "2013", caption=""),
        ])
        assert PDFProcessor()._title_lines(rows, 4) == ["CONSOLIDATED STATEMENTS OF CASH FLOWS", "(In thousands)"]

    @pytest.mark.parametrize("line", [
        "The following table provides a summary of net revenue by segment for 2022 and 2021:",
        "See accompanying Notes to Consolidated Financial Statements.",
        "Advanced Micro Devices, Inc.",
        "(1) Includes the effect of the reorganisation.",
    ])
    def test_a_sentence_or_a_footnote_ends_the_title(self, line):
        rows = _page([row(label("CONSOLIDATED BALANCE SHEETS")), row(label(line, x1=400.0)),
                      row(label("(In millions)"))])
        assert PDFProcessor()._title_lines(rows, 3) == ["(In millions)"]

    def test_a_line_beside_the_table_or_a_sentence_naming_a_month_ends_the_title(self):
        rows = _page([
            row(label("Allowance for credit losses"), cell("Selected impacts of investment securities", 520, 200)),
            row(label("allowance for credit losses on impaired AFS securities as of", x1=290.0)),
            row(label("March 31, 2021 and 2020.", x1=161.0), cell("Three months ended March 31,", 558, 100)),
        ])
        assert PDFProcessor()._title_lines(rows, 3, 0, (320.0, 560.0)) == []
        prose = _page([row(label("Regulatory Assets")),
                       row(label("Presented in the table below is the composition as of December 31:", x1=286.0))])
        assert PDFProcessor()._title_lines(prose, 2) == []

    def test_a_run_whose_header_was_not_read_takes_no_title(self):
        rows = [
            row(label("Loans")),
            row(label("Nonaccrual loans")),
            values("Loans reported as trading assets", "2,517", "368", "(2,149)"),
            values("Loans", "967", "829", "(138)"),
            values("Total", "3,484", "1,197", "(2,287)"),
        ]
        [table] = tables(rows)
        assert table.header[1:] == ["Column 2", "Column 3", "Column 4"] and table.title == ""

    def test_lines_from_the_previous_page_are_not_a_title(self):
        previous = [row(label("CONSOLIDATED STATEMENTS OF EARNINGS"), top=700.0)]
        current = [row(label("(In millions)"), top=80.0)]
        assert PDFProcessor()._title_lines(previous + current, 2) == ["(In millions)"]

    def test_a_ruled_grid_opening_its_page_takes_no_title_from_the_page_before(self):
        # The previous page's last line, then a grid at the top of the next page.
        recent = [row(label("See accompanying notes to consolidated financial statements"), top=740.0)]
        grid = {"headers": [], "rows": [["Net revenue", "3,930", "3,777"]], "page_range": [57, 57],
                "regions": [(57, (50.0, 72.0, 560.0, 400.0))]}
        assert PDFProcessor()._ruled_section(grid, "", recent).caption == ""
        own = [row(label("CONSOLIDATED BALANCE SHEETS"), top=40.0)]
        assert PDFProcessor()._ruled_section(grid, "", recent + own).caption == "CONSOLIDATED BALANCE SHEETS"

    def test_a_title_that_cannot_be_read_is_no_title(self, monkeypatch):
        # Decision 88: reading a title is optional; the default is the caption without one.
        monkeypatch.setenv("STRICT_OPTIONAL_STEPS", "0")
        processor = PDFProcessor()

        def unreadable(*args, **kwargs):
            raise ValueError("unreadable")

        monkeypatch.setattr(processor, "_title_lines", unreadable)
        recent = _page([row(label("Consolidated Statements of Operations")), row(label("(In millions)"))])
        grid = {"headers": [], "rows": [["Operating revenues", "3,930"]], "page_range": [86, 86]}
        assert processor._ruled_section(grid, "", recent).caption == "(In millions)"

    def test_a_ruled_grid_on_one_page_takes_the_title_over_its_header(self):
        recent = _page([
            row(label("American Water Works Company, Inc. and Subsidiary Companies")),
            row(label("Consolidated Statements of Operations")),
            row(label("(In millions, except per share data)")),
            header("2021", "2020", "2019", caption=""),
        ])
        grid = {"headers": [], "rows": [["Operating revenues", "3,930", "3,777", "3,610"]], "page_range": [86, 86]}
        section = PDFProcessor()._ruled_section(grid, "", recent)
        assert section.caption == ("American Water Works Company, Inc. and Subsidiary Companies "
                                   "Consolidated Statements of Operations (In millions, except per share data)")
        stitched = PDFProcessor()._ruled_section({**grid, "page_range": [62, 63]}, "", recent)
        assert stitched.caption == "(In millions, except per share data)"


class TestPeriodCells:
    @pytest.mark.parametrize("text", ["2023", "FY2023", "Q4 2023", "December 28, 2019", "Dec. 31, 2019",
                                      "31 December 2019", "3/25/2023"])
    def test_a_period_cell(self, text):
        assert PDFProcessor._period_row(row(cell(text, 420), cell(text, 480)))

    @pytest.mark.parametrize("text", ["December 31,", "2022 vs. 2021", "Change", "May 26,", "2023 Restated"])
    def test_not_a_period_cell(self, text):
        assert not PDFProcessor._period_row(row(cell(text, 420), cell(text, 480)))


def write_glued_statement_pdf(directory):
    """
    A cash-flow page whose "$" signs sit flush left in each column, glued by
    the PDF's own text runs to the amount of the column before ("5,070  $").
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page()

    def right(x1, y, text, size=9):
        page.insert_text((x1 - fitz.get_text_length(text, fontsize=size), y), text, fontsize=size)

    page.insert_text((72, 60), "ACME CORP.", fontsize=14)
    page.insert_text((72, 80), "CONSOLIDATED STATEMENTS OF CASH FLOWS", fontsize=12)
    page.insert_text((430, 104), "YEAR ENDED MAY 31,", fontsize=9)
    page.insert_text((72, 118), "(Dollars in millions)", fontsize=9)
    for year, x1 in zip(("2023", "2022", "2021"), YEARS):
        right(x1, 118, year)
    y = 136
    lines = [
        ("Net income", ("5,070", "6,046", "5,727"), True),
        ("Depreciation", ("703", "717", "744"), False),
        ("Stock-based compensation", ("755", "638", "611"), False),
        ("Cash and equivalents, end of year", ("7,441", "8,574", "9,889"), True),
    ]
    for name, figures, signed in lines:
        page.insert_text((72, y), name, fontsize=9)
        if signed:
            page.insert_text((392, y), "$", fontsize=9)
            # The amount and the next column's sign as one text run.
            for figure, x1 in zip(figures[:2], YEARS[:2]):
                start = x1 - fitz.get_text_length(figure, fontsize=9)
                page.insert_text((start, y), figure + "      $", fontsize=9)
            right(YEARS[2], y, figures[2])
        else:
            for figure, x1 in zip(figures, YEARS):
                right(x1, y, figure)
        y += 16
    path = directory / "acme_cash_flows.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_a_generated_statement_keeps_its_years(tmp_path):
    from src.data_processing.ingest_pipeline import extract_and_chunk

    path = write_glued_statement_pdf(tmp_path)
    doc = extract_and_chunk(str(path), path.name, "deal-headers", "financial")
    rows = {r["line_item"]: r["values"] for c in doc.chunks
            if c.get("table_representation") == "row_by_row" for r in c["structured_rows"]}
    assert set(rows["Net income"]) == {"YEAR ENDED MAY 31, 2023", "YEAR ENDED MAY 31, 2022",
                                       "YEAR ENDED MAY 31, 2021"}
    assert rows["Net income"]["YEAR ENDED MAY 31, 2023"]["raw_value"] == 5070
    assert rows["Cash and equivalents, end of year"]["YEAR ENDED MAY 31, 2021"]["raw_value"] == 9889
    assert rows["Depreciation"]["YEAR ENDED MAY 31, 2022"]["raw_value"] == 717
    grids = [c["text"] for c in doc.chunks if c.get("table_representation") == "markdown"]
    assert len(grids) == 1 and "Column 2" not in grids[0]
