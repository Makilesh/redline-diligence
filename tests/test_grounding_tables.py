"""
Numeric grounding on financial tables as the table converter writes them
(Decision 94): the period a row-by-row cell names, row labels longer than a
"Label:" (PDF dot leaders), grid headers that say more than their years, a
figure whose period nothing states, and the converter's unit note.

Every table here is in the form ingestion indexes a FinanceBench statement:
the converter's markdown grid, row-by-row and narrative representations, each
under the statement's title and the converter's unit note. No model is loaded.
"""

import pytest

from src.verification.claim_checker import _FigureView
from src.verification.numeric_grounding import (
    SCALE_HINT_RE,
    UNIT_NOTE_RE,
    ContextIndex,
    _body_units,
    _cell_header,
    _column_header,
    _is_converter_row,
    _is_row_by_row,
    extract_numbers,
    figure_periods,
    ground_texts,
    states_scale,
)
from src.verification.table_context import headed_window


def _periods(text: str) -> list[tuple[str, str]]:
    spans = [(m.start, m.end) for m in extract_numbers(text, claims_only=False)]
    return [(text[s:e], p) for (s, e), p in zip(spans, figure_periods(text, spans))]


def _check(text: str, claim: str, table: bool = True):
    chunk = {"text": text, "source_file": "f.pdf", "is_table": 1 if table else 0}
    [[check]] = ground_texts([claim], ContextIndex([chunk]))
    return check


def _confirmed(text: str, claim: str) -> bool:
    chunk = {"text": text, "source_file": "f.pdf", "is_table": 1}
    [checks] = ground_texts([claim], ContextIndex([chunk]))
    return _FigureView(checks).confirmed


# ── A period that labels its figure ───────────────────────────────────────────


class TestLabelledPeriods:
    def test_a_row_by_row_figure_belongs_to_the_period_before_its_equals_sign(self):
        # Before: 7,207 sat as close to " 2018" after it as to "2019=" before
        # it, and the tie went to the period after.
        assert _periods("Operating income: 2019=7,207 2018=7,820") == [("7,207", "2019"), ("7,820", "2018")]
        assert _periods("Securities: Fiscal Year 2019=- Fiscal Year 2018=1.2 Fiscal Year 2017=0.8") == [
            ("1.2", "2018"), ("0.8", "2017"),
        ]
        assert _periods("Investment gains (losses), net: 2016=(1,570) 2015=961 2014=1,156") == [
            ("1,570", "2016"), ("961", "2015"), ("1,156", "2014"),
        ]

    def test_a_label_is_the_figure_after_it_and_no_other(self):
        # The PDF parser put the first column's value in the row label: the
        # 2014 that labels the next figure does not date it.
        assert _periods("Revenue ....... $ 4,795,511: Fiscal Years 2014=$ 4,147,065 Fiscal Years 2013=$ 4,055,240") == [
            ("4,795,511", ""), ("4,147,065", "2014"), ("4,055,240", "2013"),
        ]
        # Nor does a period labelling an empty cell date the footnote before it.
        assert _periods("Notional value settled (2): 2021=— 2020=1,600") == [("2", ""), ("1,600", "2020")]
        assert _periods("Notional value settled (2): 2021: —;") == [("2", "")]
        # A period followed by a colon and words still dates the figure before it.
        assert _periods("Revenue was $5.0 million in FY2023: a record.") == [("$5.0 million", "2023")]

    def test_prose_and_claims_read_as_before(self):
        assert _periods("Revenue was $452.8 million in FY2023, up from $387.1 million in FY2022.") == [
            ("$452.8 million", "2023"), ("$387.1 million", "2022"),
        ]
        assert _periods("Net sales: 2019: 32,136; 2018: 32,765") == [("32,136", "2019"), ("32,765", "2018")]
        # A label scopes forward: the growth after "FY2023:" is FY2023's.
        assert _periods("Revenue FY2023: $452.8M (+17.0%)") == [("$452.8M", "2023"), ("17.0%", "2023")]

    def test_a_row_that_labels_its_cells_takes_no_header(self):
        # The footnote and the one value make two figures, as many as the
        # grid's columns: the header would date the footnote "2021".
        grid = {"text": "| Line Item | 2021 | 2020 |\n|---|---|---|\n| Notional value entered into | 4,650 | 5,500 |\n",
                "source_file": "f.pdf", "is_table": 1}
        rows = {"text": "Notional value settled (2): 2021=— 2020=1,600\n", "source_file": "f.pdf", "is_table": 1}
        index = ContextIndex([grid, rows])
        [[check]] = ground_texts(["In fiscal 2021, notional value settled was $2."], index)
        assert not check.is_confirmed
        [[check]] = ground_texts(["In fiscal 2020, notional value settled was $1,600."], index)
        assert check.is_confirmed

    def test_a_borrowed_header_that_contradicts_the_labels_is_not_used(self):
        # The file's only two-column header is another table's, in the other
        # order: the row's own labels win.
        other = {"text": "| Line Item | 2018 | 2019 |\n|---|---|---|\n| Cash | 1.0 | 2.0 |\n",
                 "source_file": "f.pdf", "is_table": 1}
        rows = {"text": "Operating income: 2019=7,207 2018=7,820\n", "source_file": "f.pdf", "is_table": 1}
        [[check]] = ground_texts(["Operating income was $7,207 in fiscal 2018."], ContextIndex([other, rows]))
        assert check.binding == "conflict"


# ── Row labels longer than a "Label:" ────────────────────────────────────────

BALANCE_SHEET = (
    "ADOBE SYSTEMS INCORPORATED\n"
    "Values are as stated in the source; the table gives no currency or scale.\n"
    "Cash and cash equivalents..................................................... $ 876,560 $ 1,117,400:\n"
    "Short-term investments ........................................................: 2015=3,111,524 2014=2,622,091\n"
    "Trade receivables, net of allowances for doubtful accounts of $7,293 and $7,867, "
    "respectively...............: 2015=672,006 2014=591,800\n"
    "Deferred income taxes...........................................................: 2015=— 2014=95,279\n"
    "Prepaid expenses and other current assets ......................................: 2015=161,802 2014=175,758\n"
)
NARRATIVE_LEADERS = (
    "ADOBE SYSTEMS INCORPORATED\n"
    "Financial table with 2 rows and 2 columns (2015, 2014). Values are as stated in the source; "
    "the table gives no currency or scale.\n"
    "Investment in lease receivable......................................................: 2015: 80,439; 2014: 80,439\n"
    "Other assets.........................................................................: 2015: 159,832; 2014: 126,315\n"
)


class TestLongRowLabels:
    def test_every_leadered_line_is_a_row(self):
        rows = [BALANCE_SHEET[s:e] for s, e, tabular, _ in _body_units(BALANCE_SHEET) if tabular]
        assert [r.split(".")[0].strip() for r in rows] == [
            "Cash and cash equivalents", "Short-term investments", "Trade receivables, net of allowances "
            "for doubtful accounts of $7,293 and $7,867, respectively", "Deferred income taxes",
            "Prepaid expenses and other current assets",
        ]

    def test_another_years_figure_is_a_conflict(self):
        # Decision 90's set: bound before, the table read as one paragraph.
        check = _check(BALANCE_SHEET, "In fiscal 2014, short-term investments was $3,111,524.")
        assert check.binding == "conflict" and "2015" in check.binding_note

    def test_the_neighbouring_rows_figure_is_not_confirmed(self):
        for claim in ("In fiscal 2014, short-term investments was $95,279.",
                      "In fiscal 2014, deferred income taxes was $175,758."):
            assert not _check(BALANCE_SHEET, claim).is_confirmed, claim

    def test_the_rows_own_figure_is_confirmed(self):
        assert _confirmed(BALANCE_SHEET, "In fiscal 2014, short-term investments was $2,622,091.")
        assert _confirmed(BALANCE_SHEET, "In fiscal 2015, prepaid expenses and other current assets was $161,802.")

    def test_a_narrative_line_with_leaders(self):
        # The second column of one row read with the next row's label.
        assert not _check(NARRATIVE_LEADERS, "In fiscal 2014, other assets was $80,439.").is_confirmed
        assert _confirmed(NARRATIVE_LEADERS, "In fiscal 2014, other assets was $126,315.")

    def test_a_long_label_without_leaders(self):
        text = ("GENERAL MILLS\nValues are as stated in the source, in millions.\n"
                "Earnings before income taxes and after-tax earnings from joint ventures: Fiscal Year 2022=3,209.6 "
                "Fiscal Year 2021=2,857.4 Fiscal Year 2020=2,600.2\n"
                "Income taxes: Fiscal Year 2022=586.3 Fiscal Year 2021=629.1 Fiscal Year 2020=480.5\n")
        label = "earnings before income taxes and after-tax earnings from joint ventures"
        assert _check(text, f"In fiscal 2020, {label} was $2,857.4.").binding == "conflict"
        assert _confirmed(text, f"In fiscal 2021, {label} was $2,857.4.")

    def test_the_row_by_row_form_is_recognised_whatever_the_label(self):
        assert _is_row_by_row("Earnings before income taxes and after-tax earnings from joint ventures: "
                              "Fiscal Year 2022=3,209.6")
        assert _is_row_by_row("Revenue: 2019=32,136")
        assert not _is_row_by_row("In accordance with the agreement: Payment: $5M")
        assert not _is_row_by_row("Revenue grew strongly this year")

    def test_an_ellipsis_in_prose_is_not_a_leader(self):
        text = "The outlook is uncertain... Revenue was $452.8 million in FY2023 and margins held."
        assert not any(tabular for _, _, tabular, _ in _body_units(text))

    def test_a_short_table_under_a_long_preamble_is_still_a_table(self):
        # Title, unit caption and unit note outnumber the two rows: read as
        # prose, one "sentence" ran from the first line with figures to the
        # last row, and the next row's figure was "employment taxes" for 2021.
        for rows in (
            "Employment taxes: 2021=1,225 2020=618 2019=1,044\n"
            "Property and other taxes: 2021=1,756 2020=2,093 2019=1,551\n",
            "Employment taxes: 2021: 1,225; 2020: 618; 2019: 1,044\n"
            "Property and other taxes: 2021: 1,756; 2020: 2,093; 2019: 1,551\n",
        ):
            text = ("Verizon Communications Inc.\n(dollars in millions)\n"
                    "Values are as stated in the source, in millions.\n"
                    "Income taxes, net of amounts refunded $ 3,040 $ 2,725 $ 3,583:\n" + rows)
            assert not _check(text, "In fiscal 2021, employment taxes was $1,756.").is_confirmed, rows
            assert _confirmed(text, "In fiscal 2021, employment taxes was $1,225.")
            assert _confirmed(text, "In fiscal 2021, property and other taxes was $1,756 million.")

    def test_a_converter_row_is_recognised(self):
        assert _is_converter_row("Employment taxes: 2021=1,225 2020=618")
        assert _is_converter_row("Employment taxes: 2021: 1,225; 2020: 618")
        assert _is_converter_row("Goodwill impairments: For the Years Ended December 31, 2018: 6,149; …")
        assert not _is_converter_row("Note: the following applies: all amounts are restated")
        assert not _is_converter_row("Revenue grew strongly in 2023")


# ── Grid headers that say more than their years ──────────────────────────────

CVS = (
    "CVS HEALTH CORPORATION\n"
    "*Values are as stated in the source, in millions.*\n"
    "\n"
    "| Line Item | For the Years Ended December 31, 2018 | For the Years Ended December 31, 2017 "
    "| For the Years Ended December 31, 2016 |\n"
    "|---|---|---|---|\n"
    "| Goodwill impairments | 6,149 | 181 | — |\n"
    "| Loss on early extinguishment of debt | — | — | 643 |\n"
    "| Hedge derivatives (2) | 35.1 | 13.5 | 4.1 |\n"
)


class TestGridHeaders:
    def test_a_cell_may_say_more_than_its_year(self):
        assert _column_header(CVS.split("\n")[3]) == ("2018", "2017", "2016")
        for line, years in (
            ("| Line Item | At December 31, 2018 | At December 31, 2017 |", ("2018", "2017")),
            ("| Line Item | YEAR ENDED MAY 31, 2019 | YEAR ENDED MAY 31, 2018 |", ("2019", "2018")),
            ("| Line Item | 2019 (1) | 2018 (1) |", ("2019", "2018")),
            ("| Line Item | 52 Weeks Ended February 1, 2020 | 52 Weeks Ended February 2, 2019 |", ("2020", "2019")),
        ):
            assert _column_header(line) == years, line

    def test_what_is_not_a_header(self):
        for line in (
            # A quarter's or a half-year's column is not a fiscal year's.
            "| Line Item | Three months ended June 30, 2019 | Three months ended June 30, 2018 |",
            "| Line Item | Q4 2019 | Q4 2018 |",
            # Two columns of one year: which is which?
            "| Line Item | December 31, 2019 | December 31, 2019 |",
            # A row, not a header.
            "| Notes due 2025 | 2,522 | 2026 |",
            "| Line Item | Column 2 | Column 3 |",
        ):
            assert _cell_header(line) == (), line

    def test_an_empty_cell_does_not_shift_the_columns(self):
        assert _check(CVS, "In fiscal 2017, goodwill impairments was $6,149.").binding == "conflict"
        assert _confirmed(CVS, "In fiscal 2018, goodwill impairments was $6,149.")
        assert _confirmed(CVS, "In fiscal 2016, loss on early extinguishment of debt was $643.")

    def test_a_footnote_in_the_label_cell_does_not_shift_the_columns(self):
        assert _confirmed(CVS, "In fiscal 2017, hedge derivatives was $13.5.")
        assert _check(CVS, "In fiscal 2018, hedge derivatives was $13.5.").binding == "conflict"
        # And the footnote marker is no year's figure.
        assert not _check(CVS, "In fiscal 2018, hedge derivatives was $2.").is_confirmed

    def test_the_header_is_recorded_for_the_figures_it_dates(self):
        index = ContextIndex([{"text": CVS, "source_file": "f.pdf", "is_table": 1}])
        for fig in index.figures:
            assert bool(fig.header_line) == bool(fig.period), fig.mention.raw


# ── Which row a claim names ───────────────────────────────────────────────────


class TestNamedRows:
    def test_a_sign_parenthetical_is_not_another_name_for_the_row(self):
        text = CVS + ("| Net income (loss) | (596) | 6,623 | 5,319 |\n"
                      "| Net (income) loss attributable to noncontrolling interests | 2 | (1) | (2) |\n")
        # "(income)" says which sign is which; the claim does not name the
        # noncontrolling interests' row by using the word "income".
        assert not _check(text, "In fiscal 2017, net income (loss) was -$1.").is_confirmed
        assert _confirmed(text, "In fiscal 2017, net income (loss) was $6,623.")
        assert _confirmed(text, "In fiscal 2018, net loss was $596.")

    def test_a_real_alias_still_names_the_row(self):
        text = ("*Values are as stated in the source, in millions.*\n\n| Line Item | 2023 | 2022 |\n|---|---|---|\n"
                "| Operating Income (EBIT) | 68.0 | 51.4 |\n| Net Income | 46.4 | 34.3 |\n")
        assert _confirmed(text, "EBIT was $68.0 million in 2023.")

    def test_a_row_named_whole_outranks_one_named_in_part(self):
        text = ("GENERAL MILLS\nValues are as stated in the source, in millions.\n"
                "Other intangible assets: 2019=7,166.8 2018=7,445.1\n"
                "Other assets: 2019=974.9 2018=943.0\n")
        check = _check(text, "In fiscal 2018, other assets was $7,445.1.")
        assert check.binding == "conflict" and "other" in check.binding_note
        assert _confirmed(text, "In fiscal 2018, other assets was $943.0.")
        assert _confirmed(text, "In fiscal 2018, other intangible assets was $7,445.1.")


# ── A figure whose period nothing states ──────────────────────────────────────


class TestUndatedFigures:
    UNDATED = "Recurring vs Non-recurring Revenue:\n  Total Revenue        $452.8     100.0%\n"

    def test_it_confirms_no_year(self):
        check = _check(self.UNDATED, "Aurora's FY2022 revenue was $452.8 million.", table=False)
        assert check.status == "grounded" and check.binding == "unbound"
        assert "do not say which period" in check.binding_note
        assert not check.is_confirmed

    def test_a_claim_that_names_no_year_is_still_bound(self):
        assert _check(self.UNDATED, "Aurora's revenue was $452.8 million.", table=False).is_confirmed


# ── The converter's unit note ─────────────────────────────────────────────────

ROWS_IN_MILLIONS = (
    "ACME CORP\n"
    "Values are as stated in the source, in millions; per-share rows are not in millions.\n"
    "Net sales: 2019=32,136 2018=32,765\n"
    "Diluted earnings per share: 2019=2.45 2018=2.10\n"
    "Gross margin: 2019=45.2% 2018=44.1%\n"
)


class TestUnitNote:
    def test_the_note_states_the_scale(self):
        for note, scale in (
            ("Values are as stated in the source, in millions.", "millions"),
            ("*Values are as stated in the source, in thousands (rows 31-60 of 90).*", "thousands"),
            ("Values are as stated in the source, in USD billions.", "billions"),
            ("Financial table with 3 rows and 2 columns (2019, 2018). Values are as stated in the source, "
             "in millions; percentage rows are not in millions.", "millions"),
        ):
            m = UNIT_NOTE_RE.search(note)
            assert m and m.group(1) == scale and states_scale(note), note
        for line in ("Values are as stated in the source; the table gives no currency or scale.",
                     "Values are as stated in the source, in USD.",
                     "Results are reported in millions of customers worldwide."):
            assert not UNIT_NOTE_RE.search(line), line

    def test_the_registrys_shared_pattern_is_unchanged(self):
        # SCALE_HINT_RE is the financial verifier's too; the note is read
        # through a pattern of grounding's own.
        assert not SCALE_HINT_RE.search("Values are as stated in the source, in millions.")

    def test_a_figure_with_its_scale_word_is_found_and_confirmed(self):
        assert _confirmed(ROWS_IN_MILLIONS, "In fiscal 2019, net sales was $32,136 million.")
        assert _confirmed(ROWS_IN_MILLIONS, "In fiscal 2019, net sales was $32.1 billion.")
        assert _confirmed(ROWS_IN_MILLIONS, "In fiscal 2019, net sales was $32,136.")

    def test_another_scale_is_not_found(self):
        for claim in ("In fiscal 2019, net sales was $32,136 billion.",
                      "In fiscal 2019, net sales was $32,136 thousand."):
            assert _check(ROWS_IN_MILLIONS, claim).status == "unsupported", claim

    def test_a_per_share_figure_is_not_read_in_millions(self):
        assert _check(ROWS_IN_MILLIONS, "In fiscal 2019, diluted earnings per share was $2.45 million.").status == (
            "unsupported")
        assert _confirmed(ROWS_IN_MILLIONS, "In fiscal 2019, diluted earnings per share was $2.45.")

    def test_another_years_scaled_figure_is_a_conflict(self):
        assert _check(ROWS_IN_MILLIONS, "In fiscal 2018, net sales was $32,136 million.").binding == "conflict"

    def test_the_figure_veto_shows_the_note(self):
        start = ROWS_IN_MILLIONS.index("Net sales")
        end = ROWS_IN_MILLIONS.index("\n", start)
        assert headed_window(ROWS_IN_MILLIONS, (start, end)) == (
            "ACME CORP\n"
            "Values are as stated in the source, in millions; per-share rows are not in millions.\n"
            "Net sales: 2019=32,136 2018=32,765"
        )


# ── The revision: an undated figure's chunk, a parenthetical label ────────────

QOE = ("QUALITY OF EARNINGS — FY2023\nRecurring vs Non-recurring Revenue:\n"
       "  Total Recurring Revenue        $331.7     73.3%\n")


class TestChunkPeriods:
    def test_an_undated_figure_takes_its_chunks_only_period(self):
        assert _check(QOE, "Total recurring revenue was $331.7 million in FY2023.", table=False).is_confirmed
        check = _check(QOE, "Total recurring revenue was $331.7 million in FY2022.", table=False)
        assert check.binding == "unbound" and "do not say which period" in check.binding_note

    def test_none_with_two_periods_in_the_chunk(self):
        text = QOE + "The prior year (FY2022) is compared in Note 4.\n"
        assert not _check(text, "Total recurring revenue was $331.7 million in FY2023.", table=False).is_confirmed

    def test_a_chunk_that_states_none_takes_its_documents_name(self):
        section = "SECTION 4: REVENUE QUALITY\n  Total Recurring Revenue        $331.7     73.3%\n"
        claim = "Total recurring revenue was $331.7 million in FY2023."

        def check(source):
            [checks] = ground_texts([claim], ContextIndex([{"text": section, "source_file": source}]))
            return checks[0]

        assert check("quality_of_earnings_report_fy2023.txt").is_confirmed
        assert check("quality_of_earnings_report_fy2022.txt").binding == "unbound"
        # A name that states no year, several, a quarter or a budget dates nothing.
        for source in ("quality_of_earnings_report.txt", "financials_2022_2023.xlsx",
                       "Q3_2023_management_accounts.xlsx", "FY2023_budget.xlsx"):
            assert check(source).binding == "unbound", source

    def test_a_quarter_or_a_plan_in_the_chunk_dates_nothing(self):
        for line in ("QUARTERLY REVIEW — Q3 2023", "THREE MONTHS ENDED MARCH 31, 2023", "FY2023 BUDGET"):
            text = QOE.replace("QUALITY OF EARNINGS — FY2023", line)
            assert not _check(text, "Total recurring revenue was $331.7 million in FY2023.", table=False).is_confirmed

    def test_a_row_of_several_figures_is_still_column_unknown(self):
        text = "KEY METRICS — FY2023\n  EBITDA      $92.8     $74.0\n"
        check = _check(text, "EBITDA was $74.0 million in FY2023.", table=False)
        assert check.binding == "unbound" and "which column" in check.binding_note

    def test_a_label_in_parentheses_does_not_strand_the_figure_before_it(self):
        assert _periods("SaaS ARR Growth: 27.9% YoY (SaaS ARR as of Dec 31, 2023: $210.5M)") == [
            ("27.9%", "2023"), ("$210.5M", "2023"),
        ]
        # A converter row's label is no parenthetical: its first value stays undated.
        assert _periods("Revenue ....... $ 4,795,511: Fiscal Years 2014=$ 4,147,065")[0] == ("4,795,511", "")


# ── What the row says about the rest of the claim ─────────────────────────────

GM_ROWS = ("GENERAL MILLS\nValues are as stated in the source, in millions.\n"
           "Net sales: Fiscal Year 2019=16,865.2 Fiscal Year 2018=15,740.4 Fiscal Year 2017=15,619.8\n"
           "Net earnings (loss): Fiscal Year 2019=1,752.7 Fiscal Year 2018=(2,131.0) Fiscal Year 2017=1,657.5\n")


def _filed(text: str, claim: str, filer: str = "GENERAL MILLS, INC.", source: str = "GENERALMILLS_2019_10K.pdf"):
    chunk = {"text": text, "source_file": source, "is_table": 1, "filing_company": filer}
    [checks] = ground_texts([claim], ContextIndex([chunk]))
    return checks


class TestDirection:
    def test_a_rise_the_row_contradicts_is_a_conflict(self):
        [check] = _filed(GM_ROWS, "Net sales fell to $16,865.2 in fiscal 2019.")
        assert check.binding == "conflict" and "15,740.4 for 2018" in check.binding_note
        [check] = _filed(GM_ROWS, "Net sales rose to $16,865.2 in fiscal 2019.")
        assert check.is_confirmed

    def test_a_change_between_the_verb_and_the_level(self):
        checks = _filed(GM_ROWS, "Net sales increased year over year by $1,124.8 million to $16,865.2 million "
                                 "in fiscal 2019.")
        assert checks[-1].is_confirmed
        checks = _filed(GM_ROWS, "Net sales decreased year over year to $16,865.2 million in fiscal 2019.")
        assert checks[-1].binding == "conflict"

    def test_nothing_to_compare_nothing_checked(self):
        # The earliest column has no prior year in the row; a loss is never
        # read for a direction.
        [check] = _filed(GM_ROWS, "Net sales fell to $15,619.8 in fiscal 2017.")
        assert check.is_confirmed
        [check] = _filed(GM_ROWS, "Net loss rose to $2,131.0 in fiscal 2018.")
        assert check.binding == "bound"


class TestSign:
    def test_the_side_the_printed_sign_selects(self):
        assert _filed(GM_ROWS, "In fiscal 2018, net loss was $2,131.0.")[0].is_confirmed
        assert _filed(GM_ROWS, "In fiscal 2019, net earnings was $1,752.7.")[0].is_confirmed
        [check] = _filed(GM_ROWS, "In fiscal 2018, net earnings was $2,131.0.")
        assert check.binding == "conflict" and "loss side" in check.binding_note
        assert _filed(GM_ROWS, "In fiscal 2019, net loss was $1,752.7.")[0].binding == "conflict"

    def test_a_written_minus_names_the_other_side(self):
        assert _filed(GM_ROWS, "In fiscal 2018, net earnings was -$2,131.0.")[0].is_confirmed

    def test_a_pair_printed_negative_side_first_after_a_word(self):
        # General Mills' "Net actuarial (loss) income": the word before "(loss)" is no side.
        rows = GM_ROWS + ("Net actuarial (loss) income: Fiscal Year 2019=(253.4) Fiscal Year 2018=140.1 "
                          "Fiscal Year 2017=197.9\n")
        assert _filed(rows, "In fiscal 2019, net actuarial loss was $253.4.")[0].is_confirmed
        assert _filed(rows, "In fiscal 2018, net actuarial income was $140.1.")[0].is_confirmed
        [check] = _filed(rows, "In fiscal 2019, net actuarial income was $253.4.")
        assert check.binding == "conflict" and "loss side" in check.binding_note
        assert _filed(rows, "In fiscal 2017, net actuarial loss was $197.9.")[0].binding == "conflict"

    @pytest.mark.parametrize("label, pair", [
        ("Net income (loss)", ("income", "loss")),
        ("(Increase) decrease in receivables", ("decrease", "increase")),
        ("Net actuarial (loss) income", ("income", "loss")),
        ("Net (loss) gain on mark-to-market valuation of commodity positions", ("gain", "loss")),
        ("Net (decrease) increase in cash, cash equivalents and restricted funds", ("increase", "decrease")),
        ("Other (income) expense, net", ("expense", "income")),
        ("Provision for (benefit from) income taxes", None),
        ("Operating Income (EBIT)", None),
        ("(Gain) or loss on sale of businesses", None),
    ])
    def test_the_sign_pair_of_a_label(self, label, pair):
        from src.verification.numeric_grounding import _sign_pair

        assert _sign_pair(label) == pair


class TestCompany:
    def test_the_filer_named_is_confirmed(self):
        assert _filed(GM_ROWS, "General Mills reported net sales of $16,865.2 in fiscal 2019.")[0].is_confirmed
        assert _filed(GM_ROWS, "In fiscal 2019, net sales was $16,865.2.")[0].is_confirmed

    def test_another_company_is_left_to_the_judge(self):
        for name in ("PepsiCo", "Kraft Heinz"):
            [check] = _filed(GM_ROWS, f"{name} reported net sales of $16,865.2 in fiscal 2019.")
            assert check.status == "grounded" and check.binding == "unbound", name
            assert "GENERAL MILLS" in check.binding_note

    def test_the_filings_file_name_names_it_too(self):
        # "Coca-Cola" against THE COCA-COLA COMPANY; "AMD" against a registrant
        # named in full, through AMD_2015_10K.pdf.
        text = GM_ROWS.replace("GENERAL MILLS", "ADVANCED MICRO DEVICES")
        assert _filed(text, "AMD reported net sales of $16,865.2 in fiscal 2019.",
                      filer="ADVANCED MICRO DEVICES, INC.", source="AMD_2015_10K.pdf")[0].is_confirmed
        assert _filed(GM_ROWS, "Coca-Cola reported net sales of $16,865.2 in fiscal 2019.",
                      filer="THE COCA-COLA COMPANY", source="COCACOLA_2017_10K.pdf")[0].is_confirmed

    def test_no_registrant_no_check(self):
        chunk = {"text": GM_ROWS, "source_file": "financials.txt", "is_table": 1}
        [[check]] = ground_texts(["PepsiCo reported net sales of $16,865.2 in fiscal 2019."], ContextIndex([chunk]))
        assert check.is_confirmed

    def test_a_first_word_names_the_filer_only_written_as_a_name(self):
        # Decision 98: "general" in "selling, general, and administrative" named
        # GENERAL MILLS, INC. by its first word, read without case.
        rows = GM_ROWS + ("Selling, general, and administrative expenses: Fiscal Year 2019=2,935.8 "
                          "Fiscal Year 2018=2,752.6\n")
        [check] = _filed(rows, "PepsiCo reported selling, general, and administrative expenses of $2,935.8 in "
                               "fiscal 2019.")
        assert check.binding == "unbound" and "GENERAL MILLS" in check.binding_note
        assert _filed(rows, "General Mills reported selling, general, and administrative expenses of $2,935.8 in "
                            "fiscal 2019.")[0].is_confirmed
        # Written as a name, the first word still names its registrant.
        text = GM_ROWS.replace("GENERAL MILLS", "ADOBE SYSTEMS")
        for claim in ("Adobe reported net sales of $16,865.2 in fiscal 2019, ahead of Photoshop's plan.",
                      "ADOBE reported net sales of $16,865.2 in fiscal 2019, ahead of Photoshop's plan."):
            assert _filed(text, claim, filer="ADOBE SYSTEMS INCORPORATED", source="filing.pdf")[0].is_confirmed
        [check] = _filed(text, "adobe reported net sales of $16,865.2 in fiscal 2019, ahead of Photoshop's plan.",
                         filer="ADOBE SYSTEMS INCORPORATED", source="filing.pdf")
        assert check.binding == "unbound" and "ADOBE SYSTEMS" in check.binding_note
