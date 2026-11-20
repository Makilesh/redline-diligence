"""
The registry behind the deal-level figure tie-out (Decision 92): which
figures two documents share, how a table's periods are read, and how a data
room is read document by document.

Offline: hand-made payloads, no models.
"""

from __future__ import annotations

import pytest

from src.utils.numerical_registry import (
    NumericalRegistry,
    column_basis,
    column_period,
    is_generic_label,
    normalize_metric_label,
)


def text_chunk(source: str, text: str, **extra) -> dict:
    return {"source_file": source, "text": text, **extra}


def statement(source: str, revenue: str, ebitda: str = "$92.8", **extra) -> dict:
    return text_chunk(
        source,
        "(in millions of USD)\n"
        "                     FY2023      FY2022\n"
        f"Revenue              {revenue}      $387.1\n"
        f"EBITDA               {ebitda}       $74.0\n",
        **extra,
    )


def tie(chunks: list[dict]):
    registry, _ = NumericalRegistry.from_documents(chunks)
    return registry.tie_out()


def keyed(result) -> dict[tuple[str, str], str]:
    return {(normalize_metric_label(f.metric_name), f.fiscal_year): f.status for f in result.figures}


# ==============================================================================
# Comparison
# ==============================================================================


class TestComparison:
    def test_documents_that_agree_and_one_that_conflicts(self):
        result = tie([
            statement("financials.txt", "$452.8"),
            statement("cim.pdf", "$452.8"),
            statement("qoe.pdf", "$452.8", ebitda="$99.0"),
        ])
        assert keyed(result)[("revenue", "FY2023")] == "agree"
        assert keyed(result)[("ebitda", "FY2023")] == "conflict"
        conflict = result.conflicts[0]
        assert conflict.metric_name == "EBITDA"
        assert {e.source_file for e in conflict.entries} == {"financials.txt", "cim.pdf", "qoe.pdf"}
        assert conflict.max_deviation_pct == pytest.approx((99.0 - 92.8) / 99.0 * 100, abs=0.01)
        # One entry per document, as printed.
        assert sorted(e.as_printed for e in conflict.entries) == ["$92.8", "$92.8", "$99.0"]

    def test_conflicts_come_first_largest_gap_first(self):
        result = tie([
            statement("a.txt", "$452.8", ebitda="$92.8"),
            statement("b.txt", "$460.0", ebitda="$99.0"),
        ])
        statuses = [f.status for f in result.figures]
        assert statuses == sorted(statuses, key=lambda s: s != "conflict")
        gaps = [f.max_deviation_pct for f in result.conflicts]
        assert gaps == sorted(gaps, reverse=True)

    def test_scale_and_printed_rounding_are_not_conflicts(self):
        thousands = text_chunk(
            "model.txt",
            "(in thousands of USD)\n            FY2023       FY2022\nRevenue    452,800      387,100\n",
        )
        rounded = text_chunk(
            "deck.txt", "($ in billions)\n          FY2023    FY2022\nRevenue     $0.5      $0.4\n",
        )
        result = tie([statement("financials.txt", "$452.8"), thousands, rounded])
        assert keyed(result)[("revenue", "FY2023")] == "agree"
        assert keyed(result)[("revenue", "FY2022")] == "agree"

    def test_a_gap_beyond_the_printed_rounding_is_a_conflict(self):
        assert keyed(tie([statement("a.txt", "$452.8"), statement("b.txt", "$452.76")]))[
            ("revenue", "FY2023")] == "agree"
        assert keyed(tie([statement("a.txt", "$452.8"), statement("b.txt", "$452.9")]))[
            ("revenue", "FY2023")] == "conflict"

    def test_single_source_figures_are_counted_not_listed(self):
        result = tie([statement("a.txt", "$452.8"), text_chunk(
            "b.txt", "(in millions)\n          FY2023\nHeadcount growth     12.0%\n")])
        assert result.figures == []
        assert result.single_source == 5  # a.txt's four and b.txt's one

    def test_a_percentage_and_an_amount_under_one_label_are_different_figures(self):
        result = tie([
            text_chunk("a.txt", "(in millions)\n             FY2023\nGross margin    60.0%\n"),
            text_chunk("b.txt", "(in millions)\n             FY2023\nGross margin    $271.7\n"),
        ])
        assert result.conflicts == []
        assert result.single_source == 2

    def test_labels_that_name_nothing_are_not_compared(self):
        result = tie([
            text_chunk("a.txt", "(in millions)\n        FY2023\nTotal         $56\nOther, net    $3\n"),
            text_chunk("b.txt", "(in millions)\n        FY2023\nTotal         $90\nOther, net    $7\n"),
        ])
        assert result.figures == [] and result.generic == 2
        assert is_generic_label("Balance at December 31, 2021") and is_generic_label("Row 3")
        assert not is_generic_label("Total assets")

    def test_a_document_stating_one_label_twice_is_not_compared(self):
        # A segment table and the consolidated statement both say "Revenue".
        segment = text_chunk("10k.pdf", "(in millions)\n          FY2023\nRevenue       $310.0\n")
        result = tie([statement("10k.pdf", "$452.8"), segment, statement("release.pdf", "$452.8")])
        assert ("revenue", "FY2023") not in keyed(result)
        assert result.ambiguous == 1

    def test_figures_with_no_period_are_counted(self):
        result = tie([text_chunk("a.txt", "Revenue    $452.8\n"), statement("b.txt", "$452.8")])
        assert result.without_period == 0  # the lone row has no header, so it is not read at all
        structured = {
            "source_file": "model.xlsx", "text": "",
            "structured_rows": [{"line_item": "Revenue", "values": {"Column 2": {
                "raw": "452.8", "raw_value": 452.8, "normalized_value": 452.8e6, "period": "",
            }}}],
        }
        assert tie([structured]).without_period == 1


# ==============================================================================
# Periods
# ==============================================================================


class TestPeriods:
    def test_prose_naming_a_year_is_not_a_header(self):
        # American Water Works' 2022 10-K: the section heading before a table
        # ("... In November 2020, Cal Am refiled ...") headed its rows as 2020.
        chunk = text_chunk(
            "awk_2022.pdf",
            "Application, which remains pending before the Coastal Commission. In November 2020, "
            "Cal Am refiled the Original Jurisdiction Application.\n"
            "(In millions)  Effective Date  Amount\n"
            "| Total general rate case authorizations | - | - | $ | 90 |\n",
        )
        registry = NumericalRegistry.from_chunks([chunk])
        assert len(registry) == 0

    def test_a_caption_in_parentheses_is_part_of_a_header_a_wrapped_paragraph_is_not(self):
        credit = text_chunk(
            "credit_agreement_summary.txt",
            "SECTION 1: OUTSTANDING INDEBTEDNESS (in millions of USD, as of December 31, 2023)\n"
            "  Total Debt                                                 $91.4\n"
            "  Net Debt / Reported EBITDA (FY2023)                          0.2x\n"
            "  Net Debt / Reported EBITDA (FY2022)                          0.7x\n",
        )
        note = text_chunk(
            "financials.txt",
            "Note 12 - Subsequent Events: On February 15, 2024, the Company's Board of Directors approved the exploration\n"
            "  Strategic alternatives                                      $5.0\n",
        )
        keys = set(NumericalRegistry.from_chunks([credit, note])._entries)
        # A row whose label names another period than its column is not read.
        assert keys == {("total debt", "FY2023"), ("net debt / ebitda fy2023", "FY2023")}

    def test_quarter_and_half_year_columns_are_their_own_periods(self):
        ten_q = text_chunk(
            "bby_q2.pdf",
            " | Three Months Ended |  |  | Six Months Ended |  | \n"
            " | July 29, 2023 |  | July 30, 2022 |  | July 29, 2023 |  | July 30, 2022 | \n"
            "Revenue | $ | 9,583 |  | $ | 10,329 |  | $ | 19,050 |  | $ | 20,976 | \n",
            filing_form="10-Q",
        )
        registry = NumericalRegistry.from_chunks([ten_q])
        assert {p for (_, p) in registry._entries} == {
            "3M ending 2023-07-29", "3M ending 2022-07-30", "6M ending 2023-07-29", "6M ending 2022-07-30",
        }

    def test_a_10q_bare_year_is_not_the_10k_fiscal_year(self):
        ten_k = statement("bby_10k.pdf", "$46,298", filing_form="10-K")
        ten_q = text_chunk(
            "bby_q2.pdf", "(in millions)\n           2023      2022\nRevenue    $9,583    $10,329\n",
            filing_form="10-Q",
        )
        result = tie([ten_k, ten_q])
        assert result.conflicts == []

    def test_balance_sheet_dates_compare_by_date(self):
        ten_k = text_chunk(
            "bby_10k.pdf", "($ in millions)\n         January 28, 2023    January 29, 2022\n"
            "Total assets         $15,803             $17,504\n", filing_form="10-K",
        )
        ten_q = text_chunk(
            "bby_q2.pdf", "($ in millions)\n      July 29, 2023    January 28, 2023    July 30, 2022\n"
            "Total assets     $15,105          $15,803             $16,450\n", filing_form="10-Q",
        )
        result = tie([ten_k, ten_q])
        assert keyed(result) == {("total assets", "2023-01-28"): "agree"}

    def test_a_31_december_balance_is_that_fiscal_year_in_an_annual_document(self):
        credit = text_chunk(
            "credit_agreement_summary.txt",
            "(in millions of USD, as of December 31, 2023)\n"
            "  Less: Cash and Cash Equivalents                            ($70.8)\n",
        )
        balance = text_chunk(
            "aurora_financials_fy2023.txt",
            "(in millions of USD, as of December 31)\nASSETS          FY2023          FY2022\n"
            "  Cash and Cash Equivalents                $89.4           $72.1\n",
        )
        assert keyed(tie([credit, balance])) == {("cash and cash equivalents", "FY2023"): "conflict"}

    def test_estimates_are_not_actuals(self):
        result = tie([
            text_chunk("cim.pdf", "(in millions)\n         2023A     2024E\nRevenue   $452.8    $520.0\n"),
            text_chunk("10k.pdf", "(in millions)\n          2024      2023\nRevenue   $498.1    $452.8\n"),
        ])
        assert keyed(result) == {("revenue", "FY2023"): "agree"}

    def test_a_january_year_end_is_keyed_by_its_end(self):
        assert column_period("Fiscal Year Ended January 28, 2023") == "FY ending 2023-01"
        assert column_period("Year Ended December 31, 2023") == "FY2023"
        assert column_period("July 29, 2023", interim=True) == "2023-07-29"
        assert column_period("July 29, 2023 Three Months Ended", interim=True) == "3M ending 2023-07-29"
        assert column_period("2023", interim=True) is None
        assert column_period("FY2023 Budget") is None
        assert column_period("Six Months Ended") is None
        assert column_period("Column 3") is None

    def test_event_dates_are_not_periods(self):
        # PepsiCo's acquisitions table, stitched to the charges table below it
        # (whose columns are fiscal years): both 10-Ks keyed the charges by
        # acquisition date, one year apart.
        def acquisitions(source: str, cost: list[str]) -> dict:
            return {"source_file": source, "structured_rows": [{"line_item": "Cost of sales", "values": {
                date: {"raw": c, "raw_value": float(c), "normalized_value": float(c), "period": "FY2020"}
                for date, c in zip(["March 23, 2020", "April 24, 2020", "June 1, 2020"], cost)
            }}]}

        result = tie([acquisitions("pep_2021.pdf", ["1", "32", "34"]),
                      acquisitions("pep_2022.pdf", ["0", "1", "32"])])
        assert result.figures == [] and result.without_period == 6
        # Two balance dates of one year, as a 10-Q prints them, are periods.
        assert column_period("January 28, 2023") == "2023-01-28"

    def test_spans_the_registry_cannot_place_are_not_compared(self):
        registry = NumericalRegistry.from_chunks([text_chunk(
            "a.pdf", "Three Months Ended     Six Months Ended     Nine Months Ended\n"
            "     2023        2022\nRevenue    $100    $90\n")])
        assert len(registry) == 0


# ==============================================================================
# Reading real table text
# ==============================================================================


class TestTableText:
    def test_a_footnote_marker_in_the_label_cell_is_not_a_value(self):
        chunk = text_chunk(
            "a.pdf", "(In millions)  2021  2020  2019\nRegulated revenues (2) | 180 |  |  | 83 |  |  | 79 | \n",
        )
        entries = NumericalRegistry.from_chunks([chunk])._entries
        assert {p: es[0].normalized_value for (_, p), es in entries.items()} == {
            "FY2021": 180e6, "FY2020": 83e6, "FY2019": 79e6}

    def test_footnote_markers_do_not_split_one_line_item(self):
        assert normalize_metric_label("Changes in working capital (b)") == normalize_metric_label(
            "Changes in working capital (a)")
        assert normalize_metric_label("Deferred Revenue (Non-Current)") != "deferred revenue"

    def test_a_percent_printed_in_its_own_cell_is_a_percentage(self):
        chunk = text_chunk("a.pdf", "(In millions)  2021  2020\nEfficiency ratio | 34.3 |  | % |  | 34.5 |  | %\n")
        entry = NumericalRegistry.from_chunks([chunk])._entries[("efficiency ratio", "FY2021")][0]
        assert entry.kind == "percent" and entry.as_printed == "34.3%"

    def test_a_dollar_cell_carries_its_column_to_the_amount(self):
        row = {"line_item": "Net income", "values": {
            "For the Years Ended December 31, 2021": {"raw": "$"},
            "Column 3": {"raw": "1,263", "raw_value": 1263.0, "normalized_value": 1263e6, "period": ""},
            "2020": {"raw": "$"},
            "Column 6": {"raw": "709", "raw_value": 709.0, "normalized_value": 709e6, "period": ""},
        }}
        entries = NumericalRegistry.from_chunks([{"source_file": "a.pdf", "structured_rows": [row]}])._entries
        assert {p: es[0].normalized_value for (_, p), es in entries.items()} == {
            "FY2021": 1263e6, "FY2020": 709e6}

    def test_a_column_the_converter_could_not_name_is_never_read_as_figures(self):
        chunk = text_chunk("a.pdf", "(In millions)  2021  2020  2019\n| Line Item | Column 2 | Column 3 | Column 4 |\n")
        assert len(NumericalRegistry.from_chunks([chunk])) == 0


# ==============================================================================
# A data room, document by document
# ==============================================================================


class TestDocumentOrder:
    def test_a_table_section_without_a_header_reads_its_documents_only_one(self):
        # The sample financials print "FY2023  FY2022" over their statements and
        # none over the "KEY FINANCIAL METRICS" table below them.
        statements = text_chunk(
            "financials.txt",
            "(in millions of USD)\n             FY2023      FY2022\nRevenue     $452.8      $387.1\n\n"
            "ASSETS       FY2023      FY2022\nCash         $89.4       $72.1\n",
            chunk_index=0, is_table=1,
        )
        metrics = text_chunk(
            "financials.txt", "KEY FINANCIAL METRICS AND NOTES\n  Adjusted EBITDA     $97.3     $74.0\n",
            chunk_index=1, is_table=1,
        )
        qoe = text_chunk("qoe.txt", "(in millions)\n               FY2023     FY2022\nAdjusted EBITDA    $99.0    $73.8\n")
        result = tie([metrics, statements, qoe])
        assert keyed(result)[("adjusted ebitda", "FY2023")] == "conflict"
        # A lone chunk lends nothing: the verifier's view is unchanged.
        assert ("adjusted ebitda", "FY2023") not in NumericalRegistry.from_chunks([metrics])._entries

    def test_a_document_with_two_headers_lends_none(self):
        chunks = [
            text_chunk("10k.pdf", "(in millions)\n        2023    2022    2021\nRevenue  $3  $2  $1\n",
                       chunk_index=0, is_table=1),
            text_chunk("10k.pdf", "(in millions)\n        2023    2022\nAssets   $9  $8\n", chunk_index=1, is_table=1),
            text_chunk("10k.pdf", "Debt     $5    $4\n", chunk_index=2, is_table=1),
        ]
        registry, _ = NumericalRegistry.from_documents(chunks)
        assert not any(label == "debt" for label, _ in registry._entries)

    def test_a_header_printed_once_or_a_public_filing_lends_none(self):
        once = [
            text_chunk("memo.txt", "(in millions)\n        FY2023    FY2022\nRevenue  $3  $2\n", chunk_index=0, is_table=1),
            text_chunk("memo.txt", "Debt     $5    $4\n", chunk_index=1, is_table=1),
        ]
        twice = [{**c, "text": c["text"] + "Assets        FY2023    FY2022\nCash  $1  $1\n"} if i == 0 else c
                 for i, c in enumerate(once)]
        filing = [{**c, "filing_form": "10-K"} for c in twice]
        for chunks, lends in ((once, False), (twice, True), (filing, False)):
            registry, _ = NumericalRegistry.from_documents(chunks)
            assert (("debt", "FY2023") in registry._entries) is lends

    def test_a_table_cut_in_two_keeps_its_header(self):
        first = text_chunk("a.txt", "(in millions)\n            FY2023    FY2022\nRevenue    $452.8    $387.1\n",
                           chunk_index=0, table_id="t1")
        second = text_chunk("a.txt", "Net income    $46.4    $34.3\n", chunk_index=1, table_id="t1")
        registry, _ = NumericalRegistry.from_documents([second, first])
        assert ("net income", "FY2023") in registry._entries

    def test_a_stop_request_ends_the_build_between_documents(self):
        calls = iter([False, True])
        registry, read = NumericalRegistry.from_documents(
            [statement("a.txt", "$1"), statement("b.txt", "$2")], should_stop=lambda: next(calls))
        assert read == 1 and {e.source_file for es in registry._entries.values() for e in es} == {"a.txt"}


# ==============================================================================
# Which line item a figure is (Decision 93)
# ==============================================================================


def cell(raw: str, scale: float = 1e6) -> dict:
    value = float(raw.strip("()").replace(",", ""))
    return {"raw": raw, "raw_value": value, "normalized_value": value * scale, "scale_factor": scale}


def rows_chunk(source: str, rows: list[tuple[str, dict]], table_id: str = "t1", span=(1, 20), **extra) -> dict:
    structured = [{"line_item": label, "values": values} for label, values in rows]
    return {"source_file": source, "table_id": table_id, "is_table": 1, "table_row_range": list(span),
            "structured_rows": structured, "content_type": "table_row_by_row", "chunk_index": 0, **extra}


class TestIdentity:
    def test_a_balance_and_a_flow_of_one_name_are_never_one_figure(self):
        balance = {"December 31, 2020": cell("50")}
        result = tie([
            rows_chunk("awk_2020.pdf", [("Accrued taxes", balance)]),
            rows_chunk("awk_2021.pdf", [("Accrued taxes", balance)]),
            rows_chunk("awk_2022.pdf", [("Accrued taxes", {"For the Years Ended December 31, 2020": cell("3")})]),
        ])
        assert not result.conflicts
        [agreement] = result.agreements
        assert {e.source_file for e in agreement.entries} == {"awk_2020.pdf", "awk_2021.pdf"}

    def test_a_date_before_the_first_year_makes_every_column_a_balance(self):
        # "December 31, 2023  2022" keys both columns as balances; the 2022 one is a balance too.
        balances = text_chunk("a.pdf", "December 31, 2023  2022\nAccrued taxes  50  45\n")
        flows = text_chunk("b.pdf", "Year Ended December 31, 2022\nAccrued taxes  3\n")
        assert not tie([balances, flows]).conflicts

    def test_a_table_continued_in_the_next_chunk_keeps_its_columns_basis(self):
        head = text_chunk("a.pdf", "December 31, 2023  2022\nAccrued taxes  50  45\n",
                          table_id="t9", chunk_index=0)
        rest = text_chunk("a.pdf", "Deferred revenue  12  10\n", table_id="t9", chunk_index=1)
        flows = text_chunk("b.pdf", "Year Ended December 31, 2023\nDeferred revenue  3\n")
        registry, _ = NumericalRegistry.from_documents([head, rest, flows])
        assert {e.basis for e in registry._entries[("deferred revenue", "FY2023")]} == {"balance", "flow"}
        assert not registry.tie_out().conflicts

    def test_a_row_of_nils_is_a_line_item_not_a_subheading(self):
        chunk = rows_chunk("a.pdf", [
            ("Swaptions:", {}),
            ("Notional value sold", {"2021": cell("2,000")}),
            ("Amortization of intangibles", {"2021": {"raw": "—"}, "2020": {"raw": "$ —"}}),
            ("Notional value sold (2)", {"2021": cell("1,000")}),
        ])
        registry, _ = NumericalRegistry.from_documents([chunk])
        assert {label for label, _ in registry._entries} == {"swaptions notional value sold"}

    def test_a_balance_still_meets_a_figure_whose_header_does_not_say(self):
        result = tie([
            rows_chunk("a.pdf", [("Total deferred tax assets", {"December 31, 2020": cell("1,083")})]),
            rows_chunk("b.pdf", [("Total deferred tax assets", {"2020": cell("1,083")})]),
        ])
        assert [f.status for f in result.figures] == ["agree"]

    def test_a_line_item_repeated_under_subheadings_is_read_with_its_subheading(self):
        older = rows_chunk("vz_2021.pdf", [
            ("Treasury Rate Locks:", {}),
            ("Pre-tax gain recognized in Interest expense", {"2020": cell("15")}),
            ("Swaptions:", {}),
            ("Pre-tax gain recognized in Interest expense (2)", {"2021": cell("11")}),
        ])
        newer = rows_chunk("vz_2022.pdf", [
            ("Interest Rate Swaps:", {}),
            ("Pre-tax gain recognized in Interest expense", {"2021": cell("2")}),
        ])
        result = tie([older, newer])
        assert not result.conflicts
        registry, _ = NumericalRegistry.from_documents([older])
        assert {label for label, _ in registry._entries} == {
            "treasury rate locks pre tax gain recognized in interest expense",
            "swaptions pre tax gain recognized in interest expense"}

    def test_a_repeated_line_item_whose_subheading_is_out_of_reach_is_not_read(self):
        tail = rows_chunk("vz_2021.pdf", [
            ("Pre-tax gain recognized in Interest expense", {"2021": cell("11")}),
            ("Pre-tax gain recognized in Interest expense (2)", {"2021": cell("4")}),
        ], span=(21, 22))
        assert len(NumericalRegistry.from_chunks([tail])) == 0

    def test_a_subheading_carries_to_the_next_chunk_of_its_table(self):
        head = rows_chunk("a.pdf", [("Swaptions:", {}), ("Notional value sold", {"2021": cell("2,000")})],
                          span=(1, 2), chunk_index=0)
        rest = rows_chunk("a.pdf", [("Notional value sold (2)", {"2021": cell("1,000")})], span=(3, 3), chunk_index=1)
        registry, _ = NumericalRegistry.from_documents([rest, head])
        assert ("swaptions notional value sold", "FY2021") in registry._entries

    def test_a_table_is_read_once_from_its_structured_rows(self):
        dates = {"January 28, 2017": cell("(185)"), "January 30, 2016": cell("123")}
        rows = rows_chunk("bby_2017.pdf", [("Receivables", dates)], span=(1, 1))
        grid = {"source_file": "bby_2017.pdf", "table_id": "t1", "is_table": 1, "table_row_range": [1, 1],
                "content_type": "table_markdown", "chunk_index": 1,
                "text": "| Fiscal Years Ended | January 28, 2017 | January 30, 2016 |\n|---|---|---|\n"
                        "| Receivables | (185) | 123 |\n"}
        registry, _ = NumericalRegistry.from_documents([rows, grid])
        assert sorted(period for label, period in registry._entries if label == "receivables") == [
            "2016-01-30", "2017-01-28"]

    def test_a_grid_whose_rows_name_no_period_is_still_read_from_its_text(self):
        rows = rows_chunk("awk.pdf", [("Total general rate case authorizations",
                                       {"Column 2": cell("56"), "Column 3": cell("45")})], span=(1, 1))
        grid = {"source_file": "awk.pdf", "table_id": "t1", "is_table": 1, "table_row_range": [1, 1],
                "content_type": "table_markdown", "chunk_index": 1,
                "text": "(In millions)  2020  2019\n| General rate cases | Column 2 | Column 3 |\n"
                        "| Total general rate case authorizations | 56 | 45 |\n"}
        registry, _ = NumericalRegistry.from_documents([rows, grid])
        assert ("total general rate case authorizations", "FY2020") in registry._entries

    def test_a_number_too_long_for_a_float_is_no_figure(self):
        header = "Three Months Ended  August 31, 2022\n"
        result = tie([text_chunk("a.pdf", header + "Net revenue  " + "7" * 400 + "\n"),
                      text_chunk("b.pdf", header + "Net revenue  0\n")])
        assert result.figures == []

    @pytest.mark.parametrize("header, interim, key, basis", [
        ("12 Weeks Ended 3/25/2023", True, "3M ending 2023-03-25", "flow"),
        ("3/25/2023", True, "2023-03-25", "balance"),
        ("12/31/2022", False, "FY2022", "balance"),
        ("For the Years Ended December 31, 2020", False, "FY2020", "flow"),
        ("At December 31, 2020", False, "FY2020", "balance"),
        ("2020", False, "FY2020", ""),
    ])
    def test_column_period_and_basis(self, header, interim, key, basis):
        assert column_period(header, interim) == key
        assert column_basis(header) == basis
