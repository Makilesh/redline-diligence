"""
The table path: a figure is never rewritten on its way into the index.

Covers the converter (cells printed as stated, the table's scale applied only
where it belongs and only in structured_rows), the derived metrics (computed
only when the arithmetic is defined), long tables (every chunk keeps its
header), the workbook reader, scale/currency detection, and the Financial
Verifier's registry actually seeing converter tables.

No network, no LLM, no model weights.
"""

from __future__ import annotations

import pytest

from src.data_processing.excel_normalizer import ExcelNormalizer
from src.data_processing.financial_table_converter import (
    UNIT_COUNT,
    UNIT_CURRENCY,
    UNIT_PER_SHARE,
    UNIT_PERCENT,
    UNIT_RATIO,
    annual_columns,
    classify_row,
    fmt_exact,
    parse_cell,
    registry_period,
)
from src.data_processing.ingest_pipeline import _table_sections, extract_and_chunk
from src.data_processing.multi_page_table_stitcher import MultiPageTableStitcher
from src.utils.numerical_registry import NumericalRegistry
from tests.fixtures import stress_docs as sx

DEAL = "deal-test"
MAX_TOKENS = 800

# The audit's table: one "in millions" caption over rows that are not all in
# millions.
MIXED_UNIT_ROWS = [
    ["($ in millions, except per share)", "FY2022", "FY2023"],
    ["Revenue", "398.1", "452.8"],
    ["Cost of revenue", "(158.3)", "(181.1)"],
    ["Diluted EPS", "2.31", "2.74"],
    ["Headcount", "1,150", "1,250"],
    ["Net leverage", "3.2", "2.9"],
    ["Gross margin", "60.0%", "61.5%"],
]


def _by_representation(rows, heading="Income Statement", **kwargs) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for section in _table_sections(rows, "", heading, 3, "d_t000", **kwargs):
        grouped.setdefault(section["table_representation"], []).append(section)
    return grouped


def _structured(rows, **kwargs) -> dict[str, dict]:
    reps = _by_representation(rows, **kwargs)
    return {
        row["line_item"]: row
        for section in reps["row_by_row"]
        for row in section["structured_rows"]
    }


def _metrics(rows) -> dict[str, float]:
    reps = _by_representation(rows)
    if "metrics_summary" not in reps:
        return {}
    return {name: m["value"] for name, m in reps["metrics_summary"][0]["metrics"].items()}


# ==============================================================================
# Figures are printed as stated
# ==============================================================================


class TestFiguresAreNeverRewritten:
    def test_no_representation_multiplies_a_cell_by_the_table_scale(self):
        reps = _by_representation(MIXED_UNIT_ROWS)
        assert set(reps) == {"narrative", "row_by_row", "metrics_summary", "markdown"}
        for name in ("narrative", "row_by_row", "markdown"):
            text = reps[name][0]["text"]
            for stated in ("398.1", "452.8", "2.74", "1,250", "2.9", "61.5%", "(181.1)"):
                assert stated in text, f"{name} lost {stated}"
            # What the converter used to manufacture under "in millions".
            for invented in ("452,800,000", "2,740,000", "1,250,000,000", "2,900,000"):
                assert invented not in text, f"{name} invented {invented}"

    def test_the_unit_is_stated_once_per_representation(self):
        reps = _by_representation(MIXED_UNIT_ROWS)
        for name in ("narrative", "row_by_row", "markdown"):
            section = reps[name][0]
            assert "in USD millions" in section["text"]
            assert "per-share" in section["text"] and "not in millions" in section["text"]
            assert section["scale_factor"] == 1_000_000.0
            assert section["currency"] == "USD"

    def test_grid_prints_every_cell_exactly(self):
        rows = [
            ["Per share data", "FY2023"],
            ["Dividend per share", "0.0375"],
            ["Exchange ratio", "1.23456"],
            ["Headcount", "1,150"],
        ]
        grid = _by_representation(rows)["markdown"][0]["text"]
        assert "| Dividend per share | 0.0375 |" in grid
        assert "| Exchange ratio | 1.23456 |" in grid
        assert "| Headcount | 1,150 |" in grid
        assert "0.04" not in grid and "1,150.00" not in grid

    def test_only_currency_rows_take_the_scale_in_structured_rows(self):
        rows = _structured(MIXED_UNIT_ROWS)

        revenue = rows["Revenue"]["values"]["FY2023"]
        assert rows["Revenue"]["unit"] == UNIT_CURRENCY
        assert revenue["raw"] == "452.8"
        assert revenue["raw_value"] == 452.8
        assert revenue["normalized_value"] == pytest.approx(452_800_000)
        assert revenue["scale_factor"] == 1_000_000.0

        cost = rows["Cost of revenue"]["values"]["FY2023"]
        assert cost["raw"] == "(181.1)" and cost["raw_value"] == -181.1

        for label, unit, value in (
            ("Diluted EPS", UNIT_PER_SHARE, 2.74),
            ("Headcount", UNIT_COUNT, 1250.0),
            ("Net leverage", UNIT_RATIO, 2.9),
            ("Gross margin", UNIT_PERCENT, 61.5),
        ):
            cell = rows[label]["values"]["FY2023"]
            assert rows[label]["unit"] == unit
            assert cell["scale_factor"] == 1.0
            assert cell["normalized_value"] == value
        assert rows["Gross margin"]["values"]["FY2023"]["kind"] == "percent"

    def test_caption_outside_the_grid_sets_the_unit(self):
        rows = [["Segment", "FY2022", "FY2023"], ["Platform subscriptions", "301.2", "362.9"]]
        without = _by_representation(rows)["row_by_row"][0]
        assert without["scale_factor"] == 1.0
        assert "gives no currency or scale" in without["text"]

        with_caption = _by_representation(rows, caption="(USD in millions)")["row_by_row"][0]
        assert with_caption["scale_factor"] == 1_000_000.0
        assert "(USD in millions)" in with_caption["text"]
        assert "362.9" in with_caption["text"]

    def test_non_numeric_table_is_indexed_verbatim(self):
        sections = _table_sections(
            [["Party", "Role"], ["Meridian", "Buyer"]], "Party | Role\nMeridian | Buyer", "Parties",
            None, "d_t001",
        )
        assert [s["content_type"] for s in sections] == ["table_text"]
        assert sections[0]["text"] == "Party | Role\nMeridian | Buyer"


class TestCells:
    @pytest.mark.parametrize(
        ("raw", "value", "kind"),
        [
            ("452.8", 452.8, "number"),
            ("(1,234)", -1234.0, "number"),
            ("-1,234", -1234.0, "number"),
            ("−8.2", -8.2, "number"),  # U+2212, what PDFs print
            ("$ 310,000", 310000.0, "number"),
            ("$(4,300)", -4300.0, "number"),
            ("12,34,567", 1234567.0, "number"),  # Indian grouping
            ("61.5%", 61.5, "percent"),
            ("(3.2%)", -3.2, "percent"),
            ("7.5x", 7.5, "multiple"),
            (452.8, 452.8, "number"),
            (1250, 1250.0, "number"),
        ],
    )
    def test_numbers(self, raw, value, kind):
        cell = parse_cell(raw)
        assert cell.value == value
        assert cell.kind == kind

    @pytest.mark.parametrize(
        "raw",
        [
            "(1,234",    # the ")" landed in the next PDF cell: sign unknown
            "1,234)",
            "1,5",       # 15 in one locale, 1.5 in another
            "1,23",
            "—",
            "n/a",
            "",
            None,
            "See note 4",
            "FY2023",
        ],
    )
    def test_anything_ambiguous_stays_text(self, raw):
        cell = parse_cell(raw)
        assert cell.value is None
        assert cell.printed == ("" if raw is None else raw)

    @pytest.mark.parametrize(
        ("value", "printed"),
        [
            (387.1, "387.1"),
            (0.0375, "0.0375"),
            (1.23456, "1.23456"),
            (452_800_000.0, "452,800,000"),
            (1250.0, "1,250"),
            (-8.2, "-8.2"),
            (0.1 + 0.2, "0.3"),
            (0.0, "0"),
        ],
    )
    def test_fmt_exact_never_rounds(self, value, printed):
        assert fmt_exact(value) == printed

    @pytest.mark.parametrize(
        ("label", "values", "unit"),
        [
            ("Total revenues", ["452,800"], UNIT_CURRENCY),
            ("Diluted earnings (loss) per share", ["1.38"], UNIT_PER_SHARE),
            ("Basic EPS", ["1.41"], UNIT_PER_SHARE),
            ("Weighted-average diluted shares", ["24,058"], UNIT_COUNT),
            ("Headcount (period end)", ["1,412"], UNIT_COUNT),
            ("Net debt / EBITDA", ["2.9"], UNIT_RATIO),
            ("EV / EBITDA", ["7.5x"], UNIT_RATIO),
            ("Gross margin %", ["60.0"], UNIT_PERCENT),
            ("Effective tax rate", ["23.0%"], UNIT_PERCENT),
            # Labels that look like counts or percentages and are amounts.
            ("Gross margin", ["271,700"], UNIT_CURRENCY),
            ("Employee benefits expense", ["96,400"], UNIT_CURRENCY),
            ("Customer deposits", ["12,300"], UNIT_CURRENCY),
            ("Repurchase of shares", ["(5,000)"], UNIT_CURRENCY),
        ],
    )
    def test_row_units(self, label, values, unit):
        assert classify_row(label, [parse_cell(v) for v in values]) == unit


# ==============================================================================
# Derived metrics
# ==============================================================================


class TestDerivedMetrics:
    def test_consecutive_years(self):
        metrics = _metrics(MIXED_UNIT_ROWS)
        assert metrics["Revenue YoY FY2023"] == 13.74
        assert metrics["Revenue CAGR"] == 13.74

    def test_newest_first_columns_are_ordered_by_year_not_position(self):
        rows = [
            ["(in thousands)", "2023", "2022", "2021"],
            ["Total revenues", "452,800", "387,100", "341,900"],
            ["Gross profit", "271,700", "228,800", "198,300"],
        ]
        metrics = _metrics(rows)
        assert metrics["Revenue YoY 2023"] == 16.97
        assert metrics["Revenue YoY 2022"] == 13.22
        assert metrics["Revenue CAGR"] == 15.08
        assert metrics["Gross Margin 2023"] == 60.0

    def test_cagr_uses_the_year_span_and_skips_a_gap_and_a_quarter(self):
        rows = [["USD", "FY2019", "FY2023", "FY2023 Q4"], ["Revenue", "100", "200", "60"]]
        reps = _by_representation(rows)
        summary = reps["metrics_summary"][0]
        # 100 -> 200 over four years; it was -22.54% ("FY2019 -> FY2023 Q4, n=2").
        assert summary["metrics"]["Revenue CAGR"]["value"] == 18.92
        assert "n=4" in summary["metrics"]["Revenue CAGR"]["citation_chain"]
        # A four-year change is not "YoY", and a quarter is not a year.
        assert not [name for name in summary["metrics"] if "YoY" in name]
        assert "Q4" not in summary["text"]

    def test_estimate_and_budget_columns_take_no_part_in_growth(self):
        rows = [
            ["USD", "FY2022", "FY2023 Budget", "FY2023E", "FY2024 Forecast"],
            ["Revenue", "100", "150", "160", "200"],
        ]
        assert _metrics(rows) == {}

    def test_change_column_is_not_a_period(self):
        rows = [
            ["USD", "FY2022", "FY2023", "% change"],
            ["Revenue", "100", "120", "20.0%"],
            ["Gross profit", "60", "72", "20.0%"],
        ]
        metrics = _metrics(rows)
        assert metrics["Gross Margin FY2023"] == 60.0
        assert metrics["Revenue YoY FY2023"] == 20.0
        assert not [name for name in metrics if "change" in name.lower()]

    def test_no_growth_from_a_zero_or_negative_base(self):
        rows = [["USD", "FY2021", "FY2022", "FY2023"], ["Revenue", "0", "50", "75"]]
        metrics = _metrics(rows)
        assert "Revenue YoY FY2022" not in metrics
        assert "Revenue CAGR" not in metrics
        assert metrics["Revenue YoY FY2023"] == 50.0

    def test_cost_of_revenue_is_not_revenue(self):
        rows = [
            ["USD", "FY2022", "FY2023"],
            ["Cost of revenue", "100", "120"],
            ["Deferred revenue", "10", "40"],
        ]
        assert _metrics(rows) == {}

    def test_restated_column_supersedes_the_one_first_reported(self):
        rows = [
            ["USD", "FY2022", "FY2022 (Restated)", "FY2023"],
            ["Revenue", "100", "110", "121"],
        ]
        assert _metrics(rows)["Revenue YoY FY2023"] == 10.0

    def test_two_unmarked_columns_for_one_year_are_left_alone(self):
        assert annual_columns(["FY2022", "FY2023", "FY2023 (2)"]) == [(2022, 0)]

    def test_summary_is_marked_as_derived(self):
        summary = _by_representation(MIXED_UNIT_ROWS)["metrics_summary"][0]
        assert summary["content_type"] == "computed_metric"
        assert "not stated in the source" in summary["text"]

    @pytest.mark.parametrize(
        ("column", "key"),
        [
            ("FY2023", "FY2023"),
            ("2023", "FY2023"),
            ("FY23", "FY2023"),
            ("Year ended December 31, 2023", "FY2023"),
            ("FY2023 Budget", "FY2023BUDGET"),
            ("FY2023 Q4", "FY2023Q4"),
            ("LTM Sep-23", "LTMSEP-23"),
            ("Amount", ""),
            ("% change", None),
            ("Variance", None),
        ],
    )
    def test_registry_period_keys(self, column, key):
        assert registry_period(column) == key


# ==============================================================================
# Long tables
# ==============================================================================


class TestLongTables:
    ROWS = [["($ in thousands)", "FY2021", "FY2022", "FY2023", "FY2024E"]] + [
        [f"Line item number {i} of the schedule", f"{1000 + i}.5", f"{2000 + i}.5",
         f"{3000 + i}.5", f"{4000 + i}.5"]
        for i in range(90)
    ]

    def _chunks(self):
        from src.data_processing.semantic_chunker import SemanticChunker
        from src.data_processing.structural_chunker import StructuralChunker

        sections = _table_sections(self.ROWS, "", "Schedule A", 7, "d_t001")
        children, _parents = SemanticChunker().chunk_with_parents(StructuralChunker().chunk(sections))
        grouped: dict[str, list] = {}
        for child in children:
            grouped.setdefault(child.metadata.get("table_representation"), []).append(child)
        return sections, grouped

    def test_every_grid_chunk_carries_heading_unit_and_header(self):
        sections, grouped = self._chunks()
        windows = [s for s in sections if s["table_representation"] == "markdown"]
        assert len(windows) > 1
        # The chunker never has to cut a window, which is what used to leave
        # rows in a chunk with no header above them.
        assert len(grouped["markdown"]) == len(windows)
        assert len(grouped["row_by_row"]) == len(windows)
        for chunk in grouped["markdown"]:
            assert chunk.token_count <= MAX_TOKENS
            assert chunk.text.startswith("Schedule A")
            assert "in USD thousands" in chunk.text
            assert "| FY2021 | FY2022 | FY2023 | FY2024E |" in chunk.text
        for chunk in grouped["row_by_row"]:
            assert chunk.token_count <= MAX_TOKENS
            assert chunk.text.startswith("Schedule A")
            assert "in USD thousands" in chunk.text

    def test_no_row_is_lost_or_repeated(self):
        sections, grouped = self._chunks()
        ranges = sorted(
            tuple(s["table_row_range"]) for s in sections if s["table_representation"] == "markdown"
        )
        assert ranges[0][0] == 1 and ranges[-1][1] == 90
        assert all(prev[1] + 1 == nxt[0] for prev, nxt in zip(ranges, ranges[1:]))
        for i in range(90):
            holders = [c for c in grouped["markdown"] if f"number {i} of the schedule |" in c.text]
            assert len(holders) == 1, f"row {i} is in {len(holders)} grid chunks"
            assert f"| {3000 + i}.5 |" in holders[0].text

    def test_overview_names_every_row_once(self):
        _sections, grouped = self._chunks()
        assert len(grouped["narrative"]) == 1
        overview = grouped["narrative"][0].text
        assert "90 rows" in overview
        assert "Line item number 89 of the schedule" in overview


# ==============================================================================
# The verifier's registry sees converter tables
# ==============================================================================


def _payloads(rows, source, **kwargs):
    return [
        {**section, "source_file": source}
        for section in _table_sections(rows, "", "Income Statement", 3, f"{source}_t000", **kwargs)
    ]


class TestRegistrySeesConverterTables:
    def test_two_documents_disagreeing_are_reported(self):
        other = [list(row) for row in MIXED_UNIT_ROWS]
        other[1] = ["Revenue", "398.1", "448.1"]
        registry = NumericalRegistry.from_chunks(
            _payloads(MIXED_UNIT_ROWS, "cim.pdf") + _payloads(other, "model.xlsx")
        )
        found = registry.find_inconsistencies()
        assert len(found) == 1
        assert found[0]["fiscal_year"] == "FY2023"
        values = found[0]["values_found"]
        assert {v["source"] for v in values} == {"cim.pdf", "model.xlsx"}
        assert sorted(v["value"] for v in values) == pytest.approx([448_100_000, 452_800_000])
        # The reviewer is shown the figures as the documents print them.
        assert {v["as_printed"] for v in values} == {"452.8", "448.1"}

    def test_same_figure_in_thousands_and_in_millions_agrees(self):
        thousands = [
            ["(USD in thousands, except per share data)", "2023", "2022"],
            ["Revenue", "452,800", "398,100"],
            ["Diluted EPS", "2.74", "2.31"],
        ]
        registry = NumericalRegistry.from_chunks(
            _payloads(MIXED_UNIT_ROWS, "cim.pdf") + _payloads(thousands, "10k.pdf")
        )
        assert registry.cross_checked_count() >= 4  # revenue and EPS, two years each
        assert registry.find_inconsistencies() == []

    def test_budget_is_not_reported_as_contradicting_actuals(self):
        budget = [
            ["($ in millions)", "FY2023 Actual", "FY2023 Budget", "Variance"],
            ["Revenue", "452.8", "440.0", "12.8"],
        ]
        registry = NumericalRegistry.from_chunks(
            _payloads(MIXED_UNIT_ROWS, "cim.pdf") + _payloads(budget, "board_pack.xlsx")
        )
        assert registry.cross_checked_count() >= 1  # the actuals are compared
        assert registry.find_inconsistencies() == []

    def test_percentage_row_agrees_with_a_text_table(self):
        text_chunk = {
            "text": "(in millions of USD)\n               FY2023\nGross margin   61.5%\n",
            "source_file": "financials.txt",
        }
        registry = NumericalRegistry.from_chunks(_payloads(MIXED_UNIT_ROWS, "cim.pdf") + [text_chunk])
        assert registry.cross_checked_count() >= 1
        assert registry.find_inconsistencies() == []

    def test_structured_rows_reach_the_payload_of_a_real_workbook(self, tmp_path):
        path = sx.write_stress_xlsx(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        row_by_row = [
            c for c in doc.chunks
            if c.get("sheet_name") == "P&L" and c["table_representation"] == "row_by_row"
        ]
        assert len(row_by_row) == 1
        rows = {r["line_item"]: r for r in row_by_row[0]["structured_rows"]}

        revenue = rows["Total revenues"]["values"]["2023"]
        assert revenue["normalized_value"] == pytest.approx(452_800_000)
        assert revenue["period"] == "FY2023"
        # "(USD in thousands, except per share data)": not thousands of dollars
        # per share, not thousands of employees, not thousands of percent.
        assert rows["Diluted EPS"]["values"]["2023"]["normalized_value"] == 1.38
        assert rows["Headcount (period end)"]["values"]["2023"]["normalized_value"] == 1412
        assert rows["Gross margin %"]["values"]["2022"]["normalized_value"] == pytest.approx(59.11)

        registry = NumericalRegistry.from_chunks(
            [{**c, "source_file": "model.xlsx"} for c in doc.chunks]
        )
        assert len(registry) > 0


# ==============================================================================
# Workbooks
# ==============================================================================


@pytest.fixture(scope="module")
def workbook(tmp_path_factory):
    path = sx.write_stress_xlsx(tmp_path_factory.mktemp("xlsx"))
    return extract_and_chunk(str(path), path.name, DEAL)


def _sheet(doc, name: str) -> dict[str, dict]:
    return {
        c["table_representation"]: c for c in doc.chunks if c.get("sheet_name") == name
    }


class TestWorkbook:
    def test_every_chunk_fits_and_nothing_is_called_unnamed(self, workbook):
        assert workbook.chunks
        for chunk in workbook.chunks:
            assert chunk["token_count"] <= MAX_TOKENS
            assert "Unnamed" not in chunk["text"]

    def test_caption_above_the_header_sets_scale_and_currency(self, workbook):
        pnl = _sheet(workbook, "P&L")
        assert set(pnl) == {"narrative", "row_by_row", "metrics_summary", "markdown"}
        assert pnl["row_by_row"]["scale_label"] == "thousands"
        assert pnl["row_by_row"]["currency"] == "USD"
        assert "(USD in thousands, except per share data)" in pnl["markdown"]["text"]

    def test_numeric_year_headers_and_as_stated_amounts(self, workbook):
        grid = _sheet(workbook, "P&L")["markdown"]["text"]
        assert "| 2021 | 2022 | 2023 |" in grid
        assert "| Total revenues | 341,900 | 387,100 | 452,800 |" in grid
        assert "| Diluted EPS | -0.09 | 0.92 | 1.38 |" in grid
        assert "452,800,000" not in grid and "1,380" not in grid

    def test_percentage_format_is_applied(self, workbook):
        grid = _sheet(workbook, "P&L")["markdown"]["text"]
        # Stored as 0.5911 with a "0.0%" format: 59.11%, exactly.
        assert "| Gross margin % | 58% | 59.11% | 60% |" in grid
        assert "0.5911" not in grid

    def test_merged_two_row_header_names_every_column(self, workbook):
        grid = _sheet(workbook, "Budget vs Actual")["markdown"]["text"]
        assert "| FY2023 Actual | FY2023 Budget | FY2024 Actual | FY2024 Budget |" in grid
        assert "| Revenue | 452.8 | 440 | 498.1 | 505 |" in grid

    def test_uncalculated_formulas_are_flagged_not_read_as_empty(self, workbook):
        formulas = _sheet(workbook, "Formulas")
        assert "| Gross Profit | [formula not calculated] | [formula not calculated] |" in (
            formulas["markdown"]["text"]
        )
        gross = next(
            r for r in formulas["row_by_row"]["structured_rows"] if r["line_item"] == "Gross Profit"
        )
        assert all("normalized_value" not in cell for cell in gross["values"].values())
        assert any("formula" in w and "Formulas" in w for w in workbook.warnings)

    def test_hidden_sheet_is_indexed_and_marked(self, workbook):
        hidden = _sheet(workbook, "Hidden Adjustments")
        assert "4,200" in hidden["markdown"]["text"]
        assert all(c["sheet_state"] == "hidden" for c in hidden.values())
        assert any("hidden" in w and "Hidden Adjustments" in w for w in workbook.warnings)
        visible = _sheet(workbook, "P&L")
        assert all("sheet_state" not in c for c in visible.values())

    def test_text_sheet_is_verbatim_and_empty_sheet_is_silent(self, workbook):
        notes = _sheet(workbook, "Assumptions")
        assert set(notes) == {"verbatim"}
        assert notes["verbatim"]["content_type"] == "table_text"
        assert "twenty-one percent" in notes["verbatim"]["text"]
        assert not _sheet(workbook, "Empty")
        assert not any("Empty" in w for w in workbook.warnings)


class TestWorkbookLayouts:
    """Layouts built per test: stacked tables, an offset block, a merged amount."""

    @staticmethod
    def _ingest(tmp_path, build):
        from openpyxl import Workbook

        wb = Workbook()
        build(wb.active)
        path = tmp_path / "layout.xlsx"
        wb.save(str(path))
        return extract_and_chunk(str(path), path.name, DEAL)

    def test_two_tables_stacked_on_one_sheet_are_two_tables(self, tmp_path):
        def build(ws):
            ws.title = "Summary"
            ws.append(["USD in millions"])
            ws.append(["Income statement", "FY2022", "FY2023"])
            ws.append(["Revenue", 398.1, 452.8])
            ws.append(["EBITDA", 66.0, 77.0])
            ws.append([None])
            ws.append(["EUR in thousands"])
            ws.append(["Balance sheet", "FY2022", "FY2023"])
            ws.append(["Cash", 41200, 53900])

        doc = self._ingest(tmp_path, build)
        grids = [c for c in doc.chunks if c.get("table_representation") == "markdown"]
        assert len(grids) == 2
        assert len({c["table_id"] for c in grids}) == 2
        income = next(c for c in grids if "Revenue" in c["text"])
        balance = next(c for c in grids if "Cash" in c["text"])
        # Each table keeps its own unit; the second caption is not a data row
        # of the first table.
        assert (income["currency"], income["scale_label"]) == ("USD", "millions")
        assert (balance["currency"], balance["scale_label"]) == ("EUR", "thousands")
        assert "Cash" not in income["text"] and "Revenue" not in balance["text"]
        assert "| Cash | 41,200 | 53,900 |" in balance["text"]

    def test_blank_spacer_rows_inside_a_statement_do_not_split_it(self, tmp_path):
        def build(ws):
            ws.append(["Line Item", "FY2022", "FY2023"])
            ws.append(["Revenue", 398.1, 452.8])
            ws.append([None])
            ws.append(["Operating expenses", 300.0, 340.0])
            ws.append([None])
            ws.append(["Operating income", 98.1, 112.8])

        doc = self._ingest(tmp_path, build)
        grids = [c for c in doc.chunks if c.get("table_representation") == "markdown"]
        assert len(grids) == 1
        for label in ("Revenue", "Operating expenses", "Operating income"):
            assert f"| {label} |" in grids[0]["text"]

    def test_table_that_does_not_start_at_a1(self, tmp_path):
        def build(ws):
            ws["C4"] = "Line Item"
            ws["D4"] = "FY2022"
            ws["E4"] = "FY2023"
            ws["C5"] = "Revenue"
            ws["D5"] = 398.1
            ws["E5"] = 452.8
            ws["C6"] = "Gross Profit"
            ws["D6"] = "=D5*0.6"   # never calculated
            ws["E6"] = 271.7

        doc = self._ingest(tmp_path, build)
        grid = next(c for c in doc.chunks if c.get("table_representation") == "markdown")["text"]
        assert "| Line Item | FY2022 | FY2023 |" in grid
        assert "| Revenue | 398.1 | 452.8 |" in grid
        # The formula cell is found at its own position, not at an A1-relative one.
        assert "| Gross Profit | [formula not calculated] | 271.7 |" in grid
        assert any("1 formula cell" in w for w in doc.warnings)

    def test_merged_amount_is_not_repeated_across_periods(self, tmp_path):
        def build(ws):
            ws.append(["Line Item", "FY2022", "FY2023"])
            ws.append(["Revenue", 398.1, 452.8])
            ws.append(["Goodwill impairment", 12.5, None])
            ws.merge_cells("B3:C3")

        doc = self._ingest(tmp_path, build)
        grid = next(c for c in doc.chunks if c.get("table_representation") == "markdown")["text"]
        assert "| Goodwill impairment | 12.5 | - |" in grid


# ==============================================================================
# Scale and currency detection
# ==============================================================================


class TestScaleAndCurrencyDetection:
    @pytest.mark.parametrize(
        ("header", "factor"),
        [
            ("(in thousands, except per share data)", 1_000.0),
            ("($000s)", 1_000.0),
            ("USD thousands", 1_000.0),
            ("$ in millions", 1_000_000.0),
            ("USD mm", 1_000_000.0),
            ("EUR mn", 1_000_000.0),
            ("(€ millions)", 1_000_000.0),
            ("GBP bn", 1_000_000_000.0),
            # The first scale stated is the table's (Apple's 10-K caption).
            ("(In millions, except number of shares, which are reflected in thousands, "
             "and per-share amounts)", 1_000_000.0),
            # No scale at all.
            ("Telecomm revenue", 1.0),
            ("Summary of programmes", 1.0),
            ("Revenue FY2023", 1.0),
        ],
    )
    def test_scale(self, header, factor):
        assert ExcelNormalizer().detect_scale([header]).scale_factor == factor

    @pytest.mark.parametrize(
        ("header", "currency"),
        [
            ("USD in thousands", "USD"),
            ("C$ in millions", "CAD"),
            ("(€ millions)", "EUR"),
            ("GBP bn", "GBP"),
            # Substrings of ordinary words are not currency codes.
            ("Audited results", "UNKNOWN"),
            ("Europe revenue", "UNKNOWN"),
            ("Cadence of reporting", "UNKNOWN"),
            ("Line Item", "UNKNOWN"),
        ],
    )
    def test_currency(self, header, currency):
        assert ExcelNormalizer().detect_scale([header]).currency == currency


# ==============================================================================
# PDF table fragments
# ==============================================================================


class TestHeaderRowDetection:
    @pytest.mark.parametrize(
        "row",
        [
            ["Segment", "FY2022", "FY2023"],
            ["", "2023", "2022", "2021"],
            ["Line Item", "Year Ended December 31, 2023", "Year Ended December 31, 2022"],
            ["", "Actual", "Budget"],
        ],
    )
    def test_headers(self, row):
        assert MultiPageTableStitcher()._is_header_row(row)

    @pytest.mark.parametrize(
        "row",
        [
            ["Other", "(2.6)", "(1.9)"],          # parenthesised negatives
            ["Marketplace fees", "8.4", "11.3"],
            ["Gross margin", "60.0%", "61.5%"],
            ["Total revenues", "$ 452,800", "$ 387,100"],
            ["Headcount", "1,412", "1,304"],
            ["", "", ""],
        ],
    )
    def test_data_rows(self, row):
        assert not MultiPageTableStitcher()._is_header_row(row)
