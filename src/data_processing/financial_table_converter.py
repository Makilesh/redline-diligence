# src/data_processing/financial_table_converter.py
"""
Generates the chunk representations of a financial table, all sharing a table_id.

Two rules, both absolute:

1. A figure is never rewritten. Every representation prints each cell exactly
   as the source states it ("452.8", "(181,100)", "60.0%") and states the
   table's unit once. The converter used to multiply every plain number by the
   table's scale, so under "$ in millions, except per share" an EPS of 2.74 was
   indexed as "2,740,000" and a headcount of 1,250 as "1,250,000,000" — wrong
   figures manufactured by ingestion, which numeric grounding then accepted as
   source text. The normalised value now lives only in `structured_rows`, and
   only currency rows are scaled there.

2. Derived metrics (margins, CAGR, YoY growth) are computed deterministically,
   never by an LLM, and only when the arithmetic is defined: growth between
   fiscal years that are actually consecutive, CAGR over the real year span,
   margins only in columns that hold a period. When a period cannot be parsed
   or a base is zero or negative, nothing is emitted — a missing metric costs a
   reviewer a moment, a wrong one misprices a deal.

Every computed value carries a citation_chain showing its inputs, and
content_type="computed_metric" marks it as derived rather than verbatim.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from src.data_processing.excel_normalizer import TableNormalizationMeta
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

RepresentationType = Literal["narrative", "row_by_row", "metrics_summary", "markdown"]

# Row units. Only "currency" rows take the table's scale.
UNIT_CURRENCY = "currency"
UNIT_PERCENT = "percent"
UNIT_PER_SHARE = "per_share"
UNIT_RATIO = "ratio"
UNIT_COUNT = "count"
UNIT_TEXT = "text"

# Token budget for the rows of one table window. With the heading, the unit
# note and the header row on top, a window stays under the 800-token chunk cap,
# so the semantic chunker never has to cut a table — which is the only way a
# row ends up in a chunk without the header that says what its columns are.
TABLE_WINDOW_TOKENS = 560

_CURRENCY_SIGNS = "$€£¥₹"
_MINUS_SIGNS = "-−–"
_EMPTY_CELLS = {"", "nan", "none", "-", "—", "–", "−", "n/a", "na", "n.a.", "nm", "n.m.", "n/m"}

# A plain amount: optional sign, currency sign and parentheses around digits
# that are either ungrouped or grouped correctly ("1,234,567", Indian
# "12,34,567"). "1,5" is NOT a number here: read as English it would become 15,
# read as German it is 1.5, and guessing between them invents a figure — the
# cell stays text and is printed as stated.
_NUMBER_BODY = r"(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3}|\d+)?(?:\.\d+)?"
_PLAIN_AMOUNT = re.compile(
    rf"^(?P<lead>[{_MINUS_SIGNS}]?)\s*[{_CURRENCY_SIGNS}]?\s*(?P<open>\(?)\s*[{_CURRENCY_SIGNS}]?\s*"
    rf"(?P<inner>[{_MINUS_SIGNS}]?)\s*(?P<num>{_NUMBER_BODY})\s*(?P<close>\)?)$"
)
_PERCENT_CELL = re.compile(
    rf"^(?P<open>\(?)\s*(?P<sign>[{_MINUS_SIGNS}+]?)\s*(?P<num>{_NUMBER_BODY})\s*%\s*(?P<close>\)?)$"
)
_MULTIPLE_CELL = re.compile(rf"^(?P<num>{_NUMBER_BODY})\s*[xX×]$")

# Row labels whose values are not currency amounts, so the table's "in
# millions" does not apply to them.
_PER_SHARE_LABEL = re.compile(
    r"\bper\s+(?:\w+\s+){0,2}(?:share|unit|ads|adr)s?\b|\beps\b|\bdps\b", re.IGNORECASE
)
_PERCENT_LABEL = re.compile(
    r"%|\bpercent(?:age)?\b|\bmargin\b|\bgrowth\b|\brate\b|\byield\b|\bas a share of\b",
    re.IGNORECASE,
)
_RATIO_LABEL = re.compile(
    r"\bratio\b|\bmultiple\b|\bleverage\b|\bcoverage\b|\bturns?\b|\bev\s*/|/\s*ebitda\b"
    r"|\bdays\b|\bdso\b|\bdpo\b|\(x\)",
    re.IGNORECASE,
)
# Counts are matched narrowly: "Employee benefits expense", "Customer deposits"
# and "Repurchase of shares" are amounts.
_COUNT_LABEL = re.compile(
    r"\bhead\s?count\b|\bftes?\b|\bfull[- ]time equivalents?\b|\bnumber of\b|#\s*of\b"
    r"|\bno\. of\b|^\s*(?:total\s+|average\s+)?employees\b"
    r"|^\s*(?:total\s+|active\s+|paying\s+|new\s+|enterprise\s+)?customers\b"
    r"|\b(?:active|monthly|daily|paying|registered)\s+users\b|\bsubscribers\b"
    r"|\bshares outstanding\b|\bweighted[- ]average\b.*\bshares\b"
    r"|\b(?:basic|diluted)\s+shares\b|\bshares used\b|\bstore count\b"
    r"|\bunits (?:sold|shipped)\b",
    re.IGNORECASE,
)

# Footnote markers and annotations dropped before matching a row label.
_LABEL_NOISE = re.compile(r"\(\s*(?:\d{1,2}|[a-z])\s*\)|\[\s*\w{1,2}\s*\]|[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§]")

# Row-label aliases, most specific first. Matching is on the whole normalised
# label, never a substring: "Cost of revenue" must not be taken for revenue.
_REVENUE_LABELS = (
    "total revenues", "total revenue", "total net revenues", "total net revenue",
    "net revenues", "net revenue", "revenues", "revenue", "revenues net", "revenue net",
    "total net sales", "net sales", "total sales", "sales", "turnover",
)
_MARGIN_NUMERATORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Gross Margin", ("gross profit", "gross income")),
    ("Operating Margin", (
        "operating income", "income from operations", "operating profit", "ebit",
    )),
    ("Net Margin", ("net income", "net profit", "net earnings", "profit for the year")),
    ("EBITDA Margin", ("ebitda",)),
    ("Adjusted EBITDA Margin", ("adjusted ebitda",)),
)

# Columns that hold something derived from the periods, not a period.
_DERIVED_COLUMN = re.compile(
    r"%|\bchange\b|\bchg\b|\bvariance\b|\bvar\b|\bgrowth\b|\byoy\b|\by/y\b|\bcagr\b|\bdelta\b"
    r"|Δ|\bvs\b|\bversus\b|\bdiff(?:erence)?\b|\bbps\b|\bnotes?\b|\bref\b|\bcomment",
    re.IGNORECASE,
)
_SUB_ANNUAL = re.compile(
    r"\bq[1-4]\b|\b[1-4]q\b|\bh[12]\b|\b[12]h\b|\bquarter|\bhalf\b|\bmonths?\b|\bweeks?\b",
    re.IGNORECASE,
)
# A rolling window is twelve months long without being a fiscal year.
_ROLLING = re.compile(
    r"\bltm\b|\bttm\b|\bntm\b|\bytd\b|\bmtd\b|\bqtd\b|\blast\b|\btrailing\b|\bnext\b"
    r"|\brun[- ]rate\b",
    re.IGNORECASE,
)
_FULL_YEAR = re.compile(
    r"\b(?:twelve|12)\s+months\b|\byears?\s+end(?:ed|ing)\b|\bfiscal\s+year\b|\bfull\s+year\b",
    re.IGNORECASE,
)
_NOT_ACTUAL = re.compile(
    r"\bbudget|\bforecast|\bplan\b|\bproject(?:ed|ion)|\bestimate|\bguidance\b|\boutlook\b"
    r"|\btarget\b|\bpro[\s-]?forma\b|\bpf\b|\bcase\b|\bscenario\b",
    re.IGNORECASE,
)
_RESTATED = re.compile(r"\brestated\b|\brevised\b|\brecast\b", re.IGNORECASE)
_YEAR_4 = re.compile(r"(?<!\d)((?:19|20)\d{2})(?P<suffix>[AaEeFfPp])?(?![\d.,]\d)(?!\d)")
_YEAR_2 = re.compile(r"\b(?:FY|CY)\s?'?(\d{2})(?P<suffix>[AaEeFfPp])?\b", re.IGNORECASE)


# ==============================================================================
# Table model
# ==============================================================================


@dataclass
class TableCell:
    """One cell: the text exactly as the source prints it, and its number."""
    printed: str
    value: float | None = None
    kind: str = UNIT_TEXT  # "number" | "percent" | "multiple" | "text"


@dataclass
class TableRow:
    """A labelled row and the unit its values are in."""
    label: str
    cells: list[TableCell]
    unit: str = UNIT_CURRENCY


@dataclass
class TableData:
    """A table as the converter reads it: period columns and labelled rows."""
    columns: list[str]
    rows: list[TableRow]
    label_header: str = "Line Item"
    notes: list[str] = field(default_factory=list)  # footnotes kept with the grid

    @property
    def has_numbers(self) -> bool:
        return any(c.value is not None for r in self.rows for c in r.cells)


@dataclass(frozen=True)
class ColumnPeriod:
    """What a column header says about the period it holds."""
    year: int | None
    annual: bool
    actual: bool
    restated: bool


def fmt_exact(value: float) -> str:
    """
    Formats a number with thousands separators and no rounding at all.

    `:,.0f` turned 387.1 into "387", `:,.2f` turned a 0.0375 dividend into
    "0.04", and four fixed decimals turned 1.23456 into "1.2346" — each a figure
    the source never stated. 15 significant digits is everything a double
    holds, so this drops only binary noise (0.1 + 0.2 prints as 0.3).

    Args:
        value: Numeric value.

    Returns:
        E.g. "452,800,000", "387.1", "-8.2", "0.0375".
    """
    if value is None or isinstance(value, bool):
        return ""
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return str(value)
    text = format(Decimal(format(number, ".15g")), ",f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


# Kept under its old name for callers that format a single amount.
_fmt_number = fmt_exact


def representation_content_type(representation: str) -> str:
    """
    Maps a table representation to the payload content_type.

    metrics_summary maps to "computed_metric" because the synthesizer and the
    financial verifier key on that value to label derived (non-verbatim)
    numbers; the others are "table_<representation>".

    Args:
        representation: narrative | row_by_row | metrics_summary | markdown.

    Returns:
        content_type string for the Qdrant payload.
    """
    if representation == "metrics_summary":
        return "computed_metric"
    return f"table_{representation}"


def _number(text: str) -> float | None:
    """float() of a digit string with thousands separators, None if empty."""
    cleaned = text.replace(",", "")
    if not cleaned or cleaned == ".":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_cell(value: Any) -> TableCell:
    """
    Reads one raw cell into its printed text and, when it is one, its number.

    Parentheses and a leading minus both mean negative. An unbalanced
    parenthesis ("(1,234" — PDF extraction sometimes leaves the ")" in the
    next cell) is NOT read as a number: the sign cannot be established, so the
    cell is printed as stated and takes no part in any arithmetic.

    Args:
        value: Raw cell value (str, int, float, None).

    Returns:
        TableCell. `value` is None for text and empty cells.
    """
    if value is None:
        return TableCell("")
    if isinstance(value, bool):
        return TableCell(str(value))
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number):
            return TableCell("")
        return TableCell(fmt_exact(number), number, "number")

    text = " ".join(str(value).split())
    if text.lower() in _EMPTY_CELLS:
        # "—" is how a statement prints "nil"; keep it visible in the grid.
        return TableCell(text if text not in ("nan", "None", "none") else "")

    match = _PLAIN_AMOUNT.match(text)
    if match and match.group("num"):
        number = _number(match.group("num"))
        balanced = bool(match.group("open")) == bool(match.group("close"))
        if number is not None and balanced:
            negative = bool(match.group("open")) or bool(match.group("lead")) \
                or bool(match.group("inner"))
            return TableCell(text, -number if negative else number, "number")
        return TableCell(text)

    match = _PERCENT_CELL.match(text)
    if match and match.group("num"):
        number = _number(match.group("num"))
        balanced = bool(match.group("open")) == bool(match.group("close"))
        if number is not None and balanced:
            negative = bool(match.group("open")) or match.group("sign") in tuple(_MINUS_SIGNS)
            return TableCell(text, -number if negative else number, "percent")
        return TableCell(text)

    match = _MULTIPLE_CELL.match(text)
    if match and match.group("num"):
        number = _number(match.group("num"))
        if number is not None:
            return TableCell(text, number, "multiple")

    return TableCell(text)


def normalize_row_label(label: str) -> str:
    """
    Canonical form of a row label for alias matching.

    "Net income (loss) (1)" -> "net income"; "Total revenues:" -> "total revenues".

    Args:
        label: Row label as printed.

    Returns:
        Lower-case label without footnote markers, "(loss)" and punctuation.
    """
    text = _LABEL_NOISE.sub(" ", str(label).replace("_", " "))
    text = re.sub(r"\(\s*loss(?:es)?\s*\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9&/% ]", " ", text.lower())
    return " ".join(text.split())


def _dedupe(labels: list[str]) -> list[str]:
    """Suffixes repeated labels so every row and column stays addressable."""
    seen: dict[str, int] = {}
    result = []
    for label in labels:
        count = seen.get(label, 0)
        seen[label] = count + 1
        result.append(label if count == 0 else f"{label} ({count + 1})")
    return result


def classify_row(label: str, cells: list[TableCell]) -> str:
    """
    Decides what unit a row is in, i.e. whether the table's scale applies.

    The cells decide first (a "%" or an "x" in the cell is unambiguous), then
    the label. Anything not recognised as a percentage, per-share figure,
    ratio or count is a currency amount.

    Args:
        label: Row label.
        cells: Parsed cells of the row.

    Returns:
        One of the UNIT_* constants.
    """
    numeric = [c for c in cells if c.value is not None]
    if not numeric:
        return UNIT_TEXT
    kinds = {c.kind for c in numeric}
    if kinds == {"percent"}:
        return UNIT_PERCENT
    if kinds == {"multiple"}:
        return UNIT_RATIO
    if _PER_SHARE_LABEL.search(label):
        return UNIT_PER_SHARE
    if _RATIO_LABEL.search(label):
        return UNIT_RATIO
    if _COUNT_LABEL.search(label):
        return UNIT_COUNT
    if _PERCENT_LABEL.search(label):
        # "Gross margin" labels a percentage in one statement and the gross
        # profit amount in another. Without a "%" anywhere, only values that
        # can be percentages are treated as such.
        if "%" in label or all(abs(c.value) <= 100 for c in numeric):
            return UNIT_PERCENT
    return UNIT_CURRENCY


def table_from_rows(header: list[Any], rows: list[list[Any]]) -> TableData | None:
    """
    Builds a TableData from a header row and data rows.

    Returns None when the rows cannot be read as a labelled numeric table
    (fewer than two columns, no data rows, or no number at all) — the caller
    then indexes the table verbatim instead of fabricating representations.

    Args:
        header: Header cells; the first cell heads the row-label column.
        rows: Data rows.

    Returns:
        TableData, or None.
    """
    if not header or len(header) < 2 or not rows:
        return None

    width = len(header)
    columns = []
    for j, cell in enumerate(header[1:], start=2):
        name = "" if cell is None else " ".join(str(cell).split())
        if isinstance(cell, float) and cell.is_integer():
            name = str(int(cell))  # a year read from a numeric cell: 2023.0
        if not name or name.lower() == "nan" or name.startswith("Unnamed:"):
            name = f"Column {j}"
        columns.append(name)
    columns = _dedupe(columns)

    labels: list[str] = []
    parsed: list[list[TableCell]] = []
    for i, row in enumerate(rows, start=1):
        raw = list(row)[:width] + [None] * max(0, width - len(row))
        cells = [parse_cell(c) for c in raw[1:]]
        label_cell = raw[0]
        label = "" if label_cell is None or (
            isinstance(label_cell, float) and math.isnan(label_cell)
        ) else " ".join(str(label_cell).split())
        if not label and all(not c.printed for c in cells):
            continue
        labels.append(label or f"Row {i}")
        parsed.append(cells)

    if not parsed:
        return None
    labels = _dedupe(labels)
    table = TableData(
        columns=columns,
        rows=[TableRow(label, cells, classify_row(label, cells))
              for label, cells in zip(labels, parsed)],
        label_header=" ".join(str(header[0]).split()) if header[0] not in (None, "") and not (
            isinstance(header[0], float) and math.isnan(header[0])) else "Line Item",
    )
    return table if table.has_numbers else None


def table_from_frame(df) -> TableData:
    """
    Builds a TableData from a DataFrame (row labels as the index, periods as
    columns) — the shape callers passed before TableData existed.

    Args:
        df: pandas DataFrame.

    Returns:
        TableData. Uses the TableData attached by frame_from_rows when present,
        so the as-printed text survives the round trip.
    """
    attached = getattr(df, "attrs", {}).get("table")
    if isinstance(attached, TableData):
        return attached
    columns = [str(c) for c in df.columns]
    rows = []
    for label in df.index:
        values = df.loc[label]
        cells = [parse_cell(values.iloc[j]) for j in range(len(columns))]
        rows.append(TableRow(str(label), cells, classify_row(str(label), cells)))
    return TableData(columns=columns, rows=rows)


def frame_from_rows(header: list[Any], rows: list[list[Any]]):
    """
    DataFrame view of table_from_rows: row labels as the index, period labels
    as columns, floats for numbers and the printed text otherwise.

    Args:
        header: Header cells; the first cell heads the row-label column.
        rows: Data rows.

    Returns:
        DataFrame (object dtype) with the TableData in `.attrs["table"]`, or
        None when the rows are not a labelled numeric table.
    """
    table = table_from_rows(header, rows)
    if table is None:
        return None
    import pandas as pd

    data = [
        [c.value if c.kind == "number" else (c.printed or None) for c in row.cells]
        for row in table.rows
    ]
    df = pd.DataFrame(data, index=[r.label for r in table.rows], columns=table.columns,
                      dtype=object)
    df.attrs["table"] = table
    return df


def parse_column_period(label: str) -> ColumnPeriod | None:
    """
    Reads a column header as a period.

    Args:
        label: Column header ("FY2023", "2023", "FY23E", "FY2022 (Restated)",
               "Year ended December 31, 2023", "Q4 2023", "% change").

    Returns:
        ColumnPeriod, or None when the column holds something derived from the
        periods (a change, a variance, a note) rather than a period.
    """
    text = str(label)
    if _DERIVED_COLUMN.search(text):
        return None
    years = {int(m.group(1)) for m in _YEAR_4.finditer(text)}
    suffixes = {(m.group("suffix") or "").upper() for m in _YEAR_4.finditer(text)}
    if not years:
        short = list(_YEAR_2.finditer(text))
        years = {2000 + int(m.group(1)) for m in short}
        suffixes = {(m.group("suffix") or "").upper() for m in short}
    estimate = bool(suffixes & {"E", "F", "P"}) or bool(_NOT_ACTUAL.search(text))
    return ColumnPeriod(
        year=next(iter(years)) if len(years) == 1 else None,
        annual=not _ROLLING.search(text)
        and bool(_FULL_YEAR.search(text) or not _SUB_ANNUAL.search(text)),
        actual=not estimate,
        restated=bool(_RESTATED.search(text)),
    )


def annual_columns(columns: list[str]) -> list[tuple[int, int]]:
    """
    The columns that hold one full fiscal year of actuals, as (year, index),
    oldest first, at most one column per year.

    When a year appears twice the restated column is used — a restatement
    supersedes the figure first reported. If neither or both are restated the
    year is ambiguous (actual vs budget with no marker, two entities) and is
    left out: growth computed against the wrong one of two columns is a wrong
    number, and there is no way to choose.

    Args:
        columns: Column headers in table order.

    Returns:
        Sorted list of (fiscal year, column index).
    """
    by_year: dict[int, list[tuple[int, ColumnPeriod]]] = {}
    for index, label in enumerate(columns):
        period = parse_column_period(label)
        if period and period.year and period.annual and period.actual:
            by_year.setdefault(period.year, []).append((index, period))

    chosen: list[tuple[int, int]] = []
    for year, candidates in by_year.items():
        if len(candidates) == 1:
            chosen.append((year, candidates[0][0]))
            continue
        restated = [index for index, period in candidates if period.restated]
        if len(restated) == 1:
            chosen.append((year, restated[0]))
    return sorted(chosen)


def _find_row(table: TableData, aliases: tuple[str, ...]) -> TableRow | None:
    """First row whose normalised label is one of the aliases, in alias order."""
    by_label: dict[str, TableRow] = {}
    for row in table.rows:
        by_label.setdefault(normalize_row_label(row.label), row)
    for alias in aliases:
        row = by_label.get(alias)
        if row is not None and row.unit == UNIT_CURRENCY:
            return row
    return None


def _unit_note(table: TableData, meta: TableNormalizationMeta) -> str:
    """
    One sentence stating the unit of the as-printed values.

    Args:
        table: Table (to see which kinds of row it has).
        meta: Detected currency and scale.

    Returns:
        E.g. "Values are as stated in the source, in USD millions; per-share,
        percentage and count rows are not in millions."
    """
    currency = "" if meta.currency == "UNKNOWN" else meta.currency
    if meta.scale_factor == 1.0:
        unit = f"in {currency}" if currency else ""
    else:
        unit = f"in {currency} {meta.scale_label}".replace("  ", " ")
    if not unit:
        return "Values are as stated in the source; the table gives no currency or scale."

    names = {
        UNIT_PER_SHARE: "per-share",
        UNIT_PERCENT: "percentage",
        UNIT_RATIO: "ratio",
        UNIT_COUNT: "count",
    }
    present = [names[u] for u in names if any(r.unit == u for r in table.rows)]
    note = f"Values are as stated in the source, {unit}"
    if present and meta.scale_factor != 1.0:
        listed = present[0] if len(present) == 1 else ", ".join(present[:-1]) + " and " + present[-1]
        note += f"; {listed} rows are not in {meta.scale_label}"
    return note + "."


def split_row_windows(table: TableData, budget: int = TABLE_WINDOW_TOKENS) -> list[tuple[int, int]]:
    """
    Cuts a table's rows into consecutive windows that each fit one chunk.

    Args:
        table: Table to split.
        budget: Token budget for the rows of one window.

    Returns:
        List of (start, end) row index pairs, end exclusive. One pair for a
        table that fits.
    """
    from src.utils.token_counter import count_tokens

    # A window is rendered twice, as row_by_row lines and as grid rows; it has
    # to fit in whichever is longer.
    costs = [
        max(count_tokens(_row_by_row_line(table, row)), count_tokens(_markdown_row(row))) + 1
        for row in table.rows
    ]
    if sum(costs) <= budget:
        return [(0, len(table.rows))]
    windows: list[tuple[int, int]] = []
    start, used = 0, 0
    for index, cost in enumerate(costs):
        if index > start and used + cost > budget:
            windows.append((start, index))
            start, used = index, 0
        used += cost
    windows.append((start, len(table.rows)))
    return windows


def _narrative_line(table: TableData, row: TableRow) -> str:
    parts = [f"{col}: {cell.printed}" for col, cell in zip(table.columns, row.cells) if cell.printed]
    return f"{row.label}: {'; '.join(parts)}" if parts else ""


def _row_by_row_line(table: TableData, row: TableRow) -> str:
    parts = [f"{col}={cell.printed}" for col, cell in zip(table.columns, row.cells) if cell.printed]
    return " ".join([f"{row.label}:"] + parts)


def _markdown_cell(text: str) -> str:
    return (text or "-").replace("|", "\\|")


def _markdown_row(row: TableRow) -> str:
    cells = [_markdown_cell(row.label)] + [_markdown_cell(c.printed) for c in row.cells]
    return "| " + " | ".join(cells) + " |"


def structured_rows(table: TableData, meta: TableNormalizationMeta, start: int = 0,
                    end: int | None = None) -> list[dict]:
    """
    Machine-readable rows for the Financial Verifier's figure registry.

    Shape, per row:
        {"line_item": "Total revenues", "unit": "currency",
         "values": {"FY2023": {"raw": "452,800", "raw_value": 452800.0,
                               "normalized_value": 452800000.0,
                               "scale_factor": 1000.0, "currency": "USD",
                               "period": "FY2023", "kind": "currency",
                               "decimals": 0}},
         "columns": ["FY2023"]}

    `raw` is the cell as printed; `raw_value` its number as stated;
    `normalized_value` = raw_value x scale_factor. The table's scale is applied
    to currency rows only — for percent, per_share, ratio and count rows
    scale_factor is 1.0 and normalized_value equals raw_value, so "in millions,
    except per share" can never turn an EPS into millions. Text cells carry
    only `raw`.

    Args:
        table: Table.
        meta: Detected currency and scale.
        start: First row index.
        end: One past the last row index (default: all rows).

    Returns:
        List of row dicts.
    """
    periods = [registry_period(col) for col in table.columns]
    result = []
    for row in table.rows[start:end]:
        scale = meta.scale_factor if row.unit == UNIT_CURRENCY else 1.0
        values: dict[str, dict] = {}
        for col, period, cell in zip(table.columns, periods, row.cells):
            if not cell.printed:
                continue
            if cell.value is None or period is None:
                # Text, or a "% change" / "Variance" column: its number is not
                # the line item's value for any period, so it is kept as
                # printed and never offered for cross-source comparison.
                values[col] = {"raw": cell.printed}
                continue
            fraction = re.search(r"\.(\d+)", cell.printed)
            values[col] = {
                "raw": cell.printed,
                "raw_value": cell.value,
                "normalized_value": cell.value * scale,
                "scale_factor": scale,
                "currency": meta.currency,
                "period": period,
                # What the cell itself prints ("61.5%", "7.5x"), and to how
                # many decimals — the registry's rounding tolerance.
                "kind": cell.kind if cell.kind in ("percent", "multiple") else "currency",
                "decimals": len(fraction.group(1)) if fraction else 0,
            }
        # `columns` keeps the left-to-right order: a JSON object's key order does not
        # survive a Qdrant server reached over gRPC, a list does (Decision 105;
        # numerical_registry.row_cells).
        result.append({"line_item": row.label, "unit": row.unit, "values": values, "columns": list(values)})
    return result


def registry_period(column: str) -> str | None:
    """
    The key under which a column's figures may be compared across documents.

    A full fiscal year of actuals is "FY2023", whatever the header calls it
    ("2023", "Year ended December 31, 2023", "FY23"). Every other column keeps
    its own label ("FY2023BUDGET", "FY2023Q4", "LTMSEP-23"), so a budget is
    compared with another document's budget and never reported as
    contradicting the actuals.

    Args:
        column: Column header.

    Returns:
        Period key; "" for a column that names no period ("Amount", "Total"),
        whose figures are registered but have nothing to be compared on; None
        for a column derived from the periods ("% change", "Variance").
    """
    period = parse_column_period(column)
    if period is None:
        return None
    if period.year and period.annual and period.actual:
        return f"FY{period.year}"
    text = str(column)
    if not (_YEAR_4.search(text) or _YEAR_2.search(text) or _ROLLING.search(text)
            or _SUB_ANNUAL.search(text)):
        return ""
    return re.sub(r"[\s']", "", text.upper())


# ==============================================================================
# Converter
# ==============================================================================


class FinancialTableConverter:
    """
    Generates the representations of a financial table, all sharing a table_id.

    Representations:
        1. narrative — one line per row in prose-like form, for dense matching
        2. row_by_row — "label: period=value" lines for exact lookup; carries
           `structured_rows` for the Financial Verifier
        3. metrics_summary — derived metrics (margins, CAGR, YoY), computed here
        4. markdown — the grid, cell for cell as printed

    A table too long for one chunk is cut into row windows; each window gets
    its own row_by_row and markdown chunk with the header repeated, and the
    narrative becomes a single overview of the table's rows and columns.
    """

    def generate_all_representations(
        self,
        df,
        meta: TableNormalizationMeta,
        table_id: str,
        source_metadata: dict,
    ) -> list[dict]:
        """
        Generate the representations of one table.

        Args:
            df: TableData, or a DataFrame with row labels as the index and
                period labels as columns.
            meta: Normalization metadata (currency, scale_factor, scale_label).
            table_id: Identifier shared by every representation.
            source_metadata: Extra keys copied onto every representation.

        Returns:
            Representation dicts in the order narrative, row_by_row,
            metrics_summary, markdown (row_by_row and markdown once per row
            window for a long table). Each has text, table_representation,
            table_id, table_row_range ([first, last], 1-based) and
            table_row_count.
        """
        table = df if isinstance(df, TableData) else table_from_frame(df)
        windows = split_row_windows(table)
        base = {
            "table_id": table_id,
            "is_table": 1,
            "content_type": "clean",
            "currency": meta.currency,
            "scale_factor": meta.scale_factor,
            "scale_label": meta.scale_label,
            "table_row_count": len(table.rows),
            **source_metadata,
        }
        note = _unit_note(table, meta)

        representations = [self._narrative(table, note, windows, base)]
        row_by_row = [self._row_by_row(table, meta, note, w, windows, base) for w in windows]
        markdown = [self._markdown(table, note, w, windows, base) for w in windows]
        representations += row_by_row
        representations.append(self._metrics_summary(table, base))
        representations += markdown

        logger.info(
            "Table representations generated",
            extra={
                "table_id": table_id,
                "shape": f"{len(table.rows)}x{len(table.columns)}",
                "windows": len(windows),
                "currency": meta.currency,
                "scale_label": meta.scale_label,
            },
        )
        return representations

    @staticmethod
    def _window_label(window: tuple[int, int], windows: list[tuple[int, int]], total: int) -> str:
        """ "(rows 31-60 of 90)" for a split table, "" otherwise."""
        if len(windows) == 1:
            return ""
        return f" (rows {window[0] + 1}-{window[1]} of {total})"

    def _narrative(self, table: TableData, note: str, windows: list[tuple[int, int]],
                   base: dict) -> dict:
        """
        Natural-language rendering of the table for semantic search.

        A table that fits one chunk lists every row with its values. A longer
        one is described instead — its columns and row labels — because its
        values are already in the per-window chunks and repeating them here
        would only split this representation too.
        """
        columns = ", ".join(table.columns)
        lines = [
            f"Financial table with {len(table.rows)} rows and {len(table.columns)} columns "
            f"({columns}). {note}"
        ]
        if len(windows) == 1:
            lines += [line for row in table.rows if (line := _narrative_line(table, row))]
        else:
            lines.append("Rows: " + "; ".join(row.label for row in table.rows))
        return {
            **base,
            "text": "\n".join(lines),
            "table_representation": "narrative",
            "table_row_range": [1, len(table.rows)],
        }

    def _row_by_row(self, table: TableData, meta: TableNormalizationMeta, note: str,
                    window: tuple[int, int], windows: list[tuple[int, int]], base: dict) -> dict:
        """One "label: period=value" line per row, plus the structured rows."""
        start, end = window
        lines = [note.rstrip(".") + self._window_label(window, windows, len(table.rows)) + "."]
        lines += [_row_by_row_line(table, row) for row in table.rows[start:end]]
        return {
            **base,
            "text": "\n".join(lines),
            "table_representation": "row_by_row",
            "structured_rows": structured_rows(table, meta, start, end),
            "table_row_range": [start + 1, end],
        }

    def _markdown(self, table: TableData, note: str, window: tuple[int, int],
                  windows: list[tuple[int, int]], base: dict) -> dict:
        """The grid, every cell exactly as printed, header repeated per window."""
        start, end = window
        header_note = f"*{note.rstrip('.')}{self._window_label(window, windows, len(table.rows))}.*"
        header_row = "| " + " | ".join(
            [_markdown_cell(table.label_header)] + [_markdown_cell(c) for c in table.columns]
        ) + " |"
        separator = "|---|" + "|".join(["---"] * len(table.columns)) + "|"
        rows = [_markdown_row(row) for row in table.rows[start:end]]
        text = "\n".join([header_note, "", header_row, separator, *rows])
        if table.notes and end == len(table.rows):
            text += "\n\n" + "\n".join(table.notes)
        return {
            **base,
            "text": text,
            "table_representation": "markdown",
            "table_row_range": [start + 1, end],
        }

    def _metrics_summary(self, table: TableData, base: dict) -> dict:
        """
        Derived metrics: revenue CAGR and YoY growth, and margins.

        Computed from the values as stated (a table-wide scale cancels in every
        ratio). See the module docstring for when a metric is withheld.
        """
        metrics: dict[str, dict] = {}
        lines: list[str] = []

        revenue = _find_row(table, _REVENUE_LABELS)
        years = annual_columns(table.columns)

        if revenue is not None:
            points = [
                (year, table.columns[index], revenue.cells[index])
                for year, index in years
                if revenue.cells[index].value is not None
            ]
            if len(points) >= 2:
                (first_year, first_col, first), (last_year, last_col, last) = points[0], points[-1]
                span = last_year - first_year
                # A negative or zero endpoint has no real growth rate.
                if span >= 1 and first.value > 0 and last.value > 0:
                    cagr = (last.value / first.value) ** (1 / span) - 1
                    metrics["Revenue CAGR"] = {
                        "value": round(cagr * 100, 2),
                        "unit": "%",
                        "citation_chain": (
                            f"CAGR({revenue.label}[{first_col}={first.printed} → "
                            f"{last_col}={last.printed}], n={span})"
                        ),
                        "content_type": "computed_metric",
                    }
                    lines.append(
                        f"Revenue CAGR: {cagr * 100:.2f}% ({first_col}: {first.printed} → "
                        f"{last_col}: {last.printed}, {span} year{'s' if span != 1 else ''})"
                    )

            for (prev_year, prev_col, prev), (year, col, current) in zip(points, points[1:]):
                # Year over year means exactly one fiscal year apart, and a
                # change measured against a zero or negative base is not a
                # growth rate.
                if year - prev_year != 1 or prev.value <= 0:
                    continue
                yoy = (current.value - prev.value) / prev.value
                metrics[f"Revenue YoY {col}"] = {
                    "value": round(yoy * 100, 2),
                    "unit": "%",
                    "citation_chain": (
                        f"YoY({revenue.label}[{prev_col}={prev.printed} → {col}={current.printed}])"
                    ),
                    "content_type": "computed_metric",
                }
                lines.append(
                    f"Revenue YoY {col}: {yoy * 100:.2f}% ({prev_col}: {prev.printed} → "
                    f"{col}: {current.printed})"
                )

            for metric_name, aliases in _MARGIN_NUMERATORS:
                numerator = _find_row(table, aliases)
                if numerator is None:
                    continue
                for index, col in enumerate(table.columns):
                    if parse_column_period(col) is None:
                        continue  # "% change", "Variance": not a period
                    num, den = numerator.cells[index], revenue.cells[index]
                    if num.value is None or den.value is None or den.value <= 0:
                        continue
                    margin = num.value / den.value
                    metrics[f"{metric_name} {col}"] = {
                        "value": round(margin * 100, 2),
                        "unit": "%",
                        "citation_chain": (
                            f"{metric_name}({numerator.label}[{col}]={num.printed} / "
                            f"{revenue.label}[{col}]={den.printed})"
                        ),
                        "content_type": "computed_metric",
                    }
                    lines.append(
                        f"{metric_name} {col}: {margin * 100:.2f}% "
                        f"({numerator.label} {num.printed} / {revenue.label} {den.printed})"
                    )

        header = "Computed Metrics Summary (derived from the table above, not stated in the source):"
        return {
            **base,
            "text": "\n".join([header, *lines]),
            "table_representation": "metrics_summary",
            "metrics": metrics,
            "content_type": "computed_metric",
            "table_row_range": [1, len(table.rows)],
        }
