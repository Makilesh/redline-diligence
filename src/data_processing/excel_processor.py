"""
Excel processor — reads a workbook cell by cell with openpyxl.

Each sheet is cut into its tables and every table goes through the
FinancialTableConverter. The workbook is read with openpyxl directly rather
than through pandas because the things that decide whether a figure is right
are not in a DataFrame:

- number formats: a cell holding 0.5911 formatted "0.0%" is 59.11%, not 0.5911
  of the table's "USD in thousands";
- merged cells: a "FY2023" header merged over "Actual | Budget" names two
  columns, and pandas calls the second one "Unnamed: 2";
- sheet state: a hidden sheet is still part of the workbook a reviewer was
  sent, so it is indexed — and flagged;
- formulas with no cached value: a workbook written by a script and never
  opened in Excel has formulas but no results, and reading it silently yields
  empty cells where the totals should be.

Only .xlsx is supported: openpyxl cannot read the legacy binary .xls format.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data_processing.excel_normalizer import ExcelNormalizer, TableNormalizationMeta
from src.data_processing.financial_table_converter import (
    FinancialTableConverter,
    TableData,
    fmt_exact,
    representation_content_type,
    table_from_rows,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Above this size a workbook is opened in openpyxl's streaming mode, which
# holds one row at a time but cannot report merged cells.
_FULL_LOAD_MAX_BYTES = 25 * 1024 * 1024


def _skip_embedded_images() -> None:
    """
    Stops openpyxl opening a workbook's embedded pictures with Pillow.

    A full (non-streaming) load hands every picture in an uploaded workbook to
    PIL.Image.open to read its size, the one path by which an upload reached
    Pillow's image parsers, the subject of most published Pillow advisories
    (Decision 104). Nothing here reads pictures, only cells, so they are
    dropped the way openpyxl drops them when Pillow is not installed. Charts
    are still read, and cell values are unchanged
    (tests/test_excel_untrusted_images.py).
    """
    import openpyxl.reader.drawings as drawings

    drawings.PILImage = False
# A header is looked for in the first rows of a block only.
_HEADER_SEARCH_ROWS = 15
UNCALCULATED = "[formula not calculated]"


class ExcelProcessingError(Exception):
    """Raised when no sheet of a workbook could be processed."""


@dataclass
class ExcelSheet:
    """One table extracted from a sheet (a sheet can hold several)."""
    sheet_name: str
    table: TableData | None = None
    normalization_meta: TableNormalizationMeta = None
    table_id: str = ""
    # FinancialTableConverter output: one dict per representation, all sharing table_id
    representations: list[dict] = field(default_factory=list)
    header_rows: list[str] = field(default_factory=list)
    caption: str = ""
    sheet_state: str = "visible"


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _is_year(value: Any) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and float(value).is_integer() and 1900 <= value <= 2100
    )


def _header_like(value: Any) -> bool:
    """A cell that can sit in a column header: text, a date, or a year."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (str, dt.date, dt.datetime)) or _is_year(value)


def _header_text(value: Any) -> str:
    """Header cell as text: 2023.0 -> "2023", a date -> ISO."""
    if _is_blank(value):
        return ""
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if _is_year(value):
        return str(int(value))
    if isinstance(value, float):
        return fmt_exact(value)
    return " ".join(str(value).split())


def _cell_value(value: Any, number_format: str) -> Any:
    """
    A data cell as the converter should see it.

    Numbers stay numbers, except percentages: Excel stores 59.11% as 0.5911
    with a "%" number format, so the format is applied here and the cell is
    handed on as "59.11%" — its exact value, in the unit the sheet shows.
    """
    if isinstance(value, bool) or _is_blank(value):
        return None if _is_blank(value) else str(value)
    if isinstance(value, (int, float)):
        if "%" in (number_format or ""):
            return f"{fmt_exact(float(value) * 100)}%"
        return value
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time(0) else value.isoformat(sep=" ")
    if isinstance(value, (dt.date, dt.time)):
        return value.isoformat()
    return " ".join(str(value).split())


def _row_is_header(row: list[Any]) -> bool:
    """
    Whether a row reads as column headers: at least two filled cells, every one
    of them text, a date or a year. A title or a "(USD in thousands)" caption
    fills one cell; a data row has amounts.
    """
    filled = [v for v in row if not _is_blank(v)]
    # Distinct values: a title merged across the sheet repeats in every cell.
    if len(filled) < 2 or len({str(v) for v in filled}) < 2:
        return False
    return all(_header_like(v) for v in filled)


class ExcelProcessor:
    """
    Processes Excel files into structured, normalized table data.

    Pipeline, per sheet:
    1. Read values, number formats, merged ranges and the sheet's visibility.
    2. Cut the sheet into tables at blank rows that are followed by a new
       header row.
    3. Find each table's header (one or two rows), read the rows above it as
       its caption, and detect scale and currency from both.
    4. Generate the representations via FinancialTableConverter.

    Sheet-level failures are recorded in `failed_sheets` and logged at error.
    If every non-empty sheet fails, process() raises instead of returning an
    empty list — an upload that silently yields zero chunks looks like success.
    Things a reviewer must know but that are not errors (uncalculated
    formulas, hidden sheets) are collected in `warnings`.
    """

    def __init__(self):
        self._normalizer = ExcelNormalizer()
        self._converter = FinancialTableConverter()
        self.failed_sheets: list[dict] = []
        self.warnings: list[str] = []

    def process(
        self,
        excel_path: str,
        doc_id: str,
        table_id_prefix: str | None = None,
    ) -> list[ExcelSheet]:
        """
        Processes an Excel file into one ExcelSheet per table.

        Args:
            excel_path: Absolute path to the Excel file.
            doc_id: Document identifier for metadata.
            table_id_prefix: Prefix for table_ids (default: doc_id).
                The ingestion pipeline passes "{deal_id}_{doc_id}".

        Returns:
            List of ExcelSheet objects with normalized data and representations.

        Raises:
            FileNotFoundError: If excel_path does not exist.
            ExcelProcessingError: If sheets had content but none could be processed.
        """
        from openpyxl import load_workbook

        path = Path(excel_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")

        logger.info("Processing Excel file", extra={"path": excel_path, "doc_id": doc_id})

        prefix = table_id_prefix or doc_id
        self.failed_sheets = []
        self.warnings = []
        streaming = path.stat().st_size > _FULL_LOAD_MAX_BYTES

        _skip_embedded_images()
        values_wb = load_workbook(excel_path, data_only=True, read_only=streaming)
        # Second pass with formulas visible: the only way to tell an empty cell
        # from a formula whose result was never cached.
        formulas_wb = load_workbook(excel_path, data_only=False, read_only=True)

        sheets: list[ExcelSheet] = []
        try:
            for worksheet in values_wb.worksheets:
                name = worksheet.title
                try:
                    tables = self._process_sheet(
                        worksheet,
                        formulas_wb[name] if name in formulas_wb.sheetnames else None,
                        doc_id,
                        table_id=f"{prefix}_t{len(sheets):03d}",
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to process sheet '{name}'",
                        extra={"error": str(e), "doc_id": doc_id},
                        exc_info=True,
                    )
                    self.failed_sheets.append({"sheet_name": name, "error": str(e)})
                    continue
                for table in tables or []:
                    table.table_id = f"{prefix}_t{len(sheets):03d}"
                    for rep in table.representations:
                        rep["table_id"] = table.table_id
                    sheets.append(table)
        finally:
            values_wb.close()
            formulas_wb.close()

        if not sheets and self.failed_sheets:
            raise ExcelProcessingError(
                f"None of {len(self.failed_sheets)} sheet(s) could be processed: "
                + "; ".join(f"{f['sheet_name']}: {f['error']}" for f in self.failed_sheets)
            )

        logger.info(
            "Excel processing complete",
            extra={
                "doc_id": doc_id,
                "total_tables": len(sheets),
                "failed_sheets": [f["sheet_name"] for f in self.failed_sheets],
                "sheet_names": sorted({s.sheet_name for s in sheets}),
                "warnings": len(self.warnings),
            },
        )
        return sheets

    # ── Reading ──────────────────────────────────────────────────────────────

    def _read_grid(self, worksheet, formula_sheet) -> tuple[list[list[Any]], int]:
        """
        Reads a sheet into a rectangular grid of converter-ready cell values.

        Returns:
            (grid with fully empty leading/trailing columns removed, number of
            formula cells that have no cached value).
        """
        formulas: set[tuple[int, int]] = set()
        if formula_sheet is not None:
            for r, row in enumerate(formula_sheet.iter_rows()):
                for c, cell in enumerate(row):
                    if cell.data_type == "f" or (
                        isinstance(cell.value, str) and cell.value.startswith("=")
                    ):
                        formulas.add((r, c))

        raw: list[list[Any]] = []
        uncalculated = 0
        for r, row in enumerate(worksheet.iter_rows()):
            values = []
            for c, cell in enumerate(row):
                value = cell.value
                if _is_blank(value) and (r, c) in formulas:
                    uncalculated += 1
                    values.append(UNCALCULATED)
                    continue
                values.append((value, getattr(cell, "number_format", "") or ""))
            raw.append(values)

        # A merged range holds its value in the top-left cell only. Copy it
        # across the range so a header merged over two columns names both.
        # Only header-like values are copied: repeating a merged AMOUNT into
        # the cells it spans would state a figure for periods it was never
        # printed under.
        for merged in getattr(getattr(worksheet, "merged_cells", None), "ranges", []) or []:
            top, left = merged.min_row - 1, merged.min_col - 1
            if top >= len(raw) or left >= len(raw[top]):
                continue
            anchor = raw[top][left]
            if isinstance(anchor, str) or not _header_like(anchor[0]):
                continue
            for r in range(top, min(merged.max_row, len(raw))):
                for c in range(left, min(merged.max_col, len(raw[r]))):
                    raw[r][c] = anchor

        width = max((len(row) for row in raw), default=0)
        grid = [row + [(None, "")] * (width - len(row)) for row in raw]
        used = [
            c for c in range(width)
            if any(not self._blank(row[c]) for row in grid)
        ]
        if not used:
            return [], uncalculated
        grid = [row[used[0]:used[-1] + 1] for row in grid]
        return grid, uncalculated

    @staticmethod
    def _blank(cell: Any) -> bool:
        return not isinstance(cell, str) and _is_blank(cell[0])

    @staticmethod
    def _raw(cell: Any) -> Any:
        """The stored value of a grid cell (the formula marker is its own value)."""
        return cell if isinstance(cell, str) else cell[0]

    @staticmethod
    def _data(cell: Any) -> Any:
        return cell if isinstance(cell, str) else _cell_value(cell[0], cell[1])

    def _split_blocks(self, grid: list[list[Any]]) -> list[list[list[Any]]]:
        """
        Cuts a sheet into stacked tables.

        A blank row separates two tables only when what follows it starts a new
        table — a header row — within the next few rows. Blank spacer rows
        inside one statement do not split it.
        """
        blocks: list[list[list[Any]]] = []
        current: list[list[Any]] = []
        i = 0
        while i < len(grid):
            row = grid[i]
            if all(self._blank(c) for c in row):
                j = i
                while j < len(grid) and all(self._blank(c) for c in grid[j]):
                    j += 1
                lookahead = grid[j:j + 4]
                starts_table = any(
                    _row_is_header([self._raw(c) for c in r]) for r in lookahead
                ) and any(
                    isinstance(self._raw(c), (int, float)) and not _is_year(self._raw(c))
                    for r in current for c in r[1:]
                )
                if current and starts_table:
                    blocks.append(current)
                    current = []
                i = j
                continue
            current.append(row)
            i += 1
        if current:
            blocks.append(current)
        return blocks

    def _process_sheet(
        self,
        worksheet,
        formula_sheet,
        doc_id: str,
        table_id: str,
    ) -> list[ExcelSheet] | None:
        """
        Processes a single sheet into its tables.

        Args:
            worksheet: openpyxl worksheet opened with cached values.
            formula_sheet: The same sheet opened with formulas, or None.
            doc_id: Document identifier.
            table_id: Identifier for the sheet's first table.

        Returns:
            List of ExcelSheet (one per table), or None if the sheet is empty.
        """
        name = worksheet.title
        state = getattr(worksheet, "sheet_state", "visible") or "visible"
        if not hasattr(worksheet, "iter_rows"):
            return None  # chartsheet: no cells

        grid, uncalculated = self._read_grid(worksheet, formula_sheet)
        if not grid:
            return None

        if uncalculated:
            self.warnings.append(
                f"Sheet '{name}': {uncalculated} formula cell(s) have no calculated value "
                "(the workbook was saved without being calculated); they are indexed as "
                f"'{UNCALCULATED}', not as numbers"
            )
        if state != "visible":
            self.warnings.append(
                f"Sheet '{name}' is hidden in the workbook; it was indexed and is marked hidden"
            )

        tables = []
        for block in self._split_blocks(grid):
            sheet = self._process_block(block, name, table_id)
            if sheet is not None:
                sheet.sheet_state = state
                tables.append(sheet)
        return tables or None

    def _process_block(self, block: list[list[Any]], name: str, table_id: str) -> ExcelSheet | None:
        """Turns one table's rows into an ExcelSheet."""
        raw_rows = [[self._raw(c) for c in row] for row in block]

        header_idx = next(
            (i for i, row in enumerate(raw_rows[:_HEADER_SEARCH_ROWS]) if _row_is_header(row)),
            None,
        )
        width = len(raw_rows[0])
        if header_idx is None:
            # No header row at all (label/value pairs, or amounts from the
            # first row): every row is data and the columns stay unnamed.
            pre_header: list[str] = []
            header = [""] * width
            data_start = 0
        else:
            pre_header = []
            for row in raw_rows[:header_idx]:
                parts = list(dict.fromkeys(_header_text(v) for v in row if not _is_blank(v)))
                if parts:
                    pre_header.append(" ".join(parts))
            header = [_header_text(v) for v in raw_rows[header_idx]]
            data_start = header_idx + 1

        # A second header row ("Actual | Budget" under a merged "FY2023", or the
        # years under "Year ended December 31,") has no row label and no amount.
        if header_idx is not None and data_start < len(raw_rows):
            second = raw_rows[data_start]
            body = [v for v in second[1:] if not _is_blank(v)]
            if _is_blank(second[0]) and len(body) >= 2 and all(_header_like(v) for v in body) and (
                data_start + 1 < len(raw_rows)
            ):
                sub = [_header_text(v) for v in second]
                top = list(header)
                # Fill the gaps a merged or centred-across top header leaves.
                for c in range(2, len(top)):
                    if not top[c] and sub[c]:
                        top[c] = top[c - 1]
                header = [
                    " ".join(part for part in (top[c], sub[c]) if part) if c else top[0]
                    for c in range(len(top))
                ]
                data_start += 1

        rows = [[self._data(c) for c in row] for row in block[data_start:]]
        rows = [row for row in rows if any(not _is_blank(v) for v in row)]
        if not rows and not any(header):
            return None

        caption = "\n".join(pre_header)
        norm_meta = self._normalizer.detect_scale(header + pre_header)

        table = table_from_rows(header, rows)
        if table is not None:
            representations = self._converter.generate_all_representations(
                df=table,
                meta=norm_meta,
                table_id=table_id,
                source_metadata={"sheet_name": name},
            )
        else:
            # No numeric table (a notes or assumptions sheet): index the cells
            # verbatim rather than inventing financial representations.
            lines = [" | ".join(header)] + [
                " | ".join("" if _is_blank(v) else (fmt_exact(v) if isinstance(v, float) else str(v))
                           for v in row)
                for row in rows
            ]
            representations = [{
                "text": "\n".join(line for line in lines if line.strip(" |")),
                "table_representation": "verbatim",
                "table_id": table_id,
                "currency": norm_meta.currency,
                "scale_factor": norm_meta.scale_factor,
                "scale_label": norm_meta.scale_label,
            }]

        return ExcelSheet(
            sheet_name=name,
            table=table,
            normalization_meta=norm_meta,
            table_id=table_id,
            representations=representations,
            header_rows=header,
            caption=caption,
        )

    def to_chunks(self, sheets: list[ExcelSheet]) -> list[dict]:
        """
        Converts ExcelSheet objects to sections for the ingestion pipeline.
        Each representation becomes a separate section sharing the table's
        table_id, so sibling retrieval can pull every representation back.

        Args:
            sheets: List of ExcelSheet from process().

        Returns:
            List of section dicts ready for chunking.
        """
        chunks = []
        for sheet in sheets:
            prefix = "\n".join(part for part in (sheet.sheet_name, sheet.caption) if part)
            for rep in sheet.representations:
                rep_type = rep["table_representation"]
                text = rep.get("text", "")
                # A metrics summary with no computable metric is only its title.
                if rep_type == "metrics_summary" and not rep.get("metrics"):
                    continue
                if not text.strip():
                    continue
                chunk = {
                    "text": f"{prefix}\n{text}",
                    "section_heading": sheet.sheet_name,
                    "page_number": None,
                    "section_type": "table",
                    "is_table": 1,
                    "sheet_name": sheet.sheet_name,
                    "table_id": sheet.table_id,
                    "table_representation": rep_type,
                    "content_type": (
                        "table_text" if rep_type == "verbatim"
                        else representation_content_type(rep_type)
                    ),
                    "currency": rep.get("currency", "UNKNOWN"),
                    "scale_factor": rep.get("scale_factor", 1.0),
                    "scale_label": rep.get("scale_label", "units"),
                }
                if sheet.sheet_state != "visible":
                    chunk["sheet_state"] = sheet.sheet_state
                for key in ("metrics", "structured_rows", "table_row_range", "table_row_count"):
                    if rep.get(key):
                        chunk[key] = rep[key]
                chunks.append(chunk)

        return chunks
