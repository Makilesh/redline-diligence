"""
PDF layout processor using PyMuPDF (fitz).

Extracts text with layout awareness: headings detected by font size,
paragraphs, tables, and page numbers. Integrates with LegalClauseSegmenter
for legal PDFs and MultiPageTableStitcher for ruled tables.

Text is rebuilt as visual rows. PyMuPDF reports each table cell of a
financial statement as its own "line", so the old line-by-line text put
"Total revenues", "452,800", "387,100" and "341,900" on four lines and the year
header on three more — the figures survived, which column they belonged to did
not. A row now keeps its cells on one line ("Total revenues  452,800
387,100  341,900"), and a run of such rows under a period header — the layout
of every 10-K statement, which has no ruling lines for pdfplumber to find — is
emitted as a table, so it gets the same representations as a ruled one.

Also removed from the text: running page headers and footers ("Form 10-K |
Page 41"), and the text of ruled tables, which used to be indexed twice — as a
table and again, one cell per line, as body text.

A file that cannot be read fails with a reason a user can act on (password,
corrupt, no text layer), and pages with no text layer are reported as a
warning instead of being dropped silently.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from src.data_processing.errors import DocumentExtractionError, NoExtractableContentError
from src.data_processing.legal_clause_segmenter import LegalClauseSegmenter
from src.data_processing.multi_page_table_stitcher import MultiPageTableStitcher
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Running headers/footers are looked for in these bands of the page height.
_TOP_BAND = 0.08
_BOTTOM_BAND = 0.12
_PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE)
# A value cell of a statement row: an amount, percentage or multiple, or the
# dash a statement prints for nil.
_VALUE = re.compile(r"^[\s$€£¥₹(\-−–+]*\d[\d,]*\.?\d*\s*[%xX×]?\s*\)?$")
_NIL = re.compile(r"^[\-—–−·]+$")
_PERIOD = re.compile(
    r"^(?:(?:FY|CY)\s?'?\d{2,4}[AEFP]?|(?:19|20)\d{2}[AEFP]?|Q[1-4]\s?(?:FY)?\s?'?\d{2,4}"
    r"|H[12]\s?'?\d{2,4}|LTM|TTM|YTD)$",
    re.IGNORECASE,
)
# Consecutive section-label rows a statement may carry between figures.
_MAX_LABEL_ROWS = 3
# A word in a table cell: what a row label has and a strip of amounts does not.
_LABEL = re.compile(r"[A-Za-z]{3,}")
_FOOTNOTE = re.compile(r"^(?:\(\d{1,2}\)|\d{1,2}\)|\*{1,3}|†|‡)\s+\S")
_CAPTION = re.compile(
    r"^\(.*\)$|\bin\s+(?:thousands|millions|billions)\b|\b(?:USD|EUR|GBP|\$)\s*(?:000|m|mm|bn)\b",
    re.IGNORECASE,
)


@dataclass
class PDFSection:
    """A structural section extracted from a PDF."""
    text: str
    page_number: int
    page_range: list[int] = field(default_factory=list)
    section_heading: str = ""
    section_type: str = "body"  # heading, body, table, footer, header
    font_size: float = 0.0
    is_table: bool = False
    clause_id: str | None = None
    # Header row first, then data rows — lets the ingestion pipeline build the
    # FinancialTableConverter representations instead of indexing flat text.
    table_rows: list[list[str]] | None = None
    # Unit line printed next to the table ("(in thousands, except per share
    # data)") and footnotes printed under it.
    caption: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class _Cell:
    text: str
    x0: float
    x1: float


@dataclass
class _Row:
    cells: list[_Cell]
    top: float
    bottom: float
    size: float

    @property
    def text(self) -> str:
        return "  ".join(c.text for c in self.cells)


@dataclass
class _Table:
    """A borderless table found in the rows of one page."""
    start: int
    end: int  # exclusive, over the page's rows
    header: list[str]
    rows: list[list[str]]
    caption: str = ""
    notes: list[str] = field(default_factory=list)
    note_rows: int = 0


def _ranges(numbers: list[int]) -> str:
    """[2, 3, 4, 7] -> "2-4, 7"."""
    out, start, prev = [], None, None
    for n in numbers:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(out)


class PDFProcessor:
    """
    PDF layout processor that extracts structured sections from PDF documents.

    Strategy:
    1. Open and validate the file (password, corruption, text layer)
    2. Find ruled tables (pdfplumber, stitched across pages) and their regions
    3. Rebuild each page as visual rows, minus running headers/footers and the
       ruled tables' regions
    4. Headings by font size (> median * 1.2); borderless statement tables by
       layout; everything else is body text
    5. For legal PDFs: clause segmentation over the whole document

    Args:
        legal_mode: If True, use clause segmentation instead of font-based headings.
    """

    def __init__(self, legal_mode: bool = False):
        self.legal_mode = legal_mode
        self._clause_segmenter = LegalClauseSegmenter() if legal_mode else None
        self._table_stitcher = MultiPageTableStitcher()
        self.warnings: list[str] = []

    # ── Entry point ───────────────────────────────────────────────────────────

    def process(self, pdf_path: str, doc_id: str) -> list[PDFSection]:
        """
        Processes a PDF file into structured sections.

        Args:
            pdf_path: Absolute path to the PDF file.
            doc_id: Document identifier for metadata.

        Returns:
            List of PDFSection objects with text, page numbers, and headings.
            Non-fatal problems (pages without a text layer) are in self.warnings.

        Raises:
            FileNotFoundError: If pdf_path does not exist.
            DocumentExtractionError: The file is password-protected or not a
                readable PDF.
            NoExtractableContentError: No page has a text layer (a scan).
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        self.warnings = []

        logger.info(
            "Processing PDF",
            extra={"pdf_path": pdf_path, "doc_id": doc_id, "legal_mode": self.legal_mode},
        )

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise DocumentExtractionError(
                "The file is not a readable PDF (it may be corrupt or truncated)"
            ) from e

        try:
            if doc.needs_pass:
                raise DocumentExtractionError(
                    "The PDF is password-protected; upload a copy without the password"
                )
            # PyMuPDF "repairs" a damaged file to zero pages rather than failing.
            if doc.page_count == 0:
                raise DocumentExtractionError(
                    "The file is not a readable PDF (it may be corrupt or truncated)"
                )
            total_pages = doc.page_count

            pages = [self._page_rows(doc[i]) for i in range(total_pages)]
            empty = [i + 1 for i, rows in enumerate(pages) if not rows]
            if len(empty) == total_pages:
                raise NoExtractableContentError(
                    f"The PDF has no text layer on any of its {total_pages} page(s); it looks "
                    "like a scan. Run OCR on it and upload the result."
                )
            if empty:
                self.warnings.append(
                    f"Page(s) {_ranges(empty)} of {total_pages} have no text layer (likely "
                    "scanned images) and were not indexed"
                )

            ruled = self._extract_tables(pdf_path)
            self._drop_running_lines(pages, [doc[i].rect.height for i in range(total_pages)])

            if self.legal_mode:
                sections = self._process_legal(pages, ruled)
            else:
                sections = self._process_layout(pages, ruled)
        finally:
            doc.close()

        logger.info(
            "PDF processing complete",
            extra={
                "doc_id": doc_id,
                "total_sections": len(sections),
                "total_pages": total_pages,
                "tables": sum(1 for s in sections if s.is_table),
                "warnings": len(self.warnings),
            },
        )
        return sections

    # ── Rows ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _page_rows(page) -> list[_Row]:
        """
        A page's text as visual rows: lines whose vertical extents overlap are
        one row, their cells ordered left to right.
        """
        lines = []
        for block in page.get_text("dict", sort=True)["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                if not text:
                    continue
                x0, y0, x1, y1 = line["bbox"]
                size = max(span["size"] for span in line["spans"])
                lines.append((y0, y1, x0, x1, text, size))
        lines.sort(key=lambda item: ((item[0] + item[1]) / 2, item[2]))

        rows: list[_Row] = []
        for y0, y1, x0, x1, text, size in lines:
            row = rows[-1] if rows else None
            if row is not None:
                overlap = min(row.bottom, y1) - max(row.top, y0)
                if overlap >= 0.5 * min(row.bottom - row.top, y1 - y0):
                    row.cells.append(_Cell(text, x0, x1))
                    row.top, row.bottom = min(row.top, y0), max(row.bottom, y1)
                    row.size = max(row.size, size)
                    continue
            rows.append(_Row([_Cell(text, x0, x1)], y0, y1, size))
        for row in rows:
            row.cells.sort(key=lambda c: c.x0)
        return rows

    @staticmethod
    def _drop_running_lines(pages: list[list[_Row]], heights: list[float]) -> None:
        """
        Removes running headers and footers: rows in the top or bottom band of
        the page that recur, digits aside, on at least two pages (and half of
        them), plus bare page numbers in those bands.
        """
        def in_band(row: _Row, height: float) -> bool:
            return row.bottom <= height * _TOP_BAND or row.top >= height * (1 - _BOTTOM_BAND)

        def key(row: _Row) -> str:
            return re.sub(r"\d+", "#", row.text.lower()).strip()

        counts: dict[str, int] = {}
        for rows, height in zip(pages, heights):
            for k in {key(r) for r in rows if in_band(r, height)}:
                counts[k] = counts.get(k, 0) + 1
        needed = max(2, (len(pages) + 1) // 2)
        running = {k for k, n in counts.items() if n >= needed}
        for i, (rows, height) in enumerate(zip(pages, heights)):
            pages[i] = [
                r for r in rows
                if not (in_band(r, height) and (key(r) in running or _PAGE_NUMBER.match(r.text)))
            ]

    # ── Layout mode ───────────────────────────────────────────────────────────

    def _process_layout(self, pages: list[list[_Row]], ruled: list[dict]) -> list[PDFSection]:
        """
        Headings, body text and tables in reading order.

        Body text is closed at every page break so each section is cited with
        the page its text is on, not the page its heading started on.
        """
        sections: list[PDFSection] = []
        heading = ""
        body: list[str] = []
        body_page = 1
        recent: list[_Row] = []  # rows since the last heading, for captions
        previous_table: tuple[PDFSection, list[str]] | None = None

        def flush() -> None:
            nonlocal body
            if body:
                sections.append(PDFSection(
                    text="\n".join(body), page_number=body_page,
                    section_heading=heading, section_type="body",
                ))
            body = []

        regions = self._regions_by_page(ruled)
        for page_index, rows in enumerate(pages):
            page_number = page_index + 1
            flush()
            body_page = page_number
            sizes = [r.size for r in rows]
            median = statistics.median(sizes) if sizes else 12.0
            threshold = median * 1.2

            page_regions = regions.get(page_number, [])
            kept = [r for r in rows if not any(self._inside(r, bbox) for bbox, _t, _f in page_regions)]
            borderless = self._borderless_tables(kept)
            starts = {t.start: t for t in borderless}
            markers = sorted(
                ((bbox[1], table, first) for bbox, table, first in page_regions),
                key=lambda m: m[0],
            )

            i = 0
            while i < len(kept) or markers:
                if markers and (i >= len(kept) or markers[0][0] <= kept[i].top):
                    _top, table, first = markers.pop(0)
                    if first:
                        flush()
                        sections.append(self._ruled_section(table, heading, recent))
                    continue
                row = kept[i]
                if i in starts:
                    flush()
                    found = starts[i]
                    section = PDFSection(
                        text="\n".join([found.caption, "  ".join(found.header)]
                                       + ["  ".join(r) for r in found.rows]).strip(),
                        page_number=page_number,
                        page_range=[page_number, page_number],
                        section_heading=heading,
                        section_type="table",
                        is_table=True,
                        table_rows=[found.header] + found.rows,
                        caption=found.caption,
                        notes=found.notes,
                    )
                    if (
                        previous_table is not None
                        and previous_table[1] == found.header
                        and previous_table[0].page_range[1] == page_number - 1
                    ):
                        # The statement continues from the previous page under
                        # the same column header: one table, not two.
                        prev = previous_table[0]
                        prev.table_rows.extend(found.rows)
                        prev.notes.extend(found.notes)
                        prev.page_range[1] = page_number
                        prev.text += "\n" + "\n".join("  ".join(r) for r in found.rows)
                    else:
                        sections.append(section)
                        previous_table = (section, found.header)
                    i = found.end + found.note_rows
                    recent = []
                    continue
                if len(row.cells) == 1 and row.size >= threshold and len(row.text) < 200:
                    flush()
                    heading = row.text
                    sections.append(PDFSection(
                        text=row.text, page_number=page_number, section_heading=row.text,
                        section_type="heading", font_size=row.size,
                    ))
                    recent = []
                else:
                    body.append(row.text)
                    recent.append(row)
                i += 1
        flush()
        return sections

    @staticmethod
    def _inside(row: _Row, bbox) -> bool:
        x0, top, x1, bottom = bbox
        middle = (row.top + row.bottom) / 2
        return top - 1 <= middle <= bottom + 1 and any(
            c.x0 >= x0 - 2 and c.x1 <= x1 + 2 for c in row.cells
        )

    @staticmethod
    def _regions_by_page(ruled: list[dict]) -> dict[int, list]:
        by_page: dict[int, list] = {}
        for table in ruled:
            for n, (page, bbox) in enumerate(table.get("regions") or []):
                by_page.setdefault(page, []).append((bbox, table, n == 0))
        return by_page

    def _ruled_section(self, table: dict, heading: str, recent: list[_Row]) -> PDFSection:
        """A stitched ruled table, under the heading in force where it starts."""
        rows_text = []
        if table.get("headers"):
            rows_text.append(" | ".join(str(h) for h in table["headers"]))
        rows_text += [" | ".join(str(c) for c in row) for row in table.get("rows", [])]
        page_range = table.get("page_range", [1])
        caption = " ".join(r.text for r in recent[-2:] if _CAPTION.search(r.text))
        return PDFSection(
            text="\n".join(rows_text),
            page_number=page_range[0],
            page_range=list(page_range),
            section_heading=heading or table.get("sheet_name_equivalent", "Table"),
            section_type="table",
            is_table=True,
            table_rows=(
                [list(table["headers"])] + [list(r) for r in table.get("rows", [])]
                if table.get("headers") else None
            ),
            caption=caption,
        )

    # ── Borderless tables ─────────────────────────────────────────────────────

    @staticmethod
    def _is_value(text: str) -> bool:
        return bool(_VALUE.match(text) or _NIL.match(text))

    def _value_row(self, row: _Row) -> bool:
        """A label followed by at least two value cells and nothing else."""
        if len(row.cells) < 3 or not re.search(r"[A-Za-z]", row.cells[0].text):
            return False
        # "(Dollars in millions)  2019  2018" is the header of most statements,
        # not their first line: bare years parse as amounts, and taking the
        # header for a value row left the table with "Column 2" / "Column 3".
        if self._period_row(row):
            return False
        return all(self._is_value(c.text) for c in row.cells[1:])

    @staticmethod
    def _label_row(row: _Row) -> bool:
        """A section label inside a statement ("Operating expenses:"): one short cell."""
        return len(row.cells) == 1 and len(row.text) < 80

    @staticmethod
    def _period_row(row: _Row) -> bool:
        cells = [c.text for c in row.cells]
        periods = [c for c in cells if _PERIOD.match(c)]
        return len(periods) >= 2 and len(periods) >= len(cells) - 1

    def _borderless_tables(self, rows: list[_Row]) -> list[_Table]:
        """
        Runs of statement rows under a period header.

        A run is at least two value rows sharing the same columns (cells
        matched to header columns by their right edge, since amounts are
        right-aligned), optionally interrupted by label-only rows ("Operating
        expenses:"). Without a period header a run needs three rows and its
        columns stay unnamed. A row whose cells cannot be placed in distinct
        columns ends the run — the table is not guessed.
        """
        tables: list[_Table] = []
        i = 0
        while i < len(rows):
            if not self._value_row(rows[i]):
                i += 1
                continue

            header_index = None
            for j in range(i - 1, max(i - 2 - _MAX_LABEL_ROWS, -1), -1):
                if self._period_row(rows[j]):
                    header_index = j
                    break
                # Past the row just above, only section labels ("ASSETS",
                # "Current assets:") may sit between a header and its figures —
                # the layout of every balance sheet.
                if j < i - 1 and not self._label_row(rows[j]):
                    break
            if header_index is not None:
                header_cells = [c for c in rows[header_index].cells if _PERIOD.match(c.text)]
            else:
                header_cells = rows[i].cells[1:]
            anchors = [c.x1 for c in header_cells]
            names = [c.text for c in header_cells] if header_index is not None else [
                f"Column {n + 2}" for n in range(len(anchors))]

            # Rows between the header and the first figures are section labels;
            # they belong to the table (they used to be skipped with it, and lost).
            data: list[list[str]] = [
                [rows[k].text] + [""] * len(anchors)
                for k in range(header_index + 1 if header_index is not None else i, i)
            ]
            j = i
            while j < len(rows):
                row = rows[j]
                if self._value_row(row):
                    placed = self._place(row.cells[1:], anchors)
                    if placed is None:
                        break
                    data.append([row.cells[0].text] + placed)
                    j += 1
                    continue
                # Label-only rows inside the statement ("Operating expenses:",
                # "LIABILITIES AND EQUITY" / "Current liabilities:").
                labels = 0
                while (j + labels < len(rows) and labels < _MAX_LABEL_ROWS
                       and self._label_row(rows[j + labels])):
                    labels += 1
                if labels and j + labels < len(rows) and self._value_row(rows[j + labels]):
                    data.extend([rows[k].text] + [""] * len(anchors) for k in range(j, j + labels))
                    j += labels
                    continue
                break
            value_rows = sum(1 for r in data if any(r[1:]))
            if value_rows < (2 if header_index is not None else 3):
                i = j if j > i else i + 1
                continue

            start = header_index if header_index is not None else i
            label_header = ""
            caption_rows = []
            if header_index is not None:
                header_row = rows[header_index]
                label_header = header_row.cells[0].text if not _PERIOD.match(
                    header_row.cells[0].text) else ""
                # A spanning caption over the period columns ("Year Ended
                # December 31,") belongs to every column header.
                above = rows[header_index - 1] if header_index >= 1 else None
                if above is not None and len(above.cells) == 1 and above.cells[0].x0 > (
                        rows[i].cells[0].x1) and not _CAPTION.search(above.text):
                    names = [f"{above.text} {n}" for n in names]
                    start -= 1
            for k in range(start - 1, max(start - 3, -1), -1):
                if _CAPTION.search(rows[k].text) and len(rows[k].text) < 120:
                    caption_rows.insert(0, rows[k].text)
                else:
                    break
            notes = []
            k = j
            while k < len(rows) and _FOOTNOTE.match(rows[k].text):
                notes.append(rows[k].text)
                k += 1
            tables.append(_Table(
                start=start, end=j, header=[label_header] + names, rows=data,
                caption=" ".join(caption_rows), notes=notes, note_rows=k - j,
            ))
            i = k
        # Caption rows stay in the body; a table's start is its header.
        for table in tables:
            while table.start < len(rows) and rows[table.start].text in table.caption:
                table.start += 1
        return tables

    @staticmethod
    def _place(cells: list[_Cell], anchors: list[float]) -> list[str] | None:
        """Puts value cells into the columns whose right edge is nearest; None if two collide."""
        if not anchors:
            return None
        spacing = min((b - a for a, b in zip(anchors, anchors[1:])), default=200.0)
        placed = [""] * len(anchors)
        for cell in cells:
            column = min(range(len(anchors)), key=lambda n: abs(anchors[n] - cell.x1))
            if abs(anchors[column] - cell.x1) > 0.6 * spacing or placed[column]:
                return None
            placed[column] = cell.text
        return placed

    # ── Legal mode ────────────────────────────────────────────────────────────

    def _process_legal(self, pages: list[list[_Row]], ruled: list[dict]) -> list[PDFSection]:
        """
        Clause segmentation over the whole document (so a clause that runs over
        a page break stays one clause), on the cleaned rows.
        """
        regions = self._regions_by_page(ruled)
        page_texts = []
        for page_index, rows in enumerate(pages):
            boxes = [bbox for bbox, _t, _f in regions.get(page_index + 1, [])]
            kept = [r for r in rows if not any(self._inside(r, b) for b in boxes)]
            page_texts.append((page_index + 1, "\n".join(r.text for r in kept)))

        sections = [
            PDFSection(
                text=clause["text"],
                page_number=clause["page_number"],
                page_range=clause.get("page_range") or [],
                section_heading=clause.get("clause_id", ""),
                section_type="clause",
                clause_id=clause.get("clause_id"),
            )
            for clause in self._clause_segmenter.segment_document(page_texts)
        ]
        sections += [self._ruled_section(t, "", []) for t in ruled]
        return sections

    # ── Ruled tables ──────────────────────────────────────────────────────────

    def _extract_tables(self, pdf_path: str) -> list[dict]:
        """
        Ruled tables via MultiPageTableStitcher; an extraction failure costs the
        tables, not the document, and is reported as a warning.
        """
        try:
            tables = self._table_stitcher.stitch_document_tables(pdf_path)
        except Exception as e:
            logger.warning(
                "Table extraction failed, continuing without tables",
                extra={"error": str(e)},
            )
            self.warnings.append("Tables could not be extracted from this PDF; its text was indexed")
            return []
        kept = [t for t in tables if self._has_row_labels(t)]
        if len(kept) < len(tables):
            logger.info(
                "Ruled tables without row labels left to the text path",
                extra={"dropped": len(tables) - len(kept), "kept": len(kept)},
            )
        return kept

    @staticmethod
    def _has_row_labels(table: dict) -> bool:
        """
        Whether a ruled table carries the labels of its rows.

        Filings rendered from EDGAR HTML often rule only the amount columns (a
        rule under each figure), so pdfplumber's "table" is a strip of numbers
        — "$ | 39,117", "21,643", ... — with every line item outside it. Taken
        as a table, its region was removed from the page text, labels and all:
        on NIKE's FY2019 10-K, 32 of 34 ruled tables were such strips and the
        balance sheet reached the index as 100 unlabelled figures. A table where
        fewer than a quarter of the rows have a word in them is left to the
        text path, which keeps each row's label next to its figures.
        """
        rows = [r for r in table.get("rows") or [] if any(str(c or "").strip() for c in r)]
        if not rows:
            return True
        labelled = sum(1 for r in rows if any(_LABEL.search(str(c or "")) for c in r))
        return labelled >= 0.25 * len(rows)
