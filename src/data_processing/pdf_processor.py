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

Such a table keeps the period header printed over its figures (Decision 93):
a "$" printed apart from its amount, a line PyMuPDF glued from two cells, a
wrapped label or a label whose figures sit a line lower no longer ends it
under "Column 2", "Column 3"; dates and the span and date lines above the
years ("Three months ended June 30," / "December 2,") name the columns, and so
does a span printed in the label column ("Fiscal Years Ended"); a run the
header did not reach takes it from further up the page, or from the statement
it continues on the previous page. The title printed over a table ("CONSOLIDATED
STATEMENTS OF CASH FLOWS") is its caption, so every chunk of it names the
statement, and each page's part of a statement is its own table, cited at its
own page.

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
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from src.data_processing.errors import DocumentExtractionError, NoExtractableContentError
from src.data_processing.legal_clause_segmenter import LegalClauseSegmenter
from src.data_processing.multi_page_table_stitcher import MultiPageTableStitcher
from src.utils.fail_open import fail_open
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Running headers/footers are looked for in these bands of the page height.
_TOP_BAND = 0.08
_BOTTOM_BAND = 0.12
_PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE)
# A value cell of a statement row: an amount, percentage or multiple, or the
# dash a statement prints for nil.
_VALUE = re.compile(r"^[\s$€£¥₹(\-−–+]*\d[\d,]*\.?\d*\s*[%xX×]?\s*\)?$")
_NIL = re.compile(r"^(?:[$€£¥₹]\s*)?[\-—–−·]+$")
# A currency sign printed apart from its amount: statements set it flush left
# in the column and the figure flush right.
_CURRENCY_SIGN = re.compile(r"^(?:[$€£¥₹]|US\$)$")
_PERIOD_TOKEN = (
    r"(?:FY|CY)\s?'?\d{2,4}[AEFP]?|(?:19|20)\d{2}[AEFP]?|Q[1-4]\s?(?:FY)?\s?'?\d{2,4}"
    r"|H[12]\s?'?\d{2,4}|LTM|TTM|YTD"
)
_PERIOD = re.compile(rf"^(?:{_PERIOD_TOKEN})$", re.IGNORECASE)
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
# A balance date heading a column: "December 28, 2019", "28 December 2019",
# "3/25/2023". Kraft Heinz prints its balance sheet under two such dates and
# PepsiCo its quarterly release under "3/25/2023  3/19/2022"; neither header was
# read, and both statements were indexed as "Column 2" / "Column 3".
_DATE_TOKEN = (
    rf"{_MONTH}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}|\d{{1,2}}\s+{_MONTH},?\s+(?:19|20)\d{{2}}"
    r"|\d{1,2}/\d{1,2}/(?:19|20)\d{2}"
)
_DATE = re.compile(rf"^(?:{_DATE_TOKEN})$", re.IGNORECASE)
_PERIODS_IN = re.compile(rf"(?<![\w/])(?:{_DATE_TOKEN}|{_PERIOD_TOKEN})(?![\w/])", re.IGNORECASE)
# What a line over the period cells says about them: the span of each column
# ("Three months ended", "52 Weeks Ended", "Years Ended") and the day and month
# its year is printed under ("December 2,", "May 31,").
_SPAN = re.compile(
    r"\b(?:months?|weeks?|years?|quarters?|periods?)\s+end(?:ed|ing)\b|\byear[\s-]to[\s-]date\b",
    re.IGNORECASE,
)
_MONTH_DAY = re.compile(rf"\b{_MONTH}\s+\d{{1,2}}\b", re.IGNORECASE)
# A single line over every column names periods with one of these; a
# statement's title centred on the page ("CONSOLIDATED STATEMENTS OF INCOME")
# is no part of a column's name.
_PERIOD_WORD = re.compile(rf"\b{_MONTH}(?!\w)|\bend(?:ed|ing)\b|\bfiscal\b", re.IGNORECASE)
# Header lines read above the period cells: a span, a month and a day line
# ("13 Weeks Ended" / "January" / "28,") over the years of an earnings release.
_MAX_HEADER_LINES = 3
# A header piece is a few words outside parentheses ("FOR THE TRAILING FOUR
# QUARTERS ENDED MAY 31,"), never a sentence; a caption that names no period
# ("Health Care and Life") is fewer still.
_HEADER_WORD = re.compile(r"[A-Za-z]{2,}")
_PARENTHETICAL = re.compile(r"\([^()]*\)")
_MAX_HEADER_WORDS = 8
_MAX_CAPTION_WORDS = 5
_SENTENCE = re.compile(r"[a-z]{2}[.:;!?](?:\s+[A-Z]|\s*$)")
# Consecutive section-label rows a statement may carry between figures.
_MAX_LABEL_ROWS = 3
# A word in a table cell: what a row label has and a strip of amounts does not.
_LABEL = re.compile(r"[A-Za-z]{3,}")
_FOOTNOTE = re.compile(r"^(?:\(\d{1,2}\)|\d{1,2}\)|\*{1,3}|†|‡)\s+\S")
_CAPTION = re.compile(
    r"^\(.*\)$|\bin\s+(?:thousands|millions|billions)\b|\b(?:USD|EUR|GBP|\$)\s*(?:000|m|mm|bn)\b",
    re.IGNORECASE,
)
_SPACE = re.compile(r"\s")
# Lines longer than this are prose, never cells glued together: their words
# are not kept.
_MAX_CELL_CHARS = 120
# A table's title: up to this many short lines printed right above its header
# ("ADOBE SYSTEMS INCORPORATED", "CONSOLIDATED STATEMENTS OF CASH FLOWS"),
# none of them a sentence.
_MAX_TITLE_LINES = 4
_MAX_TITLE_CHARS = 120
_MAX_TITLE_WORDS = 12


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
    # Each word with its horizontal extent, for a short line of several words:
    # PyMuPDF sometimes reports two cells of a statement as one line ("5,070
    # $", "— $", "1,603,451  1,499,218", "December 28, 2019  December 29,
    # 2018"), and only the words' own edges say which column each belongs to.
    words: tuple[tuple[str, float, float], ...] | None = None


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
    # Right edges of the period columns; empty when the header names no period.
    anchors: list[float] = field(default_factory=list)
    # The title lines printed above the table ("CONSOLIDATED STATEMENTS OF CASH
    # FLOWS"), kept apart from the unit caption until the section is built.
    title: str = ""


def _words(chars: list[dict]) -> tuple[tuple[str, float, float], ...]:
    """A line's words — runs of non-space characters — with their left and right edges."""
    words: list[tuple[str, float, float]] = []
    run: list[dict] = []
    for ch in [*chars, None]:
        if ch is not None and not ch["c"].isspace():
            run.append(ch)
            continue
        if run:
            text = unicodedata.normalize("NFC", "".join(c["c"] for c in run))
            words.append((text, run[0]["bbox"][0], run[-1]["bbox"][2]))
            run = []
    return tuple(words)


def _cell_periods(cell: _Cell) -> list[_Cell]:
    """
    The periods a header cell prints, each with its own edges: the cell itself
    when it is one ("2023", "December 28, 2019"); each of them when PyMuPDF
    read two header cells as one line ("December 28, 2019  December 29, 2018",
    "3/25/2023  3/19/2022"), cut out at its words; none otherwise.
    """
    if _PERIOD.match(cell.text) or _DATE.match(cell.text):
        return [cell]
    if not cell.words or len(cell.words) < 2:
        return []
    found = list(_PERIODS_IN.finditer(cell.text))
    if len(found) < 2 or _PERIODS_IN.sub("", cell.text).strip():
        return []
    spans = [m.span() for m in re.finditer(r"\S+", cell.text)]
    if [cell.text[a:b] for a, b in spans] != [w[0] for w in cell.words]:
        return []
    periods = []
    for match in found:
        inside = [n for n, (a, b) in enumerate(spans) if a >= match.start() and b <= match.end()]
        if not inside:
            return []
        periods.append(_Cell(match.group(0), cell.words[inside[0]][1], cell.words[inside[-1]][2]))
    return periods


def _pieces(cell: _Cell, size: float) -> list[_Cell]:
    """
    A header line cut where its words are further apart than a space: PyMuPDF
    reads "January 28,   January 29," over two columns as one line.
    """
    if not cell.words or len(cell.words) < 2:
        return [cell]
    runs = [[cell.words[0]]]
    for word in cell.words[1:]:
        if word[1] - runs[-1][-1][2] > 0.6 * size:
            runs.append([])
        runs[-1].append(word)
    if len(runs) == 1:
        return [cell]
    return [_Cell(" ".join(w[0] for w in run), run[0][1], run[-1][2]) for run in runs]


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
        # "rawdict" is "dict" with each character's box: a line's text is the
        # same either way (checked on every page of four filings), and the
        # boxes give the words of a line their own edges (_Cell.words).
        for block in page.get_text("rawdict", sort=True)["blocks"]:
            for line in block.get("lines", []):
                chars = [ch for span in line["spans"] for ch in span["chars"]]
                # Canonical form: a font's ToUnicode map may name a code point
                # canonically equal to the plain one. In 210 of 232 CUAD
                # contracts every semicolon came out as U+037E GREEK QUESTION
                # MARK, which the identity patterns and every exact match
                # downstream read as another character (Decision 89). NFC
                # changes only how canonically equal text is written.
                text = unicodedata.normalize("NFC", "".join(ch["c"] for ch in chars)).strip()
                if not text:
                    continue
                x0, y0, x1, y1 = line["bbox"]
                size = max(span["size"] for span in line["spans"])
                words = _words(chars) if _SPACE.search(text) and len(text) <= _MAX_CELL_CHARS else None
                lines.append((y0, y1, x0, x1, text, size, words))
        lines.sort(key=lambda item: ((item[0] + item[1]) / 2, item[2]))

        rows: list[_Row] = []
        for y0, y1, x0, x1, text, size, words in lines:
            row = rows[-1] if rows else None
            if row is not None:
                overlap = min(row.bottom, y1) - max(row.top, y0)
                if overlap >= 0.5 * min(row.bottom - row.top, y1 - y0):
                    row.cells.append(_Cell(text, x0, x1, words))
                    row.top, row.bottom = min(row.top, y0), max(row.bottom, y1)
                    row.size = max(row.size, size)
                    continue
            rows.append(_Row([_Cell(text, x0, x1, words)], y0, y1, size))
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

        def flush() -> None:
            nonlocal body
            if body:
                sections.append(PDFSection(
                    text="\n".join(body), page_number=body_page,
                    section_heading=heading, section_type="body",
                ))
            body = []

        ruled = self._ruled_with_headers(pages, ruled)
        regions = self._regions_by_page(ruled)
        carried: _Table | None = None
        for page_index, rows in enumerate(pages):
            page_number = page_index + 1
            flush()
            body_page = page_number
            sizes = [r.size for r in rows]
            median = statistics.median(sizes) if sizes else 12.0
            threshold = median * 1.2

            page_regions = regions.get(page_number, [])
            kept = [r for r in rows if not any(self._inside(r, bbox) for bbox, _t, _f in page_regions)]
            continued = carried
            borderless = self._borderless_tables(
                kept, continued, tuple(bbox[1] for bbox, _t, _f in page_regions))
            carried = self._carried(kept, borderless, page_regions)
            starts = {t.start: t for t in borderless}
            opening = min(borderless, key=lambda t: t.start) if borderless else None
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
                        # Lines above one grid are not the next one's caption.
                        recent = []
                    continue
                row = kept[i]
                if i in starts:
                    flush()
                    found = starts[i]
                    title = found.title
                    if (not title and continued is not None and found is opening
                            and found.header == continued.header
                            and not any(_SENTENCE.search(r.text) or len(r.cells) > 1 and not self._period_row(r)
                                        for r in kept[:found.start])):
                        # The previous page's statement goes on here under its
                        # header printed again, with no title of its own.
                        title = continued.title
                    caption = " ".join(part for part in (title, found.caption) if part)
                    # Each page's part of a statement is its own table, cited at
                    # its own page. Main joined a table to the last one of the
                    # previous page whenever their headers matched, whatever lay
                    # between, and cited every row at the first page: Best Buy's
                    # statements of earnings, comprehensive income and cash flows
                    # became one table "on page 56" (Decision 93).
                    sections.append(PDFSection(
                        text="\n".join([caption, "  ".join(found.header)]
                                       + ["  ".join(r) for r in found.rows]).strip(),
                        page_number=page_number,
                        page_range=[page_number, page_number],
                        section_heading=heading,
                        section_type="table",
                        is_table=True,
                        table_rows=[found.header] + found.rows,
                        caption=caption,
                        notes=found.notes,
                    ))
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

    def _ruled_with_headers(self, pages: list[list[_Row]], ruled: list[dict]) -> list[dict]:
        """
        The ruled tables to keep: all but those whose grid names no column and
        whose rows the text path reads as a statement under a period header.

        Filings rendered from EDGAR HTML often rule a statement's figures and
        print its years above the rules, outside pdfplumber's table, whose
        first row is then a section label ("ASSETS", "Net revenue:") with empty
        cells beside it: every column was "Column N", around "$" and spacer
        columns. On FinanceBench's dev evidence pages, 23 of the 26 converter
        tables the borderless rules still left headed that way were such grids
        (Decision 93). A grid is left to the text path, as Decision 89 leaves a
        box around text, only when on every page it covers at least half its
        rows fall in borderless tables under a period header; any other ruled
        table is kept as it was.
        """
        unnamed = {id(t) for t in ruled
                   if t.get("headers") and not any(str(c or "").strip() for c in t["headers"][1:])}
        if not unnamed:
            return ruled
        claimed: dict[int, bool] = {}
        for page_number, entries in self._regions_by_page(ruled).items():
            if not any(id(table) in unnamed for _bbox, table, _first in entries) or page_number > len(pages):
                continue
            kept_regions = [bbox for bbox, table, _first in entries if id(table) not in unnamed]
            rows = [r for r in pages[page_number - 1] if not any(self._inside(r, b) for b in kept_regions)]
            headed = [(t.start, t.end) for t in self._borderless_tables(rows) if t.anchors]
            for bbox, table, _first in entries:
                if id(table) not in unnamed:
                    continue
                inside = [n for n, r in enumerate(rows) if self._inside(r, bbox)]
                covered = sum(1 for n in inside if any(start <= n < end for start, end in headed))
                claimed[id(table)] = claimed.get(id(table), True) and bool(inside) and 2 * covered >= len(inside)
        left = [t for t in ruled if not claimed.get(id(t), False)]
        if len(left) < len(ruled):
            logger.info(
                "Ruled grids that name no column left to the text path, which reads their period header",
                extra={"left_to_text": len(ruled) - len(left), "kept": len(left)},
            )
        return left

    @staticmethod
    def _carried(rows: list[_Row], tables: list[_Table], regions: list) -> _Table | None:
        """
        The statement the next page may continue: the page's last borderless
        table, when it is under a period header and nothing but a few short
        lines ("See accompanying notes", a page number) and no ruled table
        follow it on the page.
        """
        if not tables:
            return None
        last = max(tables, key=lambda t: t.end)
        after = rows[last.end + last.note_rows:]
        if not last.anchors or len(after) > 3 or any(len(r.cells) > 1 or len(r.text) >= 80 for r in after):
            return None
        bottom = rows[last.end - 1].bottom
        if any(bbox[1] > bottom for bbox, _t, _f in regions):
            return None
        return last

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
        units = [r.text for r in recent[-2:] if _CAPTION.search(r.text)]
        # The title over a grid on one page, past the header lines printed above
        # its rules (see _title_lines). A grid stitched over pages is left as it
        # was: its rows may run into the next statement, which the first
        # page's title does not name.
        regions = table.get("regions") or []
        bbox = regions[0][1] if regions and regions[0][1] else None
        extent = (bbox[0], bbox[2]) if bbox else None
        # Optional (Decision 88): a title that cannot be read is no title.
        title = fail_open(
            "pdf_table_title", self._title_lines, recent, len(recent), 0, extent, bbox[1] if bbox else None,
            default=[],
        ) if page_range[0] == page_range[-1] else []
        caption = " ".join(title + [u for u in units if u not in title])
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
        """A label followed by at least two values and nothing else."""
        return self._value_cells(row) is not None

    def _value_cells(self, row: _Row) -> list[_Cell] | None:
        """
        The values of a statement row — a label followed by at least two
        values and nothing else — or None for any other row.

        A currency sign printed apart from its amount is joined to it, and a
        line PyMuPDF glued from two cells is read word by word. Statements set
        the "$" flush left in its column and the figure flush right, so the
        first line of a statement and every total read "Net income | $ | 5,070
        $ | 6,046 $ | 5,727" (the sign of the next column glued to the amount
        before it) or "Cash | $ | 2,306,072 | $ | 1,011,315". Neither was a
        value row, so each ended the table there and the figures below lost the
        period header above them — the commonest of the causes behind the 66
        of 85 converter tables on FinanceBench's dev evidence pages headed
        "Column 2", "Column 3" (Decisions 90, 93). Decision 65 held a "$"-join
        back because the extra tables crowded the reranker's candidates;
        Decision 68's one slot per table in the rerank pool took that cost away.
        """
        if len(row.cells) < 2 or not re.search(r"[A-Za-z]", row.cells[0].text):
            return None
        # "(Dollars in millions)  2019  2018" is the header of most statements,
        # not their first line: bare years parse as amounts, and taking the
        # header for a value row left the table with "Column 2" / "Column 3".
        if self._period_row(row):
            return None
        values = self._values(row.cells[1:])
        return values if values is not None and len(values) >= 2 else None

    def _hanging_values(self, row: _Row) -> list[_Cell] | None:
        """
        The figures of a row with no label, set a little lower than the label
        line above them ("TOTAL CURRENT LIABILITIES" over "27,194  26,532"),
        or None.
        """
        if any(re.search(r"[A-Za-z]", c.text) for c in row.cells) or self._period_row(row):
            return None
        return self._values(row.cells)

    def _values(self, cells: list[_Cell]) -> list[_Cell] | None:
        """Cells read as values, signs joined and glued cells cut at their words; None if any is not."""
        pieces: list[_Cell] = []
        for cell in cells:
            if cell.words and len(cell.words) > 1 and not self._is_value(cell.text):
                pieces.extend(_Cell(text, x0, x1) for text, x0, x1 in cell.words)
            else:
                pieces.append(cell)
        values: list[_Cell] = []
        for piece in pieces:
            if values and _CURRENCY_SIGN.match(values[-1].text) and self._is_value(piece.text):
                sign = values.pop()
                piece = _Cell(f"{sign.text} {piece.text}", sign.x0, piece.x1)
            values.append(piece)
        if not values or not all(self._is_value(v.text) for v in values):
            return None
        return values

    @staticmethod
    def _label_row(row: _Row, left: float | None = None) -> bool:
        """
        A section label inside a statement ("Operating expenses:"): one cell,
        short, or ending left of the statement's figures (`left`, the left edge
        of its value columns). A line item's label wrapped over two lines
        ("Adjustments to reconcile net income to net cash provided by operating
        activities:", "(Increase) decrease in prepaid expenses, operating lease
        right-of-use assets and") runs past 80 characters, and ended the table
        there; a line of prose runs on under the figures.
        """
        if len(row.cells) != 1:
            return False
        return len(row.text) < 80 or (left is not None and row.cells[0].x1 < left)

    @staticmethod
    def _period_row(row: _Row) -> bool:
        return PDFProcessor._period_cells(row) is not None

    @staticmethod
    def _period_cells(row: _Row) -> list[_Cell] | None:
        """
        The period cells of a column header — at least two periods and at most
        one other cell, the label column's caption ("(Dollars in millions)") —
        or None for any other row.
        """
        periods: list[_Cell] = []
        others = 0
        for cell in row.cells:
            found = _cell_periods(cell)
            periods.extend(found)
            others += not found
        return periods if len(periods) >= 2 and others <= 1 else None

    def _borderless_tables(self, rows: list[_Row], carried: _Table | None = None,
                           ruled_tops: tuple[float, ...] = ()) -> list[_Table]:
        """
        Runs of statement rows under a period header.

        A run is at least two value rows sharing the same columns (cells
        matched to header columns by their right edge, since amounts are
        right-aligned), optionally interrupted by label-only rows ("Operating
        expenses:"). Without a period header a run needs three rows and its
        columns stay unnamed. A row whose cells cannot be placed in distinct
        columns ends the run — the table is not guessed — unless the next row
        goes on under the same columns; then it stays in the table as text.

        A run with no period row within reach takes the one above it on the
        page that its columns continue under (_inherited_header), or, at the
        top of a page, `carried`: the previous page's statement, when its
        columns sit where the run's do and no ruled table (`ruled_tops`, the
        tops of the page's ruled regions) comes between.
        """
        tables: list[_Table] = []
        titles: dict[int, str] = {}  # header row -> the title over it
        i = floor = 0
        while i < len(rows):
            first = self._value_cells(rows[i])
            if first is None:
                i += 1
                continue
            left = min(c.x0 for c in first)

            label_x1 = rows[i].cells[0].x1
            header_index = None
            header = None
            for j in range(i - 1, max(i - 2 - _MAX_LABEL_ROWS, -1), -1):
                if self._period_row(rows[j]):
                    # A period row whose columns cannot all be told apart is no
                    # header: the run is read as if none were within reach.
                    header = self._header(rows, j, label_x1)
                    header_index = j if header is not None else None
                    break
                # Past the row just above, only section labels ("ASSETS",
                # "Current assets:") may sit between a header and its figures —
                # the layout of every balance sheet.
                if j < i - 1 and not self._label_row(rows[j], left):
                    break
            source = header_index
            borrowed = None
            if source is None:
                source = self._inherited_header(rows, i, first, left)
                if source == -1:
                    source = None
                    if (carried is not None and len(carried.anchors) == len(first)
                            and self._place(first, carried.anchors) is not None
                            and all(top > rows[i].top for top in ruled_tops)):
                        borrowed = carried
                elif source is not None:
                    header = self._header(rows, source, label_x1)
                    if header is None:
                        source = None
            if source is not None:
                header_cells, names, top, skipped = header
                anchors = [c.x1 for c in header_cells]
            elif borrowed is not None:
                anchors, names = list(borrowed.anchors), list(borrowed.header[1:])
            else:
                anchors = [c.x1 for c in first]
                names = [f"Column {n + 2}" for n in range(len(anchors))]

            # Rows between the header and the first figures are section labels;
            # they belong to the table (they used to be skipped with it, and
            # lost). A unit line there ("(In millions, except par value
            # amounts)" under the years) is the table's caption: as a label row
            # its unit was not read, and the table said it gave none.
            data: list[list[str]] = []
            unit_rows: list[str] = []
            for k in range(header_index + 1 if header_index is not None else i, i):
                if len(rows[k].cells) == 1 and _CAPTION.search(rows[k].text) and len(rows[k].text) < 120:
                    unit_rows.append(rows[k].text)
                else:
                    data.append([rows[k].text] + [""] * len(anchors))
            j = i
            while j < len(rows):
                row = rows[j]
                values = self._value_cells(row)
                if values is not None:
                    placed = self._place(values, anchors)
                    if placed is None:
                        # A row the columns cannot hold ("BASIC NET INCOME PER
                        # SHARE  $ 0.29  3 $ 1.51", a footnote marker read as a
                        # figure) stays in the table as text when the statement
                        # goes on under the same columns; otherwise it ends it.
                        following = self._value_cells(rows[j + 1]) if j + 1 < len(rows) else None
                        if (following is None or self._place(following, anchors) is None
                                or not any(any(r[1:]) for r in data)):
                            break
                        data.append([row.text] + [""] * len(anchors))
                        j += 1
                        continue
                    data.append([row.cells[0].text] + placed)
                    left = min(left, *(c.x0 for c in values))
                    j += 1
                    continue
                # Label-only rows inside the statement ("Operating expenses:",
                # "LIABILITIES AND EQUITY" / "Current liabilities:").
                labels = 0
                while (j + labels < len(rows) and labels < _MAX_LABEL_ROWS
                       and self._label_row(rows[j + labels], left)):
                    labels += 1
                if labels and j + labels < len(rows):
                    if self._value_row(rows[j + labels]):
                        data.extend([rows[k].text] + [""] * len(anchors) for k in range(j, j + labels))
                        j += labels
                        continue
                    # A line item whose figures sit a little lower than its
                    # label ("TOTAL CURRENT LIABILITIES" over "27,194  26,532")
                    # is one row; PyMuPDF puts them on two.
                    hanging = self._hanging_values(rows[j + labels])
                    if hanging is not None and len(hanging) == len(anchors):
                        placed = self._place(hanging, anchors)
                        if placed is not None:
                            data.extend([rows[k].text] + [""] * len(anchors) for k in range(j, j + labels - 1))
                            data.append([rows[j + labels - 1].text] + placed)
                            j += labels + 1
                            continue
                break
            value_rows = sum(1 for r in data if any(r[1:]))
            if value_rows < (2 if source is not None or borrowed is not None else 3):
                i = j if j > i else i + 1
                continue

            start = header_index if header_index is not None else i
            label_header = ""
            caption_rows: list[str] = []
            if source is None:
                top, skipped = start, []
            else:
                header_row = rows[source]
                label_header = "" if _cell_periods(header_row.cells[0]) else header_row.cells[0].text
                names, label_header = self._label_column_span(names, label_header)
                if header_index is not None:
                    start = top
            if borrowed is not None:
                label_header = borrowed.header[0]
                caption_rows = [borrowed.caption] if borrowed.caption else []
            else:
                # The unit line over the header; an inherited header brings its own.
                for k in range(top - 1, max(top - 3, -1), -1):
                    if _CAPTION.search(rows[k].text) and len(rows[k].text) < 120:
                        caption_rows.insert(0, rows[k].text)
                    else:
                        break
            # The statement's title, printed above its header (and unit line).
            # A run under a header further up the page is that statement's; one
            # that continues the previous page's statement takes its title when
            # the page prints none of its own.
            # A run whose header was not read has no title read either: the lines
            # right above its first figures are its own section labels.
            title = ""
            if source is not None and header_index is None and source in titles:
                title = titles[source]
            elif source is not None or borrowed is not None:
                above = top - (len(caption_rows) if borrowed is None else 0)
                extent = (rows[i].cells[0].x0, max(anchors))
                title = " ".join(fail_open("pdf_table_title", self._title_lines, rows, above, floor, extent,
                                           default=[]))
                if not title and borrowed is not None:
                    title = borrowed.title
            if source is not None and header_index is not None:
                titles[source] = title
            notes = []
            k = j
            while k < len(rows) and _FOOTNOTE.match(rows[k].text):
                notes.append(rows[k].text)
                k += 1
            tables.append(_Table(
                start=start, end=j, header=[label_header] + names, rows=data,
                caption=" ".join(caption_rows + skipped + unit_rows), notes=notes, note_rows=k - j,
                anchors=anchors if source is not None or borrowed is not None else [],
                title=title,
            ))
            i = floor = k
        # Caption rows stay in the body; a table's start is its header.
        for table in tables:
            while table.start < len(rows) and rows[table.start].text in table.caption:
                table.start += 1
        return tables

    def _inherited_header(self, rows: list[_Row], i: int, first: list[_Cell],
                          left: float) -> int | None:
        """
        The period row a run of figures continues under when none is within
        reach above it: the nearest one above it on the page, reached over
        nothing but rows of figures in as many columns and lines that start
        left of the figures (section labels, a wrapped label, a heading), and
        whose period cells sit over the run's columns. A statement broken by a
        row that could not be placed — a figure glued to its footnote marker,
        an extra column — keeps its header that way. Any other row stops the
        search: a header that could not be read ("Domestic | International",
        a caption over the columns) belongs to the run, and is not another
        table's to give.

        Returns:
            The index of that period row; -1 when the search reached the top of
            the page over nothing that stops it (the run may continue the
            previous page's statement); None otherwise.
        """
        for j in range(i - 1, -1, -1):
            row = rows[j]
            periods = self._period_cells(row)
            if periods is not None:
                if len(periods) == len(first) and self._place(first, [p.x1 for p in periods]) is not None:
                    return j
                return None
            values = self._value_cells(row) or self._hanging_values(row)
            if values is not None:
                if len(values) != len(first):
                    return None
                continue
            if len(row.cells) == 1 and row.cells[0].x0 < left and (
                    len(row.text) < 80 or row.cells[0].x1 < left):
                continue
            return None
        return -1

    def _title_lines(self, rows: list[_Row], index: int, floor: int = 0,
                     extent: tuple[float, float] | None = None, top: float | None = None) -> list[str]:
        """
        The title printed right above a table's header: up to four short lines,
        none a sentence, read upwards from row `index` (exclusive) and no
        further than `floor`, in reading order; with `extent`, the table's left
        and right edges, a line set wholly beside the table (the other column
        of a two-column page) ends it; with `top`, the table's top on its
        page, a line at or below it ends it, as one printed on an earlier
        page: over a ruled grid that opens its page, the rows since the last
        heading end with the previous page's last line ("See accompanying
        notes…", a running footer), and no line of the grid's own page
        comes after it to show the page break.

        A statement's title ("ADOBE SYSTEMS INCORPORATED" / "CONSOLIDATED
        STATEMENTS OF CASH FLOWS") is printed over its column header, often in
        the body's font, so it was not a heading and stayed in the text
        before the table: the table's chunks named no statement, and their
        heading was whatever larger line came last — on statement pages, often
        the previous page's "See accompanying Notes…". On FinanceBench's dev
        questions, five of the eight whose evidence page left the top ten were
        statements whose title and first figures main had kept together in one
        text chunk (Decision 93). Header lines over a ruled grid (its years,
        its span: short pieces, each naming a period) are passed over; a
        multi-cell row, a sentence, a footnote of the table above, or a line
        with no word ends the title.
        """
        found: list[str] = []
        for k in range(index - 1, floor - 1, -1):
            row = rows[k]
            text = row.text.strip()
            if k + 1 < len(rows) and row.top > rows[k + 1].top:
                break  # the previous page: rows since a heading can run across a page break
            if top is not None and row.top >= top:
                break  # the previous page's last lines, over a ruled grid that opens its page
            if extent is not None and all(c.x1 < extent[0] or c.x0 > extent[1] for c in row.cells):
                break  # the other column of the page
            if not found and (self._period_row(row) or not _CAPTION.search(text) and all(
                    (_PERIOD_WORD.search(c.text) or _cell_periods(c)) and not c.text.endswith((".", ":"))
                    and len(_HEADER_WORD.findall(c.text)) <= _MAX_CAPTION_WORDS for c in row.cells)):
                continue  # header lines the columns did not take ("Years Ended", "November 27,")
            words = len(_HEADER_WORD.findall(_PARENTHETICAL.sub(" ", text)))
            if (len(row.cells) != 1 or len(text) > _MAX_TITLE_CHARS or words > _MAX_TITLE_WORDS
                    or _SENTENCE.search(text) or text.endswith((".", ":")) or _FOOTNOTE.match(text)
                    or not _LABEL.search(text)):
                break
            found.append(text)
            if len(found) == _MAX_TITLE_LINES:
                break
        return found[::-1]

    @staticmethod
    def _label_column_span(names: list[str], label: str) -> tuple[list[str], str]:
        """
        The column names and label-column header of a period row whose label
        cell prints the columns' span or day ("Fiscal Years Ended | January 28,
        2017 | ...", "Three months ended March 31, | 2021 | 2020", "At December
        31, | 2021 | 2020"): the span heads every column, as it would printed
        over them. Read from the label column it reached only the markdown
        grid's first cell, so the registry keyed one Best Buy statement twice —
        "FY ending 2017-01" from the grid and the balance date "2017-01-28"
        from its rows (Decision 93) — and a 10-Q's years had no period at all.
        A unit note in the cell ("(In millions)") stays in the label column.
        """
        if not label:
            return names, label
        words = _PARENTHETICAL.sub(" ", label)
        span = " ".join(words.split())
        if (not span or _PERIODS_IN.search(span) or _SENTENCE.search(span)
                or len(_HEADER_WORD.findall(span)) > _MAX_HEADER_WORDS):
            return names, label
        if _SPAN.search(span):
            if any(_SPAN.search(n) for n in names):
                return names, label
        elif _MONTH_DAY.search(span):
            if any(_MONTH_DAY.search(n) or _DATE.search(n) for n in names):
                return names, label
        else:
            return names, label
        unit = " ".join(_PARENTHETICAL.findall(label))
        return [f"{span} {n}" for n in names], unit

    def _header(self, rows: list[_Row], index: int,
                label_x1: float) -> tuple[list[_Cell], list[str], int, list[str]] | None:
        """
        The header a period row gives a table: its period cells, the names
        stacking gives the columns, the index of its first line and the
        label-column lines passed over (see _stacked_names). None when two
        columns would bear the same name. A repeated name is a span line that
        was not read: on FinanceBench's dev filings it named a quarter by its
        year ("2020", "2020 (2)" for four quarters of General Mills' fiscal
        2020) and an earnings release's 13-week column like its 52-week one;
        such a header is no header, and its columns stay unnamed.
        """
        periods = self._period_cells(rows[index])
        if periods is None:
            return None
        names, top, skipped = self._stacked_names(rows, index, periods, label_x1)
        if len(set(names)) < len(names):
            return None
        return periods, names, top, skipped

    def _stacked_names(self, rows: list[_Row], header_index: int, periods: list[_Cell],
                       label_x1: float) -> tuple[list[str], int, list[str]]:
        """
        Column names from the period cells and the header lines over them.

        A span over every column ("Years Ended", "YEAR ENDED MAY 31,"), one
        piece per column ("December 2, | December 3, | November 27,") or one per
        block of columns ("Three months ended June 30, | Six months ended June
        30," over four years; "Pension | Retiree Medical") belongs to the
        columns it sits over. Read alone, a 10-Q's "2022 | 2021 | 2022 | 2021"
        named a quarter and a half-year alike, and an earnings release's "2023
        | 2022" lost the "13 Weeks Ended" and "52 Weeks Ended" that set its two
        statements apart; the registry reads neither as a period (Decision 92).
        At most three lines are read; one line set wholly in the label column
        ("In Millions, Except Per") is passed over, and any other line whose
        pieces do not sit over the columns stops the reading.

        Returns:
            The names, the index of the first header line, and the text of the
            line passed over (kept as caption text, not lost).
        """
        names = [p.text for p in periods]
        start = header_index
        skipped_at = None
        spacing = min((b.x1 - a.x1 for a, b in zip(periods, periods[1:])), default=200.0)
        read = 0
        k = header_index - 1
        while read < _MAX_HEADER_LINES and k >= 0:
            above = rows[k]
            if skipped_at is None and all(c.x1 <= periods[0].x0 for c in above.cells):
                skipped_at = k
                k -= 1
                continue
            prefixes = self._header_prefixes(above, periods, label_x1, 0.3 * spacing)
            if prefixes is None:
                break
            names = [f"{p} {n}" for p, n in zip(prefixes, names)]
            start = k
            read += 1
            k -= 1
        # The passed-over line is part of the header only when a line above it was read.
        skipped = [rows[skipped_at].text] if skipped_at is not None and start < skipped_at else []
        return names, start, skipped

    def _header_prefixes(self, above: _Row, periods: list[_Cell], label_x1: float,
                         tolerance: float) -> list[str] | None:
        """What one header line above the period cells adds to each column's name, or None."""
        if _CAPTION.search(above.text) or any(c.x0 <= label_x1 for c in above.cells):
            return None
        if self._value_cells(above) is not None or self._period_cells(above) is not None:
            return None
        pieces = [piece for cell in above.cells for piece in _pieces(cell, above.size)]
        # A line of a paragraph set over the columns is no caption: a header
        # piece is a few words and not a sentence.
        if any(len(_HEADER_WORD.findall(_PARENTHETICAL.sub(" ", p.text))) > _MAX_HEADER_WORDS
               or _SENTENCE.search(p.text) for p in pieces):
            return None
        if len(pieces) == 1:
            piece = pieces[0]
            # Two spans or two dates in one piece: which column has which?
            if len(_SPAN.findall(piece.text)) > 1 or len(_MONTH_DAY.findall(piece.text)) > 1:
                return None
            # A span names periods; another caption heads every column only when
            # it is short and set across all of them ("Health Care and Life",
            # "As of"), not a title centred on the page or a caption over one
            # block of columns.
            if not _PERIOD_WORD.search(piece.text) and (
                    len(_HEADER_WORD.findall(piece.text)) > _MAX_CAPTION_WORDS
                    or piece.x0 > periods[0].x1 + tolerance or piece.x1 < periods[-1].x0 - tolerance):
                return None
            return [piece.text] * len(periods)
        if len(periods) % len(pieces):
            return None
        width = len(periods) // len(pieces)
        for n, piece in enumerate(pieces):
            block = periods[n * width:(n + 1) * width]
            centre = (piece.x0 + piece.x1) / 2
            if not block[0].x0 - tolerance <= centre <= block[-1].x1 + tolerance:
                return None
        return [pieces[n // width].text for n in range(len(periods))]

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
        kept = [t for t in tables if self._has_row_labels(t) and not self._is_text_frame(t)]
        if len(kept) < len(tables):
            logger.info(
                "Ruled tables without row labels, and boxes around text, left to the text path",
                extra={"dropped": len(tables) - len(kept), "kept": len(kept)},
            )
        return kept

    @staticmethod
    def _is_text_frame(table: dict) -> bool:
        """
        Whether a ruled "table" is a box drawn around running text.

        Exhibits printed from HTML, and word-processor files with page or
        paragraph borders, rule a box around their text, and pdfplumber reports
        the box as a one-column table whose cell holds the whole page with its
        line breaks collapsed. Taken as a table, its region was removed from the
        page text: in 35 of the 232 CUAD contracts tuned on (Decision 89) more
        than half the text was indexed as flattened table cells — no clause
        boundaries, no parent chunks, tables appended after the body, and no
        title line or party clause for the document's identity. A grid with at
        most one non-empty column has no columns to keep apart, so its text is
        read as text.
        """
        rows = [table.get("headers") or [], *(table.get("rows") or [])]
        columns = {i for row in rows for i, cell in enumerate(row) if str(cell or "").strip()}
        return len(columns) <= 1

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
