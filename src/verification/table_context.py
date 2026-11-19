"""
A table row as a reader needs it — for the figure veto (claim_checker.FigureVeto).

The veto scores a claim whose every figure numeric grounding confirmed against
the passage each figure came from. When that passage is a table row, the row
alone ("Income Tax Expense  $15.5  $11.5") does not say which column is which
year, what unit the amounts are in, or whose statement it is — and a model
asked whether it supports "Income tax expense was $15.5 million in FY2023"
says no. Decision 64 measured exactly that: the correct claims FactCG's veto
sent to the judge were mostly table rows read without their column headers.

So a row is shown with the lines a reader would look at (Decision 90):

    title       the chunk's first line, when it is a heading rather than a
                table row or a sentence (a statement title, a company name)
    unit line   the nearest line above the row that states a unit or scale
                ("(in millions of USD)", the converter's "Values are … in USD
                millions.")
    header      the column-header line numeric grounding read the figure's
                period from — the line that made the figure "confirmed" for
                that period, from this chunk or another of the same file
    row         the whole line(s) the figure sits on

A claim whose figures sit in several rows ("Total assets were $560.4 million
and total liabilities were $288.0 million") can be shown them together, in one
window (joined_window): one row alone holds half of such a claim.

Every line is the document's text, whitespace collapsed; nothing is rewritten
or realigned. Only table rows get this: a line in a chunk ingestion marked as a
table, or a line with a column gap anywhere. A "Label: value" line in prose
("- Offer: $55 - $60 per share") keeps its sentence windows, where the party
or the clause it belongs to is. This parses counterparty-supplied text, so it
walks lines with plain string operations and linear regexes.
"""

from __future__ import annotations

from src.verification.numeric_grounding import (
    _COLUMN_GAP_RE,
    SCALE_HINT_RE,
    _is_rule_line,
    extract_numbers,
)

# A first line longer than this is prose, not a title; a unit line or header
# longer than this is not one either (the converter's unit note is ~20 words).
TITLE_MAX_WORDS = 20
CONTEXT_MAX_WORDS = 40


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _columned(line: str) -> bool:
    """True for a line set out in columns (a gap of 2+ spaces, a tab or a pipe)."""
    return bool(_COLUMN_GAP_RE.search(line.strip()))


def _is_sentence(line: str) -> bool:
    """A line of prose ("Results improved…, as the table below shows."), not a
    heading: eight words or more ending in a full stop. "AURORA TECHNOLOGIES
    INC." is a heading."""
    text = line.strip()
    return text.endswith((".", "!", "?")) and len(text.split()) >= 8


def is_table_chunk(chunk: dict) -> bool:
    """True for a chunk ingestion indexed as a table (any representation)."""
    return chunk.get("is_table") in (1, True, "1") or str(chunk.get("content_type") or "").startswith("table")


def headed_window(body: str, row_span: tuple[int, int], header_line: str = "",
                  in_table: bool = True) -> str:
    """
    The row at `row_span` of `body`, with its title, unit line and header.

    Args:
        body: A chunk's evidence text (evidence.chunk_evidence_text), the text
            numeric grounding parsed.
        row_span: (start, end) of the row's line(s) in `body`
            (NumericCheck.evidence_row).
        header_line: The column-header line grounding read the period from
            (NumericCheck.evidence_header), "" when none did.
        in_table: Whether the chunk is a table chunk (is_table_chunk).

    Returns:
        Newline-joined lines, the row last; "" when the span does not lie
        inside `body`, holds no text, or is not a table row — the caller then
        uses the sentence windows.
    """
    parts = headed_parts(body, row_span, header_line, in_table)
    if parts is None:
        return ""
    context, row = parts
    return "\n".join(dict.fromkeys([*context, row]))


def joined_window(rows: list[tuple[int, str, tuple[int, int], str, bool]]) -> str:
    """
    One window for a claim whose figures sit in several table rows.

    A claim joining two rows ("Total assets were $560.4 million and total
    liabilities were $288.0 million") is only supported by a passage holding
    both; one window per row holds half of it each, and FactCG scored such
    correct claims below the veto line (Decision 90's integration review). So
    the rows are shown together: per chunk, in the order the claim's figures
    first reach it, its title, unit line and header line(s), then the claim's
    rows of that chunk in document order. Every line appears once.

    Args:
        rows: (chunk key, evidence text, row span, header line, is a table
            chunk) per distinct row, in the order the claim's figures reach
            them.

    Returns:
        Newline-joined lines; "" when no row is a table row.
    """
    by_chunk: dict[int, list[tuple[int, list[str], str]]] = {}
    for key, body, span, header, in_table in rows:
        parts = headed_parts(body, span, header, in_table)
        if parts is not None:
            by_chunk.setdefault(key, []).append((span[0], *parts))
    lines: list[str] = []
    for items in by_chunk.values():
        for _, context, _ in items:
            lines.extend(context)
        lines.extend(row for _, _, row in sorted(items, key=lambda item: item[0]))
    return "\n".join(dict.fromkeys(lines))


def headed_parts(body: str, row_span: tuple[int, int], header_line: str = "",
                 in_table: bool = True) -> tuple[list[str], str] | None:
    """
    The pieces of headed_window: (title / unit / header lines, the row), or
    None when the span is not a table row of `body`.
    """
    start, end = row_span
    if not (0 <= start < end <= len(body)):
        return None
    raw_row = body[start:end]
    row = _collapse(raw_row)
    if not row or not (in_table or _columned(raw_row)):
        return None

    above = body[:start].splitlines()
    lines: list[str] = []

    title = next((ln for ln in above if ln.strip()), "")
    if (
        title.strip()
        and not _is_rule_line(title)
        and not _columned(title)
        and len(title.split()) <= TITLE_MAX_WORDS
        and not _is_sentence(title)
    ):
        lines.append(_collapse(title))

    # Not another row of figures that happens to name a unit: that would show
    # the model a second row's values.
    unit = next(
        (ln for ln in reversed(above)
         if SCALE_HINT_RE.search(ln) and not (_columned(ln) and extract_numbers(ln, claims_only=False))),
        "",
    )
    if unit.strip() and len(unit.split()) <= CONTEXT_MAX_WORDS:
        lines.append(_collapse(unit))

    header = _collapse(header_line)
    if header and len(header.split()) <= CONTEXT_MAX_WORDS:
        lines.append(header)

    return lines, row
