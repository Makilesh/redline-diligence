"""
Maps indexed chunk text back to character offsets in its source document,
and scores character spans the way LegalBench-RAG does.

The payload has no character offsets: the text sectioner strips lines and
drops rule lines, and the semantic chunker re-joins sentences with single
spaces, so a chunk is rarely a verbatim substring of the file. It is, almost
always, a substring once whitespace is collapsed on both sides. Alignment is
therefore done on whitespace-collapsed text with an index back to the
original offsets; a chunk whose text was altered beyond whitespace (a dropped
"=====" rule, a table rendered as generated prose) is aligned piecewise and
the pieces that cannot be found are counted, so the share of chunk text that
could not be placed is part of the report rather than a silent error.

Spans are half-open (start, end) offsets into the decoded document text — the
convention of the benchmark's own files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

_WS = re.compile(r"\s+")
# Characters a parser or chunker may substitute; folded on both sides.
_FOLD = str.maketrans({" ": " ", "’": "'", "‘": "'", "“": '"', "”": '"'})

PIECE_CHARS = 48        # piecewise alignment granularity
_MIN_PIECE = 12         # shorter pieces are not placed (false matches)
_SEARCH_WINDOW = 4_000  # how far past the previous piece the next may start


class NormalisedText:
    """Whitespace-collapsed text with a map from each kept character to its offset."""

    def __init__(self, text: str):
        chars: list[str] = []
        offsets: list[int] = []
        pending_space = False
        for i, ch in enumerate(text.translate(_FOLD)):
            if ch.isspace():
                pending_space = bool(chars)
                continue
            if pending_space:
                chars.append(" ")
                offsets.append(i - 1)
                pending_space = False
            chars.append(ch)
            offsets.append(i)
        self.text = "".join(chars)
        self.offsets = offsets
        self.length = len(text)

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        """Original (start, end) for the normalised half-open range [start, end)."""
        return self.offsets[start], self.offsets[end - 1] + 1


def normalise(text: str) -> str:
    return _WS.sub(" ", text.translate(_FOLD)).strip()


@dataclass
class Alignment:
    """Where one chunk's text sits in its source document."""
    spans: list[tuple[int, int]] = field(default_factory=list)
    matched_chars: int = 0   # normalised chunk characters placed in the source
    total_chars: int = 0     # normalised chunk characters
    norm_start: int | None = None  # normalised start (the next chunk's search hint)

    @property
    def coverage(self) -> float:
        return 1.0 if not self.total_chars else self.matched_chars / self.total_chars


def _pieces(text: str, size: int = PIECE_CHARS) -> list[tuple[int, int]]:
    """[start, end) pieces of about `size` characters, cut at spaces."""
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > start else end
        pieces.append((start, end))
        start = end + (1 if end < len(text) and text[end] == " " else 0)
    return pieces


def align(chunk_text: str, source: NormalisedText, hint: int = 0) -> Alignment:
    """
    Aligns one chunk to its source.

    Fast path: the whole collapsed chunk occurs in the collapsed source (first
    occurrence at or after `hint`, else anywhere). Otherwise the chunk is cut
    into ~48-character pieces placed in order (a piece that is not found is
    split in two and retried), each within a window after the previous one;
    consecutive placed pieces form one span.

    Args:
        chunk_text: The chunk's payload text.
        source: The normalised source document.
        hint: Normalised offset to search from first (chunks come in document
            order, so the previous chunk's start is a good hint).

    Returns:
        Alignment with original-offset spans (merged) and match coverage.
    """
    needle = normalise(chunk_text)
    result = Alignment(total_chars=len(needle))
    if not needle:
        return result
    hay = source.text
    at = hay.find(needle, hint)
    if at < 0:
        at = hay.find(needle)
    if at >= 0:
        result.spans = [source.original_span(at, at + len(needle))]
        result.matched_chars = len(needle)
        result.norm_start = at
        return result

    cursor: int | None = None
    norm_spans: list[list[int]] = []

    def place(piece: str) -> None:
        nonlocal cursor
        if len(piece) < _MIN_PIECE:
            return  # too short to place without false matches
        if cursor is None:
            found = hay.find(piece, hint)
            if found < 0:
                found = hay.find(piece)
        else:
            found = hay.find(piece, cursor, cursor + _SEARCH_WINDOW + len(piece))
        if found < 0:
            # The piece straddles an alteration: place its halves instead.
            middle = piece.find(" ", len(piece) // 2)
            if middle < 0 or len(piece) < 2 * _MIN_PIECE:
                return
            place(piece[:middle])
            place(piece[middle + 1:])
            return
        result.matched_chars += len(piece)
        if norm_spans and found <= norm_spans[-1][1] + 8:
            norm_spans[-1][1] = max(norm_spans[-1][1], found + len(piece))
        else:
            norm_spans.append([found, found + len(piece)])
        cursor = found + len(piece)

    for start, end in _pieces(needle):
        place(needle[start:end])
    result.spans = merge_spans(source.original_span(s, e) for s, e in norm_spans)
    if norm_spans:
        result.norm_start = norm_spans[0][0]
    return result


# ==============================================================================
# Span arithmetic and the LegalBench-RAG metrics
# ==============================================================================


def merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted union of half-open spans."""
    out: list[list[int]] = []
    for start, end in sorted(s for s in spans if s[1] > s[0]):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(s, e) for s, e in out]


def span_length(spans: Sequence[tuple[int, int]]) -> int:
    return sum(e - s for s, e in spans)


def overlap_length(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]]) -> int:
    """Characters in both unions (each side merged first)."""
    a, b = merge_spans(a), merge_spans(b)
    i = j = total = 0
    while i < len(a) and j < len(b):
        low, high = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if high > low:
            total += high - low
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


FileSpans = dict  # file -> list[(start, end)]


def char_precision_recall(retrieved: FileSpans, truth: FileSpans) -> tuple[float, float]:
    """
    Character-level precision and recall of retrieved spans against the truth.

    LegalBench-RAG's definition (run_benchmark.QAResult), with one deliberate
    difference: retrieved spans are merged before counting. The benchmark's
    chunkers never overlap; this pipeline's do (sentence overlap between
    neighbours, and a parent contains its children), and without the merge a
    character retrieved twice would count twice — recall could pass 1.

    Args:
        retrieved: file -> spans retrieved.
        truth: file -> ground-truth spans.

    Returns:
        (precision, recall); precision is 0 when nothing was retrieved.
    """
    retrieved_total = sum(span_length(merge_spans(v)) for v in retrieved.values())
    truth_total = sum(span_length(merge_spans(v)) for v in truth.values())
    relevant = sum(overlap_length(retrieved.get(f, []), spans) for f, spans in truth.items())
    precision = relevant / retrieved_total if retrieved_total else 0.0
    recall = relevant / truth_total if truth_total else 0.0
    return precision, recall


def touches(spans: Sequence[tuple[int, int]], target: tuple[int, int]) -> bool:
    """True if any span shares at least one character with the target."""
    return any(min(e, target[1]) > max(s, target[0]) for s, e in spans)
