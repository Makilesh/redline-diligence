"""
Splitting an answer into atomic, checkable claims.

Verification works claim by claim: a single unsupported sentence should be named,
not averaged away inside an answer-level score. The splitter keeps what asserts
something about the documents and drops what does not:

- headings, horizontal rules, table header/separator rows;
- lines that are only citation markers;
- list lead-ins ("Key adjustments:") that carry no figure;
- statements of absence ("the data room does not disclose X") — they assert
  that something is NOT in the evidence, which an entailment check against the
  evidence cannot confirm, and they are the designed way to decline;
- fragments too short to assert anything.

Each claim remembers the citation markers that support it: its own, or — when
a sentence has none — the nearest marker in the same paragraph, which is how
synthesis answers usually cite ("Sentence one. Sentence two [source].").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.verification.numeric_grounding import extract_numbers, strip_citations

# A citation marker is any bracketed group containing a pipe — the format the
# synthesis prompt specifies:
#   [📄 FileName | FiscalYear | p.PageNum | Section | Version]
#   [📄 FileName | Section]            (sources without pages, e.g. .txt)
# Matching on the pipe avoids colliding with ordinary markdown links.
CITATION_MARKER = re.compile(r"\[([^\[\]]*\|[^\[\]]*)\]")
PAGE_IN_MARKER = re.compile(r"(?:p\.|pg\.|page\s*)(\d+)", re.IGNORECASE)

# Wording that declines rather than asserts: a statement about what the
# EVIDENCE does not say, which an entailment check against that evidence cannot
# confirm and which is the designed way to decline. Kept separate from the
# synthesizer's usable-answer guard (_DECLINES_TO_ANSWER there) so that what
# the verifier skips never loosens what synthesis accepts.
#
# Every alternative is anchored on the evidence or on information being
# available — "the documents do not state", "is not disclosed", "cannot be
# calculated", "is missing from the data room". The earlier pattern matched
# bare keywords, so assertions about the deal were exempt from verification
# altogether: "There is no evidence of any pending litigation against the
# Company", "The buyer will be required to pay a reverse termination fee",
# "Management considers the insurance coverage insufficient" and "The key
# customer consent is missing from the executed agreement" each produced no
# claim, and a fabricated one passed unchecked.
_EVIDENCE_NOUN = (
    r"(?:data\s+room|documents?|context|materials?|sources?|excerpts?|passages?"
    r"|disclosures?|information\s+provided|provided\s+information)"
)
_INFO_VERB = (
    r"(?:contain|include|state|specify|disclose|provide|mention|address|give|identify"
    r"|quantify|break\s+out|itemi[sz]e|report|indicate|show)"
)
DECLINES_TO_ANSWER = re.compile(
    # The evidence does not contain / state / disclose …
    rf"\b{_EVIDENCE_NOUN}\b[^.;]{{0,120}}\b(?:do(?:es)?|did)\s+not\s+{_INFO_VERB}"
    # Neither X nor Y is included in the provided context.
    rf"|\b(?:neither|none|no)\b[^.;]{{0,160}}\b(?:included|contained|found|provided|present"
    rf"|available|disclosed|stated)\s+in\s+(?:the|any|all)\s+(?:provided\s+|retrieved\s+"
    rf"|available\s+)?{_EVIDENCE_NOUN}\b"
    # "Context missing", "Retention pool at closing: Undisclosed".
    rf"|\b{_EVIDENCE_NOUN}\s+(?:is\s+|are\s+)?(?:missing|absent|unavailable|insufficient)\b"
    r"|:\s*undisclosed\b"
    # … is not disclosed / stated / available: information availability.
    r"|\b(?:is|are|was|were)\s+not\s+(?:disclosed|provided|available|specified|stated|given"
    r"|quantified|itemi[sz]ed|broken\s+out|reported|identified|detailed)\b"
    r"|\bnot\s+(?:disclosed|provided|stated|specified|available|found|present|included"
    r"|contained|mentioned)\s+in\s+the\b"
    r"|\b(?:cannot|can\s+not|could\s+not)\s+be\s+(?:calculated|computed|determined"
    r"|established|verified|confirmed|derived|quantified)\b"
    r"|\bunable\s+to\s+(?:find|determine|calculate|compute|confirm|verify|locate)\b"
    r"|\bno\s+(?:information|mention|disclosure|reference|record|data|details?|breakdown)"
    r"\s+(?:on|of|about|regarding|for|is|are|was|were|has|have)\b"
    r"|\b(?:insufficient|not\s+sufficient)\s+(?:information|data|context|evidence|detail)s?\b"
    rf"|\bno\s+evidence\b[^.;]{{0,40}}\bin\s+the\s+(?:provided\s+|retrieved\s+|available\s+)?"
    rf"{_EVIDENCE_NOUN}\b"
    # "… is missing" / "is missing from the data room" — but not "is missing
    # from the executed agreement", which is a finding about the agreement.
    rf"|\b(?:is|are|remains?)\s+(?:missing|absent)(?:\s+from\s+(?:the|any|all)\s+(?:provided\s+"
    rf"|retrieved\s+|available\s+)?{_EVIDENCE_NOUN}\b|(?!\s+from\b))"
    r"|\bmissing\s+(?:information|data|details?|documents?|context|figures?)\b"
    r"|\b(?:would|will)\s+(?:need|be\s+needed|be\s+required)\s+to\s+(?:answer|confirm|determine"
    r"|quantify|calculate|compute|verify|assess|establish)\b"
    r"|\b(?:is|are)\s+(?:needed|required)\s+to\s+(?:answer|confirm|determine|quantify|calculate"
    r"|compute|verify)\b",
    re.IGNORECASE,
)

# Abbreviations that end in a period without ending a sentence.
_ABBREVIATIONS = {
    "inc", "corp", "ltd", "co", "llc", "plc", "no", "nos", "vs", "v", "e.g", "i.e",
    "etc", "approx", "u.s", "u.k", "mr", "ms", "dr", "st", "sec", "art", "fig",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
    "dec", "p", "pp", "ref", "est", "avg", "al",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[*A-Z0-9$€£])")
_LEADING_MARKERS = re.compile(r"^\s*(?:\[[^\[\]]*\|[^\[\]]*\]\s*[.,;]?\s*)+")
_BULLET = re.compile(r"^\s*(?:[-*•+]|\d{1,2}[.)])\s+")
# Single underscores are left alone: they occur in file names and identifiers far
# more often than as emphasis in model output.
_MARKDOWN_EMPHASIS = re.compile(r"(\*\*|__|\*|`)")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
_HRULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,}|={3,})\s*$")
_WORD = re.compile(r"[A-Za-z]{2,}")

# Below this many words a line with no figure does not assert anything worth
# checking ("Key risks", "In summary").
_MIN_WORDS = 4


@dataclass
class Claim:
    """
    One checkable statement from the answer.

    Attributes:
        index: Position in the answer's claim list.
        text: Clean claim text — citation markers and markdown stripped.
        raw: The claim as written, markers included.
        markers: Citation marker bodies supporting this claim.
        is_table_row: True when the claim was a markdown table row.
        has_numbers: True when the claim states at least one figure.
    """

    index: int
    text: str
    raw: str
    markers: list[str] = field(default_factory=list)
    is_table_row: bool = False
    has_numbers: bool = False


def _clean(text: str) -> str:
    """Strips citation markers, markdown emphasis and bullets; collapses spaces."""
    text = strip_citations(text)
    text = _BULLET.sub("", text)
    text = _MARKDOWN_EMPHASIS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip()


def _split_sentences(line: str) -> list[str]:
    """
    Splits one paragraph into sentences, keeping citations with their sentence.

    Models often put the marker after the full stop ("…growth. [📄 file | p.3]"),
    which a plain splitter would hand to the NEXT sentence. Leading markers are
    therefore moved back onto the sentence they follow.

    Markers are masked while splitting: live markers quote section text such as
    "…114% in FY2022. Gross", whose full stop would otherwise split the marker.
    """
    saved: list[str] = []

    def _mask(m: re.Match) -> str:
        saved.append(m.group(0))
        return f"[\x00{len(saved) - 1}|\x00]"

    masked = CITATION_MARKER.sub(_mask, line)
    pieces = [
        re.sub(r"\[\x00(\d+)\|\x00\]", lambda m: saved[int(m.group(1))], p)
        for p in _SENTENCE_BOUNDARY.split(masked)
    ]
    merged: list[str] = []
    for piece in pieces:
        if merged:
            prev = merged[-1].rstrip()
            last_word = re.findall(r"([A-Za-z.]+)\.$", prev)
            if last_word and last_word[0].lower().rstrip(".") in _ABBREVIATIONS:
                merged[-1] = f"{merged[-1]} {piece}"
                continue
            lead = _LEADING_MARKERS.match(piece)
            if lead:
                merged[-1] = f"{merged[-1]} {lead.group(0).strip()}"
                piece = piece[lead.end():]
                if not piece.strip():
                    continue
        merged.append(piece)
    return [p.strip() for p in merged if p.strip()]


def _is_heading(line: str) -> bool:
    """True for markdown headings and whole-line bold labels without figures."""
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    if re.fullmatch(r"\*\*[^*]+\*\*:?", stripped) and not extract_numbers(stripped):
        return True
    return False


def _table_cells(line: str) -> list[str]:
    """Splits a markdown table row into cleaned cells."""
    body = line.strip().strip("|")
    return [_clean(c) for c in body.split("|")]


def _is_checkable(text: str, raw: str) -> bool:
    """
    True when a cleaned line asserts something worth verifying.

    Args:
        text: Cleaned claim text.
        raw: Original text, markers included.

    Returns:
        False for fragments, lead-ins and statements of absence.
    """
    return _sentence_role(text) == "claim"


def _sentence_role(text: str) -> str:
    """
    What a cleaned line is to the verifier.

    Args:
        text: Cleaned claim text.

    Returns:
        "claim" — asserts something checkable; "decline" — states what the
        evidence does not say (not checked, but counted); "skip" — a fragment
        or lead-in that asserts nothing.
    """
    if not text or not _WORD.search(text):
        return "skip"
    if extract_numbers(text):
        return "claim"
    if len(text.split()) < _MIN_WORDS:
        return "skip"
    if text.endswith(":"):
        return "skip"
    if DECLINES_TO_ANSWER.search(text):
        return "decline"
    return "claim"


def split_claims(answer: str, declined: list[str] | None = None) -> list[Claim]:
    """
    Splits an answer into atomic claims.

    Args:
        answer: Generated answer text (markdown, with inline citation markers).
        declined: If given, the statements of absence that were set aside are
            appended to it — they are not verified, and a reader of the result
            should be able to see how many sentences that was.

    Returns:
        Checkable claims in answer order. Empty for an empty answer or one that
        only declines.
    """
    if not answer:
        return []

    def checkable(text: str) -> bool:
        role = _sentence_role(text)
        if role == "decline" and declined is not None:
            declined.append(text)
        return role == "claim"

    claims: list[Claim] = []
    lines = answer.splitlines()
    table_header: list[str] | None = None

    for li, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _HRULE.match(stripped) or _is_heading(stripped):
            table_header = None if not stripped.startswith("|") else table_header
            continue

        # ── Markdown tables ────────────────────────────────────────────────
        if stripped.startswith("|"):
            if _TABLE_SEPARATOR.match(stripped):
                continue
            nxt = lines[li + 1].strip() if li + 1 < len(lines) else ""
            if _TABLE_SEPARATOR.match(nxt):
                table_header = _table_cells(stripped)
                continue
            cells = _table_cells(stripped)
            if not any(cells):
                continue
            # "Label: FY2023 $198.4M; FY2022 $172.3M" reads as a statement,
            # which both the NLI model and the judge handle far better than a
            # bare pipe-delimited row.
            label, rest = cells[0], cells[1:]
            parts = []
            for ci, cell in enumerate(rest, start=1):
                if not cell:
                    continue
                head = table_header[ci] if table_header and ci < len(table_header) else ""
                parts.append(f"{head} {cell}".strip())
            text = f"{label}: " + "; ".join(parts) if parts else label
            if checkable(text):
                claims.append(Claim(
                    index=len(claims),
                    text=text,
                    raw=stripped,
                    markers=CITATION_MARKER.findall(stripped),
                    is_table_row=True,
                    has_numbers=bool(extract_numbers(text)),
                ))
            continue

        table_header = None
        if stripped.startswith(">"):
            stripped = stripped.lstrip("> ")

        # Citation-only line: attaches to nothing checkable itself.
        if not strip_citations(stripped).strip(" .,;*-"):
            continue

        line_markers = CITATION_MARKER.findall(stripped)
        for sentence in _split_sentences(stripped):
            text = _clean(sentence)
            if not checkable(text):
                continue
            markers = CITATION_MARKER.findall(sentence)
            if not markers and line_markers:
                # Inherit the nearest marker in the paragraph: the next one if
                # the sentence precedes it, otherwise the last one before it.
                pos = stripped.find(sentence)
                after = CITATION_MARKER.search(stripped, pos + len(sentence)) if pos >= 0 else None
                markers = [after.group(1)] if after else [line_markers[-1]]
            claims.append(Claim(
                index=len(claims),
                text=text,
                raw=sentence,
                markers=markers,
                has_numbers=bool(extract_numbers(text)),
            ))

    return claims


def resolve_cited_chunks(markers: list[str], chunks: list[dict]) -> list[int]:
    """
    Maps citation marker bodies to indices of the chunks they name.

    Same rule as the synthesizer's citation selection: the file must match, and
    a page in the marker narrows to that page. Markers without a page — the
    `[file | Section]` form used for unpaginated sources — match every chunk of
    that file.

    Args:
        markers: Marker bodies (text inside the brackets).
        chunks: Context chunks with source_file / page_number.

    Returns:
        Chunk indices in retrieval order, de-duplicated.
    """
    parsed = []
    for body in markers:
        page = PAGE_IN_MARKER.search(body)
        parsed.append((body.lower(), int(page.group(1)) if page else None))

    selected = []
    for ci, chunk in enumerate(chunks):
        source = (chunk.get("source_file") or "").lower()
        if not source:
            continue
        for body, page in parsed:
            if source not in body:
                continue
            chunk_page = chunk.get("page_number")
            if page is not None and chunk_page not in (None, "", 0):
                try:
                    if int(chunk_page) != page:
                        continue
                except (TypeError, ValueError):
                    pass
            selected.append(ci)
            break
    return selected
