"""
Splitting an answer into atomic, checkable claims.

Verification works claim by claim: a single unsupported sentence should be named,
not averaged away inside an answer-level score. The splitter keeps what asserts
something about the documents and drops what does not:

- headings, horizontal rules, table header/separator rows;
- lines that are only citation markers;
- list lead-ins ("Key adjustments:", "Atlas Data Holdings (Private Equity
  Consortium)") and "[Sources: ...]" lines that carry no figure;
- fragments too short to assert anything.

A statement of absence ("the data room does not disclose X", "the indemnity
cap is not specified") is kept, and marked: it is the designed way to decline,
but it is also a claim that can be false, and a false one tells the reviewer a
term or a risk does not exist. An entailment model cannot confirm an absence,
so these are put to the judge only (see claim_checker). The items of a list
under a lead-in that says what the evidence lacks ("The data room lacks:")
are statements of absence too.

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
# At most six digits: the page is read with int(), which refuses a string of
# more than 4,300 digits (Python 3.11+), so a marker the model wrote with a
# runaway page number failed the whole request after synthesis (Decision 88).
# No document has a millionth page; a longer number is no page at all.
PAGE_IN_MARKER = re.compile(r"(?:p\.|pg\.|page\s*)(\d{1,6})(?!\d)", re.IGNORECASE)

# Wording that states what the EVIDENCE does not say. Kept separate from the
# synthesizer's usable-answer guard (_DECLINES_TO_ANSWER there) so that changing
# how the verifier treats a decline never loosens what synthesis accepts.
#
# Every alternative is anchored on the evidence or on information being
# available — "the documents do not state", "is not disclosed", "cannot be
# calculated", "is missing from the data room". Two earlier versions got this
# wrong in the fail-open direction. The first matched bare keywords and dropped
# the sentence: "There is no evidence of any pending litigation against the
# Company" and "The buyer will be required to pay a reverse termination fee"
# produced no claim at all. The second anchored the pattern but still dropped
# what matched, so "The indemnity cap is not specified in the merger agreement"
# — false, the cap is $69.6 million — was never checked and did not lower
# confidence. A match now only decides HOW the sentence is checked.
_EVIDENCE_NOUN = (
    r"(?:data\s+room|documents?|context|materials?|sources?|excerpts?|passages?"
    r"|disclosures?|information\s+provided|provided\s+information"
    # A named document is evidence too: "the merger agreement does not state…".
    r"|agreements?|reports?|memo(?:randum)?s?|schedules?|statements?|filings?"
    r"|decks?|presentations?)"
)
_INFO_VERB = (
    r"(?:contain|include|state|specify|disclose|provide|mention|address|give|identify"
    r"|quantify|break\s+out|itemi[sz]e|report|indicate|show|list|discuss|detail)"
)
# "does not explicitly state", "are not fully detailed".
_HEDGE = r"(?:(?:explicitly|expressly|specifically|directly|clearly|fully|separately)\s+)?"
DECLINES_TO_ANSWER = re.compile(
    # The evidence does not contain / state / disclose …
    rf"\b{_EVIDENCE_NOUN}\b[^.;]{{0,120}}\b(?:do(?:es)?|did)\s+not\s+{_HEDGE}{_INFO_VERB}"
    # "The provided agreement contains no provision …", "The data room lacks:".
    # Phrasings measured on the dev half of the live-answer replay
    # (eval/run_answer_replay.py, Decision 72), where correct declines written
    # this way were scored as ordinary claims and flagged "not supported".
    # Only the copy in front of the model or the room as a whole: "The BAA has
    # no assignment restriction" is a term of the contract, not a gap in the
    # evidence, and stays an ordinary claim.
    rf"|\b(?:(?:provided|retrieved|available)\s+{_EVIDENCE_NOUN}|data\s+room|documents"
    rf"|context|materials|excerpts|passages)\b[^.;]{{0,60}}\b(?:contains?|includes?|provides?)\s+no\b"
    rf"|\b{_EVIDENCE_NOUN}\s+(?:currently\s+)?lack(?:s|ed)?\b"
    rf"|\bnone\s+of\s+the\s+(?:provided\s+|retrieved\s+|available\s+)?{_EVIDENCE_NOUN}\b"
    # Neither X nor Y is included in the provided context.
    rf"|\b(?:neither|none|no)\b[^.;]{{0,160}}\b(?:included|contained|found|provided|present"
    rf"|available|disclosed|stated)\s+in\s+(?:the|any|all)\s+(?:provided\s+|retrieved\s+"
    rf"|available\s+)?{_EVIDENCE_NOUN}\b"
    # "Context missing", "Retention pool at closing: Undisclosed".
    rf"|\b{_EVIDENCE_NOUN}\s+(?:is\s+|are\s+)?(?:missing|absent|unavailable|insufficient)\b"
    r"|:\s*undisclosed\b"
    # … is not disclosed / stated / available: information availability.
    rf"|\b(?:is|are|was|were)\s+not\s+{_HEDGE}(?:disclosed|provided|available|specified|stated"
    r"|given|quantified|itemi[sz]ed|broken\s+out|reported|identified|detailed|mentioned"
    r"|addressed|discussed)\b"
    # "Aurora has not disclosed any …", "no figure has been specified": the
    # disclosure is what is negated, not the fact. Narrower than the line
    # above on purpose — "has not been reported to regulators" is a finding.
    r"|\b(?:has|have|had)\s+not\s+(?:been\s+)?(?:disclosed|specified|quantified|itemi[sz]ed)\b"
    r"|\bnot\s+(?:disclosed|provided|stated|specified|available|found|present|included"
    r"|contained|mentioned)\s+in\s+the\b"
    r"|\b(?:cannot|can\s+not|could\s+not)\s+be\s+(?:calculated|computed|determined"
    r"|established|verified|confirmed|derived|quantified)\b"
    r"|\bunable\s+to\s+(?:find|determine|calculate|compute|confirm|verify|locate)\b"
    # "no information on …", "There is no information available regarding …",
    # "There are no mentions of …", "There are no figures provided for …".
    r"|\bno\s+(?:information|mentions?|disclosures?|references?|records?|data|details?"
    r"|breakdowns?|figures)\s+(?:(?:is|are|was|were)\s+)?(?:(?:available|provided|given)\s+)?"
    r"(?:on|of|about|regarding|for|to|is|are|was|were|has|have)\b"
    # "No churn rates or retention figures are provided for Palantir."
    r"|\bno\b[^.;:]{0,80}\b(?:is|are|was|were)\s+(?:provided|disclosed)\s+(?:for|on|about"
    r"|regarding)\b"
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
_WORD_OR_DOT = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.")


def _word_before_final_period(text: str) -> str | None:
    """
    What re.findall(r"([A-Za-z.]+)\\.$", text.rstrip()) finds — the run of
    letters and dots before a closing full stop — read from the end.

    The regex scanned the whole text, and the text grows each time an
    abbreviation joins the next sentence to it, so an answer repeating
    "Inc." was quadratic: 0.2 s at 12,000 characters, about 1.4 s for an
    8,192-token answer (Decision 88).
    """
    end = len(text)
    while end and text[end - 1].isspace():
        end -= 1
    if not end or text[end - 1] != ".":
        return None
    start = end - 1
    while start and text[start - 1] in _WORD_OR_DOT:
        start -= 1
    return text[start:end - 1] or None


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
# checking ("Key risks", "In summary"). Words are counted with _WORD, not
# str.split: "Deductible / Basket Threshold" is three words, not four, and was
# checked — and flagged "not supported" — as a claim (Decision 72). A
# "Label: value" pair keeps the old whitespace count ("Maturity: June 30,
# 2026", "Certainty Level: HIGH ."): counted with _WORD alone these fell under
# the line on the replay's dev half, and a wrong value would never have been
# checked.
_MIN_WORDS = 4
_LABEL_VALUE = re.compile(r"^[^:]{2,60}:\s*\S")

# "Atlas Data Holdings (Private Equity Consortium — DECLINE)": a name and a
# descriptor in brackets, written as the lead-in of the bullets under it. With
# no figure and a title-case name (no lower-case word, so no verb outside the
# brackets) it asserts nothing the bullets do not; "Northstar requires consent
# (not yet obtained)" has a lower-case verb and stays a claim.
_NAME_WITH_DESCRIPTOR = re.compile(r"^(?P<name>[^()]{2,80}?)\s*\([^()]{1,80}\)\s*:?$")
_TITLE_SMALL_WORDS = {"of", "and", "the", "for", "to", "in", "on", "at", "by", "a", "an", "&"}
# "[Sources: 📄 a.txt; 📄 b.txt]" — a list of files, not a statement.
_SOURCES_LINE = re.compile(r"^\[\s*(?:sources?|citations?|references?)\s*:[^\[\]]*\]$", re.IGNORECASE)


def _is_name_lead_in(text: str) -> bool:
    """True for "Name (descriptor)" with a title-case name and no figure."""
    m = _NAME_WITH_DESCRIPTOR.match(text)
    if not m:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'.-]*", m.group("name"))
    return bool(words) and all(w[0].isupper() or w.lower() in _TITLE_SMALL_WORDS for w in words)


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
        is_absence: True when the claim says the evidence does NOT state
            something ("the FY2024 budget is not disclosed").
    """

    index: int
    text: str
    raw: str
    markers: list[str] = field(default_factory=list)
    is_table_row: bool = False
    has_numbers: bool = False
    is_absence: bool = False


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
            last_word = _word_before_final_period(merged[-1])
            if last_word and last_word.lower().rstrip(".") in _ABBREVIATIONS:
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


def _sentence_role(text: str) -> str:
    """
    What a cleaned line is to the verifier.

    Args:
        text: Cleaned claim text.

    Returns:
        "claim" — asserts something about the deal; "absence" — states what the
        evidence does not say; "skip" — a fragment or lead-in that asserts
        nothing.
    """
    if not text or not _WORD.search(text) or _SOURCES_LINE.match(text):
        return "skip"
    if extract_numbers(text):
        return "claim"
    if len(text.split()) < _MIN_WORDS or (
        len(_WORD.findall(text)) < _MIN_WORDS and not _LABEL_VALUE.match(text)
    ):
        return "skip"
    if text.endswith(":") or _is_name_lead_in(text):
        return "skip"
    if DECLINES_TO_ANSWER.search(text):
        return "absence"
    return "claim"


def split_claims(answer: str) -> list[Claim]:
    """
    Splits an answer into atomic claims.

    Args:
        answer: Generated answer text (markdown, with inline citation markers).

    Returns:
        Claims in answer order, statements of absence included (is_absence).
        Empty for an empty answer or one made only of headings and fragments.
    """
    if not answer:
        return []

    claims: list[Claim] = []
    lines = answer.splitlines()
    table_header: list[str] | None = None
    # True while reading the list under a lead-in that states what the
    # evidence lacks ("The data room currently lacks:"). Each item is then a
    # statement of absence even though it reads as a bare noun phrase
    # ("- Quarterly financial statements for FY2024."): scored on its own it
    # is entailed by nothing and was flagged "not supported" (Decision 72).
    absence_list = False

    for li, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _HRULE.match(stripped) or _is_heading(stripped):
            table_header = None if not stripped.startswith("|") else table_header
            if stripped:
                absence_list = False
            continue

        # ── Markdown tables ────────────────────────────────────────────────
        if stripped.startswith("|"):
            absence_list = False
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
            role = _sentence_role(text)
            if role != "skip":
                claims.append(Claim(
                    index=len(claims),
                    text=text,
                    raw=stripped,
                    markers=CITATION_MARKER.findall(stripped),
                    is_table_row=True,
                    has_numbers=bool(extract_numbers(text)),
                    is_absence=role == "absence",
                ))
            continue

        table_header = None
        if stripped.startswith(">"):
            stripped = stripped.lstrip("> ")

        # Citation-only line: attaches to nothing checkable itself.
        if not strip_citations(stripped).strip(" .,;*-"):
            continue

        # Only a paragraph opens or closes the list; a bullet that ends with a
        # colon does not, so an unrelated sibling bullet never inherits it.
        is_item = bool(_BULLET.match(stripped))
        lead_in = False
        line_markers = CITATION_MARKER.findall(stripped)
        for sentence in _split_sentences(stripped):
            text = _clean(sentence)
            role = _sentence_role(text)
            lead_in = role == "skip" and text.endswith(":") and bool(DECLINES_TO_ANSWER.search(text))
            if role == "skip":
                continue
            # A figure is never folded into an absence: it is grounded as usual.
            if role == "claim" and is_item and absence_list and not extract_numbers(text):
                role = "absence"
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
                is_absence=role == "absence",
            ))
        if not is_item:
            absence_list = lead_in

    return claims


def cited_documents(markers: list[str], chunks: list[dict]) -> list[str]:
    """
    The retrieved documents a claim's citation markers name.

    Document level, not chunk level: a figure the cited document states in
    another section, page or table chunk than the marker names is attributed
    to the right source (Decision 78). A file is named by its full name, as the
    synthesis prompt asks, or by its name without the extension.

    Args:
        markers: Marker bodies (text inside the brackets).
        chunks: Context chunks with source_file.

    Returns:
        Source file names, in context order, de-duplicated; empty when no
        marker names a retrieved document.
    """
    bodies = [body.lower() for body in markers]
    named: list[str] = []
    for chunk in chunks:
        source = chunk.get("source_file") or ""
        if not source or source in named:
            continue
        full = source.lower()
        stem = full.rsplit(".", 1)[0] if "." in full else full
        # Word-bounded stem: "financials" must not name "aurora_financials.txt".
        stem_re = re.compile(r"(?<![\w.-])" + re.escape(stem) + r"(?![\w-])")
        if any(full in body or stem_re.search(body) for body in bodies):
            named.append(source)
    return named


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
