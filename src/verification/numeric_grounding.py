"""
Deterministic numeric grounding.

Every figure an answer states is extracted, normalised to a plain magnitude
(`$452.8 million` -> 452_800_000) and looked up among the figures present in
the retrieved context, normalised the same way. A figure is:

- grounded            — the same value appears in the context (within rounding);
- derived             — not verbatim, but reproducible by one arithmetic step
                        (difference, sum, ratio, percentage change) from two
                        figures the answer itself states AND that are grounded;
- derived_unverified  — not reproducible, but the sentence presents it as a
                        computation ("an increase of", "combined", "approximately"),
                        so it is reported as a warning rather than a fabrication;
- unsupported         — none of the above.

Why deterministic: the previous validator asked a lite model whether a stronger
model's numbers were right. That check cost an LLM call per query and was
unreliable in exactly the place it matters — a lite model will wave through a
transposed digit. String-level grounding cannot be talked out of a mismatch.

Why scale-agnostic matching: data-room tables state their unit once in a header
("(in millions of USD)") and print bare figures ("$452.8"), while answers write
"$452.8 million" or "$452,800,000". When one side of a comparison carries no
explicit unit, the other is allowed to differ by a power of 1,000.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, asdict
from typing import Iterable

# ─── Extraction ────────────────────────────────────────────────────────────────

_SCALE_WORDS = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
}

# One pattern for every figure shape found in the sample data room and in the
# answers recorded in RESULTS.md. Group order matters: the scale must be tried
# before the unit so "12.0M" is a scaled count, not "12.0" followed by noise.
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])                                   # not the tail of FY2023, p.3, v2
    (?P<cur>US\$|USD\s?|EUR\s?|GBP\s?|\$|€|£)?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?:
        (?P<scale_attached>MM|mm|[Bb]n|BN|mn|M|B|K|k|m|b)(?![A-Za-z])
      | \s?(?P<scale_word>thousand|million|billion|trillion|mn|bn|tn)s?\b
      | \s(?P<scale_spaced>MM|M|B|K)\b          # "$452.8 MM" — currency only
    )?
    (?P<cur_suffix>\s?(?:USD|EUR|GBP)\b)?
    (?P<unit>
        \s?%
      | \s?(?:percentage\s+points?|ppts?|pp)\b
      | \s?(?:percent|per\s?cent)\b
      | \s?(?:bps|basis\s+points?)\b
      | [x×](?![A-Za-z])
    )?
    (?![\w])                                      # rejects 280G, 1st, 3rd
    """,
    re.VERBOSE,
)

# Citation markers carry page numbers, fiscal years and — in live output — whole
# section titles such as "Revenue Growth: 17.0% YoY". None of that is a claim.
_CITATION_MARKER_RE = re.compile(r"\[[^\[\]]*\|[^\[\]]*\]")

# Figures that are references, not quantities: "Section 4.3", "Note 5", "p. 3".
# A leading reference word disqualifies the number that follows it.
_REFERENCE_PREFIX_RE = re.compile(
    r"(?:\b(?:section|sections|§|article|note|notes|schedule|exhibit|item|clause|"
    r"rule|page|pages|pp?\.|appendix|annex|tab|row|sheet|slide|step|tier|level|"
    r"phase|series|form|reg\.?|regulation)\s*$)",
    re.IGNORECASE,
)

# Month-name dates: the day number is not a figure.
_MONTH_BEFORE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*$",
    re.IGNORECASE,
)

_NUMERIC_DATE_RE = re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b")


@dataclass(frozen=True)
class NumericMention:
    """
    One figure found in text, normalised.

    Attributes:
        raw: The matched text, e.g. "$452.8 million".
        value: Magnitude with any explicit scale applied (sign dropped — answers
            say "declined by $1.8M" where tables print "($1.8)").
        kind: currency | percent | pp | bps | multiple | count | plain.
        explicit_scale: True when the text itself stated the unit (M, million).
        scale: The explicit scale factor (1.0 when none).
        decimals: Digits after the decimal point as printed — drives the
            rounding tolerance.
        start: Offset in the source text.
        end: End offset in the source text.
    """

    raw: str
    value: float
    kind: str
    explicit_scale: bool
    scale: float
    decimals: int
    start: int
    end: int

    @property
    def half_unit(self) -> float:
        """Half of the last printed digit, in normalised units — rounding slack."""
        return 0.5 * (10 ** -self.decimals) * self.scale

    @property
    def is_claim(self) -> bool:
        """
        True for figures an answer asserts as a quantity.

        Plain integers without separators or units ("284", "5", "12") are
        excluded: in prose they are overwhelmingly list numbers, clause numbers,
        day counts and headcounts, and treating them as financial claims
        produces noise that trains reviewers to ignore the check.
        """
        return self.kind != "plain"


def strip_citations(text: str) -> str:
    """
    Removes inline citation markers, preserving offsets' meaning for the rest.

    Args:
        text: Answer text.

    Returns:
        Text with every `[... | ...]` marker replaced by a space.
    """
    return _CITATION_MARKER_RE.sub(" ", text or "")


def _is_year(num: str, has_unit: bool) -> bool:
    """True for a bare four-digit 19xx/20xx integer used as a year."""
    return (
        not has_unit
        and len(num) == 4
        and num.isdigit()
        and num[:2] in ("19", "20")
    )


def extract_numbers(text: str, claims_only: bool = True) -> list[NumericMention]:
    """
    Extracts every figure from text.

    Args:
        text: Text to scan. Citation markers should already be stripped when
            scanning an answer (see strip_citations).
        claims_only: When True, drop "plain" numbers (no unit, scale, or
            thousands separator). Answers are scanned with True; context is
            scanned with False, because tables print bare figures.

    Returns:
        NumericMention list in text order.
    """
    if not text:
        return []

    # Blank out numeric dates first so "12/31/2023" does not yield 12, 31, 2023.
    scrubbed = _NUMERIC_DATE_RE.sub(lambda m: " " * len(m.group(0)), text)

    mentions: list[NumericMention] = []
    for m in _NUMBER_RE.finditer(scrubbed):
        num = m.group("num")
        cur = (m.group("cur") or "").strip() or (m.group("cur_suffix") or "").strip()
        scale_token = (
            m.group("scale_attached") or m.group("scale_word") or m.group("scale_spaced")
        )
        unit = (m.group("unit") or "").strip().lower()

        # Lower-case single-letter scales are only a scale next to a currency:
        # "5m" is as likely minutes or metres; "$5m" is five million. A spaced
        # letter ("12 M") is likewise only trusted after a currency.
        if not cur and (
            scale_token in ("m", "b", "k") or m.group("scale_spaced")
        ):
            # Re-read without the scale: still a figure, just unscaled.
            scale_token = None

        has_unit = bool(cur or scale_token or unit)
        if _is_year(num, has_unit):
            continue

        prefix = scrubbed[max(0, m.start() - 24): m.start()]
        if _REFERENCE_PREFIX_RE.search(prefix) and not (cur or unit):
            continue
        if _MONTH_BEFORE_RE.search(prefix) and not has_unit:
            continue

        mantissa = float(num.replace(",", ""))
        decimals = len(num.split(".")[1]) if "." in num else 0
        scale = _SCALE_WORDS.get(scale_token.lower(), 1.0) if scale_token else 1.0

        # "percentage points" must be tested before "percent", which prefixes it.
        if unit.startswith("percentage") or unit.startswith("pp"):
            kind = "pp"
        elif unit.startswith("%") or unit.startswith("per"):
            kind = "percent"
        elif unit.startswith("bps") or unit.startswith("basis"):
            kind = "bps"
        elif unit in ("x", "×"):
            kind = "multiple"
        elif cur:
            kind = "currency"
        elif scale_token or "," in num:
            kind = "count"
        else:
            kind = "plain"

        mention = NumericMention(
            raw=m.group(0).strip(),
            value=abs(mantissa) * scale,
            kind=kind,
            explicit_scale=scale_token is not None,
            scale=scale,
            decimals=decimals,
            start=m.start(),
            end=m.end(),
        )
        if claims_only and not mention.is_claim:
            continue
        mentions.append(mention)

    return mentions


# ─── Units and periods a document states ──────────────────────────────────────

# A unit header: "(in millions of USD)", "$ in thousands", "USD m". Shared with
# the financial verifier's registry so both read a table's unit the same way.
SCALE_HINT_RE = re.compile(
    r"\(\s*(?:\$|USD|US\$|EUR|GBP)?\s*(?:in\s+)?(thousands|millions|billions)\b"
    r"|\bin\s+(thousands|millions|billions)\s+of\b"
    r"|\b(?:USD|\$)\s*(thousands|millions|billions|000s|mm|m)\b",
    re.IGNORECASE,
)
SCALE_HINT_WORDS = {
    "thousands": 1e3, "000s": 1e3,
    "millions": 1e6, "mm": 1e6, "m": 1e6,
    "billions": 1e9,
}

# A fiscal period or a calendar year: FY2023, FY'23, Q3 FY2023, fiscal 2023,
# "December 31, 2023". Docket-style numbers ("2023-CV-04821") are not periods.
PERIOD_RE = re.compile(
    r"\b(?:(?:Q[1-4]|H[12])\s?)?"
    r"(?:FY\s?'?(?P<fy>\d{4}|\d{2})|fiscal(?:\s+year)?\s+(?P<fiscal>(?:19|20)\d{2})"
    r"|(?P<year>(?:19|20)\d{2}))[EA]?"
    r"(?!\w|-[A-Za-z]{2}-|\.\d)",
    re.IGNORECASE,
)

# A figure quoted per unit is never "in millions", whatever the table header says.
_PER_UNIT_RE = re.compile(
    r"per[\s-](?:share|seat|unit|user|employee)|\bEPS\b|share\s+price|closing\s+price"
    r"|\bVWAP\b|52-week|current\s+price|price\s+floor",
    re.IGNORECASE,
)

# Wording that ties a figure to "the other period" without naming it:
# "up from $387.1M", "against $72.1M a year earlier".
_COMPARISON_LEAD_RE = re.compile(
    r"\b(?:from|versus|vs\.?|against|compared\s+(?:with|to)|prior|previous(?:ly)?)\s*[($]*$",
    re.IGNORECASE,
)
_COMPARISON_TRAIL_RE = re.compile(
    r"^\W*(?:\w+\s+){0,2}(?:a\s+year\s+(?:earlier|before|ago)|the\s+(?:year|period)\s+before"
    r"|(?:in\s+)?the\s+(?:prior|previous)\s+(?:year|period))",
    re.IGNORECASE,
)

_RULE_LINE_RE = re.compile(r"^\s*[=\-_*~|:\s]{4,}\s*$")
_COLUMN_GAP_RE = re.compile(r"\S(?:\s{2,}|\t|\s*\|\s*)\S")
_LABEL_COLON_RE = re.compile(r"^\s*[^:\n]{2,60}:\s")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'$])")
_BINDING_WORD_RE = re.compile(r"[a-z][a-z&]{2,}")
_BINDING_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "has",
    "had", "have", "its", "which", "into", "than", "their", "also", "been", "under",
    "over", "per", "all", "any", "not", "but", "such", "these", "those", "company",
    "total", "add", "less", "plus", "minus", "note", "back",
    # Units and scale words say nothing about WHICH figure this is.
    "million", "billion", "thousand", "trillion", "usd", "eur", "gbp", "percent",
    "percentage", "basis", "point", "points", "bps",
    "approximately", "about", "roughly",
}

# Share of a row label's words the claim must use, and share of the claim's
# words a prose sentence must hold, before a matching figure counts as the SAME
# figure (see ContextIndex.bind).
LABEL_COVERAGE = 0.6
SENTENCE_COVERAGE = 0.5
# A period this close to a figure, with no other figure between, belongs to it:
# "$452.8 million in FY2023", "FY2022: 387.1".
_PERIOD_ADJACENCY = 16


def binding_tokens(text: str) -> set[str]:
    """
    The words that say WHAT a figure measures — "gross", "margin" — with
    numbers, periods, units and filler removed and plurals folded.

    Args:
        text: A claim, a table row or a sentence.

    Returns:
        Lower-case content tokens.
    """
    lowered = PERIOD_RE.sub(" ", text or "").lower().replace("-", " ").replace("/", " ")
    out = set()
    for token in _BINDING_WORD_RE.findall(lowered):
        if token in _BINDING_STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        out.add(token)
    return out


def _periods_in(text: str) -> list[tuple[int, int, str]]:
    """(start, end, four-digit year) of every period stated in text."""
    out = []
    for m in PERIOD_RE.finditer(text or ""):
        year = m.group("fy") or m.group("fiscal") or m.group("year")
        out.append((m.start(), m.end(), f"20{year}" if len(year) == 2 else year))
    return out


def figure_periods(text: str, spans: list[tuple[int, int]]) -> list[str]:
    """
    The period each figure in a piece of text is stated for.

    For every figure, in order: a period right next to it with no other figure
    in between ("$452.8 million in FY2023", "FY2022 $387.1M"); "*" when the
    wording points at a comparison period without naming it ("up from
    $387.1M", "against $72.1M a year earlier"); the only period the text
    mentions, unless that period belongs to such a comparison ("284 (up from
    231 in FY2022)" says nothing about when 284 was); otherwise "".

    Args:
        text: A claim, a table row or a sentence.
        spans: (start, end) of every figure in text, in order.

    Returns:
        One of a four-digit year, "*" (another, unnamed period) or "" (not
        stated) per span.
    """
    periods = _periods_in(text)
    if not periods:
        return [
            "*" if _is_comparison(text, start, end) else "" for start, end in spans
        ]

    adjacent: list[str] = []
    for start, end in spans:
        best: tuple[int, int, str] | None = None
        for p_start, p_end, year in periods:
            if p_end <= start:
                gap, lo, hi, after = start - p_end, p_end, start, 1
            elif p_start >= end:
                gap, lo, hi, after = p_start - end, end, p_start, 0
            else:
                continue
            if gap > _PERIOD_ADJACENCY:
                continue
            if any(s < hi and e > lo for s, e in spans if (s, e) != (start, end)):
                continue
            if best is None or (gap, after) < best[:2]:
                best = (gap, after, year)
        adjacent.append(best[2] if best else "")

    comparison = [_is_comparison(text, start, end) for start, end in spans]
    years = {year for _, _, year in periods}
    only = next(iter(years)) if len(years) == 1 else ""
    # The single period is "taken" when it sits next to a comparison figure.
    if only and any(a == only and c for a, c in zip(adjacent, comparison)):
        only = ""
    out = []
    for a, c in zip(adjacent, comparison):
        out.append(a or ("*" if c else only))
    return out


def _is_comparison(text: str, start: int, end: int) -> bool:
    """True when the wording around a figure marks it as the comparison value."""
    return bool(
        _COMPARISON_LEAD_RE.search(text[max(0, start - 24): start])
        or _COMPARISON_TRAIL_RE.match(text[end: end + 40])
    )


# ─── Context index ─────────────────────────────────────────────────────────────

# Ratios state their own unit and are never "in millions"; amounts may be.
_RATIO_KINDS = {"percent", "pp", "bps", "multiple"}
_SCALES = (1.0, 1e3, 1e6, 1e9)
# What an unscaled "$452.8" may mean when no unit header was retrieved:
# financial statements are printed in thousands or millions.
_DEFAULT_AMOUNT_SCALES = (1e3, 1e6)

# Floor on relative tolerance, for a claim printed with more precision than the
# context rounds to ("$92,800,000" against a thousands table's "$92,812").
_REL_TOLERANCE = 5e-4


def _kinds_compatible(claim_kind: str, ctx_kind: str, ctx_cell: bool = False) -> bool:
    """
    True when a claim of one kind may be grounded by a context figure of another.

    A ratio (%, pp, bps, x) is grounded by the same kind of ratio, or by a bare
    number sitting in a table cell — tables print "17.0" under a "%" header. It
    is never grounded by a count or an amount: measured on the sample data
    room, letting "1,850 employees" stand in for "2%" and "65 days" for "65%"
    made 46.5% of random integer percentages come back grounded.

    Args:
        claim_kind: Kind of the figure in the answer.
        ctx_kind: Kind of the figure in the context.
        ctx_cell: True when the context figure is a table cell.

    Returns:
        Whether the two may be the same figure.
    """
    if claim_kind == ctx_kind:
        return True
    if {claim_kind, ctx_kind} <= {"percent", "pp"}:
        return True
    if claim_kind in _RATIO_KINDS or ctx_kind in _RATIO_KINDS:
        return ctx_kind == "plain" and ctx_cell
    # Amounts: tables and converted spreadsheets drop the currency symbol, so
    # an amount may be grounded by a separated count, and by a bare cell.
    return ctx_kind != "plain" or ctx_cell


@dataclass
class ContextFigure:
    """
    A figure found in a context chunk, with where it came from.

    Attributes:
        mention: The normalised figure.
        chunk_index: Index of the chunk in the context.
        source_file: File the chunk came from.
        snippet: Text around the figure, for display.
        cell: True when the figure is a cell of a table row.
        row: Index into ContextIndex.rows of that row, else -1.
        anchor_tokens: binding_tokens of the row or sentence the figure sits in.
        period: Year the figure is stated for — its column header, or a period
            next to it — "" when the document does not say.
    """

    mention: NumericMention
    chunk_index: int
    source_file: str
    snippet: str
    cell: bool = False
    row: int = -1
    anchor_tokens: frozenset = frozenset()
    period: str = ""


@dataclass
class _Row:
    """One table row: its label words and the figures in it."""

    tokens: frozenset
    figures: list[int]


@dataclass
class Binding:
    """
    Whether a matching context figure is the SAME figure the claim states.

    status: "bound"    — the claim names the row or sentence the figure sits
                         in, for the period it sits in;
            "conflict" — it names that row, but for another period, or it
                         names a more specific row whose figure is different;
            "unbound"  — the value occurs, in a place the claim does not name.
    """

    status: str
    note: str = ""
    figure: ContextFigure | None = None


class ContextIndex:
    """
    Sorted index of every figure in the retrieved context.

    A lookup is a range query, so grounding a whole answer is O(n log m) rather
    than a nested scan. Each figure is inserted at the magnitudes it can
    honestly stand for:

    - a figure that states its unit ("$452.8 million", "17.0%", "8.5x") — that
      value only;
    - an unscaled amount ("$452.8") — also the unit its document declares in a
      header ("(in millions of USD)"), or thousands and millions when no header
      was retrieved, unless it is a per-share figure or is printed to the cent
      or in full dollars ("$58.00", "$260,000");
    - a bare number in a table cell — also the unit declared by the nearest
      header above it in the same chunk.

    The previous index inserted every unscaled number at four magnitudes and
    let any bare number ground any kind of claim. Measured with the
    random-figure test in eval/run_verification_eval.py, that reported 33.8% of
    random "$X.Y billion" figures and 46.5% of random integer percentages as
    grounded.
    """

    def __init__(self, chunks: Iterable[dict]):
        self.figures: list[ContextFigure] = []
        self.rows: list[_Row] = []
        # (value at one magnitude, figure id, True when that magnitude is assumed)
        keyed: list[tuple[float, int, bool]] = []

        from src.verification.evidence import chunk_evidence_text

        # Unit headers and column headers are properties of a document: a table
        # chunk is often retrieved without the heading that states them. So
        # every body is parsed first, and what each file declares is shared.
        parsed = []
        file_scales: dict[str, set[float]] = {}
        file_headers: dict[str, set[tuple[str, ...]]] = {}
        seen_bodies: set[str] = set()
        for ci, chunk in enumerate(chunks):
            body = chunk_evidence_text(chunk)
            # Sibling chunks expand to the same parent; index its figures once.
            if not body or body in seen_bodies:
                continue
            seen_bodies.add(body)
            source = chunk.get("source_file", "") or ""
            hints = [
                (m.start(), SCALE_HINT_WORDS[next(g for g in m.groups() if g).lower()])
                for m in SCALE_HINT_RE.finditer(body)
            ]
            units = _body_units(body)
            file_scales.setdefault(source, set()).update(s for _, s in hints)
            file_headers.setdefault(source, set()).update(u[3] for u in units if u[3])
            parsed.append((ci, source, body, hints, units))

        for ci, source, body, hints, units in parsed:
            mentions = extract_numbers(body, claims_only=False)
            declared = sorted(file_scales.get(source, ()))
            header: tuple[str, ...] = ()
            cursor = 0
            for start, end, tabular, unit_header in units:
                if unit_header:
                    header = unit_header
                    continue
                while cursor < len(mentions) and mentions[cursor].start < start:
                    cursor += 1
                in_unit = []
                while cursor < len(mentions) and mentions[cursor].start < end:
                    in_unit.append(mentions[cursor])
                    cursor += 1
                if not in_unit:
                    continue

                text = body[start:end]
                tokens = frozenset(binding_tokens(text))
                spans = [(m.start - start, m.end - start) for m in in_unit]
                stated = figure_periods(text, spans)
                per_unit = bool(_PER_UNIT_RE.search(text))
                nearest = next((s for pos, s in reversed(hints) if pos <= start), None)

                row_id = -1
                columns: tuple[str, ...] = ()
                if tabular:
                    row_id = len(self.rows)
                    self.rows.append(_Row(tokens=tokens, figures=[]))
                    # A column header applies to a row of like figures of the
                    # same count: "Revenue  $452.8  $387.1" under "FY2023
                    # FY2022", not "$189.6  41.9%".
                    families = {m.kind in _RATIO_KINDS for m in in_unit}
                    if len(families) == 1:
                        options = [header] if header else sorted(
                            h for h in file_headers.get(source, ()) if len(h) == len(in_unit)
                        )
                        if len(options) == 1 and len(options[0]) == len(in_unit):
                            columns = options[0]

                for k, mention in enumerate(in_unit):
                    period = columns[k] if columns else stated[k]
                    lo = max(0, mention.start - 60)
                    idx = len(self.figures)
                    self.figures.append(ContextFigure(
                        mention=mention,
                        chunk_index=ci,
                        source_file=source,
                        snippet=body[lo: mention.end + 20].replace("\n", " ").strip(),
                        cell=tabular,
                        row=row_id,
                        anchor_tokens=tokens,
                        period="" if period == "*" else period,
                    ))
                    if row_id >= 0:
                        self.rows[row_id].figures.append(idx)
                    keyed.append((mention.value, idx, False))
                    for s in _assumed_scales(mention, tabular, per_unit, declared, nearest):
                        keyed.append((mention.value * s, idx, True))

        keyed.sort(key=lambda entry: entry[0])
        self._values = [value for value, _, _ in keyed]
        self._ids = [idx for _, idx, _ in keyed]
        self._assumed = [assumed for _, _, assumed in keyed]

    def __len__(self) -> int:
        return len(self.figures)

    def candidates(self, mention: NumericMention) -> list[ContextFigure]:
        """
        Every context figure a claimed figure could be, within rounding.

        A ratio is compared as printed. An amount that states its unit is
        compared at that value; one that does not ("$452.8" quoted from a
        millions table) may also equal a context figure that does — but only
        one side of a comparison is ever given an assumed unit.

        Args:
            mention: Figure from the answer.

        Returns:
            Matching ContextFigures, de-duplicated, in context order.
        """
        if mention.explicit_scale or mention.kind in _RATIO_KINDS:
            scales: tuple[float, ...] = (1.0,)
        else:
            scales = _SCALES
        found: dict[int, ContextFigure] = {}
        for s in scales:
            target = mention.value * s
            tol = max(mention.half_unit * s, _REL_TOLERANCE * target, 1e-9)
            lo = bisect.bisect_left(self._values, target - tol)
            hi = bisect.bisect_right(self._values, target + tol)
            for k in range(lo, hi):
                if s != 1.0 and self._assumed[k]:
                    continue
                fig = self.figures[self._ids[k]]
                if _kinds_compatible(mention.kind, fig.mention.kind, fig.cell):
                    found.setdefault(self._ids[k], fig)
        return [found[i] for i in sorted(found)]

    def lookup(self, mention: NumericMention) -> ContextFigure | None:
        """
        Finds a context figure matching a claimed figure, within rounding.

        Args:
            mention: Figure from the answer.

        Returns:
            The first compatible matching ContextFigure, or None.
        """
        found = self.candidates(mention)
        return found[0] if found else None

    def bind(
        self,
        candidates: list[ContextFigure],
        claim_tokens: set[str],
        claim_period: str,
        claim_kind: str,
        explained_rows: set[int],
    ) -> Binding:
        """
        Decides whether any matching context figure is the figure the claim means.

        A value that occurs somewhere in ~64k characters of context is weak
        evidence: a language model's usual numeric error is the right table and
        the wrong row or column, and that error is by construction a value the
        context contains. On the benchmark (eval/verification_set.json) 73
        claims attach a real figure to the wrong metric or period; accepting
        any occurrence passed 70 of them. So a match must also be tied to what
        the claim says the figure IS:

        - a table cell — the claim uses at least LABEL_COVERAGE of the row's
          label words, the cell's column period is the period the claim states
          for it, and the claim does not name another row more fully ("Adjusted
          EBITDA Margin" when the match sits in "EBITDA Margin") that none of
          its own figures match;
        - a figure in prose — the sentence holds at least SENTENCE_COVERAGE of
          the claim's words (or the claim holds the label of a short "Label:
          value" line), for the same period.

        The test is lexical and deliberately one-sided: a paraphrase sharing no
        words with its row comes back "unbound" and is left to the entailment
        model or the judge, which costs a call; a wrong figure that came back
        "bound" would be shown as verified.

        Args:
            candidates: ContextIndex.candidates for the figure.
            claim_tokens: binding_tokens of the claim.
            claim_period: Year the claim states for this figure, "*" or "".
            claim_kind: Kind of the claimed figure.
            explained_rows: Rows matched by some figure of the same claim.

        Returns:
            The best Binding over the candidates: bound, else conflict, else
            unbound.
        """
        best = Binding("unbound")
        for fig in candidates:
            anchor = fig.anchor_tokens
            shared = len(anchor & claim_tokens)
            if fig.cell:
                tied = bool(anchor) and shared / len(anchor) >= LABEL_COVERAGE
            else:
                tied = bool(claim_tokens) and (
                    shared / len(claim_tokens) >= SENTENCE_COVERAGE
                    or (0 < len(anchor) <= 6 and shared / len(anchor) >= LABEL_COVERAGE)
                )
            if not tied:
                continue
            note = ""
            if fig.period and claim_period not in ("", "*") and fig.period != claim_period:
                note = f"the documents give this figure for {fig.period}, not {claim_period}"
            elif fig.cell:
                rival = self._rival_row(fig, claim_tokens, claim_kind, shared, explained_rows)
                if rival:
                    note = f"the statement names another line item ({rival}) with a different figure"
            if not note:
                return Binding("bound", figure=fig)
            if best.status == "unbound":
                best = Binding("conflict", note=note, figure=fig)
        return best

    def _rival_row(
        self,
        fig: ContextFigure,
        claim_tokens: set[str],
        claim_kind: str,
        shared: int,
        explained_rows: set[int],
    ) -> str:
        """
        Label words of a row the claim names more fully than fig's own row.

        "Professional Services revenue was $452.8 million" matches the Revenue
        row on one word and the Professional Services row on two. A row that
        one of the claim's other figures matches is not a rival — the claim
        simply states both ("revenue of $452.8M and gross profit of $271.7M").
        """
        for ri, row in enumerate(self.rows):
            if ri == fig.row or ri in explained_rows or not row.tokens:
                continue
            overlap = len(row.tokens & claim_tokens)
            if overlap <= shared or overlap / len(row.tokens) < LABEL_COVERAGE:
                continue
            if any(
                _kinds_compatible(claim_kind, self.figures[i].mention.kind, True)
                for i in row.figures
            ):
                return " ".join(sorted(row.tokens))
        return ""


def _assumed_scales(
    mention: NumericMention,
    cell: bool,
    per_unit: bool,
    declared: list[float],
    nearest: float | None,
) -> tuple[float, ...]:
    """
    Magnitudes beyond the printed value at which a context figure is indexed.

    Args:
        mention: The context figure.
        cell: True when it is a table cell.
        per_unit: True when its row or sentence quotes a per-share/per-unit figure.
        declared: Unit headers found anywhere in the figure's document.
        nearest: The unit header closest above the figure in its chunk, if any.

    Returns:
        Scale factors (never including 1.0).
    """
    if mention.explicit_scale or mention.kind in _RATIO_KINDS or per_unit:
        return ()
    if mention.kind == "currency":
        if declared:
            return tuple(declared)
        # No header retrieved. "$452.8" and "$453" are statement figures;
        # "$58.00" is a price and "$260,000" is already in dollars.
        if mention.decimals >= 2 or mention.value >= 1000:
            return ()
        return _DEFAULT_AMOUNT_SCALES
    # A bare or separated number is an amount only as a cell under a header in
    # its own chunk — "30 days" and a section number "8.2" are never "$30
    # million" or "$8.2 billion".
    if cell and nearest is not None:
        return (nearest,)
    return ()


def _body_units(body: str) -> list[tuple[int, int, bool, tuple[str, ...]]]:
    """
    Splits evidence text into the rows and sentences its figures sit in.

    Data-room text is hard-wrapped prose interleaved with fixed-width tables
    and "Label: value" lists. A paragraph whose lines mostly look tabular (a
    column gap, a pipe, or a leading "Label:") yields one unit per line; other
    paragraphs yield one unit per sentence, across line breaks. A line that
    only names columns ("FY2023   FY2022", "ASSETS   FY2023   FY2022") is
    returned as a column header.

    Args:
        body: Evidence text of one chunk.

    Returns:
        (start, end, tabular, header) in text order; header is the tuple of
        years of a column-header line and () otherwise.
    """
    units: list[tuple[int, int, bool, tuple[str, ...]]] = []
    for para in re.finditer(r"(?:[^\n]*\S[^\n]*(?:\n|$))+", body):
        lines = []
        offset = para.start()
        for raw in para.group(0).splitlines(keepends=True):
            if raw.strip() and not _RULE_LINE_RE.match(raw):
                lines.append((offset, offset + len(raw.rstrip("\r\n")), raw))
            offset += len(raw)
        if not lines:
            continue
        listed = sum(
            1 for _, _, raw in lines
            if _COLUMN_GAP_RE.search(raw.strip()) or _LABEL_COLON_RE.match(raw)
        )
        if listed >= max(1, len(lines) // 2):
            for start, end, raw in lines:
                if _COLUMN_GAP_RE.search(raw.strip()):
                    units.append((start, end, True, _column_header(raw)))
                    continue
                # "Revenue Growth: 17.0% YoY. The Company was in compliance…":
                # the label owns its own sentence, not the rest of the line.
                cut = 0
                for m in _SENTENCE_END_RE.finditer(body[start:end]):
                    units.append((start + cut, start + m.start(), cut == 0, ()))
                    cut = m.end()
                units.append((start + cut, end, cut == 0, ()))
            continue
        start, end = lines[0][0], lines[-1][1]
        text = body[start:end]
        cut = 0
        for m in _SENTENCE_END_RE.finditer(text):
            units.append((start + cut, start + m.start(), False, ()))
            cut = m.end()
        units.append((start + cut, end, False, ()))
    return units


def _column_header(line: str) -> tuple[str, ...]:
    """
    Years named by a column-header line, () for any other line.

    A header holds two or more periods, each set off by a column gap, and no
    figure: "FY2023   FY2022", optionally after a table title ("ASSETS").
    "Period Under Review: FY2022 - FY2023" is a sentence about periods, not a
    header.
    """
    periods = list(PERIOD_RE.finditer(line))
    if len(periods) < 2:
        return ()
    if extract_numbers(PERIOD_RE.sub(" ", line), claims_only=False):
        return ()
    for m in periods:
        if not re.search(r"(?:^|\s{2,}|\t|\|\s*)$", line[: m.start()]):
            return ()
    return tuple(year for _, _, year in _periods_in(line))


# ─── Grounding ────────────────────────────────────────────────────────────────

# Wording that presents a figure as a computation or approximation. A figure
# introduced this way that cannot be reproduced is flagged for review, not
# failed. Only the text just before the figure is searched (_CUE_WINDOW chars).
_CUE_WINDOW = 45

# Wording that marks a whole claim as a calculation ("Implied EV / EBITDA:
# 7.50x", "calculated as"). Its figures may draw on inputs stated elsewhere in
# the answer, so an unreproducible one is a warning wherever it sits.
_COMPUTED_CLAIM_RE = re.compile(
    r"\b(?:implied|calculat\w*|comput\w*|derived|pro forma|annuali[sz]ed)\b",
    re.IGNORECASE,
)
_COMPUTATION_CUE_RE = re.compile(
    r"\b(?:increas\w*|decreas\w*|grew|grow\w*|declin\w*|rose|fell|drop\w*|"
    r"change[ds]?|differen\w*|combined|aggregate|sum|impl(?:ied|ies|y)|"
    r"approximately|approx\.?|roughly|nearly|almost|exceed\w*|higher|lower|"
    r"more than|less than|delta|cagr|average|equat\w*|net of|reduction|"
    r"improv\w*|expan\w*|contract(?:ed|ion)|yoy|year-over-year)\b|[~≈+=−]",
    re.IGNORECASE,
)


@dataclass
class NumericCheck:
    """Grounding verdict for one figure in the answer."""

    raw: str
    value: float
    kind: str
    status: str  # grounded | derived | derived_unverified | unsupported
    evidence_source: str = ""
    evidence_snippet: str = ""
    formula: str = ""
    text_index: int = 0
    # For a grounded figure: is the matching context figure the one the claim
    # means? "bound" | "conflict" | "unbound" (see ContextIndex.bind); "" for
    # figures that were not found verbatim.
    binding: str = ""
    binding_note: str = ""

    @property
    def is_confirmed(self) -> bool:
        """True when the figure is reproduced AND tied to what the claim says it is."""
        return self.status == "derived" or (self.status == "grounded" and self.binding == "bound")

    def to_dict(self) -> dict:
        """Serialisable form for AgentState."""
        return asdict(self)


def _fmt(v: float) -> str:
    """Compact human-readable number for formulas."""
    if v >= 1e9:
        return f"{v / 1e9:,.4g}B"
    if v >= 1e6:
        return f"{v / 1e6:,.4g}M"
    if v >= 1e3:
        return f"{v:,.0f}"
    return f"{v:,.4g}"


def _derive(
    mention: NumericMention,
    operands: list[tuple[float, str]],
) -> str | None:
    """
    Tries to reproduce a figure from two grounded figures stated alongside it.

    Deliberately restricted to grounded operands from the SAME claim — the
    sentence or table row the figure sits in. Searching pairs of context
    numbers, or even of the whole answer's figures, finds a coincidental match
    for almost any target: tried against the full RESULTS.md answer set, a
    whole-answer operand pool "derived" 86.3% as (24.8M - 181.1M) / 181.1M.
    Analysts state their inputs next to their result ("+$65.7M (+17.0%)" in the
    row holding $452.8M and $387.1M; "exceeds the $3.48M deductible by $4.52
    million" beside the $8.0M exposure), so same-claim operands keep the real
    derivations while making coincidences rare.

    Args:
        mention: Ungrounded figure from the answer.
        operands: (normalised value, kind) of grounded figures in the same claim.

    Returns:
        A formula string when reproducible, else None.
    """
    kind = mention.kind
    scales = (
        (1.0,)
        if mention.explicit_scale or kind in ("percent", "pp", "bps", "multiple")
        else _SCALES
    )

    def close(target: float, got: float, s: float) -> bool:
        tol = max(mention.half_unit * s, _REL_TOLERANCE * abs(target), 1e-9)
        return abs(target - got) <= tol

    amounts = [(v, k) for v, k in operands if k in ("currency", "count")]
    percents = [v for v, k in operands if k == "percent"]
    multiples = [v for v, k in operands if k == "multiple"]
    basis_points = [v for v, k in operands if k == "bps"]

    def amount_pairs():
        """Ordered pairs of distinct amounts of the same kind."""
        for i, (a, ka) in enumerate(amounts):
            for j, (b, kb) in enumerate(amounts):
                if i != j and ka == kb and b:
                    yield a, b

    for s in scales:
        target = mention.value * s
        if kind in ("currency", "count"):
            for a, b in amount_pairs():
                if close(target, abs(a - b), s):
                    return f"|{_fmt(a)} - {_fmt(b)}|"
                if close(target, a + b, s):
                    return f"{_fmt(a)} + {_fmt(b)}"
        elif kind == "percent":
            for a, b in amount_pairs():
                if close(target, abs(a - b) / b * 100, s):
                    return f"({_fmt(a)} - {_fmt(b)}) / {_fmt(b)}"
                if a < b and close(target, a / b * 100, s):
                    return f"{_fmt(a)} / {_fmt(b)}"
            for i, a in enumerate(percents):
                for b in percents[i + 1:]:
                    if close(target, abs(a - b), s):
                        return f"|{a:g}% - {b:g}%|"
            for a in basis_points:
                if close(target, a / 100, s):
                    return f"{a:g} bps / 100"
        elif kind == "pp":
            for i, a in enumerate(percents):
                for b in percents[i + 1:]:
                    if close(target, abs(a - b), s):
                        return f"|{a:g}% - {b:g}%|"
        elif kind == "bps":
            for i, a in enumerate(percents):
                for b in percents[i + 1:]:
                    if close(target, abs(a - b) * 100, s):
                        return f"|{a:g}% - {b:g}%| x 100"
            for a in percents:
                if close(target, a * 100, s):
                    return f"{a:g}% x 100"
        elif kind == "multiple":
            for a, b in amount_pairs():
                if close(target, a / b, s):
                    return f"{_fmt(a)} / {_fmt(b)}"
            for i, a in enumerate(multiples):
                for b in multiples[i + 1:]:
                    if close(target, abs(a - b), s):
                        return f"|{a:g}x - {b:g}x|"
    return None


def ground_texts(texts: list[str], index: ContextIndex) -> list[list[NumericCheck]]:
    """
    Grounds every figure in a list of answer segments (claims or sentences).

    Two passes per segment: every figure is first looked up verbatim; the ones
    not found are then tried as derivations from the segment's own grounded
    figures (see _derive), and failing that are classed by wording.

    Args:
        texts: Answer segments, citation markers already stripped.
        index: ContextIndex over the retrieved chunks.

    Returns:
        One list of NumericCheck per input text, in order.
    """
    results: list[list[NumericCheck]] = []

    for ti, text in enumerate(texts):
        checks: list[NumericCheck] = []
        pending: list[tuple[NumericMention, NumericCheck]] = []
        operands: list[tuple[float, str]] = []
        # (figure, its check, its value in the context's normalisation)
        grounded: list[tuple[NumericMention, NumericCheck, float]] = []

        mentions = extract_numbers(text, claims_only=True)
        found = [index.candidates(mention) for mention in mentions]
        claim_tokens = binding_tokens(text)
        # Periods are read off every figure in the text, plain ones included:
        # in "12 months ended FY2023, $452.8M" the 12 sits between the two.
        every = extract_numbers(text, claims_only=False)
        stated = dict(zip(
            ((m.start, m.end) for m in every),
            figure_periods(text, [(m.start, m.end) for m in every]),
        ))
        explained = {fig.row for group in found for fig in group if fig.row >= 0}

        for mention, group in zip(mentions, found):
            if group:
                binding = index.bind(
                    group,
                    claim_tokens,
                    stated.get((mention.start, mention.end), ""),
                    mention.kind,
                    explained,
                )
                fig = binding.figure or group[0]
                # Operand value in the context's normalisation, so a claim
                # printed as "$452.8" (unscaled) still computes in dollars when
                # the context says "$452.8 million".
                value = mention.value
                if not mention.explicit_scale and fig.mention.explicit_scale:
                    value = fig.mention.value
                operands.append((value, mention.kind))
                check = NumericCheck(
                    raw=mention.raw,
                    value=mention.value,
                    kind=mention.kind,
                    status="grounded",
                    evidence_source=fig.source_file,
                    evidence_snippet=fig.snippet[:160],
                    text_index=ti,
                    binding=binding.status,
                    binding_note=binding.note,
                )
                checks.append(check)
                grounded.append((mention, check, value))
            else:
                check = NumericCheck(
                    raw=mention.raw,
                    value=mention.value,
                    kind=mention.kind,
                    status="unsupported",
                    text_index=ti,
                )
                checks.append(check)
                pending.append((mention, check))

        # A figure the context holds somewhere the claim does not name is still
        # confirmed when the claim's own confirmed figures reproduce it: "up
        # 17.0% from $387.1 million" needs no "Revenue Growth: 17.0%" line.
        confirmed = [(value, c.kind) for _, c, value in grounded if c.binding == "bound"]
        for mention, check, _ in grounded:
            if check.binding == "bound":
                continue
            formula = _derive(mention, confirmed)
            if formula:
                check.binding = "bound"
                check.binding_note = ""
                check.formula = formula

        # Two rounds, so a figure computed from a derived one still resolves:
        # "margin rose 1.74pp" from two margins the sentence itself computed.
        for _ in range(2):
            progressed = False
            for mention, check in pending:
                if check.status == "derived":
                    continue
                formula = _derive(mention, operands)
                if formula:
                    check.status = "derived"
                    check.formula = formula
                    operands.append((mention.value, mention.kind))
                    progressed = True
            if not progressed:
                break

        for mention, check in pending:
            if check.status == "derived":
                continue
            # The cue must introduce THIS figure ("an increase of $X",
            # "approximately 7.5x"), not merely occur somewhere in the sentence —
            # otherwise "leverage fell to 0.2x and the balance was $142.0M" would
            # excuse a fabricated $142.0M because of "fell". So the lead-in stops
            # at the previous figure.
            previous_end = max(
                (m.end for m in mentions if m.end <= mention.start), default=0
            )
            lead_in = text[max(previous_end, mention.start - _CUE_WINDOW): mention.start]
            if _COMPUTATION_CUE_RE.search(lead_in) or _COMPUTED_CLAIM_RE.search(text):
                check.status = "derived_unverified"

        # A figure restating the one just before it in other units —
        # "$0.02 million ($20,000)", "$97.3M ($97,300,000)" — is the same claim
        # twice, so it shares that figure's verdict instead of being judged on
        # its own (where "$20,000" matches nothing verbatim).
        for i in range(1, len(checks)):
            prev, cur = checks[i - 1], checks[i]
            if (
                cur.status in ("unsupported", "derived_unverified")
                and prev.status != "unsupported"
                and _kinds_compatible(cur.kind, prev.kind)
                and abs(cur.value - prev.value) <= max(mentions[i].half_unit, _REL_TOLERANCE * prev.value)
            ):
                cur.status = prev.status
                cur.formula = f"restates {prev.raw}"
                cur.evidence_source = prev.evidence_source
                cur.evidence_snippet = prev.evidence_snippet
                cur.binding = prev.binding
                cur.binding_note = prev.binding_note

        results.append(checks)

    return results


def ground_answer_numbers(answer: str, chunks: list[dict]) -> list[dict]:
    """
    Grounds every figure in an answer against the context — the answer's
    `numerical_claims`.

    Args:
        answer: Generated answer text (citation markers are stripped here).
        chunks: Retrieved context chunks.

    Returns:
        List of NumericCheck dicts, in answer order.
    """
    from src.verification.claims import split_claims  # deferred: avoids a cycle

    claims = split_claims(answer)
    if not claims:
        return []
    index = ContextIndex(chunks)
    grouped = ground_texts([c.text for c in claims], index)
    return [check.to_dict() for checks in grouped for check in checks]


def summarize_numeric(checks: Iterable[dict | NumericCheck]) -> dict[str, int]:
    """
    Counts numeric checks by status.

    Args:
        checks: NumericCheck objects or their dicts.

    Returns:
        {status: count} over the four statuses, zeros included.
    """
    counts = {"grounded": 0, "derived": 0, "derived_unverified": 0, "unsupported": 0}
    for c in checks:
        status = c["status"] if isinstance(c, dict) else c.status
        counts[status] = counts.get(status, 0) + 1
    return counts


__all__ = [
    "NumericMention",
    "NumericCheck",
    "ContextIndex",
    "Binding",
    "SCALE_HINT_RE",
    "SCALE_HINT_WORDS",
    "PERIOD_RE",
    "binding_tokens",
    "figure_periods",
    "extract_numbers",
    "strip_citations",
    "ground_texts",
    "ground_answer_numbers",
    "summarize_numeric",
]
