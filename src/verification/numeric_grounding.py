"""
Deterministic numeric grounding.

Every figure an answer states is extracted, normalised to a plain magnitude
(`$452.8 million` -> 452_800_000) and looked up among the figures present in
the retrieved context, normalised the same way. A figure is:

- grounded            — the same value appears in the context (within rounding),
                        at a unit the document states;
- derived             — not verbatim, but reproducible by one arithmetic step
                        (difference, sum, ratio, percentage change) from two
                        figures the answer itself states AND that are grounded,
                        or the same rate in the other notation (225 bps = 2.25%);
- derived_unverified  — not reproducible, but the sentence presents it as a
                        computation ("an increase of", "combined", "approximately")
                        AND states figures it could have been computed from, so
                        it is reported as a warning rather than a fabrication;
- unsupported         — none of the above.

A grounded figure also carries a binding — is the matching context figure the
one the claim means, or just the same number somewhere else? See
ContextIndex.bind. Only a bound figure confirms a claim by itself.

Why deterministic: the previous validator asked a lite model whether a stronger
model's numbers were right. That check cost an LLM call per query and was
unreliable in exactly the place it matters — a lite model will wave through a
transposed digit. String-level grounding cannot be talked out of a mismatch.

Why units are read from the document: data-room tables state their unit once in
a header ("(in millions of USD)") and print bare figures ("$452.8"), while
answers write "$452.8 million" or "$452,800,000". So an unscaled context amount
is also read at the unit its document declares — and at no other. An earlier
version let either side differ by any power of 1,000 and let any bare number
stand in for any kind of figure; see ContextIndex for what that cost.
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
        kind: currency | percent | pp | bps | multiple | count | quantity | plain.
            A quantity is a bare number with the thing it counts: "18 months",
            "30 days", "700 employees".
        explicit_scale: True when the text itself stated the unit (M, million).
        scale: The explicit scale factor (1.0 when none).
        decimals: Digits after the decimal point as printed — drives the
            rounding tolerance.
        start: Offset in the source text.
        end: End offset in the source text.
        currency: "USD", "EUR" or "GBP" when the text names one, else "".
        unit: For a quantity, what is counted, singular: "month", "employee".
    """

    raw: str
    value: float
    kind: str
    explicit_scale: bool
    scale: float
    decimals: int
    start: int
    end: int
    currency: str = ""
    unit: str = ""

    @property
    def half_unit(self) -> float:
        """Half of the last printed digit, in normalised units — rounding slack."""
        return 0.5 * (10 ** -self.decimals) * self.scale

    @property
    def is_claim(self) -> bool:
        """
        True for figures an answer asserts as a quantity.

        Plain integers without separators or units ("284", "5", "12") are
        excluded: in prose they are overwhelmingly list numbers and clause
        numbers, and treating them as financial claims produces noise that
        trains reviewers to ignore the check. A bare number followed by what it
        counts ("24 months", "45 days") IS a claim — survival, notice and
        non-compete periods are among the most consequential numbers in a
        contract — but a weaker one: see the quantity rules in claim_checker.
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

        raw = m.group(0).strip()
        counted = _QUANTITY_UNIT_RE.match(scrubbed, m.end()) if kind == "plain" else None
        if counted:
            kind = "quantity"
            raw = f"{raw} {counted.group('noun')}{counted.group('plural')}"

        mentions.append(NumericMention(
            raw=raw,
            value=abs(mantissa) * scale,
            kind=kind,
            explicit_scale=scale_token is not None,
            scale=scale,
            decimals=decimals,
            start=m.start(),
            end=m.end(),
            currency=_CURRENCY_CODES.get(cur.upper(), "") if cur else "",
            unit=counted.group("noun").lower() if counted else "",
        ))

    mentions = _share_range_units(scrubbed, mentions)
    if claims_only:
        mentions = [mention for mention in mentions if mention.is_claim]
    return mentions


_CURRENCY_CODES = {"$": "USD", "US$": "USD", "USD": "USD", "€": "EUR", "EUR": "EUR", "£": "GBP", "GBP": "GBP"}

# What a bare number counts, directly after it: "18 months", "(36) months",
# "60-day", "30 calendar days", "700 employees".
_QUANTITY_UNIT_RE = re.compile(
    r"\)?[ -]?(?:(?:calendar|business|working|consecutive|full)\s)?"
    r"(?P<noun>day|week|month|quarter|year|hour|employee|customer|patent|share|director"
    r"|executive|officer|contractor)(?P<plural>s?)\b",
    re.IGNORECASE,
)

# What may stand between the two ends of a range: "225 to 175 basis points",
# "$3.5 – $8.0 million", "between 10 and 15%".
_RANGE_JOIN_RE = re.compile(r"\s*(?:to|and|or|through|[-–—])\s*")


def _share_range_units(text: str, mentions: list[NumericMention]) -> list[NumericMention]:
    """
    Gives the first figure of a range the unit stated once, after the second.

    "From 225 to 175 basis points" and "$3.5 to $8.0 million" state the unit
    one time for both ends. Read literally the first end is a bare 225 (not a
    figure at all) or $3.5 (three dollars fifty), so a correct range could
    neither be grounded nor used as an operand.

    Args:
        text: The text the mentions were extracted from.
        mentions: Figures in text order.

    Returns:
        The mentions, with range starts completed.
    """
    out = list(mentions)
    for i in range(len(out) - 1):
        first, second = out[i], out[i + 1]
        if second.start - first.end > 12 or not _RANGE_JOIN_RE.fullmatch(
            text[first.end: second.start]
        ):
            continue
        stated_once = second.kind in ("percent", "pp", "bps", "multiple", "quantity")
        if first.kind == "plain" and (
            stated_once or (second.kind == "count" and second.explicit_scale)
        ):
            kind, scale, explicit = second.kind, second.scale, second.explicit_scale
        elif (
            first.kind == second.kind == "currency"
            and second.explicit_scale
            and not first.explicit_scale
        ):
            kind, scale, explicit = first.kind, second.scale, True
        else:
            continue
        out[i] = NumericMention(
            raw=first.raw,
            value=first.value * scale,
            kind=kind,
            explicit_scale=explicit,
            scale=scale,
            decimals=first.decimals,
            start=first.start,
            end=first.end,
            currency=first.currency,
            unit=second.unit,
        )
    return out


# ─── Units and periods a document states ──────────────────────────────────────

# A unit header: "(in millions of USD)", "$ in thousands", "USD m". Shared with
# the financial verifier's registry so both read a table's unit the same way.
# Every quantifier is followed by something it cannot itself match: this runs
# over counterparty-supplied text, where "(" plus a long run of spaces made the
# earlier `\(\s*(?:…)?\s*` form quadratic.
SCALE_HINT_RE = re.compile(
    r"\(\s*(?:(?:\$|USD|US\$|EUR|GBP)\s*)?(?:in\s+)?(thousands|millions|billions)\b"
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

_RULE_CHARS = frozenset("=-_*~|: \t")
_PARENTHETICAL_RE = re.compile(r"\(([^()\n]*)\)")
_COLUMN_GAP_RE = re.compile(r"\S(?:\s{2,}|\t|\s*\|\s*)\S")
# Applied to a left-stripped line (see _is_listed).
_LABEL_COLON_RE = re.compile(r"[^:\n]{2,60}:\s")
# A sentence ends before a capital, a figure or a list bullet: without the
# bullet, "…$10 million per fiscal year;\n  - Dispose of assets … $20 million"
# is one sentence, and every figure in a bulleted list shares one anchor.
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?;])\s+(?=[-•*]\s|[A-Z0-9(\"'$])|(?<=:)\s+(?=[-•*]\s)"
)
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

    # Only the nearest period on each side can be adjacent, and only the
    # neighbouring figure can sit in between — so this is a bisect per figure,
    # not a scan of every period (a document line can hold thousands of both).
    period_starts = [p[0] for p in periods]
    comparison = [_is_comparison(text, start, end) for start, end in spans]
    adjacent: list[str] = []
    for i, (start, end) in enumerate(spans):
        best: tuple[int, int, str] | None = None
        k = bisect.bisect_left(period_starts, end)
        if k < len(periods):
            p_start, _, year = periods[k]
            next_figure = spans[i + 1][0] if i + 1 < len(spans) else p_start
            if p_start - end <= _PERIOD_ADJACENCY and next_figure >= p_start:
                best = (p_start - end, 0, year)
        j = bisect.bisect_right(period_starts, start) - 1
        # "…in FY2023, up from $387.1M": the period before a comparison figure
        # belongs to the figure it is being compared with.
        if j >= 0 and periods[j][1] <= start and not comparison[i]:
            _, p_end, year = periods[j]
            previous_figure = spans[i - 1][1] if i else p_end
            gap = start - p_end
            if (
                gap <= _PERIOD_ADJACENCY
                and previous_figure <= p_end
                and (best is None or (gap, 1) < best[:2])
            ):
                best = (gap, 1, year)
        adjacent.append(best[2] if best else "")

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
# Kinds that are never read at an assumed magnitude.
_UNSCALED_KINDS = _RATIO_KINDS | {"quantity"}
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
    # A quantity is grounded by the same count of the same thing, or by a bare
    # number ("12 of the Company's 47 issued U.S. patents" for "12 patents"),
    # and grounds nothing else: "30 days" is not "$30 million" or "30%".
    if claim_kind == "quantity":
        return ctx_kind in ("quantity", "plain")
    if ctx_kind == "quantity":
        return False
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
        unit: Index into ContextIndex.units of the row or sentence it sits in.
        anchor_tokens: binding_tokens of that row or sentence.
        alias_tokens: For a cell, the label words inside parentheses —
            "Operating Income (EBIT)" is also just "EBIT".
        period: Year the figure is stated for — its column header, or a period
            next to it — "" when the document does not say.
        column_unknown: True for a cell in a row of several like figures whose
            columns no retrieved header names: one of them is this period's
            figure, and nothing retrieved says which.
    """

    mention: NumericMention
    chunk_index: int
    source_file: str
    snippet: str
    cell: bool = False
    unit: int = -1
    anchor_tokens: frozenset = frozenset()
    alias_tokens: frozenset = frozenset()
    period: str = ""
    column_unknown: bool = False

    def named_by(self, claim_tokens: set[str]) -> bool:
        """
        True when a claim names the row or sentence this figure sits in.

        A cell: the claim uses LABEL_COVERAGE of the row's label words — of the
        whole label, of the label without its parenthetical, or all of the
        parenthetical alone. A figure in prose: the sentence holds
        SENTENCE_COVERAGE of the claim's words, or the claim holds the label of
        a short "Label: value" line.
        """
        anchor = self.anchor_tokens
        shared = len(anchor & claim_tokens)
        if not self.cell:
            return bool(claim_tokens) and (
                shared / len(claim_tokens) >= SENTENCE_COVERAGE
                or (0 < len(anchor) <= 6 and shared / len(anchor) >= LABEL_COVERAGE)
            )
        if anchor and shared / len(anchor) >= LABEL_COVERAGE:
            return True
        alias = self.alias_tokens
        if not alias:
            return False
        main = anchor - alias
        return alias <= claim_tokens or (
            bool(main) and len(main & claim_tokens) / len(main) >= LABEL_COVERAGE
        )


@dataclass
class _Unit:
    """A table row or a sentence that holds figures: its words and its figures."""

    tokens: frozenset
    figures: list[int]
    tabular: bool
    kinds: frozenset = frozenset()


@dataclass
class Binding:
    """
    Whether a matching context figure is the SAME figure the claim states.

    status: "bound"    — the claim names the row or sentence the figure sits
                         in, for the period it sits in;
            "conflict" — it names that row, but for another period, or it
                         names another line more fully whose figure differs;
            "unbound"  — the value occurs, in a place the claim does not name
                         (or in a row whose columns cannot be told apart).
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
        self.units: list[_Unit] = []
        self._units_by_token: dict[str, list[int]] = {}
        # (value at one magnitude, figure id, True when that magnitude is assumed)
        keyed: list[tuple[float, int, bool]] = []

        from src.verification.evidence import chunk_evidence_text

        # Unit headers and column headers are properties of a document: a table
        # chunk is often retrieved without the heading that states them (the
        # sample financials' key-metrics table has neither). So every body is
        # parsed first, and what each file declares is shared across its chunks.
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
            hint_positions = [pos for pos, _ in hints]
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
                stated = figure_periods(text, [(m.start - start, m.end - start) for m in in_unit])
                per_unit = bool(_PER_UNIT_RE.search(text))
                at = bisect.bisect_right(hint_positions, start) - 1
                nearest = hints[at][1] if at >= 0 else None

                alias = frozenset(
                    binding_tokens(" ".join(_PARENTHETICAL_RE.findall(text)))
                ) if tabular else frozenset()

                unit_id = len(self.units)
                self.units.append(_Unit(
                    tokens=tokens,
                    figures=[],
                    tabular=tabular,
                    kinds=frozenset(m.kind for m in in_unit),
                ))
                for token in tokens:
                    self._units_by_token.setdefault(token, []).append(unit_id)

                # A column header applies to a row of like figures of the same
                # count: "Revenue  $452.8  $387.1" under "FY2023  FY2022", not
                # "$189.6  41.9%".
                columns: tuple[str, ...] = ()
                alike = tabular and len(in_unit) > 1 and len({m.kind for m in in_unit}) == 1
                if alike:
                    options = [header] if header else sorted(
                        h for h in file_headers.get(source, ()) if len(h) == len(in_unit)
                    )
                    if len(options) == 1 and len(options[0]) == len(in_unit):
                        columns = options[0]

                for k, mention in enumerate(in_unit):
                    period = columns[k] if columns else stated[k]
                    period = "" if period == "*" else period
                    lo = max(0, mention.start - 60)
                    idx = len(self.figures)
                    self.figures.append(ContextFigure(
                        mention=mention,
                        chunk_index=ci,
                        source_file=source,
                        snippet=body[lo: mention.end + 20].replace("\n", " ").strip(),
                        cell=tabular,
                        unit=unit_id,
                        anchor_tokens=tokens,
                        alias_tokens=alias,
                        period=period,
                        column_unknown=alike and not period,
                    ))
                    self.units[unit_id].figures.append(idx)
                    keyed.append((mention.value, idx, False))
                    for s in _assumed_scales(mention, tabular, per_unit, declared, nearest):
                        keyed.append((mention.value * s, idx, True))

        keyed.sort(key=lambda entry: entry[0])
        self._values = [value for value, _, _ in keyed]
        self._ids = [idx for _, idx, _ in keyed]
        self._assumed = [assumed for _, _, assumed in keyed]

    def __len__(self) -> int:
        return len(self.figures)

    def _within(self, target: float, tol: float) -> range:
        """Positions in the sorted index whose value is within tol of target."""
        return range(
            bisect.bisect_left(self._values, target - tol),
            bisect.bisect_right(self._values, target + tol),
        )

    def candidates(self, mention: NumericMention) -> list[ContextFigure]:
        """
        Every context figure a claimed figure could be, within rounding.

        A ratio is compared as printed. An amount that states its unit is
        compared at that value; one that does not ("$452.8" quoted from a
        millions table) may also equal a context figure that does — but only
        one side of a comparison is ever given an assumed unit. An amount in
        one currency is never grounded by the same number in another. A
        percentage and a basis-point figure are the same quantity in two
        notations, so "2.25%" is grounded by "225 bps" and the reverse.

        Args:
            mention: Figure from the answer.

        Returns:
            Matching ContextFigures, de-duplicated, in context order.
        """
        if mention.explicit_scale or mention.kind in _UNSCALED_KINDS:
            scales: tuple[float, ...] = (1.0,)
        else:
            scales = _SCALES
        found: dict[int, ContextFigure] = {}
        for s in scales:
            target = mention.value * s
            tol = max(mention.half_unit * s, _REL_TOLERANCE * target, 1e-9)
            if mention.kind == "quantity":
                tol = 1e-9  # "24 months" is not "24.3 months" rounded
            for k in self._within(target, tol):
                if s != 1.0 and self._assumed[k]:
                    continue
                fig = self.figures[self._ids[k]]
                if not _kinds_compatible(mention.kind, fig.mention.kind, fig.cell):
                    continue
                if mention.unit and fig.mention.unit and mention.unit != fig.mention.unit:
                    continue
                if mention.currency and fig.mention.currency and mention.currency != fig.mention.currency:
                    continue
                found.setdefault(self._ids[k], fig)

        # A margin quoted in basis points, restated as a percentage: "SOFR plus
        # 2.25%" for "225 bps". Only this direction — answers convert to
        # percentages for readability; grounding every "N bps" against every
        # small percentage raised the random-figure rate for bps from 16% to 25%.
        if mention.kind == "percent":
            target = mention.value * 100.0
            tol = max(mention.half_unit * 100.0, _REL_TOLERANCE * target, 1e-9)
            for k in self._within(target, tol):
                fig = self.figures[self._ids[k]]
                if fig.mention.kind == "bps" and not self._assumed[k]:
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
        explained_units: set[int],
        overlaps: dict[int, int] | None = None,
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
          label words;
        - a figure in prose — the sentence holds at least SENTENCE_COVERAGE of
          the claim's words (or the claim holds the label of a short "Label:
          value" line);

        and, for a match tied that way:

        - the period the document states for the figure (its column header, or
          a period next to it) is the period the claim states for it;
        - the claim does not name another row or sentence more fully ("Adjusted
          EBITDA Margin" when the match sits in "EBITDA Margin") that none of
          its own figures match.

        A tied match that fails one of those is a "conflict": evidence the
        figure belongs elsewhere. A conflict outranks a tied match whose period
        simply is not stated — "Total Revenue $452.8" in an undated table must
        not vouch for "FY2022 revenue was $452.8 million" when the income
        statement puts $452.8 under FY2023.

        The test is lexical and deliberately one-sided: a paraphrase sharing no
        words with its row comes back "unbound" and is left to the entailment
        model or the judge, which costs a call; a wrong figure that came back
        "bound" would be shown as verified.

        Args:
            candidates: ContextIndex.candidates for the figure.
            claim_tokens: binding_tokens of the claim.
            claim_period: Year the claim states for this figure, "*" or "".
            claim_kind: Kind of the claimed figure.
            explained_units: Rows and sentences matched by a figure of the claim.
            overlaps: unit_overlaps(claim_tokens), when the caller binds several
                figures of one claim and has computed it already.

        Returns:
            The Binding for the figure.
        """
        year = claim_period if claim_period not in ("", "*") else ""
        conflict: Binding | None = None
        undated: ContextFigure | None = None
        ambiguous: ContextFigure | None = None
        # The claim's strongest rival is the same for every candidate, so it is
        # found once, and only if a candidate gets that far. Looking for it per
        # candidate made binding candidates x units: 4.2 s for a 40-claim answer
        # against a 1,000-row schedule that repeats one figure, fourfold per
        # doubling — on a document a counterparty supplies.
        rival: tuple[int, str] | None = None

        for fig in candidates:
            if not fig.named_by(claim_tokens):
                continue
            shared = len(fig.anchor_tokens & claim_tokens)

            if year and fig.period and fig.period != year:
                conflict = conflict or Binding(
                    "conflict",
                    note=f"the documents give this figure for {fig.period}, not {year}",
                    figure=fig,
                )
                continue
            if rival is None:
                rival = self._strongest_rival(
                    self.unit_overlaps(claim_tokens) if overlaps is None else overlaps,
                    claim_tokens,
                    claim_kind,
                    explained_units | {f.unit for f in candidates},
                )
            if rival[0] > shared:
                conflict = conflict or Binding(
                    "conflict",
                    note=f"the statement names another item ({rival[1]}) whose figure is different",
                    figure=fig,
                )
                continue
            if year and not fig.period:
                if fig.column_unknown:
                    ambiguous = ambiguous or fig
                else:
                    undated = undated or fig
                continue
            return Binding("bound", figure=fig)

        if conflict:
            return conflict
        if undated:
            return Binding("bound", figure=undated)
        if ambiguous:
            return Binding(
                "unbound",
                note=f"the retrieved table does not show which column is {year}",
                figure=ambiguous,
            )
        return Binding("unbound")

    def unit_overlaps(self, claim_tokens: set[str]) -> dict[int, int]:
        """
        How many of a claim's words each row or sentence shares with it.

        Read off an inverted index, so the cost is the number of (word, unit)
        pairs that actually match — not a scan of every unit per figure.

        Args:
            claim_tokens: binding_tokens of the claim.

        Returns:
            {unit index: shared word count} for units sharing at least one word.
        """
        counts: dict[int, int] = {}
        for token in claim_tokens:
            for ui in self._units_by_token.get(token, ()):
                counts[ui] = counts.get(ui, 0) + 1
        return counts

    def _strongest_rival(
        self,
        overlaps: dict[int, int],
        claim_tokens: set[str],
        claim_kind: str,
        excluded_units: set[int],
    ) -> tuple[int, str]:
        """
        The row or sentence, other than those the claim's figures match, that
        the claim names most fully.

        "Professional Services revenue was $452.8 million" matches the Revenue
        row on one word and the Professional Services row on two; "Gross
        revenue retention was 118%" matches the net-retention sentence on two
        words and the gross-retention sentence on three. A row or sentence that
        one of the claim's figures does match is not a rival — the claim simply
        states both ("revenue of $452.8M and gross profit of $271.7M"). A
        candidate is in conflict when the rival shares more words with the
        claim than the candidate's own row or sentence does.

        Args:
            overlaps: unit_overlaps(claim_tokens).
            claim_tokens: binding_tokens of the claim.
            claim_kind: Kind of the claimed figure; a rival must hold a figure
                that could be mistaken for it.
            excluded_units: Units matched by a figure of the claim.

        Returns:
            (shared word count, the shared words) of the strongest rival, or
            (0, "") when there is none.
        """
        best_overlap, best_unit = 0, -1
        for ui, overlap in overlaps.items():
            if overlap <= best_overlap or ui in excluded_units:
                continue
            unit = self.units[ui]
            if unit.tabular:
                if overlap / len(unit.tokens) < LABEL_COVERAGE:
                    continue
            elif overlap / len(claim_tokens) < SENTENCE_COVERAGE:
                continue
            if any(_kinds_compatible(claim_kind, kind, True) for kind in unit.kinds):
                best_overlap, best_unit = overlap, ui
        if best_unit < 0:
            return 0, ""
        return best_overlap, " ".join(sorted(self.units[best_unit].tokens & claim_tokens))


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
    if mention.explicit_scale or mention.kind in _UNSCALED_KINDS or per_unit:
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


def _is_rule_line(line: str) -> bool:
    """True for a separator line ("=====", "-----  -----", "|---|---|")."""
    stripped = line.strip()
    return len(stripped) >= 4 and _RULE_CHARS.issuperset(stripped)


def _is_listed(line: str) -> bool:
    """True for a line that is a table row or a "Label: value" entry."""
    stripped = line.strip()
    return bool(_COLUMN_GAP_RE.search(stripped) or _LABEL_COLON_RE.match(line.lstrip()))


def _body_units(body: str) -> list[tuple[int, int, bool, tuple[str, ...]]]:
    """
    Splits evidence text into the rows and sentences its figures sit in.

    Data-room text is hard-wrapped prose interleaved with fixed-width tables
    and "Label: value" lists. A paragraph whose lines mostly look tabular (a
    column gap, a pipe, or a leading "Label:") yields one unit per line; other
    paragraphs yield one unit per sentence, across line breaks. A line that
    only names columns ("FY2023   FY2022", "ASSETS   FY2023   FY2022") is
    returned as a column header.

    Lines are walked with plain string operations: this runs over
    counterparty-supplied text, and a paragraph regex with nested quantifiers
    took cubic time on a long whitespace-padded line.

    Args:
        body: Evidence text of one chunk.

    Returns:
        (start, end, tabular, header) in text order; header is the tuple of
        years of a column-header line and () otherwise.
    """
    units: list[tuple[int, int, bool, tuple[str, ...]]] = []
    paragraph: list[tuple[int, int, str]] = []

    def flush() -> None:
        if not paragraph:
            return
        listed = sum(1 for _, _, raw in paragraph if _is_listed(raw))
        if listed >= max(1, len(paragraph) // 2):
            for start, end, raw in paragraph:
                if _COLUMN_GAP_RE.search(raw.strip()):
                    units.append((start, end, True, _column_header(raw)))
                    continue
                # "Revenue Growth: 17.0% YoY. The Company was in compliance…":
                # the label owns its own sentence, not the rest of the line.
                labelled = bool(_LABEL_COLON_RE.match(raw.lstrip()))
                cut = 0
                for m in _SENTENCE_END_RE.finditer(body[start:end]):
                    units.append((start + cut, start + m.start(), labelled and cut == 0, ()))
                    cut = m.end()
                units.append((start + cut, end, labelled and cut == 0, ()))
        else:
            start, end = paragraph[0][0], paragraph[-1][1]
            cut = 0
            for m in _SENTENCE_END_RE.finditer(body[start:end]):
                units.append((start + cut, start + m.start(), False, ()))
                cut = m.end()
            units.append((start + cut, end, False, ()))
        paragraph.clear()

    offset = 0
    for raw in body.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        if not line.strip():
            flush()
        elif not _is_rule_line(line):
            paragraph.append((offset, offset + len(line), line))
        offset += len(raw)
    flush()
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
        before = line[: m.start()]
        if before.strip() and not before.endswith(("  ", "\t")) and not before.rstrip(" ").endswith("|"):
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
    # Where the matching context figure sits — its chunk and the figure as the
    # document prints it — so the claim checker can score the claim against
    # the exact passage the number came from. Internal: not serialised.
    evidence_chunk: int = -1
    evidence_figure: str = ""

    @property
    def is_confirmed(self) -> bool:
        """True when the figure is reproduced AND tied to what the claim says it is.

        A figure computed from the claim's own figures has no binding of its
        own (""): its operands carry theirs.
        """
        return self.status in ("grounded", "derived") and self.binding in ("bound", "")

    def to_dict(self) -> dict:
        """Serialisable form for AgentState."""
        out = asdict(self)
        del out["evidence_chunk"], out["evidence_figure"]
        return out


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
            for i, a in enumerate(basis_points):
                for b in basis_points[i + 1:]:
                    if close(target, abs(a - b), s):
                        return f"|{a:g} bps - {b:g} bps|"
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

    Per segment, every figure is first looked up in the context and bound to
    the row or sentence it sits in (ContextIndex.bind); the ones not found are
    tried as derivations from the segment's own grounded figures (see _derive).
    What is still unexplained is "derived_unverified" only if it is worded as a
    calculation AND there are figures it could have been calculated from, in
    its claim or elsewhere in the answer; otherwise it is "unsupported".

    Args:
        texts: Answer segments, citation markers already stripped.
        index: ContextIndex over the retrieved chunks.

    Returns:
        One list of NumericCheck per input text, in order.
    """
    results: list[list[NumericCheck]] = []
    segments: list[tuple] = []
    answer_operands: list[tuple[float, str]] = []

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
        explained = {fig.unit for group in found for fig in group}
        overlaps = index.unit_overlaps(claim_tokens) if any(found) else {}

        for mention, group in zip(mentions, found):
            if group:
                binding = index.bind(
                    group,
                    claim_tokens,
                    stated.get((mention.start, mention.end), ""),
                    mention.kind,
                    explained,
                    overlaps,
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
                    evidence_chunk=fig.chunk_index,
                    evidence_figure=fig.mention.raw,
                )
                if fig.mention.kind == "bps" and mention.kind == "percent":
                    # Found in the other notation: a conversion, shown as one.
                    check.status = "derived"
                    check.formula = f"{fig.mention.value:g} bps / 100"
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

        segments.append((text, mentions, checks, pending, list(operands)))
        answer_operands.extend(
            (value, check.kind) for _, check, value in grounded if check.is_confirmed
        )

    # Second pass: what to make of a figure that is neither in the context nor
    # reproducible from its own claim. It needs every claim's confirmed figures,
    # so it runs after all of them are grounded.
    answer_operands = list(dict.fromkeys(answer_operands))[:_MAX_ANSWER_OPERANDS]
    attempts = 0
    for text, mentions, checks, pending, operands in segments:
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
            if not (_COMPUTATION_CUE_RE.search(lead_in) or _COMPUTED_CLAIM_RE.search(text)):
                continue
            # Wording alone does not make a figure a calculation: "revenue
            # increased to $999.9 million" used to be downgraded from a failure
            # to a warning by the word "increased". There must be something it
            # could have been calculated FROM — two like figures in the same
            # claim, or inputs stated elsewhere in the answer that reproduce it.
            plausible = _has_inputs(mention, operands)
            if not plausible and attempts < _MAX_ANSWER_DERIVATIONS:
                attempts += 1
                plausible = _derive(mention, answer_operands) is not None
            if plausible:
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
                cur.evidence_chunk = prev.evidence_chunk
                cur.evidence_figure = prev.evidence_figure

        results.append(checks)

    return results


# Bounds on the answer-wide plausibility check: an answer is model output, and
# trying every pair of its figures for every unexplained one is cubic.
_MAX_ANSWER_OPERANDS = 200
_MAX_ANSWER_DERIVATIONS = 50


def _has_inputs(mention: NumericMention, operands: list[tuple[float, str]]) -> bool:
    """
    True when a claim states figures its unexplained figure could come from.

    Args:
        mention: The figure that is not in the context.
        operands: (value, kind) of the claim's grounded and derived figures.

    Returns:
        Whether at least two like figures stand beside it: two amounts for an
        amount; two amounts or two ratios for a ratio. A quantity ("27 months")
        is never treated as a calculation.
    """
    if mention.kind == "quantity":
        return False
    amounts = sum(1 for _, kind in operands if kind in ("currency", "count"))
    if mention.kind in ("currency", "count"):
        return amounts >= 2
    ratios = sum(1 for _, kind in operands if kind in _RATIO_KINDS)
    return amounts >= 2 or ratios >= 2


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
