"""
Numerical registry — collects labelled financial figures across documents and
detects cross-document inconsistencies deterministically.

Used by Agent 4 (Financial Verifier) on a question's retrieved context, and by
the deal-level tie-out report (src/reports/tie_out.py) on a whole data room.
Figures come from three places:

- structured payloads written at ingestion (`structured_rows` on row-by-row
  table chunks, or a chunk-level `metric_name` / `normalized_value`);
- the text of converted tables (PDF, DOCX, XLSX), in the forms the table
  converter writes: "Revenue: FY2023=452,800,000 FY2022=387,100,000" (row by
  row), "Revenue: FY2023: 452,800,000; FY2022: 387,100,000" (narrative) and a
  markdown grid. Ingestion builds `structured_rows` but does not carry it into
  the stored payload, so without this the registry saw no converted table at
  all and the verifier reported "no labelled figures" for every non-.txt
  document;
- fixed-width tables in text chunks, parsed line by line: a period header
  ("FY2023   FY2022") followed by label + value rows ("Revenue  $452.8  $387.1").
  A row of a pipe-separated table ("Net income | $ | 1,263 | | $ | 709") takes
  its label from its first cell only, so a footnote marker in the label
  ("Other (2)") is never read as the first value.

An inconsistency is the same labelled metric, for the same period, reported
with different values by DIFFERENT sources, after unit normalisation and
allowing for the printed rounding. Two sources disagreeing is a fact about the
data room that a reviewer must see; a single source restating a figure is not.

Periods are read so that two figures share one only when they cover the same
time (Decision 92). A header names fiscal years ("FY2023", "2023") unless its
block says otherwise: "Three Months Ended" or "26 weeks ended" columns are
keyed by their span and end date ("3M ending 2023-07-29"), a balance-sheet date
by the date itself ("2023-07-29"; a 31 December date in an annual document is
that fiscal year), a fiscal year ending in January–March by its end month (such
years are named either way), and an estimate as "FY2024E". A bare year in a
10-Q, an 8-K or an earnings release is not taken for a fiscal year, since those
documents print quarters and year-to-date columns under the same years. A line
of prose that happens to name a year is not a header.

Which line item a figure is (Decision 93): a figure read under a balance date
is never compared with one read under a year's or a quarter's span, though
both may be keyed "FY2020"; a line item a table prints more than once, under
different subheadings, is read with its subheading ("Swaptions: Pre-tax gain
recognized in Interest expense"); and a table is read once — where its
structured rows carry its periods, its markdown and narrative forms are not
read again. A month/day/year date ("3/25/2023") is read as the date.

CRITICAL: Compare normalised values only. Raw values across documents with
different scale factors produce false inconsistencies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

from src.utils.casefold import ascii_lower
from src.utils.logger import setup_logger
from src.verification.numeric_grounding import (
    SCALE_HINT_RE as _SCALE_HINT,
    SCALE_HINT_WORDS as _SCALE_WORD,
    extract_numbers,
)

logger = setup_logger(__name__)

# Period tokens recognised in table headers and structured-row columns.
_PERIOD = re.compile(
    r"\b(?:FY\s?'?(?:19|20)?\d{2}|(?:Q[1-4]|H[12])\s?(?:FY)?\s?'?(?:19|20)?\d{2}|"
    r"(?:19|20)\d{2}[EA]?|LTM|TTM)\b",
    re.IGNORECASE,
)
_RULE_LINE = re.compile(r"^\s*[=\-_*~]{4,}\s*$")
# "FY2023=452,800,000" / "FY2023: 452,800,000" — a period-keyed value in the
# text of a converted table. The period is the same token _PERIOD matches.
_KEYED_VALUE = re.compile(
    r"(?P<period>" + _PERIOD.pattern + r")\s?[=:]\s?(?P<value>\(?-?\$?\d[\d,]*(?:\.\d+)?\)?%?)",
    re.IGNORECASE,
)
# A line-item label is a few words. Anything longer, or with sentence
# punctuation, is prose sitting in a label position — and labels are quoted into
# the synthesis prompt, so that prose could be an instruction (see
# prompt_safety.wrap_finding).
_MAX_LABEL_CHARS = 80
_SENTENCE_IN_LABEL = re.compile(r"[.!?]\s+\S")
_LABEL_PREFIX = re.compile(r"^(?:add(?:\s+back)?|less|plus|minus)\s*:\s*", re.IGNORECASE)
# Parentheticals that annotate a figure without changing what it measures.
# "(Non-Current)" or "(excluding restructuring)" DO change it, so they stay.
_ANNOTATION = re.compile(
    r"\(\s*(?:note\s*\d+[a-z]?|as\s+reported|reported|audited|unaudited|restated)\s*\)",
    re.IGNORECASE,
)
# Footnote markers: "(a)", "(ii)", "(1)", "[2]", "*", "†". One filing writes
# "Changes in working capital (b)" and the next "(a)" for the same line.
_FOOTNOTE = re.compile(
    r"\(\s*(?:[a-z]|i{1,3}|iv|v|vi{1,3}|ix|x|\d{1,2})\s*\)|\[\s*\w{1,2}\s*\]|[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§]",
    re.IGNORECASE,
)
_SCALES = (1.0, 1e3, 1e6, 1e9)
# Rows a caption's "except ..." clause typically excludes from its scale.
_PER_ROW_UNIT = re.compile(
    # Not "shares": share counts are usually stated in the caption's unit.
    r"\b(?:per\s+(?:share|unit|employee)|eps|headcount|employees|ratio|leverage|"
    r"coverage|days|months)\b",
    re.IGNORECASE,
)

# ── Periods (Decision 92) ──────────────────────────────────────────────────

_MONTH_WORD = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_MONTH_NUMBER = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_ONLY = re.compile(rf"\b{_MONTH_WORD}\b\.?", re.IGNORECASE)
# A day and month printed right before a year: "December 31, 2023",
# "Dec. 31 2023", "31 December 2023".
_DATE_BEFORE_YEAR = re.compile(
    rf"(?:\b(?P<m1>{_MONTH_WORD})\.?\s+(?P<d1>\d{{1,2}})|\b(?P<d2>\d{{1,2}})\s+(?P<m2>{_MONTH_WORD})\.?)"
    r",?\s*$",
    re.IGNORECASE,
)
# A day and month with no year after it: "Three Months Ended March 31," above a
# row of years.
_MONTH_DAY = re.compile(
    rf"\b(?P<m>{_MONTH_WORD})\.?\s+(?P<d>\d{{1,2}})\b(?!,?\s*(?:19|20)\d{{2}})",
    re.IGNORECASE,
)
# A day and month, a year after it or not: a column headed by a date.
_MONTH_DAY_ANY = re.compile(rf"\b{_MONTH_WORD}\.?\s+\d{{1,2}}\b", re.IGNORECASE)
# A date written month/day/year, as PepsiCo's releases head their columns
# ("12 Weeks Ended 3/25/2023").
_NUMERIC_DATE = re.compile(r"(?<![\d/])\d{1,2}/\d{1,2}/(?:19|20)\d{2}(?![\d/])")
_NUMERIC_DATE_BEFORE_YEAR = re.compile(r"(?<![\d/])(?P<m>\d{1,2})/(?P<d>\d{1,2})/$")
# What a column's figures cover. A quarter is three months, a 13-week quarter
# too; "52 weeks" and "twelve months" are a year.
_SPAN = re.compile(
    r"\b(?P<months>three|six|nine|twelve|3|6|9|12)[\s-]+months?\b"
    r"|\b(?P<weeks>twelve|thirteen|fourteen|sixteen|seventeen|twenty[\s-]?(?:four|six|seven|eight)"
    r"|thirty[\s-]?(?:six|nine)|forty|fifty[\s-]?(?:two|three)|\d{1,2})[\s-]+weeks?\b"
    r"|\b(?P<quarter>(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter|quarters?\s+end(?:ed|ing))\b"
    r"|\b(?P<half>half[\s-]year|first\s+half|second\s+half)\b"
    r"|\b(?P<ytd>ytd|year[\s-]to[\s-]date|interim)\b"
    r"|\b(?P<annual>years?\s+end(?:ed|ing)|fiscal\s+years?|full[\s-]year)\b",
    re.IGNORECASE,
)
_SPAN_MONTHS = {"three": 3, "six": 6, "nine": 9, "twelve": 12}
_SPAN_WEEKS = {
    "twelve": 12, "thirteen": 13, "fourteen": 14, "sixteen": 16, "seventeen": 17,
    "twentyfour": 24, "twentysix": 26, "twentyseven": 27, "twentyeight": 28,
    "thirtysix": 36, "thirtynine": 39, "forty": 40, "fiftytwo": 52, "fiftythree": 53,
}
# Columns that are not actuals: never compared, even with each other.
_NOT_ACTUAL = re.compile(
    r"\bbudget|\bforecast|\bplan\b|\bproject(?:ed|ion)|\bestimate|\bguidance\b|\boutlook\b"
    r"|\btarget\b|\bpro[\s-]?forma\b|\bscenario\b",
    re.IGNORECASE,
)
# A header is a short line. Prose that happens to name a year ("Application,
# which remains pending ... In November 2020, Cal Am refiled ...") headed every
# row after it in a real 10-K, and those rows were then compared as 2020 figures.
_WORD = re.compile(r"[A-Za-z]{2,}")
# Words outside parentheses: a caption in them is part of a header ("SECTION
# 1: OUTSTANDING INDEBTEDNESS (in millions of USD, as of December 31, 2023)"),
# while a wrapped paragraph line ("Note 12 — Subsequent Events: On February
# 15, 2024, the Company's Board of Directors approved the exploration") runs on.
_MAX_HEADER_WORDS = 8
_PARENTHETICAL = re.compile(r"\([^()]*\)")
_MAX_CONTEXT_WORDS = 10
_PROSE_WORDS = 12
_SENTENCE_END = re.compile(r"[a-z]{2}[.!?](?:\s+[A-Z]|\s*$)")
_COMPARISON = re.compile(r"\b(?:vs|versus|compared)\b", re.IGNORECASE)
# The converter's name for a column whose header it could not read.
_COLUMN_PLACEHOLDER = re.compile(r"\bColumn \d+\b")
# A nil cell as statements print it — a dash, or the replacement character a
# PDF's em dash often decodes to — ending a row's label: "Principal payments
# $  —  $" (a dash inside a label, "Revenue - Software", is part of it).
_NIL_CELL = re.compile(r"(?:^|\s)[—–\-�](?:\s+\$)?\s*$")
_CURRENCY_CELL = {"$", "€", "£", "¥", "₹", "US$", "USD", "EUR", "GBP"}
# Filings whose bare years are not fiscal years: a 10-Q prints quarter and
# year-to-date columns under "2023 2022", and an 8-K or a release usually is one.
_INTERIM_FORMS = {"10-Q", "8-K", "EARNINGS_RELEASE"}

# Words that never make a line-item label on their own: "Total", "Other, net"
# or "Balance at December 31" name a different figure in every table they
# close, so two documents' "Total" for 2023 are not one figure.
_GENERIC_LABEL_WORDS = frozenset(
    "total totals subtotal sub other others net balance balances amount amounts beginning "
    "end ending of at the year years period periods and as for all adjustment adjustments "
    "elimination eliminations corporate unallocated consolidated reconciling items item "
    "various miscellaneous line row column value values in on current noncurrent non long "
    "short term change changes increase decrease less plus add before after from to by "
    "per percent cent following prior same average weighted basic diluted high low "
    "domestic international foreign".split()
)
# A row whose first value is exactly 100 is a rebased index — a 10-K's
# performance graph restarts at 100 in a different year in every filing — or a
# common-size statement's 100%.
_INDEX_BASE = 100.0
_MONTH_TOKENS = frozenset(
    "jan january feb february mar march apr april may jun june jul july aug august sep sept "
    "september oct october nov november dec december".split()
)


def normalize_metric_label(label: str) -> str:
    """
    Canonical form of a line-item label, for matching across documents.

    Drops add/less prefixes, annotation parentheticals ("(Note 1)", "(as
    reported)"), footnote markers ("(a)", "(ii)", "(1)", "†") and the word
    "reported", so the QoE report's "Reported EBITDA" and the financials'
    "EBITDA" compare as one metric — which they are. Other parentheticals are
    kept: "Deferred Revenue (Non-Current)" is not "Deferred Revenue".

    Args:
        label: Raw label text.

    Returns:
        Lower-case, punctuation-free label.
    """
    text = _LABEL_PREFIX.sub("", label.strip())
    text = _ANNOTATION.sub(" ", text)
    text = _FOOTNOTE.sub(" ", text)
    text = re.sub(r"\breported\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9%/&+ ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def is_generic_label(label: str) -> bool:
    """
    True for a label that names no figure on its own ("Total", "Other, net",
    "Balance at December 31", "Row 3").

    Args:
        label: Raw or normalised label.

    Returns:
        Whether every word of it is generic, a month or a number.
    """
    tokens = normalize_metric_label(label).split()
    return not tokens or all(
        t in _GENERIC_LABEL_WORDS or t in _MONTH_TOKENS or t.isdigit() for t in tokens
    )


def _is_line_item_label(label: str) -> bool:
    """True for text that reads as a table row's label rather than as prose."""
    return (
        len(re.findall(r"[A-Za-z]", label)) >= 3
        and len(label) <= _MAX_LABEL_CHARS
        and not _SENTENCE_IN_LABEL.search(label)
    )


def normalize_period(token: str) -> str:
    """Canonical period form: "FY 2023", "FY'23", "fy23" -> "FY2023"; "2024E" -> "FY2024E"."""
    t = re.sub(r"[\s']", "", token.upper())
    m = re.fullmatch(r"FY(\d{2})", t)
    if m:
        return f"FY20{m.group(1)}"
    m = re.fullmatch(r"(19|20)(\d{2})([EA]?)", t)
    if m:
        # An estimate column is not the year's actuals (Decision 92).
        return f"FY{m.group(1)}{m.group(2)}" + ("E" if m.group(3) == "E" else "")
    return t


def _is_interim(chunk: dict) -> bool:
    """True for a chunk of a 10-Q, an 8-K or an earnings release (filing_metadata.py)."""
    form = str(chunk.get("filing_form") or "").upper()
    return form.removesuffix("/A") in _INTERIM_FORMS


_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _horizon(chunk: dict) -> tuple[int, int, int] | None:
    """
    The last day a public filing reports on (filing_metadata.py): its period
    end, else the end of its latest fiscal year. None for any other document.
    """
    end = chunk.get("filing_period_end")
    found = _ISO_DATE.fullmatch(end) if isinstance(end, str) else None
    if found:
        return int(found.group(1)), int(found.group(2)), int(found.group(3))
    years = [y for y in chunk.get("filing_fiscal_years") or [] if isinstance(y, int) and not isinstance(y, bool)]
    return (max(years), 12, 31) if years else None


def _is_future(period: str, horizon: tuple[int, int, int] | None) -> bool:
    """
    True for a period after the last one its filing reports: a debt maturity
    or lease schedule ("2024  2025  2026") states what is due then as seen at
    the filing date, so the next year's filing states other figures for the
    same years — never a restatement, and not a figure to tie out.
    """
    if horizon is None:
        return False
    if period.startswith("FY") and period[2:6].isdigit() and len(period) >= 6:
        return int(period[2:6]) > horizon[0]
    found = _ISO_DATE.search(period)
    if found:
        return (int(found.group(1)), int(found.group(2)), int(found.group(3))) > horizon
    found = re.search(r"FY ending (\d{4})-(\d{2})", period)
    if found:
        return (int(found.group(1)), int(found.group(2))) > horizon[:2]
    return False


def _month_day(month: str, day: str) -> tuple[int, int] | None:
    """(month, day) for a matched month word and day, or None for an impossible date."""
    number = _MONTH_NUMBER.get(ascii_lower(month)[:3])
    try:
        d = int(day)
    except ValueError:
        return None
    return (number, d) if number and 1 <= d <= 31 else None


def _date_before(line: str, start: int) -> tuple[int, int] | None:
    """The (month, day) printed immediately before position `start`, if any."""
    numeric = _NUMERIC_DATE_BEFORE_YEAR.search(line[max(0, start - 6):start])
    if numeric:
        # "12 Weeks Ended 3/25/2023": a US filing's month/day/year.
        month, day = int(numeric.group("m")), int(numeric.group("d"))
        return (month, day) if 1 <= month <= 12 and 1 <= day <= 31 else None
    found = _DATE_BEFORE_YEAR.search(line[max(0, start - 24):start])
    if not found:
        return None
    if found.group("m1"):
        return _month_day(found.group("m1"), found.group("d1"))
    return _month_day(found.group("m2"), found.group("d2"))


def column_basis(header: str) -> str:
    """
    What a column's figures are, as its header says: "flow" under a span (a
    year, quarter or year-to-date ended), "balance" under a date with no span
    ("December 31, 2020", "At May 31, 2023"), "" when the header does not say
    (a bare year, "FY2023").

    The registry keys a 31 December balance in an annual document as that
    fiscal year, so that it meets the same balance printed under "2020"; it
    then met a cash-flow line of the same name too — American Water Works'
    "Accrued taxes", a balance of 50 against a change of 3 (Decision 93).
    Figures of the two kinds are never compared.

    Args:
        header: Column header, with any header lines over it.

    Returns:
        "flow", "balance" or "".
    """
    if _SPAN.search(header):
        return "flow"
    if _MONTH_DAY_ANY.search(header) or _NUMERIC_DATE.search(header):
        return "balance"
    return ""


def _span_kind(match: re.Match) -> str | None:
    """'FY', '3M', '6M', '9M' or 'YTD' for one span marker; None when unreadable."""
    if match.group("months"):
        word = ascii_lower(match.group("months"))
        n = _SPAN_MONTHS.get(word) or (int(word) if word.isdigit() else 0)
        return {3: "3M", 6: "6M", 9: "9M", 12: "FY"}.get(n)
    if match.group("weeks"):
        word = re.sub(r"[\s-]", "", ascii_lower(match.group("weeks")))
        n = _SPAN_WEEKS.get(word) or (int(word) if word.isdigit() else 0)
        if n in (52, 53):
            return "FY"
        if 12 <= n <= 17:
            return "3M"
        if 24 <= n <= 28:
            return "6M"
        if 36 <= n <= 40:
            return "9M"
        return None
    if match.group("quarter"):
        return "3M"
    if match.group("half"):
        return "6M"
    if match.group("ytd"):
        return "YTD"
    return "FY"


def _reads_like_header(line: str) -> bool:
    """A short line of column titles, not a sentence that happens to name a year."""
    text = _MONTH_ONLY.sub(" ", _PERIOD.sub(" ", _PARENTHETICAL.sub(" ", line)))
    if len(_WORD.findall(text)) > _MAX_HEADER_WORDS:
        return False
    if _SENTENCE_END.search(line) or line.rstrip(" |").endswith("."):
        return False
    return not _COMPARISON.search(line)


def _token_key(token: re.Match, line: str, kind: str, shared: tuple[int, int] | None,
               interim: bool) -> str | None:
    """
    The period one header token heads, or None when it cannot be compared.

    Args:
        token: A _PERIOD match in `line`.
        line: The header line.
        kind: The span heading the token's column: "" (none stated), "FY",
            "3M", "6M", "9M" or "YTD".
        shared: A (month, day) the header block prints once for every column.
        interim: Whether the document is a 10-Q, 8-K or earnings release.

    Returns:
        Period key.
    """
    compact = re.sub(r"[\s']", "", token.group(0).upper())
    if compact in ("LTM", "TTM"):
        return None  # a trailing twelve months ending when?
    if compact[0] in "QH":
        return normalize_period(token.group(0))  # "Q1 2023" names its own span
    if compact.startswith("FY"):
        return normalize_period(token.group(0)) if kind in ("", "FY") else None
    year, suffix = int(compact[:4]), compact[4:]
    if suffix == "E":
        return f"FY{year}E" if kind in ("", "FY") else None
    month_day = _date_before(line, token.start()) or shared
    if kind == "":
        if month_day:
            if month_day == (12, 31) and not interim:
                return f"FY{year}"
            return f"{year}-{month_day[0]:02d}-{month_day[1]:02d}"
        return None if interim else f"FY{year}"
    if kind == "FY":
        if month_day and month_day[0] <= 3:
            # A year ending in January–March is named for either calendar
            # year, so it is keyed by its end (Decision 70's problem).
            return f"FY ending {year}-{month_day[0]:02d}"
        return f"FY{year}"
    if month_day:
        return f"{kind} ending {year}-{month_day[0]:02d}-{month_day[1]:02d}"
    return None


def _header_keys(line: str, context: list[str], interim: bool) -> list[str | None]:
    """
    One period key per period token of a header line.

    The span of each column ("Three Months Ended", "Year Ended") may be printed
    on the line itself or on the header lines above it (`context`); when two
    spans head the columns ("Three Months Ended | Six Months Ended" over four
    dates) they are dealt out to the columns in equal blocks, and when they
    cannot be, no column is compared.

    Args:
        line: The header line.
        context: Up to two header-like lines printed above it.
        interim: Whether the document is a 10-Q, 8-K or earnings release.

    Returns:
        Keys, None for a column whose period cannot be read safely.
    """
    tokens = list(_PERIOD.finditer(line))
    kinds = [_span_kind(m) for m in _SPAN.finditer("\n".join([*context, line]))]
    count = len(tokens)
    if not kinds:
        per_token = [""] * count
    elif None in kinds:
        return [None] * count
    elif len(set(kinds)) == 1:
        per_token = [kinds[0]] * count
    elif count % len(kinds) == 0:
        per_token = [k for k in kinds for _ in range(count // len(kinds))]
    else:
        return [None] * count

    # A month and day printed once for every column: "December 31," above the
    # years, or before the first of them only ("December 31, 2023  2022").
    shared = None
    days = {md for md in (_month_day(m.group("m"), m.group("d"))
                          for c in context for m in _MONTH_DAY.finditer(c)) if md}
    if len(days) == 1:
        shared = days.pop()
    elif not days and tokens:
        dated = [_date_before(line, t.start()) for t in tokens]
        if dated[0] and not any(dated[1:]):
            shared = dated[0]
    keys = [_token_key(t, line, kind, shared, interim) for t, kind in zip(tokens, per_token)]
    return [None] * count if _event_dates(keys) else keys


def _event_dates(keys: list[str | None]) -> bool:
    """
    True when a header's balance-date columns hold three or more different
    dates of one calendar year: those are events — acquisitions, issuances —
    or quarters whose span the header lost, not reporting dates. PepsiCo's
    "Acquisition date | March 23, 2020 | April 24, 2020 | June 1, 2020" was
    stitched to the charges table below it, whose columns are fiscal years,
    and its rows were compared as of those dates across two 10-Ks.
    """
    by_year: dict[str, set[str]] = {}
    for key in keys:
        if key and _ISO_DATE.fullmatch(key):
            by_year.setdefault(key[:4], set()).add(key)
    return any(len(dates) >= 3 for dates in by_year.values())


def column_period(header: str, interim: bool = False) -> str | None:
    """
    The period a converted table's column header names, read as a text header
    is read (see _header_keys), or None.

    Args:
        header: Column header as the converter keeps it ("2023", "Year Ended
            December 31, 2023", "July 29, 2023 Three Months Ended",
            "FY2023 Budget", "Column 3").
        interim: Whether the document is a 10-Q, 8-K or earnings release.

    Returns:
        Period key, or None for a column with no single readable period, or
        one that is not actuals.
    """
    if _NOT_ACTUAL.search(header) or len(_PERIOD.findall(header)) != 1:
        return None
    return _header_keys(header, [], interim)[0]


def _pipe_label(line: str) -> tuple[str, int]:
    """The first cell of a pipe-separated row, and where its value cells start."""
    offset = len(line) - len(line.lstrip())
    if line[offset:offset + 1] == "|":
        offset += 1
    end = line.find("|", offset)
    if end < 0:
        return line[offset:].strip(), len(line)
    return line[offset:end].strip(), end


@dataclass
class NumericalEntry:
    """A single labelled figure from a source document."""

    metric_name: str
    raw_value: float
    normalized_value: float
    currency: str = "USD"
    scale_factor: float = 1.0
    source_file: str = ""
    page_number: int = 0
    fiscal_year: str = ""
    is_computed: bool = False
    citation_chain: str = ""
    kind: str = "currency"
    decimals: int = 1
    scale_known: bool = True
    raw_text: str = ""
    # Where the figure sits in its document, for a reader without page numbers.
    section_heading: str = ""
    # "flow" or "balance" when its column header says which (column_basis).
    basis: str = ""

    @property
    def tolerance(self) -> float:
        """Half of the last printed digit, in normalised units."""
        return 0.5 * (10 ** -self.decimals) * self.scale_factor

    @property
    def as_printed(self) -> str:
        """The figure as its document prints it."""
        return self.raw_text.split("  ")[-1] if self.raw_text else ""


@dataclass
class MetricComparison:
    """Comparison result for a metric/period across sources."""

    metric_name: str
    entries: list[NumericalEntry] = field(default_factory=list)
    is_consistent: bool = True
    max_deviation_pct: float = 0.0
    discrepancy_detail: str = ""
    fiscal_year: str = ""


@dataclass
class FigureTieOut:
    """One labelled figure stated by two or more documents (Decision 92)."""

    metric_name: str
    fiscal_year: str
    kind: str  # "amount" | "percent" | "multiple"
    status: str  # "conflict" | "agree"
    entries: list[NumericalEntry] = field(default_factory=list)  # one per document
    max_deviation_pct: float = 0.0


@dataclass
class TieOutResult:
    """Every multi-source figure of a registry, and what was left out and why."""

    figures: list[FigureTieOut] = field(default_factory=list)  # conflicts first
    single_source: int = 0  # figures one document states
    ambiguous: int = 0  # a document states two different values for one label and period
    generic: int = 0  # labels that name no figure on their own ("Total", "Other")
    without_period: int = 0  # figures read with no period to compare them on

    @property
    def conflicts(self) -> list[FigureTieOut]:
        return [f for f in self.figures if f.status == "conflict"]

    @property
    def agreements(self) -> list[FigureTieOut]:
        return [f for f in self.figures if f.status == "agree"]


def _values_agree(a: NumericalEntry, b: NumericalEntry) -> bool:
    """
    True when two entries state the same figure, allowing for rounding and — if
    either side's unit is unknown — for a power-of-1,000 scale difference.
    """
    if a.kind != b.kind and {a.kind, b.kind} & {"percent", "multiple"}:
        return False
    if a.scale_known and b.scale_known:
        return abs(a.normalized_value - b.normalized_value) <= max(a.tolerance, b.tolerance, 1e-9)
    # Rescale only the side whose unit is unknown, and its rounding slack with
    # it. Multiplying the combined slack by the scale — as this used to — let a
    # millions table's 50,000 slack grow to 5e10 and call 452.8 and $448.1M equal.
    for x, y in ((a, b), (b, a)):
        if y.scale_known:
            continue
        # A zero is zero in any unit; rescaling it only rescaled its slack, so
        # an unstated "0" agreed with anything up to half a billion.
        for s in _SCALES if y.normalized_value else (1.0,):
            if abs(x.normalized_value - y.normalized_value * s) <= max(x.tolerance, y.tolerance * s, 1e-9):
                return True
    return False


def _kind_class(entry: NumericalEntry) -> str:
    """Percentages and multiples are other figures than the amount a label also names."""
    return entry.kind if entry.kind in ("percent", "multiple") else "amount"


def _aligned(entry: NumericalEntry, reference: NumericalEntry) -> float:
    """`entry`'s value at the scale that best matches `reference`, when a scale is unknown."""
    if entry.scale_known:
        return entry.normalized_value
    return min(
        (entry.normalized_value * s for s in _SCALES),
        key=lambda v: abs(v - reference.normalized_value),
    )


def _spread_pct(entries: list[NumericalEntry]) -> float:
    """How far apart the documents' values are, as a share of the larger one."""
    reference = next((e for e in entries if e.scale_known), entries[0])
    values = [_aligned(e, reference) for e in entries]
    top = max(abs(v) for v in values)
    return round((max(values) - min(values)) / max(top, 1e-9) * 100, 2)


class NumericalRegistry:
    """
    Collects labelled figures and compares them across documents.

    Entries are keyed by (normalised metric label, period). Only entries with a
    period take part in consistency checks: "Revenue $452.8" and "Revenue
    $387.1" are not a contradiction if one is FY2023 and the other FY2022, and a
    figure whose period is unknown cannot be compared safely.
    """

    def __init__(self):
        self._entries: dict[tuple[str, str], list[NumericalEntry]] = {}
        # Per table (document, table_id): the line items it prints more than
        # once, and the subheading in force where its last chunk read ended.
        self._repeats: dict[tuple[str, str], set[str]] = {}
        self._subheadings: dict[tuple[str, str], tuple[int, str]] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def register(self, entry: NumericalEntry) -> None:
        """
        Registers a figure, ignoring exact duplicates from the same source.

        Args:
            entry: NumericalEntry to register.
        """
        if not math.isfinite(entry.normalized_value):
            # A run of digits too long for a float is no figure: compared, its
            # infinite value made a conflict's spread NaN (found by the fuzz
            # profile, on main's registry too).
            return
        key = (normalize_metric_label(entry.metric_name), entry.fiscal_year)
        bucket = self._entries.setdefault(key, [])
        for i, existing in enumerate(bucket):
            if existing.source_file == entry.source_file and _values_agree(existing, entry):
                # The same figure is often seen twice: read from a table's text
                # (scale unknown) and from its structured rows (scale stated).
                # Chunk order decides which arrives first, so keep the one that
                # knows its scale rather than whichever came first.
                if entry.scale_known and not existing.scale_known:
                    bucket[i] = entry
                return
        bucket.append(entry)

    # ── Building from chunks ───────────────────────────────────────────────

    @classmethod
    def from_chunks(cls, chunks: list[dict]) -> "NumericalRegistry":
        """
        Builds a registry from retrieved chunks.

        Args:
            chunks: Context chunk payloads.

        Returns:
            Populated NumericalRegistry.
        """
        registry = cls()
        registry._prepare(chunks)
        ranges = _structured_ranges(chunks)
        for chunk in chunks:
            if not _read_elsewhere(chunk, ranges):
                registry.add_chunk(chunk)
        return registry

    @classmethod
    def from_documents(
        cls,
        chunks: list[dict],
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple["NumericalRegistry", int]:
        """
        Builds a registry from every chunk of a data room, document by document.

        A retrieved context is a handful of chunks in rank order; a data room is
        whole documents, read in their own order (`chunk_index`), so a header
        can head what follows it in the same document: a table the chunker cut
        in two keeps its period header in the second half (same `table_id`),
        and a table section that prints no header of its own reads the
        document's only one — the sample financials' "KEY FINANCIAL METRICS"
        rows sit under the "FY2023  FY2022" of the statements above them. A
        document that prints more than one period header lends none. A table
        is read once (see _read_elsewhere), and a line item it repeats under
        subheadings is read with its subheading (see _register_structured).

        Args:
            chunks: Child chunk payloads of one deal. Documents (by
                `source_file`, the unit figures are compared between) are read
                in the order they first appear.
            should_stop: Polled before each document; True stops the build.

        Returns:
            (registry, documents_read): fewer than all documents when
            `should_stop` ended the build.
        """
        registry = cls()
        registry._prepare(chunks)
        ranges = _structured_ranges(chunks)
        documents: dict[str, list[dict]] = {}
        for chunk in chunks:
            documents.setdefault(str(chunk.get("source_file") or "unknown"), []).append(chunk)
        for read, document in enumerate(documents.values()):
            if should_stop is not None and should_stop():
                return registry, read
            ordered = sorted(
                document,
                key=lambda c: (
                    str(c.get("doc_id") or ""),
                    c.get("chunk_index") if isinstance(c.get("chunk_index"), int) else 0,
                ),
            )
            scheme = _document_scheme(ordered)
            last_header: dict[str, list[str | None]] = {}
            for chunk in ordered:
                if _read_elsewhere(chunk, ranges):
                    continue
                table_id = chunk.get("table_id") or ""
                inherited = last_header.get(table_id) if table_id else None
                if inherited is None and chunk.get("is_table"):
                    inherited = scheme
                trailing = registry.add_chunk(chunk, inherited=inherited)
                if table_id and trailing:
                    last_header[table_id] = trailing
        return registry, len(documents)

    def _prepare(self, chunks: list[dict]) -> None:
        """Notes, per table, the line items it prints more than once (see _register_structured)."""
        rows_by_table: dict[tuple[str, str], list[dict]] = {}
        for chunk in chunks:
            if chunk.get("table_id") and chunk.get("structured_rows"):
                rows_by_table.setdefault(_table_key(chunk), []).extend(chunk["structured_rows"])
        for key, rows in rows_by_table.items():
            self._repeats[key] = _repeated_labels(rows)

    def _subheading_before(self, chunk: dict) -> str:
        """The subheading over a chunk's first row: its table's previous chunk's last, when read just before."""
        span = _row_range(chunk)
        carried = self._subheadings.get(_table_key(chunk)) if chunk.get("table_id") else None
        if carried and span and carried[0] == span[0] - 1:
            return carried[1]
        return ""

    def _subheading_after(self, chunk: dict, subheading: str) -> None:
        span = _row_range(chunk)
        if chunk.get("table_id") and span:
            self._subheadings[_table_key(chunk)] = (span[1], subheading)

    def add_chunk(self, chunk: dict, inherited: list[str | None] | None = None) -> list[str | None]:
        """
        Registers every figure one chunk states.

        Args:
            chunk: Chunk payload.
            inherited: Period keys heading the chunk's first rows, from earlier
                in its document (see from_documents); None for a lone chunk.

        Returns:
            The period keys in force at the end of the chunk's text.
        """
        source = chunk.get("source_file", "") or "unknown"
        page = chunk.get("page_number") or 0
        self._register_structured(chunk, source, page)
        text = chunk.get("text", "") or ""
        parent = chunk.get("parent_text", "") or ""
        trailing: list[str | None] = []
        if text:
            trailing = self._register_text_tables(text, chunk, source, page, inherited)
        if parent:
            self._register_text_tables(parent, chunk, source, page)
        return trailing

    def _register_structured(self, chunk: dict, source: str, page) -> None:
        """
        Registers figures from structured payload fields, when present.

        A line item a table prints more than once, under different subheadings
        (Verizon's "Pre-tax gain recognized in Interest expense" under
        "Treasury Rate Locks:" and under "Swaptions:"), is that subheading's
        line: it is read as "Swaptions: Pre-tax gain recognized in Interest
        expense", and when its subheading is not within reach (an earlier chunk
        of the table that was not read) it is not read at all. Read by its
        label alone, the 2021 10-K's swaptions row met the 2022 10-K's
        interest-rate-swaps row as one figure (Decision 93).
        """
        interim = _is_interim(chunk)
        horizon = _horizon(chunk)
        section = str(chunk.get("section_heading") or "")
        rows = chunk.get("structured_rows") or []
        repeated = self._repeats.get(_table_key(chunk)) if chunk.get("table_id") else None
        if repeated is None:
            repeated = _repeated_labels(rows)
        subheading = self._subheading_before(chunk)
        for row in rows:
            label = str(row.get("line_item", "")).strip()
            if not label:
                continue
            cells = list((row.get("values") or {}).items())
            if not _has_values(row):
                subheading = label
                continue
            if _rebased_index([c.get("raw_value") for _, c in cells
                               if isinstance(c, dict) and "normalized_value" in c]):
                continue
            name = label
            if normalize_metric_label(label) in repeated:
                if not subheading:
                    continue
                name = f"{subheading.rstrip(' :')}: {label}"
            # The row's column headers, read once: dates of events are no periods.
            events = _event_dates([column_period(str(col), interim) for col, _ in cells])
            for index, (col, cell) in enumerate(cells):
                if not isinstance(cell, dict) or "normalized_value" not in cell:
                    continue
                try:
                    value = abs(float(cell["normalized_value"]))
                    raw = float(cell.get("raw_value", value))
                    scale = float(cell.get("scale_factor", 1.0) or 1.0)
                except (TypeError, ValueError):
                    continue
                period = None if events else self._structured_period(cells, index, interim)
                if period and _is_future(period, horizon):
                    continue
                # The basis of the column the period was read from ("$" cells
                # push an amount one column right of its header).
                header = str(col)
                if period and index and column_period(header, interim) != period:
                    header = str(cells[index - 1][0])
                raw_str = repr(raw)
                decimals = len(raw_str.split(".")[1]) if "." in raw_str and not raw_str.endswith(".0") else 0
                # Converter rows state the printed precision and whether the
                # cell is a percentage or a multiple; without the kind a "61.5%"
                # row would be compared with a text table's as a currency amount.
                if isinstance(cell.get("decimals"), int):
                    decimals = cell["decimals"]
                kind = cell.get("kind") if cell.get("kind") in ("percent", "multiple") else "currency"
                self.register(NumericalEntry(
                    metric_name=name,
                    raw_value=raw,
                    normalized_value=value,
                    currency=cell.get("currency", "USD") or "USD",
                    scale_factor=scale,
                    source_file=source,
                    page_number=page,
                    fiscal_year=period or "",
                    kind=kind,
                    decimals=decimals,
                    # A table that states no unit is written with scale 1 —
                    # "Values are as stated in the source; the table gives no
                    # currency or scale" — so 1 is not a stated scale: Best
                    # Buy's "Total rent expense 790" (millions, unit unstated)
                    # was a 100% conflict with "$790" million in the next 10-K.
                    scale_known=kind != "currency" or scale != 1.0 or abs(abs(raw) - value) > 1e-9,
                    # Two spaces before the value: find_inconsistencies reports
                    # what follows them as the figure "as printed".
                    raw_text=(
                        f"{name}  {cell['raw']}" if isinstance(cell.get("raw"), str)
                        else f"{name} {col}={raw}"
                    ),
                    section_heading=section,
                    basis=column_basis(header),
                ))
        self._subheading_after(chunk, subheading)

        if chunk.get("metric_name") and chunk.get("normalized_value") is not None:
            try:
                value = abs(float(chunk["normalized_value"]))
            except (TypeError, ValueError):
                return
            fy = chunk.get("fiscal_year") or ""
            self.register(NumericalEntry(
                metric_name=str(chunk["metric_name"]),
                raw_value=float(chunk.get("raw_value") or value),
                normalized_value=value,
                currency=chunk.get("currency", "USD") or "USD",
                scale_factor=float(chunk.get("scale_factor", 1.0) or 1.0),
                source_file=source,
                page_number=page,
                fiscal_year=normalize_period(str(fy)) if fy else "",
                is_computed=chunk.get("content_type") == "computed_metric",
                scale_known=True,
                section_heading=section,
            ))

    @staticmethod
    def _structured_period(cells: list[tuple[str, object]], index: int, interim: bool) -> str | None:
        """
        The period of one structured cell, read from its column header.

        The converter's own `period` ("FY2023" for any column naming 2023) is
        kept only as its verdict that a column names no period at all (""); the
        header itself is read as a text header is, so a 10-Q's "July 29, 2023"
        is that date and not fiscal 2023. A "$" printed in its own cell pushes
        the amount one column right of the header it belongs to ("| $ | 1,263 |"
        under "2021 | Column 3"); the amount takes the "$" cell's column.

        Args:
            cells: The row's (column header, cell) pairs in table order.
            index: Position of the cell being read.
            interim: Whether the document is a 10-Q, 8-K or earnings release.

        Returns:
            Period key, or None.
        """
        col, cell = cells[index]
        stated = cell.get("period") if isinstance(cell, dict) else None
        period = column_period(str(col), interim) if stated != "" else None
        if period is None and index:
            prev_col, prev = cells[index - 1]
            if (isinstance(prev, dict) and "normalized_value" not in prev
                    and str(prev.get("raw", "")).strip() in _CURRENCY_CELL):
                period = column_period(str(prev_col), interim)
        return period

    def _register_text_tables(self, text: str, chunk: dict, source: str, page,
                              inherited: list[str | None] | None = None) -> list[str | None]:
        """
        Parses fixed-width tables in text: a period header, then label/value rows.

        A row is registered only when its value count matches the header's
        period count, so a prose line that happens to end in a number is never
        mistaken for a table row. A line of prose resets the header.

        Returns:
            The period keys in force at the end of the text.
        """
        hint = _SCALE_HINT.search(text)
        chunk_scale = None
        if hint:
            # Read as numeric grounding reads it (ascii_lower: see src/utils/casefold.py).
            word = ascii_lower(next(g for g in hint.groups() if g))
            chunk_scale = _SCALE_WORD.get(word)
        if chunk_scale is None and chunk.get("scale_factor"):
            try:
                chunk_scale = float(chunk["scale_factor"])
            except (TypeError, ValueError):
                chunk_scale = None
            # Ingestion writes 1 when the document states no unit (see
            # _register_structured): that is not a stated scale.
            if chunk_scale == 1.0:
                chunk_scale = None
        # A table's caption scale does not apply to every row: "($ in millions,
        # except per share)" leaves EPS, headcount and ratios as printed, and a
        # text line does not say which rows are excepted. A converter table's
        # typed values come from its structured_rows (read above, per row), so
        # its text forms are registered scale-unknown — they still agree with
        # the structured figures at any power of 1,000, and can no longer
        # stamp EPS 2.74 as $2,740,000 (audit I1, here on the reading side).
        if chunk.get("table_id"):
            chunk_scale = None
        excepted = bool(re.search(r"\bexcept\b", text[:400], re.IGNORECASE))
        interim = _is_interim(chunk)
        horizon = _horizon(chunk)
        section = str(chunk.get("section_heading") or "")

        periods: list[str | None] = list(inherited or [])
        # What each column's figures are (column_basis); a table continued
        # from the chunk before keeps its header's (see _Header).
        carried = getattr(inherited, "bases", None)
        bases: list[str] = list(carried) if carried and len(carried) == len(periods) else [""] * len(periods)
        context: list[str] = []  # header lines printed above a period header
        for line in text.splitlines():
            if not line.strip() or _RULE_LINE.match(line):
                continue
            if _COLUMN_PLACEHOLDER.search(line):
                # "| Line Item | Column 2 | ...": a header the converter could
                # not read. Its column numbers are not figures.
                context = []
                continue
            # A chunk that carries structured_rows has been read from them; its
            # text is the same table again.
            if not chunk.get("structured_rows") and self._register_keyed_row(
                line, source, page, interim, section, horizon
            ):
                continue

            if _PERIOD.search(line):
                remainder = _PERIOD.sub(" ", line)
                if not extract_numbers(remainder, claims_only=False):
                    periods = _header_keys(line, context, interim) if _reads_like_header(line) else []
                    bases = _header_bases(line, context) if periods else []
                    context = []
                    continue

            numbers = extract_numbers(line, claims_only=False)
            piped = "|" in line
            if piped:
                # The label is the first cell: a footnote marker in it ("Other
                # (2)") must not be read as the row's first value.
                label, values_start = _pipe_label(line)
                numbers = [n for n in numbers if n.start >= values_start]
            if not numbers:
                if not piped and _is_prose(line):
                    periods, context = [], []
                elif _is_context_line(line):
                    context = [*context, line][-2:]
                continue
            context = []
            if not periods or len(numbers) != len(periods):
                if not piped and _is_prose(line, numbers):
                    periods = []
                continue
            if not piped:
                # "($8.2)" leaves its opening parenthesis on the label side.
                label = line[: numbers[0].start].strip(" |").rstrip("(").strip()
                if _NIL_CELL.search(label):
                    # "Principal payments  $  —  $ 1,000": a nil printed
                    # before the first value, so the values no longer sit
                    # under the columns the header names.
                    continue
            label = label.rstrip(":").strip()
            # Values must be the tail of the line — a table row, not prose. A
            # pipe table may print a value's "%" in a cell of its own.
            tail = line[numbers[-1].end:].strip(" )|%" if piped else " )|")
            if tail or not _is_line_item_label(label) or _rebased_index([n.value for n in numbers]):
                continue
            # A row whose label names its own period ("Net Debt / EBITDA
            # (FY2022)  0.7x" under a 2023 column) is not that column's figure.
            named = {normalize_period(t) for t in _PERIOD.findall(label)}
            if named and not all(p in named for p in periods if p):
                continue

            # Under a caption with an "except ..." clause, a per-share or count
            # row keeps its printed value; its scale is not the caption's.
            row_scale = None if (excepted and _PER_ROW_UNIT.search(label)) else chunk_scale
            for i, (period, num) in enumerate(zip(periods, numbers)):
                if not period or _is_future(period, horizon):
                    continue  # a column whose period cannot be read, or a schedule ahead
                following = line[num.end: numbers[i + 1].start if i + 1 < len(numbers) else len(line)]
                percent_cell = piped and re.fullmatch(r"[\s|]*%[\s|)]*", following) is not None
                unit_scaled = num.kind in ("percent", "pp", "bps", "multiple") or percent_cell
                scale_known = num.explicit_scale or unit_scaled or row_scale is not None
                scale = 1.0 if (num.explicit_scale or unit_scaled) else (row_scale or 1.0)
                self.register(NumericalEntry(
                    metric_name=label,
                    raw_value=num.value / (num.scale or 1.0),
                    normalized_value=num.value * scale,
                    scale_factor=num.scale * scale,
                    source_file=source,
                    page_number=page,
                    fiscal_year=period,
                    kind="percent" if (num.kind in ("percent", "pp") or percent_cell) else (
                        "multiple" if num.kind == "multiple" else "currency"
                    ),
                    decimals=num.decimals,
                    scale_known=scale_known,
                    raw_text=f"{label}  {num.raw}" + ("%" if percent_cell else ""),
                    section_heading=section,
                    basis=bases[i] if i < len(bases) else "",
                ))
        return _Header(periods, bases)

    def _register_keyed_row(self, line: str, source: str, page, interim: bool = False,
                            section: str = "", horizon: tuple[int, int, int] | None = None) -> bool:
        """
        Registers a row of a converted table from its text form.

        The table converter writes each row as "Label: FY2023=452,800,000
        FY2022=387,100,000" (row by row) or "Label: FY2023: 452,800,000;
        FY2022: 387,100,000" (narrative). Whether those values are multiplied
        out to full units or printed as the source states them ("452.8", unit
        stated once) depends on the converter version, so the scale is recorded
        as unknown and compared across a power of 1,000 (see _values_agree).
        This is the fallback for chunks without `structured_rows` — the sibling
        representations of a table, and payloads written before ingestion
        carried the structured form.

        Args:
            line: One line of chunk text.
            source: File the chunk came from.
            page: Page number, if any.
            interim: Whether the document is a 10-Q, 8-K or earnings release:
                its bare years are not fiscal years.
            section: Section heading of the chunk.
            horizon: The last day the filing reports on (see _is_future).

        Returns:
            True when the line was a period-keyed row (registered or not), so
            the fixed-width parser should leave it alone.
        """
        label, sep, rest = line.partition(":")
        if not sep or not _is_line_item_label(label.strip(" |-*")) or _PERIOD.search(label):
            return False
        pairs = list(_KEYED_VALUE.finditer(rest))
        # The whole row must be period-keyed values: "SaaS ARR Growth: 27.9% YoY
        # (SaaS ARR as of Dec 31, 2023: $210.5M)" has a period and a colon too.
        if not pairs or _KEYED_VALUE.sub("", rest).strip(" ;,|"):
            return False
        label = label.strip(" |-*")
        values = [extract_numbers(pair.group("value"), claims_only=False) for pair in pairs]
        if _rebased_index([found[0].value for found in values if found]):
            return True
        for pair, found in zip(pairs, values):
            period = _keyed_period(pair.group("period"), interim)
            if not found or not period or _is_future(period, horizon):
                continue
            num = found[0]
            self.register(NumericalEntry(
                metric_name=label,
                raw_value=num.value,
                normalized_value=num.value,
                scale_factor=1.0,
                source_file=source,
                page_number=page,
                fiscal_year=period,
                kind="percent" if num.kind in ("percent", "pp") else (
                    "multiple" if num.kind == "multiple" else "currency"
                ),
                decimals=num.decimals,
                scale_known=num.kind in ("percent", "pp", "multiple"),
                raw_text=f"{label}  {num.raw}",
                section_heading=section,
            ))
        return True

    # ── Comparison ─────────────────────────────────────────────────────────

    def check_consistency(self) -> list[MetricComparison]:
        """
        Compares every metric/period reported by more than one source.

        Returns:
            One MetricComparison per (metric, period) with 2+ distinct sources.
        """
        results = []
        for (_metric, period), entries in self._entries.items():
            if not period:
                continue
            sources = {e.source_file for e in entries}
            if len(sources) < 2:
                continue

            disagreements = []
            max_dev = 0.0
            for i, a in enumerate(entries):
                for b in entries[i + 1:]:
                    if a.source_file == b.source_file or _bases_clash(a, b) or _values_agree(a, b):
                        continue
                    base = max(abs(a.normalized_value), 1e-9)
                    dev = abs(a.normalized_value - b.normalized_value) / base * 100
                    max_dev = max(max_dev, dev)
                    disagreements.append(
                        f"{a.source_file}: {a.raw_text.split('  ')[-1] or a.normalized_value} vs "
                        f"{b.source_file}: {b.raw_text.split('  ')[-1] or b.normalized_value}"
                    )
            results.append(MetricComparison(
                metric_name=entries[0].metric_name,
                entries=entries,
                is_consistent=not disagreements,
                max_deviation_pct=round(max_dev, 2),
                discrepancy_detail="; ".join(disagreements),
                fiscal_year=period,
            ))
        return results

    def find_inconsistencies(self) -> list[dict]:
        """
        Cross-document disagreements, in the shape the synthesizer consumes.

        Severity is by relative size of the gap: >5% high, >1% medium, else low.
        Every item carries `method: "deterministic"` — these are measured facts
        about the documents, not a model's opinion.

        Returns:
            List of inconsistency dicts.
        """
        out = []
        for comp in self.check_consistency():
            if comp.is_consistent:
                continue
            dev = comp.max_deviation_pct
            severity = "high" if dev > 5 else "medium" if dev > 1 else "low"
            values = [
                {
                    "source": e.source_file,
                    "value": e.normalized_value,
                    "as_printed": e.raw_text.split("  ")[-1],
                    "page": e.page_number or None,
                }
                for e in comp.entries
            ]
            out.append({
                "metric": comp.metric_name,
                "fiscal_year": comp.fiscal_year,
                "values_found": values,
                "discrepancy_type": "value_mismatch",
                "severity": severity,
                "max_deviation_pct": dev,
                "explanation": (
                    f"{comp.metric_name} for {comp.fiscal_year} differs across sources: "
                    f"{comp.discrepancy_detail} ({dev:.1f}% apart)"
                ),
                "method": "deterministic",
            })
        return out

    def tie_out(self) -> TieOutResult:
        """
        Every labelled figure two or more documents state, agreeing or not.

        Stricter than check_consistency, because a deal-level report sets one
        document against another for every figure and a reviewer must be able
        to trust each conflict it lists:

        - percentages, multiples and amounts under one label are different
          figures ("Gross margin 60.0%" and a "gross margin" of $271.7M);
        - a label that names nothing on its own ("Total", "Other, net") is
          left out and counted;
        - a document that states two different values for one label and
          period (a segment table and the consolidated statement both say
          "Revenue") has no single figure to compare: left out and counted.

        Returns:
            TieOutResult with conflicts first (largest gap first), then
            agreements, and the counts of what was not compared.
        """
        result = TieOutResult()
        for (metric, period), entries in self._entries.items():
            if not period:
                result.without_period += len(entries)
                continue
            groups: dict[str, list[NumericalEntry]] = {}
            for entry in entries:
                groups.setdefault(_kind_class(entry), []).append(entry)
            for kind, group in groups.items():
                if is_generic_label(metric):
                    result.generic += 1
                    continue
                parts, left_out = _basis_parts(group)
                result.ambiguous += left_out
                for part in parts:
                    by_source: dict[str, list[NumericalEntry]] = {}
                    for entry in part:
                        by_source.setdefault(entry.source_file, []).append(entry)
                    if len(by_source) < 2:
                        result.single_source += 1
                        continue
                    stated = [_distinct_values(es) for es in by_source.values()]
                    if any(len(values) > 1 for values in stated):
                        result.ambiguous += 1
                        continue
                    figures = [values[0] for values in stated]
                    conflict = any(not _values_agree(a, b) for a, b in combinations(figures, 2))
                    result.figures.append(FigureTieOut(
                        metric_name=figures[0].metric_name,
                        fiscal_year=period,
                        kind=kind,
                        status="conflict" if conflict else "agree",
                        entries=figures,
                        max_deviation_pct=_spread_pct(figures) if conflict else 0.0,
                    ))
        result.figures.sort(key=lambda f: (
            f.status != "conflict",
            -f.max_deviation_pct,
            normalize_metric_label(f.metric_name),
            f.fiscal_year,
        ))
        return result

    def cross_checked_count(self) -> int:
        """Number of metric/period pairs reported by more than one source."""
        return len(self.check_consistency())

    def to_dict(self) -> dict:
        """
        Serialises the registry for AgentState.

        Returns:
            {"<metric> | <period>": {metric, fiscal_year, values, is_consistent}}.
        """
        consistency = {
            (normalize_metric_label(c.metric_name), c.fiscal_year): c.is_consistent
            for c in self.check_consistency()
        }
        out = {}
        for (metric, period), entries in self._entries.items():
            out[f"{metric} | {period or 'unknown period'}"] = {
                "metric": entries[0].metric_name,
                "fiscal_year": period,
                "values": [
                    {
                        "source": e.source_file,
                        "raw_value": e.raw_value,
                        "normalized_value": e.normalized_value,
                        "currency": e.currency,
                        "fiscal_year": e.fiscal_year,
                        "page": e.page_number or None,
                    }
                    for e in entries
                ],
                "is_consistent": consistency.get((metric, period), True),
            }
        return out


def _is_prose(line: str, numbers: list | None = None) -> bool:
    """A sentence rather than a row or a header: it ends the header above it."""
    if numbers and not line[numbers[-1].end:].strip(" )|%"):
        return False  # ends in a value: a row
    return len(_WORD.findall(line)) >= _PROSE_WORDS or bool(_SENTENCE_END.search(line))


def _rebased_index(values: list) -> bool:
    """True for a row of three or more values that starts at exactly 100 (see _INDEX_BASE)."""
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return len(numbers) >= 3 and abs(abs(float(numbers[0])) - _INDEX_BASE) < 1e-9


def _is_context_line(line: str) -> bool:
    """A short line with no figure, such as "Three Months Ended" above a row of dates."""
    return (
        bool(line.strip(" |"))
        and len(_WORD.findall(line)) <= _MAX_CONTEXT_WORDS
        and not _SENTENCE_END.search(line)
        and not extract_numbers(line, claims_only=False)
    )


def _keyed_period(token: str, interim: bool) -> str | None:
    """The period of a "period=value" pair in a converted table's text."""
    compact = re.sub(r"[\s']", "", token.upper())
    if compact in ("LTM", "TTM"):
        return None
    if interim and re.fullmatch(r"(?:19|20)\d{2}A?", compact):
        return None  # a 10-Q's "2023" column is a quarter or year-to-date
    return normalize_period(token)


def _document_scheme(chunks: list[dict]) -> list[str | None] | None:
    """
    The period header a document prints, when it prints only that one, and
    more than once.

    A schedule or a set of statements ("FY2023  FY2022" over the income
    statement and both halves of the balance sheet) has one column scheme
    throughout. A public filing does not: it prints dozens of tables, a 10-K's
    columns change from three years to two to five, and a table whose header
    was lost there is not under any other table's — so a filing lends none.

    Args:
        chunks: The document's chunks in document order.

    Returns:
        Period keys of its only header, or None.
    """
    if any(chunk.get("filing_form") for chunk in chunks):
        return None
    schemes: set[tuple[str | None, ...]] = set()
    printed = 0
    for chunk in chunks:
        context: list[str] = []
        for line in (chunk.get("text") or "").splitlines():
            if not line.strip() or _RULE_LINE.match(line):
                continue
            if not _PERIOD.search(line):
                context = [*context, line][-2:] if _is_context_line(line) else []
                continue
            if not extract_numbers(_PERIOD.sub(" ", line), claims_only=False) and _reads_like_header(line):
                keys = tuple(_header_keys(line, context, False))
                if any(keys):
                    schemes.add(keys)
                    printed += 1
                if len(schemes) > 1:
                    return None
            context = []
    return list(schemes.pop()) if schemes and printed > 1 else None


def _distinct_values(entries: list[NumericalEntry]) -> list[NumericalEntry]:
    """One entry per different value a document states, the scale-known copy first."""
    distinct: list[NumericalEntry] = []
    for entry in sorted(entries, key=lambda e: not e.scale_known):
        if not any(_values_agree(entry, d) for d in distinct):
            distinct.append(entry)
    return distinct


class _Header(list):
    """
    The period keys in force at the end of a chunk's text, with what each
    column's figures are (column_basis). The next chunk of the same table
    starts under both: with the keys alone, a balance sheet continued in
    the next chunk met a cash-flow line of the same name again.
    """

    def __init__(self, keys=(), bases=()):
        super().__init__(keys)
        bases = list(bases)
        self.bases = bases if len(bases) == len(self) else [""] * len(self)


def _header_bases(line: str, context: list[str]) -> list[str]:
    """
    What each period column of a text header holds (see column_basis): flows
    under a span on the line or the lines above it, balances under a date, ""
    otherwise. A date printed before the first year only ("December 31,
    2023  2022") heads every column, as _header_keys reads it.
    """
    tokens = list(_PERIOD.finditer(line))
    if _SPAN.search("\n".join([*context, line])):
        return ["flow"] * len(tokens)
    if any(_MONTH_DAY.search(c) for c in context):
        return ["balance"] * len(tokens)
    dated = [_date_before(line, t.start()) for t in tokens]
    if dated and dated[0] and not any(dated[1:]):
        return ["balance"] * len(tokens)
    return ["balance" if d else "" for d in dated]


def _bases_clash(a: NumericalEntry, b: NumericalEntry) -> bool:
    """True for a balance and a flow: never one figure, whatever their labels and periods."""
    return {a.basis, b.basis} == {"flow", "balance"}


def _basis_parts(entries: list[NumericalEntry]) -> tuple[list[list[NumericalEntry]], bool]:
    """
    A figure's entries split into what can be compared: all of them, or — when
    both balances and flows are among them — the flows apart from the balances.
    Entries whose basis is unknown then fit neither, and are left out.

    Returns:
        (parts, left_out): left_out is True when entries were left out.
    """
    bases = {e.basis for e in entries}
    if not {"flow", "balance"} <= bases:
        return [entries], False
    parts = [[e for e in entries if e.basis == basis] for basis in ("flow", "balance")]
    return parts, "" in bases


# What a statement prints in a value column for nil or "not meaningful": the
# converter keeps such a cell as printed, with no value.
_NIL_MARKS = {"-", "—", "–", "−", "n/a", "na", "n.a.", "nm", "n.m.", "n/m"}


def _has_values(row: dict) -> bool:
    """
    Whether a structured row states a figure, or a nil in a value column: a
    subheading ("Swaptions:") states neither. A row of dashes
    ("Amortization of intangibles  —  —") is a line item with nothing to
    compare, not a subheading over the rows below it.
    """
    for cell in (row.get("values") or {}).values():
        if not isinstance(cell, dict):
            continue
        if "normalized_value" in cell:
            return True
        if str(cell.get("raw") or "").replace("$", " ").strip().lower() in _NIL_MARKS:
            return True
    return False


def _repeated_labels(rows: list[dict]) -> set[str]:
    """The line items, other than generic ones, that one table prints on more than one row."""
    counts: dict[str, int] = {}
    for row in rows:
        if _has_values(row):
            label = normalize_metric_label(str(row.get("line_item", "")))
            counts[label] = counts.get(label, 0) + 1
    return {label for label, n in counts.items() if n > 1 and not is_generic_label(label)}


def _row_range(chunk: dict) -> tuple[int, int] | None:
    """The (first, last) table rows a chunk holds, when it says."""
    span = chunk.get("table_row_range")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        try:
            return int(span[0]), int(span[1])
        except (TypeError, ValueError):
            return None
    return None


# The text forms of a converted table's rows, besides the row-by-row chunks that
# carry them as structured_rows.
_TEXT_FORMS = {"table_markdown", "table_narrative"}


def _table_key(chunk: dict) -> tuple[str, str]:
    """A table's identity: its document and its table_id."""
    return str(chunk.get("source_file") or ""), str(chunk.get("table_id") or "")


def _structured_ranges(chunks: list[dict]) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """
    Per table_id, the row ranges read from structured_rows whose columns name a
    period. Rows under "Column 2", "Column 3" are still read from their text
    too: a ruled grid's years printed above its rules reach only the text, as
    its caption, and American Water's rate-case tables were compared on them.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    for chunk in chunks:
        span = _row_range(chunk)
        if not (chunk.get("table_id") and chunk.get("structured_rows") and span):
            continue
        interim = _is_interim(chunk)
        headers = {str(col) for row in chunk["structured_rows"] for col in (row.get("values") or {})}
        if any(column_period(header, interim) for header in headers):
            ranges.setdefault(_table_key(chunk), []).append(span)
    return ranges


def _read_elsewhere(chunk: dict, ranges: dict[tuple[str, str], list[tuple[int, int]]]) -> bool:
    """
    True for a markdown or narrative form of table rows whose structured rows
    are read too: one table is read once. Read twice, the two forms could key
    one figure two ways — Best Buy's cash-flow rows as of the balance date
    "2017-01-28" from the rows and as "FY ending 2017-01" from the grid, whose
    label column printed "Fiscal Years Ended" — and the tie-out counted four
    conflicts twice (Decision 93).
    """
    if chunk.get("structured_rows") or chunk.get("content_type") not in _TEXT_FORMS:
        return False
    span = _row_range(chunk)
    covered = ranges.get(_table_key(chunk)) if chunk.get("table_id") else None
    if not span or not covered:
        return False
    return all(any(a <= n <= b for a, b in covered) for n in range(span[0], span[1] + 1))
