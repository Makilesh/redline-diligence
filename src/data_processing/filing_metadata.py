"""
Filing metadata — which company and which period a public filing covers,
read from the filing itself.

A data room for a public target holds several years of the same company's
10-Ks, 10-Qs, 8-Ks and earnings releases. They share boilerplate, segment
names and most of their figures (each 10-K repeats the prior years as
comparatives), so a question about FY2018 retrieves the FY2019 filing about as
readily as the FY2018 one. Measured on FinanceBench (Decision 65): the top
reranked chunk came from the wrong filing on 43 of 150 questions, 35 of them
another filing of the same company.

The file name usually says which filing it is, but a data room cannot be
relied on to name files ("10-K.pdf", "Annual report final v3.pdf"), so nothing
here reads it. Every SEC form states its period and its registrant on the
cover page, in fixed language:

    FORM 10-K ... For the fiscal year ended December 31, 2018
    Commission file number 1-3285   3M COMPANY (Exact name of registrant ...)

and an earnings release names its company and ticker in the opening line
("PepsiCo, Inc. (NASDAQ: PEP) today reported results for the first quarter
2023"). Those are deterministic patterns, so the extraction is regexes over the
first pages — no model, no network. A document that matches none of them (a
CIM, a contract, management accounts) gets no filing metadata at all, and
everything that uses it then does nothing.

Output, written onto every child payload of the document when present:

    filing_form           "10-K" | "10-Q" | "8-K" | "earnings_release"
    filing_period_end     ISO date the period ended (10-K / 10-Q)
    filing_date           ISO date of the report (8-K / earnings release)
    filing_fiscal_years   fiscal years a question may call this filing by
    filing_entity_keys    identities of the registrant ("sec:1-3285",
                          "name:3m", "ticker:pep"); two filings are the same
                          company when they share any key
    filing_company        the registrant's name as printed (display only)
"""

from __future__ import annotations

import re
from datetime import date

# Only the opening pages are read: the cover of an SEC form is page 1, and an
# earnings release states its subject in its first lines. Reading further would
# pick up the dates of comparative periods and exhibits.
COVER_PAGES = 3
# For documents without page numbers (.txt, .docx), the same budget in characters.
COVER_CHARS = 12_000
# Where the cover's fixed language must appear: the Commission's name near the
# top, the form right after it, and a release's company and ticker in its lead.
_SEC_HEADER_WITHIN = 800
_SEC_FORM_WITHIN = 400
_RELEASE_HEAD_CHARS = 1_500

_MONTHS = {
    m: i for i, m in enumerate(
        ("january", "february", "march", "april", "may", "june", "july", "august",
         "september", "october", "november", "december"), start=1)
}
_DATE = r"(January|February|March|April|May|June|July|August|September|October|November|December)" \
        r"\s+(\d{1,2})\s*,?\s*((?:19|20)\d\d)"
_DATE_RE = re.compile(_DATE, re.IGNORECASE)

# Every cover opens "UNITED STATES SECURITIES AND EXCHANGE COMMISSION
# Washington, D.C. 20549"; a contract or a release that merely mentions the
# Commission does not follow it with its address.
_SEC_HEADER = re.compile(r"SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION\W{0,10}Washington", re.IGNORECASE)
_SEC_FORM = re.compile(r"\bFORM\s+(10-K|10-Q|8-K)\b", re.IGNORECASE)
# The cover's layout is sometimes interleaved by the PDF text layer
# ("For the fiscal year ended Commission file December 31, 2022 number 1-5805"),
# so a few words may sit between the phrase and its date.
_PERIOD_PATTERNS = {
    "10-K": re.compile(r"fiscal\s+year\s+ended\b.{0,60}?" + _DATE, re.IGNORECASE | re.DOTALL),
    "10-Q": re.compile(r"quarterly\s+period\s+ended\b.{0,60}?" + _DATE, re.IGNORECASE | re.DOTALL),
    "8-K": re.compile(r"Date\s+of\s+report\b.{0,80}?" + _DATE, re.IGNORECASE | re.DOTALL),
}

# SEC file numbers look like "1-3285", "001-07882", "0-15175". An IRS employer
# number is 2-7 digits ("41-0417775") and a phone number is preceded by its area
# code in parentheses, so neither matches.
_FILE_NUMBER = re.compile(r"(?<![\d\-)])(?<!\)\s)(\d{1,3})-(\d{4,6})(?![\d\-])")
# "Commission File Number", also when a two-column cover interleaves the label
# with its neighbour: "(Commission (IRS Employer of incorporation) File Number)".
_COMMISSION_FILE = re.compile(r"Commission\b.{0,60}?\bFile\b", re.IGNORECASE | re.DOTALL)
_REGISTRANT_MARKER = re.compile(r"\(\s*Exact\s+name\s+of\s+(?:the\s+)?registrant", re.IGNORECASE)

# Earnings release: "Ulta Beauty, Inc. (NASDAQ: ULTA) today announced ..."
_EXCHANGE_TICKER = re.compile(r"\(\s*(?:NYSE|NASDAQ|Nasdaq|NYSE American)\s*:\s*([A-Z][A-Z.]{0,5})\b")
_RESULTS = re.compile(r"\bresults\b", re.IGNORECASE)
_RELEASE_YEARS = (
    re.compile(r"\b(?:fiscal(?:\s+year)?|full[\s-]year|(?:first|second|third|fourth)[\s-]quarter"
               r"(?:\s+(?:of\s+)?(?:fiscal\s+)?)?|Q[1-4])\s*((?:19|20)\d\d)\b", re.IGNORECASE),
    re.compile(r"\b((?:19|20)\d\d)\s+(?:first|second|third|fourth|full)[\s-](?:quarter|year)\b",
               re.IGNORECASE),
)
# A standalone number (a date, a file number) ends the field before a name;
# a digit inside a name ("3M") does not.
_NUMBER_TOKEN = r"(?<![A-Za-z])\d[\d,\-]*(?![A-Za-z])"
_NAME_SPLIT = re.compile(_NUMBER_TOKEN + r"|\)|--|\s[-–—]\s|:")

# Words that do not distinguish one registrant from another.
_NAME_STOPWORDS = frozenset({
    "the", "inc", "incorporated", "corporation", "corp", "co", "company", "companies",
    "plc", "ltd", "limited", "llc", "lp", "sa", "nv", "ag", "holdings", "holding", "group",
    "and",
})
_WHITESPACE = re.compile(r"\s+")


def _iso(match: re.Match, offset: int = 0) -> date | None:
    """The date captured by _DATE at group offset+1..offset+3."""
    month = _MONTHS[match.group(offset + 1).lower()]
    try:
        return date(int(match.group(offset + 3)), month, int(match.group(offset + 2)))
    except ValueError:
        return None


def normalize_company_name(name: str) -> str:
    """
    "The Coca-Cola Company" -> "coca cola"; "PepsiCo, Inc." -> "pepsico".

    Lower-cased words with punctuation and legal-form words removed, so the
    same registrant printed two ways ("COCA COLA CO", "The Coca-Cola Company")
    gets one key. Two different names still get two keys ("Square, Inc." and
    "Block, Inc."); the SEC file number is what links a renamed registrant.

    Args:
        name: Company name as printed.

    Returns:
        Normalised name ("" when nothing distinctive is left).
    """
    words = re.sub(r"[^0-9a-z]+", " ", name.lower()).split()
    return " ".join(w for w in words if w not in _NAME_STOPWORDS)


def _letters(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def _cover_text(sections: list[dict]) -> str:
    """Whitespace-collapsed text of the opening pages."""
    parts, used, shown = [], 0, set()
    for section in sections:
        page = section.get("page_number")
        if page is not None and page > COVER_PAGES:
            break
        # Tables are skipped: their representations restate the cover's grid
        # several times over and would push a release's opening sentence out of
        # the window it is looked for in.
        if section.get("is_table"):
            continue
        text = section.get("text") or ""
        # A processor may lift a line it reads as a heading ("FORM 10-K") out of
        # the text; the cover's fixed language must still be found, once.
        heading = section.get("section_heading") or ""
        if heading and _letters(heading) not in shown and _letters(heading) not in _letters(text):
            text = f"{heading} {text}"
        shown.add(_letters(heading))
        shown.add(_letters(text))
        parts.append(text)
        used += len(text)
        if page is None and used >= COVER_CHARS:
            break
    return _WHITESPACE.sub(" ", " ".join(parts)).strip()


def _sec_file_number(cover: str) -> str | None:
    """The Commission file number closest to its label, normalised ("001-07882" -> "1-7882")."""
    best = None
    for label in _COMMISSION_FILE.finditer(cover):
        start = max(0, label.start() - 120)
        window = cover[start: label.end() + 120]
        for m in _FILE_NUMBER.finditer(window):
            distance = abs(start + m.start() - label.start())
            if best is None or distance < best[0]:
                best = (distance, f"{int(m.group(1))}-{int(m.group(2))}")
    return best[1] if best else None


def _registrant_name(cover: str) -> str:
    """The name printed just before "(Exact name of registrant as specified in its charter)"."""
    marker = _REGISTRANT_MARKER.search(cover)
    if not marker:
        return ""
    before = cover[max(0, marker.start() - 120): marker.start()]
    # The name follows the file number or the report date on every layout seen;
    # whatever precedes the last of those is the cover's previous field.
    tail = re.split(_NUMBER_TOKEN + r"|Number\s*:?|No\.", before)[-1]
    name = tail.strip(" _.,:;()|-–—")
    return name if re.search(r"[A-Za-z]{2}", name) else ""


def _sec_filing(cover: str, form: str) -> dict:
    out: dict = {"filing_form": form}
    match = _PERIOD_PATTERNS[form].search(cover)
    when = _iso(match) if match else None
    if when is not None:
        if form == "8-K":
            out["filing_date"] = when.isoformat()
            out["filing_fiscal_years"] = [when.year]
        else:
            out["filing_period_end"] = when.isoformat()
            if form == "10-K":
                # A fiscal year ending in January-March is named after either
                # calendar year it touches: Walmart and Best Buy call the year
                # ended 2023-01-28 "fiscal 2023", Ulta Beauty calls it "fiscal
                # 2022". Both are kept rather than guessed.
                years = {when.year} | ({when.year - 1} if when.month <= 3 else set())
            else:
                # A quarter belongs to the fiscal year that contains it, which
                # ends later in the same calendar year or in the next one
                # (Amcor's Q2 FY2023 ended 2022-12-31; Best Buy's Q2 FY2024
                # ended 2023-07-29).
                years = {when.year, when.year + 1}
            out["filing_fiscal_years"] = sorted(years)
    keys = []
    file_number = _sec_file_number(cover)
    if file_number:
        keys.append(f"sec:{file_number}")
    name = _registrant_name(cover)
    if name:
        out["filing_company"] = name
        if normalize_company_name(name):
            keys.append(f"name:{normalize_company_name(name)}")
    if keys:
        out["filing_entity_keys"] = keys
    return out


def _earnings_release(head: str) -> dict:
    ticker = _EXCHANGE_TICKER.search(head)
    if not ticker or not _RESULTS.search(head):
        return {}
    out: dict = {"filing_form": "earnings_release"}
    keys = [f"ticker:{ticker.group(1).lower()}"]
    name = _NAME_SPLIT.split(head[max(0, ticker.start() - 80): ticker.start()])[-1]
    name = name.strip(" _.,;|–—-")
    if re.search(r"[A-Za-z]{2}", name):
        out["filing_company"] = name
        if normalize_company_name(name):
            keys.append(f"name:{normalize_company_name(name)}")
    out["filing_entity_keys"] = keys
    years = set()
    first_date = _DATE_RE.search(head)
    if first_date and _iso(first_date):
        out["filing_date"] = _iso(first_date).isoformat()
        years.add(_iso(first_date).year)
    for pattern in _RELEASE_YEARS:
        years |= {int(m.group(1)) for m in pattern.finditer(head)}
    if years:
        out["filing_fiscal_years"] = sorted(years)
    return out


def extract_filing_metadata(sections: list[dict]) -> dict:
    """
    Company and period of a public filing, from its opening pages.

    Args:
        sections: The document's sections in order, as extract_sections
            produces them (reads `text` and `page_number`).

    Returns:
        The filing_* payload fields that could be read, or {} when the
        document is not recognisably an SEC form or an earnings release.
        Only filing_fiscal_years and filing_entity_keys are used for
        retrieval; a filing missing either is simply never preferred or
        set aside.
    """
    cover = _cover_text(sections)
    if not cover:
        return {}
    # A cover page opens with the Commission's name and the form a few lines
    # later. A press release mentions both too ("... in our Form 10-Q filed
    # with the Securities and Exchange Commission"), but pages in.
    header = _SEC_HEADER.search(cover[:_SEC_HEADER_WITHIN])
    form = _SEC_FORM.search(cover, header.end(), header.end() + _SEC_FORM_WITHIN) if header else None
    if form:
        return _sec_filing(cover, form.group(1).upper())
    return _earnings_release(cover[:_RELEASE_HEAD_CHARS])
