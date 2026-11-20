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

and an earnings release names its company and period in its opening line
("PepsiCo, Inc. (NASDAQ: PEP) today reported results for the first quarter
2023"; "Amcor reports fiscal 2023 results"). Those are deterministic patterns,
so the extraction is regexes over the first pages — no model, no network.

It fails safe: a field is written only when its fixed language is found, and
a document that is recognisably an SEC form of a kind not modelled here (S-1,
6-K, a proxy statement, an OCR-garbled form name) gets no metadata at all
rather than a guess. Everything that uses the fields then does nothing.

Output, written onto every child payload of the document when present:

    filing_form           "10-K" | "10-Q" | "8-K" | "20-F" | "40-F" (each
                          optionally "/A") | "earnings_release"
    filing_period_end     ISO date the period ended (annual forms, 10-Q, and a
                          release that states it)
    filing_date           ISO date of the report (8-K / earnings release)
    filing_fiscal_years   fiscal years a question may call this filing by
    filing_fiscal_quarter 4 for an annual report or a fourth-quarter/full-year
                          release, 1-3 for a 10-Q or release whose quarter is
                          stated; absent when not stated
    filing_entity_keys    identities of the registrant ("sec:1-3285",
                          "name:3m", "ticker:pep"); two filings are the same
                          company when they share any key
    filing_company        the registrant's name as printed: the cover's or,
                          when the cover prints none, the signature page's
                          (display, and the verifier's company check)

Naming a fiscal year (Decision 70). A registrant names its fiscal year itself
(dei:DocumentFiscalYearFocus is that label), and for most year ends every
convention agrees: the year ended June 30, 2023 is fiscal 2023. Two cases need
a rule, and both are decided by the year's *nominal* month, its end date less
a week (a 52/53-week year ends within a few days of the month it is anchored
to, so January 1, 2023 is a December year):
  - a nominal December year is named after that calendar year only. Johnson &
    Johnson's 52-week year ended January 1, 2023 is "full-year 2022" in its
    own release and its 10-K; it spends 358 of its 364 days in 2022, and the
    SEC's calendar frames (CY#### = 365 days +/- 30) put it in CY2022;
  - a nominal January-March year is named both ways, because registrants
    disagree: Walmart and Best Buy call the year ended in January 2023 "fiscal
    2023", Target, Home Depot and Ulta Beauty call it "fiscal 2022", Japanese
    March year ends are usually named after the start year, and question
    writers follow neither (FinanceBench calls Ulta's fiscal 2022 "FY2023").
A quarter belongs to the fiscal year that ends one to three quarters after it
(exactly 4 - q quarters after when its quarter q is stated), named by the
same rule.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from src.utils.casefold import ascii_lower
from src.utils.fail_open import fail_open

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
# A release recognised by its headline alone ("Amcor reports fiscal 2023
# results") must say so at the very top of the document.
_HEADLINE_CHARS = 250

_MONTH_NAMES = ("january", "february", "march", "april", "may", "june", "july", "august",
                "september", "october", "november", "december")
_MONTHS = {m: i for i, m in enumerate(_MONTH_NAMES, start=1)}
_MONTH = "(" + "|".join(m.capitalize() for m in _MONTH_NAMES) + ")"
_YEAR = r"((?:19|20)\d\d)"
_DATE = _MONTH + r"\s+(\d{1,2})\s*,?\s*" + _YEAR                  # December 31, 2022
_DATE_DMY = r"(\d{1,2})\s+" + _MONTH + r"\s*,?\s*" + _YEAR + r"(?!\d)"   # 31 December 2022 (non-US covers)
_DATE_RE = re.compile(_DATE, re.IGNORECASE)
_ANY_DATE = re.compile(f"(?:{_DATE})|(?:{_DATE_DMY})", re.IGNORECASE)
# A month name, full or abbreviated, between a cover's label and the date it
# reads means that date is not the label's: the label's own date was garbled
# ("December 3l, 2O22" on a scanned cover) or abbreviated.
_MONTH_WORD = re.compile(
    r"\b(?:" + "|".join(_MONTH_NAMES) + r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\b", re.IGNORECASE)

# Every cover opens "UNITED STATES SECURITIES AND EXCHANGE COMMISSION
# Washington, D.C. 20549"; a contract or a release that merely mentions the
# Commission does not follow it with its address.
_SEC_HEADER = re.compile(r"SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION\W{0,10}Washington", re.IGNORECASE)
# The first form name after the header is the form ("FORM 10-K", "FORM 10-K/A",
# "FORM 20-F", "FORM S-1", "FORM 6-K").
_SEC_FORM = re.compile(r"\bFORM\s+([0-9A-Z]{1,3}-[0-9A-Z]{1,4})\b(\s*/\s*A\b)?", re.IGNORECASE)
# Forms whose period is a fiscal year, stated as "For the fiscal year ended".
# A 20-F or 40-F is a foreign private issuer's annual report.
_ANNUAL_FORMS = {"10-K", "20-F", "40-F"}
_SUPPORTED_FORMS = _ANNUAL_FORMS | {"10-Q", "8-K"}
# The cover's layout is sometimes interleaved by the PDF text layer
# ("For the fiscal year ended Commission file December 31, 2022 number 1-5805"),
# so a few words may sit between the label and its date.
_PERIOD_LABELS = {
    "annual": (re.compile(r"fiscal\s+year\s+ended\b", re.IGNORECASE), 60),
    "10-Q": (re.compile(r"quarterly\s+period\s+ended\b", re.IGNORECASE), 60),
    "8-K": (re.compile(r"Date\s+of\s+report\b", re.IGNORECASE), 80),
}
# A 10-Q's statements cover the quarter and the year to date: "three and six
# months ended June 30" is the second quarter. Read where it names the cover's
# own quarter end (month and day), so another period's statement cannot count.
_YTD = re.compile(
    r"\b(three|six|nine|twelve|thirteen|sixteen|twenty[\s-]?(?:four|six|eight)|thirty[\s-]?(?:six|nine)|forty"
    r"|3|6|9|12|13|16|24|26|28|36|39|40)\s+(months|weeks)\s+ended\s+" + _MONTH + r"\s+(\d{1,2})\b",
    re.IGNORECASE)
_YTD_QUARTER = {
    "months": {"three": 1, "3": 1, "six": 2, "6": 2, "nine": 3, "9": 3},
    "weeks": {"twelve": 1, "12": 1, "thirteen": 1, "13": 1, "sixteen": 1, "16": 1,
              "twentyfour": 2, "24": 2, "twentysix": 2, "26": 2, "twentyeight": 2, "28": 2,
              "thirtysix": 3, "36": 3, "thirtynine": 3, "39": 3, "forty": 3, "40": 3},
}

# SEC file numbers look like "1-3285", "001-07882", "0-15175". An IRS employer
# number is 2-7 digits ("41-0417775") and a phone number is preceded by its area
# code in parentheses, so neither matches.
_FILE_NUMBER = re.compile(r"(?<![\d\-)])(?<!\)\s)(\d{1,3})-(\d{4,6})(?![\d\-])")
# "Commission File Number", also when a two-column cover interleaves the label
# with its neighbour: "(Commission (IRS Employer of incorporation) File Number)".
_COMMISSION_FILE = re.compile(r"Commission\b.{0,60}?\bFile\b", re.IGNORECASE | re.DOTALL)
_REGISTRANT_MARKER = re.compile(r"\(\s*Exact\s+name\s+of\s+(?:the\s+)?registrant", re.IGNORECASE)
# A foreign private issuer prints its name in its own language and then in
# English: "SONY GROUP CORPORATION (Translation of Registrant's name into English)".
_TRANSLATION_MARKER = re.compile(r"\(\s*Translation\s+of\s+(?:the\s+)?registrant", re.IGNORECASE)
# The signature page names the registrant too, in the fixed language of every
# 10-K, 10-Q and 8-K, with a label under the name:
#
#     Pursuant to the requirements of Section 13 or 15(d) of the Securities
#     Exchange Act of 1934, the Registrant has duly caused this report to be
#     signed on its behalf by the undersigned, thereunto duly authorized.
#     THE COCA-COLA COMPANY
#     (Registrant)
#
# It is read only when the cover yields no name: Coca-Cola's 2017 10-K prints
# its name on the cover as an image. The sentence's subject varies ("the
# registrant", "PepsiCo"), as do "this report" / "this Annual Report on Form
# 10-K", "by" / "of the undersigned", "thereunto" / "hereunto" and a date or a
# place before its full stop. Most signature pages print no "(Registrant)"
# label (31 of the 42 SEC forms among FinanceBench's dev filings: "NIKE, INC.
# By: /s/ ...", "Date: ... SQUARE, INC."); without it nothing marks where the
# name ends, so they give none.
_SIGNED = re.compile(
    r"\bhas\s+duly\s+caused\s+this\s+(?:\S+\s+){0,6}?to\s+be\s+signed\s+on\s+(?:its\s+)?behalf\s+(?:by|of)\s+"
    r"the\s+undersigned,?\s+(?:there|here)unto\s+duly\s+authori[sz]ed\b[^.]{0,120}\.",
    re.IGNORECASE)
_SIGNED_LABEL = re.compile(r"\(\s*Registrant\s*\)", re.IGNORECASE)
# The label follows the name; it must start this close to the sentence's end.
_SIGNED_NAME_CHARS = 100

# Earnings release: "Ulta Beauty, Inc. (NASDAQ: ULTA) today announced ..."
_EXCHANGE_TICKER = re.compile(r"\(\s*(?:NYSE|NASDAQ|Nasdaq|NYSE American)\s*:\s*([A-Z][A-Z.]{0,5})\b")
_RESULTS = re.compile(r"\bresults\b", re.IGNORECASE)
# "Amcor reports fiscal 2023 results": the company is whatever opens the headline.
_HEADLINE = re.compile(r"\s*(.{2,80}?)\s+(?:reports|reported|announces|announced|posts|delivers)\b",
                       re.IGNORECASE)
# A line break inside a hyphenated word survives as "second- quarter".
_RELEASE_YEARS = (
    re.compile(r"\b(?:fiscal(?:\s+year)?|full[\s-]+year|(?:first|second|third|fourth)[\s-]+quarter"
               r"(?:\s+(?:of\s+)?(?:fiscal\s+)?)?|Q[1-4])\s*((?:19|20)\d\d)\b", re.IGNORECASE),
    re.compile(r"\b((?:19|20)\d\d)\s+(?:first|second|third|fourth|full)[\s-]+(?:quarter|year)\b",
               re.IGNORECASE),
)
_RELEASE_QUARTERS = (
    (re.compile(r"\b(?:first[\s-]+quarter|Q1|1Q)\b", re.IGNORECASE), 1),
    (re.compile(r"\b(?:second[\s-]+quarter|Q2|2Q)\b", re.IGNORECASE), 2),
    (re.compile(r"\b(?:third[\s-]+quarter|Q3|3Q)\b", re.IGNORECASE), 3),
    (re.compile(r"\b(?:fourth[\s-]+quarter|Q4|4Q|full[\s-]+year|fiscal\s+year|fifty[\s-]+two[\s-]+week"
                r"|fifty[\s-]+three[\s-]+week|annual)\b", re.IGNORECASE), 4),
    # "fiscal 2023 results": a fiscal year's results with no quarter named.
    (re.compile(r"\bfiscal\s+(?:19|20)\d\d\s+results\b", re.IGNORECASE), 4),
)
# What a release says it reports is the clause around "results"; guidance in
# the same lead ("provides outlook for fiscal 2024", "full-year 2023 guidance")
# names a period it does not cover.
_RESULTS_BEFORE = 90
_RESULTS_AFTER = 160
_GUIDANCE = re.compile(r"\b(?:guidance|outlook|expects?|forecasts?|targets?)\b", re.IGNORECASE)
_CLAUSE_END = re.compile(r"[.;!?]\s|\s[-–—]\s")
_ENDED = re.compile(r"\bended\s+", re.IGNORECASE)
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

# A 52/53-week year ends within a week of the month end it is anchored to.
_NOMINAL_SHIFT = timedelta(days=7)
_QUARTER = timedelta(weeks=13)


def _date_from(match: re.Match) -> date | None:
    """The date matched by _ANY_DATE (month-day-year or day-month-year)."""
    g = match.groups()
    month, day, year = (g[0], g[1], g[2]) if g[0] else (g[4], g[3], g[5])
    try:
        # ascii_lower: "Auguſt" (long s) matches the case-insensitive month
        # names, and .lower() alone is no key of _MONTHS (Decision 88).
        return date(int(year), _MONTHS[ascii_lower(month)], int(day))
    except ValueError:
        return None


def _date_after(text: str, start: int, max_gap: int) -> date | None:
    """
    The date that follows position `start` within `max_gap` characters, or None.

    The gap may hold words of an interleaved column ("Commission file") but not
    a month name or a digit: those are a garbled or abbreviated date of the
    label's own ("Decem ber 3l, 2O22", "Dec. 31, 2022"), and the next
    well-formed date is then some other field's.
    """
    match = _ANY_DATE.search(text, start, start + max_gap + 30)
    if not match or match.start() - start > max_gap:
        return None
    gap = text[start: match.start()]
    if _MONTH_WORD.search(gap) or re.search(r"\d", gap):
        return None
    return _date_from(match)


def annual_fiscal_years(end: date) -> list[int]:
    """
    The fiscal years a question may call a year ending on `end` (see module docstring).

    Args:
        end: Last day of the fiscal year.

    Returns:
        [nominal year] for a nominal April-December year end; [year - 1, year]
        for a nominal January-March year end.
    """
    nominal = end - _NOMINAL_SHIFT
    return [nominal.year - 1, nominal.year] if nominal.month <= 3 else [nominal.year]


def quarter_fiscal_years(end: date, quarter: int | None = None) -> list[int]:
    """
    The fiscal years a question may call the fiscal year a quarter ending on `end` belongs to.

    That year ends 4 - quarter quarters later. When the quarter is stated as the
    second or third, that is the estimate (within a week either way); otherwise
    it is anywhere from one to three quarters later. A first quarter is read
    from the absence of six- and nine-month statements, which a cover that
    omits its statements also shows, so it gets the wide window too.

    Args:
        end: Last day of the quarter.
        quarter: Fiscal quarter (1-3) when stated, else None.

    Returns:
        Sorted fiscal years.
    """
    if quarter in (2, 3):
        centre = end + _QUARTER * (4 - quarter)
        low, high = centre - _NOMINAL_SHIFT, centre + _NOMINAL_SHIFT
    else:
        low, high = end + _QUARTER - _NOMINAL_SHIFT, end + 3 * _QUARTER + _NOMINAL_SHIFT
    years: set[int] = set()
    day = low
    while day <= high:
        years.update(annual_fiscal_years(day))
        day += timedelta(days=1)
    return sorted(years)


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


def _cover_text(sections: list[dict], tables: bool = False) -> str:
    """Whitespace-collapsed text of the opening pages."""
    parts, used, shown = [], 0, set()
    for section in sections:
        page = section.get("page_number")
        if page is not None and page > COVER_PAGES:
            break
        # Tables are skipped for the cover's fields: their representations
        # restate the cover's grid several times over and would push a
        # release's opening sentence out of the window it is looked for in.
        # A 10-Q's table of contents may be one, so its quarter is read with them.
        if section.get("is_table") and not tables:
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


def _name_before(cover: str, marker_re: re.Pattern, after_label: bool = False) -> str:
    """
    The name printed just before a "(Exact name of registrant ...)"-style label;
    with after_label, the name also starts after the previous label's ")" (the
    English name of a 20-F follows the "(Exact name ...)" label of the original).
    """
    marker = marker_re.search(cover)
    if not marker:
        return ""
    before = cover[max(0, marker.start() - 120): marker.start()]
    # The name follows the file number or the report date on every layout seen;
    # whatever precedes the last of those is the cover's previous field.
    tail = re.split(_NUMBER_TOKEN + r"|Number\s*:?|No\." + (r"|\)" if after_label else ""), before)[-1]
    name = tail.strip(" _.,:;()|-–—")
    return name if re.search(r"[A-Za-z]{2}", name) else ""


def _signature_name(sections: list[dict]) -> str:
    """
    The registrant's name as its signature page prints it, or "".

    The name is what lies between the sentence a signature block opens with
    and the "(Registrant)" label under the name (_SIGNED), and only when that
    is a name alone: no number (a date, a page), no field ("Date:", "By:"), no
    "/s/" signature. A document whose blocks name different registrants gives
    none. Tables are skipped: the sentence is prose, and a block the
    processor read as a grid has no reading order to trust.
    """
    text = _WHITESPACE.sub(" ", " ".join(str(s.get("text") or "") for s in sections if not s.get("is_table")))
    names: dict[str, str] = {}
    for m in _SIGNED.finditer(text):
        label = _SIGNED_LABEL.search(text, m.end(), m.end() + _SIGNED_NAME_CHARS)
        if not label:
            continue
        name = text[m.end(): label.start()].strip(" _.,:;()|-–—")
        key = normalize_company_name(name)
        if key and re.search(r"[A-Za-z]{2}", name) and not _NAME_SPLIT.search(name) and "/" not in name:
            names.setdefault(key, name)
    return next(iter(names.values())) if len(names) == 1 else ""


def _quarter_of_10q(text: str, end: date) -> int | None:
    """Fiscal quarter of a 10-Q from its year-to-date statements, or None."""
    found = set()
    for m in _YTD.finditer(text):
        if (_MONTHS[ascii_lower(m.group(3))], int(m.group(4))) != (end.month, end.day):
            continue
        length = re.sub(r"[\s-]", "", ascii_lower(m.group(1)))
        quarter = _YTD_QUARTER[ascii_lower(m.group(2))].get(length)
        if quarter:
            found.add(quarter)
    year_to_date = found - {1}
    if len(year_to_date) > 1:
        return None      # six and nine months to the same date: not a 10-Q layout seen
    if year_to_date:
        return year_to_date.pop()
    return 1 if found else None


def _sec_filing(cover: str, with_tables: str, form: str, sections: list[dict]) -> dict:
    out: dict = {"filing_form": form}
    base = form.removesuffix("/A")
    label, gap = _PERIOD_LABELS["annual" if base in _ANNUAL_FORMS else base]
    when = None
    for m in label.finditer(cover):
        when = _date_after(cover, m.end(), gap)
        if when:
            break
    if base in _ANNUAL_FORMS:
        out["filing_fiscal_quarter"] = 4
    if when is not None:
        if base == "8-K":
            out["filing_date"] = when.isoformat()
            out["filing_fiscal_years"] = [when.year]
        elif base in _ANNUAL_FORMS:
            out["filing_period_end"] = when.isoformat()
            out["filing_fiscal_years"] = annual_fiscal_years(when)
        else:
            out["filing_period_end"] = when.isoformat()
            quarter = _quarter_of_10q(with_tables, when)
            if quarter:
                out["filing_fiscal_quarter"] = quarter
            out["filing_fiscal_years"] = quarter_fiscal_years(when, quarter)
    keys = []
    file_number = _sec_file_number(cover)
    if file_number:
        keys.append(f"sec:{file_number}")
    name = _name_before(cover, _REGISTRANT_MARKER)
    if name:
        out["filing_company"] = name
        if normalize_company_name(name):
            keys.append(f"name:{normalize_company_name(name)}")
    english = _name_before(cover, _TRANSLATION_MARKER, after_label=True)
    if english and normalize_company_name(english) and f"name:{normalize_company_name(english)}" not in keys:
        keys.append(f"name:{normalize_company_name(english)}")
        out.setdefault("filing_company", english)
    if "filing_company" not in out:
        # Optional (Decision 88): if the signature page cannot be read, the
        # filing keeps what its cover gave.
        signed = fail_open("filing_signature_name", _signature_name, sections, default="")
        if signed:
            # Named, but not keyed. Two filings are one registrant when they
            # share any key, and that is what lets the period preference set
            # one filing aside for another's (src/vector_db/filing_period.py).
            # The file number already links a registrant's SEC filings; a
            # name key would add only its releases, and a misread name would
            # join another company's filings (Decision 98).
            out["filing_company"] = signed
    if keys:
        out["filing_entity_keys"] = keys
    return out


def _results_clauses(head: str) -> list[str]:
    """
    The text around each "results" that says what the release reports: the
    headline or sentence up to it ("PepsiCo Reports First-Quarter 2023 Results")
    and what follows "results for" ("results for the fourth quarter ended
    January 28, 2023"), each cut at a clause end or a guidance word.
    """
    clauses = []
    for m in _RESULTS.finditer(head):
        before = head[max(0, m.start() - _RESULTS_BEFORE): m.start()]
        before = _CLAUSE_END.split(before)[-1]
        guidance = list(_GUIDANCE.finditer(before))
        if guidance:
            before = before[guidance[-1].end():]
        clauses.append(before + m.group(0))
        after = head[m.end(): m.end() + _RESULTS_AFTER]
        if re.match(r"\s+for\b", after, re.IGNORECASE):
            after = _CLAUSE_END.split(after)[0]
            guidance_after = _GUIDANCE.search(after)
            clauses.append(after[: guidance_after.start()] if guidance_after else after)
    return clauses


def _release_period(head: str) -> dict:
    """Fiscal years, quarter and period end a release says it reports; {} when it does not say."""
    clauses = _results_clauses(head)
    years = {int(m.group(1)) for clause in clauses for p in _RELEASE_YEARS for m in p.finditer(clause)}
    quarters = {q for clause in clauses for p, q in _RELEASE_QUARTERS if p.search(clause)}
    # "second quarter of fiscal year 2023" is a second-quarter release; a
    # fourth-quarter or full-year one names no other quarter.
    named = quarters - {4}
    quarter = max(named) if named else (4 if quarters else None)
    end = None
    for clause in clauses:
        for m in _ENDED.finditer(clause):
            end = _date_after(clause, m.end(), 0)
            if end:
                break
        if end:
            break
    out: dict = {}
    if quarter:
        out["filing_fiscal_quarter"] = quarter
    if end:
        out["filing_period_end"] = end.isoformat()
        years.update(annual_fiscal_years(end) if quarter == 4
                     else quarter_fiscal_years(end, quarter) if quarter
                     else annual_fiscal_years(end) + quarter_fiscal_years(end))
    if years:
        out["filing_fiscal_years"] = sorted(years)
    return out


def _earnings_release(head: str) -> dict:
    ticker = _EXCHANGE_TICKER.search(head)
    period = _release_period(head)
    keys: list[str] = []
    if ticker and _RESULTS.search(head):
        keys.append(f"ticker:{ticker.group(1).lower()}")
        name = _NAME_SPLIT.split(head[max(0, ticker.start() - 80): ticker.start()])[-1]
    else:
        # No ticker in the lead: a release is still recognisable by a headline
        # at the very top that names its company and the period it reports.
        lead = _DATE_RE.sub(" ", head[:_HEADLINE_CHARS], count=1)
        headline = _HEADLINE.match(lead)
        if not headline or not period.get("filing_fiscal_years") or not _RESULTS.search(lead):
            return {}
        name = headline.group(1)
    out: dict = {"filing_form": "earnings_release"}
    name = name.strip(" _.,;|–—-")
    if re.search(r"[A-Za-z]{2}", name):
        out["filing_company"] = name
        if normalize_company_name(name):
            keys.append(f"name:{normalize_company_name(name)}")
    if not keys:
        return {}
    out["filing_entity_keys"] = keys
    out.update(period)
    # A release is called by the period it reports and also by the year it is
    # dated in, which is the year its guidance is about: Johnson & Johnson's
    # "Q4 and Full-Year 2022 Results" of January 24, 2023 is where its 2023
    # outlook is, so it is named [2022, 2023] (Decision 70, after its test look).
    first_date = _DATE_RE.search(head)
    if first_date:
        when = _date_from(_ANY_DATE.match(head, first_date.start()))
        if when:
            out["filing_date"] = when.isoformat()
            out["filing_fiscal_years"] = sorted({*out.get("filing_fiscal_years", []), when.year})
    return out


def extract_filing_metadata(sections: list[dict]) -> dict:
    """
    Company and period of a public filing, from its opening pages (and the
    registrant's name from its signature page when the cover prints none).

    Args:
        sections: The document's sections in order, as extract_sections
            produces them (reads `text` and `page_number`).

    Returns:
        The filing_* payload fields that could be read, or {} when the
        document is not recognisably a supported SEC form or an earnings
        release. Only filing_fiscal_years, filing_fiscal_quarter and
        filing_entity_keys are used for retrieval; a filing missing the years
        or the keys is simply never preferred or set aside.
    """
    cover = _cover_text(sections)
    if not cover:
        return {}
    # A cover page opens with the Commission's name and the form a few lines
    # later. A press release mentions both too ("... in our Form 10-Q filed
    # with the Securities and Exchange Commission"), but pages in.
    header = _SEC_HEADER.search(cover[:_SEC_HEADER_WITHIN])
    if header:
        form = _SEC_FORM.search(cover, header.end(), header.end() + _SEC_FORM_WITHIN)
        name = form.group(1).upper() if form else ""
        if name not in _SUPPORTED_FORMS:
            # An SEC cover of a form not modelled here (S-1, 6-K, a proxy
            # statement) or whose form name is unreadable: nothing, rather than
            # a release read off a registration statement.
            return {}
        return _sec_filing(cover, _cover_text(sections, tables=True), name + ("/A" if form.group(2) else ""),
                           sections)
    return _earnings_release(cover[:_RELEASE_HEAD_CHARS])
