"""
Document identity — which agreement a document is and who its parties are,
read from the document's own opening.

A diligence question usually names the agreement it is about: "the Northwind
NDA", "the SPA with Atlas", "the Acquisition Agreement between Parent X and
Target Y". In a room of similar contracts that name is what tells them
apart, and retrieval did not use it: document focus (Decision 67) searches
inside whichever document holds the most fused score, and with all 714
LegalBench-RAG files indexed that was the wrong document for 41 of 94
ContractNLI and 11 of 102 MAUD dev questions (Decision 85).

Every agreement states its identity in fixed language at the top:

    MUTUAL NON-DISCLOSURE AGREEMENT
    This Agreement is made ... by and between HYDRON TECHNOLOGIES, INC., a New
    York corporation ... ("Hydron") and MIAMI DOLPHINS, LTD., ... ("Dolphins").

    AGREEMENT AND PLAN OF MERGER by and among STERIS plc, Solar New US Holding
    Co, LLC, Crystal Merger Sub 1, LLC and Cantel Medical Corp.

so the extraction is regexes over the opening — no model, no network:

    doc_title     the document's own title ("MUTUAL NON-DISCLOSURE AGREEMENT")
    doc_parties   the names in the first party clause, each cut at its
                  descriptor ("a New York corporation", an address), with the
                  short names the clause gives them ("Hydron"); parties the
                  first lines introduce otherwise ("AG Projects, a Dutch
                  corporation", 'INaturals ("Company")'); organisation names
                  with a legal form (Inc., LLC, plc, ...) near the top. Role
                  words ("the Company", "Buyer") are not names.

Nothing here reads the file name. A data room's names are not reliable
("NDA_final_v3.pdf"), and on the benchmark the file name is where the
question's own wording came from, so reading it would measure the benchmark's
construction rather than the documents (see Decision 86).

The opening is the text of the document's first child chunks in chunk order,
so ingestion and a backfill over an existing index read exactly the same
input — the payloads — and write exactly the same fields. A chunk flagged as
containing PII is skipped: its text is withheld from default retrieval, and
nothing derived from it is copied onto every other chunk of the document.
A field is written only when found; a document with neither is left without
identity, and the query step (src/vector_db/named_document.py) then keeps the
fused-score choice for it.
"""

from __future__ import annotations

import re
import unicodedata

# The opening read for identity, in characters of chunk text. The parties of a
# contract are named in its first paragraph; a merger agreement's cover lists
# them above its table of contents. Further in, "between" and organisation
# names belong to recitals and third parties.
OPENING_CHARS = 4_000
# Where a party clause may start, and how far organisation names with a legal
# form are looked for (a template's issuer, a cover's list of parties).
PARTY_CLAUSE_WITHIN = 3_000
ORG_FALLBACK_CHARS = 1_500
_CLAUSE_MAX_CHARS = 800
MAX_PARTIES = 12
MAX_NAME_CHARS = 100
_MAX_NAME_WORDS = 10
MAX_TITLE_CHARS = 120

_WS = re.compile(r"[ \t\r\f\v]+")

# Kinds of document a title names.
_DOC_TYPE = re.compile(
    r"\b(?:agreements?|contracts?|amendments?|addend(?:um|a)|leases?|licen[cs]es?|polic(?:y|ies)"
    r"|plan\s+of\s+merger|indentures?|deeds?|memorand(?:um|a)|letters?|notes?|charters?|bylaws"
    r"|terms|undertakings?|guarant(?:ee|y)|consents?|waivers?|statement\s+of\s+work|protocols?"
    r"|declarations?|certificates?|instruments?|arrangements?|NDA|MOU|LOI)\b",
    re.IGNORECASE,
)
# Lines above a title that are not it: exhibit numbers, version stamps and
# confidentiality legends.
_TITLE_NOISE = re.compile(
    r"^(?:exhibit\s+[\w.\-()]+|execution\s+(?:version|copy|draft)|confidential(?:\s+treatment\s+requested)?"
    r"|final(?:\s+version)?|draft|redacted(?:\s+copy)?)\b[\s:.\-]*",
    re.IGNORECASE,
)
# "THIS SPONSORSHIP AGREEMENT (...)", "This Mutual Non-Disclosure Agreement is ..."
_THIS_TITLE = re.compile(
    r"\b(?:THIS|This)\s+((?:[A-Z][\w\-/&']*,?\s+(?:AND\s+|and\s+|OF\s+|of\s+)?){0,8}?"
    r"(?:AGREEMENT|Agreement|CONTRACT|Contract|AMENDMENT|Amendment|LEASE|Lease|LICENSE|License))\b"
)

# The first party clause: "between", "by and between", "among", "by and among".
# Not \b at the end: a template's blank follows the word directly ("between____").
_PARTY_CLAUSE = re.compile(r"\b(?:by\s+and\s+)?(?:between|among(?:st)?)(?![A-Za-z])\s*:?\s*", re.IGNORECASE)
# Where the clause ends: a sentence end, the recitals, the cover's date, the
# table of contents. A period after an abbreviation ("Inc.", "N.A.") is not one.
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_CLAUSE_END = re.compile(
    r"(?P<period>\.)\s+(?=[A-Z\"“])"
    r"|;|\bWHEREAS\b|\bRECITALS?\b|\bWITNESSETH\b|\bBACKGROUND\b|\bPREAMBLE\b|\b(?i:table\s+of\s+contents?)\b"
    r"|\b[Dd]ated\b|\bDATED\b|\beffective\s+as\s+of\b|\bas\s+of\s+the\s+date\b|\bIN\s+WITNESS\b"
    r"|\b(?i:" + _MONTH + r")\s+\d",
)
_ABBREVIATION = re.compile(r"(?:\b[A-Za-z]|\b(?i:inc|ltd|co|corp|no|st|jr|sr|mr|ms|mrs|dr|bros|intl|int'l))$")
# Inside a clause, a party's name ends where its description begins: its form
# of organisation, its address, its registration.
_DESCRIPTOR = re.compile(
    r"(?:^|,?\s+)(?:a|an)\s+(?=[A-Za-z])"     # ", a Delaware corporation", " an individual"
    r"|\s+(?:with|having|whose|located|residing|represented|organized|organised|existing|duly"
    r"|incorporated\s+(?:in|under)|registered|hereinafter|herein|acting|through|on\s+behalf"
    r"|together\s+with|as\s+(?:company|seller|buyer|purchaser|repairer|agent|trustee)|of\s+the\s+one"
    r"|of\s+the\s+other|of\s+the\s+first|of\s+the\s+second)\b"
    r"|,?\s+(?:company\s+)?registration\s+(?:number|no)\b"
    r"|\s+of\s+(?=\d)|\s+at\s+(?=\d)|,\s*(?=\d)|\s+(?=\d{2,})|_{2,}|:",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"[\"“”'‘’]([^\"“”'‘’()]{1,60})[\"“”'‘’]")
# Legal-form words after a name: "Product Development Experts, Inc." keeps its
# comma; "STERIS plc, Solar New US Holding Co, LLC, ..." splits at the other commas.
_CORPORATE_FORMS = (
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "l.l.c", "ltd", "limited",
    "plc", "lp", "l.p", "llp", "l.l.p", "sa", "s.a", "s/a", "ag", "gmbh", "nv", "n.v", "bv", "b.v",
    "se", "s.p.a", "spa", "srl", "s.r.l", "sas", "s.a.s", "pty", "pte", "ab", "asa", "a/s",
    "oy", "oyj", "kk", "k.k", "kg", "gie", "g.i.e", "ulc", "pllc", "p.c", "lllp",
)
_FORM = r"(?i:" + "|".join(re.escape(f) for f in sorted(_CORPORATE_FORMS, key=len, reverse=True)) + r")\.?"
_FORM_ONLY = re.compile(r"^\s*" + _FORM + r"\s*$")
_FORM_END = re.compile(r",?\s+" + _FORM + r"\s*$")
# A cover lists one party per line; flattened, "PERSPECTA INC. JAGUAR PARENTCO
# INC." must still be two names: a legal form ends a name.
_AFTER_FORM = re.compile(r"(\b" + _FORM + r")\s+(?=[A-Z0-9])")
_COMMA_SPLIT = re.compile(r",\s+(?!" + _FORM + r"(?:[\s,.;()]|$))(?=[A-Za-z0-9\"“])")
_AND_SPLIT = re.compile(r"\s+(?:and|AND|And)\s+|\s*\bor\b\s*(?=[A-Z])")
# A capitalised name of up to eight words ("Arkansas Western Pipeline Company",
# "LONDON SCHOOL OF ECONOMICS & POLITICAL SCIENCE"), with its legal form.
_NAME = (r"(?=[\w&'’.\-]*[A-Za-z])[A-Z0-9][\w&'’.\-]*"
         r"(?:[ \t]+(?:(?=[\w&'’.\-]*[A-Za-z])[A-Z0-9&][\w&'’.\-]*|&|of|de|du|for))*?"
         r"(?:,?[ \t]+" + _FORM + r")?")
# Organisation names with a legal form near the top: "THE BLEECKER CONSULTING
# GROUP, LLC", "Lyophilization Technology, Inc.". Not "a Dutch corporation".
_ORG = re.compile(
    r"(?<![Aa][ \t])(?<![Aa]n[ \t])\b((?:[A-Z0-9][\w&'’.\-]*[ \t]+(?:&[ \t]+|of[ \t]+|de[ \t]+|and[ \t]+)?){0,6}"
    r"[A-Z0-9][\w&'’.\-]*),?[ \t]+(" + _FORM + r")(?![\w])",
)
# Not the description itself ("a Delaware corporation ("NTC")").
_NOT_DESCRIPTION = r"(?<![Aa][ \t])(?<![Aa]n[ \t])\b"
# A party introduced by its description: "AG Projects, a Dutch corporation",
# "HYDRON TECHNOLOGIES, INC., a New York corporation".
_DESCRIBED = re.compile(
    _NOT_DESCRIPTION + r"(?P<name>" + _NAME + r"),?[ \t]+(?:a|an)[ \t]+(?:[\w.\-]+[ \t]+){0,5}?"
    r"(?:corporation|company|limited[ \t]+liability[ \t]+company|(?:limited[ \t]+)?partnership|association"
    r"|banking[ \t]+(?:association|corporation)|national[ \t]+bank|bank|trust|cooperative|soci[eé]t[eé]"
    r"|public[ \t]+limited[ \t]+company|private[ \t]+limited[ \t]+company|statutory[ \t]+body)\b",
)
# A party followed by the role the document will call it by: 'INaturals ("Company")',
# 'M5 Computer Security ("Recipient")'. Other defined terms ("Plan", "Common
# Stock") follow things that are not parties.
_BEFORE_DEFINED = re.compile(
    _NOT_DESCRIPTION + r"(?P<name>" + _NAME + r")[ \t]*\([ \t]*(?:hereinafter[ \t]+(?:referred[ \t]+to[ \t]+as[ \t]+"
    r"|called[ \t]+)?)?(?:the[ \t]+)?[\"“](?P<term>[^\"”()]{1,40})[\"”][ \t]*\)",
)
# A short name the document defines in parentheses: ("SINTEC"), (the "LTI").
_DEFINED_SHORT = re.compile(
    r"\([^()]{0,60}?[\"“]([A-Z][A-Za-z0-9&.\-]{1,20}(?:[ \t][A-Z][A-Za-z0-9&.\-]{1,20}){0,2})[\"”][ \t]*\)"
)
_NAME_CONNECTORS = frozenset({"of", "and", "de", "du", "der", "la", "le", "for", "the", "&"})
_NAME_ACRONYM = re.compile(
    _NOT_DESCRIPTION + r"(?P<name>" + _NAME + r")[ \t]*\([ \t]*(?P<short>[A-Z][A-Z0-9&]{1,7})[ \t]*\)"
)
_LEADING_NOISE = re.compile(r"^(?:(?:exhibit|annex|schedule)\s+[\w.\-()]+\s+|\d[\d\s.,\-/]*\s+(?=[A-Za-z])"
                            r"|(?:and|of|by|with|to)\s+)+", re.IGNORECASE)
# Where the names around a party clause are read.
NAMES_WITHIN = 1_500

# The roles a contract defines its parties by: a defined name that is one of
# these is not a name, but a name followed by one is a party ('INaturals ("Company")').
_PARTY_ROLES = frozenset({
    "company", "parent", "buyer", "seller", "purchaser", "vendor", "recipient", "discloser",
    "disclosing party", "receiving party", "party", "parties", "each party", "licensee",
    "licensor", "sublicensee", "distributor", "supplier", "customer", "client", "contractor",
    "subcontractor", "consultant", "employee", "employer", "executive", "lender", "lenders", "borrower",
    "landlord", "tenant", "lessor", "lessee", "guarantor", "agent", "broker", "merger sub", "merger sub i",
    "merger sub ii", "merger subsidiary", "acquiror", "acquirer", "acquisition sub", "target", "investor",
    "investors", "shareholder", "stockholder", "member", "manager", "partner", "sponsor", "owner", "operator",
    "manufacturer", "reseller", "provider", "service provider", "franchisee", "franchisor", "transporter",
    "shipper", "developer", "publisher", "producer", "participant", "participants", "inventor", "mentor",
    "user", "you", "we", "us", "holder", "holders", "assignor", "assignee", "grantor", "grantee", "trustee",
    "issuer", "underwriter", "depositary", "escrow agent", "bank", "individual", "university",
    "institution", "student", "repairer", "professional", "endorser", "talent", "artist",
    "author", "affiliate", "affiliates", "subsidiary", "collaborator", "marketer",
    "sales agent", "representative", "company or individual", "beneficiary", "obligor", "obligee", "carrier",
})
# Defined terms and template words that are neither names nor roles.
_NOT_NAMES = frozenset({
    "agreement", "amendment", "effective date", "purpose", "information", "confidential information",
    "proprietary information", "territory", "products", "product", "services", "term", "closing", "merger",
    "transaction", "transactions", "business", "surviving corporation", "surviving company", "surviving entity",
    "effective time", "date", "execution date", "confidentiality", "registration number", "organization name",
})
# A name that holds one of these is a heading or a template blank, not a party.
_NOT_NAME_WORDS = re.compile(
    r"\b(?:agreements?|contracts?|amendments?|addend(?:um|a)|memorand(?:um|a)|NDA|insert|yyyy|xx+"
    r"|name|address|street|avenue|ave|road|drive|boulevard|blvd|suite|floor|highway|hwy|p\.?o\.?\s+box"
    r"|article|section|representations|warranties|between|among|means)\b",
    re.IGNORECASE)
_SENTENCE_WORDS = frozenset({"now", "therefore", "it", "is", "as", "follows", "hereby", "agreed", "agree",
                             "shall", "will", "may", "that", "this", "which", "whereas", "be", "are", "by"})
# Words no organisation's or person's name holds: one of them makes a phrase of
# a sentence ("ANY INFORMATION", "WHO ARE GIVEN ACCESS", "BE CONFIDENTIAL").
_NON_NAME_WORDS = frozenset({"any", "by", "who", "whom", "whose", "which", "whatever", "whoever", "are", "is", "be",
                             "been", "being", "was", "were", "shall", "will", "may", "must", "should", "would",
                             "could", "this", "these", "those", "such", "not", "than", "where", "when", "if"})

# Tokens that never tell one document from another: function words, legal
# forms and party roles. Kept small — rarity across the documents being
# compared (named_document.py's weighting) does the rest.
STOPWORDS = frozenset({
    "a", "an", "the", "of", "and", "or", "in", "on", "for", "to", "by", "with", "between", "among", "amongst",
    "this", "that", "these", "those", "its", "it", "is", "are", "be", "as", "at", "from", "into", "under",
    "s", "de", "du", "la", "le", "des", "der", "und", "y", "et",
    *(f.replace(".", "").replace("/", "") for f in _CORPORATE_FORMS),
    "company", "companies", "corporation", "incorporated", "limited", "holdings", "holding",
    "party", "parties",
})
_TOKEN = re.compile(r"[a-z0-9]+")
# Compounds whose parts a question may write joined or apart: "i-on", "Coca-Cola", "theglobe.com".
_COMPOUND = re.compile(r"[a-z0-9]+(?:[\-./'’][a-z0-9]+)+")
_ACRONYM_SKIP = frozenset({"and", "of", "the", "for", "to", "a", "an", "in", "on", "by", "with"})


def _ascii_fold(text: str) -> str:
    """NFKD with combining marks removed: "Aéreas" -> "Aereas", "MOËT" -> "MOET"."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def name_tokens(text: str) -> set[str]:
    """
    The words a name or a question is matched on.

    Lower-cased, accents removed, split on everything but letters and digits;
    a hyphenated, dotted or apostrophised compound also counts joined ("i-on"
    -> "i", "on", "ion"), so "Coca-Cola" and "CocaCola" meet. Stopwords and
    one-letter words are dropped.

    Args:
        text: A party name, a title, or a question.

    Returns:
        Set of tokens (may be empty).
    """
    folded = _ascii_fold(text or "").lower()
    tokens = set(_TOKEN.findall(folded))
    tokens.update(re.sub(r"[\-./'’]", "", m.group(0)) for m in _COMPOUND.finditer(folded))
    return {t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())}


def title_acronyms(title: str) -> set[str]:
    """
    Short forms a question may call the title by: "Non-Disclosure Agreement"
    -> {"nda"}, "Mutual Non-Disclosure Agreement" -> {"nda", "mnda"},
    "Share Purchase Agreement" -> {"spa"}. The initials of each trailing run of
    three or more significant words (hyphenated parts count as words) of the
    title and of each part of it a dash or colon sets apart ("DEPARTMENT OF
    HOMELAND SECURITY — NON-DISCLOSURE AGREEMENT" -> "dhs" too), plus an
    acronym the title itself gives in parentheses ("(NDA)").
    """
    out: set[str] = set()
    folded = _ascii_fold(title or "")
    for part in {folded, *re.split(r"\s[—–-]\s|:|\|", folded)}:
        words = [w for w in re.split(r"[\s\-/]+", part.lower()) if w and w[0].isalnum() and w not in _ACRONYM_SKIP]
        words = [w for w in (re.sub(r"[^a-z0-9]", "", w) for w in words) if w]
        out |= {"".join(w[0] for w in words[i:]) for i in range(len(words)) if len(words) - i >= 3}
    out.update(m.lower() for m in re.findall(r"\(([A-Z]{2,6})\)", title or ""))
    return out


def document_opening(payloads: list[dict], max_chars: int = OPENING_CHARS) -> str:
    """
    The opening of a document from its child chunk payloads.

    Chunks are read in chunk_index order; a chunk flagged contains_pii=1 is
    skipped. A section heading a processor lifted out of the text is put back
    on its own line, once, so the title is found whichever processor parsed
    the file.

    Args:
        payloads: The document's child payloads, any order (reads chunk_index,
            text, section_heading, contains_pii).
        max_chars: Characters of chunk text to read.

    Returns:
        The opening, lines kept.
    """
    parts: list[str] = []
    used = 0
    last_heading = ""
    for payload in sorted(payloads, key=lambda p: p.get("chunk_index") or 0):
        if payload.get("contains_pii"):
            continue
        text = payload.get("text") or ""
        heading = (payload.get("section_heading") or "").strip()
        if heading and heading != last_heading and heading.lower() not in text.lower():
            text = f"{heading}\n{text}"
        last_heading = heading or last_heading
        parts.append(text)
        used += len(text)
        if used >= max_chars:
            break
    return "\n".join(parts)[:max_chars]


def _clean(text: str) -> str:
    return _WS.sub(" ", text.replace("\n", " ")).strip(" ,;:-–—\"“”'‘’()[]")


def _heading_like(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 4:
        return False
    upper = sum(1 for c in letters if c.isupper()) / len(letters)
    words = [w for w in re.findall(r"[A-Za-z][\w\-']*", line) if w.lower() not in _ACRONYM_SKIP]
    titled = bool(words) and all(w[0].isupper() for w in words)
    return upper >= 0.6 or titled


def extract_title(opening: str) -> str:
    """
    The document's own title: the first short heading-like line that names a
    kind of document ("AGREEMENT AND PLAN OF MERGER", "Mutual Non-Disclosure
    Agreement"), exhibit numbers and version stamps removed; else the phrase
    of its opening sentence ("THIS SPONSORSHIP AGREEMENT"); else "".
    """
    for raw in opening.splitlines()[:40]:
        line = _clean(raw)
        while True:
            stripped = _TITLE_NOISE.sub("", line).strip()
            if stripped == line:
                break
            line = stripped
        if not line or len(line) > MAX_TITLE_CHARS or len(line.split()) > 15:
            continue
        if re.match(r"(?i)(?:this|the)\s", line) or line.endswith("."):
            continue
        if _DOC_TYPE.search(line) and _heading_like(line):
            return line
    # "THIS AGREEMENT" alone names no kind of agreement.
    for match in _THIS_TITLE.finditer(opening[:PARTY_CLAUSE_WITHIN]):
        title = _clean(match.group(1))
        if len(title.split()) >= 2:
            return title
    return ""


def _key(name: str) -> str:
    return re.sub(r"\s+", " ", _ascii_fold(name).lower()).strip(" .,").removeprefix("the ")


def _is_party_role(term: str) -> bool:
    return _key(term) in _PARTY_ROLES


def _is_role(name: str) -> bool:
    """A role or a defined term, not a name ("Receiving Party", "Entrusting party")."""
    key = _key(name.replace(",", " "))
    return key in _PARTY_ROLES or key in _NOT_NAMES or bool(re.search(r"\bpart(?:y|ies)$", key))


def _as_name(text: str) -> str:
    """`text` cleaned, if it reads as a party's name; else ""."""
    name = _LEADING_NOISE.sub("", _clean(text))
    if not name or len(name) > MAX_NAME_CHARS or len(name.split()) > _MAX_NAME_WORDS:
        return ""
    # A name starts with a capital or a digit ("3M"), or carries a legal form
    # ("salesforce.com, inc."); it is not only a role or a legal form, a state
    # or postal code ("CA", "V6B 1N2"), or a date.
    if not (name[0].isupper() or name[0].isdigit() or _FORM_END.search(name)) \
            or _is_role(name) or _FORM_ONLY.match(name):
        return ""
    letters = sum(c.isalpha() for c in name)
    if letters < 3 or sum(c.isdigit() for c in name) * 3 > letters or re.match(_MONTH + r"\b", name):
        return ""
    # A sentence, a heading or a template blank is not a name ("Shipper is
    # engaged in business as", "NOW THEREFORE IT IS AGREED", "Non-Disclosure
    # Agreement", "[insert name]").
    words = name.split()
    if sum(1 for w in words if w[0].islower() and w not in _NAME_CONNECTORS) > 1 \
            or sum(1 for w in words if w.lower().strip(".,") in _SENTENCE_WORDS) > 1 \
            or any(w.lower().strip(".,;:") in _NON_NAME_WORDS for w in words) \
            or _NOT_NAME_WORDS.search(name):
        return ""
    return name if name_tokens(name) else ""


def _all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 20 and sum(c.isupper() for c in letters) >= 0.8 * len(letters)


def _as_short_name(text: str) -> str:
    """A quoted short name ("Hydron", "LTI", "i-on"), or "" for a role, a term or a blank."""
    short = _clean(text)
    if not short or _is_role(short) or set(short) <= set("_ .") or _NOT_NAME_WORDS.search(short):
        return ""
    if " " in short and not short[0].isupper():
        return ""
    return short if name_tokens(short) else ""


def _names_in_part(part: str, needs_form: bool = False) -> list[str]:
    """
    Names in one stretch of a clause between "and"s and asides. The first
    description ends the names ("HEALTHCARE CAPITAL CORP., of 1100 West 73rd
    Avenue, Vancouver, British Columbia" is one party); before it, commas
    separate a list ("STERIS plc, Solar New US Holding Co, LLC, ...").
    With needs_form, only names that end in a legal form count.
    """
    head = _DESCRIPTOR.split(part, maxsplit=1)[0]
    names: list[str] = []
    for segment in _COMMA_SPLIT.split(head):
        if _FORM_ONLY.match(segment) and names:
            names[-1] = f"{names[-1]}, {_clean(segment)}"
            continue
        name = _as_name(segment)
        if name:
            names.append(name)
    return [n for n in names if _FORM_END.search(n)] if needs_form else names


def _clause_end(rest: str) -> int:
    """Offset where a party clause ends in `rest` (its length when nothing ends it)."""
    for end in _CLAUSE_END.finditer(rest):
        if end.group("period") and _ABBREVIATION.search(rest[: end.start()]):
            continue
        return end.start()
    return len(rest)


def _parties_in_clause(clause: str) -> list[str]:
    """Names and defined short names in one party clause."""
    names: list[str] = []
    # Parenthesised asides hold the defined short names ("Hydron") and roles
    # ("as Company"); they also end the name before them.
    for aside in re.findall(r"\(([^()]*)\)", clause):
        # 'the "Company"', 'hereinafter referred to as "LTI"', or a bare "(SAMED)".
        quoted = _QUOTED.findall(aside) or re.findall(r"^\s*([A-Z][A-Z0-9&]{1,7})\s*$", aside)
        for raw in quoted:
            short = _as_short_name(raw)
            if short:
                names.append(short)
    # Asides and a template's blanks ("between ______ and ...") end a name too.
    body = re.sub(r"\([^()]*\)|\[[^\[\]]*\]|_{2,}", " ; ", clause)
    body = _AFTER_FORM.sub(r"\1 ; ", body)
    # In capitals every word looks like a name: "BETWEEN NATURAL OR LEGAL
    # PERSONS WHO ARE GIVEN ACCESS ..." is a sentence. There a name must end
    # in a legal form ("ATLAS MERGER SUB, INC."); quoted short names still count.
    needs_form = _all_caps(body)
    for piece in re.split(r"\s;\s", body):
        for part in _AND_SPLIT.split(piece):
            names += _names_in_part(part, needs_form)
    return names


def _is_alias(short: str, names: list[str]) -> bool:
    """A defined short name that stands for a known name: a word of it, or its initials."""
    key = re.sub(r"[^a-z0-9]", "", _ascii_fold(short).lower())
    for name in names:
        words = [w for w in re.findall(r"[a-z0-9]+", _ascii_fold(name).lower()) if w not in STOPWORDS]
        if key in words or (len(key) >= 2 and key == "".join(w[0] for w in words)[: len(key)]):
            return True
    return False


def _top_names(top: str, known: list[str]) -> list[str]:
    """
    Parties the first lines introduce without a "between" clause: by their
    description ("AG Projects, a Dutch corporation"), or followed by the role
    the document calls them by ('INaturals ("Company")'); and the short names
    defined for any party found ('(the "LTI")').
    """
    names = [m.group("name") for m in _DESCRIBED.finditer(top)]
    names += [m.group("name") for m in _BEFORE_DEFINED.finditer(top) if _is_party_role(m.group("term"))]
    # "South African Medical Devices Industry Association (SAMED)": the name and its acronym.
    acronyms = [(m.group("name"), m.group("short")) for m in _NAME_ACRONYM.finditer(top)]
    names += [name for name, _ in acronyms]
    out = [n for n in (_as_name(raw) for raw in names) if n]
    out += [s for s in (_as_short_name(short) for name, short in acronyms if _as_name(name)) if s]
    out += [s for s in (_as_short_name(raw) for raw in _DEFINED_SHORT.findall(top))
            if s and _is_alias(s, known + out)]
    return out


def extract_parties(opening: str) -> list[str]:
    """
    The parties a document names: its first party clause, with the short
    names that clause defines; the parties its first lines introduce by
    description ("AG Projects, a Dutch corporation") or by the role they will
    be called ('INaturals ("Company")'); and organisation names with a legal
    form near the top. At most MAX_PARTIES, duplicates (by their words) once.
    """
    head = opening[:PARTY_CLAUSE_WITHIN]
    flat = _WS.sub(" ", head.replace("\n", " "))
    names: list[str] = []
    for match in _PARTY_CLAUSE.finditer(flat):
        rest = flat[match.end(): match.end() + _CLAUSE_MAX_CHARS]
        names = _parties_in_clause(rest[: _clause_end(rest)])
        if names:
            break
    # Line breaks kept: a name does not run across lines, so the title line
    # above "AG Projects, a Dutch corporation" is not read as part of it.
    top = _WS.sub(" ", opening[:NAMES_WITHIN])
    names += _top_names(top, names)
    # Organisations with a legal form in the first lines: a template's issuer
    # ("THE BLEECKER CONSULTING GROUP, LLC"), a cover listing its parties one
    # per line ("PENN VIRGINIA CORPORATION AND LONESTAR RESOURCES US INC.").
    top = _WS.sub(" ", opening[:ORG_FALLBACK_CHARS])
    orgs = [n for n in (_as_name(f"{m.group(1)}, {m.group(2)}") for m in _ORG.finditer(top))
            if n and not _is_role(n)]
    names += orgs
    names += [s for s in (_as_short_name(raw) for raw in _DEFINED_SHORT.findall(top))
              if s and _is_alias(s, orgs)]
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = " ".join(sorted(name_tokens(name)))
        if key and key not in seen:
            seen.add(key)
            out.append(name[:MAX_NAME_CHARS])
    return out[:MAX_PARTIES]


def extract_document_identity(payloads: list[dict]) -> dict:
    """
    doc_title and doc_parties for one document, from its child payloads.

    Args:
        payloads: The document's child chunk payloads (as written to the index).

    Returns:
        {"doc_title": str, "doc_parties": [str]} with only the fields found;
        {} when neither is.
    """
    opening = document_opening(payloads)
    if not opening.strip():
        return {}
    identity: dict = {}
    title = extract_title(opening)
    if title:
        identity["doc_title"] = title
    parties = extract_parties(opening)
    if parties:
        identity["doc_parties"] = parties
    return identity
