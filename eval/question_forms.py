"""
The ways a deal team writes the names in a question, as deterministic
rewrites of a LegalBench-RAG question (Decision 91).

LegalBench-RAG writes every question formally: "Consider the Sponsorship
Agreement between Hydron Technologies, Inc. and Miami Dolphins, Ltd.; ...".
People type "does the nda with northwind have a non-solicit?", "change of
control in the atlas spa", or a name pasted in capitals. Named-document focus
(src/vector_db/named_document.py) reads names by their capitals, so the form
matters; these rewrites measure how much.

    original     as the benchmark writes it
    lower        fully lower-cased
    upper        ALL CAPS
    short        the forms users drop: a legal suffix after a name removed
                 ("Inc.", "LLC", "Ltd.", "Corp.", "plc", ...), and "the
                 <Title> between <A> and <B>" rewritten to name one party,
                 without "between" — "the <party> <Title>" or "the <Title>
                 with <party>" — quotes, role words (Parent, Target), asides
                 and a trailing "for <purpose>" dropped; "Non-Disclosure
                 Agreement" written "NDA". Case as the benchmark writes it.
    short-lower  short, lower-cased ("consider the nda with velco; ...")

Which party and which wording "short" uses are drawn from
Random(f"{SEED}:<query id>"), so a question's form depends on nothing else:
not on the other questions, the order they are read in, or the run. The
ground truth is untouched: a rewrite changes how a question names its
agreement, never which agreement it asks about.
"""

from __future__ import annotations

import random
import re

FORMS = ("original", "lower", "upper", "short", "short-lower")
# The seed Decision 91's measurement drew its short forms with.
SEED = "case10"

# Legal suffixes as names write them. Case-sensitive on purpose: "limited to",
# "co-branding", "se" in a question's own words are no suffixes. "Company" is
# left alone: it is also a defined term ("Company Material Adverse Effect").
_SUFFIX_WORDS = (
    "Inc", "INC", "Incorporated", "INCORPORATED", "LLC", "L.L.C", "Ltd", "LTD", "Limited", "LIMITED", "Corp", "CORP",
    "Corporation", "CORPORATION", "plc", "PLC", "Plc", "LP", "L.P", "LLP", "L.L.P", "N.A", "S.A", "AG", "SE", "GmbH",
    "N.V", "NV", "B.V", "BV", "Co", "CO", "S.p.A", "SpA", "Pty", "PTY", "Pte", "PTE", "ULC", "FSB", "LLLP", "PLLC",
)
_SUFFIX = re.compile(
    r",?[ \t]+(?:" + "|".join(re.escape(s) for s in sorted(_SUFFIX_WORDS, key=len, reverse=True)) + r")\.?"
    r"(?=[\s,;:\"'”’)\]]|$)"
)
_DANGLING = re.compile(r"[ \t]*[&,][ \t]*(?=[\"'”’;)]|$)")
_PREFIX = re.compile(r"^(Consider )the (.+?) between (.+?); (.*)$", re.S)
_DOUBLE_QUOTED = re.compile(r"[\"“]([^\"“”]+)[\"”]")
_SINGLE_QUOTED = re.compile(r"(?:^|(?<=\s))'(.+?)'(?=\s|$|,)")
_NDA = re.compile(r"\bNon-Disclosure Agreement\b")


def strip_suffixes(text: str) -> str:
    """Legal suffixes after names removed: "Hydron Technologies, Inc. and Miami Dolphins, Ltd." -> without them."""
    previous = None
    while previous != text:
        previous = text
        text = _SUFFIX.sub("", text)
    return _DANGLING.sub("", text)


def _parties(spec: str) -> list[str]:
    quoted = _DOUBLE_QUOTED.findall(spec) or _SINGLE_QUOTED.findall(spec)
    if quoted:
        return [strip_suffixes(q).strip(" ,") for q in quoted]
    spec = strip_suffixes(spec)
    spec = re.sub(r"\s*\([^()]*\)", "", spec)
    # A trailing description of the deal is no party: "... and Kathy Whitworth
    # with Intellectual Property Rights ...", "... and GINARES GROUP AG for
    # Renewable Energy Projects".
    spec = re.sub(r"\s+(?:for|with)\s+.*$", "", spec)
    parts = [p.strip(" ,") for p in re.split(r",?\s+and\s+|,\s+", spec) if p.strip(" ,")]
    return parts or [spec]


def _clean_party(party: str, last: bool) -> str:
    if last:
        party = re.sub(r"\s+for\s+.*$", "", party)
    party = re.sub(r"^(?:Parent|Target|The|the)\s+", "", party.strip())
    return party.strip(" ,.\"'“”")


def short_form(query: str, query_id: str) -> str:
    """The question as a deal team shortens it (the "short" form, case kept)."""
    rng = random.Random(f"{SEED}:{query_id}")
    match = _PREFIX.match(query)
    if match:
        consider, title, spec, body = match.groups()
        parties = _parties(spec)
        pick_last = rng.random() < 0.5
        party = _clean_party(parties[-1] if pick_last else parties[0], last=pick_last)
        title = _NDA.sub("NDA", title)
        if rng.random() < 0.5:
            out = f"{consider}the {party} {title}; {strip_suffixes(body)}"
        else:
            out = f"{consider}the {title} with {party}; {strip_suffixes(body)}"
    else:
        out = strip_suffixes(_NDA.sub("NDA", query))
    return re.sub(r"[ \t]{2,}", " ", out)


def question_form(query: str, query_id: str, form: str) -> str:
    """The question rewritten in one of FORMS."""
    if form == "original":
        return query
    if form == "lower":
        return query.lower()
    if form == "upper":
        return query.upper()
    if form == "short":
        return short_form(query, query_id)
    if form == "short-lower":
        return short_form(query, query_id).lower()
    raise ValueError(f"unknown question form {form!r}; one of {FORMS}")
