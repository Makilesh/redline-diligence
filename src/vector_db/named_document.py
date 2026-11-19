"""
Prefer the document a question names (Decision 86).

Document focus (Decision 67) searches again inside one document, the one
holding the most fused score, and lets the cross-encoder choose among both
sets. On a room of near-identical contracts the fused score points at the
wrong one: with all 714 LegalBench-RAG files indexed, focus searched the
wrong agreement for 41 of 94 ContractNLI and 11 of 102 MAUD dev questions
(Decision 85), although every one of those questions names its agreement
("Consider the Acquisition Agreement between Parent X and Target Y; ...").
A deal team's questions do the same: "the Northwind NDA", "the SPA with Atlas".

Each document carries its identity from ingestion — its title and the parties
its opening names (src/data_processing/document_identity.py, read from the
text, never the file name). A question NAMES a document when its words match
that document's identity and no other retrieved document's nearly as well:

  - the words compared are those the question writes with a capital — names
    are proper nouns; "does the document state that ..." names nothing —
    against each document's title and party names (accents, case and
    punctuation ignored, legal forms dropped). A title's short form ("NDA",
    "SPA") counts when the question writes it in capitals;
  - a question whose capitals mark no name — typed in lower case ("does the
    nda with northwind have a non-solicit?"), with a capital only where a
    sentence starts, or in ALL CAPS — is CASELESS (Decision 91). Its words are
    compared by what the retrieved text does with them instead: a word that
    text writes in lower case anywhere ("state", "general", "services",
    "document") is an ordinary word there and names nothing; a word it never
    writes in lower case ("northwind", "velco", "bankshares") counts as in a
    capitalised question. A name made only of ordinary words ("community
    bankers trust", "product development experts") counts by its run: two or
    more question words in a row that are also in a row in one retrieved
    party's name count whatever the text does with them. A short form counts
    when, besides, the retrieved text writes it in capitals ("NDA"), so "app"
    in "does this app sell my data?" is no "Agreement for Prospective
    Purchasers". Before this an ALL CAPS question counted every word as a
    name and every short word as a short form;
  - a word counts by how rare it is among the documents the first search
    retrieved, in their text as well as their identities. A party name occurs
    in one of them; a word of the question's topic ("confidential",
    "information", "merger") occurs in most, because the search retrieved
    them for that topic, and counts for almost nothing. That is what keeps a
    title such as "Critical Energy Infrastructure Information NDA" from being
    named by every question about confidential information;
  - the named document must score at least half as much again as the next
    one (two documents named about as strongly is a question about both). So
    "the Northwind NDA" names the Northwind NDA over the Northwind MSA by the
    title's short form, and "Northwind's agreement" names neither.

Only where focus already runs, and only the choice of document changes: the
named document's passages are searched instead of the fused leader's, and
nothing is removed from the rerank pool, so the cross-encoder still decides.
When nothing is named — no identity on the retrieved documents (an index
built before Decision 86 and not backfilled), no matching word, a tie, two
documents named — the fused-score choice stands, exactly as before.
NAMED_DOCUMENT_FOCUS=0 turns it off.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Iterable

from src.data_processing.document_identity import STOPWORDS, _ascii_fold, name_tokens, title_acronyms

# Least evidence that names a document (a word that only one of four or more
# retrieved documents holds weighs >= 1.0), and the lead it needs over the
# next one: s1 >= (1 + NAMED_MARGIN) * s2. Chosen on LegalBench-RAG dev
# (Decision 86): at 0.5 the named document was right 143 times and wrong 3
# (the fused choice was wrong too), never displacing a right choice; 1.0 named
# 17 fewer, all right, for 0.5 point less recall.
MIN_EVIDENCE = 1.0
NAMED_MARGIN = 0.5


def named_document_enabled() -> bool:
    """NAMED_DOCUMENT_FOCUS=0 keeps the fused-score choice everywhere (on by default)."""
    return os.getenv("NAMED_DOCUMENT_FOCUS", "1").strip().lower() not in {"0", "false", "off", "no"}


@dataclass
class NamedDocument:
    """The document a question names, if any (for the focus step, logs and the trace)."""
    document: str | None = None          # doc_id
    source_file: str = ""
    score: float = 0.0
    runner_up: float = 0.0
    matched: list[str] = field(default_factory=list)
    documents: int = 0                   # documents retrieved by the first search
    reason: str = ""                     # why nothing was named, or "named"


def identity_terms(payload: dict) -> set[str]:
    """The words a document can be named by: those of its parties' names and of its title."""
    terms: set[str] = set()
    for party in payload.get("doc_parties") or []:
        terms |= name_tokens(str(party))
    return terms | name_tokens(str(payload.get("doc_title") or ""))


def identity_acronyms(payload: dict) -> set[str]:
    """The title's short forms ("nda", "spa"); they name the document only when written in capitals."""
    return title_acronyms(str(payload.get("doc_title") or "")) - identity_terms(payload)


# An acronym as a question writes it: "the Northwind NDA", "DHS's agreement".
_CAPITALS = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")
# A word the question writes with a capital: "Northwind's", "Co-Branding", "M5-Systems".
_CAPITALISED = re.compile(r"(?<![\w&'’.\-])[A-Z][\w&'’.\-]*")
# After these a capital starts a sentence; it marks no name ("Does the nda ...").
_SENTENCE_END = ".!?;:\n"
# A word a retrieved text writes in lower case, standing on its own: not part
# of a URL, an e-mail address, a domain or a dotted abbreviation
# ("www.northwind.com", "legal@atlas.com", "e.g.").
_LOWER_WORD = re.compile(r"(?<![\w@/.\\])[a-z]+(?:[\-'’][a-z]+)*(?![\w@/\\]|\.\w)")
# A short form a retrieved text writes in capitals: "NDA", "SPA".
_CAPITALS_WORD = re.compile(r"(?<![\w@/.\\])[A-Z]{2,8}(?![\w@/\\]|\.\w)")


def caseless(query: str) -> bool:
    """
    True when the question's capitals mark no name: ALL CAPS (fewer than one
    letter in ten in lower case), or a capital only where a sentence starts
    (and "I"). "does the nda with northwind ...", "Does the nda with
    northwind ...", "DOES THE NDA WITH NORTHWIND ..." are; "Does the NDA ...",
    "the Northwind nda" are not.
    """
    letters = [c for c in query if c.isalpha()]
    if sum(c.islower() for c in letters) < 0.1 * len(letters):
        return True
    for match in _CAPITALISED.finditer(query):
        before = query[: match.start()].rstrip()
        if match.group(0) != "I" and before and before[-1] not in _SENTENCE_END:
            return False
    return True


# A word as name_tokens reads one, in order (pieces of a compound, no joined form).
_WORD = re.compile(r"[a-z0-9]+")


def _words_in_order(text: str) -> list[str]:
    folded = _ascii_fold(text or "").lower()
    return [w for w in _WORD.findall(folded) if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]


def party_runs(query: str, parties: Iterable[str]) -> set[str]:
    """
    The question's words that stand in a run of two or more which is also a
    run of one party's name: "the community bankers trust merger agreement"
    and the party "Community Bankers Trust Corporation" give "community",
    "bankers", "trust", although the retrieved text writes each in lower case.
    """
    pairs: set[tuple[str, str]] = set()
    for party in parties:
        words = _words_in_order(str(party))
        pairs.update(zip(words, words[1:]))
    seq = _words_in_order(query)
    return {w for a, b in zip(seq, seq[1:]) if (a, b) in pairs for w in (a, b)}


def written_case(texts: Iterable[str]) -> tuple[set[str], set[str]]:
    """
    How retrieved texts write their words: (the words they write in lower
    case, the 2-8-letter words they write in capitals), accents folded. A
    lower-case compound counts by its pieces and joined, as name_tokens reads
    it: "co-branding" -> "co", "branding", "cobranding".
    """
    lower: set[str] = set()
    capitals: set[str] = set()
    for text in texts:
        folded = _ascii_fold(text or "")
        for word in set(_LOWER_WORD.findall(folded)):
            pieces = re.split(r"[\-'’]", word)
            lower.update(p for p in pieces if len(p) > 1)
            if len(pieces) > 1:
                lower.add("".join(pieces))
        capitals.update(w.lower() for w in _CAPITALS_WORD.findall(folded))
    return lower, capitals


def name_document(
    query: str,
    payloads: Iterable[dict],
    *,
    margin: float = NAMED_MARGIN,
    min_evidence: float = MIN_EVIDENCE,
) -> NamedDocument:
    """
    The document among the retrieved ones that the question names.

    Args:
        query: The query or sub-question.
        payloads: Payloads of the first search's dense and sparse results.
        margin: Lead over the next document's score, as a fraction of it.
        min_evidence: Least score that names a document.

    Returns:
        NamedDocument; `document` is None when nothing is named.
    """
    if not named_document_enabled():
        return NamedDocument(reason="off")
    texts: dict[str, list[str]] = {}
    identity: dict[str, set[str]] = {}
    acronyms: dict[str, set[str]] = {}
    parties: dict[str, list] = {}
    files: dict[str, str] = {}
    for payload in payloads:
        doc = payload.get("doc_id")
        if not doc:
            continue
        texts.setdefault(doc, []).append(payload.get("text") or "")
        files.setdefault(doc, payload.get("source_file") or "")
        if doc not in identity and (payload.get("doc_parties") or payload.get("doc_title")):
            identity[doc] = identity_terms(payload)
            acronyms[doc] = identity_acronyms(payload)
            parties[doc] = list(payload.get("doc_parties") or [])
    outcome = NamedDocument(documents=len(texts))
    if len(texts) < 2:
        outcome.reason = "one document"
        return outcome
    if not identity:
        outcome.reason = "no identity"
        return outcome

    if caseless(query):
        # Nothing in the question's case tells a name from a word: "does the
        # nda with northwind ...", or ALL CAPS, where every word has capitals.
        # The retrieved text tells them apart instead. A word it writes in
        # lower case ("state", "general", "document") is an ordinary word
        # there, whatever else it is; one it never does ("northwind",
        # "bankshares") is read as the name it is in the identity, and so is
        # a run of words that is a run of a party's name ("community bankers
        # trust"). A short form needs the text to write it in capitals as
        # well ("NDA"). Only the words that could match are looked up.
        tokens = name_tokens(query)
        known = set().union(*identity.values())
        short = set().union(*acronyms.values())
        words, capitals = set(), set()
        if tokens & (known | short):
            lower, written_capitals = written_case(t for parts in texts.values() for t in parts)
            words = (tokens & known) - lower
            words |= party_runs(query, (p for ps in parties.values() for p in ps)) & known
            capitals = ((tokens & short) - lower) & written_capitals
    else:
        # A question names a document with proper nouns: "Northwind", "Atlas",
        # "Kentucky Bancshares". A word it writes in lower case ("does the
        # document state that ...", "considered") is not a name, however rarely
        # the retrieved text uses it.
        words = set().union(*(name_tokens(w) for w in _CAPITALISED.findall(query)))
        # A title's short form counts only as the question writes it, in capitals:
        # "NDA" names the non-disclosure agreement, "app" is not "Agreement for
        # Prospective Purchasers".
        capitals = {m.lower() for m in _CAPITALS.findall(query)}
    matches = {doc: (words & identity[doc]) | (capitals & acronyms[doc]) for doc in identity}
    if not any(matches.values()):
        outcome.reason = "no match"
        return outcome
    # Every term that can match is weighed: the question's words and its
    # capitals. A short form that is also a legal form ("SPA" for a Share
    # Purchase Agreement, as for an S.p.A.) is no word of name_tokens' and
    # counts through the capitals alone. The retrieved text is read only now,
    # and only for those terms.
    terms = words | capitals
    text = {doc: name_tokens("\n".join(parts)) & terms for doc, parts in texts.items()}
    n = len(text)
    df = {w: sum(1 for d in text if w in text[d] or w in identity.get(d, ()) or w in acronyms.get(d, ()))
          for w in terms}
    weight = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items() if c}
    scores = sorted(((sum(weight[w] for w in m), doc) for doc, m in matches.items()), reverse=True)
    (best, doc), runner_up = scores[0], (scores[1][0] if len(scores) > 1 else 0.0)
    matched = sorted(matches[doc])
    outcome.score, outcome.runner_up, outcome.matched = round(best, 3), round(runner_up, 3), matched
    if best < min_evidence:
        outcome.reason = "no match"
    elif best < (1 + margin) * runner_up:
        outcome.reason = "ambiguous"
    else:
        outcome.document, outcome.source_file, outcome.reason = doc, files.get(doc, ""), "named"
    return outcome
