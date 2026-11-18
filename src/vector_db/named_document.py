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

from src.data_processing.document_identity import name_tokens, title_acronyms

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
    outcome = NamedDocument(documents=len(texts))
    if len(texts) < 2:
        outcome.reason = "one document"
        return outcome
    if not identity:
        outcome.reason = "no identity"
        return outcome

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
