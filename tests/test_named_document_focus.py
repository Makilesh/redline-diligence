"""
Named-document focus (Decision 86): document focus searches the document a
question names, by the identity its opening states, instead of the one the
fused list leans to — and keeps the fused choice whenever nothing is named.

Offline: Qdrant, the embedder and the cross-encoder are stubbed.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.agents.retrieval_executor as rx
from src.agents.retrieval_strategy import RETRIEVAL_CONFIGS
from src.vector_db.named_document import identity_acronyms, identity_terms, name_document
from tests.conftest import make_scored_point

NDA_TEXT = ("The Receiving Party shall hold the Confidential Information in confidence and shall not disclose "
            "Confidential Information to third parties.")


def _payload(chunk: str, doc: str, text: str = NDA_TEXT, **identity) -> dict:
    return {"chunk_id": chunk, "doc_id": doc, "source_file": f"{doc}.txt", "text": text, **identity}


# Five near-identical NDAs; the question names one by its party.
ROOM = [
    _payload("a1", "A", doc_title="MUTUAL NON-DISCLOSURE AGREEMENT", doc_parties=["Harbor Analytics"]),
    _payload("a2", "A", doc_title="MUTUAL NON-DISCLOSURE AGREEMENT", doc_parties=["Harbor Analytics"]),
    _payload("b1", "B", doc_title="NON-DISCLOSURE AGREEMENT", doc_parties=["Gamma Systems"]),
    _payload("c1", "C", doc_title="CONFIDENTIALITY AGREEMENT", doc_parties=["Delta Logistics, Inc."]),
    _payload("d1", "D", doc_title="NON-DISCLOSURE AGREEMENT", doc_parties=["Epsilon Foods"]),
    _payload("e1", "E", doc_title="NON-DISCLOSURE AGREEMENT", doc_parties=["Zeta Mining plc"]),
]


class TestNameDocument:
    def test_the_party_a_question_names(self):
        named = name_document("Consider the Non-Disclosure Agreement between Gamma Systems and Buyer; "
                              "may the Receiving Party share Confidential Information?", ROOM)
        assert named.document == "B" and named.reason == "named"
        assert "gamma" in named.matched

    def test_a_short_form_of_the_title_counts_when_the_party_alone_is_shared(self):
        # A data room: one counterparty, two agreements, and other agreements.
        room = [
            _payload("n1", "N", text="Notice of termination ends the obligations.",
                     doc_title="Non-Disclosure Agreement", doc_parties=["Northwind Traders"]),
            _payload("m1", "M", text="Either party may terminate on notice.", doc_title="Master Services Agreement",
                     doc_parties=["Northwind Traders"]),
            *[_payload(f"{k}1", k, text="Termination on ninety days notice.", doc_title=f"{t} Agreement",
                       doc_parties=[p]) for k, t, p in (("L", "Lease", "Harbor Realty"),
                                                         ("S", "Supply", "Gamma Systems"),
                                                         ("K", "Credit", "Delta Bank"),
                                                         ("E", "Employment", "Jane Roe"))],
        ]
        assert name_document("Does the Northwind NDA allow disclosure to advisers?", room).document == "N"
        assert name_document("What does Northwind's agreement say about termination?", room).document is None

    def test_a_short_form_that_is_also_a_legal_form_is_weighed(self):
        # "spa" is a stopword (S.p.A.), so it is no word of the question's;
        # written in capitals it still names the Share Purchase Agreement, and
        # weighing it must not fail.
        room = [
            _payload("p1", "P", text="The Seller shall sell the Shares to the Buyer at Closing.",
                     doc_title="SHARE PURCHASE AGREEMENT", doc_parties=["Atlas Holdings Inc.", "Borealis Ltd"]),
            _payload("n1", "N", doc_title="MUTUAL NON-DISCLOSURE AGREEMENT", doc_parties=["Cygnus Corp"]),
            _payload("m1", "M", text="The Supplier shall provide the Services.",
                     doc_title="MASTER SERVICES AGREEMENT", doc_parties=["Epsilon Inc"]),
            _payload("l1", "L", text="The Tenant shall pay the rent monthly.",
                     doc_title="LEASE AGREEMENT", doc_parties=["Harbor Realty"]),
        ]
        named = name_document("What is the termination fee in the SPA?", room)
        assert named.document == "P" and named.matched == ["spa"]
        named = name_document("What is the termination fee in the SPA with Atlas?", room)
        assert named.document == "P" and named.matched == ["atlas", "spa"]
        # Two share purchase agreements: the short form alone names neither.
        twin = room + [_payload("q1", "Q", text="The Seller shall sell the Shares.",
                                doc_title="SHARE PURCHASE AGREEMENT", doc_parties=["Corvid Partners"])]
        assert name_document("What is the termination fee in the SPA?", twin).document is None

    def test_a_topic_word_in_a_title_names_nothing(self):
        # Every retrieved NDA speaks of confidential information, so a title
        # holding those words is not named by a question about them.
        room = ROOM + [_payload("x1", "X", doc_title="CONFIDENTIAL INFORMATION AGREEMENT",
                                doc_parties=["Receiving Party"])]
        named = name_document("Does the document require Confidential Information to be marked?", room)
        assert named.document is None

    def test_two_documents_named_as_strongly_is_a_question_about_both(self):
        named = name_document("Compare the agreements of Gamma Systems and Epsilon Foods", ROOM)
        assert named.document is None and named.reason == "ambiguous"

    @pytest.mark.parametrize("query, reason", [
        ("May the Receiving Party keep copies?", "no match"),
    ])
    def test_nothing_named(self, query, reason):
        assert name_document(query, ROOM).reason == reason

    def test_an_index_without_identity_keeps_the_fused_choice(self):
        bare = [{k: v for k, v in p.items() if k not in ("doc_title", "doc_parties")} for p in ROOM]
        named = name_document("the Non-Disclosure Agreement between Gamma Systems and Buyer", bare)
        assert named.document is None and named.reason == "no identity"

    def test_one_document_retrieved(self):
        assert name_document("Gamma Systems", ROOM[2:3]).reason == "one document"

    def test_switch_off(self, monkeypatch):
        monkeypatch.setenv("NAMED_DOCUMENT_FOCUS", "0")
        assert name_document("the agreement of Gamma Systems", ROOM).reason == "off"

    def test_identity_terms(self):
        payload = {"doc_title": "Share Purchase Agreement", "doc_parties": ["Atlas Holdings, Inc."]}
        assert {"atlas", "share", "purchase"} <= identity_terms(payload)
        assert not {"inc", "holdings", "spa"} & identity_terms(payload)
        assert identity_acronyms(payload) == {"spa"}

    def test_a_word_in_lower_case_is_no_name(self):
        room = ROOM + [_payload("s1", "S", text="Notices go to the bank under the laws of the state of New York.",
                                doc_title="Escrow Agreement", doc_parties=["State Street Bank"])]
        # A question that capitalises names: its lower-case words are not names.
        assert name_document("Does the NDA state that copies must be returned?", room).document is None
        # A caseless one (Decision 91): the retrieved text writes "state" in lower case, so it is that word.
        assert name_document("Does the document state that copies must be returned?", room).document is None
        assert name_document("Does the State Street escrow allow early release?", room).document == "S"

    def test_a_short_form_counts_only_in_capitals(self):
        # "Agreement for Prospective Purchasers" -> "app": an ordinary word.
        room = ROOM + [_payload("p1", "P", text="Purchasers keep information confidential.",
                                doc_title="Agreement for Prospective Purchasers", doc_parties=["Vendor Agent"])]
        assert name_document("Does this app sell customer information?", room).document is None
        assert name_document("Does the APP require the purchasers to return documents?", room).document == "P"


# ---------------------------------------------------------------------------
# Through the retrieval executor
# ---------------------------------------------------------------------------

def _point(chunk: str, doc: str, text: str = NDA_TEXT, **identity):
    point = make_scored_point(chunk, 0.5)
    point.payload.update(_payload(chunk, doc, text, **identity))
    return point


class _Stubs:
    def __init__(self, monkeypatch, corpus_wide, in_document):
        self.focus_calls: list[dict] = []
        self.reranked: list[str] = []

        async def embed(texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

        async def search(**kwargs):
            return corpus_wide

        async def dense(**kwargs):
            self.focus_calls.append(kwargs)
            return in_document.get(kwargs["doc_id"], [])[: kwargs["top_k"]]

        async def rerank(pairs):
            self.reranked = [p for _, p in pairs]
            return np.asarray([0.9 if "answer" in p else 0.1 for _, p in pairs], dtype=np.float32)

        async def expand(**kwargs):
            return list(kwargs["chunks"])

        monkeypatch.setattr(rx, "embed_texts_async", embed)
        monkeypatch.setattr(rx, "compute_sparse_bm25", lambda text: "sparse")
        monkeypatch.setattr(rx, "hybrid_search", search)
        monkeypatch.setattr(rx, "dense_search", dense)
        monkeypatch.setattr(rx, "rerank_pairs_async", rerank)
        monkeypatch.setattr(rx, "expand_context", expand)


def _state(query: str, focus_k: int = 5) -> dict:
    return {"current_query": query, "query_type": "legal", "parsed_intent": {}, "deal_id": "deal-n",
            "include_pii": False, "rewrite_iteration": 0,
            "retrieval_config": {**RETRIEVAL_CONFIGS["legal"], "document_focus_k": focus_k},
            "extracted_filters": {}, "sub_questions": []}


def _corpus_wide(with_identity: bool = True):
    """The fused list leans to A (three chunks); B's preamble, which names it, is retrieved once."""
    ident = {"A": {"doc_title": "NON-DISCLOSURE AGREEMENT", "doc_parties": ["Harbor Analytics"]},
             "B": {"doc_title": "NON-DISCLOSURE AGREEMENT", "doc_parties": ["Gamma Systems"]},
             "C": {"doc_title": "NON-DISCLOSURE AGREEMENT", "doc_parties": ["Delta Logistics"]}}
    if not with_identity:
        ident = {k: {} for k in ident}
    dense = [_point("a1", "A", **ident["A"]), _point("a2", "A", **ident["A"]), _point("c1", "C", **ident["C"])]
    sparse = [_point("a1", "A", **ident["A"]), _point("b_pre", "B", "Gamma Systems and Buyer agree", **ident["B"]),
              _point("a3", "A", **ident["A"])]
    return dense, sparse


IN_DOCUMENT = {"A": [_point("a_clause", "A", "A's own clause")],
               "B": [_point("b_clause", "B", "B's clause: the answer")]}
QUESTION = ("Consider the Non-Disclosure Agreement between Gamma Systems and Buyer; "
            "may the Receiving Party share Confidential Information?")


@pytest.mark.asyncio
async def test_focus_searches_the_document_the_question_names(monkeypatch):
    stubs = _Stubs(monkeypatch, _corpus_wide(), IN_DOCUMENT)
    out = await rx.retrieval_executor_node(_state(QUESTION))

    assert [c["doc_id"] for c in stubs.focus_calls] == ["B"]
    assert out["reranked_results"][0]["chunk_id"] == "b_clause"
    # Nothing removed: the fused leader's chunks are all still scored.
    ids = {c["chunk_id"] for c in out["reranked_results"]}
    assert {"a1", "a2", "a3", "b_pre", "c1", "b_clause"} <= ids


@pytest.mark.asyncio
@pytest.mark.parametrize("query, with_identity", [
    ("may the Receiving Party share Confidential Information?", True),    # names no document
    ("Compare the NDAs of Gamma Systems and Delta Logistics", True),       # names two
    (QUESTION, False),                                                     # index without identity
])
async def test_the_fused_choice_stands_when_nothing_is_named(monkeypatch, query, with_identity):
    stubs = _Stubs(monkeypatch, _corpus_wide(with_identity), IN_DOCUMENT)
    await rx.retrieval_executor_node(_state(query))

    assert [c["doc_id"] for c in stubs.focus_calls] == ["A"]


@pytest.mark.asyncio
async def test_focus_off_names_nothing(monkeypatch):
    stubs = _Stubs(monkeypatch, _corpus_wide(), IN_DOCUMENT)
    await rx.retrieval_executor_node(_state(QUESTION, focus_k=0))

    assert stubs.focus_calls == []


@pytest.mark.asyncio
async def test_switched_off(monkeypatch):
    monkeypatch.setenv("NAMED_DOCUMENT_FOCUS", "0")
    stubs = _Stubs(monkeypatch, _corpus_wide(), IN_DOCUMENT)
    await rx.retrieval_executor_node(_state(QUESTION))

    assert [c["doc_id"] for c in stubs.focus_calls] == ["A"]
