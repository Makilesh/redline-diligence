"""
Named-document focus without capital letters (Decision 91): a question typed
in lower case, or in capitals, names its agreement by the words the retrieved
documents never write in lower case, and by runs of a party's name — and a
word they do write in lower case ("state", "general") names nothing however
rare it is among them.

Offline: payloads only, no search.
"""

from __future__ import annotations

import pytest

from src.vector_db.named_document import caseless, name_document, party_runs, written_case

NDA_TEXT = ("The Receiving Party shall hold the Confidential Information in confidence, as the laws of the state "
            "of New York require, and shall not disclose it in general or to third parties.")


def _payload(chunk: str, doc: str, text: str = NDA_TEXT, **identity) -> dict:
    return {"chunk_id": chunk, "doc_id": doc, "source_file": f"{doc}.txt", "text": text, **identity}


ROOM = [
    _payload("a1", "A", doc_title="MUTUAL NON-DISCLOSURE AGREEMENT", doc_parties=["Harbor Analytics"]),
    _payload("b1", "B", doc_title="NON-DISCLOSURE AGREEMENT", doc_parties=["Gamma Systems"]),
    _payload("c1", "C", doc_title="CONFIDENTIALITY AGREEMENT", doc_parties=["Delta Logistics, Inc."]),
    _payload("d1", "D", doc_title="NON-DISCLOSURE AGREEMENT", doc_parties=["Epsilon Foods"]),
    _payload("s1", "S", text="Notices go to the escrow agent at its office.", doc_title="Escrow Agreement",
             doc_parties=["State Street Bank"]),
    _payload("g1", "G", text="The supplier ships the goods.", doc_title="Supply Agreement",
             doc_parties=["General Freight"]),
]


class TestCaseless:
    @pytest.mark.parametrize("query", [
        "does the nda with gamma systems allow disclosure?",
        "Does the nda with gamma systems allow disclosure?",           # a capital only where a sentence starts
        "consider the agreement; does it bind gamma systems? i think not.",
        "DOES THE NDA WITH GAMMA SYSTEMS ALLOW DISCLOSURE?",
    ])
    def test_capitals_that_mark_no_name(self, query):
        assert caseless(query)

    @pytest.mark.parametrize("query", [
        "Does the NDA with Gamma Systems allow disclosure?",
        "does the nda with Gamma allow disclosure?",
        'Consider "Fiverr"\'s privacy policy; is my data sold?',
        "Consider the Merger Agreement between 'UNITED BANKSHARES, INC.' and 'COMMUNITY BANKERS TRUST'; what?",
    ])
    def test_a_capital_that_marks_a_name(self, query):
        assert not caseless(query)


class TestWrittenCase:
    def test_lower_case_words_and_capital_short_forms(self):
        lower, capitals = written_case(["The NDA binds the parties; co-branding is excluded. See www.northwind.com, "
                                        "write to legal@atlas.com, e.g. for the SPA."])
        assert {"the", "binds", "parties", "co", "branding", "cobranding", "is", "excluded", "write", "to"} <= lower
        assert "see" not in lower                       # written "See"
        # A URL, an e-mail address and a dotted abbreviation are not words written in lower case.
        assert not {"northwind", "www", "com", "atlas", "legal", "eg"} & lower
        assert {"nda", "spa"} <= capitals and "see" not in capitals

    def test_accents_are_folded(self):
        lower, _ = written_case(["the société générale"])
        assert {"societe", "generale"} <= lower


class TestPartyRuns:
    def test_a_run_of_a_party_name(self):
        parties = ["Community Bankers Trust Corporation", "United Bankshares, Inc."]
        assert party_runs("the community bankers trust merger agreement", parties) == {"community", "bankers", "trust"}
        assert party_runs("THE UNITED BANKSHARES MERGER", parties) == {"united", "bankshares"}

    def test_words_out_of_order_or_alone_are_no_run(self):
        parties = ["Community Bankers Trust Corporation"]
        assert party_runs("trust the bankers of the community", parties) == set()
        assert party_runs("the community agreement", parties) == set()

    def test_a_run_lies_within_one_party(self):
        # "community" ends one party and "trust" starts another: not a run of either.
        assert party_runs("the community trust", ["Health Community", "Trust Bank"]) == set()


class TestNameDocumentWithoutCapitals:
    @pytest.mark.parametrize("query", [
        "consider the non-disclosure agreement between gamma systems and buyer; may the receiving party share "
        "confidential information?",
        "CONSIDER THE NON-DISCLOSURE AGREEMENT BETWEEN GAMMA SYSTEMS AND BUYER; MAY THE RECEIVING PARTY SHARE "
        "CONFIDENTIAL INFORMATION?",
        "Does the nda with gamma systems allow disclosure?",
    ])
    def test_the_party_a_caseless_question_names(self, query):
        named = name_document(query, ROOM)
        assert named.document == "B" and "gamma" in named.matched

    def test_a_word_the_retrieved_text_writes_in_lower_case_names_nothing(self):
        # "state" and "general" are words of two parties' names, and words the
        # retrieved text uses in lower case: in a caseless question they are
        # those words, not the names.
        assert name_document("does the document state that copies must be returned?", ROOM).document is None
        assert name_document("DOES THE DOCUMENT STATE THAT COPIES MUST BE RETURNED?", ROOM).document is None
        assert name_document("is the general rule that notice is written?", ROOM).document is None
        # The whole name still names it.
        assert name_document("does the state street escrow allow early release?", ROOM).document == "S"
        assert name_document("when does general freight ship?", ROOM).document == "G"

    def test_a_name_made_of_ordinary_words_names_by_its_run(self):
        # Another agreement writes "community", "bankers" and "trust" in lower
        # case, so word by word they name nothing; in a row they are the party.
        room = [
            _payload("m1", "M", text="The Bank Holding Company Act governs the merger.",
                     doc_title="AGREEMENT AND PLAN OF MERGER",
                     doc_parties=["United Bankshares, Inc.", "Community Bankers Trust Corporation"]),
            _payload("p1", "P", text="The community bank holds deposits in trust for its bankers.",
                     doc_title="AGREEMENT AND PLAN OF MERGER", doc_parties=["Pinnacle Bank"]),
            *[_payload(f"{k}1", k, text="The parties shall cooperate.", doc_title="AGREEMENT AND PLAN OF MERGER",
                       doc_parties=[p]) for k, p in (("Q", "Quorum Holdings"), ("R", "Riverview Bancorp"))],
        ]
        named = name_document("consider the community bankers trust merger agreement; what is the termination fee?",
                              room)
        assert named.document == "M" and {"community", "bankers", "trust"} <= set(named.matched)
        # Out of order the same words are the ordinary words the text uses.
        assert name_document("is the trust of the bankers in the community limited?", room).document is None

    def test_a_run_of_a_title_is_no_name(self):
        # Titles are the phrases a question's topic repeats ("intellectual property").
        room = ROOM + [_payload("i1", "I", text="The intellectual property stays with the licensor.",
                                doc_title="Intellectual Property Agreement", doc_parties=["Orbit Labs"])]
        assert name_document("how is intellectual property ownership assigned?", room).document is None

    def test_a_caseless_short_form_needs_the_text_to_write_it_in_capitals(self):
        room = [
            _payload("n1", "N", text="Notice of termination ends the obligations.",
                     doc_title="Non-Disclosure Agreement", doc_parties=["Northwind Traders"]),
            _payload("m1", "M", text="Either party may terminate on notice.", doc_title="Master Services Agreement",
                     doc_parties=["Northwind Traders"]),
            *[_payload(f"{k}1", k, text="Termination on ninety days notice.", doc_title=f"{t} Agreement",
                       doc_parties=[p]) for k, t, p in (("L", "Lease", "Harbor Realty"),
                                                         ("S", "Supply", "Gamma Systems"),
                                                         ("K", "Credit", "Delta Bank"))],
        ]
        # No retrieved text writes "NDA": "nda" may be any word, and Northwind has two agreements.
        assert name_document("does the northwind nda allow sharing with advisers?", room).document is None
        room[0] = _payload("n1", "N", text="This NDA ends on notice of termination.",
                           doc_title="Non-Disclosure Agreement", doc_parties=["Northwind Traders"])
        assert name_document("does the northwind nda allow sharing with advisers?", room).document == "N"
        assert name_document("DOES THE NORTHWIND NDA ALLOW SHARING WITH ADVISERS?", room).document == "N"

    def test_an_ordinary_word_is_no_short_form(self):
        # "Agreement for Prospective Purchasers" -> "app"; the privacy policies write "app" in lower case.
        room = ROOM + [
            _payload("p1", "P", text="PURCHASERS KEEP INFORMATION CONFIDENTIAL. THE APP IS LICENSED.",
                     doc_title="Agreement for Prospective Purchasers", doc_parties=["Vendor Agent"]),
            _payload("q1", "Q", text="Our app collects your location when the app is open.",
                     doc_title="Privacy Policy"),
        ]
        assert name_document("does this app sell customer information?", room).document is None
        assert name_document("DOES THIS APP SELL CUSTOMER INFORMATION?", room).document is None

    def test_a_question_that_names_nothing(self):
        for query in ("may the receiving party share confidential information?",
                      "MAY THE RECEIVING PARTY SHARE CONFIDENTIAL INFORMATION?", "is my information secure"):
            assert name_document(query, ROOM).document is None

    def test_lower_case_and_capitals_name_alike(self):
        query = "the nda between gamma systems and buyer: may advisers see it?"
        lower, upper = name_document(query.lower(), ROOM), name_document(query.upper(), ROOM)
        assert (lower.document, lower.matched, lower.score) == (upper.document, upper.matched, upper.score)

    def test_a_capitalised_question_is_read_as_before(self):
        # Capitals mark the names: lower-case words are not read as names.
        room = ROOM + [_payload("w1", "W", text="Shipping terms apply.", doc_title="Shipping Agreement",
                                doc_parties=["Walabot Logistics"])]
        assert name_document("Does the Gamma Systems NDA cover walabot data?", room).document == "B"
