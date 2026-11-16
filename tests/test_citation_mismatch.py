"""
Tests for the citation check (claim_checker._check_citations, Decision 78).

Numeric grounding finds a figure anywhere in the retrieved context. These
tests pin what is added on top: a figure must be stated by a document its
claim cites. A figure stated only outside the cited documents is a citation
mismatch — named on the claim and in the flags, "passed" becomes "warning",
and nothing else changes (no failure, no revision, no change to the claim's
own verdict). The same document in another chunk, page or table sibling, and
any one of several documents stating the value, are the right source.
"""

import pytest

from src.verification.claim_checker import revision_feedback, verify_answer
from src.verification.claims import cited_documents
from src.verification.numeric_grounding import ContextIndex, ground_texts

MERGER = "merger_agreement_v2_final.txt"
MEMO = "integration_memo.txt"
DECK = "board_deck_strategic_review_mar2024.txt"
FIN = "aurora_financials_fy2023.txt"

MERGER_TEXT = (
    "Section 8.2 Indemnification Cap. The aggregate liability of the Company for "
    "breaches of representations shall not exceed $69.6 million, being 10% of the "
    "aggregate Merger Consideration."
)
# A planted document: the figure an attacker wants repeated, stated next to the
# words a cap claim would use, so that grounding binds it.
PLANTED_TEXT = (
    "Integration note. The indemnification cap for breaches of representations is "
    "$500.0 million under the merger agreement."
)
DECK_TEXT = (
    "SLIDE 3: PRELIMINARY INDICATIONS OF INTEREST. Vertex Capital Partners offered "
    "$55 - $60 per share in cash, with committed financing."
)
MERGER_PRICE_TEXT = (
    "Section 1.2 Merger Consideration. Each share shall be converted into the right "
    "to receive $58.00 per share in cash. The aggregate Merger Consideration is "
    "approximately $696.0 million."
)

# One table split across two chunks of one document, as the chunker does for a
# long table: the header and first rows in one, the rest in the other.
TABLE_HEAD = """CONSOLIDATED INCOME STATEMENT (in millions of USD)
                                          FY2023          FY2022
Revenue                                   $452.8          $387.1
Gross Profit                              $271.7          $228.8
"""
TABLE_TAIL = """CONSOLIDATED INCOME STATEMENT (continued)
                                          FY2023          FY2022
Operating Income                           $67.9           $51.5
Net Income                                 $46.4           $35.0
"""


def chunk(source: str, text: str, section: str = "", page=None, chunk_id: str = "") -> dict:
    return {
        "chunk_id": chunk_id or f"{source}-{section or 'body'}-{page}",
        "source_file": source,
        "text": text,
        "section_heading": section,
        "page_number": page,
    }


async def supports_everything(query, claims, documents):
    """A judge that upholds every claim: whatever is left is the citation check's."""
    return {c["id"]: {"status": "supported", "reason": "stated"} for c in claims}, "stub-judge"


async def verify(answer: str, context: list[dict]):
    return await verify_answer(answer, context, "q", judge=supports_everything, use_nli=False)


class TestLaundering:
    @pytest.mark.asyncio
    async def test_a_planted_figure_cited_to_a_clean_document_is_named(self):
        context = [chunk(MERGER, MERGER_TEXT, "Section 8.2"), chunk(MEMO, PLANTED_TEXT)]
        answer = f"The indemnification cap for breaches of representations is $500.0 million [📄 {MERGER} | Section 8.2]."
        report = await verify(answer, context)

        check = report.claim_checks[0]
        assert check["citation_mismatch"] == [{"figure": "$500.0 million", "found_in": [MEMO]}]
        assert check["cited_documents"] == [MERGER]
        assert report.validation_status == "warning"
        assert any(
            f.startswith("Citation mismatch: $500.0 million is stated in integration_memo.txt, "
                         f"not in the cited {MERGER}")
            for f in report.flags
        )
        assert report.citation_counts["figures_citation_mismatch"] == 1

    @pytest.mark.asyncio
    async def test_the_same_figure_cited_to_the_document_that_states_it_passes(self):
        context = [chunk(MERGER, MERGER_TEXT, "Section 8.2"), chunk(MEMO, PLANTED_TEXT)]
        answer = f"The indemnification cap for breaches of representations is $500.0 million [📄 {MEMO} | Integration note]."
        report = await verify(answer, context)
        assert "citation_mismatch" not in report.claim_checks[0]
        assert report.validation_status == "passed"

    @pytest.mark.asyncio
    async def test_a_mismatch_never_fails_an_answer_or_asks_for_a_revision(self):
        context = [chunk(MERGER, MERGER_TEXT, "Section 8.2"), chunk(MEMO, PLANTED_TEXT)]
        answer = f"The indemnification cap for breaches of representations is $500.0 million [📄 {MERGER} | Section 8.2]."
        report = await verify(answer, context)
        check = report.claim_checks[0]
        # The claim's own verdict is untouched: the figure is in the documents.
        assert check["status"] == "supported"
        assert report.validation_status != "failed"
        assert revision_feedback(report.claim_checks, report.flags) == []

    @pytest.mark.asyncio
    async def test_a_failed_answer_stays_failed_and_keeps_both_findings(self):
        context = [chunk(MERGER, MERGER_TEXT, "Section 8.2"), chunk(MEMO, PLANTED_TEXT)]
        answer = (
            f"The indemnification cap for breaches of representations is $500.0 million [📄 {MERGER} | Section 8.2]. "
            f"The basket is $812.3 million [📄 {MERGER} | Section 8.2]."
        )
        report = await verify(answer, context)
        assert report.validation_status == "failed"
        assert any(f.startswith("Citation mismatch") for f in report.flags)
        assert any(f.startswith("Figure not found") for f in report.flags)

    @pytest.mark.asyncio
    async def test_a_sentence_inheriting_a_clean_marker_is_checked_too(self):
        # "Sentence one. Sentence two [source]." — the marker covers both.
        context = [chunk(MERGER, MERGER_TEXT, "Section 8.2"), chunk(MEMO, PLANTED_TEXT)]
        answer = (
            "The indemnification cap for breaches of representations is $500.0 million. "
            f"Liability for breaches is capped under the agreement [📄 {MERGER} | Section 8.2]."
        )
        report = await verify(answer, context)
        assert report.claim_checks[0]["citation_mismatch"][0]["figure"] == "$500.0 million"
        assert report.validation_status == "warning"

    @pytest.mark.asyncio
    async def test_a_marker_naming_no_retrieved_document_does_not_vouch(self):
        context = [chunk(MERGER, MERGER_TEXT, "Section 8.2"), chunk(MEMO, PLANTED_TEXT)]
        answer = (
            "The indemnification cap for breaches of representations is $500.0 million "
            "[📄 audited_financials_fy2023.pdf | p.4]."
        )
        report = await verify(answer, context)
        check = report.claim_checks[0]
        assert check["cited_documents"] == []
        assert report.validation_status == "warning"
        assert any("not in any retrieved document its citation names" in f for f in report.flags)


class TestRightSourceElsewhereInIt:
    @pytest.mark.asyncio
    async def test_a_table_sibling_chunk_of_the_cited_document_is_the_right_source(self):
        # Cited by the section of the first chunk; the figure sits in the second.
        context = [
            chunk(FIN, TABLE_HEAD, "CONSOLIDATED INCOME STATEMENT", chunk_id="fin-1"),
            chunk(FIN, TABLE_TAIL, "CONSOLIDATED INCOME STATEMENT (continued)", chunk_id="fin-2"),
        ]
        answer = f"Net Income for FY2023 was $46.4 million [📄 {FIN} | CONSOLIDATED INCOME STATEMENT]."
        report = await verify(answer, context)
        assert "citation_mismatch" not in report.claim_checks[0]
        assert report.validation_status == "passed"
        assert report.citation_counts["figures_in_cited_documents"] == 1

    @pytest.mark.asyncio
    async def test_another_page_of_the_cited_document_is_the_right_source(self):
        context = [
            chunk(FIN, TABLE_HEAD, page=4, chunk_id="fin-p4"),
            chunk(FIN, TABLE_TAIL, page=5, chunk_id="fin-p5"),
        ]
        answer = f"Net Income for FY2023 was $46.4 million [📄 {FIN} | p.4]."
        report = await verify(answer, context)
        assert "citation_mismatch" not in report.claim_checks[0]
        assert report.validation_status == "passed"

    @pytest.mark.asyncio
    async def test_a_figure_two_documents_state_is_right_cited_to_either(self):
        # $58.00 is in the merger agreement and in the deck's summary line; the
        # grounding binds it in the merger agreement, the claim cites the deck.
        deck = DECK_TEXT + " The Board accepted Vertex's final price of $58.00 per share."
        context = [chunk(MERGER, MERGER_PRICE_TEXT, "Section 1.2"), chunk(DECK, deck, "SLIDE 3")]
        for cited in (DECK, MERGER):
            answer = f"Each share converts into the right to receive $58.00 per share in cash [📄 {cited} | x]."
            report = await verify(answer, context)
            assert "citation_mismatch" not in report.claim_checks[0], cited
            assert report.validation_status == "passed"

    @pytest.mark.asyncio
    async def test_a_claim_citing_two_documents_is_right_for_figures_of_both(self):
        context = [chunk(MERGER, MERGER_PRICE_TEXT, "Section 1.2"), chunk(DECK, DECK_TEXT, "SLIDE 3")]
        answer = (
            "The $58.00 per share price sits inside Vertex's $55 - $60 per share indication "
            f"[📄 {MERGER} | Section 1.2] [📄 {DECK} | SLIDE 3]."
        )
        report = await verify(answer, context)
        assert "citation_mismatch" not in report.claim_checks[0]

    @pytest.mark.asyncio
    async def test_one_marker_for_figures_of_two_documents_names_the_other(self):
        # The live shape behind most mismatches on the replay (pm:comp_02).
        context = [chunk(MERGER, MERGER_PRICE_TEXT, "Section 1.2"), chunk(DECK, DECK_TEXT, "SLIDE 3")]
        answer = (
            "The $58.00 per share price sits inside Vertex's $55 - $60 per share indication "
            f"[📄 {DECK} | SLIDE 3]."
        )
        report = await verify(answer, context)
        mismatch = report.claim_checks[0]["citation_mismatch"]
        assert [m["figure"] for m in mismatch] == ["$58.00"]
        assert mismatch[0]["found_in"] == [MERGER]


class TestWhatIsNotPenalised:
    @pytest.mark.asyncio
    async def test_a_claim_without_any_marker_is_counted_not_flagged(self):
        context = [chunk(MERGER, MERGER_TEXT, "Section 8.2"), chunk(MEMO, PLANTED_TEXT)]
        answer = "The indemnification cap for breaches of representations is $500.0 million."
        report = await verify(answer, context)
        assert "citation_mismatch" not in report.claim_checks[0]
        assert report.citation_counts["claims_without_citation"] == 1
        assert report.citation_counts["figures_without_citation"] == 1
        assert not any(f.startswith("Citation mismatch") for f in report.flags)
        assert report.validation_status == "passed"

    @pytest.mark.asyncio
    async def test_a_quantity_is_not_held_to_its_citation(self):
        context = [
            chunk(MERGER, "Section 8.1 Survival. Representations survive for 18 months.", "Section 8.1"),
            chunk(MEMO, "Integration note. Retention awards vest over 24 months."),
        ]
        answer = f"Retention awards vest over 24 months [📄 {MERGER} | Section 8.1]."
        report = await verify(answer, context)
        assert "citation_mismatch" not in report.claim_checks[0]
        assert report.citation_counts.get("figures", 0) == 0

    @pytest.mark.asyncio
    async def test_a_figure_the_answer_computes_is_not_held_to_a_location(self):
        context = [chunk(FIN, TABLE_HEAD, "CONSOLIDATED INCOME STATEMENT")]
        answer = (
            "Revenue grew from $387.1 million in FY2022 to $452.8 million in FY2023, "
            f"an increase of $65.7 million [📄 {FIN} | CONSOLIDATED INCOME STATEMENT]."
        )
        report = await verify(answer, context)
        assert "citation_mismatch" not in report.claim_checks[0]
        assert report.citation_counts["figures"] == 2

    @pytest.mark.asyncio
    async def test_a_computed_figure_that_also_occurs_elsewhere_is_still_computed(self):
        # Verification benchmark f070.3: 60% = $3.9M / $6.5M, and "60%" also
        # sits in the board deck ("60% cash / 40% stock"). The figure rests on
        # the cited document's two amounts, not on the deck.
        retention = (
            "SECTION 3: RETENTION POOL. The Board approved a retention pool of $6,500,000 on "
            "March 1, 2024. Engineering (41 employees) receives $3,900,000 of the retention pool."
        )
        deck = "SLIDE 3. Meridian Software Corp offered 60% cash / 40% stock consideration."
        source = "employment_and_retention_agreements.txt"
        context = [chunk(source, retention, "SECTION 3"), chunk(DECK, deck, "SLIDE 3")]
        answer = (
            "Engineering's $3,900,000 allocation is 60% of the $6,500,000 retention pool "
            f"[📄 {source} | SECTION 3: RETENTION POOL]."
        )
        report = await verify(answer, context)
        assert "citation_mismatch" not in report.claim_checks[0]
        assert report.citation_counts["figures"] == 2  # the two amounts; 60% is computed


class TestNamingDocuments:
    def test_a_marker_names_a_document_by_its_file_name_or_stem(self):
        context = [chunk(FIN, "x"), chunk(MERGER, "y"), chunk("financials.txt", "z")]
        assert cited_documents([f"📄 {FIN} | p.4"], context) == [FIN]
        # The stem "financials" inside "aurora_financials_fy2023" is not a
        # citation of financials.txt, and the reverse holds too.
        assert cited_documents(["📄 aurora_financials_fy2023 | p.4"], context) == [FIN]
        assert cited_documents(["📄 financials | p.4"], context) == ["financials.txt"]
        assert cited_documents(["📄 Section 8.2 | Indemnification"], context) == []

    def test_every_location_is_kept_internally_and_none_is_serialised(self):
        context = [
            chunk(MERGER, MERGER_PRICE_TEXT),
            chunk(DECK, DECK_TEXT + " Final price $58.00 per share."),
        ]
        [[check]] = ground_texts(["The price is $58.00 per share."], ContextIndex(context))
        assert set(check.evidence_files) == {MERGER, DECK}
        assert "evidence_files" not in check.to_dict()


class TestValidatorTrace:
    @pytest.mark.asyncio
    async def test_the_trace_carries_counts_and_each_mismatch(self, monkeypatch):
        import src.agents.hallucination_validator as hv

        async def no_judge(**kwargs):
            raise AssertionError("a figure-confirmed claim needs no judge")

        monkeypatch.setenv("VERIFY_NLI", "0")
        monkeypatch.setattr(hv, "call_verification_agent_with_model", no_judge)
        out = await hv.hallucination_validator_node({
            "generated_answer": (
                "The indemnification cap for breaches of representations is $500.0 million "
                f"[📄 {MERGER} | Section 8.2]."
            ),
            "expanded_context": [chunk(MERGER, MERGER_TEXT, "Section 8.2"), chunk(MEMO, PLANTED_TEXT)],
            "original_query": "What is the cap?",
            "agent_trace": [],
        })
        assert out["validation_status"] == "warning"
        trace = out["agent_trace"][0]
        assert trace["citation_counts"]["figures_citation_mismatch"] == 1
        assert trace["citation_mismatches"][0]["found_in"] == [MEMO]
        assert trace["citation_mismatches"][0]["cited"] == [MERGER]
        assert trace["claim_checks"][0]["citation_mismatch"] is True
        assert out["claim_checks"][0]["citation_mismatch"][0]["figure"] == "$500.0 million"
