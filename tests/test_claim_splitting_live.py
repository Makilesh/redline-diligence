"""
Claim splitting on the shapes live answers take (Decision 72).

Measured on the dev half of the live-answer replay (eval/run_answer_replay.py):
correct declines phrased in ways the absence pattern missed, the bare noun
phrases listed under "The data room lacks:", heading fragments and bullet
lead-ins were scored as ordinary claims and flagged "not supported". The safety
rule of claims.py is unchanged and tested here too: only wording about what the
documents disclose is a statement of absence, never a bare negation of a fact,
and a statement of absence is still checked (by the judge), never dropped.
"""

from __future__ import annotations

import pytest

from src.verification.claims import split_claims


def _absence(text: str) -> list[bool]:
    return [c.is_absence for c in split_claims(text)]


class TestDeclinesAnchoredOnDisclosure:
    DECLINES = [
        "Based on the provided documents, Aurora Technologies Inc. has not disclosed any "
        "environmental remediation liabilities for manufacturing sites.",
        "Based on the provided documents, there is no information available regarding "
        "Aurora's quarterly revenue figures for Q1 or Q2 of FY2024.",
        "Competitor Churn Data: There are no mentions of churn rates, retention rates, or any "
        "specific customer health metrics for Palantir, Snowflake, Databricks, or DataRobot.",
        "No competitor churn/retention data: There are no figures provided for the churn rates "
        "or retention rates of Palantir, Snowflake, Databricks, or DataRobot.",
        "Absence of Buyer Termination Fee: The provided agreement contains no provision "
        "requiring the Buyer to pay a reverse termination fee to the Company.",
        "While the memo confirms no notification obligations arose, it does not explicitly "
        "state if any data privacy approvals are required.",
        "Competitor Churn Rates: No churn rates or retention figures are provided for Palantir.",
        "None of the documents mention a FY2025 operating plan.",
    ]
    FACTS = [
        # Bare negations of facts stay ordinary claims, checked by the
        # entailment model like any other.
        "The Company has not sought a consent or waiver from the lending syndicate.",
        "No waiver or amendment has been requested or granted.",
        "The loss has not been reported to regulators.",
        "No customer exercised a termination right as a result.",
        "There is no evidence of any pending litigation against the Company.",
        "The key customer consent is missing from the executed agreement.",
        "No governmental investigation is pending against the Company.",
        # A term of a contract, not a gap in the evidence: an earlier draft of
        # this change read "agreement ... has no" as a decline and moved the
        # verification benchmark's f081.3 off the entailment model.
        "The Business Associate Agreement has no assignment restriction and survives a "
        "change of control.",
        "The merger agreement contains no financing condition.",
    ]

    @pytest.mark.parametrize("sentence", DECLINES)
    def test_a_decline_is_a_statement_of_absence(self, sentence):
        assert _absence(sentence) == [True]

    @pytest.mark.parametrize("sentence", FACTS)
    def test_a_negated_fact_is_not(self, sentence):
        assert _absence(sentence) == [False]


class TestListsUnderAnAbsenceLeadIn:
    def test_items_under_a_lead_in_about_what_the_documents_lack_are_absences(self):
        answer = (
            "### Missing Information\n"
            "The provided context is insufficient to answer the question as it lacks:\n"
            "*   Quarterly financial statements for FY2024.\n"
            "*   Interim management reports or trading updates for the first half of 2024.\n"
        )
        claims = split_claims(answer)
        assert [c.text for c in claims] == [
            "Quarterly financial statements for FY2024.",
            "Interim management reports or trading updates for the first half of 2024.",
        ]
        assert all(c.is_absence for c in claims)

    def test_the_list_ends_at_the_next_paragraph(self):
        answer = (
            "The data room currently lacks:\n"
            "* Interim financial statements for FY2024.\n"
            "\n"
            "The board approved the strategic review unanimously.\n"
            "* The board approved the strategic review unanimously again.\n"
        )
        assert _absence(answer) == [True, False, False]

    def test_an_item_with_a_figure_is_still_grounded(self):
        # A figure is never folded into an absence: it would escape the figure check.
        answer = "The data room lacks:\n* Revenue of $452.8 million for Q1 FY2024.\n"
        (claim,) = split_claims(answer)
        assert claim.has_numbers and not claim.is_absence

    def test_an_ordinary_lead_in_passes_nothing_on(self):
        answer = "Key terms:\n* The board approved the transaction unanimously.\n"
        assert _absence(answer) == [False]

    def test_a_bullet_ending_in_a_colon_opens_no_list(self):
        answer = (
            "* The documents do not state:\n"
            "* The board approved the transaction unanimously.\n"
        )
        assert _absence(answer) == [False]


class TestFragments:
    @pytest.mark.parametrize("line", [
        "Deductible / Basket Threshold",
        "**Atlas Data Holdings** (Private Equity Consortium — DECLINE)",
        "Northstar Financial Group (Consent Required)",
        "[Sources: 📄 aurora_financials_fy2023.txt; 📄 board_deck_strategic_review_mar2024.txt]",
    ])
    def test_a_heading_fragment_or_lead_in_is_not_a_claim(self, line):
        assert split_claims(line) == []

    @pytest.mark.parametrize("line", [
        # A verb outside the brackets: a statement.
        "Northstar requires written consent (not yet obtained)",
        # A figure: always checked.
        "Atlas Data Holdings ($48 - $52 per share)",
        # Four words by the module's own count.
        "Austria and Canada: not reportable.",
        # "Label: value" pairs keep the whitespace count: fewer than four words
        # by the word pattern, but a wrong value here must still be checked.
        "Maturity: June 30, 2026 [📄 credit_agreement_summary.txt | SECTION 2].",
        "Certainty Level: HIGH [📄 board_deck_strategic_review_mar2024.txt | SLIDE 3].",
    ])
    def test_a_statement_is_still_a_claim(self, line):
        assert len(split_claims(line)) == 1
