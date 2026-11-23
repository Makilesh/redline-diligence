"""
An answer that says the documents searched do not contain what was asked ends as
"not_found", not "answered" (Decision 112).

Round 16's wrong answer — "The provided documents do not contain specific
information about the revenue contribution from the top ten customers", when a
schedule in the room states 47.8% — shipped as `answered`, and its absence
statements could be confirmed only against the passages a filter had chosen. The
rule that recognises such an answer reads its opening section (its direct answer)
only, and only an opening that declines in full: a cited or figured opening is a
partial answer and stays `answered`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.verification.claims import not_found_lead
from src.workflow import orchestrator

ROUND_16 = (
    "### Direct Answer\n"
    "The provided documents do not contain specific information about the revenue "
    "contribution from the top ten customers for FY2023. Therefore, it is not possible to "
    "determine what share of FY2023 revenue comes from the top ten customers based on the "
    "given context.\n\n"
    "### Supporting Evidence with Citations\n"
    "The documents include detailed financial statements for FY2023 "
    "[📄 aurora_financials_fy2023.txt | p.1 | CONSOLIDATED INCOME STATEMENT].\n"
)


@pytest.mark.parametrize("answer", [
    ROUND_16,
    # No heading; the opening paragraph is the direct answer.
    "Based on the provided documents, a direct comparison of customer churn rates is not "
    "possible because the data room does not contain churn metrics for Aurora's competitors."
    "\n\n### 1. Direct Answer\nAurora reports NRR of 118% [📄 deck.txt | Slide 2].",
    "### 1. Direct Answer\nThe provided context **does not contain** quarterly revenue figures "
    "for either Q1 or Q2 of FY2024.\n\n---\n\n### 2. Evidence\n* FY2023 revenue: $452.8M "
    "[📄 aurora_financials_fy2023.txt | p.1 | INCOME STATEMENT]",
    "Based on the provided documents, the specific projected revenue for FY2025 is **not "
    "available**. The provided context is limited to historical data for FY2022 and FY2023 "
    "and does not include forward-looking projections.\n\n### 1. Supporting Evidence\n- x",
    "The implied EV/EBITDA multiple on adjusted EBITDA cannot be calculated from the provided "
    "documents.\n\n**Missing Information:**\nThe share count is not provided.",
    "No. None of the provided documents mention an MSCI ESG rating for the company.",
])
def test_a_whole_decline_is_recognised(answer):
    assert not_found_lead(answer)


@pytest.mark.parametrize("answer", [
    # Round 17's answer to the same question, after Decision 109.
    "The top ten customers represent 47.8% of Aurora's FY2023 revenue "
    "[📄 customer_contracts_schedule.txt | p.1 | SECTION 1].",
    # Declines the total, then answers in part inside the same section.
    "### Direct Answer\n\nAn explicit total Free Cash Flow figure for FY2023 is **not provided** "
    "in the context chunks.\n\nHowever, based on the available context:\n"
    "* **Operating Cash Flow:** **$52.1 million** [📄 credit_agreement_summary.txt | Section 1]"
    "\n\n---\n\n### Components",
    # Partially disclosed: states what the documents do give.
    "Based on the provided documents, the total cash required at closing is only partially "
    "disclosed. The documents provide specific figures for the revolving credit facility but "
    "do not contain the amounts for executive severance.\n\n### 1. Revolver\n- $85,000,000",
    # A finding about a contract term, not a gap in the evidence.
    "### Direct Answer\nNo, the merger agreement does not contain a financing condition.\n\n"
    "### Evidence\nSection 6 lists the conditions [📄 merger_agreement.txt | Section 6].",
    # A cited decline asserts what its source says.
    "The data room does not contain a churn figure [📄 deck.txt | Slide 2].",
    # A figure in the opening is an answer.
    "The documents do not state FY2024 revenue; FY2023 revenue was $452.8 million.",
    # A decline followed by an uncited assertion is not only a decline.
    "The provided documents do not contain the top-ten share. Meridian Health is the largest "
    "customer.",
    # Found in review: findings that read like gaps in the evidence.
    "The transaction documents contain no financing condition.\n\n## Details\nx",
    "In the context of the merger, a financing condition is not provided for.\n\n## Details\nx",
    "The target could not identify a successor CEO before signing.\n\n## Details\nx",
    "The outcome of the DataFlow litigation cannot be determined at this time.\n\n## Details\nx",
    "The seller provided insufficient information to the regulator.\n\n## Details\nx",
    "",
    None,
])
def test_anything_else_is_not(answer):
    assert not_found_lead(answer) is None


def test_on_the_recorded_live_answers_it_picks_exactly_the_reviewed_declines():
    """
    The 123 answers of the committed end-to-end runs (eval/answer_replay_answers.json),
    classified and read one by one on 2026-11-24: these eight decline the whole
    question, and no other answer is touched. Among those kept: aug:fin_03, which
    declines the total but answers from the components, and pm:mh_06, partial.
    """
    replay = json.loads(Path("eval/answer_replay_answers.json").read_text(encoding="utf-8"))
    picked = {f"{r['run']}:{r['id']}" for r in replay["records"] if not_found_lead(r["answer"])}
    assert picked == {
        "pm:ctrl_04", "am:ctrl_04", "am:ctrl_05", "am:mh_05",
        "aug:mh_04", "aug:ctrl_02", "aug:ctrl_04", "aug:ctrl_05",
    }


@pytest.mark.parametrize(("state", "expected"), [
    ({"generated_answer": ROUND_16, "validation_status": "passed"}, "not_found"),
    # How the run ended outranks a failed validation; the status still travels.
    ({"generated_answer": ROUND_16, "validation_status": "failed"}, "not_found"),
    ({"generated_answer": ROUND_16, "force_refusal": True}, "refused"),
    (
        {"generated_answer": ROUND_16, "agent_trace": [{"agent": "synthesis_unavailable"}]},
        "synthesis_failed",
    ),
])
def test_run_outcome(state, expected):
    assert orchestrator.run_outcome(state) == expected
