"""
The FinanceBench dev/test split keeps each company's filings on one side and is
fixed by its seed (Decision 68). Synthetic questions; no download needed.
"""

from __future__ import annotations

from eval.realdocs_data import (
    EvidencePage,
    FinanceQuestion,
    financebench_company_split,
    select_financebench_split,
)


def _questions() -> list[FinanceQuestion]:
    return [
        FinanceQuestion(id=f"{c}{y}", question="q", answer="a", question_type="t", question_reasoning="",
                        doc_name=f"{c}_{y}_10K", evidence=[EvidencePage(f"{c}_{y}_10K", 1, "")], company=c)
        for c in "ABCDEFGH" for y in (2021, 2022)
    ]


def test_companies_stay_together_and_the_split_is_fixed():
    questions = _questions()

    split = financebench_company_split(questions)

    assert split == financebench_company_split(list(reversed(questions)))
    assert len(split["dev"]) == len(split["test"]) == 4
    assert not set(split["dev"]) & set(split["test"])
    dev_docs = select_financebench_split(questions, "dev")
    assert {d.split("_")[0] for d in dev_docs} == set(split["dev"])
    assert len(dev_docs) == 8  # both years of every dev company


def test_another_seed_gives_another_split():
    questions = _questions()

    assert financebench_company_split(questions, seed=1) != financebench_company_split(questions, seed=2)
