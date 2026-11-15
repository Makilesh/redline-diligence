"""
The figure veto (claim_checker.FigureVeto): a claim whose every figure is
confirmed is still scored by a model that has a veto, against the passages the
figures were found in, and a low score sends it to the judge.

No weights are loaded: the scorer is a stub and the loaded model is patched.
"""

import asyncio

from src.verification import claim_checker, nli
from src.verification.nli import NLIScore
from src.verification.numeric_grounding import ContextIndex, ground_texts

XSMALL = "cross-encoder/nli-deberta-v3-xsmall"
SMALL = "cross-encoder/nli-deberta-v3-small"
SOURCE = "customer_concentration_analysis.txt"
CHUNKS = [
    {
        "chunk_id": "c1",
        "text": "Pinnacle Health accounted for 8.1% of total revenue in FY2023. "
                "No other customer exceeded 5% of total revenue.",
        "source_file": SOURCE,
        "page_number": None,
        "section_heading": "Customer concentration",
    },
    {
        "chunk_id": "c2",
        "text": "Pinnacle Health renewed its master services agreement in FY2023 "
                "for a further three years.",
        "source_file": "material_contracts_summary.txt",
        "page_number": None,
        "section_heading": "Renewals",
    },
]
MARKER = f" [📄 {SOURCE} | Customer concentration]"
# Every figure confirmed: the 8.1% is this sentence's, for FY2023.
PLAIN = "Pinnacle Health accounted for 8.1% of total revenue in FY2023." + MARKER
# Same figure, same sentence, opposite meaning — and a negation cue.
NEGATED = "Pinnacle Health did not account for 8.1% of total revenue in FY2023." + MARKER


def _verify(monkeypatch, answer, support, model=XSMALL, judge=None, loaded_after=None):
    seen: list[tuple[str, str]] = []
    state = {"loaded": model}

    async def scorer(pairs):
        seen.extend(pairs)
        if loaded_after is not None:
            state["loaded"] = loaded_after  # e.g. a checker replaced by a fallback
        return [NLIScore(support, 0.0, 1.0 - support) for _ in pairs]

    monkeypatch.setattr(nli, "ascore_pairs", scorer)
    monkeypatch.setattr(nli, "loaded_model_name", lambda: state["loaded"])
    report = asyncio.run(claim_checker.verify_answer(answer, CHUNKS, "concentration?", judge=judge, use_nli=True))
    [check] = report.claim_checks
    return report, check, seen


class TestConfiguration:
    def test_only_xsmall_has_a_veto(self):
        veto = claim_checker.thresholds_for(XSMALL).figure_veto
        assert veto == claim_checker.FigureVeto(line=0.0031, evidence="figure", cues_only=True)
        assert claim_checker.thresholds_for(nli.FACTCG).figure_veto is None
        assert claim_checker.thresholds_for(SMALL).figure_veto is None
        assert claim_checker.thresholds_for(None).figure_veto is None


class TestVeto:
    def test_figures_confirmed_but_unsupported_goes_to_the_judge(self, monkeypatch):
        report, check, seen = _verify(monkeypatch, NEGATED, 0.001)
        assert seen, "a figure-confirmed claim with a cue is scored"
        assert check["status"] == "unverified" and check["suspected_misstatement"]
        assert check["evidence_source"] == SOURCE
        # Never failed, never flagged as unsupported: the judge decides.
        assert report.validation_status == "warning"
        assert report.status_counts["unsupported"] == report.status_counts["contradicted"] == 0

    def test_the_judge_has_the_last_word(self, monkeypatch):
        async def judge(query, payload, documents):
            assert any(d["source_file"] == SOURCE for d in documents)
            return {1: {"status": "contradicted", "reason": "it did", "document": 1}}, "stub"

        report, check, _ = _verify(monkeypatch, NEGATED, 0.001, judge=judge)
        assert check["status"] == "contradicted" and check["method"] == "llm"
        assert report.validation_status == "failed"

    def test_enough_support_keeps_the_figure_verdict(self, monkeypatch):
        report, check, seen = _verify(monkeypatch, NEGATED, 0.5)
        assert seen
        assert check["status"] == "supported" and check["method"] == "numeric"
        assert check["score"] == 0.5
        assert report.validation_status == "passed"

    def test_evidence_is_the_passage_holding_the_figure(self, monkeypatch):
        _, _, seen = _verify(monkeypatch, NEGATED, 0.5)
        assert seen and all("8.1%" in premise for premise, _ in seen)
        assert all(hyp.startswith("Pinnacle Health did not account") for _, hyp in seen)

    def test_no_cue_no_scoring(self, monkeypatch):
        report, check, seen = _verify(monkeypatch, PLAIN, 0.001)
        assert seen == []
        assert check["status"] == "supported" and check["method"] == "numeric"
        assert report.validation_status == "passed"

    def test_a_model_without_a_veto_adds_no_pairs(self, monkeypatch):
        report, check, seen = _verify(monkeypatch, NEGATED, 0.001, model=nli.FACTCG)
        assert seen == []
        assert check["status"] == "supported" and check["method"] == "numeric"

    def test_lines_follow_the_model_that_loaded(self, monkeypatch):
        # Planned for -xsmall, scored by -small (no veto): the scores are ignored.
        _, check, seen = _verify(monkeypatch, NEGATED, 0.001, model=XSMALL, loaded_after=SMALL)
        assert seen and check["status"] == "supported" and check["method"] == "numeric"

    def test_no_nli_no_veto(self, monkeypatch):
        called = []

        async def scorer(pairs):
            called.append(pairs)
            return []

        monkeypatch.setattr(nli, "ascore_pairs", scorer)
        monkeypatch.setattr(nli, "loaded_model_name", lambda: XSMALL)
        report = asyncio.run(claim_checker.verify_answer(NEGATED, CHUNKS, "q", use_nli=False))
        assert called == [] and report.claim_checks[0]["status"] == "supported"


class TestFigureWindows:
    def test_a_longer_number_is_not_the_figure(self):
        windows = {0: [("Revenue was $452.85 million.", {"revenue"}),
                       ("Revenue was $452.8 million in FY2023.", {"revenue", "fy2023"})]}
        index = ContextIndex([{"text": "Revenue was $452.8 million in FY2023.", "source_file": "f.txt"}])
        [checks] = ground_texts(["Revenue was $452.8 million in FY2023."], index)
        got = claim_checker._figure_windows({"revenue", "fy2023"}, checks, windows)
        assert [w for _, w, _ in got] == ["Revenue was $452.8 million in FY2023."]

    def test_where_the_figure_was_found_is_not_serialised(self):
        index = ContextIndex([{"text": "Revenue was $452.8 million in FY2023.", "source_file": "f.txt"}])
        [[check]] = ground_texts(["Revenue was $452.8 million in FY2023."], index)
        assert check.evidence_chunk == 0 and "452.8" in check.evidence_figure
        assert "evidence_chunk" not in check.to_dict() and "evidence_figure" not in check.to_dict()
