"""
Reporting logic of the retrieval eval harness (eval/run_retrieval_eval.py).

No models and no Qdrant: these cover the parts that turn per-question results
into claims —

- the gate statistic is the graph's own routing decision, so the harness cannot
  report "admitted" for a context the router sends to the rewriter;
- every mean carries a bootstrap interval and every comparison a paired one,
  because on a few dozen questions most differences are noise;
- a before/after comparison pairs questions by id across two result files.
"""

from __future__ import annotations

import pytest

import eval.run_retrieval_eval as harness
from eval import metrics as m


def _state(query_type: str = "comparative") -> dict:
    return {"query_type": query_type, "rewrite_iteration": 0, "original_query": "q",
            "current_query": "q", "sub_questions": []}


def _chunks(scores: list[float]) -> list[dict]:
    return [{"chunk_id": f"c{i}", "reranker_score": s} for i, s in enumerate(scores)]


class TestGateIsTheRoutersDecision:
    @pytest.mark.asyncio
    async def test_context_the_router_rejects_is_not_reported_as_admitted(self):
        """
        The heuristic passes on the best score alone; the router also requires
        precision >= 0.25. The harness used to read the heuristic's flag, so it
        reported this context (the shape of comp_02 with decomposition in the
        committed results: one chunk at 0.31, the rest just above the floor) as
        admitted while production sent it to the rewriter.
        """
        from src.agents.quality_assessor import _heuristic_assessment

        reranked = _chunks([0.31, 0.12, 0.11, 0.11, 0.12, 0.11])
        heuristic = _heuristic_assessment({**_state(), "reranked_results": reranked})
        assert heuristic is not None and heuristic["force_refusal"] is False

        gate = await harness.gate_decision(_state(), reranked, laya=False)

        assert gate["decision"] == "refused"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scores,expected", [
        ([0.9, 0.8, 0.7], "admitted"),
        ([0.01, 0.02], "refused"),
        ([0.2, 0.01], "llm_fallback"),
        ([], "refused"),
    ])
    async def test_the_three_outcomes(self, scores, expected):
        gate = await harness.gate_decision(_state("financial"), _chunks(scores), laya=False)
        assert gate["decision"] == expected

    def test_an_answerability_veto_is_a_refusal(self):
        verdict = {"context_quality_score": 0.9, "force_refusal": True, "answerability_veto": True,
                   "quality_breakdown": {"relevance": 0.9, "completeness": 1.0, "precision": 0.9}}
        assert harness._decision(_state(), verdict) == "refused"
        # Without the veto the node does not refuse either (it sets both together).
        assert harness._decision(
            _state(), {**verdict, "answerability_veto": False, "force_refusal": False}
        ) == "admitted"

    def test_decision_agrees_with_the_edge_on_every_band(self):
        """Whatever the thresholds become, the harness reports what the edge does."""
        from src.agents.quality_assessor import _heuristic_assessment
        from src.workflow.conditional_edges import route_after_quality_check

        for top in (0.02, 0.06, 0.2, 0.31, 0.5, 0.95):
            for rest in ([], [0.11] * 5, [0.6, 0.5]):
                state = {**_state(), "reranked_results": _chunks([top, *rest])}
                verdict = _heuristic_assessment(state)
                if verdict is None:
                    continue
                route = route_after_quality_check({**state, **verdict, "answerability_veto": False})
                assert (harness._decision(state, verdict) == "admitted") == (
                    route == "answer_synthesizer")


class TestBootstrap:
    def test_interval_brackets_the_mean_and_is_deterministic(self):
        values = [1.0, 1.0, 0.5, 0.0, 1.0, 0.75, 1.0, 0.25, 1.0, 1.0]
        low, high = m.bootstrap_ci(values)

        assert low < sum(values) / len(values) < high
        assert (low, high) == m.bootstrap_ci(values)
        assert 0.0 <= low and high <= 1.0

    def test_constant_values_have_a_zero_width_interval(self):
        assert m.bootstrap_ci([0.8] * 12) == pytest.approx((0.8, 0.8))

    def test_undefined_questions_are_left_out(self):
        assert m.bootstrap_ci([None, 1.0, None, 0.0]) == m.bootstrap_ci([1.0, 0.0])
        assert m.bootstrap_ci([None, 1.0]) is None
        assert m.bootstrap_ci([]) is None

    def test_interval_narrows_with_more_questions(self):
        few = m.bootstrap_ci([0.0, 1.0] * 5)
        many = m.bootstrap_ci([0.0, 1.0] * 50)
        assert (many[1] - many[0]) < (few[1] - few[0])


class TestPairedDelta:
    def test_consistent_gain_clears_zero(self):
        before = [0.5] * 20
        after = [0.6] * 15 + [0.5] * 5

        delta = m.paired_delta(before, after)

        assert delta["mean"] == pytest.approx(0.075)
        assert (delta["improved"], delta["regressed"], delta["n"]) == (15, 0, 20)
        assert delta["low"] > 0 and delta["clears_zero"] is True

    def test_one_question_flipping_does_not(self):
        """The situation of most single-question differences on a 35-question set."""
        before = [1.0] * 34 + [0.0]
        after = [1.0] * 35

        delta = m.paired_delta(before, after)

        assert delta["improved"] == 1
        assert delta["low"] == pytest.approx(0.0)
        assert delta["clears_zero"] is False

    def test_mixed_changes_span_zero(self):
        delta = m.paired_delta([0.5] * 10, [1.0, 0.0] * 5)
        assert delta["mean"] == pytest.approx(0.0)
        assert delta["low"] < 0 < delta["high"]
        assert delta["clears_zero"] is False

    def test_pairs_with_an_undefined_side_are_dropped(self):
        delta = m.paired_delta([0.0, None, 0.2, 0.4], [1.0, 1.0, None, 0.4])
        assert delta["n"] == 2
        assert m.paired_delta([None, 0.1], [0.2, None]) is None


def _question(qid: str, runs: dict[str, dict], is_control: bool = False) -> dict:
    return {"id": qid, "is_control": is_control,
            "runs": {a: {"metrics": metrics} for a, metrics in runs.items()}}


class TestComparisons:
    def test_stage_comparison_pairs_ablations_within_a_run(self):
        questions = [
            _question(f"q{i}", {"dense": {"recall@10": 0.4, "fact_coverage@10": 0.5},
                                "hybrid_rrf": {"recall@10": 0.6, "fact_coverage@10": 0.5}})
            for i in range(6)
        ] + [_question("ctrl", {"dense": {"recall@10": None}, "hybrid_rrf": {"recall@10": None}},
                       is_control=True)]

        stages = harness.stage_comparisons(questions, ["dense", "hybrid_rrf"], [10])

        assert [(s["before"], s["after"]) for s in stages] == [("dense", "hybrid_rrf")]
        recall = stages[0]["metrics"]["recall@10"]
        assert recall["mean"] == pytest.approx(0.2) and recall["n"] == 6
        assert recall["clears_zero"] is True
        assert stages[0]["metrics"]["fact_coverage@10"]["changed"] == []

    def test_previous_run_is_paired_by_question_id(self):
        def run(values: dict[str, float]) -> dict:
            questions = [_question(q, {"production": {"fact_coverage@context": v}})
                         for q, v in values.items()]
            return {
                "generated_at": "then", "command": "cmd", "questions": questions,
                "golden_set": {"sha256": "abc", "answerable": len(questions)},
                "config": {"k": [10]},
                "aggregates": {"production": {"latency_ms": {"p50": 800.0, "p95": 1200.0}}},
            }

        previous = run({"a": 0.2, "b": 1.0, "c": 1.0, "gone": 0.0})
        current = run({"c": 1.0, "b": 1.0, "a": 1.0, "new": 0.0})  # different order on purpose

        comparison = harness.compare_with_previous(current, previous)

        delta = comparison["ablations"]["production"]["fact_coverage@context"]
        assert delta["n"] == 3, "questions present in only one run are not compared"
        assert delta["changed"] == [{"id": "a", "before": 0.2, "after": 1.0}]
        assert comparison["same_golden_set"] is True
        assert comparison["latency_ms"]["production"]["before"]["p50"] == 800.0

    def test_report_states_whether_a_difference_clears_the_noise(self):
        answerable = [
            _question(f"q{i}", {"production": {"fact_coverage@10": 1.0, "fact_coverage@context": 1.0,
                                               "recall@10": 0.5, "mrr@10": 0.5, "ndcg@10": 0.5},
                                "production_decomp": {
                                    "fact_coverage@10": 1.0,
                                    "fact_coverage@context": 1.0,
                                    "recall@10": 0.5, "mrr@10": 0.5 if i else 1.0, "ndcg@10": 0.5}})
            for i in range(8)
        ]
        for q in answerable:
            q["query_type"] = "financial"
            for run in q["runs"].values():
                run.update({"latency_ms": 1.0, "sub_questions": [],
                            "gate": {"decision": "admitted"}})
        ablations = ["production", "production_decomp"]
        report = {
            "config": {"k": [10]},
            "golden_set": {"answerable": 8},
            "questions": answerable,
            "aggregates": {a: harness.aggregate(answerable, a, [10]) for a in ablations},
            "stage_comparisons": harness.stage_comparisons(answerable, ablations, [10]),
        }

        text = "\n".join(harness.render_uncertainty(report))

        assert "no — within noise" in text  # one question's MRR moved
        assert "100.0 [100.0, 100.0]" in text
        assert report["aggregates"]["production"]["ci95"]["recall@10"] == pytest.approx((0.5, 0.5))


class TestDistractorMode:
    """Noise documents are candidates only: never relevant, never a source of facts."""

    QUESTION = {"expected_answer_contains": ["$85", "term loan"],
                "expected_citations": [{"source_pattern": "credit_agreement"}]}

    @staticmethod
    def _chunk(chunk_id: str, source: str, text: str, **extra) -> dict:
        return {"chunk_id": chunk_id, "source_file": source, "text": text, **extra}

    def test_a_fact_found_only_in_a_distractor_does_not_count(self):
        from eval.corpus import DISTRACTOR_PREFIX

        ranking = [
            self._chunk("d1", f"{DISTRACTOR_PREFIX}0001.txt", "a fee of $85 on the term loan"),
            self._chunk("c1", "credit_agreement_summary.txt", "the term loan matures in 2027"),
        ]
        run = {"ranking": ranking, "context": ranking, "latency_ms": 1.0,
               "gate": {"decision": "admitted"}, "sub_questions": []}

        detail = harness.score_run(run, self.QUESTION, {"c1": 1}, [10])

        assert detail["metrics"]["fact_coverage@10"] == 0.5
        assert detail["metrics"]["fact_coverage@context"] == 0.5
        assert detail["facts_missing_from_context"] == ["$85"]
        assert detail["distractors_in_top"] == 1
        assert detail["metrics"]["mrr@10"] == 0.5  # the distractor still takes rank 1

    def test_masking_keeps_the_context_budget_honest(self):
        from eval.corpus import DISTRACTOR_PREFIX

        noise = self._chunk("d1", f"{DISTRACTOR_PREFIX}0001.txt", "x" * 40, parent_text="y" * 900)
        real = self._chunk("c1", "merger.txt", "kept", parent_text="kept in full")

        masked = harness._mask_distractors([noise, real])

        assert masked[0]["text"] == "#" * 40 and masked[0]["parent_text"] == "#" * 900
        assert masked[1] == real
        assert noise["text"] == "x" * 40, "the retrieval output itself must not be modified"

    def test_same_contracts_every_time(self, monkeypatch, tmp_path):
        """Pinned revision, sorted names: --distractors N means the same N files anywhere."""
        import huggingface_hub

        from eval import corpus

        seen = {}

        class Api:
            def list_repo_files(self, repo, repo_type, revision):
                seen["revision"] = revision
                return [f"{corpus.DISTRACTOR_SUBDIR}/{n}" for n in ("b.txt", "C.TXT", "a.txt")] + [
                    "CUAD_v1/master_clauses.csv", "CUAD_v1/full_contract_pdf/a.pdf"]

        def download(repo, name, repo_type, revision, local_dir):
            return str(tmp_path / name.rsplit("/", 1)[-1])

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

        paths = corpus.fetch_distractors(2, cache_dir=tmp_path)

        assert [p.name for p in paths] == ["C.TXT", "a.txt"]
        assert seen["revision"] == corpus.DISTRACTOR_REVISION
        with pytest.raises(RuntimeError):
            corpus.fetch_distractors(4, cache_dir=tmp_path)

    def test_the_baseline_gate_is_not_available_with_distractors(self):
        with pytest.raises(SystemExit):
            harness.parse_args(["--distractors", "10", "--baseline", "eval/baseline.json"])
        assert harness.parse_args([]).distractors == 0
