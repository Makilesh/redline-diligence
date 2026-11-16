"""
The answer-replay harness (eval/run_answer_replay.py) and the retrieval trace
field it relies on (Decision 72).

What matters here is that the measurement cannot quietly measure the wrong
thing: the split is fixed, a replay that did not find the live context is
caught, labelled claims keep their sampling weight, and a claim the splitter
stops producing counts against the change that dropped it.
"""

from __future__ import annotations

import pytest

import src.agents.retrieval_executor as rx
from eval import run_answer_replay as ar
from tests.test_retrieval_batching import _point, _Recorder, _state


class TestRetrievalTrace:
    @pytest.mark.asyncio
    async def test_the_trace_names_the_chunks_the_answer_is_written_from(self, monkeypatch):
        _Recorder(monkeypatch, {
            "alpha beta": ([_point("c1", "nothing"), _point("c2", "alpha"), _point("c3", "alpha beta")], []),
        })

        out = await rx.retrieval_executor_node(_state())

        (trace,) = out["agent_trace"]
        assert trace["retrieved_chunk_ids"] == [c["chunk_id"] for c in out["reranked_results"]]
        assert trace["retrieved_chunk_ids"] == ["c3", "c2", "c1"]


class TestSplit:
    IDS = ["fin_01", "fin_02", "fin_03", "legal_01", "legal_02", "ctrl_01", "ctrl_02", "mh_01"]

    def test_fixed_and_independent_of_order(self):
        assert ar.split_of(self.IDS) == ar.split_of(list(reversed(self.IDS)))

    def test_every_family_is_on_both_sides(self):
        split = ar.split_of(self.IDS)
        for prefix in ("fin", "legal", "ctrl"):
            sides = {split[q] for q in self.IDS if q.startswith(prefix)}
            assert sides == {"dev", "test"}

    def test_a_new_question_does_not_move_others_out_of_their_family(self):
        # Adding a question to one family leaves other families' assignment alone.
        before = ar.split_of(self.IDS)
        after = ar.split_of(self.IDS + ["sum_01"])
        assert {q: after[q] for q in self.IDS} == before


class TestOutcomes:
    @pytest.mark.parametrize("check, outcome", [
        ({"status": "supported", "method": "nli"}, "pass"),
        ({"status": "unsupported", "method": "nli"}, "flag"),
        ({"status": "unsupported", "method": "numeric"}, "flag"),
        ({"status": "unverified", "method": "none", "reason": ""}, "judge"),
        ({"status": "unverified", "method": "none",
          "reason": "not checked: beyond the first 40 statements"}, "unchecked"),
    ])
    def test_replay(self, check, outcome):
        assert ar.claim_outcome(check) == outcome

    def test_a_live_judge_verdict_is_judge_bound_whatever_it_said(self):
        assert ar.live_class({"status": "unsupported", "method": "llm"}) == "judge"
        assert ar.live_class({"status": "supported", "method": "numeric"}) == "pass"


def _rec(run="pm", citations=None, sources=("a.txt",)):
    return {
        "run": run, "id": "fin_01",
        "citations": citations or [{"chunk_id": "c1", "source_file": "a.txt", "section_heading": "S"}],
        "retrievals": [{"sources": list(sources)}],
    }


class TestContextReproduced:
    def test_all_citations_and_the_same_sources(self):
        replayed = {"context": [{"chunk_id": "c1", "source_file": "a.txt"}], "sources": ["a.txt"]}
        assert ar.context_reproduced(_rec(), replayed)["reproduced"]

    def test_a_cited_chunk_missing_from_the_replay_is_not_a_reproduction(self):
        replayed = {"context": [{"chunk_id": "c9", "source_file": "a.txt"}], "sources": ["a.txt"]}
        out = ar.context_reproduced(_rec(), replayed)
        assert not out["reproduced"] and out["missing_citations"] == ["c1"]

    def test_other_sources_are_not_a_reproduction(self):
        replayed = {"context": [{"chunk_id": "c1", "source_file": "a.txt"}], "sources": ["a.txt", "b.txt"]}
        assert not ar.context_reproduced(_rec(), replayed)["reproduced"]

    def test_a_run_with_incomparable_ids_matches_on_section(self):
        cit = [{"chunk_id": "old-uuid_0003", "source_file": "a.txt", "section_heading": "Revenue"}]
        replayed = {"context": [{"chunk_id": "c1", "source_file": "a.txt", "section_heading": "REVENUE "}],
                    "sources": ["a.txt"]}
        assert ar.context_reproduced(_rec(run="aug", citations=cit), replayed)["reproduced"]


class TestReplayState:
    def test_the_rewritten_query_and_iteration_are_used_and_noted(self):
        rec = {
            "query": "q", "golden_query_type": "legal",
            "query_intelligence": {"query_type": "legal", "reformulated_query": "q2",
                                   "metadata_filters": {"document_category": "legal", "doc_id": "x"},
                                   "sub_questions": []},
            "retrievals": [{"sub_questions": []}, {"sub_questions": []}],
            "rewrites": ["q3"],
        }
        state, caveats = ar.replay_state(rec, "deal", "pm-run")
        assert state["current_query"] == "q3" and state["rewrite_iteration"] == 1
        # Only whitelisted filters, exactly as Agent 1 forwards them.
        assert state["extracted_filters"] == {"document_category": "legal"}
        # The pm run predates document focus (Decision 67).
        assert state["retrieval_config"]["document_focus_k"] == 0
        assert any("rewrit" in c for c in caveats)

    def test_current_settings_keep_document_focus(self):
        rec = {
            "query": "q", "golden_query_type": "legal",
            "query_intelligence": {"query_type": "legal", "metadata_filters": {}, "sub_questions": ["s"]},
            "retrievals": [{"sub_questions": ["s"]}], "rewrites": [],
        }
        state, _ = ar.replay_state(rec, "deal", "current")
        assert state["retrieval_config"]["document_focus_k"] > 0
        assert state["sub_questions"] == ["s"] and state["current_query"] == "q"


class TestRates:
    LABELS = {
        "strata": {"flag": {"weight": 1.0}, "pass": {"weight": 4.0}},
        "labels": {
            "pm:fin_01:aaa": {"label": "supported", "stratum": "flag", "cause": "multi_fact"},
            "pm:fin_01:bbb": {"label": "supported", "stratum": "pass"},
            "pm:fin_01:ccc": {"label": "not_supported", "stratum": "pass"},
            "pm:fin_01:ddd": {"label": "not_supported", "stratum": "flag"},
        },
    }
    ROWS = [{"claims": [
        {"key": "pm:fin_01:aaa", "outcome": "flag", "method": "nli"},
        {"key": "pm:fin_01:bbb", "outcome": "pass", "method": "numeric"},
        {"key": "pm:fin_01:ddd", "outcome": "flag", "method": "numeric"},
    ]}]

    def test_strata_weights_and_dropped_claims(self):
        items = ar.labelled_items(self.ROWS, self.LABELS, {"fin_01": "dev"}, "dev", set())
        by_key = {i["key"]: i for i in items}
        # ccc is no longer produced by the splitter: a wrong claim nobody checks.
        assert by_key["pm:fin_01:ccc"]["outcome"] == "skipped"
        rates = ar._weighted_rates(items)
        # Correct claims: weight 1 flagged, weight 4 passed.
        assert rates["false_alarm"] == pytest.approx(1 / 5)
        assert rates["false_alarm_nli"] == pytest.approx(1 / 5)
        # Wrong claims: weight 4 skipped (a false pass), weight 1 flagged.
        assert rates["false_pass"] == pytest.approx(4 / 5)

    def test_other_split_and_unreproduced_answers_are_left_out(self):
        assert ar.labelled_items(self.ROWS, self.LABELS, {"fin_01": "test"}, "dev", set()) == []
        assert ar.labelled_items(self.ROWS, self.LABELS, {"fin_01": "dev"}, "dev", {"pm:fin_01"}) == []

    def test_intervals_resample_questions(self):
        items = ar.labelled_items(self.ROWS, self.LABELS, {"fin_01": "dev"}, "dev", set())
        out = ar.rates_with_intervals(items, samples=50)
        assert out["questions"] == 1
        # One question: every resample is the same, so the interval is a point.
        assert out["false_alarm"]["ci95"] == [0.2, 0.2]
