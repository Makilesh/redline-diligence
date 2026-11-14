"""
The quality gate's precision dimension must not fall when retrieval finds more
relevant evidence (audit W8; the Decision 17 failure in another dimension).
"""

import pytest

import src.agents.quality_assessor as qa
from src.workflow.conditional_edges import route_after_quality_check


def _state(scores, query_type="comparative"):
    return {
        "original_query": "Compare the change of control severance multiples.",
        "current_query": "Compare the change of control severance multiples.",
        "query_type": query_type,
        "sub_questions": [],
        "reranked_results": [
            {"chunk_id": f"c{i}", "text": f"passage {i}", "reranker_score": s}
            for i, s in enumerate(scores)
        ],
        "rewrite_iteration": 0,
        "force_refusal": False,
        "answerability_veto": False,
    }


def _route(scores, query_type="comparative"):
    state = _state(scores, query_type)
    out = qa._heuristic_assessment(state)
    return out, route_after_quality_check({**state, **out})


def test_comp_05_is_not_refused_for_retrieving_one_more_relevant_passage():
    # Measured on the golden set: without context headers comp_05 retrieved
    # [0.459, 0.212, 0.164] and was admitted; with them retrieval also found the
    # 280G passage (0.145) and the old mean-of-usable precision fell to 0.245.
    before, route_before = _route([0.459, 0.212, 0.164, 0.04])
    after, route_after = _route([0.459, 0.212, 0.164, 0.145, 0.04])
    assert route_before == route_after == "answer_synthesizer"
    assert after["quality_breakdown"]["precision"] == before["quality_breakdown"]["precision"]


QUERY_TYPES = ["financial", "legal", "comparative", "summary", "multi_hop"]


def _precision(scores, query_type):
    return qa._heuristic_assessment(_state(scores, query_type))["quality_breakdown"]["precision"]


@pytest.mark.parametrize("query_type", QUERY_TYPES)
def test_precision_does_not_fall_once_the_expected_evidence_is_there(query_type):
    expected = qa.EXPECTED_EVIDENCE_COUNT[query_type]
    scores = [0.62, 0.48, 0.33][:expected]
    previous = _precision(scores, query_type)
    for extra in (0.29, 0.2, 0.15, 0.11, 0.10):
        scores = [*scores, extra]
        precision = _precision(scores, query_type)
        assert precision >= previous
        previous = precision


@pytest.mark.parametrize("query_type", QUERY_TYPES)
def test_precision_is_never_below_the_old_mean_of_usable(query_type):
    for scores in ([0.62, 0.31], [0.62, 0.31, 0.29, 0.2, 0.11], [0.459, 0.212, 0.164, 0.145]):
        usable = [s for s in scores if s >= qa.RELEVANCE_FLOOR]
        assert _precision(scores, query_type) >= round(sum(usable) / len(usable), 3)


def test_weak_evidence_alone_is_still_refused_on_precision():
    # The top `expected` usable chunks are all weak: precision stays below the
    # router's 0.25 even though relevance passes.
    out, route = _route([0.31, 0.12, 0.11, 0.10])
    assert out["quality_breakdown"]["precision"] < 0.25
    assert route == "query_rewriter"
