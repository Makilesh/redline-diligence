"""
Validator time per answer in the answer replay (Decision 75): premise changes
are bounded by what they cost per answer, so the replay reports it — and only
when the model scored the pairs, never from a score cache.
"""

from __future__ import annotations

from eval import run_answer_replay as ar


def test_latency_summary_nearest_rank():
    out = ar.latency_summary([float(v) for v in range(1, 21)] + [None])
    assert out == {"answers": 20, "p50_ms": 10.0, "p95_ms": 19.0, "max_ms": 20.0}
    assert ar.latency_summary([]) is None
    assert ar.latency_summary([None]) is None


def _report(misses):
    row = {"key": "pm:fin_01", "run": "pm", "id": "fin_01", "judge_called": False,
           "context_reproduced": True, "validator_ms": 120.0, "claims": []}
    repro = {"pm:fin_01": {"reproduced": True, "missing_citations": [], "same_sources": True,
                           "verdicts": {"comparable": False}}}
    return {"generated_at": "t", "as_of": "pm-run", "nli": "m", "score_cache_misses": misses,
            "summary": ar.summarise([row], repro)}


def test_times_are_shown_only_when_the_model_scored():
    assert "validator time per answer" in ar.render(_report(None))
    assert "p50 120.0 ms" in ar.render(_report(None))
    # Served from a score cache, the times say nothing about the model.
    assert "validator time per answer" not in ar.render(_report(0))
    assert "validator time per answer" not in ar.render(_report(12))
