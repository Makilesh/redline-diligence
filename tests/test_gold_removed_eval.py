"""
eval/run_gold_removed_eval.py: decisions recomputed per gate setting, and the
paired comparison of two runs, on hand-made rows (no models, no store).
"""

from eval import run_gold_removed_eval as gr


def _row(qid, condition, p, heuristic="admitted", consulted=True, doc="d1", subset="cuad"):
    return {"id": qid, "condition": condition, "heuristic": heuristic, "consulted": consulted,
            "document": doc, "subset": subset,
            "matrix": {f: {"q": [p, 0.0, 0.0, 0.0, 0.0]} for f in ("header", "title", "bare")}}


def test_parse_setting():
    assert gr.parse_setting("title,1,0.3") == ("title", 1, 0.3)


def test_decisions_follow_the_gate():
    rows = [_row("a", "removed", 0.30), _row("b", "removed", 0.50),
            _row("c", "removed", 0.10, heuristic="refused", consulted=False),
            _row("d", "removed", 0.20, heuristic="llm_fallback")]
    out = gr.decisions(rows, ("bare", 3, 0.35))
    assert out == {("a", "removed"): "refused", ("b", "removed"): "admitted",
                   ("c", "removed"): "refused", ("d", "removed"): "refused"}


def test_compare_pairs_runs_by_query_and_condition():
    a = {"rows": [_row("q1", "removed", 0.30, doc="d1"), _row("q2", "removed", 0.50, doc="d2"),
                  _row("q1", "present", 0.60, doc="d1"), _row("q2", "present", 0.60, doc="d2")]}
    b = {"rows": [_row("q1", "removed", 0.50, doc="d1"), _row("q2", "removed", 0.50, doc="d2"),
                  _row("q1", "present", 0.30, doc="d1"), _row("q2", "present", 0.60, doc="d2")]}
    result = gr.compare(a, b, ("bare", 3, 0.35), ("bare", 3, 0.35))
    assert result["removed"]["a"]["point"] == 0.5 and result["removed"]["b"]["point"] == 0.0
    assert result["removed"]["b_minus_a"]["delta"] == -0.5
    assert result["present"]["b"]["point"] == 0.5
    assert result["removed"]["by_subset"]["cuad"] == {"n": 2, "a": 1, "b": 0}
    assert "Gold document removed" in gr.render_compare(result, "off", "on")
