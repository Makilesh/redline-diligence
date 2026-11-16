"""
eval/answerability_grid.py: the offline (form, top-k, threshold) recomputation
must reproduce the gate's own aggregation, and the selection must follow the
rule written down before the grid was run.
"""

import asyncio

import pytest

from eval import answerability_grid as ag
from src.decisions import answerability as ans


def _chunk(text, score, header=""):
    return {"text": text, "reranker_score": score, "context_header": header}


HEADER = "Document: credit agreement summary\nSection: FINANCIAL COVENANTS"


def test_passage_forms():
    c = _chunk("Leverage below 4.0x.", 0.9, HEADER)
    assert ag.passage_text(c, "header") == f"{HEADER}\n\nLeverage below 4.0x."
    assert ag.passage_text(c, "title") == "Document: credit agreement summary\n\nLeverage below 4.0x."
    assert ag.passage_text(c, "bare") == "Leverage below 4.0x."
    bare = _chunk("No header here.", 0.9)
    assert {ag.passage_text(bare, f) for f in ag.FORMS} == {"No header here."}
    with pytest.raises(ValueError):
        ag.passage_text(c, "sideways")


def test_header_form_is_what_the_gate_sends():
    chunks = [_chunk("a", 0.2, HEADER), _chunk("b", 0.9), _chunk("", 0.95), _chunk("c", 0.5, HEADER)]
    expected = ans.top_passages(chunks, k=3)
    assert [ag.passage_text(c, "header") for c in ag.ranked_chunks(chunks, 3)] == expected


def test_pair_scores_scores_each_distinct_pair_once(monkeypatch):
    calls = []

    async def fake_batch(requests):
        calls.append(requests)
        return [{"stated": {"noul": 0.1 * (i + 1)}} for i in range(len(requests))]

    monkeypatch.setattr("src.decisions.laya_client.adecide_batch", fake_batch)
    chunks = [_chunk("x", 0.9, HEADER), _chunk("y", 0.8)]
    matrix = asyncio.run(ag.pair_scores(["Q?"], chunks, "instr", k=5))
    # x in three forms + y once (no header): 4 distinct pairs.
    assert len(calls) == 1 and len(calls[0]) == 4
    assert matrix["bare"]["Q?"][1] == matrix["header"]["Q?"][1] == matrix["title"]["Q?"][1]
    assert len({matrix[f]["Q?"][0] for f in ag.FORMS}) == 3


def test_score_and_decision_follow_the_gate():
    matrix = {"header": {"Q": [0.2, 0.5, 0.3], "sub": [0.1, 0.1, 0.6]}}
    assert ag.score_at(matrix, "header", 1) == 0.2
    assert ag.score_at(matrix, "header", 2) == 0.5
    assert ag.score_at(matrix, "header", 3) == 0.6
    # Same aggregation as answerability.aggregate.
    _, score, vetoed, _ = ans.aggregate({f: ps[:2] for f, ps in matrix["header"].items()}, 0.55)
    assert score == ag.score_at(matrix, "header", 2) and vetoed
    assert ag.laya_decision("refused", 0.9, 0.35) == "refused"
    assert ag.laya_decision("admitted", 0.30, 0.35) == "refused"
    assert ag.laya_decision("llm_fallback", 0.40, 0.35) == "llm_fallback"
    assert ag.laya_decision("admitted", None, 0.35) == "admitted"


def _row(label, p, heuristic="admitted"):
    return {"label": label, "heuristic": heuristic,
            "matrix": {f: {"q": [p, 0.0, 0.0, 0.0, 0.0]} for f in ag.FORMS}}


def test_evaluate_counts_and_ignores_heuristic_refusals():
    rows = [_row("answerable", 0.50), _row("answerable", 0.45),
            _row("unanswerable", 0.30), _row("unanswerable", 0.40),
            _row("unanswerable", 0.10, heuristic="refused"), _row("retrieval_miss", 0.01)]
    cell = ag.evaluate(rows, "header", 3, 0.35)
    assert cell["false_vetoes"] == 0 and cell["caught"] == 1
    assert cell["refused_unanswerable"] == 2  # the heuristic refusal counts as refused
    assert cell["lowest_answerable"] == 0.45 and cell["auc"] == 1.0
    # A route refusal that was not a forced one is still scored, as the node does.
    routed = dict(_row("unanswerable", 0.10, heuristic="refused"), consulted=True)
    assert ag.evaluate(rows + [routed], "header", 3, 0.35)["caught"] == 2


def test_select_respects_margin_then_tie_breaks():
    rows = [_row("answerable", 0.45), _row("unanswerable", 0.30), _row("unanswerable", 0.38)]
    cells = ag.grid(rows)
    best = ag.select(cells)
    # 0.40 would catch both but sits within 0.04 of the lowest answerable (0.45 - 0.04 = 0.41 ok,
    # 0.425 is not): the best admissible threshold catches both at 0.40.
    assert best["caught"] == 2 and best["threshold"] == 0.4
    # Ties: k=3 and the header form are preferred, then the lowest such threshold.
    assert best["k"] == 3 and best["form"] == "header"
    assert ag.select([dict(c, false_vetoes=1) for c in cells]) is None
