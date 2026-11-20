"""
The checklist grid's harness (eval/run_checklist_eval.py, Decision 95): the
metrics, the pre-registered threshold rules and the evidence choice it fits the
shipped status models with. Offline: synthetic cells, no data.
"""

from __future__ import annotations

import pytest

from eval import run_checklist_eval as harness


def _cell(present: bool, status: str, hit1: bool = False, hit3: bool = False, key: str = "a",
          category: str = "Insurance") -> dict:
    return {"present": present, "status": status, "hit1": hit1, "hit3": hit3 or hit1, "key": key,
            "category": category}


def test_the_metrics_count_what_the_reviewer_is_told():
    cells = [
        _cell(True, "found", hit1=True),       # right, with the clause shown
        _cell(True, "found", hit1=False),      # there, but another passage shown
        _cell(True, "not_found"),              # a false absence
        _cell(True, "unclear", hit3=True),
        _cell(False, "not_found"),             # right
        _cell(False, "found"),                 # a false presence
        _cell(False, "unclear"),
    ]
    m = harness.cell_metrics(cells)
    assert m["present"] == 4 and m["absent"] == 3
    assert m["FA"] == pytest.approx(1 / 4)
    assert m["NF"] == pytest.approx(1 / 3)
    assert m["FP"] == pytest.approx(1 / 3) and m["FP_lenient"] == pytest.approx(2 / 3)
    assert m["DC"] == pytest.approx(2 / 7) and m["decided"] == pytest.approx(5 / 7)
    assert m["hit1"] == pytest.approx(1 / 4) and m["hit3"] == pytest.approx(2 / 4)


def test_the_absent_threshold_keeps_at_most_the_allowed_share_of_present_cells_below():
    cells = [_cell(True, "") for _ in range(100)] + [_cell(False, "") for _ in range(50)]
    p_present = [0.5005 + i / 1000 for i in range(100)] + [0.01] * 50
    p_present[:2] = [0.05, 0.08]  # two present cells score low: 2% may fall below
    t = harness.absent_threshold(p_present, cells, fa_max=0.02)
    assert sum(p < t for p, c in zip(p_present, cells) if c["present"]) == 2
    assert t == pytest.approx(0.502)  # the next grid step would put a third (0.5025) below


def test_the_found_threshold_is_the_lowest_at_which_found_cells_are_right_often_enough():
    cells = [_cell(True, "", hit1=True) for _ in range(9)] + [_cell(False, "")] + [_cell(False, "") for _ in range(10)]
    p_correct = [0.9] * 9 + [0.95] + [0.2] * 10
    assert harness.found_threshold(p_correct, cells, fp_min=0.9) == pytest.approx(0.201)
    assert harness.found_threshold(p_correct, cells, fp_min=0.95) is None


def test_statuses_put_found_first():
    assert harness.statuses([0.01, 0.5, 0.01, 0.01], [0.9, 0.1, 0.1, 0.1], [0.05, 0.05, 0.05, None],
                            t_found=0.8) == ["found", "unclear", "not_found", "unclear"]
    assert harness.statuses([0.9], [0.99], [0.05], t_found=None) == ["unclear"]


def test_an_item_needs_fifty_clauses_for_a_not_found_bar_of_its_own():
    common = [{"item": "governing_law", "present": True} for _ in range(60)]
    rare = [{"item": "most_favoured_nation", "present": True} for _ in range(49)]
    absent = [{"item": "governing_law", "present": False} for _ in range(10)]
    cells = common + rare + absent
    p_present = [0.3005 + i / 1000 for i in range(60)] + [0.2] * 49 + [0.01] * 10
    p_present[0] = 0.05  # one common clause scores low: 2% of 60 allows one below
    bars = harness.item_absent_bars(p_present, cells, pooled=0.5)
    assert set(bars) == {"governing_law"}
    assert bars["governing_law"] == pytest.approx(0.301)  # just above its second-lowest clause would put two below
    assert harness.item_absent_bars(p_present, cells, pooled=0.1)["governing_law"] == pytest.approx(0.1)


def test_the_evidence_is_the_best_window_within_the_depth_and_alternates_do_not_overlap_it():
    windows = [
        {"chunk": "c1", "rank": 0, "start": 0, "end": 100, "raw": {"document": 0.2}},
        {"chunk": "c1", "rank": 0, "start": 50, "end": 150, "raw": {"document": 0.6}},
        {"chunk": "c2", "rank": 1, "start": 0, "end": 100, "raw": {"document": 0.9}},
        {"chunk": "c1", "rank": 0, "start": 150, "end": 250, "raw": {"document": 0.5}},
    ]
    best, alternates = harness.best_window(windows, "document", 1, lambda s: s)
    assert (best["chunk"], best["start"]) == ("c1", 50)
    assert [(w["chunk"], w["start"]) for w in alternates] == [("c1", 150)]
    best, _ = harness.best_window(windows, "document", 2, lambda s: s)
    assert best["chunk"] == "c2"


def test_the_halves_are_seeded_per_group_and_stable():
    groups = {f"cuad/doc{i}.txt": ("mini-dev" if i < 5 else "fresh-dev") for i in range(25)}
    halves = harness.checklist_halves(groups)
    assert halves == harness.checklist_halves(dict(reversed(list(groups.items()))))
    assert sum(h == "tune" for d, h in halves.items() if groups[d] == "mini-dev") == 3
    assert sum(h == "tune" for d, h in halves.items() if groups[d] == "fresh-dev") == 10


def test_gold_spans_index_the_stored_text():
    context = "Recitals. This Agreement is governed by the laws of Delaware."
    answers = [{"answer_start": 10, "text": "This Agreement is governed by the laws of Delaware."}]
    assert harness._spans(context, context, answers) == [[10, 61]]
    # A context that differs from the stored text is located by the span's text.
    assert harness._spans(context, "X" + context, answers) == [[11, 62]]
    assert harness._spans(context, context, [{"answer_start": 0, "text": "nope"}]) is None


def test_the_cpu_host_is_emulated_with_the_gpu_hidden(monkeypatch):
    # An empty CUDA_VISIBLE_DEVICES is deleted on Windows (the C runtime's
    # putenv), which shows the GPU again: the CPU measurements must say "-1".
    import argparse
    import os

    for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RERANKER_MODEL", "RERANKER_FP16",
                 "WARM_MODELS"):
        monkeypatch.setenv(name, "unset")
    harness._configure(argparse.Namespace(reranker="minilm", device="cpu", threads=2))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "2"
    assert os.environ["RERANKER_MODEL"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert os.environ["RERANKER_FP16"] == "0"
