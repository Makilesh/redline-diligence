"""Tests for eval/decisions/stats.py — the intervals behind the decision evals."""

from __future__ import annotations

import pytest

from eval.decisions import stats


def test_wilson_interval_does_not_collapse_at_zero():
    # 0 false blocks out of 124 is the query guard's headline; the interval must
    # still admit a non-zero rate, which the normal approximation would not.
    low, high = stats.wilson_interval(0, 124)
    assert low == 0.0
    assert 0.02 < high < 0.04


def test_wilson_interval_brackets_the_proportion():
    low, high = stats.wilson_interval(23, 27)
    assert low < 23 / 27 < high
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)


def test_rule_of_three():
    assert stats.rule_of_three(26) == pytest.approx(3 / 26)
    assert stats.rule_of_three(0) == 1.0


def test_auc_counts_ties_as_half():
    assert stats.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert stats.auc([0.5], [0.5]) == 0.5
    assert stats.auc([], [0.5]) is None


def _f1(rows):
    tp = sum(1 for gold, pred in rows if gold and pred)
    fp = sum(1 for gold, pred in rows if not gold and pred)
    fn = sum(1 for gold, pred in rows if gold and not pred)
    return stats.f1_counts(tp, fp, fn)


def test_bootstrap_ci_is_seeded_and_brackets_the_point():
    rows = [(True, True)] * 12 + [(True, False)] * 8 + [(False, True)] * 3 + [(False, False)] * 40
    first = stats.bootstrap_ci(rows, _f1, replicates=300)
    second = stats.bootstrap_ci(rows, _f1, replicates=300)
    assert first == second
    assert first["low"] < first["point"] < first["high"]
    # 20 positives cannot pin F1 down to a few points.
    assert first["high"] - first["low"] > 0.15


def test_bootstrap_ci_skips_undefined_replicates():
    # One positive in five rows: about a third of resamples draw none.
    rows = [(True, True)] + [(False, False)] * 4
    ci = stats.bootstrap_ci(rows, _f1, replicates=200)
    assert ci["skipped"] > 0
    assert ci["point"] == 1.0


def test_cluster_bootstrap_is_wider_than_row_bootstrap():
    # Errors concentrated in one document: resampling rows hides that.
    rows = [("doc_a", True, True)] * 20 + [("doc_b", True, False)] * 20 + [("doc_c", True, True)] * 20

    def recall(sample):
        return sum(pred for _, _, pred in sample) / len(sample)

    by_row = stats.bootstrap_ci(rows, recall, replicates=400)
    by_doc = stats.bootstrap_ci(rows, recall, replicates=400, cluster=lambda r: r[0])
    assert by_doc["high"] - by_doc["low"] > by_row["high"] - by_row["low"]


def test_paired_bootstrap_sees_a_consistent_gain_that_separate_intervals_miss():
    # The candidate fixes 6 of the baseline's misses and breaks nothing: the
    # two designs' own intervals overlap, the paired difference does not cross 0.
    rows = (
        [{"gold": True, "a": True, "b": True}] * 20
        + [{"gold": True, "a": False, "b": True}] * 6
        + [{"gold": True, "a": False, "b": False}] * 14
    )

    def recall(key):
        return lambda sample: sum(r[key] for r in sample) / len(sample)

    a = stats.bootstrap_ci(rows, recall("a"), replicates=500)
    b = stats.bootstrap_ci(rows, recall("b"), replicates=500)
    assert a["high"] > b["low"]
    diff = stats.paired_bootstrap(rows, recall("a"), recall("b"), replicates=500)
    assert diff["delta"] == pytest.approx(0.15)
    assert diff["low"] > 0
    assert diff["p_better"] > 0.97


def test_paired_bootstrap_cannot_separate_a_one_row_difference():
    rows = (
        [{"gold": True, "a": True, "b": True}] * 5
        + [{"gold": True, "a": False, "b": True}]
        + [{"gold": True, "a": False, "b": False}] * 3
    )

    def recall(key):
        return lambda sample: sum(r[key] for r in sample) / len(sample)

    diff = stats.paired_bootstrap(rows, recall("a"), recall("b"), replicates=500)
    assert diff["low"] <= 0 <= diff["high"]


def test_fmt_ci():
    assert stats.fmt_ci({"point": 0.6584, "low": 0.512, "high": 0.781}) == "0.658 [0.51, 0.78]"
    assert stats.fmt_ci({"point": None, "low": None, "high": None}) == "—"
