"""
Uncertainty for the decision evals — how much of a reported gain is noise.

The labelled sets behind src/decisions are small (tens of positives), so a
difference of one or two rows moves F1 by several points. Every table in
eval/decisions/ and eval/results/ therefore carries an interval next to its
point estimate, computed here:

  wilson_interval    a proportion k/n (false vetoes, blocks, accuracy)
  bootstrap_ci       any statistic of a list of rows (F1, AUC, recall)
  paired_bootstrap   the difference between two designs scored on the SAME
                     rows — the test an adoption decision actually needs,
                     because two designs' errors on one set are correlated and
                     their separate intervals overstate the uncertainty of the
                     difference

Percentile bootstrap, seeded, numpy only. Rows are resampled with replacement;
pass `cluster` to resample whole groups instead (chunks of one document are
not independent draws). A replicate on which the statistic is undefined (no
positives drawn) is skipped and counted, not silently treated as zero.
"""

from __future__ import annotations

import math
from typing import Callable, Hashable, Sequence, TypeVar

import numpy as np

Row = TypeVar("Row")
Statistic = Callable[[Sequence[Row]], "float | None"]

DEFAULT_REPLICATES = 2000
DEFAULT_SEED = 20261114


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because the proportions here sit
    at or near 0 and 1 (0 false blocks out of 124), where the normal interval
    collapses to a point.

    Args:
        k: Successes.
        n: Trials.
        z: Normal quantile (1.96 = 95%).

    Returns:
        (low, high); (0.0, 1.0) when n is 0.
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rule_of_three(n: int) -> float:
    """
    95% upper bound on an event rate after n trials with zero events (3/n).

    "0 false vetoes out of 26" does not mean the rate is zero; it means the
    rate is plausibly anything below this.
    """
    return 1.0 if n <= 0 else min(1.0, 3.0 / n)


def _resample_indices(
    n: int, rng: np.random.Generator, clusters: list[list[int]] | None
) -> list[int]:
    if clusters is None:
        return rng.integers(0, n, size=n).tolist()
    picked = rng.integers(0, len(clusters), size=len(clusters))
    return [i for c in picked for i in clusters[c]]


def _clusters(rows: Sequence[Row], cluster: Callable[[Row], Hashable] | None) -> list[list[int]] | None:
    if cluster is None:
        return None
    groups: dict[Hashable, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(cluster(row), []).append(i)
    return list(groups.values())


def bootstrap_ci(
    rows: Sequence[Row],
    statistic: Statistic,
    *,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    cluster: Callable[[Row], Hashable] | None = None,
) -> dict:
    """
    Percentile bootstrap interval for a statistic of labelled rows.

    Args:
        rows: The scored rows (any objects the statistic understands).
        statistic: rows -> value, or None when undefined on that resample.
        replicates: Bootstrap resamples.
        seed: RNG seed — reports must be reproducible.
        alpha: 1 - coverage.
        cluster: row -> group key; resamples groups instead of rows.

    Returns:
        {"point", "low", "high", "n", "skipped"}; low/high are None when no
        replicate was defined.
    """
    rng = np.random.default_rng(seed)
    groups = _clusters(rows, cluster)
    values, skipped = [], 0
    for _ in range(replicates):
        sample = [rows[i] for i in _resample_indices(len(rows), rng, groups)]
        value = statistic(sample)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            skipped += 1
        else:
            values.append(float(value))
    low, high = (
        (float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2)))
        if values else (None, None)
    )
    return {"point": statistic(rows), "low": low, "high": high, "n": len(rows), "skipped": skipped}


def paired_bootstrap(
    rows: Sequence[Row],
    statistic_a: Statistic,
    statistic_b: Statistic,
    *,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    cluster: Callable[[Row], Hashable] | None = None,
) -> dict:
    """
    Interval for statistic_b - statistic_a over the same resampled rows.

    Args:
        rows: Rows carrying both designs' predictions.
        statistic_a: The baseline design's statistic.
        statistic_b: The candidate design's statistic.
        replicates, seed, alpha, cluster: As bootstrap_ci.

    Returns:
        {"delta", "low", "high", "p_better", "n", "skipped"}. p_better is the
        share of resamples where the candidate beats the baseline (ties count
        half) — near 0.5 means the set cannot tell them apart.
    """
    rng = np.random.default_rng(seed)
    groups = _clusters(rows, cluster)
    deltas, skipped = [], 0
    for _ in range(replicates):
        sample = [rows[i] for i in _resample_indices(len(rows), rng, groups)]
        a, b = statistic_a(sample), statistic_b(sample)
        if a is None or b is None:
            skipped += 1
        else:
            deltas.append(float(b) - float(a))
    a, b = statistic_a(rows), statistic_b(rows)
    if not deltas:
        return {"delta": None if a is None or b is None else b - a, "low": None, "high": None,
                "p_better": None, "n": len(rows), "skipped": skipped}
    arr = np.asarray(deltas)
    return {
        "delta": None if a is None or b is None else float(b) - float(a),
        "low": float(np.quantile(arr, alpha / 2)),
        "high": float(np.quantile(arr, 1 - alpha / 2)),
        "p_better": float(((arr > 0).sum() + 0.5 * (arr == 0).sum()) / len(arr)),
        "n": len(rows),
        "skipped": skipped,
    }


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """
    P(a positive scores above a negative), ties counted half.

    Args:
        positives: Scores of the rows that should score high.
        negatives: Scores of the rows that should score low.

    Returns:
        AUC in [0, 1], or None when either side is empty.
    """
    if not positives or not negatives:
        return None
    pos = np.asarray(positives, dtype=float)[:, None]
    neg = np.asarray(negatives, dtype=float)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def f1_counts(tp: int, fp: int, fn: int) -> float | None:
    """F1 from counts; None when there is nothing to find and nothing was flagged."""
    if tp + fp + fn == 0:
        return None
    return 2 * tp / (2 * tp + fp + fn)


def fmt_ci(ci: dict, digits: int = 3) -> str:
    """`0.658 [0.51, 0.78]` — the cell format used in the result tables."""
    point = ci.get("point", ci.get("delta"))
    if point is None:
        return "—"
    if ci.get("low") is None:
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{ci['low']:.2f}, {ci['high']:.2f}]"
