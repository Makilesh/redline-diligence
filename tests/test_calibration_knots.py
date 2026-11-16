"""
The calibration harness's score maps stay valid profiles when a reranker's sigmoid output crowds toward 1.

Ettin's regression logits sit mostly above zero (a median pair near 0.97 after the sigmoid), so distinct quantiles
round to the same 6-decimal knot. The maps used to compare before rounding and emitted a duplicate knot, which
RerankerProfile rejects ("a score map must be strictly increasing") — the harness crashed for ettin-reranker-68m-v1.
"""

import numpy as np

from eval.decisions.run_reranker_calibration import (
    _append_knot,
    candidate_profile,
    probability_matched_map,
    quantile_map,
)


def _strictly_increasing(knots):
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    return all(b > a for a, b in zip(xs, xs[1:])) and all(b > a for a, b in zip(ys, ys[1:]))


def test_append_knot_rounds_before_comparing():
    knots = [(0.0, 0.0), (0.999999, 0.5)]
    _append_knot(knots, 0.9999991, 0.6)  # rounds to the previous x: not a new knot
    assert knots == [(0.0, 0.0), (0.999999, 0.5)]
    _append_knot(knots, 0.9999996, 0.7)  # rounds to 1.0: the end point, not an interior knot
    assert knots == [(0.0, 0.0), (0.999999, 0.5)]


def test_quantile_map_of_saturated_scores_is_a_valid_profile():
    rng = np.random.default_rng(77)
    logits = rng.normal(9.0, 3.0, 2000)  # most of the mass far up the sigmoid
    raw = (1 / (1 + np.exp(-logits))).astype(np.float32).astype(float).tolist()
    reference = rng.beta(0.3, 3.0, 2000).tolist()
    knots = quantile_map(raw, reference)
    assert knots[0] == (0.0, 0.0) and knots[-1] == (1.0, 1.0)
    assert _strictly_increasing(knots)
    candidate_profile("saturated", knots)  # raises if the map is not a valid profile


def test_probability_matched_map_of_saturated_scores_is_a_valid_profile():
    rng = np.random.default_rng(78)
    labels = (rng.random(600) < 0.1).astype(int)
    logits = rng.normal(8.0, 3.0, 600) + 3.0 * labels
    target = (1 / (1 + np.exp(-logits))).tolist()
    reference = np.clip(rng.beta(0.3, 3.0, 600) + 0.5 * labels, 0, 0.999).tolist()
    knots = probability_matched_map(target, reference, labels.tolist())
    assert _strictly_increasing(knots)
    candidate_profile("saturated", knots)


def test_unsaturated_maps_are_unchanged_by_rounding_first():
    # Comparing the rounded value only drops would-be duplicates; a map without collisions is identical.
    rng = np.random.default_rng(79)
    raw = rng.beta(2.0, 5.0, 1500).tolist()  # no two quantiles share 6 decimals
    reference = rng.beta(2.0, 3.0, 1500).tolist()
    knots = quantile_map(raw, reference)
    from eval.decisions.run_reranker_calibration import QUANTILE_LEVELS

    old = [(0.0, 0.0)]  # the previous rule: compare unrounded, store rounded
    for r, b in zip(np.quantile(raw, QUANTILE_LEVELS).tolist(), np.quantile(reference, QUANTILE_LEVELS).tolist()):
        if r > old[-1][0] and b > old[-1][1] and r < 1.0 and b < 1.0:
            old.append((round(r, 6), round(b, 6)))
    old.append((1.0, 1.0))
    assert knots == old
