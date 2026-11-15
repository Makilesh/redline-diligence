"""
The answerability gate's CPU early exit must never change its verdict.

The veto fires only when EVERY (facet, passage) pair is below the threshold, so
scoring can stop at the first pair that clears it. On a CPU that is the
difference between one ModernBERT-large pass and up to fifteen. These tests pin
that the shortcut is exact: same veto, same unanswered facets whenever it
vetoes, and every pair scored before a veto. Laya is stubbed.
"""

import random

import pytest

import src.decisions.answerability as ans


def _chunks(*scores):
    return [{"chunk_id": f"c{i}", "text": f"passage {i}", "reranker_score": s}
            for i, s in enumerate(scores)]


def _stub_laya(monkeypatch, table: dict[tuple[str, str], float]):
    calls = []

    async def fake(requests):
        calls.append(len(requests))
        return [{"stated": {"type": "noul", "noul": table[(s["question"], s["passage"])]}}
                for s, _ in requests]

    monkeypatch.setattr(ans, "adecide_batch", fake)
    return calls


@pytest.mark.asyncio
async def test_early_exit_stops_at_the_first_pair_that_clears_the_veto(monkeypatch):
    monkeypatch.setenv("LAYA_GATE_EARLY_EXIT", "1")
    passages = [f"passage {i}" for i in range(3)]
    table = {(f, p): 0.1 for f in ("Q?", "sub?") for p in passages}
    table[("sub?", "passage 0")] = 0.6
    calls = _stub_laya(monkeypatch, table)
    verdict = await ans.assess_answerability("Q?", ["sub?"], _chunks(0.9, 0.8, 0.7))
    # Best passage first, question before sub-question: Q? x p0, then sub? x p0 clears it.
    assert calls == [1, 1]
    assert not verdict.vetoed and verdict.pairs_scored == 2 and verdict.early_exit


@pytest.mark.asyncio
async def test_early_exit_scores_every_pair_before_vetoing(monkeypatch):
    monkeypatch.setenv("LAYA_GATE_EARLY_EXIT", "1")
    passages = [f"passage {i}" for i in range(3)]
    table = {(f, p): 0.2 for f in ("Q?", "sub?") for p in passages}
    calls = _stub_laya(monkeypatch, table)
    verdict = await ans.assess_answerability("Q?", ["sub?"], _chunks(0.9, 0.8, 0.7))
    assert sum(calls) == 6 and verdict.vetoed and not verdict.early_exit
    assert verdict.unanswered == ["Q?", "sub?"]


@pytest.mark.asyncio
async def test_batched_mode_is_one_call(monkeypatch):
    monkeypatch.setenv("LAYA_GATE_EARLY_EXIT", "0")
    passages = [f"passage {i}" for i in range(3)]
    calls = _stub_laya(monkeypatch, {(f, p): 0.9 for f in ("Q?", "sub?") for p in passages})
    verdict = await ans.assess_answerability("Q?", ["sub?"], _chunks(0.9, 0.8, 0.7))
    assert calls == [6] and verdict.pairs_scored == 6 and not verdict.early_exit


@pytest.mark.asyncio
async def test_early_exit_never_changes_the_verdict(monkeypatch):
    rng = random.Random(7)
    facets = ["Q?", "a?", "b?"]
    passages = [f"passage {i}" for i in range(3)]
    for _ in range(200):
        table = {(f, p): rng.choice([0.05, 0.2, 0.34, 0.35, 0.36, 0.6, 0.9])
                 for f in facets for p in passages}
        verdicts = []
        for mode in ("0", "1"):
            monkeypatch.setenv("LAYA_GATE_EARLY_EXIT", mode)
            _stub_laya(monkeypatch, table)
            verdicts.append(await ans.assess_answerability("Q?", facets[1:], _chunks(0.9, 0.8, 0.7)))
        batched, early = verdicts
        assert batched.vetoed == early.vetoed
        assert (batched.score >= ans.VETO_THRESHOLD) == (early.score >= ans.VETO_THRESHOLD)
        if batched.vetoed:
            assert early.facet_scores == batched.facet_scores
            assert early.unanswered == batched.unanswered


@pytest.mark.parametrize("device,env,expected", [
    ("cpu", None, True), ("cuda", None, False), ("cuda", "1", True), ("cpu", "0", False),
])
def test_early_exit_defaults_to_the_laya_device(monkeypatch, device, env, expected):
    monkeypatch.setattr(ans, "laya_device", lambda: device)
    if env is None:
        monkeypatch.delenv("LAYA_GATE_EARLY_EXIT", raising=False)
    else:
        monkeypatch.setenv("LAYA_GATE_EARLY_EXIT", env)
    assert ans.early_exit_enabled() is expected
