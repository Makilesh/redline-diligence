"""
The calibration harness can calibrate a reranker that has no profile yet (Decision 87).

Loading an unlisted reranker raises UncalibratedReranker, and the error tells the
operator to derive a profile with eval/decisions/run_reranker_calibration.py — which
loaded the target the same way and failed with the same error, so Decisions 77 and 87
registered profiles by hand first. The target is now collected under an identity
placeholder, asserted for its collection only, at the legal focus depth the caller
gives. No model loads here: the index, the warm-up and the collection are stubbed.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from eval.decisions import run_reranker_calibration as cal
from src.decisions import reranker_calibration as rc

NEW = "D:/checkpoints/a-new-reranker"
MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("RERANKER_PROFILE", raising=False)
    monkeypatch.delenv("DOCUMENT_FOCUS_K", raising=False)


def test_a_new_target_is_collected_under_a_placeholder(clean_env):
    profiles = rc.PROFILES
    with pytest.raises(rc.UncalibratedReranker):
        rc.resolve_profile(NEW, 512)
    with cal.collection_profile(NEW, 10) as profile:
        assert profile.name == cal.PLACEHOLDER_PROFILE and not profile.knots
        assert profile.document_focus_k == 10
        assert rc.resolve_profile(NEW, 512) is profile
    # Gone afterwards: the placeholder never outlives the collection.
    assert rc.PROFILES is profiles
    with pytest.raises(rc.UncalibratedReranker):
        rc.resolve_profile(NEW, 512)


def test_the_placeholder_is_removed_when_collection_fails(clean_env):
    profiles = rc.PROFILES
    with pytest.raises(RuntimeError), cal.collection_profile(NEW, None):
        raise RuntimeError("collection failed")
    assert rc.PROFILES is profiles
    with pytest.raises(rc.UncalibratedReranker):
        rc.resolve_profile(NEW, 512)


def test_an_operators_assertion_is_respected(monkeypatch):
    import os

    monkeypatch.setenv("RERANKER_PROFILE", rc.MINILM_PROFILE.name)
    with cal.collection_profile(NEW, 10) as profile:
        # An asserted profile is the operator's statement about the target: it is used, not replaced.
        assert profile is rc.MINILM_PROFILE
    assert os.environ["RERANKER_PROFILE"] == rc.MINILM_PROFILE.name


def test_a_profiled_target_keeps_its_profile(clean_env):
    profiles = rc.PROFILES
    with cal.collection_profile(MINILM, 30) as profile:
        assert profile is rc.MINILM_PROFILE
    assert rc.PROFILES is profiles


def test_collect_reaches_a_new_target_at_the_requested_depth(clean_env, monkeypatch):
    from src.agents.retrieval_strategy import document_focus_depth
    from src.vector_db import reranker

    for name in ("RERANKER_MODEL_NAME", "RERANKER_MAX_LENGTH", "RERANKER_FP16",
                 "_reranker_model", "_reranker_profile"):
        monkeypatch.setattr(reranker, name, getattr(reranker, name))
    depths = {}

    async def warm_models():
        # What loading does: resolve the configured model's profile at its length (raises if none).
        rc.resolve_profile(reranker.RERANKER_MODEL_NAME, reranker.RERANKER_MAX_LENGTH)
        depths[reranker.RERANKER_MODEL_NAME] = document_focus_depth(30)
        return {}

    async def build_index():
        return object(), None

    async def collect_one(questions):
        return {"pairs": {"[\"q\", \"p\"]": 0.5}, "contexts": {}}

    monkeypatch.setattr(reranker, "warm_models", warm_models)
    monkeypatch.setattr(cal, "build_index", build_index)
    monkeypatch.setattr(cal, "use_client", lambda client: contextlib.nullcontext())
    monkeypatch.setattr(cal, "_collect_one", collect_one)

    out = asyncio.run(cal.collect([], NEW, "0", target_focus_k=10))

    assert depths == {cal.REFERENCE["model"]: 30, NEW: 10}
    assert out["target"]["collected_under"] == {"profile": cal.PLACEHOLDER_PROFILE, "document_focus_k": 10}
    assert "collected_under" not in out["reference"]
    with pytest.raises(rc.UncalibratedReranker):
        rc.resolve_profile(NEW, 512)
