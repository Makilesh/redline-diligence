"""
Per-reranker calibration of the context gate, and the reranker's sequence length.

The gate's thresholds sit on bge-reranker-v2-m3's score scale. These tests pin
the properties that make swapping the cross-encoder safe:

  1. bge's profile is the identity, so the measured configuration is unchanged.
  2. Every other profile is a strictly increasing map onto that scale — it can
     move a score across a threshold but never reorder two passages.
  3. A reranker without a profile fails loudly (at load, and so at startup)
     instead of being gated against another model's floors.
  4. The sequence length comes from the model's own limits, so MiniLM no
     longer needs RERANKER_MAX_LENGTH=512 to avoid crashing.

No model weights load here: CrossEncoder is stubbed.
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import src.agents.quality_assessor as qa
from src.decisions import reranker_calibration as rc
from src.workflow import conditional_edges as edges

MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE = "BAAI/bge-reranker-v2-m3"


# ==============================================================================
# Profiles
# ==============================================================================


def test_reference_profile_is_the_identity():
    grid = [i / 1000 for i in range(1001)] + [1e-6, 3.3e-5, 0.0999, 0.30001]
    assert all(rc.REFERENCE_PROFILE.calibrate(s) == s for s in grid)
    assert rc.REFERENCE_PROFILE.model == BGE and rc.REFERENCE_PROFILE.max_length == 1024


@pytest.mark.parametrize("profile", [p for p in rc.PROFILES if p.knots], ids=lambda p: p.name)
def test_score_maps_are_strictly_increasing_onto_unit_interval(profile):
    grid = sorted({i / 20000 for i in range(20001)} | {k[0] for k in profile.knots})
    mapped = [profile.calibrate(s) for s in grid]
    assert mapped[0] == 0.0 and mapped[-1] == 1.0
    assert all(b > a for a, b in zip(mapped, mapped[1:]))
    # Out-of-range input is clamped, never extrapolated.
    assert profile.calibrate(-0.5) == 0.0 and profile.calibrate(1.5) == 1.0


def test_minilm_profile_is_derived_not_a_placeholder():
    profile = rc.resolve_profile(MINILM, 512)
    assert profile.knots, "MiniLM must carry a measured map, not the identity"
    assert "answerability_dev" in profile.derivation


@pytest.mark.parametrize("knots", [
    ((0.0, 0.0), (0.5, 0.4), (0.4, 0.6), (1.0, 1.0)),   # x not increasing
    ((0.0, 0.0), (0.4, 0.5), (0.5, 0.5), (1.0, 1.0)),   # y not increasing
    ((0.1, 0.0), (1.0, 1.0)),                           # does not start at the origin
])
def test_malformed_maps_are_rejected(knots):
    with pytest.raises(ValueError):
        rc.RerankerProfile("bad", "x/y", 512, knots=knots)


def test_known_models_resolve(monkeypatch):
    monkeypatch.delenv("RERANKER_PROFILE", raising=False)
    assert rc.resolve_profile(BGE, 1024) is rc.REFERENCE_PROFILE
    assert rc.resolve_profile(MINILM, 512) is rc.MINILM_PROFILE
    assert rc.resolve_profile(MINILM) is rc.MINILM_PROFILE  # length unknown: not checked


def test_unknown_reranker_fails_loudly(monkeypatch):
    monkeypatch.delenv("RERANKER_PROFILE", raising=False)
    with pytest.raises(rc.UncalibratedReranker, match="No calibration profile"):
        rc.resolve_profile("cross-encoder/ms-marco-TinyBERT-L-2-v2", 512)


def test_known_reranker_at_another_length_fails_loudly(monkeypatch):
    monkeypatch.delenv("RERANKER_PROFILE", raising=False)
    with pytest.raises(rc.UncalibratedReranker, match="max_length 1024"):
        rc.resolve_profile(BGE, 512)


def test_override_asserts_a_profile_and_rejects_unknown_names(monkeypatch):
    monkeypatch.setenv("RERANKER_PROFILE", rc.MINILM_PROFILE.name)
    assert rc.resolve_profile("someone/minilm-finetune", 256) is rc.MINILM_PROFILE
    monkeypatch.setenv("RERANKER_PROFILE", "nonexistent")
    with pytest.raises(rc.UncalibratedReranker, match="names no profile"):
        rc.resolve_profile(BGE, 1024)


# ==============================================================================
# The gate reads calibrated scores
# ==============================================================================


def _chunks(*scores):
    return [{"chunk_id": f"c{i}", "text": f"passage {i}", "reranker_score": s}
            for i, s in enumerate(scores)]


def test_assessor_maps_scores_through_the_active_profile():
    chunks = _chunks(0.004, 0.2, 0.9)
    with rc.use_profile(rc.MINILM_PROFILE):
        mapped = qa._calibrated_scores(chunks)
    assert mapped == [rc.MINILM_PROFILE.calibrate(s) for s in (0.004, 0.2, 0.9)]
    with rc.use_profile(rc.REFERENCE_PROFILE):
        assert qa._calibrated_scores(chunks) == [0.004, 0.2, 0.9]


def test_default_reranker_gate_is_unchanged():
    """No profile forced, nothing loaded: the configured default (bge) is the identity."""
    from src.vector_db import reranker

    assert reranker.RERANKER_MODEL_NAME == BGE
    assert rc.active_profile() is rc.REFERENCE_PROFILE
    state = {"query_type": "financial", "reranked_results": _chunks(0.95, 0.04, 0.2)}
    out = qa._heuristic_assessment(state)
    assert out["quality_breakdown"]["relevance"] == 0.95
    assert out["quality_breakdown"]["precision"] == round((0.95 + 0.2) / 2, 3)


def test_profile_decides_which_side_of_a_floor_a_raw_score_lands():
    """The same raw best score is noise to bge and a weak-but-real hit to MiniLM, or vice versa."""
    raw = next(s / 10000 for s in range(1, 10000)
               if (rc.MINILM_PROFILE.calibrate(s / 10000) >= rc.CONFIDENT_REFUSAL_CEILING)
               != (s / 10000 >= rc.CONFIDENT_REFUSAL_CEILING))
    state = {"query_type": "financial", "reranked_results": _chunks(raw)}
    with rc.use_profile(rc.REFERENCE_PROFILE):
        bge = qa._heuristic_assessment(state)
    with rc.use_profile(rc.MINILM_PROFILE):
        minilm = qa._heuristic_assessment(state)
    assert (bge is not None and bge["force_refusal"]) != (minilm is not None and minilm["force_refusal"])


def test_router_thresholds_come_from_the_common_scale():
    for thresholds in edges.THRESHOLDS.values():
        assert thresholds == rc.ROUTER_THRESHOLDS
    assert qa.RELEVANCE_FLOOR == rc.RELEVANCE_FLOOR
    assert qa.CONFIDENT_REFUSAL_CEILING == rc.CONFIDENT_REFUSAL_CEILING
    assert qa.CONFIDENT_PASS_FLOOR == rc.CONFIDENT_PASS_FLOOR


# ==============================================================================
# Reranker loading: sequence length and the profile check
# ==============================================================================


@pytest.fixture
def reranker_module(monkeypatch):
    """A fresh reranker module per test, restored afterwards."""
    def load(**env):
        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        monkeypatch.delenv("RERANKER_PROFILE", raising=False)
        sys.modules.pop("src.vector_db.reranker", None)
        return importlib.import_module("src.vector_db.reranker")

    yield load
    # Restore the environment BEFORE re-importing: monkeypatch's own teardown
    # runs after this one, and the module reads RERANKER_MODEL at import time.
    monkeypatch.undo()
    sys.modules.pop("src.vector_db.reranker", None)
    importlib.import_module("src.vector_db.reranker")


def _fake_cross_encoder(model_type: str, positions: int, tokenizer_limit, pad: int = 0):
    model = MagicMock()
    model.config = SimpleNamespace(model_type=model_type, max_position_embeddings=positions,
                                   pad_token_id=pad)
    model.tokenizer = SimpleNamespace(model_max_length=tokenizer_limit)
    return model


@pytest.mark.parametrize("model_type,positions,tokenizer_limit,pad,expected", [
    ("bert", 512, 512, 0, 512),                 # ms-marco-MiniLM-L-6-v2
    ("xlm-roberta", 8194, 8192, 1, 8192),       # bge-reranker-v2-m3: positions offset by pad+1
    ("roberta", 514, int(1e30), 1, 512),        # tokenizer without a limit (sentinel)
    ("bert", 512, 256, 0, 256),
])
def test_model_max_length_reads_the_models_own_limits(
        reranker_module, model_type, positions, tokenizer_limit, pad, expected):
    mod = reranker_module()
    model = _fake_cross_encoder(model_type, positions, tokenizer_limit, pad)
    assert mod._model_max_length(model) == expected


def test_minilm_loads_at_the_default_length_without_crashing(reranker_module):
    """RERANKER_MAX_LENGTH unset (1024): MiniLM is clipped to its 512 positions."""
    mod = reranker_module(RERANKER_MODEL=MINILM, RERANKER_MAX_LENGTH=None)
    fake = _fake_cross_encoder("bert", 512, 512)
    with patch.object(mod, "CrossEncoder", return_value=fake) as ctor:
        mod._reranker_model = None
        model = mod._get_reranker_model()

    assert ctor.call_args.kwargs["max_length"] == 1024
    assert model.max_length == 512
    assert mod._reranker_profile is rc.MINILM_PROFILE
    assert rc.active_profile() is rc.MINILM_PROFILE


def test_bge_keeps_its_measured_length(reranker_module):
    mod = reranker_module(RERANKER_MODEL=None, RERANKER_MAX_LENGTH=None)
    fake = _fake_cross_encoder("xlm-roberta", 8194, 8192, pad=1)
    fake.max_length = 1024
    with patch.object(mod, "CrossEncoder", return_value=fake):
        mod._reranker_model = None
        model = mod._get_reranker_model()
    assert model.max_length == 1024 and mod._reranker_profile is rc.REFERENCE_PROFILE


def test_uncalibrated_reranker_does_not_load(reranker_module):
    mod = reranker_module(RERANKER_MODEL="cross-encoder/ms-marco-TinyBERT-L-2-v2",
                          RERANKER_MAX_LENGTH=None)
    with patch.object(mod, "CrossEncoder", return_value=_fake_cross_encoder("bert", 512, 512)):
        mod._reranker_model = None
        with pytest.raises(rc.UncalibratedReranker):
            mod._get_reranker_model()
    assert mod._reranker_model is None and mod._reranker_profile is None


@pytest.mark.asyncio
async def test_uncalibrated_reranker_stops_startup(reranker_module, monkeypatch):
    """warm_models swallows load failures — except this one, which is configuration."""
    mod = reranker_module(RERANKER_MODEL="cross-encoder/ms-marco-TinyBERT-L-2-v2",
                          RERANKER_MAX_LENGTH=None)
    monkeypatch.setattr(mod, "_get_embedding_model", lambda: MagicMock())
    with patch.object(mod, "CrossEncoder", return_value=_fake_cross_encoder("bert", 512, 512)):
        mod._reranker_model = None
        with pytest.raises(rc.UncalibratedReranker):
            await mod.warm_models()
