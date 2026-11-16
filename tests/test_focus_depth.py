"""
The document-focus depth a deployment runs (Decision 71).

Decision 67 chose 30 focus candidates for legal questions with
bge-reranker-v2-m3 on a GPU. MiniLM, the CPU host's reranker, recalled more
with 10 (Decision 71), so a reranker's calibration profile may cap the depth,
and DOCUMENT_FOCUS_K overrides both. An explicit config (a rewrite, an eval's
override) is never re-capped.

Offline: no weights load; profiles are forced with use_profile.
"""

from __future__ import annotations

import pytest

import src.agents.retrieval_executor as rx
import src.agents.retrieval_strategy as rs
from src.agents.retrieval_strategy import RETRIEVAL_CONFIGS, document_focus_depth, get_retrieval_config
from src.decisions import reranker_calibration as rc
from tests.test_document_focus import CORPUS_WIDE, IN_A, _Stubs

LEGAL_DEPTH = RETRIEVAL_CONFIGS["legal"]["document_focus_k"]


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("DOCUMENT_FOCUS_K", raising=False)
    monkeypatch.delenv("RERANKER_PROFILE", raising=False)


def test_gpu_reference_keeps_the_measured_depth():
    with rc.use_profile(rc.REFERENCE_PROFILE):
        assert rc.REFERENCE_PROFILE.document_focus_k is None
        assert get_retrieval_config("legal", {})["document_focus_k"] == LEGAL_DEPTH == 30


def test_cpu_host_profile_caps_the_depth():
    ceiling = rc.MINILM_PROFILE.document_focus_k
    assert ceiling is not None and 0 <= ceiling < LEGAL_DEPTH
    with rc.use_profile(rc.MINILM_PROFILE):
        assert get_retrieval_config("legal", {})["document_focus_k"] == ceiling


def test_cpu_host_depth_is_the_measured_one():
    # Chosen on LegalBench-RAG dev under the emulated 2-core host (Decision 71).
    # Change it only with a new measurement.
    assert rc.MINILM_PROFILE.document_focus_k == 10


def test_a_ceiling_never_raises_the_depth():
    generous = rc.RerankerProfile("generous", "x/y", 512, document_focus_k=100)
    with rc.use_profile(generous):
        assert get_retrieval_config("legal", {})["document_focus_k"] == LEGAL_DEPTH


@pytest.mark.parametrize("query_type", [t for t in RETRIEVAL_CONFIGS if t != "legal"])
def test_types_without_focus_stay_without_it(monkeypatch, query_type):
    monkeypatch.setenv("DOCUMENT_FOCUS_K", "30")
    with rc.use_profile(rc.MINILM_PROFILE):
        assert get_retrieval_config(query_type, {}).get("document_focus_k", 0) == 0
    assert document_focus_depth(0) == 0


@pytest.mark.parametrize("value, expected", [("30", 30), ("0", 0), ("500", 30), ("-4", 0), (" 12 ", 12)])
def test_env_overrides_the_profile_within_bounds(monkeypatch, value, expected):
    monkeypatch.setenv("DOCUMENT_FOCUS_K", value)
    with rc.use_profile(rc.MINILM_PROFILE):
        assert get_retrieval_config("legal", {})["document_focus_k"] == expected


def test_a_malformed_override_is_ignored(monkeypatch):
    monkeypatch.setenv("DOCUMENT_FOCUS_K", "lots")
    with rc.use_profile(rc.MINILM_PROFILE):
        assert get_retrieval_config("legal", {})["document_focus_k"] == rc.MINILM_PROFILE.document_focus_k


def test_an_uncalibrated_reranker_does_not_fail_here(monkeypatch):
    def boom():
        raise rc.UncalibratedReranker("no profile")

    monkeypatch.setattr(rs, "active_profile", boom)
    assert get_retrieval_config("legal", {})["document_focus_k"] == LEGAL_DEPTH


def test_negative_ceiling_is_rejected():
    with pytest.raises(ValueError, match="document_focus_k"):
        rc.RerankerProfile("bad", "x/y", 512, document_focus_k=-1)


def _fresh_state() -> dict:
    # No retrieval_config: the node selects one, as on a first retrieval.
    return {"current_query": "closing condition", "query_type": "legal", "parsed_intent": {},
            "deal_id": "deal-1", "include_pii": False, "rewrite_iteration": 0,
            "extracted_filters": {}, "sub_questions": []}


@pytest.mark.asyncio
async def test_the_node_searches_the_document_at_the_hosts_depth(monkeypatch):
    stubs = _Stubs(monkeypatch, CORPUS_WIDE, {"A": IN_A})
    with rc.use_profile(rc.MINILM_PROFILE):
        out = await rx.retrieval_executor_node(_fresh_state())
    assert [c["top_k"] for c in stubs.focus_calls] == [rc.MINILM_PROFILE.document_focus_k]
    assert out["retrieval_config"]["document_focus_k"] == rc.MINILM_PROFILE.document_focus_k


@pytest.mark.asyncio
async def test_an_explicit_config_is_honoured(monkeypatch):
    """A config already in state (a rewrite, or an eval's override) is not re-capped."""
    stubs = _Stubs(monkeypatch, CORPUS_WIDE, {"A": IN_A})
    state = {**_fresh_state(), "retrieval_config": {**RETRIEVAL_CONFIGS["legal"], "document_focus_k": 30}}
    with rc.use_profile(rc.MINILM_PROFILE):
        await rx.retrieval_executor_node(state)
    assert [c["top_k"] for c in stubs.focus_calls] == [30]
