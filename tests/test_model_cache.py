"""
Models load from the local Hugging Face cache first (Decision 84).

transformers 4.57 calls the Hub's model_info API whenever it loads a
large-vocabulary tokenizer by Hub id and does not catch a connection error, so
with the Hub unreachable bge-m3, bge-reranker-v2-m3, FactCG and the DeBERTa NLI
model all failed to load although they were cached. These tests pin that a
cached model is loaded from its snapshot directory, found without touching the
network, and that every load site in the app goes through that path. No model
weights load here.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

import src.utils.model_cache as mc

COMMIT = "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
OTHER = "1111111111111111111111111111111111111111"


@pytest.fixture
def no_network(monkeypatch):
    """Any HTTP request from huggingface_hub fails the test."""
    def refuse(*args, **kwargs):
        raise AssertionError("the cache lookup must not touch the network")

    monkeypatch.setattr(requests.Session, "send", refuse)


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    """A Hugging Face cache with org/cached at two commits, 'main' at COMMIT."""
    import huggingface_hub.constants as constants

    repo = tmp_path / "models--org--cached"
    for commit in (COMMIT, OTHER):
        snap = repo / "snapshots" / commit
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(COMMIT, encoding="utf-8")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    return repo


# ==============================================================================
# cached_snapshot
# ==============================================================================


def test_a_cached_model_resolves_to_its_snapshot_without_the_network(hub_cache, no_network):
    assert mc.cached_snapshot("org/cached") == str(hub_cache / "snapshots" / COMMIT)


def test_a_pinned_revision_resolves_to_that_commit(hub_cache, no_network):
    assert mc.cached_snapshot("org/cached", OTHER) == str(hub_cache / "snapshots" / OTHER)


def test_an_uncached_model_or_revision_is_not_found(hub_cache, no_network):
    assert mc.cached_snapshot("org/elsewhere") is None
    assert mc.cached_snapshot("org/cached", "2" * 40) is None


def test_a_local_directory_is_its_own_snapshot(tmp_path, no_network):
    assert mc.cached_snapshot(str(tmp_path)) == str(tmp_path)


# ==============================================================================
# load_cached_first
# ==============================================================================


def test_the_loader_gets_the_snapshot_when_cached(monkeypatch):
    seen = []
    monkeypatch.setattr(mc, "cached_snapshot", lambda repo_id, revision=None: f"/cache/{revision}")
    assert mc.load_cached_first(lambda s: seen.append(s) or "model", "org/m", "rev") == "model"
    assert seen == ["/cache/rev"]


def test_the_loader_gets_the_id_when_not_cached(monkeypatch):
    seen = []
    monkeypatch.setattr(mc, "cached_snapshot", lambda repo_id, revision=None: None)
    mc.load_cached_first(seen.append, "org/m")
    assert seen == ["org/m"]


def test_a_cached_copy_that_does_not_load_falls_back_to_the_id(monkeypatch):
    seen = []

    def load(source):
        seen.append(source)
        if source.startswith("/cache"):
            raise OSError("model.safetensors is missing")
        return "model"

    monkeypatch.setattr(mc, "cached_snapshot", lambda repo_id, revision=None: "/cache/snap")
    assert mc.load_cached_first(load, "org/m") == "model"
    assert seen == ["/cache/snap", "org/m"]


def test_a_failure_by_id_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(mc, "cached_snapshot", lambda repo_id, revision=None: None)

    def load(source):
        raise requests.ConnectionError("Hub unreachable")

    with pytest.raises(requests.ConnectionError):
        mc.load_cached_first(load, "org/m")


# ==============================================================================
# Every load site goes through the cache
# ==============================================================================


@pytest.fixture
def snapshot_of(monkeypatch):
    """Pretends every model is cached at /cache/<id>@<revision>; records the lookups."""
    lookups = []

    def fake(repo_id, revision=None):
        lookups.append((repo_id, revision))
        return f"/cache/{repo_id}@{revision}"

    monkeypatch.setattr(mc, "cached_snapshot", fake)
    return lookups


@pytest.fixture
def reranker_module(monkeypatch):
    def load(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        sys.modules.pop("src.vector_db.reranker", None)
        return importlib.import_module("src.vector_db.reranker")

    yield load
    sys.modules.pop("src.vector_db.reranker", None)
    importlib.import_module("src.vector_db.reranker")


def test_the_reranker_loads_from_the_cache_and_keeps_its_profile(reranker_module, snapshot_of):
    from src.decisions import reranker_calibration as rc

    minilm = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    mod = reranker_module(RERANKER_MODEL=minilm, RERANKER_MAX_LENGTH="512")
    fake = MagicMock()
    fake.config = SimpleNamespace(model_type="bert", max_position_embeddings=512, pad_token_id=0)
    fake.tokenizer = SimpleNamespace(model_max_length=512)
    with patch.object(mod, "CrossEncoder", return_value=fake) as ctor:
        mod._reranker_model = None
        mod._get_reranker_model()

    assert ctor.call_args.args[0] == f"/cache/{minilm}@None"
    # The calibration profile follows the configured name, not the load path.
    assert mod._reranker_profile is rc.MINILM_PROFILE


def test_the_embedder_loads_from_the_cache(reranker_module, snapshot_of):
    mod = reranker_module(EMBEDDING_MODEL="BAAI/bge-m3")
    with patch.object(mod, "SentenceTransformer", return_value=MagicMock()) as ctor:
        mod._embedding_model = None
        mod._get_embedding_model()
    assert ctor.call_args.args[0] == "/cache/BAAI/bge-m3@None"


def test_the_grounding_checker_loads_its_pinned_commit_from_the_cache(snapshot_of, monkeypatch):
    import transformers

    from src.verification import nli

    sources = []

    def recorder(kind):
        def from_pretrained(source, *args, **kwargs):
            sources.append((kind, source, kwargs.get("revision")))
            return MagicMock()
        return MagicMock(from_pretrained=from_pretrained)

    monkeypatch.setattr(transformers, "AutoConfig", recorder("config"))
    monkeypatch.setattr(transformers, "AutoTokenizer", recorder("tokenizer"))
    monkeypatch.setattr(transformers, "AutoModelForSequenceClassification", recorder("model"))
    name = "yaxili96/FactCG-DeBERTa-v3-Large"
    pinned = nli._CHECKERS[name]["revision"]

    nli._Checker(name, "cpu")

    assert snapshot_of == [(name, pinned)]
    assert {source for _, source, _ in sources} == {f"/cache/{name}@{pinned}"}
    assert {kind for kind, _, _ in sources} == {"config", "tokenizer", "model"}


def test_the_nli_fallback_loads_from_the_cache(snapshot_of, monkeypatch):
    import sentence_transformers

    from src.verification import nli

    fake = MagicMock()
    fake.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    ctor = MagicMock(return_value=fake)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", ctor)
    monkeypatch.setattr(nli, "_device", lambda: "cpu")
    # _load sets the module's model globals; restored after the test.
    for attr in ("_model", "_model_name", "_model_kind", "_label_index"):
        monkeypatch.setattr(nli, attr, getattr(nli, attr))
    name = "cross-encoder/nli-deberta-v3-xsmall"

    nli._load(name)

    assert ctor.call_args.args[0] == f"/cache/{name}@None"


def test_the_token_counter_loads_from_the_cache(snapshot_of, monkeypatch):
    from src.utils import token_counter as tc

    seen = []
    monkeypatch.setattr(tc, "AutoTokenizer", MagicMock(from_pretrained=lambda s: seen.append(s) or "tok"))
    tc.get_tokenizer.cache_clear()
    try:
        assert tc.get_tokenizer() == "tok"
    finally:
        tc.get_tokenizer.cache_clear()
    assert seen == ["/cache/BAAI/bge-m3@None"]
