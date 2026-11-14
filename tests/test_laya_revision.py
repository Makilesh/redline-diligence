"""
The Laya checkpoint is loaded at a pinned Hub commit, not at whatever `main` is.

No weights are loaded here: snapshot_download and laya.Router are stubbed, so
these tests check what is ASKED for — the revision, the subfolder, the files.
"""

import sys
import types

import pytest

from src.decisions import laya_client


@pytest.fixture
def downloads(monkeypatch):
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return f"/hub/snapshots/{kwargs['revision']}"

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls


def test_default_revision_is_a_full_commit_sha(monkeypatch):
    monkeypatch.delenv("LAYA_REVISION", raising=False)
    assert laya_client.laya_revision() == laya_client.DEFAULT_REVISION
    assert len(laya_client.DEFAULT_REVISION) == 40
    int(laya_client.DEFAULT_REVISION, 16)  # a commit, not a branch name


def test_pinned_checkpoint_downloads_only_its_subfolder_at_the_revision(monkeypatch, downloads):
    monkeypatch.delenv("LAYA_REVISION", raising=False)
    spec = laya_client._pinned_checkpoint("typed-decisions")

    assert spec == (f"/hub/snapshots/{laya_client.DEFAULT_REVISION}", "typed-decisions")
    (repo, kwargs), = downloads
    assert repo == laya_client.BUNDLE_REPO
    assert kwargs["revision"] == laya_client.DEFAULT_REVISION
    assert all(p.startswith("typed-decisions/") for p in kwargs["allow_patterns"])
    assert "typed-decisions/model.safetensors" in kwargs["allow_patterns"]


def test_revision_override(monkeypatch, downloads):
    monkeypatch.setenv("LAYA_REVISION", "main")
    laya_client._pinned_checkpoint("english")
    (_, kwargs), = downloads
    assert kwargs["revision"] == "main"
    # The root checkpoint's files, without a subfolder prefix.
    assert "model.safetensors" in kwargs["allow_patterns"]


def test_unknown_checkpoint_key_is_left_to_the_router(downloads):
    assert laya_client._pinned_checkpoint("my-own-finetune") is None
    assert downloads == []


def test_router_receives_the_pinned_local_snapshot(monkeypatch, downloads):
    built = {}

    class FakeRouter:
        def __init__(self, models=None, device=None, max_loaded=None):
            built.update(models=models, device=device, max_loaded=max_loaded)

    laya = types.ModuleType("laya")
    laya.Router = FakeRouter
    monkeypatch.setitem(sys.modules, "laya", laya)
    monkeypatch.setenv("LAYA_ENABLED", "1")
    monkeypatch.setenv("LAYA_DEVICE", "cpu")
    monkeypatch.delenv("LAYA_MODEL", raising=False)
    monkeypatch.delenv("LAYA_REVISION", raising=False)
    monkeypatch.setattr(laya_client, "_router", None)
    monkeypatch.setattr(laya_client, "_load_failed", False)

    laya_client._get_router()

    assert built["models"] == {
        "typed-decisions": (f"/hub/snapshots/{laya_client.DEFAULT_REVISION}", "typed-decisions")
    }
    assert built["max_loaded"] == 1
    assert laya_client.loaded_revision() == laya_client.DEFAULT_REVISION
