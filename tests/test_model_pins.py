"""
The default models load at pinned Hub commits, and the image bakes the same ones.

Audit M5: the CPU image fetched bge-m3, the MiniLM reranker and the xsmall NLI
model at whatever their repositories' `main` was on the day of the build, and
an empty cache fetched them the same way at runtime. The thresholds, the
reranker calibration profiles and every benchmark were measured on particular
commits, so a new upload could move them unannounced. No weights are loaded
here: these tests check what is asked for.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import src.utils.model_cache as mc

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile.hf"


def _build_args() -> dict[str, str]:
    return dict(re.findall(r"(?m)^ARG\s+(\w+)=(\S+)\s*$", DOCKERFILE.read_text(encoding="utf-8")))


def test_every_pin_is_a_full_commit():
    for repo, commit in mc.PINNED_REVISIONS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", commit), repo  # a commit, not a branch name


@pytest.mark.parametrize(
    ("model_arg", "revision_arg"),
    [
        ("EMBEDDING_MODEL", "EMBEDDING_REVISION"),
        ("RERANKER_MODEL", "RERANKER_REVISION"),
        ("NLI_MODEL", "NLI_REVISION"),
    ],
)
def test_the_image_bakes_the_commit_the_app_loads(model_arg, revision_arg):
    args = _build_args()
    model = args[model_arg]
    assert args[revision_arg] == mc.PINNED_REVISIONS[model], (
        f"Dockerfile.hf {revision_arg} differs from PINNED_REVISIONS[{model!r}]"
    )


def test_the_image_fetches_each_model_at_its_build_arg():
    text = DOCKERFILE.read_text(encoding="utf-8")
    for model_arg, revision_arg in (
        ("EMBEDDING_MODEL", "EMBEDDING_REVISION"),
        ("RERANKER_MODEL", "RERANKER_REVISION"),
        ("NLI_MODEL", "NLI_REVISION"),
    ):
        assert f"('${{{model_arg}}}', revision='${{{revision_arg}}}')" in text, model_arg
    # No unpinned fetch of a default model is left.
    assert not re.search(r"(?:SentenceTransformer|CrossEncoder)\('\$\{\w+\}'\)", text)


def test_the_image_points_main_at_each_pin():
    """A download at a commit writes no refs/main; the image writes it, so an
    unpinned load by id in the image (found by the round-13 Docker check's
    smoke, offline) resolves to the pin instead of failing or fetching main."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    step = text[text.index("refs/main at the pins") - 1200:text.index("refs/main at the pins")]
    for model_arg, revision_arg in (
        ("EMBEDDING_MODEL", "EMBEDDING_REVISION"),
        ("RERANKER_MODEL", "RERANKER_REVISION"),
        ("NLI_MODEL", "NLI_REVISION"),
    ):
        assert f"('${{{model_arg}}}', '${{{revision_arg}}}')" in step, model_arg
    assert "'refs'" in step and "'main'" in step and "assert not missing" in step


def test_an_override_or_a_local_directory_is_not_pinned(tmp_path):
    assert mc.pinned_revision("my-org/my-reranker") is None
    assert mc.pinned_revision(str(tmp_path)) is None


def test_a_model_missing_from_the_cache_is_fetched_at_its_pin(monkeypatch):
    """The by-id fallback asks the Hub for the pinned commit, not for main."""
    asked = []
    monkeypatch.setattr(mc, "cached_snapshot", lambda repo_id, revision=None: asked.append(revision))

    import src.utils.token_counter as tc

    seen = []
    monkeypatch.setattr(
        tc, "AutoTokenizer",
        MagicMock(from_pretrained=lambda s, revision=None: seen.append((s, revision)) or "tok"),
    )
    tc.get_tokenizer.cache_clear()
    try:
        tc.get_tokenizer()
    finally:
        tc.get_tokenizer.cache_clear()

    pinned = mc.PINNED_REVISIONS["BAAI/bge-m3"]
    assert asked == [pinned]
    assert seen == [("BAAI/bge-m3", pinned)]


def test_the_cached_lookup_asks_for_the_commit(monkeypatch, tmp_path):
    """cached_snapshot hands the revision to the Hub cache lookup, offline."""
    import huggingface_hub

    calls = []

    def fake_snapshot_download(repo_id, revision=None, local_files_only=False):
        calls.append((repo_id, revision, local_files_only))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    pinned = mc.PINNED_REVISIONS["BAAI/bge-m3"]
    assert mc.cached_snapshot("BAAI/bge-m3", pinned) == str(tmp_path)
    assert calls == [("BAAI/bge-m3", pinned, True)]
