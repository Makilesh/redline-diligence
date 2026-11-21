"""
Load Hugging Face models from the local cache first.

transformers 4.57 calls the Hub's model_info API every time it loads, by Hub
id, a tokenizer whose vocabulary exceeds 100,000 tokens (its Mistral-regex
check, `PreTrainedTokenizerBase._patch_mistral_regex`), and does not catch a
connection error. That covers four of the models this system loads —
bge-m3, bge-reranker-v2-m3, FactCG-DeBERTa-v3-Large and nli-deberta-v3-xsmall
— so with the Hub unreachable (an outage, a DNS failure, a host without
egress) each failed to load although every file was already in the cache.
A round-7 benchmark crashed this way during a network outage (Decision 84).

Neither `local_files_only=True` nor the cache avoids the call; loading from a
local directory does. So a model that is in the Hub cache is loaded from its
snapshot directory, which also saves the per-load Hub round trips, and only a
model that is not cached is fetched by id as before. Laya resolves its pinned
snapshot the same way (src/decisions/laya_client.py).

A cached model is used as cached: an unpinned model is no longer silently
updated when its Hub repository changes. To move to a new revision, pin it
or clear that model from the cache.

The default models are pinned (PINNED_REVISIONS): each is loaded at the Hub
commit its thresholds, calibration profiles and benchmarks were measured on,
from the cache when that commit is there and from the Hub at that commit when
it is not. A new upload to one of these repositories used to reach a fresh
image build, or a machine with an empty cache, unannounced (audit M5).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

T = TypeVar("T")

# Hub commits of the default models, as measured. Each is the commit the
# development cache's "main" pointed to when the round-12 gates and benchmarks
# ran (every refs/main file predates them), and upstream "main" when it was
# pinned (2026-11-21). Dockerfile.hf bakes the CPU image's three at the same commits
# (its *_REVISION build args; tests/test_model_pins.py keeps the two equal).
# ms-marco-MiniLM-L-6-v2 was renamed upstream to ms-marco-MiniLM-L6-v2 in
# August 2026; the old id redirects, and that commit changed only the README
# (the weights, config and tokenizer are byte-identical to the July commit).
# To move a model: pin the new commit here and in Dockerfile.hf, then re-run
# the benchmarks and the calibration that depend on it.
PINNED_REVISIONS: dict[str, str] = {
    "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "cross-encoder/nli-deberta-v3-xsmall": "a150876415327c80daeff35ca6f68f5ed8cf5c24",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "cross-encoder/nli-deberta-v3-small": "fa2804872c3b4bd748f38c0185cc85775361e735",
}


def pinned_revision(repo_id: str) -> str | None:
    """
    The pinned Hub commit of a default model, or None for any other model.

    Args:
        repo_id: Hub id, or a local directory (never pinned).

    Returns:
        The commit, or None: an override or a local copy is loaded as given.
    """
    return PINNED_REVISIONS.get(repo_id)


def cached_snapshot(repo_id: str, revision: str | None = None) -> str | None:
    """
    The local snapshot directory of a cached model, without touching the network.

    Args:
        repo_id: Hub id ("BAAI/bge-m3"), or a local directory, which is
            returned as is.
        revision: Branch, tag or commit; None means the cached "main".

    Returns:
        The directory, or None when the model (at that revision) is not cached.
    """
    if os.path.isdir(repo_id):
        return repo_id
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id, revision=revision, local_files_only=True)
    except Exception:  # not cached, or no huggingface_hub: the caller fetches by id
        return None


def load_cached_first(load: Callable[[str], T], repo_id: str, revision: str | None = None) -> T:
    """
    Calls `load` on the cached snapshot of `repo_id`, else on the id itself.

    A cached copy that fails to load (an interrupted download that left files
    out) falls back to the id, which fetches what is missing — the behaviour
    before this module. The id is also what a missing model gets.

    Args:
        load: Takes a model path or Hub id and returns the loaded object.
        repo_id: Hub id or local directory.
        revision: Revision the snapshot must be at (a pinned commit); `load`
            is expected to pass the same revision when given the id.

    Returns:
        Whatever `load` returns.
    """
    local = cached_snapshot(repo_id, revision)
    if local is not None and local != repo_id:
        try:
            return load(local)
        except Exception as e:  # noqa: BLE001 — retried by id below
            logger.warning(
                f"Cached copy of {repo_id} did not load; fetching it from the Hub",
                extra={"error": f"{type(e).__name__}: {str(e)[:200]}"},
            )
    return load(repo_id)
