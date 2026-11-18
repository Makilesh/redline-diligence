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
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

T = TypeVar("T")


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
