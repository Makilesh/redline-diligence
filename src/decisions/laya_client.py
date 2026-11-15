"""
Laya decision-model client — one lazily loaded model per process.

Laya (convaiinnovations/laya, Apache-2.0) answers typed questions about a piece
of text in a single encoder forward pass: `noul` returns P(true), `choice` a
distribution over named options, `score` a position on an ordinal scale. It
generates no text, so it cannot hallucinate an answer — only be wrong about a
probability, which is measurable.

The default checkpoint is `typed-decisions`, the authors' fine-tune. The base
checkpoints are near chance zero-shot per the model card, so they are not used.
Inputs are truncated at 1024 tokens: pose questions over a chunk or a
(question, passage) pair, never a whole document.

Every caller must keep a non-Laya path. `LAYA_ENABLED=0`, a missing package or a
failed download all raise LayaUnavailable, and the caller falls back to the
rule or heuristic that ran before Laya existed.

The checkpoint is pinned to a Hub commit. laya.Router resolves the bundle repo
at whatever `main` is, and `main` moved between September and November: the
weights happened to be byte-identical, but an upstream change would have moved
every threshold in this package without a line of this repository changing.
The snapshot is therefore resolved here, at LAYA_REVISION, and handed to the
Router as a local path.

Env:
    LAYA_ENABLED   "0" disables every Laya decision (default on).
    LAYA_MODEL     Router checkpoint key (default "typed-decisions").
    LAYA_DEVICE    "cuda" / "cpu"; default picks CUDA when available.
    LAYA_REVISION  Hub commit of convaiinnovations/laya to load (default
                   DEFAULT_REVISION). "main" follows upstream — every number in
                   eval/ must then be re-measured.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_MODEL = "typed-decisions"
# The bundle repo and where each Router key lives in it (laya 0.3.20,
# router.DEFAULT_MODELS). A key outside this table is left to the Router.
BUNDLE_REPO = "convaiinnovations/laya"
BUNDLE_SUBFOLDERS: dict[str, str | None] = {
    "english": None,
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}
# The commit every number in eval/ was measured on (typed-decisions/
# model.safetensors sha256 4fa56de7…a24e, identical at the September commit
# 55cf4c4). Bump it only together with a re-run of the decision evals.
DEFAULT_REVISION = "85cc74ddae6109077ebe2853a9b3077fb6606a78"
# Upper bound on (state, question) sequences per forward pass. Laya's own
# batch_size counts states, and each state is encoded once per question, so a
# fixed state count lets an 11-question ingest pass balloon to 176 sequences —
# enough to exhaust memory on a small CPU host. Batches are sized from this.
MAX_SEQUENCES_PER_BATCH = 64

_router = None
_load_failed = False
# Router() is cheap; the checkpoint loads on the first predict. Until one
# prediction has succeeded, a failure is treated as a load failure and
# remembered, so an offline host is not re-downloaded on every upload.
_has_predicted = False
_load_lock = threading.Lock()
# The Router is not documented as thread-safe; to_thread callers share it.
_predict_lock = threading.Lock()


class LayaUnavailable(RuntimeError):
    """Raised when Laya is disabled or cannot be loaded; callers fall back."""


def laya_enabled() -> bool:
    """True unless LAYA_ENABLED is set to a falsy value."""
    return os.getenv("LAYA_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def laya_model() -> str:
    """The Router checkpoint key in use."""
    return os.getenv("LAYA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def laya_revision() -> str:
    """The Hub commit (or branch) of the bundle repo that is loaded."""
    return os.getenv("LAYA_REVISION", DEFAULT_REVISION).strip() or DEFAULT_REVISION


def _pinned_checkpoint(model: str) -> tuple[str, str | None] | None:
    """
    Local snapshot of `model` at laya_revision(), as a Router model spec.

    Mirrors laya.Agent's own download (same files, same subfolder) with the one
    argument it does not expose: `revision`. A commit already in the Hub cache
    resolves without the network.

    Args:
        model: Router checkpoint key.

    Returns:
        (snapshot directory, subfolder), or None for a key that is not one of
        the bundle's checkpoints.
    """
    if model not in BUNDLE_SUBFOLDERS:
        return None
    from huggingface_hub import snapshot_download

    subfolder = BUNDLE_SUBFOLDERS[model]
    prefix = f"{subfolder}/" if subfolder else ""
    directory = snapshot_download(
        BUNDLE_REPO,
        revision=laya_revision(),
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=[
            prefix + name
            for name in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
        ],
    )
    return directory, subfolder


def laya_device() -> str:
    """The device Laya runs on (or will, once loaded): LAYA_DEVICE, else CUDA when available."""
    return _device()


def _device() -> str:
    configured = os.getenv("LAYA_DEVICE", "").strip()
    if configured:
        return configured
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_router():
    """
    Loads the Laya Router once per process (thread-safe, lazy).

    A load failure is remembered, so a missing package or an offline host costs
    one attempt per process rather than one download attempt per query.

    Returns:
        The laya.Router.

    Raises:
        LayaUnavailable: If disabled, not installed, or failed to load.
    """
    global _router, _load_failed

    if not laya_enabled():
        raise LayaUnavailable("LAYA_ENABLED=0")
    if _router is not None:
        return _router
    if _load_failed:
        raise LayaUnavailable("Laya failed to load earlier in this process")

    with _load_lock:
        if _router is not None:
            return _router
        try:
            from laya import Router

            device = _device()
            pinned = _pinned_checkpoint(laya_model())
            logger.info(
                f"Loading Laya ({laya_model()}) on {device.upper()}",
                extra={"revision": laya_revision() if pinned else "unpinned"},
            )
            # max_loaded=1: only the one checkpoint is ever requested, and each
            # resident checkpoint is ~1.7GB.
            _router = Router(
                models={laya_model(): pinned} if pinned else None,
                device=device,
                max_loaded=1,
            )
            return _router
        except Exception as e:
            _load_failed = True
            raise LayaUnavailable(f"Laya could not be loaded: {e}") from e


def decide_batch(requests: list[tuple[dict, dict]]) -> list[dict[str, dict[str, Any]]]:
    """
    Answers typed questions for many states synchronously. Call via adecide_batch.

    Args:
        requests: (state, questions) pairs. `state` maps field names to text;
            `questions` maps a name to a Laya question dict whose instructions
            refer to state fields in backticks.

    Returns:
        One {question_name: answer} dict per request, in order. A `noul`
        answer carries "noul" (P(true)); a `choice` answer "choice",
        "probabilities" and "confidence".

    Raises:
        LayaUnavailable: If Laya is disabled or cannot be loaded, or inference
            fails.
    """
    global _has_predicted, _load_failed

    if not requests:
        return []
    router = _get_router()
    batch = [{"state": s, "questions": q, "model": laya_model()} for s, q in requests]
    widest = max(len(q) for _, q in requests) or 1
    batch_size = max(1, MAX_SEQUENCES_PER_BATCH // widest)
    try:
        with _predict_lock:
            outputs = router.predict_batch(batch, batch_size=batch_size)
    except Exception as e:
        if not _has_predicted:
            _load_failed = True
        raise LayaUnavailable(f"Laya inference failed: {e}") from e
    _has_predicted = True
    return [out.get("answers", {}) for out in outputs]


async def adecide_batch(requests: list[tuple[dict, dict]]) -> list[dict[str, dict[str, Any]]]:
    """
    Answers typed questions off the event loop.

    Args:
        requests: (state, questions) pairs — see decide_batch.

    Returns:
        One {question_name: answer} dict per request, in order.

    Raises:
        LayaUnavailable: If Laya is disabled or cannot be loaded.
    """
    if not requests:
        return []
    return await asyncio.to_thread(decide_batch, requests)


def noul(answer: dict[str, Any]) -> float:
    """P(true) from a `noul` answer."""
    return float(answer.get("noul", 0.0))


def loaded_model() -> str | None:
    """Checkpoint key if Laya has been loaded in this process, else None."""
    return laya_model() if _router is not None else None


def loaded_revision() -> str | None:
    """Hub revision of the loaded checkpoint (None if not loaded or not a bundle key)."""
    if _router is None or laya_model() not in BUNDLE_SUBFOLDERS:
        return None
    return laya_revision()


async def warm_laya() -> float | None:
    """
    Loads and exercises Laya so the first query or upload does not pay for it.

    Mirrors warm_models / warm_nli_model: failures are logged and swallowed — a
    warmup must never stop the API from starting.

    Returns:
        Seconds taken, or None when disabled or failed.
    """
    if not laya_enabled():
        return None
    start = time.monotonic()
    probe = {"passage": "The Company is not party to any pending litigation."}
    question = {
        "q": {"type": "noul", "instructions": "Does `passage` disclose pending litigation?"}
    }
    try:
        await adecide_batch([(probe, question)])
    except LayaUnavailable as e:
        logger.warning(f"Laya warmup failed: {e}")
        return None
    elapsed = round(time.monotonic() - start, 2)
    logger.info("Laya warmed", extra={"model": laya_model(), "seconds": elapsed})
    return elapsed
