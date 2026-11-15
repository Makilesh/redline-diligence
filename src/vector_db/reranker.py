"""
Embedding and reranking model management with async wrappers.

CRITICAL: SentenceTransformer.encode() and CrossEncoder.predict() are synchronous
CPU/GPU operations. Calling them directly inside async functions blocks the entire
asyncio event loop — every other coroutine is frozen until encoding completes.

All calls MUST go through the async wrappers which use run_in_executor with
SEPARATE ThreadPoolExecutor pools to prevent embed/rerank from starving each other.

Uses asyncio.get_running_loop() — NOT get_event_loop() which is deprecated in 3.10+
and broken in 3.12.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ==============================================================================
# Module-level executors — separate pools prevent embed/rerank from starving
# each other. With a single executor of max_workers=2, one request uses both
# threads (embed + rerank), blocking any concurrent request entirely.
# ==============================================================================
_embed_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")
_rerank_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rerank")

# ==============================================================================
# Model selection
# ==============================================================================
# The defaults are the models every measurement in RESULTS.md was produced with.
# They are overridable because the cross-encoder does not fit a CPU-only host:
# measured on 2 vCPU (the free-tier deployment target), bge-reranker-v2-m3 takes
# 111s to score 40 passages, and a decomposed query reranks four times — roughly
# 7.5 minutes per question, against 0.12s for the whole dense embedding step.
#
# Quantization is not enough to close that: int8 buys 2-4x where ~30x is needed.
# Only a smaller cross-encoder does, so the model itself is the knob.
# `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params) measured 3.2s for the same
# 40 passages — 34x — at the cost of being English-only.
#
# The embedding model is deliberately overridable too, but changing it is a much
# bigger commitment than changing the reranker: reranking is query-time only,
# while a new embedding model invalidates every vector already in Qdrant and
# requires a full re-index.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Upper bound on the sequence length the cross-encoder pads pairs to. Halving it
# to 512 is worth 1.6x on its own and costs nothing when chunks are 512 tokens to
# begin with. The length actually used is this or the model's own limit,
# whichever is smaller (see _model_max_length): MiniLM has 512 position
# embeddings, and at 1024 the first pair longer than 512 tokens (594, on the
# golden set) crashed the forward pass, so it used to need RERANKER_MAX_LENGTH=512
# set by hand.
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))

# Half precision for the cross-encoder. "auto" (the default) enables it on CUDA
# only: fp16 on a CPU is slower than fp32, not faster. Set RERANKER_FP16=0 to
# force full precision on a GPU.
RERANKER_FP16 = os.getenv("RERANKER_FP16", "auto").strip().lower()

# Pairs per forward pass. Pairs are sorted by length before batching (see
# _score_pairs), so a batch is padded to the length of its own longest member
# rather than to the longest passage in the whole candidate set — which is why
# small batches win here: fewer short pairs are padded up to a long neighbour.
# Measured on an RTX 5070 Ti laptop GPU in fp16, p50 over the eval questions,
# 20 pairs per pass (one pass / every pass of a decomposed question in one call):
#   batch 4: 116 / 412 ms   batch 8: 137 / 397 ms
#   batch 16: 187 / 414 ms  batch 32: 210 / 534 ms
# A GPU with more headroom may prefer a larger value.
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "8"))

# ==============================================================================
# Model singletons — lazily loaded on first use
# ==============================================================================
_embedding_model: SentenceTransformer | None = None
_reranker_model: CrossEncoder | None = None
# The calibration profile of the loaded cross-encoder (src/decisions/
# reranker_calibration.py), resolved with its effective sequence length.
_reranker_profile = None


def _get_embedding_model() -> SentenceTransformer:
    """
    Lazily loads the configured embedding model on first use.
    Always resident in VRAM (every query uses it).

    The model is EMBEDDING_MODEL_NAME, which defaults to BAAI/bge-m3. Any
    override must produce 1024-dimensional vectors to match VECTOR_SIZE, and
    changing it invalidates everything already indexed.

    Returns:
        SentenceTransformer model on CUDA when available, else CPU.
    """
    global _embedding_model
    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {EMBEDDING_MODEL_NAME} embedding model to {device.upper()}")
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=device,
        )
        logger.info(
            f"{EMBEDDING_MODEL_NAME} embedding model loaded successfully on {device.upper()}"
        )
    return _embedding_model


def _get_reranker_model() -> CrossEncoder:
    """
    Lazily loads the configured cross-encoder on first use.
    Always resident in VRAM (every query uses it).

    The model is RERANKER_MODEL_NAME, which defaults to BAAI/bge-reranker-v2-m3.
    Unlike the embedding model this is safe to change at any time — reranking is
    query-time only and touches nothing already stored.

    CRITICAL: cross-encoders output RAW LOGITS (unbounded) by default.
    Without sigmoid normalization, scores can be negative or >1, which silently
    breaks the Quality Assessor's relevance floors, which are calibrated on
    this [0, 1] scale.
    activation_fct=torch.nn.Sigmoid() normalizes output to [0, 1]. This applies
    to any substituted model, not just the default — swapping the model without
    it would leave every threshold comparing against a different scale, silently.

    The sigmoid puts every model on [0, 1], not on the same scale. The gate's
    thresholds are calibrated per model, so the model's calibration profile is
    resolved here, at its effective sequence length, and an uncalibrated model
    fails to load rather than being gated against another model's floors.

    Returns:
        CrossEncoder model with sigmoid activation for [0,1] normalized scores.

    Raises:
        UncalibratedReranker: No calibration profile for this model and length.
    """
    global _reranker_model, _reranker_profile
    if _reranker_model is None:
        from src.decisions.reranker_calibration import resolve_profile

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Loading {RERANKER_MODEL_NAME} to {device.upper()}",
            extra={"max_length": RERANKER_MAX_LENGTH},
        )
        model = CrossEncoder(
            RERANKER_MODEL_NAME,
            max_length=RERANKER_MAX_LENGTH,
            device=device,
            default_activation_function=torch.nn.Sigmoid(),  # ← MANDATORY for [0,1] scores
        )
        limit = _model_max_length(model)
        max_length = RERANKER_MAX_LENGTH if limit is None else min(RERANKER_MAX_LENGTH, limit)
        if max_length != RERANKER_MAX_LENGTH:
            model.max_length = max_length
        if limit is None:
            logger.warning(
                f"{RERANKER_MODEL_NAME} does not state its sequence limit; "
                "calibration length not checked"
            )
        profile = resolve_profile(RERANKER_MODEL_NAME, None if limit is None else max_length)
        half = _use_half_precision(device)
        if half:
            model.model.half()
        _reranker_model, _reranker_profile = model, profile
        logger.info(
            f"{RERANKER_MODEL_NAME} loaded successfully on {device.upper()} (sigmoid activated)",
            extra={"half_precision": half, "max_length": max_length,
                   "calibration_profile": profile.name},
        )
    return _reranker_model


def _model_max_length(model) -> int | None:
    """
    The longest sequence the model can encode, from its own config and tokenizer.

    The smaller of the tokenizer's model_max_length and the position-embedding
    count. RoBERTa-family encoders (bge-reranker-v2-m3 is XLM-RoBERTa) offset
    positions by padding_idx + 1, so 8194 embeddings hold 8192 tokens.

    Args:
        model: A loaded CrossEncoder.

    Returns:
        The limit in tokens, or None when the model states neither.
    """
    limits = []
    tokenizer_limit = getattr(getattr(model, "tokenizer", None), "model_max_length", None)
    # Tokenizers without a limit report a huge sentinel (int(1e30)).
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        limits.append(tokenizer_limit)
    config = getattr(model, "config", None)
    positions = getattr(config, "max_position_embeddings", None)
    if isinstance(positions, int) and positions > 0:
        if "roberta" in str(getattr(config, "model_type", "")):
            pad = getattr(config, "pad_token_id", 1)
            positions -= (pad if isinstance(pad, int) else 1) + 1
        limits.append(positions)
    return min(limits) if limits else None


def _use_half_precision(device: str) -> bool:
    """
    Whether the cross-encoder weights should be converted to fp16.

    Args:
        device: "cuda" or "cpu", as chosen at load time.

    Returns:
        True on CUDA unless RERANKER_FP16 disables it; never on CPU.
    """
    if RERANKER_FP16 in ("0", "false", "no", "off"):
        return False
    return device == "cuda"


def _score_pairs(pairs: list[tuple[str, str]]) -> np.ndarray:
    """
    Scores (query, passage) pairs with the cross-encoder. Synchronous.

    Pairs are scored longest-first and the result is put back in input order.
    CrossEncoder.predict pads every batch to its longest member, and passage
    length is skewed — on the sample deal a pair is 144 tokens at the median and
    575 at the maximum — so in input order one long passage made every pair in
    its batch as expensive as itself.

    The activation runs in float32 whatever the precision of the weights: fp16
    resolves about 0.0005 near 1.0, which is enough to tie the strongest chunks.

    Args:
        pairs: (query, passage) pairs, any order, queries may differ.

    Returns:
        np.ndarray of shape (len(pairs),), float32, in the order of `pairs`.
    """
    model = _get_reranker_model()
    order = sorted(
        range(len(pairs)),
        key=lambda i: len(pairs[i][0]) + len(pairs[i][1]),
        reverse=True,
    )
    options = {}
    # The model's own activation (sigmoid, set at load), moved to float32. A
    # stand-in model without one keeps whatever its predict() does by default.
    activation = getattr(model, "default_activation_function", None)
    if activation is not None:
        options["activation_fct"] = lambda logits: activation(logits.float())
    sorted_scores = model.predict(
        [list(pairs[i]) for i in order],
        batch_size=RERANKER_BATCH_SIZE,
        show_progress_bar=False,
        **options,
    )
    scores = np.empty(len(pairs), dtype=np.float32)
    scores[order] = np.asarray(sorted_scores, dtype=np.float32).reshape(len(pairs))
    return scores


# ==============================================================================
# Async wrappers — NEVER call model.encode() or model.predict() directly
# in async context. Always use these wrappers.
# ==============================================================================


async def embed_texts_async(texts: list[str]) -> np.ndarray:
    """
    Non-blocking wrapper for synchronous SentenceTransformer.encode().
    Always use this in async context — never call model.encode() directly.

    Returns np.ndarray (not list[list[float]]). Callers must call .tolist()
    when passing to Qdrant or other APIs that expect plain Python lists.

    Args:
        texts: List of text strings to embed.

    Returns:
        np.ndarray of shape (len(texts), 1024) with L2-normalized embeddings.

    Raises:
        RuntimeError: If no running event loop is available.
        ValueError: If texts is empty.
    """
    if not texts:
        raise ValueError("Cannot embed empty text list")

    model = _get_embedding_model()
    loop = asyncio.get_running_loop()  # ← NOT get_event_loop()

    logger.info("Embedding texts asynchronously", extra={"num_texts": len(texts)})

    result = await loop.run_in_executor(
        _embed_executor,
        lambda: model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        ),
    )

    logger.info(
        "Embedding complete",
        extra={"num_texts": len(texts), "output_shape": list(result.shape)},
    )
    return result


async def rerank_async(query: str, passages: list[str]) -> np.ndarray:
    """
    Non-blocking wrapper for synchronous cross-encoder reranking.

    Returns np.ndarray (not list[float]). CrossEncoder.predict() returns
    numpy array. Callers must convert individual scores: float(score).

    Expected score range after sigmoid: [0.0, 1.0]
    Scores > 0.5 indicate positive relevance. The Quality Assessor heuristics
    assume this normalized range.

    Args:
        query: The search query string.
        passages: List of passage texts to score against the query.

    Returns:
        np.ndarray of shape (len(passages),) with sigmoid-normalized scores in [0, 1].

    Raises:
        RuntimeError: If no running event loop is available.
        ValueError: If passages is empty.
    """
    if not passages:
        raise ValueError("Cannot rerank empty passage list")

    return await rerank_pairs_async([(query, p) for p in passages])


async def rerank_pairs_async(pairs: list[tuple[str, str]]) -> np.ndarray:
    """
    Non-blocking cross-encoder scoring of arbitrary (query, passage) pairs.

    Unlike rerank_async the pairs need not share a query, so a decomposed
    question can score the candidates of every sub-question in one call instead
    of one call per pass.

    Args:
        pairs: (query, passage) pairs.

    Returns:
        np.ndarray of shape (len(pairs),) with sigmoid-normalized scores in
        [0, 1], in the order of `pairs`.

    Raises:
        RuntimeError: If no running event loop is available.
        ValueError: If pairs is empty.
    """
    if not pairs:
        raise ValueError("Cannot rerank empty pair list")

    loop = asyncio.get_running_loop()  # ← NOT get_event_loop()

    logger.info("Reranking passages asynchronously", extra={"num_passages": len(pairs)})

    result = await loop.run_in_executor(_rerank_executor, lambda: _score_pairs(pairs))

    logger.info(
        "Reranking complete",
        extra={
            "num_passages": len(pairs),
            "min_score": float(np.min(result)),
            "max_score": float(np.max(result)),
        },
    )
    return result


def get_embed_executor() -> ThreadPoolExecutor:
    """
    Returns the module-level embedding executor for use by other modules
    that need to wrap synchronous embedding-related operations (e.g., BM25).

    Returns:
        ThreadPoolExecutor dedicated to embedding operations.
    """
    return _embed_executor


async def warm_models() -> dict[str, float]:
    """
    Loads and exercises every local model so the first query does not pay for it.

    All three models load lazily on first use: BAAI/bge-m3 for dense embeddings,
    bge-reranker-v2-m3 for cross-encoding, and FastEmbed's BM25 for sparse
    vectors. Together that is several gigabytes of weights, so the first query
    after a restart was tens of seconds slower than every query after it —
    fine for a batch evaluation, bad in front of a live audience where the first
    question is the one being watched.

    Loading is not enough on its own; each model is also run once, because the
    first forward pass allocates buffers and triggers kernel selection that a
    bare constructor call does not.

    Failures are logged and swallowed. A warmup is an optimisation, and it must
    never be the reason the API refuses to start — with one exception: a
    reranker without a calibration profile. That is a configuration error, not
    a load failure, and serving with it would gate every query against another
    model's score scale, so it stops startup here instead of the first query.

    Returns:
        Mapping of model name to seconds taken. Missing keys indicate a failure.

    Raises:
        UncalibratedReranker: The configured reranker has no profile.
    """
    import asyncio
    import time

    from src.decisions.reranker_calibration import UncalibratedReranker
    from src.vector_db.hybrid_search import compute_sparse_bm25

    timings: dict[str, float] = {}
    loop = asyncio.get_running_loop()
    probe = "Aurora Technologies reported total revenue of $452.8 million in FY2023."

    async def _timed(name: str, fn) -> None:
        start = time.monotonic()
        try:
            await loop.run_in_executor(_embed_executor, fn)
            timings[name] = round(time.monotonic() - start, 2)
        except UncalibratedReranker:
            raise
        except Exception as e:
            logger.warning(f"Model warmup failed for {name}: {e}")

    await _timed("dense_embedding", lambda: _get_embedding_model().encode([probe]))
    await _timed("reranker", lambda: _score_pairs([(probe, probe)]))
    await _timed("sparse_bm25", lambda: compute_sparse_bm25(probe))

    logger.info("Local model warmup complete", extra={"seconds": timings})
    return timings
