"""
Agent 2 — Retrieval Strategy (Deterministic Code — Zero LLM Calls).

This is a pure function, not an LLM agent. Called at the start of the
executor node. Returns retrieval configuration based on query type
and parsed intent signals.

RETRIEVAL_CONFIGS maps query types to optimal dense/sparse weights,
top-k values, reranker thresholds, and expansion flags.

Env:
    DOCUMENT_FOCUS_K  Document-focus depth for the query types that use it
                      (legal), overriding the reranker profile's ceiling;
                      clamped to [0, 30], 0 turns focus off. For a reranker
                      or host its profile was not measured on — see
                      document_focus_depth.
"""

import os

from src.decisions.reranker_calibration import UncalibratedReranker, active_profile
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

RETRIEVAL_CONFIGS: dict[str, dict] = {
    "legal": {
        "dense_weight": 0.4,
        "sparse_weight": 0.6,
        "top_k_dense": 30,
        "top_k_sparse": 40,
        "reranker_top_k": 20,
        "final_top_k": 10,
        "use_parent_expansion": True,
        "use_sibling_expansion": True,
        # Dense candidates from inside the leading document, added to the
        # rerank pool (retrieval_executor._candidates_for_query). Chosen on
        # LegalBench-RAG's dev documents (10 / 20 / 30 simulated); on the
        # held-out half final-context character recall 73.2 -> 85.3%,
        # +12.1 points [+5.2, +21.2], MAUD 21 -> 61% — Decision 67. Measured
        # with bge-reranker-v2-m3; a reranker's profile may lower it
        # (document_focus_depth; MiniLM runs 10, Decision 71).
        "document_focus_k": 30,
    },
    "financial": {
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
        "top_k_dense": 40,
        "top_k_sparse": 40,
        "reranker_top_k": 20,
        "final_top_k": 10,
        "use_parent_expansion": True,
        "use_sibling_expansion": True,
    },
    "comparative": {
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "top_k_dense": 30,
        "top_k_sparse": 30,
        "reranker_top_k": 15,
        "final_top_k": 8,
        "use_parent_expansion": True,
        "use_sibling_expansion": False,
    },
    "summary": {
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "top_k_dense": 40,
        "top_k_sparse": 20,
        "reranker_top_k": 20,
        "final_top_k": 10,
        "use_parent_expansion": True,
        "use_sibling_expansion": False,
    },
    "multi_hop": {
        "dense_weight": 0.55,
        "sparse_weight": 0.45,
        "top_k_dense": 50,
        "top_k_sparse": 50,
        "reranker_top_k": 25,
        "final_top_k": 12,
        "use_parent_expansion": True,
        "use_sibling_expansion": True,
    },
}


# Sane ranges for every retrieval knob. The Query Rewriter (an LLM) may adjust
# these; anything it returns is whitelisted and clamped here, because values
# like reranker_top_k=500 turn one query into minutes of cross-encoder CPU.
RETRIEVAL_CONFIG_BOUNDS: dict[str, tuple[type, float, float]] = {
    "dense_weight": (float, 0.0, 1.0),
    "sparse_weight": (float, 0.0, 1.0),
    "top_k_dense": (int, 5, 100),
    "top_k_sparse": (int, 5, 100),
    "reranker_top_k": (int, 5, 50),
    "final_top_k": (int, 3, 20),
    "document_focus_k": (int, 0, 30),
}
RETRIEVAL_CONFIG_FLAGS = frozenset({"use_parent_expansion", "use_sibling_expansion"})

# Filter keys an LLM may set, with their allowed values (None = remove filter).
REWRITABLE_FILTER_VALUES: dict[str, frozenset] = {
    "document_category": frozenset({
        "financial", "legal", "board", "audit", "regulatory", "operational", "other",
    }),
}


def clamp_retrieval_config(config: dict) -> dict:
    """
    Whitelists and clamps retrieval config values to RETRIEVAL_CONFIG_BOUNDS.

    Unknown keys and non-numeric values are dropped; numbers are coerced to the
    key's type and clamped into range; expansion flags are coerced to bool.

    Args:
        config: Retrieval config (possibly LLM-modified).

    Returns:
        New dict containing only valid, in-range keys.
    """
    clean: dict = {}
    for key, value in (config or {}).items():
        if key in RETRIEVAL_CONFIG_FLAGS:
            if isinstance(value, bool):
                clean[key] = value
            continue
        bounds = RETRIEVAL_CONFIG_BOUNDS.get(key)
        if bounds is None or isinstance(value, bool):
            continue
        cast, low, high = bounds
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number:  # NaN
            continue
        number = min(max(number, low), high)
        clean[key] = int(round(number)) if cast is int else float(number)
    return clean


def apply_config_overrides(base: dict, overrides: dict | None) -> dict:
    """
    Merges LLM-proposed overrides into a config, keeping only safe values.

    Args:
        base: Current retrieval config.
        overrides: Proposed updates (e.g. the rewriter's updated_retrieval_config).

    Returns:
        Merged, clamped config. Base keys the overrides do not touch are kept.
    """
    merged = dict(base or {})
    if isinstance(overrides, dict):
        merged.update(clamp_retrieval_config(overrides))
    # Re-clamp the whole result too: the base may itself have come from state.
    return {**merged, **clamp_retrieval_config(merged)}


def apply_filter_overrides(base: dict, overrides: dict | None) -> dict:
    """
    Merges LLM-proposed metadata filter updates, whitelisting keys and values.

    A value of None removes that filter (widening the search). include_pii,
    is_current_version and any other key are never accepted from an LLM.

    Args:
        base: Current extracted filters.
        overrides: Proposed updates (e.g. the rewriter's updated_metadata_filters).

    Returns:
        New filters dict.
    """
    merged = dict(base or {})
    if not isinstance(overrides, dict):
        return merged
    for key, value in overrides.items():
        allowed = REWRITABLE_FILTER_VALUES.get(key)
        if allowed is None:
            continue
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, str) and value in allowed:
            merged[key] = value
    return merged


def document_focus_depth(configured: int) -> int:
    """
    The document-focus depth this deployment runs for a query type configured
    with `configured` (0 = the type does not focus, and stays off).

    The depth in RETRIEVAL_CONFIGS was chosen with bge-reranker-v2-m3 on a GPU.
    Whether more in-document candidates help depends on how well the reranker
    orders them, and what they cost on the host it runs on, so the reranker's
    calibration profile may cap it: MiniLM, the CPU host's reranker, recalled
    more with 10 than with 30 (Decision 71). DOCUMENT_FOCUS_K overrides both,
    for a reranker or host the profile was not measured on.

    Args:
        configured: The query type's document_focus_k.

    Returns:
        The depth to use, in RETRIEVAL_CONFIG_BOUNDS' range.
    """
    if configured <= 0:
        return 0
    _, low, high = RETRIEVAL_CONFIG_BOUNDS["document_focus_k"]
    override = os.getenv("DOCUMENT_FOCUS_K", "").strip()
    if override:
        try:
            return int(min(max(int(override), low), high))
        except ValueError:
            logger.warning("DOCUMENT_FOCUS_K is not an integer; ignored",
                           extra={"value": override})
    try:
        ceiling = getattr(active_profile(), "document_focus_k", None)
    except UncalibratedReranker:
        # The reranker itself refuses to load for the same reason, loudly;
        # the depth is not the place to report it.
        ceiling = None
    return configured if ceiling is None else min(configured, ceiling)


def get_retrieval_config(query_type: str, parsed_intent: dict) -> dict:
    """
    Deterministic retrieval config selection. No LLM.
    Augments base config with intent signals from Agent 1.
    Called ONCE on first retrieval, then state["retrieval_config"] is used
    on rewrite iterations (the rewriter may have modified it).

    Args:
        query_type: One of "financial", "legal", "comparative", "summary", "multi_hop".
        parsed_intent: Parsed intent dict from Agent 1, containing
                       requires_numerical_precision and requires_cross_document.

    Returns:
        Dict with retrieval configuration parameters.

    Raises:
        KeyError: If query_type is not in RETRIEVAL_CONFIGS.
    """
    if query_type not in RETRIEVAL_CONFIGS:
        logger.warning(
            f"Unknown query_type '{query_type}', defaulting to 'summary'",
            extra={"query_type": query_type},
        )
        query_type = "summary"

    config = RETRIEVAL_CONFIGS[query_type].copy()
    if "document_focus_k" in config:
        config["document_focus_k"] = document_focus_depth(config["document_focus_k"])

    # Augment with intent signals
    if parsed_intent.get("requires_cross_document"):
        config["top_k_dense"] = min(config["top_k_dense"] + 10, 60)
        config["top_k_sparse"] = min(config["top_k_sparse"] + 10, 60)
        logger.info(
            "Cross-document retrieval required, increased top-k",
            extra={
                "top_k_dense": config["top_k_dense"],
                "top_k_sparse": config["top_k_sparse"],
            },
        )

    logger.info(
        "Retrieval config selected",
        extra={"query_type": query_type, "config": config},
    )
    return config
