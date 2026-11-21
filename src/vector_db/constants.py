"""
Shared constants for vector database operations.

IMPORTANT: QUANTIZED_SEARCH_PARAMS is defined here and used in hybrid_search.py.
To avoid implicit cross-module imports, import explicitly:
    from src.vector_db.constants import QUANTIZED_SEARCH_PARAMS
"""

from qdrant_client.models import (
    SearchParams,
    QuantizationSearchParams,
)

# ==============================================================================
# Collection Constants (Immutable Post-Creation)
# ==============================================================================

VECTOR_SIZE: int = 1024  # BAAI/bge-m3 output dimension — DO NOT CHANGE
COLLECTION_NAME: str = "manda_due_diligence"
PARENT_COLLECTION_NAME: str = "manda_parent_chunks"

# ==============================================================================
# Quantization Search Parameters
# ==============================================================================

# Use these search params on EVERY dense vector search call.
# Without rescore=True and oversampling, scalar quantization silently degrades
# recall on the precise numeric chunks most critical for M&A accuracy.
QUANTIZED_SEARCH_PARAMS = SearchParams(
    hnsw_ef=128,            # Higher than ef_construct default — better search recall
    exact=False,
    quantization=QuantizationSearchParams(
        ignore=False,
        rescore=True,       # Re-score top candidates with original float32 vectors
        oversampling=2.0,   # Fetch 2× candidates before rescoring to compensate for
                            # quantization-induced ranking noise
    ),
)

# ==============================================================================
# Qdrant Operation Constants
# ==============================================================================

# Retry configuration for Qdrant operations
QDRANT_MAX_RETRIES: int = 3
QDRANT_BASE_DELAY_S: float = 1.0
QDRANT_MAX_DELAY_S: float = 8.0

# Batch upsert configuration
QDRANT_BATCH_SIZE: int = 100  # vectors per upsert call

# A first-time ingest writes its points with this field in place of deal_id,
# and moves the value to deal_id only once every point is written
# (ingest_pipeline.index_document). Every reader filters on deal_id, so an
# upload cut short by a crash or a cancelled request is never searched,
# listed or counted; deal purges and the sandbox sweeper match this field too.
STAGED_DEAL_FIELD: str = "staged_deal_id"
