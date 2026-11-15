"""
Hybrid search module — BM25 sparse vectors and search execution.

FastEmbed BM25 sparse vector computation for Qdrant native sparse support.
The full hybrid search function is implemented here alongside the BM25 encoder.

Metadata constraints (fiscal year, document category, etc.) are passed as
Qdrant filter parameters directly into the dense and sparse search calls.
There is NO third "metadata retrieval channel" via scroll.
"""

import asyncio
import re
import time
import unicodedata

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    SparseVector,
    ScoredPoint,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

from src.vector_db.constants import (
    COLLECTION_NAME,
    QUANTIZED_SEARCH_PARAMS,
)
from src.vector_db.qdrant_client import get_qdrant_client
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ==============================================================================
# BM25 Sparse Embedding — Module-level model, load once at startup
# ==============================================================================

_bm25_model: SparseTextEmbedding | None = None


def _get_bm25_model() -> SparseTextEmbedding:
    """
    Lazily loads the BM25 sparse embedding model.

    Returns:
        SparseTextEmbedding model for Qdrant/bm25.
    """
    global _bm25_model
    if _bm25_model is None:
        logger.info("Loading Qdrant/bm25 sparse embedding model")
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("Qdrant/bm25 sparse embedding model loaded successfully")
    return _bm25_model


# Version of the text normalisation applied before BM25 encoding. Stored on
# every child point as `sparse_encoding` so that vectors written by an older
# encoding can be found: a query encoded one way does not match a document
# encoded another, and nothing else would say so.
#   1 (or absent)  raw text through FastEmbed 0.3 — case-sensitive
#   2              normalize_for_bm25
SPARSE_ENCODING_VERSION = 2

_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# A number keeps its decimal / section dots ("452.8", "8.1"); a word is a run of
# letters and digits that starts with a letter ("fy2023", "q3"). Everything else
# separates tokens.
_BM25_TOKEN = re.compile(r"\d+(?:\.\d+)*|[^\W\d_][^\W_]*")


def normalize_for_bm25(text: str) -> str:
    """
    Normalises text into the tokens BM25 should match on, for documents and
    queries alike.

    FastEmbed 0.3 (the pinned range) feeds text to an NLTK-style word tokenizer
    and never lower-cases it, so on its own:
      - "Revenue" and "revenue" are different terms, and "What"/"The" survive
        the (lower-case) stopword list — "What is the Revenue and EBITDA?" shares
        no term with "revenue and ebitda were strong";
      - a sentence-final period stays on the word unless it ends the whole text
        ("FY2023.", "months."), because the tokenizer expects one sentence;
      - "$452.8M", "7.5x", "EV/EBITDA" and "change-of-control" are single terms,
        so "452.8", "EBITDA" or "change of control" in a question miss them.
    Figures and defined terms are what a diligence question keys on, so the
    text is reduced to plain tokens here, independently of the FastEmbed version
    (0.4+ lower-cases by itself; lower-casing twice changes nothing).

    Measured on the retrieval eval, sparse channel alone (35 questions): fact
    coverage@10 73.1% -> 94.3% (paired bootstrap 95% CI +10.1 to +34.0 points,
    10 questions improved, none regressed), recall@10 46.6% -> 77.3%. All of
    that comes from lower-casing. The punctuation and unit handling cannot be
    told apart from lower-casing alone on that set (every interval includes
    zero); it is here for the term mismatches listed above, which are
    deterministic and each pinned by a test.

    Lower-cases (after NFKC), drops thousands separators, and splits on
    everything that is not a letter or digit, except dots between digits. A
    number is split from a unit glued to it ("452.8m" -> "452.8 m", "280g" ->
    "280 g"). Stopwords and stemming are still FastEmbed's.

    Args:
        text: Chunk text or query.

    Returns:
        Space-separated tokens (may be empty if the text has no letters or digits).
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _THOUSANDS_SEPARATOR.sub("", text)
    return " ".join(_BM25_TOKEN.findall(text))


def compute_sparse_bm25(text: str) -> SparseVector:
    """
    Computes BM25 sparse vector for a text string (document or query), after
    normalize_for_bm25.
    FastEmbed returns embeddings as objects with .indices and .values arrays.
    These must be converted to lists before passing to Qdrant.

    NOTE: This is a synchronous CPU computation. When called from async context,
    wrap in run_in_executor using the _embed_executor pool.

    Args:
        text: Input text to encode.

    Returns:
        SparseVector with indices and values for Qdrant sparse search.

    Raises:
        ValueError: If text is empty or None.
    """
    if not text:
        raise ValueError("Cannot compute BM25 for empty text")

    model = _get_bm25_model()
    embeddings = list(model.embed([normalize_for_bm25(text)]))
    return SparseVector(
        indices=embeddings[0].indices.tolist(),
        values=embeddings[0].values.tolist(),
    )


def compute_sparse_bm25_batch(texts: list[str], batch_size: int = 64) -> list[SparseVector]:
    """
    Computes BM25 sparse vectors for many texts in one model call.

    Ingestion used to call compute_sparse_bm25 once per chunk, each call a
    separate executor round-trip; FastEmbed batches internally, so one call
    for the whole document is both simpler and faster.

    NOTE: Synchronous CPU work — call from a worker thread in async code.

    Args:
        texts: Non-empty texts to encode.
        batch_size: FastEmbed batch size.

    Returns:
        SparseVectors in the same order as texts.

    Raises:
        ValueError: If any text is empty.
    """
    if any(not t for t in texts):
        raise ValueError("Cannot compute BM25 for empty text")
    if not texts:
        return []

    model = _get_bm25_model()
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in model.embed([normalize_for_bm25(t) for t in texts], batch_size=batch_size)
    ]


# ==============================================================================
# Filter Building
# ==============================================================================


VALID_DOCUMENT_CATEGORIES = frozenset({
    "financial", "legal", "board", "audit", "regulatory", "operational", "other",
})

# Payload fields a query may narrow on. Everything else in metadata_filters is
# dropped: the dict is LLM-authored (Agent 1 / Agent 6), and a key that is not
# on the payload (fiscal_year, currency) would silently match nothing.
#
# Only document_category. doc_id, content_type and is_table used to be accepted
# as well, but nothing except model output ever supplied them, and each is a
# hard filter: a model that echoed a doc_id or "is_table": 1 confined every
# search of the request to it, and the rewrite loop cannot remove a key it is
# not allowed to set. A trusted caller that needs to scope a search to one
# document should get its own argument, as include_superseded did.
ALLOWED_FILTER_KEYS = frozenset({"document_category"})


def _valid_filter_value(key: str, value) -> bool:
    """
    Checks a whitelisted filter value has a type/value the payload can match.

    Args:
        key: Filter key (already whitelisted).
        value: Scalar or list value.

    Returns:
        True if the value should become a filter condition.
    """
    values = value if isinstance(value, list) else [value]
    if not values:
        return False
    if key == "document_category":
        return all(isinstance(v, str) and v in VALID_DOCUMENT_CATEGORIES for v in values)
    return False


def _build_filter(
    deal_id: str,
    metadata_filters: dict,
    include_superseded: bool = False,
    doc_id: str | None = None,
) -> Filter:
    """
    Builds a Qdrant Filter from deal_id (always applied) and optional metadata.
    deal_id is mandatory — never execute a search without it.

    Default exclusions applied automatically:
    - is_current_version=1 — ALWAYS, unless the trusted `include_superseded`
      argument is set. metadata_filters cannot turn it off: that dict comes
      from the LLM, and Agent 1's schema used to echo "is_current_version": 1,
      whose mere presence skipped this condition while the whitelist then
      discarded it — superseded documents were searchable on every query.
    - contains_pii=0 (PII-flagged content excluded by default per compliance policy;
      pass include_pii=True in metadata_filters to override for authorized users.
      The retrieval executor sets that key from the authenticated request after
      discarding any LLM-supplied value.)

    Args:
        deal_id: Mandatory deal isolation filter.
        metadata_filters: Dict of payload fields to filter on (untrusted keys
                          are dropped; see ALLOWED_FILTER_KEYS).
        include_superseded: Trusted, non-LLM override to search superseded
                            versions too (e.g. an explicit version-history view).
        doc_id: Trusted scope to one document (the retrieval executor's
                document-focus pass). Never read from metadata_filters.

    Returns:
        Qdrant Filter with all conditions applied.
    """
    conditions = [FieldCondition(key="deal_id", match=MatchValue(value=deal_id))]
    if doc_id:
        conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))

    if not include_superseded:
        conditions.append(
            FieldCondition(key="is_current_version", match=MatchValue(value=1))
        )

    # CRITICAL: Exclude PII-flagged content by default.
    # NOTE: Use .get() on a shallow copy — NOT .pop() on the original dict.
    # _build_filter() is called once per query expansion via asyncio.gather().
    # .pop() on the shared dict would consume include_pii on the first call.
    # A model can return null or a list where the filters object belongs.
    local_filters = dict(metadata_filters) if isinstance(metadata_filters, dict) else {}
    include_pii = local_filters.pop("include_pii", False)
    if not include_pii:
        conditions.append(
            FieldCondition(key="contains_pii", match=MatchValue(value=0))
        )

    dropped = []
    for key, value in local_filters.items():
        if value is None:
            continue
        if key not in ALLOWED_FILTER_KEYS or not _valid_filter_value(key, value):
            dropped.append(key)
            continue
        if isinstance(value, list):
            conditions.append(
                FieldCondition(key=key, match=MatchAny(any=value))
            )
        else:
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

    if dropped:
        logger.debug("Ignored filter keys", extra={"dropped_filter_keys": dropped})

    return Filter(must=conditions)


# ==============================================================================
# Hybrid Search
# ==============================================================================


async def hybrid_search(
    query_text: str,
    query_vector: list[float],
    query_sparse: SparseVector,
    deal_id: str,
    metadata_filters: dict,
    top_k_dense: int = 40,
    top_k_sparse: int = 40,
    client: AsyncQdrantClient | None = None,
    include_superseded: bool = False,
) -> tuple[list[ScoredPoint], list[ScoredPoint]]:
    """
    Executes dense and sparse searches in parallel with metadata constraints
    applied as Qdrant filter parameters — NOT as a separate retrieval channel.

    Args:
        query_text: Original query string (used for logging).
        query_vector: Dense embedding of the query.
        query_sparse: BM25 sparse vector of the query.
        deal_id: Mandatory deal isolation filter.
        metadata_filters: Dict of payload fields to filter on
                          (e.g., {"document_category": "financial"}). Version
                          filtering is not controlled here — see include_superseded.
        top_k_dense: Number of candidates from dense search.
        top_k_sparse: Number of candidates from sparse search.
        client: AsyncQdrantClient instance.
        include_superseded: Trusted override to also search superseded
                            document versions. Never derived from LLM output.

    Returns:
        Tuple of (dense_results, sparse_results) as ScoredPoint lists.

    Raises:
        QdrantException: On connection failure.
    """
    if client is None:
        client = get_qdrant_client()

    qdrant_filter = _build_filter(
        deal_id, metadata_filters, include_superseded=include_superseded
    )

    start = time.monotonic()

    dense_task = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        query_filter=qdrant_filter,
        limit=top_k_dense,
        search_params=QUANTIZED_SEARCH_PARAMS,  # rescore=True, oversampling=2.0
        with_payload=True,
    )
    sparse_task = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_sparse,
        using="sparse",
        query_filter=qdrant_filter,
        limit=top_k_sparse,
        with_payload=True,
    )

    dense_response, sparse_response = await asyncio.gather(dense_task, sparse_task)
    dense_results = dense_response.points
    sparse_results = sparse_response.points

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "Hybrid search complete",
        extra={
            "deal_id": deal_id,
            "dense_results": len(dense_results),
            "sparse_results": len(sparse_results),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )

    return dense_results, sparse_results


async def dense_search(
    query_vector: list[float],
    deal_id: str,
    metadata_filters: dict,
    top_k: int,
    doc_id: str | None = None,
    client: AsyncQdrantClient | None = None,
) -> list[ScoredPoint]:
    """
    Dense search alone, under the same server-enforced filters as hybrid_search.

    Used by the retrieval executor's document-focus pass, which searches one
    document's chunks; measured on LegalBench-RAG, a BM25 search inside the
    document added nothing on top of this one (Decision 67).

    Args:
        query_vector: Dense embedding of the query.
        deal_id: Mandatory deal isolation filter.
        metadata_filters: As for hybrid_search (untrusted keys dropped).
        top_k: Number of results.
        doc_id: Trusted scope to one document; never derived from LLM output.
        client: AsyncQdrantClient instance.

    Returns:
        ScoredPoints, best first.
    """
    if client is None:
        client = get_qdrant_client()
    response = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        query_filter=_build_filter(deal_id, metadata_filters, doc_id=doc_id),
        limit=top_k,
        search_params=QUANTIZED_SEARCH_PARAMS,
        with_payload=True,
    )
    return response.points


# ==============================================================================
# Fetch Chunks by IDs
# ==============================================================================


async def fetch_chunks_by_ids(
    chunk_ids: list[str],
    client: AsyncQdrantClient | None = None,
) -> list[dict]:
    """
    Fetches full chunk payloads by chunk_id from Qdrant.
    Uses client.scroll() with filter on chunk_id field.
    Returns list of payload dicts in same order as input ids.

    NOTE: scroll is acceptable here because we are doing deterministic lookup
    by chunk_id (fetching known points), not relevance-based retrieval.

    DEAL ISOLATION: This function does NOT add a deal_id filter because
    chunk_ids are globally unique (format: "{deal_id}_{doc_id}_{seq}").

    Args:
        chunk_ids: List of chunk_id strings to retrieve.
        client: AsyncQdrantClient instance. If None, uses get_qdrant_client().

    Returns:
        List of payload dicts in the same order as input chunk_ids.
        Missing IDs are silently skipped (logged as warnings).

    Raises:
        QdrantException: On connection failure after retries.
    """
    if not chunk_ids:
        return []

    if client is None:
        client = get_qdrant_client()

    start = time.monotonic()

    results = await client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids))
            ]
        ),
        limit=len(chunk_ids),
        with_payload=True,
        with_vectors=False,
    )

    # Return in original rank order — preserves RRF ranking
    payload_map = {p.payload["chunk_id"]: p.payload for p in results[0]}
    ordered = [payload_map[cid] for cid in chunk_ids if cid in payload_map]

    if len(ordered) < len(chunk_ids):
        missing = set(chunk_ids) - set(payload_map.keys())
        logger.warning(
            f"fetch_chunks_by_ids: {len(missing)} IDs not found",
            extra={"missing_ids": list(missing)},
        )

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "fetch_chunks_by_ids complete",
        extra={
            "requested": len(chunk_ids),
            "found": len(ordered),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )

    return ordered
