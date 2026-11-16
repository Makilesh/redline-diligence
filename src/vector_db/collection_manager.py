"""
Qdrant collection creation and management.

These values CANNOT change after the collection is created.
Changing VECTOR_SIZE requires re-embedding and re-indexing the entire corpus.

Collections:
    - manda_due_diligence: Main child-chunk collection with hybrid vector support
    - manda_parent_chunks: Parent-chunk collection for context expansion
"""

import asyncio
import time

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    SparseVectorParams,
    SparseIndexParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    PayloadSchemaType,
)

from src.vector_db.constants import (
    VECTOR_SIZE,
    COLLECTION_NAME,
    PARENT_COLLECTION_NAME,
    QDRANT_MAX_RETRIES,
    QDRANT_BASE_DELAY_S,
    QDRANT_MAX_DELAY_S,
)
from src.vector_db.qdrant_client import get_qdrant_client
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def _retry_operation(operation, operation_name: str) -> None:
    """
    Executes a Qdrant operation with exponential backoff retry.

    Args:
        operation: Async callable to execute.
        operation_name: Human-readable name for logging.

    Raises:
        Exception: Re-raises the last exception after max retries.
    """
    for attempt in range(QDRANT_MAX_RETRIES):
        try:
            start = time.monotonic()
            await operation()
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(
                f"{operation_name} completed",
                extra={"elapsed_ms": round(elapsed_ms, 2), "attempt": attempt + 1},
            )
            return
        except Exception as e:
            if attempt == QDRANT_MAX_RETRIES - 1:
                logger.error(
                    f"{operation_name} failed after {QDRANT_MAX_RETRIES} attempts",
                    extra={"error": str(e)},
                )
                raise
            delay = min(
                QDRANT_BASE_DELAY_S * (2 ** attempt),
                QDRANT_MAX_DELAY_S,
            )
            logger.warning(
                f"{operation_name} attempt {attempt + 1} failed, retrying in {delay}s",
                extra={"error": str(e), "delay_s": delay},
            )
            await asyncio.sleep(delay)


async def create_collection(
    client: AsyncQdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    Creates the main child-chunk collection with hybrid vector support.
    HNSW parameters are explicitly set — do not rely on Qdrant defaults,
    which use ef_construct=100 (lower recall at scale than ef_construct=200).

    Args:
        client: Initialized AsyncQdrantClient instance. If None, uses singleton.
        collection_name: Collection to create (defaults to the standard name;
                         overridable so an eval harness can index elsewhere).

    Raises:
        QdrantException: If collection creation fails after retries.
    """
    if client is None:
        client = get_qdrant_client()

    # Check if collection already exists
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]
    if collection_name in existing_names:
        logger.info(
            f"Collection '{collection_name}' already exists, skipping creation"
        )
        await ensure_sparse_idf(client, collection_name)
        return

    async def _create():
        await client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=VECTOR_SIZE,           # 1024 — immutable
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=16,                   # Connections per layer
                        ef_construct=200,       # Build-time search width
                        full_scan_threshold=10_000,
                    ),
                    quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,    # Quantized INT8 vectors in RAM
                        )
                    ),
                    # The float32 originals are kept as well, for rescoring. They
                    # are in RAM too: on_disk is not set, so nothing here puts
                    # them on disk (an earlier comment said it did).
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,          # Sparse BM25 index in RAM for speed
                    ),
                    # FastEmbed's Qdrant/bm25 emits only the term-frequency half
                    # of BM25 and leaves IDF to Qdrant; see ensure_sparse_idf.
                    modifier=Modifier.IDF,
                )
            },
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=4,
                indexing_threshold=20_000,
            ),
        )

    await _retry_operation(_create, f"Create collection '{collection_name}'")
    logger.info(
        f"Collection '{collection_name}' created successfully",
        extra={
            "vector_size": VECTOR_SIZE,
            "distance": "cosine",
            "hnsw_m": 16,
            "hnsw_ef_construct": 200,
            "quantization": "INT8",
        },
    )


async def ensure_sparse_idf(
    client: AsyncQdrantClient,
    collection_name: str = COLLECTION_NAME,
    vector_name: str = "sparse",
) -> bool:
    """
    Makes sure the sparse vector is scored with Qdrant's IDF modifier.

    FastEmbed's `Qdrant/bm25` stores only the term-frequency / length-normalised
    part of BM25 in each vector and expects Qdrant to multiply by IDF at query
    time ("expected to be used with modifier=idf" — its own docstring). Without
    the modifier every query term weighs the same, so "the" in a question counts
    as much as "280G". Collections created before this was set are upgraded in
    place: IDF is computed from the index at query time, so nothing is re-indexed.

    Best effort. A server that refuses the update leaves sparse search exactly
    as it was, which is degraded ranking rather than an outage, so the failure
    is logged and startup continues.

    Args:
        client: AsyncQdrantClient instance.
        collection_name: Collection to check.
        vector_name: Name of the sparse vector.

    Returns:
        True if the modifier is active after the call.
    """
    try:
        info = await client.get_collection(collection_name)
        sparse = (info.config.params.sparse_vectors or {}).get(vector_name)
        if sparse is None:
            logger.warning(
                f"Collection '{collection_name}' has no '{vector_name}' sparse vector"
            )
            return False
        if sparse.modifier == Modifier.IDF:
            return True

        await client.update_collection(
            collection_name=collection_name,
            sparse_vectors_config={
                vector_name: SparseVectorParams(index=sparse.index, modifier=Modifier.IDF)
            },
        )
        info = await client.get_collection(collection_name)
        active = info.config.params.sparse_vectors[vector_name].modifier == Modifier.IDF
    except Exception as e:
        logger.warning(
            f"Could not enable the IDF modifier on '{collection_name}'; "
            "sparse search stays un-weighted until the collection is recreated",
            extra={"error": str(e)},
        )
        return False

    if active:
        logger.info(f"Enabled the IDF modifier on '{collection_name}.{vector_name}'")
    else:
        logger.warning(
            f"Qdrant accepted the update but '{collection_name}.{vector_name}' still has "
            "no IDF modifier; sparse search stays un-weighted until the collection is recreated"
        )
    return active


async def create_parent_collection(
    client: AsyncQdrantClient | None = None,
    collection_name: str = PARENT_COLLECTION_NAME,
) -> None:
    """
    Creates the parent-chunk collection for context expansion.
    Parent text blobs are large — do NOT set always_ram on payload.
    At 50k chunks × ~8KB parent text = ~400MB per deal; forcing this
    into RAM exhausts the 32GB system RAM budget at scale.
    Vectors are on disk; payload (containing parent_text) goes to disk.

    INGESTION COMPLIANCE NOTE:
    The ingestion pipeline MUST write the `contains_pii` (0 or 1) and
    `is_current_version` (0 or 1) metadata fields to both the main child-chunk
    collection payloads AND the parent-chunk collection payloads.

    Args:
        client: Initialized AsyncQdrantClient instance. If None, uses singleton.
        collection_name: Collection to create (defaults to the standard name;
                         overridable so an eval harness can index elsewhere).

    Raises:
        QdrantException: If collection creation fails after retries.
    """
    if client is None:
        client = get_qdrant_client()

    # Check if collection already exists
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]
    if collection_name in existing_names:
        logger.info(
            f"Collection '{collection_name}' already exists, skipping creation"
        )
        return

    async def _create():
        await client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            },
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=10_000,  # Use memory-mapped files for large collections
            ),
        )

    await _retry_operation(_create, f"Create collection '{collection_name}'")
    logger.info(f"Collection '{collection_name}' created successfully")


async def create_payload_indexes(
    client: AsyncQdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
) -> None:
    """
    Creates payload indexes for filtered search.

    NOTE on booleans: Qdrant does not have a dedicated BOOL PayloadSchemaType
    in all versions. Store boolean fields as integers (0/1) and index as INTEGER.

    Args:
        client: AsyncQdrantClient instance. If None, uses singleton.
        collection_name: Child collection to index.
        parent_collection_name: Parent collection to index.

    Raises:
        QdrantException: If index creation fails after retries.
    """
    if client is None:
        client = get_qdrant_client()

    # Main collection indexes
    indexes = [
        ("deal_id",             PayloadSchemaType.KEYWORD),
        ("document_category",   PayloadSchemaType.KEYWORD),
        ("file_type",           PayloadSchemaType.KEYWORD),
        ("is_table",            PayloadSchemaType.INTEGER),    # 0 or 1 — safe across versions
        ("fiscal_year",         PayloadSchemaType.KEYWORD),
        ("table_id",            PayloadSchemaType.KEYWORD),
        ("is_current_version",  PayloadSchemaType.INTEGER),    # 0 or 1
        ("currency",            PayloadSchemaType.KEYWORD),
        ("contains_pii",        PayloadSchemaType.INTEGER),    # 0 or 1
        ("content_type",        PayloadSchemaType.KEYWORD),
        # Re-ingest deletes a document's stale points and supersession
        # retires them — both filter on doc_id.
        ("doc_id",              PayloadSchemaType.KEYWORD),
        ("chunk_id",            PayloadSchemaType.KEYWORD),
        # Indexed so /deals can facet distinct documents per deal without
        # scrolling the whole collection. Qdrant refuses to facet an unindexed
        # field outright, so this is a requirement of that endpoint, not a
        # performance nicety.
        ("source_file",         PayloadSchemaType.KEYWORD),
        # Which BM25 text normalisation wrote the point's sparse vector; counted
        # by sparse_index_status to find points that need re-indexing.
        ("sparse_encoding",     PayloadSchemaType.INTEGER),
        # Section-table lookup filters on it (sibling_retrieval.fetch_section_tables);
        # a server in strict mode rejects a filter on an unindexed field.
        ("section_heading",     PayloadSchemaType.KEYWORD),
        # The same-registrant, named-period lookup filters on these
        # (filing_period.prefer_named_period). Absent on non-filings.
        ("filing_entity_keys",  PayloadSchemaType.KEYWORD),
        ("filing_fiscal_years", PayloadSchemaType.INTEGER),
        ("filing_fiscal_quarter", PayloadSchemaType.INTEGER),
    ]

    for field_name, field_type in indexes:
        async def _create_index(fn=field_name, ft=field_type):
            await client.create_payload_index(
                collection_name=collection_name,
                field_name=fn,
                field_schema=ft,
            )

        await _retry_operation(
            _create_index,
            f"Create index '{field_name}' on '{collection_name}'",
        )

    logger.info(
        f"Created {len(indexes)} payload indexes on '{collection_name}'"
    )

    # Parent collection indexes (used during context expansion)
    parent_indexes = [
        ("chunk_id",            PayloadSchemaType.KEYWORD),
        ("deal_id",             PayloadSchemaType.KEYWORD),
        ("doc_id",              PayloadSchemaType.KEYWORD),
        ("is_current_version",  PayloadSchemaType.INTEGER),
        ("contains_pii",        PayloadSchemaType.INTEGER),
    ]

    for field_name, field_type in parent_indexes:
        async def _create_parent_index(fn=field_name, ft=field_type):
            await client.create_payload_index(
                collection_name=parent_collection_name,
                field_name=fn,
                field_schema=ft,
            )

        await _retry_operation(
            _create_parent_index,
            f"Create index '{field_name}' on '{parent_collection_name}'",
        )

    logger.info(
        f"Created {len(parent_indexes)} payload indexes on '{parent_collection_name}'"
    )


async def setup_collections(
    client: AsyncQdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
) -> None:
    """
    Complete collection setup: creates both collections and all payload indexes.
    Idempotent — safe to call on every startup.

    Args:
        client: AsyncQdrantClient instance. If None, uses singleton.
        collection_name: Child collection name.
        parent_collection_name: Parent collection name.
    """
    if client is None:
        client = get_qdrant_client()

    logger.info("Starting Qdrant collection setup")
    await create_collection(client, collection_name)
    await create_parent_collection(client, parent_collection_name)
    await create_payload_indexes(client, collection_name, parent_collection_name)
    try:
        await sparse_index_status(client, collection_name)
    except Exception as e:
        # A diagnostic must not be the reason startup fails.
        logger.warning("Could not check the sparse index encoding", extra={"error": str(e)})
    logger.info("Qdrant collection setup complete")


async def sparse_index_status(
    client: AsyncQdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """
    How many indexed chunks carry sparse vectors from an older BM25 encoding.

    The text normalisation applied before BM25 (hybrid_search.normalize_for_bm25)
    is applied to queries as well as documents. A point indexed before it — or
    under an earlier version of it — holds terms a current query no longer
    produces ("Revenue" was a different term from "revenue"), so the keyword
    channel quietly misses it while dense search still works. Nothing fails, so
    it is counted here and logged, and can be shown by a health endpoint.

    The fix for a non-zero count is to re-ingest the affected documents
    (`python run_demo.py --reindex` for the demo deal).

    Args:
        client: AsyncQdrantClient instance. If None, uses singleton.
        collection_name: Child collection to inspect.

    Returns:
        {"current_encoding": int, "total_points": int, "stale_points": int}
    """
    from src.vector_db.hybrid_search import SPARSE_ENCODING_VERSION

    if client is None:
        client = get_qdrant_client()

    total = (await client.count(collection_name=collection_name, exact=True)).count
    stale = (await client.count(
        collection_name=collection_name,
        count_filter=Filter(must_not=[
            FieldCondition(key="sparse_encoding", match=MatchValue(value=SPARSE_ENCODING_VERSION)),
        ]),
        exact=True,
    )).count

    status = {
        "current_encoding": SPARSE_ENCODING_VERSION,
        "total_points": total,
        "stale_points": stale,
    }
    if stale:
        logger.warning(
            f"{stale} of {total} chunks in '{collection_name}' were indexed with an older "
            "BM25 encoding and are invisible to keyword search for many queries. "
            "Re-ingest their documents (python run_demo.py --reindex for the demo deal).",
            extra=status,
        )
    return status
