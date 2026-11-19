"""
Backfill document identity onto an index written before Decision 86.

    python -m src.vector_db.identity_backfill                    # every deal, configured Qdrant
    python -m src.vector_db.identity_backfill --deal deal-1
    python -m src.vector_db.identity_backfill --path <store>/qdrant --dry-run

Ingestion writes doc_title and doc_parties on every child chunk
(document_identity.py). An index built before that has neither, and the
named-document step (named_document.py) then keeps the fused-score choice for
its documents: nothing breaks, nothing improves. This reads each document's
child payloads back from the index, computes the identity with the same
function and from the same input ingestion uses — the payloads' chunk_index,
text, section_heading and contains_pii — and writes it onto that document's
child points. A backfilled document therefore carries exactly what
re-ingesting the file would write, without the file.

Idempotent: a document whose points already carry the computed identity is
left alone, and a field the extractor no longer finds is removed. Parents
are not touched (searches return children).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.data_processing.document_identity import extract_document_identity
from src.vector_db.constants import COLLECTION_NAME
from src.utils.fail_open import fail_open
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

IDENTITY_FIELDS = ("doc_title", "doc_parties")
# Chunks read per document: the opening is a few thousand characters, and a
# merger agreement's cover can be split into a dozen tiny chunks before it.
_OPENING_CHUNKS = 64
_PAGE = 1_000
_READ_FIELDS = ["deal_id", "doc_id", "chunk_index", "text", "section_heading", "contains_pii",
                *IDENTITY_FIELDS]


def _canonical(identity: dict) -> str:
    return json.dumps(identity, sort_keys=True, ensure_ascii=False)


async def backfill_document_identity(
    client: AsyncQdrantClient,
    *,
    deal_id: str | None = None,
    collection_name: str = COLLECTION_NAME,
    dry_run: bool = False,
) -> dict:
    """
    Computes and writes doc_title / doc_parties for every document of an index.

    Args:
        client: Qdrant client (server or local).
        deal_id: Only this deal's documents (all deals when None).
        collection_name: Child collection.
        dry_run: Compute and count, write nothing.

    Returns:
        Counts: documents, updated, unchanged, with_title, with_parties, and
        failed (identity extraction raised; the document was left as it was).
    """
    scroll_filter = (Filter(must=[FieldCondition(key="deal_id", match=MatchValue(value=deal_id))])
                     if deal_id else None)
    point_ids: dict[tuple, list] = defaultdict(list)
    opening: dict[tuple, list[dict]] = defaultdict(list)
    # The identities the document's points carry now (one, when consistent).
    current: dict[tuple, set[str]] = defaultdict(set)
    offset = None
    while True:
        records, offset = await client.scroll(
            collection_name=collection_name, scroll_filter=scroll_filter, limit=_PAGE, offset=offset,
            with_payload=_READ_FIELDS, with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            key = (payload.get("deal_id"), payload.get("doc_id"))
            if not key[1]:
                continue
            point_ids[key].append(record.id)
            opening[key].append(payload)
            # Keep the opening only; the rest of a long document is not read.
            if len(opening[key]) > 2 * _OPENING_CHUNKS:
                opening[key] = sorted(opening[key], key=lambda p: p.get("chunk_index") or 0)[:_OPENING_CHUNKS]
            current[key].add(_canonical({f: payload.get(f) for f in IDENTITY_FIELDS if payload.get(f)}))
        if offset is None:
            break

    stats = {"documents": len(point_ids), "updated": 0, "unchanged": 0, "with_title": 0, "with_parties": 0,
             "failed": 0}
    for key, ids in point_ids.items():
        payloads = sorted(opening[key], key=lambda p: p.get("chunk_index") or 0)[:_OPENING_CHUNKS]
        # A document whose identity cannot be read is left exactly as it is —
        # not given {}, which would delete fields it may already carry — and
        # the run goes on to the next (src/utils/fail_open.py).
        identity = fail_open("document_identity", extract_document_identity, payloads, default=None)
        if identity is None:
            stats["failed"] += 1
            continue
        stats["with_title"] += "doc_title" in identity
        stats["with_parties"] += "doc_parties" in identity
        if current[key] == {_canonical(identity)}:
            stats["unchanged"] += 1
            continue
        stats["updated"] += 1
        if dry_run:
            continue
        stale = [f for f in IDENTITY_FIELDS if f not in identity]
        if identity:
            await client.set_payload(collection_name=collection_name, payload=identity, points=ids, wait=True)
        if stale:
            await client.delete_payload(collection_name=collection_name, keys=stale, points=ids, wait=True)
    logger.info("Document identity backfill complete", extra={**stats, "deal_id": deal_id, "dry_run": dry_run})
    return stats


async def _main(args) -> dict:
    if args.path:
        client = AsyncQdrantClient(path=args.path)
    else:
        from src.vector_db.qdrant_client import get_qdrant_client
        client = get_qdrant_client()
    try:
        return await backfill_document_identity(client, deal_id=args.deal, dry_run=args.dry_run)
    finally:
        await client.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deal", help="only this deal_id")
    parser.add_argument("--path", help="a local-mode Qdrant directory instead of the configured server")
    parser.add_argument("--dry-run", action="store_true", help="compute and count, write nothing")
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(_main(args))), file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
