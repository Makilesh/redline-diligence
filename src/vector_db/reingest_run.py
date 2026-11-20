"""
Re-ingest, from their original files, the documents reingest_report marks (Decision 97).

    python -m src.vector_db.reingest_run --files /srv/originals --deal deal-1 --dry-run
    python -m src.vector_db.reingest_run --files /srv/originals --deal deal-1 --log reingest.jsonl
    python -m src.vector_db.reingest_run --path <store>/qdrant --files <dir> --types pdf

WRITES TO THE INDEX. Snapshot it first and stop the API (DEPLOYMENT.md,
"Re-ingesting an existing index"); a local-mode store cannot be opened by two
processes anyway.

Each document is indexed again by index_document, the function POST /ingest
calls, under what the index holds for it now: its stored name (source_file),
category, version flag and supersedes link. Its file is found by its bytes,
so the name on disk does not matter. The document is then read back and
checked as the report checks it: every point stamped with the current
ingest_version, children and parents linked, the counts index_document
reported. A run that fails or leaves the document inconsistent is repeated
(--retries): a complete run upserts every point of the document and deletes
the rest, whatever an interrupted one left. A file the pipeline refuses
(IngestionError) is not retried.

The file is copied first, and its copy's bytes must still be the document's
(compute_doc_id(deal, sha256) == doc_id): a file edited after the report is
refused at once, never indexed as a second, current document beside the old.

Nothing else is written: documents the report does not mark for re-ingestion,
or whose file is not given, are left as they are and listed. Documents the
report leaves to an operator (version state, name, category or supersedes
link in conflict across their points, or parents with no child) are listed
with the values in conflict and never indexed: guessing would choose a state. Each attempt is
one JSON line in --log, and each document one `document_reingested` audit
event. Afterwards run the identity backfill, then the report again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME
from src.vector_db.reingest_report import build_report, document_problems, read_document
from src.utils import audit_log
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

FILE_CHANGED = "file changed since the report: its bytes are no longer this document's"


def plan_reingest(report: dict, types: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """
    The documents to re-ingest that have a file, and those that have none.

    Args:
        report: build_report(..., files=...).
        types: Only these file types (all when None).

    Returns:
        (rows to re-ingest, rows that need it but have no file), report rows.
    """
    todo, missing = [], []
    for deal in report["deals"].values():
        for row in deal["documents"]:
            if row["action"] != "reingest" or (types and row["file_type"] not in types):
                continue
            (todo if row.get("file") else missing).append(row)
    return todo, missing


async def _verify(client, row: dict, result: dict | None, collection_name: str,
                  parent_collection_name: str) -> list[str]:
    """Problems with the document as it now stands; empty when it is one complete current run."""
    from src.data_processing.ingest_pipeline import INGEST_VERSION

    doc = await read_document(client, row["deal_id"], row["doc_id"], collection_name, parent_collection_name)
    problems = document_problems(doc)
    if any(p.get("ingest_version") != INGEST_VERSION for _, p in doc.children + doc.parents):
        problems.append("not_current_version")
    if result is not None and (len(doc.children), len(doc.parents)) != (
            result["chunks_created"], result["parent_chunks_created"]):
        problems.append("counts_differ_from_run")
    return problems


async def reingest_document(
    client,
    row: dict,
    *,
    retries: int = 2,
    reclassify: bool = False,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
    embed_fn=None,
    sparse_fn=None,
    backoff_s: float = 2.0,
    log=None,
) -> dict:
    """
    Indexes one document again from its file, checks it, and repeats if needed.

    Args:
        client: Qdrant client.
        row: The document's report row (with "file").
        retries: Further attempts after the first.
        reclassify: Let the classifier choose the category (as a fresh upload
            does) instead of keeping the indexed one.
        collection_name: Child collection.
        parent_collection_name: Parent collection.
        embed_fn, sparse_fn: As index_document (tests pass stubs).
        backoff_s: Wait before the second attempt, doubled after each.
        log: Open text file; one JSON line per attempt.

    Returns:
        {"deal_id", "doc_id", "source_file", "ok", "attempts", "problems", "error",
         "chunks_before", "chunks_after", "parents_before", "parents_after", "seconds"}
    """
    from src.data_processing.ingest_pipeline import compute_content_hash, compute_doc_id

    outcome = {
        "deal_id": row["deal_id"], "doc_id": row["doc_id"], "source_file": row["source_file"],
        "file": row["file"], "ok": False, "attempts": 0, "problems": [], "error": None,
        "chunks_before": row["children"], "parents_before": row["parents"],
        "ingest_version_before": row["ingest_version"],
        "chunks_after": None, "parents_after": None, "seconds": 0.0,
    }
    started = time.perf_counter()
    # The file is indexed from a private copy whose bytes are checked against
    # the document first. A file edited after the report would otherwise be
    # indexed as a new, current document beside the old one (doc_id comes from
    # the bytes), and each retry would repeat it.
    workdir = tempfile.mkdtemp(prefix="reingest_")
    try:
        copy = Path(workdir) / f"original{Path(row['file']).suffix}"
        shutil.copyfile(row["file"], copy)
        if compute_doc_id(row["deal_id"], compute_content_hash(str(copy))) != row["doc_id"]:
            outcome.update(error=FILE_CHANGED, seconds=round(time.perf_counter() - started, 2))
            if log is not None:
                log.write(json.dumps({**outcome, "attempt": 0}, ensure_ascii=False) + "\n")
                log.flush()
        else:
            await _attempts(client, row, str(copy), outcome, started, retries=retries, reclassify=reclassify,
                            collection_name=collection_name, parent_collection_name=parent_collection_name,
                            embed_fn=embed_fn, sparse_fn=sparse_fn, backoff_s=backoff_s, log=log)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if not outcome["ok"]:
        logger.error("Re-ingest did not complete the document",
                     extra={"doc_id": row["doc_id"], "attempts": outcome["attempts"], "error": outcome["error"],
                            "problems": outcome["problems"]})
    audit_log.record(
        "document_reingested",
        deal_id=row["deal_id"], doc_id=row["doc_id"], file_name=row["source_file"],
        ok=outcome["ok"], attempts=outcome["attempts"], error=outcome["error"], problems=outcome["problems"],
        chunks_before=outcome["chunks_before"], chunks_after=outcome["chunks_after"],
        parents_before=outcome["parents_before"], parents_after=outcome["parents_after"],
        ingest_version_before=outcome["ingest_version_before"],
    )
    return outcome


async def _attempts(client, row: dict, path: str, outcome: dict, started: float, *, retries: int,
                    reclassify: bool, collection_name: str, parent_collection_name: str, embed_fn, sparse_fn,
                    backoff_s: float, log) -> None:
    """index_document, read back, verify; repeated until whole or out of attempts. Fills `outcome`."""
    from src.data_processing.errors import IngestionError
    from src.data_processing.ingest_pipeline import index_document

    for attempt in range(1, retries + 2):
        outcome["attempts"] = attempt
        t0 = time.perf_counter()
        result, error, final = None, None, False
        try:
            result = await index_document(
                path, row["source_file"], row["deal_id"],
                None if reclassify else row["document_category"],
                is_current_version=bool(row["is_current_version"]),
                supersedes_doc_id=row["supersedes_doc_id"] or None,
                client=client, collection_name=collection_name,
                parent_collection_name=parent_collection_name,
                embed_fn=embed_fn, sparse_fn=sparse_fn,
            )
        except IngestionError as e:
            # The file itself is refused (unreadable, unsafe, empty): another
            # attempt would refuse it again.
            error, final = f"{type(e).__name__}: {e}", True
        except Exception as e:  # noqa: BLE001 — a store or model fault: verify, then retry
            error = f"{type(e).__name__}: {e}"
        if result is not None and result["doc_id"] != row["doc_id"]:
            # The checked copy makes this impossible; if it happens, stop.
            error, final = f"indexed as {result['doc_id']}, not {row['doc_id']}", True
        try:
            problems = await _verify(client, row, result, collection_name, parent_collection_name) \
                if not final else []
        except Exception as e:  # noqa: BLE001 — the read-back failed: the document's state is unknown
            problems = [f"not_verified: {type(e).__name__}"]
        outcome.update(problems=problems, error=error, ok=error is None and not problems)
        if result is not None:
            outcome.update(chunks_after=result["chunks_created"], parents_after=result["parent_chunks_created"])
        outcome["seconds"] = round(time.perf_counter() - started, 2)
        if log is not None:
            log.write(json.dumps({**outcome, "attempt": attempt,
                                  "attempt_seconds": round(time.perf_counter() - t0, 2)},
                                 ensure_ascii=False) + "\n")
            log.flush()
        if outcome["ok"] or final:
            break
        if attempt <= retries:
            logger.warning("Re-ingest attempt did not complete the document; retrying",
                           extra={"doc_id": row["doc_id"], "attempt": attempt, "error": error,
                                  "problems": problems})
            await asyncio.sleep(backoff_s * 2 ** (attempt - 1))


async def run(
    client,
    files: str | Path,
    *,
    deal_id: str | None = None,
    types: set[str] | None = None,
    dry_run: bool = False,
    retries: int = 2,
    reclassify: bool = False,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
    embed_fn=None,
    sparse_fn=None,
    backoff_s: float = 2.0,
    log=None,
) -> dict:
    """
    Re-ingests every document the report marks that has a file.

    Returns:
        {"planned", "done", "failed", "without_file", "files_not_indexed", "documents": [outcomes]}
    """
    report = await build_report(client, deal_id=deal_id, files=files, collection_name=collection_name,
                                parent_collection_name=parent_collection_name)
    todo, missing = plan_reingest(report, types)
    rows = [r for deal in report["deals"].values() for r in deal["documents"]]
    summary = {
        "planned": len(todo),
        "without_file": [{k: r[k] for k in ("deal_id", "doc_id", "source_file", "file_type")} for r in missing],
        # Never indexed by the driver: which state is right is a person's call.
        "for_operator": [{k: r[k] for k in ("deal_id", "doc_id", "source_file", "problems", "conflicts")}
                         for r in rows if r["action"] == "operator"],
        "identity_failed": [{k: r[k] for k in ("deal_id", "doc_id", "source_file")}
                            for r in rows if r["action"] == "identity_failed"],
        "files_not_indexed": report["files"]["not_indexed"],
        "dry_run": dry_run,
        "documents": [],
    }
    if dry_run:
        summary["documents"] = [{k: r[k] for k in ("deal_id", "doc_id", "source_file", "file", "reasons")}
                                for r in todo]
        return summary
    for n, row in enumerate(todo, start=1):
        outcome = await reingest_document(
            client, row, retries=retries, reclassify=reclassify, collection_name=collection_name,
            parent_collection_name=parent_collection_name, embed_fn=embed_fn, sparse_fn=sparse_fn,
            backoff_s=backoff_s, log=log)
        summary["documents"].append(outcome)
        logger.info("Re-ingest progress", extra={"done": n, "planned": len(todo), "doc_id": row["doc_id"],
                                                 "ok": outcome["ok"], "seconds": outcome["seconds"]})
    summary["done"] = sum(1 for d in summary["documents"] if d["ok"])
    summary["failed"] = [d for d in summary["documents"] if not d["ok"]]
    return summary


async def _main(args) -> dict:
    if args.path:
        client = AsyncQdrantClient(path=args.path)
    else:
        from src.vector_db.qdrant_client import get_qdrant_client
        client = get_qdrant_client()
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    try:
        return await run(client, args.files, deal_id=args.deal,
                         types=set(args.types.split(",")) if args.types else None,
                         dry_run=args.dry_run, retries=args.retries, reclassify=args.reclassify, log=log)
    finally:
        if log is not None:
            log.close()
        await client.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--files", required=True, help="directory of the original files (matched by their bytes)")
    parser.add_argument("--deal", help="only this deal_id")
    parser.add_argument("--path", help="a local-mode Qdrant directory instead of the configured server")
    parser.add_argument("--types", help="only these file types, comma-separated (e.g. pdf)")
    parser.add_argument("--dry-run", action="store_true", help="list what would be re-ingested; write nothing")
    parser.add_argument("--retries", type=int, default=2, help="further attempts for a document (default 2)")
    parser.add_argument("--reclassify", action="store_true",
                        help="let the classifier choose each category, as a new upload does")
    parser.add_argument("--log", help="append one JSON line per attempt to this file")
    args = parser.parse_args(argv)
    summary = asyncio.run(_main(args))
    print(json.dumps({k: v for k, v in summary.items() if k != "documents"}, ensure_ascii=False, default=str))
    if args.dry_run:
        for row in summary["documents"]:
            print(f"would re-ingest {row['source_file']} ({row['deal_id']}/{row['doc_id']}) from {row['file']}: "
                  f"{'; '.join(row['reasons'])}")
    for row in summary["for_operator"]:
        print(f"left for an operator: {row['source_file']} ({row['deal_id']}/{row['doc_id']}): "
              f"{', '.join(row['problems'])} {json.dumps(row['conflicts'], ensure_ascii=True)}")
    return 1 if summary.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
