"""
Which documents of an index need re-ingesting, and why — read-only (Decision 97).

    python -m src.vector_db.reingest_report                            # every deal, configured Qdrant
    python -m src.vector_db.reingest_report --deal deal-1 --files /srv/originals
    python -m src.vector_db.reingest_report --path <store>/qdrant --json report.json

A point's id comes from its chunk's position, not its content, so after an
extraction change nothing in a point said which code wrote it. Ingestion now
stamps every child and parent with `ingest_version` (ingest_pipeline.
INGEST_VERSION; CURRENT_SINCE says, per file type, from which version on the
chunks are what the current code writes). This reads every document's points
— payload fields only, no vectors, and the opening's text for identity — and
says what each needs:

    reingest           index its file again: its table rows were stored
                       without their column order (Decision 108), its
                       type's extraction changed
                       after its stamp (an unstamped PDF was indexed before
                       Decision 97, by code that cannot be told apart), its
                       BM25 encoding is old, or its points do not form one
                       document (a re-index cut short: two stamps, children
                       their parent does not hold, parents nothing points to)
    operator           left to a person, never to the driver: its points
                       disagree on its version state (some retired, some
                       current), its name, category or supersedes link, or
                       it has parents and no child; the row lists the values
    identity_failed    the identity extractor raised on it
    backfill_identity  `python -m src.vector_db.identity_backfill` would change
                       its doc_title / doc_parties
    nothing

Points a first-time upload staged and never committed (its run was cut short
by a crash or a cancelled request; ingest_pipeline.index_document) carry no
deal_id, so no reader sees them. They are listed apart, under "staged", by the
deal they were meant for: purging that deal deletes them, and uploading the
same file again commits it.

It writes nothing. With --files, the files under a directory are matched to
documents by their bytes (doc_id = uuid5(deal, sha256(bytes))): a renamed file
still finds its document, an edited one finds none.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, IsEmptyCondition, MatchValue, PayloadField

from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME, STAGED_DEAL_FIELD
from src.vector_db.identity_backfill import IdentityPlan
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_PAGE = 1_000

# What a document's consistency and its plan are read from (children), on top
# of what the identity backfill reads (IdentityPlan.FIELDS).
CHILD_FIELDS = [
    "deal_id", "doc_id", "chunk_id", "chunk_index", "parent_chunk_id", "source_file", "file_type",
    "document_category", "is_current_version", "superseded_by", "supersedes_doc_id", "sparse_encoding",
    "ingest_version", "doc_title", "doc_parties", "structured_rows",
]
PARENT_FIELDS = ["deal_id", "doc_id", "chunk_id", "child_chunk_ids", "ingest_version", "is_current_version",
                 "superseded_by"]

# What collect() keeps of a child's structured_rows (see rows_unordered).
ROWS_UNORDERED = "_rows_unordered"


def rows_unordered(rows) -> bool:
    """Whether any table row was stored without its column order (Decision 111)."""
    from src.utils.numerical_registry import column_order_known

    return any(isinstance(r, dict) and not column_order_known(r) for r in rows or [])


ACTIONS = ("reingest", "operator", "identity_failed", "backfill_identity", "nothing")

# Per document, fields every child of one upload carries with one value.
DOCUMENT_FIELDS = ("source_file", "document_category", "supersedes_doc_id")

# Problems the driver must not resolve by guessing: which version state, name,
# category or supersedes link is right, or whether a document that has no
# child left should exist at all, is an operator's call (action "operator").
OPERATOR_PROBLEMS = ("mixed_version_state", "mixed_document_fields", "orphan_parents_only")


@dataclass
class DocumentPoints:
    """One document's points, as (point id, payload) pairs, payload fields only."""

    deal_id: str
    doc_id: str
    children: list[tuple] = field(default_factory=list)
    parents: list[tuple] = field(default_factory=list)


def _deal_filter(deal_id: str | None, doc_id: str | None = None) -> Filter | None:
    must = []
    if deal_id:
        must.append(FieldCondition(key="deal_id", match=MatchValue(value=deal_id)))
    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    return Filter(must=must) if must else None


async def _scroll(client, collection_name: str, scroll_filter, fields: list[str]):
    offset = None
    while True:
        records, offset = await client.scroll(
            collection_name=collection_name, scroll_filter=scroll_filter, limit=_PAGE, offset=offset,
            with_payload=fields, with_vectors=False,
        )
        for record in records:
            yield record
        if offset is None:
            return


def _common(values) -> object:
    """The most frequent value (ties: the first in sorted order of their repr)."""
    counts = Counter(values)
    if not counts:
        return None
    best = max(counts.values())
    return sorted((v for v, n in counts.items() if n == best), key=repr)[0]


def _stamp(payload: dict) -> int:
    """A point's ingest_version; 0 when it has none (written before Decision 97)."""
    value = payload.get("ingest_version")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


async def read_document(
    client,
    deal_id: str,
    doc_id: str,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
) -> DocumentPoints:
    """Reads one document's points (state fields only) from both collections."""
    doc = DocumentPoints(deal_id, doc_id)
    flt = _deal_filter(deal_id, doc_id)
    async for record in _scroll(client, collection_name, flt, CHILD_FIELDS):
        doc.children.append((str(record.id), record.payload or {}))
    async for record in _scroll(client, parent_collection_name, flt, PARENT_FIELDS):
        doc.parents.append((str(record.id), record.payload or {}))
    return doc


def document_problems(doc: DocumentPoints) -> list[str]:
    """
    Why a document's points do not form one document written by one run.

    Every check holds for a document index_document wrote completely, at any
    version: one stamp on all its points, one version state, each child's id
    derived from its chunk id and position, each child held by the parent it
    names, each parent holding only children that name it, no parent without
    a child. A re-index cut short breaks one of them when it changed the
    chunks: the points then carry two stamps (or the new stamp beside none),
    and new children point at parents an old run wrote.

    Returns:
        Sorted problem codes; empty when consistent.
    """
    from src.data_processing.ingest_pipeline import point_id_for

    problems: set[str] = set()
    points = doc.children + doc.parents
    if doc.parents and not doc.children:
        # Parents with no child: a first upload whose rollback could not
        # delete them. Nothing of it is searchable; whether it should exist
        # is not the file's to say.
        problems.add("orphan_parents_only")
    if len({_stamp(p) for _, p in points}) > 1:
        problems.add("mixed_stamps")
    if len({int(p.get("is_current_version", 1) or 0) for _, p in points}) > 1 \
            or len({p.get("superseded_by") or "" for _, p in points}) > 1:
        problems.add("mixed_version_state")
    # What one upload writes on every child: two values mean two uploads under
    # one doc_id, cut short (the same bytes re-uploaded under another name,
    # category override or supersedes link).
    if any(len({p.get(f) or "" for _, p in doc.children}) > 1 for f in DOCUMENT_FIELDS):
        problems.add("mixed_document_fields")

    children: dict[str, dict] = {}
    for point_id, payload in doc.children:
        chunk_id = payload.get("chunk_id") or ""
        index = payload.get("chunk_index")
        if chunk_id in children:
            problems.add("duplicate_chunk_id")
        children[chunk_id] = payload
        expected = f"{doc.deal_id}_{doc.doc_id}_{index:04d}" if isinstance(index, int) else None
        if chunk_id != expected or point_id != point_id_for(chunk_id):
            problems.add("foreign_point_id")
    parents: dict[str, dict] = {}
    for point_id, payload in doc.parents:
        chunk_id = payload.get("chunk_id") or ""
        if chunk_id in parents:
            problems.add("duplicate_chunk_id")
        parents[chunk_id] = payload
        if point_id != point_id_for(chunk_id):
            problems.add("foreign_point_id")

    referenced: set[str] = set()
    for chunk_id, payload in children.items():
        parent_id = payload.get("parent_chunk_id")
        if not parent_id:
            continue  # tables and redlines have no parent
        referenced.add(parent_id)
        parent = parents.get(parent_id)
        if parent is None:
            problems.add("missing_parent")
        elif chunk_id not in (parent.get("child_chunk_ids") or []):
            problems.add("child_not_held_by_parent")
    for chunk_id, payload in parents.items():
        if chunk_id not in referenced:
            problems.add("orphan_parent")
        for child_id in payload.get("child_chunk_ids") or []:
            child = children.get(child_id)
            if child is None or child.get("parent_chunk_id") != chunk_id:
                problems.add("parent_holds_other_children")
    return sorted(problems)


def plan_document(doc: DocumentPoints, identity: dict | None, identity_unchanged: bool | None,
                  identity_failed: bool = False) -> dict:
    """
    What one document needs, and why.

    `operator` (before anything else): a problem the driver must not resolve
    by guessing (OPERATOR_PROBLEMS); the row lists the values in conflict.
    `identity_failed`: nothing else to do, but the identity extractor raised
    on the document, so the backfill would leave it as it is (as it counts it,
    "failed") — a bug to look at, not "nothing".

    Args:
        doc: Its points.
        identity: The identity the backfill would write (None: not computed,
            or the extractor raised).
        identity_unchanged: Whether its points already carry exactly that.
        identity_failed: The extractor raised on this document's opening.

    Returns:
        The document's row of the report.
    """
    from src.data_processing.ingest_pipeline import CURRENT_SINCE, INGEST_VERSION
    from src.vector_db.hybrid_search import SPARSE_ENCODING_VERSION

    child_payloads = [p for _, p in doc.children]
    file_type = _common(p.get("file_type") for p in child_payloads if p.get("file_type"))
    stamps = sorted({_stamp(p) for _, p in doc.children + doc.parents})
    encodings = sorted({p.get("sparse_encoding") for p in child_payloads}, key=repr)
    problems = document_problems(doc)

    reasons = []
    if problems:
        reasons.append("inconsistent: " + ", ".join(problems))
    if not child_payloads:
        reasons.append("no child points")
    if any(e != SPARSE_ENCODING_VERSION for e in encodings):
        reasons.append(f"BM25 encoding {', '.join(map(str, encodings))} (current {SPARSE_ENCODING_VERSION})")
    # Table rows stored without their column order (Decision 108): a Qdrant
    # server read over gRPC returns `values` shuffled, and for header-named
    # columns nothing else says the order. Read from the rows themselves, not
    # the stamp: rows whose columns are placeholders, or which have one cell,
    # read correctly at any version.
    unordered = sum(
        1 for p in child_payloads
        if p.get(ROWS_UNORDERED, rows_unordered(p.get("structured_rows")))
    )
    if unordered:
        reasons.append(f"{unordered} table chunk(s) store rows without their column order "
                       "(written before Decision 108; read over gRPC their columns are shuffled)")
    since = CURRENT_SINCE.get(file_type or "")
    if since is not None and min(stamps, default=0) < since:
        written = "not stamped (indexed before Decision 97)" if min(stamps, default=0) == 0 \
            else f"stamped {min(stamps)}"
        reasons.append(f"{file_type} extraction changed at ingest_version {since}; {written}")

    for_operator = [p for p in problems if p in OPERATOR_PROBLEMS]
    if for_operator:
        action = "operator"
        reasons.insert(0, "for an operator, not the driver: " + ", ".join(for_operator))
    elif reasons:
        action = "reingest"
    elif identity_failed:
        action = "identity_failed"
        reasons.append("the identity extractor raised on this document (the backfill counts it failed)")
    elif identity is not None and identity_unchanged is False:
        action = "backfill_identity"
        reasons.append("the identity backfill would change doc_title / doc_parties")
    else:
        action = "nothing"

    conflicts = {}
    if "mixed_version_state" in problems:
        conflicts["superseded_by"] = sorted({p.get("superseded_by") or "" for _, p in doc.children + doc.parents})
        conflicts["is_current_version"] = sorted({int(p.get("is_current_version", 1) or 0)
                                                  for _, p in doc.children + doc.parents})
    if "mixed_document_fields" in problems:
        for f in DOCUMENT_FIELDS:
            values = sorted({p.get(f) or "" for p in child_payloads})
            if len(values) > 1:
                conflicts[f] = values

    return {
        "deal_id": doc.deal_id,
        "doc_id": doc.doc_id,
        "source_file": _common(p.get("source_file") for p in child_payloads if p.get("source_file")),
        "file_type": file_type,
        "document_category": _common(p.get("document_category") for p in child_payloads
                                     if p.get("document_category")),
        "is_current_version": _common(int(p.get("is_current_version", 1) or 0) for p in child_payloads),
        "superseded_by": _common(p.get("superseded_by") or "" for p in child_payloads),
        "supersedes_doc_id": _common(p.get("supersedes_doc_id") or "" for p in child_payloads),
        "children": len(doc.children),
        "parents": len(doc.parents),
        "unordered_table_chunks": unordered,
        "ingest_version": stamps[0] if len(stamps) == 1 else stamps,
        "current_ingest_version": INGEST_VERSION,
        "sparse_encoding": encodings[0] if len(encodings) == 1 else encodings,
        "identity": {
            "doc_title": any(p.get("doc_title") for p in child_payloads),
            "doc_parties": any(p.get("doc_parties") for p in child_payloads),
            "backfill_changes": None if identity is None or identity_unchanged is None else not identity_unchanged,
            "failed": identity_failed,
        },
        "problems": problems,
        "conflicts": conflicts,
        "action": action,
        "reasons": reasons,
    }


async def collect(
    client,
    *,
    deal_id: str | None = None,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
) -> tuple[dict[tuple, DocumentPoints], IdentityPlan]:
    """Reads every document's points (both collections) and the backfill's plan."""
    docs: dict[tuple, DocumentPoints] = {}
    plan = IdentityPlan()
    flt = _deal_filter(deal_id)
    fields = sorted(set(CHILD_FIELDS) | set(IdentityPlan.FIELDS) | {STAGED_DEAL_FIELD})
    async for record in _scroll(client, collection_name, flt, fields):
        payload = record.payload or {}
        key = (payload.get("deal_id") or "", payload.get("doc_id") or "")
        if not key[1] or _uncommitted(payload):
            continue
        plan.add(record.id, payload)
        state = {k: payload[k] for k in CHILD_FIELDS if k in payload}
        # A table's rows are read for one flag, not kept for the whole corpus.
        if "structured_rows" in state:
            state[ROWS_UNORDERED] = rows_unordered(state.pop("structured_rows"))
        docs.setdefault(key, DocumentPoints(*key)).children.append((str(record.id), state))
    async for record in _scroll(client, parent_collection_name, flt, [*PARENT_FIELDS, STAGED_DEAL_FIELD]):
        payload = record.payload or {}
        key = (payload.get("deal_id") or "", payload.get("doc_id") or "")
        if not key[1] or _uncommitted(payload):
            continue
        docs.setdefault(key, DocumentPoints(*key)).parents.append((str(record.id), payload))
    return docs, plan


def _uncommitted(payload: dict) -> bool:
    """A point a first-time upload staged and never committed: no deal_id yet."""
    return not payload.get("deal_id") and bool(payload.get(STAGED_DEAL_FIELD))


async def collect_staged(
    client,
    *,
    deal_id: str | None = None,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
) -> list[dict]:
    """
    Documents whose first-time upload staged points and never committed them.

    Returns:
        [{"deal_id" (the deal they were staged for), "doc_id", "source_file",
          "children", "parents"}], sorted.
    """
    must = [IsEmptyCondition(is_empty=PayloadField(key="deal_id"))]
    if deal_id:
        must.append(FieldCondition(key=STAGED_DEAL_FIELD, match=MatchValue(value=deal_id)))
    flt = Filter(must=must)
    found: dict[tuple, dict] = {}
    for name, count in ((collection_name, "children"), (parent_collection_name, "parents")):
        async for record in _scroll(client, name, flt, [STAGED_DEAL_FIELD, "doc_id", "source_file"]):
            payload = record.payload or {}
            if not _uncommitted(payload) or not payload.get("doc_id"):
                continue
            key = (str(payload[STAGED_DEAL_FIELD]), str(payload["doc_id"]))
            row = found.setdefault(key, {"deal_id": key[0], "doc_id": key[1], "source_file": None,
                                         "children": 0, "parents": 0})
            row[count] += 1
            row["source_file"] = row["source_file"] or payload.get("source_file")
    return [found[k] for k in sorted(found)]


def match_files(directory: str | Path, indexed: set[tuple]) -> tuple[dict[tuple, Path], list[Path]]:
    """
    The files under `directory` that are indexed documents, found by their bytes.

    doc_id is uuid5(deal, sha256(bytes)) (ingest_pipeline.compute_doc_id), so
    a file is the document whose id its bytes give in that deal, whatever its
    name; the same bytes in two deals are two documents.

    Args:
        directory: Where the original files are (searched recursively).
        indexed: The (deal_id, doc_id) pairs of the index.

    Returns:
        ({(deal_id, doc_id): path} for the indexed documents found, the files
        that are no indexed document, including any that cannot be read). Of
        two identical files, the first by path is used.
    """
    from src.data_processing.ingest_pipeline import compute_content_hash, compute_doc_id

    deals = sorted({deal for deal, _ in indexed})
    matched: dict[tuple, Path] = {}
    unmatched: list[Path] = []
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        try:
            digest = compute_content_hash(str(path))
        except OSError as e:
            logger.warning("Original file could not be read", extra={"path": str(path), "error": str(e)})
            unmatched.append(path)
            continue
        hits = [key for key in ((deal, compute_doc_id(deal, digest)) for deal in deals) if key in indexed]
        for key in hits:
            matched.setdefault(key, path)
        if not hits:
            unmatched.append(path)
    return matched, unmatched


async def build_report(
    client,
    *,
    deal_id: str | None = None,
    files: str | Path | None = None,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
) -> dict:
    """
    The report: one row per document, totals per deal, and the file matching.

    Args:
        client: Qdrant client (server or local); only read.
        deal_id: Only this deal (all deals when None).
        files: Directory of original files to match to the documents.
        collection_name: Child collection.
        parent_collection_name: Parent collection.

    Returns:
        {"ingest_version", "current_since", "sparse_encoding", "deals": {deal: {"documents": [...],
         "totals": {...}}}, "staged": [collect_staged rows], "files": {...} (with `files`)}
    """
    from src.data_processing.ingest_pipeline import CURRENT_SINCE, INGEST_VERSION
    from src.vector_db.hybrid_search import SPARSE_ENCODING_VERSION

    docs, plan = await collect(client, deal_id=deal_id, collection_name=collection_name,
                               parent_collection_name=parent_collection_name)

    def rows() -> list[dict]:
        out = []
        for key in sorted(docs):
            doc = docs[key]
            identity = plan.identity(key) if key in plan.point_ids else None
            unchanged = plan.unchanged(key, identity) if identity is not None else None
            failed = key in plan.point_ids and identity is None
            out.append(plan_document(doc, identity, unchanged, identity_failed=failed))
        return out

    # The identity extractor is regex work over each opening: off the event loop.
    documents = await asyncio.to_thread(rows)
    matched: dict[tuple, Path] = {}
    report: dict = {
        "ingest_version": INGEST_VERSION,
        "current_since": dict(CURRENT_SINCE),
        "sparse_encoding": SPARSE_ENCODING_VERSION,
        "deals": {},
        "staged": await collect_staged(client, deal_id=deal_id, collection_name=collection_name,
                                       parent_collection_name=parent_collection_name),
    }
    if files is not None:
        matched, unmatched = await asyncio.to_thread(
            match_files, files, {(d["deal_id"], d["doc_id"]) for d in documents})
        report["files"] = {
            "directory": str(files),
            "matched": sum(1 for d in documents if (d["deal_id"], d["doc_id"]) in matched),
            "not_indexed": [str(p) for p in unmatched],
        }
    for row in documents:
        if files is not None:
            path = matched.get((row["deal_id"], row["doc_id"]))
            row["file"] = str(path) if path else None
        deal = report["deals"].setdefault(row["deal_id"], {"documents": [], "totals": {}})
        deal["documents"].append(row)
    for deal in report["deals"].values():
        deal["totals"] = _totals(deal["documents"], with_files=files is not None)
    return report


def _totals(rows: list[dict], with_files: bool) -> dict:
    by_action = Counter(r["action"] for r in rows)
    groups = Counter(
        (r["file_type"] or "?", json.dumps(r["ingest_version"]), _identity_state(r["identity"]), r["action"])
        for r in rows)
    totals = {
        "documents": len(rows),
        "by_action": {a: by_action.get(a, 0) for a in ACTIONS},
        "by_type_stamp_identity_action": [
            {"file_type": t, "ingest_version": json.loads(s), "identity": i, "action": a, "documents": n}
            for (t, s, i, a), n in sorted(groups.items())],
        "inconsistent": sum(1 for r in rows if r["problems"]),
        "identity_failed": sum(1 for r in rows if r["identity"].get("failed")),
    }
    if with_files:
        totals["reingest_without_file"] = sum(1 for r in rows if r["action"] == "reingest" and not r.get("file"))
    return totals


def _identity_state(identity: dict) -> str:
    has = "title+parties" if identity["doc_title"] and identity["doc_parties"] else \
        "title" if identity["doc_title"] else "parties" if identity["doc_parties"] else "none"
    if identity.get("failed"):
        has += ", extractor failed"
    elif identity["backfill_changes"]:
        has += ", backfill changes it"
    return has


def format_report(report: dict) -> str:
    """The report as text: per deal, documents by type, stamp, identity and action."""
    lines = [f"ingest_version {report['ingest_version']}, current since {report['current_since']}, "
             f"BM25 encoding {report['sparse_encoding']}"]
    for deal_id, deal in sorted(report["deals"].items()):
        totals = deal["totals"]
        # ASCII only: a Windows console's code page cannot print every dash.
        lines.append(f"\n{deal_id}: {totals['documents']} documents: "
                     + ", ".join(f"{a} {n}" for a, n in totals["by_action"].items())
                     + (f"; inconsistent {totals['inconsistent']}" if totals["inconsistent"] else "")
                     + (f"; to re-ingest with no file {totals['reingest_without_file']}"
                        if totals.get("reingest_without_file") else ""))
        for g in totals["by_type_stamp_identity_action"]:
            lines.append(f"  {g['documents']:5d}  {g['file_type']:<5} stamp {json.dumps(g['ingest_version']):<8} "
                         f"identity {g['identity']:<34} -> {g['action']}")
        for row in deal["documents"]:
            if row["action"] != "nothing":
                file_note = "" if "file" not in row else f"  file: {row['file'] or 'NOT PROVIDED'}"
                lines.append(f"    {row['action']:<17} {row['source_file']} ({row['doc_id']}): "
                             f"{'; '.join(row['reasons'])}{file_note}")
                for field_name, values in row.get("conflicts", {}).items():
                    lines.append(f"        {field_name}: {json.dumps(values, ensure_ascii=True)}")
    if report.get("staged"):
        lines.append(f"\nstaged, never committed: {len(report['staged'])} documents (an upload cut short; "
                     "nothing reads these points; purging the deal deletes them, uploading the same file "
                     "again commits it)")
        for row in report["staged"]:
            lines.append(f"    {row['deal_id']}: {row['source_file']} ({row['doc_id']}): "
                         f"{row['children']} children, {row['parents']} parents")
    if "files" in report:
        files = report["files"]
        lines.append(f"\nfiles under {files['directory']}: {files['matched']} matched to a document, "
                     f"{len(files['not_indexed'])} match none")
    return "\n".join(lines)


async def _main(args) -> dict:
    if args.path:
        client = AsyncQdrantClient(path=args.path)
    else:
        from src.vector_db.qdrant_client import get_qdrant_client
        client = get_qdrant_client()
    try:
        return await build_report(client, deal_id=args.deal, files=args.files)
    finally:
        await client.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deal", help="only this deal_id")
    parser.add_argument("--path", help="a local-mode Qdrant directory instead of the configured server")
    parser.add_argument("--files", help="directory of original files to match to the documents by their bytes")
    parser.add_argument("--json", help="also write the full report (every document) to this file")
    args = parser.parse_args(argv)
    report = asyncio.run(_main(args))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(format_report(report), file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
