"""
Retrieval on real documents — LegalBench-RAG (contracts, character spans) and
FinanceBench (10-K/10-Q PDFs, evidence pages). Zero LLM calls.

    python -m eval.run_realdocs_eval legalbench
    python -m eval.run_realdocs_eval legalbench --subsets maud cuad
    python -m eval.run_realdocs_eval financebench --max-documents 20
    python -m eval.run_realdocs_eval financebench --documents 3M_2018_10K PEPSICO_2022_10K
    python -m eval.run_realdocs_eval legalbench --compare eval/results/realdocs_legalbench_<date>.json

Why: every other quality number in this repository is measured on nine
synthetic plain-text documents (~13K tokens) that fit in one prompt, so no
retrieval design can be told apart from context stuffing, and every PDF
claim rests on generated fixtures. These two public benchmarks are real.

What runs is the production path, as in eval/run_retrieval_eval.py: each
document goes through index_document (parsing, chunking, PII flagging, bge-m3
dense + BM25 sparse) into an in-memory Qdrant, and each question through
dense / sparse / hybrid_rrf (the production hybrid_search and RRF lists) and
production (retrieval_executor_node: RRF -> cross-encoder -> threshold ->
parent/sibling expansion). The retrieval config is RETRIEVAL_CONFIGS["legal"]
for LegalBench-RAG and ["financial"] for FinanceBench, with no metadata
filters; Agent 1 (an LLM) is not run, so decomposition is not measured — the
benchmarks ship no sub-questions and producing them needs LLM calls.

Laya's ingest decisions are off (LAYA_ENABLED=0 for indexing, as in the
distractor mode): they do not change what is retrieved except through the
PII second opinion, and the rule-based PII flag still applies — a PII-flagged
chunk is unreachable exactly as in production, and the share of ground truth
that sits in such chunks is reported as the reachable ceiling.

LegalBench-RAG scoring (the benchmark's own metrics; eval/span_alignment.py):
chunks are mapped back to character offsets of their source file, then
    char_precision@k / char_recall@k   characters of the top-k chunks that lie in
                                       ground-truth spans / of the ground truth
                                       retrieved (retrieved spans merged first)
    snippet_recall@k                   share of ground-truth snippets touched by
                                       a top-k chunk
    hit@k, mrr@10                      a top-k chunk touches the ground truth
    doc_hit@k                          a top-k chunk comes from the right file
and @context: the same on the final context exactly as the synthesizer would
receive it (parents, one copy each, MAX_CONTEXT_CHARS budget).

FinanceBench scoring: the evidence page (FinanceBench gives a zero-based page;
the pipeline numbers pages from 1) of the right filing among the top-k chunks
(page_hit@k, mrr@10) and in the final context (page_hit@context), plus
evidence-token recall — the share of the annotator's evidence tokens present
in the final context, and in everything the index holds for that page (the
ingestion ceiling: what the parser kept of the page at all).

Intervals are percentile bootstraps that resample DOCUMENTS, not questions:
mini has ~10 queries per document, and those are not independent draws.
Writes eval/results/realdocs_<benchmark>_<date>.json and .md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval import realdocs_data as data
from eval import span_alignment as sa
from eval.decisions import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

ABLATIONS = ("dense", "sparse", "hybrid_rrf", "production")
ABLATION_LABELS = {
    "dense": "dense-only",
    "sparse": "sparse-only (BM25)",
    "hybrid_rrf": "hybrid RRF (no rerank)",
    "production": "production (hybrid + rerank + expansion)",
}
RANK_DEPTH = 10
DEFAULT_K = (1, 5, 10)
QUERY_TYPE = {"legalbench": "legal", "financebench": "financial"}
DEAL_ID = {"legalbench": "realdocs_legalbench", "financebench": "realdocs_financebench"}
STAGE_PAIRS = (("dense", "hybrid_rrf"), ("sparse", "hybrid_rrf"), ("hybrid_rrf", "production"))


# ==============================================================================
# Index
# ==============================================================================


async def new_index():
    from qdrant_client import AsyncQdrantClient

    from src.vector_db.collection_manager import setup_collections
    from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME

    client = AsyncQdrantClient(location=":memory:")
    await setup_collections(client, COLLECTION_NAME, PARENT_COLLECTION_NAME)
    return client


async def open_store(path: Path):
    """An on-disk local Qdrant (created on first use) and its ingestion log."""
    from qdrant_client import AsyncQdrantClient

    from src.vector_db.collection_manager import setup_collections
    from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME

    path.mkdir(parents=True, exist_ok=True)
    client = AsyncQdrantClient(path=str(path / "qdrant"))
    if not await client.collection_exists(COLLECTION_NAME):
        await setup_collections(client, COLLECTION_NAME, PARENT_COLLECTION_NAME)
    log = path / "ingestion.json"
    return client, (json.loads(log.read_text(encoding="utf-8")) if log.exists() else [])


async def scroll_all(client, collection: str) -> list[dict]:
    payloads, offset = [], None
    while True:
        points, offset = await client.scroll(collection_name=collection, limit=1000, offset=offset,
                                             with_payload=True, with_vectors=False)
        payloads.extend(p.payload for p in points)
        if offset is None:
            return payloads


async def index_files(client, files: list[tuple[Path, str]], deal_id: str) -> list[dict]:
    """
    index_document for each (path, display filename), timing each and keeping
    failures (with the pipeline's own reason) instead of stopping the run.
    """
    from src.data_processing.errors import IngestionError
    from src.data_processing.ingest_pipeline import index_document

    previous = os.environ.get("LAYA_ENABLED")
    os.environ["LAYA_ENABLED"] = "0"
    results = []
    try:
        for n, (path, filename) in enumerate(files, start=1):
            started = time.perf_counter()
            try:
                result = await index_document(str(path), filename, deal_id, None, client=client)
                results.append({**{k: result[k] for k in (
                    "filename", "document_category", "chunks_created", "parent_chunks_created",
                    "table_count", "warnings")}, "seconds": round(time.perf_counter() - started, 2),
                    "ok": True})
            except IngestionError as e:
                results.append({"filename": filename, "ok": False, "error_type": type(e).__name__,
                                "error": str(e), "seconds": round(time.perf_counter() - started, 2)})
            print(f"[realdocs] indexed {n}/{len(files)} {filename} "
                  f"({results[-1]['seconds']}s{'' if results[-1]['ok'] else ', FAILED'})",
                  file=sys.stderr)
    finally:
        if previous is None:
            os.environ.pop("LAYA_ENABLED", None)
        else:
            os.environ["LAYA_ENABLED"] = previous
    return results


def is_reachable(chunk: dict, deal_id: str) -> bool:
    """hybrid_search._build_filter's non-negotiable conditions with include_pii=False."""
    return (chunk.get("deal_id") == deal_id and chunk.get("is_current_version") == 1
            and chunk.get("contains_pii") == 0)


# ==============================================================================
# Retrieval runs (production functions, production order)
# ==============================================================================


async def candidate_runs(query: str, config: dict, deal_id: str) -> dict:
    """dense / sparse / hybrid_rrf rankings from one hybrid_search (as run_retrieval_eval)."""
    from src.vector_db.hybrid_search import compute_sparse_bm25, fetch_chunks_by_ids, hybrid_search
    from src.vector_db.reranker import embed_texts_async, get_embed_executor
    from src.vector_db.rrf_fusion import reciprocal_rank_fusion

    start = time.perf_counter()
    query_vector = (await embed_texts_async([query]))[0].tolist()
    loop = asyncio.get_running_loop()
    query_sparse = await loop.run_in_executor(get_embed_executor(), lambda: compute_sparse_bm25(query))
    dense, sparse = await hybrid_search(
        query_text=query, query_vector=query_vector, query_sparse=query_sparse, deal_id=deal_id,
        metadata_filters={"include_pii": False}, top_k_dense=config.get("top_k_dense", 40),
        top_k_sparse=config.get("top_k_sparse", 40))
    search_ms = (time.perf_counter() - start) * 1000
    fused = reciprocal_rank_fusion(dense_results=dense, sparse_results=sparse, k=60,
                                   dense_weight=config.get("dense_weight", 0.6),
                                   sparse_weight=config.get("sparse_weight", 0.4))
    fused_chunks = await fetch_chunks_by_ids([cid for cid, _ in fused[: config.get("reranker_top_k", 20)]])
    total_ms = (time.perf_counter() - start) * 1000
    return {
        "dense": {"ranking": [p.payload for p in dense], "latency_ms": search_ms},
        "sparse": {"ranking": [p.payload for p in sparse], "latency_ms": search_ms},
        "hybrid_rrf": {"ranking": fused_chunks, "latency_ms": total_ms},
    }


async def node_run(query: str, query_type: str, deal_id: str) -> dict:
    """One call of the real retrieval_executor_node, plus the heuristic refusal gate."""
    from eval.run_retrieval_eval import gate_decision
    from src.agents.retrieval_executor import retrieval_executor_node

    state = {"original_query": query, "current_query": query, "deal_id": deal_id,
             "query_type": query_type, "parsed_intent": {}, "extracted_filters": {},
             "sub_questions": [], "include_pii": False, "rewrite_iteration": 0}
    start = time.perf_counter()
    out = await retrieval_executor_node(state)
    latency_ms = (time.perf_counter() - start) * 1000
    gate = await gate_decision(state, out["reranked_results"], laya=False)
    return {"ranking": out["reranked_results"], "context": out["expanded_context"],
            "latency_ms": latency_ms, "gate": gate["decision"],
            "max_reranker_score": gate["max_reranker_score"]}


def context_bodies(chunks: list[dict]) -> list[tuple[dict, str]]:
    """
    (chunk, body) for each <document> the synthesizer would show, in order.

    Mirrors answer_synthesizer._format_context_for_synthesis (one copy per
    parent, rank order, MAX_CONTEXT_CHARS, fall back to the child's own text
    past the budget); run() checks the count against the real formatter.
    """
    from src.agents.answer_synthesizer import MAX_CONTEXT_CHARS
    from src.verification.evidence import chunk_evidence_text, parent_key

    out, shown, used = [], set(), 0
    for chunk in chunks:
        key = parent_key(chunk)
        if key is not None and key in shown:
            continue
        body = chunk_evidence_text(chunk)
        if used + len(body) > MAX_CONTEXT_CHARS:
            body = chunk.get("text", "") or ""
            key = None
            if used + len(body) > MAX_CONTEXT_CHARS:
                continue
        if key is not None:
            shown.add(key)
        used += len(body)
        out.append((chunk, body))
    return out


def _check_context_mirror(chunks: list[dict], bodies: list) -> None:
    from src.agents.answer_synthesizer import _format_context_for_synthesis

    rendered = _format_context_for_synthesis(chunks)
    count = len(re.findall(r"<document\b", rendered))
    if count != len(bodies):
        raise RuntimeError(f"context mirror drifted from the synthesizer: {len(bodies)} bodies, "
                           f"{count} <document> elements")


# ==============================================================================
# LegalBench-RAG
# ==============================================================================


class SpanIndex:
    """chunk_id / parent id -> (corpus file, source spans), built once per run."""

    def __init__(self, chunks: list[dict], parents: list[dict], source_of: dict[str, str],
                 texts: dict[str, str]):
        self.file_of = source_of  # source_file (display name) -> corpus path
        normalised = {path: sa.NormalisedText(text) for path, text in texts.items()}
        self.spans: dict[str, tuple[str, list[tuple[int, int]]]] = {}
        self.coverage: list[float] = []
        self.unaligned_chars = 0
        self.total_chars = 0
        by_doc: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            by_doc[chunk["source_file"]].append(chunk)
        for source_file, doc_chunks in by_doc.items():
            path = source_of[source_file]
            hint = 0
            for chunk in sorted(doc_chunks, key=lambda c: c["chunk_index"]):
                alignment = sa.align(chunk.get("text") or "", normalised[path], hint)
                if alignment.norm_start is not None:
                    hint = alignment.norm_start
                self.spans[chunk["chunk_id"]] = (path, alignment.spans)
                self.coverage.append(alignment.coverage)
                self.total_chars += alignment.total_chars
                self.unaligned_chars += alignment.total_chars - alignment.matched_chars
        for parent in parents:
            path = source_of[parent["source_file"]]
            alignment = sa.align(parent.get("text") or "", normalised[path])
            self.spans[parent["chunk_id"]] = (path, alignment.spans)

    def chunk_spans(self, chunk: dict) -> tuple[str, list[tuple[int, int]]]:
        return self.spans.get(chunk["chunk_id"], (self.file_of.get(chunk.get("source_file"), ""), []))

    def body_spans(self, chunk: dict, body: str) -> tuple[str, list[tuple[int, int]]]:
        """Spans of what a context <document> shows: the parent, the child, or both."""
        path, spans = self.chunk_spans(chunk)
        parent_id = chunk.get("parent_chunk_id")
        if chunk.get("parent_text") and parent_id in self.spans and body != (chunk.get("text") or ""):
            parent_path, parent_spans = self.spans[parent_id]
            if body == chunk.get("parent_text"):
                return parent_path, parent_spans
            return path, sa.merge_spans(spans + parent_spans)
        return path, spans


def _file_spans(items: list[tuple[str, list[tuple[int, int]]]]) -> dict:
    out: dict[str, list] = defaultdict(list)
    for path, spans in items:
        if path:
            out[path].extend(spans)
    return out


def score_spans(items: list[tuple[str, list[tuple[int, int]]]], query: data.SpanQuery,
                ks: list[int], context: list | None = None) -> dict:
    """LegalBench-RAG metrics for one ranking (and context) of one query."""
    truth = _file_spans([(p, [(s, e)]) for p, s, e in query.snippets])
    metrics: dict = {}
    for k in ks:
        top = _file_spans(items[:k])
        precision, recall = sa.char_precision_recall(top, truth)
        metrics[f"char_precision@{k}"] = precision
        metrics[f"char_recall@{k}"] = recall
        metrics[f"snippet_recall@{k}"] = sum(
            1 for p, s, e in query.snippets if sa.touches(top.get(p, []), (s, e))) / len(query.snippets)
        metrics[f"hit@{k}"] = float(metrics[f"snippet_recall@{k}"] > 0)
        metrics[f"doc_hit@{k}"] = float(any(p == query.document for p, _ in items[:k]))
    first = next((rank for rank, (p, spans) in enumerate(items[:RANK_DEPTH], start=1)
                  if any(sa.touches(spans, (s, e)) for q, s, e in query.snippets if q == p)), None)
    metrics[f"mrr@{RANK_DEPTH}"] = 1.0 / first if first else 0.0
    if context is not None:
        ctx = _file_spans(context)
        precision, recall = sa.char_precision_recall(ctx, truth)
        metrics["char_precision@context"] = precision
        metrics["char_recall@context"] = recall
        metrics["snippet_recall@context"] = sum(
            1 for p, s, e in query.snippets if sa.touches(ctx.get(p, []), (s, e))) / len(query.snippets)
        # The system's own fact-coverage view: a snippet counts when the
        # context holds (nearly) all of it, not just one character.
        metrics["snippet_full@context"] = sum(
            1 for p, s, e in query.snippets
            if sa.overlap_length(ctx.get(p, []), [(s, e)]) >= 0.9 * (e - s)) / len(query.snippets)
    return metrics


def reachable_ceiling(query: data.SpanQuery, reachable_spans: dict) -> float:
    """Share of the ground-truth characters inside any chunk production can return."""
    truth = _file_spans([(p, [(s, e)]) for p, s, e in query.snippets])
    return sa.char_precision_recall(reachable_spans, truth)[1]


def write_corpus(documents: dict[str, str], directory: Path) -> tuple[list[tuple[Path, str]], dict[str, str]]:
    """
    Writes each document as UTF-8 .txt under a generated name and returns the
    (path, display filename) list and display filename -> corpus path.

    The display name is what an uploader would send: the corpus file's base
    name, sanitised the way POST /ingest sanitises it.
    """
    from src.data_processing.ingest_pipeline import sanitize_filename

    files, source_of = [], {}
    for n, (corpus_path, text) in enumerate(sorted(documents.items())):
        name = sanitize_filename(corpus_path.split("/")[-1])
        if not name.lower().endswith(".txt"):
            name += ".txt"
        if name in source_of:
            name = f"{n:03d}_{name}"
        path = directory / f"doc_{n:03d}.txt"
        path.write_text(text, encoding="utf-8", newline="")
        files.append((path, name))
        source_of[name] = corpus_path
    return files, source_of


async def run_legalbench(args) -> dict:
    from eval.corpus import use_client
    from src.agents.retrieval_strategy import get_retrieval_config
    from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME
    from src.vector_db.reranker import warm_models

    deal_id = DEAL_ID["legalbench"]
    mini = data.load_legalbench_mini(subsets=tuple(args.subsets))
    queries = mini.queries
    if args.max_queries:
        per: Counter = Counter()
        kept = []
        for q in queries:
            if per[q.subset] < args.max_queries:
                kept.append(q)
                per[q.subset] += 1
        queries = kept
    documents = {p: mini.documents[p] for p in sorted({s[0] for q in queries for s in q.snippets})}

    started = time.perf_counter()
    client = await new_index()
    with tempfile.TemporaryDirectory(prefix="realdocs_") as tmp:
        files, source_of = write_corpus(documents, Path(tmp))
        ingestion = await index_files(client, files, deal_id)
    index_s = time.perf_counter() - started
    chunks = await scroll_all(client, COLLECTION_NAME)
    parents = await scroll_all(client, PARENT_COLLECTION_NAME)
    span_index = SpanIndex(chunks, parents, source_of, documents)
    reachable = _file_spans([span_index.chunk_spans(c) for c in chunks if is_reachable(c, deal_id)])
    config = get_retrieval_config(QUERY_TYPE["legalbench"], {})

    report_queries = []
    with use_client(client):
        await warm_models()
        for n, query in enumerate(queries, start=1):
            runs: dict = {}
            if set(args.ablations) & {"dense", "sparse", "hybrid_rrf"}:
                runs.update({a: r for a, r in (await candidate_runs(query.query, config, deal_id)).items()
                             if a in args.ablations})
            if "production" in args.ablations:
                runs["production"] = await node_run(query.query, QUERY_TYPE["legalbench"], deal_id)
            scored = {}
            for ablation, run in runs.items():
                items = [span_index.chunk_spans(c) for c in run["ranking"]]
                context = None
                if "context" in run:
                    bodies = context_bodies(run["context"])
                    if n <= 5:
                        _check_context_mirror(run["context"], bodies)
                    context = [span_index.body_spans(c, b) for c, b in bodies]
                metrics = score_spans(items, query, args.k, context)
                entry = {"metrics": {k: round(v, 4) for k, v in metrics.items()},
                         "latency_ms": round(run["latency_ms"], 1)}
                if "gate" in run:
                    entry.update(gate=run["gate"], max_reranker_score=run["max_reranker_score"],
                                 context_documents=len(context or []),
                                 context_chars=sum(sa.span_length(sa.merge_spans(s)) for _, s in context or []),
                                 # For diagnosis: where the top five came from.
                                 top=[{"file": c.get("source_file"), "chunk_index": c.get("chunk_index"),
                                       "is_table": c.get("is_table"),
                                       "relevant": any(sa.touches(sp, (s, e)) for p, s, e in query.snippets
                                                       if p == path),
                                       "reranker_score": round(float(c.get("reranker_score", 0.0)), 4)}
                                      for c, (path, sp) in zip(run["ranking"][:5], items[:5])])
                scored[ablation] = entry
            report_queries.append({
                "id": query.id, "subset": query.subset, "document": query.document,
                "query": query.query, "n_snippets": len(query.snippets),
                "truth_chars": sum(e - s for _, s, e in query.snippets),
                "reachable_ceiling": round(reachable_ceiling(query, reachable), 4),
                "runs": scored,
            })
            if n % 50 == 0 or n == len(queries):
                print(f"[realdocs] {n}/{len(queries)} queries", file=sys.stderr)

    coverage = sorted(span_index.coverage)
    return {
        "benchmark": "legalbench",
        "dataset": {
            "name": "LegalBench-RAG mini (authors' benchmark.py selection)",
            "zip_sha256": data.LEGALBENCH_SHA256, "code_revision": data.LEGALBENCH_CODE_REVISION,
            "licences": data.LEGALBENCH_LICENCES, "subsets": list(args.subsets),
            "queries": len(queries), "documents": len(documents),
            "characters": sum(len(t) for t in documents.values()),
        },
        "config": {"k": args.k, "rank_depth": RANK_DEPTH, "ablations": list(args.ablations),
                   "query_type": QUERY_TYPE["legalbench"], "retrieval_config": config},
        "index": {
            "chunks": len(chunks), "parents": len(parents),
            "reachable_chunks": sum(1 for c in chunks if is_reachable(c, deal_id)),
            "pii_flagged_chunks": sum(1 for c in chunks if c.get("contains_pii")),
            "table_chunks": sum(1 for c in chunks if c.get("is_table")),
            "index_seconds": round(index_s, 1),
            "alignment": {
                "chunks_fully_aligned": sum(1 for c in coverage if c >= 1.0),
                "chunks": len(coverage),
                "unaligned_char_share": round(span_index.unaligned_chars / max(1, span_index.total_chars), 4),
                "p05_coverage": round(coverage[len(coverage) // 20], 4) if coverage else None,
            },
            "documents": ingestion,
        },
        "queries": report_queries,
    }


# ==============================================================================
# FinanceBench
# ==============================================================================

_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def evidence_tokens(text: str) -> set[str]:
    """
    Distinct tokens of an evidence extract: lower-case words of 3+ letters and
    numbers with separators removed ("1,577.0" -> "1577.0"). FinanceBench's
    extracts come from a different PDF parser ("Flow s", one cell per line),
    so a token view is what survives both parsers.
    """
    text = re.sub(r"(?<=\d),(?=\d)", "", (text or "").lower())
    return {t for t in _TOKEN.findall(text) if len(t) >= 3 or t[0].isdigit()}


def token_recall(evidence: set[str], text: str) -> float | None:
    if not evidence:
        return None
    return len(evidence & evidence_tokens(text)) / len(evidence)


def chunk_pages(chunk: dict) -> set[int]:
    """Pages a chunk is cited with: its page, or the whole range of a stitched table."""
    pages = {int(chunk["page_number"])} if chunk.get("page_number") else set()
    page_range = chunk.get("page_range") or []
    if len(page_range) == 2:
        pages |= set(range(int(page_range[0]), int(page_range[1]) + 1))
    return pages


def score_pages(run: dict, question: data.FinanceQuestion, filename_of: dict[str, str],
                ks: list[int], pages_of_parent: dict[str, set[int]]) -> dict:
    targets = {(filename_of.get(e.doc_name), e.page) for e in question.evidence}
    docs = {f for f, _ in targets}

    def hits(chunk: dict) -> bool:
        return any((chunk.get("source_file"), p) in targets for p in chunk_pages(chunk))

    ranking = run["ranking"]
    metrics = {}
    for k in ks:
        metrics[f"page_hit@{k}"] = float(any(hits(c) for c in ranking[:k]))
        metrics[f"doc_hit@{k}"] = float(any(c.get("source_file") in docs for c in ranking[:k]))
    first = next((r for r, c in enumerate(ranking[:RANK_DEPTH], start=1) if hits(c)), None)
    metrics[f"mrr@{RANK_DEPTH}"] = 1.0 / first if first else 0.0
    if "context" in run:
        bodies = context_bodies(run["context"])
        hit = False
        for chunk, body in bodies:
            pages = set(chunk_pages(chunk))
            if body != (chunk.get("text") or "") and chunk.get("parent_chunk_id"):
                pages |= pages_of_parent.get(chunk["parent_chunk_id"], set())
            hit |= any((chunk.get("source_file"), p) in targets for p in pages)
        metrics["page_hit@context"] = float(hit)
        context_text = "\n".join(b for c, b in bodies if c.get("source_file") in docs)
        tokens = set().union(*(evidence_tokens(e.text) for e in question.evidence))
        metrics["evidence_token_recall@context"] = token_recall(tokens, context_text)
        top_text = "\n".join(c.get("text") or "" for c in ranking[:5] if c.get("source_file") in docs)
        metrics["evidence_token_recall@5"] = token_recall(tokens, top_text)
    return metrics


def raw_page_text(path: Path, page: int) -> str:
    """PyMuPDF's plain text of one 1-based page ("" when out of range)."""
    import fitz

    with fitz.open(path) as pdf:
        return pdf[page - 1].get_text("text") if 1 <= page <= pdf.page_count else ""


def pdf_page_stats(path: Path) -> dict:
    """Pages, pages without a text layer and characters — the parser-independent facts."""
    import fitz

    with fitz.open(path) as pdf:
        texts = [page.get_text("text") for page in pdf]
    return {"pages": len(texts), "pages_without_text": sum(1 for t in texts if not t.strip()),
            "characters": sum(len(t) for t in texts)}


async def run_financebench(args) -> dict:
    from eval.corpus import use_client
    from src.agents.retrieval_strategy import get_retrieval_config
    from src.vector_db.constants import COLLECTION_NAME
    from src.vector_db.reranker import warm_models

    deal_id = DEAL_ID["financebench"]
    questions = data.load_financebench()
    if args.documents:
        names = sorted(args.documents)
    else:
        names = data.select_financebench_documents(questions, args.max_documents, seed=args.seed)
    chosen = set(names)
    questions = [q for q in questions if q.doc_name in chosen
                 and all(e.doc_name in chosen for e in q.evidence)]

    started = time.perf_counter()
    files = [(data.financebench_pdf(name), f"{name}.pdf") for name in names]
    if args.store:
        # Parsing every filing takes about an hour on this machine (0.2-0.4 s a
        # page, 12,013 pages), longer than one run should hold the machine. An
        # on-disk store lets ingestion run in batches (--ingest-only
        # --max-new-documents N, repeated) and the questions run once at the end.
        client, ingestion = await open_store(Path(args.store))
        done = {d["filename"] for d in ingestion}
        todo = [f for f in files if f[1] not in done]
        if args.max_new_documents is not None:
            todo = todo[: args.max_new_documents]
        new = await index_files(client, todo, deal_id)
        for entry, (path, _) in zip(new, todo):
            entry.update(pdf_page_stats(path))
        ingestion += new
        (Path(args.store) / "ingestion.json").write_text(json.dumps(ingestion, indent=1), encoding="utf-8")
        missing = [f for _, f in files if f not in {d["filename"] for d in ingestion}]
        if args.ingest_only or missing:
            await client.close()
            print(f"[realdocs] store {args.store}: {len(ingestion)} filings ingested, "
                  f"{len(missing)} still to do", file=sys.stderr)
            return {"benchmark": "financebench", "ingest_only": True, "remaining": missing}
        ingestion = [d for d in ingestion if d["filename"] in {f for _, f in files}]
    else:
        client = await new_index()
        ingestion = await index_files(client, files, deal_id)
        for entry, (path, _) in zip(ingestion, files):
            entry.update(pdf_page_stats(path))
    index_s = time.perf_counter() - started
    filename_of = {name: f"{name}.pdf" for name in names}
    chunks = await scroll_all(client, COLLECTION_NAME)
    pages_of_parent: dict[str, set[int]] = defaultdict(set)
    page_text: dict[tuple[str, int], list[str]] = defaultdict(list)
    for chunk in chunks:
        if chunk.get("parent_chunk_id"):
            pages_of_parent[chunk["parent_chunk_id"]] |= chunk_pages(chunk)
        for page in chunk_pages(chunk):
            page_text[(chunk["source_file"], page)].append(chunk.get("text") or "")
    for entry in ingestion:
        ft = [c for c in chunks if c["source_file"] == entry["filename"]]
        entry["content_types"] = dict(Counter(c.get("content_type") for c in ft))
        entry["pii_flagged_chunks"] = sum(1 for c in ft if c.get("contains_pii"))
    config = get_retrieval_config(QUERY_TYPE["financebench"], {})

    report_questions = []
    with use_client(client):
        await warm_models()
        for n, question in enumerate(questions, start=1):
            runs: dict = {}
            if set(args.ablations) & {"dense", "sparse", "hybrid_rrf"}:
                runs.update({a: r for a, r in (await candidate_runs(question.question, config, deal_id)).items()
                             if a in args.ablations})
            if "production" in args.ablations:
                runs["production"] = await node_run(question.question, QUERY_TYPE["financebench"], deal_id)
            tokens = set().union(*(evidence_tokens(e.text) for e in question.evidence))
            ceiling_text = "\n".join(t for e in question.evidence
                                     for t in page_text.get((filename_of[e.doc_name], e.page), []))
            raw_text = "\n".join(raw_page_text(data.financebench_pdf(e.doc_name), e.page)
                                 for e in question.evidence)
            scored = {}
            for ablation, run in runs.items():
                metrics = score_pages(run, question, filename_of, args.k, pages_of_parent)
                entry = {"metrics": {k: (None if v is None else round(v, 4)) for k, v in metrics.items()},
                         "latency_ms": round(run["latency_ms"], 1)}
                if "gate" in run:
                    entry.update(gate=run["gate"], max_reranker_score=run["max_reranker_score"],
                                 top=[{"file": c.get("source_file"), "pages": sorted(chunk_pages(c)),
                                       "content_type": c.get("content_type"),
                                       "reranker_score": round(float(c.get("reranker_score", 0.0)), 4)}
                                      for c in run["ranking"][:5]])
                scored[ablation] = entry
            report_questions.append({
                "id": question.id, "document": question.doc_name, "question": question.question,
                "question_type": question.question_type, "reasoning": question.question_reasoning,
                "evidence_pages": [[e.doc_name, e.page] for e in question.evidence],
                "evidence_page_indexed": all(page_text.get((filename_of[e.doc_name], e.page))
                                             for e in question.evidence),
                "evidence_token_ceiling": None if not tokens else round(token_recall(tokens, ceiling_text), 4),
                # The same against PyMuPDF's plain text of the page: what any
                # parser could have kept. A gap between the two is lost by ours.
                "evidence_token_raw": None if not tokens else round(token_recall(tokens, raw_text), 4),
                "runs": scored,
            })
            if n % 25 == 0 or n == len(questions):
                print(f"[realdocs] {n}/{len(questions)} questions", file=sys.stderr)

    return {
        "benchmark": "financebench",
        "dataset": {
            "name": "FinanceBench open sample", "repo": data.FINANCEBENCH_REPO,
            "revision": data.FINANCEBENCH_REVISION, "licence": data.FINANCEBENCH_LICENCE,
            "documents": names, "selection": (
                "explicit --documents" if args.documents else
                "all" if args.max_documents is None else
                f"seeded random sample of {args.max_documents} filings (seed {args.seed}); every "
                "question whose evidence is in the sample"),
            "questions": len(questions),
        },
        "config": {"k": args.k, "rank_depth": RANK_DEPTH, "ablations": list(args.ablations),
                   "query_type": QUERY_TYPE["financebench"], "retrieval_config": config},
        "index": {"chunks": len(chunks), "index_seconds": round(index_s, 1),
                  "reachable_chunks": sum(1 for c in chunks if is_reachable(c, deal_id)),
                  "documents": ingestion},
        "queries": report_questions,
    }


# ==============================================================================
# Aggregation and reporting
# ==============================================================================


def metric_names(report: dict, ablation: str) -> list[str]:
    names: list[str] = []
    for q in report["queries"]:
        for name in (q["runs"].get(ablation) or {}).get("metrics", {}):
            if name not in names:
                names.append(name)
    return names


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate(report: dict, group_key: str | None) -> dict:
    """
    {group: {ablation: {metric: {"mean", "low", "high"}}}} with document-clustered
    bootstrap intervals; group "all" pools everything.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in report["queries"]:
        groups["all"].append(q)
        if group_key:
            groups[q[group_key]].append(q)
    out: dict = {}
    for group, rows in groups.items():
        out[group] = {"n": len(rows), "documents": len({q["document"] for q in rows})}
        for ablation in report["config"]["ablations"]:
            have = [q for q in rows if ablation in q["runs"]]
            if not have:
                continue
            cell = {}
            for name in metric_names(report, ablation):
                ci = stats.bootstrap_ci(
                    have, lambda s, n=name, a=ablation: _mean(q["runs"][a]["metrics"].get(n) for q in s),
                    cluster=lambda q: q["document"])
                cell[name] = {"mean": ci["point"], "low": ci["low"], "high": ci["high"]}
            lat = sorted(q["runs"][ablation]["latency_ms"] for q in have)
            cell["latency_ms"] = {"p50": lat[len(lat) // 2], "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))]}
            if ablation == "production":
                cell["gate"] = dict(Counter(q["runs"][ablation]["gate"] for q in have))
            out[group][ablation] = cell
        ceiling_key = "reachable_ceiling" if report["benchmark"] == "legalbench" else "evidence_token_ceiling"
        out[group][ceiling_key] = _mean(q.get(ceiling_key) for q in rows)
    return out


def stage_deltas(report: dict, metrics: list[str]) -> list[dict]:
    """Paired, document-clustered differences between consecutive stages."""
    out = []
    for before, after in STAGE_PAIRS:
        if before not in report["config"]["ablations"] or after not in report["config"]["ablations"]:
            continue
        rows = [q for q in report["queries"] if before in q["runs"] and after in q["runs"]]
        for name in metrics:
            if any(name not in q["runs"][before]["metrics"] or name not in q["runs"][after]["metrics"]
                   for q in rows):
                continue
            d = stats.paired_bootstrap(
                rows,
                lambda s, n=name, b=before: _mean(q["runs"][b]["metrics"][n] for q in s),
                lambda s, n=name, a=after: _mean(q["runs"][a]["metrics"][n] for q in s),
                cluster=lambda q: q["document"])
            out.append({"before": before, "after": after, "metric": name, **d})
    return out


def compare(report: dict, previous: dict, metrics: list[str]) -> list[dict]:
    """Paired before/after of a code change: same queries, same harness."""
    before = {q["id"]: q for q in previous["queries"]}
    out = []
    for ablation in report["config"]["ablations"]:
        rows = [(before[q["id"]], q) for q in report["queries"]
                if q["id"] in before and ablation in q["runs"] and ablation in before[q["id"]]["runs"]]
        for name in metrics:
            pairs = [(b["runs"][ablation]["metrics"].get(name), a["runs"][ablation]["metrics"].get(name), a)
                     for b, a in rows]
            pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
            if len(pairs) < 2:
                continue
            d = stats.paired_bootstrap(
                pairs, lambda s: _mean(p[0] for p in s), lambda s: _mean(p[1] for p in s),
                cluster=lambda p: p[2]["document"])
            d["improved"] = sum(1 for b, a, _ in pairs if a > b + 1e-9)
            d["regressed"] = sum(1 for b, a, _ in pairs if a < b - 1e-9)
            out.append({"ablation": ablation, "metric": name, **d})
    return out


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}"


def _cell(c: dict | None) -> str:
    if not c or c.get("mean") is None:
        return "—"
    if c.get("low") is None:
        return _pct(c["mean"])
    return f"{_pct(c['mean'])} [{_pct(c['low'])}, {_pct(c['high'])}]"


HEADLINE = {
    "legalbench": ["char_recall@1", "char_precision@1", "char_recall@10", "char_precision@10",
                   "snippet_recall@10", "hit@1", "mrr@10", "doc_hit@1", "char_recall@context",
                   "char_precision@context", "snippet_full@context"],
    "financebench": ["page_hit@1", "page_hit@5", "page_hit@10", "mrr@10", "doc_hit@1", "doc_hit@10",
                     "page_hit@context", "evidence_token_recall@5", "evidence_token_recall@context"],
}


# The paired tables (stage effects, --compare) show these, to stay readable.
DELTA_METRICS = {
    "legalbench": ["char_recall@1", "char_recall@10", "hit@1", "mrr@10", "char_recall@context",
                   "snippet_full@context"],
    "financebench": ["page_hit@1", "page_hit@10", "mrr@10", "page_hit@context",
                     "evidence_token_recall@context"],
}


def render_markdown(report: dict) -> str:
    bench = report["benchmark"]
    ds, idx = report["dataset"], report["index"]
    aggs = report["aggregates"]
    headline = [m for m in HEADLINE[bench] if any(
        m in metric_names(report, a) for a in report["config"]["ablations"])]
    lines = [f"# Real-document retrieval — {bench} — {report['generated_at'][:10]}", ""]
    if bench == "legalbench":
        lines += [
            f"{ds['name']}: {ds['queries']} queries over {ds['documents']} real contracts / policies "
            f"({ds['characters']:,} characters), zip SHA-256 {ds['zip_sha256'][:12]}… · index: "
            f"{idx['chunks']:,} chunks ({idx['reachable_chunks']:,} reachable, "
            f"{idx['pii_flagged_chunks']} PII-flagged, {idx['table_chunks']:,} indexed as tables), "
            f"{idx['index_seconds']} s · device {report['environment']['device']} · LLM calls: 0",
            "",
            f"Alignment of chunks back to source offsets: {idx['alignment']['chunks_fully_aligned']:,} "
            f"of {idx['alignment']['chunks']:,} chunks verbatim (whitespace aside); "
            f"{idx['alignment']['unaligned_char_share'] * 100:.2f}% of chunk characters could not be "
            "placed (text the pipeline altered or generated; it can never count as a retrieved "
            "ground-truth character).",
            "",
            "Percentages, mean [95% CI, documents resampled]. char_* = LegalBench-RAG's "
            "character-level precision/recall (retrieved spans merged); snippet_full@context = "
            "ground-truth snippets ≥90% present in the final context.",
            "",
        ]
    else:
        ok = [d for d in idx["documents"] if d.get("ok")]
        lines += [
            f"{ds['name']} ({ds['licence']}), revision {ds['revision'][:12]}: {ds['questions']} "
            f"questions over {len(ds['documents'])} filings ({ds['selection']}) · index: "
            f"{idx['chunks']:,} chunks from {len(ok)} parsed PDFs, {idx['index_seconds']} s · device "
            f"{report['environment']['device']} · LLM calls: 0",
            "",
            "Percentages, mean [95% CI, filings resampled]. page_hit = the evidence page of the right "
            "filing (FinanceBench page + 1) is cited by a retrieved chunk; evidence_token_recall = "
            "share of the annotator's evidence tokens present in the retrieved text.",
            "",
        ]
    groups = [g for g in aggs if g != "all"] + ["all"]
    for group in groups:
        g = aggs[group]
        title = "all" if group == "all" else group
        lines += [f"## {title} — {g['n']} queries, {g['documents']} documents", ""]
        lines += ["| ablation | " + " | ".join(headline) + " | p50 ms |",
                  "|---|" + "---|" * (len(headline) + 1)]
        for ablation in report["config"]["ablations"]:
            if ablation not in g:
                continue
            row = g[ablation]
            lines.append(f"| {ABLATION_LABELS[ablation]} | "
                         + " | ".join(_cell(row.get(m)) for m in headline)
                         + f" | {row['latency_ms']['p50']:.0f} |")
        ceiling = g.get("reachable_ceiling", g.get("evidence_token_ceiling"))
        label = ("ground truth inside a reachable chunk" if bench == "legalbench"
                 else "evidence tokens anywhere in the index on the evidence page")
        lines += ["", f"Ceiling ({label}): {_pct(ceiling)}%."]
        if "production" in g and g["production"].get("gate"):
            gate = g["production"]["gate"]
            lines.append("Refusal gate on these answerable questions (heuristic, no LLM): "
                         + ", ".join(f"{k} {v}" for k, v in sorted(gate.items())) + ".")
        lines.append("")
    deltas = report.get("stage_deltas") or []
    if deltas:
        lines += ["## Stage effects (paired, all queries, documents resampled)", "",
                  "| comparison | metric | Δ points | 95% CI | P(after better) |", "|---|---|---|---|---|"]
        for d in deltas:
            lines.append(f"| {ABLATION_LABELS[d['before']]} → {ABLATION_LABELS[d['after']]} | {d['metric']} | "
                         f"{d['delta'] * 100:+.1f} | [{d['low'] * 100:+.1f}, {d['high'] * 100:+.1f}] | "
                         f"{d['p_better']:.2f} |")
        lines.append("")
    comparison = report.get("comparison")
    if comparison:
        lines += [f"## Against {comparison['previous']}", "",
                  "| ablation | metric | Δ points | 95% CI | improved | regressed |", "|---|---|---|---|---|---|"]
        for d in comparison["deltas"]:
            lines.append(f"| {d['ablation']} | {d['metric']} | {d['delta'] * 100:+.1f} | "
                         f"[{d['low'] * 100:+.1f}, {d['high'] * 100:+.1f}] | {d['improved']} | {d['regressed']} |")
        lines.append("")
    if bench == "financebench":
        lines += render_ingestion(report)
    return "\n".join(lines)


def render_ingestion(report: dict) -> list[str]:
    docs = report["index"]["documents"]
    ok = [d for d in docs if d.get("ok")]
    failed = [d for d in docs if not d.get("ok")]
    lines = ["## Ingestion of the real filings", "",
             f"{len(ok)} of {len(docs)} PDFs indexed; {len(failed)} failed"
             + (": " + "; ".join(f"{d['filename']} ({d['error_type']}: {d['error']})" for d in failed)
                if failed else "") + ".",
             f"Pages: {sum(d.get('pages', 0) for d in docs):,}, of which "
             f"{sum(d.get('pages_without_text', 0) for d in docs)} have no text layer. Parse + index time "
             f"{sum(d['seconds'] for d in docs):.0f} s "
             f"({sum(d['seconds'] for d in docs) / max(1, sum(d.get('pages', 0) for d in docs)):.2f} s/page).",
             f"Tables found: {sum(d.get('table_count', 0) for d in ok):,}; chunks by content type: "
             + ", ".join(f"{k} {v:,}" for k, v in sorted(sum(
                 (Counter(d.get("content_types", {})) for d in ok), Counter()).items())) + ".",
             f"Warnings: {sum(1 for d in ok if d.get('warnings'))} documents "
             + ("(" + "; ".join(f"{d['filename']}: {w}" for d in ok for w in d.get("warnings", []))[:1500] + ")"
                if any(d.get("warnings") for d in ok) else "") + ".",
             "",
             "| filing | pages | no text | s | chunks | tables | category |", "|---|---|---|---|---|---|---|"]
    for d in docs:
        lines.append(f"| {d['filename']} | {d.get('pages', '—')} | {d.get('pages_without_text', '—')} | "
                     f"{d['seconds']:.0f} | {d.get('chunks_created', '—')} | {d.get('table_count', '—')} | "
                     f"{d.get('document_category', d.get('error_type', '—'))} |")
    q = report["queries"]
    lines += ["", f"Evidence page has indexed text for {sum(1 for x in q if x['evidence_page_indexed'])} "
              f"of {len(q)} questions. Evidence tokens kept by the pipeline on the evidence page "
              f"(all chunks of that page): {_pct(_mean(x['evidence_token_ceiling'] for x in q))}%; "
              f"in PyMuPDF's plain page text: {_pct(_mean(x['evidence_token_raw'] for x in q))}%. "
              "The gap is what ingestion loses.", ""]
    return lines


def _device() -> str:
    from eval.run_retrieval_eval import _device as device

    return device()


async def run(args) -> dict:
    report = await (run_legalbench(args) if args.benchmark == "legalbench" else run_financebench(args))
    if report.get("ingest_only"):
        return report
    from src.vector_db.reranker import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME

    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["command"] = "python -m eval.run_realdocs_eval " + " ".join(sys.argv[1:])
    report["environment"] = {"device": _device(), "embedding_model": EMBEDDING_MODEL_NAME,
                             "reranker_model": RERANKER_MODEL_NAME, "python": sys.version.split()[0],
                             "vector_store": "qdrant-client local mode (:memory:), exact search",
                             "laya_at_ingest": False}
    report["aggregates"] = aggregate(report, "subset" if args.benchmark == "legalbench" else "question_type")
    report["stage_deltas"] = stage_deltas(report, DELTA_METRICS[args.benchmark])
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval.run_realdocs_eval",
                                     description="Retrieval on LegalBench-RAG mini and FinanceBench (no LLM calls).")
    parser.add_argument("benchmark", choices=("legalbench", "financebench"))
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=list(ABLATIONS))
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--subsets", nargs="+", choices=data.LEGALBENCH_SUBSETS,
                        default=list(data.LEGALBENCH_SUBSETS), help="LegalBench-RAG sub-corpora")
    parser.add_argument("--max-queries", type=int, help="LegalBench-RAG: first N mini queries per subset (debugging)")
    parser.add_argument("--max-documents", type=int, help="FinanceBench: seeded sample of N filings")
    parser.add_argument("--documents", nargs="+", metavar="DOC_NAME", help="FinanceBench: exactly these filings")
    parser.add_argument("--seed", type=int, default=20261115, help="FinanceBench sampling seed")
    parser.add_argument("--store", metavar="DIR",
                        help="FinanceBench: on-disk index (outside the repo) so ingestion can run in batches")
    parser.add_argument("--max-new-documents", type=int, metavar="N",
                        help="with --store: index at most N filings not yet in the store")
    parser.add_argument("--ingest-only", action="store_true", help="with --store: stop after ingesting")
    parser.add_argument("--compare", metavar="RESULTS_JSON", help="paired before/after against a previous run")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.k = sorted(set(k for k in args.k if 1 <= k <= RANK_DEPTH))
    return args


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    if not args.verbose:
        logging.disable(logging.WARNING)
    try:
        report = asyncio.run(run(args))
    except Exception as e:
        logging.disable(logging.NOTSET)
        logging.getLogger(__name__).exception("Real-document eval failed to run")
        print(f"[realdocs] FAILED to run: {e}", file=sys.stderr)
        return 2
    if report.get("ingest_only"):
        return 0
    if args.compare:
        previous = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        report["comparison"] = {"previous": args.compare,
                                "deltas": compare(report, previous, DELTA_METRICS[args.benchmark])}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"realdocs_{args.benchmark}_{report['generated_at'][:10]}{('_' + args.tag) if args.tag else ''}"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    (out / f"{stem}.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"\n[realdocs] wrote {out / stem}.json/.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
