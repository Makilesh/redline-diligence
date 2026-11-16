"""
Retrieval latency on a Qdrant server vs local mode, for the round-4/5 query-side features.

Decisions 67, 68, 70 and 71 timed their extra searches on local-mode Qdrant only, where a
filtered search pair costs 1.5-2 s at 45K points. This copies an existing on-disk eval store
into a throwaway Qdrant server (same image as docker-compose, gRPC like production), then runs
every question through the real retrieval_executor_node in four combinations

    {local, server} x {feature off, feature on}

in an order rotated per question (paired, interleaved). Feature:
    legal    document focus: document_focus_k 0 vs the configured depth (30 on GPU)
    finance  filing-period preference: FILING_PERIOD_PREFERENCE=0 vs default (year + quarter)

Recorded per run: node latency, time inside each Qdrant-touching call, and the top-10 chunk ids,
so the server's HNSW results can be compared with local mode's exact search.

Start a throwaway server first (no volume, ports clear of docker-compose's):

    docker run -d --rm --name redline-bench-qdrant -p 6339:6333 -p 6340:6334 qdrant/qdrant:v1.18.0
    python -m eval.run_qdrant_server_latency legal   --store <LegalBench-RAG --store dir> --out legal.json
    python -m eval.run_qdrant_server_latency finance --store <FinanceBench --store dir> --out fin.json

The stores are run_realdocs_eval.py --store directories. Decision 79.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

TIMES: dict = defaultdict(float)


def _timed(name, fn):
    async def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return await fn(*a, **k)
        finally:
            TIMES[name] += (time.perf_counter() - t0) * 1000
    return wrapper


def instrument():
    import src.agents.retrieval_executor as rx

    rx.hybrid_search = _timed("hybrid_search", rx.hybrid_search)
    rx.prefer_named_period = _timed("period_pref", rx.prefer_named_period)
    rx.dense_search = _timed("focus_search", rx.dense_search)
    rx.rerank_pairs_async = _timed("rerank", rx.rerank_pairs_async)
    rx.expand_context = _timed("expand", rx.expand_context)


async def copy_store(src, dst) -> dict:
    """Every point of both collections, vectors and payloads, into the server."""
    from qdrant_client import models

    from src.vector_db.collection_manager import setup_collections
    from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME

    await setup_collections(dst, COLLECTION_NAME, PARENT_COLLECTION_NAME)
    counts = {}
    for name in (COLLECTION_NAME, PARENT_COLLECTION_NAME):
        if not await src.collection_exists(name):
            continue
        if not await dst.collection_exists(name):
            params = (await src.get_collection(name)).config.params
            await dst.create_collection(name, vectors_config=params.vectors,
                                        sparse_vectors_config=params.sparse_vectors)
        offset, n = None, 0
        while True:
            points, offset = await src.scroll(collection_name=name, limit=256, offset=offset,
                                              with_payload=True, with_vectors=True)
            if points:
                await dst.upsert(collection_name=name, wait=True, points=[
                    models.PointStruct(id=p.id, vector=p.vector or {}, payload=p.payload)
                    for p in points])
                n += len(points)
            if offset is None:
                break
        counts[name] = n
        print(f"[server-latency] copied {n} points into {name}", file=sys.stderr, flush=True)
    # Let the optimizer finish building HNSW before anything is timed.
    for name in list(counts):
        for _ in range(600):
            info = await dst.get_collection(name)
            if str(info.status).lower().endswith("green"):
                break
            await asyncio.sleep(1)
        counts[name + ":indexed_vectors"] = (await dst.get_collection(name)).indexed_vectors_count
    return counts


def legal_questions(stride: int):
    from eval import realdocs_data as data
    from eval import run_realdocs_eval as R

    mini = data.load_legalbench_mini()
    split = R.split_legalbench_documents(mini.queries, 20261115)
    dev = [q for q in mini.queries if split.get(q.document) == "dev"]
    return [(q.id, q.query) for q in dev[::stride]]


def finance_questions(store_names: set[str]):
    from eval import realdocs_data as data

    qs = data.load_financebench()
    dev = set(data.financebench_company_split(qs)["dev"])
    return [(q.id, q.question)
            for q in qs
            if q.company in dev and f"{q.doc_name}.pdf" in store_names]


async def run(args) -> dict:
    from qdrant_client import AsyncQdrantClient

    from eval import run_realdocs_eval as R
    from eval.corpus import use_client
    from src.agents.retrieval_strategy import get_retrieval_config
    from src.vector_db import reranker as rr

    local, ingestion = await R.open_store(Path(args.store))
    server = AsyncQdrantClient(url=args.url, grpc_port=args.grpc_port, prefer_grpc=True, timeout=30)
    copied = await copy_store(local, server)

    if args.benchmark == "legal":
        qtype, deal_id = "legal", R.DEAL_ID["legalbench"]
        questions = legal_questions(args.stride)
        base = get_retrieval_config("legal", {})
        on_depth = base.get("document_focus_k") or 30
        settings = {"off": ({**base, "document_focus_k": 0}, None),
                    "on": ({**base, "document_focus_k": on_depth}, None)}
    else:
        qtype, deal_id = "financial", R.DEAL_ID["financebench"]
        questions = finance_questions({d["filename"] for d in ingestion})
        settings = {"off": (None, "0"), "on": (None, "1")}
    if args.limit:
        questions = questions[: args.limit]

    combos = [(b, s) for b in ("local", "server") for s in ("off", "on")]
    clients = {"local": local, "server": server}
    instrument()
    await rr.warm_models()

    async def one(backend, setting, query):
        config, pref = settings[setting]
        if pref is not None:
            os.environ["FILING_PERIOD_PREFERENCE"] = pref
        TIMES.clear()
        with use_client(clients[backend]):
            run = await R.node_run(query, qtype, deal_id, config)
        return {"latency_ms": round(run["latency_ms"], 1),
                "components_ms": {k: round(v, 1) for k, v in TIMES.items()},
                "top10": [c.get("chunk_id") for c in run["ranking"][:10]]}

    for b, s in combos:  # untimed warm-up per combination
        await one(b, s, questions[0][1])
    rows, started = [], time.perf_counter()
    for n, (qid, query) in enumerate(questions):
        order = combos[n % 4:] + combos[: n % 4]
        runs = {f"{b}:{s}": await one(b, s, query) for b, s in order}
        rows.append({"id": qid, "runs": runs})
        if (n + 1) % 10 == 0 or n + 1 == len(questions):
            print(f"[server-latency] {args.benchmark} {n + 1}/{len(questions)} "
                  f"{time.perf_counter() - started:.0f}s", file=sys.stderr, flush=True)
    os.environ.pop("FILING_PERIOD_PREFERENCE", None)
    await local.close()
    await server.close()
    return {"benchmark": args.benchmark, "store": args.store, "copied": copied,
            "settings": {k: {"config": v[0] and {"document_focus_k": v[0].get("document_focus_k")},
                             "FILING_PERIOD_PREFERENCE": v[1]} for k, v in settings.items()},
            "reranker": rr.RERANKER_MODEL_NAME, "rows": rows}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", choices=("legal", "finance"))
    p.add_argument("--store", required=True)
    p.add_argument("--url", default="http://localhost:6339")
    p.add_argument("--grpc-port", type=int, default=6340)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    logging.disable(logging.WARNING)
    report = asyncio.run(run(args))
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[server-latency] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
