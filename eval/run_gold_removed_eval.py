"""
Unanswerable questions on real contracts: LegalBench-RAG with the gold document removed.

    python -m eval.run_gold_removed_eval --store <store> --work-copy <dir> --tag on
    python -m eval.run_gold_removed_eval --compare <off.json> <on.json> \\
        --settings-a header,3,0.35 --settings-b header,3,0.35

LegalBench-RAG has no unanswerable queries (eval/README.md), so on real
documents the answerability gate has only ever been tested for false refusals.
This makes the missing half. Every mini query names its contract ("Consider the
Non-Disclosure Agreement between X and Y; ..."), so with that one document
deleted from the index the right answer is "not in the data room" — while the
other contracts of the same kind still offer on-topic clauses that a relevance
score cannot tell apart from the answer. That is the gate's hard case, on text
nobody wrote for it.

For each document of the chosen split (default dev, seed 20261115, exactly as
run_realdocs_eval.py --split), up to --max-per-document of its queries run
through the production retrieval node twice: with the document present
(answerable) and with its child and parent points deleted from a WORKING COPY
of the store (unanswerable). The points are restored, vectors and payloads
unchanged, before the next document. Then Laya scores the top MAX_K passages of
every context in every passage form (eval/answerability_grid.py), so any gate
setting's decisions are recomputed offline.

The store is never written: --work-copy is a fresh copy made at start. Zero
LLM calls. --compare reports refusal rates for two runs (e.g. headers off and
on), each at its own gate setting, with a paired bootstrap that resamples
documents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval import answerability_grid as ag
from eval import realdocs_data as data
from eval.decisions import stats

QUERY_TYPE = "legal"
DEAL_ID = "realdocs_legalbench"


async def _points_of(client, collection: str, filename: str) -> list:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    flt = Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=filename))])
    points, offset = [], None
    while True:
        batch, offset = await client.scroll(collection_name=collection, scroll_filter=flt, limit=1000,
                                            offset=offset, with_payload=True, with_vectors=True)
        points.extend(batch)
        if offset is None:
            return points


async def remove_document(client, filename: str) -> dict:
    """Deletes one document's child and parent points; returns them for restore_document."""
    from qdrant_client.models import PointIdsList

    from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME

    removed = {}
    for collection in (COLLECTION_NAME, PARENT_COLLECTION_NAME):
        points = await _points_of(client, collection, filename)
        if points:
            await client.delete(collection_name=collection,
                                points_selector=PointIdsList(points=[p.id for p in points]))
        removed[collection] = points
    return removed


async def restore_document(client, removed: dict) -> None:
    from qdrant_client.models import PointStruct

    for collection, points in removed.items():
        if points:
            await client.upsert(collection_name=collection, points=[
                PointStruct(id=p.id, vector=p.vector or {}, payload=p.payload) for p in points])


async def _retrieve(query: str) -> tuple[list[dict], dict]:
    """The node's reranked list, the heuristic route, and whether Laya would be consulted."""
    from eval.run_retrieval_eval import gate_decision
    from src.agents.quality_assessor import _heuristic_assessment
    from src.agents.retrieval_executor import retrieval_executor_node

    state = {"original_query": query, "current_query": query, "deal_id": DEAL_ID,
             "query_type": QUERY_TYPE, "parsed_intent": {}, "extracted_filters": {},
             "sub_questions": [], "include_pii": False, "rewrite_iteration": 0}
    reranked = (await retrieval_executor_node(state))["reranked_results"]
    gate = await gate_decision(state, reranked, laya=False)
    verdict = _heuristic_assessment({**state, "reranked_results": reranked})
    return reranked, {"heuristic": gate["decision"], "max_reranker_score": gate["max_reranker_score"],
                      "consulted": not (verdict is not None and verdict.get("force_refusal"))}


async def run(args) -> dict:
    import src.decisions.answerability as ans
    from eval.corpus import use_client
    from eval.run_realdocs_eval import open_store, split_legalbench_documents, write_corpus
    from eval.run_retrieval_eval import release_retrieval_models
    from src.data_processing.chunk_context import context_headers_enabled
    from src.vector_db.reranker import warm_models

    mini = data.load_legalbench_mini()
    documents = {p: mini.documents[p] for p in sorted({s[0] for q in mini.queries for s in q.snippets})}
    split = split_legalbench_documents(mini.queries, args.split_seed)
    by_doc: dict[str, list] = defaultdict(list)
    for q in mini.queries:
        if split.get(q.document) == args.split:
            by_doc[q.document].append(q)
    with tempfile.TemporaryDirectory(prefix="gold_removed_") as tmp:
        _, source_of = write_corpus(documents, Path(tmp))
    filename_of = {path: name for name, path in source_of.items()}

    work = Path(args.work_copy)
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(args.store, work)
    started = time.perf_counter()
    client, _ = await open_store(work)
    rows, contexts = [], []
    with use_client(client):
        await warm_models()
        chosen = sorted(by_doc)[: args.max_documents] if args.max_documents else sorted(by_doc)
        for n, document in enumerate(chosen, start=1):
            queries = by_doc[document][: args.max_per_document]
            name = filename_of[document]
            for q in queries:
                reranked, gate = await _retrieve(q.query)
                rows.append({"id": q.id, "subset": q.subset, "document": document,
                             "condition": "present", "label": "answerable", **gate})
                contexts.append(reranked)
            removed = await remove_document(client, name)
            if not any(removed.values()):
                raise RuntimeError(f"{name}: no points found to remove")
            try:
                for q in queries:
                    reranked, gate = await _retrieve(q.query)
                    if any(c.get("source_file") == name for c in reranked):
                        raise RuntimeError(f"{name} still retrieved after removal")
                    rows.append({"id": q.id, "subset": q.subset, "document": document,
                                 "condition": "removed", "label": "unanswerable", **gate})
                    contexts.append(reranked)
            finally:
                await restore_document(client, removed)
            print(f"[gold-removed] {n}/{len(chosen)} {q.subset} {name}: {len(queries)} queries, "
                  f"removed {sum(len(v) for v in removed.values())} points", file=sys.stderr)
    retrieval_s = time.perf_counter() - started
    await client.close()

    release_retrieval_models()
    for row, reranked in zip(rows, contexts):
        query = next(q.query for q in by_doc[row["document"]] if q.id == row["id"])
        row["matrix"] = await ag.pair_scores([query], reranked, ans.ANSWERABILITY_INSTRUCTION)
        row["top"] = [{"file": c.get("source_file"), "is_table": bool(c.get("is_table")),
                       "reranker_score": round(float(c.get("reranker_score", 0.0)), 4)}
                      for c in ag.ranked_chunks(reranked)]
    shutil.rmtree(work, ignore_errors=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m eval.run_gold_removed_eval " + " ".join(sys.argv[1:]),
        "store": str(args.store),
        "chunk_context_headers_env": context_headers_enabled(),
        "split": args.split, "split_seed": args.split_seed,
        "max_per_document": args.max_per_document,
        "documents": len(chosen),
        "instruction": ans.ANSWERABILITY_INSTRUCTION,
        "retrieval_seconds": round(retrieval_s, 1),
        "total_seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
    }


# ==============================================================================
# Comparison
# ==============================================================================


def parse_setting(text: str) -> tuple[str, int, float]:
    form, k, threshold = text.split(",")
    return form, int(k), float(threshold)


def decisions(rows: list[dict], setting: tuple[str, int, float]) -> dict[tuple[str, str], str]:
    """(query id, condition) -> the gate's decision at `setting`."""
    form, k, threshold = setting
    return {(r["id"], r["condition"]): ag.laya_decision(r["heuristic"], ag.row_score(r, form, k), threshold)
            for r in rows}


def rate(side: str):
    """Share of rows refused on `side` ("a" or "b"), as a bootstrap statistic."""
    return lambda rows: (sum(r[side] for r in rows) / len(rows)) if rows else None


def compare(report_a: dict, report_b: dict, setting_a, setting_b) -> dict:
    """
    Refusal rates of A and B (each at its own setting) on the queries both ran,
    for the removed (should refuse) and present (should not) conditions, with
    document-resampled intervals and a paired B - A difference.
    """
    da, db = decisions(report_a["rows"], setting_a), decisions(report_b["rows"], setting_b)
    heur_a = {(r["id"], r["condition"]): r["heuristic"] for r in report_a["rows"]}
    heur_b = {(r["id"], r["condition"]): r["heuristic"] for r in report_b["rows"]}
    subset = {(r["id"], r["condition"]): (r["document"], r["subset"]) for r in report_a["rows"]}
    out = {}
    for condition in ("removed", "present"):
        keys = sorted(key for key in da if key[1] == condition and key in db)
        rows = [{"doc": subset[key][0], "subset": subset[key][1], "a": da[key] == "refused",
                 "b": db[key] == "refused", "ha": heur_a[key], "hb": heur_b[key],
                 "a_dec": da[key], "b_dec": db[key]} for key in keys]
        entry = {
            "n": len(rows),
            "a": stats.bootstrap_ci(rows, rate("a"), cluster=lambda r: r["doc"]),
            "b": stats.bootstrap_ci(rows, rate("b"), cluster=lambda r: r["doc"]),
            "b_minus_a": stats.paired_bootstrap(rows, rate("a"), rate("b"), cluster=lambda r: r["doc"]),
            "decisions_a": {d: sum(r["a_dec"] == d for r in rows) for d in ("admitted", "llm_fallback", "refused")},
            "decisions_b": {d: sum(r["b_dec"] == d for r in rows) for d in ("admitted", "llm_fallback", "refused")},
            "heuristic_a": {d: sum(r["ha"] == d for r in rows) for d in ("admitted", "llm_fallback", "refused")},
            "heuristic_b": {d: sum(r["hb"] == d for r in rows) for d in ("admitted", "llm_fallback", "refused")},
            "by_subset": {},
        }
        for sub in sorted({r["subset"] for r in rows}):
            rs = [r for r in rows if r["subset"] == sub]
            entry["by_subset"][sub] = {"n": len(rs), "a": sum(r["a"] for r in rs), "b": sum(r["b"] for r in rs)}
        out[condition] = entry
    return {"setting_a": setting_a, "setting_b": setting_b, **out}


def render_compare(result: dict, name_a: str, name_b: str) -> str:
    lines = [f"# Gold document removed — {name_a} {result['setting_a']} vs {name_b} {result['setting_b']}", ""]
    for condition, what in (("removed", "refused (correct: the answer is not in the room)"),
                            ("present", "refused (false refusal: the document is there)")):
        e = result[condition]
        d = e["b_minus_a"]
        lines += [f"## {condition}: share {what}, n={e['n']}", "",
                  f"- {name_a}: {stats.fmt_ci(e['a'])}  decisions {e['decisions_a']}  heuristic {e['heuristic_a']}",
                  f"- {name_b}: {stats.fmt_ci(e['b'])}  decisions {e['decisions_b']}  heuristic {e['heuristic_b']}",
                  f"- {name_b} - {name_a}: {stats.fmt_ci(d)} (p_better {d['p_better']})",
                  "- by subset (refused a / b of n): " + ", ".join(
                      f"{s} {v['a']}/{v['b']} of {v['n']}" for s, v in e["by_subset"].items()), ""]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval.run_gold_removed_eval",
                                     description="LegalBench-RAG with the gold document removed (no LLM calls).")
    parser.add_argument("--store", help="a store built by run_realdocs_eval legalbench --store (read only)")
    parser.add_argument("--work-copy", help="directory for the working copy (deleted and recreated)")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--split-seed", type=int, default=20261115)
    parser.add_argument("--max-per-document", type=int, default=6)
    parser.add_argument("--max-documents", type=int, help="first N documents of the split (smoke runs)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    parser.add_argument("--settings-a", default="header,3,0.35", help="form,k,threshold for A")
    parser.add_argument("--settings-b", default="header,3,0.35", help="form,k,threshold for B")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    if not args.verbose:
        logging.disable(logging.INFO)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.compare:
        a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args.compare)
        result = compare(a, b, parse_setting(args.settings_a), parse_setting(args.settings_b))
        markdown = render_compare(result, Path(args.compare[0]).stem, Path(args.compare[1]).stem)
        (out / f"gold_removed_compare{'_' + args.tag if args.tag else ''}.json").write_text(
            json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(markdown)
        return 0
    if not (args.store and args.work_copy):
        print("--store and --work-copy are required", file=sys.stderr)
        return 2
    if Path(args.work_copy).resolve() == Path(args.store).resolve():
        print("--work-copy must differ from --store", file=sys.stderr)
        return 2
    report = asyncio.run(run(args))
    path = out / f"gold_removed{'_' + args.tag if args.tag else ''}.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(report['rows'])} rows, {report['total_seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
