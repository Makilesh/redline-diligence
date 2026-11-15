"""
Derives a reranker's calibration profile on the dev set — where the score maps
in src/decisions/reranker_calibration.py come from.

    python -m eval.decisions.run_reranker_calibration
    python -m eval.decisions.run_reranker_calibration --target cross-encoder/ms-marco-MiniLM-L-6-v2
    python -m eval.decisions.run_reranker_calibration --from-cache <dir>/reranker_calibration_pairs.json

The context gate's thresholds were measured on BAAI/bge-reranker-v2-m3's score
scale (Decision 17). A different cross-encoder scores the same (query, passage)
pair on its own scale — ms-marco-MiniLM puts noise near 0.008 where bge puts it
near 0.04 — so its scores are mapped onto bge's scale before the gate reads
them. This harness measures that map:

  1. Indexes data/sample_deal and runs every question of
     eval/answerability_dev.json through the production retrieval node once per
     reranker, recording every (query, passage) pair the cross-encoder scored
     and the final reranked context. The candidate pool comes from dense + BM25
     fusion, so both rerankers score exactly the same pairs.
  2. Fits a monotone map from the target's raw scores to the reference's:
     quantile mapping over the pooled pairs (adopted), and, for comparison,
     isotonic calibration of each reranker onto P(chunk states the dev
     evidence), matched through that common probability.
  3. Replays the Quality Assessor's heuristic gate and the router on each dev
     context, per reranker and per map, and reports how often the target agrees
     with the reference's decision.

The golden set is NOT touched: it is the held-out test set, reported by
eval/run_retrieval_eval.py with RERANKER_MODEL set to the target. Makes zero
LLM calls and never loads Laya. Writes reranker_calibration.json (map knots,
gate tables) and reranker_calibration_pairs.json (the scored pairs, reusable
with --from-cache) to --output-dir.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval.corpus import PROJECT_ROOT, build_index, use_client

DEV_SET_PATH = PROJECT_ROOT / "eval" / "answerability_dev.json"
GUARD_SET_PATH = PROJECT_ROOT / "eval" / "query_guard_set.json"
NOISE_CATEGORIES = ("off_topic", "jailbreak", "injection")
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

REFERENCE = {"model": "BAAI/bge-reranker-v2-m3", "max_length": 1024, "fp16": "auto"}
DEFAULT_TARGET = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Quantile levels of the map's knots. Dense in the upper tail, where the gate's
# thresholds sit (bge 0.05-0.30 is roughly the 60th-90th percentile of the
# candidate pool) and where a 2.5% bin holds ~40 pairs on the dev set.
QUANTILE_LEVELS = tuple(round(q, 4) for q in (
    [i / 20 for i in range(0, 12)]            # 0.00 .. 0.55 every 5%
    + [0.55 + i * 0.025 for i in range(1, 15)]  # 0.575 .. 0.90 every 2.5%
    + [0.90 + i * 0.01 for i in range(1, 10)]   # 0.91 .. 0.99 every 1%
    + [0.995, 1.0]
))


def _normalise(text: str) -> str:
    return " ".join((text or "").split())


def load_dev_set(path: Path = DEV_SET_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def load_noise_prompts(path: Path = GUARD_SET_PATH) -> list[dict]:
    """
    Junk prompts of the query-guard set as gate questions whose whole context is noise.

    Only the categories with no genuine question inside (`mixed` appends an
    injection to a real one). They calibrate nothing; they show how much refusal
    power a profile keeps on queries that should never reach synthesis.
    """
    prompts = json.loads(path.read_text(encoding="utf-8"))["positives"]
    return [{"id": f"noise_{i:02d}", "query": p["prompt"], "query_type": "summary", "noise": True,
             "answerable": False}
            for i, p in enumerate(prompts) if p["category"] in NOISE_CATEGORIES]


def _configure_reranker(model: str, max_length: int | None, fp16: str) -> None:
    """Points the reranker module at `model` and drops any loaded instance."""
    from src.vector_db import reranker

    reranker.RERANKER_MODEL_NAME = model
    if max_length is not None:
        reranker.RERANKER_MAX_LENGTH = max_length
    reranker.RERANKER_FP16 = fp16
    reranker._reranker_model = None
    if hasattr(reranker, "_reranker_profile"):
        reranker._reranker_profile = None


async def _collect_one(questions: list[dict]) -> dict:
    """Runs every dev question through retrieval with the configured reranker."""
    import src.agents.retrieval_executor as rx
    from eval.run_retrieval_eval import _node_state

    scored: dict[str, float] = {}
    original = rx.rerank_pairs_async

    async def recording(pairs):
        scores = await original(pairs)
        for (query, passage), score in zip(pairs, scores):
            scored[json.dumps([query, passage], ensure_ascii=False)] = float(score)
        return scores

    contexts = {}
    rx.rerank_pairs_async = recording
    try:
        for q in questions:
            state = _node_state({"query": q["query"], "query_type": q["query_type"]}, [])
            reranked = (await rx.retrieval_executor_node(state))["reranked_results"]
            contexts[q["id"]] = [
                {"chunk_id": c.get("chunk_id"), "score": float(c["reranker_score"]),
                 "text": c.get("text", "")}
                for c in reranked
            ]
    finally:
        rx.rerank_pairs_async = original
    return {"pairs": scored, "contexts": contexts}


async def collect(questions: list[dict], target: str, target_fp16: str) -> dict:
    """Scored pairs and final contexts for the reference and the target reranker."""
    from src.vector_db.reranker import warm_models

    client, _ = await build_index()
    out = {}
    with use_client(client):
        for name, spec in (("reference", REFERENCE),
                           ("target", {"model": target, "max_length": None, "fp16": target_fp16})):
            _configure_reranker(spec["model"], spec["max_length"], spec["fp16"])
            await warm_models()
            started = time.perf_counter()
            out[name] = {"model": spec["model"], **await _collect_one(questions)}
            out[name]["seconds"] = round(time.perf_counter() - started, 1)
            print(f"[calibration] {name} {spec['model']}: {len(out[name]['pairs'])} pairs",
                  file=sys.stderr)
    return out


# ==============================================================================
# Maps
# ==============================================================================


def quantile_map(raw: list[float], reference: list[float],
                 levels: tuple[float, ...] = QUANTILE_LEVELS) -> list[tuple[float, float]]:
    """
    Knots (raw, reference-equivalent) matching the two marginal distributions.

    Monotone by construction and label-free: it uses every scored pair, not just
    the few labelled ones. Knots that do not advance on both axes are dropped so
    the map is strictly increasing between its end points (0, 0) and (1, 1).
    """
    rq = np.quantile(np.asarray(raw), levels)
    bq = np.quantile(np.asarray(reference), levels)
    knots = [(0.0, 0.0)]
    for r, b in zip(rq.tolist(), bq.tolist()):
        if r > knots[-1][0] and b > knots[-1][1] and r < 1.0 and b < 1.0:
            knots.append((round(r, 6), round(b, 6)))
    knots.append((1.0, 1.0))
    return knots


def apply_map(knots: list[tuple[float, float]], score: float) -> float:
    """Piecewise-linear interpolation through the knots (identity when empty)."""
    if not knots:
        return score
    xs = [k[0] for k in knots]
    i = bisect.bisect_right(xs, score)
    if i <= 0:
        return knots[0][1]
    if i >= len(knots):
        return knots[-1][1]
    (x0, y0), (x1, y1) = knots[i - 1], knots[i]
    return y0 + (y1 - y0) * (score - x0) / (x1 - x0) if x1 > x0 else y1


def _isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators fit of y on x; returns (sorted x, fitted y)."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order].astype(float)
    blocks: list[list[float]] = []  # [sum, count]
    for v in ys:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    fitted = np.concatenate([[s / c] * int(c) for s, c in blocks])
    return xs, fitted


def probability_matched_map(target_scores: list[float], reference_scores: list[float],
                            labels: list[int]) -> list[tuple[float, float]]:
    """
    The comparison map: both rerankers calibrated onto P(relevant) by isotonic
    regression on the labelled pairs, then target -> reference through equal
    probability. Uses only the labelled pairs (answerable dev questions), so it
    rests on far fewer positives than the quantile map.
    """
    t = np.asarray(target_scores)
    r = np.asarray(reference_scores)
    y = np.asarray(labels)
    tx, tp = _isotonic(t, y)
    rx_, rp = _isotonic(r, y)
    knots = [(0.0, 0.0)]
    for level in QUANTILE_LEVELS[1:-1]:
        x = float(np.quantile(t, level))
        p = float(np.interp(x, tx, tp))
        # Smallest reference score whose calibrated P reaches p.
        idx = int(np.searchsorted(rp, p, side="left"))
        b = float(rx_[min(idx, len(rx_) - 1)])
        if x > knots[-1][0] and b > knots[-1][1] and x < 1.0 and b < 1.0:
            knots.append((round(x, 6), round(b, 6)))
    knots.append((1.0, 1.0))
    return knots


# ==============================================================================
# Gate replay
# ==============================================================================


def candidate_profile(model: str, knots: list, refusal_ceiling: float | None = None):
    """A profile to replay the gate with (not registered anywhere)."""
    from src.decisions import reranker_calibration as rc

    return rc.RerankerProfile(
        name="candidate", model=model, max_length=0,
        knots=tuple(tuple(k) for k in knots),
        refusal_ceiling=rc.CONFIDENT_REFUSAL_CEILING if refusal_ceiling is None else refusal_ceiling)


def gate(question: dict, context: list[dict], profile) -> dict:
    """The production heuristic gate and router on this context, under `profile`."""
    from eval.run_retrieval_eval import _decision, _node_state
    from src.agents.quality_assessor import _heuristic_assessment
    from src.decisions import reranker_calibration as rc

    state = _node_state({"query": question["query"], "query_type": question["query_type"]}, [])
    chunks = [{"chunk_id": c["chunk_id"], "text": c["text"], "reranker_score": c["score"]}
              for c in context]
    full = {**state, "reranked_results": chunks}
    with rc.use_profile(profile):
        verdict = _heuristic_assessment(full)
        decision = _decision(full, verdict)
    best = max((profile.calibrate(c["score"]) for c in context), default=0.0)
    return {"decision": decision, "max": round(best, 4)}


# Grid and margin for a target's refusal ceiling (common scale). The ceiling is
# the largest grid value at most CEILING_MARGIN x the lowest best score of any
# dev question whose evidence IS in the target's context — the Laya veto's
# rule (threshold below the lowest answerable dev score, not at it) applied to
# the heuristic's outright refusal.
CEILING_GRID = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04, 0.05)
CEILING_MARGIN = 0.8


def choose_refusal_ceiling(questions: list[dict], data: dict, knots: list) -> dict:
    """The target's refusal ceiling, with the sweep behind it."""
    target = data["target"]["model"]
    contexts = data["target"]["contexts"]
    mapped = candidate_profile(target, knots)
    answerable = {q["id"]: max(mapped.calibrate(c["score"]) for c in contexts[q["id"]])
                  for q in questions if label(q, contexts[q["id"]]) == "answerable"}
    lowest = min(answerable.values())
    eligible = [c for c in CEILING_GRID if c <= CEILING_MARGIN * lowest]
    chosen = max(eligible) if eligible else CEILING_GRID[0]
    sweep = []
    for ceiling in CEILING_GRID:
        profile = candidate_profile(target, knots, ceiling)
        decisions = {q["id"]: (label(q, contexts[q["id"]]), gate(q, contexts[q["id"]], profile))
                     for q in questions}
        sweep.append({
            "ceiling": ceiling,
            "answerable_refused": sorted(i for i, (lab, g) in decisions.items()
                                         if lab == "answerable" and g["decision"] == "refused"),
            "unanswerable_refused": sum(lab == "unanswerable" and g["decision"] == "refused"
                                        for lab, g in decisions.values()),
            "noise_refused": sum(lab == "noise" and g["decision"] == "refused"
                                 for lab, g in decisions.values()),
            "noise": sum(lab == "noise" for lab, _ in decisions.values()),
        })
    return {"chosen": chosen, "lowest_answerable_best": round(lowest, 4),
            "lowest_answerable_id": min(answerable, key=answerable.get), "sweep": sweep}


def label(question: dict, context: list[dict]) -> str:
    if question.get("noise"):
        return "noise"
    if question.get("note"):
        return "excluded"
    if not question["answerable"]:
        return "unanswerable"
    evidence = _normalise(question["evidence"])
    return ("answerable" if any(evidence in _normalise(c["text"]) for c in context)
            else "retrieval_miss")


def tables(questions: list[dict], data: dict, profiles: dict) -> dict:
    """Per-profile gate counts and agreement with the reference's decisions."""
    from src.decisions import reranker_calibration as rc

    ref_ctx = data["reference"]["contexts"]
    tgt_ctx = data["target"]["contexts"]
    rows = []
    for q in questions:
        row = {"id": q["id"], "label": label(q, ref_ctx[q["id"]]),
               "reference": gate(q, ref_ctx[q["id"]], rc.REFERENCE_PROFILE)}
        for name, profile in profiles.items():
            row[name] = gate(q, tgt_ctx[q["id"]], profile)
        rows.append(row)
    out = {}
    for name in ["reference", *profiles]:
        counts = {}
        for lab in ("answerable", "unanswerable", "retrieval_miss", "excluded", "noise"):
            counts[lab] = {d: sum(1 for r in rows if r["label"] == lab and r[name]["decision"] == d)
                           for d in ("admitted", "llm_fallback", "refused")}
        agree = sum(r[name]["decision"] == r["reference"]["decision"] for r in rows)
        out[name] = {"counts": counts, "agrees_with_reference": agree, "n": len(rows),
                     "disagreements": [
                         f"{r['id']} {r['reference']['decision']}->{r[name]['decision']} "
                         f"(ref max {r['reference']['max']}, max {r[name]['max']})"
                         for r in rows if r[name]["decision"] != r["reference"]["decision"]]}
    return {"summary": out, "questions": rows}


def pair_labels(questions: list[dict], data: dict) -> tuple[list[str], list[int]]:
    """Pair keys scored by both rerankers, with 1 where the passage states the evidence."""
    evidence = {q["query"]: _normalise(q["evidence"]) for q in questions
                if q.get("answerable") and not q.get("note")}
    keys = sorted(set(data["reference"]["pairs"]) & set(data["target"]["pairs"]))
    labelled_keys, labels = [], []
    for key in keys:
        query, passage = json.loads(key)
        if query in evidence:
            labelled_keys.append(key)
            labels.append(int(evidence[query] in _normalise(passage)))
    return labelled_keys, labels


def distribution(scores: list[float]) -> dict:
    a = np.asarray(scores)
    return {f"p{p}": round(float(np.percentile(a, p)), 4) for p in (10, 25, 50, 75, 90, 95, 99)}


def analyse(questions: list[dict], data: dict) -> dict:
    # The map is fitted on the dev questions' pairs only: noise prompts are a
    # population production rarely scores (the query guard blocks them).
    dev_queries = {q["query"] for q in questions if not q.get("noise")}
    keys = sorted(k for k in set(data["reference"]["pairs"]) & set(data["target"]["pairs"])
                  if json.loads(k)[0] in dev_queries)
    ref = [data["reference"]["pairs"][k] for k in keys]
    tgt = [data["target"]["pairs"][k] for k in keys]
    qmap = quantile_map(tgt, ref)
    lkeys, labels = pair_labels(questions, data)
    pmap = probability_matched_map([data["target"]["pairs"][k] for k in lkeys],
                                   [data["reference"]["pairs"][k] for k in lkeys], labels)
    mapped = [apply_map(qmap, s) for s in tgt]
    rank_corr = float(np.corrcoef(np.argsort(np.argsort(ref)), np.argsort(np.argsort(tgt)))[0, 1])
    pos = [i for i, k in enumerate(keys) if k in set(np.asarray(lkeys)[np.asarray(labels) == 1])]
    neg = [i for i in range(len(keys)) if i not in set(pos)]
    ceiling = choose_refusal_ceiling(questions, data, qmap)
    target = data["target"]["model"]
    return {
        "pairs": len(keys),
        "labelled_pairs": len(lkeys),
        "positive_pairs": int(sum(labels)),
        "spearman": round(rank_corr, 4),
        "distribution": {"reference": distribution(ref), "target_raw": distribution(tgt),
                         "target_quantile_mapped": distribution(mapped)},
        "positives_median": {"reference": round(float(np.median([ref[i] for i in pos])), 4),
                             "target_raw": round(float(np.median([tgt[i] for i in pos])), 4)}
        if pos else None,
        "others_median": {"reference": round(float(np.median([ref[i] for i in neg])), 4),
                          "target_raw": round(float(np.median([tgt[i] for i in neg])), 4)},
        "maps": {"quantile": qmap, "probability_matched": pmap},
        "refusal_ceiling": ceiling,
        "gate": tables(questions, data, {
            "target_raw": candidate_profile(target, []),
            "target_quantile": candidate_profile(target, qmap),
            "target_quantile_ceiling": candidate_profile(target, qmap, ceiling["chosen"]),
            "target_probability_matched": candidate_profile(target, pmap),
        }),
    }


def render(report: dict) -> str:
    a = report["analysis"]
    lines = [f"# Reranker calibration — dev set, {report['generated_at'][:10]}", "",
             f"Reference {report['reference']}, target {report['target']}. "
             f"{a['pairs']} (query, passage) pairs from {report['questions']} dev questions "
             f"(+ {report['noise_prompts']} noise prompts, gate replay only); "
             f"Spearman {a['spearman']}. LLM calls: 0.", "",
             "| scores | p10 | p25 | p50 | p75 | p90 | p95 | p99 |", "|---|---|---|---|---|---|---|---|"]
    for name, d in a["distribution"].items():
        lines.append(f"| {name} | " + " | ".join(str(d[f"p{p}"]) for p in (10, 25, 50, 75, 90, 95, 99))
                     + " |")
    lines += ["", "| gate | answerable adm/llm/ref | unanswerable adm/llm/ref | "
              "retrieval miss adm/llm/ref | noise prompts adm/llm/ref | agrees with reference |",
              "|---|---|---|---|---|---|"]
    def f(d: dict) -> str:
        return f"{d['admitted']}/{d['llm_fallback']}/{d['refused']}"

    for name, s in a["gate"]["summary"].items():
        c = s["counts"]
        lines.append(f"| {name} | {f(c['answerable'])} | {f(c['unanswerable'])} | "
                     f"{f(c['retrieval_miss'])} | {f(c['noise'])} | "
                     f"{s['agrees_with_reference']}/{s['n']} |")
    lines.append("")
    for name, s in a["gate"]["summary"].items():
        if s["disagreements"]:
            lines.append(f"- {name}: " + "; ".join(s["disagreements"]))
    c = a["refusal_ceiling"]
    lines += ["", f"Refusal ceiling: {c['chosen']} (lowest best score of an answerable dev "
              f"question with its evidence in the target's context: {c['lowest_answerable_best']}, "
              f"{c['lowest_answerable_id']}; margin {CEILING_MARGIN}).", "",
              "| ceiling | answerable refused | unanswerable refused | noise refused |",
              "|---|---|---|---|"]
    for s in c["sweep"]:
        lines.append(f"| {s['ceiling']} | {', '.join(s['answerable_refused']) or '—'} | "
                     f"{s['unanswerable_refused']} | {s['noise_refused']}/{s['noise']} |")
    lines += ["", "Quantile map knots (raw -> reference scale):", "",
              "    " + repr([tuple(k) for k in a["maps"]["quantile"]])]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.decisions.run_reranker_calibration",
        description="Map a reranker's scores onto bge-reranker-v2-m3's scale (dev set, no LLM).")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--target-fp16", default="0",
                        help="RERANKER_FP16 for the target (default 0: fp32, as on the CPU host)")
    parser.add_argument("--from-cache", help="reuse a reranker_calibration_pairs.json")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    logging.disable(logging.INFO)
    questions = load_dev_set() + load_noise_prompts()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.from_cache:
        data = json.loads(Path(args.from_cache).read_text(encoding="utf-8"))
    else:
        data = asyncio.run(collect(questions, args.target, args.target_fp16))
        (out / "reranker_calibration_pairs.json").write_text(
            json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m eval.decisions.run_reranker_calibration " + " ".join(sys.argv[1:]),
        "dev_set": str(DEV_SET_PATH.relative_to(PROJECT_ROOT)),
        "reference": data["reference"]["model"],
        "target": data["target"]["model"],
        "questions": sum(not q.get("noise") for q in questions),
        "noise_prompts": sum(bool(q.get("noise")) for q in questions),
        "analysis": analyse(questions, data),
    }
    (out / "reranker_calibration.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
