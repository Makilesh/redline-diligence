"""
The GPU figure veto, measured: which evidence, which claims, which line (Decision 90).

    python -m eval.run_figure_veto_eval score --set benchmark --work <dir>        # GPU (FactCG)
    python -m eval.run_figure_veto_eval select --work <dir>                       # dev half only
    python -m eval.run_figure_veto_eval score --set realdocs --data D:/redline_data/veto10 --work <dir>
    python -m eval.run_figure_veto_eval confirm --work <dir> --chosen <bench>/chosen.json
    python -m eval.run_figure_veto_eval confirm --work <bench> --split test --chosen <bench>/chosen.json

A claim whose every figure numeric grounding confirms is passed on its figures
alone, so the grounding checker never reads its wording: on GPU every wrong
benchmark claim that still passes is such a claim (Decision 63). A figure veto
scores those claims too, against the passages their figures came from, and a
score below its line sends the claim to the LLM judge — never to a failure
(claim_checker.FigureVeto). Decision 64 found FactCG's veto too costly on GPU:
the correct claims it vetoed were mostly table rows scored without their
column headers. This harness measures the remedy, the "headed" evidence
(table_context.headed_window), against the plain figure windows.

score    verifies every case under five configurations — production FactCG
         with no veto, and evidence {figure, headed} x scope {all
         figure-confirmed claims, cue claims only} at a line no score reaches
         — first with a stub to collect every (premise, claim) pair, then with
         FactCG's real scores (each unique pair scored once, fp16, GPU). Each
         case keeps its production outcome and, per configuration, the exact
         best veto score of each of its claims, so any line is evaluated
         without scoring again: a claim below the line moves from pass to the
         judge, and nothing else changes. Also times, per case, the veto pairs
         alone on the GPU — the cost a veto adds to an answer.
select   the pre-registered dev rule on the benchmark's dev half: fewest wrong
         claims passed, with at most 2 of 109 fewer correct claims passed
         locally and no more false alarms; ties to more correct claims passed,
         cue claims before all, figure windows before headed, the higher line.
         The pick is replayed in full and must reproduce the fast count.
confirm  the chosen configuration, unchanged, against production on a set (or
         split): paired outcomes, intervals that resample clusters (filings
         for the FinanceBench claims, fact families for the benchmark), per
         kind, and Decision 90's adoption rule (--rule single), or the rule
         for claims joining two rows (--rule two-row).

Evidence forms: "figure" (sentence windows), "headed" (one window per table
row, Decision 90 as first measured), "joined" (headed, with one window holding
all the rows of a claim that joins several), "single" (headed, with no veto for
such a claim). The last three give a claim within one row the same window.

Cases: the benchmark (eval/verification_set.json, contexts from
eval/verification_corpus.py, clustered by family) or the FinanceBench claims
eval/figure_veto_set.py builds (clustered by filing). Outcomes are
run_verification_eval's: pass / judge / warn / fail / skipped, no judge
attached. Makes zero LLM calls.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261119
BOOTSTRAP = 10_000
NEGATIVE = ("unsupported", "contradicted")
EVIDENCE = ("figure", "headed", "joined", "single")
# Evidence name -> (FigureVeto.evidence, FigureVeto.rows).
_EVIDENCE = {"figure": ("figure", "each"), "headed": ("headed", "each"),
             "joined": ("headed", "joined"), "single": ("headed", "single")}
SCOPES = ("all", "cue")
CONFIGS = ("none",) + tuple(f"{e}/{s}" for e in EVIDENCE for s in SCOPES)
# Every claim in scope is vetoed at this line: support scores are probabilities.
ALL_LINE = 2.0
VETO_REASON = "every figure is in the documents, but the passages holding them"
SUPPORTED_SLACK = 2  # dev: at most 2 of 109 fewer correct claims passed locally
ADOPT_PP = 0.02


# ==============================================================================
# Cases
# ==============================================================================


def benchmark_cases() -> list[dict]:
    """Every benchmark item, as run_verification_eval verifies it."""
    from eval import run_verification_eval as rve

    return [
        {"id": item["id"], "split": item["split"], "cluster": item["family"],
         "label": item["label"], "kind": item["kind"], "claim": item["claim"],
         "answer": rve.item_answer(item), "chunks": rve.item_context(item)}
        for item in rve.load_items()
    ]


def realdocs_cases(data_dir: Path) -> list[dict]:
    """The FinanceBench claims of eval/figure_veto_set.py, clustered by filing."""
    from eval.figure_veto_set import load_cases

    return [{**case, "split": "realdocs", "cluster": case["doc"]} for case in load_cases(data_dir)]


def load_set(args) -> list[dict]:
    if args.set == "benchmark":
        return benchmark_cases()
    return realdocs_cases(Path(args.data))


# ==============================================================================
# Verification under one configuration
# ==============================================================================


def thresholds(config: str, line: float = ALL_LINE):
    """Production FactCG lines, with the configuration's veto (or none)."""
    from src.verification import claim_checker as cc
    from src.verification import nli

    base = dataclasses.replace(cc.MODEL_THRESHOLDS[nli.FACTCG], figure_veto=None)
    if config == "none":
        return base
    name, scope = config.split("/")
    evidence, rows = _EVIDENCE[name]
    return dataclasses.replace(base, figure_veto=cc.FigureVeto(
        line=line, evidence=evidence, cues_only=scope == "cue", rows=rows))


async def verify(case: dict, th, scorer, exact: bool = False):
    """verify_answer as production runs it, under thresholds `th`."""
    from src.verification import claim_checker as cc

    saved = cc.thresholds_for
    cc.thresholds_for = lambda _model: th
    if exact:
        # The check's score is rounded for display; the line needs it exact.
        cc.round = lambda value, digits=None: value
    try:
        return await cc.verify_answer(case["answer"], case["chunks"], case["claim"],
                                      scorer=scorer, judge=None, use_nli=True)
    finally:
        cc.thresholds_for = saved
        if exact:
            del cc.round


def pair_key(premise: str, hypothesis: str) -> str:
    return hashlib.sha1(f"{premise}\x00{hypothesis}".encode("utf-8")).hexdigest()


class ScoreCache:
    """
    FactCG's score for every (premise, claim) pair, each scored once.

    A miss is scored by the production model (nli.score_pairs, fp16 on the
    GPU) in the call that needs it, so every configuration and the later full
    replays read the same number for the same pair.
    """

    def __init__(self, scores: dict[str, list[float]] | None = None, offline: bool = False):
        self.scores = dict(scores or {})
        self.pairs: dict[str, list[str]] = {}
        self.offline = offline
        self.seen: list[str] = []

    async def __call__(self, ps):
        from src.verification import nli

        keys = [pair_key(p, h) for p, h in ps]
        for k, (p, h) in zip(keys, ps):
            self.pairs.setdefault(k, [p, h])
        todo = [(k, pair) for k, pair in zip(keys, ps) if k not in self.scores]
        if todo:
            if self.offline:
                raise KeyError(f"{len(todo)} pairs were never scored")
            fresh = nli.score_pairs([pair for _, pair in dict(todo).items()])
            if nli.loaded_model_name() != nli.FACTCG:
                raise RuntimeError(f"scored with {nli.loaded_model_name()}, not {nli.FACTCG}")
            for (k, _), s in zip(dict(todo).items(), fresh):
                self.scores[k] = [s.entailment, s.contradiction, s.neutral]
        self.seen.extend(keys)
        return [nli.NLIScore(*self.scores[k]) for k in keys]


def outcome(report) -> str:
    from eval.run_verification_eval import outcome_of

    return outcome_of(report)


def _vetoed(report, cue_only: bool) -> list[float]:
    """Exact best veto score of each vetoed claim (a line no score reaches vetoes all)."""
    from src.verification.claim_checker import _VETO_CUE

    return [
        c["score"] for c in report.claim_checks
        if (c.get("reason") or "").startswith(VETO_REASON)
        and (not cue_only or _VETO_CUE.search(c["claim"]))
    ]


async def run_cases(cases: list[dict], cache: ScoreCache,
                    evidences: tuple[str, ...] = EVIDENCE) -> tuple[list[dict], dict]:
    """
    Per case: the production outcome and, per configuration, every vetoed
    claim's exact best score — and the pairs each configuration scored.

    Cue claims are a subset of all figure-confirmed claims, and a claim's
    windows do not depend on which others are scored, so a cue configuration's
    scores are the "all" configuration's, kept for the claims with a cue.
    """
    rows, per_case = [], {c: {} for c in ("none", *(f"{e}/all" for e in evidences))}
    for case in cases:
        row = {k: case[k] for k in ("id", "split", "cluster", "label", "kind", "claim")}
        cache.seen = []
        report = await verify(case, thresholds("none"), cache)
        per_case["none"][case["id"]] = cache.seen
        row["base"] = outcome(report)
        row["claims"] = len(report.claim_checks)
        row["veto"] = {}
        for evidence in evidences:
            cache.seen = []
            report = await verify(case, thresholds(f"{evidence}/all"), cache, exact=True)
            per_case[f"{evidence}/all"][case["id"]] = cache.seen
            row["veto"][f"{evidence}/all"] = _vetoed(report, cue_only=False)
            row["veto"][f"{evidence}/cue"] = _vetoed(report, cue_only=True)
        rows.append(row)
    return rows, per_case


def time_veto_pairs(cases: list[dict], pairs: dict, per_case: dict, sample: int) -> dict:
    """GPU time of each case's veto pairs scored alone — the cost a veto adds to an answer."""
    import torch

    from src.verification import nli

    out = {}
    nli.score_pairs([("warm up premise.", "warm up.")] * 4)
    for config in (c for c in per_case if c != "none"):
        per = []
        ids = [c["id"] for c in cases
               if [k for k in per_case[config][c["id"]] if k not in set(per_case["none"][c["id"]])]]
        ids = set(random.Random(SEED).sample(ids, min(sample, len(ids))))
        for case in cases:
            if case["id"] not in ids:
                continue
            base = set(per_case["none"][case["id"]])
            extra = [k for k in per_case[config][case["id"]] if k not in base]
            batch = [tuple(pairs[k]) for k in extra]
            torch.cuda.synchronize()
            started = time.perf_counter()
            nli.score_pairs(batch)
            torch.cuda.synchronize()
            per.append((len(batch), (time.perf_counter() - started) * 1000))
        per.sort(key=lambda t: t[1])
        n = len(per)
        out[config] = {
            "cases_with_veto_pairs": n,
            "pairs_mean": round(sum(p for p, _ in per) / n, 2) if n else 0,
            "ms_mean": round(sum(m for _, m in per) / n, 1) if n else 0,
            "ms_p50": round(per[n // 2][1], 1) if n else 0,
            "ms_p95": round(per[min(n - 1, int(n * 0.95))][1], 1) if n else 0,
        }
    return out


# ==============================================================================
# Outcomes at a line, counts, intervals
# ==============================================================================


def outcome_at(row: dict, config: str, line: float) -> str:
    """A claim below the line goes to the judge; nothing else moves."""
    if config == "none" or row["base"] not in ("pass", "judge"):
        return row["base"]
    return "judge" if any(s < line for s in row["veto"][config]) else row["base"]


def counts(rows: list[dict], outcomes: list[str]) -> dict:
    neg = [o for r, o in zip(rows, outcomes) if r["label"] in NEGATIVE]
    sup = [o for r, o in zip(rows, outcomes) if r["label"] == "supported"]
    return {
        "n": len(rows), "negative": len(neg), "supported": len(sup),
        "false_pass": sum(o in ("pass", "skipped") for o in neg),
        "false_alarm": sum(o in ("warn", "fail") for o in sup),
        "supported_pass": sum(o == "pass" for o in sup),
        "judge": sum(o == "judge" for o in outcomes),
    }


def rates(c: dict) -> dict:
    def share(a, b):
        return round(a / b, 4) if b else None
    return {
        "false_pass_rate": share(c["false_pass"], c["negative"]),
        "false_alarm_rate": share(c["false_alarm"], c["supported"]),
        "supported_pass_rate": share(c["supported_pass"], c["supported"]),
        "judge_load": share(c["judge"], c["n"]),
    }


def paired_intervals(rows: list[dict], before: list[str], after: list[str],
                     samples: int = BOOTSTRAP, seed: int = SEED) -> dict:
    """95% intervals of the change in each rate, resampling clusters."""
    clusters: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        clusters[r["cluster"]].append(i)
    names = sorted(clusters)
    fields = ("negative", "supported", "n", "false_pass", "false_alarm", "supported_pass", "judge")
    per = []
    for name in names:
        idx = clusters[name]
        sub = [rows[i] for i in idx]
        b = counts(sub, [before[i] for i in idx])
        a = counts(sub, [after[i] for i in idx])
        per.append((b, a))
    rng = random.Random(seed)
    deltas: dict[str, list[float]] = defaultdict(list)
    pairs = (("false_pass", "negative"), ("false_alarm", "supported"),
             ("supported_pass", "supported"), ("judge", "n"))
    for _ in range(samples):
        draw = [per[rng.randrange(len(per))] for _ in per]
        tot_b = {f: sum(d[0][f] for d in draw) for f in fields}
        tot_a = {f: sum(d[1][f] for d in draw) for f in fields}
        for num, den in pairs:
            if tot_b[den]:
                deltas[num].append(tot_a[num] / tot_a[den] - tot_b[num] / tot_b[den])
    out = {}
    for num, den in pairs:
        values = sorted(deltas[num])
        if not values:
            continue
        b, a = counts(rows, before), counts(rows, after)
        point = (a[num] / a[den] - b[num] / b[den]) if b[den] else None
        out[num] = {
            "change_pp": round(100 * point, 2) if point is not None else None,
            "ci95_pp": [round(100 * values[int(0.025 * len(values))], 2),
                        round(100 * values[min(len(values) - 1, int(0.975 * len(values)))], 2)],
        }
    return out


def by_kind(rows: list[dict], before: list[str], after: list[str]) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[f"{r['label']}/{r['kind']}"].append(i)
    out = {}
    for name, idx in sorted(groups.items()):
        out[name] = {"n": len(idx),
                     "before": dict(Counter(before[i] for i in idx)),
                     "after": dict(Counter(after[i] for i in idx))}
    return out


# ==============================================================================
# Commands
# ==============================================================================


def cmd_score(args) -> int:
    from src.verification import nli

    os.environ["NLI_MODEL"] = nli.FACTCG
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cases = load_set(args)
    evidences = tuple(args.evidence.split(","))
    started = time.perf_counter()
    # Seeded with an earlier run's scores, a pair it already scored is read,
    # not scored again: windows that did not change cost nothing and score
    # exactly the same.
    seed = json.loads(Path(args.seed_scores).read_text(encoding="utf-8")) if args.seed_scores else {}
    cache = ScoreCache(seed)
    rows, per_case = asyncio.run(run_cases(cases, cache, evidences))
    fresh = len(cache.scores) - len(seed)
    print(f"[veto] {len(cases)} cases, {len(cache.pairs)} pairs seen, {fresh} newly scored "
          f"({time.perf_counter() - started:.0f}s)", file=sys.stderr)
    timing = {} if args.no_timing else time_veto_pairs(cases, cache.pairs, per_case, args.timing_sample)
    (work / "pairs.json").write_text(json.dumps({"pairs": cache.pairs, "per_case": per_case}), encoding="utf-8")
    (work / "scores.json").write_text(json.dumps(cache.scores), encoding="utf-8")
    (work / "cases.json").write_text(json.dumps({"set": args.set, "evidences": evidences, "rows": rows,
                                                 "timing": timing, "newly_scored": fresh}, indent=0),
                                     encoding="utf-8")
    base = counts(rows, [r["base"] for r in rows])
    print(json.dumps({"production": base, "newly_scored": fresh, "timing": timing}, indent=1))
    return 0


def _candidates(rows: list[dict], evidences: tuple[str, ...]) -> list[dict]:
    """Every (evidence, scope, line) on these rows, with its counts."""
    base = counts(rows, [r["base"] for r in rows])
    out = [{"config": "none", "line": 0.0, "counts": base}]
    for config in (f"{e}/{s}" for e in evidences for s in SCOPES):
        scores = sorted({s for r in rows for s in r["veto"][config]})
        for s, above in zip(scores, scores[1:] + [1.0]):
            line = (s + above) / 2
            out.append({"config": config, "line": line,
                        "counts": counts(rows, [outcome_at(r, config, line) for r in rows])})
    return out


def choose(rows: list[dict], evidences: tuple[str, ...] = ("figure", "headed")
           ) -> tuple[dict | None, dict, list[dict]]:
    """The pre-registered dev rule (Decision 90); ties prefer earlier evidences."""
    base = counts(rows, [r["base"] for r in rows])
    cands = _candidates(rows, evidences)
    ok = [c for c in cands
          if base["supported_pass"] - c["counts"]["supported_pass"] <= SUPPORTED_SLACK
          and c["counts"]["false_alarm"] <= base["false_alarm"]]

    def rank(c):
        evidence, scope = c["config"].split("/") if c["config"] != "none" else (evidences[0], "cue")
        return (c["counts"]["false_pass"], -c["counts"]["supported_pass"],
                0 if scope == "cue" else 1, evidences.index(evidence), -c["line"])

    ok.sort(key=rank)
    best = ok[0] if ok and ok[0]["counts"]["false_pass"] < base["false_pass"] else None
    return best, base, cands


async def _full_replay(cases: list[dict], scores: dict, config: str, line: float) -> list[str]:
    """Every case verified under the chosen configuration, from the cached scores."""
    scorer = ScoreCache(scores, offline=True)
    return [outcome(await verify(case, thresholds(config, line), scorer)) for case in cases]


def _frontier(cands: list[dict]) -> list[str]:
    """Per configuration, the fewest false passes reachable at each supported-pass level."""
    lines = []
    for config in dict.fromkeys(c["config"] for c in cands if c["config"] != "none"):
        best: dict[int, dict] = {}
        for c in cands:
            if c["config"] != config:
                continue
            k = c["counts"]["supported_pass"]
            if k not in best or c["counts"]["false_pass"] < best[k]["counts"]["false_pass"]:
                best[k] = c
        pts = sorted(best.values(), key=lambda c: -c["counts"]["supported_pass"])
        lines.append(f"  {config:<12} " + "  ".join(
            f"supP {c['counts']['supported_pass']}/FP {c['counts']['false_pass']}@{c['line']:.4f}"
            for c in pts[:8]))
    return lines


def cmd_select(args) -> int:
    work = Path(args.work)
    data = json.loads((work / "cases.json").read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r["split"] == "dev"]
    evidences = tuple(args.evidence.split(","))
    best, base, cands = choose(rows, evidences)
    log = [f"dev production: {base}  (candidates: {', '.join(evidences)})"]
    log += _frontier(cands)
    chosen: dict = {"dev_production": base}
    if best is None:
        log.append("no candidate lowers dev false passes within the constraint -> negative")
        chosen["config"] = None
    else:
        evidence, scope = best["config"].split("/")
        chosen.update(config=best["config"], evidence=evidence, scope=scope,
                      line=round(best["line"], 6), dev_fast=best["counts"])
        cases = [c for c in benchmark_cases() if c["split"] == "dev"]
        scores = json.loads((work / "scores.json").read_text(encoding="utf-8"))
        full = asyncio.run(_full_replay(cases, scores, best["config"], chosen["line"]))
        by_id = {r["id"]: r for r in rows}
        full_counts = counts([by_id[c["id"]] for c in cases], full)
        chosen["dev_full_replay"] = full_counts
        chosen["replay_matches"] = full_counts == best["counts"]
        log.append(f"chosen: {best['config']} line {chosen['line']} -> {best['counts']} "
                   f"(full replay {full_counts}, match={chosen['replay_matches']})")
    out = Path(args.out) if args.out else work / "chosen.json"
    out.write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    out.with_suffix(".log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 0


def evaluate(rows: list[dict], config: str, line: float) -> dict:
    before = [r["base"] for r in rows]
    after = [outcome_at(r, config, line) for r in rows]
    b, a = counts(rows, before), counts(rows, after)
    return {"config": config, "line": line, "before": b, "after": a,
            "rates_before": rates(b), "rates_after": rates(a),
            "paired": paired_intervals(rows, before, after), "by_kind": by_kind(rows, before, after),
            "vetoed_claims": sum(1 for x, y in zip(before, after) if x != y)}


def adoption(result: dict) -> dict:
    """Decision 90's rule, on the confirmation set."""
    p = result["paired"]
    fp_falls = (result["after"]["false_pass"] < result["before"]["false_pass"]
                and p["false_pass"]["ci95_pp"][1] < 0)
    fa_ok = p["false_alarm"]["change_pp"] <= 100 * ADOPT_PP
    sup_ok = p["supported_pass"]["change_pp"] >= -100 * ADOPT_PP
    return {"false_pass_falls_with_interval_below_zero": fp_falls,
            "false_alarm_within_2pp": fa_ok, "supported_pass_within_2pp": sup_ok,
            "adopt": fp_falls and fa_ok and sup_ok}


def adoption_two_row(result: dict) -> dict:
    """
    The rule for claims joining two rows (Decision 90's integration review):
    correct claims passed locally fall by at most 2pp, false alarms rise by at
    most 2pp, wrong claims passed do not rise — point estimates.
    """
    p = result["paired"]
    sup_ok = p["supported_pass"]["change_pp"] >= -100 * ADOPT_PP
    fa_ok = p["false_alarm"]["change_pp"] <= 100 * ADOPT_PP
    fp_ok = result["after"]["false_pass"] <= result["before"]["false_pass"]
    return {"supported_pass_within_2pp": sup_ok, "false_alarm_within_2pp": fa_ok,
            "false_pass_not_higher": fp_ok, "adopt": sup_ok and fa_ok and fp_ok}


def cmd_confirm(args) -> int:
    work = Path(args.work)
    data = json.loads((work / "cases.json").read_text(encoding="utf-8"))
    chosen = json.loads(Path(args.chosen).read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if args.split is None or r["split"] == args.split]
    if not chosen.get("config"):
        print("no configuration was chosen on dev; nothing to confirm")
        return 0
    config, line = chosen["config"], chosen["line"]
    if config not in rows[0]["veto"]:
        print(f"{config} was not scored on this set (evidences {data.get('evidences')})")
        return 1
    report = {"set": data["set"], "split": args.split, "chosen": chosen,
              "result": evaluate(rows, config, line)}
    report["rule_name"] = args.rule
    report["rule"] = (adoption_two_row if args.rule == "two-row" else adoption)(report["result"])
    report["secondary_same_scope_and_line"] = {
        other: evaluate(rows, other, line) for other in rows[0]["veto"]
        if other != config and other.endswith("/" + chosen["scope"])
    }
    fc = [r for r in rows if r["veto"][config]]
    report["figure_confirmed_scored"] = {
        "claims": len(fc),
        "supported": sum(r["label"] == "supported" for r in fc),
        "wrong": sum(r["label"] in NEGATIVE for r in fc),
    }
    report["timing"] = data.get("timing", {})
    if args.replay:
        cases = load_set(args)
        cases = [c for c in cases if args.split is None or c["split"] == args.split]
        scores = json.loads((work / "scores.json").read_text(encoding="utf-8"))
        full = asyncio.run(_full_replay(cases, scores, config, line))
        by_id = {r["id"]: r for r in rows}
        fast = [outcome_at(by_id[c["id"]], config, line) for c in cases]
        report["replay_matches"] = full == fast
    name = args.name or f"confirm_{data['set']}{'_' + args.split if args.split else ''}.json"
    (work / name).write_text(json.dumps(report, indent=1), encoding="utf-8")
    r = report["result"]
    print(json.dumps({"before": r["before"], "after": r["after"], "paired": r["paired"],
                      "rule": report["rule"], "replay_matches": report.get("replay_matches"),
                      "figure_confirmed_scored": report["figure_confirmed_scored"]}, indent=1))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval.run_figure_veto_eval",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("--set", choices=("benchmark", "realdocs"), required=True)
    p.add_argument("--data", help="eval/figure_veto_set.py output directory (realdocs)")
    p.add_argument("--work", required=True)
    p.add_argument("--evidence", default=",".join(EVIDENCE),
                   help=f"evidence forms to score, comma-separated (of {', '.join(EVIDENCE)})")
    p.add_argument("--seed-scores", help="an earlier run's scores.json: pairs it holds are not scored again")
    p.add_argument("--no-timing", action="store_true")
    p.add_argument("--timing-sample", type=int, default=150,
                   help="cases per configuration whose veto pairs are timed alone on the GPU")
    p = sub.add_parser("select")
    p.add_argument("--work", required=True)
    p.add_argument("--evidence", default="figure,headed",
                   help="candidate evidence forms, in tie-break order (Decision 90: figure,headed)")
    p.add_argument("--out", help="where to write the pick (default <work>/chosen.json)")
    p = sub.add_parser("confirm")
    p.add_argument("--work", required=True)
    p.add_argument("--chosen", required=True)
    p.add_argument("--split", help="benchmark split (test = the repeat look)")
    p.add_argument("--set", choices=("benchmark", "realdocs"))
    p.add_argument("--data")
    p.add_argument("--rule", choices=("single", "two-row"), default="single",
                   help="Decision 90's rule (single-figure set) or the two-row rule")
    p.add_argument("--name", help="output file name in <work>")
    p.add_argument("--replay", action="store_true", help="also replay the chosen line in full")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.disable(logging.WARNING)
    args = parse_args(argv)
    return {"score": cmd_score, "select": cmd_select, "confirm": cmd_confirm}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
