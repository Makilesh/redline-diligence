"""
Numeric grounding on financial tables, measured (Decision 94).

    python -m eval.run_grounding_eval ground --data D:/redline_data/ground11/set --out <dir>/ground.json
    python -m eval.run_grounding_eval verify --data D:/redline_data/ground11/set --out <dir>/verify.json  # GPU
    python -m eval.run_grounding_eval compare --before <main>/ground.json --after <branch>/ground.json \
        --verify-before <main>/verify.json --verify-after <branch>/verify.json --out <dir>/compare.json

A wrong figure shown as verified is the failure the validator exists to
prevent, and on a table it is usually a real figure: another year's, or the
neighbouring row's. Numeric grounding confirms a claim's figure only when it
is tied to the row and period the claim names (numeric_grounding.ContextIndex
.bind); this harness measures how often that tie is wrong on the tables the
production ingestion writes.

ground   numeric grounding alone (ContextIndex + ground_texts, no model) on
         every claim of a figure_veto_set build: per claim "confirmed" (every
         figure confirmed, as claim_checker._FigureView reads it), else
         "unsupported" (a figure not found), "conflict" (found, for another
         row or period) or "unbound" (found, not tied to what the claim says).
verify   the production verifier on every claim: verify_answer with the
         configured checker (FactCG on GPU, with its figure veto), no judge;
         outcomes as run_verification_eval reads them (pass / judge / warn /
         fail / skipped).
compare  two runs over the same claims — main's code and the branch's —
         paired by claim: counts by label and kind, 95% intervals of each
         change that resample filings, and Decision 94's adoption rule.

Run `ground` and `verify` from a checkout of each code version: the claims and
their contexts come from the set, the code from the working directory. Makes
zero LLM calls. FinanceBench is CC BY-NC 4.0: outputs stay outside the
repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

# Grounding outcomes, read as run_figure_veto_eval.counts reads verifier
# outcomes: a confirmed claim passes on its figures; a figure not found fails
# the claim; a conflict or an unbound figure leaves it to a model or the judge.
AS_VERIFIER = {"confirmed": "pass", "conflict": "judge", "unbound": "judge", "unsupported": "fail"}
# Decision 94's rule: supported claims confirmed may fall by at most this much.
SUPPORTED_SLACK_PP = 1.0
# The supported claims that state a figure as the table prints it: the ones the
# unit note's reading cannot add to, so a loss among them cannot be masked.
PRINTED_KINDS = ("plain",)
# Decision 94's bootstrap seed, written down before anything was scored. (The
# first pass left it to run_figure_veto_eval's default, Decision 90's 20261119.)
SEED = 20261120


def grounding_outcome(checks: list) -> str:
    """
    What numeric grounding alone makes of one claim.

    Args:
        checks: The claim's NumericChecks (ground_texts).

    Returns:
        "confirmed" when claim_checker._FigureView would pass the claim on its
        figures; else "unsupported", "conflict" or "unbound", the first that
        any figure is.
    """
    figures = [n for n in checks if n.kind != "quantity"]
    missing_quantities = [n for n in checks if n.kind == "quantity" and n.status != "grounded"]
    if figures and all(n.is_confirmed for n in figures) and not missing_quantities:
        return "confirmed"
    if any(n.status == "unsupported" for n in figures):
        return "unsupported"
    if any(n.binding == "conflict" for n in figures):
        return "conflict"
    return "unbound"


def _rows(cases: list[dict]) -> list[dict]:
    return [{k: case.get(k) for k in ("id", "doc", "label", "kind", "claim")} for case in cases]


def cmd_ground(args) -> int:
    from eval.figure_veto_set import load_cases
    from src.verification.claims import split_claims
    from src.verification.numeric_grounding import ContextIndex, ground_texts

    started = time.perf_counter()
    cases = load_cases(Path(args.data))
    rows = _rows(cases)
    for row, case in zip(rows, cases):
        texts = [c.text for c in split_claims(case["answer"])]
        groups = ground_texts(texts, ContextIndex(case["chunks"]))
        checks = [n for group in groups for n in group]
        row["outcome"] = grounding_outcome(checks)
        row["figures"] = [
            {"raw": n.raw, "status": n.status, "binding": n.binding, "note": n.binding_note}
            for n in checks
        ]
    _write(args.out, "ground", rows, time.perf_counter() - started)
    return 0


def cmd_verify(args) -> int:
    from eval.figure_veto_set import load_cases
    from eval.run_verification_eval import outcome_of
    from src.verification import nli
    from src.verification.claim_checker import verify_answer

    started = time.perf_counter()
    cases = load_cases(Path(args.data))
    rows = _rows(cases)

    async def run() -> None:
        for row, case in zip(rows, cases):
            report = await verify_answer(case["answer"], case["chunks"], case["claim"],
                                         judge=None, use_nli=True)
            row["outcome"] = outcome_of(report)

    asyncio.run(run())
    model = nli.loaded_model_name()
    if args.require_model and model != args.require_model:
        raise RuntimeError(f"verified with {model}, not {args.require_model}")
    _write(args.out, "verify", rows, time.perf_counter() - started, model=model)
    return 0


def _write(path: str, mode: str, rows: list[dict], seconds: float, **extra) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": mode, "seconds": round(seconds, 1), **extra,
               "outcomes": dict(Counter(r["outcome"] for r in rows)), "rows": rows}
    out.write_text(json.dumps(payload, indent=0, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"mode": mode, "claims": len(rows), "seconds": payload["seconds"], **extra,
                      "outcomes": payload["outcomes"]}, indent=1))


def compare_runs(before: dict, after: dict, kinds: tuple[str, ...] | None = None) -> dict:
    """
    Two runs over the same claims, paired by claim id.

    Args:
        before: A `ground` or `verify` output (main's code).
        after: The same mode's output for the same set (the branch's).
        kinds: Restrict to claims of these kinds (all claims when None).

    Returns:
        Counts before and after (run_figure_veto_eval.counts: false_pass =
        wrong claims passed, supported_pass = supported claims passed,
        false_alarm = supported claims warned or failed), paired 95%
        intervals resampling filings, and outcomes by label and kind.
    """
    from eval.run_figure_veto_eval import by_kind, counts, paired_intervals

    if before["mode"] != after["mode"]:
        raise ValueError(f"cannot pair a {before['mode']} run with a {after['mode']} run")
    old = {r["id"]: r for r in before["rows"]}
    if set(old) != {r["id"] for r in after["rows"]}:
        raise ValueError("the two runs hold different claims")
    read = AS_VERIFIER.get if before["mode"] == "ground" else (lambda o: o)
    rows, b, a = [], [], []
    for r in after["rows"]:
        if kinds is not None and r["kind"] not in kinds:
            continue
        rows.append({"id": r["id"], "cluster": r["doc"], "label": r["label"], "kind": r["kind"]})
        b.append(read(old[r["id"]]["outcome"]))
        a.append(read(r["outcome"]))
    return {"before": counts(rows, b), "after": counts(rows, a),
            "paired": paired_intervals(rows, b, a, seed=SEED), "by_kind": by_kind(rows, b, a),
            "moved": sum(x != y for x, y in zip(b, a))}


def adoption(ground: dict, printed: dict, verify: dict | None) -> dict:
    """
    Decision 94's rule on the new set (the benchmark's two conditions are
    read from its own runs):
        (a) wrong claims confirmed by grounding alone fall, the paired 95%
            interval (filings resampled) entirely below zero;
        (b) supported claims confirmed fall by at most 1pp, on all supported
            claims and on those stated as printed;
        (e) the production verifier passes no more wrong claims.
    """
    p = ground["paired"]
    falls = ground["after"]["false_pass"] < ground["before"]["false_pass"] and p["false_pass"]["ci95_pp"][1] < 0
    supported = p["supported_pass"]["change_pp"] >= -SUPPORTED_SLACK_PP
    as_printed = printed["paired"]["supported_pass"]["change_pp"] >= -SUPPORTED_SLACK_PP
    production = verify is None or verify["after"]["false_pass"] <= verify["before"]["false_pass"]
    return {"a_wrong_confirmed_fall_interval_below_zero": falls,
            "b_supported_confirmed_within_1pp": supported,
            "b_as_printed_supported_confirmed_within_1pp": as_printed,
            "e_production_wrong_passed_not_higher": production,
            "adopt_on_this_set": falls and supported and as_printed and production}


def cmd_compare(args) -> int:
    def load(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    ground = compare_runs(load(args.before), load(args.after))
    printed = compare_runs(load(args.before), load(args.after), kinds=PRINTED_KINDS)
    verify = None
    if args.verify_before and args.verify_after:
        verify = compare_runs(load(args.verify_before), load(args.verify_after))
    report = {"grounding": ground, "grounding_as_printed": printed, "production": verify,
              "rule": adoption(ground, printed, verify)}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    brief = {name: {"before": r["before"], "after": r["after"], "paired": r["paired"]}
             for name, r in (("grounding", ground), ("grounding_as_printed", printed), ("production", verify))
             if r is not None}
    print(json.dumps({**brief, "rule": report["rule"]}, indent=1))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval.run_grounding_eval",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ground")
    p.add_argument("--data", required=True, help="a figure_veto_set build directory")
    p.add_argument("--out", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--require-model", help="fail unless this checker scored (e.g. the FactCG model id)")
    p = sub.add_parser("compare")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--verify-before")
    p.add_argument("--verify-after")
    p.add_argument("--out")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.disable(logging.WARNING)
    args = parse_args(argv)
    return {"ground": cmd_ground, "verify": cmd_verify, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
