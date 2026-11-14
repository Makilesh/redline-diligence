"""
Verification benchmark — does the validator pass what it should, and only that?

    python -m eval.run_verification_eval                      # dev + test, NLI as configured
    python -m eval.run_verification_eval --split dev          # tuning runs: dev only
    python -m eval.run_verification_eval --no-nli             # deterministic stages alone (no model)
    python -m eval.run_verification_eval --baseline eval/verification_baseline.json   # exit 1 on regression
    python -m eval.run_verification_eval --answers tests/e2e_validation_results.json  # recorded answers

Every item of eval/verification_set.json is one claim about the sample data
room with a hand-assigned label — supported (verbatim / paraphrase / derived /
a true statement that the data room lacks something), unsupported (absent), or
contradicted (changed figure, figure taken from another metric or period,
swapped party, flipped negation, wrong period, wrong unit, wrong arithmetic, a
false statement that the data room lacks something it states). The claim is
verified exactly as an answer would be: wrapped
with a citation marker and passed to `verify_answer` with a production-shaped
context (eval/verification_corpus.py). No LLM judge is attached, so what is
measured is what the deterministic and local-model stages decide on their own,
and how much they leave for the judge:

    pass     every claim supported, no unverifiable calculation  -> "passed"
    judge    nothing flagged, but at least one claim undecided   -> goes to the LLM judge
    warn     a claim flagged unsupported by NLI, or a calculation
             that could not be reproduced                        -> "warning"
    fail     a figure not in the context, or a contradiction     -> "failed" + re-synthesis
    skipped  the splitter produced no checkable claim            -> never verified

The two error directions, reported separately because they are not equally bad:

    false pass   an unsupported or contradicted claim ends at `pass` or `skipped`.
                 This is the number that must not rise: a wrong figure shown as
                 verified is the failure the project exists to prevent.
    false alarm  a supported claim ends at `warn` or `fail`. An inconvenience —
                 a correct answer shown with a warning, or re-synthesised.

Families (a fact and its perturbations) are split dev/test as a unit, so a
threshold tuned on dev has not seen any variant of a test fact.

Also reported: the share of RANDOM figures that numeric grounding accepts
against data-room-sized contexts — a fabricated figure should essentially never
be "found".

Makes zero LLM calls. Writes verification.json / verification.md to
eval/results/ (regenerated output, not the gate); the gate is
eval/verification_baseline.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval.verification_corpus import (
    CORPUS_DIR,
    PROJECT_ROOT,
    build_context,
    corpus_chunks,
    file_context,
)

SET_PATH = PROJECT_ROOT / "eval" / "verification_set.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
OUTCOMES = ("pass", "judge", "warn", "fail", "skipped")
NEGATIVE_LABELS = ("unsupported", "contradicted")

# Random-figure grounding: shapes a fabricated figure takes in an answer.
FIGURE_SHAPES = ("$X.Y million", "$X million", "$X.Y billion", "N%", "N.N%", "N.Nx", "N bps")
FIGURE_SEED = 7
FIGURE_CONTEXTS = 60
FIGURES_PER_SHAPE = 50


# ==============================================================================
# The set
# ==============================================================================


def load_items(path: Path = SET_PATH, split: str | None = None) -> list[dict]:
    """
    Flattens the benchmark into one dict per claim.

    Args:
        path: verification_set.json.
        split: "dev", "test", or None for both.

    Returns:
        Items carrying their family's source, section, evidence and split.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for family in data["families"]:
        if split and family["split"] != split:
            continue
        for item in family["items"]:
            items.append({
                **item,
                "family": family["id"],
                "split": family["split"],
                "source": family["source"],
                "section": family["section"],
                "evidence": family["evidence"],
            })
    return items


def item_answer(item: dict) -> str:
    """The claim as the synthesizer would write it: one cited sentence."""
    return f"{item['claim']} [📄 {item['source']} | {item['section']}]"


def item_context(item: dict) -> list[dict]:
    """Production-shaped context: the evidence chunk(s) plus on-topic distractors."""
    evidence = [(e["source"], e["excerpt"]) for e in item["evidence"]]
    return build_context(item["claim"], evidence)


# ==============================================================================
# Scoring
# ==============================================================================


def outcome_of(report) -> str:
    """
    What the local stages decided for one verified answer.

    Args:
        report: VerificationReport from verify_answer (no judge attached).

    Returns:
        One of OUTCOMES.
    """
    checks = report.claim_checks
    if not checks:
        return "skipped"
    statuses = {c["status"] for c in checks}
    numeric_failure = any(
        c["status"] == "unsupported" and c["method"] == "numeric" for c in checks
    )
    if "contradicted" in statuses or numeric_failure:
        return "fail"
    unverifiable = any(n["status"] == "derived_unverified" for n in report.numeric_checks)
    if "unsupported" in statuses or unverifiable:
        return "warn"
    if "unverified" in statuses:
        return "judge"
    # verify_answer's own verdict is the tiebreak: anything it does not call
    # "passed" is not a pass here either.
    return "pass" if report.validation_status == "passed" else "warn"


async def evaluate(items: list[dict], *, scorer=None, use_nli: bool | None = None) -> list[dict]:
    """
    Verifies every item and records what the local stages decided.

    Args:
        items: Items from load_items.
        scorer: NLI scorer override (tests, backend comparisons); defaults to
            the configured local model.
        use_nli: Force NLI on/off; None follows VERIFY_NLI.

    Returns:
        One row per item: id, label, kind, split, outcome, and the per-claim
        detail needed to debug a miss.
    """
    from src.verification.claim_checker import verify_answer

    rows = []
    for item in items:
        report = await verify_answer(
            item_answer(item), item_context(item), item["claim"],
            scorer=scorer, judge=None, use_nli=use_nli,
        )
        rows.append({
            "id": item["id"],
            "split": item["split"],
            "label": item["label"],
            "kind": item["kind"],
            "claim": item["claim"],
            "outcome": outcome_of(report),
            "checks": [
                {k: c.get(k) for k in ("status", "method", "score", "reason", "numbers")}
                for c in report.claim_checks
            ],
            "elapsed_ms": report.elapsed_ms,
        })
    return rows


def _rates(rows: list[dict]) -> dict:
    counts = Counter(r["outcome"] for r in rows)
    n = len(rows)
    out = {"n": n, **{o: counts.get(o, 0) for o in OUTCOMES}}
    return out


def summarise(rows: list[dict]) -> dict:
    """
    Headline rates and the per-label / per-kind outcome tables.

    Args:
        rows: Output of evaluate.

    Returns:
        {"headline": {...}, "by_label": {...}, "by_kind": {...}}.
    """
    supported = [r for r in rows if r["label"] == "supported"]
    negative = [r for r in rows if r["label"] in NEGATIVE_LABELS]

    def share(group: list[dict], outcomes: tuple[str, ...]) -> float | None:
        if not group:
            return None
        return round(sum(r["outcome"] in outcomes for r in group) / len(group), 4)

    by_label: dict[str, dict] = {}
    by_kind: dict[str, dict] = {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["kind"])].append(r)
    for label in ("supported", *NEGATIVE_LABELS):
        by_label[label] = _rates([r for r in rows if r["label"] == label])
    for (label, kind), group in sorted(groups.items()):
        by_kind[f"{label}/{kind}"] = _rates(group)

    return {
        "headline": {
            "items": len(rows),
            "supported": len(supported),
            "negative": len(negative),
            # The hard number: wrong claims shown as verified (or never checked).
            "false_pass_rate": share(negative, ("pass", "skipped")),
            "false_passes": sum(r["outcome"] in ("pass", "skipped") for r in negative),
            "negative_caught_locally": share(negative, ("warn", "fail")),
            "negative_to_judge": share(negative, ("judge",)),
            # The inconvenience: correct claims flagged.
            "false_alarm_rate": share(supported, ("warn", "fail")),
            "false_alarms": sum(r["outcome"] in ("warn", "fail") for r in supported),
            "supported_pass_rate": share(supported, ("pass",)),
            "supported_to_judge": share(supported, ("judge",)),
            "judge_load": share(rows, ("judge",)),
        },
        "by_label": by_label,
        "by_kind": by_kind,
    }


# ==============================================================================
# Random-figure grounding
# ==============================================================================


def _random_figure(shape: str, rng: random.Random) -> str:
    if shape == "$X.Y million":
        return f"${rng.randint(1, 999)}.{rng.randint(0, 9)} million"
    if shape == "$X million":
        return f"${rng.randint(1, 999)} million"
    if shape == "$X.Y billion":
        return f"${rng.randint(1, 9)}.{rng.randint(0, 9)} billion"
    if shape == "N%":
        return f"{rng.randint(1, 99)}%"
    if shape == "N.N%":
        return f"{rng.randint(0, 99)}.{rng.randint(0, 9)}%"
    if shape == "N.Nx":
        return f"{rng.randint(0, 24)}.{rng.randint(0, 9)}x"
    if shape == "N bps":
        return f"{rng.randint(1, 60) * 5} bps"
    raise ValueError(shape)


def random_figure_grounding(
    seed: int = FIGURE_SEED,
    contexts: int = FIGURE_CONTEXTS,
    per_shape: int = FIGURES_PER_SHAPE,
) -> dict:
    """
    Share of random figures that numeric grounding reports as `grounded`.

    Each trial draws a context of 8 random ~8,000-character windows of the
    sample data room (about one synthesis context) and looks up random figures
    of each shape in it. A random figure is fabricated by construction, so
    every hit is a fabricated figure the validator would have accepted. Some
    hits are unavoidable — the data room does contain round percentages — which
    is why the same-kind, same-scale rate is reported next to it as the floor.

    Returns:
        {shape: {"n", "grounded", "floor"}} with rates in [0, 1].
    """
    from src.verification import numeric_grounding as ng

    rng = random.Random(seed)
    windows = []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for start in range(0, max(len(text) - 4000, 1), 4000):
            windows.append(text[start: start + 8000])

    totals = {shape: [0, 0, 0] for shape in FIGURE_SHAPES}
    for _ in range(contexts):
        chunks = [{"source_file": "window", "text": w} for w in rng.sample(windows, 8)]
        index = ng.ContextIndex(chunks)
        for shape in FIGURE_SHAPES:
            for _ in range(per_shape):
                mention = ng.extract_numbers(_random_figure(shape, rng))[0]
                tol = max(mention.half_unit, ng._REL_TOLERANCE * mention.value, 1e-9)
                same_kind = any(
                    f.mention.kind == mention.kind
                    and f.mention.explicit_scale == mention.explicit_scale
                    and abs(f.mention.value - mention.value) <= tol
                    for f in index.figures
                )
                totals[shape][0] += 1
                totals[shape][1] += index.lookup(mention) is not None
                totals[shape][2] += same_kind
    return {
        shape: {"n": n, "grounded": round(hit / n, 4), "floor": round(floor / n, 4)}
        for shape, (n, hit, floor) in totals.items()
    }


# ==============================================================================
# Recorded answers
# ==============================================================================


async def evaluate_answers(
    answers_path: Path,
    contexts_path: Path | None,
    *,
    scorer=None,
    use_nli: bool | None = None,
) -> dict:
    """
    Re-verifies recorded pipeline answers (tests/e2e_validation_results.json).

    The recorded file holds answers and the files they cite, not the retrieved
    chunks. With `contexts_path` ({question id: [chunk dicts]}, e.g. dumped
    from the retrieval harness) the real context is used; otherwise every chunk
    of each cited file stands in for it — a superset of what was retrieved, so
    numeric grounding is, if anything, more lenient than it was live.

    Returns:
        {"answers": [...per answer...], "status_counts": {...}, "claim_counts": {...}}.
    """
    from src.verification.claim_checker import verify_answer

    records = json.loads(answers_path.read_text(encoding="utf-8"))
    contexts = json.loads(contexts_path.read_text(encoding="utf-8")) if contexts_path else {}
    out = []
    for rec in records:
        answer = rec.get("answer") or ""
        if rec.get("refused") or not answer.strip():
            continue
        chunks = contexts.get(rec["id"])
        if chunks is None:
            cited = sorted({c["source_file"] for c in rec.get("citations") or []})
            chunks = file_context(cited)
        report = await verify_answer(
            answer, chunks, rec.get("query", ""), scorer=scorer, judge=None, use_nli=use_nli,
        )
        out.append({
            "id": rec["id"],
            "expect_refusal": bool(rec.get("expect_refusal")),
            "outcome": outcome_of(report),
            "claims": len(report.claim_checks),
            "status_counts": report.status_counts,
            "method_counts": report.method_counts,
            "numeric_counts": report.numeric_counts,
            "not_supported": [
                {"claim": c["claim"], "status": c["status"], "method": c["method"],
                 "score": c["score"], "reason": c["reason"]}
                for c in report.claim_checks if c["status"] != "supported"
            ],
        })
    claims = Counter()
    for a in out:
        for status, n in a["status_counts"].items():
            claims[status] += n
    return {
        "answers": out,
        "outcome_counts": dict(Counter(a["outcome"] for a in out)),
        "claim_counts": dict(claims),
        "context": "saved retrieval contexts" if contexts_path else "all chunks of each cited file",
    }


# ==============================================================================
# Report
# ==============================================================================


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def baseline_metrics(report: dict) -> dict:
    """The gated numbers, per split, plus the random-figure grounding rates."""
    out = {"splits": {}, "random_figure_grounding": {}}
    for split, summary in report["splits"].items():
        h = summary["headline"]
        out["splits"][split] = {
            "false_pass_rate": h["false_pass_rate"],
            "false_alarm_rate": h["false_alarm_rate"],
            "supported_pass_rate": h["supported_pass_rate"],
        }
    for shape, r in (report.get("random_figure_grounding") or {}).items():
        out["random_figure_grounding"][shape] = r["grounded"]
    return out


def compare_to_baseline(current: dict, baseline: dict, tolerance: float) -> list[str]:
    """
    Regressions against a recorded baseline.

    The false-pass rate and the random-figure grounding rates may not rise at
    all beyond rounding (tolerance applies to the other two): a change that
    passes one more wrong claim is not an improvement whatever else it buys.

    Returns:
        Human-readable regression lines; empty when none.
    """
    problems = []
    for split, base in (baseline.get("splits") or {}).items():
        cur = (current.get("splits") or {}).get(split)
        if cur is None:
            continue
        if cur["false_pass_rate"] > base["false_pass_rate"] + 1e-9:
            problems.append(
                f"{split}: false-pass rate {_pct(base['false_pass_rate'])} -> "
                f"{_pct(cur['false_pass_rate'])}"
            )
        if cur["false_alarm_rate"] > base["false_alarm_rate"] + tolerance:
            problems.append(
                f"{split}: false-alarm rate {_pct(base['false_alarm_rate'])} -> "
                f"{_pct(cur['false_alarm_rate'])}"
            )
        if cur["supported_pass_rate"] < base["supported_pass_rate"] - tolerance:
            problems.append(
                f"{split}: supported pass rate {_pct(base['supported_pass_rate'])} -> "
                f"{_pct(cur['supported_pass_rate'])}"
            )
    for shape, base_rate in (baseline.get("random_figure_grounding") or {}).items():
        cur_rate = (current.get("random_figure_grounding") or {}).get(shape)
        if cur_rate is not None and cur_rate > base_rate + 0.005:
            problems.append(
                f"random figures `{shape}` grounded {_pct(base_rate)} -> {_pct(cur_rate)}"
            )
    return problems


def render_markdown(report: dict) -> str:
    """Summary tables for humans."""
    lines = [
        f"# Verification benchmark — {report['generated_at'][:10]}", "",
        f"`{report['command']}`", "",
        f"NLI: {report['nli']} · device: {report['device']} · LLM calls: 0 · "
        f"{report['total_seconds']}s", "",
    ]
    for split, summary in report["splits"].items():
        h = summary["headline"]
        lines += [
            f"## {split} — {h['items']} claims ({h['supported']} supported, {h['negative']} not)", "",
            "| | rate | count |", "|---|---|---|",
            f"| **False pass** (wrong claim passed or never checked) | "
            f"**{_pct(h['false_pass_rate'])}** | {h['false_passes']}/{h['negative']} |",
            f"| Wrong claim caught locally (warn/fail) | {_pct(h['negative_caught_locally'])} | |",
            f"| Wrong claim left to the LLM judge | {_pct(h['negative_to_judge'])} | |",
            f"| **False alarm** (supported claim flagged) | **{_pct(h['false_alarm_rate'])}** | "
            f"{h['false_alarms']}/{h['supported']} |",
            f"| Supported claim passed locally | {_pct(h['supported_pass_rate'])} | |",
            f"| Supported claim left to the LLM judge | {_pct(h['supported_to_judge'])} | |",
            f"| Judge load (all claims) | {_pct(h['judge_load'])} | |", "",
            "| label / kind | n | pass | judge | warn | fail | skipped |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, r in summary["by_kind"].items():
            lines.append(
                f"| {name} | {r['n']} | {r['pass']} | {r['judge']} | {r['warn']} | "
                f"{r['fail']} | {r['skipped']} |"
            )
        lines.append("")
    figures = report.get("random_figure_grounding")
    if figures:
        lines += [
            "## Random figures reported as grounded", "",
            f"{FIGURE_CONTEXTS} contexts of 8 random ~8,000-character windows, "
            f"{FIGURES_PER_SHAPE} random figures per shape per context, seed {FIGURE_SEED}. "
            "Floor = a context figure of the same kind and scale within rounding.", "",
            "| fabricated shape | grounded | floor |", "|---|---|---|",
        ]
        for shape, r in figures.items():
            lines.append(f"| `{shape}` | {_pct(r['grounded'])} | {_pct(r['floor'])} |")
        lines.append("")
    answers = report.get("answers")
    if answers:
        lines += [
            f"## Recorded answers ({len(answers['answers'])}; context: {answers['context']})", "",
            "Outcome without the LLM judge: "
            + ", ".join(f"{k} {v}" for k, v in sorted(answers["outcome_counts"].items())) + ". "
            "Claims: " + ", ".join(f"{k} {v}" for k, v in sorted(answers["claim_counts"].items()))
            + ".", "",
        ]
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> dict:
    """Runs the selected splits (and optional extras) and assembles the report."""
    from src.verification import nli

    started = time.perf_counter()
    use_nli = False if args.no_nli else None
    corpus_chunks()  # chunk once, before timing anything per item

    splits = ("dev", "test") if args.split == "all" else (args.split,)
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m eval.run_verification_eval " + " ".join(sys.argv[1:]),
        "splits": {},
        "rows": [],
    }
    for split in splits:
        items = load_items(Path(args.set), split)
        if args.limit:
            items = items[: args.limit]
        rows = await evaluate(items, use_nli=use_nli)
        report["splits"][split] = summarise(rows)
        report["rows"].extend(rows)
        print(f"[verification] {split}: {len(rows)} claims", file=sys.stderr)

    if not args.no_random_figures:
        report["random_figure_grounding"] = random_figure_grounding()
    if args.answers:
        report["answers"] = await evaluate_answers(
            Path(args.answers), Path(args.contexts) if args.contexts else None, use_nli=use_nli,
        )

    enabled = nli.nli_enabled() and not args.no_nli
    report["nli"] = (nli.loaded_model_name() or nli.nli_model_name()) if enabled else "off"
    report["device"] = _device() if enabled else "n/a"
    report["total_seconds"] = round(time.perf_counter() - started, 1)
    return report


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_verification_eval",
        description="Labelled claim benchmark for the answer validator (no LLM calls).")
    parser.add_argument("--set", default=str(SET_PATH), help="benchmark file")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--no-nli", action="store_true",
                        help="deterministic stages only; loads no model")
    parser.add_argument("--no-random-figures", action="store_true",
                        help="skip the random-figure grounding measurement")
    parser.add_argument("--answers", help="recorded answers to re-verify "
                        "(tests/e2e_validation_results.json format)")
    parser.add_argument("--contexts", help="{question id: [chunks]} for --answers")
    parser.add_argument("--limit", type=int, help="first N claims per split (smoke runs)")
    parser.add_argument("--baseline", help="exit 1 if the gated rates regress against this file")
    parser.add_argument("--update-baseline", help="write the gated rates to this file")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="allowed worsening of the false-alarm and pass rates (default 0.02)")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    if not args.verbose:
        logging.disable(logging.WARNING)
    report = asyncio.run(run(args))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_no_nli" if args.no_nli else ""
    (out / f"verification{suffix}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    (out / f"verification{suffix}.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)

    current = baseline_metrics(report)
    if args.update_baseline:
        payload = {"generated_at": report["generated_at"], "command": report["command"],
                   "nli": report["nli"], **current}
        Path(args.update_baseline).write_text(
            json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        problems = compare_to_baseline(current, baseline, args.tolerance)
        for problem in problems:
            print(f"[verification] REGRESSION {problem}", file=sys.stderr)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
