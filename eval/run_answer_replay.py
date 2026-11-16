"""
Answer replay — the verifier measured on answers the live pipeline wrote.

    python -m eval.run_answer_replay --build-answers           # refresh eval/answer_replay_answers.json
    python -m eval.run_answer_replay --output-dir <dir>         # replay retrieval + verify (GPU: FactCG)
    python -m eval.run_answer_replay --contexts <dir>/contexts.json --output-dir <dir>   # reuse contexts
    python -m eval.run_answer_replay --contexts ... --labels eval/answer_replay_labels.json --split dev

eval/verification_set.json is 542 short, single-fact, hand-written claims. Live
answers are longer, multi-fact, written in tables and bullet lists, and they
decline in their own words; a verifier tuned on the first need not behave the
same on the second. This harness puts the second in front of it.

For every recorded live answer (eval/answer_replay_answers.json, extracted from
the committed end-to-end runs by --build-answers):

1. The context is rebuilt by the production retrieval node
   (retrieval_executor_node) on the sample data room in an in-memory Qdrant,
   from what the trace recorded: Agent 1's query type, reformulated query,
   whitelisted category filter and sub-questions; after a rewrite, the
   rewritten query at the same rewrite iteration. When the trace carries
   `retrieved_chunk_ids` (runs recorded after Decision 72) those exact chunks
   are expanded instead of searching again.
2. The answer is verified by the production verify_answer with the LLM judge
   replaced by a stub that records which claims it would have been asked about
   and answers nothing — so every number here is what the local stages
   (numeric grounding, the entailment model) decide on their own.
3. Reproduction is checked, because a replayed context that differs from the
   live one measures nothing: every chunk the answer cites must be in the
   rebuilt context and the reranked sources must match the trace; for the run
   whose verifier code is the current one, each claim's live verdict class
   (passed locally / flagged locally / sent to the judge) is compared too.
   Answers that fail the context check are reported and left out of the rates.

With --labels, claims carry a hand label (supported / not_supported /
not_checkable, against the nine documents of the data room, see the guide in
the labels file) and the report gives, on the chosen split, the false-alarm rate
(correct claims flagged), the false-pass rate (wrong claims passed) and the share
sent to the judge, per stage, with 95% bootstrap intervals that resample
questions. Labelled claims carry the weight of the stratum they were sampled
from, so the rates estimate the whole population of replayed claims.

`--as-of pm-run` (default) replays retrieval as it was when the reference run
was recorded: Decision 67's document focus and Decision 68's table cap were
merged after it, so both are switched off; `--as-of current` uses today's
settings. Makes zero LLM calls.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANSWERS_PATH = PROJECT_ROOT / "eval" / "answer_replay_answers.json"
LABELS_PATH = PROJECT_ROOT / "eval" / "answer_replay_labels.json"

# The recorded runs, oldest last. `verifier_current` marks the run whose
# verification code is today's (src/verification unchanged since it was
# recorded), so its per-claim live verdicts can be compared one to one.
RUNS = {
    "pm": {
        "source": "tests/e2e_validation_results.json",
        "commit": "a6031a4",
        "recorded": "2026-11-15 pm (FactCG on GPU + lite judge)",
        "verifier_current": True,
        "chunk_ids_comparable": True,
    },
    "am": {
        "source": "tests/e2e_validation_results.json",
        "commit": "564d4f4",
        "recorded": "2026-11-15 am (nli-deberta-v3-small + lite judge)",
        "verifier_current": False,
        "chunk_ids_comparable": True,
    },
    "aug": {
        "source": "tests/e2e_validation_results.json",
        "commit": "5d73540",
        "recorded": "2026-08-19 (LLM-only validator, earlier chunker)",
        "verifier_current": False,
        # Documents then got random ids, so chunk ids from that run name
        # nothing in today's index; sources and section headings still do.
        "chunk_ids_comparable": False,
    },
}

# Fixed before any measurement (Decision 72): questions are split by id prefix,
# shuffled with this seed inside each prefix, and dealt dev, test, dev, ...
SPLIT_SEED = 72
BOOTSTRAP_SEED = 72
BOOTSTRAP_SAMPLES = 2000

LABELS = ("supported", "not_supported", "not_checkable")
# What the local stages decided for one claim, judge stubbed.
OUTCOMES = ("pass", "flag", "judge", "unchecked", "skipped")


# ==============================================================================
# Recorded answers
# ==============================================================================


def _git_show(commit: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout


def _trace(rec: dict, agent: str) -> list[dict]:
    return [t for t in rec.get("agent_trace") or [] if isinstance(t, dict) and t.get("agent") == agent]


def extract_record(run: str, rec: dict) -> dict:
    """
    What the replay needs from one recorded answer, and nothing else.

    Args:
        run: Run key (RUNS).
        rec: One entry of a recorded tests/e2e_validation_results.json.

    Returns:
        A compact record: the question, the answer, its citations, Agent 1's
        output, every retrieval pass and rewrite, and every validation.
    """
    qi = (_trace(rec, "query_intelligence") or [{}])[0].get("output") or {}
    return {
        "run": run,
        "id": rec["id"],
        "query": rec["query"],
        "golden_query_type": rec.get("query_type"),
        "answer": rec.get("answer") or "",
        "refused": bool(rec.get("refused")),
        "expect_refusal": bool(rec.get("expect_refusal")),
        "validation_status": rec.get("validation_status"),
        "citations": [
            {k: c.get(k) for k in ("chunk_id", "source_file", "section_heading")}
            for c in rec.get("citations") or []
        ],
        "query_intelligence": {
            "query_type": qi.get("query_type"),
            "reformulated_query": qi.get("reformulated_query"),
            "metadata_filters": qi.get("metadata_filters") or {},
            "sub_questions": qi.get("sub_questions") or [],
            "requires_cross_document": qi.get("requires_cross_document"),
            "requires_numerical_precision": qi.get("requires_numerical_precision"),
        },
        "retrievals": [
            {k: t.get(k) for k in ("sub_questions", "sources", "reranked_count",
                                   "retrieved_chunk_ids") if k in t}
            for t in _trace(rec, "retrieval_executor")
        ],
        "rewrites": [t.get("rewritten_query") for t in _trace(rec, "query_rewriter")],
        "validations": [
            {k: t.get(k) for k in ("attempt", "validation_status", "llm_calls", "nli_model",
                                   "llm_model", "status_counts", "method_counts", "claim_checks")
             # Per-claim live verdicts are only comparable for a run whose
             # verifier is today's; elsewhere they are dead weight.
             if k in t and (k != "claim_checks" or RUNS[run]["verifier_current"])}
            for t in _trace(rec, "hallucination_validator")
        ],
    }


def build_answers(path: Path = ANSWERS_PATH) -> dict:
    """Extracts every run's answers from git history into one committed file."""
    runs = {}
    records = []
    for run, info in RUNS.items():
        raw = json.loads(_git_show(info["commit"], info["source"]))
        runs[run] = {k: v for k, v in info.items()}
        runs[run]["answers"] = len(raw)
        records.extend(extract_record(run, rec) for rec in raw)
    data = {
        "_about": "Live answers from the committed end-to-end runs, as recorded "
                  "(eval/run_answer_replay.py --build-answers). Regenerate, do not edit.",
        "runs": runs,
        "records": records,
    }
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def load_answers(path: Path = ANSWERS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_of(question_ids: list[str], seed: int = SPLIT_SEED) -> dict[str, str]:
    """
    Question id -> "dev" | "test". Fixed rule (notes in DECISIONS_LOG 72).

    Grouped by id prefix ("fin", "legal", "ctrl", ...), each group sorted,
    shuffled with `seed` and dealt alternately starting with dev — so every
    question family is on both sides.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for qid in sorted(set(question_ids)):
        groups[qid.split("_")[0]].append(qid)
    out = {}
    for prefix in sorted(groups):
        ids = groups[prefix]
        random.Random(f"{seed}:{prefix}").shuffle(ids)
        for i, qid in enumerate(ids):
            out[qid] = "dev" if i % 2 == 0 else "test"
    return out


def claim_key(run: str, qid: str, text: str) -> str:
    """Stable id of one claim: run, question and a hash of its clean text."""
    return f"{run}:{qid}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


# ==============================================================================
# Context replay
# ==============================================================================


def replay_state(rec: dict, deal_id: str, as_of: str) -> tuple[dict, list[str]]:
    """
    The AgentState the retrieval node saw on the pass that fed the final answer.

    Returns:
        (state, caveats) — caveats name anything the trace did not record.
    """
    from src.llm.schemas import QUERY_TYPES
    from src.agents.retrieval_strategy import apply_filter_overrides, get_retrieval_config

    qi = rec["query_intelligence"]
    caveats = []
    query_type = qi.get("query_type")
    if query_type not in QUERY_TYPES:
        caveats.append(f"query type {query_type!r} not recorded; golden type used")
        query_type = rec.get("golden_query_type") or "summary"
    parsed_intent = {
        "query_type": query_type,
        "requires_cross_document": bool(qi.get("requires_cross_document")),
        "requires_numerical_precision": bool(qi.get("requires_numerical_precision")),
    }
    passes = rec["retrievals"]
    last = passes[-1] if passes else {}
    rewrites = [q for q in rec["rewrites"] if q]
    current = rewrites[-1] if rewrites else (qi.get("reformulated_query") or rec["query"])
    if rewrites:
        caveats.append("rewritten: the rewriter's config/filter proposals are not traced")
    sub_questions = last.get("sub_questions")
    if sub_questions is None:
        sub_questions = qi.get("sub_questions") or []
    config = get_retrieval_config(query_type, parsed_intent)
    if as_of == "pm-run":
        config["document_focus_k"] = 0
    state = {
        "original_query": rec["query"],
        "current_query": current,
        "deal_id": deal_id,
        "query_type": query_type,
        "parsed_intent": parsed_intent,
        "extracted_filters": apply_filter_overrides({}, qi.get("metadata_filters")),
        "sub_questions": [s for s in sub_questions if isinstance(s, str) and s.strip()],
        "include_pii": False,
        "rewrite_iteration": len(rewrites),
        "retrieval_config": config,
    }
    return state, caveats


async def _exact_context(client, chunk_ids: list[str], config: dict,
                         deal_id: str) -> tuple[list[dict], list[dict]]:
    """Expands exactly the chunks a live trace recorded (no search)."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    from src.vector_db.constants import COLLECTION_NAME
    from src.vector_db.parent_child_retrieval import expand_context

    points, _ = await client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids))]),
        limit=len(chunk_ids) + 10, with_payload=True, with_vectors=False,
    )
    by_id = {p.payload["chunk_id"]: p.payload for p in points}
    reranked = [dict(by_id[c]) for c in chunk_ids if c in by_id]
    return reranked, await expand_context(
        chunks=reranked,
        include_parents=config.get("use_parent_expansion", True),
        include_siblings=config.get("use_sibling_expansion", True),
        include_pii=False,
        deal_id=deal_id,
    )


def _jsonable(chunk: dict) -> dict:
    return json.loads(json.dumps(chunk, default=float))


# The categories the live runs ingested the data room with
# (tests/run_end_to_end_validation.py pins all nine). They decide what Agent 1's
# category filter keeps, so the retrieval eval's three pins (eval/corpus.py) do
# not reproduce a live context: with them the credit agreement summary was
# classified legal, and every financial question lost it.
LIVE_CATEGORY_OVERRIDES = {
    "aurora_financials_fy2023.txt": "financial",
    "quality_of_earnings_report_fy2023.txt": "financial",
    "credit_agreement_summary.txt": "financial",
    "merger_agreement_v2_final.txt": "legal",
    "customer_contracts_schedule.txt": "legal",
    "employment_and_retention_agreements.txt": "legal",
    "ip_portfolio_and_litigation_schedule.txt": "legal",
    "regulatory_and_data_privacy_memo.txt": "regulatory",
    "board_deck_strategic_review_mar2024.txt": "board",
}


async def build_live_index():
    """The sample data room in an in-memory Qdrant, categorised as the live runs were."""
    from qdrant_client import AsyncQdrantClient

    from eval.corpus import CORPUS_DIR, DEAL_ID
    from src.data_processing.ingest_pipeline import index_document
    from src.vector_db.collection_manager import setup_collections
    from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME

    client = AsyncQdrantClient(location=":memory:")
    await setup_collections(client, COLLECTION_NAME, PARENT_COLLECTION_NAME)
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        await index_document(str(path), path.name, DEAL_ID,
                             LIVE_CATEGORY_OVERRIDES.get(path.name), client=client)
    return client


async def replay_contexts(records: list[dict], as_of: str) -> dict[str, dict]:
    """
    Rebuilds every answer's context with the production retrieval node.

    Returns:
        {"run:id": {"reranked_ids", "sources", "context", "caveats", "elapsed_ms"}}.
    """
    from eval.corpus import DEAL_ID, use_client
    from src.agents.retrieval_executor import retrieval_executor_node
    from src.vector_db.reranker import warm_models

    if as_of == "pm-run":
        os.environ["TABLE_REPRESENTATION_CAP"] = "0"
    print("[replay] indexing the sample data room ...", file=sys.stderr)
    client = await build_live_index()
    out = {}
    with use_client(client):
        await warm_models()
        for rec in records:
            key = f"{rec['run']}:{rec['id']}"
            if not rec["answer"].strip() or rec["refused"] or not rec["retrievals"]:
                continue
            state, caveats = replay_state(rec, DEAL_ID, as_of)
            start = time.perf_counter()
            recorded_ids = rec["retrievals"][-1].get("retrieved_chunk_ids")
            if recorded_ids:
                reranked, context = await _exact_context(
                    client, recorded_ids, state["retrieval_config"], DEAL_ID)
                caveats.append("exact: recorded retrieved_chunk_ids expanded")
            else:
                result = await retrieval_executor_node(state)
                reranked, context = result["reranked_results"], result["expanded_context"]
            out[key] = {
                "reranked_ids": [c.get("chunk_id") for c in reranked],
                "sources": sorted({c.get("source_file", "") for c in reranked} - {""}),
                "context": [_jsonable(c) for c in context],
                "caveats": caveats,
                "state": {k: state[k] for k in ("current_query", "query_type", "extracted_filters",
                                                "sub_questions", "rewrite_iteration")},
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
            }
    return out


def context_reproduced(rec: dict, replayed: dict) -> dict:
    """
    Whether the rebuilt context is the one the live answer was verified against.

    Two conditions: every chunk the answer cites is in the rebuilt context, and
    the reranked sources equal the trace's. For a run whose chunk ids name
    nothing today (RUNS[...]["chunk_ids_comparable"]), a citation counts when a
    rebuilt chunk has its source file and section heading.
    """
    comparable = RUNS[rec["run"]]["chunk_ids_comparable"]
    context = replayed["context"]
    ids = {c.get("chunk_id") for c in context}
    sections = {(c.get("source_file"), (c.get("section_heading") or "").strip().lower()) for c in context}
    missing = []
    for cit in rec["citations"]:
        if comparable and cit.get("chunk_id"):
            if cit["chunk_id"] not in ids:
                missing.append(cit["chunk_id"])
        elif (cit.get("source_file"), (cit.get("section_heading") or "").strip().lower()) not in sections:
            missing.append(f"{cit.get('source_file')} | {cit.get('section_heading')}")
    recorded_sources = sorted((rec["retrievals"][-1].get("sources") or []) if rec["retrievals"] else [])
    same_sources = recorded_sources == replayed["sources"]
    return {
        "reproduced": not missing and same_sources,
        "missing_citations": missing,
        "same_sources": same_sources,
        "recorded_sources": recorded_sources,
        "replayed_sources": replayed["sources"],
    }


# ==============================================================================
# Verification with the judge stubbed
# ==============================================================================


class ScoreCache:
    """
    Entailment scores keyed by (premise, hypothesis), written next to the run.

    Splitter experiments change which claims exist, not how an unchanged claim
    is scored, so re-scoring every pair on the GPU for each variant measures
    nothing new. Pairs the cache lacks are scored by the real model.
    """

    def __init__(self, path: Path | None):
        self.path = path
        self.model: str | None = None
        self.scores: dict[str, list[float]] = {}
        self.misses = 0
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.model, self.scores = data.get("model"), data.get("scores", {})

    @staticmethod
    def _key(premise: str, hypothesis: str) -> str:
        return hashlib.sha1(f"{premise}\x00{hypothesis}".encode("utf-8")).hexdigest()

    async def score(self, pairs):
        from src.verification import nli

        keys = [self._key(p, h) for p, h in pairs]
        todo = [(k, pair) for k, pair in zip(keys, pairs) if k not in self.scores]
        if todo:
            self.misses += len(todo)
            fresh = await nli.ascore_pairs([pair for _, pair in todo])
            model = nli.loaded_model_name()
            if self.model and model != self.model:
                raise RuntimeError(f"score cache holds {self.model} scores; {model} loaded")
            self.model = model
            for (k, _), s in zip(todo, fresh):
                self.scores[k] = [s.entailment, s.contradiction, s.neutral]
        return [nli.NLIScore(*self.scores[k]) for k in keys]

    def save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps({"model": self.model, "scores": self.scores}),
                                 encoding="utf-8")


def claim_outcome(check: dict) -> str:
    """
    pass / flag / judge / unchecked for one claim check, judge stubbed.

    "judge" is every claim the local stages left open — including those past
    the judge's per-answer cap, which stay unverified live as well; "unchecked"
    is only a claim past the entailment budget.
    """
    if check["status"] == "supported":
        return "pass"
    if check["status"] in ("unsupported", "contradicted"):
        return "flag"
    if (check.get("reason") or "").startswith("not checked: beyond"):
        return "unchecked"
    return "judge"


def live_class(check: dict) -> str:
    """The same classes for a live claim check (the judge's verdicts folded into "judge")."""
    if check.get("method") == "llm":
        return "judge"
    if check.get("status") == "supported":
        return "pass"
    if check.get("status") in ("unsupported", "contradicted"):
        return "flag"
    return "judge" if check.get("status") == "unverified" else "unchecked"


async def verify_records(records: list[dict], contexts: dict, cache: ScoreCache | None,
                         use_nli: bool = True) -> list[dict]:
    """
    Verifies each answer against its rebuilt context with the judge stubbed.

    Returns:
        One row per answer, with per-claim detail.
    """
    from src.verification import claim_checker
    from src.verification.claims import split_claims

    rows = []
    for rec in records:
        key = f"{rec['run']}:{rec['id']}"
        replayed = contexts.get(key)
        if replayed is None:
            continue
        asked: list[str] = []

        async def stub_judge(query, claims, documents, _asked=asked):
            _asked.extend(c["claim"] for c in claims)
            return {}, "stub"

        scorer = cache.score if (cache and use_nli) else None
        with _cached_model_name(cache):
            report = await claim_checker.verify_answer(
                rec["answer"], replayed["context"], rec["query"],
                scorer=scorer, judge=stub_judge, use_nli=use_nli,
            )
        claims = split_claims(rec["answer"])
        asked_set = set(asked)
        checks = []
        for claim, check in zip(claims, report.claim_checks):
            checks.append({
                "key": claim_key(rec["run"], rec["id"], claim.text),
                "index": claim.index,
                "claim": claim.text,
                "outcome": claim_outcome(check),
                # False for an open claim past the judge's per-answer cap.
                "asked_judge": claim.text in asked_set,
                "status": check["status"],
                "method": check["method"],
                "score": check["score"],
                "reason": check["reason"],
                "is_absence": claim.is_absence,
                "is_table_row": claim.is_table_row,
                "has_numbers": claim.has_numbers,
                "numbers": [{k: n.get(k) for k in ("raw", "status", "binding")} for n in check["numbers"]],
                "evidence": check.get("evidence", ""),
                "evidence_source": check.get("evidence_source", ""),
                # Figures stated only outside the documents the claim cites (Decision 78).
                "citation_mismatch": check.get("citation_mismatch") or [],
            })
        rows.append({
            "key": key,
            "run": rec["run"],
            "id": rec["id"],
            "validation_status": report.validation_status,
            "judge_called": bool(asked),
            "judge_claims": len(asked),
            "nli_model": report.nli_model,
            "citation_counts": report.citation_counts,
            "claims": checks,
        })
    return rows


@contextlib.contextmanager
def _cached_model_name(cache: ScoreCache | None):
    """
    While verifying from a score cache, loaded_model_name() falls back to the
    model that produced the cached scores.

    verify_answer picks its decision lines by the model that scored
    (thresholds_for(nli.loaded_model_name())). When every pair comes from the
    cache no model loads, and without this the cached FactCG scores would be
    read against the cross-encoders' lines.
    """
    from src.verification import nli

    if cache is None or not cache.model:
        yield
        return
    original = nli.loaded_model_name
    nli.loaded_model_name = lambda: original() or cache.model
    try:
        yield
    finally:
        nli.loaded_model_name = original


def verdict_reproduction(rec: dict, row: dict) -> dict:
    """Per-claim agreement with the live run's final validation (current-verifier runs only)."""
    if not RUNS[rec["run"]]["verifier_current"] or not rec["validations"]:
        return {"comparable": False}
    live = rec["validations"][-1].get("claim_checks") or []
    replay = row["claims"]
    agree = 0
    pairs = []
    for i, check in enumerate(replay):
        lc = live[i] if i < len(live) else None
        same_text = bool(lc) and lc.get("claim", "")[:200] == check["claim"][:200]
        live_c = live_class(lc) if lc else None
        ok = same_text and live_c == check["outcome"]
        agree += ok
        pairs.append({"key": check["key"], "live": live_c, "live_status": lc.get("status") if lc else None,
                      "live_method": lc.get("method") if lc else None, "same_text": same_text, "agree": ok})
    return {
        "comparable": True,
        "live_claims": len(live),
        "replay_claims": len(replay),
        "agree": agree,
        "claims": pairs,
    }


# ==============================================================================
# Rates
# ==============================================================================


def load_labels(path: Path = LABELS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _weighted_rates(items: list[dict]) -> dict:
    """
    Weighted shares over labelled claims.

    items: {"label", "outcome", "method", "weight"}.
    """
    def share(group, pred):
        total = sum(i["weight"] for i in group)
        return (sum(i["weight"] for i in group if pred(i)) / total) if total else None

    correct = [i for i in items if i["label"] == "supported"]
    wrong = [i for i in items if i["label"] == "not_supported"]
    nuisance = [i for i in items if i["label"] in ("supported", "not_checkable")]
    flagged = lambda i: i["outcome"] == "flag"  # noqa: E731
    return {
        "false_alarm": share(correct, flagged),
        "false_alarm_numeric": share(correct, lambda i: flagged(i) and i["method"] == "numeric"),
        "false_alarm_nli": share(correct, lambda i: flagged(i) and i["method"] == "nli"),
        "flag_incl_not_checkable": share(nuisance, flagged),
        "correct_to_judge": share(correct, lambda i: i["outcome"] == "judge"),
        "correct_passed": share(correct, lambda i: i["outcome"] == "pass"),
        "false_pass": share(wrong, lambda i: i["outcome"] in ("pass", "skipped")),
        "wrong_flagged": share(wrong, flagged),
        "wrong_to_judge": share(wrong, lambda i: i["outcome"] == "judge"),
        "judge_share": share(items, lambda i: i["outcome"] == "judge"),
    }


def rates_with_intervals(items: list[dict], samples: int = BOOTSTRAP_SAMPLES,
                         seed: int = BOOTSTRAP_SEED) -> dict:
    """Point estimates plus 95% percentile intervals, resampling questions."""
    point = _weighted_rates(items)
    by_q: dict[str, list[dict]] = defaultdict(list)
    for i in items:
        by_q[i["question"]].append(i)
    qids = sorted(by_q)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        pick = [i for q in (rng.choice(qids) for _ in qids) for i in by_q[q]]
        for k, v in _weighted_rates(pick).items():
            if v is not None:
                draws[k].append(v)
    out = {}
    for k, v in point.items():
        d = sorted(draws.get(k, []))
        ci = [round(d[int(0.025 * (len(d) - 1))], 4), round(d[int(0.975 * (len(d) - 1))], 4)] if d else None
        out[k] = {"value": None if v is None else round(v, 4), "ci95": ci}
    counts = Counter(i["label"] for i in items)
    out["labelled"] = dict(counts)
    out["questions"] = len(qids)
    return out


def labelled_items(rows: list[dict], labels: dict, split: dict[str, str], which: str,
                   excluded: set[str]) -> list[dict]:
    """
    Joins replay outcomes with labels for one split.

    A labelled claim that the splitter no longer produces is "skipped"; its
    weight is the inverse of its sampling probability (labels["strata"]).
    """
    present = {c["key"]: (row, c) for row in rows for c in row["claims"]}
    items = []
    for key, lab in labels["labels"].items():
        run, qid, _ = key.split(":")
        if which != "all" and split.get(qid) != which:
            continue
        if f"{run}:{qid}" in excluded:
            continue
        hit = present.get(key)
        items.append({
            "key": key,
            "question": qid,
            "label": lab["label"],
            "outcome": hit[1]["outcome"] if hit else "skipped",
            "method": hit[1]["method"] if hit else "none",
            "weight": labels["strata"][lab["stratum"]]["weight"],
            "cause": lab.get("cause"),
            "ambiguous": bool(lab.get("ambiguous")),
        })
    return items


def cause_breakdown(items: list[dict]) -> dict:
    """Remaining false alarms and judge-bound correct claims by labelled cause."""
    out = {"flagged_correct": Counter(), "judge_correct": Counter()}
    for i in items:
        if i["label"] not in ("supported", "not_checkable"):
            continue
        cause = i.get("cause") or ("fragment" if i["label"] == "not_checkable" else "unlabelled")
        if i["outcome"] == "flag":
            out["flagged_correct"][cause] += 1
        elif i["outcome"] == "judge":
            out["judge_correct"][cause] += 1
    return {k: dict(v.most_common()) for k, v in out.items()}


# ==============================================================================
# Driver
# ==============================================================================


async def run(args: argparse.Namespace) -> dict:
    data = load_answers(Path(args.answers))
    records = [r for r in data["records"] if r["run"] in args.runs]
    if args.questions:
        records = [r for r in records if r["id"] in set(args.questions)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    if args.contexts:
        contexts = json.loads(Path(args.contexts).read_text(encoding="utf-8"))
    else:
        contexts = await replay_contexts(records, args.as_of)
        (output_dir / "contexts.json").write_text(json.dumps(contexts, ensure_ascii=False), encoding="utf-8")
    retrieval_s = time.perf_counter() - started

    cache = ScoreCache(Path(args.score_cache)) if args.score_cache else None
    rows = await verify_records(records, contexts, cache, use_nli=not args.no_nli)
    if cache:
        cache.save()

    by_key = {f"{r['run']}:{r['id']}": r for r in records}
    split = split_of([r["id"] for r in data["records"]])
    reproduction = {}
    for row in rows:
        rec = by_key[row["key"]]
        repro = context_reproduced(rec, contexts[row["key"]])
        repro["verdicts"] = verdict_reproduction(rec, row)
        repro["caveats"] = contexts[row["key"]].get("caveats", [])
        reproduction[row["key"]] = repro
        row["split"] = split.get(row["id"])
        row["context_reproduced"] = repro["reproduced"]

    excluded = {k for k, v in reproduction.items() if not v["reproduced"]}
    summary = summarise(rows, reproduction)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": args.as_of,
        "runs": {k: RUNS[k] for k in args.runs},
        "split_seed": SPLIT_SEED,
        "split": split,
        "nli": "off" if args.no_nli else (cache.model if cache and cache.model else
                                          next((r["nli_model"] for r in rows if r["nli_model"]), None)),
        "score_cache_misses": cache.misses if cache else None,
        "retrieval_seconds": round(retrieval_s, 1),
        "summary": summary,
        "reproduction": reproduction,
        "answers": rows,
    }
    if args.labels:
        labels = load_labels(Path(args.labels))
        report["rates"] = {}
        for which in args.split:
            items = labelled_items(rows, labels, split, which, excluded)
            report["rates"][which] = {
                **rates_with_intervals(items),
                "causes": cause_breakdown(items),
                # The same rates without the labels that turned on a judgement call.
                "unambiguous": rates_with_intervals([i for i in items if not i["ambiguous"]]),
            }
    return report


def summarise(rows: list[dict], reproduction: dict) -> dict:
    """Counts per run: answers, reproduction, claim outcomes, judge calls."""
    out = {}
    for run in sorted({r["run"] for r in rows}):
        rs = [r for r in rows if r["run"] == run]
        repro = [reproduction[r["key"]] for r in rs]
        verd = [v["verdicts"] for v in repro if v["verdicts"].get("comparable")]
        kept = [r for r in rs if r["context_reproduced"]]
        out[run] = {
            "answers": len(rs),
            "context_reproduced": sum(v["reproduced"] for v in repro),
            "citations_all_found": sum(not v["missing_citations"] for v in repro),
            "same_sources": sum(v["same_sources"] for v in repro),
            "verdict_agreement": (
                {"claims": sum(v["replay_claims"] for v in verd), "agree": sum(v["agree"] for v in verd),
                 "same_claim_count": sum(v["live_claims"] == v["replay_claims"] for v in verd)}
                if verd else None
            ),
            "verdict_agreement_reproduced": (
                {"claims": sum(v["verdicts"]["replay_claims"] for v in repro
                               if v["reproduced"] and v["verdicts"].get("comparable")),
                 "agree": sum(v["verdicts"]["agree"] for v in repro
                              if v["reproduced"] and v["verdicts"].get("comparable"))}
                if verd else None
            ),
            "claim_outcomes_all": dict(Counter(c["outcome"] for r in rs for c in r["claims"])),
            "claim_outcomes_reproduced": dict(Counter(c["outcome"] for r in kept for c in r["claims"])),
            "flag_methods_reproduced": dict(Counter(
                c["method"] for r in kept for c in r["claims"] if c["outcome"] == "flag")),
            "judge_called_reproduced": sum(r["judge_called"] for r in kept),
            "answers_reproduced": len(kept),
        }
    return out


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def render(report: dict) -> str:
    lines = [f"# Answer replay ({report['generated_at']}, retrieval as of {report['as_of']}, "
             f"NLI {report['nli']})", ""]
    for run, s in report["summary"].items():
        lines.append(f"## {run}: {s['answers']} answers, context reproduced {s['context_reproduced']} "
                     f"(citations found {s['citations_all_found']}, same sources {s['same_sources']})")
        if s["verdict_agreement"]:
            va, vr = s["verdict_agreement"], s["verdict_agreement_reproduced"]
            lines.append(f"- live verdict class agreement: {va['agree']}/{va['claims']} claims "
                         f"(reproduced contexts: {vr['agree']}/{vr['claims']}); same claim count "
                         f"{va['same_claim_count']} answers")
        lines.append(f"- claim outcomes (reproduced answers): {s['claim_outcomes_reproduced']}; "
                     f"flags by stage {s['flag_methods_reproduced']}; judge called on "
                     f"{s['judge_called_reproduced']}/{s['answers_reproduced']} answers")
        lines.append("")
    for which, r in (report.get("rates") or {}).items():
        lines.append(f"## Rates on {which} ({r['questions']} questions, labelled {r['labelled']})")
        for k, v in r.items():
            if isinstance(v, dict) and "value" in v:
                val = "—" if v["value"] is None else f"{100 * v['value']:.1f}%"
                ci = "" if not v["ci95"] else f" [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]"
                lines.append(f"- {k}: {val}{ci}")
        lines.append(f"- causes: {r['causes']}")
        un = r["unambiguous"]
        lines.append("- without ambiguous labels: " + ", ".join(
            f"{k} {_pct(un[k]['value'])}" for k in ("false_alarm", "false_pass", "judge_share")))
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build-answers", action="store_true",
                        help="extract eval/answer_replay_answers.json from git history and exit")
    parser.add_argument("--answers", default=str(ANSWERS_PATH))
    parser.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    parser.add_argument("--questions", nargs="+", help="restrict to these question ids")
    parser.add_argument("--as-of", choices=("pm-run", "current"), default="pm-run",
                        help="retrieval settings: as when the pm run was recorded, or today's")
    parser.add_argument("--contexts", help="reuse a contexts.json from an earlier replay")
    parser.add_argument("--score-cache", help="entailment score cache (read and extended)")
    parser.add_argument("--no-nli", action="store_true", help="numeric grounding only")
    parser.add_argument("--labels", help="hand labels (eval/answer_replay_labels.json)")
    parser.add_argument("--split", nargs="+", default=["dev"], choices=("dev", "test", "all"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "eval" / "results" / "answer_replay"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    if not args.verbose:
        logging.disable(logging.INFO)
    if args.build_answers:
        data = build_answers()
        print(f"[replay] wrote {ANSWERS_PATH} ({len(data['records'])} records)", file=sys.stderr)
        return 0
    try:
        report = asyncio.run(run(args))
    except Exception as e:
        logging.disable(logging.NOTSET)
        logging.getLogger(__name__).exception("Answer replay failed to run")
        print(f"[replay] FAILED to run: {e}", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir)
    (output_dir / "replay.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    text = render(report)
    (output_dir / "replay.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
