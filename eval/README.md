# Retrieval eval

Measures retrieval alone — no synthesis, so no model choice or quota state can move the numbers — on the 41-question golden set (`tests/golden_qa_set.json`: 35 answerable, 6 unanswerable controls).

**What runs.** `data/sample_deal/*.txt` is indexed into an in-memory Qdrant through `index_document` (bge-m3 dense + BM25 sparse, the same path as `POST /ingest`). Each question then goes through five configurations, all built from production code: `dense`, `sparse` and `hybrid_rrf` (the lists from `hybrid_search` and `reciprocal_rank_fusion`), `production` (`retrieval_executor_node`: RRF, then the bge-reranker-v2-m3 cross-encoder, then the threshold, then parent/sibling expansion) and `production_decomp` (the same node given Agent 1's sub-questions). The retrieval config comes from the golden `query_type`, with no metadata filters. Agent 1's own classification and category guess are not measured here.

**Metrics** (answerable questions; means overall and per `query_type`):
- `fact_coverage@k`: share of a question's expected facts found anywhere in the top-k chunk texts. The README's decomposition numbers use this metric. `@context` measures the final context the synthesizer receives, rendered by the synthesizer's own `_format_context_for_synthesis` with full parent sections, one copy per parent, up to its character budget. Parent sections are about 2k tokens, so this number sits well above `@10`.
- `recall@k`, `mrr@10`, `ndcg@10` (linear gain) and `source_hit@k` use derived labels. A chunk counts as relevant if it comes from a file matching an expected `source_pattern` **and** contains at least one expected fact. Its grade is the number of distinct facts it contains. Chunks that production can never return (PII-flagged, superseded) are excluded from the labels.
- Refusal gate: for every question, where the graph's own edge (`route_after_quality_check`) sends the heuristic assessment of the reranked context: *admitted* (to synthesis), *refused* (not to synthesis on this retrieval — in the full pipeline the rewrite loop runs next), or *ambiguous*, where the node would ask an LLM. The harness calls the edge rather than reading the assessor's `force_refusal`, because the two are separate decisions: the edge also applies the per-type precision and completeness thresholds. For the controls, admitting the context is the failure. Latency is reported as p50/p95 per pass.

**Uncertainty.** 35 questions is a small sample: one question flipping moves a mean by up to 2.9 points. The summary therefore prints a 95% bootstrap interval for every overall mean (questions resampled, 10,000 resamples, fixed seed) and, for each stage (dense → hybrid, hybrid → reranked, production → decomposition), the paired per-question difference with its interval and the number of questions improved and regressed. A difference whose paired interval includes zero is reported as "within noise". `--compare <previous retrieval_*.json>` adds the same paired table between a previous run and this one, which is how a code change should be reported.

**Distractor mode** (`--distractors N`). The sample data room never makes retrieval choose among more than ~110 chunks. With `--distractors N` the first N contracts of CUAD v1 (510 commercial contracts, The Atticus Project, CC BY 4.0; Hugging Face dataset `theatticusproject/cuad` at a pinned revision, taken in sorted file-name order) are indexed into the *same deal* through `index_document`, so the golden questions have to be answered out of thousands of chunks of real contract text. The files are downloaded on first use into `~/.cache/redline-eval` (override with `REDLINE_EVAL_CACHE`) and are never committed. A distractor chunk is never relevant and its text never counts towards fact coverage (it is blanked, at unchanged length, before matching — a contract can contain "$85" by coincidence), but it competes for every rank and for the context budget; the summary reports how many reach the final top 10. Laya's ingest decisions are skipped for these documents. The regression baseline is defined on the sample data room alone, so `--baseline` and `--update-baseline` are refused together with `--distractors`; use `--compare` between two distractor runs.

**Reference row.** The sample corpus is about 13K tokens and fits in one prompt, so the last row of the summary table shows what handing over the whole corpus would score on fact coverage (the ceiling). Fact coverage cannot separate a retrieval design from that on this corpus; ranking metrics at a fixed depth and the refusal gate can.

**Matching.** Matching is case-, whitespace- and markdown-insensitive. `$` before a digit and thousands separators are ignored, so `$452.8` matches a table cell `452.8`. Numbers match only at number boundaries, so `$85` does not match `$850`, but `$47` does match `$47.00`. A fact that is a list matches if any one of its variants appears.

**Limits.** The labels come from substring matches, not human judgement. A chunk that says the same thing in other words is scored as a miss. A short fact such as `12` or `peg` can match by accident. The corpus is 9 synthetic documents, so absolute numbers are optimistic compared with a real data room, and the comparison between ablations is the meaningful part. Local-mode Qdrant searches exactly, while the server uses HNSW with int8 quantization and rescoring.

**Run** (GPU if available). Measured: about 2 minutes on an RTX 5070 Ti, about 15 minutes on a 24-thread CPU, and slower on a 4-vCPU CI runner:

```bash
python -m eval.run_retrieval_eval                                 # all ablations, k=5,10
python -m eval.run_retrieval_eval --ablations production production_decomp --k 5 10
python -m eval.run_retrieval_eval --baseline eval/baseline.json   # exit 1 on regression
python -m eval.run_retrieval_eval --compare eval/results/retrieval_<date>.json   # paired before/after
```

The run writes `eval/results/retrieval_<date>.json` with full per-question detail and `eval/results/latest.md` with the summary.

**Zero LLM calls.** Sub-questions are read from `eval/sub_questions.json`. That file was produced once by `python -m eval.generate_sub_questions`, which runs the real Agent 1 node on the lite agent ladder, one call per question, and records its provenance. Regenerate it only when Agent 1's prompt or the golden set changes. Entries whose query text no longer matches are ignored. Without the file, decomposition is reported as not measured.

**Baseline.** `eval/baseline.json` holds the overall gated metrics for `production` and `production_decomp`. With `--baseline`, a drop larger than the tolerance (default 2pp, `--tolerance` to override) exits 1. After an intentional retrieval change, rerun with `--update-baseline` and commit the new file alongside the change. The CI job (`retrieval-eval` in `.github/workflows/ci.yml`: manual and weekly) runs on CPU against a baseline recorded on a GPU, so it uses a 3pp tolerance.

## Laya decisions on the query path

Two local Laya checks (see `src/decisions/`) are measured here, both LLM-free and deterministic.

**Answerability gate** (`src/decisions/answerability.py`, used by the Quality Assessor). The reranker heuristic measures relevance, so an on-topic question whose figure is absent (FY2024 capex when only FY2023 is in the data room) passes it. Laya asks, for the question and each sub-question against the top 3 reranked chunks, "Does `passage` state the information needed to answer `question`?", and vetoes admission only when no facet scores 0.35 or more.

- The phrasing, the top-3 cut and the threshold come from the **dev set** `eval/answerability_dev.json`: 29 answerable and 29 unanswerable questions, mostly minimal pairs, written for this purpose with different wording from the golden set. Each answerable entry carries a verbatim corpus excerpt. Each unanswerable entry carries regexes that must match nowhere in the corpus. `tests/test_decisions_query.py` re-checks both on every CI run. `python -m eval.run_answerability_eval --compare-phrasings` writes `results/answerability_dev.md` with the threshold sweep. An answerable question whose evidence retrieval failed to surface is reported as `retrieval_miss`, not held against the gate.
- The **golden set is the held-out test set**. `run_retrieval_eval.py` reports the heuristic gate and the Laya-augmented gate side by side on the same context, per ablation (`--no-laya` skips it). The decision reported is the node's own `combine_with_answerability`, so the harness measures production logic.

**Public query guard** (`src/decisions/query_guard.py`, used by `/query` and `/query/stream` for non-admin callers). `python -m eval.run_query_guard_eval` scores `eval/query_guard_set.json`: 124 genuine questions (the golden and dev sets plus 25 casual on-topic variants) against 27 junk prompts (off-topic, jailbreak, injection, and injections appended to a real question). It reports precision/recall at the configured thresholds and a sweep of each one. `results/query_guard.md` has the numbers.

Latency on a small CPU host: set `LAYA_DEVICE=cpu` and pass `--threads 2` to either script. The timing run writes `*_timing.md`.
