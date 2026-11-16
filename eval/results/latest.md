# Retrieval eval — 2026-11-15

Golden set: 35 answerable + 6 control questions · corpus: 9 documents, 111 chunks (111 retrievable) · device: cuda (NVIDIA GeForce RTX 5070 Ti Laptop GPU) · LLM calls: 0 · fact-coverage ceiling (facts present in any retrievable chunk): 99.0%

Percentages. fact_cov = share of expected facts present in the top-k chunk texts (any file); ctx = the final context the synthesizer receives (reranked + parent/sibling expansion, formatted by the synthesizer). recall/MRR/nDCG use substring-derived chunk labels — see eval/README.md.

| ablation | fact_cov@5 | fact_cov@10 | fact_cov@context | recall@5 | recall@10 | mrr@10 | ndcg@10 | source_hit@5 | source_hit@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense-only | 77.5 | 88.8 | — | 52.8 | 69.8 | 60.1 | 59.8 | 94.3 | 98.6 | 20 | 23 |
| sparse-only (BM25) | 91.0 | 95.2 | — | 65.4 | 78.0 | 68.7 | 68.9 | 97.1 | 97.1 | 20 | 23 |
| hybrid RRF (no rerank) | 89.8 | 97.1 | — | 68.6 | 80.1 | 69.0 | 69.6 | 98.6 | 98.6 | 21 | 24 |
| production (hybrid + rerank) | 93.3 | 94.3 | 97.1 | 72.4 | 78.7 | 77.2 | 73.8 | 97.1 | 98.6 | 168 | 205 |
| production + decomposition | 94.3 | 95.2 | 98.1 | 75.1 | 81.2 | 77.7 | 75.6 | 98.6 | 98.6 | 194 | 553 |
| *whole corpus in the prompt (reference)* | 99.0 | 99.0 | 99.0 | — | — | — | — | — | — | — | — |

Reference row: the retrievable corpus is 12,643 tokens (58,432 characters in 111 chunks) and fits in one prompt, so handing over everything scores the fact-coverage ceiling. On a corpus this small, fact coverage cannot separate a retrieval design from context stuffing; ranking quality (recall/MRR/nDCG at a fixed depth) and the refusal gate can.

## Uncertainty

95% percentile-bootstrap intervals over the 35 answerable questions (10,000 resamples, fixed seed). A difference between two rows of the table above means something only if its paired interval below excludes zero.

| ablation | fact_cov@10 | fact_cov@context | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|
| dense-only | 88.8 [79.0, 97.1] | — | 69.8 [58.4, 81.1] | 60.1 [47.9, 72.2] | 59.8 [49.2, 70.1] |
| sparse-only (BM25) | 95.2 [89.5, 100.0] | — | 78.0 [67.7, 87.4] | 68.7 [56.8, 80.6] | 68.9 [59.1, 78.5] |
| hybrid RRF (no rerank) | 97.1 [92.4, 100.0] | — | 80.1 [69.8, 89.4] | 69.0 [57.5, 80.1] | 69.6 [60.6, 78.1] |
| production (hybrid + rerank) | 94.3 [87.6, 99.0] | 97.1 [92.4, 100.0] | 78.7 [68.3, 88.3] | 77.2 [66.7, 87.1] | 73.8 [65.2, 81.5] |
| production + decomposition | 95.2 [89.5, 100.0] | 98.1 [94.3, 100.0] | 81.2 [71.7, 89.9] | 77.7 [67.8, 87.2] | 75.6 [68.1, 82.9] |

Paired per-question effect of each stage (same questions, so each is its own control):

| comparison | metric | Δ points | 95% CI | improved | regressed | clears zero |
|---|---|---|---|---|---|---|
| dense-only → hybrid RRF (no rerank) | fact_cov@10 | +8.3 | [+0.7, +17.6] | 4 | 0 | yes |
| dense-only → hybrid RRF (no rerank) | recall@10 | +10.3 | [+1.0, +20.7] | 7 | 3 | yes |
| dense-only → hybrid RRF (no rerank) | mrr@10 | +8.9 | [+1.8, +16.6] | 11 | 3 | yes |
| dense-only → hybrid RRF (no rerank) | ndcg@10 | +9.8 | [+4.6, +15.6] | 19 | 5 | yes |
| hybrid RRF (no rerank) → production (hybrid + rerank) | fact_cov@10 | -2.9 | [-7.6, +0.0] | 0 | 2 | no — within noise |
| hybrid RRF (no rerank) → production (hybrid + rerank) | recall@10 | -1.4 | [-6.9, +4.0] | 3 | 4 | no — within noise |
| hybrid RRF (no rerank) → production (hybrid + rerank) | mrr@10 | +8.2 | [-0.8, +17.6] | 10 | 3 | no — within noise |
| hybrid RRF (no rerank) → production (hybrid + rerank) | ndcg@10 | +4.1 | [-2.2, +10.7] | 15 | 13 | no — within noise |
| production (hybrid + rerank) → production + decomposition | fact_cov@10 | +1.0 | [+0.0, +2.9] | 1 | 0 | no — within noise |
| production (hybrid + rerank) → production + decomposition | fact_cov@context | +1.0 | [+0.0, +2.9] | 1 | 0 | no — within noise |
| production (hybrid + rerank) → production + decomposition | recall@10 | +2.6 | [+0.0, +6.1] | 3 | 0 | no — within noise |
| production (hybrid + rerank) → production + decomposition | mrr@10 | +0.5 | [-7.6, +8.6] | 2 | 2 | no — within noise |
| production (hybrid + rerank) → production + decomposition | ndcg@10 | +1.9 | [-3.1, +7.1] | 5 | 3 | no — within noise |

## production (hybrid + rerank) — by query type

| type | n | fact_cov@5 | fact_cov@10 | fact_cov@context | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|---|
| comparative | 5 | 100.0 | 100.0 | 100.0 | 83.3 | 76.7 | 77.6 |
| financial | 8 | 100.0 | 100.0 | 100.0 | 100.0 | 64.6 | 74.2 |
| legal | 8 | 100.0 | 100.0 | 100.0 | 83.5 | 93.8 | 85.7 |
| multi_hop | 10 | 83.3 | 86.7 | 90.0 | 67.3 | 80.0 | 70.3 |
| summary | 4 | 83.3 | 83.3 | 100.0 | 48.8 | 63.3 | 52.9 |
| **all** | 35 | 93.3 | 94.3 | 97.1 | 78.7 | 77.2 | 73.8 |

## production + decomposition — by query type

| type | n | fact_cov@5 | fact_cov@10 | fact_cov@context | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|---|
| comparative | 5 (5 dec.) | 100.0 | 100.0 | 100.0 | 83.3 | 63.3 | 67.7 |
| financial | 8 (2 dec.) | 100.0 | 100.0 | 100.0 | 100.0 | 70.8 | 78.9 |
| legal | 8 (2 dec.) | 100.0 | 100.0 | 100.0 | 85.3 | 85.4 | 83.8 |
| multi_hop | 10 (9 dec.) | 86.7 | 90.0 | 93.3 | 74.8 | 90.0 | 79.5 |
| summary | 4 (0 dec.) | 83.3 | 83.3 | 100.0 | 48.8 | 63.3 | 52.9 |
| **all** | 35 (18 dec.) | 94.3 | 95.2 | 98.1 | 81.2 | 77.7 | 75.6 |

Decomposition, per question on fact_coverage@context: 1 improved, 0 regressed, 34 unchanged (18 of the 35 answerable questions were decomposed by Agent 1).

Changed: mh_05 66.7→100.0

## Refusal gate (Quality Assessor heuristic, no LLM)

Decided by the graph's own edge (route_after_quality_check) on the heuristic assessment. *refused* = not sent to synthesis on this retrieval; in the full pipeline the rewrite loop runs next and refusal follows only if it fails too.

| ablation | group | admitted | ambiguous → LLM | refused |
|---|---|---|---|---|
| production (hybrid + rerank) | controls | 3 | 1 | 2 |
| production (hybrid + rerank) | answerable | 33 | 1 | 1 |
| production + decomposition | controls | 3 | 1 | 2 |
| production + decomposition | answerable | 33 | 1 | 1 |

Controls (production): ctrl_01 max=0.0261 refused, ctrl_02 max=0.5654 admitted, ctrl_03 max=0.0518 llm_fallback, ctrl_04 max=0.345 admitted, ctrl_05 max=0.9586 admitted, ctrl_06 max=0.0378 refused

## Refusal gate — heuristic vs heuristic + Laya answerability

Laya can only take admission away (veto an admitted context, or refuse in the ambiguous band without the LLM); see src/decisions/answerability.py. Thresholds were chosen on eval/answerability_dev.json, not on this set.

| ablation | group | gate | admitted | ambiguous → LLM | refused |
|---|---|---|---|---|---|
| production (hybrid + rerank) | controls | heuristic | 3 | 1 | 2 |
| production (hybrid + rerank) | controls | + Laya | 2 | 0 | 4 |
| production (hybrid + rerank) | answerable | heuristic | 33 | 1 | 1 |
| production (hybrid + rerank) | answerable | + Laya | 33 | 1 | 1 |
| production + decomposition | controls | heuristic | 3 | 1 | 2 |
| production + decomposition | controls | + Laya | 2 | 0 | 4 |
| production + decomposition | answerable | heuristic | 33 | 1 | 1 |
| production + decomposition | answerable | + Laya | 33 | 1 | 1 |

Laya latency (production (hybrid + rerank), device cuda): p50 58 ms, p95 77 ms over 38 assessments.
Laya latency (production + decomposition, device cuda): p50 74 ms, p95 265 ms over 39 assessments.

Changed by Laya: ctrl_02 (production) admitted→refused P=0.2941, ctrl_02 (production_decomp) admitted→refused P=0.2941, ctrl_03 (production) llm_fallback→refused P=0.1803, ctrl_03 (production_decomp) llm_fallback→refused P=0.1803

Controls, Laya answerability (production): ctrl_01 P=None, ctrl_02 P=0.2941, ctrl_03 P=0.1803, ctrl_04 P=0.5206, ctrl_05 P=0.4181, ctrl_06 P=None

Baseline check (eval/baseline.json, tolerance 2.0pp): **PASS**

