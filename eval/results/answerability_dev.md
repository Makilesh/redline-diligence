# Answerability gate — dev set, 2026-11-14

eval\answerability_dev.json: 27 answerable (evidence in context), 28 unanswerable, 2 answerable-but-retrieval-missed, 1 excluded (debatable label). Adopted: `Does `passage` state the information needed to answer `question`?` over the top 3 reranked chunks, veto below 0.35. LLM calls: 0.

| label | gate | admitted | ambiguous → LLM | refused |
|---|---|---|---|---|
| answerable | heuristic | 25 | 2 | 0 |
| answerable | + Laya | 25 | 2 | 0 |
| unanswerable | heuristic | 19 | 7 | 2 |
| unanswerable | + Laya | 12 | 4 | 12 |
| retrieval_miss | heuristic | 2 | 0 | 0 |
| retrieval_miss | + Laya | 1 | 0 | 1 |

## P1: `Does `passage` state the information needed to answer `question`?` — AUC 0.801

| veto below | false vetoes (answerable) | unanswerable caught | of which heuristic-admitted | of which ambiguous | retrieval misses vetoed |
|---|---|---|---|---|---|
| 0.25 | 0/27 | 4/28 | 2/19 | 2 | 1 |
| 0.3 | 0/27 | 6/28 | 3/19 | 3 | 1 |
| 0.325 | 0/27 | 10/28 | 6/19 | 3 | 1 |
| 0.35 | 0/27 | 11/28 | 7/19 | 3 | 1 |
| 0.375 | 0/27 | 11/28 | 7/19 | 3 | 1 |
| 0.4 | 0/27 | 12/28 | 8/19 | 3 | 1 |
| 0.425 | 1/27 | 13/28 | 8/19 | 4 | 1 |
| 0.45 | 2/27 | 15/28 | 10/19 | 4 | 1 |

Laya latency per assessment (cuda): p50 50 ms, p95 63 ms (n=57).

Lowest-scoring answerable questions: dev_a38 0.4087, dev_a40 0.4328, dev_a22 0.4656
