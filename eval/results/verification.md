# Verification benchmark — 2026-11-14

`python -m eval.run_verification_eval `

NLI: cross-encoder/nli-deberta-v3-small · device: cuda · LLM calls: 0 · 23.4s

## dev — 276 claims (109 supported, 167 not)

| | rate | count |
|---|---|---|
| **False pass** (wrong claim passed or never checked) | **8.4%** | 14/167 |
| Wrong claim caught locally (warn/fail) | 32.9% | |
| Wrong claim left to the LLM judge | 58.7% | |
| **False alarm** (supported claim flagged) | **2.8%** | 3/109 |
| Supported claim passed locally | 67.0% | |
| Supported claim left to the LLM judge | 30.3% | |
| Judge load (all claims) | 47.5% | |

| label / kind | n | pass | judge | warn | fail | skipped |
|---|---|---|---|---|---|---|
| contradicted/absence | 6 | 0 | 6 | 0 | 0 | 0 |
| contradicted/derived | 7 | 0 | 0 | 2 | 5 | 0 |
| contradicted/figure | 24 | 0 | 1 | 0 | 23 | 0 |
| contradicted/misattributed | 33 | 2 | 31 | 0 | 0 | 0 |
| contradicted/negation | 46 | 4 | 39 | 3 | 0 | 0 |
| contradicted/party | 20 | 6 | 11 | 3 | 0 | 0 |
| contradicted/period | 8 | 2 | 6 | 0 | 0 | 0 |
| contradicted/unit | 3 | 0 | 0 | 0 | 3 | 0 |
| supported/absence | 6 | 0 | 6 | 0 | 0 | 0 |
| supported/derived | 14 | 9 | 5 | 0 | 0 | 0 |
| supported/paraphrase | 76 | 55 | 18 | 3 | 0 | 0 |
| supported/verbatim | 13 | 9 | 4 | 0 | 0 | 0 |
| unsupported/absent | 11 | 0 | 1 | 10 | 0 | 0 |
| unsupported/absent_figure | 9 | 0 | 3 | 0 | 6 | 0 |

## test — 266 claims (101 supported, 165 not)

| | rate | count |
|---|---|---|
| **False pass** (wrong claim passed or never checked) | **9.7%** | 16/165 |
| Wrong claim caught locally (warn/fail) | 32.1% | |
| Wrong claim left to the LLM judge | 58.2% | |
| **False alarm** (supported claim flagged) | **5.0%** | 5/101 |
| Supported claim passed locally | 64.4% | |
| Supported claim left to the LLM judge | 30.7% | |
| Judge load (all claims) | 47.7% | |

| label / kind | n | pass | judge | warn | fail | skipped |
|---|---|---|---|---|---|---|
| contradicted/absence | 6 | 0 | 6 | 0 | 0 | 0 |
| contradicted/derived | 5 | 1 | 1 | 2 | 1 | 0 |
| contradicted/figure | 27 | 0 | 5 | 0 | 22 | 0 |
| contradicted/misattributed | 40 | 6 | 34 | 0 | 0 | 0 |
| contradicted/negation | 40 | 5 | 29 | 6 | 0 | 0 |
| contradicted/party | 15 | 4 | 10 | 1 | 0 | 0 |
| contradicted/period | 4 | 0 | 4 | 0 | 0 | 0 |
| contradicted/unit | 8 | 0 | 2 | 0 | 6 | 0 |
| supported/absence | 6 | 0 | 6 | 0 | 0 | 0 |
| supported/derived | 16 | 11 | 5 | 0 | 0 | 0 |
| supported/paraphrase | 59 | 35 | 19 | 5 | 0 | 0 |
| supported/verbatim | 20 | 19 | 1 | 0 | 0 | 0 |
| unsupported/absent | 13 | 0 | 3 | 10 | 0 | 0 |
| unsupported/absent_figure | 7 | 0 | 2 | 0 | 5 | 0 |

## Random figures reported as grounded

60 contexts of 8 random ~8,000-character windows, 50 random figures per shape per context, seed 7. Floor = a context figure of the same kind and scale within rounding.

| fabricated shape | grounded | floor |
|---|---|---|
| `$X.Y million` | 2.2% | 0.7% |
| `$X million` | 9.8% | 3.6% |
| `$X.Y billion` | 2.1% | 0.7% |
| `N%` | 34.9% | 30.1% |
| `N.N%` | 6.4% | 4.3% |
| `N.Nx` | 11.1% | 4.5% |
| `N bps` | 9.5% | 3.4% |

