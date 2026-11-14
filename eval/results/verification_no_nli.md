# Verification benchmark — 2026-11-14

`python -m eval.run_verification_eval --no-nli --update-baseline eval/verification_baseline.json`

NLI: off · device: n/a · LLM calls: 0 · 13.5s

## dev — 276 claims (109 supported, 167 not)

| | rate | count |
|---|---|---|
| **False pass** (wrong claim passed or never checked) | **3.0%** | 5/167 |
| Wrong claim caught locally (warn/fail) | 23.4% | |
| Wrong claim left to the LLM judge | 73.7% | |
| **False alarm** (supported claim flagged) | **0.0%** | 0/109 |
| Supported claim passed locally | 48.6% | |
| Supported claim left to the LLM judge | 51.4% | |
| Judge load (all claims) | 64.9% | |

| label / kind | n | pass | judge | warn | fail | skipped |
|---|---|---|---|---|---|---|
| contradicted/absence | 6 | 0 | 6 | 0 | 0 | 0 |
| contradicted/derived | 7 | 0 | 0 | 2 | 5 | 0 |
| contradicted/figure | 24 | 0 | 1 | 0 | 23 | 0 |
| contradicted/misattributed | 33 | 2 | 31 | 0 | 0 | 0 |
| contradicted/negation | 46 | 0 | 46 | 0 | 0 | 0 |
| contradicted/party | 20 | 3 | 17 | 0 | 0 | 0 |
| contradicted/period | 8 | 0 | 8 | 0 | 0 | 0 |
| contradicted/unit | 3 | 0 | 0 | 0 | 3 | 0 |
| supported/absence | 6 | 0 | 6 | 0 | 0 | 0 |
| supported/derived | 14 | 9 | 5 | 0 | 0 | 0 |
| supported/paraphrase | 76 | 35 | 41 | 0 | 0 | 0 |
| supported/verbatim | 13 | 9 | 4 | 0 | 0 | 0 |
| unsupported/absent | 11 | 0 | 11 | 0 | 0 | 0 |
| unsupported/absent_figure | 9 | 0 | 3 | 0 | 6 | 0 |

## test — 266 claims (101 supported, 165 not)

| | rate | count |
|---|---|---|
| **False pass** (wrong claim passed or never checked) | **6.7%** | 11/165 |
| Wrong claim caught locally (warn/fail) | 21.8% | |
| Wrong claim left to the LLM judge | 71.5% | |
| **False alarm** (supported claim flagged) | **0.0%** | 0/101 |
| Supported claim passed locally | 45.5% | |
| Supported claim left to the LLM judge | 54.5% | |
| Judge load (all claims) | 65.0% | |

| label / kind | n | pass | judge | warn | fail | skipped |
|---|---|---|---|---|---|---|
| contradicted/absence | 6 | 0 | 6 | 0 | 0 | 0 |
| contradicted/derived | 5 | 1 | 1 | 2 | 1 | 0 |
| contradicted/figure | 27 | 0 | 5 | 0 | 22 | 0 |
| contradicted/misattributed | 40 | 4 | 36 | 0 | 0 | 0 |
| contradicted/negation | 40 | 3 | 37 | 0 | 0 | 0 |
| contradicted/party | 15 | 3 | 12 | 0 | 0 | 0 |
| contradicted/period | 4 | 0 | 4 | 0 | 0 | 0 |
| contradicted/unit | 8 | 0 | 2 | 0 | 6 | 0 |
| supported/absence | 6 | 0 | 6 | 0 | 0 | 0 |
| supported/derived | 16 | 11 | 5 | 0 | 0 | 0 |
| supported/paraphrase | 59 | 17 | 42 | 0 | 0 | 0 |
| supported/verbatim | 20 | 18 | 2 | 0 | 0 | 0 |
| unsupported/absent | 13 | 0 | 13 | 0 | 0 | 0 |
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

