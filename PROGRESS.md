# Status

Where the project stands, as of round 18 (2026-11-24). Why each statement holds, with
the evidence, is in [DECISIONS_LOG.md](DECISIONS_LOG.md) under the decision named;
how to deploy and operate it is in [DEPLOYMENT.md](DEPLOYMENT.md). This page replaces
the build-phase tracker, which still read "COMPLETE — 8 git commits" (its history is
in git).

## Verdict

**Not ready for production with confidential data rooms.** Ready for a limited pilot
on non-confidential material, with users told to check every figure against its
citation. What stands between the two is hosting, legal and access decisions
(below), not code.

## What is verified, and how to re-run it

| What | Evidence | Re-run |
|---|---|---|
| Offline test suite, lint, fuzz | 2,968 passed + 3 expected failures (1 Linux-only test skipped on Windows); lint clean; fuzz 72/72 | `pytest`, `ruff check .`, `HYPOTHESIS_PROFILE=fuzz pytest tests/test_fuzz_properties.py` |
| Benchmarks unchanged by a change | verification, FactCG, golden retrieval (PASS), answerability: field-by-field diff against the previous round | `python -m eval.run_verification_eval …` and siblings (Decision 104 lists the gate set) |
| Models load at pinned commits, also offline | 7/7, and 7/7 with the network off (Decisions 84, 100) | `pytest -m models` (with `HF_HUB_OFFLINE=1`) |
| A real LLM end to end | sample deal, local qwen2.5:14b, deployed CPU profile, API and console: 6 of 6 answerable questions answered with the expected figures and cited once the guessed category stopped filtering retrieval (Decisions 106, 109); out-of-scope refused; an outage reported as one | the questions, settings and checks are in Decision 106; no hosted LLM was called |
| A large upload does not stall questions | a 4.4 MB 10-K ingested while questions run: the question during its parse 0.56 s in retrieval (33.3 s before), the PDF parsed in a child process with identical output on 28 filings (Decision 110); a large workbook and contract no longer slow a question's model work either (Decision 114) | the contention and parity scripts named in Decisions 110 and 114 |
| A context too long for the local window is answered | a 10-K's 64,000 characters refused at 19,387 tokens by qwen2.5:14b, cut to fit and answered with the filing's figures, cited, at 16,384 and 8,192 (Decision 113) | the live script named in Decision 113 |
| Backups restore | restore drill and the re-ingestion runbook on Qdrant servers, matching a fresh ingest once table columns kept their order (Decisions 105, 108) | `scripts/qdrant_backup.py` (DEPLOYMENT.md, "Backups and restore") |
| Images build and run | amd64 and arm64 (emulated): API (3.09 / 3.04 GB) and Streamlit (192 / 184 MB) images build, pin and smoke-test; amd64 also serves (Decision 104) | Dockerfile.hf, Dockerfile.streamlit |
| Dependencies have no unaccepted advisory | weekly Dependency audit workflow; 9 accepted, each unreachable, review by 2027-02-28 (Decision 104) | `scripts/check_vulnerabilities.py` |
| CI | every push: ruff, offline pytest, verification gate, web typecheck/lint/test/build, golden retrieval eval | GitHub Actions |

## Owner decisions

**Made**

- LLM provider: free tiers for now (2026-11-22, Decision 107). Blocker B1 therefore
  stands: no confidential documents on this deployment.
- A "the documents do not contain X" answer is its own outcome, `not_found`, shown
  like a refusal (2026-11-24, option (a); Decision 112).
- Upload notice: drafted and shown on both upload panels while a Gemini key is
  configured; the wording is the owner's to change with `UPLOAD_NOTICE` (2026-11-25,
  Decision 116).
- Dependabot: alerts, security updates and version updates on (2026-11-25,
  Decision 116).
- GitGuardian incident 40889490: a false positive (a documented `--api-key` flag
  name); the owner closes it in the GitGuardian dashboard.

**Open** (numbered as in the round-14 report)

1. Licensing, with counsel: PyMuPDF (AGPL; seven options researched), the MS MARCO
   terms behind the CPU reranker, Vercel's Hobby plan.
3. Access model: for a pilot, one client per instance behind an authenticated proxy
   or VPN with the user in the audit log; for production, per-user access to deals.
4. Security options A–H: fields that still carry document text to callers.
5. Re-ingestion: when, and which deals (the runbook now has a server rehearsal,
   Decision 105); keep the original uploads? log failed ingests to the audit file?
6. Retention: purging a deal also removes its audit lines? query text kept out of
   container logs? per-document delete?
7. Capacity: ingestion as a background job (changes the API), rate limits on admin
   requests.
8. Deleting `D:/redline_data/pdf11/fb_int9_copy`, `pdf11/fb_pdf11`, `pdf11/rows_v1`
   (~1.57 GB); keep `int9/fb_main`.
9. Company-name follow-ups: the signature-page name as an identity key; filings that
   print no name label.
11. Base-image pinning by digest (reproducible builds vs security patches; the base
    image's setuptools carries two build-time advisories).

**New**

- Backups (H3): where they are kept off the VM, encryption at rest, schedule,
  retention (the tooling is done: DEPLOYMENT.md, "Backups and restore").

## Production blockers

- **B1** Confidential text would go to the Gemini free tier, whose terms say not to
  send it confidential data (kept by Decision 107).
- **B2** No user accounts or per-client access control; one shared admin key.
- **B3** Licensing undecided.
- **H1** Never run with the production LLM (no hosted call by rule) or on the real ARM
  server; the local-model run (Decision 106) covers the rest of the path.
- **H2** Capacity unmeasured on the ARM host; the free tier allows ~20 requests a day
  per model and key. On the development machine a large upload no longer stalls
  questions (Decision 110); while its chunks are embedded a question shares the CPU.
- **H3** Backup tooling verified; where, encrypted how, and how often are open.
- **H4** Purging a deal leaves its query text in the audit file and container logs.

## Outstanding technical issues

1. A "not found" is recognised from the wording of the answer's opening section
   (Decision 112): a decline worded outside the rule stays `answered`, as every
   decline was before, and a finding phrased as a decline of the evidence is read as
   one (Decision 120 removed the subject-changing cases a review found); a new model's
   phrasing should be checked on its first answers.
2. Indexes already written to a Qdrant server keep the columns of header-named
   table rows shuffled (about one row in four in the rehearsal batch) until each
   document is re-ingested from its file; new ingests store the order (Decision
   108), and the re-ingestion report names the affected documents (Decision 111).
3. When the local model's window does not hold the context, it answers from the
   best-ranked evidence that fits (Decision 113) and says so under the answer, with how
   many passages and how much of their text (Decision 118); the validator still checks
   the answer against everything retrieved.
4. Dependency upgrades held back: transformers 5 with sentence-transformers ≥ 5.6,
   and torch 2.13, fix the 8 accepted model-library advisories and move every model
   score; a measured round of their own. litellm ≥ 1.95 needs a live provider
   check. The npm lint toolchain's fix, after 2026-11-30 (Decision 104); Dependabot
   #61–#66 and #68 pass CI and await the owner's merge (Decision 119).
   The query rewriter's prompt still says "null to remove the category filter",
   though since Decision 109 the category only ranks. Left as it is on purpose: a
   prompt change moves the model's output, nulling the category no longer narrows
   anything, and the change wants a live before/after on recorded rewriter inputs.
5. Injection-screen limits, kept on purpose (Decision 76): an instruction split
   across chunks, an echo split across list items, look-alike characters.
6. Staged ingest (Decision 101): two processes uploading the same bytes at once are
   not serialised; a cut-short upload's hidden points stay until a purge or re-upload.
7. Company-name reading limits (Decision 98).
8. With the network off, the rewrite loop still runs two retrieval passes (~0.4 s).
9. Development machine only: Windows timers tick in 15.6 ms steps; an outage takes
   ~40 s to report there (each refused local connection ~8 s); with
   `CUDA_VISIBLE_DEVICES=""` a CUDA build of torch reports a GPU with no device
   (use `-1`).
10. Ingest embeds a document's chunks in one call on the two-thread pool queries
    share; while it runs, a question's retrieval took 1.3–1.4 s (0.3 s idle) and its
    NLI validation ~40% longer on the development machine (Decision 110). All four
    parsed formats now parse in the child process (Decisions 110, 114), and so is the
    classifier's sample of each (Decision 121; ~0.2 s an upload); the embedding is CPU sharing, to be
    measured on the host (H2).
11. A parser child that died while idle used to stop every later upload until a
    restart; the pool is now replaced (Decision 115). The likely killer in that CI run
    was the test suite itself: a console test left its Streamlit script as the
    process's main, which every parser child ran first (fixed, Decision 117). A child
    that dies after an upload was queued to it still fails that one upload.

## How to check a change

```bash
.venv/Scripts/python scripts/update_constraints.py --check    # the lock matches the environment
pytest && ruff check .                                         # offline suite and lint
pytest -m models                                               # pinned models, also with HF_HUB_OFFLINE=1
python -m eval.run_retrieval_eval --baseline eval/baseline.json --tolerance 0.03
```

Then diff the benchmark outputs against the previous round's, field by field, and
record what changed in DECISIONS_LOG.md.
