# Status

Where the project stands, as of round 17 (2026-11-23). Why each statement holds, with
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
| Offline test suite, lint, fuzz | 2,878 passed + 3 expected failures (1 Linux-only test skipped on Windows); lint clean; fuzz 72/72 | `pytest`, `ruff check .`, `HYPOTHESIS_PROFILE=fuzz pytest tests/test_fuzz_properties.py` |
| Benchmarks unchanged by a change | verification, FactCG, golden retrieval (PASS), answerability: field-by-field diff against the previous round | `python -m eval.run_verification_eval …` and siblings (Decision 104 lists the gate set) |
| Models load at pinned commits, also offline | 7/7, and 7/7 with the network off (Decisions 84, 100) | `pytest -m models` (with `HF_HUB_OFFLINE=1`) |
| A real LLM end to end | sample deal, local qwen2.5:14b, deployed CPU profile, API and console: 6 of 6 answerable questions answered with the expected figures and cited once the guessed category stopped filtering retrieval (Decisions 106, 109); out-of-scope refused; an outage reported as one | the questions, settings and checks are in Decision 106; no hosted LLM was called |
| A large upload does not stall questions | a 4.4 MB 10-K ingested while questions run: the question during its parse 0.56 s in retrieval (33.3 s before), the PDF parsed in a child process with identical output on 28 filings (Decision 110) | the contention and parity scripts named in Decision 110 |
| Backups restore | restore drill and the re-ingestion runbook on Qdrant servers, matching a fresh ingest once table columns kept their order (Decisions 105, 108) | `scripts/qdrant_backup.py` (DEPLOYMENT.md, "Backups and restore") |
| Images build and run | amd64 and arm64 (emulated): API (3.09 / 3.04 GB) and Streamlit (192 / 184 MB) images build, pin and smoke-test; amd64 also serves (Decision 104) | Dockerfile.hf, Dockerfile.streamlit |
| Dependencies have no unaccepted advisory | weekly Dependency audit workflow; 9 accepted, each unreachable, review by 2027-02-28 (Decision 104) | `scripts/check_vulnerabilities.py` |
| CI | every push: ruff, offline pytest, verification gate, web typecheck/lint/test/build, golden retrieval eval | GitHub Actions |

## Owner decisions

**Made**

- LLM provider: free tiers for now (2026-11-22, Decision 107). Blocker B1 therefore
  stands: no confidential documents on this deployment.

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
- A notice on the upload panel that uploaded text goes to a third-party AI service on
  a free tier (wording is the owner's; Decision 107).
- Dependabot: turn on alerts and security updates (Settings → Code security), and
  rename `.github/dependabot.example.yml` to `dependabot.yml` for version updates.
- How a "the documents do not contain X" answer is presented: today it is labelled
  `answered`, and the judge can confirm its absence statements against the retrieved
  passages only (Decision 109). Options: a distinct outcome ("not found") shown like a
  refusal, or keeping the label and badging absence statements "not confirmed for the
  whole data room". Either changes what users see on every such answer.

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

1. A "the documents do not contain X" answer is labelled `answered`, and its absence
   statements can be confirmed against the retrieved passages only. Its known cause,
   the guessed category filtering retrieval, is removed (Decision 109); how such an
   answer is presented is an owner decision above.
2. Indexes already written to a Qdrant server keep the columns of header-named
   table rows shuffled (about one row in four in the rehearsal batch) until each
   document is re-ingested from its file; new ingests store the order (Decision
   108), and the re-ingestion report names the affected documents (Decision 111).
3. The local model refuses a synthesis context that fills its budget (64,000
   characters, ~16.7k tokens, over the default 16,384 window) instead of answering
   from fewer chunks (Decision 106).
4. Dependency upgrades held back: transformers 5 with sentence-transformers ≥ 5.6,
   and torch 2.13, fix the 8 accepted model-library advisories and move every model
   score; a measured round of their own. litellm ≥ 1.95 needs a live provider
   check. The npm lint toolchain's fix, after 2026-11-30 (Decision 104).
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
    NLI validation ~40% longer on the development machine (Decision 110). Other formats
    (DOCX, XLSX, PPTX) still parse in the API's process; only PDF parsing was measured
    as a stall.

## How to check a change

```bash
.venv/Scripts/python scripts/update_constraints.py --check    # the lock matches the environment
pytest && ruff check .                                         # offline suite and lint
pytest -m models                                               # pinned models, also with HF_HUB_OFFLINE=1
python -m eval.run_retrieval_eval --baseline eval/baseline.json --tolerance 0.03
```

Then diff the benchmark outputs against the previous round's, field by field, and
record what changed in DECISIONS_LOG.md.
