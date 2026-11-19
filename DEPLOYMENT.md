# Deployment

A free, end-to-end public deployment: the Next.js frontend on Vercel, and the
FastAPI backend, Qdrant and Postgres together on one Linux VM behind Caddy,
which provides HTTPS automatically.

```
Vercel (web/, free)  ──HTTPS──▶  https://<name>.duckdns.org
                                        │
                           Linux VM (≥ 2 cores / 8GB RAM)
                     Caddy :443 → api:7860 → qdrant, postgres
                          (docker-compose.prod.yml)
```

| Layer | Host | Cost |
|---|---|---|
| Frontend | Vercel Hobby | Free (non-commercial use) |
| Backend + Qdrant + Postgres | Oracle Cloud Always Free ARM (2 OCPU / 12GB), or any VM with ≥ 8GB RAM | Free on Oracle; card verification required at sign-up |
| Hostname + HTTPS | DuckDNS + Caddy (Let's Encrypt) | Free |
| LLM | Gemini API free tier | Per-key daily quota, multiplied by key count |

Oracle only reclaims an Always Free VM as idle when CPU, network *and* memory
all stay under 20% for a week; with the models loaded, memory alone sits well
above that.

---

## Why this shape, and not a smaller one

The backend is the constraint, and it is not close. `bge-m3` alone is 2.27GB
resident and the whole API holds about 5–6GB once the reranker and NLI model are
loaded (the CPU host uses the small `nli-deberta-v3-xsmall`; on a GPU machine the
FactCG grounding checker adds about 0.6GB of GPU memory at peak). The 512MB free tiers at Render, Fly and Koyeb are not near-misses — they
are an order of magnitude short. What fits is a VM with at least 8GB of RAM:
Oracle Cloud's Always Free ARM instance (2 OCPU / 12GB) is the free option, and
co-locating Qdrant and Postgres on it removes two external accounts.

**The reranker had to change.** Measured at two threads on x86 (an emulation of the 2-vCPU host; not measured on ARM), which is what a free VM
provides:

| Reranker | `max_length` | 40 passages | ×4 passes (a decomposed query) |
|---|---|---|---|
| `bge-reranker-v2-m3` (568M) | 1024 | 111.2s | **444.9s** |
| `bge-reranker-v2-m3` (568M) | 512 | 67.8s | 271.2s |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M) | 512 | **3.2s** | **12.9s** |

Seven and a half minutes per question is not a demo. Quantization does not close
that gap — int8 buys 2–4× where ~30× is needed — so the deployed profile uses a
smaller cross-encoder, set through `RERANKER_MODEL`.

Two consequences worth stating plainly:

- **The deployed reranker is English-only.** The data room is English, so this
  costs nothing here, but it is a real capability difference from the local
  configuration.
- **The retrieval numbers in `RESULTS.md` were measured with the default
  reranker.** They do not describe the deployed profile until the eval is re-run
  against it (last step below).
- **Document focus runs 10 deep, not 30.** Legal questions add passages from the
  leading document to the rerank pool (Decision 67). With MiniLM, more than ten
  of them lowered recall on LegalBench-RAG's dev documents (78.6% at 10, 75.3% at
  30), so the MiniLM calibration profile caps the depth at 10, at about +0.3 s per
  legal question on two cores. `DOCUMENT_FOCUS_K` overrides it (0–30) if you run
  a different reranker or host (Decision 71).
- **Backfill document identity on an index built before Decision 86.** A legal
  question that names its agreement ("the Northwind NDA") now focuses on it, by
  the title and parties ingestion reads from each document's opening — with
  MiniLM at depth 10, +1.5 points of final-context recall [+0.5, +2.7] on
  LegalBench-RAG's sealed held-out documents (ranked on a GPU, as before). An
  existing index has neither field and keeps the old behaviour until you run,
  once, `python -m src.vector_db.identity_backfill` (`--dry-run` counts first,
  `--deal <id>` limits it to one deal). It reads the chunks already in Qdrant —
  the files are not needed — and writes exactly what re-ingesting them would; on
  the 714-file benchmark index (49,559 chunks, local mode) it took 135 s, and
  35 s to re-run when most documents were already up to date. The
  choice itself adds no search and no rerank pairs: it is Python over the
  retrieved payloads, about 17 ms per legal question on the development laptop
  (two ARM cores not measured). `NAMED_DOCUMENT_FOCUS=0` turns it off.
- **Re-ingest contract PDFs indexed before Decision 89, then re-run the identity
  backfill.** A PDF whose text sits inside a ruled box — an exhibit printed from
  HTML, a document with page borders — was indexed as flattened one-column table
  cells: no clause boundaries, no parent chunks, out of reading order, and often
  no title or parties for named-document focus (35 of 232 CUAD contracts, more
  than half their text). Uploading the same file again re-indexes it in place:
  its doc_id comes from its bytes, so its old points are replaced, not
  duplicated. The identity read from PDF text changed too (no-break spaces,
  lines wrapped after "a", titles centred over several lines), and
  `python -m src.vector_db.identity_backfill` applies that to every other
  document without the files; on the 714-file benchmark index it changes the
  identity of 6 of its 712 distinct documents, each a correction. The backfill
  cannot repair a boxed PDF's chunks: only re-ingesting it does.
- **MiniLM is still the best fit for two cores, not the best small reranker.**
  Six Apache-2.0 alternatives were measured on the same emulated host
  (Decision 77). Only `ettin-reranker-68m-v1` beat it on LegalBench-RAG's dev
  documents (84.1% against 78.6%, matching bge on a GPU), at 306 ms a pair
  against 74 ms — about 8 s more per legal question. Its own int8 ONNX build
  (the ARM file, timed on x86) is 2.2× faster but still adds 2.5 s per legal
  question at p50, twice the budget, and reorders the top passage of a third of
  queries against full precision (Decision 81). `ettin-reranker-17m-v1`
  is faster than MiniLM (40 ms a pair) but no better. The Ettin models are
  saved in sentence-transformers 5's format and load *silently wrong* (random
  classification head) under this repository's pinned 2.x; do not point
  `RERANKER_MODEL` at one without the conversion described in Decision 77.
- **A distilled MiniLM is measured, not deployed (Decision 87).** MiniLM-L6
  fine-tuned to imitate bge on LegalBench-RAG costs the same 84 ms a pair. On
  held-out contracts at focus 30 it recalls 74.0%, against MiniLM's 63.0% at 10
  and bge's 74.6% (measured before Decision 86). It refuses one more golden
  answerable question and gains nothing on FinanceBench, so it is not the CPU
  host's reranker. Its checkpoint is kept outside the repository and was never
  published.
- **Keep `QDRANT_MODE=local` for development.** The embedded index searches
  exactly but slowly: with FinanceBench's 84 filings (45K chunks) indexed, a
  filing question's retrieval took 9.8 s at p50 in local mode against 0.6 s on
  a Qdrant server, with the same evidence pages retrieved (Decision 79).
- **Uploads index slowly.** bge-m3 on two cores embeds about one chunk every
  1.5–3 s: the 111-chunk sample deal takes about 3 minutes, and a document of a
  few hundred chunks several minutes. Seed the demo corpus before going live.

**The embedding model does not change.** It costs 0.12s per query — it was never
the problem — and swapping it would invalidate every vector already indexed.
That is why no re-indexing is needed when moving between profiles.

---

## 1 · A server with a public IP

Any Ubuntu 22.04/24.04 VM with a public IPv4 address, at least 2 cores, 8GB RAM
and 40GB of disk. On Oracle Cloud: Compute → Instances → Create instance →
Canonical Ubuntu 24.04, shape **VM.Standard.A1.Flex** at 2 OCPU / 12GB (it shows
"Always Free-eligible"), assign a public IPv4 address, and paste your SSH public
key. "Out of host capacity" is common — retry later or in another availability
domain.

A home connection does not work as the server. Mobile and most residential
broadband sit behind carrier-grade NAT and change address, so nothing on the
internet can reach a port on them.

## 2 · Open ports 80 and 443 — in both firewalls

- **Cloud firewall.** On Oracle: Networking → Virtual cloud networks → your VCN →
  Security Lists → Default → Add Ingress Rules: source `0.0.0.0/0`, TCP,
  destination port `80`; repeat for `443`.
- **Host firewall.** Oracle's Ubuntu image rejects everything but SSH in
  iptables, and skipping this is the usual reason a site is unreachable with no
  error anywhere:

  ```bash
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
  sudo netfilter-persistent save
  ```

Do not open 5432, 6333 or 6334. The production compose file publishes nothing
but Caddy's ports.

## 3 · A hostname

Create a subdomain at <https://www.duckdns.org> and set its IP to the **server's**
public IP. DuckDNS pre-fills the address you are browsing from, which is your
own connection, not the server — overwrite it. Check with:

```bash
nslookup <name>.duckdns.org   # must print the VM's public IP
```

Caddy cannot obtain a certificate until this resolves to the server.

## 4 · Docker and the code

```bash
ssh -i ~/.ssh/<key> ubuntu@<server-ip>
curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker ubuntu
exit   # log back in so the docker group applies
git clone https://github.com/Makilesh/redline-diligence.git && cd redline-diligence
```

## 5 · Configure

Copy the backend block of `.env.deploy.example` to `.env` in the repository root
and fill in the secrets:

```bash
cp .env.deploy.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # run twice: POSTGRES_PASSWORD, ADMIN_API_KEY
nano .env   # DOMAIN, POSTGRES_PASSWORD, GEMINI_API_KEYS, ADMIN_API_KEY, CORS_ORIGINS
```

`docker compose` refuses to start while `DOMAIN`, `POSTGRES_PASSWORD`,
`ADMIN_API_KEY` or `CORS_ORIGINS` is unset, rather than falling back to a
default password or an unauthenticated admin.

## 6 · Start the stack

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api caddy
```

The first build downloads CPU PyTorch and bakes the embedding, reranker and NLI
weights into the image — 20–30 minutes on 2 ARM cores. Every Python
dependency installs from a prebuilt wheel on both architectures (the install
runs with `--only-binary :all:`), so nothing is compiled. A build that stops
making progress for more than a few minutes is a fault, not patience — the
network fetches in the Dockerfile are all bounded. The model
layers come before the code, so a later `git pull && up -d --build` that only
changes Python files rebuilds in seconds instead of re-downloading ~5GB.

The API then needs a few more minutes to load the models; Caddy waits for its
health check. When it is up:

```bash
curl https://<name>.duckdns.org/health
curl -H "X-Admin-Key: $ADMIN_API_KEY" https://<name>.duckdns.org/ready
```

`/ready` is 200 once Qdrant and the graph are ready; check that its
`degraded` list is empty and `checkpointer.durable` is true.

## 7 · Seed the demo data room

Writing to the demo deal is an admin operation. From the server:

```bash
export ADMIN_API_KEY=$(grep ^ADMIN_API_KEY .env | cut -d= -f2-)
for f in data/sample_deal/*.txt; do
  curl -sS -H "X-Admin-Key: $ADMIN_API_KEY" \
       -F "deal_id=aurora_vertex_2024" -F "file=@$f" \
       https://<name>.duckdns.org/api/v1/ingest; echo
done
curl -s https://<name>.duckdns.org/api/v1/deals
```

Nine documents, roughly 110 chunks. Re-running the loop is safe: document and
point IDs are derived from content, so identical files replace themselves
instead of duplicating.

## 8 · Frontend on Vercel, and close the CORS loop

1. Import the repository at <https://vercel.com/new> and set **Root Directory**
   to `web`.
2. Set `NEXT_PUBLIC_API_URL` to `https://<name>.duckdns.org` (no trailing
   slash). It is inlined at **build** time: set it, then redeploy — a build made
   without it calls `http://localhost:8000` from every visitor's browser.
3. Vercel deploys the default branch. Set Settings → Git → Production Branch, or
   merge your work into `main`.
4. Put the Vercel URL in `CORS_ORIGINS` in the server's `.env` and recreate the
   API container:
   `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d api`.
   Get CORS wrong and the failure is silent on the server — the browser blocks
   every request and the UI shows an empty deal list.

Finally, set the repository variable `BACKEND_URL` (Settings → Secrets and
variables → Actions → Variables) to `https://<name>.duckdns.org`; the uptime
workflow in `.github/workflows/keepalive.yml` then checks `/health` every 6
hours.

### Updating

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

The named volumes keep the index, the quota counters and the TLS certificates
across rebuilds.

## 9 · Re-run the eval against the deployed profile

The last step, and the one that keeps the project's headline claims honest.

**Pin the synthesis model, or the comparison is worthless.** Recall on this set
is dominated by which model answered, not by retrieval. Measured inside a single
run — one index, identical retrieval — mean recall was:

| Synthesis model | n | mean recall |
|---|---|---|
| `gemini-3.6-flash` | 1 | 100.0% |
| `gemini-3-flash-preview` | 10 | 93.3% |
| `gemini-3.5-flash` | 20 | 65.8% |
| `gemini-2.5-flash` | 4 | 50.0% |

A 50-point spread from model availability alone — and it is not quota that
causes it. With 5 keys each rung has ~95 requests/day, and after two full eval
runs `gemini-3.7-flash` still had 86 left. What actually moves the mix is Google
returning 503 "this model is currently experiencing high demand", which puts the
model into an escalating process-wide cooldown (90s doubling to 720s). One rung
sat out 12 minutes of a 25-minute run that way, and synthesis fell to whatever
was still answering.

So an unpinned local-vs-deployed comparison measures which models Google happened
to be serving that hour, at least as much as it measures the reranker.

`SYNTHESIS_MODEL_PIN` restricts synthesis to one model and never substitutes.
It distinguishes the two ways a pin can be unavailable: a 503 cooldown is
transient, so it waits (up to `SYNTHESIS_PIN_MAX_WAIT_SECONDS`, default 1800) and
retries; exhausted daily quota is terminal, so it raises immediately.

**Which model to pin.** `gemini-3.7-flash` is the top of the ladder and had the
most headroom in practice, so it is the better pin for new measurements. Note
that the historical figures in `RESULTS.md` were produced on `gemini-3.6-flash`
— comparing against those specifically requires pinning to that instead.

**Set these on the API server, not on the eval command.** The harness is a pure
HTTP client — model selection, retrieval and reranking all happen inside the
FastAPI process, so an env var exported for the harness reaches nothing that
acts on it. This is easy to get wrong and fails silently: the run completes and
reports an unpinned mix.

Use the *same* pin for both runs, on a day with quota to spare:

```bash
SYNTHESIS_MODEL_PIN=gemini/gemini-3.7-flash python run_api.py
```

```bash
python tests/run_end_to_end_validation.py
```

Then restart the API with the deployed retrieval profile and repeat:

```bash
SYNTHESIS_MODEL_PIN=gemini/gemini-3.7-flash RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 python run_api.py
```

The report records both the retrieval profile and the pin under **Run
conditions**, so the two runs are self-labelling rather than two undated tables.

`RESULTS.md` and the README currently report one configuration. Once the live
demo runs a different reranker, they describe two, and both should be stated —
the local profile the accuracy was measured on, and the deployed profile a
visitor actually exercises.

The Quality Assessor's relevance floors in `src/agents/quality_assessor.py`
(`RELEVANCE_FLOOR`, `CONFIDENT_REFUSAL_CEILING`, `CONFIDENT_PASS_FLOOR`) were
calibrated against `bge-reranker-v2-m3`'s sigmoid output. The deployment runs
the MiniLM cross-encoder, whose scores are on a comparable but not identical scale,
so if the eval shows more refusals than expected, those floors are the first
place to look — `python -m eval.run_retrieval_eval` reports the gate's decision
per question and runs with `RERANKER_MODEL` set to either model.

---

## What the deployment does not have

**The local Ollama fallback.** The model ladder's last rung is a local
Qwen2.5-14B served by Ollama, which exists to answer when every cloud key is
spent. A 2-core VM cannot run 14B weights alongside the retrieval models at a
usable speed.

The practical effect: on the deployed instance, exhausting the Gemini daily
quota is terminal for that day rather than a quiet downgrade. Since keys
multiply capacity and rotation drains one key per model before stepping down a
rung, the way to push that ceiling out is more keys in `GEMINI_API_KEYS`, not a
different host.

**One key per Google Cloud project.** Gemini's free-tier quota belongs to the
project, not to the key: two keys created in the same project share one
quota, and listing both only makes the engine believe it has twice the
capacity it has. Create each key in its own project. A key that answers 401
(revoked, or created in a project without the API enabled) is retired for the
process; remove it from `GEMINI_API_KEYS` rather than leaving it to be retried
on every restart.

---

## Operating the service

**One worker, by design.** Rate limits, the concurrency gate, the daily query
cap, the sandbox registry and the model-routing state (cooldowns, failure
classification, token-per-minute metering) all live in the API process. A
second worker would double every limit and load a second copy of every model.
`run_api.py` and the app's startup refuse a worker count above one
(`WEB_CONCURRENCY`, `--workers`) unless `ALLOW_MULTIPLE_WORKERS=1` is set, in
which case `/health` reports it as degraded. Scale with `MAX_CONCURRENT_PIPELINES`
and a bigger host, not with processes.

**Health, readiness, metrics.**

| Endpoint | What it says | Fails when |
|---|---|---|
| `GET /health` | Liveness, plus `phase`, `checkpointer` and every active `degraded` mode | Never, while the process serves |
| `GET /ready` | Graph compiled, Qdrant answering, not draining — plus the Qdrant backend, stale-BM25 count, checkpointer durability and each model's load state | Any hard check is false (503) |
| `GET /metrics` | Prometheus text (`?format=json` for JSON with p50/p95): query outcomes, per-node latency, LLM calls and tokens per node, in-flight pipelines | Admin key required unless `METRICS_PUBLIC=1` |

A degraded mode is a fallback the process is running on — in-memory
checkpoints, a model whose warmup failed, chunks indexed under an old BM25
encoding, more than one worker, a wildcard CORS origin. Each is logged once at
ERROR as `DEGRADED_MODE` and listed by reason in `/health` and, with an
explanation, in `/ready`. In production `REQUIRE_DURABLE_CHECKPOINTS=1` turns
the most common one — Postgres unreachable — into a failed start instead.

**Models load from the image, not the Hub.** The image bakes every model into
its Hugging Face cache, and the app loads a cached model from that snapshot
without contacting the Hub. Before Decision 84 it loaded them by Hub id, and
transformers then called the Hub for four of them on every start: with the Hub
unreachable (an outage, a DNS failure, a host without egress) the API could not
embed a question or check an answer. A model that is not cached is still
downloaded. The cache is therefore what runs: to move a model to a newer
revision, rebuild the image (or clear that model from the cache) — an upstream
change is no longer picked up on restart.

**Per-query timings.** Every response carries `node_timings` (one row per
node: `duration_ms`, `llm_calls`, tokens, models) and `llm_calls`; each node
also logs a `Node completed` line. To compare two builds, run the same
questions against each and diff `GET /metrics?format=json` →
`redline_node_duration_seconds`.

**Lifecycle.** A query whose client disconnects is cancelled (no further LLM
calls are made for it). A query that runs past `QUERY_TIMEOUT_SECONDS`
(default 900; longer when `SYNTHESIS_MODEL_PIN` is set) ends with 504, or an
SSE `error` event with `code: "timeout"`. On SIGTERM the process stops
accepting new queries (503, `/ready` 503) and gives in-flight ones
`SHUTDOWN_GRACE_SECONDS` (25) to finish.

**Checkpoints.** A run's checkpoints are deleted when it completes; those of
failed or timed-out runs are kept for `CHECKPOINT_TTL_SECONDS` (24h) for
diagnosis, and a deal's are deleted with the deal.

**Who can read what.** Anonymous callers may query and browse only the deals
in `PUBLIC_DEAL_IDS` (default `aurora_vertex_2024`) and their own sandbox
deals; any other deal answers 404 without the admin key.

## Not yet verified

- **The image has not been built on ARM.** The dependency set was checked
  statically for both architectures (`uv pip compile requirements.txt
  --python-platform aarch64-manylinux_2_28 --python-version 3.11
  --only-binary :all:`, and the same for x86_64): all 162 packages resolve to
  wheels. The previous blocker — fastembed 0.3.x requiring PyStemmer, which has
  no aarch64 wheel and whose source build downloaded with no timeout — is gone
  with fastembed 0.5+. The merged compose file and the Caddyfile were validated
  (`docker compose config`, `caddy validate`).
- **Quality Assessor floor recalibration** for the MiniLM reranker — see step 9.
