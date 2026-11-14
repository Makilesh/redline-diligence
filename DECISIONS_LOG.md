# M&A Due Diligence Intelligence Engine — Decisions Log

All architecture decisions that deviated from or resolved ambiguities in p4.md are recorded here.

---

## Decision 1: Qdrant Quantization Config
- **Date**: Session 1
- **Context**: p4.md specifies INT8 quantization but doesn't specify always_ram setting
- **Resolution**: Set `always_ram=True` on ScalarQuantization for fast retrieval. Trade-off: higher memory usage, lower latency.
- **Impact**: collection_manager.py

## Decision 2: BM25 via FastEmbed
- **Date**: Session 2
- **Context**: p4.md says "FastEmbed BM25" but doesn't specify exact class/method
- **Resolution**: Used `fastembed.SparseTextEmbedding(model_name="Qdrant/bm25")` which produces Qdrant-native SparseVector objects directly
- **Impact**: hybrid_search.py

## Decision 3: asyncio.Lock() creation timing
- **Date**: Session 2
- **Context**: p4.md warns "asyncio.Lock() requires running event loop" but doesn't specify where to create rate limiter locks
- **Resolution**: Rate limiters in BudgetTracker are created lazily via `_get_rate_limiter()` on first use, not at class definition time. This prevents the "no running event loop" error.
- **Impact**: budget_tracker.py, rate_limiter.py

## Decision 4: Agent 5 Heuristic vs LLM Split
- **Date**: Session 2
- **Context**: p4.md says "Primary: heuristics (~60% of queries), Fallback: LLM on ambiguity"
- **Resolution**: Implemented three heuristic paths: (1) clearly good (mean_score >= 0.7), (2) clearly bad (mean_score < 0.2), (3) ambiguous → LLM. Thresholds chosen empirically based on typical reranker score distributions.
- **Impact**: quality_assessor.py

## Decision 5: TOCTOU Budget Race Condition
- **Date**: Session 2
- **Context**: p4.md's budget tracker used separate _budget_available() + _increment() calls
- **Resolution**: Replaced with atomic _try_consume() using conditional UPDATE. Single SQL statement: `UPDATE ... SET used_today = used_today + 1 WHERE used_today < limit`. Prevents two concurrent requests from both passing the check and overshooting.
- **Impact**: budget_tracker.py

## Decision 6: gemini model string format
- **Date**: Session 2
- **Context**: p4.md references "gemini-3.5-flash" and "gemini-3.1-flash-lite" but LiteLLM model strings need prefix
- **Resolution**: Used "gemini/gemini-3.5-flash" and "gemini/gemini-3.1-flash-lite" format for LiteLLM routing. Marked with ⚠ VERIFY in code for runtime validation.
- **Impact**: budget_tracker.py

## Decision 7: Rewriter JSON key → State key mapping
- **Date**: Session 2
- **Context**: p4.md's Agent 6 output uses `updated_retrieval_config` and `updated_metadata_filters` but AgentState uses `retrieval_config` and `extracted_filters`
- **Resolution**: query_rewriter_node explicitly maps keys when returning partial state. Documented in prompt template docstring.
- **Impact**: query_rewriter.py

## Decision 8: Docker Compose Ollama connectivity
- **Date**: Session 2
- **Context**: Agents 4 and 8 use local Ollama which runs on the host, not in Docker
- **Resolution**: Set OLLAMA_API_BASE to `http://host.docker.internal:11434` in Docker Compose. This resolves to the host machine on both Docker Desktop (Mac/Windows) and newer Docker Engine (Linux with --add-host).
- **Impact**: docker-compose.yml

## Decision 9: Missing dependencies beyond p4.md requirements.txt
- **Date**: Session 3
- **Context**: p4.md's requirements.txt lists `asyncpg` and `psycopg2-binary` for Postgres but does NOT list `langgraph-checkpoint-postgres` or `psycopg[binary]`. At runtime, `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver` (used in orchestrator.py per p4.md's own code) fails with `ModuleNotFoundError: No module named 'langgraph.checkpoint.postgres'`. The `langgraph` pip package does NOT bundle the postgres checkpoint adapter — it's a separate package. Furthermore, `langgraph-checkpoint-postgres` depends on `psycopg` (v3, async-native), not `psycopg2-binary` (v2, sync-only). The binary wheel (`psycopg[binary]`) is needed on systems without `libpq` development headers.
- **Resolution**: Added `langgraph-checkpoint-postgres>=2.0.0,<3.0.0` and `psycopg[binary]>=3.1.0,<4.0.0` to requirements.txt. Pinned `<3.0.0` to avoid breaking changes with `langgraph>=0.2.0,<0.3.0` (version 3.x of checkpoint-postgres requires langgraph 0.3+). Validated: 76/76 tests pass after addition.
- **Impact**: requirements.txt

## Decision 10: Docker stub directory cleanup
- **Date**: Session 4
- **Context**: Phase 0 scaffold created `docker/Dockerfile`, `docker/docker-compose.yml`, and `docker/qdrant_config.yaml` as stubs. Phase 6 then placed the real implementations at project root (`Dockerfile`, `Dockerfile.streamlit`, `docker-compose.yml`) and the Qdrant config at `config/qdrant_config.yaml`. The `docker/` stubs were never updated and contained only placeholder comments.
- **Resolution**: Deleted `docker/` directory entirely. Fixed `README.md` reference from `docker compose -f docker/docker-compose.yml up -d` to `docker compose up -d`.
- **Impact**: docker/, README.md

## Decision 11: Streamlit import path and API base URL
- **Date**: Session 5
- **Context**: `streamlit run app/streamlit_app.py` crashed on its first import with `ModuleNotFoundError: No module named 'app'`. Streamlit's `bootstrap._fix_sys_path()` inserts only the *script's* directory (`app/`) into `sys.path`, never the project root, so the absolute `app.components.*` imports could not resolve — locally or in Docker. Separately, `API_URL` was hardcoded to `http://localhost:8000/api/v1` while `docker-compose.yml` injects `API_URL=http://api:8000`; inside the container `localhost` is the Streamlit container itself. Either bug alone made the Dockerized UI non-functional.
- **Resolution**: Prepend the resolved project root to `sys.path` in `streamlit_app.py` before the component imports (works identically bare-metal and in Docker, no env var required), and read the API base from `os.getenv("API_URL", "http://localhost:8000")`, appending `/api/v1`. Pinned Streamlit/requests in `Dockerfile.streamlit`, added a `/_stcore/health` healthcheck to the compose service, and copied `.streamlit/` into the image so the theme applies.
- **Impact**: app/streamlit_app.py, Dockerfile.streamlit, docker-compose.yml, .streamlit/config.toml

## Decision 12: include_pii lives on the state root, not in extracted_filters
- **Date**: Session 5
- **Context**: `QueryRequest.include_pii` was accepted by the API but never reached retrieval, so the flag was inert (failing safe, but over-promising in the contract). The obvious fix — put it in `extracted_filters` — is wrong: Agents 1 and 6 deliberately `pop("include_pii")` from anything an LLM produced, precisely so a model cannot escalate its own data access.
- **Resolution**: Added `include_pii` as a top-level `AgentState` field, set once by `run_query()` from the authenticated request. `retrieval_executor_node` merges it into the filter dict passed to `hybrid_search()` and forwards it to `expand_context()`. The LLM-stripping rule is untouched, and the query route records every authorized use on its `AUDIT_LOG` line.
- **Impact**: state_definitions.py, orchestrator.py, retrieval_executor.py, api/routes/query.py

## Decision 13: Document registry + version retirement
- **Date**: Session 5
- **Context**: `RiskSignalExtractor` and the risk/version UI components were fully implemented but unreachable — nothing produced their data. `document_count` on a deal never left 0. Worse, uploading a replacement document left *both* versions marked `is_current_version=1`, so retrieval's default version filter returned superseded terms alongside the ones that replaced them.
- **Resolution**: Added an in-process `_documents` registry in `api/routes/deals.py` (same pattern and same single-worker constraint as `_deals`), populated by the ingestion route, exposed via `GET /deals/{id}/documents` and `GET /deals/{id}/risk-signals`. Wired `RiskSignalExtractor` into ingestion — regex only, no LLM, no added latency — writing `risk_signals` per chunk and aggregating to document level. Ingesting with `supersedes_doc_id` now issues a Qdrant `set_payload` flipping the prior document's chunks to `is_current_version=0` and stamping `superseded_by`. Severity ranking lives in the API layer, not the extractor: the extractor reports what it matched, the dashboard ranks it.
- **Trade-off**: The registry is process-local and lost on restart, and it is now the only record of version chains and risk signals. Moving it to Postgres is the first step of the scale-out path.
- **Impact**: api/routes/deals.py, api/routes/ingest.py, api/models/response_models.py

## Decision 14: AsyncPostgresSaver lifetime
- **Date**: Session 5
- **Context**: `get_compiled_graph()` called `AsyncPostgresSaver.from_conn_string(url).setup()`. That factory returns an `_AsyncGeneratorContextManager`, which has no `.setup()` — so the call raised `AttributeError` on **every** startup and was swallowed by the `except` that falls back to `MemorySaver`. Postgres checkpointing had therefore never worked in any environment, and the warning it logged read like "Postgres is down" even when it was up. Durable checkpoints and crash recovery were silently absent.
- **Resolution**: Enter the context manager into a module-level `AsyncExitStack` that lives as long as the compiled graph, and unwind it from the FastAPI lifespan via a new `close_checkpointer()`. The `MemorySaver` fallback is retained for genuine connection failures.
- **Note**: Verified against the library contract (`from_conn_string` returns an async context manager with no `setup` attribute). The connected path could not be exercised locally — no Postgres instance was available — so first Docker run should be checked for `"compiled with PostgresSaver"` rather than the MemorySaver warning.
- **Impact**: src/workflow/orchestrator.py, api/main.py

## Decision 15: Escape currency before Streamlit renders LLM prose
- **Date**: Session 5
- **Context**: Streamlit's markdown parses `$…$` as inline LaTeX. An answer reading "revenue was $184.2M, up from $151.7M" had the span between the two dollar signs rendered as math: both `$` characters and the enclosing bold markers were silently swallowed, so figures disappeared from the rendered answer. Verified in the DOM (`class="language-math math-inline"`).
- **Resolution**: `app/styles.py::escape_currency()` normalizes then escapes `$` before any LLM-produced string reaches `st.markdown`. Applied to the answer body, hallucination flags, and risk descriptions.
- **Rationale**: For an engine whose stated premise is that a wrong number is a hard transaction failure, a number vanishing at the render layer is a correctness defect, not a styling one.
- **Impact**: app/styles.py, app/components/answer_display.py, app/components/risk_dashboard.py

## Decision 16: Citation precision — investigated, deliberately not changed
- **Date**: Session 5
- **Context**: With citations now carrying real metadata (Decision 12's sibling change), the UI displays section headings — and a live run exposed that they are wrong. An answer citing only "SECTION 8. INDEMNIFICATION" three times produced citations labelled *SECTION 7. LITIGATION*, *SECTION 5. CHANGE OF CONTROL*, and a mid-sentence fragment. Root cause: `answer_synthesizer` matches `source_file in answer`, which is document-level, so every retrieved chunk from that document is returned.
- **Attempted**: Narrowing to chunks whose `section_heading` also appears in the answer, falling back to file-level when none match. **Reverted** — it changed nothing in practice. `section_heading` is the heading of the *chunk*, and a chunk routinely spans several sections (a 998-byte test agreement produced 3 chunks covering 4 sections), so the answer's section heading is simply not present in the chunk metadata to match against.
- **Resolution**: Left the existing behaviour in place and documented the imprecision in-code, rather than shipping speculative logic with no demonstrated benefit. A real fix is one of: (a) per-chunk heading granularity from the chunker, or (b) parsing the inline `[file | p.N | SECTION]` citation markers the synthesis prompt already instructs the model to emit — (b) is cheaper and self-validating, but couples the parser to the prompt contract and needs its own tests.
- **Impact**: src/agents/answer_synthesizer.py (comment only)


## Decision 17: Calibrate the quality gate against the reranker's real score distribution
- **Date**: Session 6
- **Context**: 10 of 16 completed golden questions were refused. The Quality Assessor scored context as `mean(top_5)` / `min(top_5)` of reranker scores and required `mean >= 0.5 and min >= 0.2`; the per-type gate then demanded e.g. `precision >= 0.8` for financial queries. Nobody had measured what BAAI/bge-reranker-v2-m3 actually outputs on this corpus. Labelling every reranked chunk relevant/irrelevant using the golden set's expected facts gave the answer: relevant chunks score 0.24-0.99 (median 0.241), noise sits at median 0.006. The distribution is bimodal, so (a) averaging across the top-k means **retrieving more candidates lowers the score** - the gate punished recall - and (b) `min(top_5) >= 0.2` required the fifth-best chunk to be excellent, which such distributions rarely satisfy. Evidence: fin_02 and mh_04 scored 0.638 and 0.645 on genuinely good context and were refused.
- **Resolution**: Score the *usable* evidence instead of the whole retrieved set. `relevance = max(score)`; `precision = mean(scores >= RELEVANCE_FLOOR)`; `completeness = len(usable) / EXPECTED_EVIDENCE_COUNT[query_type]`. `RELEVANCE_FLOOR = 0.10` is the peak-F1 point of a sweep over the labelled data (F1 0.534). Thresholds re-derived the same way: the 18 questions whose retrieval genuinely contained the answer scored relevance >= 0.521 / precision >= 0.298 / completeness >= 0.333, while the one true retrieval failure scored 0.009 / 0.000 / 0.000 - so 0.30 / 0.25 / 0.30 sits in a wide empty band rather than near a dense part of the distribution.
- **Side effect**: the confident bands now cover the whole observed range, so the LLM fallback fired 0/19 times - one fewer model call per query.
- **Guarded by**: `test_one_decisive_chunk_is_enough`, `test_retrieving_more_noise_does_not_lower_quality`, plus an adversarial all-noise check that must still refuse.
- **Impact**: quality_assessor.py, conditional_edges.py

## Decision 18: Progressive relaxation of the document_category filter
- **Date**: Session 6
- **Context**: Agent 1 infers a `document_category` and it was applied as a hard Qdrant `must`. When the inference is wrong the answer is removed from the search space and the rewrite loop cannot recover it, because rewriting changes query text and never filters. Measured on the golden set: legal_01 ("per-share merger consideration") was classified `financial`, filtering out the merger agreement - the only document containing the answer; fin_05 (credit facility terms) was classified `legal` and excluded the financials; comp_02 (valuation methodologies) was classified `financial` and excluded the board deck. All three burned both rewrite attempts against a search space that could not contain the answer. It also compounds two error sources, since the category being matched was itself assigned by a heuristic classifier at ingestion.
- **Resolution**: Keep the category filter on the first attempt, where it buys precision, and drop it once `rewrite_iteration >= 1` - the point at which the system is explicitly trading precision for recall. `deal_id`, `is_current_version` and `contains_pii` are never relaxed: those are isolation and compliance constraints, not relevance hints.
- **Guarded by**: `TestProgressiveFilterRelaxation`.
- **Impact**: retrieval_executor.py

## Decision 19: The hallucination validator must see the synthesizer's evidence
- **Date**: Session 6
- **Context**: 10 of 19 answers came back `validation_status="failed"`, several with 100% fact recall against the golden set. fin_01 was flagged for claiming revenue was $452.8M/$387.1M - the correct, sourced figures. Cause: the validator formatted context as `c["text"][:500]`, while the synthesizer received the full chunk plus parent context. The validator was asked "is this claim supported?" while being shown roughly a sixth of the supporting text, so grounded numbers past the cutoff looked unsupported.
- **Resolution**: Give the validator the full chunk text and parent context - the same evidence the writer had. Flagged answers dropped from 11/19 to 2/19 and `passed` rose from 8 to 17.
- **Why it matters beyond the metric**: a safety control with a high false-positive rate is worse than none, because reviewers learn to ignore it.
- **Impact**: hallucination_validator.py

## Decision 20: Verification agents default to cloud, with local retained
- **Date**: Session 6
- **Context**: Agents 4 and 8 always used `ollama/qwen2.5:14b`. That was a deliberate cost decision - verification is the highest-volume agent traffic, and keeping it local held the pipeline inside the free Gemini tier - but it made the project unrunnable without a 12GB-VRAM host and a 9GB model pull, and local inference dominated latency.
- **Resolution**: `VERIFICATION_BACKEND` selects the backend, defaulting to `cloud`. The cloud path routes through `BudgetTracker.get_model_for_agent()`, so it takes a rate-limiter slot, debits the daily quota, and automatically returns the local model when the quota is spent. Setting `VERIFICATION_BACKEND=local` restores the original behaviour. Both paths are live.
- **Measured**: mean latency 55.1s -> 18.2s. Cost: cloud calls per query roughly doubled, which is what exposed Decision 21.
- **Impact**: litellm_wrapper.py, financial_verifier.py, hallucination_validator.py

## Decision 21: Rate-limit by upstream model, not by budget bucket
- **Date**: Session 6
- **Context**: After Decision 20 the golden-set run failed with `RateLimitError`. `_get_rate_limiter` keyed limiters by budget bucket - `synthesis_primary` at 5 RPM and `agent_workhorse` at 15 RPM - but both buckets resolve to the same `gemini-3.1-flash-lite`, so the client permitted 20 requests/minute against a single 15 RPM provider quota. The bug pre-dated the change and stayed hidden only because local verification kept cloud volume low. Separately, `call_structured_agent` retried transport errors with no delay, so a 429 burned all three attempts in milliseconds.
- **Resolution**: Key rate limiters by upstream model string via `BUCKET_MODELS` / `MODEL_RPM_LIMITS`, so every bucket spending against the same model shares one limiter. Daily budgets stay per-bucket - that is cost allocation, a separate concern from provider metering. Added exponential backoff to structured-agent retries, with a longer pause for rate-limit errors since the window they enforce is measured in seconds.
- **Result**: 23/23 queries completed with zero rate-limit or connection errors.
- **Guarded by**: `TestSharedModelRateLimiter`.
- **Impact**: budget_tracker.py, litellm_wrapper.py

## Decision 22: Unanswerable control questions in the golden set
- **Date**: Session 6
- **Context**: After Decisions 17 and 18 the engine answered 19/19 answerable questions and refused none. On a set where every question is answerable, a 0% refusal rate is unfalsifiable: "correctly answers everything" and "gate is broken and never refuses" produce identical numbers.
- **Resolution**: Added 4 control questions (`ctrl_*`) whose answers are absent from the corpus by construction - environmental liabilities, FY2024 quarterlies, headcount by office, churn vs competitors - and scored them on whether the engine fabricated the missing figure.
- **Finding**: the first scoring pass reported 1/3 and looked like hallucination. It was the metric, not the engine: a binary refused/answered flag scored *partial* answers as failures. The engine had replied "The provided documents do not contain the revenue figures for Q1 FY2024" with a citation, and for the churn question reported the retention and competitor data that do exist while stating churn does not. That is better due-diligence behaviour than a blanket refusal. Re-scored on "did it fabricate the missing fact": 4/4.
- **Also surfaced**: one control crashed the pipeline - RRF raised `ValueError` when both retrievers returned empty, turning a legitimate no-results query into a 500. Now returns an empty list so the Quality Assessor takes the refusal path.
- **Impact**: tests/golden_qa_set.json, tests/run_end_to_end_validation.py, rrf_fusion.py

## Decision 23: Streamlit example queries must write to the widget's own state key
- **Date**: Session 6
- **Context**: Clicking an example query filled the text area visually but left the Search button disabled. The handler wrote to a separate `query_text` key passed as `value=`; once a keyed widget exists Streamlit serves its stored state and ignores `value=`, so the widget still returned an empty string and `disabled=not query` stayed true. Found by driving the real UI, not by reading the code - the DOM and the widget state disagreed.
- **Resolution**: Write directly to `st.session_state["query_input"]` (the text area's own key) and call `st.rerun()`.
- **Impact**: app/components/query_interface.py

## Decision 24: Sub-question decomposition for multi-hop and comparative queries
- **Date**: Session 7
- **Context**: Query expansion generated paraphrases of the *same* question. For a question whose answer requires combining facts from different documents, that is structurally incapable of working: "what is the implied EV/EBITDA multiple?" rephrased five ways still never asks for the per-share price or the share count, so the passages holding them were never retrieved and the engine correctly reported it could not compute the multiple.
- **Resolution**: Agent 1 now emits `sub_questions` alongside `query_expansions`, gated to `DECOMPOSABLE_QUERY_TYPES = {multi_hop, comparative}` and capped at `MAX_SUB_QUESTIONS = 4`. Each sub-question gets its own retrieval pass, **reranked against itself** — that is the whole mechanism: a passage containing only a share price scores near zero against the parent question and high against "what is the per-share merger consideration?". Results merge round-robin by rank so every facet is represented, then re-sort by score.
- **Two sub-bugs, both found by running it rather than reading it**:
  - Sub-questions inherited the parent's `document_category` filter, so all three passes of the EV/EBITDA query returned chunks from the *same* document. Decomposition had worked and the filter discarded the benefit. Sub-questions exist precisely to look elsewhere, so they now drop that filter (Decision 18 relaxes the same filter for a related reason).
  - Splitting a fixed `final_top_k` across passes starved the parent query: comp_05 (severance multiples) went from 20% to **0%** fact coverage because the one table the parent had already found was squeezed out. The budget is now additive — the parent keeps its full `final_top_k` and sub-questions add on top.
- **Measured on retrieval alone, not end-to-end**: mean fact coverage in retrieved context **64.2% → 84.7% (+20.4pp)** across the 15 decomposable golden questions, **zero regressions** (comp_02 10%→100%, mh_06 33%→100%, mh_07 50%→100%, mh_04 33%→67%). End-to-end recall could not attribute the change — see Decision 25.
- **Guarded by**: quota-merge, additive-budget, and filter-inheritance tests in `tests/test_agents.py`.
- **Impact**: query_intelligence.py, retrieval_executor.py, prompt_templates/query_intelligence.py, state_definitions.py, orchestrator.py

## Decision 25: Report end-to-end recall as a range, because the benchmark measures quota state
- **Date**: Session 7
- **Context**: Three runs of the identical 41-question set scored 86.6%, 73.9% and 71.0%. The obvious reading was that Decision 24 had backfired. It had not. Question types that are **never decomposed** moved in lockstep — legal 100%→78%→73%, summary 72%→69%→50% — while the synthesis model mix drifted from mostly `gemini-3.6-flash` to mostly `gemini-3.5-flash` as the 20-request daily quota on the stronger model drained.
- **Resolution**: Report recall as a range (**71–87%**) with the cause stated, rather than the single flattering figure from the best run, and measure Decision 24 on retrieval alone where model choice cannot interfere.
- **Why this is the finding, not the caveat**: the headline metric was substantially a function of what time of day the run started. Any single number quoted from it — including the good one — would have been an artifact. A benchmark that silently varies by 15 points with the provider's rate limiter is measuring the rate limiter.
- **Impact**: README.md, RESULTS.md, scratchpad A/B harness

## Decision 26: Durable checkpointing was dead on Windows — uvicorn hardcodes the Proactor loop
- **Date**: Session 7
- **Context**: Every startup logged `Failed to initialize PostgresSaver: Psycopg cannot use the 'ProactorEventLoop'` and fell back to `MemorySaver`. One WARNING line was the only evidence that crash recovery and session resume did not exist. An earlier fix (holding the `AsyncPostgresSaver` context open in an `AsyncExitStack`) was necessary and insufficient — the connection could never open in the first place.
- **Why the obvious fix fails**: setting `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` does nothing. uvicorn 0.49's `loops/asyncio.py` returns `asyncio.ProactorEventLoop` as an explicit loop **factory**, and a factory overrides the policy. It also loads the ASGI app from inside `Server.serve()`, so module-scope code in `api.main` runs when the loop already exists. Both routes are closed.
- **Resolution**: `run_api.py` constructs the loop itself — `asyncio.Runner(loop_factory=asyncio.SelectorEventLoop)` — and hands uvicorn a `Server` to run on it. Linux is untouched and keeps uvloop, since its default loop is already selector-based. Result: `LangGraph state machine compiled with PostgresSaver` for the first time.
- **Trade-off**: the selector loop caps around 512 sockets and cannot spawn subprocesses. This process talks HTTP to Qdrant and Postgres and spawns nothing, so neither binds; `--reload` is off for that reason.
- **Impact**: run_api.py (new), api/main.py

## Decision 27: Bind dates as dates — schema drift hid a total failure behind a green test suite
- **Date**: Session 7
- **Context**: On a Postgres volume created fresh from the current schema, **every** query failed with `invalid input for query argument $1: '2026-08-14' ('str' object has no attribute 'toordinal')`. The budget tracker passed `date.today().isoformat()` into `reset_date = $1::date`; asyncpg encodes by the type it infers from the statement, and the server-side `::date` cast happens after encoding, so it cannot rescue a str.
- **Why it survived**: the long-lived development database still held `reset_date` as text from an older schema revision, so string binding worked there, and every existing tracker test sets `_is_mock = True`, which skips SQL entirely. The suite was green while the code could not complete a single query against a correctly-created database. This is the failure mode worth naming: it was never a dev-only quirk — it broke every *first* deploy, and only the first, which is the one nobody has logs for.
- **Resolution**: `_utc_today()` returns a `date` for anything crossing into SQL; `_utc_today_iso()` returns the string form for the in-Python comparisons and the mock backend.
- **Guarded by**: `TestDateBindingToPostgres`, which drives the real SQL path through a recording connection and asserts no ISO-shaped string is ever bound to a `::date` parameter. Verified to fail against the pre-fix behaviour before being kept.
- **Impact**: budget_tracker.py, tests/test_rate_limiter.py

## Decision 28: Container packaging — exclude live state, pass the whole key pool
- **Date**: Session 7
- **Context**: `docker compose up` failed outright: with no `.dockerignore`, the build context included `docker_data/`, a bind-mount target held open by the running containers, so the build died on `open docker_data: Access is denied` before writing a layer. Separately, the `api` service passed only `GEMINI_API_KEY` while the budget tracker's primary path reads `GEMINI_API_KEYS` — the container silently collapsed a five-key rotation pool to one key and would hit a daily cap the host process survives.
- **Resolution**: Added `.dockerignore` (live DB state, secrets, virtualenvs, VCS, caches, and `data/` which is mounted at runtime); compose now passes `GEMINI_API_KEYS` alongside the legacy singular form.
- **Impact**: .dockerignore (new), docker-compose.yml

## Decision 29: Pin the Qdrant client to the server — a version skew failed every write in silence
- **Date**: Session 7
- **Context**: On a clean environment, all 9 documents failed to ingest with `Vector dimension error: expected dim: 1024, got 0` and `Conversion between sparse and regular vectors failed`, and the harness then ran all 41 questions against an empty index and scored 0% recall on every one. `requirements.txt` allowed `qdrant-client>=1.9.0,<2.0.0`, which resolved to 1.18.0, while `docker-compose.yml` pinned the server to `v1.12.1` — six minor versions apart, against a documented contract of at most one.
- **Diagnosis**: probing the same upsert over both transports isolated it immediately — `prefer_grpc=True` failed, `prefer_grpc=False` succeeded. The newer client's gRPC vector encoding was unreadable by the older server. REST, health checks, collection creation and search all worked, so every signal short of the actual write said the system was fine.
- **Why it was invisible**: the client *does* detect this and raises a `UserWarning`, which a JSON log handler swallows. Nothing downstream treats an empty index as an error — retrieval returns no chunks, the quality gate scores 0.00, and the engine correctly refuses. A total ingestion failure and a genuinely unanswerable corpus are indistinguishable in the output.
- **Resolution**: server pinned to `v1.18.0`, client range narrowed to `>=1.17.0,<1.20.0`, with the coupling stated in both files. `assert_server_compatible()` now runs before anything writes and **raises** on a skew rather than logging. The eval harness aborts when fewer documents ingest than the corpus contains, instead of producing a full results file measuring the ingestion failure.
- **Migration note**: Qdrant 1.18 cannot read segments written by 1.12 (`unknown variant 'on_disk'`), so the volume had to be recreated. It held zero points, so nothing was lost — on a populated deployment this would require a re-index.
- **Impact**: requirements.txt, docker-compose.yml, src/vector_db/qdrant_client.py, api/main.py, tests/run_end_to_end_validation.py

## Decision 30: The models were on CPU the whole time
- **Date**: Session 8
- **Context**: A live query took 51s, with 24s of it inside retrieval across two passes. `torch.__version__` was `2.12.1+cpu` on a machine with an idle RTX 5070 Ti (12GB). BAAI/bge-m3 and bge-reranker-v2-m3 had been running on CPU for the life of the project.
- **Why it was invisible**: the device-selection code was correct — `"cuda" if torch.cuda.is_available() else "cpu"` — and it faithfully logged `CPU` at every startup. Nothing failed. Answers were identical. The only symptom was that everything was about five times slower than it needed to be, which is not a symptom anyone notices without a baseline. `requirements.txt` even documented installing a CUDA wheel; the instruction did not work, which is the actual defect.
- **Root cause of the bad instruction**: `pip install torch --index-url https://download.pytorch.org/whl/cu128` reports *"Requirement already satisfied"* whenever any torch is present. The CPU and CUDA wheels share a version and differ only by local tag (`2.12.1+cpu` vs `+cu128`), which pip does not treat as an upgrade. The documented command was a no-op on every machine that already had torch — which is every machine that had installed `requirements.txt` first, since sentence-transformers pulls torch in.
- **Resolution**: `pip install --force-reinstall --no-deps torch --index-url .../cu128`, with the reason written into `requirements.txt` and a one-line verification step in the README. cu128 or later is required for Blackwell (sm_120); earlier CUDA builds install cleanly and fail at the first forward pass.
- **Measured**: retrieval 6–18s per pass → **0.8–3.8s for up to four passes**. API startup including model warmup 76s → 35s. End-to-end 51.5s → 32.5s on the same question.
- **Impact**: requirements.txt, README.md

## Decision 31: Warm the local models at startup
- **Date**: Session 8
- **Context**: All three local models load lazily. The first query after a restart therefore paid several gigabytes of weight loading — fine for a batch evaluation, bad in front of an audience, where the first question is the one being watched.
- **Resolution**: `warm_models()` runs during the API lifespan, loading *and exercising* each model, since the first forward pass allocates buffers and selects kernels that a bare constructor does not. Failures are logged and swallowed — a warmup must never be why the API refuses to start. `WARM_MODELS=0` skips it for fast development restarts.
- **Measured**: ~8.6s moved off the first query (dense 5.4s, reranker 2.9s, BM25 0.3s).
- **Impact**: src/vector_db/reranker.py, api/main.py

## Decision 32: The classifier must not veto the decomposer
- **Date**: Session 8
- **Context**: Sub-questions were kept only when `query_type` was `multi_hop` or `comparative`. Asked live for the implied EV/EBITDA multiple on adjusted rather than reported EBITDA — the exact question sub-question decomposition was built for (Decision 24) — Agent 1 decomposed it correctly into price, share count and EBITDA, *and* classified it `financial`. The veto discarded all three sub-questions and the engine answered about the offer price instead.
- **Resolution**: honour sub-questions whenever the model returns them. Emitting them is the model's judgment that the answer spans several facts; classification is a separate judgment made for routing. Gating one on the other meant the feature worked only when two independent guesses agreed and failed invisibly when they did not. The prompt decides when to decompose; `MAX_SUB_QUESTIONS` bounds the cost.
- **Measured on the same question**: 1 retrieval pass → 4; sub-questions 0 → 3; rewrite loops 1 → 0; confidence 0.85 → 1.00; and the answer went from the offer price to the multiple itself, **7.03x on Adjusted EBITDA of $99.0M versus 7.15x reported**.
- **Not yet re-measured**: the 41-question evaluation predates this change. Removing a veto can only add decomposition, so the effect should be neutral-to-positive — but that is an argument, not a measurement, and the reported numbers say so.
- **Guarded by**: `test_sub_questions_survive_an_unexpected_query_type`, `test_sub_questions_are_capped`.
- **Impact**: src/agents/query_intelligence.py, tests/test_agents.py

## Decision 33: The UI must reflect what is indexed, not what this process remembers
- **Date**: Session 8
- **Context**: Opening the dashboard against a fully populated index showed "No deals found", an empty version browser and an empty risk dashboard. `_deals` and `_documents` are in-memory dicts populated only by `POST /deals` and the ingest route, so anything ingested directly (how the evaluation harness loads the corpus) was invisible, and *everything* was invisible after a restart while Qdrant kept the data on disk. The engine answered questions correctly about a deal the UI said did not exist.
- **Resolution**: `GET /deals` now merges the in-memory registry with deals discovered by faceting Qdrant on `deal_id`; documents and risk signals are reconstructed by scrolling chunk payloads when the registry is empty. Added a `source_file` payload index, since Qdrant refuses to facet an unindexed field.
- **Shape mismatch worth noting**: chunk payloads store risk signals as bare type strings while the registry stores the extractor's full dicts. Handling only the dict shape produced a dashboard that rendered nothing and reported no error — the reconstruction normalises both.
- **Result**: 9 documents and 19 risk signals (7 high severity) now visible after a cold start.
- **Impact**: api/routes/deals.py, src/vector_db/collection_manager.py

## Decision 34: One command to start the demo
- **Date**: Session 8
- **Context**: Starting the stack by hand has ordering constraints that are easy to get wrong and expensive to get wrong live: Qdrant must be healthy before the API opens its client, the API must be up before ingestion, and the index must be populated before the UI is worth showing. Every failure mode had already happened at least once — Docker not running, a version-skewed Qdrant, an empty index after a volume reset, cold models on the first question.
- **Resolution**: `run_demo.py` performs and *checks* each step, starting Docker Desktop itself if needed. It fails with the specific remedy rather than a stack trace.
- **Also fixed here**: the Qdrant healthcheck ran `curl`, which that image does not ship — 1932 consecutive failed checks on one container. Since `api` waits on `condition: service_healthy`, the fully containerized `docker compose up` path could never have started the API or the UI. Replaced with a bash `/dev/tcp` probe.
- **Impact**: run_demo.py (new), docker-compose.yml

## Decision 35: A 503 is not a quota refusal, and neither is a timeout
- **Date**: Session 8
- **Context**: Re-running the evaluation after Decision 32, `gemini-3.6-flash` began returning 503 *"This model is currently experiencing high demand"* on roughly a third of synthesis calls. Each failure cost ~125 seconds before the provider dropped the connection, average query latency went from 30s to 80s, and five answerable questions never produced an answer at all.
- **Three separate defects, all exposed by one bad afternoon**:
  - **No request timeout.** LiteLLM defaults to 600s, which is not a timeout so much as its absence. `STRUCTURED_TIMEOUT_SECONDS = 60` and `PROSE_TIMEOUT_SECONDS = 120` are generous against healthy latencies (1-3s and 10-30s) and only fire when something is genuinely wrong.
  - **503 misclassified.** It fell through to the generic transport-retry path, so a model that was down provider-wide got three retries with backoff before anything else was tried. A 429 means *this key* is spent and another key is the right move; a 503 means the *model* is down and only another model helps. `is_service_unavailable()` now separates them, and `skip_model_for_request()` puts the model in a 90-second cooldown **without debiting quota** — the opposite mistake would retire the best synthesis model for a whole day over a blip.
  - **Uncited answers were published.** Under stress the model returned answers at 10-22% of their normal length with zero citation markers. One shipped at `validation=passed` with `confidence=1.0` and no sources; another leaked the model's own working notes ("Let's double-check all details to ensure accuracy") as the answer body. `_is_usable_answer()` now treats a response with no citations, or with scratchpad markers, as a failed generation and retries on another rung. Deliberately narrow: a genuine refusal has nothing to cite and passes through untouched.
- **Guarded by**: `TestRequestTimeouts`, `TestServiceUnavailableHandling`, `TestUsableAnswerGuard`.
- **Impact**: litellm_wrapper.py, budget_tracker.py, answer_synthesizer.py

## Decision 36: Evaluation reports must carry their run conditions
- **Date**: Session 8
- **Context**: The end-to-end recall metric has now been misread twice for reasons that had nothing to do with the engine. First when it swung 15 points between identical runs purely on which synthesis model the daily quota allowed (Decision 25). Then when the provider incident above produced truncated, uncited answers and a 62.1% score that looked exactly like a retrieval regression — while a retrieval-only A/B on the same index showed a **+5.4pp improvement with zero regressions**.
- **Resolution**: `RESULTS.md` is generated with a **Run conditions** section recording the synthesis model mix, how many queries were decomposed, and how many answers were lost to upstream synthesis failure. A non-zero failure count is the signal that the run is not comparable to a clean one.
- **Why this rather than more careful reading**: the number gets quoted out of the document it lives in. Attaching the conditions to the artefact is the only version that survives being copied into a slide.
- **Measured this run**: 62.1% mean recall, 30/35 answered, 6/6 controls held, 80.4s average latency, 38/41 syntheses on `gemini-3.6-flash`, 6 answers lost upstream. The comparable clean run scored 85.9% on the same model mix.
- **Impact**: tests/run_end_to_end_validation.py, RESULTS.md, README.md

## Decision 37: The decomposition fix, measured on a clean provider day
- **Date**: Session 9
- **Context**: Decision 36 recorded a 62.1% run that was dominated by a Gemini incident rather than by any code change. Re-run once `gemini-3.6-flash` quota reset and the provider was healthy (probed first: 4/4 successful calls, no 503s).
- **Result**: **86.3% mean fact recall**, 35/35 answered, 26/35 with every expected fact present, 33/35 citation match, 6/6 controls held, **0/41 answers lost upstream**. Same synthesis model mix as the 85.9% baseline (38 of 41 on `gemini-3.6-flash`), so the comparison is like-for-like.
- **Honest reading**: end-to-end recall is **unchanged**. +0.4pp across 35 questions, with 6 improving and 3 regressing, is run-to-run variance in the synthesis model, not a demonstrated effect. The claim that decomposition works rests on the retrieval measurement — +5.4pp fact coverage with zero regressions, concentrated in comparative (+16.0) and multi-hop (+10.8) — not on this number.
- **Why report both**: the two together say something more useful than either alone. Decomposition reliably puts more of the required evidence in front of the synthesizer; on a 9-document corpus the synthesizer was already finding enough for most questions, so the end-to-end ceiling is set elsewhere. That is an argument for the corpus being too small to show the gain, not for the gain being absent — and it predicts where the difference would appear at data-room scale.
- **Also confirmed**: the uncited-answer guard from Decision 35 fired zero times on this run, which is the expected behaviour when the provider is healthy.
- **Impact**: RESULTS.md, README.md

## Decision 38: A UI test that needs the backend is not a UI test
- **Date**: Session 9
- **Context**: Two Streamlit tests failed when run with the API stopped. They were not broken — they had always required a live backend, and had only ever been run while one happened to be up. The deal selector is populated from `GET /deals`, so with no API the app returns early at "select a deal" and never renders the query interface the tests assert on.
- **Resolution**: added a `?deal_id=` deep link as the last fallback in the deal selector, and the tests now use it. No network call is needed to select a deal, so the tests exercise the UI and nothing else. Panels whose endpoints are unreachable degrade to empty, which is the behaviour the dashboard is built for regardless.
- **Not a test-only change**: the deep link is independently useful — it makes a data room shareable as a URL rather than "open the dashboard, then pick Aurora from the dropdown".
- **Guarded by**: `test_deep_link_selects_a_deal`, plus the existing example-query tests now running API-free.
- **Impact**: app/components/deal_manager.py, tests/test_streamlit_ui.py

## Decision 39: The budget panel reported yesterday's spend
- **Date**: Session 9
- **Context**: After the daily quota reset, the sidebar read "26/95 left" on `gemini-3.6-flash` while all 95 were in fact available. The daily reset is applied lazily inside `_try_consume`, on each slot's next call, so the stored counter stays at yesterday's value until then. `_budget_available` already handles this — a stale `reset_date` means a full allowance — but `get_budget_status` read `used_today` raw.
- **Why it mattered**: the panel contradicted the router. A status display that disagrees with the thing it reports on is worse than no display, and this one is the first thing a reviewer looks at to understand the quota engineering.
- **Resolution**: `get_budget_status` treats a stale `reset_date` as zero usage, matching what the consumption path will do on the next call, and reports today's date rather than the stored one.
- **Guarded by**: `TestBudgetStatusReporting`, including an invariant test that the reported remaining capacity never contradicts `_budget_available`.
- **Impact**: src/llm/budget_tracker.py

## Decision 40: The category filter must come off *before* the search, not after it
- **Date**: Session 9
- **Symptom**: the EV/EBITDA question answered differently run to run — sometimes computing 7.50x, sometimes reporting the multiple "cannot be determined". Live sampling put the required `$696 million` aggregate merger consideration in the retrieved context in only **3 of 8 runs**.
- **False leads, both measured and discarded**:
  - *Sub-question depth.* The `$696M` chunk ranks 1st, 5th, 4th and 1st depending on how Agent 1 phrases the value sub-question, and `MIN_CHUNKS_PER_SUB_QUESTION = 2` truncates before rank 4. Plausible — but a sweep over depth 2→6 on the whole golden set moved mean coverage **not at all** (73.8% flat), because the sweep froze one decomposition per question and happened to freeze a favourable one for this query. A measurement that cannot see the failure is not evidence of its absence.
  - *Agent 1 temperature.* Already `0.0`. The phrasing variance is model nondeterminism, not a setting.
- **Actual cause**: Agent 1 classifies this question `financial` or `legal`, and `document_category` is applied as a hard filter on the *first* attempt. `$696 million` is stated in the regulatory memo and in the merger agreement; either filter removes the chunk that states it. Deterministic, no LLM involved: filter `none` → found, `financial` → missing, `legal` → missing, `regulatory` → found.
- **Why progressive relaxation (Decision 18) did not save it**: relaxation fires on *low context quality*. This context scores well — it is full of plausible, high-scoring financial chunks. It is **incomplete, not weak**, and nothing downstream can tell those two apart. A gate that only reacts to bad scores cannot rescue a good score with a hole in it.
- **Resolution**: once a query has been decomposed, the parent pass drops `document_category` too — not just the sub-question passes. Decomposing is Agent 1 stating that the answer spans several facts; those facts routinely sit in different document categories, so constraining the parent search to one contradicts the judgment that produced the sub-questions. Undecomposed queries keep the filter, where it still buys precision.
- **Measured**: `$696M` present at every category filter and at the smallest depth (15 chunks, no increase in context size). Retrieval A/B over the golden set improved from **+5.4pp to +10.4pp**, multi-hop coverage **81.7% → 94.7%**, comparative **+31.0pp**, still **zero regressions**. Live, `$696` and `$92.8` now reach the answer in **4/4** runs against 3/8 before.
- **Residual, stated plainly**: the computed `7.5x` still appears in only 2 of 4 live runs — and both misses were on `gemini-3.5-flash-lite`, both hits on `gemini-3.6-flash`. Retrieval is now reliable; the arithmetic step is model-dependent. That is honest degradation rather than a retrieval defect, and the answer still reports the inputs it was unable to combine.
- **Guarded by**: `test_sub_questions_do_not_inherit_the_category_filter` (updated), `test_undecomposed_query_keeps_the_category_filter`.
- **Impact**: src/agents/retrieval_executor.py, tests/test_agents.py

## Decision 41: Half the fallback ladder was unreachable — for two different reasons
- **Date**: Session 9
- **Context**: while tracing a synthesis failure, the log showed `404 models/gemini-3-flash is not found for API version v1alpha`. Probing every laddered model: `gemini-3-flash`, `gemini-2.5-flash` and `gemini-2.5-flash-lite` all 404. Three of six cloud rungs were dead, so whenever the working rungs were spent or failing — exactly when the ladder matters — the router descended into models that could only fail, burning an attempt and a timeout on each.
- **First conclusion, and the correction to it**: I initially recorded all three as nonexistent and deleted them. That was half wrong, and worth recording as written. Listing the API's own catalogue showed **`gemini-3-flash-preview`** — the console's "Gemini 3 Flash" is served under a `-preview` suffix. The 404 meant *I asked for the wrong string*, not *the model is absent*, and reading it the first way cost a working reasoning rung. It is back on the ladder under its real id.
- **What was genuinely absent**: `gemini-2.5-flash` and `gemini-2.5-flash-lite` appear in the rate-limit console **and** in ListModels, and 404 on every `generateContent` call for this key. A catalogue entry is not an availability guarantee — which is why the check that matters is a live call, not a listing.
- **What was being missed entirely**: `gemini-3.7-flash` exists, is newer than everything on the ladder, and was not in the registry at all. Nothing fails when a newer model ships; the pipeline quietly keeps running on the older one. It now leads the synthesis ladder, with a test asserting that it does.
- **Resolution**: registry rebuilt from the provider console for quotas and from live calls for availability. Synthesis is `3.7-flash → 3.6-flash → 3.5-flash → 3-flash-preview → 3.5-flash-lite → 3.1-flash-lite → local`; agents `3.5-flash-lite → 3.1-flash-lite → local`. Per key: 950 agent calls/day (~237 queries) and 76 reasoning-grade syntheses across four rungs.
- **Why the existing tests could not catch any of it**: every registry test checked *internal consistency* — laddered models have declared quotas, allowances fit under caps, ladders are ordered correctly. All of those passed throughout. Availability and freshness are network facts; consistency tests cannot see them.
- **Guarded by**: `TestLadderModelsExist` (pins ladders to the probed set, and records the wrong-id case explicitly so it is not re-derived), `test_console_quotas_are_recorded_faithfully`, `test_newest_reasoning_model_leads_synthesis`.
- **Impact**: src/llm/model_registry.py, config/litellm_config.yaml, README.md, tests/test_rate_limiter.py

## Decision 42: Model availability is per (key, model), not per model
- **Date**: Session 9
- **Context**: Decision 41 recorded `gemini-2.5-flash` and `gemini-2.5-flash-lite` as unreachable and dropped them. Probing every model against **every configured key** showed that was too coarse: both answer normally on keys 3 and 4, and return `404 "This model is no longer available to new users"` on keys 0 and 1. Google grandfathers existing keys when a model closes to new sign-ups, so the same model string is simultaneously valid and invalid depending on which credential carries it.
- **Why the existing failure taxonomy could not express this**: every branch retires the wrong scope. A 429 retires the (key, model) pair *for a day*; a 503 retires the *model* for every key, briefly; an auth error retires the *key* for every model. None of them says "this credential may never use this model" — permanent, and scoped to a single slot.
- **Resolution**: `is_model_unavailable_for_key()` classifies it and `mark_slot_unavailable()` retires that one slot for the process lifetime. Deliberately not persisted: it is a fact about credentials that a restart re-learns in one call, and writing it to Postgres would make a provider policy change sticky in a way nobody would think to clear.
- **Consequence**: both models are back on the ladders. Cost is one wasted call per unavailable slot per process; benefit is **+19 reasoning-grade syntheses and +19 agent calls per day on each grandfathered key**. Per key that is now 969 agent calls and 95 reasoning syntheses on a grandfathered key, 950 and 76 on a newer one.
- **Also confirmed**: key index 2 returns `401 invalid authentication credentials` for every model — the long-standing bad entry in `GEMINI_API_KEYS`, now pinned down as a whole-key problem rather than a model one.
- **Guarded by**: `TestPerKeyModelAvailability` — retiring a slot leaves the model usable on another key, leaves other models usable on that key, and selection actually skips the retired slot.
- **Impact**: litellm_wrapper.py, budget_tracker.py, answer_synthesizer.py, model_registry.py, config/litellm_config.yaml, README.md

## Decision 43: Two corrections to guards added earlier this session
- **Date**: Session 9
- **The uncited-answer guard was rejecting correct refusals.** Decision 35 added `_is_usable_answer()` to stop unsourced answers shipping. It demanded a citation marker whenever context was present — but an answer saying *"the transaction value is not stated in the provided context"* has nothing to cite. Asked for a multiple the corpus cannot support, `gemini-3.6-flash`, `gemini-3.5-flash` and `gemini-3-flash-preview` each declined correctly, each was scored a failed generation, and the ladder burned three scarce 20-RPD reasoning slots in a row, spent 183 seconds, and then accepted the *weakest* model's answer. A guard that spends the good models to reach the worst one is worse than no guard. An uncited answer now passes when it declines and is rejected when it asserts — the distinction that mattered from the start.
- **A 503 should not be retried on the same model.** `call_prose_agent` retried three times with backoff before raising, so `gemini-3.7-flash` returning 503 cost ~48s before the ladder could try `gemini-3.6-flash`, which answered first time. Since 503 is already classified as *model unavailable*, that retry is time spent waiting for the same answer. It now raises immediately and lets the ladder move.
- **Measured**: the same query went 183s to 113s to **75s and 50s** across the two fixes, answering on `gemini-3.6-flash` with full citations instead of degrading to `flash-lite`.
- **Guarded by**: `test_uncited_decline_is_allowed_through`, `test_uncited_assertion_is_still_rejected`, `TestUnavailableModelIsNotRetried`.
- **Impact**: answer_synthesizer.py, litellm_wrapper.py

## Decision 44: Live verification of the session's changes, and where heading cleanup stops
- **Date**: Session 9
- **What was checked**: six queries across financial, legal, multi-hop, comparative and control, plus a full pass through the Streamlit UI driven by AppTest against the live API.
- **Result**: all six answered correctly. Latency 13-36s (down from 75-183s before this session's routing fixes). Controls still refuse. The UI renders all five tabs, the deal context strip (9 documents, 19 risk signals, 7 high severity), the VALIDATED pill, and 26 citation rows.
- **Routing behaved as designed**: `gemini-3.7-flash` returned 503 twice, was put in cooldown, and the query descended to `gemini-3.6-flash` — costing one attempt rather than the three it cost before the fast-bail fix. The uncited-answer guard fired once, correctly: a 601-character uncited assertion was rejected and the retry produced an 11-citation answer that passed validation.
- **Not exercised live**: the per-key slot retirement from Decision 42. The ladder never descended far enough to reach `gemini-2.5-flash`, so that path remains unit-tested only. Worth stating rather than implying the whole mechanism has been seen working.
- **Citation headings, and the limit of fixing them in the viewer**: the panel was still showing prose the chunker had labelled as headings — `VERTEX CAPITAL PARTNERS LLC, a Delaware limited liability company (`. Rather than guess a rule, all 109 distinct `section_heading` values in the corpus were audited. Dropping long strings that end on a preposition, article, conjunction or dangling punctuation removes 24 with no legitimate heading caught; tightening the terminal-punctuation threshold to 70 characters removes 7 more, while keeping table rows that legitimately end in a period. 78 of 109 survive.
- **What is left, deliberately**: fragments that end on an ordinary noun — "Dr. James Wu, Chief Technology Officer, is named as sole or first inventor on 12" — are not distinguishable from a terse heading without parsing them. The thresholds are tuned to this corpus and openly labelled as such. More numbers here would be overfitting; the durable repair is per-heading granularity in the chunker (Decision 16).
- **One self-inflicted note**: killing the Streamlit process directly also took down the API, because `run_demo.py` supervises both and shuts one down when the other exits. That is the intended behaviour — a half-dead stack is worse than a stopped one — but it means restarts should go through the launcher.
- **Impact**: app/components/citation_viewer.py, tests/test_streamlit_ui.py

## Decision 45: A timeout is unavailability, not a transient fault
- **Date**: Session 9
- **Context**: after the quota reset, a live query returned **nothing at all** — the client's 300-second budget expired with no response. `gemini-3.7-flash` had begun timing out consistently, and a timeout matched none of the failure classifiers, so it fell through to the generic retry path: three attempts at `PROSE_TIMEOUT_SECONDS = 120` each. The ladder was never reached. The user got a dead request instead of an answer from the next rung down.
- **The reasoning is the same as for a 503**, which made the omission easy to miss: a model that has just consumed its entire deadline and produced nothing is not more likely to answer on an identical retry, and each retry costs another full deadline. Retrying a timeout is the single most expensive thing this pipeline can do.
- **Resolution**: `is_timeout_error()` classifies it, `call_prose_agent` raises on the first occurrence instead of retrying, and both ladders route it to the same short model cooldown as a 503 — skip the model, do not debit quota, try a different one.
- **Measured on the same query**: 300s and no answer → **140s** (one timeout, then `gemini-3.6-flash` answered with 4 citations) and **19s** on the next call, once the cooldown had taken `3.7-flash` out of rotation.
- **Guarded by**: `TestTimeoutIsTreatedAsUnavailability`, including an assertion that one timeout plus a fallback attempt still fits inside the client's request budget — the invariant that was actually violated.
- **Note on the model itself**: `gemini-3.7-flash` is new and currently unreliable — 503s earlier in the session, hard timeouts now. It stays at the top of the ladder because it answers when it is healthy, and the cooldown makes an unhealthy day cost one attempt rather than the request.
- **Impact**: litellm_wrapper.py, answer_synthesizer.py, tests/test_rate_limiter.py

## Decision 46: Model cooldown escalates; a fixed one is wrong in both directions
- **Date**: Session 9
- **Context**: with `gemini-3.7-flash` timing out persistently, the flat 90-second cooldown from Decision 35 expired between queries. Every query therefore re-tried it, paid the full 120-second timeout, descended to `gemini-3.6-flash` and answered — putting a 41-question evaluation on course for 95 minutes instead of 30, and spending a scarce 20-RPD slot each time to learn something already known.
- **Why not just lengthen it**: a long fixed cooldown strands a healthy model after a brief 503 spike, which is the case Decision 35 was written for. The two failure shapes need different treatment and a single constant cannot provide it.
- **Resolution**: the cooldown doubles per consecutive failure — 90s, 180s, 360s, capped at 30 minutes — and **one success clears the streak**, so a recovered model is back in rotation immediately rather than serving out an escalated penalty. Failure counts are per model, so a sick rung never drags a healthy one into backoff.
- **Observed live during the evaluation**: `gemini-3.7-flash` escalated 90s → 180s → 360s while `gemini-3.6-flash` and `gemini-3.5-flash` each took an independent 90s after their own single blips.
- **Guarded by**: `TestEscalatingModelCooldown` — doubling, the cap, per-model isolation, and that a success resets the streak rather than merely clearing the current cooldown.
- **Impact**: budget_tracker.py, answer_synthesizer.py

## Decision 47: The README is a README, not a second decision log
- **Date**: Session 9
- **Context**: the README had grown to 445 lines and was carrying the full narrative of every bug found — three competing A/B tables for one feature, a paragraph contradicting a table three screens earlier, and long war stories that belong here. A reader evaluating the project had to mine it for what the system actually does.
- **Resolution**: restructured to 333 lines. It now opens with a worked multi-hop answer showing real citations, then architecture, the ingestion and retrieval mechanisms, the quota engineering, results, and setup. Bug narratives collapsed to six short "engineering notes" that state the lesson and point here for the full account.
- **What was deliberately kept in full**: the five due-diligence ingestion mechanisms (PDF streaming, multi-page table stitching, the 4-representation table model, parent-child expansion, layout-aware headings), the RAG strategy list, and the technology stack table. These are the concrete substance of the system rather than commentary on it — an earlier draft compressed them and lost the specifics that make the work legible.
- **Also corrected**: stale figures throughout — the summary table still showed a superseded run, the run history listed four runs when six had happened, and the roadmap listed two completed items as outstanding. Verified programmatically that the quota table matches `MODEL_LIMITS`, the capacity figures match what the registry computes, and every headline number matches `RESULTS.md`.
- **Impact**: README.md

## Decision 48: Report both clean runs, not the better one
- **Date**: Session 9
- **Context**: the evaluation was re-run with every fix from this session in place — timeouts, 503 and timeout classification, escalating model cooldown, per-key availability, the category-filter fix, the corrected answer guard. It scored **80.1%**, against **86.3%** on the previous clean run. Both completed 41/41 with **zero answers lost upstream**, so both are valid measurements rather than one being spoiled.
- **What actually moved**: 24 of 35 questions unchanged, 3 improved, 8 regressed — on identical code and an identical index. The synthesis mix was slightly weaker (33 `gemini-3.6-flash` / 4 `3.5-flash` / 1 `3-flash-preview`, versus 38 all on `3.6-flash`), which accounts for part of it and not all.
- **The largest single move was traced rather than assumed**: `legal_05` went 100% → 0%. Retrieval was correct — `customer_contracts_schedule.txt` was in the retrieved context, exactly as on the good run. The synthesizer cited only the merger agreement and answered from its *definition* of a material contract instead of the schedule listing the contracts. A synthesis choice, not a retrieval failure.
- **Resolution**: the README reports both runs side by side, with the diff and the traced explanation. Quoting 86.3% alone would have been defensible — it is the more recent measurement of the older code path — and it would also have been the number chosen because it flattered. A document that spends several paragraphs warning that this metric moves with the provider cannot then quote its best draw from it.
- **What this says about the system**: end-to-end recall on 35 questions has a run-to-run spread of roughly six points with everything held constant. That is the honest resolution of the measurement, and it is why every retrieval-quality change in this project is measured on retrieval alone.
- **Impact**: README.md, RESULTS.md

## Decision 49: Delete the reranker threshold instead of wiring it in
- **Date**: Session 10
- **Context**: every entry in `RETRIEVAL_CONFIGS` carried a `reranker_threshold` (0.25–0.4), `requires_numerical_precision` raised it, and the Query Rewriter's prompt told the model to lower it when good results were being filtered out. Nothing read it. A review flagged it as a knob that claimed a behaviour the system did not have, and the first fix wired it into the executor with a floor of three chunks.
- **Measured before keeping it**: the new committed retrieval harness (`eval/`) ran production retrieval with and without the cut-off on the same index. With it, final-context fact coverage was **85.2%**; without it, **91.6%** (with decomposition: 87.1% → 92.6%). Recall@10 went 62.0% → 76.4%. `sum_03` fell from 100% to 0% under the threshold; `comp_02`'s only relevant chunk scored 0.04 against the parent question and was cut.
- **Why it hurts**: a cross-encoder score answers "is this passage about the question?", and a passage carrying one needed fact of four can score low on the whole question while being indispensable. A fixed cut-off discards exactly those. Irrelevant context is already handled downstream by the Quality Assessor, whose floors were calibrated on labelled data (Decision 17).
- **Resolution**: the threshold, its clamp bound, the precision bump and the rewriter instruction are deleted, and a test asserts no retrieval config reintroduces one. The baseline in `eval/baseline.json` was regenerated from the threshold-free code.
- **Also re-measured**: the +10.4pp decomposition gain quoted in Decision 37 came from an uncommitted harness on the old ingestion. On the committed harness and the reworked ingestion it is **+0.9pp** on fact coverage@10 (+1.0pp on the final context), one question improved and none regressed; multi-hop 83.3% → 86.7%. The README now quotes the committed figure.
- **Impact**: src/agents/retrieval_executor.py, src/agents/retrieval_strategy.py, src/llm/prompt_templates/query_rewriter.py, eval/baseline.json, README.md

## Decision 50: A local decision model for judgments — adopted where it measured better, and nowhere else
- **Date**: Session 11
- **Context**: several pipeline steps are typed judgments, not generation — is a risk disclosed, what kind of document is this, is a question about the deal, does a passage actually state the answer. They were made by regexes (with known negation and false-positive bugs) or by spending an LLM call. Two decision models were considered. TypeSafe's hosted Jev was rejected: paid after a one-off credit, sign-ups paused, it sends data-room text to a third party, its own docs list numbers, dates and negations as weaknesses, and the key on hand did not authenticate with TypeSafe at all (401), so it came from an unofficial reseller. Laya (Apache-2.0, local) was chosen, using the authors' fine-tuned `typed-decisions` checkpoint without further fine-tuning.
- **Method**: a labelled set per decision, thresholds and instruction phrasing chosen on a dev split, results reported on a held-out split, compared against the existing method; adopt only on a win, always keep the old path as the fallback.
- **Adopted**: risk signals as a regex + Laya hybrid (test F1 0.583 → 0.658; confirm-only on CPU, P 1.00, to keep ingest time acceptable on a 2-core host), document category (accuracy 0.77 → 0.95), a public query guard (23/27 junk prompts blocked, 0/124 genuine questions blocked), and the answerability gate (below).
- **Not adopted**: PII. Laya lowered precision on the test split and would have excluded from retrieval a chunk that golden question `sum_04` depends on — a regex false positive is visible; a model false positive silently removes evidence. The regex itself was fixed instead.
- **Answerability gate**: one Laya question — does `passage` state the information needed to answer `question`? — for the question and each sub-question over the top 3 reranked chunks; it may only *veto* admission (best score < 0.35), never admit, because Laya scores unrelated passages around 0.55. Phrasing and threshold chosen on a 58-question dev set of minimal pairs (1/28 → 10/28 unanswerable refused, 0/26 answerable vetoed). On the held-out golden set: controls refused 2 → 4 of 6 with none left to the LLM, answerables admitted unchanged (33, or 34 with decomposition) — one reasoning-model synthesis and one lite call saved per pass. The margin is thin (fin_03 at 0.39). On by default on GPU; `docker-compose.prod.yml` turns it off for the 2-core CPU host, where it measured 6.6 s per facet.
- **Tried and dropped**: a Laya fabrication signal for grading control answers — correct declines and fabricated answers scored in the same 0.47–0.60 band, so it was not added.
- **Honest limits**: the labelled sets are small and single-annotator; the regex fixes were written after reading the corpus, which flatters the fixed-rule baselines on corpus rows.
- **Impact**: src/decisions/, src/data_processing/{risk_signal_extractor,document_classifier,pii_detector,ingest_pipeline}.py, src/agents/quality_assessor.py, api/routes/query.py, eval/decisions/, eval/results/{query_guard,answerability_dev}.md, README.md

## Decision 51: One call layer for every agent, exact accounting — and why not LiteLLM's Router
- **Date**: Session 12
- **Context**: the failure taxonomy of Decisions 35, 42, 45 and 46 lived in two hand-written ladder loops (synthesis, verification). Agents 1, 5 and 6 selected a slot once and called it, so one 429 on that key failed the query — after 20s + 40s of backoff against the key that had refused. Separately, the quota count was wrong in both directions: retries inside the wrapper were free (up to three provider requests per debited unit), while a refused request kept its unit and *any* 429 set the slot's counter to its daily limit.
- **Measured before** (scripted provider, real tracker, virtual time; key 0 on the top rung fails the stated way, two keys): a 429 on Agents 1/5/6 → query fails after 3 calls and 60s; the same 429 on the judge → answered, but 476 units recorded for one served request; "every cloud model spent" on the judge → 19 calls, 360s asleep; malformed JSON on Agents 1/5/6 → query fails after three identical temperature-0 requests billed as one.
- **Router, evaluated against the installed litellm 1.89.3 with mocked providers**: `order` plus `fallbacks` reproduces "drain every key on model A, then model B", and a 429 rotates keys. Everything else this project's routing depends on is absent or opposite. There is no per-day limit — usage-based routing keys its counters by `%H-%M` with a 60s TTL, and RPD is the binding free-tier constraint. A 503 or timeout is retried on the *same* deployment (three attempts by default) and then on the *next key of the same model*: six calls on a sick model with two keys before falling back, where Decision 35 says a 503 is model-wide. Cooldowns are per deployment and constant (no escalation, no "until the daily reset", no permanent retirement), a single-deployment group is never cooled for a 429 or 503 under default settings, an HTTP-400 "API key not valid" is not cooled at all, and nothing links one key's deployments across models, so a dead key cannot be retired. State is in memory or Redis; there is no refund.
- **Resolution**: keep the quota layer, delete the duplication. `classify_failure` (src/llm/failures.py) returns exactly one failure class per exception, from LiteLLM's typed exceptions first and message text only to refine. `call_structured_agent` descends the agent ladder itself, so every caller gets the taxonomy without owning a loop; the verification path uses the same one. Accounting is settled at the call boundary: selection pays for one request, each retry pays for itself, and a request the provider refused is refunded.
- **Three changes of substance to the taxonomy**: (1) a 429 is split by the quota the provider names — per-minute rests the slot for the retry delay, only per-day spends it; (2) a caller's "this answer was no use" (`skip_model_for_request`) is scoped to its request, and the process-wide cooldown is fed only by the wrapper, which is the code that actually sees a 503 — one query whose answers were all rejected on content used to leave every synthesis model cooling for the next user; (3) the quota day is the provider's (midnight Pacific), not UTC.
- **Refund rule, and why it is narrow**: refunded on 503, per-minute 429, rejected key, revoked model and a connection that never got a response; **not** on a timeout or a 5xx. The provider's documentation does not say whether admitted-but-failed requests count, and the two errors are not symmetric: an under-count costs one wasted call before the next 429 corrects it, whereas an over-count on a 20-RPD slot is 5% of the day each time. The roadmap line said "refunded on 503/timeout"; the timeout half was deliberately not done.
- **Structured output**: JSON mode, then a parser that insists on an object, then one paid repair request (the model is shown its own reply, or given double the room if it was cut off), then another model for that request only. Agents 1 and 6 read the result through lenient Pydantic models, so a null filter dict or an unknown query type becomes a default instead of a crash, and Agent 1's filter proposals pass the same whitelist as the rewriter's. Gemini's native response schema was **not** adopted: LiteLLM maps it as `response_json_schema` for some rungs and `response_schema` for another, reports no schema support for three laddered models that are missing from its capability map, and none of that can be validated without a live call per rung.
- **Measured after** (same harness): every single-failure row for Agents 1/5/6 is answered in 2 calls with no sleep; a per-minute 429 records 1 unit instead of 476; "every cloud model spent" on the judge is 7 calls and no sleep; a 503 on every cloud model is one call per model, then local, with zero units recorded for zero served requests; malformed JSON is 3 calls, 3 units, answered on the next model.
- **Not the goal that was hoped for**: the brief was "with less code if possible". The wrapper and tracker grew — the duplicate ladder went, but refunds, charged retries, the per-minute split, request scoping and token metering are new behaviour, not rearrangement.
- **Guarded by**: `tests/test_llm_failures.py`, `tests/test_budget_accounting.py`, `tests/test_llm_routing.py`, `tests/test_agent_output_validation.py`; the cooldown tests in `tests/test_rate_limiter.py` now call `report_model_unavailable`, which owns the behaviour they assert.
- **Impact**: src/llm/{failures,schemas,litellm_wrapper,budget_tracker,model_registry,rate_limiter}.py, src/llm/prompt_templates/{query_intelligence,query_rewriter}.py, src/agents/{query_intelligence,query_rewriter}.py, config/litellm_config.yaml, README.md

## Decision 52: The keyword channel was case-sensitive and IDF-less
- **Date**: Session 12
- **Context**: FastEmbed 0.3's BM25 never lower-cases, keeps sentence-final periods on words and treats "$452.8M" or "EV/EBITDA" as one term, and the sparse vector was created without Qdrant's IDF modifier although the encoder relies on it. "What is the Revenue and EBITDA?" shared no term with "revenue and ebitda were strong". Found by reading the FastEmbed changelog (fixed upstream in 0.4.0), not by any test.
- **Resolution**: `normalize_for_bm25` runs on documents and queries alike, idempotent and independent of the FastEmbed version; collections are created with `Modifier.IDF` and existing ones upgraded in place; every chunk records the `sparse_encoding` that produced its vector, and `sparse_index_status()` logs at startup how many chunks predate the current one.
- **Measured** (35 answerable questions, paired bootstrap 95% CI): sparse-only fact coverage@10 73.1 → 94.3 (+21.2 [+10.0, +33.7]); hybrid without rerank 85.3 → 96.2 (+10.9 [+3.2, +20.6]); after rerank +3.6 [−3.4, +11.8] — within noise on the sample data room. With 60 real contracts (CUAD, ~3,300 chunks) added to the deal: production fact coverage@10 78.9 → 91.6 (+12.7 [+3.4, +23.6]), MRR@10 67.5 → 77.6 (+10.1 [+1.5, +20.6]). IDF alone: no measurable effect at either size.
- **Why it was invisible**: the cross-encoder repaired most of the damage on a 112-chunk corpus, and the harness reported means without intervals.
- **Impact**: src/vector_db/{hybrid_search,collection_manager}.py, src/data_processing/ingest_pipeline.py; existing indexes must be rebuilt (`python run_demo.py --reindex`).

## Decision 53: One rerank call per question, in half precision
- **Date**: Session 12
- **Context**: the cross-encoder was ~780 ms of an 800 ms retrieval pass and ran once per sub-question, one after another, so decomposed questions took 2.2–4.6 s in this node.
- **Resolution**: one embed call and one rerank call cover every pass of a question, each candidate still scored against the query that fetched it; pairs are sorted by length and scored in batches of 8; the cross-encoder runs in fp16 on CUDA with the activation in float32 (`RERANKER_FP16=0` turns it off).
- **Measured** (same harness, one job on 8 pinned cores): production p50 799 → 153 ms, p95 1069 → 187 ms; with decomposition p50 934 → 164 ms, p95 4078 → 530 ms. Rankings and refusal-gate decisions identical on every question; fp16 moves scores by at most 0.0034.
- **Rejected**: batching alone in fp32 (2.8 s → 2.6 s — the cost is padded tokens times precision, not call overhead); larger batches (more padding, slower); the Qdrant Query API (its weighted RRF is a different function from ours, and the candidate stage is ~20 ms of a ~150 ms pass).

## Decision 54: The Qdrant backend is configured, never guessed
- **Date**: Session 12
- **Context**: `get_qdrant_client()` probed the server port for one second and, if nothing answered, switched for the life of the process to an embedded database under `./qdrant_local_db`. That database starts empty: an API started while Qdrant was still booting came up healthy, accepted uploads the server never saw, and refused every question. A `qdrant_local_db/` directory in the developer checkout showed it had fired.
- **Resolution**: `QDRANT_MODE=server|local`, default server. An unreachable server fails startup with an error that names the switch; `get_qdrant_backend()` reports the live backend for `/ready`.
- **Impact**: src/vector_db/qdrant_client.py, .env.example, api/main.py (readiness).

## Decision 55: A before/after needs an interval, and a corpus big enough to disagree
- **Date**: Session 12
- **Context**: the retrieval harness printed means over 35 questions, where one question is worth up to 2.9 points. Bootstrap intervals on the committed results were ±8–14 points, and the decomposition gain quoted in Decision 49 rested on one question. The corpus is ~13K tokens (the README's "66K" was a character count), so retrieval could not be told apart from handing over the whole data room.
- **Resolution**: every mean carries a 95% bootstrap interval, every stage a paired per-question delta, `--compare` gives a paired before/after between two runs, a whole-corpus reference row shows the ceiling, the refusal gate is read from the router's own `route_after_quality_check`, and `--distractors N` adds real contracts (CUAD, CC BY 4.0, pinned revision, downloaded on first use, never committed). The verification layer got the same treatment (Decision 56), and the decision evals gained Wilson and clustered/paired bootstrap intervals (`eval/decisions/stats.py`).
- **Rejected on this evidence**: tuning RRF weights or k (every difference within one question), raising rerank depth (the candidate pool is flat at 98.1 against a 99.0 ceiling), bge-m3's learned sparse vectors instead of BM25 (80.4 vs 94.3 fact coverage@10), ColBERT as a third fusion list (91.4 vs 97.1).

## Decision 56: A figure is grounded only where the claim says it is
- **Date**: Session 12
- **Context**: an audit measured the share of *random* figures that numeric grounding accepted on sample-corpus contexts: integer percentages 48.5%, "$X.Y billion" 33.5%, multiples 16.2%. Percent and multiple claims were rescaled by 10^3–10^9, every unitless number in the context was indexed at four magnitudes ("2%" was grounded by "1,850 employees"), and a figure that appeared anywhere settled the claim — "Net income was $452.8M", where $452.8M is revenue, came back `supported`, confidence 1.0, with no judge call. Claims past the 40th were never checked, and any sentence containing "not disclosed" or "missing" was skipped. Nothing measured this layer: there was no labelled set.
- **Benchmark first**: `eval/verification_set.json` — 542 claims in 147 families, hand-written against `data/sample_deal`, every evidence excerpt checked verbatim by a test, split dev/test by family. Labels cover verbatim, paraphrase and arithmetic support; facts the room never states; and contradictions of every audited kind (changed figure, real figure on the wrong metric or period, swapped party, flipped negation, wrong unit, wrong arithmetic, false "X is not disclosed"). `eval/run_verification_eval.py` reports false pass and false alarm separately and gates the random-figure grounding rate.
- **Resolution**: ratios are never rescaled; amounts are read at the unit their document declares; a match must name the claim's row or sentence and period with no better-matching rival line, otherwise the claim goes to the judge; quantities ("24 months") are figures; every claim's figures are grounded, the 40-claim budget limits only NLI and judge work, and a claim past it is "not checked", never supported; statements of absence are checked, not skipped; a failed synthesis is not verified as an answer.
- **Measured** (deterministic stages, test half): wrong claims passed or never checked 40.9% → 6.7%; supported claims flagged 2.1% → 0.0%; a real figure attached to the wrong metric or period passed 70/73 → 12/73; random "$X.Y billion" grounded 33.8% → 2.1%. With the production NLI model in the loop, false pass 13.9% → 9.7%.
- **Cost, stated plainly**: fewer supported claims pass on local checks alone, so LLM-judge load rose from ~37% to ~65% of claims (lite-tier calls, not reasoning-tier).
- **Remaining weakness**: with NLI, 9–10% of wrong claims still pass, mostly flipped negations and swapped parties — `nli-deberta-v3-small` entails "$13.92M termination fee on a Superior Proposal" at 0.98 from the sentence giving $13.92M as the *reverse* fee. `-base` was measured and rejected (fewer false passes, more false alarms, 8× slower); purpose-trained checkers (MiniCheck, FactCG, HHEM) were not measured.
- **Impact**: src/verification/, src/agents/{hallucination_validator,answer_synthesizer}.py, src/utils/numerical_registry.py, src/llm/prompt_templates/{answer_synthesizer,hallucination_validator}.py, eval/verification_*

## Decision 57: Index table figures as the document states them
- **Date**: Session 12
- **Context**: the table converter multiplied every plain numeric cell by the table's scale and printed the product as document text. Under "($ in millions, except per share)" EPS 2.74 was indexed as "2,740,000" and a headcount of 1,250 as "1,250,000,000" — and numeric grounding then accepted those strings as source text: the system manufactured wrong numbers and then certified them. The grid rounded every cell to two decimals, growth and CAGR were computed over column positions rather than years, and `structured_rows` were built and dropped before the payload, so the cross-source figure registry saw nothing from any PDF, DOCX, PPTX or XLSX table. The sample corpus is all `.txt`, which bypasses the converter, so no eval could see any of this.
- **Resolution**: every representation prints each cell exactly as stated and states the unit once; the normalised value lives only in `structured_rows`, where only currency rows take the scale; growth is computed between fiscal years that are actually consecutive and CAGR over the real span, and nothing is emitted when the arithmetic is undefined. The registry reads each row's kind, precision and period, prefers the copy of a figure that knows its scale, and no longer applies a caption's scale to every row of a table's text — which had re-created the EPS defect on the reading side when the ingestion and verification branches were merged.
- **Also**: PDFs are rebuilt as visual rows so borderless 10-K statements become tables; encrypted, corrupt, scanned and binary files fail with a specific reason instead of indexing empty text; Word content controls and PowerPoint group shapes are read; a plain-text table keeps a digit-free column header with its rows; a retired document stays retired on re-index, a failed re-index keeps the serving copy, and `mark_superseded` reports whether it happened.
- **Guarded by**: generated stress documents (`tests/fixtures/stress_docs.py`) with hand-computed expectations, `tests/test_table_conversion.py`, `tests/test_ingestion_robustness.py`, `tests/test_ingest_versioning.py`, `tests/test_text_tables.py`.
- **Honest limits**: every format claim rests on generated fixtures, not real data-room files; borderless-table detection is heuristic; there is no OCR.

## Decision 58: The service tells the truth about its own state
- **Date**: Session 12
- **Context**: degraded modes were silent — an in-memory checkpointer, a local Qdrant or a missing NLI model all left `/ready` green. A client that disconnected mid-stream left the graph running and spending quota. Per-minute limits assumed one worker without enforcing it. Checkpoints were written on every superstep and never deleted, a reused `session_id` returned the previous query's trace, `/query/stream` sent raw exception text to public callers, and any caller could read any non-sandbox deal.
- **Resolution**: disconnects cancel the run on both query routes; runs and nodes have deadlines (504, or an SSE error with `code: "timeout"`); SIGTERM drains. More than one worker is refused at startup unless explicitly allowed — chosen over moving rate limits to Postgres because about eight pieces of per-process state, the LLM-routing windows among them, would have to be shared, and the VM cannot hold two copies of the models. Every fallback is recorded in `runtime_status` and shown by `/health` and `/ready`. Checkpoint threads are cleared before a run, deleted on completion, expire on failure and are purged with their deal, and thread ids can no longer collide. Public callers read only `PUBLIC_DEAL_IDS` and their own sandbox; error frames carry a request id, never exception text. Every node records its timing, LLM calls and tokens (`/metrics`, admin only), and a terminal `outcome` replaces the ambiguous `is_refusal`.
- **ARM deployment**: the image build was blocked by one package — `fastembed<0.4` requires `PyStemmer<3`, which has no aarch64 wheel, and its `setup.py` downloads sources with no timeout, which explains a build that hung silently. `Dockerfile.hf` now compiles it from checksum-pinned sources with bounded downloads on ARM only, pins CPU torch, and caches model layers before the code copy. Not yet built on aarch64.
- **Impact**: api/, src/workflow/, src/utils/{runtime_status,audit_log}.py, Dockerfile.hf, docker-compose*.yml, DEPLOYMENT.md

## Decision 59: Executive compensation is diligence content, not PII to hide
- **Date**: Session 12
- **Context**: the PII rework (audit I11) added a rule that flags a compensation table — named people with salary and bonus columns — and it fires correctly on the sample deal's executive pay and severance tables. PII-flagged chunks are excluded from retrieval, so two golden questions (severance multiples; cash needed at closing) would then be refused.
- **Decision**: not adopted. In an M&A data room, named-executive pay, change-of-control severance and 280G exposure are what a reviewer is there to read. The PII rules stay focused on identifiers that should never reach an answer: SSNs, account numbers, IBANs (mod-97 checked) and payment cards (Luhn checked).
- **Measured**: the rule flagged exactly the two labelled corpus chunks (executive compensation and severance tables) and no other sample-deal chunk; labelled F1 would have been 0.759 against 0.615 without it (+0.199 [0, +0.429]) — but those labels were written under the old policy, which counted executive pay as PII. Excluding the severance table would refuse golden questions comp_05 (compare severance multiples) and mh_06 (total cash needed at closing). It also flagged 2 of 3 adversarial title-case line items.
- **Impact**: src/data_processing/pii_detector.py

## Decision 60: LangGraph 1.x and FastEmbed 0.5+, which also unblocks the ARM image
- **Date**: Session 12
- **Context**: `langgraph<0.3` was kept because the graph is the project's whole control flow, at the cost of a standing "incompatible versions of langgraph and checkpoint-postgres" warning (requirements.txt carried the reasoning). `fastembed<0.4` turned out to be the ARM deployment blocker (Decision 58) and to carry the case-insensitive BM25 bug (Decision 52).
- **Measured** (an overlay environment, so the shared one was untouched until the result was in): langgraph 0.2.76 → 1.2.21 with langgraph-checkpoint-postgres 3.2.0 (langchain-core follows to 1.8, imported nowhere here), and fastembed 0.3.6 → 0.7.4. Offline suite unchanged with no code change (the one difference was the overlay's own path handling of pywin32). Retrieval eval, paired on the same questions: production and production + decomposition identical on every metric (final-context fact coverage 97.1 / 98.1, MRR@10 77.2 / 77.7), both refusal-gate tables identical, heuristic and + Laya; only the BM25-only diagnostic row moves (recall@5 66.3 → 65.4), within noise — the project normalises BM25 text itself, so the new stemmer changes almost nothing. `uv pip compile --only-binary :all:` resolves all 162 packages to wheels on aarch64 and x86_64.
- **Resolution**: `langgraph>=1.2,<2`, `langgraph-checkpoint-postgres>=3.2,<4`, `fastembed>=0.5,<0.8`, no explicit langchain-core pin. `Dockerfile.hf` drops the bounded PyStemmer source build and its compiler, and installs with `--only-binary :all:`, so a future pin that would need a source build fails the build instead of hanging it.
- **Not verified**: the Postgres saver under 3.2.0 (one live start in the final run) and an actual aarch64 build. Checkpoints written by 2.x are not read back — every run clears its thread first — so old checkpoint tables can be dropped.
- **Impact**: requirements.txt, Dockerfile.hf, DEPLOYMENT.md
