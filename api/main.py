"""
FastAPI application entry point with lifespan.

Manages startup/shutdown of Qdrant client, BudgetTracker, and LangGraph.
Immutable audit log on every query.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from src.vector_db.qdrant_client import (
    get_qdrant_client,
    close_qdrant_client,
    assert_server_compatible,
)
from src.vector_db.collection_manager import setup_collections
from src.llm.budget_tracker import BudgetTracker
from src.workflow.orchestrator import (
    get_compiled_graph,
    close_checkpointer,
    sweep_stale_checkpoints,
)
from src.utils import metrics, runtime_status
from src.utils.logger import setup_logger
from api.security import (
    REQUEST_ID_HEADER,
    BodySizeLimitMiddleware,
    RequestIdMiddleware,
    enforce_single_worker,
    is_admin,
    is_development,
    metrics_public,
    pipelines_in_flight,
    require_metrics_access,
)

logger = setup_logger(__name__)

# Windows defaults asyncio to ProactorEventLoop, which psycopg refuses to run
# on, so AsyncPostgresSaver raises at startup and checkpointing degrades to
# MemorySaver. See run_api.py for the full account.
#
# This covers callers that import the app before any loop exists (tests,
# scripts, `python -c`). It does NOT cover `uvicorn api.main:app`, because
# uvicorn imports the app from inside Server.serve(), by which point its loop is
# already running and a policy change has no effect — that path needs
# run_api.py. Setting it in both places is deliberate: neither one covers the
# other's entry.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Module-level reference to compiled graph
_app_graph = None

# Serialises rebuilds of the graph after a lost checkpointer connection, so a
# burst of requests that all notice the same dead connection reconnect once.
_rebuild_lock = asyncio.Lock()

_checkpoint_sweeper_task: asyncio.Task | None = None


def _postgres_url() -> str:
    return os.getenv(
        "POSTGRES_URL",
        "postgresql://manda_user:password@localhost:5432/manda_rag",
    )


async def _warm_local_models() -> None:
    """
    Loads the local models and records how each one came up.

    Every warmup function swallows its own failures by design — a warmup must
    not stop the API from starting. The cost of that design was that a failed
    load looked identical to a successful one until the first query paid for
    it. The outcome is now kept in runtime_status, so `/ready` and `/health`
    say which models are loaded and which are not.
    """
    from src.vector_db.reranker import warm_models
    from src.verification.nli import nli_enabled, warm_nli_model
    from src.decisions.laya_client import laya_enabled, warm_laya

    timings = await warm_models()
    for name in ("dense_embedding", "reranker", "sparse_bm25"):
        runtime_status.set_model(
            name,
            runtime_status.MODEL_READY if name in timings else runtime_status.MODEL_FAILED,
        )

    for name, enabled, warm in (
        ("nli", nli_enabled(), warm_nli_model),
        ("laya", laya_enabled(), warm_laya),
    ):
        if not enabled:
            runtime_status.set_model(name, runtime_status.MODEL_DISABLED)
            continue
        seconds = await warm()
        runtime_status.set_model(
            name,
            runtime_status.MODEL_READY if seconds is not None else runtime_status.MODEL_FAILED,
        )


# Two exact counts against Qdrant; cheap at demo scale but not something a
# healthcheck polling every 30s should repeat each time.
_SPARSE_STATUS_TTL_SECONDS = 60.0
_sparse_status_read_at: float = 0.0


async def _refresh_sparse_index_status(force: bool = False) -> None:
    """Re-reads the stale-BM25 count into runtime_status, at most once a minute."""
    global _sparse_status_read_at
    loop = asyncio.get_running_loop()
    if not force and loop.time() - _sparse_status_read_at < _SPARSE_STATUS_TTL_SECONDS:
        return
    _sparse_status_read_at = loop.time()
    try:
        from src.vector_db.collection_manager import sparse_index_status
        from src.vector_db import qdrant_client as qdrant_module

        status = await asyncio.wait_for(
            sparse_index_status(qdrant_module.get_qdrant_client()), timeout=5
        )
    except Exception as e:
        logger.warning(
            "Could not read the sparse index status", extra={"error_type": type(e).__name__}
        )
        status = None
    runtime_status.set_sparse_index(status)


# Ceiling on the startup identity check. It runs as a background task, so this
# bounds how long it may keep reading, not how long startup waits (it does not).
_IDENTITY_CHECK_TIMEOUT_SECONDS = 30.0
_identity_check_task: asyncio.Task | None = None


async def _check_identity_index() -> None:
    """
    Samples the index for documents the identity backfill would update and
    records the result in runtime_status (Decision 88). Read-only, bounded,
    and never the reason startup is slow or fails: it runs beside startup,
    and any error is logged and recorded as "not read".
    """
    try:
        from src.vector_db.collection_manager import identity_index_status
        from src.vector_db import qdrant_client as qdrant_module

        status = await asyncio.wait_for(
            identity_index_status(qdrant_module.get_qdrant_client()),
            timeout=_IDENTITY_CHECK_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(
            "Could not check the index for documents without identity",
            extra={"error_type": type(e).__name__},
        )
        status = None
    runtime_status.set_identity_index(status)


async def _checkpoint_sweeper_loop() -> None:
    """Sweeps expired checkpoint threads at startup and then on an interval."""
    interval = max(60, int(os.getenv("CHECKPOINT_SWEEP_INTERVAL_SECONDS", "3600")))
    while True:
        try:
            await sweep_stale_checkpoints(_app_graph)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Same reasoning as the sandbox sweeper: one bad pass must not end
            # the loop and silently turn retention off.
            logger.error(
                "Checkpoint sweep failed", extra={"error_type": type(e).__name__}
            )
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler — startup and shutdown logic.
    """
    global _app_graph, _checkpoint_sweeper_task, _identity_check_task

    logger.info("Starting M&A Due Diligence Intelligence Engine")
    runtime_status.set_phase("starting")

    # Before anything expensive: a second worker would load its own copy of
    # every model only to enforce limits that are not shared with the first.
    enforce_single_worker()

    # Initialize Qdrant
    client = get_qdrant_client()
    # Before anything writes: a client/server skew makes every gRPC upsert fail
    # while leaving reads and health checks working, so it must stop startup.
    await assert_server_compatible(client)
    await setup_collections(client)
    await _refresh_sparse_index_status(force=True)
    # Documents the identity backfill would update (Decision 86) — beside
    # startup, not in its way.
    _identity_check_task = asyncio.create_task(_check_identity_index())

    # Initialize budget tracker
    postgres_url = _postgres_url()
    await BudgetTracker.get_instance(postgres_url)

    # Compile LangGraph
    _app_graph = await get_compiled_graph(postgres_url)

    # Load the local models now rather than on the first query. Costs a slower
    # startup and removes a tens-of-seconds penalty from whichever question is
    # asked first. Set WARM_MODELS=0 to skip (faster restarts while developing).
    if os.getenv("WARM_MODELS", "1") != "0":
        await _warm_local_models()
    else:
        for name in ("dense_embedding", "reranker", "sparse_bm25", "nli", "laya"):
            runtime_status.set_model(name, runtime_status.MODEL_LAZY)

    from api.routes.deals import start_sandbox_sweeper, stop_sandbox_sweeper

    await start_sandbox_sweeper()
    _checkpoint_sweeper_task = asyncio.create_task(_checkpoint_sweeper_loop())

    runtime_status.set_phase("ready")
    logger.info("Application startup complete", extra=runtime_status.describe())

    yield  # Application runs here

    # Shutdown
    runtime_status.set_phase("draining")
    logger.info(
        "Shutting down application", extra={"pipelines_in_flight": pipelines_in_flight()}
    )
    for task in (_checkpoint_sweeper_task, _identity_check_task):
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    _checkpoint_sweeper_task = _identity_check_task = None
    await stop_sandbox_sweeper()
    # A PDF parse still running in its child process is stopped rather than waited
    # for, which could outlast the shutdown grace period (Decision 110).
    from src.data_processing import parse_isolation
    parse_isolation.shutdown()
    await close_checkpointer()
    await close_qdrant_client()
    await BudgetTracker.close()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="M&A Due Diligence Intelligence Engine",
    description="Production-grade Hybrid Agentic RAG for M&A Due Diligence",
    version="1.0.0",
    lifespan=lifespan,
)

# The Next.js frontend is served from a different origin than this API in every
# deployment (localhost:3000 vs :8000 in development, Vercel vs the container
# host in production), so browsers preflight every request. CORS_ORIGINS is a
# comma-separated allowlist; the default covers a local `npm run dev`.
#
# Deliberately not `allow_credentials=True`: this API has no cookie session, and
# the combination of credentials with a permissive origin list is the mistake
# that turns CORS from a control into a formality.
_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
if "*" in _cors_origins and not is_development():
    # Not refused: without credentials a wildcard exposes nothing a direct
    # request could not get. But it is almost never what was meant, and it lets
    # any page drive a visitor's browser against the rate limits.
    runtime_status.mark_degraded(
        "cors_wildcard",
        "CORS_ORIGINS contains '*': every website may call this API from a visitor's "
        "browser. Set it to the frontend's origin.",
    )

# Middleware runs outermost-last-added. The body limit is innermost so that its
# 413 still passes back out through CORS (a browser cannot read an error
# response that lacks the CORS headers) and picks up the request id.
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    # Lets the browser read the correlation id that generic error messages cite,
    # and how long to wait after a 429 or 503 (hidden from fetch() otherwise).
    expose_headers=[REQUEST_ID_HEADER, "Retry-After"],
)
app.add_middleware(RequestIdMiddleware)


def get_graph():
    """Returns the compiled LangGraph application."""
    return _app_graph


async def rebuild_graph():
    """
    Recompiles the graph on a fresh checkpointer connection.

    Called from the request path when the Postgres checkpointer reports its
    connection closed (see orchestrator.checkpointer_connection_broken). If
    Postgres is still down the rebuild lands on the in-memory fallback — which
    serves queries and is reported as degraded — rather than failing every
    request until someone restarts the container.

    Returns:
        The compiled graph now in use.
    """
    global _app_graph
    from src.workflow.orchestrator import checkpointer_connection_broken

    async with _rebuild_lock:
        # Another request may have rebuilt it while this one waited.
        if _app_graph is None or checkpointer_connection_broken(_app_graph):
            logger.error("Checkpointer connection lost; rebuilding the graph")
            await close_checkpointer()
            _app_graph = await get_compiled_graph(_postgres_url())
        return _app_graph


# Register routes
from api.routes.query import router as query_router
from api.routes.ingest import router as ingest_router
from api.routes.deals import router as deals_router

app.include_router(query_router, prefix="/api/v1", tags=["queries"])
app.include_router(ingest_router, prefix="/api/v1", tags=["ingestion"])
app.include_router(deals_router, prefix="/api/v1", tags=["deals"])


@app.get("/health")
async def health_check():
    """
    Liveness probe. Dependency-free: the keepalive workflow and the container
    healthcheck hit it, and a liveness check that fails when Qdrant blips would
    get a healthy process restarted. It is always 200 while the process serves.

    It is no longer silent about degradation, though. `degraded` lists every
    fallback this process is running on (in-memory checkpoints, a model whose
    warmup failed, ...) from state the process already holds — no I/O — so the
    one URL everybody checks cannot say "healthy" over a degraded service
    without also saying how.
    """
    status = runtime_status.describe()
    return {
        "status": "healthy",
        "service": "manda-rag",
        "phase": status["phase"],
        "uptime_seconds": status["uptime_seconds"],
        "degraded": status["degraded"],
        "checkpointer": status["checkpointer"]["backend"],
    }


@app.get("/ready")
async def readiness_check(request: Request):
    """
    Readiness probe: can this process actually answer a query?

    `checks` are the hard requirements — the LangGraph pipeline compiled
    (startup finished), Qdrant answering within a few seconds, and the process
    not draining for shutdown. Any false one gives 503, so a deploy can wait on
    this rather than on `/health`.

    Everything that is degraded-but-working is reported next to them without
    failing the probe: which Qdrant backend is in use (server or embedded, no
    credentials), whether indexed chunks carry an outdated BM25 encoding, which
    checkpointer is in use and whether it is durable, each local model's load
    state, and the list of active degraded modes with an explanation of each.

    Returns:
        JSON with overall status, each check's result, and the degraded detail.
    """
    from src.vector_db import qdrant_client as qdrant_module

    await _refresh_sparse_index_status()
    status = runtime_status.describe()
    checks = {
        "graph_compiled": _app_graph is not None,
        "qdrant_reachable": False,
        "accepting_queries": not runtime_status.is_draining(),
    }
    try:
        # Looked up at call time so a replaced client (tests, reconnects) is used.
        await asyncio.wait_for(qdrant_module.get_qdrant_client().get_collections(), timeout=5)
        checks["qdrant_reachable"] = True
    except Exception as e:
        logger.warning("Readiness check: Qdrant unreachable", extra={"error": str(e)})

    ready = all(checks.values())
    content = {"status": "ready" if ready else "not_ready", "checks": checks}
    # The detail names internal hosts, models and failure classes. Deploy
    # tooling only needs the status code; the operator presents the admin key.
    if not (is_admin(request) or metrics_public()):
        return JSONResponse(status_code=200 if ready else 503, content=content)
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            **content,
            "qdrant": qdrant_module.get_qdrant_backend(),
            "sparse_index": runtime_status.sparse_index(),
            "document_identity": runtime_status.identity_index(),
            "checkpointer": status["checkpointer"],
            "models": status["models"],
            "degraded": status["degraded"],
            "degraded_detail": status["degraded_detail"],
            "pipelines_in_flight": pipelines_in_flight(),
        },
    )


@app.get("/metrics", dependencies=[Depends(require_metrics_access)])
async def metrics_endpoint(request: Request):
    """
    Process metrics: request outcomes, per-node latency, LLM calls and tokens,
    in-flight pipelines, checkpointer durability and active degraded modes.

    Prometheus text format by default; `?format=json` returns the same series
    as JSON with a mean and bucketed p50/p95 per histogram, which is the easier
    form for a before/after latency comparison. Admin-only unless
    METRICS_PUBLIC=1.
    """
    if request.query_params.get("format") == "json":
        return JSONResponse(metrics.snapshot())
    return PlainTextResponse(
        metrics.render_prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )
