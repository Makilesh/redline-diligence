"""
Guardrails for running the API as an unauthenticated public demo.

The deployed backend spends the owner's free-tier Gemini quota on every query,
and anyone who finds the URL can script it. Nothing here is user authentication
— there are no accounts — it is cost and availability control:

  * An optional admin key (`ADMIN_API_KEY`, sent as `X-Admin-Key`) that lifts
    every limit and is the only way to touch non-sandbox data.
  * Per-client rate limits, keyed on the client IP.
  * A process-wide cap on concurrent pipeline runs, which fails fast instead of
    queueing: on a 2 vCPU Space a queued request is a request that will time out
    at the proxy anyway, after holding a connection the whole time.
  * A global daily query cap — the last line of defence when per-IP limits are
    evaded by rotating addresses.
  * A per-request correlation id, so a generic error shown to a visitor can be
    matched to the full exception in the server log.

Everything is in-process and in-memory. That matches the deployment — one
uvicorn worker on one host — and a restart resetting the counters is harmless:
Gemini's own quotas are the hard ceiling, these only keep one visitor from
spending all of it.

Because the state is per-process, a second worker silently doubles every
limit and halves nothing: N workers means N concurrency gates, N daily caps and
N copies of each client's window. `enforce_single_worker()` therefore refuses
to start under a declared worker count above one, instead of leaving that as a
sentence in the README.

Configuration is read from the environment at call time rather than at import:
`.env` is loaded as a side effect of importing litellm, which may happen after
this module is imported, and tests need to change limits per case.
"""

import hmac
import itertools
import math
import os
import re
import secrets
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.utils import metrics, runtime_status
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

ADMIN_HEADER = "X-Admin-Key"
REQUEST_ID_HEADER = "X-Request-ID"

# A pipeline slot older than this is assumed leaked and reclaimed. The streaming
# endpoint releases its slot from a generator `finally`, which never runs if the
# client disconnects before the first byte is sent; without a reaper, each such
# disconnect would permanently shrink capacity until the Space restarted. Set
# well above the slowest real query (minutes on CPU) so a live run is never
# reclaimed out from under itself.
_SLOT_STALE_SECONDS = 15 * 60
# Kept above the run deadline by this margin. The two were independent numbers
# that happened to be equal; raising QUERY_TIMEOUT_SECONDS (a pinned eval run
# waits up to 35 minutes) would have had live runs reaped at 15.
_SLOT_STALE_MARGIN_SECONDS = 120


def _slot_stale_seconds() -> float:
    """Age at which a held slot is presumed leaked: never below the run deadline."""
    # Imported here: the orchestrator pulls in every agent, and this module is
    # imported by code that should not pay for that at import time.
    from src.workflow.orchestrator import query_timeout_seconds

    deadline = query_timeout_seconds()
    if not deadline:
        return _SLOT_STALE_SECONDS
    return max(_SLOT_STALE_SECONDS, deadline + _SLOT_STALE_MARGIN_SECONDS)


# ==============================================================================
# Configuration
# ==============================================================================


def _env_int(name: str, default: int) -> int:
    """Reads a non-negative integer env var, falling back on junk values."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(f"Ignoring non-integer {name}={raw!r}; using {default}")
        return default


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def is_development() -> bool:
    """True only when ENVIRONMENT is explicitly `development`.

    Unset counts as production. The Space has no `.env`, so a missing value
    must fail closed rather than hand every visitor admin rights.
    """
    return os.getenv("ENVIRONMENT", "").strip().lower() == "development"


# ==============================================================================
# Caller identity
# ==============================================================================


def is_admin(request: Request) -> bool:
    """
    Decides whether the caller may bypass limits and touch non-sandbox data.

    With ADMIN_API_KEY set, only a matching `X-Admin-Key` header qualifies. With
    it unset, every caller is admin in development (so the local UI, Streamlit
    and the eval harness keep working unchanged) and nobody is otherwise.

    Args:
        request: Incoming request.

    Returns:
        True if the caller has admin rights.
    """
    configured = os.getenv("ADMIN_API_KEY", "")
    if configured:
        presented = request.headers.get(ADMIN_HEADER, "")
        # Constant-time compare: the key is the only secret guarding deletes.
        return bool(presented) and hmac.compare_digest(
            presented.encode("utf-8"), configured.encode("utf-8")
        )
    return is_development()


def client_ip(request: Request) -> str:
    """
    Returns the address rate limits are keyed on.

    Behind the Hugging Face proxy every request arrives from the proxy's own
    address, so without X-Forwarded-For all visitors would share one bucket.
    The header is only honoured when TRUST_PROXY_HEADERS says a proxy is in
    front: without one, any client could set it and pick its own bucket.

    Args:
        request: Incoming request.

    Returns:
        Client IP string, or "unknown".
    """
    if _env_flag("TRUST_PROXY_HEADERS"):
        first_hop = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if first_hop:
            return first_hop[:64]
    return request.client.host if request.client else "unknown"


def get_request_id(request: Request) -> str:
    """Returns the correlation id RequestIdMiddleware assigned, minting one if absent."""
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = secrets.token_hex(8)
        request.state.request_id = request_id
    return request_id


@dataclass(frozen=True)
class ClientContext:
    """Who is calling, resolved once per request by the rate-limit dependency."""

    ip: str
    is_admin: bool
    request_id: str


def client_context(request: Request) -> ClientContext:
    """Builds the ClientContext for a request."""
    return ClientContext(
        ip=client_ip(request),
        is_admin=is_admin(request),
        request_id=get_request_id(request),
    )


async def require_admin(request: Request) -> None:
    """
    FastAPI dependency: rejects non-admin callers with 403.

    Args:
        request: Incoming request.

    Raises:
        HTTPException: 403 when the caller is not admin.
    """
    if not is_admin(request):
        raise HTTPException(
            status_code=403,
            detail=f"This operation requires the {ADMIN_HEADER} header.",
        )


def public_error_detail(operation: str, request_id: str) -> str:
    """
    The client-facing text for an internal failure.

    Exception strings carry file paths, Qdrant URLs and provider error bodies —
    none of which a public caller should see. The request id is what lets the
    owner find the real error in the log.

    Args:
        operation: What failed, e.g. "Query".
        request_id: Correlation id for this request.

    Returns:
        A generic message naming the request id.
    """
    return f"{operation} failed due to an internal error (request id: {request_id})."


# ==============================================================================
# Per-client rate limiting
# ==============================================================================

# Bucket name -> (per-minute env var, default, per-day env var, default).
# Upload defaults are lower than query defaults: ingestion does not touch Gemini
# but embeds on the Space's CPU, and a few large PDFs can pin it for minutes.
_BUCKETS: dict[str, tuple[str, int, str, int]] = {
    "query": ("RATE_LIMIT_QUERY_PER_MINUTE", 5, "RATE_LIMIT_QUERY_PER_DAY", 50),
    "ingest": ("RATE_LIMIT_INGEST_PER_MINUTE", 3, "RATE_LIMIT_INGEST_PER_DAY", 20),
    # Sandbox creation. It was the one unauthenticated write with no limit at
    # all: each POST /deals adds a record to an in-memory registry, so a loop
    # could grow the process until the host killed it.
    "deal": ("RATE_LIMIT_DEAL_PER_MINUTE", 10, "RATE_LIMIT_DEAL_PER_DAY", 100),
    # Listings, document/risk reads, purges and the budget panel. Each fans out
    # to Qdrant or Postgres (a deal listing is one facet per deal) and none was
    # limited. Generous, because the UI makes several of these per question.
    "read": ("RATE_LIMIT_READ_PER_MINUTE", 120, "RATE_LIMIT_READ_PER_DAY", 0),
}

_MINUTE = 60.0
_DAY = 24 * 60 * 60.0

# Bound on tracked clients before a full prune. Stale keys are otherwise only
# dropped when the same client returns, so a scan from many addresses would grow
# the table without limit.
_MAX_TRACKED_KEYS = 10_000


class _SlidingWindowLimiter:
    """Sliding-window counter per key, holding one timestamp per admitted hit."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}

    def reset(self) -> None:
        self._hits.clear()

    def hit(self, key: str, per_minute: int, per_day: int, now: float) -> float | None:
        """
        Records a hit unless it would exceed a limit.

        Args:
            key: Client/bucket key.
            per_minute: Allowed hits in any 60s window (0 = unlimited).
            per_day: Allowed hits in any 24h window (0 = unlimited).
            now: Current monotonic time.

        Returns:
            None if admitted, else seconds until the caller may retry.
        """
        if len(self._hits) > _MAX_TRACKED_KEYS:
            self._prune(now)

        window = self._hits.setdefault(key, deque())
        while window and now - window[0] >= _DAY:
            window.popleft()

        if per_day and len(window) >= per_day:
            return _DAY - (now - window[0])

        if per_minute:
            recent = [t for t in window if now - t < _MINUTE]
            if len(recent) >= per_minute:
                return _MINUTE - (now - recent[0])

        window.append(now)
        return None

    def _prune(self, now: float) -> None:
        for key in [k for k, w in self._hits.items() if not w or now - w[-1] >= _DAY]:
            del self._hits[key]


_limiter = _SlidingWindowLimiter()


def rate_limit(bucket: str):
    """
    Builds a FastAPI dependency enforcing one bucket's per-client limits.

    The dependency returns the ClientContext so the route can reuse the
    admin/IP/request-id resolution rather than repeating it.

    Args:
        bucket: Key into _BUCKETS ("query" or "ingest").

    Returns:
        An async dependency callable.
    """
    minute_var, minute_default, day_var, day_default = _BUCKETS[bucket]

    async def dependency(request: Request) -> ClientContext:
        ctx = client_context(request)
        if ctx.is_admin:
            return ctx

        retry_after = _limiter.hit(
            f"{bucket}:{ctx.ip}",
            per_minute=_env_int(minute_var, minute_default),
            per_day=_env_int(day_var, day_default),
            now=time.monotonic(),
        )
        if retry_after is not None:
            metrics.RATE_LIMITED.inc(bucket=bucket)
            logger.warning(
                "Rate limit exceeded",
                extra={"bucket": bucket, "client_ip": ctx.ip, "request_id": ctx.request_id},
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded for this demo. Please wait and try again.",
                headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
            )
        return ctx

    dependency.__name__ = f"{bucket}_rate_limit"
    return dependency


query_rate_limit = rate_limit("query")
ingest_rate_limit = rate_limit("ingest")
deal_rate_limit = rate_limit("deal")
read_rate_limit = rate_limit("read")


# ==============================================================================
# Pipeline concurrency + global daily cap
# ==============================================================================


class PipelineSlot:
    """A held pipeline slot. `release()` is idempotent, so it can be wired to
    more than one cleanup path (generator `finally` and a background task)."""

    def __init__(self, gate: "_PipelineGate", token: int) -> None:
        self._gate = gate
        self._token = token

    def release(self) -> None:
        self._gate._active.pop(self._token, None)

    def __enter__(self) -> "PipelineSlot":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class _PipelineGate:
    """Counts in-flight pipeline runs and pipeline runs per UTC day."""

    def __init__(self) -> None:
        self._active: dict[int, float] = {}
        self._tokens = itertools.count()
        self._day: str = ""
        self._day_count = 0

    def reset(self) -> None:
        self._active.clear()
        self._day, self._day_count = "", 0

    @property
    def in_flight(self) -> int:
        return len(self._active)

    def acquire(self, ctx: ClientContext, count_toward_daily_cap: bool = True) -> PipelineSlot:
        """
        Takes a slot or raises immediately.

        Admin runs are never rejected, but they still occupy a slot and count
        toward the daily total — they spend the same CPU and the same quota.

        Args:
            ctx: Caller context from the rate-limit dependency.
            count_toward_daily_cap: False for work that does not spend Gemini
                quota (ingestion), which should hold a slot but not a query.

        Returns:
            A PipelineSlot to release when the run ends.

        Raises:
            HTTPException: 429 when the daily cap is spent, 503 when every
                slot is busy.
        """
        now = time.monotonic()
        stale_after = _slot_stale_seconds()
        for token, started in list(self._active.items()):
            if now - started > stale_after:
                logger.warning("Reclaiming stale pipeline slot", extra={"slot": token})
                self._active.pop(token, None)

        utc_now = datetime.now(timezone.utc)
        today = utc_now.date().isoformat()
        if today != self._day:
            self._day, self._day_count = today, 0

        if not ctx.is_admin:
            daily_cap = _env_int("DAILY_QUERY_CAP", 300)
            if count_toward_daily_cap and daily_cap and self._day_count >= daily_cap:
                midnight = datetime.combine(
                    utc_now.date() + timedelta(days=1), datetime.min.time(), timezone.utc
                )
                metrics.RATE_LIMITED.inc(bucket="daily_cap")
                logger.warning(
                    "Global daily query cap reached",
                    extra={"cap": daily_cap, "request_id": ctx.request_id},
                )
                raise HTTPException(
                    status_code=429,
                    detail="The demo's daily query budget is spent. It resets at 00:00 UTC.",
                    headers={"Retry-After": str(int((midnight - utc_now).total_seconds()) + 1)},
                )

            max_concurrent = max(1, _env_int("MAX_CONCURRENT_PIPELINES", 2))
            if len(self._active) >= max_concurrent:
                metrics.RATE_LIMITED.inc(bucket="concurrency")
                raise HTTPException(
                    status_code=503,
                    detail="The engine is busy with other questions. Please retry shortly.",
                    headers={"Retry-After": "15"},
                )

        token = next(self._tokens)
        self._active[token] = now
        if count_toward_daily_cap:
            self._day_count += 1
        return PipelineSlot(self, token)


_gate = _PipelineGate()
metrics.PIPELINES_IN_FLIGHT.set_function(lambda: _gate.in_flight)


def pipelines_in_flight() -> int:
    """Pipeline slots currently held (queries and ingestions)."""
    return _gate.in_flight


def acquire_pipeline_slot(ctx: ClientContext, count_toward_daily_cap: bool = True) -> PipelineSlot:
    """Module-level entry point to the shared gate; see `_PipelineGate.acquire`."""
    return _gate.acquire(ctx, count_toward_daily_cap=count_toward_daily_cap)


def reset_limits() -> None:
    """Clears every counter. For tests; nothing in production calls it."""
    _limiter.reset()
    _gate.reset()


# ==============================================================================
# Sandbox deal guard (for the ingestion route)
# ==============================================================================


async def require_sandbox_deal(deal_id: str, request: Request) -> None:
    """
    Allows a write into `deal_id` only if the caller may write there.

    Admin may write anywhere. Everyone else may only write into a live sandbox
    deal — otherwise any visitor could inject documents into the curated demo
    corpus, or into a permanent deal, and those uploads would never be swept.

    Args:
        deal_id: Target deal.
        request: Incoming request (for the admin check).

    Raises:
        HTTPException: 422 for a malformed id, 403 for a non-sandbox target,
            410 for an expired sandbox.
    """
    # Imported here: deals.py imports this module at load time.
    from api.models.request_models import DEAL_ID_REGEX
    from api.routes.deals import is_sandbox_id, sandbox_is_expired

    if not DEAL_ID_REGEX.fullmatch(deal_id or ""):
        raise HTTPException(status_code=422, detail="Invalid deal_id.")
    if is_admin(request):
        return
    if not is_sandbox_id(deal_id):
        raise HTTPException(
            status_code=403,
            detail="Uploads are only accepted into a temporary sandbox deal.",
        )
    if sandbox_is_expired(deal_id):
        raise HTTPException(status_code=410, detail="This sandbox deal has expired.")


# ==============================================================================
# Read access to deals
# ==============================================================================

# The deal the hosted demo is built around (scripts/seed_demo.sh, run_demo.py).
DEFAULT_PUBLIC_DEAL_IDS = "aurora_vertex_2024"


def public_deal_ids() -> set[str]:
    """Deals any caller may read (env PUBLIC_DEAL_IDS, comma-separated)."""
    raw = os.getenv("PUBLIC_DEAL_IDS", DEFAULT_PUBLIC_DEAL_IDS)
    return {part.strip() for part in raw.split(",") if part.strip()}


def can_read_deal(deal_id: str, admin: bool) -> bool:
    """
    Whether a caller may query, list or read `deal_id`.

    The admin key used to guard writes only. Reads were open: any visitor could
    list every permanent deal and run the full pipeline over it, so the first
    real data room ingested on a public instance would have been public too.
    Public callers now reach exactly two kinds of deal — the ones named in
    PUBLIC_DEAL_IDS, and sandboxes, whose unguessable id is the capability.

    Args:
        deal_id: Deal being read.
        admin: Whether the caller holds admin rights.

    Returns:
        True if the read is allowed.
    """
    if admin:
        return True
    from api.routes.deals import is_sandbox_id

    return is_sandbox_id(deal_id) or deal_id in public_deal_ids()


def require_readable_deal(deal_id: str, admin: bool) -> None:
    """
    Raises 404 when the caller may not read `deal_id`.

    404 rather than 403: "exists but is private" is itself information, and it
    keeps a private deal indistinguishable from one that was never created.

    Raises:
        HTTPException: 404 for a deal the caller may not read.
    """
    if not can_read_deal(deal_id, admin):
        raise HTTPException(status_code=404, detail="Deal not found.")


# ==============================================================================
# Public response hygiene
# ==============================================================================

# Keys whose value is exception text by construction (`"error": str(e)` in the
# synthesizer's and validator's trace entries).
_EXCEPTION_KEYS = {"error", "exception", "traceback"}
_REDACTED = "internal error (details in the server log)"

# Human-readable notes that embed an exception after a fixed phrase:
#   "validation unavailable: RuntimeError: litellm.AuthenticationError: key=…"
#   "… LLM judge failed (APIConnectionError: http://10.0.0.7:6333 …)"
# The phrase is kept — it is the useful part — and everything after it is cut.
_EXCEPTION_TAIL = re.compile(
    r"(validation unavailable|LLM judge failed|judge failed|synthesis failed)\b.*",
    re.IGNORECASE | re.DOTALL,
)


# The only fields whose strings are rewritten. Deliberately narrow: answer text,
# claims and evidence quotes are document content and must never be cut because
# a sentence happens to contain one of the phrases above.
_NOTE_KEYS = {"hallucination_flags", "reason", "notes", "note", "warnings"}


def scrub_public(value, _key: str | None = None):
    """
    Removes internal exception text from a payload bound for a public caller.

    The error *responses* were already generic. The successful ones were not:
    agents record failures they recovered from in `agent_trace[].error`,
    `hallucination_flags` and claim notes, and those fields are returned
    verbatim — so a provider error body, with its hostnames and key fragments,
    reached the browser inside a 200. Admin callers still get everything.

    Args:
        value: Any JSON-like structure.

    Returns:
        A copy with exception-bearing keys replaced and, in note fields only,
        everything after a known failure phrase removed.
    """
    if isinstance(value, dict):
        return {
            k: (_REDACTED if k in _EXCEPTION_KEYS and v else scrub_public(v, k))
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [scrub_public(v, _key) for v in value]
    if isinstance(value, str) and _key in _NOTE_KEYS:
        return _EXCEPTION_TAIL.sub(r"\1", value)
    return value


# ==============================================================================
# Correlation id middleware
# ==============================================================================


class RequestIdMiddleware:
    """
    Assigns every HTTP request an id, exposed as `request.state.request_id`
    and echoed in the `X-Request-ID` response header.

    Pure ASGI rather than `@app.middleware("http")`: BaseHTTPMiddleware wraps
    the response body, which interferes with the SSE endpoint's streaming.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Always minted server-side: echoing a client-supplied id into logs would
        # let a caller forge correlation with someone else's request.
        request_id = secrets.token_hex(8)
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        await self.app(scope, receive, send_with_id)


# ==============================================================================
# Request body size limit
# ==============================================================================

INGEST_PATH = "/api/v1/ingest"

# Room for the multipart envelope around the file: boundaries, part headers and
# the four small form fields.
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024


def max_upload_bytes() -> int:
    """Upload size cap in bytes (env MAX_UPLOAD_MB, default 25)."""
    return _env_int("MAX_UPLOAD_MB", 25) * 1024 * 1024


def max_body_bytes(path: str) -> int:
    """
    The largest request body accepted on `path`.

    Args:
        path: Request path.

    Returns:
        Bytes: the upload cap plus envelope for the ingest route, and
        MAX_JSON_BODY_KB (default 64) everywhere else — the largest legitimate
        JSON body is a 2000-character query.
    """
    if path.rstrip("/") == INGEST_PATH:
        return max_upload_bytes() + _MULTIPART_OVERHEAD_BYTES
    return max(1, _env_int("MAX_JSON_BODY_KB", 64)) * 1024


class BodySizeLimitMiddleware:
    """
    Rejects oversized request bodies before they are buffered or spooled.

    The route-level upload cap ran too late to protect anything. Starlette
    parses a multipart body in full — spooling the file to the temp directory —
    before the route function is called, so a 10GB upload was written to disk
    in its entirety and only then answered with 413. JSON bodies had no cap at
    all: `/query` validates `query` to 2000 characters, but only after the whole
    body has been read into memory.

    This counts bytes as they arrive. A declared Content-Length over the limit
    is refused without reading anything; a body that lies about its length, or
    is chunked and declares none, is cut off at the limit.

    Pure ASGI for the same reason as RequestIdMiddleware: it must not buffer.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") in {"GET", "HEAD", "OPTIONS"}:
            await self.app(scope, receive, send)
            return

        limit = max_body_bytes(scope.get("path", ""))
        too_large = f"Request body exceeds the {_human_bytes(limit)} limit for this endpoint."

        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None:
            try:
                over = int(declared) > limit
            except ValueError:
                over = False
            if over:
                response = JSONResponse({"detail": too_large}, status_code=413)
                await response(scope, receive, send)
                return

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # An HTTPException, not a bare error: FastAPI re-raises those
                    # from its body parsing and turns everything else into a
                    # generic 400.
                    raise HTTPException(status_code=413, detail=too_large)
            return message

        await self.app(scope, counting_receive, send)


def _human_bytes(count: int) -> str:
    if count >= 1024 * 1024:
        return f"{count // (1024 * 1024)} MB"
    return f"{count // 1024} KB"


# ==============================================================================
# Single-worker enforcement
# ==============================================================================


def declared_worker_count(
    argv: list[str] | None = None, environ: dict[str, str] | None = None
) -> int:
    """
    How many worker processes this server was asked to run.

    A worker cannot count its siblings, but it can read what the launcher was
    told: uvicorn and gunicorn both honour WEB_CONCURRENCY, and both take
    `--workers N` / `-w N` on a command line that worker processes inherit.

    Args:
        argv: Command line (defaults to sys.argv).
        environ: Environment (defaults to os.environ).

    Returns:
        The largest worker count found; 1 when none is declared.
    """
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ

    found = [1]
    for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS", "API_WORKERS"):
        raw = (environ.get(name) or "").strip()
        if raw.isdigit():
            found.append(int(raw))
    for i, arg in enumerate(argv):
        value = None
        if arg in ("--workers", "-w") and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith("--workers="):
            value = arg.split("=", 1)[1]
        if value is not None and value.strip().isdigit():
            found.append(int(value))
    return max(found)


def enforce_single_worker() -> int:
    """
    Refuses to run as one of several workers, unless explicitly overridden.

    Rate limits, the concurrency gate, the daily cap, the deal registry and the
    model cooldowns all live in this process. With N workers each visitor gets
    N times every limit and the engine runs N times MAX_CONCURRENT_PIPELINES
    pipelines — on a host sized for one — and each worker also loads its own
    multi-GB copy of every model.

    ALLOW_MULTIPLE_WORKERS=1 downgrades the refusal to a recorded degraded
    mode, for a deployment that has put real shared limits in front of the API.

    Returns:
        The declared worker count.

    Raises:
        RuntimeError: when more than one worker is declared and not overridden.
    """
    workers = declared_worker_count()
    if workers <= 1:
        return workers
    if _env_flag("ALLOW_MULTIPLE_WORKERS"):
        runtime_status.mark_degraded(
            "multiple_workers",
            f"{workers} workers declared: rate limits, the concurrency gate and the daily "
            "cap are per process, so every limit is multiplied by the worker count.",
        )
        return workers
    raise RuntimeError(
        f"{workers} workers requested, but this API keeps its rate limits, concurrency "
        "gate and daily query cap in process memory and must run as a single worker. "
        "Start it with one worker (python run_api.py), or set ALLOW_MULTIPLE_WORKERS=1 "
        "to accept limits multiplied by the worker count."
    )


# ==============================================================================
# Metrics access
# ==============================================================================


async def require_metrics_access(request: Request) -> None:
    """
    FastAPI dependency guarding `/metrics`.

    Metrics name the models in use and how fast quota is being spent, and the
    production stack has no network boundary between this route and the
    internet (Caddy proxies everything). So it needs the admin key unless
    METRICS_PUBLIC=1 says a scraper on a private network is the only reader.

    Raises:
        HTTPException: 403 when the caller may not read metrics.
    """
    if _env_flag("METRICS_PUBLIC") or is_admin(request):
        return
    raise HTTPException(
        status_code=403, detail=f"Metrics require the {ADMIN_HEADER} header."
    )
