"""
Query routes — main ask endpoint with immutable audit logging.

Two transports over one pipeline: `/query` returns the finished result in a
single response, `/query/stream` pushes an event per agent as the graph runs.
Both end at the same `QueryResponse`, built by one shared function so the
streamed payload can never describe a different result than the blocking one.

Request lifecycle, the same on both transports:

  * A run stops when nobody is waiting for it. A closed tab used to leave the
    pipeline running to completion — up to three more LLM calls on a free-tier
    quota, for an answer with no reader. The stream is cancelled by the server
    when the connection drops; the blocking route polls for the disconnect.
  * A run has a deadline (QUERY_TIMEOUT_SECONDS). Expiry is a 504 / an `error`
    event with code "timeout", not a hung connection.
  * While the process is draining for shutdown, new runs are refused with 503.
"""

import asyncio
import contextlib
import json
import os
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from api.models.request_models import QueryRequest
from api.models.response_models import QueryResponse, Citation
from api.security import (
    ClientContext,
    acquire_pipeline_slot,
    public_error_detail,
    query_rate_limit,
    read_rate_limit,
    require_readable_deal,
    scrub_public,
)
from src.decisions.laya_client import LayaUnavailable
from src.decisions.query_guard import BLOCKED_MESSAGE, check_query, query_guard_enabled
from src.workflow.orchestrator import (
    checkpointer_connection_broken,
    node_timings,
    run_outcome,
    run_query,
    stream_query,
)
from src.utils import audit_log, metrics, runtime_status
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# How often the blocking route checks whether its client is still connected.
# The check is a non-blocking read of the ASGI receive channel, so the cost is
# negligible; the interval only bounds how long an abandoned run keeps going.
_DISCONNECT_POLL_SECONDS = 1.0

# Marks the end of the pipeline's event queue (see query_stream_endpoint).
_STREAM_DONE = object()


class _ClientDisconnected(Exception):
    """The caller of a blocking query went away before the pipeline finished."""


def _heartbeat_seconds() -> float:
    """Idle interval after which the stream sends a comment frame (0 = never)."""
    try:
        return max(0.0, float(os.getenv("SSE_HEARTBEAT_SECONDS", "15")))
    except ValueError:
        return 15.0


def _build_response(
    result: dict, session_id: str, request_id: str = "", public: bool = True
) -> QueryResponse:
    """
    Converts a terminal AgentState into the wire response.

    Args:
        result: Final state from the pipeline.
        session_id: Session ID this query ran under.
        request_id: Correlation id of the HTTP request.
        public: False for admin callers, who get diagnostics unredacted.

    Returns:
        Populated QueryResponse.
    """
    citations = [
        Citation(
            chunk_id=c.get("chunk_id", ""),
            source_file=c.get("source_file", ""),
            page_number=c.get("page_number"),
            section_heading=c.get("section_heading", ""),
            is_current_version=c.get("is_current_version", 1) == 1,
            content_type=c.get("content_type", "text"),
            is_redline=bool(c.get("is_redline", False)),
            superseded_by=c.get("superseded_by", ""),
        )
        for c in result.get("citations", [])
    ]

    # A refusal is any terminal state the pipeline reached without usable context —
    # either the quality gate forced it, or the rewrite loop exhausted its budget.
    is_refusal = bool(result.get("force_refusal", False)) or not result.get(
        "generated_answer", ""
    )

    timings = node_timings(result.get("agent_trace", []))

    # Diagnostic fields can carry exception text the agents recovered from;
    # only admin callers get it verbatim (see security.scrub_public).
    diagnostics = {
        "hallucination_flags": result.get("hallucination_flags", []),
        "claim_checks": result.get("claim_checks", []),
        "agent_trace": result.get("agent_trace", []),
    }
    if public:
        diagnostics = scrub_public(diagnostics)

    return QueryResponse(
        answer=result.get("generated_answer", ""),
        query_type=result.get("query_type", "summary"),
        confidence_score=result.get("confidence_score", 0.0),
        # A run that reached here without a status was not verified; it must
        # not be reported as "passed" (the frontends badge that as Verified).
        validation_status=result.get("validation_status") or "warning",
        citations=citations,
        hallucination_flags=diagnostics["hallucination_flags"],
        claim_checks=diagnostics["claim_checks"],
        total_latency_ms=result.get("total_latency_ms", 0.0),
        session_id=session_id,
        rewrite_iterations=result.get("rewrite_iteration", 0),
        agent_trace=diagnostics["agent_trace"],
        is_refusal=is_refusal,
        context_quality_score=result.get("context_quality_score", 0.0),
        outcome=result.get("outcome") or run_outcome(result),
        node_timings=timings,
        llm_calls=sum(row["llm_calls"] for row in timings),
        request_id=request_id,
    )


def _audit(request: QueryRequest, response: QueryResponse, transport: str) -> None:
    """
    Writes the audit record for a completed query (see src/utils/audit_log.py).

    Args:
        request: The originating request.
        response: The response being returned.
        transport: "blocking" or "stream" — which endpoint served it.
    """
    audit_log.record(
        "query_completed",
        **{
            "transport": transport,
            "deal_id": request.deal_id,
            "session_id": response.session_id,
            "request_id": response.request_id,
            "query": request.query,
            "query_type": response.query_type,
            "confidence_score": response.confidence_score,
            "validation_status": response.validation_status,
            "latency_ms": response.total_latency_ms,
            "citations_count": len(response.citations),
            "rewrite_iterations": response.rewrite_iterations,
            # Compliance: an authorized PII query must be attributable after the fact.
            "include_pii": request.include_pii,
            "is_refusal": response.is_refusal,
            "outcome": response.outcome,
            "llm_calls": response.llm_calls,
        },
    )


def _record(transport: str, outcome: str, started: float | None = None) -> None:
    """Counts one query outcome and, for runs that started, its duration."""
    metrics.QUERIES.inc(transport=transport, outcome=outcome)
    if started is not None:
        metrics.QUERY_SECONDS.observe(
            time.monotonic() - started, transport=transport, outcome=outcome
        )


async def _guard_public_query(request: QueryRequest, client: ClientContext) -> None:
    """
    Refuses off-topic, jailbreak and injection queries from public callers.

    Runs before a pipeline slot is taken, so a blocked query spends no Gemini
    call and no share of the daily cap. Admin callers are never judged. Fails
    open: if Laya is unavailable the query runs as it would have without the
    guard.

    Raises:
        HTTPException: 422 with a plain explanation when the query is blocked.
    """
    if client.is_admin or not query_guard_enabled():
        return
    try:
        verdict = await check_query(request.query)
    except LayaUnavailable as e:
        logger.warning("Query guard unavailable, allowing query", extra={"error": str(e)})
        return
    if verdict.blocked:
        logger.info(
            "Query blocked by guard",
            extra={
                "deal_id": request.deal_id,
                "reasons": verdict.reasons,
                "scores": verdict.scores,
                "request_id": client.request_id,
            },
        )
        raise HTTPException(status_code=422, detail=BLOCKED_MESSAGE)


def _require_graph():
    """Returns the compiled graph, or raises 503 if startup has not finished."""
    from api.main import get_graph

    graph = get_graph()
    if graph is None:
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    return graph


async def _usable_graph():
    """
    The compiled graph, rebuilt first if its checkpointer connection has died.

    Raises:
        HTTPException: 503 while the process is draining or not yet started.
    """
    if runtime_status.is_draining():
        raise HTTPException(
            status_code=503,
            detail="The engine is restarting. Please retry shortly.",
            headers={"Retry-After": "30"},
        )
    graph = _require_graph()
    if checkpointer_connection_broken(graph):
        from api.main import rebuild_graph

        graph = await rebuild_graph()
    return graph


def _enforce_pii_policy(request: QueryRequest, client: ClientContext) -> None:
    """
    Forces include_pii off for public callers.

    The flag is client-set, so on a public endpoint it is a request, not an
    authorization. Mutating the request (rather than passing a separate value)
    keeps the audit record truthful about what was actually served.
    """
    if request.include_pii and not client.is_admin:
        request.include_pii = False


def _timeout_detail(request_id: str) -> str:
    return f"The query exceeded the time limit and was stopped (request id: {request_id})."


async def _run_unless_disconnected(http_request: Request, pipeline):
    """
    Awaits a pipeline run, cancelling it if the HTTP client goes away.

    A blocking request gives the server no signal when its caller leaves: the
    handler is simply awaiting, and nothing is written until the end. So the
    run goes in its own task and the connection is polled alongside it.

    Args:
        http_request: The incoming request, for `is_disconnected()`.
        pipeline: The `run_query(...)` coroutine.

    Returns:
        The pipeline's result.

    Raises:
        _ClientDisconnected: the client closed the connection first.
    """
    task = asyncio.ensure_future(pipeline)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=_DISCONNECT_POLL_SECONDS)
            if done:
                return task.result()
            if await http_request.is_disconnected():
                raise _ClientDisconnected()
    finally:
        # Also reached when this handler is itself cancelled (shutdown).
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    http_request: Request,
    client: ClientContext = Depends(query_rate_limit),
):
    """
    Main query endpoint — runs the full agentic RAG pipeline.
    Logs every query to the immutable audit log.

    Args:
        request: QueryRequest with query, deal_id, optional session_id.
        http_request: The raw request, watched for a client disconnect.
        client: Caller context; resolving it enforces the per-client rate limit.

    Returns:
        QueryResponse with answer, citations, confidence, trace. 504 when the
        run exceeds its deadline.
    """
    require_readable_deal(request.deal_id, client.is_admin)
    graph = await _usable_graph()
    session_id = request.session_id or str(uuid.uuid4())
    _enforce_pii_policy(request, client)
    try:
        await _guard_public_query(request, client)
    except HTTPException:
        _record("blocking", "blocked")
        raise

    logger.info(
        "Query received",
        extra={
            "deal_id": request.deal_id,
            "session_id": session_id,
            "request_id": client.request_id,
            "query_length": len(request.query),
        },
    )

    slot = acquire_pipeline_slot(client)
    started = time.monotonic()
    try:
        result = await _run_unless_disconnected(
            http_request,
            run_query(
                app=graph,
                query=request.query,
                deal_id=request.deal_id,
                session_id=session_id,
                include_pii=request.include_pii,
            ),
        )
        if result.get("status") == "error":
            if result.get("error_kind") == "timeout":
                raise TimeoutError(result.get("error") or "pipeline timeout")
            raise ValueError(result.get("error", "Unknown pipeline error"))
    except _ClientDisconnected:
        _record("blocking", "cancelled", started)
        logger.info(
            "Query cancelled: client disconnected",
            extra={
                "deal_id": request.deal_id,
                "session_id": session_id,
                "request_id": client.request_id,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            },
        )
        # Nobody is left to read this; 499 is the conventional "client closed
        # request" status and keeps the access log honest.
        return Response(status_code=499)
    except TimeoutError:
        _record("blocking", "timeout", started)
        logger.error(
            "Query pipeline timed out",
            extra={"deal_id": request.deal_id, "request_id": client.request_id},
        )
        raise HTTPException(status_code=504, detail=_timeout_detail(client.request_id))
    except Exception as e:
        _record("blocking", "error", started)
        logger.error(
            "Query pipeline failed",
            extra={"error": str(e), "deal_id": request.deal_id, "request_id": client.request_id},
        )
        raise HTTPException(status_code=500, detail=public_error_detail("Query", client.request_id))
    finally:
        slot.release()

    response = _build_response(
        result, session_id, client.request_id, public=not client.is_admin
    )
    _record("blocking", "refusal" if response.is_refusal else "answered", started)
    _audit(request, response, transport="blocking")
    return response


def _sse(event: str, payload: dict) -> str:
    """
    Formats one Server-Sent Event frame.

    `json.dumps` is what makes this safe: the spec terminates a frame at a blank
    line, so a raw newline anywhere in the payload — and answers are full of
    them — would split one event into two malformed ones. JSON escaping collapses
    every newline to `\\n` before it can reach the wire.

    Args:
        event: Event name the client listens for.
        payload: JSON-serialisable body.

    Returns:
        An SSE frame, terminated by the required blank line.
    """
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


@router.post("/query/stream")
async def query_stream_endpoint(
    request: QueryRequest, client: ClientContext = Depends(query_rate_limit)
):
    """
    Streams pipeline progress as Server-Sent Events, then the finished result.

    The pipeline runs eight agents and takes tens of seconds, so the blocking
    endpoint leaves the client with nothing to show for most of the wait. This
    emits each agent's finding as it lands — what the query was classified as,
    how many chunks survived reranking, what the quality gate scored — and closes
    with the same QueryResponse `/query` would have returned.

    Event sequence: `start`, then one `stage` per completed agent — interleaved
    with `token` ({text}) frames while the answer is written and `answer_reset`
    ({reason}) if that draft must be discarded — then exactly one of `result`
    or `error`. Token frames are an unverified draft; `result` is the checked
    answer. They pass through the generic branch below unchanged.

    An `error` event is always {detail, code, session_id, request_id}: `detail`
    is generic text naming the request id, and `code` is "timeout" or
    "pipeline_error". Between events, an idle stream carries SSE comment frames
    (`: keepalive`) every SSE_HEARTBEAT_SECONDS so that proxies and mobile
    networks do not drop a connection that is merely waiting on a slow node;
    comment frames are not events and carry no data.

    Args:
        request: QueryRequest with query, deal_id, optional session_id.
        client: Caller context; resolving it enforces the per-client rate limit.

    Returns:
        StreamingResponse of `text/event-stream`.
    """
    require_readable_deal(request.deal_id, client.is_admin)
    graph = await _usable_graph()
    session_id = request.session_id or str(uuid.uuid4())
    _enforce_pii_policy(request, client)
    try:
        await _guard_public_query(request, client)
    except HTTPException:
        _record("stream", "blocked")
        raise
    # Taken before the response starts, so a saturated engine is a real 503
    # status rather than an in-band error event.
    slot = acquire_pipeline_slot(client)

    logger.info(
        "Streamed query received",
        extra={
            "deal_id": request.deal_id,
            "session_id": session_id,
            "request_id": client.request_id,
            "query_length": len(request.query),
        },
    )

    def error_frame(code: str) -> str:
        detail = (
            _timeout_detail(client.request_id)
            if code == "timeout"
            else public_error_detail("Query", client.request_id)
        )
        return _sse(
            "error",
            {
                "detail": detail,
                "code": code,
                "session_id": session_id,
                "request_id": client.request_id,
            },
        )

    async def event_source() -> AsyncIterator[str]:
        started = time.monotonic()
        # Stays "cancelled" unless the pipeline reaches a terminal event: the
        # only way out of this generator without one is the consumer closing it
        # (client disconnect) or the task being cancelled (shutdown).
        outcome = "cancelled"

        # The pipeline runs in its own task and hands events over a queue, so
        # this generator can wake up on a timer as well as on an event. Waiting
        # on the graph directly leaves no way to send a heartbeat during a
        # node that takes a minute, and no way to notice a dead connection
        # until the next event is written to it.
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)

        async def pump() -> None:
            try:
                async for item in stream_query(
                    app=graph,
                    query=request.query,
                    deal_id=request.deal_id,
                    session_id=session_id,
                    include_pii=request.include_pii,
                ):
                    await queue.put(item)
            except Exception as e:
                await queue.put(e)
            finally:
                await queue.put(_STREAM_DONE)

        producer = asyncio.ensure_future(pump())
        heartbeat = _heartbeat_seconds() or None
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=heartbeat)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if item is _STREAM_DONE:
                    break
                if isinstance(item, Exception):
                    raise item

                event, payload = item
                if event == "result":
                    response = _build_response(
                        payload, session_id, client.request_id, public=not client.is_admin
                    )
                    outcome = "refusal" if response.is_refusal else "answered"
                    _audit(request, response, transport="stream")
                    yield _sse("result", response.model_dump())
                elif event == "error":
                    # Rebuilt here rather than forwarded: whatever the
                    # orchestrator put in the payload, only a generic message
                    # and the request id go to the client.
                    code = payload.get("code") if isinstance(payload, dict) else None
                    code = code if code in ("timeout", "pipeline_error") else "pipeline_error"
                    outcome = "timeout" if code == "timeout" else "error"
                    yield error_frame(code)
                elif event == "stage" and not client.is_admin:
                    # Stage details repeat the validator's flags and notes.
                    yield _sse(event, scrub_public(payload))
                else:
                    yield _sse(event, payload)
        except Exception as e:
            # The status line is long gone by now — a 500 is not available, so the
            # only way to tell the client is an in-band error event.
            outcome = "error"
            logger.error(
                "Streamed query failed",
                extra={
                    "error": str(e),
                    "deal_id": request.deal_id,
                    "request_id": client.request_id,
                },
            )
            yield error_frame("pipeline_error")
        finally:
            if not producer.done():
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await producer
            slot.release()
            _record("stream", outcome, started)
            if outcome == "cancelled":
                logger.info(
                    "Streamed query cancelled: client disconnected",
                    extra={
                        "deal_id": request.deal_id,
                        "session_id": session_id,
                        "request_id": client.request_id,
                        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                    },
                )

    return StreamingResponse(
        event_source(),
        # Second release path: the generator's `finally` never runs if the
        # client drops before the first chunk. Release is idempotent.
        background=BackgroundTask(slot.release),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # Without this, nginx-style proxies buffer the whole response and
            # deliver every event at once at the end — which is precisely the
            # behaviour this endpoint exists to avoid.
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _without_key_breakdown(value):
    """Drops every `keys` entry (the per-API-key usage list) from a budget payload."""
    if isinstance(value, dict):
        return {k: _without_key_breakdown(v) for k, v in value.items() if k != "keys"}
    if isinstance(value, list):
        return [_without_key_breakdown(v) for v in value]
    return value


@router.get("/budget")
async def get_budget_status(
    http_request: Request, client: ClientContext = Depends(read_rate_limit)
):
    """
    Returns current API budget status for all models.

    Public callers get the per-model totals the UI shows. The per-key breakdown
    — how many Gemini keys are configured and how close each one is to
    exhaustion — is for admin callers only.
    """
    from src.llm.budget_tracker import BudgetTracker

    try:
        tracker = await BudgetTracker.get_instance()
        status = await tracker.get_budget_status()
        return status if client.is_admin else _without_key_breakdown(status)
    except Exception as e:
        request_id = client.request_id
        logger.error(
            "Budget status failed", extra={"error": str(e), "request_id": request_id}
        )
        raise HTTPException(
            status_code=500, detail=public_error_detail("Budget status", request_id)
        )
