"""
LangGraph State Machine — M&A Due Diligence Query Pipeline.

CRITICAL RULES:
- All nodes are async def. Invoke with await app.ainvoke()
- PostgresSaver checkpoints state to (deal_id, session_id)
- Edge: rewriter → executor is EXPLICIT (not conditional)
- operator.add reducer for accumulating fields (picklable)
- Do NOT use lambda reducers — breaks PostgresSaver

State Machine Topology:
    START → query_intelligence → retrieval_executor → route_to_financial_verifier
    [financial_verifier | quality_assessor] → route_after_quality_check
    [answer_synthesizer | query_rewriter → retrieval_executor | insufficient_context]
    answer_synthesizer → hallucination_validator → route_after_validation → END
"""

import asyncio
import functools
import os
import time
import uuid
from contextlib import AsyncExitStack
from datetime import datetime, timedelta, timezone

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver

from src.workflow.state_definitions import AgentState
from src.workflow.conditional_edges import (
    _as_score,
    route_to_financial_verifier,
    route_after_quality_check,
    route_after_synthesis,
    route_after_validation,
)
from src.agents.query_intelligence import query_intelligence_node
from src.agents.retrieval_executor import retrieval_executor_node
from src.agents.financial_verifier import financial_verifier_node
from src.agents.quality_assessor import quality_assessor_node
from src.agents.query_rewriter import query_rewriter_node
from src.agents.answer_synthesizer import answer_synthesizer_node, synthesize_answer
from src.agents.hallucination_validator import hallucination_validator_node
from src.llm.litellm_wrapper import set_trace_context
from src.verification.claim_checker import revision_feedback
from src.utils import metrics, runtime_status
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Holds the open AsyncPostgresSaver context for the process lifetime.
# None when the MemorySaver fallback is in use.
_checkpointer_stack: AsyncExitStack | None = None


# ==============================================================================
# Configuration
# ==============================================================================


def _env_seconds(name: str, default: float) -> float:
    """Reads a non-negative number of seconds; junk or unset gives the default."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(f"Ignoring non-numeric {name}={raw!r}; using {default}")
        return default


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Whole-request ceiling. Before this existed a run had no deadline of its own:
# a provider that accepted the connection and never answered held a pipeline
# slot until the slot reaper took it back at 15 minutes — while the run itself
# kept going. 900s keeps that horizon, so nothing that completes today is cut.
DEFAULT_QUERY_TIMEOUT_SECONDS = 900.0


def query_timeout_seconds() -> float:
    """
    Deadline for one pipeline run, in seconds (env QUERY_TIMEOUT_SECONDS; 0 = none).

    With SYNTHESIS_MODEL_PIN set and no explicit timeout, the default stretches
    to cover the pin's own wait: a pinned measurement run is told to sit out a
    provider cooldown for up to SYNTHESIS_PIN_MAX_WAIT_SECONDS, and a 15-minute
    deadline would turn that deliberate wait into a failure.

    Returns:
        Seconds, or 0.0 when disabled.
    """
    if os.getenv("QUERY_TIMEOUT_SECONDS", "").strip():
        return _env_seconds("QUERY_TIMEOUT_SECONDS", DEFAULT_QUERY_TIMEOUT_SECONDS)
    if os.getenv("SYNTHESIS_MODEL_PIN", "").strip():
        pin_wait = _env_seconds("SYNTHESIS_PIN_MAX_WAIT_SECONDS", 1800.0)
        return max(DEFAULT_QUERY_TIMEOUT_SECONDS, pin_wait + 300.0)
    return DEFAULT_QUERY_TIMEOUT_SECONDS


def node_timeout_seconds() -> float:
    """
    Ceiling for a single graph step (env NODE_TIMEOUT_SECONDS; 0 = none, the default).

    Off by default because the right value is deployment-specific: the same
    rerank is 3s on a GPU and minutes on two CPU cores. The LLM calls inside a
    node already carry their own 60s/120s timeouts; this is the backstop for
    everything else a node does.
    """
    return _env_seconds("NODE_TIMEOUT_SECONDS", 0.0)


# ==============================================================================
# Node instrumentation
# ==============================================================================


def _attach_perf(result, scope: dict, node: str):
    """
    Adds a `perf` block to the trace entry a node returned.

    Every node appends exactly one `agent_trace` entry; the block goes on that
    entry so the trace stays one row per node execution. A node that returned
    no entry gets a minimal one rather than vanishing from the timings.

    Args:
        result: The node's state delta.
        scope: The finished metrics scope for this execution.
        node: Graph node name.

    Returns:
        The delta with `perf` attached; non-dict results pass through untouched.
    """
    if not isinstance(result, dict):
        return result
    perf = {
        "duration_ms": scope.get("duration_ms", 0.0),
        "llm_calls": scope["llm_calls"],
        "prompt_tokens": scope["prompt_tokens"],
        "completion_tokens": scope["completion_tokens"],
        "models": list(scope["models"]),
        # One per LLM call, in order; "length" is a reply cut off at max_tokens.
        "finish_reasons": list(scope.get("finish_reasons") or []),
    }
    trace = result.get("agent_trace")
    if isinstance(trace, list) and trace and isinstance(trace[-1], dict):
        trace = [*trace[:-1], {**trace[-1], "perf": perf}]
    else:
        trace = [*(trace or []), {"agent": node, "perf": perf}]
    return {**result, "agent_trace": trace}


def _instrument(node: str, fn):
    """
    Wraps a graph node so its latency and LLM usage are measured and reported.

    The measurement lives here rather than in each agent so that it is uniform
    (a node cannot forget to time itself) and so that both transports get it:
    the SSE path derived durations from the gaps between stream events, which
    folds the checkpoint write into the node, and `/query` had nothing at all.

    `functools.wraps` matters: LangGraph inspects a node's signature to decide
    whether to pass `config`, and must see the original's.

    Args:
        node: Graph node name.
        fn: The async node function.

    Returns:
        An async function with the same signature.
    """

    @functools.wraps(fn)
    async def instrumented(*args, **kwargs):
        with metrics.node_scope(node) as scope:
            result = await fn(*args, **kwargs)
        state = args[0] if args and isinstance(args[0], dict) else {}
        logger.info(
            "Node completed",
            extra={
                "node": node,
                "session_id": state.get("session_id"),
                "duration_ms": scope.get("duration_ms"),
                "llm_calls": scope["llm_calls"],
                "prompt_tokens": scope["prompt_tokens"],
                "completion_tokens": scope["completion_tokens"],
            },
        )
        return _attach_perf(result, scope, node)

    return instrumented


def node_timings(agent_trace: list[dict]) -> list[dict]:
    """
    Flattens a run's trace into one timing row per node execution, in order.

    Args:
        agent_trace: The accumulated `agent_trace` of a finished run.

    Returns:
        [{agent, duration_ms, llm_calls, prompt_tokens, completion_tokens, models}]
        for every entry that carries a `perf` block.
    """
    rows = []
    for entry in agent_trace or []:
        perf = entry.get("perf") if isinstance(entry, dict) else None
        if isinstance(perf, dict):
            rows.append({"agent": entry.get("agent", ""), **perf})
    return rows


async def insufficient_context_node(state: AgentState) -> dict:
    """
    Terminal node for queries where context is insufficient after max rewrites.
    Sets force_refusal=True so the synthesizer generates a refusal.

    Args:
        state: Current AgentState.

    Returns:
        State with force_refusal and generated refusal answer.
    """
    logger.info("Insufficient context — generating refusal")
    return {
        "force_refusal": True,
        "generated_answer": (
            "I was unable to find sufficient relevant information in the data room "
            "to answer this question, even after refining the search. This may mean "
            "the relevant documents haven't been uploaded yet, or the question falls "
            "outside the scope of the available materials.\n\n"
            f"Search attempts: {state.get('rewrite_iteration', 0) + 1}\n"
            f"Best quality score achieved: {_as_score(state.get('context_quality_score')):.2f}"
        ),
        "confidence_score": 0.0,
        "validation_status": "passed",  # No validation needed for refusal
        "agent_trace": [
            {"agent": "insufficient_context", "force_refusal": True}
        ],
    }


async def synthesis_unavailable_node(state: AgentState) -> dict:
    """
    Terminal node for a run whose synthesis failed on every model.

    The synthesizer has already written its apology as `generated_answer`.
    This marks the run as declined, so the response cannot present that text
    as a verified answer: no claim checks, confidence 0, `is_refusal` true.
    `outcome` ("synthesis_failed", see run_outcome) is what tells it apart from
    a refusal for lack of context.

    Args:
        state: Current AgentState.

    Returns:
        State delta marking the run as declined.
    """
    logger.warning("Synthesis unavailable — returning the degraded refusal unverified")
    return {
        "force_refusal": True,
        "confidence_score": 0.0,
        "hallucination_flags": [],
        "claim_checks": [],
        "validation_status": "passed",  # nothing was asserted, so nothing to fail
        "agent_trace": [{"agent": "synthesis_unavailable", "force_refusal": True}],
    }


# Terminal outcomes, one per way a run can end. The state carries several
# overlapping signals (force_refusal, an empty answer, validation_status, the
# synthesizer's failure mark) that callers used to combine themselves — and
# combined differently. This is the one place they are reduced to a verdict.
OUTCOME_ANSWERED = "answered"
OUTCOME_ANSWERED_UNVERIFIED = "answered_unverified"  # failed validation twice; shipped flagged
OUTCOME_REFUSED = "refused"  # no usable context, or the assessor said so
OUTCOME_SYNTHESIS_FAILED = "synthesis_failed"  # every model failed; apology returned


def run_outcome(final_state: dict) -> str:
    """
    Reduces a finished run's state to one terminal outcome.

    Args:
        final_state: The run's final state.

    Returns:
        One of the OUTCOME_* values.
    """
    agents = {
        entry.get("agent")
        for entry in final_state.get("agent_trace") or []
        if isinstance(entry, dict)
    }
    if "synthesis_unavailable" in agents:
        return OUTCOME_SYNTHESIS_FAILED
    if final_state.get("force_refusal") or not final_state.get("generated_answer"):
        return OUTCOME_REFUSED
    if final_state.get("validation_status") == "failed":
        return OUTCOME_ANSWERED_UNVERIFIED
    return OUTCOME_ANSWERED


async def retry_synthesis_node(state: AgentState) -> dict:
    """
    Re-runs synthesis after a failed validation, with the failures fed back.

    Re-running the identical prompt — what this node used to do — mostly
    reproduces the identical answer, since synthesis runs at temperature 0.1.
    The contradicted and unsupported claims from the validator are now appended
    to the prompt with an instruction to remove or correct them.

    Args:
        state: Current AgentState with claim_checks / hallucination_flags.

    Returns:
        Updated state from re-synthesis.
    """
    feedback = revision_feedback(
        state.get("claim_checks") or [], state.get("hallucination_flags") or []
    )
    logger.info(
        "Retrying synthesis after validation failure",
        extra={
            "attempt": state.get("validation_attempt", 0),
            "feedback_items": len(feedback),
        },
    )
    return await synthesize_answer(state, revision_feedback=feedback)


def build_graph() -> StateGraph:
    """
    Constructs the LangGraph StateGraph with all nodes and edges.
    Does NOT compile — call .compile(checkpointer=...) to get a runnable.

    Returns:
        Uncompiled StateGraph.
    """
    graph = StateGraph(AgentState)
    metrics.install_llm_usage_tap()

    # Add nodes — each wrapped so its latency and LLM usage reach the trace.
    for name, fn in (
        ("query_intelligence", query_intelligence_node),
        ("retrieval_executor", retrieval_executor_node),
        ("financial_verifier", financial_verifier_node),
        ("quality_assessor", quality_assessor_node),
        ("query_rewriter", query_rewriter_node),
        ("answer_synthesizer", answer_synthesizer_node),
        ("hallucination_validator", hallucination_validator_node),
        ("insufficient_context", insufficient_context_node),
        ("retry_synthesis", retry_synthesis_node),
        ("synthesis_unavailable", synthesis_unavailable_node),
    ):
        graph.add_node(name, _instrument(name, fn))

    # Set entry point
    graph.set_entry_point("query_intelligence")

    # Explicit edges
    graph.add_edge("query_intelligence", "retrieval_executor")

    # After retrieval → conditional: financial verifier or quality assessor
    graph.add_conditional_edges(
        "retrieval_executor",
        route_to_financial_verifier,
        {
            "financial_verifier": "financial_verifier",
            "quality_assessor": "quality_assessor",
        },
    )

    # After financial verifier → quality assessor (always)
    graph.add_edge("financial_verifier", "quality_assessor")

    # After quality check → conditional: synthesize, rewrite, or refuse
    graph.add_conditional_edges(
        "quality_assessor",
        route_after_quality_check,
        {
            "answer_synthesizer": "answer_synthesizer",
            "query_rewriter": "query_rewriter",
            "insufficient_context": "insufficient_context",
        },
    )

    # CRITICAL: rewriter → executor is EXPLICIT edge (self-correction loop)
    graph.add_edge("query_rewriter", "retrieval_executor")

    # After synthesis → validation, unless synthesis failed on every model
    after_synthesis = {
        "hallucination_validator": "hallucination_validator",
        "synthesis_unavailable": "synthesis_unavailable",
    }
    graph.add_conditional_edges("answer_synthesizer", route_after_synthesis, after_synthesis)
    graph.add_edge("synthesis_unavailable", END)

    # After validation → conditional: retry or end
    graph.add_conditional_edges(
        "hallucination_validator",
        route_after_validation,
        {
            "retry_synthesis": "retry_synthesis",
            "end": END,
        },
    )

    # Retry synthesis → validation again (same guard as the first synthesis)
    graph.add_conditional_edges("retry_synthesis", route_after_synthesis, after_synthesis)

    # Insufficient context → end (after generating refusal)
    graph.add_edge("insufficient_context", END)

    return graph


async def get_compiled_graph(postgres_url: str):
    """
    Compiles the graph with PostgresSaver checkpointer.
    Checkpoints are keyed to (deal_id, session_id) via the config dict.

    `AsyncPostgresSaver.from_conn_string()` returns an async *context manager*,
    not a saver — the connection must stay open for as long as the graph is
    used, so it is entered into a module-level AsyncExitStack that the API
    lifespan unwinds via close_checkpointer(). Awaiting `.setup()` on the
    context manager directly raises AttributeError and silently drops every
    deployment back to MemorySaver, losing crash recovery.

    Args:
        postgres_url: PostgreSQL connection string.

    Returns:
        Compiled LangGraph application ready for ainvoke().
    """
    global _checkpointer_stack

    graph = build_graph()
    stack = AsyncExitStack()
    try:
        # Bounded: a Postgres host that drops packets instead of refusing them
        # would otherwise hold startup for the OS connect timeout (minutes).
        checkpointer = await asyncio.wait_for(
            stack.enter_async_context(AsyncPostgresSaver.from_conn_string(postgres_url)),
            timeout=_env_seconds("CHECKPOINTER_CONNECT_TIMEOUT_SECONDS", 15.0) or None,
        )
        await checkpointer.setup()
        app = graph.compile(checkpointer=checkpointer)
        _checkpointer_stack = stack
        runtime_status.set_checkpointer("postgres")
        logger.info("LangGraph state machine compiled with PostgresSaver")
    except Exception as e:
        await stack.aclose()
        await close_checkpointer()
        if _env_flag("REQUIRE_DURABLE_CHECKPOINTS"):
            # The deployment said Postgres is not optional. Refusing to start is
            # the only response that cannot be missed: the alternative is a
            # service that looks healthy and forgets every in-flight run.
            logger.error(
                "PostgresSaver unavailable and REQUIRE_DURABLE_CHECKPOINTS is set; "
                "refusing to start",
                extra={"error_type": type(e).__name__, "error": str(e)},
            )
            raise RuntimeError(
                "Durable checkpointing is required (REQUIRE_DURABLE_CHECKPOINTS=1) but "
                f"the Postgres checkpointer could not be initialised: {type(e).__name__}"
            ) from e
        # Degrade rather than fail: the pipeline is fully functional without
        # durable checkpoints, it just cannot resume a session after a restart.
        # The degradation is recorded, not just logged — see runtime_status.
        logger.warning(
            f"Failed to initialize PostgresSaver: {e}. Falling back to MemorySaver."
        )
        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)
        runtime_status.set_checkpointer("memory", error=type(e).__name__)
        logger.info("LangGraph state machine compiled with MemorySaver")

    step_timeout = node_timeout_seconds()
    if step_timeout:
        app.step_timeout = step_timeout
    return app


async def close_checkpointer() -> None:
    """
    Closes the Postgres checkpointer connection, if one was opened.

    Called from the FastAPI lifespan shutdown. Safe to call when the fallback
    MemorySaver is in use — it is a no-op.
    """
    global _checkpointer_stack

    if _checkpointer_stack is None:
        return

    try:
        await _checkpointer_stack.aclose()
        logger.info("PostgresSaver checkpointer closed")
    except Exception as e:
        logger.warning(f"Error closing checkpointer: {e}")
    finally:
        _checkpointer_stack = None


def checkpointer_connection_broken(app) -> bool:
    """
    True when the graph's Postgres checkpointer has lost its connection.

    The saver holds ONE connection for the life of the process and never
    reconnects. If Postgres restarts — a deploy, an OOM kill, the host
    rebooting under it — every later checkpoint write raises, and since a
    checkpoint is written after every node, every query fails until the API is
    restarted. psycopg knows the connection is gone (`closed` is set as soon as
    an operation fails on it); this reads that flag without doing any I/O, so
    the request path can check it for free and rebuild the graph.

    Args:
        app: Compiled LangGraph application.

    Returns:
        True if the checkpointer exposes a connection that reports closed.
    """
    conn = getattr(getattr(app, "checkpointer", None), "conn", None)
    return bool(getattr(conn, "closed", False))


# ==============================================================================
# Checkpoint retention
# ==============================================================================
# A run writes one checkpoint per graph step (7-12 for a typical query), each a
# full copy of the state — retrieved chunk text included. Nothing ever deleted
# them: every query left a thread behind for good, and a client that reused a
# session id got something worse, because LangGraph applies a new run's input
# ON TOP of the thread's existing state. For plain fields that is an overwrite,
# but `agent_trace` and `rewrite_history` use an `operator.add` reducer, so the
# second question in a session returned the first question's trace as well
# (measured: 2 trace entries after run one, 4 after run two on the same thread),
# and the rewriter was handed the previous question's rewrite history.
#
# The policy now, in order of when it applies:
#   * before a run   — the thread is cleared, so a run always starts clean and a
#                      thread never holds more than one run;
#   * after success  — the thread is deleted (CHECKPOINT_KEEP_COMPLETED=1 keeps
#                      the finished run for inspection). The answer has been
#                      returned and audited; nothing reads the checkpoints again;
#   * after failure  — the thread is left alone: an interrupted run is the one
#                      case where the saved state has diagnostic value;
#   * on a TTL       — whatever is left is swept after CHECKPOINT_TTL_SECONDS;
#   * on deal purge  — a deal's threads go with it, so a visitor's uploaded text
#                      does not outlive their sandbox inside a checkpoint.
#
# Every operation here is best-effort: retention must never fail a query.


def thread_id_for(deal_id: str, session_id: str) -> str:
    """
    The checkpoint thread key for one (deal, session).

    Joined with ":" because both ids are validated to `[A-Za-z0-9_-]`, so the
    separator cannot occur inside either. The previous "_" could: the pairs
    ("sbx-1_a", "b") and ("sbx-1", "a_b") both produced "sbx-1_a_b", and two
    callers sharing a thread share its accumulated state.
    """
    return f"{deal_id}:{session_id}"


def _keep_completed() -> bool:
    return _env_flag("CHECKPOINT_KEEP_COMPLETED")


async def _delete_thread(app, thread_id: str, reason: str) -> bool:
    """
    Deletes one checkpoint thread, swallowing (and logging) any failure.

    Args:
        app: Compiled LangGraph application.
        thread_id: Thread to delete.
        reason: Why, for the metric and the log.

    Returns:
        True if the delete was issued without error.
    """
    checkpointer = getattr(app, "checkpointer", None)
    delete = getattr(checkpointer, "adelete_thread", None)
    if delete is None:
        return False
    try:
        await delete(thread_id)
    except Exception as e:
        logger.warning(
            "Checkpoint thread delete failed",
            extra={"thread_id": thread_id, "reason": reason, "error_type": type(e).__name__},
        )
        return False
    metrics.CHECKPOINT_THREADS_DELETED.inc(reason=reason)
    return True


async def _thread_last_seen(app) -> dict[str, str]:
    """
    Maps every checkpoint thread to the ISO timestamp of its newest checkpoint.

    Uses the saver's public `alist`, so it works on both backends. That reads
    whole checkpoints, which is only acceptable because the table is kept small
    by the policy above; it is called from the periodic sweep and from deal
    purges, never from the request path.
    """
    checkpointer = getattr(app, "checkpointer", None)
    if checkpointer is None or not hasattr(checkpointer, "alist"):
        return {}
    last_seen: dict[str, str] = {}
    async for item in checkpointer.alist(None):
        thread_id = (item.config.get("configurable") or {}).get("thread_id")
        if not thread_id:
            continue
        ts = str((item.checkpoint or {}).get("ts") or "")
        if ts > last_seen.get(thread_id, ""):
            last_seen[thread_id] = ts
        else:
            last_seen.setdefault(thread_id, ts)
    return last_seen


async def purge_deal_checkpoints(app, deal_id: str) -> int:
    """
    Deletes every checkpoint thread that belongs to a deal.

    Thread ids are `<deal_id>:<session_id>` (see thread_id_for), so the prefix
    `<deal_id>:` identifies the deal exactly. Threads written under the old
    `<deal_id>_<session_id>` form are not matched; the TTL sweep removes those.

    Args:
        app: Compiled LangGraph application (None is tolerated).
        deal_id: Deal being purged.

    Returns:
        Number of threads deleted.
    """
    if app is None:
        return 0
    prefix = f"{deal_id}:"
    deleted = 0
    try:
        threads = await _thread_last_seen(app)
    except Exception as e:
        logger.warning(
            "Could not list checkpoint threads for deal purge",
            extra={"deal_id": deal_id, "error_type": type(e).__name__},
        )
        return 0
    for thread_id in threads:
        if thread_id.startswith(prefix) and await _delete_thread(app, thread_id, "deal_purged"):
            deleted += 1
    return deleted


async def sweep_stale_checkpoints(app, max_age_seconds: float | None = None) -> int:
    """
    Deletes checkpoint threads whose newest checkpoint is older than the TTL.

    Args:
        app: Compiled LangGraph application (None is tolerated).
        max_age_seconds: TTL; defaults to CHECKPOINT_TTL_SECONDS (24h). 0 disables.

    Returns:
        Number of threads deleted.
    """
    if app is None:
        return 0
    if max_age_seconds is None:
        max_age_seconds = _env_seconds("CHECKPOINT_TTL_SECONDS", 24 * 60 * 60.0)
    if not max_age_seconds:
        return 0
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()
    deleted = 0
    for thread_id, ts in (await _thread_last_seen(app)).items():
        # ISO-8601 UTC timestamps order lexicographically. A thread with no
        # readable timestamp is left alone rather than guessed at.
        if ts and ts < cutoff and await _delete_thread(app, thread_id, "expired"):
            deleted += 1
    if deleted:
        logger.info("Checkpoint sweep complete", extra={"threads_deleted": deleted})
    return deleted


def _build_initial_state(
    query: str,
    deal_id: str,
    session_id: str,
    include_pii: bool,
) -> AgentState:
    """
    Builds the zero-value AgentState every run starts from.

    Extracted so `run_query` and `stream_query` cannot drift apart. A field
    missing from one but not the other is not a crash — TypedDict access on an
    absent key raises only where it is read — so the two would diverge silently
    until some agent read a key the streaming path never seeded.

    Args:
        query: User's natural language question.
        deal_id: Deal identifier for retrieval isolation.
        session_id: Thread key for checkpointing.
        include_pii: Compliance override from the authenticated caller.

    Returns:
        A fully populated initial AgentState.
    """
    return {
        "original_query": query,
        "current_query": query,
        "query_type": "summary",  # Will be overwritten by Agent 1
        "parsed_intent": {},
        "extracted_filters": {},
        "sub_questions": [],
        "retrieval_config": {},
        "dense_results": [],
        "sparse_results": [],
        "fused_results": [],
        "retrieval_passes": [],
        "reranked_results": [],
        "expanded_context": [],
        "context_quality_score": 0.0,
        "quality_breakdown": {},
        "quality_method": "heuristic",
        "missing_aspects": [],
        "answerability_veto": False,
        "rewrite_iteration": 0,
        "rewrite_history": [],
        "agent_trace": [],
        "numerical_registry": {},
        "inconsistencies": [],
        "generated_answer": "",
        "answer_truncated": False,
        "citations": [],
        "numerical_claims": [],
        "confidence_score": 0.0,
        "hallucination_flags": [],
        "claim_checks": [],
        "validation_status": "passed",
        "validation_attempt": 0,
        "force_refusal": False,
        "deal_id": deal_id,
        "session_id": session_id,
        "include_pii": include_pii,
        "total_latency_ms": 0.0,
        "status": "running",
        "error": None,
    }


async def run_query(
    app,
    query: str,
    deal_id: str,
    session_id: str | None = None,
    include_pii: bool = False,
    timeout_seconds: float | None = None,
) -> AgentState:
    """
    Executes a full query through the pipeline.

    Args:
        app: Compiled LangGraph application.
        query: User's natural language query.
        deal_id: Deal identifier for isolation.
        session_id: Optional session ID (generated if not provided).
        include_pii: Compliance override from the authenticated caller. Defaults
            to False so PII-flagged chunks stay excluded unless explicitly
            authorized. Never set from LLM output — see AgentState.include_pii.
        timeout_seconds: Deadline for the run. None uses QUERY_TIMEOUT_SECONDS;
            0 disables it.

    Returns:
        Final AgentState with answer and metadata. On failure, the initial
        state with status "error", `error`, and `error_kind` ("timeout" or
        "error"). Cancellation is not caught: a caller that cancels this
        coroutine (client gone, shutdown) gets CancelledError, and LangGraph
        cancels the node in flight on the way out.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    start = time.monotonic()

    initial_state = _build_initial_state(query, deal_id, session_id, include_pii)

    thread_id = thread_id_for(deal_id, session_id)
    config = {"configurable": {"thread_id": thread_id}}

    set_trace_context(deal_id=deal_id, session_id=session_id)

    logger.info(
        "Starting query pipeline",
        extra={"query": query, "deal_id": deal_id, "session_id": session_id},
    )

    # A run always starts on an empty thread — see "Checkpoint retention".
    await _delete_thread(app, thread_id, "reset")

    try:
        # CRITICAL: Use ainvoke() — NOT invoke() which deadlocks with async nodes.
        # asyncio.timeout(None) is a no-op, so a disabled deadline costs nothing.
        async with asyncio.timeout(_deadline_delay(timeout_seconds)):
            result = await app.ainvoke(initial_state, config=config)
        elapsed_ms = (time.monotonic() - start) * 1000
        result["total_latency_ms"] = round(elapsed_ms, 2)
        result["status"] = "completed"
        result["outcome"] = run_outcome(result)

        logger.info(
            "Query pipeline completed",
            extra={
                "session_id": session_id,
                "elapsed_ms": round(elapsed_ms, 2),
                "outcome": result["outcome"],
                "validation_status": result.get("validation_status"),
                "confidence_score": result.get("confidence_score"),
                **_run_summary(result),
            },
        )
        if not _keep_completed():
            await _delete_thread(app, thread_id, "completed")

        return result

    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        timed_out = isinstance(e, TimeoutError)
        logger.error(
            "Query pipeline timed out" if timed_out else "Query pipeline failed",
            extra={
                "session_id": session_id,
                "elapsed_ms": round(elapsed_ms, 2),
                # A step timeout raises a bare TimeoutError whose str() is empty.
                "error": str(e) or type(e).__name__,
                "error_type": type(e).__name__,
            },
        )
        initial_state["status"] = "error"
        initial_state["error"] = str(e) or type(e).__name__
        # Read by the route to answer 504 rather than 500. Not an AgentState
        # field: this dict never re-enters the graph.
        initial_state["error_kind"] = "timeout" if timed_out else "error"
        initial_state["total_latency_ms"] = round(elapsed_ms, 2)
        return initial_state


def _deadline_delay(timeout_seconds: float | None) -> float | None:
    """Resolves a run's deadline: explicit value, else the configured default; None = off."""
    if timeout_seconds is None:
        timeout_seconds = query_timeout_seconds()
    return timeout_seconds or None


def _run_summary(final_state: dict) -> dict:
    """Per-node latency and LLM-call totals of a finished run, for its log line."""
    timings = node_timings(final_state.get("agent_trace") or [])
    per_node: dict[str, float] = {}
    for row in timings:
        per_node[row["agent"]] = round(per_node.get(row["agent"], 0.0) + row["duration_ms"], 2)
    return {
        "node_ms": per_node,
        "llm_calls": sum(row["llm_calls"] for row in timings),
    }


# ==============================================================================
# Streaming
# ==============================================================================
# The pipeline takes tens of seconds end to end, which is a long time to show a
# spinner. Every node already reports what it did through `agent_trace`, so the
# same information can be pushed to the client as each node lands rather than
# being withheld until the last one finishes.

# Display metadata for the graph's nodes. Keyed by node name so an unknown node
# (added later, or one of LangGraph's internal keys) degrades to a title-cased
# label rather than disappearing from the timeline.
_STAGE_LABELS: dict[str, tuple[str, str]] = {
    "query_intelligence": (
        "Query Intelligence",
        "Classifying intent and decomposing the question",
    ),
    "retrieval_executor": (
        "Retrieval Executor",
        "Hybrid search, rank fusion and reranking",
    ),
    "financial_verifier": (
        "Financial Verifier",
        "Cross-checking figures against source tables",
    ),
    "quality_assessor": (
        "Quality Assessor",
        "Scoring whether the context can support an answer",
    ),
    "query_rewriter": ("Query Rewriter", "Reformulating after thin retrieval"),
    "answer_synthesizer": ("Answer Synthesizer", "Writing the cited answer"),
    "retry_synthesis": ("Answer Synthesizer", "Re-writing after a validation failure"),
    "hallucination_validator": (
        "Hallucination Validator",
        "Grounding every claim in retrieved text",
    ),
    "insufficient_context": ("Insufficient Context", "Declining rather than guessing"),
    "synthesis_unavailable": (
        "Synthesis Unavailable",
        "No model returned a usable answer; nothing to verify",
    ),
}

# The nodes that always run, in topological order. Sent up-front so the client
# can paint the whole timeline as pending instead of growing it a row at a time.
# Conditional nodes (verifier, rewriter, refusal) are appended only if reached.
_STAGE_ORDER: list[str] = [
    "query_intelligence",
    "retrieval_executor",
    "quality_assessor",
    "answer_synthesizer",
    "hallucination_validator",
]


def _summarize_stage(node: str, delta: dict) -> tuple[str, dict]:
    """
    Turns a node's state delta into a one-line summary and a small detail bag.

    The summary is what the client renders on the timeline row, so it is written
    to read as a finding ("decomposed into 3 sub-questions") rather than a status
    ("query_intelligence complete") — the point of showing the pipeline is that
    each step reports something the user could not have guessed.

    Args:
        node: Graph node name that just completed.
        delta: The partial state that node returned.

    Returns:
        (summary, detail) — detail holds the structured values behind the line.
    """
    # Every node appends exactly one trace entry; it carries the richest data.
    trace = (delta.get("agent_trace") or [{}])[-1]

    if node == "query_intelligence":
        subs = delta.get("sub_questions") or []
        qtype = delta.get("query_type", "summary")
        filters = delta.get("extracted_filters") or {}
        summary = f"Classified as {qtype}"
        if subs:
            plural = "s" if len(subs) > 1 else ""
            summary += f" - decomposed into {len(subs)} sub-question{plural}"
        return summary, {
            "query_type": qtype,
            "sub_questions": subs,
            "filters": {k: v for k, v in filters.items() if v},
            "model": trace.get("model", ""),
        }

    if node == "retrieval_executor":
        reranked = trace.get("reranked_count", 0)
        passes = trace.get("retrieval_passes", 0)
        sources = trace.get("sources", []) or []
        pass_plural = "es" if passes != 1 else ""
        doc_plural = "s" if len(sources) != 1 else ""
        summary = (
            f"{reranked} chunks reranked from {passes} retrieval pass{pass_plural} "
            f"across {len(sources)} document{doc_plural}"
        )
        return summary, {
            "reranked_count": reranked,
            "retrieval_passes": passes,
            "sub_questions": trace.get("sub_questions", []),
            "sources": sources,
            "elapsed_ms": trace.get("elapsed_ms", 0),
        }

    if node == "financial_verifier":
        inconsistencies = delta.get("inconsistencies") or []
        figures = trace.get("figures", len(delta.get("numerical_registry") or {}))
        cross = trace.get("cross_checked", 0)
        noun = "inconsistency" if len(inconsistencies) == 1 else "inconsistencies"
        summary = (
            f"{figures} figures read, {cross} cross-checked across sources - "
            f"{len(inconsistencies)} {noun} found"
        )
        return summary, {
            "figures": figures,
            "cross_checked": cross,
            "inconsistencies": inconsistencies,
            "method": trace.get("method", "deterministic"),
        }

    if node == "quality_assessor":
        score = _as_score(delta.get("context_quality_score"))
        missing = delta.get("missing_aspects") or []
        method = delta.get("quality_method", "heuristic")
        breakdown = delta.get("quality_breakdown", {}) or {}
        summary = f"Context quality {score:.2f} ({method})"
        if delta.get("answerability_veto"):
            # The reranker alone would have passed this; say why it did not.
            summary += " - relevant passages do not state the answer"
        elif missing:
            plural = "s" if len(missing) > 1 else ""
            summary += f" - {len(missing)} aspect{plural} missing"
        return summary, {
            "score": score,
            "method": method,
            "missing_aspects": missing,
            "breakdown": breakdown,
            "answerability": breakdown.get("answerability"),
            "answerability_veto": bool(delta.get("answerability_veto", False)),
        }

    if node == "query_rewriter":
        iteration = delta.get("rewrite_iteration", 0)
        return f"Reformulated the query (attempt {iteration + 1})", {
            "rewritten_query": delta.get("current_query", ""),
            "iteration": iteration,
        }

    if node in ("answer_synthesizer", "retry_synthesis"):
        citations = delta.get("citations") or []
        plural = "s" if len(citations) != 1 else ""
        verb = "Answer revised" if trace.get("revision") else "Answer drafted"
        summary = f"{verb} with {len(citations)} citation{plural}"
        if trace.get("revision"):
            items = trace.get("revision_items", 0)
            summary += f" - {items} flagged claim{'s' if items != 1 else ''} fed back"
        if delta.get("answer_truncated"):
            summary += " - cut off at the output limit"
        return summary, {
            "citation_count": len(citations),
            "answer_length": trace.get("answer_length", 0),
            "model": trace.get("model", ""),
            "revision": bool(trace.get("revision")),
            "answer_truncated": bool(delta.get("answer_truncated")),
        }

    if node == "hallucination_validator":
        # Missing means the validator did not decide; never report that as passed.
        status = delta.get("validation_status") or "warning"
        flags = delta.get("hallucination_flags") or []
        confidence = delta.get("confidence_score", 0.0)
        counts = trace.get("status_counts") or {}
        methods = trace.get("method_counts") or {}
        checked = trace.get("claims_checked", 0)
        summary = f"Validation {status} - confidence {confidence:.0%}"
        if trace.get("validation_unavailable"):
            summary += " - verifier unavailable, answer kept"
        elif checked:
            summary += f" - {counts.get('supported', 0)}/{checked} claims supported"
            bad = counts.get("contradicted", 0) + counts.get("unsupported", 0)
            if bad:
                summary += f", {bad} flagged"
            used = ", ".join(f"{k} {v}" for k, v in sorted(methods.items()))
            if used:
                summary += f" ({used}"
                summary += f"; {trace.get('llm_calls', 0)} LLM call"
                summary += "s)" if trace.get("llm_calls", 0) != 1 else ")"
        return summary, {
            "validation_status": status,
            "confidence_score": confidence,
            "hallucination_flags": flags,
            "status_counts": counts,
            "method_counts": methods,
            "numeric_counts": trace.get("numeric_counts") or {},
            "claim_checks": trace.get("claim_checks") or [],
            "nli_model": trace.get("nli_model"),
            "llm_model": trace.get("llm_model"),
            "llm_calls": trace.get("llm_calls", 0),
            "model": trace.get("model", ""),
        }

    if node == "insufficient_context":
        return "Refused - no passage in the data room supports an answer", {
            "force_refusal": True,
        }

    if node == "synthesis_unavailable":
        return "Declined - the language models did not return a usable answer", {
            "force_refusal": True,
        }

    return f"{node} completed", {}


async def stream_query(
    app,
    query: str,
    deal_id: str,
    session_id: str | None = None,
    include_pii: bool = False,
    timeout_seconds: float | None = None,
):
    """
    Runs the pipeline, yielding an event per completed node then the final state.

    `timeout_seconds` is the run's deadline (None = QUERY_TIMEOUT_SECONDS, 0 =
    off). On expiry, or on any pipeline failure, the stream ends with an "error"
    event carrying a generic `detail` and a `code` ("timeout" or
    "pipeline_error") — never the exception text.

    Uses `stream_mode=["updates", "values"]`: "updates" carries each node's own
    delta — the only place per-node output is visible — while "values" carries
    the accumulated state, whose last emission is the final result. Requesting
    both avoids re-implementing LangGraph's reducers to rebuild the end state
    from deltas, which is exactly where a hand-rolled accumulator would go wrong:
    `agent_trace` uses an `operator.add` reducer, not replacement, so naive
    dict-merging of deltas would keep only the last agent's trace entry.

    Args:
        app: Compiled LangGraph application.
        query: User's natural language query.
        deal_id: Deal identifier for isolation.
        session_id: Optional session ID (generated if not provided).
        include_pii: Compliance override from the authenticated caller.

    Answer tokens are streamed too: the run is started with
    `configurable.stream_tokens=True` and stream mode "custom", which the
    synthesizer detects and answers by streaming its completion through
    LangGraph's stream writer (see answer_synthesizer._token_sink). run_query
    sets neither, so the blocking path is unchanged.

    Yields:
        (event_name, payload) tuples: one "start", many "stage" interleaved with
        "token" ({text}) and "answer_reset" ({reason}) while the answer is being
        written, then exactly one of "result" or "error". Tokens are a draft;
        the validated answer arrives in "result".
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    start = time.monotonic()
    initial_state = _build_initial_state(query, deal_id, session_id, include_pii)
    thread_id = thread_id_for(deal_id, session_id)
    config = {
        "configurable": {
            "thread_id": thread_id,
            "stream_tokens": True,
        }
    }
    set_trace_context(deal_id=deal_id, session_id=session_id)
    delay = _deadline_delay(timeout_seconds)
    deadline = None if delay is None else asyncio.get_running_loop().time() + delay

    logger.info(
        "Starting streamed query pipeline",
        extra={"query": query, "deal_id": deal_id, "session_id": session_id},
    )

    yield (
        "start",
        {
            "session_id": session_id,
            "planned_stages": [
                {
                    "agent": node,
                    "label": _STAGE_LABELS[node][0],
                    "description": _STAGE_LABELS[node][1],
                }
                for node in _STAGE_ORDER
            ],
        },
    )

    final_state: dict | None = None
    last_mark = start
    seq = 0

    # A run always starts on an empty thread — see "Checkpoint retention".
    await _delete_thread(app, thread_id, "reset")

    events = app.astream(
        initial_state, config=config, stream_mode=["updates", "values", "custom"]
    ).__aiter__()

    try:
        while True:
            # The deadline wraps each wait on the graph, never a `yield`: a
            # timeout scope left open across a yield would fire while the
            # consumer — not this generator — is the code that is running.
            try:
                async with asyncio.timeout_at(deadline):
                    mode, chunk = await events.__anext__()
            except StopAsyncIteration:
                break

            if mode == "values":
                final_state = chunk
                continue

            if mode == "custom":
                # Emitted by the synthesizer: {"type": "token", "text": ...} or
                # {"type": "answer_reset", "reason": ...}.
                if isinstance(chunk, dict) and chunk.get("type") == "token":
                    yield ("token", {"text": chunk.get("text", "")})
                elif isinstance(chunk, dict) and chunk.get("type") == "answer_reset":
                    yield ("answer_reset", {"reason": chunk.get("reason", "")})
                continue

            # mode == "updates": {node_name: delta}. One key in practice, but the
            # loop keeps parallel-branch support free if the graph ever gains one.
            for node, delta in (chunk or {}).items():
                if not isinstance(delta, dict):
                    continue
                now = time.monotonic()
                summary, detail = _summarize_stage(node, delta)
                label, description = _STAGE_LABELS.get(
                    node, (node.replace("_", " ").title(), "")
                )
                seq += 1
                yield (
                    "stage",
                    {
                        "seq": seq,
                        "agent": node,
                        "label": label,
                        "description": description,
                        "summary": summary,
                        "detail": detail,
                        "duration_ms": round((now - last_mark) * 1000, 2),
                        "elapsed_ms": round((now - start) * 1000, 2),
                        # The node's own measurement. `duration_ms` above is the
                        # gap between stream events, so it also contains the
                        # checkpoint write; the difference is that overhead.
                        "perf": (delta.get("agent_trace") or [{}])[-1].get("perf"),
                    },
                )
                last_mark = now

    except Exception as e:
        timed_out = isinstance(e, TimeoutError)
        logger.error(
            "Streamed query pipeline timed out" if timed_out else "Streamed query pipeline failed",
            extra={
                "session_id": session_id,
                "elapsed_ms": round((time.monotonic() - start) * 1000, 2),
                "error": str(e) or type(e).__name__,
                "error_type": type(e).__name__,
            },
        )
        # The exception text stays in the log. It used to be the event's
        # `detail`, and the route forwards events verbatim — so a Qdrant URL or
        # a provider error body went straight to a public browser.
        yield (
            "error",
            {
                "detail": "The pipeline timed out." if timed_out else "The pipeline failed.",
                "code": "timeout" if timed_out else "pipeline_error",
                "session_id": session_id,
            },
        )
        return
    finally:
        # Reached on normal exit, on error, and when the consumer stops early
        # (client disconnect closes this generator at a yield). Closing the
        # graph stream is what cancels a node that is still running.
        aclose = getattr(events, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass

    if final_state is None:
        yield (
            "error",
            {
                "detail": "Pipeline produced no state",
                "code": "pipeline_error",
                "session_id": session_id,
            },
        )
        return

    elapsed_ms = (time.monotonic() - start) * 1000
    final_state["total_latency_ms"] = round(elapsed_ms, 2)
    final_state["status"] = "completed"
    final_state["outcome"] = run_outcome(final_state)

    logger.info(
        "Streamed query pipeline completed",
        extra={
            "session_id": session_id,
            "elapsed_ms": round(elapsed_ms, 2),
            "outcome": final_state["outcome"],
            "validation_status": final_state.get("validation_status"),
            **_run_summary(final_state),
        },
    )
    if not _keep_completed():
        await _delete_thread(app, thread_id, "completed")

    yield ("result", final_state)
