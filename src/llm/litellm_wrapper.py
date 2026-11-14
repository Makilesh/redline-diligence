"""
LiteLLM wrapper for structured agent calls.

Enforces JSON mode via response_format parameter for agents that expect
structured JSON output (Agents 1, 5, 6, 8), guarantees the result is a JSON
object, and descends the agent ladder when the selected slot fails.

All agents that return JSON MUST use call_structured_agent().
Agent 7 (Answer Synthesizer) returns prose and uses call_prose_agent().

Every provider request made here is accounted for in BudgetTracker: selection
pays for the first, a retry pays for itself, and a request the provider refused
gives its unit back (see "Accounting at the call boundary").
"""

import asyncio
import contextvars
import json
import os
import time

import litellm

from src.llm.failures import (
    Failure,
    FailureKind,
    ProseAgentFailed,
    StructuredOutputError,
    classify_failure,
)
from src.llm.model_registry import AGENT_LADDER, LOCAL_MODEL, is_local
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── Observability (optional, free) ────────────────────────────────────────────
# Per-request context attached to every LLM call's metadata: which deal/session
# the call served. Set once per query by the orchestrator; contextvars propagate
# into LangGraph's node tasks, so no agent has to thread it through.
_trace_context: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "llm_trace_context", default=None
)

# The model that actually answered the most recent verification call in this
# context. See active_verification_model().
_last_verification_model: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "last_verification_model", default=None
)


def set_trace_context(**fields) -> None:
    """
    Records request-scoped metadata (deal_id, session_id) for LLM call tracing.

    Args:
        **fields: Values to attach to every subsequent call in this context.
    """
    _trace_context.set({k: v for k, v in fields.items() if v is not None})

    # The orchestrator calls this once per query, which makes it the start of
    # a request as far as routing is concerned: model skips recorded by an
    # earlier query in the same task must not carry over.
    from src.llm.budget_tracker import begin_request_scope  # deferred: avoids import cycle

    begin_request_scope()


def _langfuse_enabled() -> bool:
    """True when both Langfuse keys are configured."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _configure_tracing() -> bool:
    """
    Enables LiteLLM's Langfuse callbacks when LANGFUSE_PUBLIC_KEY and
    LANGFUSE_SECRET_KEY are set; a no-op otherwise.

    Langfuse is deliberately NOT a hard dependency (`pip install langfuse` to
    use it; LANGFUSE_HOST selects a self-hosted or regional instance). If the
    keys are set but the package is missing, tracing is skipped with a warning
    rather than failing every LLM call at the callback.

    Returns:
        True when tracing was enabled.
    """
    if not _langfuse_enabled():
        return False
    try:
        import langfuse  # noqa: F401  (presence check only)
    except ImportError:
        logger.warning(
            "LANGFUSE_* keys are set but the langfuse package is not installed; "
            "tracing disabled (pip install langfuse)"
        )
        return False
    for hook in (litellm.success_callback, litellm.failure_callback):
        if "langfuse" not in hook:
            hook.append("langfuse")
    logger.info("LLM tracing enabled (Langfuse via LiteLLM callbacks)")
    return True


_TRACING = _configure_tracing()


def _call_metadata(agent: str | None) -> dict | None:
    """
    LiteLLM `metadata` for a call — only when tracing is on, so untraced calls
    send exactly the kwargs they always did.

    Args:
        agent: Pipeline agent making the call.

    Returns:
        Metadata dict, or None.
    """
    if not _TRACING:
        return None
    ctx = _trace_context.get() or {}
    tags = [t for t in (agent, ctx.get("deal_id")) if t]
    return {
        "generation_name": agent or "llm_call",
        "trace_name": "manda-query",
        "session_id": ctx.get("session_id"),
        "tags": tags,
        "trace_metadata": {**ctx, "agent": agent},
    }


def _log_usage(response, model: str, agent: str | None, started: float) -> None:
    """
    Logs token usage and latency for one completed call.

    Free observability even without Langfuse: the structured log line is enough
    to see which agent spends the quota. Tolerates responses without `usage`
    (test doubles, some local backends).
    """
    usage = getattr(response, "usage", None)

    def _get(name: str):
        if usage is None:
            return None
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        return value

    logger.info(
        "LLM call usage",
        extra={
            "model": model,
            "agent": agent,
            "prompt_tokens": _get("prompt_tokens"),
            "completion_tokens": _get("completion_tokens"),
            "total_tokens": _get("total_tokens"),
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            **{k: v for k, v in (_trace_context.get() or {}).items() if k == "deal_id"},
        },
    )

# Local fallback for the verification agents. The cloud model is no longer named
# here — it comes from BudgetTracker's agent ladder, so verification routes and
# rotates across keys like every other cloud call instead of pinning one model.
LOCAL_VERIFICATION_MODEL = LOCAL_MODEL

# Retry policy for prose calls. Matches the 3-attempt budget the structured
# agent already used, with linear backoff so a transient upstream 503 gets a
# meaningful gap before the next attempt.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

# Ladder rungs to try before giving up. Bounded so a provider-wide outage
# cannot walk the whole ladder on every request.
MAX_LADDER_FALLBACKS = 6

# Same-slot transport retries for the local model only (see _structured_on_slot).
# A refused connection is excluded: nothing is listening, and waiting six
# seconds to learn that again helps nobody.
LOCAL_TRANSPORT_ATTEMPTS = 3
_LOCAL_RETRY_KINDS = frozenset({
    FailureKind.TIMEOUT,
    FailureKind.SERVER_ERROR,
    FailureKind.SERVICE_UNAVAILABLE,
    FailureKind.UNKNOWN,
})

# How much of an unusable reply is echoed back in the repair request. Enough to
# show the model what it did; bounded so a runaway reply cannot double the
# prompt.
REPAIR_ECHO_CHARS = 4000

# Per-request ceilings. LiteLLM defaults to 600s, which is not a timeout so much
# as an absence of one: when Gemini returned 503 "experiencing high demand", each
# synthesis call hung for ~125 seconds before the provider dropped it, and the
# ladder could not descend until it did. Three of those in a row is a query that
# looks hung to the user while a perfectly good fallback model sits unused.
#
# The values are generous against observed behaviour rather than tight: healthy
# structured calls finish in 1-3s and healthy synthesis in 10-30s, so these only
# fire when something is actually wrong. Exceeding them raises, which the retry
# loop already treats as a transport error — so the effect is to reach the next
# rung sooner, never to lose a call that would have succeeded.
STRUCTURED_TIMEOUT_SECONDS = 60.0
PROSE_TIMEOUT_SECONDS = 120.0


def is_quota_error(exc: Exception) -> bool:
    """
    True when the provider refused the call for rate or quota reasons.

    Distinguishing this from a generic failure is what lets a caller move down
    the model ladder instead of retrying a model the provider will keep refusing
    all day. A per-minute limit and a per-day limit both surface as 429 and both
    answer True here — another slot is the better bet either way. What differs
    is how long the refused slot stays out, and that is decided by
    BudgetTracker.mark_slot_exhausted from the quota the provider named.
    """
    return classify_failure(exc).kind is FailureKind.QUOTA


def is_service_unavailable(exc: Exception) -> bool:
    """
    True when the provider says the model itself is temporarily unavailable.

    Worth separating from both quota and transport errors because the right
    response is different. A 429 means *this key* is spent, so another key on
    the same model is the best next move. A 503 — "This model is currently
    experiencing high demand" — means the model is down provider-wide, so every
    key will fail the same way and only another *model* helps.

    Observed live: gemini-3.6-flash returned 503 on roughly a third of synthesis
    calls while gemini-3.5-flash answered every request in ~1.2s. Treating the
    503 as a generic transport error meant three retries with backoff against a
    model that was not going to answer, before anything else was tried.

    Deliberately NOT treated as quota exhaustion: marking the slot spent would
    retire that model for the rest of the day over a blip that typically clears
    in minutes.

    Also True for ProseAgentFailed — a model whose same-model retries all came
    back empty or errored. The repair is identical (try another model), and
    without it that case matched no classifier and ended the caller's ladder.
    """
    return classify_failure(exc).kind in (
        FailureKind.SERVICE_UNAVAILABLE,
        FailureKind.MODEL_FAILED,
    )


def is_model_unavailable_for_key(exc: Exception) -> bool:
    """
    True when this credential specifically cannot use this model.

    Distinct from every other failure mode here, and the distinction is not
    hypothetical. `gemini-2.5-flash` and `gemini-2.5-flash-lite` return
    404 "This model is no longer available to new users" on two of five
    configured keys and answer normally on the other two — Google grandfathers
    older keys when a model is closed to new sign-ups.

    So availability is a property of the (key, model) pair, not of the model.
    A 429 retires the pair for the day, a 503 retires the model for everyone
    briefly, an auth error retires the key entirely — none of those describe
    "this key may never use this model", which is permanent and affects one
    slot.
    """
    return classify_failure(exc).kind is FailureKind.MODEL_UNAVAILABLE_FOR_KEY


def is_timeout_error(exc: Exception) -> bool:
    """
    True when the request exceeded its own deadline.

    Treated exactly like a 503, and for the same reason: a model that just
    consumed the full timeout without answering is not more likely to answer on
    the next identical attempt, and every retry costs another full timeout.

    Measured: `gemini-3.7-flash` began timing out consistently, and because a
    timeout matched none of the other classifiers it fell through to the generic
    retry path — three attempts at 120 seconds each. The caller's 300-second
    budget expired before the ladder was ever allowed to try a healthy model, so
    the query returned nothing at all rather than a slightly worse answer.
    """
    return classify_failure(exc).kind is FailureKind.TIMEOUT


def is_auth_error(exc: Exception) -> bool:
    """
    True when the provider rejected the credential itself.

    Distinct from a quota refusal in an important way: an exhausted key recovers
    at the next daily reset, whereas a rejected credential is unusable for every
    model until someone fixes the configuration. The caller therefore retires the
    whole key rather than one (key, model) slot.

    Worth handling rather than letting it fail the request: a run with four
    configured keys lost 16 of 35 questions because one key was invalid — one
    entry in GEMINI_API_KEYS had been split by a stray comma into two fragments —
    and every request routed to it died instead of moving to a working key.
    """
    return classify_failure(exc).kind is FailureKind.AUTH


# ─── Accounting at the call boundary ─────────────────────────────────────────
#
# BudgetTracker debits one daily unit and one RPM-window entry when it SELECTS a
# slot. Everything after that happens here, where the provider's answer is
# known — so this is where the count is reconciled: a refused request gives its
# unit back, a same-slot retry pays for itself, and a 429 is annotated with the
# quota it tripped. Doing it at the call boundary means every caller gets it,
# including the ladder loops that live in the agents.


def _live_tracker():
    """The process's BudgetTracker if one has been initialised, else None."""
    from src.llm.budget_tracker import BudgetTracker  # deferred: avoids import cycle

    return BudgetTracker._instance


def _owned_slot(api_key: str | None):
    """
    Resolves the credential a request was sent with to the tracker's key index.

    Returns:
        (tracker, key_index) when a tracker is metering this credential, else
        None — a local model, a direct call with an unmanaged key, or a test
        double standing in for the tracker.
    """
    if not api_key:
        return None
    tracker = _live_tracker()
    lookup = getattr(tracker, "key_index_of", None)
    if lookup is None:
        return None
    key_index = lookup(api_key)
    return (tracker, key_index) if key_index >= 0 else None


async def _settle_failure(model: str, api_key: str | None, failure: Failure) -> None:
    """
    Reconciles the tracker with a request that failed before being served.

    Notes the quota a 429 tripped (so mark_slot_exhausted can tell a spent day
    from a busy minute) and refunds the unit when the provider did not serve
    the request. Never raises: bookkeeping must not break a fallback path.
    """
    owned = _owned_slot(api_key)
    if owned is None:
        return
    tracker, key_index = owned
    try:
        if failure.kind is FailureKind.QUOTA:
            tracker.note_quota_refusal(
                key_index, model,
                per_minute=failure.is_per_minute_quota,
                retry_after=failure.retry_after,
            )
        if failure.refundable:
            await tracker.refund_slot(key_index, model)
    except Exception as e:
        logger.error(
            "Failed to settle quota after a provider failure",
            extra={"model": model, "error": str(e)},
        )


async def _charge_retry(model: str, api_key: str | None) -> bool:
    """
    Pays for one more request on a slot that selection already paid for once.

    Returns:
        True if the retry may be sent. Always True when nothing is metering the
        call; False when the slot has no window or daily allowance left, in
        which case the caller should give up on this slot rather than retry.
    """
    owned = _owned_slot(api_key)
    if owned is None:
        return True
    tracker, key_index = owned
    try:
        return await tracker.charge_extra_request(key_index, model)
    except Exception as e:
        logger.error(
            "Failed to account for a retry; not retrying on this slot",
            extra={"model": model, "error": str(e)},
        )
        return False


# Failures that are evidence about the provider's health for a model, as
# opposed to our own connectivity or one slot's quota. Only these feed the
# process-wide cooldown.
_HEALTH_KINDS = frozenset({
    FailureKind.SERVICE_UNAVAILABLE,
    FailureKind.TIMEOUT,
    FailureKind.SERVER_ERROR,
})


def _report_unhealthy(model: str) -> None:
    """
    Tells the tracker the provider failed on this model.

    Reported here, at the call boundary, because this is the only code that
    actually sees the provider fail. Callers can only keep their own request
    off a model (BudgetTracker.skip_model_for_request); they have no way to
    put it in the process-wide cooldown, so a rejected answer can no longer be
    booked as an outage.
    """
    report = getattr(_live_tracker(), "report_model_unavailable", None)
    if report is not None:
        report(model)


async def _acompletion(kwargs: dict, model: str, api_key: str | None):
    """litellm.acompletion with the quota settled if the request fails."""
    try:
        return await litellm.acompletion(**kwargs)
    except Exception as e:
        await _settle_failure(model, api_key, classify_failure(e))
        raise


# ─── Structured (JSON) calls ─────────────────────────────────────────────────


class ServedModel:
    """
    Out-parameter naming the model that actually answered a structured call.

    call_structured_agent may descend the ladder, so the model an agent selected
    is not necessarily the one that served it. Agents pass one of these to keep
    their trace truthful; a test double that ignores it leaves the initial
    model in place.
    """

    def __init__(self, model: str):
        self.model = model


def _parse_json_object(raw, finish_reason: str | None) -> dict:
    """
    Turns a completion into the JSON object the agent asked for.

    Args:
        raw: The completion text (may be None).
        finish_reason: Provider finish reason; "length" means the output was
            cut off at max_tokens.

    Returns:
        The parsed object.

    Raises:
        StructuredOutputError: Empty, truncated, unparseable, or valid JSON
            that is not an object. The last case used to be returned as-is and
            crashed the caller on its first `.get`.
    """
    cut_off = finish_reason == "length"
    if raw is None or not str(raw).strip():
        # With a thinking model an empty reply at the token limit means the
        # budget went on reasoning, so it is reported as truncation: the repair
        # is more room, not a re-ask.
        raise StructuredOutputError(
            "truncated" if cut_off else "empty", "the completion had no content", raw
        )

    # Strip accidental markdown fences before parsing
    clean = str(raw).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        # JSON mode occasionally wraps the object in a sentence. Salvaging the
        # outermost braces is free; a repair request costs a quota unit.
        start, end = clean.find("{"), clean.rfind("}")
        parsed = None
        if 0 <= start < end:
            try:
                parsed = json.loads(clean[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            raise StructuredOutputError(
                "truncated" if cut_off else "malformed", str(e), raw
            ) from e

    if not isinstance(parsed, dict):
        raise StructuredOutputError(
            "wrong_shape", f"expected a JSON object, got {type(parsed).__name__}", raw
        )
    return parsed


def _repair_messages(messages: list[dict], error: StructuredOutputError) -> list[dict]:
    """
    Messages for the one repair request that follows an unusable reply.

    Sending the identical request again at temperature 0 mostly reproduces the
    identical reply — the old loop did that twice. A reply that was cut off gets
    the same question with more room (the caller doubles max_tokens); anything
    else is shown its own output and the parser's complaint.
    """
    if error.problem in ("empty", "truncated"):
        return messages
    return messages + [
        {"role": "assistant", "content": (error.raw or "")[:REPAIR_ECHO_CHARS]},
        {
            "role": "user",
            "content": (
                f"That reply could not be used ({error.problem}). Reply again with a "
                "single valid JSON object only — no prose before or after it, no "
                "code fences."
            ),
        },
    ]


async def _structured_request(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
    agent: str | None,
) -> dict:
    """One provider request in JSON mode, parsed. No retries of any kind."""
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},  # Enforces JSON mode
        "timeout": STRUCTURED_TIMEOUT_SECONDS,
    }
    if is_local(model):
        kwargs["num_ctx"] = 8192  # Expand context window for local Ollama to prevent truncation
    if api_key:
        # Explicit per-call credential. With key rotation the key is
        # chosen per request, so relying on the ambient GEMINI_API_KEY
        # env var would send every call to key 1 while the tracker
        # debited whichever key it thought it had picked.
        kwargs["api_key"] = api_key
    metadata = _call_metadata(agent)
    if metadata:
        kwargs["metadata"] = metadata

    logger.info(
        "Calling structured agent",
        extra={"model": model, "temperature": temperature, "max_tokens": max_tokens},
    )
    started = time.monotonic()
    response = await _acompletion(kwargs, model, api_key)
    _log_usage(response, model, agent, started)

    first = response.choices[0]
    parsed = _parse_json_object(first.message.content, getattr(first, "finish_reason", None))
    logger.info(
        "Structured agent call successful",
        extra={"model": model, "response_keys": list(parsed.keys())},
    )
    return parsed


async def _structured_on_slot(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
    agent: str | None,
) -> dict:
    """
    Everything that is worth doing on ONE (key, model) slot.

    For a metered cloud slot that is a single request, plus at most one repair
    request when the reply was unusable — and the repair is paid for. Transport
    failures are not retried here at all: a 429, a 503 or a timeout is true for
    the next identical request too, so the ladder moves instead.

    The local model is the exception. Nothing meters it and there is no rung
    below it, so a transient fault there is retried in place as it always was.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    transport_attempts = LOCAL_TRANSPORT_ATTEMPTS if is_local(model) else 1

    for attempt in range(1, transport_attempts + 1):
        try:
            try:
                return await _structured_request(
                    messages, model, temperature, max_tokens, api_key, agent
                )
            except StructuredOutputError as unusable:
                logger.warning(
                    "Agent returned unusable JSON; sending one repair request",
                    extra={"model": model, "problem": unusable.problem,
                           "raw_preview": (unusable.raw or "")[:200] or "empty"},
                )
                if not await _charge_retry(model, api_key):
                    raise
                room = max_tokens * 2 if unusable.problem in ("empty", "truncated") else max_tokens
                return await _structured_request(
                    _repair_messages(messages, unusable), model, temperature, room, api_key, agent
                )
        except StructuredOutputError:
            raise
        except Exception as e:
            logger.error(
                "Structured agent call failed",
                extra={"model": model, "attempt": attempt, "error": str(e),
                       "error_type": type(e).__name__},
            )
            retryable = classify_failure(e).kind in _LOCAL_RETRY_KINDS
            if attempt == transport_attempts or not retryable:
                raise
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError("unreachable: the loop above returns or raises")


async def _reroute_after_failure(tracker, choice, exc: Exception, excluded: set[str]) -> bool:
    """
    Applies the repair a failure calls for and says whether another rung is
    worth trying.

    This is the whole failure taxonomy in one place, each class repaired at the
    scope it is true for:

      429                → this (key, model) slot: rested if per-minute, spent
                           for the day otherwise. Next key, same model.
      rejected key       → the key, for every model.
      404 for this key   → this slot, permanently.
      503/timeout/5xx    → the model, for every key, on an escalating cooldown.
      unusable output    → the model, for this request only. Says nothing about
                           its health, so nobody else is affected.
      anything else      → stop. A malformed request fails identically on every
                           rung, and walking the ladder would only spend quota.

    Args:
        tracker: BudgetTracker (or a stand-in exposing the mark_* methods).
        choice: The slot that failed.
        exc: What it raised.
        excluded: Request-scoped set of models to avoid; updated in place.

    Returns:
        True if the ladder should select again.
    """
    key_index = getattr(choice, "key_index", -1)
    model = choice.model
    if key_index < 0 or is_local(model):
        return False

    failure = classify_failure(exc)
    kind = failure.kind

    if kind is FailureKind.AUTH:
        tracker.mark_key_unusable(key_index)
        return True
    if kind is FailureKind.QUOTA:
        await tracker.mark_slot_exhausted(key_index, model)
        logger.warning(
            "Agent rung refused by provider, descending the ladder",
            extra={"model": model, "key_index": key_index,
                   "quota_window": failure.quota_window},
        )
        return True
    if kind is FailureKind.MODEL_UNAVAILABLE_FOR_KEY:
        # This key may never use this model. Retire the pair, not the
        # key and not the model — both are still fine elsewhere.
        tracker.mark_slot_unavailable(key_index, model)
        logger.warning(
            "Model not available for this credential; retiring the slot",
            extra={"model": model, "key_index": key_index},
        )
        return True
    if failure.another_model_helps:
        # The model is down or unresponsive for everyone, so rotating
        # keys is pointless. Keep this request off it, and — when it is the
        # provider that failed rather than our own connection — tell every
        # other request too.
        tracker.skip_model_for_request(model)
        if kind in _HEALTH_KINDS:
            report = getattr(tracker, "report_model_unavailable", None)
            if report is not None:
                report(model)
        logger.warning(
            "Agent rung unavailable provider-side, trying another model",
            extra={"model": model, "failure": kind.value},
        )
        return True
    if kind is FailureKind.BAD_OUTPUT:
        excluded.add(model)
        logger.warning(
            "Agent rung returned unusable output twice, trying another model",
            extra={"model": model},
        )
        return True
    return False


def _ladder_hops(tracker) -> int:
    """
    How many rungs one request may try.

    Each failed hop removes something from future selection (a slot, a key, a
    model), so the walk is self-limiting; the bound only has to stop a request
    from touring a provider-wide outage. It grows with the number of keys
    because a 429 costs one hop per key on the same model: with five keys and a
    fixed bound of six, a stale counter on the top rung used up five hops
    rotating keys and left one for the entire rest of the ladder.
    """
    keys = getattr(tracker, "_api_keys", None) or ()
    return MAX_LADDER_FALLBACKS + sum(1 for k in keys if k)


async def _structured_with_ladder(tracker, first_choice, on_slot) -> tuple[dict, str]:
    """
    Runs a structured call down the agent ladder until a rung answers.

    Args:
        tracker: BudgetTracker (or a stand-in).
        first_choice: A slot the caller already selected, or None to select.
        on_slot: Coroutine function taking a ModelChoice and returning the
            parsed dict from that slot.

    Returns:
        (parsed dict, model that served it).

    Raises:
        Exception: The last cloud failure when no rung — including the local
            model — produced an answer.
    """
    from src.llm.budget_tracker import ModelChoice  # deferred: avoids import cycle

    excluded: set[str] = set()
    choice = first_choice
    last_error: Exception | None = None
    cloud_error: Exception | None = None

    def give_up(error: Exception) -> Exception:
        # No local model to fall back on (connection refused) is not news; the
        # cloud failure that sent us there is the one worth reporting.
        if cloud_error is not None and classify_failure(error).kind is FailureKind.CONNECTION:
            return cloud_error
        return error

    for _ in range(_ladder_hops(tracker)):
        if choice is None:
            if excluded and hasattr(tracker, "key_index_of"):
                choice = await tracker.get_model_for_agent(exclude=frozenset(excluded))
            else:
                choice = await tracker.get_model_for_agent()
        try:
            result = await on_slot(choice)
        except Exception as e:
            last_error = e
            if not is_local(choice.model):
                cloud_error = e
            if not await _reroute_after_failure(tracker, choice, e, excluded):
                break
            choice = None
            continue
        # One good answer ends a failure streak. Only the synthesizer used to
        # report this, so an agent model that had one bad spell kept its
        # escalated cooldown for the life of the process.
        note_healthy = getattr(tracker, "note_model_healthy", None)
        if note_healthy is not None:
            note_healthy(choice.model)
        return result, choice.model

    if choice is not None and is_local(choice.model):
        raise give_up(last_error) if last_error else RuntimeError("structured call failed")

    logger.warning(
        "Cloud ladder failed, falling back to local model",
        extra={"error": str(last_error), "fallback": LOCAL_MODEL},
    )
    try:
        result = await on_slot(ModelChoice(model=LOCAL_MODEL, api_key=None, key_index=-1))
    except Exception as local_error:
        raise give_up(local_error)
    return result, LOCAL_MODEL


async def call_structured_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    api_key: str | None = None,
    agent: str | None = None,
    descend: bool = True,
    served: ServedModel | None = None,
) -> dict:
    """
    Wrapper for all agents that return JSON.
    Enforces JSON mode via response_format parameter and guarantees the result
    is a JSON object.

    The caller names the slot to try first (`model`, `api_key`, as selected by
    BudgetTracker.get_model_for_agent). If that slot fails and the tracker is
    metering the credential, the call descends the agent ladder itself: next
    key on a 429, next model on a 503 or timeout, and finally the local model.
    Agents 1, 5 and 6 select once and call this, so before the descent lived
    here a single 429 on the selected key failed the whole query — after 60
    seconds of backoff against a key that was not going to answer.

    Args:
        system_prompt: System-level instructions for the agent.
        user_prompt: User query / context for the agent.
        model: LiteLLM model string (e.g., "gemini/gemini-3.5-flash-lite").
        temperature: Sampling temperature (default 0.0 for determinism).
        max_tokens: Maximum output tokens.
        api_key: Credential for this specific call. Required when multiple keys
            are configured — see the note at the kwargs assembly.
        agent: Calling agent's name, for usage logs and optional tracing.
        descend: False to try only the named slot and raise. For callers that
            run their own ladder (the verification path).
        served: Optional holder updated with the model that actually answered.

    Returns:
        Parsed JSON object from the agent response.

    Raises:
        StructuredOutputError: (a ValueError) every rung tried returned
            unusable JSON.
        Exception: The provider failure, when no rung could serve the call.
    """
    owned = _owned_slot(api_key) if descend else None
    if owned is None:
        return await _structured_on_slot(
            system_prompt, user_prompt, model, temperature, max_tokens, api_key, agent
        )

    from src.llm.budget_tracker import ModelChoice  # deferred: avoids import cycle

    tracker, key_index = owned

    async def on_slot(choice):
        return await _structured_on_slot(
            system_prompt, user_prompt, choice.model, temperature, max_tokens,
            choice.api_key, agent,
        )

    result, served_model = await _structured_with_ladder(
        tracker, ModelChoice(model=model, api_key=api_key, key_index=key_index), on_slot
    )
    if served is not None:
        served.model = served_model
    return result


# Failures a prose call hands straight to the caller's ladder: retrying the same
# slot cannot change the answer. Everything else (a 5xx, a dropped connection,
# an empty completion) is retried in place first.
_PROSE_NO_RETRY_KINDS = frozenset({
    FailureKind.SERVICE_UNAVAILABLE,
    FailureKind.TIMEOUT,
    FailureKind.QUOTA,
    FailureKind.AUTH,
    FailureKind.MODEL_UNAVAILABLE_FOR_KEY,
    FailureKind.REQUEST_INVALID,
})


async def call_prose_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 3000,
    api_key: str | None = None,
    agent: str | None = None,
) -> str:
    """
    Wrapper for agents that return prose (not JSON).
    Used by Agent 7 (Answer Synthesizer) which returns natural language answers.
    Does NOT enforce JSON mode.

    Args:
        system_prompt: System-level instructions for the agent.
        user_prompt: User query / context for the agent.
        model: LiteLLM model string.
        temperature: Sampling temperature (default 0.1 for slight variety).
        max_tokens: Maximum output tokens (default 3000 for long answers).
        api_key: Credential for this specific call.
        agent: Calling agent's name, for usage logs and optional tracing.

    Returns:
        Raw string response from the agent. Never None.

    Raises:
        ProseAgentFailed: (a RuntimeError) every attempt failed or returned an
            empty completion. Classified so a ladder moves to another model.
        Exception: A provider refusal the caller's ladder classifies — 429,
            503, timeout, rejected key, revoked model — raised on first sight.

    Retries exist because both failure modes were observed on the golden-set run:
    a transient upstream 503 ("model is currently experiencing high demand") and,
    more insidiously, a 200 response whose content was empty. The empty case used
    to be returned as None and propagate until something did `x in None`, failing
    the whole request with an opaque TypeError. Both are now retried, and an
    exhausted retry raises a typed error the synthesizer converts into a refusal —
    a refusal is a far better outcome for the caller than a 500.

    Every retry is a provider request, so every retry is charged to the slot
    (see _charge_retry). When the slot cannot pay, the retries stop.
    """
    last_error: Exception | None = None
    attempts = 0

    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        logger.info(
            "Calling prose agent",
            extra={
                "model": model,
                "attempt": attempt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

        try:
            prose_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": PROSE_TIMEOUT_SECONDS,
            }
            if api_key:
                prose_kwargs["api_key"] = api_key
            metadata = _call_metadata(agent)
            if metadata:
                prose_kwargs["metadata"] = metadata
            started = time.monotonic()
            response = await _acompletion(prose_kwargs, model, api_key)
            _log_usage(response, model, agent, started)
            content = response.choices[0].message.content

            if content and content.strip():
                logger.info(
                    "Prose agent call successful",
                    extra={
                        "model": model,
                        "attempt": attempt,
                        "response_length": len(content),
                    },
                )
                return content

            last_error = RuntimeError("empty completion")
            logger.warning(
                "Prose agent returned an empty completion",
                extra={"model": model, "attempt": attempt},
            )

        except Exception as e:
            last_error = e
            logger.warning(
                "Prose agent call failed",
                extra={"model": model, "attempt": attempt, "error": str(e)},
            )

            # A 503 means the model is down for everyone, so retrying *this*
            # model is time spent waiting for the same answer. Surface it
            # immediately and let the caller's ladder pick a different model.
            #
            # Measured: gemini-3.7-flash returning 503 cost three attempts and
            # ~48 seconds of backoff before the ladder was allowed to move to
            # gemini-3.6-flash, which answered on the first try.
            #
            # The same holds for a 429, a rejected key and a revoked model:
            # each is a property of the slot, not of this attempt. They used to
            # be retried here — three requests and six seconds against a key
            # that had already said no — before the ladder saw them.
            kind = classify_failure(e).kind
            if kind in _PROSE_NO_RETRY_KINDS:
                if kind in _HEALTH_KINDS:
                    _report_unhealthy(model)
                raise

        if attempt == MAX_RETRIES or not await _charge_retry(model, api_key):
            break
        await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    if last_error is not None and classify_failure(last_error).kind in _HEALTH_KINDS:
        _report_unhealthy(model)
    raise ProseAgentFailed(
        f"Prose agent failed after {attempts} attempts "
        f"(model={model}): {last_error}"
    ) from last_error


async def call_verification_agent_with_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1500,
    agent: str | None = None,
) -> tuple[dict, str]:
    """
    Verification call that also reports which model actually answered.

    Model routing is a deployment choice, not an architectural one, so it is
    controlled by VERIFICATION_BACKEND:

      "cloud" (default) — Gemini via the budget tracker's agent ladder. Higher-
          quality structured reasoning and no local GPU requirement.
      "local"           — Ollama / Qwen2.5:14b. Keeps verification off the
          metered API entirely, at the cost of a 12GB-VRAM machine.

    If the cloud ladder fails for any reason, the local model is tried before
    giving up, so a quota exhaustion degrades rather than fails.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User query / context.
        temperature: Sampling temperature (default 0.0).
        max_tokens: Maximum output tokens (default 1500).
        agent: Calling agent's name, for usage logs and optional tracing.

    Returns:
        (parsed JSON dict, model string that served the call).

    Raises:
        ValueError: If every configured model returns invalid JSON.
        Exception: The last cloud failure when even the local fallback fails.
    """
    backend = os.getenv("VERIFICATION_BACKEND", "cloud").strip().lower()

    if backend == "local":
        result = await call_structured_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=LOCAL_VERIFICATION_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            agent=agent,
        )
        _last_verification_model.set(LOCAL_VERIFICATION_MODEL)
        return result, LOCAL_VERIFICATION_MODEL

    # Route through the budget tracker rather than calling Gemini directly, so
    # verification takes a rate-limiter slot and debits the daily quota like
    # every other cloud call. get_model_for_agent() already returns the local
    # model when the quota is spent.
    from src.llm.budget_tracker import BudgetTracker  # deferred: avoids import cycle

    tracker = await BudgetTracker.get_instance()

    # Descend the ladder on refusals rather than failing. The tracker's
    # counters can lag the provider's (restarts reset the in-memory fallback,
    # other processes share the key), so a rung it offers may already be closed.
    # The descent itself is the same one every structured agent uses.
    async def on_slot(choice):
        return await call_structured_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=choice.model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=choice.api_key,
            agent=agent,
            descend=False,
        )

    result, model = await _structured_with_ladder(tracker, None, on_slot)
    _last_verification_model.set(model)
    return result, model


async def call_verification_agent(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1500,
    agent: str | None = None,
) -> dict:
    """
    Wrapper for the verification LLM calls. Enforces JSON mode.

    Thin form of call_verification_agent_with_model for callers that only need
    the result; the serving model is still recorded for
    active_verification_model().

    Args:
        system_prompt: System-level instructions.
        user_prompt: User query / context.
        temperature: Sampling temperature (default 0.0).
        max_tokens: Maximum output tokens (default 1500).
        agent: Calling agent's name, for usage logs and optional tracing.

    Returns:
        Parsed JSON dict from the verification model.
    """
    result, _model = await call_verification_agent_with_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        agent=agent,
    )
    return result


# Backwards-compatible alias — the verification agents were originally local-only.
call_local_agent = call_verification_agent


def active_verification_model() -> str:
    """
    The model that served the most recent verification call in this context,
    or — before any call — the model the verification path will try first.

    This used to return AGENT_LADDER[0] unconditionally, so the trace named the
    top rung even when quota exhaustion or a 503 had sent the call down the
    ladder or to the local model. The served model is now recorded by
    call_verification_agent_with_model; prefer the model that function returns
    over calling this.
    """
    served = _last_verification_model.get()
    if served:
        return served
    backend = os.getenv("VERIFICATION_BACKEND", "cloud").strip().lower()
    if backend == "local":
        return LOCAL_VERIFICATION_MODEL
    return AGENT_LADDER[0]


# ─── Streaming ────────────────────────────────────────────────────────────────


class StreamInterrupted(RuntimeError):
    """
    A streamed completion failed after tokens had already been emitted.

    Distinct from a failure before the first token: the client has partial text
    on screen, so the caller must tell it to discard that draft before falling
    back to another model.
    """

    def __init__(self, message: str, emitted_chars: int, cause: Exception):
        super().__init__(message)
        self.emitted_chars = emitted_chars
        self.cause = cause


class _UsageHolder:
    """Response-shaped holder so _log_usage can read a streamed call's usage."""

    def __init__(self, usage):
        self.usage = usage


async def stream_prose_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    on_token,
    temperature: float = 0.1,
    max_tokens: int = 3000,
    api_key: str | None = None,
    agent: str | None = None,
) -> str:
    """
    Prose completion streamed token by token, returning the full text.

    Failure semantics mirror call_prose_agent so the synthesizer's ladder logic
    is unchanged:

    - fails BEFORE the first token: a 503 / timeout / quota / auth error is
      re-raised for the ladder to classify; any other error falls back to the
      non-streamed call_prose_agent on the same model (which retries), and its
      whole answer is emitted as one token.
    - fails AFTER tokens were emitted: raises StreamInterrupted, and the caller
      emits an answer reset before trying elsewhere.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User prompt.
        model: LiteLLM model string.
        on_token: Callable receiving each text delta.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        api_key: Credential for this call.
        agent: Calling agent's name.

    Returns:
        The complete generated text. Never empty.

    Raises:
        StreamInterrupted: The stream broke after emitting text.
        RuntimeError: Empty completion.
        Exception: Provider errors the ladder classifies (see above).
    """
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": PROSE_TIMEOUT_SECONDS,
        "stream": True,
    }
    if api_key:
        kwargs["api_key"] = api_key
    metadata = _call_metadata(agent)
    if metadata:
        kwargs["metadata"] = metadata

    parts: list[str] = []
    emitted = 0
    started = time.monotonic()
    usage = None
    try:
        stream = await litellm.acompletion(**kwargs)
        async for chunk in stream:
            usage = getattr(chunk, "usage", None) or usage
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError):
                delta = None
            if delta:
                parts.append(delta)
                emitted += len(delta)
                on_token(delta)
    except Exception as e:
        if emitted:
            # A stream that drops mid-answer is the provider failing, whatever
            # the exception class says.
            _report_unhealthy(model)
            raise StreamInterrupted(
                f"stream from {model} failed after {emitted} chars: {e}", emitted, e
            ) from e
        failure = classify_failure(e)
        await _settle_failure(model, api_key, failure)
        if failure.kind in _PROSE_NO_RETRY_KINDS:
            if failure.kind in _HEALTH_KINDS:
                _report_unhealthy(model)
            raise
        # The non-streamed attempt below is a second provider request on this
        # slot, so it is paid for like any other retry.
        if not await _charge_retry(model, api_key):
            raise ProseAgentFailed(
                f"stream from {model} failed before the first token and the slot "
                f"has no allowance left for a retry: {e}"
            ) from e
        logger.warning(
            "Streaming failed before the first token; retrying without streaming",
            extra={"model": model, "error": str(e)},
        )
        text = await call_prose_agent(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            agent=agent,
        )
        on_token(text)
        return text

    _log_usage(_UsageHolder(usage), model, agent, started)

    text = "".join(parts)
    if not text.strip():
        # Typed so the caller's ladder tries another model; as a bare
        # RuntimeError it matched no classifier and ended in a refusal.
        raise ProseAgentFailed(f"empty streamed completion (model={model})")
    return text
