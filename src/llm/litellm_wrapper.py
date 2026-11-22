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
    CompletionTruncated,
    Failure,
    FailureKind,
    LocalContextExceeded,
    ProseAgentFailed,
    is_context_refusal,
    ResponseDeadlineExceeded,
    StructuredOutputError,
    TokenWindowFull,
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


def _usage_field(obj, name: str):
    """A field of a usage object or dict (or of its token-details part), if any."""
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(name)
    return value


def _log_usage(
    response,
    model: str,
    agent: str | None,
    started: float,
    finish_reason: str | None = None,
    max_tokens: int | None = None,
) -> None:
    """
    Logs token usage, latency and how the call ended, for one completed call.

    Free observability even without Langfuse: the structured log line is enough
    to see which agent spends the quota. Tolerates responses without `usage`
    (test doubles, some local backends).

    `finish_reason` "length" next to `max_tokens` is what makes a cut-off reply
    visible: the 2026-11-15 evening run had 29 synthesis calls end at 2,996 of
    3,000 tokens, and nothing in the log said so. Gemini 3.x counts thinking
    against max_tokens, so `reasoning_tokens` (LiteLLM's reading of the
    provider's thoughtsTokenCount) is logged too — it is the part of the budget
    the reader never sees.
    """
    usage = getattr(response, "usage", None)

    def _get(name: str):
        return _usage_field(usage, name)

    logger.info(
        "LLM call usage",
        extra={
            "model": model,
            "agent": agent,
            "prompt_tokens": _get("prompt_tokens"),
            "completion_tokens": _get("completion_tokens"),
            "reasoning_tokens": _reasoning_tokens(usage),
            "total_tokens": _get("total_tokens"),
            "max_tokens": max_tokens,
            "finish_reason": finish_reason,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            **{k: v for k, v in (_trace_context.get() or {}).items() if k == "deal_id"},
        },
    )


def _reasoning_tokens(usage) -> int | None:
    """Thinking tokens the provider reported, from usage.completion_tokens_details."""
    value = _usage_field(_usage_field(usage, "completion_tokens_details"), "reasoning_tokens")
    return value if isinstance(value, int) else None


class ProseOutcome:
    """
    Out-parameter describing how a prose call ended, for the caller's trace.

    Like ServedModel: a caller that passes one learns what it could not see
    from the returned text alone, and a test double that ignores it leaves the
    defaults (finish_reason None, not truncated).

    Attributes:
        finish_reason: The provider's finish reason for the request whose text
            was returned ("stop", "length", ...), or None if it did not say.
        max_tokens: That request's output budget (larger after a length retry).
        length_retries: Same-model requests made because a reply was cut off.
        reasoning_tokens: Thinking tokens the provider reported for it, if any.
    """

    def __init__(self, max_tokens: int | None = None):
        self.finish_reason: str | None = None
        self.max_tokens = max_tokens
        self.length_retries = 0
        self.reasoning_tokens: int | None = None

    @property
    def truncated(self) -> bool:
        """True when the returned text stopped at max_tokens, not where the model ended it."""
        return self.finish_reason == "length"

# Local fallback for the verification agents. The cloud model is no longer named
# here — it comes from BudgetTracker's agent ladder, so verification routes and
# rotates across keys like every other cloud call instead of pinning one model.
LOCAL_VERIFICATION_MODEL = LOCAL_MODEL

# Context window (tokens) of every call to the local Ollama model. Without it
# Ollama uses its own default — 4,096 on a GPU under 24 GiB — and a longer prompt
# is cut to its first 4 and last ~2,046 tokens: the instructions at its start are
# lost. Only JSON calls set one (8,192), so every local synthesis prompt (4,100–
# 6,100 tokens on the sample deal) was cut, came back uncited and was refused,
# and the model reloaded each time the window size changed between calls
# (Decision 106). One size for every call. Measured with qwen2.5:14b on a 12 GB
# GPU: 8,192 fits it fully (~41 tokens/s), 16,384 keeps 83% on the GPU
# (~24 tokens/s). OLLAMA_NUM_CTX sets it.
DEFAULT_LOCAL_NUM_CTX = 16_384
_LOCAL_NUM_CTX_BOUNDS = (2_048, 131_072)


def _move_ollama_flags_to_top_level() -> None:
    """
    Sends `truncate` and `shift` where Ollama reads them.

    Ollama's own guards against the window: truncate=false refuses a prompt longer
    than it (counted with the model's tokenizer, which no estimate here can match —
    it splits every digit), and shift=false ends a reply that reaches it ("length")
    instead of discarding the context's start. Both are top-level request fields.
    LiteLLM (pinned 1.89.7) puts every provider option inside `options`, where Ollama
    ignores them — measured: a prompt was truncated with both set there. This moves
    the two flags up; tests/test_local_context_window.py checks the request LiteLLM
    builds, so an upgrade that changes the builder fails there.
    """
    from litellm.llms.ollama.completion.transformation import OllamaConfig

    original = OllamaConfig.transform_request
    if getattr(original, "_moves_ollama_flags", False):
        return

    def transform_request(self, *args, **kwargs):
        data = original(self, *args, **kwargs)
        options = data.get("options") or {}
        for flag in ("truncate", "shift"):
            if flag in options:
                data[flag] = options.pop(flag)
        return data

    transform_request._moves_ollama_flags = True
    OllamaConfig.transform_request = transform_request


_move_ollama_flags_to_top_level()


def local_num_ctx() -> int:
    """
    The context window sent with every local-model call: OLLAMA_NUM_CTX, else 16,384.

    Returns:
        The window, clamped to [2,048, 131,072]; an unreadable value falls back to
        the default with a warning.
    """
    raw = os.getenv("OLLAMA_NUM_CTX", "").strip()
    if not raw:
        return DEFAULT_LOCAL_NUM_CTX
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring OLLAMA_NUM_CTX: not an integer", extra={"value": raw})
        return DEFAULT_LOCAL_NUM_CTX
    low, high = _LOCAL_NUM_CTX_BOUNDS
    return min(max(value, low), high)


def _local_options() -> dict:
    """
    The provider options of every local-model call: one context window, and Ollama's
    refusal of what does not fit it instead of silent truncation or context shifting.
    """
    return {"num_ctx": local_num_ctx(), "truncate": False, "shift": False}

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
# Exceeding them raises a timeout, which every ladder treats as the model being
# unavailable — so the effect is to reach the next rung sooner, never to lose a
# call that would have succeeded.
#
# Structured calls: 118 of them in the 2026-11-15 live run (Lite rungs, agents
# and the judge alike) took 0.8-2.5s, none longer. 20s is eight times the
# slowest, and still above the worst a healthy Lite model can take when a
# repair request doubles max_tokens to 3,000 and it spends all of them thinking
# (~14s at the slowest healthy rate measured, 214 tokens/s). It was 60s, which
# a hung Lite model charged in full on each of the ~3 structured calls a query
# makes. The local model is not held to it: it has no rung below it, and a 14B
# model on a consumer GPU reading an 8K-token prompt is slower than any cloud
# rung without being unhealthy.
STRUCTURED_TIMEOUT_SECONDS = 20.0
LOCAL_STRUCTURED_TIMEOUT_SECONDS = 60.0

# Prose: the TOTAL budget for one synthesis request. Healthy synthesis takes
# 5-14s (59 calls in the live run; one outlier at 34s). On its own this bound
# is what made a hung model cost two minutes — see first_token_deadline() for
# the deadline that now catches a hang long before it.
PROSE_TIMEOUT_SECONDS = 120.0

# ─── First-token deadline for streamed prose ─────────────────────────────────
#
# A hung or overloaded model is recognisable long before the total budget runs
# out: it does not start answering. What "start" means for a thinking model was
# measured, not assumed (live probe, 2026-11-15, gemini-3.8-flash via LiteLLM
# 1.89.3, streamed, temperature 0.1, max_tokens 3000): the request opened at
# once, and then NOTHING arrived — no SSE chunk, no keep-alive — until thinking
# had finished. The first chunk came at 138.7s; all 148 answer tokens followed
# within 0.8s. Gemini 3.x streams no bytes while it thinks.
#
# So time-to-first-token is thinking time, and thinking time is bounded by the
# request's own max_tokens, because Gemini counts thoughts against it (12 of the
# run's 59 synthesis calls ended at 2,995-2,996 of 3,000 completion tokens). The
# slowest HEALTHY rate in the run was 214 tokens/s (gemini-2.5-flash; the
# 3-flash-preview rung ran at 260-290), so a healthy model that spends its whole
# budget thinking still starts answering within ~14s at max_tokens=3000. The
# deadline is that bound at a rate 30% below the slowest healthy one, plus the
# request overhead: 2 + 3000/150 = 22s. Streamed calls in the run actually
# started answering 4.3-5.7s after the request.
#
# One of the 59 healthy calls would have been abandoned (34.0s total, 2,409
# tokens — 71 tokens/s); it would have cost its unit and a fall to the next rung.
# Each hung call it replaces cost 120s.
#
# Still keyed to the WHOLE budget now that synthesis asks for 8,192 tokens
# (56.6s), not to a guess at how much of it thinking will take. Gemini 3.x
# takes no numeric thinking cap (LiteLLM 1.89.3 maps reasoning_effort to a
# thinkingLevel and drops a `thinking` budget for Gemini 3), so any number
# below the budget is a guess the model is free to exceed — and a missed
# deadline cools the model for everyone. Re-asked live with room to spare
# (2026-11-15, gemini-3.6-flash), mh_05 thought for 3,992 tokens and started
# answering at 37.9s, then wrote a complete 17-citation answer: any deadline
# under ~38s would have abandoned and cooled a healthy model (the same model
# reached the same point in 11-14s on two other questions, so the rate varies
# more than the run suggested). The cost is a slower verdict on a model that never starts —
# 56.6s instead of 22s, still half the 120s it replaced — and the case it
# exists for is still caught: the same request on gemini-3.8-flash, thinking at
# ~18 tokens/s, first answered at 71.2s.
FIRST_TOKEN_MIN_TOKENS_PER_SECOND = 150.0
FIRST_TOKEN_OVERHEAD_SECONDS = 2.0
FIRST_TOKEN_FLOOR_SECONDS = 10.0

# After the first token a healthy stream delivers a chunk every ~0.1s. A gap
# this long means the stream has died with the connection still open.
STREAM_STALL_SECONDS = 20.0


def first_token_deadline(max_tokens: int) -> float:
    """
    Seconds a streamed prose request may take to emit its first answer token.

    Derived from max_tokens (see the measurements above) so that raising the
    output budget cannot silently turn the deadline into a guillotine for
    healthy long answers. PROSE_FIRST_TOKEN_TIMEOUT_SECONDS overrides it.

    Args:
        max_tokens: The request's output budget, thinking included.

    Returns:
        The deadline, never above PROSE_TIMEOUT_SECONDS.
    """
    override = os.getenv("PROSE_FIRST_TOKEN_TIMEOUT_SECONDS", "").strip()
    if override:
        try:
            return min(max(float(override), 1.0), PROSE_TIMEOUT_SECONDS)
        except ValueError:
            logger.warning(
                "Ignoring PROSE_FIRST_TOKEN_TIMEOUT_SECONDS: not a number",
                extra={"value": override},
            )
    derived = FIRST_TOKEN_OVERHEAD_SECONDS + max_tokens / FIRST_TOKEN_MIN_TOKENS_PER_SECOND
    return min(max(derived, FIRST_TOKEN_FLOOR_SECONDS), PROSE_TIMEOUT_SECONDS)


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

    Also True for a 500/502 and for ProseAgentFailed — a model whose same-model
    retries all came back empty or errored. The repair is identical (try
    another model), and without it those cases matched no classifier and ended
    the caller's ladder in a refusal while six other rungs sat unused.
    """
    return classify_failure(exc).kind in (
        FailureKind.SERVICE_UNAVAILABLE,
        FailureKind.SERVER_ERROR,
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


# Characters per input token, for the estimate booked before a request is sent.
# Deliberately low (English prose runs nearer 4): figures, tables and markup
# tokenise denser than prose, and an estimate that is too high only matters
# near the ceiling, where caution is the point. It is replaced by the provider's
# own prompt-token count as soon as the response arrives.
CHARS_PER_INPUT_TOKEN = 3.0


def _estimate_input_tokens(messages: list[dict]) -> int:
    """Upper-biased estimate of a request's input tokens, from its length."""
    chars = sum(len(m.get("content") or "") for m in messages)
    return int(chars / CHARS_PER_INPUT_TOKEN) + 1


def _prompt_tokens(usage) -> int | None:
    """The provider's prompt-token count from a usage object or dict, if any."""
    value = getattr(usage, "prompt_tokens", None)
    if value is None and isinstance(usage, dict):
        value = usage.get("prompt_tokens")
    return value if isinstance(value, int) else None


async def _reserve_input_tokens(kwargs: dict, model: str, api_key: str | None):
    """
    Books the request's input against the slot's tokens-per-minute window.

    Returns:
        The reservation, or None when nothing meters this call.

    Raises:
        TokenWindowFull: The request does not fit this minute. It has not been
            sent; the unit taken for it at selection is handed back and the
            refusal noted, exactly as for a per-minute 429 from the provider.
    """
    owned = _owned_slot(api_key)
    if owned is None:
        return None
    tracker, key_index = owned
    try:
        return tracker.reserve_input_tokens(
            key_index, model, _estimate_input_tokens(kwargs["messages"])
        )
    except TokenWindowFull as full:
        logger.warning(
            "Request held back: it would overrun the slot's tokens-per-minute window",
            extra={"model": model, "key_index": key_index,
                   "retry_after": round(full.retry_after, 1)},
        )
        await _settle_failure(model, api_key, classify_failure(full))
        raise


async def _acompletion(kwargs: dict, model: str, api_key: str | None):
    """
    litellm.acompletion with the accounts settled either way: input tokens
    booked before the request and corrected after it, and the quota unit
    refunded if the provider refused.
    """
    reservation = await _reserve_input_tokens(kwargs, model, api_key)
    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as e:
        failure = classify_failure(e)
        if reservation is not None and failure.refundable:
            reservation.release()
        await _settle_failure(model, api_key, failure)
        if is_local(model) and is_context_refusal(e):
            logger.warning("Local model refused a prompt longer than its context window",
                           extra={"model": model, "num_ctx": kwargs.get("num_ctx"), "error": str(e)[-300:]})
            raise LocalContextExceeded(
                f"{model} refused a prompt longer than its context window ({kwargs.get('num_ctx')} "
                f"tokens, OLLAMA_NUM_CTX): {str(e)[-200:]}") from e
        raise
    if reservation is not None:
        reservation.adjust(_prompt_tokens(getattr(response, "usage", None)))
    return response


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
        "timeout": (
            LOCAL_STRUCTURED_TIMEOUT_SECONDS if is_local(model) else STRUCTURED_TIMEOUT_SECONDS
        ),
    }
    if is_local(model):
        kwargs.update(_local_options())
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

    first = response.choices[0]
    finish_reason = getattr(first, "finish_reason", None)
    _log_usage(response, model, agent, started, finish_reason=finish_reason, max_tokens=max_tokens)
    parsed = _parse_json_object(first.message.content, finish_reason)
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
# slot cannot change the answer, or costs more than moving on. A 500 is in the
# second group — each in-place retry is an admitted request that keeps its
# daily unit, so three of them spent 15% of a reasoning slot's day on a model
# that had just failed. A dropped connection and an empty completion are still
# retried in place first.
_PROSE_NO_RETRY_KINDS = frozenset({
    FailureKind.SERVICE_UNAVAILABLE,
    FailureKind.SERVER_ERROR,
    FailureKind.TIMEOUT,
    FailureKind.QUOTA,
    FailureKind.AUTH,
    FailureKind.MODEL_UNAVAILABLE_FOR_KEY,
    FailureKind.REQUEST_INVALID,
})

# ─── Replies cut off at max_tokens ───────────────────────────────────────────
#
# Gemini 3.x counts thinking against max_tokens, and a reply that runs into the
# limit ends with finish_reason "length" and whatever text fitted. Nothing used
# to read that: in the 2026-11-15 evening run 29 of 54 synthesis calls stopped
# at 2,996 of 3,000 tokens and shipped mid-sentence — mh_05 as 528 characters
# ending "because Adjusted EBITDA is higher", scored passed at recall 0 — and
# mh_06 failed outright when three rungs in a row were cut before their first
# citation.
#
# The repair is more room on the same model, as for a cut-off structured reply,
# not another model. The model answered promptly; it needed a bigger budget.
# Moving down the ladder did not help in that run — every rung tried for
# mh_06, legal_05 and mh_04 was cut off too, and 3.5-flash is served by the same
# backend as 3.6-flash. So a cut-off reply gets ONE retry on the same slot with
# twice the room, paid for like any retry. It is not a health failure: nothing
# is cooled and nothing is refunded (the provider served the request). Only if
# the retry is cut off as well — twice the budget suggests a model that loops,
# not one that is verbose — does the caller's ladder move on, for this request
# only (CompletionTruncated is MODEL_FAILED).
LENGTH_RETRY_GROWTH = 2


async def _room_after_cutoff(
    model: str,
    api_key: str | None,
    room: int,
    outcome: ProseOutcome,
    text: str | None,
    attempts_left: int,
) -> int:
    """
    The output budget for the one retry a cut-off reply gets, paid for.

    Args:
        model: The model that was cut off.
        api_key: Its credential, for the retry charge.
        room: The budget it was cut off at.
        outcome: The call's outcome; its length_retries is updated.
        text: What the cut-off reply contained, for the log.
        attempts_left: Requests the caller may still make on this slot.

    Returns:
        The retry's budget.

    Raises:
        CompletionTruncated: The retry was already used, the caller has no
            attempt left, or the slot cannot pay for another request.
    """
    logger.warning(
        "Prose reply cut off at the output limit",
        extra={"model": model, "max_tokens": room, "reasoning_tokens": outcome.reasoning_tokens,
               "visible_chars": len(text or ""), "length_retries": outcome.length_retries},
    )
    if outcome.length_retries >= 1 or attempts_left < 1:
        raise CompletionTruncated(
            f"prose reply from {model} was cut off at {room} tokens "
            f"after {outcome.length_retries} retry with more room"
        )
    if not await _charge_retry(model, api_key):
        raise CompletionTruncated(
            f"prose reply from {model} was cut off at {room} tokens and the slot "
            f"has no allowance left for a retry"
        )
    outcome.length_retries += 1
    return room * LENGTH_RETRY_GROWTH


async def call_prose_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 3000,
    api_key: str | None = None,
    agent: str | None = None,
    stream_with_deadline: bool = False,
    outcome: ProseOutcome | None = None,
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
        stream_with_deadline: Stream the request internally so the
            first-token deadline applies (see first_token_deadline), and return
            the whole text exactly as the blocking call does. A blocking
            request cannot tell a model that is thinking from one that is hung
            until the total timeout; a streamed one can. Ignored for the local
            model, which has no rung below it to fall to.
        outcome: Optional holder told how the returned text ended (finish
            reason, final budget, length retries).

    Returns:
        Raw string response from the agent. Never None, and never a reply cut
        off at max_tokens.

    Raises:
        CompletionTruncated: A reply was cut off at max_tokens and the one
            retry with twice the room was cut off too or could not be paid for.
        ProseAgentFailed: (a RuntimeError) every attempt failed or returned an
            empty completion. Classified so a ladder moves to another model.
        StreamInterrupted: With stream_with_deadline, the stream failed after
            answer text had started; the model is reported unhealthy.
        Exception: A provider refusal the caller's ladder classifies — 429,
            503, timeout (including a missed first-token deadline), rejected
            key, revoked model — raised on first sight.

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
    streamed = stream_with_deadline and not is_local(model)
    outcome = outcome if outcome is not None else ProseOutcome()
    room = max_tokens
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    local_options = _local_options() if is_local(model) else {}

    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        logger.info(
            "Calling prose agent",
            extra={
                "model": model,
                "attempt": attempt,
                "temperature": temperature,
                "max_tokens": room,
            },
        )

        try:
            prose_kwargs = {
                "model": model,
                "messages": [dict(m) for m in messages],  # a fresh list per attempt, as before
                "temperature": temperature,
                "max_tokens": room,
                "timeout": PROSE_TIMEOUT_SECONDS,
                **local_options,
            }
            if api_key:
                prose_kwargs["api_key"] = api_key
            metadata = _call_metadata(agent)
            if metadata:
                prose_kwargs["metadata"] = metadata
            if streamed:
                probe = ProseOutcome(room)
                content = await _streamed_text(
                    prose_kwargs, model, api_key, agent,
                    on_token=_discard_token,
                    first_token_timeout=first_token_deadline(room),
                    outcome=probe,
                )
                finish_reason, reasoning = probe.finish_reason, probe.reasoning_tokens
            else:
                started = time.monotonic()
                response = await _acompletion(prose_kwargs, model, api_key)
                first = response.choices[0]
                finish_reason = getattr(first, "finish_reason", None)
                reasoning = _reasoning_tokens(getattr(response, "usage", None))
                _log_usage(response, model, agent, started,
                           finish_reason=finish_reason, max_tokens=room)
                content = first.message.content

        except StreamInterrupted:
            # The model stopped mid-answer and has already been reported; a
            # same-slot retry would ask the failing model again.
            raise
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
        else:
            outcome.finish_reason, outcome.reasoning_tokens = finish_reason, reasoning
            outcome.max_tokens = room
            if finish_reason == "length":
                # Cut off at max_tokens, with or without visible text. Never
                # returned as an answer: the model gets the room it needed,
                # once, on the same slot (see _room_after_cutoff).
                room = await _room_after_cutoff(
                    model, api_key, room, outcome, content, attempts_left=MAX_RETRIES - attempt
                )
                continue
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


def _discard_token(_text: str) -> None:
    """on_token for a stream nobody is watching (the blocking /query path)."""


async def _await_within(awaitable, seconds: float):
    """
    Awaits `awaitable` for at most `seconds`.

    asyncio.wait rather than wait_for, so that a TimeoutError raised BY the
    provider call (LiteLLM's own timeout derives from it) can never be mistaken
    for our deadline expiring, or the other way round.

    Returns:
        (True, result) when it finished in time; (False, None) when the
        deadline expired, in which case it has been cancelled.

    Raises:
        Whatever the awaitable raised (StopAsyncIteration included).
    """
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=max(0.0, seconds))
    except asyncio.CancelledError:
        task.cancel()
        raise
    if not done:
        task.cancel()
        await asyncio.wait({task})
        return False, None
    return True, task.result()


async def _close_stream(stream) -> None:
    """
    Best-effort release of an abandoned stream's connection.

    Without it an abandoned request keeps its socket until garbage collection,
    and the provider keeps generating into it. Bounded, and never raises:
    cleanup must not turn a fallback into a failure.
    """
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    try:
        await asyncio.wait_for(aclose(), timeout=2.0)
    except Exception as e:  # noqa: BLE001
        logger.debug("Closing an abandoned stream failed", extra={"error": str(e)})


async def _streamed_text(
    kwargs: dict,
    model: str,
    api_key: str | None,
    agent: str | None,
    on_token,
    first_token_timeout: float | None,
    outcome: "ProseOutcome | None" = None,
) -> str:
    """
    One streamed provider request under the wrapper's deadlines.

    Three deadlines, each answering a different question:
      first_token — has the model started answering? (see first_token_deadline)
      stall       — once it has, is it still going? (STREAM_STALL_SECONDS)
      total       — has the whole request run out of time? (PROSE_TIMEOUT_SECONDS)

    The total one is enforced HERE because LiteLLM 1.89.3 does not enforce it on
    this path: its Gemini streaming call ignores the `timeout` argument and
    sends the request on litellm.module_level_aclient, whose read timeout is
    litellm.request_timeout — 6000s unless REQUEST_TIMEOUT is set. The streamed
    synthesis path was bounded only by the pipeline's own 900s deadline while
    the blocking path had two minutes.

    Accounts are settled like a blocking request: input tokens booked before,
    corrected from the stream's usage after (requested with include_usage —
    without it LiteLLM drops usage from the stream and the booking stays an
    estimate; the live run logged prompt_tokens=null for every streamed call),
    and a refusal is refunded. A missed deadline is a TIMEOUT and, like one,
    keeps its unit: the provider admitted the request.

    Args:
        kwargs: litellm.acompletion arguments; `stream` is set here.
        model: Upstream model.
        api_key: Credential the request is sent with.
        agent: Calling agent, for the usage log.
        on_token: Receives each text delta.
        first_token_timeout: Seconds to the first text delta, or None for no
            first-token deadline (the total and stall deadlines still apply).
        outcome: Receives the stream's finish reason and reasoning tokens.
            LiteLLM puts the finish reason on the stream's last choice, so a
            stream cut off at max_tokens ends normally and only this tells.

    Returns:
        The concatenated text (possibly empty — the caller decides).

    Raises:
        ResponseDeadlineExceeded: A deadline expired before any text.
        StreamInterrupted: The stream failed or stalled after text was emitted;
            the model has been reported unhealthy.
        Exception: The provider's own failure before any text, accounts settled.
    """
    kwargs["stream"] = True
    if not is_local(model):
        kwargs["stream_options"] = {"include_usage": True}
    total = float(kwargs.get("timeout") or PROSE_TIMEOUT_SECONDS)

    loop = asyncio.get_running_loop()
    opened_at = loop.time()
    first_by = opened_at + min(first_token_timeout or total, total)
    total_by = opened_at + total

    parts: list[str] = []
    emitted = 0
    usage = None
    finish_reason = None
    stream = None
    started = time.monotonic()
    reservation = await _reserve_input_tokens(kwargs, model, api_key)
    try:
        ok, stream = await _await_within(litellm.acompletion(**kwargs), first_by - loop.time())
        if not ok:
            raise ResponseDeadlineExceeded(model, "first_token", first_by - opened_at)
        iterator = stream.__aiter__()
        last_chunk_at = opened_at
        while True:
            now = loop.time()
            if not emitted:
                phase, until = "first_token", first_by
            else:
                stall_by = last_chunk_at + STREAM_STALL_SECONDS
                phase, until = ("stall", stall_by) if stall_by < total_by else ("total", total_by)
            try:
                ok, chunk = await _await_within(iterator.__anext__(), until - now)
            except StopAsyncIteration:
                break
            if not ok:
                limit = {"first_token": first_by - opened_at, "stall": STREAM_STALL_SECONDS,
                         "total": total}[phase]
                raise ResponseDeadlineExceeded(model, phase, limit)
            last_chunk_at = loop.time()
            usage = getattr(chunk, "usage", None) or usage
            try:
                choice = chunk.choices[0]
            except (AttributeError, IndexError, TypeError):
                choice = None
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            try:
                delta = choice.delta.content
            except AttributeError:
                delta = None
            if delta:
                parts.append(delta)
                emitted += len(delta)
                on_token(delta)
    except asyncio.CancelledError:
        # The caller gave up (a node or pipeline deadline): release the
        # connection rather than leave the provider generating into it.
        await _close_stream(stream)
        raise
    except Exception as e:
        await _close_stream(stream)
        if emitted:
            # A stream that drops or stalls mid-answer is the provider failing,
            # whatever the exception class says.
            _report_unhealthy(model)
            raise StreamInterrupted(
                f"stream from {model} failed after {emitted} chars: {e}", emitted, e
            ) from e
        failure = classify_failure(e)
        if reservation is not None and failure.refundable:
            reservation.release()
        await _settle_failure(model, api_key, failure)
        if isinstance(e, ResponseDeadlineExceeded):
            logger.warning(
                "Streamed request abandoned: the model did not start answering in time",
                extra={"model": model, "deadline_seconds": round(e.seconds, 1),
                       "phase": e.phase},
            )
        raise

    _log_usage(
        _UsageHolder(usage), model, agent, started,
        finish_reason=finish_reason, max_tokens=kwargs.get("max_tokens"),
    )
    if reservation is not None:
        reservation.adjust(_prompt_tokens(usage))
    if outcome is not None:
        outcome.finish_reason = finish_reason
        outcome.reasoning_tokens = _reasoning_tokens(usage)
    return "".join(parts)


async def stream_prose_agent(
    system_prompt: str,
    user_prompt: str,
    model: str,
    on_token,
    temperature: float = 0.1,
    max_tokens: int = 3000,
    api_key: str | None = None,
    agent: str | None = None,
    outcome: ProseOutcome | None = None,
) -> str:
    """
    Prose completion streamed token by token, returning the full text.

    Failure semantics mirror call_prose_agent so the synthesizer's ladder logic
    is unchanged:

    - fails BEFORE the first token: a 503 / timeout / quota / auth error is
      re-raised for the ladder to classify — and so is a missed first-token
      deadline (first_token_deadline(max_tokens), not applied to the local
      model), which is a timeout; any other error falls back to the
      non-streamed call_prose_agent on the same model (which retries), and its
      whole answer is emitted as one token.
    - fails AFTER tokens were emitted (including a stall): raises
      StreamInterrupted, and the caller emits an answer reset before trying
      elsewhere.
    - is cut off at max_tokens (finish_reason "length"): before any text, it
      is retried once with twice the room exactly as call_prose_agent does —
      the client has seen nothing. After text, the tokens are already on the
      client's screen, so the text is returned and `outcome.truncated` tells
      the caller to mark the answer as cut off.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User prompt.
        model: LiteLLM model string.
        on_token: Callable receiving each text delta.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        api_key: Credential for this call.
        agent: Calling agent's name.
        outcome: Optional holder told how the returned text ended. Callers
            must read `outcome.truncated`: on this path a cut-off answer is
            returned, not raised.

    Returns:
        The generated text. Never empty; possibly cut off (see outcome).

    Raises:
        StreamInterrupted: The stream broke after emitting text.
        CompletionTruncated: Cut off before any text, twice.
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
    }
    if is_local(model):
        kwargs.update(_local_options())
    if api_key:
        kwargs["api_key"] = api_key
    metadata = _call_metadata(agent)
    if metadata:
        kwargs["metadata"] = metadata
    outcome = outcome if outcome is not None else ProseOutcome()
    room = max_tokens

    try:
        while True:
            kwargs["max_tokens"] = room
            probe = ProseOutcome(room)
            text = await _streamed_text(
                kwargs, model, api_key, agent, on_token,
                first_token_timeout=None if is_local(model) else first_token_deadline(room),
                outcome=probe,
            )
            outcome.finish_reason, outcome.reasoning_tokens = probe.finish_reason, probe.reasoning_tokens
            outcome.max_tokens = room
            if not (probe.truncated and not text.strip()):
                break
            # Cut off while still thinking: nothing has reached the client, so
            # the retry with more room is as invisible here as on /query.
            room = await _room_after_cutoff(model, api_key, room, outcome, text, attempts_left=1)
    except StreamInterrupted:
        raise
    except CompletionTruncated:
        raise
    except Exception as e:
        failure = classify_failure(e)
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
            max_tokens=room,
            api_key=api_key,
            agent=agent,
            outcome=outcome,
        )
        on_token(text)
        return text

    if not text.strip():
        # Typed so the caller's ladder tries another model; as a bare
        # RuntimeError it matched no classifier and ended in a refusal.
        raise ProseAgentFailed(f"empty streamed completion (model={model})")
    if outcome.truncated:
        # The answer is already on the client's screen, so it cannot be
        # silently replaced. It is returned, and the caller marks it as cut off.
        logger.warning(
            "Streamed answer cut off at the output limit; returned flagged as truncated",
            extra={"model": model, "max_tokens": room, "chars": len(text),
                   "reasoning_tokens": outcome.reasoning_tokens},
        )
    return text
