"""
Failure taxonomy for provider calls — one classifier, one answer per exception.

WHY THIS EXISTS
---------------
The routing layer repairs different failures at different scopes: a 429 spends
one (key, model) slot, a 503 sidelines a model for every key, a rejected
credential retires a key, a revoked model retires one slot for good. Those
decisions used to rest on five independent string predicates that each
re-derived the answer from `str(exc)`, so two of them could both claim the same
exception and bare digits inside an unrelated message ("time taken=503.2s")
could decide the route.

`classify_failure` returns exactly one `Failure`. LiteLLM's typed exceptions
are trusted first — it maps Gemini's HTTP 400/401/403/404/408/429/500/502/503 to
distinct classes — and the message is only consulted to refine a class that is
ambiguous by design (Gemini reports a bad API key as HTTP 400) or when the
exception is not a LiteLLM type at all (test doubles, raw transport errors).

A 429 additionally carries WHICH quota tripped. Gemini's body names it
(`quotaId: ...PerDay...` / `...PerMinute...`) and that distinction is worth a
lot on the free tier: a per-minute refusal clears in seconds, while treating it
as a spent day discards up to 475 requests on a volume slot.
"""

import asyncio
import re
from dataclasses import dataclass
from enum import Enum

import litellm


class FailureKind(str, Enum):
    """What went wrong, named by the repair it calls for."""

    QUOTA = "quota"                        # 429 — this (key, model) slot is refused
    AUTH = "auth"                          # the credential itself was rejected
    MODEL_UNAVAILABLE_FOR_KEY = "model_unavailable_for_key"  # 404 for this slot
    SERVICE_UNAVAILABLE = "service_unavailable"  # 503 — the model, for every key
    TIMEOUT = "timeout"                    # the request burned its deadline
    SERVER_ERROR = "server_error"          # 500/502/504 — provider-side fault
    CONNECTION = "connection"              # no HTTP response was received
    BAD_OUTPUT = "bad_output"              # 200, but empty / unparseable / wrong shape
    MODEL_FAILED = "model_failed"          # same-model retries exhausted
    REQUEST_INVALID = "request_invalid"    # 4xx the next rung would repeat
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Failure:
    """
    A classified provider failure.

    Attributes:
        kind: The failure class.
        quota_window: For QUOTA only — "day", "minute", or None when the
            provider did not say. None is treated as "day": the daily cap is
            the binding limit on the free tier, and assuming the short window
            would re-offer a spent slot on every request.
        retry_after: Seconds the provider asked us to wait, when it said.
    """

    kind: FailureKind
    quota_window: str | None = None
    retry_after: float | None = None

    @property
    def is_per_minute_quota(self) -> bool:
        """True for a 429 the provider attributed to a per-minute limit."""
        return self.kind is FailureKind.QUOTA and self.quota_window == "minute"

    @property
    def refundable(self) -> bool:
        """
        True when the provider answered that it did NOT serve the request, so
        the daily unit reserved for it should be handed back.

        Deliberately narrow. A timeout or a 500 is not refunded: the request
        was admitted and may well have been counted upstream, and a local
        counter that under-counts produces surprise 429s. A per-day 429 is not
        refunded either — the slot is about to be marked spent regardless.
        """
        if self.kind is FailureKind.QUOTA:
            return self.quota_window == "minute"
        return self.kind in _REFUNDABLE_KINDS

    @property
    def another_model_helps(self) -> bool:
        """True when the right next move is a different model, not another key."""
        return self.kind in _MODEL_SCOPED_KINDS


# The provider explicitly refused or never received the request.
_REFUNDABLE_KINDS = frozenset({
    FailureKind.AUTH,
    FailureKind.MODEL_UNAVAILABLE_FOR_KEY,
    FailureKind.SERVICE_UNAVAILABLE,
    FailureKind.CONNECTION,
})

# Failures that say something about the model rather than the credential.
_MODEL_SCOPED_KINDS = frozenset({
    FailureKind.SERVICE_UNAVAILABLE,
    FailureKind.TIMEOUT,
    FailureKind.SERVER_ERROR,
    FailureKind.CONNECTION,
    FailureKind.MODEL_FAILED,
})


class StructuredOutputError(ValueError):
    """
    The provider returned 200 but the content is not the JSON object asked for.

    A ValueError because that is what callers of call_structured_agent were
    always told to expect for invalid JSON. `problem` is one of "empty",
    "truncated", "malformed", "wrong_shape".
    """

    def __init__(self, problem: str, detail: str, raw: str | None = None):
        preview = (raw or "")[:200]
        super().__init__(f"Agent returned unusable JSON ({problem}): {detail}\nRaw: {preview}")
        self.problem = problem
        self.raw = raw


class TokenWindowFull(Exception):
    """
    A request was not sent because its input would overrun the slot's
    tokens-per-minute allowance.

    Raised locally, before any provider call, and classified as a per-minute
    quota refusal: the provider would have answered 429, and the repair is the
    same — rest this slot, send the request to another one.
    """

    def __init__(self, model: str, tokens: int, retry_after: float):
        super().__init__(
            f"input of ~{tokens} tokens does not fit this minute's token window "
            f"for {model}; next opening in {retry_after:.0f}s"
        )
        self.retry_after = retry_after


class ResponseDeadlineExceeded(TimeoutError):
    """
    A streamed completion missed one of the wrapper's own deadlines.

    Raised locally, not by the provider: no text by the first-token deadline,
    a silent gap after text had started, or the total budget spent. Classified
    exactly like a provider timeout (TIMEOUT): the model is unresponsive, so the
    next move is another model, and the request is not refunded — it reached
    the provider and was admitted, just as a request that runs into the HTTP
    timeout was.

    Attributes:
        phase: "first_token", "stall" or "total".
        seconds: The deadline that expired.
    """

    def __init__(self, model: str, phase: str, seconds: float):
        super().__init__(
            f"{model}: no {'text' if phase == 'first_token' else 'progress'} within "
            f"{seconds:.0f}s ({phase.replace('_', '-')} deadline)"
        )
        self.phase = phase
        self.seconds = seconds


class ProseAgentFailed(RuntimeError):
    """
    A prose call exhausted its same-model retries without a usable completion.

    Classified as MODEL_FAILED so a ladder moves to the next model. It used to
    be a bare RuntimeError that matched no classifier, which turned three empty
    completions from one model into a refusal while six other rungs sat unused.
    """


_QUOTA_ID_DAY = re.compile(r"per\s*_?day", re.IGNORECASE)
_QUOTA_ID_MINUTE = re.compile(r"per\s*_?minute", re.IGNORECASE)
_RETRY_DELAY = re.compile(
    r"retry_?delay\W{1,6}(\d+(?:\.\d+)?)\s*s|retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE
)


def _has_status(text: str, code: int) -> bool:
    """True if `code` appears as a standalone number, not inside another one."""
    return re.search(rf"(?<![\d.]){code}(?![\d]|\.\d)", text) is not None


def _retry_after(exc: Exception, text: str) -> float | None:
    """Seconds to wait, from the Retry-After header or the body's RetryInfo."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
            if value is not None:
                return max(0.0, float(value))
        except (TypeError, ValueError, AttributeError):
            pass
    match = _RETRY_DELAY.search(text)
    if match:
        return float(match.group(1) or match.group(2))
    return None


def _quota(exc: Exception, text: str) -> Failure:
    """Builds the QUOTA failure, reading which window tripped from the body."""
    # A body can list several violations; the daily one is the one that matters,
    # because it will still be true after the per-minute one has cleared.
    if _QUOTA_ID_DAY.search(text):
        window = "day"
    elif _QUOTA_ID_MINUTE.search(text):
        window = "minute"
    else:
        window = None
    return Failure(FailureKind.QUOTA, quota_window=window, retry_after=_retry_after(exc, text))


def _is_bad_key_text(text: str) -> bool:
    return (
        "api key not valid" in text
        or "api_key_invalid" in text
        or "access_token_type_unsupported" in text
        or "unauthenticated" in text
        or "permission_denied" in text
    )


def _is_revoked_model_text(text: str) -> bool:
    return (
        "no longer available" in text
        or "not found for api version" in text
        or "is not supported for generatecontent" in text
    )


def _is_unavailable_text(text: str) -> bool:
    return (
        _has_status(text, 503)
        or "unavailable" in text
        or "overloaded" in text
        or "experiencing high demand" in text
    )


def _is_timeout_text(text: str) -> bool:
    return "timed out" in text or "timeout passed" in text


def _is_quota_text(text: str) -> bool:
    return _has_status(text, 429) or "resource_exhausted" in text or "quota" in text


def _from_text(exc: Exception, name: str, text: str) -> Failure | None:
    """
    Classification from the class name and message alone.

    Used for exceptions that are not LiteLLM types, and to refine LiteLLM's
    catch-all classes. The class name outranks the message: an exception called
    RateLimitError is a rate limit whatever digits its message contains.
    """
    if "ratelimit" in name:
        return _quota(exc, text)
    if "authentication" in name or "permissiondenied" in name:
        return Failure(FailureKind.AUTH)
    if "notfound" in name:
        return Failure(FailureKind.MODEL_UNAVAILABLE_FOR_KEY) if _is_revoked_model_text(text) else None
    if "timeout" in name:
        return Failure(FailureKind.TIMEOUT)
    if "serviceunavailable" in name or "overloaded" in name:
        return Failure(FailureKind.SERVICE_UNAVAILABLE)

    if _is_bad_key_text(text):
        return Failure(FailureKind.AUTH)
    if _has_status(text, 404) and _is_revoked_model_text(text):
        return Failure(FailureKind.MODEL_UNAVAILABLE_FOR_KEY)
    if _is_quota_text(text):
        return _quota(exc, text)
    if _is_timeout_text(text):
        return Failure(FailureKind.TIMEOUT)
    if _is_unavailable_text(text):
        return Failure(FailureKind.SERVICE_UNAVAILABLE)
    return None


def classify_failure(exc: BaseException) -> Failure:
    """
    Classifies an exception raised by (or around) a provider call.

    Args:
        exc: The exception.

    Returns:
        Exactly one Failure. UNKNOWN when nothing matches — callers must treat
        that as "do not assume another rung would behave differently".
    """
    if isinstance(exc, StructuredOutputError):
        return Failure(FailureKind.BAD_OUTPUT)
    if isinstance(exc, ProseAgentFailed):
        return Failure(FailureKind.MODEL_FAILED)
    if isinstance(exc, TokenWindowFull):
        return Failure(FailureKind.QUOTA, quota_window="minute", retry_after=exc.retry_after)
    if isinstance(exc, ResponseDeadlineExceeded):
        return Failure(FailureKind.TIMEOUT)
    if not isinstance(exc, Exception):
        return Failure(FailureKind.UNKNOWN)

    name = type(exc).__name__.lower()
    text = str(exc).lower()

    # ── LiteLLM's typed exceptions: the status code already decided the class.
    if isinstance(exc, litellm.RateLimitError):
        return _quota(exc, text)
    if isinstance(exc, (litellm.AuthenticationError, litellm.PermissionDeniedError)):
        return Failure(FailureKind.AUTH)
    if isinstance(exc, litellm.NotFoundError):
        # Any model-level 404 means this key cannot reach this model; the
        # "closed to new users" wording is just the case seen in practice.
        return Failure(FailureKind.MODEL_UNAVAILABLE_FOR_KEY)
    if isinstance(exc, litellm.Timeout):
        # Kept ahead of the connection classes: Timeout derives from openai's
        # APIConnectionError, so a looser check further down would claim it.
        return Failure(FailureKind.TIMEOUT)
    if isinstance(exc, litellm.ServiceUnavailableError):
        return Failure(FailureKind.SERVICE_UNAVAILABLE)
    if isinstance(exc, litellm.BadRequestError):
        # Gemini reports an invalid API key as HTTP 400, so the class alone
        # cannot tell a dead credential from a malformed request.
        if _is_bad_key_text(text):
            return Failure(FailureKind.AUTH)
        return Failure(FailureKind.REQUEST_INVALID)
    if isinstance(exc, litellm.UnprocessableEntityError):
        return Failure(FailureKind.REQUEST_INVALID)

    # ── Catch-all classes: LiteLLM also files unmapped provider errors here, so
    # the message gets a chance to be more specific before the class decides.
    if isinstance(exc, (litellm.InternalServerError, litellm.BadGatewayError)):
        return _from_text(exc, "", text) or Failure(FailureKind.SERVER_ERROR)
    if isinstance(exc, litellm.APIConnectionError):
        return _from_text(exc, "", text) or Failure(FailureKind.CONNECTION)
    if isinstance(exc, litellm.APIError):
        return _from_text(exc, "", text) or Failure(FailureKind.SERVER_ERROR)

    # ── Not a LiteLLM type.
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return Failure(FailureKind.TIMEOUT)
    refined = _from_text(exc, name, text)
    if refined is not None:
        return refined
    if isinstance(exc, (ConnectionError, OSError)):
        return Failure(FailureKind.CONNECTION)
    return Failure(FailureKind.UNKNOWN)
