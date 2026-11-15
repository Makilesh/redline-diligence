"""
Failure classification — the one decision every routing repair hangs off.

Built from LiteLLM's real exception classes (constructed the way its Gemini
mapping constructs them) rather than look-alikes, because the classifier's
first line of evidence is the type.
"""

import json

import litellm
import pytest

from src.llm.failures import (
    FailureKind,
    ProseAgentFailed,
    StructuredOutputError,
    classify_failure,
)


def _gemini_429(quota_id: str | None, retry_delay: str | None = "31s") -> litellm.RateLimitError:
    details = []
    if quota_id:
        details.append({
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{
                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                "quotaId": quota_id,
            }],
        })
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": retry_delay})
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                      "message": "You exceeded your current quota.", "details": details}}
    return litellm.RateLimitError(
        message="GeminiException - " + json.dumps(body), llm_provider="gemini", model="m"
    )


class TestTypedExceptions:
    """Each HTTP status LiteLLM maps for Gemini lands on exactly one kind."""

    @pytest.mark.parametrize("exc, kind", [
        (litellm.ServiceUnavailableError(message="503 high demand", llm_provider="gemini", model="m"),
         FailureKind.SERVICE_UNAVAILABLE),
        (litellm.Timeout(message="Timeout passed=60.0", model="m", llm_provider="gemini"),
         FailureKind.TIMEOUT),
        (litellm.AuthenticationError(message="invalid credentials", llm_provider="gemini", model="m"),
         FailureKind.AUTH),
        (litellm.PermissionDeniedError(
            message="PERMISSION_DENIED", llm_provider="gemini", model="m",
            response=litellm.RateLimitError(message="x", llm_provider="g", model="m").response),
         FailureKind.AUTH),
        (litellm.NotFoundError(message="models/x is not found", model="m", llm_provider="gemini"),
         FailureKind.MODEL_UNAVAILABLE_FOR_KEY),
        (litellm.InternalServerError(message="An internal error has occurred", llm_provider="gemini", model="m"),
         FailureKind.SERVER_ERROR),
        (litellm.APIConnectionError(message="Connection reset by peer", llm_provider="gemini", model="m"),
         FailureKind.CONNECTION),
        (litellm.BadRequestError(message="Request contains an invalid argument", model="m", llm_provider="gemini"),
         FailureKind.REQUEST_INVALID),
        (litellm.ContextWindowExceededError(message="too long", model="m", llm_provider="gemini"),
         FailureKind.REQUEST_INVALID),
    ])
    def test_status_class_decides(self, exc, kind):
        assert classify_failure(exc).kind is kind

    def test_timeout_is_not_mistaken_for_a_connection_error(self):
        """litellm.Timeout is an openai.APIConnectionError by inheritance."""
        import openai

        exc = litellm.Timeout(message="timed out", model="m", llm_provider="gemini")
        assert isinstance(exc, openai.APIConnectionError)  # the trap being guarded
        assert classify_failure(exc).kind is FailureKind.TIMEOUT

    def test_gemini_reports_a_dead_key_as_http_400(self):
        """The class says 'bad request'; only the body says the key is dead."""
        exc = litellm.BadRequestError(
            message='GeminiException BadRequestError - {"error": {"code": 400, "message": '
                    '"API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}',
            model="m", llm_provider="gemini",
        )
        assert classify_failure(exc).kind is FailureKind.AUTH

    def test_daily_quota_wrapped_as_http_400_is_a_quota_failure(self):
        """Seen live on the streamed path: the class says 400, the body says 429."""
        exc = litellm.BadRequestError(
            message="Vertex_ai_betaException BadRequestError - b'{\\n  \"error\": {\\n    "
                    "\"code\": 429,\\n    \"message\": \"You exceeded your current quota. "
                    "Please retry in 16h36m24.27s.\",\\n    \"status\": \"RESOURCE_EXHAUSTED\",\\n"
                    "    \"details\": [{\"quotaId\": \"GenerateRequestsPerDayPerProjectPerModel-FreeTier\"}]}}'",
            model="m", llm_provider="gemini",
        )
        f = classify_failure(exc)
        assert f.kind is FailureKind.QUOTA
        assert f.quota_window == "day"

    def test_a_bare_429_in_a_bad_request_message_is_not_quota(self):
        """Only the error body's own status counts, not a number in the prose."""
        exc = litellm.BadRequestError(
            message='GeminiException BadRequestError - {"error": {"code": 400, "message": '
                    '"Invalid value at position 429", "status": "INVALID_ARGUMENT"}}',
            model="m", llm_provider="gemini",
        )
        assert classify_failure(exc).kind is FailureKind.REQUEST_INVALID

    def test_unmapped_provider_error_in_a_catch_all_class_is_refined(self):
        """LiteLLM files errors it cannot map under APIConnectionError."""
        exc = litellm.APIConnectionError(
            message='GeminiException - {"error": {"code": 503, "status": "UNAVAILABLE"}}',
            llm_provider="gemini", model="m",
        )
        assert classify_failure(exc).kind is FailureKind.SERVICE_UNAVAILABLE


class TestQuotaWindow:
    """A 429 names the quota that tripped; the repair depends on which."""

    def test_per_day_quota(self):
        f = classify_failure(_gemini_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
        assert f.kind is FailureKind.QUOTA and f.quota_window == "day"
        assert not f.is_per_minute_quota and not f.refundable

    def test_per_minute_quota(self):
        f = classify_failure(_gemini_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"))
        assert f.quota_window == "minute" and f.is_per_minute_quota
        assert f.refundable, "the provider did not serve it; the unit must come back"
        assert f.retry_after == 31.0

    def test_token_per_minute_quota_is_a_minute_window(self):
        f = classify_failure(_gemini_429("GenerateContentInputTokensPerModelPerMinute-FreeTier"))
        assert f.is_per_minute_quota

    def test_unnamed_quota_is_treated_as_daily(self):
        """No quota id: assume the binding limit, never the optimistic one."""
        f = classify_failure(_gemini_429(None, retry_delay=None))
        assert f.kind is FailureKind.QUOTA and f.quota_window is None
        assert not f.is_per_minute_quota and not f.refundable

    def test_daily_wins_when_both_are_listed(self):
        exc = litellm.RateLimitError(
            message='quotaId: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", '
                    'quotaId: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"',
            llm_provider="gemini", model="m",
        )
        assert classify_failure(exc).quota_window == "day"

    def test_retry_after_header_is_read(self):
        import httpx

        response = httpx.Response(
            429, headers={"retry-after": "12"},
            request=httpx.Request("POST", "https://example.invalid/"),
        )
        exc = litellm.RateLimitError(message="429", llm_provider="gemini", model="m",
                                     response=response)
        assert classify_failure(exc).retry_after == 12.0

    def test_retry_hint_in_the_message_is_read(self):
        exc = litellm.RateLimitError(message="Please retry in 7.5s.", llm_provider="gemini", model="m")
        assert classify_failure(exc).retry_after == 7.5


class TestUntypedExceptions:
    """Exceptions that are not LiteLLM types still classify, by name then text."""

    def test_digits_inside_another_number_do_not_decide_the_route(self):
        """'time taken=503.2' is not a 503 and '14290 tokens' is not a 429."""
        assert classify_failure(Exception("time taken=503.2 seconds")).kind is FailureKind.UNKNOWN
        assert classify_failure(Exception("prompt was 14290 tokens")).kind is FailureKind.UNKNOWN

    def test_standalone_status_codes_still_count(self):
        assert classify_failure(Exception("429 Too Many Requests")).kind is FailureKind.QUOTA
        assert classify_failure(Exception('{"code": 503}')).kind is FailureKind.SERVICE_UNAVAILABLE

    def test_builtin_transport_errors(self):
        assert classify_failure(ConnectionError("connection reset")).kind is FailureKind.CONNECTION
        assert classify_failure(TimeoutError("timed out")).kind is FailureKind.TIMEOUT

    def test_a_parse_error_is_not_a_provider_failure(self):
        assert classify_failure(ValueError("invalid JSON")).kind is FailureKind.UNKNOWN

    def test_exactly_one_kind_per_exception(self):
        """The old predicates could overlap; one classifier cannot."""
        from src.llm.litellm_wrapper import (
            is_auth_error, is_model_unavailable_for_key, is_quota_error,
            is_service_unavailable, is_timeout_error,
        )

        predicates = (is_auth_error, is_model_unavailable_for_key, is_quota_error,
                      is_service_unavailable, is_timeout_error)
        samples = [
            _gemini_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
            litellm.ServiceUnavailableError(message="503", llm_provider="g", model="m"),
            litellm.Timeout(message="t", model="m", llm_provider="g"),
            litellm.AuthenticationError(message="a", llm_provider="g", model="m"),
            litellm.NotFoundError(message="no longer available", model="m", llm_provider="g"),
            Exception("429 quota exceeded; service unavailable; timed out"),
        ]
        for exc in samples:
            assert sum(bool(p(exc)) for p in predicates) == 1, repr(exc)

    def test_a_500_sends_the_ladder_to_another_model(self):
        """It matched no predicate, so the synthesizer's ladder stopped and refused."""
        from src.llm.litellm_wrapper import is_quota_error, is_service_unavailable

        exc = litellm.InternalServerError(message="internal", llm_provider="g", model="m")
        assert is_service_unavailable(exc) and not is_quota_error(exc)


class TestOwnErrorTypes:
    def test_unusable_structured_output(self):
        exc = StructuredOutputError("malformed", "Expecting value", raw="not json")
        assert isinstance(exc, ValueError), "callers were promised ValueError for bad JSON"
        assert classify_failure(exc).kind is FailureKind.BAD_OUTPUT

    def test_exhausted_prose_retries_send_the_ladder_to_another_model(self):
        from src.llm.litellm_wrapper import is_service_unavailable

        exc = ProseAgentFailed("Prose agent failed after 3 attempts")
        assert isinstance(exc, RuntimeError)
        failure = classify_failure(exc)
        assert failure.kind is FailureKind.MODEL_FAILED and failure.another_model_helps
        assert is_service_unavailable(exc)

    def test_a_full_token_window_is_a_per_minute_refusal(self):
        from src.llm.failures import TokenWindowFull

        failure = classify_failure(TokenWindowFull("gemini/m", 9000, retry_after=17.0))
        assert failure.is_per_minute_quota and failure.refundable
        assert failure.retry_after == 17.0

    def test_refund_policy_is_narrow(self):
        """Only explicit refusals come back; an admitted request may have counted."""
        refunded = {k for k in FailureKind
                    if classify_failure(_sample(k)).refundable}
        assert refunded == {FailureKind.AUTH, FailureKind.MODEL_UNAVAILABLE_FOR_KEY,
                            FailureKind.SERVICE_UNAVAILABLE, FailureKind.CONNECTION}


def _sample(kind: FailureKind) -> Exception:
    """One representative exception per kind (per-day flavour for QUOTA)."""
    return {
        FailureKind.QUOTA: _gemini_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
        FailureKind.AUTH: litellm.AuthenticationError(message="a", llm_provider="g", model="m"),
        FailureKind.MODEL_UNAVAILABLE_FOR_KEY:
            litellm.NotFoundError(message="gone", model="m", llm_provider="g"),
        FailureKind.SERVICE_UNAVAILABLE:
            litellm.ServiceUnavailableError(message="503", llm_provider="g", model="m"),
        FailureKind.TIMEOUT: litellm.Timeout(message="t", model="m", llm_provider="g"),
        FailureKind.SERVER_ERROR:
            litellm.InternalServerError(message="internal", llm_provider="g", model="m"),
        FailureKind.CONNECTION:
            litellm.APIConnectionError(message="reset", llm_provider="g", model="m"),
        FailureKind.BAD_OUTPUT: StructuredOutputError("empty", "no content"),
        FailureKind.MODEL_FAILED: ProseAgentFailed("x"),
        FailureKind.REQUEST_INVALID:
            litellm.BadRequestError(message="invalid argument", model="m", llm_provider="g"),
        FailureKind.UNKNOWN: ValueError("x"),
    }[kind]
