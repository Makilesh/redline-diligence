"""
Provider model registry — the single source of truth for model capabilities
and the quotas the provider actually enforces.

WHY THIS EXISTS
---------------
Limits used to be scattered as literals across budget_tracker (RPM here, RPD
there, a model string in a third place). That is how the two quota bugs in this
codebase happened: separate rate limiters summing past a shared model's RPM, and
per-bucket daily allowances summing to 960 against a 500 RPD cap. Both were
arithmetic errors over numbers nobody could see side by side. Keeping every
limit in one table makes those errors visible and testable.

THE SHAPE OF THE FREE TIER
--------------------------
The numbers below are the free-tier quotas and they are lopsided in a way that
dictates the whole routing strategy:

    model                    RPM     TPM      RPD
    gemini-3.8-flash           5    250K       20     (assumed — see below)
    gemini-3.6-flash           5    250K       20
    gemini-3.5-flash           5    250K       20
    gemini-3-flash             5    250K       20
    gemini-2.5-flash           5    250K       20
    gemini-3.5-flash-lite     15    250K      500     <- volume
    gemini-3.1-flash-lite     15    250K      500     <- volume
    gemini-2.5-flash-lite     10    250K       20

Only the Lite tier has volume, and only two models in it. Every reasoning-grade
model is capped at 20 requests/day. A pipeline that spends ~4 agent calls and 1
synthesis call per query therefore cannot put agent traffic on a reasoning model
at all — five queries would exhaust it. This is why routing is split into two
ladders:

  * AGENT_LADDER   — high volume, low reasoning demand (query classification,
                     rewriting, quality assessment, validation). Must sit on the
                     500 RPD lite model.
  * SYNTHESIS_LADDER — one call per query, and the only place where reasoning
                     quality shows up in the output. Gets the good models, and
                     spills to lite when their 20 RPD each is spent.

With a single key that is ~80 premium syntheses/day (4 reasoning models x 20)
before quality degrades, and ~500 agent calls. Adding keys multiplies both,
which is what BudgetTracker's key rotation is for.

TPM IS METERED ON INPUT TOKENS
------------------------------
Tokens-per-minute is an INPUT-token limit, and it is metered per (key, model)
slot at the call boundary: the wrapper reserves an estimate before sending a
request and replaces it with the provider's own prompt-token count afterwards
(BudgetTracker.reserve_input_tokens). A request that would not fit is treated
exactly like a per-minute 429 that was never sent — the slot rests, the request
rotates to the next key.

RPM still binds first at ordinary traffic (15 requests/min against 250K TPM
needs ~16K input tokens per request, and most prompts run ~8K). The case TPM
exists for is the verification judge under concurrency: it carries up to 9,000
characters per document, and fifteen of those in a minute on one key is over
the ceiling.

QUOTAS ARE PER PROJECT, AND THE DAY IS PACIFIC
----------------------------------------------
Two facts from the provider's rate-limit documentation that the accounting
depends on (https://ai.google.dev/gemini-api/docs/rate-limits, read 2026-11-14):

  * "Rate limits are applied per project, not per API key." Key rotation
    therefore multiplies capacity only when every key in GEMINI_API_KEYS comes
    from a DIFFERENT Google Cloud project. Two keys from one project share one
    quota, and the tracker — which counts per key — will believe it has twice
    what the provider will serve. BudgetTracker warns when it sees the
    signature of that (see _warn_if_keys_share_a_project).
  * "Requests per day (RPD) quotas reset at midnight Pacific time." The daily
    counters reset on that calendar (quota_day below), not on UTC.

The same page labels TPM "Tokens per minute (input)": it is an input-token
limit. It does not publish per-model free-tier numbers — those are shown only
in the AI Studio console, which is why the table above is a transcription.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class ModelLimits:
    """Provider-enforced quotas for a single upstream model."""

    rpm: int          # requests per minute
    tpm: int          # input tokens per minute
    rpd: int          # requests per day
    reasoning: bool   # True for full Flash models, False for Lite variants


# Provider free-tier quotas, exactly as reported by the Google AI Studio console.
# Change these ONLY to match the provider — never to make an allocation fit.
MODEL_LIMITS: dict[str, ModelLimits] = {
    # Verified against the Google AI Studio rate-limit console AND against a live
    # generateContent call per model. Both checks are necessary: the console
    # lists models this key cannot actually call, and ListModels advertises them
    # too — `gemini-2.5-flash` and `gemini-2.5-flash-lite` appear in both and
    # return 404 on every request. Availability is what the endpoint does, not
    # what a catalogue says.
    #
    # Reasoning tier — 5 RPM / 250K TPM / 20 RPD each, newest first.
    #
    # gemini-3.8-flash: added 2026-11-15. Advertised by ListModels on all four
    # live keys (version "3.0", 1,048,576 input / 65,536 output tokens,
    # thinking), listed "Stable" on the provider's models page (released
    # 2026-09-02, no shutdown date), and answered a streamed generateContent
    # through LiteLLM 1.89.3 with this project's synthesis settings — slowly:
    # 138.7s to the first token that day, the same backend gemini-3.7-flash was
    # hanging on (see SERVED_BY). Its limits are ASSUMED to be the reasoning
    # tier's: per-model free-tier numbers are shown only in the AI Studio
    # console and were not re-read. If they differ, the first 429 corrects the
    # count for that slot.
    "gemini/gemini-3.8-flash":        ModelLimits(rpm=5,  tpm=250_000, rpd=20,  reasoning=True),
    "gemini/gemini-3.7-flash":        ModelLimits(rpm=5,  tpm=250_000, rpd=20,  reasoning=True),
    "gemini/gemini-3.6-flash":        ModelLimits(rpm=5,  tpm=250_000, rpd=20,  reasoning=True),
    "gemini/gemini-3.5-flash":        ModelLimits(rpm=5,  tpm=250_000, rpd=20,  reasoning=True),
    # "Gemini 3 Flash" in the console; the servable id carries the -preview
    # suffix. Calling it `gemini-3-flash` returns 404, which was previously
    # misread as the model not existing at all.
    "gemini/gemini-3-flash-preview":  ModelLimits(rpm=5,  tpm=250_000, rpd=20,  reasoning=True),

    # Volume tier — 15 RPM / 250K TPM / 500 RPD. The only models that can carry
    # sustained agent traffic.
    "gemini/gemini-3.5-flash-lite":   ModelLimits(rpm=15, tpm=250_000, rpd=500, reasoning=False),
    "gemini/gemini-3.1-flash-lite":   ModelLimits(rpm=15, tpm=250_000, rpd=500, reasoning=False),

    # Grandfathered tier — available on SOME keys only.
    #
    # Both are closed to new sign-ups: on two of five configured keys they
    # return 404 "This model is no longer available to new users", and on the
    # other two they answer normally. Availability here is a property of the
    # (key, model) pair, which is why the router retires a *slot* rather than
    # the model when it sees that 404 — see mark_slot_unavailable(). Keeping
    # them costs one wasted call per unavailable slot per process, and buys
    # real capacity on the keys that still have access.
    "gemini/gemini-2.5-flash":        ModelLimits(rpm=5,  tpm=250_000, rpd=20,  reasoning=True),
    "gemini/gemini-2.5-flash-lite":   ModelLimits(rpm=10, tpm=250_000, rpd=20,  reasoning=False),

    # NOT LISTED, deliberately:
    #   gemini-2-flash, gemini-2-flash-lite, gemini-2.5-pro, gemini-3.1-pro —
    #     console shows 0/0/0, i.e. no free-tier access at all.
    #
    # Local fallback. Not provider-metered; the numbers just mean "effectively
    # unlimited" so the same accounting code path works without special-casing.
    "ollama/qwen2.5:14b":             ModelLimits(rpm=1000, tpm=10**9, rpd=10**6, reasoning=True),
}

LOCAL_MODEL = "ollama/qwen2.5:14b"

# Fraction of each provider quota we allow ourselves to spend. The remainder
# absorbs clock skew between our day boundary and the provider's, and in-flight
# requests that were counted upstream but not yet locally.
SAFETY_MARGIN = 0.95

# Synthesis: one call per query, and the only agent whose reasoning quality is
# visible in the answer. Ordered best-first; each rung is worth ~20 requests/day
# per key, then it falls through. The lite model is last so synthesis degrades
# gracefully rather than failing once the reasoning models are spent.
SYNTHESIS_LADDER: list[str] = [
    # 3.8 above 3.7: the provider serves gemini-3.7-flash WITH gemini-3.8-flash
    # (SERVED_BY), so this rung was already the top rung's model; naming it
    # directly asks for what is served. 3.7-flash stays below it because
    # whether the two ids share one daily quota is not published — if they do
    # not, it is twenty more requests a day per key of the same model; if they
    # do, its first 429 marks each slot spent and costs one refused request.
    "gemini/gemini-3.8-flash",
    "gemini/gemini-3.7-flash",
    "gemini/gemini-3.6-flash",
    "gemini/gemini-3.5-flash",
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-2.5-flash",       # until 2026-12-01 — see RETIREMENTS
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-3.1-flash-lite",
    LOCAL_MODEL,
]

# Agents: ~4 calls per query doing classification, rewriting, assessment and
# validation — structured tasks that do not need frontier reasoning. Volume is
# the binding constraint, so this ladder is ordered by RPD, not by capability.
# Reasoning models are deliberately absent: at 20 RPD they would be drained by
# five queries and then be unavailable to synthesis, which actually needs them.
AGENT_LADDER: list[str] = [
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-3.1-flash-lite",
    "gemini/gemini-2.5-flash-lite",
    LOCAL_MODEL,
]


# ─── Ids the provider serves with another model ──────────────────────────────
#
# Provider deprecations page (https://aistudio.google.com/docs/deprecations,
# read 2026-11-15): gemini-3.7-flash "is no longer available in Google AI
# Studio; the endpoint is not shutting down and automatically routes API
# requests to gemini-3.8-flash", and gemini-3.5-flash likewise "routes API
# requests to gemini-3.6-flash".
#
# What that changes is HEALTH, not quota: two ids on one backend fail
# together, so an outage seen on either must cool both. The live run of
# 2026-11-15 shows exactly that — in all four incidents gemini-3.6-flash and
# gemini-3.5-flash returned 503 within 3-8s of each other, and the ladder paid
# a second refused request (3-8s) to learn it twice. Quota stays per id:
# whether the provider counts a routed request against the id asked for or the
# model that served it is not published.
#
# Not verified end to end: a raw generateContent to gemini-3.7-flash (to read
# the response's modelVersion) hung for 150s without answering on the day.
SERVED_BY: dict[str, str] = {
    "gemini/gemini-3.7-flash": "gemini/gemini-3.8-flash",
    "gemini/gemini-3.5-flash": "gemini/gemini-3.6-flash",
}


def serving_model(model: str) -> str:
    """The model the provider actually serves for `model`."""
    return SERVED_BY.get(model, model)


def same_backend(model: str) -> tuple[str, ...]:
    """
    Every known id served by the same model as `model`, `model` included.

    Args:
        model: Upstream model id.

    Returns:
        The ids whose health is one fact, in registry order.
    """
    backend = serving_model(model)
    ids = [m for m in MODEL_LIMITS if serving_model(m) == backend]
    return tuple(ids) if model in ids else (model, *ids)


# ─── Known retirements ───────────────────────────────────────────────────────
#
# A dated entry, not a calendar reminder: from its date the selector stops
# offering the id (BudgetTracker._select_from_ladder), and
# tests/test_model_ladder_dates.py fails until it is taken off the ladders.


@dataclass(frozen=True)
class Retirement:
    """When an id stops serving the model it names, and what replaces it."""

    on: date               # first day the id no longer serves as itself
    becomes: str | None    # what the provider serves instead; None = shut down
    source: str


RETIREMENTS: dict[str, Retirement] = {
    # "Effective December 1, 2026, all API requests to gemini-2.5-flash will
    # automatically route to gemini-3.5-flash-lite" (models page and
    # deprecations page, read 2026-11-15), "billed at Gemini 3.5 Flash-Lite
    # rates". From then on the synthesis rung would silently be a Lite model
    # under a 20-RPD reasoning cap, ranked above the real Lite rungs.
    "gemini/gemini-2.5-flash": Retirement(
        on=date(2026, 12, 1), becomes="gemini/gemini-3.5-flash-lite",
        source="https://aistudio.google.com/docs/deprecations",
    ),
    # Shutdown listed for May 7, 2027 — "the earliest possible date", with
    # gemini-3.5-flash-lite as the replacement.
    "gemini/gemini-3.1-flash-lite": Retirement(
        on=date(2027, 5, 7), becomes=None,
        source="https://aistudio.google.com/docs/deprecations",
    ),
}


def is_retired(model: str, now: datetime | None = None) -> bool:
    """
    True once `model` has reached its retirement date.

    Judged on the UTC calendar, which reaches a date before Pacific time does:
    the provider names a day but not a timezone, and retiring a rung a few
    hours early costs less than serving a substituted model under its old name.

    Args:
        model: Upstream model id.
        now: An aware datetime; defaults to the current time.

    Returns:
        True when the id should no longer be offered.
    """
    retirement = RETIREMENTS.get(model)
    if retirement is None:
        return False
    moment = now if now is not None else datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).date() >= retirement.on


# Where the provider's daily quota turns over.
QUOTA_RESET_TIMEZONE = "America/Los_Angeles"

# Used only when the IANA database is unavailable (a slim container without
# tzdata). Pacific STANDARD time, deliberately: during daylight saving the real
# reset is an hour earlier than this says, so the fallback errs towards
# treating a fresh slot as spent for one hour — never towards offering a slot
# the provider has not reset yet.
_QUOTA_RESET_FALLBACK = timezone(timedelta(hours=-8))


def _quota_timezone():
    """The provider's reset timezone, or the fixed-offset fallback."""
    try:
        return ZoneInfo(QUOTA_RESET_TIMEZONE)
    except ZoneInfoNotFoundError:
        return _QUOTA_RESET_FALLBACK


def quota_day(now: datetime | None = None) -> date:
    """
    The provider's quota day for a moment in time.

    Args:
        now: An aware datetime; defaults to the current time.

    Returns:
        The calendar date in the provider's reset timezone. Daily counters
        belong to this date and reset when it changes.
    """
    moment = now if now is not None else datetime.now(timezone.utc)
    return moment.astimezone(_quota_timezone()).date()


def usable_rpd(model: str) -> int:
    """Daily request allowance after the safety margin."""
    limits = MODEL_LIMITS.get(model)
    if limits is None:
        return 0
    return max(1, int(limits.rpd * SAFETY_MARGIN))


def rpm_for(model: str) -> int:
    """Requests-per-minute ceiling for a model."""
    limits = MODEL_LIMITS.get(model)
    return limits.rpm if limits else 5


def usable_tpm(model: str) -> int | None:
    """
    Input tokens per minute we allow ourselves on a model, after the safety
    margin — or None when the model is not metered (local, or not in the table).
    """
    limits = MODEL_LIMITS.get(model)
    if limits is None or is_local(model):
        return None
    return max(1, int(limits.tpm * SAFETY_MARGIN))


def is_local(model: str) -> bool:
    """True for models that cost nothing and bypass key rotation."""
    return model.startswith("ollama/")
