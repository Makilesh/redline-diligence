"""
Budget tracker — Postgres-backed daily API call tracking with per-model RPM rate limiting.

The BudgetTracker must be:
- Persistent across restarts — stored in Postgres, not in-memory
- Per (API key, model) — quotas are enforced per key and per model, so
  accounting at any coarser granularity mis-counts
- Thread-safe — uses DB transactions for increment
- Resetting on the provider's quota day (midnight Pacific) — checked on every call
- Exposed in the UI — Streamlit sidebar shows live budget status

INITIALIZATION:
    BudgetTracker uses asyncpg.create_pool() which is async — it CANNOT
    be called in __init__. Use the async factory classmethod:
        tracker = await BudgetTracker.get_instance(postgres_url)

RATE LIMITERS:
    asyncio.Lock() requires a running event loop. Class-level attributes
    are evaluated at import time — BEFORE any event loop is running.
    Rate limiters are created lazily on first use via _get_rate_limiter().
"""

import asyncio
import contextvars
import os
import time
from dataclasses import dataclass
from datetime import date

import asyncpg

from src.llm.model_registry import (
    AGENT_LADDER,
    LOCAL_MODEL,
    SYNTHESIS_LADDER,
    is_local,
    quota_day,
    rpm_for,
    usable_rpd,
)
from src.llm.rate_limiter import RateLimiter
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# How long to skip a model after the provider reports it unavailable, or after
# it burns its whole timeout without answering.
#
# The cooldown DOUBLES on each consecutive failure, because a fixed one is wrong
# in both directions. A brief 503 spike clears in a couple of minutes, so a long
# fixed cooldown would strand a healthy model. But when gemini-3.7-flash started
# timing out persistently, a fixed 90s meant every query re-tried it, paid the
# full 120s timeout, and only then descended — a 41-question evaluation was on
# course to take 95 minutes instead of 30, and spent a scarce 20-RPD slot each
# time to learn nothing new.
#
# Doubling makes repeated failure cheap: 90s, 180s, 360s, and so on to the cap.
# One success resets it, so a model that recovers is back in rotation at once.
SERVICE_COOLDOWN_SECONDS = 90.0
MAX_SERVICE_COOLDOWN_SECONDS = 1800.0

# How long a pinned synthesis run will wait for its model to come back before
# giving up. Sized against the cooldown ladder: it equals the longest single
# cooldown (MAX_SERVICE_COOLDOWN_SECONDS), so a model at the cap is waited out
# once before the run is abandoned.
PIN_MAX_WAIT_SECONDS = float(os.getenv("SYNTHESIS_PIN_MAX_WAIT_SECONDS", "1800"))

# How long a (key, model) slot sits out after a per-MINUTE 429 when the provider
# did not say how long to wait. Per-minute limits are enforced over a rolling
# minute, so a full minute is the longest the refusal can remain true.
RATE_LIMIT_COOLDOWN_SECONDS = 60.0
# Bounds on a provider-supplied retry delay, so a malformed or hostile value can
# neither spin the selector nor park a slot for hours.
MIN_RATE_LIMIT_COOLDOWN_SECONDS = 1.0
MAX_RATE_LIMIT_COOLDOWN_SECONDS = 300.0

# Models the current request has asked not to be offered again, as
# {model: monotonic time of the skip}. Held in a ContextVar so it belongs to one
# request: LangGraph runs each node in a task whose context is a copy of the
# request's, and the orchestrator opens a fresh scope per query.
_request_skips: contextvars.ContextVar[dict[str, float] | None] = contextvars.ContextVar(
    "llm_request_skipped_models", default=None
)

# Longest a request-scoped skip is honoured. A request is over well inside this
# (the API's own budget is 300s); the bound exists so that a skip recorded by a
# long-lived task that never opens a new scope — a script looping over queries —
# cannot quietly steer every later query away from a model.
REQUEST_SKIP_TTL_SECONDS = 300.0


def begin_request_scope() -> None:
    """Starts a query with no request-scoped model skips. Call once per query."""
    _request_skips.set({})


def _models_skipped_for_request() -> frozenset[str]:
    """Models the current request has skipped (see skip_model_for_request)."""
    skips = _request_skips.get()
    if not skips:
        return frozenset()
    horizon = time.monotonic() - REQUEST_SKIP_TTL_SECONDS
    return frozenset(m for m, at in skips.items() if at >= horizon)


# A note about which quota a 429 tripped is only trusted for this long. It is
# written by the call wrapper at the moment of failure and read by
# mark_slot_exhausted a few statements later; anything older belongs to some
# other request.
QUOTA_HINT_TTL_SECONDS = 30.0


def _quota_today() -> date:
    """
    The provider's current quota day — the unit the daily counters reset on.

    This is a date in the provider's reset timezone (midnight Pacific for
    Gemini; see model_registry.quota_day), not UTC. It was UTC, which opened a
    seven-to-eight hour window every day in which the two calendars disagreed:
    from 00:00 UTC the counters read zero while the provider's were still
    spent, every slot the selector then offered answered 429, and each 429
    stamped the slot "spent" for the NEW UTC day — so when the provider did
    reset, the tracker kept the slot closed for the remaining sixteen hours.

    Returns a `date`, not a string, because asyncpg binds parameters by the
    type it infers from the statement: `reset_date = $1::date` infers `date`
    and rejects a str with the opaque "'str' object has no attribute
    'toordinal'". The `::date` cast does not rescue it — the cast applies
    server-side, after asyncpg has already had to encode the value.

    This was invisible for a long time. The api_budget table in the running
    database predated the DATE column in the schema and still held reset_date
    as text, so string binding worked and every quota query passed. The bug
    only appeared on a database created fresh from the current schema — which
    means it was never a dev-only quirk: it broke every first deploy, and
    every query with it, while the developer's own machine stayed green.
    """
    return quota_day()


def _quota_today_iso() -> str:
    """
    The quota day as an ISO string, for comparison against ModelBudget.reset_date.

    ModelBudget carries reset_date as a string, so the in-Python comparisons
    (and the mock backend) need this form. Anything crossing into SQL must use
    `_quota_today()` instead.
    """
    return _quota_today().isoformat()


@dataclass(frozen=True)
class ModelChoice:
    """
    A resolved routing decision: which model, billed to which API key.

    Model selection returns this rather than a bare model string because with
    key rotation the key is no longer implicit — the same model may be served
    by different keys on consecutive calls, and the caller has to pass the
    matching credential or the quota accounting is a fiction.
    """

    model: str
    api_key: str | None  # None for the local model, which needs no credential
    key_index: int       # -1 for local; otherwise position in the key list


@dataclass
class ModelBudget:
    """Current budget state for a single model."""

    model_key: str
    daily_limit: int
    used_today: int
    reset_date: str  # ISO date string, provider quota day


class BudgetTracker:
    """
    Singleton that persists daily API call counts to Postgres.
    Call get_model_for_agent() before every LiteLLM call to get
    the appropriate model string given current budget state.

    Schema (auto-created on startup):
        CREATE TABLE IF NOT EXISTS api_budget (
            model_key TEXT PRIMARY KEY,
            used_today INTEGER DEFAULT 0,
            reset_date DATE DEFAULT CURRENT_DATE
        );
    """

    _instance: "BudgetTracker | None" = None

    # Quota accounting is per (API key, model) — a "slot".
    #
    # Provider quotas attach to the key, so N keys multiply capacity, and they
    # attach per model, so spend on one model does not reduce another. Anything
    # coarser mis-counts: the previous per-bucket accounting allowed 960
    # requests/day against a single model's 500 RPD cap because two buckets each
    # believed they owned a separate 480.
    #
    # Limits come from model_registry.MODEL_LIMITS, never from literals here —
    # one table, so allocation errors are visible side by side rather than
    # scattered across three constants.
    SLOT_SEPARATOR = "::"

    # Shortest plausible provider API key. Purely a fragment filter — it does
    # not validate a key, it rejects things that cannot possibly be one.
    MIN_API_KEY_LENGTH = 30

    @classmethod
    def _load_api_keys(cls) -> list[str]:
        """
        Collects Gemini API keys from the environment, in priority order.

        Accepts either form, because both are common:
            GEMINI_API_KEYS=key1,key2,key3
            GEMINI_API_KEY_1=key1  GEMINI_API_KEY_2=key2   (numbered, any count)
            GEMINI_API_KEY=key1                             (single, legacy)

        Order is meaningful: the selector drains key 1 on a given model before
        touching key 2, so the first key should be the one you most want spent.

        Each key must come from a DIFFERENT Google Cloud project. The provider
        applies rate limits per project, not per key, so two keys from one
        project share a single quota while this tracker counts them as two.

        Returns:
            De-duplicated list of keys, preserving order. Empty if none set,
            which makes the pipeline run entirely on the local model.
        """
        keys: list[str] = []

        multi = os.getenv("GEMINI_API_KEYS", "")
        keys.extend(k.strip() for k in multi.split(",") if k.strip())

        i = 1
        while True:
            numbered = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
            if not numbered:
                break
            keys.append(numbered)
            i += 1

        single = os.getenv("GEMINI_API_KEY", "").strip()
        if single:
            keys.append(single)

        seen: set[str] = set()
        ordered = [k for k in keys if not (k in seen or seen.add(k))]

        # Drop fragments loudly rather than routing to them.
        #
        # Comma-splitting is only safe if no key contains a comma. A malformed or
        # truncated paste produces a short fragment that looks like a key to the
        # selector, which then allocates it quota, rotates onto it, and burns a
        # request discovering it is invalid — on every rung of the ladder. A
        # length floor cannot validate a key, but it catches the fragment case
        # that actually occurs, and warning beats silently ignoring input the
        # user believed they had configured.
        valid, rejected = [], []
        for k in ordered:
            (valid if len(k) >= cls.MIN_API_KEY_LENGTH else rejected).append(k)

        for k in rejected:
            logger.warning(
                "Ignoring malformed API key entry — too short to be a valid key. "
                "Check for a stray comma or a truncated paste in GEMINI_API_KEYS.",
                extra={"length": len(k), "starts_with": k[:6]},
            )

        return valid

    def _slot(self, key_index: int, model: str) -> str:
        """Composite accounting identifier for one (API key, model) pair."""
        return f"{key_index}{self.SLOT_SEPARATOR}{model}"

    def _slot_model(self, slot: str) -> str:
        """Extracts the model from a slot identifier."""
        _, _, model = slot.partition(self.SLOT_SEPARATOR)
        return model or slot

    def _slot_limit(self, slot: str) -> int:
        """Daily allowance for a slot, from the registry plus safety margin."""
        return usable_rpd(self._slot_model(slot))

    def _all_slots(self) -> list[str]:
        """Every (key, model) pair this process may spend against."""
        models = [m for m in dict.fromkeys(SYNTHESIS_LADDER + AGENT_LADDER)
                  if not is_local(m)]
        return [self._slot(i, m)
                for i in range(len(self._api_keys))
                for m in models]

    # DO NOT instantiate RateLimiter at class definition time.
    # asyncio.Lock() inside RateLimiter.__init__ will fail before event loop starts.
    _rate_limiters: dict[str, RateLimiter] = {}  # populated lazily on first use
    _instance_lock: asyncio.Lock | None = None  # lazily created — see get_instance()

    @classmethod
    async def get_instance(cls, postgres_url: str | None = None) -> "BudgetTracker":
        """
        Async factory — the ONLY way to create a BudgetTracker.
        asyncpg.create_pool() is async and cannot be called in __init__.

        Uses double-checked locking to prevent race condition where two
        concurrent startup requests both see _instance is None and create
        duplicate connection pools.

        Args:
            postgres_url: PostgreSQL connection string. Required on first call.

        Returns:
            Singleton BudgetTracker instance with initialized connection pool.

        Raises:
            ValueError: If postgres_url is None on first call.
            asyncpg.PostgresError: If database connection fails.
        """
        if cls._instance is not None:
            return cls._instance

        # Lazy lock creation — safe because this is called inside a running event loop
        if cls._instance_lock is None:
            cls._instance_lock = asyncio.Lock()

        async with cls._instance_lock:
            if cls._instance is None:
                if postgres_url is None:
                    raise ValueError(
                        "postgres_url is required on first BudgetTracker initialization"
                    )
                logger.info("Initializing BudgetTracker singleton")
                instance = cls.__new__(cls)
                # Must be set before _ensure_schema(), which enumerates slots.
                instance._api_keys = cls._load_api_keys()
                instance._rate_limiters = {}
                logger.info(
                    "Gemini API keys loaded",
                    extra={"key_count": len(instance._api_keys)},
                )
                if not instance._api_keys:
                    logger.warning(
                        "No Gemini API key configured — every agent will run on "
                        "the local model. Set GEMINI_API_KEYS or GEMINI_API_KEY."
                    )
                try:
                    instance._pool = await asyncpg.create_pool(
                        postgres_url, min_size=2, max_size=5
                    )
                    instance._is_mock = False
                    await instance._ensure_schema()
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize Postgres pool: {e}. Falling back to In-Memory Budget Tracking."
                    )
                    instance._is_mock = True
                    instance._mock_budgets = {}
                cls._instance = instance
                logger.info("BudgetTracker initialized successfully")
        return cls._instance

    async def _ensure_schema(self) -> None:
        """
        Creates the api_budget table if it doesn't exist.
        Idempotent — safe to call on every startup.

        Raises:
            asyncpg.PostgresError: If schema creation fails.
        """
        if getattr(self, "_is_mock", False):
            return
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_budget (
                    model_key TEXT PRIMARY KEY,
                    used_today INTEGER DEFAULT 0,
                    reset_date DATE DEFAULT CURRENT_DATE
                )
            """)
            # Ensure rows exist for each model
            for model_key in self._all_slots():
                await conn.execute(
                    """
                    INSERT INTO api_budget (model_key, used_today, reset_date)
                    VALUES ($1, 0, $2::date)
                    ON CONFLICT (model_key) DO NOTHING
                """,
                    model_key,
                    _quota_today(),
                )
        logger.info("Budget schema verified")

    def _get_rate_limiter(self, slot: str) -> RateLimiter:
        """
        Lazily creates RateLimiter instances, one per (API key, model) slot.

        The granularity is the point. RPM is metered by the provider per key and
        per model, so a limiter any coarser under-counts and one any finer
        over-counts. An earlier version keyed limiters by budget bucket, and
        because two buckets resolved to the same model their separate 5 and 15
        RPM allowances permitted 20 requests/minute against a single 15 RPM
        quota. Keying by slot makes the client-side ceiling structurally match
        the provider's.

        Lazily constructed because asyncio.Lock() inside RateLimiter.__init__
        requires a running event loop, which does not exist at import time.

        Args:
            slot: "<key_index>::<model>" identifier.

        Returns:
            RateLimiter dedicated to that key/model pair.
        """
        if slot not in self._rate_limiters:
            self._rate_limiters[slot] = RateLimiter(
                rpm_limit=rpm_for(self._slot_model(slot))
            )
        return self._rate_limiters[slot]

    def mark_slot_unavailable(self, key_index: int, model: str) -> None:
        """
        Records that this credential may never use this model.

        Google grandfathers older API keys when a model closes to new users, so
        `gemini-2.5-flash` answers on two of five configured keys and returns
        404 "no longer available to new users" on the rest. Availability is
        therefore a property of the (key, model) pair.

        Retired for the process lifetime rather than the day: unlike a spent
        quota this does not come back at midnight. Not persisted, because it is
        a fact about credentials that a redeploy re-learns in one call.

        Args:
            key_index: Position of the API key.
            model: Upstream model this key cannot use.
        """
        if key_index < 0 or is_local(model):
            return
        if not hasattr(self, "_unavailable_slots"):
            self._unavailable_slots = set()
        self._unavailable_slots.add(self._slot(key_index, model))

    def _slot_is_unavailable(self, slot: str) -> bool:
        """True if this (key, model) pair has been found permanently unusable."""
        return slot in getattr(self, "_unavailable_slots", ())

    def key_index_of(self, api_key: str | None) -> int:
        """
        Position of a live API key, or -1 if it is unknown, retired or absent.

        Lets the call wrapper settle accounts for a request knowing only the
        credential it was sent with — the agents pass `model` and `api_key`
        down, not the ModelChoice they came from.

        Args:
            api_key: Credential a request was sent with.

        Returns:
            Index into the key list, or -1.
        """
        if not api_key:
            return -1
        try:
            return self._api_keys.index(api_key)
        except ValueError:
            return -1

    # ── Per-minute refusals ──────────────────────────────────────────────────
    #
    # A 429 does not always mean the day is spent. Gemini names the quota that
    # tripped, and a per-minute one (RPM drift between processes, or TPM, which
    # is not metered here at all) clears within the minute. Marking the slot
    # exhausted for that — the only response the tracker used to have — set
    # used_today to the limit: measured in the scenario harness, one 429 on a
    # volume slot recorded 475 units against a single request the provider had
    # not even served.

    def note_quota_refusal(
        self,
        key_index: int,
        model: str,
        per_minute: bool,
        retry_after: float | None = None,
    ) -> None:
        """
        Records which quota a 429 tripped, for the mark_slot_exhausted call that
        follows it.

        The split exists because the two facts arrive in different places: the
        call wrapper sees the exception (and so the quota id), while the ladder
        loops in the agents call mark_slot_exhausted(key_index, model) with no
        exception in hand. The note carries the detail across without changing
        that call's signature.

        Args:
            key_index: Position of the API key.
            model: Upstream model that refused.
            per_minute: True if the provider attributed the 429 to a per-minute
                limit.
            retry_after: Seconds the provider asked for, when it said.
        """
        if key_index < 0 or is_local(model):
            return
        if not hasattr(self, "_quota_hints"):
            self._quota_hints = {}
        self._quota_hints[self._slot(key_index, model)] = (
            per_minute, retry_after, time.monotonic()
        )

    def _take_quota_hint(self, slot: str) -> tuple[bool, float | None] | None:
        """Pops a fresh note left by note_quota_refusal, if there is one."""
        hints = getattr(self, "_quota_hints", None)
        if not hints:
            return None
        hint = hints.pop(slot, None)
        if hint is None:
            return None
        per_minute, retry_after, noted_at = hint
        if time.monotonic() - noted_at > QUOTA_HINT_TTL_SECONDS:
            return None
        return per_minute, retry_after

    def _cool_slot(self, slot: str, seconds: float | None) -> float:
        """Sidelines one (key, model) slot for a bounded time; returns the time."""
        wait = RATE_LIMIT_COOLDOWN_SECONDS if seconds is None else seconds
        wait = max(MIN_RATE_LIMIT_COOLDOWN_SECONDS, min(wait, MAX_RATE_LIMIT_COOLDOWN_SECONDS))
        if not hasattr(self, "_slot_cooldowns"):
            self._slot_cooldowns = {}
        self._slot_cooldowns[slot] = time.monotonic() + wait
        return wait

    def _slot_in_cooldown(self, slot: str) -> bool:
        """True while a slot is sitting out a per-minute refusal."""
        cooldowns = getattr(self, "_slot_cooldowns", None)
        if not cooldowns:
            return False
        until = cooldowns.get(slot)
        if until is None:
            return False
        if until <= time.monotonic():
            del cooldowns[slot]
            return False
        return True

    # ── Keeping the daily counter equal to what the provider served ──────────

    async def refund_slot(self, key_index: int, model: str) -> None:
        """
        Returns one daily unit for a request the provider did not serve.

        The unit is debited at selection, before the call, because that is the
        only point where the debit can be atomic with the capacity check. So a
        request the provider then refuses — 503, a rejected key, a revoked
        model, a per-minute 429 — has already been charged. On a 20-RPD
        reasoning slot that is 5% of the day per refusal, and a 503 spell
        produces them in runs.

        Never refunds a slot the provider itself has declared spent today: a
        concurrent request's refund must not reopen a rung the provider has
        closed.

        Args:
            key_index: Position of the API key, or -1 for local (ignored).
            model: Upstream model the request was sent to.
        """
        if key_index < 0 or is_local(model):
            return
        slot = self._slot(key_index, model)
        today = _quota_today()
        if getattr(self, "_provider_exhausted", {}).get(slot) == today:
            return

        if getattr(self, "_is_mock", False):
            b = self._mock_budgets.get(slot)
            if b and b["reset_date"] == today.isoformat() and b["used_today"] > 0:
                b["used_today"] -= 1
            return

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE api_budget SET used_today = used_today - 1 "
                    "WHERE model_key = $1 AND used_today > 0 AND reset_date = $2::date",
                    slot, today,
                )
        except Exception as e:
            # A lost refund over-counts by one; never worth failing the request.
            logger.error("Failed to refund slot", extra={"slot": slot, "error": str(e)})

    async def charge_extra_request(self, key_index: int, model: str) -> bool:
        """
        Accounts for one more provider request on a slot already selected.

        Selection pays for exactly one request. A same-slot retry (a JSON repair
        request, a prose retry after an empty completion) is a second request
        the provider counts against RPM and RPD, and it used to be free: up to
        three requests went out against one debited unit, so the local counter
        ran behind the provider's precisely when calls were failing.

        Never blocks. If the slot has no window or no daily allowance left the
        answer is False and the caller should move on rather than retry here.

        Args:
            key_index: Position of the API key, or -1 for local.
            model: Upstream model.

        Returns:
            True if the request is accounted for and may be sent.
        """
        if key_index < 0 or is_local(model):
            return True
        if key_index >= len(self._api_keys) or not self._api_keys[key_index]:
            return False
        slot = self._slot(key_index, model)
        if self._slot_is_unavailable(slot) or self._slot_in_cooldown(slot):
            return False
        limiter = self._get_rate_limiter(slot)
        if not await limiter.try_acquire():
            return False
        if await self._try_consume(slot):
            return True
        await limiter.release()
        return False

    def skip_model_for_request(self, model: str) -> None:
        """
        Keeps the CURRENT request off a model. Nothing else.

        This is what a caller says when a model's answer was no use to it — an
        uncited synthesis, a draft that failed a content check — or when it has
        already watched the model fail and wants the next selection to go
        elsewhere. It is a statement about one request, so it is stored in that
        request's context and no other request can see it.

        It used to write the process-wide cooldown below, despite the name.
        Measured: every synthesis model returned the same correct, cited,
        101-character answer, each was rejected as too short, and one query
        left all six models in an escalating 90-second cooldown — so the next
        user's question was answered by a lite model. A content rejection is
        not evidence about the provider, and provider health is now reported
        only by the code that observes the provider (report_model_unavailable).

        Ignored outside a running task: with no request there is nothing to
        scope the skip to, and writing it into a long-lived context would turn
        it back into the global side effect this method exists to avoid.

        Args:
            model: Upstream model this request should not be offered again.
        """
        if is_local(model):
            return
        try:
            asyncio.current_task()
        except RuntimeError:
            return
        skips = _request_skips.get()
        if skips is None:
            skips = {}
            _request_skips.set(skips)
        skips[model] = time.monotonic()

    def report_model_unavailable(self, model: str) -> None:
        """
        Marks a model as provider-side unavailable for a short cooldown.

        Distinct from `mark_slot_exhausted`, and the distinction matters. A 429
        means this key is spent on this model until the daily reset, so the slot
        is debited to its limit. A 503 — "This model is currently experiencing
        high demand" — means the model is down for every key, and typically
        clears in minutes. Debiting quota for that would retire real capacity
        over a blip.

        The cooldown is in-memory and per-process by design: it is a hint about
        the last few seconds of provider behaviour, not durable accounting.

        Called by the call wrapper when it sees the provider fail (503, timeout,
        5xx, a dropped stream) — never for a response a caller merely disliked.

        Args:
            model: Upstream model the provider reported as unavailable.
        """
        if is_local(model):
            return
        if not hasattr(self, "_model_cooldowns"):
            self._model_cooldowns = {}
        if not hasattr(self, "_model_failures"):
            self._model_failures = {}

        failures = self._model_failures.get(model, 0) + 1
        self._model_failures[model] = failures
        backoff = min(
            SERVICE_COOLDOWN_SECONDS * (2 ** (failures - 1)),
            MAX_SERVICE_COOLDOWN_SECONDS,
        )
        self._model_cooldowns[model] = time.monotonic() + backoff
        logger.info(
            "Model cooling down after consecutive failures",
            extra={"model": model, "consecutive_failures": failures,
                   "cooldown_seconds": backoff},
        )

    def note_model_healthy(self, model: str) -> None:
        """
        Clears the failure streak after a model answers successfully.

        Without this the backoff would only ever grow, so a model that recovered
        after a bad spell would stay side-lined for up to half an hour. One good
        answer is sufficient evidence that the outage is over.

        Args:
            model: Upstream model that just returned a usable response.
        """
        failures = getattr(self, "_model_failures", None)
        if failures:
            failures.pop(model, None)
        cooldowns = getattr(self, "_model_cooldowns", None)
        if cooldowns:
            cooldowns.pop(model, None)

    def _models_in_cooldown(self) -> set[str]:
        """Models currently skipped after a provider-side unavailability."""
        cooldowns = getattr(self, "_model_cooldowns", None)
        if not cooldowns:
            return set()
        now = time.monotonic()
        expired = [m for m, until in cooldowns.items() if until <= now]
        for m in expired:
            del cooldowns[m]
        return set(cooldowns)

    async def _select_from_ladder(
        self,
        ladder: list[str],
        purpose: str,
        exclude: frozenset[str] | tuple[str, ...] = (),
    ) -> ModelChoice:
        """
        Walks a model ladder and the configured keys to find capacity.

        Order is deliberate: for each model, every key is tried before dropping
        to the next model down. Keys multiply capacity at a given quality level,
        so exhausting all of them on gemini-3.6-flash before falling back to
        gemini-3.5-flash is what "keep the best for the best" actually means. The
        opposite order — cycling models within a key — would degrade answer
        quality while premium quota sat unused on another key.

        Two passes:
          1. Non-blocking. Take the first slot with both daily budget and a free
             RPM window. This is where key rotation pays off — a saturated key is
             skipped instantly rather than slept on.
          2. Blocking. If every slot is momentarily rate-limited but some still
             have daily budget, wait on the best of them. Waiting a few seconds
             for the good model beats permanently degrading to the local one.

        Falls back to the local model only when no cloud slot has daily budget
        left, which is the genuine end of the quota.

        Args:
            ladder: Models in preference order.
            purpose: "synthesis" or "agent", for logging.
            exclude: Models to pass over for THIS request only — rungs that
                already returned unusable output to the caller. Unlike a
                cooldown this says nothing about the model's health, so it
                must not affect any other request.

        Returns:
            ModelChoice naming the model and the key to bill it to.
        """
        cooled = self._models_in_cooldown()
        # A request-scoped skip means "offer me a different rung". On a ladder
        # of one — a pinned run — there is no different rung, and honouring the
        # skip would silently substitute the local model for the pin.
        if len(ladder) > 1:
            exclude = frozenset(exclude) | _models_skipped_for_request()

        # Pass 1 — immediate capacity anywhere.
        for model in ladder:
            if is_local(model):
                break
            if model in cooled or model in exclude:
                continue
            for key_index, api_key in enumerate(self._api_keys):
                if not api_key:  # retired by mark_key_unusable
                    continue
                slot = self._slot(key_index, model)
                if self._slot_is_unavailable(slot) or self._slot_in_cooldown(slot):
                    continue
                if not await self._budget_available(slot):
                    continue
                limiter = self._get_rate_limiter(slot)
                if not await limiter.try_acquire():
                    continue
                if await self._try_consume(slot):
                    return ModelChoice(model=model, api_key=api_key, key_index=key_index)
                # Lost the race for the slot's last daily unit: the request is
                # going elsewhere, so the window entry taken for it goes back.
                await limiter.release()

        # Pass 2 — everything is rate-limited; wait rather than degrade quality.
        # Cooldowns are ignored here: if the only remaining capacity is on a
        # model the provider just 503'd, trying it beats degrading to local.
        #
        # A slot the PROVIDER rate-limited is different from one our own window
        # says is full: its wait is not ours to measure, so it is passed over
        # here rather than slept on.
        for model in ladder:
            if is_local(model):
                break
            if model in exclude:
                continue
            for key_index, api_key in enumerate(self._api_keys):
                if not api_key:  # retired by mark_key_unusable
                    continue
                slot = self._slot(key_index, model)
                if self._slot_is_unavailable(slot) or self._slot_in_cooldown(slot):
                    continue
                if not await self._budget_available(slot):
                    continue
                logger.info(
                    "All slots rate-limited; waiting for capacity",
                    extra={"purpose": purpose, "model": model, "key_index": key_index},
                )
                limiter = self._get_rate_limiter(slot)
                await limiter.acquire()
                if await self._try_consume(slot):
                    return ModelChoice(model=model, api_key=api_key, key_index=key_index)
                await limiter.release()

        logger.warning(
            "All cloud quota exhausted, falling back to local model",
            extra={"purpose": purpose, "keys_configured": len(self._api_keys)},
        )
        return ModelChoice(model=LOCAL_MODEL, api_key=None, key_index=-1)

    async def get_model_for_synthesis(self) -> ModelChoice:
        """
        Picks the best available model for answer synthesis.

        Synthesis is one call per query and the only agent whose reasoning
        quality is visible in the output, so it gets the reasoning ladder —
        each rung worth ~20 requests/day per key — and only spills to a Lite
        model once those are spent.

        CRITICAL: the rate-limiter slot is taken BEFORE the daily budget is
        debited. If the task is cancelled while waiting, the budget is not
        charged for a call that was never sent.

        Returns:
            ModelChoice with the model and the API key to bill it to.

        Raises:
            RuntimeError: If SYNTHESIS_MODEL_PIN names a model with no capacity
                left. Pinned runs fail rather than substitute — see below.
        """
        pin = os.getenv("SYNTHESIS_MODEL_PIN", "").strip()
        if not pin:
            return await self._select_from_ladder(SYNTHESIS_LADDER, "synthesis")

        # Pinned: measurement mode, not a production path.
        #
        # Recall on the golden set is dominated by which model answered. Measured
        # inside a single run, on one index with identical retrieval, mean recall
        # was 100% on gemini-3.6-flash, 93% on gemini-3-flash-preview, 66% on
        # gemini-3.5-flash and 50% on gemini-2.5-flash — a 50-point spread driven
        # entirely by which rung had daily quota left at that moment.
        #
        # That makes the ladder fatal to any A/B comparison. Evaluating two
        # retrieval configurations on different days compares their synthesis
        # quota states far more than it compares their retrieval, and the ladder
        # guarantees the two runs differ because the first one spends the quota
        # the second would have used.
        #
        # So the pin restricts the ladder to exactly one model and, critically,
        # does NOT fall back. Falling back is what the unpinned ladder is for and
        # is right in production; here it would silently reintroduce the variable
        # the pin exists to remove, and the substitution is invisible in the
        # results — the run would simply report a number nobody could interpret.
        # A loud failure is recoverable; a quietly confounded measurement is not.
        if pin not in SYNTHESIS_LADDER:
            logger.warning(
                "SYNTHESIS_MODEL_PIN names a model absent from SYNTHESIS_LADDER",
                extra={"pin": pin, "ladder": SYNTHESIS_LADDER},
            )

        # Two reasons the pin can be unavailable, and they need opposite
        # responses. A 503 puts the model in an escalating, process-wide
        # cooldown (90s doubling to 1800s) — measured against this provider, the
        # reasoning rungs 503 several times an hour, so a pinned 41-question run
        # would almost certainly meet one. That is transient and worth waiting
        # out. Daily quota exhaustion is not: it clears at the daily reset, and no
        # amount of waiting inside one run will help.
        #
        # Failing on the first cooldown would make the pin unusable for exactly
        # the long measurement runs it exists to serve.
        import time

        deadline = time.monotonic() + PIN_MAX_WAIT_SECONDS

        while True:
            choice = await self._select_from_ladder([pin], "synthesis(pinned)")
            if choice.model == pin:
                return choice

            if not await self._model_has_daily_budget(pin):
                raise RuntimeError(
                    f"SYNTHESIS_MODEL_PIN={pin} has no daily quota left on any "
                    f"configured key, and a pinned run must not substitute another "
                    f"model. Wait for the daily quota reset (midnight Pacific), add "
                    f"another key, or unset "
                    f"the pin to restore normal ladder behaviour."
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"SYNTHESIS_MODEL_PIN={pin} still had daily quota but stayed "
                    f"provider-side unavailable for {PIN_MAX_WAIT_SECONDS}s. The "
                    f"model is having a bad spell; re-run the measurement later "
                    f"rather than comparing against a substituted model."
                )

            # Sleep until the cooldown lapses rather than polling: the deadline
            # is known exactly, so there is nothing to discover by waking early.
            cooldowns = getattr(self, "_model_cooldowns", {}) or {}
            wait = cooldowns.get(pin, 0) - time.monotonic()
            wait = max(1.0, min(wait, remaining))

            logger.info(
                "Pinned synthesis model is cooling down; waiting rather than "
                "substituting",
                extra={"model": pin, "wait_seconds": round(wait, 1)},
            )
            await asyncio.sleep(wait)

    async def _model_has_daily_budget(self, model: str) -> bool:
        """
        True if any configured key still has daily allowance for this model.

        Distinguishes "temporarily unavailable" from "spent for the day", which
        is the difference between a pinned run that should wait and one that
        should stop.

        Args:
            model: Upstream model identifier.

        Returns:
            True if at least one live key has budget left on this model.
        """
        for key_index, api_key in enumerate(self._api_keys):
            if not api_key:  # retired by mark_key_unusable
                continue
            slot = self._slot(key_index, model)
            if self._slot_is_unavailable(slot):
                continue
            if await self._budget_available(slot):
                return True
        return False

    async def get_model_for_agent(
        self, exclude: frozenset[str] | tuple[str, ...] = ()
    ) -> ModelChoice:
        """
        Picks a model for the high-volume agent calls.

        Agents spend ~4 calls per query on classification, rewriting, assessment
        and validation — structured work that does not need frontier reasoning.
        The ladder is ordered by daily capacity rather than capability, and
        deliberately excludes the reasoning models: at 20 RPD each they would be
        drained by five queries and then be unavailable to synthesis, which is
        where that quality actually matters.

        Args:
            exclude: Models to pass over for this request only (see
                _select_from_ladder).

        Returns:
            ModelChoice with the model and the API key to bill it to.
        """
        if exclude:
            return await self._select_from_ladder(AGENT_LADDER, "agent", exclude)
        return await self._select_from_ladder(AGENT_LADDER, "agent")

    async def mark_slot_exhausted(self, key_index: int, model: str) -> None:
        """
        Records that the provider has refused further calls on this (key, model).

        Local counters drift from the provider's for reasons that cannot be
        engineered away: the process restarts and the in-memory fallback resets
        to zero while the provider's daily counter keeps running, another process
        shares the key, or someone uses the key by hand. When that happens the
        tracker believes it has quota it does not have, and every call to that
        model returns 429.

        Observed cost of not having this: a golden-set run opened with 19
        consecutive synthesis failures — exactly gemini-3.6-flash's daily
        allowance — because the previous run had already spent it. The gate had
        passed the context with scores up to 0.945; the answers were lost purely
        to a stale counter, and the run only recovered once the local counter
        also hit its limit and the ladder moved on by itself.

        Burning the local counter makes the provider's 429 authoritative, so the
        next selection skips this rung immediately.

        That is the right response to a per-DAY refusal only. When the call
        wrapper has noted (note_quota_refusal) that this 429 was a per-minute
        limit, the slot is rested for the provider's retry delay instead and the
        day's allowance is left alone. With no note the refusal is treated as
        daily, as it always was: on the free tier the daily cap is the one that
        binds, and guessing "per-minute" would re-offer a spent slot each time.

        Args:
            key_index: Position of the API key, or -1 for local (ignored).
            model: Upstream model the provider refused.
        """
        if key_index < 0 or is_local(model):
            return

        slot = self._slot(key_index, model)
        limit = self._slot_limit(slot)
        today = _quota_today()

        hint = self._take_quota_hint(slot)
        if hint is not None and hint[0]:
            wait = self._cool_slot(slot, hint[1])
            logger.warning(
                "Provider rate-limited this slot per minute; resting it",
                extra={"model": model, "key_index": key_index,
                       "cooldown_seconds": round(wait, 1)},
            )
            return

        if not hasattr(self, "_provider_exhausted"):
            self._provider_exhausted = {}
        await self._warn_if_keys_share_a_project(slot, model, today)
        self._provider_exhausted[slot] = today

        logger.warning(
            "Provider refused further calls; marking slot exhausted for today",
            extra={"model": model, "key_index": key_index},
        )

        if getattr(self, "_is_mock", False):
            # The mock backend stores reset_date as a string, matching ModelBudget.
            self._mock_budgets[slot] = {
                "used_today": limit,
                "reset_date": today.isoformat(),
            }
            return

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE api_budget SET used_today = $1, reset_date = $2::date "
                    "WHERE model_key = $3",
                    limit, today, slot,
                )
        except Exception as e:
            # Never let bookkeeping failure break the caller's fallback path.
            logger.error("Failed to persist exhausted slot",
                         extra={"slot": slot, "error": str(e)})

    async def _warn_if_keys_share_a_project(self, slot: str, model: str, today: date) -> None:
        """
        Logs, once, when two keys look like they draw on the same quota.

        The provider's limits are per Google Cloud project, not per key, so keys
        created in one project share a single allowance — but this tracker
        counts per key and would believe it has N times that. Nothing in an API
        key says which project it belongs to, so the only evidence is
        behavioural: the provider declares a slot's day spent while our own
        count for it is well under the limit, and it has already done the same
        for another key on the same model today.

        That pattern has one other cause — counters lost in a restart on the
        in-memory fallback — and the message says so rather than guess.

        Args:
            slot: The slot the provider just refused for the day.
            model: Its model.
            today: Current quota day.
        """
        if getattr(self, "_shared_project_warned", False):
            return
        also_refused = [
            s for s, day in self._provider_exhausted.items()
            if day == today and s != slot and self._slot_model(s) == model
        ]
        if not also_refused:
            return
        try:
            budget = await self._load_budget(slot)
        except Exception:
            return
        used = budget.used_today if budget.reset_date == today.isoformat() else 0
        if used * 2 >= self._slot_limit(slot):
            return

        self._shared_project_warned = True
        logger.warning(
            "The provider declared a model's daily quota spent on two API keys "
            "while this process had counted well under the limit on the second. "
            "Gemini rate limits are per Google Cloud PROJECT, not per key: keys "
            "created in the same project share one quota and add no capacity. "
            "Check that every key in GEMINI_API_KEYS comes from a different "
            "project. (The other explanation is counters lost in a restart "
            "while running on the in-memory fallback.)",
            extra={"model": model, "slots": sorted(also_refused + [slot]),
                   "locally_counted": used, "limit": self._slot_limit(slot)},
        )

    def mark_key_unusable(self, key_index: int) -> None:
        """
        Retires a credential the provider rejected, for the life of the process.

        Unlike quota exhaustion, which is per (key, model) and clears at the next
        daily reset, a rejected credential is bad for every model until the
        configuration is fixed. Retiring the key stops every subsequent selection
        from routing to it.

        Measured cost of not doing this: a run with four configured keys lost 16
        of 35 questions to a single invalid key — one entry in GEMINI_API_KEYS had
        been split by a stray comma into two fragments — because each request that
        landed on it failed outright instead of rotating to a working key.

        The key is blanked rather than removed from the list so that key_index
        values already held by in-flight requests keep pointing at the same
        position.

        Args:
            key_index: Position of the rejected key. Negative values (local) are
                ignored.
        """
        if key_index < 0 or key_index >= len(self._api_keys):
            return
        if not self._api_keys[key_index]:
            return

        logger.error(
            "API key rejected by provider — retiring it for this process. "
            "Check GEMINI_API_KEYS for a truncated or comma-split entry.",
            extra={"key_index": key_index},
        )
        self._api_keys[key_index] = ""

    async def get_budget_status(self) -> dict:
        """
        Returns current budget status, aggregated per model across all keys.

        The UI cares about "how much gemini-3.6-flash is left", not about which
        key it sits on, so per-key slots are summed. Per-key detail stays in
        `keys`, so an exhausted individual key is still diagnosable.

        Returns:
            Dict of model -> {used, limit, remaining, reset_date, keys}.
        """
        status: dict[str, dict] = {}
        today = _quota_today_iso()

        for slot in self._all_slots():
            # A slot this credential may never use is not capacity. Before it is
            # discovered the panel overstates a grandfathered model — 95/95 for
            # gemini-2.5-flash when only two of five keys can reach it — but it
            # self-corrects on the first call, and overstating for one call beats
            # overstating for ever.
            if self._slot_is_unavailable(slot):
                continue

            model = self._slot_model(slot)
            budget = await self._load_budget(slot)
            limit = self._slot_limit(slot)

            # A slot whose stored reset_date is not today has a stale counter:
            # the daily reset is applied lazily by _try_consume, on that slot's
            # next call. Reporting the raw number means the sidebar shows
            # yesterday's spend as today's — after a quota reset it read
            # "26/95 left" while all 95 were in fact available. _budget_available
            # already treats a stale date as a full allowance; this makes the
            # display agree with the routing decision instead of contradicting it.
            used_today = budget.used_today if budget.reset_date == today else 0

            entry = status.setdefault(
                model,
                {"used": 0, "limit": 0, "remaining": 0,
                 "reset_date": today, "keys": []},
            )
            entry["used"] += used_today
            entry["limit"] += limit
            entry["remaining"] += max(0, limit - used_today)
            entry["keys"].append(
                {"key_index": int(slot.split(self.SLOT_SEPARATOR)[0]),
                 "used": used_today, "limit": limit}
            )

        return status

    async def _budget_available(self, model_key: str) -> bool:
        """
        Read-only check for UI display.
        Do NOT use for consumption decisions — use _try_consume() instead.

        Args:
            model_key: Model identifier.

        Returns:
            True if budget is available for this model today.
        """
        budget = await self._load_budget(model_key)
        today = _quota_today_iso()
        if budget.reset_date != today:
            return True
        return budget.used_today < self._slot_limit(model_key)

    async def _try_consume(self, model_key: str) -> bool:
        """
        Atomic check-and-increment: returns True and increments counter if budget
        remains, returns False without incrementing if exhausted.

        CRITICAL: This replaces the old _budget_available() + _increment() pattern
        which had a TOCTOU race: between checking and incrementing, a concurrent
        request could pass the same check, causing both to increment and overshoot
        the daily limit. The single conditional UPDATE is atomic at the DB level.

        Args:
            model_key: Model identifier.

        Returns:
            True if budget was consumed, False if exhausted.
        """
        today = _quota_today()
        today_iso = today.isoformat()
        if getattr(self, "_is_mock", False):
            if model_key not in self._mock_budgets:
                self._mock_budgets[model_key] = {"used_today": 0, "reset_date": today_iso}
            b = self._mock_budgets[model_key]
            if b["reset_date"] != today_iso:
                b["used_today"] = 0
                b["reset_date"] = today_iso
            if b["used_today"] < self._slot_limit(model_key):
                b["used_today"] += 1
                logger.info(
                    f"Budget consumed (local mock): {model_key} ({b['used_today']}/{self._slot_limit(model_key)})"
                )
                return True
            logger.warning(f"Budget exhausted (local mock) for model {model_key}")
            return False

        async with self._pool.acquire() as conn:
            # Reset if the stored day is not the current quota day (idempotent).
            # "<>" rather than "<": rows written when the day was counted in UTC
            # can carry a date AHEAD of the quota day, and _budget_available
            # already treats any other date as a full allowance — the two must
            # agree or the selector offers a slot the debit then refuses.
            await conn.execute(
                "UPDATE api_budget SET used_today = 0, reset_date = $1::date "
                "WHERE model_key = $2 AND reset_date <> $1::date",
                today,
                model_key,
            )
            # Atomic conditional increment — prevents TOCTOU race
            result = await conn.execute(
                "UPDATE api_budget SET used_today = used_today + 1 "
                "WHERE model_key = $1 AND used_today < $2",
                model_key,
                self._slot_limit(model_key),
            )
            # result is "UPDATE N" where N is rows affected
            consumed = result == "UPDATE 1"

            if consumed:
                logger.info(
                    "Budget consumed",
                    extra={"model_key": model_key},
                )
            else:
                logger.warning(
                    "Budget exhausted for model",
                    extra={"model_key": model_key},
                )

            return consumed

    async def _load_budget(self, model_key: str) -> ModelBudget:
        """
        Loads current budget state from Postgres.

        Args:
            model_key: Model identifier.

        Returns:
            ModelBudget dataclass with used_today and reset_date.
        """
        today = _quota_today_iso()
        if getattr(self, "_is_mock", False):
            if model_key not in self._mock_budgets:
                self._mock_budgets[model_key] = {"used_today": 0, "reset_date": today}
            b = self._mock_budgets[model_key]
            return ModelBudget(
                model_key=model_key,
                daily_limit=self._slot_limit(model_key),
                used_today=b["used_today"],
                reset_date=b["reset_date"],
            )

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT model_key, used_today, reset_date FROM api_budget WHERE model_key = $1",
                model_key,
            )
            if row is None:
                # First use — insert default
                await conn.execute(
                    "INSERT INTO api_budget (model_key, used_today, reset_date) "
                    "VALUES ($1, 0, $2::date)",
                    model_key,
                    _quota_today(),
                )
                return ModelBudget(
                    model_key=model_key,
                    daily_limit=self._slot_limit(model_key),
                    used_today=0,
                    reset_date=_quota_today_iso(),
                )
            return ModelBudget(
                model_key=row["model_key"],
                daily_limit=self._slot_limit(model_key),
                used_today=row["used_today"],
                reset_date=(
                    row["reset_date"].isoformat()
                    if hasattr(row["reset_date"], "isoformat")
                    else str(row["reset_date"])
                ),
            )

    @classmethod
    async def close(cls) -> None:
        """
        Closes the connection pool. Call on application shutdown.

        Raises:
            RuntimeError: If no instance exists.
        """
        if cls._instance is not None:
            if not getattr(cls._instance, "_is_mock", False) and hasattr(cls._instance, "_pool"):
                await cls._instance._pool.close()
                logger.info("BudgetTracker connection pool closed")
            cls._instance = None
