"""
In-process metrics: counters, gauges and histograms, rendered in the
Prometheus text format by `GET /metrics`.

Deliberately not `prometheus_client`. The whole need is a few dozen series from
one process, and a dependency whose wheel story has to be re-checked on aarch64
(the deployment target) is a bad trade for ~150 lines of arithmetic.

Two things are measured that nothing else in the system can see:

  * Per-node latency and LLM usage. The orchestrator wraps every graph node in
    `node_scope()`, and each completed LLM call lands in the scope that is
    active in the calling task. That is what turns "the query took 31 s" into
    "synthesis took 24 s over 2 calls".
  * Request outcomes, including the ones that never reach the pipeline (rate
    limited, engine busy) and the ones that leave it early (client disconnect,
    timeout).

LLM calls are observed through the wrapper's own structured log record rather
than a hook inside it: `src/llm/litellm_wrapper._log_usage` emits "LLM call
usage" synchronously, in the task that made the call, for every completed
call. A logging filter on that logger is therefore both complete and correctly
attributed, and needs no edit to the LLM layer. The coupling is to the message
name and its fields; tests/test_platform_observability.py pins both.
"""

import contextvars
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

_LOCK = threading.Lock()

# Seconds. Wide on purpose: a deterministic node finishes in milliseconds and a
# CPU rerank or a synthesis call that walks the model ladder takes minutes.
_DEFAULT_BUCKETS: tuple[float, ...] = (
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0,
)

LLM_USAGE_LOGGER = "src.llm.litellm_wrapper"
LLM_USAGE_MESSAGE = "LLM call usage"


def _label_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((labels or {}).items()))


def _format_labels(key: tuple[tuple[str, str], ...], extra: str = "") -> str:
    parts = [f'{k}="{_escape(v)}"' for k, v in key]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class Counter:
    """Monotonic counter, optionally labelled."""

    kind = "counter"

    def __init__(self, name: str, help_text: str) -> None:
        self.name, self.help = name, help_text
        self._values: dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = _label_key(labels)
        with _LOCK:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._values.get(_label_key(labels), 0.0)

    def reset(self) -> None:
        self._values.clear()

    def render(self) -> list[str]:
        return [f"{self.name}{_format_labels(k)} {v:g}" for k, v in sorted(self._values.items())]

    def snapshot(self) -> list[dict]:
        return [{"labels": dict(k), "value": v} for k, v in sorted(self._values.items())]


class Gauge(Counter):
    """A value that can go down. `set_function` reads it lazily at scrape time."""

    kind = "gauge"

    def __init__(self, name: str, help_text: str) -> None:
        super().__init__(name, help_text)
        self._function = None

    def set(self, value: float, **labels: str) -> None:
        with _LOCK:
            self._values[_label_key(labels)] = float(value)

    def set_function(self, function) -> None:
        self._function = function

    def _refresh(self) -> None:
        if self._function is not None:
            try:
                self.set(float(self._function()))
            except Exception:
                # A scrape must never fail because one gauge's source did.
                pass

    def render(self) -> list[str]:
        self._refresh()
        return super().render()

    def snapshot(self) -> list[dict]:
        self._refresh()
        return super().snapshot()


class Histogram:
    """Cumulative-bucket histogram, optionally labelled."""

    kind = "histogram"

    def __init__(
        self, name: str, help_text: str, buckets: tuple[float, ...] = _DEFAULT_BUCKETS
    ) -> None:
        self.name, self.help, self.buckets = name, help_text, buckets
        # key -> [per-bucket counts..., +Inf count, sum]
        self._series: dict[tuple, list[float]] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = _label_key(labels)
        with _LOCK:
            series = self._series.setdefault(key, [0.0] * (len(self.buckets) + 2))
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    series[i] += 1
            series[-2] += 1
            series[-1] += value

    def reset(self) -> None:
        self._series.clear()

    def count(self, **labels: str) -> float:
        series = self._series.get(_label_key(labels))
        return series[-2] if series else 0.0

    def render(self) -> list[str]:
        lines: list[str] = []
        for key, series in sorted(self._series.items()):
            for bound, count in zip(self.buckets, series):
                le = 'le="' + f"{bound:g}" + '"'
                lines.append(f"{self.name}_bucket{_format_labels(key, le)} {count:g}")
            inf = 'le="+Inf"'
            lines.append(f"{self.name}_bucket{_format_labels(key, inf)} {series[-2]:g}")
            lines.append(f"{self.name}_sum{_format_labels(key)} {series[-1]:.6f}")
            lines.append(f"{self.name}_count{_format_labels(key)} {series[-2]:g}")
        return lines

    def _quantile(self, series: list[float], q: float) -> float | None:
        """Upper bound of the bucket holding the q-th observation (an estimate)."""
        total = series[-2]
        if not total:
            return None
        target = q * total
        for bound, count in zip(self.buckets, series):
            if count >= target:
                return bound
        return float("inf")

    def snapshot(self) -> list[dict]:
        out = []
        for key, series in sorted(self._series.items()):
            count, total = series[-2], series[-1]
            out.append(
                {
                    "labels": dict(key),
                    "count": count,
                    "sum_seconds": round(total, 4),
                    "mean_seconds": round(total / count, 4) if count else None,
                    # Bucket upper bounds, so these read "at most".
                    "p50_le_seconds": self._quantile(series, 0.5),
                    "p95_le_seconds": self._quantile(series, 0.95),
                }
            )
        return out


_REGISTRY: dict[str, Counter | Gauge | Histogram] = {}


def _register(metric):
    _REGISTRY[metric.name] = metric
    return metric


# ─── Series ───────────────────────────────────────────────────────────────────

QUERIES = _register(
    Counter(
        "redline_queries_total",
        "Query requests by transport and outcome (answered, refusal, error, timeout, "
        "cancelled, blocked, busy, daily_cap, rate_limited, draining).",
    )
)
QUERY_SECONDS = _register(
    Histogram("redline_query_duration_seconds", "End-to-end pipeline latency per query.")
)
NODE_SECONDS = _register(
    Histogram("redline_node_duration_seconds", "Latency of each graph node execution.")
)
NODE_ERRORS = _register(
    Counter("redline_node_errors_total", "Graph node executions that raised, by node.")
)
LLM_CALLS = _register(
    Counter("redline_llm_calls_total", "Completed LLM calls by graph node and model.")
)
LLM_TOKENS = _register(
    Counter("redline_llm_tokens_total", "LLM tokens by graph node, model and kind.")
)
LLM_SECONDS = _register(
    Histogram("redline_llm_call_duration_seconds", "Latency of each completed LLM call.")
)
INGESTS = _register(Counter("redline_ingest_total", "Document uploads by outcome."))
RATE_LIMITED = _register(
    Counter("redline_rate_limited_total", "Requests rejected by a per-client limit, by bucket.")
)
PIPELINES_IN_FLIGHT = _register(
    Gauge("redline_pipelines_in_flight", "Pipeline slots currently held.")
)
CHECKPOINTER_DURABLE = _register(
    Gauge(
        "redline_checkpointer_durable",
        "1 when LangGraph checkpoints go to Postgres, 0 when the in-memory fallback is in use.",
    )
)
CHECKPOINT_THREADS_DELETED = _register(
    Counter(
        "redline_checkpoint_threads_deleted_total",
        "Checkpoint threads removed, by reason (completed, reset, deal_purged, expired).",
    )
)
DEGRADED = _register(
    Gauge("redline_degraded", "1 for each degraded mode currently active, labelled by reason.")
)
_STARTED = time.time()
UPTIME = _register(Gauge("redline_uptime_seconds", "Seconds since this process started."))
UPTIME.set_function(lambda: time.time() - _STARTED)


def render_prometheus() -> str:
    """Renders every registered series in the Prometheus text exposition format."""
    lines: list[str] = []
    for metric in _REGISTRY.values():
        lines.append(f"# HELP {metric.name} {metric.help}")
        lines.append(f"# TYPE {metric.name} {metric.kind}")
        lines.extend(metric.render())
    return "\n".join(lines) + "\n"


def snapshot() -> dict:
    """The same data as JSON, with mean and bucketed p50/p95 for histograms."""
    return {name: metric.snapshot() for name, metric in _REGISTRY.items()}


def reset_metrics() -> None:
    """Clears every series. For tests; nothing in production calls it."""
    for metric in _REGISTRY.values():
        metric.reset()


# ─── Per-node attribution ──────────────────────────────────────────────────────

# The node currently executing in this task, and what it has spent so far.
# contextvars are copied into tasks and worker threads at creation, and the
# value is a mutable dict, so calls made from `asyncio.gather` children or
# `to_thread` still land in the right node.
_node_scope: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "metrics_node_scope", default=None
)


@contextmanager
def node_scope(node: str) -> Iterator[dict]:
    """
    Times one graph node and collects the LLM calls made inside it.

    Args:
        node: Graph node name.

    Yields:
        The live scope dict: {node, llm_calls, prompt_tokens, completion_tokens,
        models, finish_reasons, duration_ms}. `duration_ms` is filled in on exit.
    """
    scope = {
        "node": node,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "models": [],
        "finish_reasons": [],
    }
    token = _node_scope.set(scope)
    # perf_counter, not monotonic: on Windows monotonic() ticks every ~15 ms,
    # which rounds every deterministic node down to 0.
    started = time.perf_counter()
    try:
        yield scope
    except BaseException:
        NODE_ERRORS.inc(node=node)
        raise
    finally:
        elapsed = time.perf_counter() - started
        scope["duration_ms"] = round(elapsed * 1000, 2)
        NODE_SECONDS.observe(elapsed, node=node)
        _node_scope.reset(token)


def record_llm_call(
    model: str,
    agent: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    latency_ms: float | None,
    finish_reason: str | None = None,
) -> None:
    """
    Records one completed LLM call against the global series and the active node.

    Args:
        model: Model that answered.
        agent: The wrapper's own agent label (used when no node scope is active,
            e.g. a call made outside the graph).
        prompt_tokens: Prompt tokens, if the provider reported usage.
        completion_tokens: Completion tokens, if reported.
        latency_ms: Wall time of the call in milliseconds.
        finish_reason: How the call ended ("stop", "length", ...), if reported.
            Kept per call, in order, so a node's perf shows a reply cut off at
            max_tokens next to its completion tokens.
    """
    scope = _node_scope.get()
    node = scope["node"] if scope else (agent or "unattributed")
    model = model or "unknown"
    LLM_CALLS.inc(node=node, model=model)
    if prompt_tokens:
        LLM_TOKENS.inc(prompt_tokens, node=node, model=model, kind="prompt")
    if completion_tokens:
        LLM_TOKENS.inc(completion_tokens, node=node, model=model, kind="completion")
    if latency_ms is not None:
        LLM_SECONDS.observe(latency_ms / 1000.0, node=node)
    if scope is not None:
        scope["llm_calls"] += 1
        scope["prompt_tokens"] += int(prompt_tokens or 0)
        scope["completion_tokens"] += int(completion_tokens or 0)
        if model not in scope["models"]:
            scope["models"].append(model)
        scope.setdefault("finish_reasons", []).append(finish_reason)


class _LLMUsageTap(logging.Filter):
    """Feeds the wrapper's "LLM call usage" records into `record_llm_call`."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg == LLM_USAGE_MESSAGE:
            try:
                record_llm_call(
                    model=getattr(record, "model", "") or "",
                    agent=getattr(record, "agent", None),
                    prompt_tokens=getattr(record, "prompt_tokens", None),
                    completion_tokens=getattr(record, "completion_tokens", None),
                    latency_ms=getattr(record, "latency_ms", None),
                    finish_reason=getattr(record, "finish_reason", None),
                )
            except Exception:
                # Observability must never break the call it observes.
                pass
        return True


_tap = _LLMUsageTap()


def install_llm_usage_tap() -> None:
    """Attaches the usage tap to the LLM wrapper's logger. Idempotent."""
    target = logging.getLogger(LLM_USAGE_LOGGER)
    if _tap not in target.filters:
        target.addFilter(_tap)
