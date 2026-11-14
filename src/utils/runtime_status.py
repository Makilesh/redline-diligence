"""
What this process is actually running on, as opposed to what it was asked for.

Every fallback in this codebase is deliberate — Postgres down means in-memory
checkpoints, a failed model warmup means a lazy load on the first query — and
each one used to be a single WARNING line at startup. That is how durable
checkpointing was dead on every Windows run for weeks (DECISIONS_LOG 14, 26):
the service answered questions, `/health` said "healthy", and the only evidence
had scrolled away.

This module is the one place those fallbacks are recorded, so `/health`,
`/ready`, `/metrics` and the logs all report the same thing. It holds plain
in-process state and does no I/O, which keeps `/health` a liveness probe.
"""

import time
from dataclasses import dataclass, field

from src.utils import metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Model warmup outcomes.
MODEL_READY = "ready"
MODEL_FAILED = "failed"  # warmup raised; the model will be retried lazily on first use
MODEL_LAZY = "lazy"  # warmup skipped (WARM_MODELS=0); loads on first use
MODEL_DISABLED = "disabled"  # switched off by configuration


@dataclass
class _RuntimeStatus:
    started_at: float = field(default_factory=time.time)
    # "starting" until the lifespan finishes, then "ready"; "draining" once
    # shutdown has begun and new pipeline runs are refused.
    phase: str = "starting"
    # "postgres" | "memory" | "none" (not initialised).
    checkpointer_backend: str = "none"
    # Exception class name only — the message can carry a connection string.
    checkpointer_error: str = ""
    models: dict[str, str] = field(default_factory=dict)
    # reason -> human-readable explanation, for every degraded mode in effect.
    degraded: dict[str, str] = field(default_factory=dict)


_status = _RuntimeStatus()


def reset_status() -> None:
    """Returns to the pristine state. For tests."""
    global _status
    for reason in list(_status.degraded):
        metrics.DEGRADED.set(0, reason=reason)
    _status = _RuntimeStatus()


def set_phase(phase: str) -> None:
    _status.phase = phase


def phase() -> str:
    return _status.phase


def is_draining() -> bool:
    return _status.phase == "draining"


def mark_degraded(reason: str, explanation: str) -> None:
    """
    Records a degraded mode and logs it at ERROR, once per reason.

    ERROR rather than WARNING on purpose: the process keeps serving, but an
    operator grepping for errors after a deploy must see that it is not running
    the configuration they asked for.

    Args:
        reason: Stable machine-readable key, e.g. "checkpointer_in_memory".
        explanation: What it means for the operator.
    """
    if reason not in _status.degraded:
        logger.error("DEGRADED_MODE", extra={"reason": reason, "explanation": explanation})
    _status.degraded[reason] = explanation
    metrics.DEGRADED.set(1, reason=reason)


def clear_degraded(reason: str) -> None:
    if _status.degraded.pop(reason, None) is not None:
        logger.info("Degraded mode cleared", extra={"reason": reason})
        metrics.DEGRADED.set(0, reason=reason)


def set_checkpointer(backend: str, error: str = "") -> None:
    """
    Records which checkpointer the compiled graph is using.

    Args:
        backend: "postgres" or "memory".
        error: Exception class name when Postgres could not be used.
    """
    _status.checkpointer_backend = backend
    _status.checkpointer_error = error
    metrics.CHECKPOINTER_DURABLE.set(1 if backend == "postgres" else 0)
    if backend == "postgres":
        clear_degraded("checkpointer_in_memory")
    else:
        mark_degraded(
            "checkpointer_in_memory",
            "LangGraph checkpoints are held in process memory: in-flight runs do not "
            "survive a restart. Postgres could not be used"
            + (f" ({error})" if error else "")
            + ".",
        )


def checkpointer_durable() -> bool:
    return _status.checkpointer_backend == "postgres"


def set_model(name: str, state: str) -> None:
    """Records one local model's warmup outcome."""
    _status.models[name] = state
    reason = f"model_warmup_failed:{name}"
    if state == MODEL_FAILED:
        mark_degraded(
            reason,
            f"Warming {name} failed at startup; it will be loaded on the first request "
            "that needs it, which will be slow and may fail the same way.",
        )
    else:
        clear_degraded(reason)


def describe() -> dict:
    """
    Snapshot for `/health` and `/ready`.

    Returns:
        {phase, uptime_seconds, checkpointer: {backend, durable, error},
         models: {name: state}, degraded: [reason...],
         degraded_detail: {reason: explanation}}
    """
    return {
        "phase": _status.phase,
        "uptime_seconds": round(time.time() - _status.started_at, 1),
        "checkpointer": {
            "backend": _status.checkpointer_backend,
            "durable": checkpointer_durable(),
            "error": _status.checkpointer_error,
        },
        "models": dict(_status.models),
        "degraded": sorted(_status.degraded),
        "degraded_detail": dict(_status.degraded),
    }
