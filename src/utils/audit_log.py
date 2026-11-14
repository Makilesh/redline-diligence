"""
Append-only audit trail for queries and ingestions.

Every completed query and every ingested document is recorded twice:

  * as an `AUDIT_LOG` line in the structured application log (stdout), which
    is what a container log driver or a log shipper collects; and
  * when AUDIT_LOG_DIR is set, as one JSON line in `audit_<UTC date>.jsonl`
    under that directory, opened in append mode and never rewritten.

The file half existed before as a module nothing imported, while the routes
wrote the log line themselves — so the "immutable audit log" the routes
promised was a stdout line, kept only as long as the container's log rotation
allowed (30MB in the production compose file). AUDIT_LOG_DIR was documented in
`.env.example` and read by nothing.

Writing is best-effort by design: a full disk must not turn every query into a
500. A failed write is logged at ERROR, so it is visible rather than silent.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger("audit_log")


def _audit_dir() -> Path | None:
    raw = os.getenv("AUDIT_LOG_DIR", "").strip()
    return Path(raw) if raw else None


def record(event: str, **fields) -> None:
    """
    Records one audit event.

    Args:
        event: Event name, e.g. "query_completed" or "document_ingested".
        **fields: JSON-serialisable event fields.
    """
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    logger.info("AUDIT_LOG", extra=entry)

    directory = _audit_dir()
    if directory is None:
        return
    path = directory / f"audit_{entry['timestamp'][:10]}.jsonl"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        logger.error(
            "Audit file write failed",
            extra={"path": str(path), "error_type": type(e).__name__, "event": event},
        )
