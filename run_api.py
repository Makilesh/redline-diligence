"""
Process entrypoint for the API server.

Exists for one reason: on Windows, psycopg refuses to run on asyncio's
ProactorEventLoop ("Psycopg cannot use the 'ProactorEventLoop' to run in async
mode"). AsyncPostgresSaver therefore raised at every startup and the
orchestrator degraded to MemorySaver — durable checkpointing was silently dead
on every Windows run, and the only trace was one WARNING line.

Setting asyncio's event loop policy does not fix this, which is the part worth
knowing. uvicorn 0.49 does not consult the policy: `uvicorn/loops/asyncio.py`
returns `asyncio.ProactorEventLoop` as an explicit loop *factory* on win32, and
`Server.run()` passes that factory straight to the runner. A factory overrides
the policy, so `set_event_loop_policy(WindowsSelectorEventLoopPolicy())` is inert
no matter how early it runs.

So the loop is constructed here instead, and uvicorn is handed a server to run
on it rather than being asked to make its own.

    python run_api.py

`uvicorn api.main:app` remains correct on Linux, where the default loop is
already selector-based — the Docker image is unaffected and keeps uvloop. On
Windows that command still degrades to in-memory checkpointing (and now says so
in `/health`).

The selector loop's limits are real but do not bind here: ~512 sockets and no
subprocess support. This process makes HTTP calls to Qdrant and Postgres and
spawns nothing.

Three more things live here because this is the one place that runs before the
app is imported and that owns the server object:

  * `.env` is loaded explicitly. It used to be loaded only as a side effect of
    importing LiteLLM, which meant API_HOST and API_PORT — read below, before
    that import — silently ignored it.
  * Exactly one worker. Rate limits, the concurrency gate and the daily cap are
    process-local (api/security.py); a worker count above one is refused here
    rather than quietly multiplying every limit.
  * Shutdown is announced. On SIGTERM uvicorn stops accepting connections and
    waits for the open ones; the app is told at that moment, so new pipeline
    runs are refused with a 503 while the in-flight ones get
    SHUTDOWN_GRACE_SECONDS to finish before they are cancelled.
"""

import asyncio
import os
import sys

import uvicorn
from dotenv import load_dotenv

# Before reading any setting. Values already in the environment win, so a
# container's injected configuration is never overridden by a stray file.
load_dotenv()


class _Server(uvicorn.Server):
    """uvicorn's server, plus a heads-up to the app when shutdown begins."""

    def handle_exit(self, sig, frame) -> None:
        # Runs in the signal handler, before uvicorn starts draining. Imported
        # here so that importing this module does not import the application.
        from src.utils import runtime_status

        runtime_status.set_phase("draining")
        super().handle_exit(sig, frame)


def _shutdown_grace_seconds() -> int:
    """How long in-flight requests get to finish after SIGTERM (default 25s)."""
    try:
        return max(1, int(os.getenv("SHUTDOWN_GRACE_SECONDS", "25")))
    except ValueError:
        return 25


def _refuse_multiple_workers() -> None:
    """Exits with an explanation if the environment asks for more than one worker."""
    from api.security import enforce_single_worker

    try:
        enforce_single_worker()
    except RuntimeError as e:
        sys.exit(str(e))


def main() -> None:
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    _refuse_multiple_workers()

    config = uvicorn.Config(
        "api.main:app",
        host=host,
        port=port,
        workers=1,
        # Without this uvicorn waits for open connections indefinitely, and an
        # SSE stream is an open connection: the container runtime's SIGKILL
        # (10s by default) would be what actually ended the process.
        timeout_graceful_shutdown=_shutdown_grace_seconds(),
    )
    server = _Server(config)

    if sys.platform != "win32":
        # uvicorn picks its own loop here, so uvloop is still used when installed.
        server.run()
        return

    # asyncio.Runner is 3.11+; loop_factory is the only way to defeat uvicorn's
    # hardcoded Proactor factory without monkeypatching its internals.
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(server.serve())


if __name__ == "__main__":
    main()
