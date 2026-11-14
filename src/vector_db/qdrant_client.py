"""
Qdrant client singleton management.

CRITICAL: AsyncQdrantClient must be created inside a running event loop.
This function lazily initializes the client on first call, which will
always happen inside an async context (from a request handler or startup).

The singleton pattern ensures all agents and request handlers share
one connection pool instead of creating per-request clients.
"""

import os
from qdrant_client import AsyncQdrantClient

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_qdrant_client: AsyncQdrantClient | None = None

# What get_qdrant_client() built, and the client it built it for. Kept beside
# the singleton rather than derived from it so get_qdrant_backend() can tell a
# client this module created from one a test or the eval harness swapped in.
_backend: dict | None = None
_backend_client: AsyncQdrantClient | None = None

QDRANT_MODES = ("server", "local")
DEFAULT_LOCAL_PATH = "./qdrant_local_db"


def qdrant_mode() -> str:
    """
    The configured Qdrant backend: "server" (default) or "local".

    Returns:
        QDRANT_MODE, lower-cased.

    Raises:
        RuntimeError: If QDRANT_MODE is set to anything else.
    """
    mode = os.getenv("QDRANT_MODE", "server").strip().lower() or "server"
    if mode not in QDRANT_MODES:
        raise RuntimeError(
            f"QDRANT_MODE={mode!r} is not supported; use one of {', '.join(QDRANT_MODES)}"
        )
    return mode


def _display_url(url: str) -> str:
    """The URL without any credentials embedded in it, for logs and health output."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if not parsed.hostname:
        return url
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def get_qdrant_client() -> AsyncQdrantClient:
    """
    Returns a singleton AsyncQdrantClient instance.

    CRITICAL: AsyncQdrantClient must be created inside a running event loop.
    This function lazily initializes the client on first call, which will
    always happen inside an async context (from a request handler or startup).

    The singleton pattern ensures all agents and request handlers share
    one connection pool instead of creating per-request clients.

    The backend is whatever QDRANT_MODE says, never a guess. This function used
    to probe the server's port for one second and, if nothing answered, switch
    to an embedded database under ./qdrant_local_db for the life of the process.
    That database starts empty, so an API started while Qdrant was still booting
    (or with a mistyped QDRANT_URL) came up healthy, accepted uploads into a
    directory the server never sees, and answered every question with "no
    relevant information". An unreachable server is now an error — at startup
    through assert_server_compatible, otherwise on first use.

    Configuration via environment variables:
        QDRANT_MODE: "server" (default) or "local". Local is an embedded on-disk
            index for development without Docker; it is single-process and has
            no HNSW or quantization.
        QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
        QDRANT_GRPC_PORT: gRPC port for faster operations (default: 6334)
        QDRANT_API_KEY: API key. Required by Qdrant Cloud; unset when self-hosted.
        QDRANT_LOCAL_PATH: Directory for local mode (default: ./qdrant_local_db)

    Returns:
        Shared AsyncQdrantClient instance.

    Raises:
        RuntimeError: If called before event loop is available, or QDRANT_MODE
            is not a supported value.
    """
    global _qdrant_client, _backend, _backend_client
    if _qdrant_client is None:
        if qdrant_mode() == "local":
            local_path = os.getenv("QDRANT_LOCAL_PATH") or DEFAULT_LOCAL_PATH
            logger.warning(
                "Initializing AsyncQdrantClient singleton (Local Mode, QDRANT_MODE=local): "
                "embedded on-disk index, not a Qdrant server",
                extra={"local_path": local_path},
            )
            _qdrant_client = AsyncQdrantClient(path=local_path)
            _backend = {"mode": "local", "location": local_path}
        else:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            api_key = os.getenv("QDRANT_API_KEY") or None
            grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

            # Managed clusters terminate TLS on the REST port and do not expose
            # plain gRPC on 6334. Preferring gRPC there produces a client that
            # constructs fine and fails on first use, so the transport follows
            # the deployment: gRPC when self-hosted, REST when managed.
            prefer_grpc = api_key is None

            logger.info(
                "Initializing AsyncQdrantClient singleton (Server Mode)",
                extra={
                    "qdrant_url": _display_url(qdrant_url),
                    "grpc_port": grpc_port,
                    "prefer_grpc": prefer_grpc,
                    "authenticated": api_key is not None,
                },
            )
            _qdrant_client = AsyncQdrantClient(
                url=qdrant_url,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,  # gRPC is faster for batch operations
                api_key=api_key,
                timeout=30,
            )
            _backend = {"mode": "server", "location": _display_url(qdrant_url)}
        _backend_client = _qdrant_client
    return _qdrant_client


def get_qdrant_backend() -> dict:
    """
    Which Qdrant backend the process is using, for health and status output.

    Does not create the client and makes no network call.

    Returns:
        {"mode": ..., "location": ...} where mode is
          "server"        a Qdrant server at `location` (URL, credentials removed)
          "local"         embedded on-disk index at `location` (QDRANT_MODE=local)
          "injected"      a client this module did not create (tests, eval harness)
          "uninitialized" get_qdrant_client() has not been called yet
    """
    if _qdrant_client is None:
        return {"mode": "uninitialized", "location": None}
    if _backend is None or _qdrant_client is not _backend_client:
        return {"mode": "injected", "location": None}
    return dict(_backend)


async def assert_server_compatible(client: AsyncQdrantClient) -> None:
    """
    Fails startup if the Qdrant server is too far from the installed client.

    Qdrant's contract is that major versions match and minor versions differ by
    at most one. The client already notices a violation — and only emits a
    `UserWarning`, which is invisible under a JSON log handler.

    That is far too quiet for what actually happens. Running client 1.18.0
    against server 1.12.1, *every* ingest failed at upsert ("Vector dimension
    error: expected dim: 1024, got 0", "Conversion between sparse and regular
    vectors failed") because the newer client's gRPC vector encoding was
    unreadable by the older server. REST on the same pair worked, so the health
    check passed, collections were created, queries ran, and the engine simply
    answered "I was unable to find sufficient relevant information" 41 times
    against an empty index. Nothing in the pipeline treats an empty index as an
    error, so the only symptom was uniformly bad results.

    A version skew that makes writes silently fail should stop the process, not
    add a line to a log nobody reads.

    Args:
        client: The initialized Qdrant client.

    Raises:
        RuntimeError: If the server version is incompatible with the client, or
            if the configured server cannot be reached at all (server mode only;
            a local or injected client has no server to check).
    """
    try:
        from qdrant_client import version as _qc_version

        client_version = _qc_version.__version__
    except Exception:
        import importlib.metadata

        client_version = importlib.metadata.version("qdrant-client")

    backend = get_qdrant_backend()
    is_server = backend["mode"] == "server" and client is _qdrant_client

    try:
        info = await client.info()
        server_version = info.version
    except Exception as e:
        if not is_server:
            # Local-storage mode has no server; nothing to be incompatible with.
            logger.info(f"Skipping Qdrant version check: {e}")
            return
        # The version endpoint failing is not proof the server is down (a proxy
        # may not route "/"), so ask it something every deployment answers
        # before refusing to start.
        try:
            await client.get_collections()
        except Exception as unreachable:
            raise RuntimeError(
                f"Qdrant server at {backend['location']} is unreachable: {unreachable}. "
                "Refusing to start without a vector store. Start Qdrant or fix "
                "QDRANT_URL; for an embedded on-disk index without a server, set "
                "QDRANT_MODE=local explicitly."
            ) from unreachable
        logger.warning(
            "Qdrant answers but did not report a version; skipping compatibility check",
            extra={"error": str(e)},
        )
        return

    def _major_minor(v: str) -> tuple[int, int]:
        parts = v.lstrip("v").split(".")
        return int(parts[0]), int(parts[1])

    try:
        c_major, c_minor = _major_minor(client_version)
        s_major, s_minor = _major_minor(server_version)
    except (ValueError, IndexError):
        logger.warning(
            "Could not parse Qdrant versions; skipping compatibility check",
            extra={"client": client_version, "server": server_version},
        )
        return

    if c_major != s_major or abs(c_minor - s_minor) > 1:
        raise RuntimeError(
            f"Qdrant client {client_version} is incompatible with server "
            f"{server_version}. Major versions must match and minor versions "
            f"must differ by at most 1. Writes will fail silently over gRPC — "
            f"documents appear to ingest and the index stays empty. Align the "
            f"image tag in docker-compose.yml with the qdrant-client pin in "
            f"requirements.txt."
        )

    logger.info(
        "Qdrant client/server versions compatible",
        extra={"client": client_version, "server": server_version},
    )


async def close_qdrant_client() -> None:
    """
    Call on application shutdown to cleanly close connections.

    Should be invoked in FastAPI's lifespan shutdown handler
    or equivalent application teardown hook.
    """
    global _qdrant_client, _backend, _backend_client
    if _qdrant_client is not None:
        logger.info("Closing AsyncQdrantClient connection")
        await _qdrant_client.close()
        _qdrant_client = None
        _backend = None
        _backend_client = None
        logger.info("AsyncQdrantClient closed successfully")
