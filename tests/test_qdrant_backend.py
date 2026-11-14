"""
The Qdrant backend is chosen by configuration, never by a failed probe.

get_qdrant_client() used to fall back to an empty embedded database under
./qdrant_local_db when the server did not answer a one-second socket probe. The
API then started healthy, accepted uploads the server never saw and refused every
question. These tests pin the replacement:

- server mode never creates a local database, whatever the server's state;
- an unreachable server stops startup with an error that names the way out;
- local mode exists only behind QDRANT_MODE=local, at a configurable path;
- get_qdrant_backend() reports which backend is live.

No test here reaches a real server (the only address used is the closed discard
port on loopback) or writes outside tmp_path.
"""

from __future__ import annotations

import pytest

import src.vector_db.qdrant_client as qc


@pytest.fixture(autouse=True)
def _isolated_singleton(monkeypatch, tmp_path):
    """Fresh singleton state per test, and a working directory nothing may litter."""
    monkeypatch.setattr(qc, "_qdrant_client", None)
    monkeypatch.setattr(qc, "_backend", None)
    monkeypatch.setattr(qc, "_backend_client", None)
    for name in ("QDRANT_MODE", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_LOCAL_PATH",
                 "QDRANT_GRPC_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


class _DownServer:
    """A server-mode client whose server does not answer."""

    async def info(self):
        raise ConnectionError("connection refused")

    async def get_collections(self):
        raise ConnectionError("connection refused")


def _install(monkeypatch, client, mode="server", location="http://qdrant:6333"):
    monkeypatch.setattr(qc, "_qdrant_client", client)
    monkeypatch.setattr(qc, "_backend", {"mode": mode, "location": location})
    monkeypatch.setattr(qc, "_backend_client", client)


class TestServerMode:
    @pytest.mark.asyncio
    async def test_unreachable_server_never_becomes_a_local_database(self, monkeypatch, tmp_path):
        """Port 9 (discard) is closed; the old code created ./qdrant_local_db here."""
        monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")

        client = qc.get_qdrant_client()

        assert qc.get_qdrant_backend() == {"mode": "server", "location": "http://127.0.0.1:9"}
        assert client is qc.get_qdrant_client()
        assert list(tmp_path.iterdir()) == []
        await qc.close_qdrant_client()

    @pytest.mark.asyncio
    async def test_unreachable_server_stops_startup(self, monkeypatch):
        client = _DownServer()
        _install(monkeypatch, client)

        with pytest.raises(RuntimeError) as err:
            await qc.assert_server_compatible(client)

        message = str(err.value)
        assert "unreachable" in message
        assert "http://qdrant:6333" in message
        assert "QDRANT_MODE=local" in message

    @pytest.mark.asyncio
    async def test_missing_version_endpoint_alone_is_not_fatal(self, monkeypatch):
        """A proxy that hides "/" must not take a working deployment down."""

        class NoVersion(_DownServer):
            async def get_collections(self):
                return []

        client = NoVersion()
        _install(monkeypatch, client)

        await qc.assert_server_compatible(client)

    def test_credentials_in_the_url_are_not_reported(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "https://user:s3cret@cluster.example.com:6333/path")
        monkeypatch.setenv("QDRANT_API_KEY", "k")

        qc.get_qdrant_client()

        assert qc.get_qdrant_backend()["location"] == "https://cluster.example.com:6333"


class TestLocalMode:
    @pytest.mark.asyncio
    async def test_local_mode_is_opt_in_and_uses_the_configured_path(self, monkeypatch, tmp_path):
        target = tmp_path / "index"
        monkeypatch.setenv("QDRANT_MODE", "local")
        monkeypatch.setenv("QDRANT_LOCAL_PATH", str(target))

        client = qc.get_qdrant_client()
        try:
            assert qc.get_qdrant_backend() == {"mode": "local", "location": str(target)}
            assert (await client.get_collections()).collections == []
            # No server to be incompatible with: startup must not be blocked.
            await qc.assert_server_compatible(client)
            assert target.is_dir()
        finally:
            await qc.close_qdrant_client()

    def test_unknown_mode_is_an_error_not_a_default(self, monkeypatch):
        monkeypatch.setenv("QDRANT_MODE", "memory")

        with pytest.raises(RuntimeError, match="QDRANT_MODE"):
            qc.get_qdrant_client()

        assert qc.get_qdrant_backend()["mode"] == "uninitialized"


class TestBackendReport:
    def test_uninitialized_before_first_use(self):
        assert qc.get_qdrant_backend() == {"mode": "uninitialized", "location": None}

    @pytest.mark.asyncio
    async def test_a_swapped_in_client_is_reported_as_injected(self, monkeypatch):
        """eval.corpus.use_client and the API tests replace the singleton directly."""
        monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
        real = qc.get_qdrant_client()
        monkeypatch.setattr(qc, "_qdrant_client", object())

        assert qc.get_qdrant_backend() == {"mode": "injected", "location": None}

        monkeypatch.setattr(qc, "_qdrant_client", real)
        assert qc.get_qdrant_backend()["mode"] == "server"
        await qc.close_qdrant_client()
        assert qc.get_qdrant_backend()["mode"] == "uninitialized"

    @pytest.mark.asyncio
    async def test_injected_client_without_a_server_does_not_block_startup(self, monkeypatch):
        client = _DownServer()
        monkeypatch.setattr(qc, "_qdrant_client", client)

        await qc.assert_server_compatible(client)
