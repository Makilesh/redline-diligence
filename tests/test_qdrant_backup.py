"""
scripts/qdrant_backup.py against an in-process stand-in for Qdrant's snapshot API.

The restore drill against a real Qdrant server is recorded in Decision 105; these tests pin
the script's own guarantees: checksums end to end, nothing restored from a backup that does
not verify, and no collection holding points replaced unless asked.
"""
from __future__ import annotations

import hashlib
import json
import sys
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qdrant_backup as qb  # noqa: E402

CHILD, PARENT = qb.COLLECTIONS


class FakeQdrant:
    """Collections of points; a snapshot is the points serialised."""

    def __init__(self, collections: dict[str, list[dict]] | None = None, bad_checksum: bool = False):
        self.points = {k: list(v) for k, v in (collections or {}).items()}
        self.snapshots: dict[tuple[str, str], bytes] = {}
        self.bad_checksum = bad_checksum
        self.uploads: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.strip("/").split("/")
        if request.url.path == "/":
            return httpx.Response(200, json={"title": "qdrant", "version": "1.18.0"})
        name = parts[1]
        if parts[2:] == ["points", "count"]:
            assert json.loads(request.content) == {"exact": True}
            if name not in self.points:
                return httpx.Response(404, json={"status": {"error": "Not found"}})
            return httpx.Response(200, json={"result": {"count": len(self.points[name])}})
        if parts[2:] == ["snapshots"] and request.method == "POST":
            blob = json.dumps(self.points[name]).encode()
            snap = f"{name}-1.snapshot"
            self.snapshots[(name, snap)] = blob
            checksum = "0" * 64 if self.bad_checksum else hashlib.sha256(blob).hexdigest()
            return httpx.Response(200, json={"result": {"name": snap, "size": len(blob), "checksum": checksum}})
        if parts[2] == "snapshots" and len(parts) == 4 and parts[3] != "upload":
            key = (name, parts[3])
            if request.method == "GET":
                return httpx.Response(200, content=self.snapshots[key])
            if request.method == "DELETE":
                del self.snapshots[key]
                return httpx.Response(200, json={"result": True})
        if parts[2:] == ["snapshots", "upload"]:
            assert request.url.params["priority"] == "snapshot"
            message = BytesParser(policy=email_policy).parsebytes(
                b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.content)
            [part] = [p for p in message.iter_parts() if p.get_param("name", header="content-disposition") == "snapshot"]
            blob = part.get_payload(decode=True)
            if hashlib.sha256(blob).hexdigest() != request.url.params["checksum"]:  # as Qdrant does
                return httpx.Response(400, json={"status": {"error": "checksum mismatch"}})
            self.points[name] = json.loads(blob)
            self.uploads.append(name)
            return httpx.Response(200, json={"result": True})
        if parts[2:] == ["points", "scroll"]:
            body = json.loads(request.content)
            ordered = sorted(self.points[name], key=lambda p: str(p["id"]))
            start = 0 if body.get("offset") is None else [str(p["id"]) for p in ordered].index(body["offset"])
            page = ordered[start:start + body["limit"]]
            nxt = str(ordered[start + body["limit"]]["id"]) if start + body["limit"] < len(ordered) else None
            return httpx.Response(200, json={"result": {"points": page, "next_page_offset": nxt}})
        return httpx.Response(500, json={"status": {"error": f"unhandled {request.method} {request.url.path}"}})

    def client(self) -> httpx.Client:
        return httpx.Client(base_url="http://qdrant.test", transport=httpx.MockTransport(self.handler))


def _points(prefix: str, n: int) -> list[dict]:
    return [{"id": f"{prefix}-{i:04d}", "payload": {"deal_id": "d1", "text": f"chunk {i}"},
             "vector": {"dense": [i / 10, 0.5], "bm25": {"indices": [i], "values": [1.0]}}} for i in range(n)]


@pytest.fixture
def source():
    return FakeQdrant({CHILD: _points("c", 600), PARENT: _points("p", 90)})


@pytest.fixture
def audit_dir(tmp_path):
    d = tmp_path / "audit"
    d.mkdir()
    (d / "audit_2026-11-22.jsonl").write_text('{"event": "document_indexed"}\n', encoding="utf-8")
    (d / "audit_2026-11-23.jsonl").write_text('{"event": "query"}\n', encoding="utf-8")
    (d / "notes.txt").write_text("not an audit file", encoding="utf-8")
    return d


def _backup(fake, out, audit_dir=None, **kw):
    with fake.client() as c:
        return qb.backup(c, out, audit_dir=audit_dir, keep_server=kw.get("keep", False),
                         with_fingerprint=kw.get("fingerprint", True))


def test_a_backup_holds_both_collections_with_checksums_and_leaves_no_server_snapshots(source, audit_dir, tmp_path):
    out = tmp_path / "bk"
    manifest = _backup(source, out, audit_dir)
    assert set(manifest["collections"]) == {CHILD, PARENT}
    for name, entry in manifest["collections"].items():
        path = out / entry["snapshot"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert entry["points_count"] == entry["points_count_after"] == len(source.points[name])
        assert entry["fingerprint"]["points"] == len(source.points[name])
    assert sorted(manifest["audit_files"]) == ["audit_2026-11-22.jsonl", "audit_2026-11-23.jsonl"]
    assert source.snapshots == {}
    assert qb.verify(out) == []


def test_server_snapshots_are_kept_when_asked(source, tmp_path):
    _backup(source, tmp_path / "bk", keep=True)
    assert len(source.snapshots) == 2


def test_a_download_that_disagrees_with_the_servers_checksum_is_an_error(tmp_path):
    fake = FakeQdrant({CHILD: _points("c", 3), PARENT: _points("p", 1)}, bad_checksum=True)
    with pytest.raises(qb.BackupError, match="differs from the server"):
        _backup(fake, tmp_path / "bk")


def test_a_backup_refuses_a_missing_collection_and_a_used_directory(tmp_path):
    with pytest.raises(qb.BackupError, match="does not exist"):
        _backup(FakeQdrant({CHILD: _points("c", 3)}), tmp_path / "bk1")
    used = tmp_path / "bk2"
    used.mkdir()
    (used / "old").write_text("x", encoding="utf-8")
    with pytest.raises(qb.BackupError, match="not empty"):
        _backup(FakeQdrant({CHILD: [], PARENT: []}), used)


def test_verify_finds_a_changed_snapshot_and_a_missing_audit_file(source, audit_dir, tmp_path):
    out = tmp_path / "bk"
    manifest = _backup(source, out, audit_dir)
    snap = out / manifest["collections"][CHILD]["snapshot"]
    data = bytearray(snap.read_bytes())
    data[10] ^= 1
    snap.write_bytes(bytes(data))
    (out / "audit" / "audit_2026-11-23.jsonl").unlink()
    problems = qb.verify(out)
    assert any("SHA-256 differs" in p for p in problems)
    assert any("audit_2026-11-23.jsonl" in p for p in problems)
    assert qb.main(["verify", str(out)]) == 1


def test_a_restore_into_an_empty_server_reproduces_every_point_and_the_audit_log(source, audit_dir, tmp_path):
    out = tmp_path / "bk"
    _backup(source, out, audit_dir)
    target = FakeQdrant()
    restored_audit = tmp_path / "restored_audit"
    with target.client() as c:
        report = qb.restore(c, out, audit_dir=restored_audit, replace=False, timeout=10)
    assert report["ok"]
    assert all(r["fingerprint_ok"] for r in report["collections"].values())
    assert target.points == source.points
    assert sorted(p.name for p in restored_audit.iterdir()) == ["audit_2026-11-22.jsonl", "audit_2026-11-23.jsonl"]


def test_a_restore_does_not_replace_a_collection_that_holds_points(source, tmp_path):
    out = tmp_path / "bk"
    _backup(source, out)
    live = FakeQdrant({CHILD: _points("live", 5), PARENT: []})
    with live.client() as c, pytest.raises(qb.BackupError, match="holds 5 points"):
        qb.restore(c, out, audit_dir=None, replace=False, timeout=10)
    assert live.uploads == [] and len(live.points[CHILD]) == 5
    with live.client() as c:
        assert qb.restore(c, out, audit_dir=None, replace=True, timeout=10)["ok"]
    assert live.points == source.points


def test_nothing_is_restored_from_a_backup_that_does_not_verify(source, tmp_path):
    out = tmp_path / "bk"
    manifest = _backup(source, out)
    (out / manifest["collections"][PARENT]["snapshot"]).write_bytes(b"corrupt")
    target = FakeQdrant()
    with target.client() as c, pytest.raises(qb.BackupError, match="does not verify"):
        qb.restore(c, out, audit_dir=None, replace=True, timeout=10)
    assert target.uploads == []


def test_a_restore_does_not_overwrite_a_different_audit_file(source, audit_dir, tmp_path):
    out = tmp_path / "bk"
    _backup(source, out, audit_dir)
    dest = tmp_path / "dest_audit"
    dest.mkdir()
    (dest / "audit_2026-11-23.jsonl").write_text('{"event": "something else"}\n', encoding="utf-8")
    with FakeQdrant().client() as c, pytest.raises(qb.BackupError, match="exists with other content"):
        qb.restore(c, out, audit_dir=dest, replace=False, timeout=10)


def test_a_fingerprint_does_not_depend_on_page_size(source):
    with source.client() as c:
        assert qb.fingerprint(c, CHILD, page=7) == qb.fingerprint(c, CHILD, page=256)
