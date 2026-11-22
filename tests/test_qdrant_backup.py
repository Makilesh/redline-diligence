"""
scripts/qdrant_backup.py against an in-process stand-in for Qdrant's snapshot API.

The restore drill against a real Qdrant server is recorded in Decision 105; these tests pin
the script's own guarantees: checksums end to end, a backup that matches one state of the
index or fails, nothing restored from a backup that does not verify, no collection holding
points replaced unless asked, no audit line ever lost, and a failure that says what it left.
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

    def __init__(self, collections: dict[str, list[dict]] | None = None, bad_checksum: bool = False,
                 on_snapshot=None, fail_upload_for: str | None = None, upload_failure="500",
                 count_fails_after_upload: bool = False, delete_status: int = 200):
        self.points = {k: list(v) for k, v in (collections or {}).items()}
        self.snapshots: dict[tuple[str, str], bytes] = {}
        self.bad_checksum = bad_checksum
        self.on_snapshot = on_snapshot  # called after a snapshot is taken: a write landing mid-backup
        self.fail_upload_for = fail_upload_for
        self.upload_failure = upload_failure  # "500", "400" or "timeout"
        self.count_fails_after_upload = count_fails_after_upload
        self.delete_status = delete_status
        self.uploads: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.strip("/").split("/")
        if request.url.path == "/":
            return httpx.Response(200, json={"title": "qdrant", "version": "1.18.0"})
        name = parts[1]
        if parts[2:] == ["points", "count"]:
            assert json.loads(request.content) == {"exact": True}
            if self.count_fails_after_upload and self.uploads:
                return httpx.Response(503, json={"status": {"error": "shards are being recovered"}})
            if name not in self.points:
                return httpx.Response(404, json={"status": {"error": "Not found"}})
            return httpx.Response(200, json={"result": {"count": len(self.points[name])}})
        if parts[2:] == ["snapshots"] and request.method == "POST":
            blob = json.dumps(self.points[name]).encode()
            snap = f"{name}-1.snapshot"
            self.snapshots[(name, snap)] = blob
            checksum = "0" * 64 if self.bad_checksum else hashlib.sha256(blob).hexdigest()
            if self.on_snapshot:
                self.on_snapshot(self, name)
            return httpx.Response(200, json={"result": {"name": snap, "size": len(blob), "checksum": checksum}})
        if parts[2] == "snapshots" and len(parts) == 4 and parts[3] != "upload":
            key = (name, parts[3])
            if request.method == "GET":
                return httpx.Response(200, content=self.snapshots[key])
            if request.method == "DELETE":
                if self.delete_status >= 300:
                    return httpx.Response(self.delete_status, json={"status": {"error": "busy"}})
                del self.snapshots[key]
                return httpx.Response(200, json={"result": True})
        if parts[2:] == ["snapshots", "upload"]:
            assert request.url.params["priority"] == "snapshot"
            if name == self.fail_upload_for:
                if self.upload_failure == "timeout":
                    raise httpx.ReadTimeout("timed out waiting for the recovery", request=request)
                if self.upload_failure == "400":
                    return httpx.Response(400, json={"status": {"error": "Wrong input: checksum mismatch"}})
                return httpx.Response(500, json={"status": {"error": "No space left on device"}})
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
                         with_fingerprint=kw.get("fingerprint", True), allow_changes=kw.get("allow", False))


def _restore(fake, out, audit_dir=None, replace=False):
    with fake.client() as c:
        return qb.restore(c, out, audit_dir=audit_dir, replace=replace, timeout=10)


def test_a_backup_holds_both_collections_with_checksums_and_leaves_no_server_snapshots(source, audit_dir, tmp_path):
    out = tmp_path / "bk"
    manifest = _backup(source, out, audit_dir)
    assert set(manifest["collections"]) == {CHILD, PARENT}
    for name, entry in manifest["collections"].items():
        path = out / entry["snapshot"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert entry["points_count"] == len(source.points[name]) and not entry["changed_during_backup"]
        assert entry["fingerprint"]["points"] == len(source.points[name])
    assert sorted(manifest["audit_files"]) == ["audit_2026-11-22.jsonl", "audit_2026-11-23.jsonl"]
    assert source.snapshots == {}
    assert qb.verify(out) == []


def test_server_snapshots_are_kept_when_asked(source, tmp_path):
    _backup(source, tmp_path / "bk", keep=True)
    assert len(source.snapshots) == 2


def test_a_failed_download_still_deletes_the_server_snapshot(tmp_path):
    fake = FakeQdrant({CHILD: _points("c", 3), PARENT: _points("p", 1)}, bad_checksum=True)
    with pytest.raises(qb.BackupError, match="differs from the server"):
        _backup(fake, tmp_path / "bk")
    assert fake.snapshots == {}


def test_a_backup_refuses_a_missing_collection_and_a_used_directory(tmp_path):
    with pytest.raises(qb.BackupError, match="does not exist"):
        _backup(FakeQdrant({CHILD: _points("c", 3)}), tmp_path / "bk1")
    used = tmp_path / "bk2"
    used.mkdir()
    (used / "old").write_text("x", encoding="utf-8")
    with pytest.raises(qb.BackupError, match="not empty"):
        _backup(FakeQdrant({CHILD: [], PARENT: []}), used)


@pytest.mark.parametrize("where", ["missing", "empty"])
def test_an_audit_directory_without_audit_files_fails_before_anything_is_taken(source, tmp_path, where):
    audit = tmp_path / "audit"
    if where == "empty":
        audit.mkdir()
    with pytest.raises(qb.BackupError, match="audit"):
        _backup(source, tmp_path / "bk", audit)
    assert source.snapshots == {} and not (tmp_path / "bk" / "collections").exists()


def _append_point(fake, name):
    fake.points[name].append({"id": f"{name}-late", "payload": {}, "vector": {"dense": [0.0, 0.0]}})


def _edit_payload(fake, name):
    fake.points[name][0] = {**fake.points[name][0], "payload": {"deal_id": "d1", "text": "edited"}}


@pytest.mark.parametrize("write, fingerprinted", [(_append_point, False), (_edit_payload, True)])
def test_a_backup_taken_while_the_index_changed_fails(tmp_path, write, fingerprinted):
    fake = FakeQdrant({CHILD: _points("c", 30), PARENT: _points("p", 5)}, on_snapshot=write)
    with pytest.raises(qb.BackupError, match="changed while it was backed up"):
        _backup(fake, tmp_path / "bk", fingerprint=fingerprinted)
    assert fake.snapshots == {}


def test_a_changed_backup_kept_on_request_is_restored_unverified(tmp_path):
    fake = FakeQdrant({CHILD: _points("c", 30), PARENT: _points("p", 5)}, on_snapshot=_edit_payload)
    manifest = _backup(fake, tmp_path / "bk", allow=True)
    entry = manifest["collections"][CHILD]
    assert entry["changed_during_backup"] and entry["points_count"] is None and "fingerprint" not in entry
    report = _restore(FakeQdrant(), tmp_path / "bk")
    assert not report["ok"] and report["collections"][CHILD]["unverified"]


def test_verify_finds_a_changed_snapshot_a_missing_audit_file_and_a_path_outside_the_backup(source, audit_dir, tmp_path):
    out = tmp_path / "bk"
    manifest = _backup(source, out, audit_dir)
    snap = out / manifest["collections"][CHILD]["snapshot"]
    data = bytearray(snap.read_bytes())
    data[10] ^= 1
    snap.write_bytes(bytes(data))
    (out / "audit" / "audit_2026-11-23.jsonl").unlink()
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    m["audit_files"]["../../escape.jsonl"] = "0" * 64
    (out / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    problems = qb.verify(out)
    assert any("SHA-256 differs" in p for p in problems)
    assert any("audit_2026-11-23.jsonl" in p for p in problems)
    assert any("escape" in p and "not an audit" in p for p in problems)
    assert qb.main(["verify", str(out)]) == 1


def test_a_restore_into_an_empty_server_reproduces_every_point_and_the_audit_log(source, audit_dir, tmp_path):
    out = tmp_path / "bk"
    _backup(source, out, audit_dir)
    target = FakeQdrant()
    restored_audit = tmp_path / "restored_audit"
    report = _restore(target, out, restored_audit)
    assert report["ok"]
    assert all(r["fingerprint_ok"] for r in report["collections"].values())
    assert target.points == source.points
    assert sorted(p.name for p in restored_audit.iterdir()) == ["audit_2026-11-22.jsonl", "audit_2026-11-23.jsonl"]


def test_a_restore_does_not_replace_a_collection_that_holds_points(source, tmp_path):
    out = tmp_path / "bk"
    _backup(source, out)
    live = FakeQdrant({CHILD: _points("live", 5), PARENT: []})
    with pytest.raises(qb.BackupError, match="holds 5 points"):
        _restore(live, out)
    assert live.uploads == [] and len(live.points[CHILD]) == 5
    assert _restore(live, out, replace=True)["ok"]
    assert live.points == source.points


def test_nothing_is_restored_from_a_backup_that_does_not_verify(source, tmp_path):
    out = tmp_path / "bk"
    manifest = _backup(source, out)
    (out / manifest["collections"][PARENT]["snapshot"]).write_bytes(b"corrupt")
    target = FakeQdrant()
    with pytest.raises(qb.BackupError, match="does not verify"):
        _restore(target, out, replace=True)
    assert target.uploads == []


def test_a_restore_never_overwrites_an_audit_line(source, audit_dir, tmp_path):
    out = tmp_path / "bk"
    _backup(source, out, audit_dir)
    live = tmp_path / "live_audit"
    live.mkdir()
    # Appended since the backup: the live file holds every saved line and more. Kept as is.
    grown = '{"event": "document_indexed"}\n{"event": "query", "later": true}\n'
    (live / "audit_2026-11-22.jsonl").write_text(grown, encoding="utf-8")
    # Shorter or different from the saved copy: the copy goes beside it, never over it.
    (live / "audit_2026-11-23.jsonl").write_text('{"event": "other"}\n', encoding="utf-8")
    report = _restore(FakeQdrant(), out, live, replace=True)
    assert (live / "audit_2026-11-22.jsonl").read_text(encoding="utf-8") == grown
    assert (live / "audit_2026-11-23.jsonl").read_text(encoding="utf-8") == '{"event": "other"}\n'
    assert (live / "audit_2026-11-23.jsonl.restored").read_text(encoding="utf-8") == '{"event": "query"}\n'
    assert report["audit"] == {"copy": [], "keep": ["audit_2026-11-22.jsonl"],
                               "conflict": ["audit_2026-11-23.jsonl.restored"]}


def test_an_audit_problem_is_found_before_any_collection_is_replaced(source, audit_dir, tmp_path):
    out = tmp_path / "bk"
    _backup(source, out, audit_dir)
    live = tmp_path / "live_audit"
    live.mkdir()
    (live / "audit_2026-11-23.jsonl").write_text('{"event": "other"}\n', encoding="utf-8")
    (live / "audit_2026-11-23.jsonl.restored").write_text("from an earlier restore\n", encoding="utf-8")
    target = FakeQdrant()
    with pytest.raises(qb.BackupError, match="earlier restore"):
        _restore(target, out, live)
    assert target.uploads == []


def test_a_restore_that_fails_halfway_says_what_it_replaced(source, tmp_path):
    out = tmp_path / "bk"
    _backup(source, out)
    target = FakeQdrant({CHILD: _points("live", 2), PARENT: _points("livep", 2)}, fail_upload_for=PARENT)
    with pytest.raises(qb.BackupError) as failed:
        _restore(target, out, replace=True)
    message = str(failed.value)
    assert CHILD in message and "already replaced" in message and "mixes backup and live data" in message
    assert target.uploads == [CHILD]


def test_a_write_to_one_collection_during_the_others_snapshot_fails_the_backup(tmp_path):
    # A child chunk and its parent come from one upload: the two snapshots must show one state.
    def child_grows_while_parent_is_snapshotted(fake, name):
        if name == PARENT:
            _append_point(fake, CHILD)

    fake = FakeQdrant({CHILD: _points("c", 30), PARENT: _points("p", 5)},
                      on_snapshot=child_grows_while_parent_is_snapshotted)
    with pytest.raises(qb.BackupError, match=f"{CHILD} 30 -> 31 points"):
        _backup(fake, tmp_path / "bk", fingerprint=False)


def test_a_refused_upload_says_the_collection_was_not_replaced(source, tmp_path):
    out = tmp_path / "bk"
    _backup(source, out)
    target = FakeQdrant(fail_upload_for=CHILD, upload_failure="400")
    with pytest.raises(qb.BackupError) as failed:
        _restore(target, out)
    message = str(failed.value)
    assert f"{CHILD} was not replaced" in message and "no other collection was replaced" in message
    assert target.uploads == []


@pytest.mark.parametrize("failure", ["timeout", "500"])
def test_an_upload_that_may_still_be_running_is_not_reported_as_not_replaced(source, tmp_path, failure):
    # Qdrant carries on a recovery it has accepted after the request ends.
    out = tmp_path / "bk"
    _backup(source, out)
    target = FakeQdrant(fail_upload_for=PARENT, upload_failure=failure)
    with pytest.raises(qb.BackupError) as failed:
        _restore(target, out)
    message = str(failed.value)
    assert "may still be restoring" in message and "not replaced" not in message
    assert f"{CHILD} already replaced" in message


def test_a_replaced_collection_that_cannot_be_checked_is_named(source, tmp_path):
    out = tmp_path / "bk"
    _backup(source, out)
    with pytest.raises(qb.BackupError, match=f"{CHILD} was replaced but could not be checked"):
        _restore(FakeQdrant(count_fails_after_upload=True), out)


def test_restored_audit_copies_are_backed_up_too(source, audit_dir, tmp_path):
    (audit_dir / "audit_2026-11-23.jsonl.restored").write_text('{"event": "query", "from": "backup"}\n',
                                                               encoding="utf-8")
    manifest = _backup(source, tmp_path / "bk", audit_dir)
    assert "audit_2026-11-23.jsonl.restored" in manifest["audit_files"]
    assert qb.verify(tmp_path / "bk") == []


def test_a_server_snapshot_that_cannot_be_deleted_is_reported(capsys, tmp_path):
    fake = FakeQdrant({CHILD: _points("c", 3), PARENT: _points("p", 1)}, delete_status=503)
    manifest = _backup(fake, tmp_path / "bk")
    assert set(manifest["collections"]) == {CHILD, PARENT}
    assert capsys.readouterr().err.count("could not delete server snapshot") == 2


def test_a_fingerprint_does_not_depend_on_page_size(source):
    with source.client() as c:
        assert qb.fingerprint(c, CHILD, page=7) == qb.fingerprint(c, CHILD, page=256)
