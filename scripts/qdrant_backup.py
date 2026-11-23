"""
Back up and restore the index (both Qdrant collections) and the audit log, and check a backup.

    python scripts/qdrant_backup.py backup  --out BACKUP_DIR [--audit-dir DIR] [--fingerprint] [--allow-changes]
    python scripts/qdrant_backup.py verify  BACKUP_DIR
    python scripts/qdrant_backup.py restore BACKUP_DIR [--audit-dir DIR] [--replace-existing]

The server is QDRANT_URL (default http://localhost:6333) with QDRANT_API_KEY, as for the API;
Its command-line URL and key options take precedence. In production Qdrant is reachable only inside the compose
network, so the script runs in a one-off container of the API image (DEPLOYMENT.md, "Backups"):

    C="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
    mkdir -p backups && sudo chown 1000:1000 backups        # the image runs as uid 1000
    $C stop api                                             # nothing writes while it runs
    $C run --rm --no-deps -v "$PWD/backups:/backups" api python scripts/qdrant_backup.py \
        backup --out /backups/$(date +%F) --audit-dir /home/user/audit --fingerprint
    $C up -d api

What is persistent is the index and the audit log (AUDIT_LOG_DIR). Postgres holds LangGraph
checkpoints (each run clears its own) and the day's quota counters (api_budget): neither is
backed up, and losing the counters at most lets the app try requests the provider then refuses
for the rest of that day. An uploaded file is deleted once it is indexed. An embedded index
(QDRANT_LOCAL_PATH) has no snapshot API: stop the API and copy its directory.

backup records each collection's exact point count (and, with --fingerprint, a hash of every
point) before and after its snapshot, and fails if they differ: a backup taken while the index
changed matches no state that can be checked after a restore. --allow-changes keeps it anyway,
unverifiable. Each snapshot is downloaded with its SHA-256 and deleted from the server, which
keeps snapshots off its disk (/qdrant/snapshots is not on a volume), even when the backup fails.

verify re-hashes every file against manifest.json. restore verifies first and plans the audit
files before writing anything; it refuses to replace a collection that holds points unless
--replace-existing, uploads each snapshot with its checksum (Qdrant refuses a damaged file
before replacing anything), and checks the point counts and fingerprints. Run it with the API
stopped. An audit file is never overwritten: the log is append-only. A live file that already
contains the backup's copy is kept; any other difference is written beside it as *.restored and
reported.

Where backups are kept, whether they are encrypted, and for how long are the owner's decisions
(DEPLOYMENT.md, "Backups"); this script writes plain files to the directory it is given.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME  # noqa: E402

FORMAT = 1
COLLECTIONS = (COLLECTION_NAME, PARENT_COLLECTION_NAME)
CHUNK = 1 << 20
# Audit files, and the copies a restore wrote beside a live file it would not overwrite.
AUDIT_NAME = re.compile(r"^audit_[A-Za-z0-9_.\-]+\.jsonl(\.restored)*$")


class BackupError(Exception):
    pass


class ServerRefused(BackupError):
    """Qdrant answered with an error status."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _client(url: str, api_key: str | None, timeout: float) -> httpx.Client:
    headers = {"api-key": api_key} if api_key else {}
    return httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=timeout)


def _result(response: httpx.Response, what: str):
    if response.status_code >= 400:
        raise ServerRefused(f"{what}: HTTP {response.status_code} {response.text[:300]}", response.status_code)
    return response.json().get("result")


def _count(client: httpx.Client, name: str) -> int | None:
    """Exact point count, or None when the collection does not exist. (The collection info's
    points_count is documented as approximate.)"""
    r = client.post(f"/collections/{quote(name)}/points/count", json={"exact": True})
    if r.status_code == 404:
        return None
    return _result(r, f"count {name}")["count"]


def fingerprint(client: httpx.Client, name: str, page: int = 256) -> dict:
    """
    SHA-256 over every point (id, payload and vectors, canonically encoded), page by page.

    Scroll returns points in id order, so the same points give the same digest on any
    server, without holding the collection in memory.
    """
    h = hashlib.sha256()
    count = 0
    offset = None
    while True:
        body = {"limit": page, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        result = _result(client.post(f"/collections/{quote(name)}/points/scroll", json=body),
                         f"scroll {name}")
        for p in result["points"]:
            h.update(json.dumps([str(p["id"]), p.get("payload"), p.get("vector")],
                                sort_keys=True, separators=(",", ":")).encode())
            count += 1
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return {"points": count, "sha256": h.hexdigest()}


def _git_commit() -> str | None:
    if os.getenv("APP_GIT_COMMIT"):
        return os.environ["APP_GIT_COMMIT"]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _audit_files(audit_dir: Path) -> list[Path]:
    if not audit_dir.is_dir():
        raise BackupError(f"--audit-dir {audit_dir} is not a directory")
    files = sorted(p for p in audit_dir.iterdir() if p.is_file() and AUDIT_NAME.match(p.name))
    if not files:
        raise BackupError(f"--audit-dir {audit_dir} holds no audit_*.jsonl file; is it AUDIT_LOG_DIR?")
    return files


def _snapshot_collection(client: httpx.Client, name: str, out: Path, keep_server: bool) -> dict:
    snap = _result(client.post(f"/collections/{quote(name)}/snapshots", params={"wait": "true"}),
                   f"snapshot {name}")
    snap_name = snap["name"]
    target = out / "collections" / name / snap_name
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        size = 0
        with client.stream("GET", f"/collections/{quote(name)}/snapshots/{quote(snap_name)}") as r:
            if r.status_code >= 400:
                raise BackupError(f"download snapshot {snap_name}: HTTP {r.status_code}")
            with open(target, "wb") as f:
                for block in r.iter_bytes(CHUNK):
                    f.write(block)
                    h.update(block)
                    size += len(block)
        digest = h.hexdigest()
        if snap.get("checksum") and snap["checksum"] != digest:
            raise BackupError(f"{snap_name}: downloaded SHA-256 {digest} differs from the server's {snap['checksum']}")
    finally:
        if not keep_server:
            # Also when the download failed: the server's snapshot directory is not on a
            # volume, and a scheduled backup that keeps failing would fill the disk.
            try:
                r = client.delete(f"/collections/{quote(name)}/snapshots/{quote(snap_name)}", params={"wait": "true"})
                if r.status_code >= 300:
                    print(f"warning: could not delete server snapshot {snap_name}: HTTP {r.status_code}",
                          file=sys.stderr)
            except httpx.HTTPError as e:
                print(f"warning: could not delete server snapshot {snap_name}: {e}", file=sys.stderr)
    return {"snapshot": target.relative_to(out).as_posix(), "bytes": size, "sha256": digest}


def _state(client: httpx.Client, name: str, with_fingerprint: bool) -> dict:
    count = _count(client, name)
    if count is None:
        raise BackupError(f"collection {name} does not exist on the server: nothing to back up")
    return {"count": count, "fingerprint": fingerprint(client, name) if with_fingerprint else None}


def backup(client: httpx.Client, out: Path, *, audit_dir: Path | None, keep_server: bool,
           with_fingerprint: bool, allow_changes: bool = False) -> dict:
    if out.exists() and any(out.iterdir()):
        raise BackupError(f"{out} is not empty; give each backup its own directory")
    audit_sources = _audit_files(audit_dir) if audit_dir is not None else []
    out.mkdir(parents=True, exist_ok=True)
    server = client.get("/").json()
    manifest: dict = {
        "format": FORMAT,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "qdrant_version": server.get("version"),
        "app_commit": _git_commit(),
        "collections": {},
        "audit_files": {},
    }
    # Both collections are read before either snapshot and after both: a child chunk and its
    # parent are written by one upload, so the two snapshots must come from one state.
    before = {name: _state(client, name, with_fingerprint) for name in COLLECTIONS}
    entries = {name: _snapshot_collection(client, name, out, keep_server) for name in COLLECTIONS}
    after = {name: _state(client, name, with_fingerprint) for name in COLLECTIONS}
    changed = [name for name in COLLECTIONS if before[name] != after[name]]
    if changed and not allow_changes:
        moved = ", ".join(f"{n} {before[n]['count']} -> {after[n]['count']} points" for n in changed)
        raise BackupError(
            f"the index changed while it was backed up ({moved}): stop the API (or every upload) "
            "and run the backup again, or pass --allow-changes to keep a backup no restore can be "
            "checked against")
    for name in COLLECTIONS:
        entry = entries[name]
        entry.update({
            # One collection changing makes both unverifiable: they no longer show one state.
            "points_count": None if changed else after[name]["count"],
            "changed_during_backup": bool(changed),
        })
        if with_fingerprint and not changed:
            entry["fingerprint"] = after[name]["fingerprint"]
        manifest["collections"][name] = entry
    if audit_sources:
        (out / "audit").mkdir(exist_ok=True)
        for f in audit_sources:
            shutil.copy2(f, out / "audit" / f.name)
            manifest["audit_files"][f.name] = _sha256(out / "audit" / f.name)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify(backup_dir: Path) -> list[str]:
    """Problems found, or an empty list."""
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"{manifest_path} is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    if manifest.get("format") != FORMAT:
        problems.append(f"manifest format {manifest.get('format')} is not {FORMAT}")
    for name in COLLECTIONS:
        entry = manifest.get("collections", {}).get(name)
        if entry is None:
            problems.append(f"collection {name} is not in the backup")
            continue
        rel = Path(entry["snapshot"])
        if rel.is_absolute() or ".." in rel.parts:
            problems.append(f"{entry['snapshot']}: not a path inside the backup")
            continue
        path = backup_dir / rel
        if not path.exists():
            problems.append(f"{entry['snapshot']} is missing")
        elif _sha256(path) != entry["sha256"]:
            problems.append(f"{entry['snapshot']}: SHA-256 differs from the manifest")
    for fname, digest in manifest.get("audit_files", {}).items():
        if not AUDIT_NAME.match(fname):
            problems.append(f"audit file name {fname!r} is not an audit_*.jsonl name")
            continue
        path = backup_dir / "audit" / fname
        if not path.exists() or _sha256(path) != digest:
            problems.append(f"audit/{fname} is missing or differs from the manifest")
    return problems


def _plan_audit(backup_dir: Path, manifest: dict, audit_dir: Path) -> list[tuple[str, Path, Path]]:
    """(action, source, destination) per audit file; nothing is written here."""
    plan = []
    for fname in manifest.get("audit_files", {}):
        src, dest = backup_dir / "audit" / fname, audit_dir / fname
        if not dest.exists():
            plan.append(("copy", src, dest))
            continue
        live, saved = dest.read_bytes(), src.read_bytes()
        if live.startswith(saved):
            plan.append(("keep", src, dest))  # the live log already holds every saved line
            continue
        beside = dest.with_name(dest.name + ".restored")
        if beside.exists():
            raise BackupError(f"{beside} exists from an earlier restore; move it away first")
        plan.append(("conflict", src, beside))
    return plan


def _unknown_state(name: str, error: Exception, before: str) -> str:
    # A recovery Qdrant has accepted carries on server-side after the request ends, with the
    # collection's shards offline: neither "replaced" nor "not replaced" can be said here.
    return (f"restoring {name} did not finish as far as this script can tell ({error}). Qdrant may still "
            f"be restoring it; {before}. Wait until {name} answers and its point count stops changing, "
            "then run the restore again with --replace-existing (with the API still stopped).")


def restore(client: httpx.Client, backup_dir: Path, *, audit_dir: Path | None, replace: bool,
            timeout: float) -> dict:
    problems = verify(backup_dir)
    if problems:
        raise BackupError("the backup does not verify, nothing restored: " + "; ".join(problems))
    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    plan = _plan_audit(backup_dir, manifest, audit_dir) if audit_dir is not None else []
    server = client.get("/").json().get("version")
    report: dict = {"qdrant_version": server, "backup_qdrant_version": manifest.get("qdrant_version"),
                    "collections": {}}
    for name in COLLECTIONS:
        existing = _count(client, name)
        if existing and not replace:
            raise BackupError(f"collection {name} holds {existing} points; "
                              "pass --replace-existing to replace it (with the API stopped)")
    replaced: list[str] = []
    for name in COLLECTIONS:
        entry = manifest["collections"][name]
        path = backup_dir / entry["snapshot"]
        before = (f"{', '.join(replaced)} already replaced, so the index now mixes backup and live data"
                  if replaced else "no other collection was replaced")
        try:
            with open(path, "rb") as f:
                # checksum: Qdrant refuses a file whose SHA-256 differs before it replaces anything.
                r = client.post(f"/collections/{quote(name)}/snapshots/upload",
                                params={"wait": "true", "priority": "snapshot", "checksum": entry["sha256"]},
                                files={"snapshot": (path.name, f, "application/octet-stream")},
                                timeout=timeout)
            _result(r, f"upload snapshot of {name}")
        except ServerRefused as e:
            if e.status < 500:  # refused up front (a bad checksum, a bad request): not replaced
                raise BackupError(f"Qdrant refused the snapshot of {name} ({e}): {name} was not "
                                  f"replaced; {before}. Fix the cause and run the restore again "
                                  "with --replace-existing.") from e
            raise BackupError(_unknown_state(name, e, before)) from e
        except httpx.HTTPError as e:
            raise BackupError(_unknown_state(name, e, before)) from e
        replaced.append(name)
        try:
            count = _count(client, name)
            expected = entry.get("points_count")
            result = {"points_count": count, "expected": expected,
                      "ok": expected is not None and count == expected}
            if expected is None:
                result["unverified"] = "the collection changed while it was backed up"
            if "fingerprint" in entry:
                got = fingerprint(client, name)
                result["fingerprint_ok"] = got == entry["fingerprint"]
                result["ok"] = result["ok"] and result["fingerprint_ok"]
        except (BackupError, httpx.HTTPError) as e:
            raise BackupError(f"{name} was replaced but could not be checked ({e}); replaced so far: "
                              f"{', '.join(replaced)}. Check the collections before starting the API.") from e
        report["collections"][name] = result
        print(f"restored {name}: {count} points ({'checked' if result['ok'] else 'NOT matching the backup'})",
              file=sys.stderr)
    if plan:
        audit_dir.mkdir(parents=True, exist_ok=True)
        for action, src, dest in plan:
            if action in ("copy", "conflict"):
                if dest.exists():
                    raise BackupError(f"{dest} exists; move it away and run the restore again")
                shutil.copy2(src, dest)
        report["audit"] = {a: sorted(d.name for act, _, d in plan if act == a) for a in ("copy", "keep", "conflict")}
        if report["audit"]["conflict"]:
            print("warning: audit files differ from the backup's copies; each copy was written beside the "
                  "live file as *.restored and nothing was overwritten: " + ", ".join(report["audit"]["conflict"]),
                  file=sys.stderr)
    report["ok"] = all(c["ok"] for c in report["collections"].values())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY") or None)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per request")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backup")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--audit-dir", type=Path)
    b.add_argument("--keep-server-snapshots", action="store_true")
    b.add_argument("--fingerprint", action="store_true")
    b.add_argument("--allow-changes", action="store_true",
                   help="keep a backup taken while the index changed (no restore can be checked against it)")
    v = sub.add_parser("verify")
    v.add_argument("backup_dir", type=Path)
    r = sub.add_parser("restore")
    r.add_argument("backup_dir", type=Path)
    r.add_argument("--audit-dir", type=Path)
    r.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            problems = verify(args.backup_dir)
            print("\n".join(problems) if problems else f"{args.backup_dir}: every file matches the manifest")
            return 1 if problems else 0
        with _client(args.url, args.api_key, args.timeout) as client:
            if args.command == "backup":
                manifest = backup(client, args.out, audit_dir=args.audit_dir,
                                  keep_server=args.keep_server_snapshots, with_fingerprint=args.fingerprint,
                                  allow_changes=args.allow_changes)
                print(json.dumps(manifest, indent=2))
                return 0
            report = restore(client, args.backup_dir, audit_dir=args.audit_dir,
                             replace=args.replace_existing, timeout=args.timeout)
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
    except (BackupError, httpx.HTTPError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
