"""
Back up and restore the index (both Qdrant collections) and the audit log, and check a backup.

    python scripts/qdrant_backup.py backup  --out BACKUP_DIR [--audit-dir DIR] [--fingerprint]
    python scripts/qdrant_backup.py verify  BACKUP_DIR
    python scripts/qdrant_backup.py restore BACKUP_DIR [--audit-dir DIR] [--replace-existing]

The server is QDRANT_URL (default http://localhost:6333) with QDRANT_API_KEY, as for the API;
--url and --api-key override them. In the production compose file Qdrant is reachable only
inside the compose network, so run it in the api container and copy the result out:

    docker compose -f docker-compose.prod.yml exec api python scripts/qdrant_backup.py \
        backup --out /tmp/backup --audit-dir /home/user/audit --fingerprint
    docker compose -f docker-compose.prod.yml cp api:/tmp/backup ./backups/2026-11-23
    docker compose -f docker-compose.prod.yml exec api rm -rf /tmp/backup

What is persistent is the index and the audit log (AUDIT_LOG_DIR). Postgres holds LangGraph
checkpoints only, and every run clears its thread first; an uploaded file is deleted once it
is indexed. An embedded index (QDRANT_LOCAL_PATH) has no snapshot API: stop the API and copy
its directory.

backup takes one snapshot per collection, downloads it with its SHA-256, deletes it from the
server (unless --keep-server-snapshots) and writes manifest.json. --fingerprint also hashes
every point (id, payload, vectors) so a restore can be compared point for point.
verify re-hashes every file against the manifest. restore verifies first, refuses to replace a
collection that holds points unless --replace-existing, uploads each snapshot, and checks the
point counts (and the fingerprints, when the backup has them).

Where backups are kept, whether they are encrypted, and for how long are the owner's decisions
(DEPLOYMENT.md, "Backups"); this script writes plain files to the directory it is given.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
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


class BackupError(Exception):
    pass


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
        raise BackupError(f"{what}: HTTP {response.status_code} {response.text[:300]}")
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


def backup(client: httpx.Client, out: Path, *, audit_dir: Path | None, keep_server: bool,
           with_fingerprint: bool) -> dict:
    if out.exists() and any(out.iterdir()):
        raise BackupError(f"{out} is not empty; give each backup its own directory")
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
    for name in COLLECTIONS:
        before = _count(client, name)
        if before is None:
            raise BackupError(f"collection {name} does not exist on the server: nothing to back up")
        snap = _result(client.post(f"/collections/{quote(name)}/snapshots", params={"wait": "true"}),
                       f"snapshot {name}")
        snap_name = snap["name"]
        target = out / "collections" / name / snap_name
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
        after = _count(client, name)
        entry = {
            "snapshot": target.relative_to(out).as_posix(),
            "bytes": size,
            "sha256": digest,
            "points_count": before,
            "points_count_after": after,
            "changed_during_backup": before != after,
        }
        if with_fingerprint:
            entry["fingerprint"] = fingerprint(client, name)
        if not keep_server:
            _result(client.delete(f"/collections/{quote(name)}/snapshots/{quote(snap_name)}",
                                  params={"wait": "true"}), f"delete server snapshot {snap_name}")
        manifest["collections"][name] = entry
    if audit_dir is not None:
        files = sorted(audit_dir.glob("audit_*.jsonl"))
        (out / "audit").mkdir(exist_ok=True)
        for f in files:
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
        path = backup_dir / entry["snapshot"]
        if not path.exists():
            problems.append(f"{entry['snapshot']} is missing")
        elif _sha256(path) != entry["sha256"]:
            problems.append(f"{entry['snapshot']}: SHA-256 differs from the manifest")
    for fname, digest in manifest.get("audit_files", {}).items():
        path = backup_dir / "audit" / fname
        if not path.exists() or _sha256(path) != digest:
            problems.append(f"audit/{fname} is missing or differs from the manifest")
    return problems


def restore(client: httpx.Client, backup_dir: Path, *, audit_dir: Path | None, replace: bool,
            timeout: float) -> dict:
    problems = verify(backup_dir)
    if problems:
        raise BackupError("the backup does not verify, nothing restored: " + "; ".join(problems))
    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    server = client.get("/").json().get("version")
    report = {"qdrant_version": server, "backup_qdrant_version": manifest.get("qdrant_version"), "collections": {}}
    for name in COLLECTIONS:
        existing = _count(client, name)
        if existing and not replace:
            raise BackupError(f"collection {name} holds {existing} points; "
                              "pass --replace-existing to replace it")
    for name in COLLECTIONS:
        entry = manifest["collections"][name]
        path = backup_dir / entry["snapshot"]
        with open(path, "rb") as f:
            # checksum: Qdrant refuses a file whose SHA-256 differs before it replaces anything.
            r = client.post(f"/collections/{quote(name)}/snapshots/upload",
                            params={"wait": "true", "priority": "snapshot", "checksum": entry["sha256"]},
                            files={"snapshot": (path.name, f, "application/octet-stream")},
                            timeout=timeout)
        _result(r, f"upload snapshot of {name}")
        count = _count(client, name)
        result = {"points_count": count, "expected": entry["points_count_after"], "ok": count == entry["points_count_after"]}
        if "fingerprint" in entry:
            got = fingerprint(client, name)
            result["fingerprint_ok"] = got == entry["fingerprint"]
            result["ok"] = result["ok"] and result["fingerprint_ok"]
        report["collections"][name] = result
    if audit_dir is not None and manifest.get("audit_files"):
        audit_dir.mkdir(parents=True, exist_ok=True)
        for fname in manifest["audit_files"]:
            src, dest = backup_dir / "audit" / fname, audit_dir / fname
            if dest.exists() and _sha256(dest) != manifest["audit_files"][fname] and not replace:
                raise BackupError(f"{dest} exists with other content; pass --replace-existing to replace it")
            shutil.copy2(src, dest)
        report["audit_files"] = len(manifest["audit_files"])
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
                                  keep_server=args.keep_server_snapshots, with_fingerprint=args.fingerprint)
                print(json.dumps(manifest, indent=2))
                changed = [n for n, c in manifest["collections"].items() if c["changed_during_backup"]]
                if changed:
                    print(f"warning: {', '.join(changed)} changed while the backup ran; "
                          "the snapshots are consistent each, but not with each other", file=sys.stderr)
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
