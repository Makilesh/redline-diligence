"""
Fail on a published advisory against a pinned dependency unless it was accepted, with a reason.

    python scripts/check_vulnerabilities.py --write-audit-input audit-input.txt
    pip-audit -r audit-input.txt --no-deps --disable-pip -s osv --format json -o audit.json
    python scripts/check_vulnerabilities.py --audit-json audit.json --requirements audit-input.txt

The audit input is every pin the images and CI install: constraints.txt
(scripts/update_constraints.py), plus torch, which they install separately at Dockerfile.hf's
TORCH_VERSION. Pinning keeps builds reproducible and freezes the versions: advisories keep
being published against them and nothing looked (on 2026-11-22, ~135 against 13 pinned
packages, all in the API image; Decision 104). The Dependency audit workflow runs this weekly
and on every change to the pins.

security/accepted-vulnerabilities.toml lists the advisories accepted for the version they
were accepted at, each with why and a date by which to look again. Exit status:
  0  every pin was audited and every advisory reported is accepted;
  1  an advisory is not accepted, an acceptance is past its review date, the file is invalid,
     or a pin is missing from the audit or was skipped by it;
  2  the audit output is missing or unreadable (the audit itself did not run).
Accepted advisories no longer reported (fixed or withdrawn) are listed so the file can be pruned.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = ROOT / "security" / "accepted-vulnerabilities.toml"
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)")
# An acceptance is a decision to look again, not to stop looking.
MAX_REVIEW_HORIZON_DAYS = 200


def _canon(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def read_pins(text: str) -> dict[str, str]:
    """name -> version for every `name==version` line."""
    pins = {}
    for line in text.splitlines():
        m = PIN.match(line.strip())
        if m:
            pins[_canon(m.group(1))] = m.group(2)
    return pins


def audit_input(root: Path = ROOT) -> str:
    """constraints.txt's pins plus torch at Dockerfile.hf's TORCH_VERSION, one per line."""
    pins = read_pins((root / "constraints.txt").read_text(encoding="utf-8"))
    m = re.search(r"^ARG TORCH_VERSION=(\S+)", (root / "Dockerfile.hf").read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit("Dockerfile.hf has no ARG TORCH_VERSION: torch would not be audited")
    pins["torch"] = m.group(1)
    return "".join(f"{name}=={version}\n" for name, version in sorted(pins.items()))


def coverage_problems(audit: dict, pins: dict[str, str]) -> list[str]:
    """Pins the audit did not report on, or skipped (no audit means nothing is known)."""
    seen = {}
    for dep in audit.get("dependencies", []):
        seen[_canon(dep["name"])] = dep
    problems = []
    for name, version in sorted(pins.items()):
        dep = seen.get(name)
        if dep is None:
            problems.append(f"{name}=={version} is not in the audit output")
        elif dep.get("skip_reason"):
            problems.append(f"{name}=={version} was skipped by the audit: {dep['skip_reason']}")
        elif dep.get("version") != version:
            problems.append(f"{name}: the audit checked {dep.get('version')}, the pin is {version}")
    return problems


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    id: str
    aliases: tuple[str, ...]
    fix_versions: tuple[str, ...]
    summary: str

    @property
    def ids(self) -> set[str]:
        return {self.id, *self.aliases}


@dataclass
class Acceptance:
    package: str
    version: str
    ids: set[str]
    reason: str
    review_by: dt.date


@dataclass
class Report:
    unaccepted: list[Finding] = field(default_factory=list)
    expired: list[Acceptance] = field(default_factory=list)
    stale: list[tuple[Acceptance, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    accepted: int = 0

    @property
    def ok(self) -> bool:
        return not (self.unaccepted or self.expired or self.errors)


def load_findings(audit: dict) -> list[Finding]:
    """One Finding per advisory and pinned version. OSV can list one advisory twice (one
    record per affected range), with different fixed versions: those are merged."""
    merged: dict[tuple[str, str, str], Finding] = {}
    for dep in audit.get("dependencies", []):
        for v in dep.get("vulns") or []:
            key = (_canon(dep["name"]), dep["version"], v["id"])
            prior = merged.get(key)
            aliases = tuple(dict.fromkeys([*(prior.aliases if prior else ()), *(v.get("aliases") or ())]))
            fixes = tuple(dict.fromkeys([*(prior.fix_versions if prior else ()), *(v.get("fix_versions") or ())]))
            merged[key] = Finding(
                package=key[0], version=key[1], id=key[2], aliases=aliases, fix_versions=fixes,
                summary=prior.summary if prior else " ".join((v.get("description") or "").split())[:160],
            )
    return list(merged.values())


def load_acceptances(data: dict, today: dt.date) -> tuple[list[Acceptance], list[str]]:
    out, errors = [], []
    for i, entry in enumerate(data.get("accepted", [])):
        where = f"accepted[{i}] ({entry.get('package', '?')})"
        missing = [k for k in ("package", "version", "ids", "reason", "review_by") if k not in entry]
        if missing:
            errors.append(f"{where}: missing {', '.join(missing)}")
            continue
        if not isinstance(entry["review_by"], dt.date):
            errors.append(f"{where}: review_by must be a date (YYYY-MM-DD), got {entry['review_by']!r}")
            continue
        if not entry["ids"] or not all(isinstance(x, str) for x in entry["ids"]):
            errors.append(f"{where}: ids must be a non-empty list of advisory ids")
            continue
        if not str(entry["reason"]).strip():
            errors.append(f"{where}: reason is empty")
            continue
        if (entry["review_by"] - today).days > MAX_REVIEW_HORIZON_DAYS:
            errors.append(f"{where}: review_by {entry['review_by']} is more than "
                          f"{MAX_REVIEW_HORIZON_DAYS} days away")
            continue
        out.append(Acceptance(_canon(entry["package"]), str(entry["version"]), set(entry["ids"]),
                              str(entry["reason"]).strip(), entry["review_by"]))
    return out, errors


def evaluate(audit: dict, accepted_data: dict, today: dt.date) -> Report:
    report = Report()
    findings = load_findings(audit)
    acceptances, report.errors = load_acceptances(accepted_data, today)
    for f in findings:
        # Accepted for this package at this version only: an upgrade retires the entry,
        # and an advisory still open on the new version must be looked at again.
        if any(a.package == f.package and a.version == f.version and a.ids & f.ids for a in acceptances):
            report.accepted += 1
        else:
            report.unaccepted.append(f)
    for a in acceptances:
        if a.review_by < today:
            report.expired.append(a)
        reported = set().union(*(f.ids for f in findings if f.package == a.package and f.version == a.version))
        report.stale.extend((a, x) for x in sorted(a.ids - reported))
    return report


def _print(report: Report, annotate: bool) -> None:
    def err(title: str, text: str) -> None:
        print(f"::error title={title}::{text}" if annotate else f"ERROR {title}: {text}")

    for e in report.errors:
        err("accepted-vulnerabilities.toml", e)
    for f in report.unaccepted:
        fix = ", ".join(f.fix_versions) or "no fixed version published"
        alias = f" ({', '.join(f.aliases)})" if f.aliases else ""
        err(f"{f.package} {f.version}", f"{f.id}{alias} not accepted; fixed in: {fix}. {f.summary}")
    for a in report.expired:
        err(f"{a.package} {a.version}", f"acceptance of {', '.join(sorted(a.ids))} was due for review by {a.review_by}")
    for a, x in report.stale:
        print(f"note: {a.package} {a.version}: {x} is accepted but no longer reported; remove it")
    print(f"{report.accepted} advisories reported and accepted; {len(report.unaccepted)} not accepted; "
          f"{len(report.expired)} acceptances past review; {len(report.errors)} file errors")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write-audit-input", type=Path,
                        help="write the pins to audit (constraints.txt plus torch) to this file and exit")
    parser.add_argument("--audit-json", type=Path, help="pip-audit --format json output")
    parser.add_argument("--requirements", type=Path,
                        help="the audit input: every pin in it must have been audited")
    parser.add_argument("--accepted", type=Path, default=ACCEPTED)
    parser.add_argument("--today", type=dt.date.fromisoformat, default=None, help="for tests")
    args = parser.parse_args(argv)
    if args.write_audit_input:
        args.write_audit_input.write_text(audit_input(), encoding="utf-8")
        return 0
    if args.audit_json is None:
        parser.error("--audit-json is required")
    try:
        audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
        if not isinstance(audit, dict) or "dependencies" not in audit:
            raise ValueError("no 'dependencies' key")
    except (OSError, ValueError) as e:
        print(f"ERROR: the audit output {args.audit_json} is missing or unreadable ({e}); "
              "the audit did not run, so nothing is known about the pins")
        return 2
    accepted = tomllib.loads(args.accepted.read_text(encoding="utf-8")) if args.accepted.exists() else {}
    report = evaluate(audit, accepted, args.today or dt.date.today())
    if args.requirements is not None:
        report.errors += coverage_problems(audit, read_pins(args.requirements.read_text(encoding="utf-8")))
    _print(report, annotate=os.environ.get("GITHUB_ACTIONS") == "true")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
