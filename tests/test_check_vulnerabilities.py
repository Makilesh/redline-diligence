"""
scripts/check_vulnerabilities.py: an advisory against a pin fails the audit unless accepted,
for that version, with a reason and a review date (Decision 104).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_vulnerabilities as cv  # noqa: E402

TODAY = dt.date(2026, 11, 23)


def _audit(*deps):
    return {"dependencies": [{"name": n, "version": v, "vulns": vulns} for n, v, vulns in deps], "fixes": []}


def _vuln(id_, aliases=(), fixes=("9.9",), desc="something bad"):
    return {"id": id_, "aliases": list(aliases), "fix_versions": list(fixes), "description": desc}


def _accept(package, version, ids, review_by=dt.date(2027, 2, 28), reason="not reachable"):
    return {"package": package, "version": version, "ids": list(ids), "reason": reason, "review_by": review_by}


def test_an_advisory_nobody_accepted_fails():
    report = cv.evaluate(_audit(("aiohttp", "3.14.1", [_vuln("PYSEC-1", ["CVE-1"])])), {}, TODAY)
    assert not report.ok
    assert [f.id for f in report.unaccepted] == ["PYSEC-1"]


def test_an_accepted_advisory_passes_by_any_of_its_ids():
    audit = _audit(("transformers", "4.57.6", [_vuln("PYSEC-2", ["CVE-2", "GHSA-2"])]))
    for listed in (["PYSEC-2"], ["CVE-2"], ["GHSA-2"]):
        report = cv.evaluate(audit, {"accepted": [_accept("transformers", "4.57.6", listed)]}, TODAY)
        assert report.ok and report.accepted == 1, listed


def test_an_acceptance_holds_for_its_version_only():
    audit = _audit(("transformers", "4.58.0", [_vuln("PYSEC-2")]))
    report = cv.evaluate(audit, {"accepted": [_accept("transformers", "4.57.6", ["PYSEC-2"])]}, TODAY)
    assert not report.ok and [f.version for f in report.unaccepted] == ["4.58.0"]
    assert [(a.version, x) for a, x in report.stale] == [("4.57.6", "PYSEC-2")]


def test_package_names_match_however_they_are_spelled():
    audit = _audit(("Sentence_Transformers", "2.7.0", [_vuln("PYSEC-3")]))
    report = cv.evaluate(audit, {"accepted": [_accept("sentence-transformers", "2.7.0", ["PYSEC-3"])]}, TODAY)
    assert report.ok


def test_an_acceptance_past_its_review_date_fails():
    audit = _audit(("litellm", "1.89.7", [_vuln("GHSA-4")]))
    accepted = {"accepted": [_accept("litellm", "1.89.7", ["GHSA-4"], review_by=dt.date(2026, 11, 22))]}
    report = cv.evaluate(audit, accepted, TODAY)
    assert not report.ok and report.expired and not report.unaccepted


def test_a_fixed_advisory_is_reported_as_stale_without_failing():
    report = cv.evaluate(_audit(("litellm", "1.89.7", [])), {"accepted": [_accept("litellm", "1.89.7", ["GHSA-4"])]}, TODAY)
    assert report.ok and [x for _, x in report.stale] == ["GHSA-4"]


def test_aliases_of_a_reported_advisory_are_not_stale():
    audit = _audit(("transformers", "4.57.6", [_vuln("PYSEC-2", ["CVE-2"])]))
    report = cv.evaluate(audit, {"accepted": [_accept("transformers", "4.57.6", ["PYSEC-2", "CVE-2"])]}, TODAY)
    assert report.ok and report.stale == []


def test_duplicate_records_of_one_advisory_count_once():
    audit = _audit(("transformers", "4.57.6", [_vuln("PYSEC-2", ["CVE-2"], ["5.0.0rc3"]),
                                               _vuln("PYSEC-2", ["CVE-2"], ["5.0.0"])]))
    [finding] = cv.load_findings(audit)
    assert finding.fix_versions == ("5.0.0rc3", "5.0.0")


@pytest.mark.parametrize("entry, message", [
    ({"package": "x", "version": "1", "ids": ["A"], "review_by": dt.date(2027, 1, 1)}, "missing reason"),
    ({**_accept("x", "1", ["A"]), "reason": "  "}, "reason is empty"),
    ({**_accept("x", "1", ["A"]), "ids": []}, "ids must be"),
    ({**_accept("x", "1", ["A"]), "review_by": "2027-01-01"}, "must be a date"),
    ({**_accept("x", "1", ["A"]), "review_by": dt.date(2028, 1, 1)}, "more than"),
])
def test_an_invalid_acceptance_fails(entry, message):
    report = cv.evaluate(_audit(), {"accepted": [entry]}, TODAY)
    assert not report.ok and any(message in e for e in report.errors), report.errors


def test_exit_codes(tmp_path, capsys):
    accepted = tmp_path / "accepted.toml"
    accepted.write_text('[[accepted]]\npackage = "a"\nversion = "1"\nids = ["X"]\nreason = "r"\n'
                        "review_by = 2027-01-01\n", encoding="utf-8")
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps(_audit(("a", "1", [_vuln("X")]))), encoding="utf-8")
    dirty = tmp_path / "dirty.json"
    dirty.write_text(json.dumps(_audit(("a", "1", [_vuln("X"), _vuln("Y")]))), encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    args = ["--accepted", str(accepted), "--today", "2026-11-23"]
    assert cv.main(["--audit-json", str(clean), *args]) == 0
    assert cv.main(["--audit-json", str(dirty), *args]) == 1
    assert cv.main(["--audit-json", str(broken), *args]) == 2
    assert cv.main(["--audit-json", str(tmp_path / "absent.json"), *args]) == 2
    assert "the audit did not run" in capsys.readouterr().out


def test_the_committed_acceptances_are_valid_and_name_pinned_versions():
    data = tomllib.loads((ROOT / "security" / "accepted-vulnerabilities.toml").read_text(encoding="utf-8"))
    entries = data["accepted"]
    # Validity, not expiry: the weekly audit is what fails on an overdue review.
    day_before_first_review = min(e["review_by"] for e in entries) - dt.timedelta(days=1)
    acceptances, errors = cv.load_acceptances(data, day_before_first_review)
    assert errors == [] and len(acceptances) == len(entries)
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+)$", (ROOT / "constraints.txt").read_text(encoding="utf-8"), re.M))
    pins = {cv._canon(k): v for k, v in pins.items()}
    for a in acceptances:
        # An upgrade must retire or re-review the entry: one for a version no longer pinned is dead.
        assert pins.get(a.package) == a.version, f"{a.package} {a.version} is accepted but {pins.get(a.package)} is pinned"
