"""
The grounding-on-tables harness (eval/run_grounding_eval.py, Decision 94) and
the grounding set's builder (eval/figure_veto_set.py --kind grounding and
grounding-cds). No model is loaded and no FinanceBench data is read. Nothing
here depends on how grounding reads converted tables: the harness measures
main's grounding as well as a branch's.
"""

from collections import Counter

import pytest

from eval import figure_veto_set as fvs
from eval import run_grounding_eval as rge
from src.verification.claim_checker import _FigureView
from src.verification.numeric_grounding import ContextIndex, ground_texts

# A table every version of grounding reads alike: a gap header and a unit header.
TABLE = (
    "ACME CORP (in millions of USD)\n"
    "                          2019          2018\n"
    "Net sales               32,136        32,765\n"
    "Operating income         7,207         7,820\n"
)


class TestOutcome:
    @pytest.mark.parametrize("claim, outcome", [
        ("In fiscal 2019, net sales was $32,136 million.", "confirmed"),
        ("In fiscal 2018, net sales was $32,136 million.", "conflict"),
        ("In fiscal 2019, net sales was $32,136 billion.", "unsupported"),
        ("In fiscal 2019, gross margin was $7,207 million.", "unbound"),
    ])
    def test_the_four_outcomes(self, claim, outcome):
        [checks] = ground_texts([claim], ContextIndex([{"text": TABLE, "source_file": "f.pdf"}]))
        assert rge.grounding_outcome(checks) == outcome
        # "confirmed" is exactly what the claim checker passes on figures alone.
        assert (outcome == "confirmed") == _FigureView(checks).confirmed


def _run(mode, outcomes):
    return {"mode": mode, "rows": [
        {"id": f"c{i}", "doc": f"doc{i % 3}", "label": label, "kind": kind, "outcome": o}
        for i, (label, kind, o) in enumerate(outcomes)
    ]}


class TestCompare:
    BEFORE = [("contradicted", "period", "confirmed")] * 6 + [("supported", "plain", "confirmed")] * 6 \
        + [("supported", "scaled", "unsupported")] * 3
    AFTER = [("contradicted", "period", "conflict")] * 6 + [("supported", "plain", "confirmed")] * 6 \
        + [("supported", "scaled", "confirmed")] * 3

    def test_paired_counts_by_label_and_kind(self):
        result = rge.compare_runs(_run("ground", self.BEFORE), _run("ground", self.AFTER))
        assert (result["before"]["false_pass"], result["after"]["false_pass"]) == (6, 0)
        assert (result["before"]["supported_pass"], result["after"]["supported_pass"]) == (6, 9)
        # A figure not found fails the claim: a false alarm for a supported one.
        assert (result["before"]["false_alarm"], result["after"]["false_alarm"]) == (3, 0)
        assert result["moved"] == 9
        printed = rge.compare_runs(_run("ground", self.BEFORE), _run("ground", self.AFTER), kinds=("plain",))
        assert printed["before"]["n"] == 6 and printed["paired"]["supported_pass"]["change_pp"] == 0.0

    def test_the_rule(self):
        before, after = _run("ground", self.BEFORE), _run("ground", self.AFTER)
        rule = rge.adoption(rge.compare_runs(before, after),
                            rge.compare_runs(before, after, kinds=rge.PRINTED_KINDS), None)
        assert rule["adopt_on_this_set"]
        # Scale-word claims newly confirmed do not mask a loss among the claims
        # stated as printed: overall 6 -> 7 of 9 confirmed, as printed 6 -> 4.
        worse = _run("ground", self.AFTER[:6] + [("supported", "plain", "unbound")] * 2 + self.AFTER[8:])
        rule = rge.adoption(rge.compare_runs(before, worse),
                            rge.compare_runs(before, worse, kinds=rge.PRINTED_KINDS), None)
        assert not rule["b_as_printed_supported_confirmed_within_1pp"] and not rule["adopt_on_this_set"]

    def test_runs_must_match(self):
        with pytest.raises(ValueError):
            rge.compare_runs(_run("ground", self.BEFORE), _run("verify", self.BEFORE))
        with pytest.raises(ValueError):
            rge.compare_runs(_run("ground", self.BEFORE), _run("ground", self.BEFORE[:-1]))


ROWS = [
    {"line_item": "Net sales", "unit": "currency", "values": {
        "2019": {"raw": "32,136", "raw_value": 32136.0, "period": "FY2019", "kind": "currency"},
        "2018": {"raw": "32,765", "raw_value": 32765.0, "period": "FY2018", "kind": "currency"}}},
    {"line_item": "Diluted earnings per share", "unit": "per_share", "values": {
        "2019": {"raw": "2.45", "raw_value": 2.45, "period": "FY2019", "kind": "currency"},
        "2018": {"raw": "2.10", "raw_value": 2.10, "period": "FY2018", "kind": "currency"}}},
    {"line_item": "Cost of sales", "unit": "currency", "values": {
        "2019": {"raw": "16,682", "raw_value": 16682.0, "period": "FY2019", "kind": "currency"},
        "2018": {"raw": "16,700", "raw_value": 16700.0, "period": "FY2018", "kind": "currency"}}},
]


class TestGroundingSet:
    def test_claims_of_a_table_in_millions(self, monkeypatch):
        monkeypatch.setattr(fvs, "GROUNDING_CELLS_PER_TABLE", 99)
        claims = fvs.grounding_claims_for_table("t1", ROWS, ["2019", "2018"], "millions")
        by_kind = {}
        for c in claims:
            by_kind.setdefault((c["label"], c["kind"]), []).append(c["claim"])
        assert "In fiscal 2019, net sales was $32,136 million." in by_kind[("supported", "scaled")]
        assert "In fiscal 2019, net sales was $32,765 million." in by_kind[("contradicted", "period_scaled")]
        assert "In fiscal 2019, net sales was $16,682." in by_kind[("contradicted", "neighbour")]
        assert "In fiscal 2019, diluted earnings per share was $2.45 million." in by_kind[("contradicted", "unit")]
        assert all(not c.endswith("million.") for c in by_kind[("supported", "plain")])
        # The wrong scale word is never the table's own.
        assert all("million" not in c for c in by_kind[("contradicted", "scale")])
        # A per-share figure takes no scale word as a supported claim.
        assert not any("per share" in c for c in by_kind[("supported", "scaled")])

    def test_a_table_without_a_scale_has_no_scaled_claims(self):
        claims = fvs.grounding_claims_for_table("t1", ROWS, ["2019", "2018"], "units")
        assert {c["kind"] for c in claims} <= {"plain", "period", "neighbour"}

    def test_only_full_fiscal_years(self):
        # A quarter's column and a budget's are not "fiscal 2019"'s figures.
        columns = ["Three months ended June 30, 2019", "Year Ended December 31, 2018", "2017 Budget"]
        assert fvs.fiscal_year_columns(columns) == {"Year Ended December 31, 2018": 2018}

    def test_company_direction_and_sign_claims_on_the_same_cells(self, monkeypatch):
        monkeypatch.setattr(fvs, "GROUNDING_CELLS_PER_TABLE", 99)
        rows = ROWS + [{"line_item": "Net income (loss)", "unit": "currency", "values": {
            "2019": {"raw": "(596)", "raw_value": -596.0, "period": "FY2019", "kind": "currency"},
            "2018": {"raw": "6,623", "raw_value": 6623.0, "period": "FY2018", "kind": "currency"}}}]
        claims = fvs.cds_claims_for_table("t1", rows, ["2019", "2018"], "Acme", ["Beta", "Gamma"])
        texts = {(c["label"], c["kind"], c["claim"]) for c in claims}
        assert ("supported", "company", "Acme reported net sales of $32,136 in fiscal 2019.") in texts
        assert any(k == "company" and lab == "contradicted" and t.startswith(("Beta", "Gamma"))
                   for lab, k, t in texts)
        # 32,136 against 32,765 the year before: a fall.
        assert ("supported", "direction", "Net sales fell to $32,136 in fiscal 2019.") in texts
        assert ("contradicted", "direction", "Net sales rose to $32,136 in fiscal 2019.") in texts
        # The printed sign selects the side: (596) is a loss.
        assert ("supported", "sign", "In fiscal 2019, net loss was $596.") in texts
        assert ("contradicted", "sign", "In fiscal 2019, net income was $596.") in texts
        # The same cells as the grounding set's.
        cells = {(c["row"], c["column"]) for c in fvs.grounding_claims_for_table("t1", rows, ["2019", "2018"], "units")}
        assert {(c["row"], c["column"]) for c in claims} == cells

    def test_sign_claims_on_every_signed_cell(self, monkeypatch):
        # Company and direction claims stay on the seeded cells; every signed
        # cell is a sign claim, seeded or not.
        monkeypatch.setattr(fvs, "GROUNDING_CELLS_PER_TABLE", 1)
        rows = ROWS + [{"line_item": "Net income (loss)", "unit": "currency", "values": {
            "2019": {"raw": "(596)", "raw_value": -596.0, "period": "FY2019", "kind": "currency"},
            "2018": {"raw": "6,623", "raw_value": 6623.0, "period": "FY2018", "kind": "currency"}}}]
        claims = fvs.cds_claims_for_table("t1", rows, ["2019", "2018"], "Acme", ["Beta"])
        kinds = Counter((c["label"], c["kind"]) for c in claims)
        assert kinds[("supported", "sign")] == kinds[("contradicted", "sign")] == 2
        assert kinds[("supported", "company")] == kinds[("contradicted", "company")] == 1
