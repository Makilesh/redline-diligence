"""
The measurement behind Decision 90: the FinanceBench claim builder
(eval/figure_veto_set.py) and the selection and adoption rules
(eval/run_figure_veto_eval.py). No store, no model: synthetic tables and rows.
"""

from eval import figure_veto_set as fvs
from eval import run_figure_veto_eval as rfv


def _cell(raw: str, value: float, period: str = "FY2019") -> dict:
    return {"raw": raw, "raw_value": value, "period": period, "kind": "currency"}


ROWS = [
    {"line_item": "Net sales", "unit": "currency",
     "values": {"2019": _cell("32,136", 32136.0), "2018": _cell("32,765", 32765.0, "FY2018")}},
    {"line_item": "Cost of sales (1)", "unit": "currency",
     "values": {"2019": _cell("16,682", 16682.0), "2018": _cell("16,682", 16682.0, "FY2018")}},
    {"line_item": "Net income (loss)", "unit": "currency",
     "values": {"2019": _cell("(1,234)", -1234.0), "2018": _cell("2,000", 2000.0, "FY2018")}},
    {"line_item": "Notes due 2025", "unit": "currency", "values": {"2019": _cell("500", 500.0)}},
    {"line_item": "Gross margin", "unit": "percent",
     "values": {"2019": {"raw": "48.1%", "raw_value": 48.1, "period": "FY2019"}}},
]


class TestClaimBuilder:
    def test_labels_read_as_a_sentence_uses_them(self):
        assert fvs.clean_label("Net cash used for investing activities ......:") == \
            "net cash used for investing activities"
        assert fvs.clean_label("Cost of sales (1)") == "cost of sales"
        assert fvs.clean_label("EBITDA") == "EBITDA"

    def test_figures_as_printed(self):
        assert fvs.render(_cell("(1,488.4)", -1488.4), "currency") == "-$1,488.4"
        assert fvs.render(_cell("$ 2,306,072", 2306072.0), "currency") == "$2,306,072"
        assert fvs.render({"raw": "61.5", "raw_value": 61.5}, "percent") == "61.5%"

    def test_columns_that_name_one_year_each(self):
        cols = ["2019", "2018", "% Change", "Column 4", "2019 vs 2018", "Q4 2019"]
        assert fvs.column_years(cols) == {"2018": 2018}  # 2019 is named twice

    def test_the_sides_of_a_signed_label(self):
        assert fvs.sign_sides("net income (loss)") == ("net income", "net loss")
        assert fvs.sign_sides("other income (expense), net") == ("other income, net", "other expense, net")
        assert fvs.sign_sides("(loss) income before taxes") == ("income before taxes", "loss before taxes")
        assert fvs.sign_sides("net sales") is None

    def test_claims_of_one_table(self):
        claims = fvs.claims_for_table("t1", ROWS, ["2019", "2018"], "Acme", ["Globex"])
        kinds = {(c["label"], c["kind"]) for c in claims}
        assert {("supported", "plain"), ("supported", "company"), ("contradicted", "company"),
                ("contradicted", "period"), ("contradicted", "neighbour")} <= kinds
        text = {c["claim"] for c in claims}
        assert "In fiscal 2019, net sales was $32,136." in text
        assert "In fiscal 2019, net sales was $32,765." in text           # the 2018 figure
        assert "Globex reported net sales of $32,136 in fiscal 2019." in text
        assert "Net sales fell to $32,136 in fiscal 2019." in text         # 32,765 -> 32,136
        assert "Net sales rose to $32,136 in fiscal 2019." in text         # the flip
        # The printed sign picks the side of "income (loss)".
        assert "In fiscal 2019, net loss was $1,234." in text
        assert "In fiscal 2019, net income was $1,234." in text
        # A label holding digits makes no claim (its footnote marker aside);
        # a row whose two years print the same figure makes no period swap.
        assert not any("notes due" in c["claim"] for c in claims)
        assert any(c["kind"] == "plain" and "cost of sales" in c["claim"] for c in claims)
        assert not any(c["kind"] == "period" and "cost of sales" in c["claim"] for c in claims)

    def test_a_neighbour_figure_differs_from_every_figure_of_the_row(self):
        claims = fvs.claims_for_table("t1", ROWS, ["2019", "2018"], "Acme", ["Globex"])
        for c in claims:
            if c["kind"] == "neighbour":
                row = ROWS[c["row"]]
                printed = {v["raw"] for v in row["values"].values()}
                assert all(p.strip("()") not in c["claim"] for p in printed)

    def test_the_sample_is_fixed_by_the_table(self):
        a = fvs.claims_for_table("t1", ROWS, ["2019", "2018"], "Acme", ["Globex", "Initech"])
        b = fvs.claims_for_table("t1", ROWS, ["2019", "2018"], "Acme", ["Globex", "Initech"])
        assert a == b


class TestTwoRowClaims:
    def test_claims_joining_two_rows(self):
        claims = fvs.two_row_claims_for_table("t1", ROWS, ["2019", "2018"], "Acme", ["Globex"])
        text = {(c["label"], c["kind"], c["claim"]) for c in claims}
        assert ("supported", "plain",
                "In fiscal 2019, net sales was $32,136 and cost of sales was $16,682.") in text
        assert ("supported", "company",
                "Acme reported net sales of $32,136 and cost of sales of $16,682 in fiscal 2019.") in text
        assert ("contradicted", "company",
                "Globex reported net sales of $32,136 and cost of sales of $16,682 in fiscal 2019.") in text
        # Every pair names two different rows of one year, printed differently.
        for c in claims:
            assert c["row"] != c["row2"] and c["figure"] != c["figure2"]

    def test_a_wrong_claim_changes_exactly_one_of_the_two_figures(self):
        claims = fvs.two_row_claims_for_table("t1", ROWS, ["2019", "2018"], "Acme", ["Globex"])
        supported = {(c["row"], c["row2"], c["column"]): c["claim"]
                     for c in claims if (c["label"], c["kind"]) == ("supported", "plain")}
        changed = [c for c in claims if c["kind"] in ("period", "neighbour")]
        assert changed
        for c in changed:
            assert c["label"] == "contradicted" and c["replaced_row"] in (c["row"], c["row2"])
            right = supported[(c["row"], c["row2"], c["column"])].split(" was ")
            wrong = c["claim"].split(" was ")
            assert sum(a != b for a, b in zip(right, wrong)) == 1

    def test_the_pairs_are_fixed_by_the_table(self):
        args = ("t1", ROWS, ["2019", "2018"], "Acme", ["Globex", "Initech"])
        assert fvs.two_row_claims_for_table(*args) == fvs.two_row_claims_for_table(*args)


def _row(rid, label, base, scores, cluster="c1", split="dev"):
    return {"id": rid, "split": split, "cluster": cluster, "label": label, "kind": "k",
            "claim": rid, "base": base,
            "veto": {c: list(scores) for c in rfv.CONFIGS[1:]}}


class TestRules:
    def test_a_claim_below_the_line_goes_to_the_judge_and_nothing_else_moves(self):
        row = _row("a", "supported", "pass", [0.3])
        assert rfv.outcome_at(row, "headed/all", 0.4) == "judge"
        assert rfv.outcome_at(row, "headed/all", 0.2) == "pass"
        assert rfv.outcome_at({**row, "base": "warn"}, "headed/all", 0.4) == "warn"
        assert rfv.outcome_at(row, "none", 0.4) == "pass"

    def test_the_dev_rule_takes_the_fewest_false_passes_within_two_correct_claims(self):
        rows = [
            _row("w1", "contradicted", "pass", [0.05]),
            _row("w2", "contradicted", "pass", [0.30]),
            _row("s1", "supported", "pass", [0.10]),
            _row("s2", "supported", "pass", [0.20]),
            _row("s3", "supported", "pass", [0.25]),
            _row("s4", "supported", "pass", [0.90]),
        ]
        best, base, _ = rfv.choose(rows)
        assert base["false_pass"] == 2 and base["supported_pass"] == 4
        # Catching w2 costs three correct claims: over the bound. Catching w1
        # alone costs none: the line sits midway to the next score above it.
        assert best["counts"]["false_pass"] == 1 and best["counts"]["supported_pass"] == 4
        assert abs(best["line"] - 0.075) < 1e-9
        assert best["config"].endswith("/cue")  # a tie: cue claims before all

    def test_no_candidate_no_veto(self):
        rows = [_row("w1", "contradicted", "pass", [0.9]),
                *[_row(f"s{i}", "supported", "pass", [0.1]) for i in range(5)]]
        best, _, _ = rfv.choose(rows)
        assert best is None

    def test_adoption_needs_an_interval_below_zero_and_the_two_bounds(self):
        rows = [_row(f"w{i}", "contradicted", "pass", [0.05], cluster=f"f{i}") for i in range(10)]
        rows += [_row(f"s{i}", "supported", "pass", [0.9], cluster=f"f{i}") for i in range(50)]
        result = rfv.evaluate(rows, "headed/all", 0.1)
        assert result["after"]["false_pass"] == 0
        assert rfv.adoption(result)["adopt"]
        # The same catches at a correct claim's price above 2pp: not adopted.
        costly = rows + [_row(f"x{i}", "supported", "pass", [0.05], cluster=f"f{i}") for i in range(3)]
        assert not rfv.adoption(rfv.evaluate(costly, "headed/all", 0.1))["adopt"]

    def test_the_two_row_rule_bounds_the_cost_only(self):
        # No catch needed on two-row claims: the cost must stay within 2pp.
        rows = [_row(f"s{i}", "supported", "pass", [0.9], cluster=f"f{i % 5}") for i in range(100)]
        rows += [_row("w", "contradicted", "pass", [0.9])]
        assert rfv.adoption_two_row(rfv.evaluate(rows, "joined/all", 0.1))["adopt"]
        rows += [_row(f"x{i}", "supported", "pass", [0.05], cluster="f0") for i in range(3)]
        verdict = rfv.adoption_two_row(rfv.evaluate(rows, "joined/all", 0.1))
        assert not verdict["supported_pass_within_2pp"] and not verdict["adopt"]
