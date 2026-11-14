"""
The verification benchmark must itself be trustworthy: a label that points at
text the data room does not contain, or a perturbation family split across dev
and test, would make every number reported from it meaningless.

Offline: no model is loaded (the smoke run uses use_nli=False).
"""

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from eval import run_verification_eval as bench
from eval.verification_corpus import (
    CONTEXT_SIZE,
    build_context,
    corpus_chunks,
    corpus_text,
    normalise,
)
from src.verification.evidence import chunk_evidence_text

LABEL_KINDS = {
    # supported/absence: a genuine decline — the data room is silent on it.
    "supported": {"verbatim", "paraphrase", "derived", "absence"},
    "unsupported": {"absent", "absent_figure"},
    # contradicted/absence: "X is not disclosed" where the data room states X.
    "contradicted": {
        "figure", "misattributed", "party", "negation", "period", "unit", "derived", "absence",
    },
}


@pytest.fixture(scope="module")
def families():
    return json.loads(bench.SET_PATH.read_text(encoding="utf-8"))["families"]


class TestSetIntegrity:
    def test_every_evidence_excerpt_is_verbatim_in_its_source(self, families):
        for family in families:
            for ev in family["evidence"]:
                assert normalise(ev["excerpt"]) in corpus_text(ev["source"]), (
                    f"{family['id']}: excerpt not found in {ev['source']}"
                )

    def test_labels_kinds_and_ids_are_valid(self, families):
        ids = [item["id"] for family in families for item in family["items"]]
        assert len(ids) == len(set(ids))
        for family in families:
            assert family["split"] in ("dev", "test")
            for item in family["items"]:
                assert item["kind"] in LABEL_KINDS[item["label"]], item["id"]
                assert item["claim"].strip()

    def test_supported_and_contradicted_claims_carry_evidence(self, families):
        # A claim the data room is silent on has no excerpt — an unsupported
        # one, or a true statement that something is absent; every other claim
        # is a statement about a specific passage.
        for family in families:
            silent = all(
                item["label"] == "unsupported" or (item["label"], item["kind"]) == ("supported", "absence")
                for item in family["items"]
            )
            if not silent:
                assert family["evidence"], family["id"]

    def test_statements_of_absence_are_recognised_as_such(self, families):
        # Both directions of the absence check are only measured if the
        # splitter classes these claims as statements of absence.
        from src.verification.claims import split_claims

        for family in families:
            for item in family["items"]:
                if item["kind"] == "absence":
                    (claim,) = split_claims(item["claim"])
                    assert claim.is_absence, item["id"]

    def test_both_splits_hold_every_label_and_kind(self, families):
        seen = {"dev": Counter(), "test": Counter()}
        for family in families:
            for item in family["items"]:
                seen[family["split"]][(item["label"], item["kind"])] += 1
        expected = {(label, kind) for label, kinds in LABEL_KINDS.items() for kind in kinds}
        for split, counts in seen.items():
            assert set(counts) == expected, f"{split} lacks {expected - set(counts)}"

    def test_no_claim_is_repeated_across_families(self, families):
        # A claim in both splits would leak a tuned threshold into the test half.
        owners: dict[str, str] = {}
        for family in families:
            for item in family["items"]:
                key = normalise(item["claim"]).lower()
                assert owners.setdefault(key, family["id"]) == family["id"], item["id"]


class TestContexts:
    def test_corpus_chunks_are_production_shaped(self):
        chunks = corpus_chunks()
        assert len({c["source_file"] for c in chunks}) == 9
        assert all(c.get("chunk_id") and c.get("text") for c in chunks)
        with_parent = [c for c in chunks if c.get("parent_text")]
        assert with_parent
        # A child is a substring of its parent, so the verifier reads the parent.
        assert all(chunk_evidence_text(c) == c["parent_text"] for c in with_parent[:10])

    def test_context_puts_the_evidence_first_and_is_deterministic(self):
        item = next(i for i in bench.load_items() if i["label"] == "supported")
        first = bench.item_context(item)
        second = bench.item_context(item)
        assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
        assert len(first) == CONTEXT_SIZE
        excerpt = normalise(item["evidence"][0]["excerpt"])
        assert excerpt in normalise(chunk_evidence_text(first[0]))

    def test_missing_evidence_is_an_error_not_a_silent_distractor(self):
        with pytest.raises(ValueError):
            build_context("x", [("aurora_financials_fy2023.txt", "not in the document")])


def _report(checks, numeric=(), status="passed"):
    return SimpleNamespace(
        claim_checks=list(checks), numeric_checks=list(numeric), validation_status=status,
    )


class TestOutcome:
    def test_precedence_fail_over_warn_over_judge_over_pass(self):
        sup = {"status": "supported", "method": "nli"}
        und = {"status": "unverified", "method": "none"}
        uns = {"status": "unsupported", "method": "nli"}
        num = {"status": "unsupported", "method": "numeric"}
        con = {"status": "contradicted", "method": "llm"}
        assert bench.outcome_of(_report([sup])) == "pass"
        assert bench.outcome_of(_report([sup, und], status="warning")) == "judge"
        assert bench.outcome_of(_report([sup, und, uns], status="warning")) == "warn"
        assert bench.outcome_of(_report([sup, uns, num], status="failed")) == "fail"
        assert bench.outcome_of(_report([sup, con], status="failed")) == "fail"

    def test_unverifiable_calculation_is_a_warning(self):
        sup = {"status": "supported", "method": "numeric"}
        assert bench.outcome_of(
            _report([sup], [{"status": "derived_unverified"}], status="warning")
        ) == "warn"

    def test_an_answer_with_no_checkable_claim_is_skipped(self):
        assert bench.outcome_of(_report([])) == "skipped"

    def test_a_skipped_wrong_claim_counts_as_a_false_pass(self):
        rows = [
            {"label": "contradicted", "kind": "negation", "outcome": "skipped"},
            {"label": "contradicted", "kind": "negation", "outcome": "fail"},
            {"label": "supported", "kind": "verbatim", "outcome": "warn"},
            {"label": "supported", "kind": "verbatim", "outcome": "pass"},
        ]
        headline = bench.summarise(rows)["headline"]
        assert headline["false_pass_rate"] == 0.5
        assert headline["false_alarm_rate"] == 0.5
        assert headline["supported_pass_rate"] == 0.5


class TestBaselineGate:
    BASE = {
        "splits": {"test": {"false_pass_rate": 0.05, "false_alarm_rate": 0.10,
                            "supported_pass_rate": 0.60}},
        "random_figure_grounding": {"N%": 0.30},
    }

    def _current(self, **changes):
        cur = json.loads(json.dumps(self.BASE))
        cur["splits"]["test"].update(changes)
        return cur

    def test_unchanged_rates_pass(self):
        assert bench.compare_to_baseline(self._current(), self.BASE, 0.02) == []

    def test_any_rise_in_false_passes_is_a_regression(self):
        # No tolerance on this one: one more wrong claim passed is not noise.
        problems = bench.compare_to_baseline(self._current(false_pass_rate=0.055), self.BASE, 0.02)
        assert problems and "false-pass" in problems[0]

    def test_pass_rate_and_false_alarms_have_tolerance(self):
        ok = self._current(supported_pass_rate=0.59, false_alarm_rate=0.11)
        assert bench.compare_to_baseline(ok, self.BASE, 0.02) == []
        worse = self._current(supported_pass_rate=0.50, false_alarm_rate=0.20)
        assert len(bench.compare_to_baseline(worse, self.BASE, 0.02)) == 2

    def test_random_figure_grounding_may_not_rise(self):
        cur = self._current()
        cur["random_figure_grounding"]["N%"] = 0.40
        assert bench.compare_to_baseline(cur, self.BASE, 0.02)


class TestRunsOffline:
    @pytest.mark.asyncio
    async def test_deterministic_stages_run_without_a_model(self):
        items = bench.load_items(split="dev")[:12]
        rows = await bench.evaluate(items, use_nli=False)
        assert len(rows) == 12
        assert {r["outcome"] for r in rows} <= set(bench.OUTCOMES)
        summary = bench.summarise(rows)
        assert summary["headline"]["items"] == 12

    def test_random_figure_grounding_reports_every_shape(self):
        rates = bench.random_figure_grounding(contexts=2, per_shape=5)
        assert set(rates) == set(bench.FIGURE_SHAPES)
        for r in rates.values():
            assert r["n"] == 10 and 0.0 <= r["grounded"] <= 1.0
