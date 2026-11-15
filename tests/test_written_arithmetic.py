"""
Tests for verifying arithmetic an answer writes out (numeric_grounding,
Decision 73): "$716.6M = $696.0M + $20.6M", "21.5% ($97.3M / $452.8M)",
"[📊 COMPUTED: $716.6M / $92.8M]", LaTeX \\frac forms, chained steps.

Strings are the shapes the live pipeline produced (eval/answer_replay_answers.json,
pm:mh_04 and pm:fin_02). Every confirming test has an adversarial twin: a wrong
result, a wrong input, an input the documents do not hold, a near miss in the
last printed digit, mixed units — each of which must not be confirmed.
"""

import pytest

from src.verification.claim_checker import verify_answer
from src.verification.numeric_grounding import (
    ContextIndex,
    extract_numbers,
    ground_texts,
    plain_math,
)

FINANCIALS = """KEY FINANCIAL METRICS AND NOTES
(in millions of USD)

                                          FY2023          FY2022
Revenue                                   $452.8          $387.1
Gross Margin                               60.0%           59.1%
Operating Margin                           15.0%           13.3%
EBITDA                                     $92.8           $72.1
EBITDA Margin                              20.5%           18.6%
Adjusted EBITDA                            $97.3           $76.4
Adjusted EBITDA Margin                     21.5%           19.7%
Total Debt                                $110.0          $122.0
Cash and Cash Equivalents                  $89.4           $72.1
Net Debt                                   $20.6           $49.9
"""

MERGER = (
    "Section 1.2 Merger Consideration. Each share shall be converted into the right "
    "to receive $58.00 in cash. The aggregate Merger Consideration is approximately "
    "$696.0 million, subject to adjustment."
)

CONTEXT = [
    {"text": FINANCIALS, "source_file": "aurora_financials_fy2023.txt"},
    {"text": MERGER, "source_file": "merger_agreement_v2_final.txt"},
]


def _ground(*claims: str, context=CONTEXT):
    return ground_texts(list(claims), ContextIndex(context))


def _by_raw(checks, raw):
    hits = [c for c in checks if c.raw == raw]
    assert hits, f"{raw!r} not among {[c.raw for c in checks]}"
    return hits[0]


def _confirmed(checks) -> bool:
    return all(c.is_confirmed for c in checks if c.kind != "quantity")


def _not_reproduced(check) -> bool:
    """Neither in the documents nor computed: a failure, or a warning when the
    claim is worded as a calculation (see ground_texts' second pass)."""
    return check.status in ("unsupported", "derived_unverified")


EV = "Implied EV = Aggregate Equity Consideration ($696.0M) + Net Debt ($20.6M) = $716.6M"


class TestPlainMath:
    """LaTeX the answers wrap their arithmetic in is read as plain text."""

    def test_frac_and_text_wrappers(self):
        text = (
            r"$$\frac{\$716.6\text{ million}}{\$92.8\text{ million}} = "
            r"\mathbf{7.72x} \quad (7.72198\text{x})$$"
        )
        plain = plain_math(text)
        assert "$716.6 million / $92.8 million = 7.72x" in plain
        raws = [m.raw for m in extract_numbers(plain)]
        assert raws == ["$716.6 million", "$92.8 million", "7.72x", "7.72198x"]

    def test_times_and_escaped_percent(self):
        assert "×" in plain_math(r"$15\% \times \$696.0\text{M}$")
        assert "15%" in plain_math(r"$15\% \times \$696.0\text{M}$")

    def test_text_without_latex_is_untouched(self):
        text = "Revenue was $452.8 million ($452.8M) in FY2023."
        assert plain_math(text) == text


class TestRatioInParentheses:
    """C (A / B): the result and the inputs are all written in the claim."""

    def test_margin_with_unbound_denominator_is_confirmed(self):
        # $452.8M is in the documents but the claim does not name revenue: it is
        # the one value consistent with the bound margin and the bound numerator.
        [checks] = _ground("Adjusted EBITDA Margin: 21.5% ($97.3M / $452.8M)")
        assert _by_raw(checks, "$452.8M").binding == "bound"
        assert _confirmed(checks)

    def test_wrong_result_is_not_confirmed(self):
        [checks] = _ground("Adjusted EBITDA Margin: 22.5% ($97.3M / $452.8M)")
        assert _not_reproduced(_by_raw(checks, "22.5%"))

    def test_wrong_input_in_documents_does_not_lift(self):
        # $387.1M is FY2022 revenue: 97.3 / 387.1 = 25.1%, not 21.5%.
        [checks] = _ground("Adjusted EBITDA Margin: 21.5% ($97.3M / $387.1M)")
        assert _by_raw(checks, "$387.1M").binding != "bound"
        assert not _confirmed(checks)

    def test_input_absent_from_documents_is_never_solved_for(self):
        # 97.3 / 452.6 = 21.498% rounds to 21.5%, and grounding's relative
        # tolerance finds $452.8 for "$452.6M" — but at the precision the
        # claim prints, $452.6M is in no document, so the equation does not
        # vouch for it.
        [checks] = _ground("Adjusted EBITDA Margin: 21.5% ($97.3M / $452.6M)")
        assert _by_raw(checks, "$452.6M").binding != "bound"
        assert not _confirmed(checks)

    def test_mixed_scales_do_not_hold(self):
        # $97.3M over $452.8B is 0.02%, not 21.5%.
        [checks] = _ground("Adjusted EBITDA Margin: 21.5% ($97.3M / $452.8B)")
        assert not _confirmed(checks)


class TestEquations:
    """A op B = C, sums of several figures, differences, bps from percentages."""

    def test_sum_equation_result_not_in_documents(self):
        [checks] = _ground("Implied EV: $696.0M + $20.6M = $716.6M")
        ev = _by_raw(checks, "$716.6M")
        assert ev.status == "derived"
        assert "696" in ev.formula and "20.6" in ev.formula

    def test_sum_equation_wrong_result_fails(self):
        [checks] = _ground("Implied EV: $696.0M + $20.6M = $717.6M")
        assert _not_reproduced(_by_raw(checks, "$717.6M"))

    def test_result_first_and_difference(self):
        [checks] = _ground("Net Debt: $20.6M = $110.0M - $89.4M")
        assert _confirmed(checks)

    def test_input_absent_from_documents_fails(self):
        # $21.0M is in no document, so $717.0M has no confirmed input to come from.
        [checks] = _ground("Implied EV: $696.0M + $21.0M = $717.0M")
        assert _by_raw(checks, "$21.0M").status == "unsupported"
        assert _by_raw(checks, "$717.0M").status == "unsupported"

    MULTI = [{"text": "Revolver $85.0 million. Capital leases $6.4 million. "
              "Term loan $18.6 million.", "source_file": "credit.txt"}]

    def test_multi_figure_sum(self):
        # No pair of the three makes $110.0M; only the written three-term sum does.
        [checks] = _ground("Total debt: $85.0M + $6.4M + $18.6M = $110.0M", context=self.MULTI)
        assert _by_raw(checks, "$110.0M").status == "derived"

    def test_multi_figure_sum_wrong_result(self):
        [checks] = _ground("Total debt: $85.0M + $6.4M + $18.6M = $111.0M", context=self.MULTI)
        assert _not_reproduced(_by_raw(checks, "$111.0M"))

    def test_signed_terms(self):
        [checks] = _ground("Net: $85.0M + $6.4M - $18.6M = $72.8M", context=self.MULTI)
        assert _by_raw(checks, "$72.8M").status == "derived"

    def test_bps_from_two_percentages(self):
        [checks] = _ground("Gross margin expanded +90 bps (60.0% - 59.1%)")
        bps = _by_raw(checks, "90 bps")
        assert bps.status == "derived"

    def test_bps_written_as_percent_does_not_hold(self):
        # 60.0% - 59.1% is 0.9 points; "90%" is not that at any precision.
        [checks] = _ground("Gross margin expanded 90% (60.0% - 59.1%)")
        assert _by_raw(checks, "90%").status != "derived"

    def test_near_miss_in_last_digit(self):
        # 716.6 / 92.8 = 7.7220 -> "7.72x"; "7.73x" is off by 0.008 > 0.005.
        _, good = _ground(EV, "Implied EV / EBITDA: $716.6M / $92.8M = 7.72x")
        _, bad = _ground(EV, "Implied EV / EBITDA: $716.6M / $92.8M = 7.73x")
        assert _by_raw(good, "7.72x").status == "derived"
        assert _not_reproduced(_by_raw(bad, "7.73x"))

    def test_more_printed_precision_is_held_to_it(self):
        # 716.6 / 92.8 = 7.721983; "7.7220x" holds, "7.7230x" does not.
        _, good = _ground(EV, "Implied EV / EBITDA: $716.6M / $92.8M = 7.7220x")
        _, bad = _ground(EV, "Implied EV / EBITDA: $716.6M / $92.8M = 7.7230x")
        assert _by_raw(good, "7.7220x").status == "derived"
        assert _not_reproduced(_by_raw(bad, "7.7230x"))

    def test_unrelated_figures_side_by_side_are_not_an_equation(self):
        # No operator, no "=": two figures in prose are never combined.
        [checks] = _ground("EBITDA was $92.8M and the multiple 7.72x on $716.6M.")
        assert _by_raw(checks, "7.72x").status == "unsupported"


class TestChaining:
    """A result computed in one claim is an input of the next."""

    ANSWER = (
        EV,
        "7.72x based on FY2023 Reported EBITDA ($92.8 million).",
        r"$$\frac{\$716.6\text{ million}}{\$92.8\text{ million}} = \mathbf{7.72x} \quad (7.72198\text{x})$$",
        "(Derived/Computed metric) [📊 COMPUTED: $716.6M / $92.8M]",
    )

    def test_ev_then_multiple(self):
        ev, plain, latex, marker = _ground(*self.ANSWER)
        assert _by_raw(ev, "$716.6M").status == "derived"
        for raw in ("$716.6 million", "7.72x", "7.72198x"):
            assert _by_raw(latex, raw).status == "derived"
        assert _by_raw(marker, "$716.6M").status == "derived"
        # Restated before the equation that computes it, beside one of its inputs.
        assert _by_raw(plain, "7.72x").status == "derived"

    def test_chain_through_unbound_input_is_not_confirmed(self):
        # $696.0M sits in a sentence this claim does not name: the EV and every
        # multiple built on it are reproduced, but left to the judge.
        ev, plain, latex, marker = _ground(*self.ANSWER)
        assert not _confirmed(ev)
        assert not _by_raw(latex, "7.72x").is_confirmed
        assert not _by_raw(plain, "7.72x").is_confirmed
        assert not _by_raw(marker, "$716.6M").is_confirmed

    def test_chain_through_confirmed_inputs_is_confirmed(self):
        # No "Net Debt" line in these documents: $20.6M exists only as the
        # first claim's result, from two figures it names.
        context = [{"text": FINANCIALS.replace(
            "Net Debt                                   $20.6           $49.9\n", ""),
            "source_file": "aurora_financials_fy2023.txt"}]
        ev, multiple = _ground(
            "Net Debt = Total Debt ($110.0M) - Cash and Cash Equivalents ($89.4M) = $20.6M",
            "Net Debt / EBITDA: $20.6M / $92.8M = 0.222x",
            context=context,
        )
        assert _confirmed(ev)
        assert _by_raw(multiple, "$20.6M").status == "derived"
        assert _by_raw(multiple, "0.222x").status == "derived"
        assert _confirmed(multiple)

    def test_restatement_needs_one_of_the_inputs(self):
        # 7.72x was computed from $716.6M and $92.8M; a claim attaching it to
        # $97.3M names a different computation and is not confirmed by it.
        _, _, restated = _ground(
            EV,
            "Implied EV / EBITDA: $716.6M / $92.8M = 7.72x",
            "7.72x based on FY2023 Company Adjusted EBITDA ($97.3 million).",
        )
        assert _not_reproduced(_by_raw(restated, "7.72x"))

    def test_wrong_chained_result_fails(self):
        _, latex = _ground(
            EV,
            r"$$\frac{\$716.6\text{ million}}{\$92.8\text{ million}} = \mathbf{7.82x}$$",
        )
        assert _not_reproduced(_by_raw(latex, "7.82x"))

    def test_chained_input_that_was_never_computed_fails(self):
        # $716.6M is not in the documents and no claim computes it.
        [marker] = _ground("(Derived/Computed metric) [📊 COMPUTED: $716.6M / $92.8M]")
        assert _not_reproduced(_by_raw(marker, "$716.6M"))

    def test_result_of_a_wrong_equation_is_not_rescued_by_restatement(self):
        # The third claim's own arithmetic is wrong (716.6 / 97.3 = 7.36); the
        # second claim's 7.72x, which shares the input $716.6M, must not
        # confirm it.
        _, _, wrong = _ground(
            EV,
            "Implied EV / EBITDA: $716.6M / $92.8M = 7.72x",
            "Implied EV / Adjusted EBITDA: $716.6M / $97.3M = 7.72x",
        )
        assert _not_reproduced(_by_raw(wrong, "7.72x"))


NO_NET_DEBT = [{"text": FINANCIALS.replace(
    "Net Debt                                   $20.6           $49.9\n", ""),
    "source_file": "aurora_financials_fy2023.txt"}]
NET_DEBT = "Net Debt = Total Debt ($110.0M) - Cash and Cash Equivalents ($89.4M) = $20.6M"

# Three "Label: value" lines a claim can name, so its inputs are confirmed.
LINES = [{"text": "Revolver: $85.0 million\nCapital leases: $6.4 million\nTerm loan: $18.6 million\n",
          "source_file": "credit.txt"}]
NEGATIVE_SUM = "Capital leases ($6.4M) + Term loan ($18.6M) - Revolver ($85.0M) = "


class TestReviewFindings:
    """False-pass paths found in review (Decision 73), each pinned."""

    def test_restatement_by_value_is_never_confirmed(self):
        # 0.222x was computed as Net Debt / EBITDA; "interest coverage" sharing
        # the value and $92.8M is another metric. Reproduced, left to the judge.
        _, restated = _ground(
            "Net Debt / EBITDA: $20.6M / $92.8M = 0.222x",
            "Interest coverage is 0.222x on EBITDA of $92.8M",
        )
        assert not _by_raw(restated, "0.222x").is_confirmed
        _, restated = _ground(
            "Net Debt / EBITDA: $20.6M / $92.8M = 0.22x",
            "Interest coverage is 0.22x on EBITDA of $92.8M",
        )
        assert not _by_raw(restated, "0.22x").is_confirmed

    def test_written_sign_must_match(self):
        [checks] = _ground(NEGATIVE_SUM + "+$60.0M", context=LINES)
        assert _by_raw(checks, "$60.0M").status != "derived"

    def test_signed_negative_result_is_confirmed(self):
        [checks] = _ground(NEGATIVE_SUM + "-$60.0M", context=LINES)
        assert _by_raw(checks, "$60.0M").status == "derived"
        assert _confirmed(checks)

    def test_unsigned_result_of_a_negative_computation_goes_to_the_judge(self):
        [checks] = _ground(NEGATIVE_SUM + "$60.0M", context=LINES)
        result = _by_raw(checks, "$60.0M")
        assert result.status == "derived" and not result.is_confirmed

    def test_direction_word_must_match(self):
        [fell] = _ground("Debt fell by $60.0M ($85.0M - $6.4M - $18.6M)", context=LINES)
        [rose] = _ground("Debt rose by $60.0M ($85.0M - $6.4M - $18.6M)", context=LINES)
        assert _by_raw(fell, "$60.0M").status != "derived"
        assert _by_raw(rose, "$60.0M").status == "derived"

    def test_spaced_dash_before_the_first_term_is_not_a_minus(self):
        # Read as a minus, -85 + 6.4 + 18.6 = -60 "held" for $60.0M; the dash
        # may be a separator, so the equation is not read at all.
        [checks] = _ground("Total - $85.0M + $6.4M + $18.6M = $60.0M", context=LINES)
        assert _by_raw(checks, "$60.0M").status != "derived"

    def test_attached_minus_and_en_dash_alike(self):
        [hyphen] = _ground("Net: -$85.0M + $6.4M + $18.6M = -$60.0M", context=LINES)
        [en_dash] = _ground("Net: –$85.0M + $6.4M + $18.6M = –$60.0M", context=LINES)
        assert _by_raw(hyphen, "$60.0M").status == "derived"
        assert _by_raw(en_dash, "$60.0M").status == "derived"

    def test_coarse_results_are_not_confirmed(self):
        # ("22%" is covered below: this fixture happens to print a 22.)
        for printed in ("0.2x", "0.22x"):
            _, multiple = _ground(
                NET_DEBT, f"Net Debt / EBITDA: $20.6M / $92.8M = {printed}", context=NO_NET_DEBT,
            )
            result = _by_raw(multiple, printed)
            assert result.status == "derived", printed
            assert not result.is_confirmed, printed
        _, latex = _ground(
            EV, r"$$\frac{\$716.6\text{ million}}{\$92.8\text{ million}} = \mathbf{8x}$$",
        )
        assert not _by_raw(latex, "8x").is_confirmed

    def test_coarse_result_binds_no_input(self):
        # "22%" (for 21.5%) would allow anything from 97.3 / 0.225 to 97.3 / 0.215.
        [checks] = _ground("Adjusted EBITDA Margin: 22% ($97.3M / $452.8M)")
        assert _by_raw(checks, "$452.8M").binding != "bound"
        assert not _confirmed(checks)

    def test_currency_and_bare_count_do_not_sum(self):
        [checks] = _ground("Total debt: $85.0M + 6.4M + $18.6M = $110.0M", context=LINES)
        assert _by_raw(checks, "$110.0M").status != "derived"

    def test_percent_with_points_or_bps_does_not_sum(self):
        for written in ("15.0% + 0.5 pp = 15.5%", "15.0% + 50 bps = 15.5%"):
            [checks] = _ground(f"Operating Margin: {written}")
            assert _by_raw(checks, "15.5%").status != "derived", written

    def test_currency_sum_does_not_give_a_bare_count(self):
        [checks] = _ground("Total debt: $85.0M + $6.4M + $18.6M = 110.0M", context=LINES)
        assert _by_raw(checks, "110.0M").status != "derived"


class TestAnswerOutcome:
    """What the verifier makes of it, with no model and no judge."""

    @pytest.mark.asyncio
    async def test_margin_with_its_computation_passes(self):
        report = await verify_answer(
            "Adjusted EBITDA Margin: 21.5% ($97.3M / $452.8M) [📄 aurora_financials_fy2023.txt | p.1]",
            CONTEXT, "What is the adjusted EBITDA margin?", use_nli=False,
        )
        assert report.validation_status == "passed"
        assert report.claim_checks[0]["method"] == "numeric"

    @pytest.mark.asyncio
    async def test_chained_multiple_is_left_to_the_judge_not_failed(self):
        report = await verify_answer(
            "\n".join(f"- {claim}" for claim in TestChaining.ANSWER),
            CONTEXT, "What is the implied EV/EBITDA multiple?", use_nli=False,
        )
        statuses = [c["status"] for c in report.claim_checks]
        assert "unsupported" not in statuses
        assert report.validation_status == "warning"

    @pytest.mark.asyncio
    async def test_wrong_multiple_still_fails(self):
        report = await verify_answer(
            "\n".join(f"- {claim}" for claim in (
                EV, r"$$\frac{\$716.6\text{ million}}{\$92.8\text{ million}} = \mathbf{7.92x}$$")),
            CONTEXT, "What is the implied EV/EBITDA multiple?", use_nli=False,
        )
        assert report.validation_status == "failed"
