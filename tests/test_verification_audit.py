"""
Regression tests for the verification findings of the independent audit.

Each class pins one finding by its audit id. The scenarios are the audit's own
where it gave one; the rest come from false passes found on the labelled
benchmark (eval/verification_set.json). No model is loaded: NLI is stubbed or
off, and the judge is a local function.
"""

import time

import pytest

from src.verification.claim_checker import verify_answer
from src.verification.nli import NLIScore
from src.verification.numeric_grounding import (
    ContextIndex,
    extract_numbers,
    figure_periods,
    ground_texts,
)

FIN = "aurora_financials_fy2023.txt"

INCOME_STATEMENT = """CONSOLIDATED INCOME STATEMENT (in millions of USD)
                                          FY2023          FY2022
                                          ------          ------
Revenue                                   $452.8          $387.1
  Software Licenses                       $198.4          $172.3
  SaaS Subscriptions                      $189.6          $148.2
  Professional Services                    $64.8           $66.6
Gross Profit                              $271.7          $228.8
  Gross Margin                             60.0%           59.1%
Operating Income (EBIT)                    $68.0           $51.4
Net Income                                 $46.4           $34.3
Earnings Per Share (Diluted)               $3.87           $2.86
"""

# The sample financials' key-metrics table is its own chunk and carries neither
# a unit header nor a column header.
KEY_METRICS = """KEY FINANCIAL METRICS AND NOTES
  EBITDA                                   $92.8           $74.0
  EBITDA Margin                            20.5%           19.1%
  Adjusted EBITDA                          $97.3           $74.0
  Adjusted EBITDA Margin                   21.5%           19.1%
"""


def _chunk(text, source=FIN, **extra):
    return {"chunk_id": f"c{abs(hash(text)) % 10_000}", "source_file": source, "text": text, **extra}


def _ground(text, chunks):
    return ground_texts([text], ContextIndex(chunks))[0]


def _status(text, chunks):
    return [(c.raw, c.status) for c in _ground(text, chunks)]


def _binding(text, chunks):
    return [(c.raw, c.binding) for c in _ground(text, chunks)]


def _scorer(entail=0.95, contradiction=0.01):
    async def score(pairs):
        return [NLIScore(entail, contradiction, max(0.0, 1 - entail - contradiction)) for _ in pairs]

    return score


# ─── V1: a figure is grounded only at a unit the document states ──────────────


class TestV1UnitsAreNotInvented:
    AUDIT_CONTEXT = [_chunk(
        "Total revenue for FY2023 was $452.8 million. Net income for FY2023 was "
        "$31.2 million. The Company had 1,850 employees at year end. Notice must be "
        "given within 30 days. The survival period is 18 months."
    )]

    def test_a_ratio_is_never_rescaled_onto_a_headcount(self):
        # The audit's case B: 2 x 1,000 and 1.9 x 1,000 both "matched" 1,850.
        assert _status("The termination fee equals 2% of equity value.", self.AUDIT_CONTEXT) == [
            ("2%", "unsupported")
        ]
        assert _status("Leverage stood at 1.9x at closing.", self.AUDIT_CONTEXT) == [
            ("1.9x", "unsupported")
        ]

    def test_a_bare_prose_number_does_not_ground_a_ratio(self):
        chunks = [_chunk("Days sales outstanding were 65 days. The lease runs 12 years.")]
        assert _status("Gross margin reached 65%.", chunks) == [("65%", "unsupported")]
        assert _status("The multiple was 12x.", chunks) == [("12x", "unsupported")]

    def test_a_section_number_is_not_billions(self):
        chunks = [_chunk("8.2 Indemnification Cap\nThe cap is described below. See 3.5 Survival.")]
        assert _status("The cap is $8.2 billion.", chunks) == [("$8.2 billion", "unsupported")]
        assert _status("Exposure is $3.5 million.", chunks) == [("$3.5 million", "unsupported")]

    def test_a_per_share_figure_is_not_read_in_the_tables_unit(self):
        chunks = [_chunk(INCOME_STATEMENT)]
        assert _status("Net income was $3.87 million in FY2023.", chunks) == [
            ("$3.87 million", "unsupported")
        ]
        assert _status("Diluted EPS was $3.87.", chunks) == [("$3.87", "grounded")]

    def test_a_price_or_a_full_dollar_amount_is_not_rescaled(self):
        chunks = [_chunk(
            "Each share is converted into the right to receive $58.00 per share in cash. "
            "Elena Marsh's base salary is $625,000. The filing fee is $260,000.",
            source="merger_agreement_v2_final.txt",
        )]
        assert _status("The consideration is $58.00 million.", chunks) == [
            ("$58.00 million", "unsupported")
        ]
        assert _status("Her salary is $625 million.", chunks) == [("$625 million", "unsupported")]
        assert _status("The filing fee is $260 million.", chunks) == [("$260 million", "unsupported")]
        assert _status("The filing fee is $260,000.", chunks) == [("$260,000", "grounded")]

    def test_a_table_is_still_read_in_the_unit_its_header_declares(self):
        chunks = [_chunk(INCOME_STATEMENT)]
        text = "Revenue was $452.8 million in FY2023 ($452,800,000)."
        assert [s for _, s in _status(text, chunks)] == ["grounded", "grounded"]
        # ... and in no other unit.
        assert _status("Revenue was $452.8 billion in FY2023.", chunks) == [
            ("$452.8 billion", "unsupported")
        ]
        assert _status("Revenue was $452.8 thousand in FY2023.", chunks) == [
            ("$452.8 thousand", "unsupported")
        ]

    def test_the_header_may_sit_in_another_chunk_of_the_same_document(self):
        # Retrieval returns the key-metrics table without the income statement's
        # "(in millions of USD)"; the header is a property of the file.
        both = [_chunk(KEY_METRICS), _chunk(INCOME_STATEMENT)]
        assert _status("EBITDA was $92.8 million.", both) == [("$92.8 million", "grounded")]
        assert _status("EBITDA was $92.8 billion.", both) == [("$92.8 billion", "unsupported")]

    def test_without_any_header_a_statement_amount_is_thousands_or_millions(self):
        alone = [_chunk(KEY_METRICS)]
        assert _status("EBITDA was $92.8 million.", alone) == [("$92.8 million", "grounded")]
        assert _status("EBITDA was $92.8 billion.", alone) == [("$92.8 billion", "unsupported")]

    def test_a_bare_cell_takes_the_unit_of_the_header_above_it(self):
        table = "INCOME STATEMENT (in millions of USD)\nRevenue        452.8        387.1\n"
        assert _status("Revenue was $452.8 million.", [_chunk(table)]) == [
            ("$452.8 million", "grounded")
        ]
        # The same bare number in prose is not an amount.
        prose = "(in millions of USD)\n\nThe meeting lasted 452.8 minutes in total, a record."
        assert _status("Revenue was $452.8 million.", [_chunk(prose)]) == [
            ("$452.8 million", "unsupported")
        ]

    def test_an_amount_is_not_grounded_by_the_same_number_in_another_currency(self):
        chunks = [_chunk(
            "A commercial dispute regarding approximately EUR 310,000 of disputed "
            "commissions. Aurora's German turnover is EUR 14.2 million."
        )]
        assert _status("The reseller disputes $310,000 of commissions.", chunks) == [
            ("$310,000", "unsupported")
        ]
        assert _status("German turnover is $14.2 million.", chunks) == [
            ("$14.2 million", "unsupported")
        ]
        assert _status("German turnover is EUR 14.2 million.", chunks) == [
            ("EUR 14.2 million", "grounded")
        ]

    def test_basis_points_restated_as_a_percentage_are_recognised(self):
        chunks = [_chunk("The facility bears interest at SOFR + 225bps.")]
        (check,) = _ground("The facility bears interest at SOFR plus 2.25%.", chunks)
        assert (check.status, check.formula) == ("derived", "225 bps / 100")
        assert _status("The facility bears interest at SOFR plus 225%.", chunks) == [
            ("225%", "unsupported")
        ]

    def test_a_range_states_its_unit_once(self):
        assert [(m.raw, m.kind, m.value) for m in extract_numbers("from 225 to 175 basis points")] == [
            ("225", "bps", 225.0), ("175 basis points", "bps", 175.0),
        ]
        assert [m.value for m in extract_numbers("exposure of $3.5 to $8.0 million")] == [3.5e6, 8.0e6]
        # Two separate figures are not a range.
        assert [m.kind for m in extract_numbers("grew from 2,412 to 2,847")] == ["count", "count"]

    def test_random_figures_are_rarely_reported_as_grounded(self):
        # The audit's Monte Carlo, smaller. Before: `$X.Y billion` 33.8%.
        from eval.run_verification_eval import random_figure_grounding

        rates = random_figure_grounding(contexts=20, per_shape=30)
        assert rates["$X.Y billion"]["grounded"] <= 0.05
        assert rates["$X.Y million"]["grounded"] <= 0.05
        assert rates["N.Nx"]["grounded"] <= 0.15


# ─── V2: a grounded figure must be the figure the claim means ─────────────────


class TestV2FiguresAreBound:
    AUDIT_CONTEXT = TestV1UnitsAreNotInvented.AUDIT_CONTEXT
    TABLES = [_chunk(INCOME_STATEMENT), _chunk(KEY_METRICS)]

    def test_correct_statements_are_bound(self):
        for text in (
            "Revenue was $452.8 million in FY2023 and $387.1 million in FY2022.",
            "EBIT rose to $68.0 million in FY2023 from $51.4 million.",
            "Revenue: FY2023 $452.8M; FY2022 $387.1M",
            "Aurora's revenue reached $452.8M in fiscal 2023, up from $387.1M the year before.",
            "EBITDA was $92.8 million in FY2023, an EBITDA margin of 20.5%.",
            "Gross Profit was $271.7 million in FY2023, a gross margin of 60.0%.",
        ):
            assert {b for _, b in _binding(text, self.TABLES)} == {"bound"}, text

    def test_the_audit_scenario_revenue_figure_called_net_income(self):
        # The value is in the context — as revenue. Nothing ties it to net income.
        (check,) = _ground("Net income for FY2023 was $452.8 million.", self.AUDIT_CONTEXT)
        assert check.status == "grounded" and check.binding == "unbound"
        assert not check.is_confirmed
        (check,) = _ground("Total revenue for FY2023 was $452.8 million.", self.AUDIT_CONTEXT)
        assert check.is_confirmed

    def test_right_row_wrong_column(self):
        (check,) = _ground("Revenue was $387.1 million in FY2023.", self.TABLES)
        assert check.binding == "conflict"
        assert "2022" in check.binding_note and "2023" in check.binding_note
        assert _binding("Aurora's FY2022 revenue was $452.8 million.", self.TABLES) == [
            ("$452.8 million", "conflict")
        ]

    def test_swapped_periods_in_one_sentence(self):
        text = "SaaS Subscriptions revenue declined from $189.6 million in FY2022 to $148.2 million in FY2023."
        assert {b for _, b in _binding(text, self.TABLES)} == {"conflict"}

    def test_right_figure_wrong_row(self):
        assert _binding("Software Licenses revenue was $189.6 million in FY2023.", self.TABLES) == [
            ("$189.6 million", "unbound")
        ]

    def test_a_more_specific_row_named_by_the_claim_is_a_conflict(self):
        # 20.5% is the EBITDA margin; the claim names the ADJUSTED margin (21.5%).
        assert _binding("The Adjusted EBITDA margin was 20.5% in FY2023.", self.TABLES) == [
            ("20.5%", "conflict")
        ]
        # $452.8 is total revenue; the claim names a sub-line.
        assert _binding("Professional Services revenue was $452.8 million in FY2023.", self.TABLES) == [
            ("$452.8 million", "conflict")
        ]

    def test_a_row_another_figure_of_the_claim_matches_is_not_a_rival(self):
        text = "Revenue was $452.8 million and gross profit was $271.7 million in FY2023."
        assert {b for _, b in _binding(text, self.TABLES)} == {"bound"}

    def test_a_better_matching_sentence_with_another_figure_is_a_conflict(self):
        chunks = [_chunk(
            "Net Revenue Retention (NRR) was 118% in FY2023 and 114% in FY2022. Gross "
            "revenue retention was 94% in FY2023."
        )]
        assert _binding("Gross revenue retention was 118% in FY2023.", chunks) == [("118%", "conflict")]
        assert _binding("Net revenue retention was 114% in FY2023.", chunks) == [("114%", "conflict")]
        assert _binding("Net revenue retention was 118% in FY2023.", chunks) == [("118%", "bound")]

    def test_an_undated_occurrence_cannot_vouch_against_a_dated_one(self):
        undated = _chunk(
            "Recurring vs Non-recurring Revenue:\n  Total Revenue        $452.8     100.0%\n",
            source="quality_of_earnings_report_fy2023.txt",
        )
        chunks = [_chunk(INCOME_STATEMENT), undated]
        assert _binding("Aurora's FY2022 revenue was $452.8 million.", chunks) == [
            ("$452.8 million", "conflict")
        ]
        # Alone, the undated table is all there is to go on.
        assert _binding("Aurora's FY2022 revenue was $452.8 million.", [undated]) == [
            ("$452.8 million", "bound")
        ]

    def test_columns_no_retrieved_header_names_are_not_guessed(self):
        alone = [_chunk(KEY_METRICS)]
        assert _binding("EBITDA was $74.0 million in FY2023.", alone) == [("$74.0 million", "unbound")]
        # No period claimed: nothing to get wrong.
        assert _binding("EBITDA was $92.8 million.", alone) == [("$92.8 million", "bound")]
        # With the document's column header retrieved in another chunk, the
        # columns are known: one is confirmed, the other contradicted.
        assert _binding("EBITDA was $92.8 million in FY2023.", self.TABLES) == [
            ("$92.8 million", "bound")
        ]
        assert _binding("EBITDA was $74.0 million in FY2023.", self.TABLES) == [
            ("$74.0 million", "conflict")
        ]

    def test_bulleted_figures_do_not_share_one_sentence(self):
        chunks = [_chunk(
            "While the facility remains outstanding, the Company may not, without lender\n"
            "consent:\n\n"
            "  - Incur additional indebtedness in excess of $15 million;\n"
            "  - Make restricted payments (including dividends) in excess of $10 million per\n"
            "    fiscal year;\n"
            "  - Dispose of assets outside the ordinary course with proceeds exceeding\n"
            "    $20 million in any fiscal year;\n",
            source="credit_agreement_summary.txt",
        )]
        assert _binding("Restricted payments are capped at $20 million per fiscal year.", chunks) != [
            ("$20 million", "bound")
        ]
        assert _binding(
            "Restricted payments, including dividends, may not exceed $10 million per fiscal year.",
            chunks,
        ) == [("$10 million", "bound")]

    def test_a_figure_the_claims_own_figures_reproduce_is_confirmed(self):
        # "17.0%" sits on a "Revenue Growth: 17.0% YoY" line the claim does not
        # quote; the two revenue figures it does state reproduce it.
        chunks = [_chunk(INCOME_STATEMENT + "\nRevenue Growth: 17.0% YoY\n")]
        text = "Revenue rose 17.0%, from $387.1 million in FY2022 to $452.8 million in FY2023."
        checks = {c.raw: c for c in _ground(text, chunks)}
        assert checks["17.0%"].is_confirmed and "387.1M" in checks["17.0%"].formula

    @pytest.mark.asyncio
    async def test_a_misattributed_figure_goes_to_the_judge_even_if_nli_entails(self):
        asked = []

        async def judge(query, claims, documents):
            asked.extend(claims)
            return {c["id"]: {"status": "contradicted", "document": 1, "reason": "FY2022"}
                    for c in claims}, "judge"

        report = await verify_answer(
            f"Revenue was $387.1 million in FY2023 [📄 {FIN} | Income Statement].",
            self.TABLES, "q", scorer=_scorer(entail=0.99), judge=judge, use_nli=True,
        )
        assert len(asked) == 1
        assert report.validation_status == "failed"
        assert report.claim_checks[0]["status"] == "contradicted"

    @pytest.mark.asyncio
    async def test_without_a_judge_a_misattributed_figure_is_not_passed(self):
        report = await verify_answer(
            f"Net income for FY2023 was $452.8 million [📄 {FIN} | Income Statement].",
            self.AUDIT_CONTEXT, "q", use_nli=False,
        )
        # Before: supported (numeric), "passed", confidence 1.0.
        assert report.validation_status == "warning"
        check = report.claim_checks[0]
        assert check["status"] == "unverified"
        assert "$452.8 million" in check["reason"]
        assert report.confidence < 1.0

    @pytest.mark.asyncio
    async def test_an_unbound_paraphrase_is_supported_when_nli_entails(self):
        report = await verify_answer(
            f"Aurora's top line reached $452.8 million [📄 {FIN} | Income Statement].",
            self.TABLES, "q", scorer=_scorer(entail=0.97), use_nli=True,
        )
        assert report.validation_status == "passed"
        assert report.claim_checks[0]["method"] == "nli"


class TestPeriodsStatedForFigures:
    def _periods(self, text):
        spans = [(m.start, m.end) for m in extract_numbers(text, claims_only=False)]
        return figure_periods(text, spans)

    def test_adjacent_period_wins(self):
        assert self._periods("$452.8 million in FY2023 and $387.1 million in FY2022") == ["2023", "2022"]
        assert self._periods("FY2023 $452.8M; FY2022 $387.1M") == ["2023", "2022"]

    def test_comparison_wording_names_another_period(self):
        assert self._periods("reached $452.8M in fiscal 2023, up from $387.1M the year before") == [
            "2023", "*",
        ]
        assert self._periods("closed FY2023 with $89.4M, against $72.1M a year earlier") == ["2023", "*"]

    def test_the_only_period_applies_unless_it_belongs_to_a_comparison(self):
        assert self._periods("In FY2023, revenue was $452.8M and EBITDA was $92.8M") == ["2023", "2023"]
        assert self._periods("Enterprise Customers: 284 (up from 231 in FY2022)") == ["", "2022"]

    def test_docket_numbers_are_not_periods(self):
        text = "Case No. 2023-CV-04821 seeks $8.0 million"
        (figure,) = extract_numbers(text)
        assert figure_periods(text, [(figure.start, figure.end)]) == [""]


# ─── Adversarial input: document text must not stall verification ─────────────


class TestPathologicalLinesAreLinear:
    """
    A security review timed the first version's separator-line pattern
    (`^\\s*[=\\-_*~|:\\s]{4,}\\s*$`) at 0.37 s for 800 spaces followed by "x",
    growing eightfold per doubling — hours for one 50,000-character line of a
    counterparty's document.
    """

    LINES = {
        "spaces-then-x": " " * 50_000 + "x",
        "dashes-then-x": "-" * 50_000 + "x",
        "paren-then-spaces": "(" + " " * 50_000 + "x",
        "trailing-spaces": "a" + " " * 50_000,
        "label-gap-colon": "Label" + " " * 50_000 + ": 5",
        "dotted": ". " * 25_000 + "x",
        "figures-and-periods": "$1.0 FY2023 " * 4_000,
    }

    # Explicit ids: pytest puts the parameter in an environment variable, and a
    # 50,000-character one does not fit.
    @pytest.mark.parametrize("line", list(LINES.values()), ids=list(LINES))
    def test_indexing_a_long_line_is_fast(self, line):
        started = time.perf_counter()
        index = ContextIndex([{"source_file": "x.txt", "text": f"Revenue was $452.8 million.\n{line}\n"}])
        ground_texts(["Revenue was $452.8 million in FY2023."], index)
        assert time.perf_counter() - started < 5.0

    def test_a_schedule_repeating_one_figure_does_not_multiply_the_work(self):
        """
        Binding looked for a rival row once per candidate figure, so a schedule
        that repeats a figure on every row cost candidates x rows: 4.2 s for a
        40-claim answer against 1,000 rows, fourfold per doubling (about a
        minute at these 4,000). The rows carry a row number, so their columns
        are not "alike" and every candidate reaches the rival check.
        """
        rows = "\n".join(
            f"Revenue line {k} gross profit        $452.8      $271.7" for k in range(4_000)
        )
        index = ContextIndex([{
            "source_file": "schedule.txt",
            "text": "SCHEDULE (in millions of USD)\n" + rows,
        }])
        claim = "Revenue was $452.8 million in FY2023 and gross profit was $271.7 million."

        started = time.perf_counter()
        results = ground_texts([claim] * 40, index)
        assert time.perf_counter() - started < 5.0
        # ... and the answer is what the slow version gave: both figures bound.
        assert {(c.status, c.binding) for c in results[0]} == {("grounded", "bound")}


# ─── V3: wording alone does not turn a fabricated figure into a calculation ───


class TestV3CueWordsDoNotExcuseAFigure:
    CONTEXT = TestV1UnitsAreNotInvented.AUDIT_CONTEXT
    DEAL = [
        _chunk("The aggregate Merger Consideration is approximately $696 million.",
               source="merger_agreement_v2_final.txt"),
        _chunk(KEY_METRICS),
    ]

    def test_a_verb_of_change_does_not_downgrade_a_fabricated_figure(self):
        # The audit's pair: "was" failed, "increased to" only warned.
        for verb in ("was", "increased to", "rose to approximately"):
            assert _status(f"Total revenue {verb} $999.9 million in FY2023.", self.CONTEXT) == [
                ("$999.9 million", "unsupported")
            ], verb

    @pytest.mark.asyncio
    async def test_the_answer_fails_either_way(self):
        report = await verify_answer(
            f"Total revenue increased to $999.9 million in FY2023 [📄 {FIN} | IS].",
            self.CONTEXT, "q", use_nli=False,
        )
        assert report.validation_status == "failed"
        assert report.confidence == 0.0

    def test_a_calculation_with_its_inputs_beside_it_is_flagged_not_failed(self):
        chunks = [_chunk(INCOME_STATEMENT)]
        text = "Revenue rose from $387.1 million to $452.8 million, an increase of roughly $99.9 million."
        assert _status(text, chunks)[-1] == ("$99.9 million", "derived_unverified")

    def test_inputs_stated_elsewhere_in_the_answer_count_if_they_reproduce_it(self):
        claims = [
            "The aggregate Merger Consideration is approximately $696 million.",
            "EBITDA was $92.8 million.",
            "Implied EV/EBITDA: 7.50x",
        ]
        index = ContextIndex(self.DEAL)
        assert ground_texts(claims, index)[2][0].status == "derived_unverified"
        # The same wording around a multiple nothing reproduces is a fabrication.
        claims[2] = "Implied EV/EBITDA: 9.90x"
        assert ground_texts(claims, index)[2][0].status == "unsupported"
        # ... and so is a "calculated" figure in an answer that states no inputs.
        alone = ground_texts(["Implied EV/EBITDA of approximately 7.50x."], index)
        assert alone[0][0].status == "unsupported"

    @pytest.mark.asyncio
    async def test_an_unreproducible_calculation_earns_no_confidence(self):
        answer = (
            "Revenue rose from $387.1 million in FY2022 to $452.8 million in FY2023, an increase "
            f"of roughly $99.9 million [📄 {FIN} | IS]."
        )
        report = await verify_answer(answer, [_chunk(INCOME_STATEMENT)], "q", use_nli=False)
        assert report.validation_status == "warning"
        # 2 of 3 figures confirmed, the claim itself unverified: (0.5 + 2/3) / 2.
        assert report.confidence == pytest.approx(0.583, abs=0.001)


# ─── V4: the claim cap does not exempt figures ────────────────────────────────


class TestV4EveryClaimsFiguresAreGrounded:
    CONTEXT = TestV1UnitsAreNotInvented.AUDIT_CONTEXT
    FORTY = "\n".join(
        f"- Total revenue for FY2023 was $452.8 million [📄 {FIN} | r{i}]." for i in range(40)
    )

    @pytest.mark.asyncio
    async def test_a_fabricated_figure_in_claim_41_fails_the_answer(self):
        answer = self.FORTY + f"\n- Adjusted EBITDA for FY2023 was $777.7 million [📄 {FIN} | x]."
        report = await verify_answer(answer, self.CONTEXT, "q", use_nli=False)
        assert len(report.claim_checks) == 41
        assert report.validation_status == "failed"
        assert report.numeric_counts["unsupported"] == 1

    @pytest.mark.asyncio
    async def test_a_confirmed_figure_past_the_cap_still_passes(self):
        answer = self.FORTY + f"\n- Net income for FY2023 was $31.2 million [📄 {FIN} | x]."
        report = await verify_answer(answer, self.CONTEXT, "q", use_nli=False)
        assert report.validation_status == "passed"
        assert report.claim_checks[40]["method"] == "numeric"

    @pytest.mark.asyncio
    async def test_an_unchecked_statement_past_the_cap_is_a_warning_and_is_said(self):
        async def judge(query, claims, documents):
            raise AssertionError("claims past the cap are not sent to the judge")

        answer = self.FORTY + f"\n- The Company intends to vigorously defend the claim [📄 {FIN} | x]."
        report = await verify_answer(
            answer, self.CONTEXT, "q", scorer=_scorer(entail=0.99), judge=judge, use_nli=True,
        )
        assert report.validation_status == "warning"
        assert report.claim_checks[40]["status"] == "unverified"
        assert any("beyond the first 40" in flag for flag in report.flags)
        assert report.confidence < 1.0


# ─── V5: durations and counts are figures too ─────────────────────────────────


class TestV5QuantitiesAreChecked:
    SOURCE = "merger_agreement_v2_final.txt"
    CONTEXT = [_chunk(
        "The representations shall survive the Closing for a period of eighteen (18) months. "
        "Notice must be given within 30 days. Severance of $2,160,000 is payable if the "
        "executive is terminated within 24 months after the change of control. "
        "The Company had 700 employees.",
        source=SOURCE,
    )]

    def test_a_bare_number_with_what_it_counts_is_extracted(self):
        def got(text):
            return [(m.raw, m.kind, m.unit) for m in extract_numbers(text)]

        assert got("The survival period is 24 months.") == [("24 months", "quantity", "month")]
        assert got("Notice must be given within 45 days.") == [("45 days", "quantity", "day")]
        assert got("The Company had 950 employees.") == [("950 employees", "quantity", "employee")]
        assert got("a 60-day exclusivity period") == [("60 day", "quantity", "day")]
        assert got("for eighteen (18) months") == [("18 months", "quantity", "month")]
        assert got("within 30 calendar days") == [("30 days", "quantity", "day")]
        assert got("between 6 and 10 months") == [
            ("6", "quantity", "month"), ("10 months", "quantity", "month"),
        ]
        # A bare number that counts nothing named is still not a claim.
        assert got("There are 12 of them, see item 5.") == []

    def test_a_quantity_is_grounded_by_the_same_count_of_the_same_thing(self):
        assert _status("The survival period is 18 months.", self.CONTEXT) == [("18 months", "grounded")]
        assert _status("The survival period is 24 days.", self.CONTEXT) == [("24 days", "unsupported")]
        assert _status("The survival period is 36 months.", self.CONTEXT) == [("36 months", "unsupported")]

    def test_a_quantity_grounds_nothing_but_a_quantity(self):
        assert _status("A $30 million escrow applies.", self.CONTEXT) == [("$30 million", "unsupported")]
        assert _status("The fee is 30%.", self.CONTEXT) == [("30%", "unsupported")]

    @pytest.mark.asyncio
    async def test_a_quantity_the_documents_do_not_state_is_not_passed_on_entailment(self):
        asked = []

        async def judge(query, claims, documents):
            asked.extend(claims)
            return {c["id"]: {"status": "contradicted", "document": 1} for c in claims}, "judge"

        answer = f"The representations survive the Closing for 36 months [📄 {self.SOURCE} | 8.1]."
        report = await verify_answer(
            answer, self.CONTEXT, "q", scorer=_scorer(entail=0.99), judge=judge, use_nli=True,
        )
        assert len(asked) == 1 and report.validation_status == "failed"

        report = await verify_answer(answer, self.CONTEXT, "q", scorer=_scorer(entail=0.99), use_nli=True)
        assert report.validation_status == "warning"
        assert "36 months" in report.claim_checks[0]["reason"]

    @pytest.mark.asyncio
    async def test_a_quantity_that_is_present_does_not_settle_the_claim_by_itself(self):
        # "18 months" is in the context; whether it is the NON-COMPETE period is
        # a question for the entailment model or the judge, as it was before.
        answer = f"The non-compete lasts 18 months [📄 {self.SOURCE} | 8.1]."
        report = await verify_answer(answer, self.CONTEXT, "q", use_nli=False)
        assert report.claim_checks[0]["status"] == "unverified"

    @pytest.mark.asyncio
    async def test_a_missing_quantity_blocks_an_otherwise_confirmed_claim(self):
        cite = f"[📄 {self.SOURCE} | 2]"
        good = await verify_answer(
            f"Severance of $2,160,000 is payable if the executive is terminated within 24 months {cite}.",
            self.CONTEXT, "q", use_nli=False,
        )
        assert good.validation_status == "passed"
        bad = await verify_answer(
            f"Severance of $2,160,000 is payable if the executive is terminated within 36 months {cite}.",
            self.CONTEXT, "q", use_nli=False,
        )
        assert bad.validation_status == "warning"


# ─── V6: only statements about the evidence are exempt from verification ──────


class TestV6OnlyStatementsOfAbsenceAreSkipped:
    ASSERTIONS = [
        "There is no evidence of any pending litigation against the Company.",
        "The buyer will be required to pay a reverse termination fee to the Company on any failure to close.",
        "Management considers the insurance coverage insufficient for the cyber incident.",
        "The key customer consent is missing from the executed agreement.",
    ]
    DECLINES = [
        "The data room does not disclose the FY2024 capital expenditure budget.",
        "The transaction value is not stated in the provided context.",
        "The implied multiple cannot be calculated from the documents.",
        "Enterprise Value (EV) / Transaction Value is absent from all context chunks.",
        "Executive Severance at Closing: Undisclosed / Context missing",
        "Neither quarterly financial reports nor quarterly breakdowns are included in the provided context.",
        "Management projections would be needed to answer this question.",
        "No information on the escrow arrangements is available.",
    ]

    @pytest.mark.parametrize("sentence", ASSERTIONS)
    def test_an_assertion_containing_a_decline_keyword_is_a_claim(self, sentence):
        from src.verification.claims import split_claims

        assert [c.text for c in split_claims(sentence)] == [sentence]

    @pytest.mark.parametrize("sentence", DECLINES)
    def test_a_statement_about_what_the_evidence_lacks_is_not(self, sentence):
        from src.verification.claims import split_claims

        declined = []
        assert split_claims(sentence, declined) == []
        assert len(declined) == 1

    @pytest.mark.asyncio
    async def test_skipped_statements_are_counted(self):
        answer = (
            f"Total revenue for FY2023 was $452.8 million [📄 {FIN} | IS]. "
            "The FY2024 budget is not disclosed. The data room does not state the capex plan."
        )
        report = await verify_answer(answer, TestV1UnitsAreNotInvented.AUDIT_CONTEXT, "q", use_nli=False)
        assert len(report.claim_checks) == 1
        assert report.claims_skipped == 2
        assert any("2 statement(s) of absence" in note for note in report.notes)

    @pytest.mark.asyncio
    async def test_a_fabricated_no_litigation_statement_reaches_the_judge(self):
        asked = []

        async def judge(query, claims, documents):
            asked.extend(c["claim"] for c in claims)
            return {c["id"]: {"status": "contradicted", "document": 1} for c in claims}, "judge"

        chunks = [_chunk("The Company is a defendant in DataFlow Systems Inc. v. Aurora Technologies Inc.")]
        report = await verify_answer(
            self.ASSERTIONS[0] + f" [📄 {FIN} | Litigation]", chunks, "q", use_nli=False, judge=judge,
        )
        assert asked == [self.ASSERTIONS[0]]
        assert report.validation_status == "failed"
