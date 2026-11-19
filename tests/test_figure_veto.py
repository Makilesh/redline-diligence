"""
The figure veto (claim_checker.FigureVeto): a claim whose every figure is
confirmed is still scored by a model that has a veto, against the passages the
figures were found in, and a low score sends it to the judge.

No weights are loaded: the scorer is a stub and the loaded model is patched.
"""

import asyncio

import pytest

from src.verification import claim_checker, nli
from src.verification.nli import NLIScore
from src.verification.numeric_grounding import ContextIndex, ground_texts
from src.verification.table_context import headed_window, joined_window

XSMALL = "cross-encoder/nli-deberta-v3-xsmall"
SMALL = "cross-encoder/nli-deberta-v3-small"
SOURCE = "customer_concentration_analysis.txt"
CHUNKS = [
    {
        "chunk_id": "c1",
        "text": "Pinnacle Health accounted for 8.1% of total revenue in FY2023. "
                "No other customer exceeded 5% of total revenue.",
        "source_file": SOURCE,
        "page_number": None,
        "section_heading": "Customer concentration",
    },
    {
        "chunk_id": "c2",
        "text": "Pinnacle Health renewed its master services agreement in FY2023 "
                "for a further three years.",
        "source_file": "material_contracts_summary.txt",
        "page_number": None,
        "section_heading": "Renewals",
    },
]
MARKER = f" [📄 {SOURCE} | Customer concentration]"
# Every figure confirmed: the 8.1% is this sentence's, for FY2023.
PLAIN = "Pinnacle Health accounted for 8.1% of total revenue in FY2023." + MARKER
# Same figure, same sentence, opposite meaning — and a negation cue.
NEGATED = "Pinnacle Health did not account for 8.1% of total revenue in FY2023." + MARKER


def _verify(monkeypatch, answer, support, model=XSMALL, judge=None, loaded_after=None):
    seen: list[tuple[str, str]] = []
    state = {"loaded": model}

    async def scorer(pairs):
        seen.extend(pairs)
        if loaded_after is not None:
            state["loaded"] = loaded_after  # e.g. a checker replaced by a fallback
        return [NLIScore(support, 0.0, 1.0 - support) for _ in pairs]

    monkeypatch.setattr(nli, "ascore_pairs", scorer)
    monkeypatch.setattr(nli, "loaded_model_name", lambda: state["loaded"])
    report = asyncio.run(claim_checker.verify_answer(answer, CHUNKS, "concentration?", judge=judge, use_nli=True))
    [check] = report.claim_checks
    return report, check, seen


class TestConfiguration:
    def test_the_models_that_have_a_veto(self):
        # CPU: cue claims, plain figure windows (Decision 64). GPU: every
        # figure-confirmed claim, table rows read with their headers (90).
        veto = claim_checker.thresholds_for(XSMALL).figure_veto
        assert veto == claim_checker.FigureVeto(line=0.0031, evidence="figure", cues_only=True)
        assert claim_checker.thresholds_for(nli.FACTCG).figure_veto == claim_checker.FigureVeto(
            line=0.03643, evidence="headed", cues_only=False, rows="joined")
        assert claim_checker.thresholds_for(SMALL).figure_veto is None
        assert claim_checker.thresholds_for(None).figure_veto is None


class TestVeto:
    def test_figures_confirmed_but_unsupported_goes_to_the_judge(self, monkeypatch):
        report, check, seen = _verify(monkeypatch, NEGATED, 0.001)
        assert seen, "a figure-confirmed claim with a cue is scored"
        assert check["status"] == "unverified" and check["suspected_misstatement"]
        assert check["evidence_source"] == SOURCE
        # Never failed, never flagged as unsupported: the judge decides.
        assert report.validation_status == "warning"
        assert report.status_counts["unsupported"] == report.status_counts["contradicted"] == 0

    def test_the_judge_has_the_last_word(self, monkeypatch):
        async def judge(query, payload, documents):
            assert any(d["source_file"] == SOURCE for d in documents)
            return {1: {"status": "contradicted", "reason": "it did", "document": 1}}, "stub"

        report, check, _ = _verify(monkeypatch, NEGATED, 0.001, judge=judge)
        assert check["status"] == "contradicted" and check["method"] == "llm"
        assert report.validation_status == "failed"

    def test_enough_support_keeps_the_figure_verdict(self, monkeypatch):
        report, check, seen = _verify(monkeypatch, NEGATED, 0.5)
        assert seen
        assert check["status"] == "supported" and check["method"] == "numeric"
        assert check["score"] == 0.5
        assert report.validation_status == "passed"

    def test_evidence_is_the_passage_holding_the_figure(self, monkeypatch):
        _, _, seen = _verify(monkeypatch, NEGATED, 0.5)
        assert seen and all("8.1%" in premise for premise, _ in seen)
        assert all(hyp.startswith("Pinnacle Health did not account") for _, hyp in seen)

    def test_no_cue_no_scoring(self, monkeypatch):
        report, check, seen = _verify(monkeypatch, PLAIN, 0.001)
        assert seen == []
        assert check["status"] == "supported" and check["method"] == "numeric"
        assert report.validation_status == "passed"

    def test_a_model_without_a_veto_adds_no_pairs(self, monkeypatch):
        report, check, seen = _verify(monkeypatch, NEGATED, 0.001, model=SMALL)
        assert seen == []
        assert check["status"] == "supported" and check["method"] == "numeric"

    def test_factcg_second_guesses_claims_without_a_cue_too(self, monkeypatch):
        report, check, seen = _verify(monkeypatch, PLAIN, 0.001, model=nli.FACTCG)
        assert seen and check["status"] == "unverified" and check["suspected_misstatement"]
        report, check, _ = _verify(monkeypatch, PLAIN, 0.2, model=nli.FACTCG)
        assert check["status"] == "supported" and check["method"] == "numeric"

    def test_lines_follow_the_model_that_loaded(self, monkeypatch):
        # Planned for -xsmall, scored by -small (no veto): the scores are ignored.
        _, check, seen = _verify(monkeypatch, NEGATED, 0.001, model=XSMALL, loaded_after=SMALL)
        assert seen and check["status"] == "supported" and check["method"] == "numeric"

    def test_no_nli_no_veto(self, monkeypatch):
        called = []

        async def scorer(pairs):
            called.append(pairs)
            return []

        monkeypatch.setattr(nli, "ascore_pairs", scorer)
        monkeypatch.setattr(nli, "loaded_model_name", lambda: XSMALL)
        report = asyncio.run(claim_checker.verify_answer(NEGATED, CHUNKS, "q", use_nli=False))
        assert called == [] and report.claim_checks[0]["status"] == "supported"


class TestFigureWindows:
    def test_a_longer_number_is_not_the_figure(self):
        windows = {0: [("Revenue was $452.85 million.", {"revenue"}),
                       ("Revenue was $452.8 million in FY2023.", {"revenue", "fy2023"})]}
        index = ContextIndex([{"text": "Revenue was $452.8 million in FY2023.", "source_file": "f.txt"}])
        [checks] = ground_texts(["Revenue was $452.8 million in FY2023."], index)
        got = claim_checker._figure_windows({"revenue", "fy2023"}, checks, windows)
        assert [w for _, w, _ in got] == ["Revenue was $452.8 million in FY2023."]

    def test_where_the_figure_was_found_is_not_serialised(self):
        index = ContextIndex([{"text": "Revenue was $452.8 million in FY2023.", "source_file": "f.txt"}])
        [[check]] = ground_texts(["Revenue was $452.8 million in FY2023."], index)
        assert check.evidence_chunk == 0 and "452.8" in check.evidence_figure
        assert "evidence_chunk" not in check.to_dict() and "evidence_figure" not in check.to_dict()
        # Nor where its row and header are (Decision 90): the serialised check
        # is exactly what it was.
        assert set(check.to_dict()) == {
            "raw", "value", "kind", "status", "evidence_source", "evidence_snippet",
            "formula", "text_index", "binding", "binding_note",
        }


# ── Headed rows (Decision 90) ────────────────────────────────────────────────
# One table in each form ingestion indexes: a fixed-width table in a .txt
# table chunk, the converter's markdown grid, row-by-row and narrative
# representations, a verbatim ruled grid, and a fixed-width table inside prose.

TXT_TABLE = (
    "CONSOLIDATED INCOME STATEMENT (in millions of USD)\n"
    "                                          FY2023          FY2022\n"
    "                                          ------          ------\n"
    "Revenue                                   $452.8          $387.1\n"
    "\n"
    "Income Before Taxes                        $61.9           $45.8\n"
    "\n"
    "Income Tax Expense                         $15.5           $11.5\n"
)
MARKDOWN = (
    "ACME CORP\n"
    "*Values are as stated in the source, in USD millions.*\n"
    "\n"
    "| Line Item | 2019 | 2018 |\n"
    "|---|---|---|\n"
    "| Net sales | 32,136 | 32,765 |\n"
    "| Cost of sales | 16,682 | 16,682 |\n"
    "| Operating income | 7,207 | 7,820 |\n"
)
ROW_BY_ROW = (
    "ACME CORP\n"
    "Values are as stated in the source, in USD millions.\n"
    "Net sales: 2019=32,136 2018=32,765\n"
    "Operating income: 2019=7,207 2018=7,820\n"
)
NARRATIVE = (
    "ACME CORP\n"
    "Financial table with 2 rows and 2 columns (2019, 2018). Values are as stated in the "
    "source, in USD millions.\n"
    "Net sales: 2019: 32,136; 2018: 32,765\n"
    "Operating income: 2019: 7,207; 2018: 7,820\n"
)
RULED = "(in millions)\nItem | 2023 | 2022\nRevenue | 452.8 | 387.1\nGross profit | 271.7 | 228.8\n"
PROSE_WITH_TABLE = (
    "Results improved across the board this year, as the table below shows for the two "
    "most recent fiscal years.\n"
    "\n"
    "                       FY2023      FY2022\n"
    "Gross Profit           $271.7      $228.8\n"
)


def _cells(text: str, claim: str, *, table: bool = True, extra: list[dict] | None = None):
    chunk = {"text": text, "source_file": "f.pdf", "is_table": 1 if table else 0,
             "content_type": "table_text" if table else "text"}
    chunks = [chunk, *(extra or [])]
    [checks] = ground_texts([claim], ContextIndex(chunks))
    return chunks, checks


def _headed(text: str, claim: str, rows: str = "each", **kwargs) -> list[str]:
    chunks, checks = _cells(text, claim, **kwargs)
    bodies = {i: (c["text"], c.get("is_table") == 1) for i, c in enumerate(chunks)}
    windows = claim_checker._build_windows(chunks)
    return [w for _, w, _ in claim_checker._figure_windows(set(), checks, windows, bodies, rows)]


class TestHeadedRows:
    def test_a_fixed_width_row_far_below_its_header(self):
        _, [tax, _] = _cells(TXT_TABLE, "Income tax expense was $15.5 million in FY2023 and "
                                        "$11.5 million in FY2022.")
        assert tax.evidence_cell and "FY2023" in tax.evidence_header
        windows = _headed(TXT_TABLE, "Income tax expense was $15.5 million in FY2023 and "
                                     "$11.5 million in FY2022.")
        # One window for both figures of the row, and the line saying what the
        # columns are: the plain windows of this row are 1-3 lines that never
        # reach the header four lines up.
        assert windows == [
            "CONSOLIDATED INCOME STATEMENT (in millions of USD)\n"
            "FY2023 FY2022\n"
            "Income Tax Expense $15.5 $11.5"
        ]

    def test_a_markdown_grid(self):
        assert _headed(MARKDOWN, "Operating income was $7,207 in fiscal 2019.") == [
            "ACME CORP\n"
            "*Values are as stated in the source, in USD millions.*\n"
            "| Line Item | 2019 | 2018 |\n"
            "| Operating income | 7,207 | 7,820 |"
        ]

    def test_a_row_by_row_line_carries_its_own_periods(self):
        assert _headed(ROW_BY_ROW, "Operating income was $7,207 in fiscal 2019.") == [
            "ACME CORP\n"
            "Values are as stated in the source, in USD millions.\n"
            "Operating income: 2019=7,207 2018=7,820"
        ]

    def test_a_narrative_line_is_shown_whole(self):
        # Grounding cuts "Net sales: 2019: 32,136;" from the line at the
        # semicolon; a reader needs the other year too.
        [window] = _headed(NARRATIVE, "Net sales were $32,136 in fiscal 2019.")
        assert window.endswith("\nNet sales: 2019: 32,136; 2018: 32,765")
        assert window.startswith("ACME CORP\nFinancial table with 2 rows and 2 columns (2019, 2018).")

    def test_a_ruled_grid_joined_with_pipes(self):
        # The caption is the first line and the unit line at once: shown once.
        assert _headed(RULED, "Revenue was $452.8 million in 2023.") == [
            "(in millions)\nItem | 2023 | 2022\nRevenue | 452.8 | 387.1"
        ]

    def test_a_fixed_width_table_inside_prose(self):
        assert _headed(PROSE_WITH_TABLE, "Gross profit was $271.7 million in FY2023.", table=False) == [
            "FY2023 FY2022\nGross Profit $271.7 $228.8"
        ]

    def test_a_header_borrowed_from_another_chunk_shows_its_periods_only(self):
        # The rows' own chunk has no header; grounding read the periods from
        # the other table of the file, whose title is not this table's.
        other = {"text": "ASSETS                     FY2023      FY2022\nCash   $89.4   $72.1\n",
                 "source_file": "f.pdf", "is_table": 1}
        rows = "KEY METRICS\nTotal Debt   $110.0   $122.0\nNet Debt   $20.6   $49.9\n"
        assert _headed(rows, "Net debt was $20.6 million in FY2023.", extra=[other]) == [
            "KEY METRICS\nFY2023 FY2022\nNet Debt $20.6 $49.9"
        ]

    def test_a_label_line_in_prose_keeps_its_sentence_windows(self):
        text = ("Party A - Vertex Capital Partners (Private Equity)\n"
                "- Offer: $55 - $60 per share (cash)\n")
        chunks, checks = _cells(text, "Vertex offered $55 - $60 per share in cash.", table=False)
        windows = claim_checker._build_windows(chunks)
        bodies = {0: (text, False)}
        headed = claim_checker._figure_windows({"vertex", "offer"}, checks, windows, bodies)
        plain = claim_checker._figure_windows({"vertex", "offer"}, checks, windows)
        # The party is on the line above: a row window would drop it.
        assert headed == plain and any("Vertex" in w for _, w, _ in headed)

    def test_a_figure_the_claim_computes_adds_no_window(self):
        windows = _headed(TXT_TABLE, "Revenue grew by $65.7 million, from $387.1 million in FY2022 "
                                     "to $452.8 million in FY2023.")
        assert windows == [
            "CONSOLIDATED INCOME STATEMENT (in millions of USD)\nFY2023 FY2022\nRevenue $452.8 $387.1"
        ]

    def test_a_span_outside_the_body_yields_nothing(self):
        assert headed_window("Revenue  452.8", (5, 99)) == ""
        assert headed_window("Revenue  452.8", (-1, -1)) == ""
        assert headed_window("Revenue  452.8\n\n", (15, 16)) == ""


TWO_ROWS = "Income tax expense was $15.5 million and revenue was $452.8 million in FY2023."


class TestJoinedRows:
    """A claim joining several rows is shown them together (Decision 90's integration review)."""

    def test_a_two_row_claim_gets_one_window_holding_both_rows(self):
        # Rows in document order, though the claim names the lower one first.
        assert _headed(TXT_TABLE, TWO_ROWS, rows="joined") == [
            "CONSOLIDATED INCOME STATEMENT (in millions of USD)\n"
            "FY2023 FY2022\n"
            "Revenue $452.8 $387.1\n"
            "Income Tax Expense $15.5 $11.5"
        ]
        # Read one row at a time, each window holds half of the claim.
        assert len(_headed(TXT_TABLE, TWO_ROWS, rows="each")) == 2

    @pytest.mark.parametrize("claim", [
        "Income tax expense was $15.5 million in FY2023 and $11.5 million in FY2022.",
        "Revenue grew by $65.7 million, from $387.1 million in FY2022 to $452.8 million in FY2023.",
    ])
    def test_a_one_row_claim_is_unchanged(self, claim):
        assert _headed(TXT_TABLE, claim, rows="joined") == _headed(TXT_TABLE, claim, rows="each")

    def test_prose_figures_keep_their_sentence_windows(self):
        prose = {"text": "The board approved a retention pool of $6.5 million in FY2023.",
                 "source_file": "f.pdf", "is_table": 0, "content_type": "text"}
        claim = ("Income tax expense was $15.5 million and revenue was $452.8 million in FY2023, "
                 "and a retention pool of $6.5 million was approved.")
        windows = _headed(TXT_TABLE, claim, rows="joined", extra=[prose])
        assert len(windows) == 2
        assert windows[0].endswith("Revenue $452.8 $387.1\nIncome Tax Expense $15.5 $11.5")
        assert "retention pool of $6.5 million" in windows[1]

    def test_rows_of_two_chunks_each_under_their_own_headers(self):
        other = ("CONSOLIDATED BALANCE SHEET (in millions of USD)\n"
                 "ASSETS                      FY2023      FY2022\n"
                 "Goodwill                    $234.5      $234.5\n")
        window = joined_window([
            (0, TXT_TABLE, _row_span(TXT_TABLE, "Income Tax Expense"), "FY2023 FY2022", True),
            (1, other, _row_span(other, "Goodwill"), "ASSETS FY2023 FY2022", True),
        ])
        assert window == (
            "CONSOLIDATED INCOME STATEMENT (in millions of USD)\nFY2023 FY2022\n"
            "Income Tax Expense $15.5 $11.5\n"
            "CONSOLIDATED BALANCE SHEET (in millions of USD)\nASSETS FY2023 FY2022\n"
            "Goodwill $234.5 $234.5"
        )

    def test_at_most_four_rows_the_first_the_claim_names(self):
        table = ("SEGMENT REVENUE (in millions of USD)\n"
                 "                 FY2023      FY2022\n"
                 "Alpha            $10.1       $9.1\n"
                 "Beta             $20.2       $19.2\n"
                 "Gamma            $30.3       $29.3\n"
                 "Delta            $40.4       $39.4\n"
                 "Epsilon          $50.5       $49.5\n")
        claim = ("Epsilon was $50.5 million, delta $40.4 million, gamma $30.3 million, beta $20.2 million "
                 "and alpha $10.1 million in FY2023.")
        [window] = _headed(table, claim, rows="joined")
        assert window.split("\n")[2:] == [
            "Beta $20.2 $19.2", "Gamma $30.3 $29.3", "Delta $40.4 $39.4", "Epsilon $50.5 $49.5",
        ]


def _row_span(body: str, label: str) -> tuple[int, int]:
    start = body.index(label)
    end = body.find("\n", start)
    return start, len(body) if end < 0 else end


HEADED = claim_checker.FigureVeto(line=0.5, evidence="headed", cues_only=False)
TABLE_CHUNKS = [{"chunk_id": "t1", "text": TXT_TABLE, "source_file": "aurora_financials.txt",
                 "page_number": None, "section_heading": "Income statement", "is_table": 1,
                 "content_type": "table_text"}]
TABLE_CLAIM = ("Income tax expense was $15.5 million in FY2023. "
               "[📄 aurora_financials.txt | Income statement]")


class TestHeadedVeto:
    def _run(self, monkeypatch, veto, support, answer=TABLE_CLAIM, chunks=TABLE_CHUNKS):
        calls: list[list[tuple[str, str]]] = []

        async def scorer(pairs):
            calls.append(list(pairs))
            return [NLIScore(support, 0.0, 1.0 - support) for _ in pairs]

        th = claim_checker.Thresholds(0.432, 0.9, 0.2, 0.5, figure_veto=veto)
        monkeypatch.setattr(claim_checker, "thresholds_for", lambda _model: th)
        monkeypatch.setattr(nli, "ascore_pairs", scorer)
        monkeypatch.setattr(nli, "loaded_model_name", lambda: nli.FACTCG)
        report = asyncio.run(claim_checker.verify_answer(answer, chunks, "tax?", use_nli=True))
        return report, calls

    def test_the_model_reads_the_row_with_its_header(self, monkeypatch):
        report, calls = self._run(monkeypatch, HEADED, 0.9)
        [pairs] = calls
        assert [p for p, _ in pairs] == [
            "CONSOLIDATED INCOME STATEMENT (in millions of USD)\nFY2023 FY2022\nIncome Tax Expense $15.5 $11.5"
        ]
        assert report.claim_checks[0]["status"] == "supported"

    def test_below_the_line_it_goes_to_the_judge_never_to_a_failure(self, monkeypatch):
        report, _ = self._run(monkeypatch, HEADED, 0.01)
        [check] = report.claim_checks
        assert check["status"] == "unverified" and check["suspected_misstatement"]
        assert report.validation_status == "warning"

    def test_one_scoring_call_per_answer(self, monkeypatch):
        # A figure-confirmed table claim and an unsettled one: both sets of
        # pairs go to the model in a single call.
        answer = TABLE_CLAIM + "\n\nRevenue growth was strong in FY2023. [📄 aurora_financials.txt | Income statement]"
        _, calls = self._run(monkeypatch, HEADED, 0.9, answer=answer)
        assert len(calls) == 1 and len(calls[0]) > 1

    def test_the_cpu_veto_keeps_its_windows(self, monkeypatch):
        # -xsmall's veto reads the plain windows: Income Tax Expense is four
        # lines below its header, and stays out of reach of it.
        xsmall = claim_checker.thresholds_for("cross-encoder/nli-deberta-v3-xsmall").figure_veto
        assert xsmall.evidence == "figure"
        negated = TABLE_CLAIM.replace("was", "was not")
        _, [pairs] = self._run(monkeypatch, xsmall, 0.9, answer=negated)
        assert pairs and all("FY2023 FY2022" not in p for p, _ in pairs)

    def test_a_window_that_cannot_be_built_fails_verification_closed(self, monkeypatch):
        # A safety step (Decision 88): verify_answer raises, and the validator
        # keeps the answer as a warning — the claim is never passed on its
        # figures alone.
        def broken(*args, **kwargs):
            raise ValueError("unreadable table")

        monkeypatch.setattr(claim_checker, "headed_window", broken)
        with pytest.raises(ValueError):
            self._run(monkeypatch, HEADED, 0.9)

    def test_joined_rows_are_one_pair_scored_in_the_one_call(self, monkeypatch):
        joined = claim_checker.FigureVeto(line=0.5, evidence="headed", cues_only=False, rows="joined")
        answer = TWO_ROWS + " [📄 aurora_financials.txt | Income statement]"
        report, [pairs] = self._run(monkeypatch, joined, 0.01, answer=answer)
        assert [p for p, _ in pairs] == [
            "CONSOLIDATED INCOME STATEMENT (in millions of USD)\nFY2023 FY2022\n"
            "Revenue $452.8 $387.1\nIncome Tax Expense $15.5 $11.5"
        ]
        assert report.claim_checks[0]["suspected_misstatement"]

    def test_single_leaves_a_claim_joining_rows_to_its_figures(self, monkeypatch):
        single = claim_checker.FigureVeto(line=0.5, evidence="headed", cues_only=False, rows="single")
        answer = TWO_ROWS + " [📄 aurora_financials.txt | Income statement]"
        report, calls = self._run(monkeypatch, single, 0.01, answer=answer)
        assert calls == []
        [check] = report.claim_checks
        assert check["status"] == "supported" and check["method"] == "numeric"
        # A claim within one row is second-guessed as before.
        report, [pairs] = self._run(monkeypatch, single, 0.01)
        assert len(pairs) == 1 and report.claim_checks[0]["suspected_misstatement"]

    def test_factcg_in_production_reads_the_headed_row(self, monkeypatch):
        # The production lines, not a patched set: FactCG's veto is headed.
        seen: list[tuple[str, str]] = []

        async def scorer(pairs):
            seen.extend(pairs)
            return [NLIScore(0.02, 0.0, 0.98) for _ in pairs]

        monkeypatch.setattr(nli, "ascore_pairs", scorer)
        monkeypatch.setattr(nli, "loaded_model_name", lambda: nli.FACTCG)
        report = asyncio.run(claim_checker.verify_answer(TABLE_CLAIM, TABLE_CHUNKS, "tax?", use_nli=True))
        assert [p for p, _ in seen] == [
            "CONSOLIDATED INCOME STATEMENT (in millions of USD)\nFY2023 FY2022\nIncome Tax Expense $15.5 $11.5"
        ]
        # 0.02 is below FactCG's line (0.0364): to the judge, not failed.
        [check] = report.claim_checks
        assert check["status"] == "unverified" and check["suspected_misstatement"]
        assert report.validation_status == "warning"

    def test_factcg_in_production_reads_a_two_row_claim_in_one_window(self, monkeypatch):
        seen: list[tuple[str, str]] = []

        async def scorer(pairs):
            seen.extend(pairs)
            return [NLIScore(0.6, 0.0, 0.4) for _ in pairs]

        monkeypatch.setattr(nli, "ascore_pairs", scorer)
        monkeypatch.setattr(nli, "loaded_model_name", lambda: nli.FACTCG)
        answer = TWO_ROWS + " [📄 aurora_financials.txt | Income statement]"
        report = asyncio.run(claim_checker.verify_answer(answer, TABLE_CHUNKS, "tax?", use_nli=True))
        assert [p for p, _ in seen] == [
            "CONSOLIDATED INCOME STATEMENT (in millions of USD)\nFY2023 FY2022\n"
            "Revenue $452.8 $387.1\nIncome Tax Expense $15.5 $11.5"
        ]
        assert report.claim_checks[0]["status"] == "supported" and report.validation_status == "passed"


# ── Known defects ─────────────────────────────────────────────────────────────
# The behaviour grounding should have, failing today for the reason given.
# Strict: the fix that makes one pass fails it until its marker is removed.


class TestKnownDefects:
    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
        "Decision 90 finding: on a row-by-row line read without a unique column "
        "header, figure_periods gives every figure the LAST column's year — the "
        "period after '7,207' ties with the one before it and wins"))
    def test_a_row_by_row_figure_belongs_to_the_period_before_its_equals_sign(self):
        index = ContextIndex([{"text": ROW_BY_ROW, "source_file": "f.pdf"}])
        [[wrong]] = ground_texts(["Operating income was $7,207 in fiscal 2018."], index)
        [[right]] = ground_texts(["Operating income was $7,207 in fiscal 2019."], index)
        # Today: the 2019 figure is bound for 2018 — a wrong figure confirmed.
        assert wrong.binding != "bound"
        assert right.binding == "bound"
