"""
Decision 70: how a filing's fiscal year is named, the quarter a filing
reports, fail-safe extraction on cover layouts beyond FinanceBench's 84, and
the quarter step of the named-period preference.

The cover fixtures are written from the public layouts of each SEC form (the
fixed labels every registrant prints: "For the fiscal year ended", "(Exact
name of registrant as specified in its charter)", "(Translation of
Registrant's name into English)", ...) with real registrants' period wording;
none was fetched from EDGAR. In-memory Qdrant, no models.
"""

from __future__ import annotations

from datetime import date

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from eval.corpus import use_client
from src.data_processing.filing_metadata import (
    annual_fiscal_years,
    extract_filing_metadata,
    quarter_fiscal_years,
)
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, VECTOR_SIZE
from src.vector_db.filing_period import prefer_named_period, question_fiscal_period
from src.vector_db.hybrid_search import hybrid_search


def _meta(text: str) -> dict:
    return extract_filing_metadata([{"text": text, "page_number": 1}])


HEADER = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549\n"


def _ten_k(period: str, form: str = "FORM 10-K") -> str:
    return (HEADER + f"{form}\n(Mark One)\n☒ ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES "
            f"EXCHANGE ACT OF 1934\nFor the fiscal year ended {period}\nor\n☐ TRANSITION REPORT PURSUANT TO "
            "SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934\nFor the transition period from to\n"
            "Commission file number 1-3215\nJOHNSON & JOHNSON\n(Exact name of registrant as specified in its "
            "charter)\nNew Jersey 22-1024240\n")


# ---------------------------------------------------------------------------
# Naming a fiscal year
# ---------------------------------------------------------------------------

class TestFiscalYearNaming:
    @pytest.mark.parametrize("end, years", [
        # A nominal December year: the calendar year it lies in.
        (date(2023, 1, 1), [2022]),     # Johnson & Johnson: "full-year 2022", 52 weeks
        (date(2016, 1, 3), [2015]),     # a 53-week year on the Saturday nearest December 31
        (date(2019, 12, 28), [2019]),   # Kraft Heinz: last Saturday of December
        (date(2022, 12, 31), [2022]),
        # A nominal January-March year: registrants disagree, both kept.
        (date(2023, 1, 28), [2022, 2023]),   # Best Buy "fiscal 2023", Ulta Beauty "fiscal 2022"
        (date(2020, 1, 31), [2019, 2020]),   # Walmart "fiscal 2020"
        (date(2024, 2, 3), [2023, 2024]),    # Target "fiscal 2023" (53 weeks)
        (date(2023, 3, 31), [2022, 2023]),   # a March year end (Japanese issuers name the start year)
        (date(2023, 4, 1), [2022, 2023]),    # a 52-week March year ending on April 1
        # Otherwise the year it ends in.
        (date(2023, 4, 30), [2023]),
        (date(2023, 5, 31), [2023]),         # Nike "fiscal 2023"
        (date(2023, 6, 30), [2023]),         # Microsoft, Amcor
        (date(2022, 10, 1), [2022]),         # a 52-week September year
    ])
    def test_annual(self, end, years):
        assert annual_fiscal_years(end) == years

    def test_johnson_and_johnson_10k_is_fiscal_2022_only(self):
        meta = _meta(_ten_k("January 1, 2023"))

        assert meta["filing_form"] == "10-K"
        assert meta["filing_period_end"] == "2023-01-01"
        assert meta["filing_fiscal_years"] == [2022]
        assert meta["filing_fiscal_quarter"] == 4
        assert meta["filing_entity_keys"] == ["sec:1-3215", "name:johnson johnson"]

    @pytest.mark.parametrize("end, quarter, years", [
        # The fiscal year ends 4 - q quarters after a stated quarter.
        (date(2022, 6, 30), 2, [2022]),        # JPMorgan Q2 2022
        (date(2023, 7, 2), 2, [2023]),         # Pfizer / J&J second quarter, 52-53-week calendar
        (date(2023, 10, 1), 3, [2023]),
        (date(2022, 12, 31), 2, [2023]),       # Amcor Q2 FY2023 (June year end)
        (date(2023, 7, 29), 2, [2023, 2024]),  # Best Buy Q2 FY2024 (year ends in January)
        # Unstated (or first) quarter: one to three quarters later.
        (date(2021, 3, 31), None, [2021]),     # a March quarter belongs to that calendar year
        (date(2023, 4, 2), 1, [2023]),         # J&J Q1 2023 (year ended December 31, 2023)
        (date(2022, 6, 30), None, [2022, 2023]),
        (date(2022, 12, 31), None, [2022, 2023]),
    ])
    def test_quarter(self, end, quarter, years):
        assert quarter_fiscal_years(end, quarter) == years


# ---------------------------------------------------------------------------
# 10-Q quarter
# ---------------------------------------------------------------------------

TEN_Q = (HEADER + "FORM 10-Q\n☒ QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT "
         "OF 1934\nFor the quarterly period ended {end}\nCommission file number 1-5805\nBIG BANK & CO.\n(Exact "
         "name of registrant as specified in its charter)\nDelaware 13-2624428\n")


class TestTenQQuarter:
    def test_three_and_six_months_is_the_second_quarter(self):
        sections = [
            {"text": TEN_Q.format(end="June 30, 2023"), "page_number": 1},
            # The table of contents often arrives as a table section.
            {"text": "Consolidated statements of income for the three and six months ended June 30, 2023 "
                     "and 2022", "page_number": 2, "is_table": 1},
        ]

        meta = extract_filing_metadata(sections)

        assert meta["filing_fiscal_quarter"] == 2
        assert meta["filing_fiscal_years"] == [2023]

    def test_twenty_six_weeks_is_the_second_quarter(self):
        text = TEN_Q.format(end="July 29, 2023") + (
            "Condensed Consolidated Statements of Earnings for the Thirteen Weeks Ended July 29, 2023 and "
            "Twenty-Six Weeks Ended July 29, 2023")

        meta = _meta(text)

        assert meta["filing_fiscal_quarter"] == 2
        assert meta["filing_fiscal_years"] == [2023, 2024]

    def test_three_months_only_is_the_first_quarter_with_the_wide_window(self):
        text = TEN_Q.format(end="March 31, 2021") + "income for the three months ended March 31, 2021 and 2020"

        meta = _meta(text)

        assert meta["filing_fiscal_quarter"] == 1
        assert meta["filing_fiscal_years"] == [2021]

    def test_statements_to_another_date_do_not_count(self):
        text = TEN_Q.format(end="September 30, 2022") + "for the six months ended December 31, 2021"

        meta = _meta(text)

        assert "filing_fiscal_quarter" not in meta
        assert meta["filing_fiscal_years"] == [2022, 2023]


# ---------------------------------------------------------------------------
# Earnings releases
# ---------------------------------------------------------------------------

class TestReleases:
    def test_fourth_quarter_release_is_named_for_the_year_it_reports_and_its_date(self):
        # 2022 is what it reports; 2023 is the year it is dated in and guides
        # ("Is growth in JnJ's adjusted EPS expected to accelerate in FY2023?").
        text = ("Johnson & Johnson Reports Q4 and Full-Year 2022 Results; Company guides 2023 adjusted "
                "operational sales growth of 3.5% to 4.5% New Brunswick, N.J. (January 24, 2023) - Johnson & "
                "Johnson (NYSE: JNJ) today announced results for fourth-quarter and full-year 2022.")

        meta = _meta(text)

        assert meta["filing_form"] == "earnings_release"
        assert meta["filing_date"] == "2023-01-24"
        assert meta["filing_fiscal_years"] == [2022, 2023]
        assert meta["filing_fiscal_quarter"] == 4
        assert meta["filing_entity_keys"] == ["ticker:jnj", "name:johnson johnson"]

    def test_headline_without_a_ticker(self):
        # Amcor's fiscal 2023 release opens with its headline and no ticker or date.
        text = ("Amcor reports fiscal 2023 results and provides outlook for fiscal 2024 June 2023 quarter: "
                "GAAP diluted EPS of 12.3 cps; Adjusted EPS of 19.3 cps Fiscal 2023 Full Year Highlights")

        meta = _meta(text)

        assert meta["filing_form"] == "earnings_release"
        assert meta["filing_fiscal_years"] == [2023]      # not the outlook year
        assert meta["filing_fiscal_quarter"] == 4
        assert meta["filing_entity_keys"] == ["name:amcor"]
        assert "filing_date" not in meta

    def test_line_broken_quarter(self):
        text = ("New Brunswick, N.J. (July 20, 2023) - Johnson & Johnson (NYSE: JNJ) today announced results for "
                "second- quarter 2023.")

        meta = _meta(text)

        assert meta["filing_fiscal_years"] == [2023] and meta["filing_fiscal_quarter"] == 2

    def test_fourth_quarter_ending_in_january_is_named_both_ways(self):
        text = ("March 9, 2023 Beauty Shop Announces Fourth Quarter Fiscal 2022 Results BOLINGBROOK, Ill. -- "
                "Beauty Shop, Inc. (NASDAQ: BSHP) today announced financial results for the thirteen-week period "
                "(\"fourth quarter\") and fifty-two-week period (\"fiscal year\") ended January 28, 2023 compared "
                "to the same periods ended January 29, 2022.")

        meta = _meta(text)

        assert meta["filing_period_end"] == "2023-01-28"
        assert meta["filing_fiscal_years"] == [2022, 2023]
        assert meta["filing_fiscal_quarter"] == 4

    def test_a_release_that_names_no_period_gets_no_years(self):
        meta = _meta("Widget Corp. (NYSE: WDG) today announced results. Revenue grew. Guidance for 2024 raised.")

        assert meta["filing_entity_keys"] == ["ticker:wdg", "name:widget"]
        assert "filing_fiscal_years" not in meta

    def test_a_release_that_names_no_period_is_named_by_its_date(self):
        meta = _meta("May 4, 2023 Widget Corp. (NYSE: WDG) today announced results. Revenue grew.")

        assert meta["filing_fiscal_years"] == [2023]

    @pytest.mark.parametrize("text", [
        # A headline-shaped sentence that names no period.
        "The Board reports results of the shareholder vote held on May 3, 2023.",
        # A company that is only stop words.
        "The Company reports fiscal 2023 results.",
    ])
    def test_headline_without_period_or_company_is_not_a_release(self, text):
        assert _meta(text) == {}


# ---------------------------------------------------------------------------
# Other cover layouts
# ---------------------------------------------------------------------------

TWENTY_F = (HEADER + "FORM 20-F (Mark One) ☐ REGISTRATION STATEMENT PURSUANT TO SECTION 12(b) OR (g) OF THE "
            "SECURITIES EXCHANGE ACT OF 1934 OR ☒ ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES "
            "EXCHANGE ACT OF 1934 For the fiscal year ended March 31, 2023 OR ☐ TRANSITION REPORT PURSUANT TO "
            "SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 For the transition period from to OR ☐ "
            "SHELL COMPANY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 Date of "
            "event requiring this shell company report Commission file number 001-06439 SONY GURUPU KABUSHIKI "
            "KAISHA (Exact name of Registrant as specified in its charter) SONY GROUP CORPORATION (Translation of "
            "Registrant's name into English) JAPAN (Jurisdiction of incorporation or organization) 1-7-1 Konan, "
            "Minato-ku, Tokyo 108-0075 Japan (Address of principal executive offices)")

FORTY_F = (HEADER + "FORM 40-F ☐ REGISTRATION STATEMENT PURSUANT TO SECTION 12 OF THE SECURITIES EXCHANGE ACT OF "
           "1934 OR ☒ ANNUAL REPORT PURSUANT TO SECTION 13(a) OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 For "
           "the fiscal year ended October 31, 2022 Commission File Number 001-13354 BANK OF MONTREAL (Exact name "
           "of Registrant as specified in its charter) Canada (Province or other jurisdiction of incorporation or "
           "organization) 6029 (Primary Standard Industrial Classification Code Number (if applicable)) Not "
           "Applicable (I.R.S. Employer Identification Number (if applicable))")

SIX_K = (HEADER + "FORM 6-K REPORT OF FOREIGN PRIVATE ISSUER PURSUANT TO RULE 13a-16 OR 15d-16 UNDER THE "
         "SECURITIES EXCHANGE ACT OF 1934 For the month of May 2023 Commission File Number: 001-06439 SONY GROUP "
         "CORPORATION (Translation of registrant's name into English) 1-7-1 Konan, Minato-ku, Tokyo 108-0075, "
         "Japan. Sony Group Corporation (NYSE: SONY) today announced results for the fiscal year ended March 31, "
         "2023.")

S_ONE = ("As filed with the Securities and Exchange Commission on March 1, 2024 Registration No. 333- "
         + HEADER + "FORM S-1 REGISTRATION STATEMENT UNDER THE SECURITIES ACT OF 1933 NEWCO HOLDINGS, INC. (Exact "
         "name of registrant as specified in its charter) Delaware 7372 (Primary Standard Industrial "
         "Classification Code Number) 12-3456789 (I.R.S. Employer Identification Number). Our fiscal year ended "
         "December 31, 2023. Newco Holdings, Inc. (NASDAQ: NEWC) results of operations.")

PROXY = (HEADER + "SCHEDULE 14A (Rule 14a-101) INFORMATION REQUIRED IN PROXY STATEMENT Proxy Statement Pursuant "
         "to Section 14(a) of the Securities Exchange Act of 1934 Filed by the Registrant ☒ Acme Corp. (NYSE: ACME) "
         "results for fiscal year 2023")

NON_US_10K = (HEADER + "FORM 10-K ☒ ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT "
              "OF 1934 For the fiscal year ended 30 June 2023 OR ☐ TRANSITION REPORT Commission File Number "
              "001-38932 GLOBAL PACKAGING PLC (Exact name of registrant as specified in its charter) Jersey "
              "98-1455367 (State or other jurisdiction of incorporation or organization) (I.R.S. Employer "
              "Identification No.) 83 Tower Road North, Warmley, Bristol, United Kingdom BS30 8XP "
              "+44 117 9753200 (Registrant's telephone number, including area code)")


class TestOtherLayouts:
    def test_10k_amendment(self):
        meta = _meta(_ten_k("December 31, 2022", form="FORM 10-K/A (Amendment No. 1)"))

        assert meta["filing_form"] == "10-K/A"
        assert meta["filing_fiscal_years"] == [2022]
        assert meta["filing_fiscal_quarter"] == 4

    def test_20f_foreign_private_issuer(self):
        meta = _meta(TWENTY_F)

        assert meta["filing_form"] == "20-F"
        assert meta["filing_period_end"] == "2023-03-31"
        assert meta["filing_fiscal_years"] == [2022, 2023]   # Sony says FY2022, Toyota would say FY2023
        assert meta["filing_entity_keys"] == ["sec:1-6439", "name:sony gurupu kabushiki kaisha", "name:sony"]

    def test_40f_canadian_issuer(self):
        meta = _meta(FORTY_F)

        assert meta["filing_form"] == "40-F"
        assert meta["filing_fiscal_years"] == [2022]
        assert meta["filing_entity_keys"] == ["sec:1-13354", "name:bank of montreal"]

    def test_non_us_10k_with_day_month_year_date(self):
        meta = _meta(NON_US_10K)

        assert meta["filing_period_end"] == "2023-06-30"
        assert meta["filing_fiscal_years"] == [2023]
        # The foreign IRS number (98-...) and the +44 phone are not file numbers.
        assert meta["filing_entity_keys"] == ["sec:1-38932", "name:global packaging"]

    @pytest.mark.parametrize("text", [SIX_K, S_ONE, PROXY], ids=["6-K", "S-1", "proxy"])
    def test_forms_without_a_modelled_period_get_nothing(self, text):
        # Not a guess from the release or the financial statements they contain.
        assert _meta(text) == {}

    def test_transition_report_gets_no_period(self):
        text = (HEADER + "FORM 10-K ☐ ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) For the fiscal year ended "
                "________ OR ☒ TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT "
                "OF 1934 For the transition period from July 1, 2022 to December 31, 2022 Commission file "
                "number 1-3285 ACME (Exact name of registrant as specified in its charter)")

        meta = _meta(text)

        assert "filing_fiscal_years" not in meta and "filing_period_end" not in meta

    @pytest.mark.parametrize("text", [
        # OCR noise in the Commission's name: not recognisably a cover.
        "UNITED STATES SECURlTlES AND EXCHANGE COMMlSSION Washington, D.C. 20549 FORM 10-K For the fiscal year "
        "ended December 31, 2022",
        # OCR noise in the form name: a cover, but of no form that can be read.
        HEADER + "FORM l0-K ANNUAL REPORT For the fiscal year ended December 31, 2022 Commission file number "
        "1-3285 ACME (Exact name of registrant as specified in its charter)",
    ])
    def test_scanned_cover_with_noise_in_the_fixed_language_gets_nothing(self, text):
        assert _meta(text) == {}

    @pytest.mark.parametrize("garbled", ["Decem ber 3l, 2O22", "Dec. 31, 2022", "December 3l, 2022"])
    def test_scanned_cover_with_a_garbled_date_gets_no_period(self, garbled):
        # A well-formed date further on (the next field) must not be read instead.
        text = (HEADER + f"FORM 10-K ANNUAL REPORT For the fiscal year ended {garbled} December 31, 2021 "
                "Commission file number 1-3285 ACME WIDGETS COMPANY (Exact name of registrant as specified in "
                "its charter)")

        meta = _meta(text)

        assert "filing_fiscal_years" not in meta
        assert meta["filing_entity_keys"] == ["sec:1-3285", "name:acme widgets"]


# ---------------------------------------------------------------------------
# The question's period
# ---------------------------------------------------------------------------

class TestQuestionPeriod:
    @pytest.mark.parametrize("question, period", [
        ("Was there any drop in Cash between FY 2023 and Q2 of FY2024?", (2024, 2)),
        ("In 2022 Q2, which segment had the highest net income?", (2022, 2)),
        ("Which segment had the lowest net revenue in 2021 Q1?", (2021, 1)),
        ("did the risk JPM faced in the second fiscal quarter of 2023 decrease?", (2023, 2)),
        ("What was revenue in the second quarter of fiscal 2023?", (2023, 2)),
        ("What percent of FY 2023 repurchases occurred in Q4 of FY2023?", (2023, 4)),
        ("As of FY2023Q1, why did PepsiCo raise guidance?", (2023, 1)),
        ("What were 2Q22 deposits?", (2022, 2)),
        ("Revenue in the first-quarter 2023 versus FY2022?", (2023, 1)),
        ("What is the FY2018 capex?", (2018, None)),
        ("Has CVS paid dividends in Q2 of FY2022 and FY2021?", (2022, 2)),
        # A form name after a quarter is not a year.
        ("Which items in the Q3 10-Q changed in 2021?", (2021, None)),
        ("Does 3M maintain a stable trend of dividend distribution?", (None, None)),
    ])
    def test_period(self, question, period):
        assert question_fiscal_period(question) == period


# ---------------------------------------------------------------------------
# Query time: ACME has an FY2022 10-K and Q2 10-Qs of 2022 and 2023.
# ---------------------------------------------------------------------------

DEAL = "deal-q"
QUERY_VECTOR = [1.0] + [0.0] * (VECTOR_SIZE - 1)
QUERY_SPARSE = SparseVector(indices=[7], values=[1.0])
ACME = ["sec:1-3285", "name:acme"]
OTHER = ["sec:1-9999", "name:other"]


def _dense(score: float) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    vector[0], vector[1] = score, (1 - score * score) ** 0.5
    return vector


async def _room(extra: list | None = None) -> AsyncQdrantClient:
    client = AsyncQdrantClient(location=":memory:")
    await setup_collections(client)
    rows = [
        ("k10-2022", 0.99, {"filing_form": "10-K", "filing_entity_keys": ACME, "filing_fiscal_years": [2022],
                            "filing_fiscal_quarter": 4}),
        ("q2-2023", 0.97, {"filing_form": "10-Q", "filing_entity_keys": ACME, "filing_fiscal_years": [2023],
                           "filing_fiscal_quarter": 2}),
        ("other", 0.95, {"filing_form": "10-K", "filing_entity_keys": OTHER, "filing_fiscal_years": [2022],
                         "filing_fiscal_quarter": 4}),
        ("q2-2022", 0.90, {"filing_form": "10-Q", "filing_entity_keys": ACME, "filing_fiscal_years": [2022],
                           "filing_fiscal_quarter": 2}),
        ("contract", 0.85, {}),
        *(extra or []),
    ]
    await client.upsert(COLLECTION_NAME, points=[
        PointStruct(id=i, vector={"dense": _dense(score), "sparse": SparseVector(indices=[7], values=[score])},
                    payload={"chunk_id": cid, "deal_id": DEAL, "doc_id": cid, "source_file": f"{cid}.pdf",
                             "text": cid, "is_current_version": 1, "contains_pii": 0, **fields})
        for i, (cid, score, fields) in enumerate(rows)
    ])
    return client


async def _ranked(client, question: str, top_k: int = 3):
    with use_client(client):
        dense, sparse = await hybrid_search(
            query_text=question, query_vector=QUERY_VECTOR, query_sparse=QUERY_SPARSE, deal_id=DEAL,
            metadata_filters={"include_pii": False}, top_k_dense=top_k, top_k_sparse=top_k)
        dense, sparse, outcome = await prefer_named_period(
            question, dense, sparse, query_vector=QUERY_VECTOR, query_sparse=QUERY_SPARSE, deal_id=DEAL,
            metadata_filters={"include_pii": False}, top_k_dense=top_k, top_k_sparse=top_k)
    return [p.payload["chunk_id"] for p in dense], [p.payload["chunk_id"] for p in sparse], outcome


class TestQuarterPreference:
    @pytest.mark.asyncio
    async def test_a_named_quarter_prefers_that_quarters_10q_over_the_10k(self):
        dense, sparse, outcome = await _ranked(await _room(), "What was ACME's Q2 2022 revenue?")

        assert dense == sparse == ["other", "q2-2022"]
        assert outcome.quarter == 2 and outcome.set_aside == ["k10-2022.pdf", "q2-2023.pdf"]

    @pytest.mark.asyncio
    async def test_the_fourth_quarter_is_the_annual_report(self):
        dense, _, outcome = await _ranked(await _room(), "What was ACME's Q4 2022 revenue?", top_k=4)

        assert dense == ["k10-2022", "other"]
        assert outcome.set_aside == ["q2-2022.pdf", "q2-2023.pdf"]

    @pytest.mark.asyncio
    async def test_a_quarter_absent_from_the_room_falls_back_to_the_year(self):
        by_quarter = await _ranked(await _room(), "What was ACME's Q1 2022 revenue?")
        by_year = await _ranked(await _room(), "What was ACME's FY2022 revenue?")

        assert by_quarter[:2] == by_year[:2] == (["k10-2022", "other", "q2-2022"],) * 2

    @pytest.mark.asyncio
    async def test_switch_reads_the_year_only(self, monkeypatch):
        monkeypatch.setenv("FILING_QUARTER_PREFERENCE", "0")

        dense, _, outcome = await _ranked(await _room(), "What was ACME's Q2 2022 revenue?")

        assert dense == ["k10-2022", "other", "q2-2022"] and outcome.quarter is None

    @pytest.mark.asyncio
    async def test_unstated_quarters_are_neither_set_aside_nor_preferred(self):
        # An 8-K of the year and a 10-Q whose quarter could not be read.
        extra = [("eightk", 0.995, {"filing_form": "8-K", "filing_entity_keys": ACME, "filing_fiscal_years": [2022]}),
                 ("q-unk", 0.80, {"filing_form": "10-Q", "filing_entity_keys": ACME,
                                  "filing_fiscal_years": [2022, 2023]})]

        dense, _, outcome = await _ranked(await _room(extra), "What was ACME's Q2 2022 revenue?", top_k=7)

        assert dense == ["eightk", "other", "q2-2022", "contract", "q-unk"]
        assert outcome.set_aside == ["k10-2022.pdf", "q2-2023.pdf"] and outcome.preferred == ["q2-2022.pdf"]

    @pytest.mark.asyncio
    async def test_documents_without_filing_metadata_are_untouched(self):
        client = AsyncQdrantClient(location=":memory:")
        await setup_collections(client)
        await client.upsert(COLLECTION_NAME, points=[
            PointStruct(id=i, vector={"dense": _dense(s), "sparse": SparseVector(indices=[7], values=[s])},
                        payload={"chunk_id": c, "deal_id": DEAL, "doc_id": c, "source_file": c, "text": c,
                                 "is_current_version": 1, "contains_pii": 0})
            for i, (c, s) in enumerate([("cim", 0.9), ("spa", 0.8)])])

        dense, _, outcome = await _ranked(client, "What was Q2 2022 revenue?")

        assert dense == ["cim", "spa"] and not outcome.applied
