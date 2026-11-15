"""
Filing metadata at ingest and the named-period preference at query time
(Decision 68). In-memory Qdrant, hand-made vectors, no models.
"""

from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from eval.corpus import use_client
from src.data_processing.filing_metadata import extract_filing_metadata, normalize_company_name
from src.data_processing.ingest_pipeline import extract_and_chunk
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, VECTOR_SIZE
from src.vector_db.filing_period import prefer_named_period, question_fiscal_year
from src.vector_db.hybrid_search import hybrid_search
from src.vector_db.representation_cap import cap_table_representations

TEN_K = """UNITED STATES
SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-K
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
For the fiscal year ended December 31, 2018
Commission file number 001-03285
ACME WIDGETS COMPANY
(Exact name of registrant as specified in its charter)
Delaware 41-0417775
Telephone number: (651) 733-1110
"""

# The text layer of some covers interleaves two columns.
TEN_Q_INTERLEAVED = """UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-Q
Quarterly report pursuant to Section 13 or 15(d) of the Securities Exchange Act of 1934
For the quarterly period ended Commission file June 30, 2022 number 1-5805
Big Bank & Co. (Exact name of registrant as specified in its charter) Delaware 13-2624428
"""

# On an 8-K the file number comes before its label.
EIGHT_K = """UNITED STATES SECURITIES AND EXCHANGE COMMISSION WASHINGTON, D.C. 20549
FORM 8-K CURRENT REPORT
Date of report (Date of earliest event reported): May 20, 2022 (May 18, 2022)
Shoe Store, Inc. (Exact name of registrant as specified in charter)
New York 1-10299 13-3513936 (State or other jurisdiction of incorporation) (Commission File Number)
(IRS Employer Identification No.) Registrant's telephone number, including area code: (212) 720-3700
"""

RELEASE = """March 9, 2023
Beauty Shop Announces Fourth Quarter Fiscal 2022 Results
BOLINGBROOK, Ill.--(BUSINESS WIRE)-- Beauty Shop, Inc. (NASDAQ: BSHP) today announced financial results
for the fourth quarter ended January 28, 2023. Our Form 10-Q filed with the Securities and Exchange
Commission describes the risks.
"""

CONTRACT = """AGREEMENT AND PLAN OF MERGER dated as of March 1, 2024 among Parent, Merger Sub and the
Company. The Company shall file with the Securities and Exchange Commission a Form 8-K within four
business days. Fiscal year 2023 results are attached as Exhibit A.
"""


def _sections(text: str, page: int | None = 1) -> list[dict]:
    return [{"text": text, "page_number": page}]


class TestExtraction:
    def test_10k_cover(self):
        meta = extract_filing_metadata(_sections(TEN_K))

        assert meta["filing_form"] == "10-K"
        assert meta["filing_period_end"] == "2018-12-31"
        assert meta["filing_fiscal_years"] == [2018]
        # Leading zeros dropped; the phone and the IRS number are not file numbers.
        assert meta["filing_entity_keys"] == ["sec:1-3285", "name:acme widgets"]
        assert meta["filing_company"] == "ACME WIDGETS COMPANY"

    def test_interleaved_10q_cover(self):
        meta = extract_filing_metadata(_sections(TEN_Q_INTERLEAVED))

        assert meta["filing_form"] == "10-Q"
        assert meta["filing_period_end"] == "2022-06-30"
        # A quarter belongs to a fiscal year ending this calendar year or next.
        assert meta["filing_fiscal_years"] == [2022, 2023]
        assert meta["filing_entity_keys"] == ["sec:1-5805", "name:big bank"]

    def test_8k_file_number_before_its_label(self):
        meta = extract_filing_metadata(_sections(EIGHT_K))

        assert meta["filing_form"] == "8-K"
        assert meta["filing_date"] == "2022-05-20"
        assert meta["filing_fiscal_years"] == [2022]
        assert meta["filing_entity_keys"] == ["sec:1-10299", "name:shoe store"]

    def test_cover_split_into_heading_sections(self):
        # How the PDF processor hands a cover over: headings lifted out as their
        # own sections and repeated as the next body's section_heading, the
        # registered-securities grid as table sections, the file number's label
        # interleaved with its neighbour's.
        sections = [
            {"text": "SECURITIES AND EXCHANGE COMMISSION", "section_heading": "SECURITIES AND EXCHANGE COMMISSION",
             "page_number": 1},
            {"text": "Washington, DC 20549", "section_heading": "SECURITIES AND EXCHANGE COMMISSION",
             "page_number": 1},
            {"text": "FORM\xa010-K", "section_heading": "FORM\xa010-K", "page_number": 1},
            {"text": "For the fiscal year ended January 28, 2023", "section_heading": "FORM\xa010-K",
             "page_number": 1},
            {"text": "ULTA BEAUTY,\xa0INC.", "section_heading": "ULTA BEAUTY,\xa0INC.", "page_number": 1},
            {"text": "(Exact name of registrant as specified in its charter) Delaware 1-33764 38-4022268 "
                     "(State) (Commission (IRS Employer of incorporation) File Number)",
             "section_heading": "ULTA BEAUTY,\xa0INC.", "page_number": 1},
            {"text": "Common stock ULTA NASDAQ " * 50, "section_heading": "x", "page_number": 1, "is_table": 1},
        ]

        meta = extract_filing_metadata(sections)

        assert meta["filing_entity_keys"] == ["sec:1-33764", "name:ulta beauty"]
        assert meta["filing_company"] == "ULTA BEAUTY, INC"

    def test_release_lead_after_its_highlight_tables(self):
        sections = [{"text": "Net revenue growth: First-Quarter: 10.2% " * 60, "is_table": 1, "page_number": 1},
                    {"text": RELEASE, "page_number": 1}]

        assert extract_filing_metadata(sections)["filing_entity_keys"] == ["ticker:bshp", "name:beauty shop"]

    def test_year_ending_in_january_is_named_both_ways(self):
        cover = TEN_K.replace("December 31, 2018", "January 28, 2023")

        assert extract_filing_metadata(_sections(cover))["filing_fiscal_years"] == [2022, 2023]

    def test_earnings_release_is_not_mistaken_for_the_form_it_mentions(self):
        meta = extract_filing_metadata(_sections(RELEASE))

        assert meta["filing_form"] == "earnings_release"
        assert meta["filing_date"] == "2023-03-09"
        assert meta["filing_fiscal_years"] == [2022, 2023]
        assert meta["filing_entity_keys"] == ["ticker:bshp", "name:beauty shop"]

    @pytest.mark.parametrize("text", [CONTRACT, "", "Quarterly results were strong in 2023."])
    def test_other_documents_get_nothing(self, text):
        assert extract_filing_metadata(_sections(text, page=None)) == {}

    def test_only_the_opening_pages_are_read(self):
        sections = [{"text": "Annual report", "page_number": 1}] + [
            {"text": "filler", "page_number": p} for p in (2, 3)] + [
            {"text": TEN_K, "page_number": 4}]

        assert extract_filing_metadata(sections) == {}

    @pytest.mark.parametrize("name, key", [
        ("The Coca-Cola Company", "coca cola"), ("COCA COLA CO", "coca cola"),
        ("PepsiCo, Inc.", "pepsico"), ("Johnson & Johnson", "johnson johnson"), ("3M COMPANY", "3m"),
    ])
    def test_company_names_normalise(self, name, key):
        assert normalize_company_name(name) == key

    def test_ingest_writes_the_fields_on_every_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAYA_ENABLED", "0")
        path = tmp_path / "cover.txt"
        path.write_text(TEN_K + "\n\nItem 1. Business\n\n" + "We make widgets. " * 200, encoding="utf-8")

        doc = extract_and_chunk(str(path), "annual_report.txt", "deal-x")

        assert doc.chunks
        assert all(c["filing_fiscal_years"] == [2018] for c in doc.chunks)
        assert all(c["filing_entity_keys"] == ["sec:1-3285", "name:acme widgets"] for c in doc.chunks)

    def test_ingest_of_an_ordinary_document_adds_no_field(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAYA_ENABLED", "0")
        path = tmp_path / "merger.txt"
        path.write_text(CONTRACT * 20, encoding="utf-8")

        doc = extract_and_chunk(str(path), "merger.txt", "deal-x")

        assert not any(k.startswith("filing_") for c in doc.chunks for k in c)


class TestQuestionYear:
    @pytest.mark.parametrize("question, year", [
        ("What is the FY2018 capital expenditure amount for 3M?", 2018),
        ("What is the FY2017 - FY2019 3 year average of capex as a % of revenue?", 2019),
        ("Does AMD have a healthy quick ratio for FY22?", 2022),
        ("Is Verizon capital intensive based on FY 2022 data?", 2022),
        ("Which region had the biggest drop in Q22023 year over year revenues?", 2023),
        ("As of Q2'2023, is Pfizer spinning off any large business segments?", 2023),
        ("Which segment had the lowest net revenue in 2021 Q1?", 2021),
        ("As of FY2023Q1, why did PepsiCo raise guidance?", 2023),
        ("As of FY 2021, how much did Verizon expect to pay for its retirees in 2024?", 2021),
        ("As of May 26, 2023, what may PepsiCo borrow?", 2023),
        ("Did revenue exceed $2,023 million?", None),
        ("Does 3M maintain a stable trend of dividend distribution?", None),
    ])
    def test_year(self, question, year):
        assert question_fiscal_year(question) == year


# ---------------------------------------------------------------------------
# Query time: a small room. ACME has FY2018 and FY2019 10-Ks; OTHER only FY2019.
# ---------------------------------------------------------------------------

DEAL = "deal-f"
QUERY_VECTOR = [1.0] + [0.0] * (VECTOR_SIZE - 1)
QUERY_SPARSE = SparseVector(indices=[7], values=[1.0])
ACME = ["sec:1-3285", "name:acme"]
OTHER = ["sec:1-9999", "name:other"]


def _dense(score: float) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    vector[0], vector[1] = score, (1 - score * score) ** 0.5
    return vector


async def _room() -> AsyncQdrantClient:
    client = AsyncQdrantClient(location=":memory:")
    await setup_collections(client)
    rows = [
        # chunk_id, score, filing fields
        ("acme2019-a", 0.99, {"filing_entity_keys": ACME, "filing_fiscal_years": [2019]}),
        ("acme2019-b", 0.97, {"filing_entity_keys": ACME, "filing_fiscal_years": [2019]}),
        ("other2019", 0.95, {"filing_entity_keys": OTHER, "filing_fiscal_years": [2019]}),
        ("acme2018", 0.90, {"filing_entity_keys": ACME, "filing_fiscal_years": [2018]}),
        ("contract", 0.85, {}),
    ]
    await client.upsert(COLLECTION_NAME, points=[
        PointStruct(id=i, vector={"dense": _dense(score),
                                  "sparse": SparseVector(indices=[7], values=[score])},
                    payload={"chunk_id": cid, "deal_id": DEAL, "doc_id": cid, "source_file": f"{cid}.pdf",
                             "text": cid, "is_current_version": 1, "contains_pii": 0, **fields})
        for i, (cid, score, fields) in enumerate(rows)
    ])
    return client


async def _ranked(client, question: str, top_k: int = 3) -> tuple[list[str], list[str], object]:
    with use_client(client):
        dense, sparse = await hybrid_search(
            query_text=question, query_vector=QUERY_VECTOR, query_sparse=QUERY_SPARSE, deal_id=DEAL,
            metadata_filters={"include_pii": False}, top_k_dense=top_k, top_k_sparse=top_k)
        dense, sparse, outcome = await prefer_named_period(
            question, dense, sparse, query_vector=QUERY_VECTOR, query_sparse=QUERY_SPARSE, deal_id=DEAL,
            metadata_filters={"include_pii": False}, top_k_dense=top_k, top_k_sparse=top_k)
    return [p.payload["chunk_id"] for p in dense], [p.payload["chunk_id"] for p in sparse], outcome


class TestNamedPeriodPreference:
    @pytest.mark.asyncio
    async def test_the_named_year_replaces_another_year_of_the_same_company(self):
        dense, sparse, outcome = await _ranked(await _room(), "What was ACME's FY2018 capex?")

        # The FY2019 ACME chunks are set aside; FY2018 was not in the first pass.
        assert dense == sparse == ["other2019", "acme2018"]
        assert outcome.applied and outcome.year == 2018

    @pytest.mark.asyncio
    async def test_another_company_without_that_year_is_kept(self):
        dense, _, _ = await _ranked(await _room(), "What was ACME's FY2018 capex?", top_k=5)

        assert "other2019" in dense and "contract" in dense

    @pytest.mark.asyncio
    @pytest.mark.parametrize("question", [
        "What was ACME's capex?",            # no year
        "What was ACME's FY2016 capex?",     # a year no ACME filing covers
        "What was ACME's FY2019 capex?",     # already the right filing
    ])
    async def test_nothing_changes_otherwise(self, question):
        dense, sparse, outcome = await _ranked(await _room(), question)

        assert dense == sparse == ["acme2019-a", "acme2019-b", "other2019"]
        assert not outcome.applied

    @pytest.mark.asyncio
    async def test_switch_off(self, monkeypatch):
        monkeypatch.setenv("FILING_PERIOD_PREFERENCE", "0")

        dense, _, outcome = await _ranked(await _room(), "What was ACME's FY2018 capex?")

        assert dense == ["acme2019-a", "acme2019-b", "other2019"] and not outcome.applied

    @pytest.mark.asyncio
    async def test_deal_and_pii_conditions_still_apply_to_the_extra_search(self):
        client = await _room()
        await client.set_payload(COLLECTION_NAME, payload={"contains_pii": 1}, points=[3])

        dense, _, outcome = await _ranked(client, "What was ACME's FY2018 capex?")

        assert "acme2018" not in dense and not outcome.applied


class TestRepresentationCap:
    PAYLOADS = {
        "n1": {"table_id": "t1", "table_representation": "narrative"},
        "r1": {"table_id": "t1", "table_representation": "row_by_row"},
        "m1": {"table_id": "t1", "table_representation": "markdown"},
        "x": {},
        "v1": {"table_id": "t2", "content_type": "table_text"},
        "v2": {"table_id": "t2", "content_type": "table_text"},
        "n3": {"table_id": "t3", "table_representation": "narrative"},
    }
    ORDER = ["n1", "r1", "x", "m1", "v1", "v2", "n3"]

    def test_cap_keeps_order_and_frees_slots(self):
        assert cap_table_representations(self.ORDER, self.PAYLOADS, per_table=1) == ["n1", "x", "v1", "v2", "n3"]
        assert cap_table_representations(self.ORDER, self.PAYLOADS, per_table=2) == [
            "n1", "r1", "x", "v1", "v2", "n3"]

    def test_one_slot_by_default_and_a_switch_off(self, monkeypatch):
        monkeypatch.delenv("TABLE_REPRESENTATION_CAP", raising=False)
        assert cap_table_representations(self.ORDER, self.PAYLOADS) == ["n1", "x", "v1", "v2", "n3"]

        monkeypatch.setenv("TABLE_REPRESENTATION_CAP", "0")
        assert cap_table_representations(self.ORDER, self.PAYLOADS) == self.ORDER
