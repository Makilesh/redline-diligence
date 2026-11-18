"""
Document identity (Decision 86): the title and parties a document's opening
states, written on every child chunk at ingest and backfilled onto an index
written before.

Offline: no model, no network; the backfill runs against in-memory Qdrant.
Fixture openings are written from common contract layouts, not copied from
any benchmark document.
"""

from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from src.data_processing.document_identity import (
    document_opening,
    extract_document_identity,
    extract_parties,
    extract_title,
    name_tokens,
    title_acronyms,
)
from src.data_processing.ingest_pipeline import extract_and_chunk
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, VECTOR_SIZE
from src.vector_db.identity_backfill import backfill_document_identity

SPONSORSHIP = (
    "SPONSORSHIP AGREEMENT\n"
    'THIS SPONSORSHIP AGREEMENT ("Agreement") is made and entered into as of this 1st day of March, 2019, '
    "by and between ACME WIDGETS, INC., a Delaware corporation with its principal offices located at "
    '100 Main Street, Suite 4, Springfield, Illinois 62701 ("Acme"), and NORTHWIND TRADERS LLC, a Texas '
    'limited liability company ("Northwind").\n'
    "WHEREAS, Acme wishes to sponsor the events Northwind organises;\n"
)
MERGER_COVER = (
    "Exhibit 2.1\n"
    "AGREEMENT AND PLAN OF MERGER\n"
    "by and among\n"
    "ATLAS HOLDINGS PLC,\n"
    "ATLAS MERGER SUB, INC.\n"
    "and\n"
    "BLUEWATER THERAPEUTICS, INC.\n"
    "Dated as of May 4, 2021\n"
    "TABLE OF CONTENTS\n"
    "ARTICLE I DEFINITIONS 1\n"
)
TEMPLATE_NDA = (
    "MUTUAL NON-DISCLOSURE AGREEMENT\n"
    "This Confidentiality Agreement is between_____________________ (“Customer”) located at"
    "______________________ and Harbor Analytics (“Company”). WHEREAS, the parties wish to "
    "discuss a possible business relationship;\n"
)
DESCRIBED_ONLY = (
    "Non-Disclosure Agreement\n"
    "Gamma Systems, a Dutch corporation with its principal place of business in Haarlem [DISCLOSING PARTY] "
    "and the [RECEIVING PARTY] represented by the person signing below.\n"
)


def _payloads(text: str) -> list[dict]:
    """The text as one child payload per line (a cover's tiny chunks), in reverse order."""
    rows = [{"chunk_index": i, "text": t, "section_heading": "", "contains_pii": 0}
            for i, t in enumerate(text.splitlines())]
    return rows[::-1]


class TestTitle:
    @pytest.mark.parametrize("opening, title", [
        (SPONSORSHIP, "SPONSORSHIP AGREEMENT"),
        (MERGER_COVER, "AGREEMENT AND PLAN OF MERGER"),
        (TEMPLATE_NDA, "MUTUAL NON-DISCLOSURE AGREEMENT"),
        ("EXECUTION VERSION\nExhibit 10.4 MASTER SERVICES AGREEMENT\nThis Agreement ...", "MASTER SERVICES AGREEMENT"),
        ('Page 1\nThis Share Purchase Agreement (the "Agreement") is made on ...', "Share Purchase Agreement"),
    ])
    def test_the_documents_own_title(self, opening, title):
        assert extract_title(opening) == title

    def test_no_title_when_none_is_stated(self):
        assert extract_title("Minutes of the meeting held on 3 May.\nPresent: the board.") == ""
        assert extract_title('THIS AGREEMENT (the "Agreement") is made ...') == ""


class TestParties:
    def test_the_party_clause_with_its_short_names(self):
        parties = extract_parties(SPONSORSHIP)
        assert "ACME WIDGETS, INC." in parties and "NORTHWIND TRADERS LLC" in parties
        assert {"Acme", "Northwind"} <= set(parties)
        # The description and the address are not names.
        assert not any("Delaware" in p or "Main Street" in p or "Springfield" in p for p in parties)

    def test_a_cover_lists_its_parties(self):
        assert extract_parties(MERGER_COVER) == [
            "ATLAS HOLDINGS PLC", "ATLAS MERGER SUB, INC.", "BLUEWATER THERAPEUTICS, INC."]

    def test_a_template_keeps_the_party_it_names_and_skips_its_blanks_and_roles(self):
        assert extract_parties(TEMPLATE_NDA) == ["Harbor Analytics"]

    def test_a_party_introduced_by_its_description(self):
        assert extract_parties(DESCRIBED_ONLY) == ["Gamma Systems"]

    def test_roles_are_not_names(self):
        opening = 'This Agreement is between the Company ("Discloser") and the Recipient ("Receiving Party").'
        assert extract_parties(opening) == []

    def test_a_sentence_in_capitals_is_not_a_list_of_names(self):
        opening = ("CONFIDENTIALITY UNDERTAKING\nTHIS UNDERTAKING IS GIVEN BETWEEN NATURAL OR LEGAL PERSONS WHO ARE "
                   "GIVEN ACCESS, FOR WHATEVER PURPOSE, TO ANY INFORMATION CONSIDERED BY ACME SA TO BE CONFIDENTIAL "
                   "AND ACME HOLDINGS, LTD.")
        assert extract_parties(opening) == ["ACME HOLDINGS, LTD."]


class TestTokens:
    def test_case_accents_punctuation_and_legal_forms(self):
        assert name_tokens("AZUL LINHAS AÉREAS S/A") == {"azul", "linhas", "aereas"}
        assert name_tokens("i-on interactive, Inc.") >= {"ion", "interactive"}
        assert "inc" not in name_tokens("i-on interactive, Inc.")
        assert name_tokens("Coca-Cola") == {"coca", "cola", "cocacola"}

    @pytest.mark.parametrize("title, acronym", [
        ("Non-Disclosure Agreement", "nda"),
        ("Mutual Non-Disclosure Agreement", "nda"),
        ("Share Purchase Agreement", "spa"),
        ("DEPARTMENT OF HOMELAND SECURITY — NON-DISCLOSURE AGREEMENT", "dhs"),
        ("NON-DISCLOSURE AGREEMENT (NDA)", "nda"),
    ])
    def test_title_acronyms(self, title, acronym):
        assert acronym in title_acronyms(title)


class TestOpening:
    def test_chunks_in_order_pii_skipped_heading_reinstated(self):
        payloads = [
            {"chunk_index": 2, "text": "third", "section_heading": "", "contains_pii": 0},
            {"chunk_index": 0, "text": "This Agreement ...", "section_heading": "LEASE AGREEMENT",
             "contains_pii": 0},
            {"chunk_index": 1, "text": "call 555-0100", "section_heading": "", "contains_pii": 1},
        ]
        assert document_opening(payloads) == "LEASE AGREEMENT\nThis Agreement ...\nthird"

    def test_identity_from_shuffled_chunks_and_only_the_fields_found(self):
        identity = extract_document_identity(_payloads(MERGER_COVER))
        assert identity == {"doc_title": "AGREEMENT AND PLAN OF MERGER",
                            "doc_parties": ["ATLAS HOLDINGS PLC", "ATLAS MERGER SUB, INC.",
                                            "BLUEWATER THERAPEUTICS, INC."]}
        assert extract_document_identity(_payloads("minutes of a meeting, nothing more")) == {}
        assert extract_document_identity([]) == {}


class TestIngest:
    def test_every_child_carries_the_identity_never_the_file_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAYA_ENABLED", "0")
        path = tmp_path / "upload.txt"
        path.write_text(SPONSORSHIP + "\n" + "1. Term. The sponsorship runs for two seasons. " * 120,
                        encoding="utf-8")

        doc = extract_and_chunk(str(path), "Zenith_Corp_NDA_final_v3.txt", "deal-i")

        assert len(doc.chunks) > 1
        assert all(c["doc_title"] == "SPONSORSHIP AGREEMENT" for c in doc.chunks)
        assert all(c["doc_parties"] == doc.chunks[0]["doc_parties"] for c in doc.chunks)
        assert not any("zenith" in name_tokens(p) for p in doc.chunks[0]["doc_parties"])

    def test_a_document_without_identity_gets_no_field(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAYA_ENABLED", "0")
        path = tmp_path / "notes.txt"
        path.write_text("revenue grew in the quarter and costs fell. " * 80, encoding="utf-8")

        doc = extract_and_chunk(str(path), "notes.txt", "deal-i")

        assert not any(k in c for c in doc.chunks for k in ("doc_title", "doc_parties"))


def _vector() -> dict:
    return {"dense": [1.0] + [0.0] * (VECTOR_SIZE - 1), "sparse": SparseVector(indices=[1], values=[1.0])}


async def _index(client, payloads: list[dict]) -> None:
    await client.upsert(COLLECTION_NAME, points=[
        PointStruct(id=i, vector=_vector(), payload=p) for i, p in enumerate(payloads)])


class TestBackfill:
    @pytest.mark.asyncio
    async def test_backfill_writes_what_ingestion_writes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAYA_ENABLED", "0")
        path = tmp_path / "merger.txt"
        path.write_text(MERGER_COVER + "\n" + "Section 1.1 The Merger. Merger Sub merges into the Company. " * 150,
                        encoding="utf-8")
        ingested = extract_and_chunk(str(path), "merger.txt", "deal-b").chunks
        expected = {k: ingested[0][k] for k in ("doc_title", "doc_parties")}
        # The same document as an index written before Decision 86 holds it.
        old = [{k: v for k, v in c.items() if k not in ("doc_title", "doc_parties")} for c in ingested]
        client = AsyncQdrantClient(location=":memory:")
        await setup_collections(client)
        await _index(client, old)

        stats = await backfill_document_identity(client)
        records, _ = await client.scroll(COLLECTION_NAME, limit=500, with_payload=True)

        assert stats["documents"] == 1 and stats["updated"] == 1
        assert all({k: r.payload.get(k) for k in expected} == expected for r in records)
        # Idempotent.
        again = await backfill_document_identity(client)
        assert again["updated"] == 0 and again["unchanged"] == 1

    def test_the_command_line_on_a_local_store(self, tmp_path, capsys):
        import asyncio
        import json

        from src.vector_db.identity_backfill import main

        async def build():
            client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
            await setup_collections(client)
            await _index(client, [{"chunk_id": f"m-{i}", "deal_id": "d", "doc_id": "m", "chunk_index": i,
                                   "text": line, "contains_pii": 0}
                                  for i, line in enumerate(MERGER_COVER.splitlines())])
            await client.close()

        def printed() -> list[dict]:
            # The counts main() prints (log records go to stdout too, and carry a message).
            rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
            return [r for r in rows if "message" not in r]

        asyncio.run(build())
        assert main(["--path", str(tmp_path / "qdrant"), "--dry-run"]) == 0
        assert printed()[-1]["updated"] == 1
        assert main(["--path", str(tmp_path / "qdrant")]) == 0
        assert main(["--path", str(tmp_path / "qdrant")]) == 0
        assert printed()[-1]["unchanged"] == 1

    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing_and_stale_fields_are_removed(self):
        client = AsyncQdrantClient(location=":memory:")
        await setup_collections(client)
        await _index(client, [
            {"chunk_id": "n-0", "deal_id": "d", "doc_id": "notes", "chunk_index": 0, "text": "revenue grew",
             "contains_pii": 0, "doc_title": "OLD TITLE", "doc_parties": ["Old Party"]},
            {"chunk_id": "o-0", "deal_id": "other", "doc_id": "o", "chunk_index": 0, "text": MERGER_COVER,
             "contains_pii": 0},
        ])

        dry = await backfill_document_identity(client, deal_id="d", dry_run=True)
        (record,), _ = await client.scroll(COLLECTION_NAME, scroll_filter=None, limit=1, with_payload=True)
        assert dry["updated"] == 1 and record.payload.get("doc_title") == "OLD TITLE"

        await backfill_document_identity(client, deal_id="d")
        records, _ = await client.scroll(COLLECTION_NAME, limit=10, with_payload=True)
        by_doc = {r.payload["doc_id"]: r.payload for r in records}
        assert "doc_title" not in by_doc["notes"] and "doc_parties" not in by_doc["notes"]
        # Another deal is not touched.
        assert "doc_title" not in by_doc["o"]
