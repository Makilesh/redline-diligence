"""
Realistic documents through the real ingestion path (extract_and_chunk).

PDFs as data rooms actually contain them — a borderless two-page 10-K
statement, a ruled table split over a page break under a caption, a contract
whose clause runs over a page break, page footers — and the files that must be
refused with a reason the uploader can act on. No network, no models.
"""

from __future__ import annotations

import pytest

from src.data_processing.ingest_pipeline import (
    DocumentExtractionError,
    NoExtractableContentError,
    extract_and_chunk,
)
from src.data_processing.text_processor import decode_text_file
from tests.fixtures import stress_docs as sx

DEAL = "deal-robust"
MAX_TOKENS = 800


def _tables(doc) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {}
    for c in doc.chunks:
        if c.get("table_id"):
            out.setdefault(c["table_id"], {}).setdefault(c.get("table_representation"), []).append(c)
    return out


def _prose(doc) -> list[dict]:
    return [c for c in doc.chunks if not c["is_table"]]


@pytest.fixture(scope="module")
def statement(tmp_path_factory):
    path = sx.write_borderless_statement_pdf(tmp_path_factory.mktemp("pdf"))
    return extract_and_chunk(str(path), path.name, DEAL, "financial")


class TestBorderlessStatement:
    def test_statement_is_one_table_across_both_pages(self, statement):
        tables = _tables(statement)
        assert len(tables) == 1
        reps = next(iter(tables.values()))
        grid = reps["markdown"][0]
        assert grid["page_range"] == [1, 2]
        assert grid["scale_label"] == "thousands"
        text = grid["text"]
        # Every row keeps its own figures in its own columns, page 2 included.
        assert "| Year Ended December 31, 2023 | Year Ended December 31, 2022 |" in text
        assert "| Total revenues | 452,800 | 387,100 | 341,900 |" in text
        assert "| Cost of revenues | (181,100) | (158,300) | (143,600) |" in text
        assert "| Net income (loss) | 33,200 | 21,900 | (2,000) |" in text
        assert "| Diluted earnings (loss) per share | 1.38 | 0.92 | (0.09) |" in text
        assert "(in thousands, except per share data)" in text
        assert sx.STATEMENT_FOOTNOTE in text

    def test_scale_applies_to_amounts_only(self, statement):
        reps = next(iter(_tables(statement).values()))
        rows = {r["line_item"]: r for c in reps["row_by_row"] for r in c["structured_rows"]}
        revenue = rows["Total revenues"]["values"]["Year Ended December 31, 2023"]
        assert revenue["normalized_value"] == pytest.approx(452_800_000)
        assert revenue["period"] == "FY2023"
        eps = rows["Diluted earnings (loss) per share"]["values"]["Year Ended December 31, 2021"]
        assert eps["normalized_value"] == -0.09 and eps["scale_factor"] == 1.0
        shares = rows["Weighted-average diluted shares"]
        assert shares["unit"] == "count"

    def test_metrics_follow_the_years_not_the_column_order(self, statement):
        reps = next(iter(_tables(statement).values()))
        metrics = reps["metrics_summary"][0]["metrics"]
        assert metrics["Revenue YoY Year Ended December 31, 2023"]["value"] == 16.97
        assert metrics["Revenue YoY Year Ended December 31, 2022"]["value"] == 13.22
        assert metrics["Revenue CAGR"]["value"] == 15.08

    def test_figures_are_not_scattered_one_per_line_in_the_prose(self, statement):
        for chunk in _prose(statement):
            assert "452,800\n387,100" not in chunk["text"]
            assert "2023\n2022" not in chunk["text"]

    def test_running_footer_is_not_indexed(self, statement):
        for chunk in statement.chunks:
            assert "Form 10-K | Page" not in chunk["text"]

    def test_every_chunk_fits(self, statement):
        assert all(c["token_count"] <= MAX_TOKENS for c in statement.chunks)


class TestRuledTable:
    @pytest.mark.parametrize("repeat_header", [True, False])
    def test_split_table_is_one_table_under_its_heading_and_caption(self, tmp_path, repeat_header):
        path = sx.write_ruled_multipage_pdf(tmp_path, repeat_header=repeat_header)
        doc = extract_and_chunk(str(path), path.name, DEAL, "financial")

        tables = _tables(doc)
        assert len(tables) == 1
        reps = next(iter(tables.values()))
        grid = reps["markdown"][0]
        assert grid["section_heading"] == "Revenue by Segment"
        assert grid["page_range"] == [1, 2]
        # The unit is printed above the grid, outside it.
        assert grid["scale_label"] == "millions" and grid["currency"] == "USD"
        for label, *values in sx.RULED_ROWS:
            assert f"| {label} | {' | '.join(values)} |" in grid["text"]

    def test_table_text_is_not_indexed_a_second_time_as_prose(self, tmp_path):
        path = sx.write_ruled_multipage_pdf(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL, "financial")
        for chunk in _prose(doc):
            assert "Platform subscriptions" not in chunk["text"]
            assert "Marketplace fees" not in chunk["text"]


class TestLegalPdf:
    @pytest.fixture(scope="class")
    def contract(self, tmp_path_factory):
        path = sx.write_legal_pdf(tmp_path_factory.mktemp("legal"))
        return extract_and_chunk(str(path), path.name, DEAL, "legal")

    def test_clause_running_over_the_page_break_stays_whole(self, contract):
        by_clause = {c["clause_id"]: c for c in contract.chunks}
        clause = by_clause["8.2(b)"]
        assert "breaches of the Fundamental" in clause["text"]
        assert "Representations, which shall be capped at the Purchase Price." in clause["text"]
        assert clause["page_number"] == 1
        # Nothing on page 2 is left without a clause.
        assert not [c for c in contract.chunks if c["clause_id"] == "PREAMBLE" and c["page_number"] == 2]

    def test_sub_clauses_name_their_parent(self, contract):
        ids = {c["clause_id"] for c in contract.chunks}
        assert {"8.1", "8.2", "8.2(a)", "8.2(b)", "8.3"} <= ids
        assert "(a)" not in ids

    def test_page_footer_is_not_part_of_any_clause(self, contract):
        for chunk in contract.chunks:
            assert "CONFIDENTIAL - Project Aurora" not in chunk["text"]


class TestUnreadableFilesFailLoudly:
    def test_scanned_pdf(self, tmp_path):
        path = sx.write_scanned_pdf(tmp_path)
        with pytest.raises(NoExtractableContentError, match="no text layer.*OCR"):
            extract_and_chunk(str(path), path.name, DEAL)

    def test_partly_scanned_pdf_is_indexed_with_a_warning(self, tmp_path):
        path = sx.write_partly_scanned_pdf(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        assert any("500 Congress Avenue" in c["text"] for c in doc.chunks)
        assert any("Page(s) 2-3 of 3 have no text layer" in w for w in doc.warnings)

    def test_encrypted_pdf(self, tmp_path):
        path = sx.write_encrypted_pdf(tmp_path)
        with pytest.raises(DocumentExtractionError, match="password-protected"):
            extract_and_chunk(str(path), path.name, DEAL)

    def test_corrupt_pdf(self, tmp_path):
        path = sx.write_corrupt_pdf(tmp_path)
        with pytest.raises(DocumentExtractionError, match="not a readable PDF"):
            extract_and_chunk(str(path), path.name, DEAL)

    def test_encrypted_or_legacy_office_file(self, tmp_path):
        path = sx.write_encrypted_office_file(tmp_path)
        with pytest.raises(DocumentExtractionError, match="password-protected or in the legacy"):
            extract_and_chunk(str(path), path.name, DEAL)

    def test_binary_file_named_txt(self, tmp_path):
        path = sx.write_binary_as_txt(tmp_path)
        with pytest.raises(DocumentExtractionError, match="not text"):
            extract_and_chunk(str(path), path.name, DEAL)


class TestTextEncodings:
    @pytest.mark.parametrize("encoding", ["cp1252", "utf-16", "utf-8-sig", "utf-8"])
    def test_currency_signs_survive(self, tmp_path, encoding):
        path = sx.write_txt_encoded(tmp_path, encoding)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        text = " ".join(c["text"] for c in doc.chunks)
        assert "£25.0 million (€29.1 million)" in text
        assert "Company’s" in text and "§12" in text
        assert "\x00" not in text and "�" not in text

    def test_decoder_rejects_binary(self):
        with pytest.raises(ValueError):
            decode_text_file(bytes(range(256)) * 4)
