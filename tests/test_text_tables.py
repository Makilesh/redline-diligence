"""Plain-text tables keep their column header with their rows."""

from src.data_processing.text_processor import parse_text_sections

COMPENSATION = """\
SECTION 1: EXECUTIVE COMPENSATION SUMMARY (FY2023)
================================================================================

Name                Title                    Base Salary   Cash Bonus   Equity Value
------------------  -----------------------  -----------   ----------   ------------
Elena Marsh         Chief Executive Officer     $625,000     $437,500      $8,120,000
Robert Chen         Chief Financial Officer     $445,000     $267,000      $2,940,000

Equity value is calculated at the merger consideration of $58.00 per share.
"""


def _tables(text: str) -> list[dict]:
    return [s for s in parse_text_sections(text) if s["is_table"]]


def test_a_header_without_digits_stays_with_its_rows():
    tables = _tables(COMPENSATION)
    assert len(tables) == 1
    body = tables[0]["text"]
    assert "Base Salary" in body and "Cash Bonus" in body
    assert body.index("Base Salary") < body.index("$625,000")


def test_the_header_is_not_also_left_behind_as_prose():
    prose = [s["text"] for s in parse_text_sections(COMPENSATION) if not s["is_table"]]
    assert not any("Base Salary" in p for p in prose)
    assert any("merger consideration" in p for p in prose)


def test_a_prose_line_above_a_table_is_not_taken_as_its_header():
    text = (
        "The following table summarises revenue by segment.\n"
        "Segment            FY2023     FY2022\n"
        "Cloud               312.4      270.1\n"
    )
    tables = _tables(text)
    assert len(tables) == 1
    assert "summarises" not in tables[0]["text"]
