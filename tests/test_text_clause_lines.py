"""
Contract clauses on one line are prose, not table rows.

EDGAR text exhibits (and the LegalBench-RAG corpora built from them) write a
whole clause on one physical line and pad its enumeration with spaces. The
whitespace-table detector read "(e)    <clause mentioning Section 1504>" as a
two-cell row with a digit; on one real merger agreement 110 of 202 sections
became "tables" (no parent section, table-sibling expansion over whole
articles). The fixture below reproduces that layout; it is generated, not a
copy of a filing.
"""

from src.data_processing.text_processor import parse_text_sections

MERGER_AGREEMENT = """\
ARTICLE I DEFINITIONS

Section 1.1    Certain Definitions. For purposes of this Agreement, the following terms have the meanings set forth below.

(e)    "Affiliated Group" means an affiliated group as defined in Section 1504 of the Code (or any analogous combined, consolidated or unitary group under state, local or non-U.S. Law).

(f)    "Antitrust Law" means, collectively, the Sherman Antitrust Act of 1890, the Clayton Antitrust Act of 1914, the HSR Act and all other applicable Laws designed to prohibit or restrict monopolization.

(m)    "Capped Call Documentation" means the letter agreements related to call options on Company Common Stock relating to $500,000,000 aggregate principal amount of the Convertible Notes.  Each such letter agreement is dated as of September 15, 2020.
"""

FEE_TABLE = """\
FEES

Item                     FY2023      FY2022
Arrangement fee           1,250       1,100
Commitment fee              400         380
"""

TERM_SHEET = """\
CREDIT FACILITY TERMS

Facility:           $250 million senior secured revolving credit facility
Interest Rate:      Term SOFR plus an applicable margin of 225 basis points
Security:           First priority lien on substantially all assets
"""


def _tables(text: str) -> list[dict]:
    return [s for s in parse_text_sections(text) if s["is_table"]]


def test_enumerated_clauses_on_one_line_are_not_tables():
    sections = parse_text_sections(MERGER_AGREEMENT)
    assert not [s for s in sections if s["is_table"]]
    body = "\n".join(s["text"] for s in sections)
    for phrase in ("Section 1504 of the Code", "Clayton Antitrust Act of 1914",
                   "$500,000,000 aggregate principal amount", "September 15, 2020"):
        assert phrase in body


def test_clauses_stay_under_their_heading():
    sections = parse_text_sections(MERGER_AGREEMENT)
    assert {s["section_heading"] for s in sections} == {"ARTICLE I DEFINITIONS"}


def test_a_real_whitespace_table_is_still_a_table():
    tables = _tables(FEE_TABLE)
    assert len(tables) == 1
    assert "Arrangement fee" in tables[0]["text"] and "380" in tables[0]["text"]


def test_a_key_value_term_sheet_is_still_one_table():
    tables = _tables(TERM_SHEET)
    assert len(tables) == 1
    assert "225 basis points" in tables[0]["text"] and "First priority lien" in tables[0]["text"]


def test_a_paragraph_with_double_spaced_sentences_does_not_extend_a_table():
    text = FEE_TABLE + (
        "The fees above are payable quarterly in arrears on the last business day of each "
        "quarter.  Any fee not paid when due bears interest at the default rate set out "
        "in the agreement until it is paid in full.\n"
    )
    tables = _tables(text)
    assert len(tables) == 1
    assert "payable quarterly" not in tables[0]["text"]
