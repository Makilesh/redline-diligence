"""
The registrant's name from a filing's signature page, when its cover prints
none (Decision 98).

Coca-Cola's 2017 10-K prints its name on the cover as an image, between the
Commission file number and "(Exact name of Registrant as specified in its
charter)", so the cover gave only the file number, and Decision 94's company
check never applied to that filing. Its signature page states the name in
text, in the fixed language of every 10-K, 10-Q and 8-K. No models, no store.
"""

from __future__ import annotations

import time

import pytest

import src.data_processing.filing_metadata as fm
from src.data_processing.filing_metadata import extract_filing_metadata
from src.data_processing.ingest_pipeline import extract_and_chunk
from src.verification.numeric_grounding import ContextIndex, ground_texts

# Page 1 of COCACOLA_2017_10K.pdf as the PDF processor reads it: the name is an
# image, so the file number's line runs straight into the name's label.
IMAGE_COVER = """UNITED STATES
SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-K
(Mark One)
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
For the fiscal year ended December 31, 2017
OR
TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
For the transition period from to
Commission File Number 001-02217
(Exact name of Registrant as specified in its charter)
Delaware  58-0628465
(State or other jurisdiction of incorporation or organization)  (I.R.S. Employer Identification No.)
One Coca-Cola Plaza, Atlanta, Georgia  30313
"""

NAMED_COVER = IMAGE_COVER.replace(
    "Commission File Number 001-02217\n", "Commission File Number 001-02217\nACME WIDGETS COMPANY\n")

SENTENCE = ("Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the "
            "Registrant has duly caused this report to be signed on its behalf by the\nundersigned, thereunto "
            "duly authorized.")

# Page 165, as read.
SIGNATURES = f"""ITEM 16.  FORM 10-K SUMMARY
None.
SIGNATURES
{SENTENCE}
THE COCA-COLA COMPANY
(Registrant)
By:  /s/ JAMES QUINCEY
James R. Quincey
President and Chief Executive Officer
Date: February 23, 2018
Pursuant to the requirements of the Securities Exchange Act of 1934, this report has been signed below by the
following persons on behalf of the Registrant and in the capacities and on the dates indicated.
"""


def _filing(cover: str = IMAGE_COVER, signatures: str = SIGNATURES, **signature_page) -> list[dict]:
    return [
        {"text": cover, "page_number": 1},
        {"text": "Item 1. Business. We make and sell beverages.", "page_number": 4},
        {"text": signatures, "page_number": 165, **signature_page},
    ]


def _signed(block: str) -> str:
    return extract_filing_metadata(_filing(signatures=block)).get("filing_company", "")


class TestSignaturePage:
    def test_the_name_a_cover_prints_as_an_image(self):
        meta = extract_filing_metadata(_filing())

        assert meta["filing_company"] == "THE COCA-COLA COMPANY"
        # Named, not keyed: the file number stays the only identity.
        assert meta["filing_entity_keys"] == ["sec:1-2217"]
        assert {k: v for k, v in meta.items() if k != "filing_company"} == extract_filing_metadata(
            _filing(signatures="None."))

    def test_the_cover_keeps_precedence(self):
        meta = extract_filing_metadata(_filing(cover=NAMED_COVER))

        assert meta["filing_company"] == "ACME WIDGETS COMPANY"
        assert meta["filing_entity_keys"] == ["sec:1-2217", "name:acme widgets"]

    @pytest.mark.parametrize("sentence, name", [
        # A 10-Q: no comma before "thereunto".
        ("Pursuant to the requirements of the Securities Exchange Act of 1934, the registrant has duly caused "
         "this report to be signed on its behalf by the undersigned thereunto duly authorized.\n"
         "JPMorgan Chase & Co.", "JPMorgan Chase & Co"),
        # An 8-K: "hereunto".
        ("the registrant has duly caused this report to be signed on its behalf by the undersigned hereunto duly "
         "authorized.\nBest Buy Co., Inc.", "Best Buy Co., Inc"),
        # The company as the subject, the report named in full, "on behalf of".
        ("PepsiCo has duly caused this Annual Report on Form 10-K to be signed on behalf of the undersigned, "
         "thereunto duly authorized.\nPepsiCo, Inc.", "PepsiCo, Inc"),
        # A date and a place before the full stop.
        ("the Registrant has duly caused this report to be signed on its behalf by the undersigned, thereunto duly "
         "authorized, in the City of Bolingbrook, State of Illinois, on March 24, 2023.\nULTA BEAUTY, INC.",
         "ULTA BEAUTY, INC"),
    ])
    def test_the_fixed_language_as_filings_print_it(self, sentence, name):
        assert _signed(f"SIGNATURES\n{sentence}\n(Registrant)\nBy: /s/ A. Person") == name
        assert _signed(f"SIGNATURES\n{sentence}\n(REGISTRANT)\nBy: /s/ A. Person") == name

    @pytest.mark.parametrize("block", [
        # Most signature pages print no label: nothing marks where the name ends.
        f"{SENTENCE}\nNIKE, INC.\nBy: /s/ MARK G. PARKER\nDate: July 24, 2018",
        f"{SENTENCE}\nDate: February 24, 2017\nSQUARE, INC.\nBy: /s/ Jack Dorsey",
        # The label far below the sentence.
        f"{SENTENCE}\nTHE COCA-COLA COMPANY " + "and its consolidated subsidiaries " * 4 + "(Registrant)",
        # The label without the sentence.
        "Exhibit 99\nTHE COCA-COLA COMPANY\n(Registrant)\nBy: /s/ JAMES QUINCEY",
        # Something that is not a name alone between them.
        f"{SENTENCE}\nDate: February 23, 2018 THE COCA-COLA COMPANY\n(Registrant)",
        f"{SENTENCE}\nBy: /s/ JAMES QUINCEY\n(Registrant)",
        f"{SENTENCE}\nTHE COCA-COLA COMPANY 165\n(Registrant)",
        f"{SENTENCE}\nTHE COMPANY\n(Registrant)",
        f"{SENTENCE}\n(Registrant)\nTHE COCA-COLA COMPANY",
    ])
    def test_nothing_but_a_labelled_name_is_read(self, block):
        assert _signed(block) == ""

    @pytest.mark.parametrize("block", [
        # Decision 98's review: an abbreviation's period is not the sentence's full stop.
        f"{SENTENCE[:-1]}, in Washington, D.C. FANNIE MAE (Registrant)",
        f"{SENTENCE[:-1]}, in the U.S. ACME CORP (Registrant)",
        # No full stop at all: the name's own periods are not one either.
        f"{SENTENCE[:-1]} AMAZON.COM, INC. (Registrant)",
        f"{SENTENCE[:-1]} ACME HOLDINGS INC. SUBSIDIARY CO (Registrant)",
        # A page break's running header or a heading before the name.
        f"{SENTENCE} Table of Contents ACME CORP (Registrant)",
        f"{SENTENCE} SIGNATURES ACME CORP (Registrant)",
        f"{SENTENCE} Index ACME CORP (Registrant)",
        f"{SENTENCE} Page ACME CORP (Registrant)",
        # Prose, and a sentence break inside what is left.
        f"{SENTENCE} The following is the Company's report ACME (Registrant)",
        f"{SENTENCE} Paul, State of Minnesota. 3M COMPANY (Registrant)",
        f"{SENTENCE} acme corp (Registrant)",
    ])
    def test_what_is_not_a_name_alone_gives_none(self, block):
        assert _signed(block) == ""

    @pytest.mark.parametrize("tail, name", [
        (", in the City of St. Paul, State of Minnesota. 3M COMPANY", "3M COMPANY"),
        (", in the City of St. Louis, State of Missouri. EMERSON ELECTRIC CO.", "EMERSON ELECTRIC CO"),
        (". J. C. PENNEY COMPANY, INC.", "J. C. PENNEY COMPANY, INC"),
        (". COCA-COLA FEMSA, S.A.B. DE C.V.", "COCA-COLA FEMSA, S.A.B. DE C.V"),
        (". AT&T INC.", "AT&T INC"),
        (". NESTLÉ S.A.", "NESTLÉ S.A"),
        (". AMAZON.COM, INC.", "AMAZON.COM, INC"),
        (". eBay Inc.", "eBay Inc"),
        (". McDonald's Corporation", "McDonald's Corporation"),
    ])
    def test_names_with_periods_initials_and_marks(self, tail, name):
        assert _signed(f"{SENTENCE[:-1]}{tail}\n(Registrant)\nBy: /s/ A. Person") == name

    def test_blocks_naming_two_registrants_give_none(self):
        other = SIGNATURES.replace("THE COCA-COLA COMPANY", "COCA-COLA ENTERPRISES, INC.")
        assert _signed(SIGNATURES + "\n" + other) == ""
        # The same registrant, printed two ways, is one name: the first as printed.
        same = SIGNATURES.replace("THE COCA-COLA COMPANY", "The Coca-Cola Company")
        assert _signed(SIGNATURES + "\n" + same) == "THE COCA-COLA COMPANY"

    def test_a_block_read_as_a_table_is_not_read(self):
        meta = extract_filing_metadata(_filing(is_table=1))
        assert "filing_company" not in meta

    def test_only_a_recognised_sec_cover(self):
        release = ("March 9, 2023\nBeauty Shop Announces Fourth Quarter Fiscal 2022 Results\n"
                   "BOLINGBROOK, Ill.--(BUSINESS WIRE)-- Beauty Shop, Inc. (NASDAQ: BSHP) today announced "
                   "financial results for the fourth quarter ended January 28, 2023.")
        meta = extract_filing_metadata(_filing(cover=release))
        assert meta["filing_company"] == "Beauty Shop, Inc"
        assert meta == extract_filing_metadata(_filing(cover=release, signatures="None."))
        contract = "AGREEMENT AND PLAN OF MERGER dated as of March 1, 2024 among Parent, Merger Sub and the Company."
        assert extract_filing_metadata(_filing(cover=contract)) == {}
        # An SEC cover of a form not modelled here gets nothing at all, as before.
        assert extract_filing_metadata(_filing(cover=IMAGE_COVER.replace("FORM 10-K", "FORM S-1"))) == {}

    def test_a_failure_keeps_what_the_cover_gave(self, monkeypatch):
        def boom(sections):
            raise KeyError("boom")

        expected = extract_filing_metadata(_filing(signatures="None."))
        monkeypatch.setattr(fm, "_signature_name", boom)
        monkeypatch.setenv("STRICT_OPTIONAL_STEPS", "0")
        assert extract_filing_metadata(_filing()) == expected
        monkeypatch.setenv("STRICT_OPTIONAL_STEPS", "1")
        with pytest.raises(KeyError):
            extract_filing_metadata(_filing())

    @pytest.mark.parametrize("unit", ["has duly caused this ", "to be signed on its behalf by the undersigned, ",
                                      "thereunto duly authorized ", "(Registrant) ", "authorized.(Registrant)",
                                      "a", "7", "Inc. ", "\n"])
    def test_linear_on_a_whole_document(self, unit):
        # A 10-K's text runs to a million characters; this reader scans all of it.
        filler = (unit * (300_000 // len(unit) + 1))[:300_000]
        sections = [{"text": IMAGE_COVER, "page_number": 1}, {"text": filler, "page_number": 100}]
        start = time.perf_counter()
        extract_filing_metadata(sections)
        assert time.perf_counter() - start < 2.0


class TestDownstream:
    def test_ingest_writes_it_on_every_chunk(self, tmp_path, monkeypatch):
        # A plain-text filing has no pages: the cover is read within its first
        # characters, the signature page wherever it is.
        monkeypatch.setenv("LAYA_ENABLED", "0")
        path = tmp_path / "10k.txt"
        path.write_text(IMAGE_COVER + "\n\nItem 1. Business\n\n" + "We make and sell beverages. " * 900
                        + "\n\n" + SIGNATURES, encoding="utf-8")
        doc = extract_and_chunk(str(path), path.name, "deal-s", "financial")

        assert len(doc.chunks) > 1
        assert all(c["filing_company"] == "THE COCA-COLA COMPANY" for c in doc.chunks)
        assert all(c["filing_entity_keys"] == ["sec:1-2217"] for c in doc.chunks)

    def test_the_company_check_now_reads_the_filing(self):
        # Decision 94's check withholds a figure-only confirmation when the
        # claim names another company than the filing's registrant; it needs
        # the registrant on the chunk.
        table = ("Values are as stated in the source, in millions.\n"
                 "Net operating revenues: Fiscal Year 2017=35,410 Fiscal Year 2016=41,863\n")
        meta = extract_filing_metadata(_filing())
        chunk = {"text": table, "source_file": "COCACOLA_2017_10K.pdf", "is_table": 1,
                 "filing_company": meta["filing_company"]}
        claim = "PepsiCo reported net operating revenues of $35,410 in fiscal 2017."
        [[check]] = ground_texts([claim], ContextIndex([chunk]))
        assert check.binding == "unbound" and "THE COCA-COLA COMPANY" in check.binding_note
        [[check]] = ground_texts(["Coca-Cola reported net operating revenues of $35,410 in fiscal 2017."],
                                 ContextIndex([chunk]))
        assert check.is_confirmed
        # Without the registrant on the chunk the wrong company was confirmed.
        del chunk["filing_company"]
        [[check]] = ground_texts([claim], ContextIndex([chunk]))
        assert check.is_confirmed

    def test_naming_one_registrant_does_not_vouch_for_anothers_figures(self):
        # Decision 98's review: once both filings name their registrant, a claim
        # naming Coca-Cola must not be confirmed by PepsiCo's figure.
        pep = {"text": "Values are as stated in the source, in millions.\n"
                       "Net revenue: Fiscal Year 2017=63,525 Fiscal Year 2016=62,799\n",
               "source_file": "filing_a.pdf", "is_table": 1, "filing_company": "PepsiCo, Inc"}
        ko = {"text": "Values are as stated in the source, in millions.\n"
                      "Goodwill: Fiscal Year 2017=9,401 Fiscal Year 2016=10,629\n",
              "source_file": "filing_b.pdf", "is_table": 1, "filing_company": "THE COCA-COLA COMPANY"}
        claim = "Coca-Cola reported net revenue of $63,525 and goodwill of $9,401 in fiscal 2017."
        checks = ground_texts([claim], ContextIndex([pep, ko]))[0]
        assert all(c.binding == "unbound" and "PepsiCo, Inc" in c.binding_note for c in checks)
        # Naming both registrants, or each figure's own, confirms.
        both = "Coca-Cola's goodwill was $9,401 and PepsiCo's net revenue $63,525 in fiscal 2017."
        assert all(c.is_confirmed for c in ground_texts([both], ContextIndex([pep, ko]))[0])
        own = "Coca-Cola reported goodwill of $9,401 in fiscal 2017."
        assert all(c.is_confirmed for c in ground_texts([own], ContextIndex([pep, ko]))[0])
