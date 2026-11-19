"""
Question forms (Decision 91): LegalBench-RAG questions rewritten as a deal team
types them, deterministically, with the ground truth untouched.
"""

from __future__ import annotations

import pytest

from eval import realdocs_data as data
from eval.question_forms import FORMS, question_form, short_form, strip_suffixes
from eval.run_realdocs_eval import parse_args

CUAD = ("Consider the Sponsorship Agreement between Hydron Technologies, Inc. and Miami Dolphins, Ltd.; "
        "What is the governing law for this contract?")
MAUD = ('Consider the Acquisition Agreement between Parent "LVMH Moët Hennessy-Louis Vuitton SE" and Target '
        '"Tiffany & Co."; What is the Definition of "Company Material Adverse Effect"')


class TestStripSuffixes:
    @pytest.mark.parametrize("text, expected", [
        ("Hydron Technologies, Inc. and Miami Dolphins, Ltd.", "Hydron Technologies and Miami Dolphins"),
        ("Tiffany & Co.", "Tiffany"),
        ("COMMUNITY BANKERS TRUST CORPORATION", "COMMUNITY BANKERS TRUST"),
        ("Morgan Stanley & Co. Incorporated", "Morgan Stanley"),
        ("Eagle Parent Holdings L.P.", "Eagle Parent Holdings"),
    ])
    def test_legal_suffixes_after_a_name(self, text, expected):
        assert strip_suffixes(text) == expected

    @pytest.mark.parametrize("text", [
        "shall be limited to technical information", "the co-branding fee",
        'the Definition of "Company Material Adverse Effect"', "the Company's representations",
    ])
    def test_words_that_are_no_suffixes(self, text):
        assert strip_suffixes(text) == text


class TestShortForm:
    def test_one_party_named_without_between(self):
        out = short_form(CUAD, "cuad_x")
        assert " between " not in out and "Inc." not in out and "Ltd." not in out
        assert out.endswith("; What is the governing law for this contract?")
        assert out in {
            "Consider the Hydron Technologies Sponsorship Agreement; What is the governing law for this contract?",
            "Consider the Sponsorship Agreement with Hydron Technologies; What is the governing law for this contract?",
            "Consider the Miami Dolphins Sponsorship Agreement; What is the governing law for this contract?",
            "Consider the Sponsorship Agreement with Miami Dolphins; What is the governing law for this contract?",
        }

    def test_quoted_parties_lose_their_quotes_roles_and_suffixes(self):
        out = short_form(MAUD, "maud_x")
        assert '"Company Material Adverse Effect"' in out           # the question's own quotes stay
        assert "Parent" not in out and "Target" not in out and " SE" not in out and "Co." not in out
        assert "Tiffany" in out or "LVMH" in out

    def test_a_non_disclosure_agreement_is_an_nda(self):
        assert short_form("Consider VELCO's Non-Disclosure Agreement; Does the document state X?", "c1") == \
            "Consider VELCO's NDA; Does the document state X?"

    def test_a_question_without_the_prefix_keeps_it(self):
        query = 'Consider "Fiverr"\'s privacy policy; who can see my information?'
        assert short_form(query, "p1") == query

    def test_a_trailing_purpose_is_no_party(self):
        query = ("Consider the Endorsement Agreement between Square Two Golf, Inc. and Kathy Whitworth with "
                 "Intellectual Property Rights and Confidentiality Provisions; Who owns the IP?")
        named = {short_form(query, f"e{i}").split(";")[0] for i in range(40)}
        assert all("Confidentiality Provisions" not in n and "Intellectual Property Rights" not in n for n in named)
        assert any("Kathy Whitworth" in n for n in named) and any("Square Two Golf" in n for n in named)

    def test_seeded_by_the_question_alone(self):
        # The same question id gives the same form, whatever else is asked and in
        # whatever order; the measured forms of Decision 91 are pinned.
        assert short_form(CUAD, "cuad_x") == short_form(CUAD, "cuad_x")
        assert short_form(
            "Consider the Wireless Content License Agreement between Twentieth Century Fox Licensing & "
            "Merchandising and Sorrent, Inc.; Does this contract include any revenue or profit-sharing "
            "arrangements?", "cuad_f1526") == (
            "Consider the Twentieth Century Fox Licensing & Merchandising Wireless Content License Agreement; "
            "Does this contract include any revenue or profit-sharing arrangements?")
        assert short_form("Consider the Non-Disclosure Agreement between IGC and LSE; Does the document allow "
                          "the Receiving Party to share some Confidential Information?", "contractnli_f0165") == (
            "Consider the NDA with IGC; Does the document allow the Receiving Party to share some Confidential "
            "Information?")


class TestQuestionForm:
    def test_every_form(self):
        assert question_form(CUAD, "q", "original") == CUAD
        assert question_form(CUAD, "q", "lower") == CUAD.lower()
        assert question_form(CUAD, "q", "upper") == CUAD.upper()
        assert question_form(CUAD, "q", "short-lower") == short_form(CUAD, "q").lower()
        with pytest.raises(ValueError):
            question_form(CUAD, "q", "title")

    def test_the_harness_option(self):
        args = parse_args(["legalbench", "--question-form", "short-lower"])
        assert args.question_form == "short-lower" and set(FORMS) >= {"original", "lower", "upper"}
        assert parse_args(["legalbench"]).question_form == "original"

    def test_the_ground_truth_is_untouched(self):
        from dataclasses import replace

        q = data.SpanQuery(id="cuad_x", subset="cuad", query=CUAD, snippets=[("cuad/a.txt", 3, 9)])
        lower = replace(q, query=question_form(q.query, q.id, "lower"))
        assert (lower.id, lower.subset, lower.snippets, lower.document) == (q.id, q.subset, q.snippets, q.document)
