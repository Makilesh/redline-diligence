"""
The Streamlit console must not overstate a result, and must not render
counterparty text as HTML.

Pure rules are tested directly; rendering is driven through AppTest with a
result placed in session state (how the app caches the last answer), so no
API is needed.
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

pytest.importorskip("streamlit.testing.v1")

from app.components.verification import answer_verdict, claim_label, result_outcome  # noqa: E402
from app.styles import pill, stat_row  # noqa: E402

APP_PATH = str(_PROJECT_ROOT / "app" / "streamlit_app.py")
DEAL = "aurora_vertex_2024"

SUPPORTED = {
    "claim": "Revenue was $452.8 million in FY2023.", "status": "supported", "method": "numeric",
    "numbers": [{"raw": "$452.8 million", "status": "grounded", "binding": "bound"}],
    "reason": "", "evidence": "Total revenue 452.8", "evidence_source": "10-K.pdf",
    "evidence_page": 4, "evidence_section": "",
}
CONTRADICTED = {
    "claim": "The termination fee is $20 million.", "status": "contradicted", "method": "llm",
    "numbers": [], "reason": "Section 8.3 sets it at $13.92 million.", "evidence": "",
    "evidence_source": "merger_agreement.txt", "evidence_page": None,
    "evidence_section": "Section 8.3",
}


def _result(**over):
    body = {
        "answer": "Revenue was $452.8 million in FY2023.", "query_type": "financial",
        "confidence_score": 0.9, "validation_status": "passed", "citations": [],
        "hallucination_flags": [], "claim_checks": [SUPPORTED], "total_latency_ms": 1000.0,
        "session_id": "s", "rewrite_iterations": 0, "agent_trace": [], "is_refusal": False,
        "context_quality_score": 0.8, "outcome": "answered", "request_id": "req-7",
    }
    body.update(over)
    return body


class TestRules:
    def test_outcome_is_read_before_is_refusal(self):
        assert result_outcome(_result(outcome="synthesis_failed", is_refusal=True)) == "synthesis_failed"
        assert result_outcome(_result(outcome="refused", is_refusal=True)) == "refused"

    def test_outcome_is_derived_for_an_older_server(self):
        legacy = _result(agent_trace=[{"agent": "synthesis_unavailable"}], is_refusal=True)
        del legacy["outcome"]
        assert result_outcome(legacy) == "synthesis_failed"
        legacy = _result(
            agent_trace=[{"agent": "insufficient_context", "reason": "assessor_unavailable"}],
            is_refusal=True,
        )
        del legacy["outcome"]
        assert result_outcome(legacy) == "synthesis_failed"
        legacy = _result(agent_trace=[{"agent": "insufficient_context"}], is_refusal=True)
        del legacy["outcome"]
        assert result_outcome(legacy) == "refused"
        legacy = _result(validation_status="failed")
        del legacy["outcome"]
        assert result_outcome(legacy) == "answered_unverified"

    def test_verified_needs_a_pass_and_all_claims_supported(self):
        assert answer_verdict(_result())[0] == "verified"
        assert answer_verdict(_result(claim_checks=[]))[0] == "no_claims"
        assert answer_verdict(_result(claim_checks=[SUPPORTED, CONTRADICTED]))[0] == "partial"

    def test_missing_status_is_not_a_pass(self):
        assert answer_verdict(_result(validation_status=""))[0] == "unknown"
        assert answer_verdict(_result(validation_status=None))[0] == "unknown"

    def test_failed_verification(self):
        key, label, detail = answer_verdict(
            _result(outcome="answered_unverified", validation_status="failed",
                    claim_checks=[SUPPORTED, CONTRADICTED])
        )
        assert key == "failed"
        assert detail.startswith("1 statement(s) could not be matched")

    def test_claim_labels(self):
        assert claim_label(SUPPORTED) == ("SUPPORTED", "ok")
        assert claim_label(CONTRADICTED) == ("CONTRADICTED", "bad")
        not_checked = {"status": "unverified", "reason": "not checked: beyond the first 40"}
        assert claim_label(not_checked)[0] == "NOT CHECKED"
        assert claim_label({"status": "something_new"})[0] == "UNVERIFIED"

    def test_markup_helpers_escape_their_text(self):
        assert "<img" not in pill("<img src=x onerror=alert(1)>")
        assert "<script>" not in stat_row([("Deal", "<script>alert(1)</script>")])


def _app_with(result: dict, deal_id: str = DEAL):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.query_params["deal_id"] = deal_id
    at.session_state["last_result"] = result
    at.session_state["last_result_deal"] = deal_id
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [w.value for w in at.warning]
    return "\n".join(str(p) for p in parts)


NOT_FOUND_ANSWER = (
    "### Direct Answer\nThe provided documents do not contain the revenue share of the top "
    "ten customers for FY2023.\n\n### Supporting Evidence\nRevenue was $452.8 million in "
    "FY2023 [📄 10-K.pdf | p.4 | Income statement]."
)


class TestNotFound:
    """Decision 112: never verified, banner first, the explanation still shown."""

    def test_the_verdict_is_not_found_even_when_every_claim_passed(self):
        key, label, detail = answer_verdict(_result(outcome="not_found", answer=NOT_FOUND_ANSWER))
        assert (key, label) == ("not_found", "NOT FOUND")
        assert "not against every document in the data room" in detail

    def test_a_failed_validation_still_reads_as_failed(self):
        key, _, _ = answer_verdict(_result(outcome="not_found", validation_status="failed",
                                           claim_checks=[CONTRADICTED]))
        assert key == "failed"

    def test_rendering(self):
        text = _text(_app_with(_result(outcome="not_found", answer=NOT_FOUND_ANSWER,
                                       confidence_score=0.93)))
        assert "NOT FOUND IN THE PASSAGES SEARCHED" in text
        assert "do not contain the revenue share" in text
        assert "✓ VERIFIED" not in text
        assert "93%" not in text
        assert "INSUFFICIENT CONTEXT" not in text


CUT = {"passages_used": 6, "passages_available": 14, "chars_used": 30_000, "chars_available": 64_000}


class TestEvidenceCut:
    """Decision 118: an answer written from part of the evidence says so."""

    def test_the_rule_keeps_only_a_real_cut(self):
        from app.components.verification import evidence_cut

        assert evidence_cut(CUT) == CUT
        assert evidence_cut(None) is None and evidence_cut("6 of 14") is None
        assert evidence_cut({**CUT, "passages_used": 14, "chars_used": 64_000}) is None
        assert evidence_cut({**CUT, "passages_used": True}) == {**CUT, "passages_used": 0}

    def test_the_note_under_the_answer(self):
        text = _text(_app_with(_result(evidence_cut=CUT)))
        assert ("Written from 6 of the 14 passages retrieved (about 47% of their text): the local "
                "model's window could not hold them all.") in text

    def test_no_note_for_the_full_context(self):
        assert "passages retrieved (about" not in _text(_app_with(_result()))

    def test_a_not_found_is_weaker_still(self):
        _, _, detail = answer_verdict(_result(outcome="not_found", answer=NOT_FOUND_ANSWER, evidence_cut=CUT))
        assert detail.endswith("judged on only 6 of the 14 passages retrieved, so the rest may hold it.")


class TestRendering:
    def test_synthesis_failure_is_not_shown_as_a_refusal(self):
        at = _app_with(_result(outcome="synthesis_failed", is_refusal=True, claim_checks=[],
                               answer="I could not generate an answer."))
        text = _text(at)
        assert "MODELS UNAVAILABLE" in text
        assert "INSUFFICIENT CONTEXT" not in text
        assert "I could not generate an answer" not in text
        assert "req-7" in text

    def test_failed_answer_lists_the_contradicted_claim_and_its_source(self):
        at = _app_with(_result(outcome="answered_unverified", validation_status="failed",
                               claim_checks=[SUPPORTED, CONTRADICTED]))
        text = _text(at)
        assert "FAILED VERIFICATION" in text
        assert "CONTRADICTED" in text
        assert "The termination fee is" in text
        assert "merger_agreement.txt · Section 8.3" in text
        assert "✓ VERIFIED" not in text

    def test_empty_claim_list_is_not_verified(self):
        text = _text(_app_with(_result(claim_checks=[])))
        assert "NOTHING TO VERIFY" in text
        assert "✓ VERIFIED" not in text

    def test_hostile_document_text_is_escaped(self):
        hostile = "<img src=x onerror=alert(1)>"
        at = _app_with(_result(
            citations=[{"source_file": hostile, "section_heading": "Section 1", "page_number": 2}],
            claim_checks=[{**CONTRADICTED, "claim": hostile, "evidence_source": hostile}],
        ))
        html_out = "\n".join(str(m.value) for m in at.markdown)
        assert "<img" not in html_out
        assert "&lt;img" in html_out

    def test_deal_id_from_the_url_is_escaped(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.query_params["deal_id"] = "<img src=x onerror=alert(1)>"
        at.run()
        assert not at.exception
        assert not any("<img" in str(m.value) for m in at.markdown)
