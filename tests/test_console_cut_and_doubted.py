"""
The Streamlit console shows the two verification signals the web client
shows: an answer cut off at the model's output limit (`answer_truncated`,
Decision 66) and a claim whose wording the local checker doubted although its
figures matched (`suspected_misstatement`, Decision 64).

The console calls the blocking /query route, which never returns a cut answer,
so `answer_truncated` is false there today; it is rendered anyway so the
console cannot misreport one if it moves to the streamed route.
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

pytest.importorskip("streamlit.testing.v1")

from app.components.verification import DOUBTED_WORDING, answer_verdict  # noqa: E402

APP_PATH = str(_PROJECT_ROOT / "app" / "streamlit_app.py")
DEAL = "aurora_vertex_2024"

SUPPORTED = {
    "claim": "Revenue was $452.8 million in FY2023.", "status": "supported", "method": "numeric",
    "numbers": [{"raw": "$452.8 million", "status": "grounded", "binding": "bound"}],
    "reason": "", "evidence": "Total revenue 452.8", "evidence_source": "10-K.pdf",
    "evidence_page": 4, "evidence_section": "",
}
UNVERIFIED = {**SUPPORTED, "claim": "Margins expanded.", "status": "unverified", "method": "none",
              "numbers": [], "reason": "no check could settle it"}
DOUBTED = {**SUPPORTED, "claim": "The buyer pays the $13.92 million termination fee.",
           "status": "unverified", "method": "llm", "suspected_misstatement": True,
           "reason": "every figure is in the documents, but the passages holding them do not "
                     "support the statement as worded; not confirmed"}


def _result(**over):
    body = {
        "answer": "Revenue was $452.8 million in FY2023, while Adjusted EBITDA",
        "query_type": "financial", "confidence_score": 0.9, "validation_status": "passed",
        "citations": [], "hallucination_flags": [], "claim_checks": [SUPPORTED],
        "total_latency_ms": 1000.0, "session_id": "s", "rewrite_iterations": 0,
        "agent_trace": [], "is_refusal": False, "context_quality_score": 0.8,
        "outcome": "answered", "request_id": "req-7",
    }
    body.update(over)
    return body


class TestVerdict:
    @pytest.mark.parametrize("status", ["warning", "passed"])
    def test_a_cut_answer_is_never_verified_and_says_why(self, status):
        key, _, detail = answer_verdict(_result(validation_status=status, answer_truncated=True))
        assert key == "partial"
        assert detail == ("All 1 statement(s) matched to the documents, but the answer was "
                          "cut off before it finished.")

    def test_the_cut_off_is_added_to_other_reasons(self):
        _, _, detail = answer_verdict(_result(validation_status="warning", answer_truncated=True,
                                              claim_checks=[SUPPORTED, UNVERIFIED]))
        assert detail.endswith(
            "1 could not be confirmed. Also, the answer was cut off before it finished.")
        key, _, detail = answer_verdict(_result(validation_status="failed", answer_truncated=True,
                                                outcome="answered_unverified"))
        assert key == "failed"
        assert detail.endswith("Also, the answer was cut off before it finished.")

    def test_an_uncut_answer_is_unchanged(self):
        assert answer_verdict(_result())[0] == "verified"
        assert answer_verdict(_result(answer_truncated="yes"))[0] == "verified", "only true counts"


def _app_with(result: dict):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.query_params["deal_id"] = DEAL
    at.session_state["last_result"] = result
    at.session_state["last_result_deal"] = DEAL
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return "\n".join(str(p) for p in [m.value for m in at.markdown] + [c.value for c in at.caption])


class TestRendering:
    def test_cut_answer_and_doubted_claim_are_both_marked(self):
        text = _text(_app_with(_result(validation_status="warning", answer_truncated=True,
                                       claim_checks=[SUPPORTED, DOUBTED])))
        assert "Cut off here" in text
        assert "the answer was cut off before it finished" in text
        assert text.count(DOUBTED_WORDING) == 1, "only the doubted claim is marked"

    def test_neither_mark_on_a_clean_answer(self):
        text = _text(_app_with(_result()))
        assert "Cut off here" not in text
        assert DOUBTED_WORDING not in text
