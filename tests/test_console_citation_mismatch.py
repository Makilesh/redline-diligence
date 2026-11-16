"""
The Streamlit console shows a citation mismatch the way the web client does
(Decision 78): a claim with a figure that no document it cites states is
marked, listed with the statements that need attention, and named in the
verdict when it is the only reason the answer is not verified.
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

pytest.importorskip("streamlit.testing.v1")

from app.components.verification import (  # noqa: E402
    MISCITED,
    answer_verdict,
    citation_note,
    is_miscited,
)

APP_PATH = str(_PROJECT_ROOT / "app" / "streamlit_app.py")
DEAL = "aurora_vertex_2024"

SUPPORTED = {
    "claim": "Revenue was $452.8 million in FY2023.", "status": "supported", "method": "numeric",
    "numbers": [{"raw": "$452.8 million", "status": "grounded", "binding": "bound"}],
    "reason": "", "evidence": "Total revenue 452.8", "evidence_source": "10-K.pdf",
    "evidence_page": 4, "evidence_section": "",
}
MISCITED_CLAIM = {
    **SUPPORTED,
    "claim": "The $58.00 price is near the top of Vertex's $55-$60 indication.",
    "numbers": [{"raw": "$58.00", "status": "grounded", "binding": "unbound"}],
    "citation_mismatch": [{"figure": "$58.00", "found_in": ["merger_agreement_v2_final.txt"]}],
    "cited_documents": ["board_deck_strategic_review_mar2024.txt"],
}


def _result(**over):
    body = {
        "answer": "Revenue was $452.8 million in FY2023.",
        "query_type": "financial", "confidence_score": 0.9, "validation_status": "warning",
        "citations": [], "hallucination_flags": [], "claim_checks": [SUPPORTED, MISCITED_CLAIM],
        "total_latency_ms": 1000.0, "session_id": "s", "rewrite_iterations": 0,
        "agent_trace": [], "is_refusal": False, "context_quality_score": 0.8,
        "outcome": "answered", "request_id": "req-78",
    }
    body.update(over)
    return body


class TestVerdict:
    @pytest.mark.parametrize("status", ["warning", "passed"])
    def test_a_miscited_answer_is_not_verified_and_says_why(self, status):
        key, _, detail = answer_verdict(_result(validation_status=status))
        assert key == "partial"
        assert detail == ("All 2 statement(s) matched to the documents, but 1 cite(s) a document "
                          "that does not state the figure.")

    def test_a_clean_answer_is_unchanged(self):
        assert answer_verdict(_result(validation_status="passed", claim_checks=[SUPPORTED]))[0] == "verified"

    def test_the_note_names_where_the_figure_is(self):
        assert citation_note(MISCITED_CLAIM) == (
            "$58.00 is stated in merger_agreement_v2_final.txt, not in the cited "
            "board_deck_strategic_review_mar2024.txt."
        )
        assert citation_note(SUPPORTED) == ""
        assert not is_miscited({**SUPPORTED, "citation_mismatch": ["junk", {"found_in": []}]})


def _app_with(result: dict):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.query_params["deal_id"] = DEAL
    at.session_state["last_result"] = result
    at.session_state["last_result_deal"] = DEAL
    at.run()
    assert not at.exception, at.exception
    return at


class TestRendering:
    def test_the_miscited_claim_is_marked_and_explained_outside_the_fold(self):
        at = _app_with(_result())
        markdown = "\n".join(str(m.value) for m in at.markdown)
        captions = "\n".join(str(c.value) for c in at.caption)
        assert markdown.count(MISCITED) == 1, "only the miscited claim is marked"
        assert "merger\\_agreement\\_v2\\_final.txt" in captions or "merger_agreement_v2_final.txt" in captions
        # The supported claim folds away; the miscited one is listed before the fold.
        assert any("1 supported statement(s)" in e.label for e in at.expander)
