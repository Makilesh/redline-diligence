"""
The Streamlit console shows instruction-like document text the way the web
client does (Decision 82): the upload result says what the ingest screen found
and where, an answer that leans on such text says so in its verdict and in one
line naming the passage, and the claim resting on it is marked and listed
before the fold. Locations only — the API sends no flagged text.
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

pytest.importorskip("streamlit.testing.v1")

from app.components.document_uploader import _describe_ingest, _md_escape  # noqa: E402
from app.components.verification import (  # noqa: E402
    INJECTION_SUSPECT,
    answer_verdict,
    injection_note,
    needs_review,
)

APP_PATH = str(_PROJECT_ROOT / "app" / "streamlit_app.py")
DEAL = "aurora_vertex_2024"
MEMO = "northwind_information_memorandum.txt"

SUPPORTED = {
    "claim": "Customer count was 1,240 accounts.", "status": "supported", "method": "numeric",
    "numbers": [{"raw": "1,240", "status": "grounded", "binding": "bound"}],
    "reason": "", "evidence": "", "evidence_source": MEMO, "evidence_page": None,
    "evidence_section": "",
}
TAINTED = {
    **SUPPORTED,
    "claim": "Total revenue for FY2023 was $900.0 million.",
    "numbers": [{"raw": "$900.0 million", "status": "grounded", "binding": "bound"}],
    "injection_suspect": True,
}
REVIEW = {
    "flagged_passages": [{"source_file": MEMO, "page_number": None,
                          "section_heading": "FINANCIAL OVERVIEW",
                          "signals": ["instruction_override"]}],
    "cited_flagged_passages": 1,
    "tainted_figures": ["$900.0 million"],
}


def _result(**over):
    body = {
        "answer": "Total revenue for FY2023 was $900.0 million.",
        "query_type": "financial", "confidence_score": 0.9, "validation_status": "warning",
        "citations": [], "hallucination_flags": [], "claim_checks": [SUPPORTED, TAINTED],
        "total_latency_ms": 1000.0, "session_id": "s", "rewrite_iterations": 0,
        "agent_trace": [], "is_refusal": False, "context_quality_score": 0.8,
        "outcome": "answered", "request_id": "req-82", "injection_review": REVIEW,
    }
    body.update(over)
    return body


class TestVerdict:
    @pytest.mark.parametrize("status", ["warning", "passed"])
    def test_an_answer_resting_on_flagged_text_is_not_verified_and_says_why(self, status):
        key, _, detail = answer_verdict(_result(validation_status=status))
        assert key == "partial"
        assert detail == (
            "All 2 statement(s) matched to the documents, but $900.0 million rests only on text "
            "that reads like instructions to an AI model."
        )

    def test_citing_a_flagged_passage_alone_is_named(self):
        review = {**REVIEW, "tainted_figures": []}
        _, _, detail = answer_verdict(_result(claim_checks=[SUPPORTED], injection_review=review))
        assert detail == ("All 1 statement(s) matched to the documents, but it cites a passage "
                          "that reads like instructions to an AI model.")

    def test_a_flagged_passage_retrieved_and_not_used_leaves_the_answer_verified(self):
        quiet = {**REVIEW, "cited_flagged_passages": 0, "tainted_figures": []}
        result = _result(validation_status="passed", claim_checks=[SUPPORTED], injection_review=quiet)
        assert answer_verdict(result)[0] == "verified"
        assert injection_note(quiet) == (
            f"Retrieved but not cited: {MEMO} · FINANCIAL OVERVIEW contains text that reads like "
            "instructions to an AI model (an instruction to ignore earlier instructions).",
            "muted",
        )

    def test_an_answer_without_a_review_is_unchanged(self):
        result = _result(validation_status="passed", claim_checks=[SUPPORTED])
        del result["injection_review"]
        assert answer_verdict(result)[0] == "verified"
        assert injection_note(None) is None

    def test_the_note_names_where_and_what(self):
        assert injection_note(REVIEW) == (
            "This answer relies on text that reads like instructions to an AI model (an instruction "
            f"to ignore earlier instructions) in {MEMO} · FINANCIAL OVERVIEW — $900.0 million is "
            "stated nowhere else. Check those statements against the document itself.",
            "warn",
        )
        assert needs_review(TAINTED) and not needs_review(SUPPORTED)


class TestUpload:
    SCREEN = {
        "flagged_passages": 2,
        "signals": ["model_address", "role_marker"],
        "locations": [
            {"page_number": 3, "section_heading": "Fees [see](http://x)", "signals": ["model_address"]},
            {"page_number": None, "section_heading": "", "signals": ["role_marker"]},
        ],
    }

    def test_the_uploader_is_told_what_was_found_and_where(self):
        lines = _describe_ingest("memo.pdf", {"status": "success", "chunks_created": 9,
                                              "document_category": "legal",
                                              "injection_screen": self.SCREEN})
        assert lines[0].startswith("✅ **memo.pdf**")  # indexed, not failed
        assert lines[1] == (
            "  - ⚠️ 2 passage(s) read like instructions to an AI model (text addressed to an AI "
            f"model, chat-role labels such as SYSTEM:): {_md_escape('p. 3 · Fees [see](http://x)')}. "
            "The file was indexed as written; an answer that cites these passages, or takes a "
            "figure only from them, is never shown as verified."
        )
        # Document words render as text, not as a link.
        assert "[see](" not in lines[1]

    def test_nothing_flagged_adds_no_line(self):
        result = {"status": "success", "chunks_created": 9, "document_category": "legal"}
        assert len(_describe_ingest("memo.pdf", result)) == 1
        empty = {**result, "injection_screen": {"flagged_passages": 0, "signals": [], "locations": []}}
        assert len(_describe_ingest("memo.pdf", empty)) == 1


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
    def test_the_note_and_the_marker_are_shown_outside_the_fold(self):
        at = _app_with(_result())
        markdown = "\n".join(str(m.value) for m in at.markdown)
        assert markdown.count(INJECTION_SUSPECT) == 1, "only the tainted claim is marked"
        assert "This answer relies on text that reads like instructions to an AI model" in markdown
        assert any("1 supported statement(s)" in e.label for e in at.expander)
