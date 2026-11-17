"""
The Streamlit version browser says which documents the ingest screen flagged
(Decisions 76, 82) — after the upload response is gone, and without echoing
any flagged text (the API sends none).
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _browser(chain):
    from app.components.version_browser import render_version_browser

    render_version_browser(chain)


def _record(**extra):
    return {
        "doc_id": "doc-1234567890abcdef", "filename": "seller_memo.pdf", "document_category": "legal",
        "chunks_created": 9, "version_label": "v1", "upload_date": "", "is_current_version": True,
        "supersedes_doc_id": "", "superseded_by": "", "has_redline": False, **extra,
    }


def _text(chain) -> str:
    at = AppTest.from_function(_browser, args=(chain,), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return "\n".join(str(m.value) for m in [*at.markdown, *at.caption])


def test_a_flagged_document_is_labelled_with_what_was_found():
    text = _text([_record(instruction_like_passages=2, instruction_signals=["role_marker", "model_address"])])
    assert "INSTRUCTION-LIKE TEXT" in text
    assert "2 passages read like instructions to an AI model" in text
    assert "text addressed to an AI model, chat-role labels such as SYSTEM:" in text


def test_ordinary_and_older_records_carry_no_label():
    text = _text([_record(), _record(doc_id="doc-2", instruction_like_passages=0)])
    assert "INSTRUCTION-LIKE TEXT" not in text
    assert "like instructions" not in text
