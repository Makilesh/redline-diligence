"""
The Streamlit tie-out tab (Decision 92): conflicts first, each value with its
document and where in it; document text shown as text; a report that could
not be fetched said to be missing, never shown as a clean one.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _panel(report):
    from app.components.tie_out import render_tie_out

    render_tie_out(report)


REPORT = {
    "deal_id": "aurora_vertex_2024",
    "conflicts": [{
        "metric": "Adjusted EBITDA", "period": "FY2023", "kind": "amount", "status": "conflict",
        "max_deviation_pct": 1.72, "more_documents": 0,
        "values": [
            {"source_file": "aurora_financials_fy2023.txt", "page_number": None,
             "section_heading": "KEY FINANCIAL METRICS", "as_printed": "$97.3", "value": 97.3e6, "scale_known": True},
            {"source_file": "qoe.pdf", "page_number": 4, "section_heading": "", "as_printed": "$99.0",
             "value": 99.0e6, "scale_known": True},
        ],
    }],
    "agreements": [{
        "metric": "Net Income", "period": "FY2023", "kind": "amount", "status": "agree",
        "max_deviation_pct": 0.0, "more_documents": 0,
        "values": [
            {"source_file": "a.pdf", "page_number": 2, "section_heading": "", "as_printed": "$46.4",
             "value": 46.4e6, "scale_known": True},
            {"source_file": "b.pdf", "page_number": 9, "section_heading": "", "as_printed": "46.4",
             "value": 46.4, "scale_known": False},
        ],
    }],
    "conflicts_total": 1, "agreements_total": 1, "single_source_figures": 412, "ambiguous_figures": 3,
    "generic_figures": 7, "figures_without_period": 90, "documents_total": 9, "documents_read": 9,
    "documents_not_read": [], "chunks_read": 111, "truncated": False, "truncation_reason": "",
    "elapsed_ms": 41.2,
}


def _run(report):
    at = AppTest.from_function(_panel, args=(report,), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return "\n".join(str(e.value) for e in [*at.markdown, *at.caption, *at.warning, *at.info, *at.success])


def test_conflicts_are_listed_first_with_document_and_place():
    at = _run(REPORT)
    conflicts = at.dataframe[0].value
    assert list(conflicts["Figure"]) == ["Adjusted EBITDA", "Adjusted EBITDA"]
    assert list(conflicts["Document"]) == ["aurora_financials_fy2023.txt", "qoe.pdf"]
    assert list(conflicts["Where"]) == ["KEY FINANCIAL METRICS", "p. 4"]
    assert list(conflicts["As printed"]) == ["$97.3", "$99.0"]
    assert list(conflicts["Gap"]) == ["1.7%", "1.7%"]
    # Agreements come after, in their own table.
    assert list(at.dataframe[1].value["Figure"]) == ["Net Income", "Net Income"]
    assert "412 figures only one document states" in _text(at)


def test_a_partial_report_names_what_it_did_not_read():
    at = _run({**REPORT, "truncated": True, "truncation_reason": "chunk_cap", "documents_read": 2,
               "documents_not_read": ["JPMORGAN_2022_10K.pdf"]})
    warning = "\n".join(str(w.value) for w in at.warning)
    assert "Partial: read 2 of 9 documents" in warning
    assert "JPMORGAN" in warning and "larger than one report reads" in warning


def test_a_report_that_does_not_say_it_is_whole_is_not_taken_for_one():
    report = dict(REPORT)
    del report["truncated"]
    assert _run(report).warning


def test_document_text_is_never_markup():
    hostile = '<img src=x onerror="alert(1)"> **Revenue** $5 and $6'
    at = _run({**REPORT, "conflicts": [{**REPORT["conflicts"][0], "metric": hostile}],
               "truncated": True, "documents_not_read": ["**bold** <b>x</b> $1 and $2.pdf"]})
    assert list(at.dataframe[0].value["Figure"])[0] == hostile
    warning = "\n".join(str(w.value) for w in at.warning)
    # Escaped for markdown: no emphasis, no HTML, no LaTeX between the dollar signs.
    assert "\\*\\*bold\\*\\*" in warning and "\\<b\\>" in warning and "\\$1" in warning


def test_an_unavailable_report_is_said_to_be_so():
    at = _run(None)
    assert any("could not be loaded" in str(i.value) for i in at.info)
    assert not at.dataframe


def test_nothing_stated_twice_is_said_plainly():
    at = _run({**REPORT, "conflicts": [], "agreements": [], "conflicts_total": 0, "agreements_total": 0})
    assert any("No figure is stated by two documents" in str(i.value) for i in at.info)


def test_the_deal_view_has_the_tab_and_survives_the_api_being_down():
    # The app's own deal view: an answer on screen, the API unreachable (port 9).
    from pathlib import Path

    app = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
    at = AppTest.from_file(app, default_timeout=60)
    at.query_params["deal_id"] = "aurora_vertex_2024"
    at.session_state["last_result"] = {"answer": "Revenue was $452.8M.", "outcome": "answered",
                                       "validation_status": "passed", "citations": [], "agent_trace": []}
    at.session_state["last_result_deal"] = "aurora_vertex_2024"
    at.run()
    assert not at.exception, at.exception
    assert any("Tie-out" in tab.label for tab in at.tabs)
    assert any("could not be loaded" in str(i.value) for i in at.info)
