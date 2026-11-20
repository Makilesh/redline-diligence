"""
The Streamlit checklist tab (Decision 95): a row per contract and a status per
item; a cell's passage shown verbatim with its page or section; document text
shown as text; a grid that could not be fetched said to be missing, never
shown as one where nothing was found.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _panel(report):
    from app.components.checklist import render_checklist

    render_checklist(report)


ASSIGNMENT = "Neither party may assign this Agreement without the prior written consent of the other party."

REPORT = {
    "deal_id": "aurora_vertex_2024",
    "items": [
        {"id": "change_of_control", "label": "Change of control", "question": "Does a change of control let a party terminate?",
         "found": 1, "unclear": 0, "not_found": 1},
        {"id": "anti_assignment", "label": "Anti-assignment", "question": "Is consent required to assign the contract?",
         "found": 1, "unclear": 1, "not_found": 0},
    ],
    "rows": [
        {"doc_id": "d1", "source_file": "merger_agreement_v2_final.txt", "document_category": "legal", "cells": [
            {"item_id": "change_of_control", "status": "found", "score": 0.91, "answerability": None,
             "passage": {"text": "Upon a Change of Control, the Company may terminate.", "page_number": None,
                         "section_heading": "Section 7.2 Termination", "chunk_id": "c1", "score": 0.91},
             "alternates": [{"text": "A Change of Control means a merger.", "page_number": None,
                             "section_heading": "Definitions", "chunk_id": "c0", "score": 0.4}]},
            {"item_id": "anti_assignment", "status": "unclear", "score": 0.2, "answerability": None,
             "passage": {"text": "This Agreement binds successors.", "page_number": 12, "section_heading": "",
                         "chunk_id": "c4", "score": 0.2}, "alternates": []},
        ]},
        {"doc_id": "d2", "source_file": "credit_agreement_summary.pdf", "document_category": "financial", "cells": [
            {"item_id": "change_of_control", "status": "not_found", "score": 0.003, "answerability": None,
             "passage": {"text": "Interest accrues monthly.", "page_number": 3, "section_heading": "",
                         "chunk_id": "c9", "score": 0.003}, "alternates": []},
            {"item_id": "anti_assignment", "status": "found", "score": 0.95, "answerability": 0.9,
             "passage": {"text": ASSIGNMENT, "page_number": 8, "section_heading": "Assignment", "chunk_id": "c8",
                         "score": 0.95}, "alternates": []},
        ]},
    ],
    "documents_total": 9, "documents_checked": 2, "documents_not_checked": [],
    "documents_excluded": [{"source_file": "aurora_financials_fy2023.txt", "document_category": "financial"}],
    "excluded_total": 1, "truncated": False, "truncation_reason": "", "reranker": "bge-reranker-v2-m3",
    "answerability": False, "timings_ms": {}, "elapsed_ms": 1312.0,
}


def _run(report):
    at = AppTest.from_function(_panel, args=(report,), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return "\n".join(str(e.value) for e in [*at.markdown, *at.caption, *at.warning, *at.info, *at.success])


def test_a_row_per_contract_and_a_status_per_item():
    at = _run(REPORT)
    grid = at.dataframe[0].value
    assert list(grid["Contract"]) == ["merger_agreement_v2_final.txt", "credit_agreement_summary.pdf"]
    assert list(grid["Change of control"]) == ["● found", "· not found"]
    assert list(grid["Anti-assignment"]) == ["? unclear", "● found"]
    assert [str(m.value) for m in at.metric] == ["2", "2", "1", "1"]
    assert "Not in the grid: 1 document that is not a contract" in _text(at)


def test_a_cell_shows_its_passage_verbatim_with_where_it_is():
    at = _run(REPORT)
    at.selectbox(key="checklist_contract").set_value(1)
    at.selectbox(key="checklist_item").set_value(1)
    at.run()
    text = _text(at)
    assert "Found" in text
    assert "Neither party may assign this Agreement" in text
    assert "p\\. 8 · Assignment" in text


def test_a_partial_grid_names_what_it_did_not_check():
    at = _run({**REPORT, "truncated": True, "truncation_reason": "time_budget",
               "documents_not_checked": ["SUPPLY_AGREEMENT.pdf"]})
    warning = "\n".join(str(w.value) for w in at.warning)
    assert "Partial: checked 2 of 3 contracts" in warning
    assert "SUPPLY\\_AGREEMENT" in warning and "ran out of time" in warning


def test_a_grid_that_does_not_say_it_is_whole_is_not_taken_for_one():
    report = dict(REPORT)
    del report["truncated"]
    assert _run(report).warning


def test_an_unknown_status_is_unclear_never_found():
    report = {**REPORT, "rows": [{**REPORT["rows"][1], "cells": [
        {**REPORT["rows"][1]["cells"][0], "status": "probably"}, REPORT["rows"][1]["cells"][1]]}]}
    at = _run(report)
    assert list(at.dataframe[0].value["Change of control"]) == ["? unclear"]


def test_document_text_is_never_markup():
    hostile = '<img src=x onerror="alert(1)"> **Assignment** $5 and $6'
    report = {**REPORT, "rows": [{**REPORT["rows"][1], "source_file": "**bold** <b>x</b>.pdf", "cells": [
        REPORT["rows"][1]["cells"][0],
        {**REPORT["rows"][1]["cells"][1], "passage": {**REPORT["rows"][1]["cells"][1]["passage"], "text": hostile}},
    ]}], "truncated": True, "documents_not_checked": ["**bold** <b>x</b> $1 and $2.pdf"]}
    at = _run(report)
    assert list(at.dataframe[0].value["Contract"]) == ["**bold** <b>x</b>.pdf"]
    at.selectbox(key="checklist_item").set_value(1)
    at.run()
    shown = "\n".join(str(m.value) for m in at.markdown)
    # Escaped for markdown: no emphasis, no HTML, no LaTeX between the dollar signs.
    assert "\\*\\*Assignment\\*\\*" in shown and "\\<img" in shown and "\\$5" in shown
    warning = "\n".join(str(w.value) for w in at.warning)
    assert "\\*\\*bold\\*\\*" in warning and "\\<b\\>" in warning and "\\$1" in warning


def test_an_unavailable_grid_is_said_to_be_so():
    at = _run(None)
    assert any("could not be loaded" in str(i.value) for i in at.info)
    assert not at.dataframe


def test_a_deal_without_contracts_is_said_plainly():
    at = _run({**REPORT, "rows": []})
    assert any("No document of this deal reads as a contract" in str(i.value) for i in at.info)


def test_the_deal_view_builds_the_grid_on_request_and_survives_the_api_being_down():
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
    assert any("Checklist" in tab.label for tab in at.tabs)
    # Nothing is fetched until the grid is asked for.
    assert not any("checklist grid could not be loaded" in str(i.value) for i in at.info)
    at.button(key="checklist_build").click()
    at.run()
    assert not at.exception, at.exception
    assert any("checklist grid could not be loaded" in str(i.value) for i in at.info)
