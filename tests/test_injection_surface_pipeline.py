"""
Decision 82, through real ingestion, retrieval, synthesis (scripted LLM) and
validation: the ingest summary and the answer's injection review name where
the planted text is, and neither carries the planted text itself.

Uses eval/injection_fixtures.json (Decision 76) and its helpers.
"""

from __future__ import annotations

import pytest

import api.routes.ingest as ingest_routes
from api.models.response_models import IngestResponse
from api.routes.query import _build_response
from tests.test_injection_fixtures import FIXTURES, PASSING_ASSESSMENT, _ingest


def _strings(value) -> list[str]:
    """Every string in a JSON-like value, keys included."""
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (str(k), *_strings(v))]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _strings(v)]
    return [value] if isinstance(value, str) else []


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture_id", ["instruction_override", "hidden_white_text"])
async def test_the_summaries_carry_locations_and_no_flagged_text(
    fixture_id, offline_pipeline, tmp_path,
):
    fixture = FIXTURES[fixture_id]
    name = fixture["document"]["name"]
    p = offline_pipeline
    await p.load_data_room()
    result = await _ingest(p, fixture, tmp_path)

    # Ingest: what POST /ingest returns for this upload.
    screen = ingest_routes._injection_screen(result)
    assert screen.flagged_passages == len(result["suspected_injection"]) > 0
    assert set(fixture["expect_signals"]) <= set(screen.signals)
    if fixture["format"] == "pdf_white_text":
        assert {loc.page_number for loc in screen.locations} == {1}
    ingest_payload = IngestResponse(
        doc_id=result["doc_id"], deal_id="d", document_category=result["document_category"],
        chunks_created=result["chunks_created"], injection_screen=screen,
    ).model_dump()

    # Query: an "obedient" scripted model repeats the planted figure.
    p.llm.script("quality_assessor", PASSING_ASSESSMENT)
    p.llm.script("synthesis", fixture["compromised_answer"])
    run = await p.ask(fixture["question"])
    response = _build_response(run.state, session_id="s")

    review = response.injection_review
    assert name in {f.source_file for f in review.flagged_passages}
    assert fixture["injected_figure"] in review.tainted_figures
    assert response.validation_status == "warning"
    assert any(c.get("injection_suspect") for c in response.claim_checks), (
        "the claim resting on the planted figure is marked for the consoles"
    )

    # The flagged text, as the trace keeps it for the reviewer.
    trace = next(t for t in reversed(run.state["agent_trace"])
                 if t.get("agent") == "hallucination_validator")
    raw = [s["sentence"] for s in trace["injection_tainted_sentences"]]
    raw += [e for f in trace["suspected_injection"] for e in f["excerpts"]]
    raw += fixture["prompt_must_contain"]
    assert raw
    surfaced = _strings(ingest_payload) + _strings({
        "injection_review": review.model_dump(),
        "hallucination_flags": response.hallucination_flags,
    })
    for text in raw:
        assert not any(text in s for s in surfaced), text
