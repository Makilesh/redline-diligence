"""
A run refused only because the quality assessor reached no model is an outage,
not a refusal.

In the ambiguous band (best chunk 0.05-0.30) the assessor needs an LLM's second
opinion; without one it degrades to heuristic numbers that never admit context,
so the rewrite loop ends in insufficient_context whatever the documents hold.
That run used to answer "the relevant documents haven't been uploaded yet" with
outcome "refused", and both consoles showed INSUFFICIENT CONTEXT: on the free
tier, every question once the daily quota was spent (found by the round-12
Docker check, which ran with no LLM reachable). It now says the service failed
and reports outcome "synthesis_failed" — the documented "no model returned a
usable answer", which both consoles already present as a service problem.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

import api.main as api_main
import api.security as security
import platform_stubs
import src.agents.quality_assessor as qa
import src.workflow.orchestrator as orchestrator
from platform_stubs import DEMO_DEAL
from src.workflow.conditional_edges import route_after_quality_check

platform_environment = platform_stubs.platform_environment
stub_pipeline = platform_stubs.stub_pipeline

QUERY = {"query": "What is the interest rate margin on the term loan?", "deal_id": DEMO_DEAL}

DEGRADED = {"agent": "quality_assessor", "method": "heuristic", "score": 0.247,
            "degraded": True, "error": "connection"}
ASSESSED = {"agent": "quality_assessor", "method": "llm", "score": 0.1}
REFUSAL_HINT = "haven't been uploaded"


# ==============================================================================
# Which assessment counts
# ==============================================================================


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        ([], None),
        ([{"agent": "retrieval_executor"}], None),
        ([ASSESSED], None),
        ([DEGRADED], "connection"),
        ([{**DEGRADED, "error": "quota"}], "quota"),
        ([{**DEGRADED, "error": None}], "unknown"),
        ([{k: v for k, v in DEGRADED.items() if k != "error"}], "unknown"),
        # The last pass is the one the router refused on.
        ([DEGRADED, {"agent": "query_rewriter"}, ASSESSED], None),
        ([ASSESSED, {"agent": "query_rewriter"}, DEGRADED], "connection"),
        ([DEGRADED, {"agent": "query_rewriter", "degraded": True}], "connection"),
        (["not a dict", DEGRADED, None], "connection"),
    ],
)
def test_assessor_unavailable_kind_reads_the_last_assessment(trace, expected):
    assert orchestrator.assessor_unavailable_kind({"agent_trace": trace}) == expected


def test_a_missing_trace_is_no_outage():
    assert orchestrator.assessor_unavailable_kind({}) is None
    assert orchestrator.assessor_unavailable_kind({"agent_trace": None}) is None


# ==============================================================================
# The terminal node and the outcome
# ==============================================================================


@pytest.mark.asyncio
async def test_outage_refusal_names_the_service_not_the_documents():
    state = {"agent_trace": [DEGRADED], "rewrite_iteration": 2, "context_quality_score": 0.247}
    out = await orchestrator.insufficient_context_node(state)

    assert out["force_refusal"] is True and out["confidence_score"] == 0.0
    assert REFUSAL_HINT not in out["generated_answer"]
    assert "service problem" in out["generated_answer"]
    assert "Search attempts: 3" in out["generated_answer"]
    # The heuristic score is not a finding about the context here.
    assert "Best quality score" not in out["generated_answer"]
    assert out["agent_trace"] == [{
        "agent": "insufficient_context", "force_refusal": True,
        "reason": "assessor_unavailable", "error": "connection",
    }]
    assert orchestrator.run_outcome({**state, **out, "agent_trace": [DEGRADED, *out["agent_trace"]]}) \
        == "synthesis_failed"


@pytest.mark.asyncio
async def test_a_real_refusal_is_unchanged():
    state = {"agent_trace": [DEGRADED, {"agent": "query_rewriter"}, ASSESSED],
             "rewrite_iteration": 2, "context_quality_score": 0.1}
    out = await orchestrator.insufficient_context_node(state)

    assert out["generated_answer"] == (
        "I was unable to find sufficient relevant information in the data room "
        "to answer this question, even after refining the search. This may mean "
        "the relevant documents haven't been uploaded yet, or the question falls "
        "outside the scope of the available materials.\n\n"
        "Search attempts: 3\n"
        "Best quality score achieved: 0.10"
    )
    assert out["agent_trace"] == [{"agent": "insufficient_context", "force_refusal": True}]
    final = {**state, **out, "agent_trace": state["agent_trace"] + out["agent_trace"]}
    assert orchestrator.run_outcome(final) == "refused"


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        ([{"agent": "insufficient_context", "force_refusal": True}], "refused"),
        ([{"agent": "insufficient_context", "reason": "assessor_unavailable"}], "synthesis_failed"),
        # A reason on another agent's entry is not this one.
        ([{"agent": "quality_assessor", "reason": "assessor_unavailable"}], "refused"),
        ([{"agent": "insufficient_context", "reason": "other"}], "refused"),
    ],
)
def test_run_outcome_reads_the_terminal_reason(trace, expected):
    state = {"generated_answer": "No.", "force_refusal": True, "agent_trace": trace}
    assert orchestrator.run_outcome(state) == expected


@pytest.mark.asyncio
async def test_the_real_assessor_on_a_dead_ladder_ends_as_an_outage(monkeypatch):
    """The trace shape the detection reads is the one the real node writes."""
    monkeypatch.setenv("LAYA_GATE", "0")

    class Choice:
        model, api_key = "ollama/qwen2.5:14b", None

    class Tracker:
        async def get_model_for_agent(self):
            return Choice()

    async def get_instance():
        return Tracker()

    async def dead_ladder(**kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(qa.BudgetTracker, "get_instance", staticmethod(get_instance))
    monkeypatch.setattr(qa, "call_structured_agent", dead_ladder)

    state = {
        "original_query": QUERY["query"], "current_query": QUERY["query"],
        "query_type": "summary", "sub_questions": [], "rewrite_iteration": 2,
        "reranked_results": [
            {"chunk_id": "c0", "text": "Margin: SOFR + 3.25%.", "reranker_score": 0.2,
             "source_file": "credit_agreement_summary.txt"},
            {"chunk_id": "c1", "text": "Unrelated.", "reranker_score": 0.12},
        ],
        "context_quality_score": 0.0, "quality_breakdown": {},
        "force_refusal": False, "answerability_veto": False, "agent_trace": [],
    }
    assessed = await qa.quality_assessor_node(state)
    state = {**state, **assessed, "agent_trace": assessed["agent_trace"]}
    assert route_after_quality_check(state) == "insufficient_context"

    out = await orchestrator.insufficient_context_node(state)
    final = {**state, **out, "agent_trace": state["agent_trace"] + out["agent_trace"]}
    assert out["agent_trace"][0]["error"] == "connection"
    assert orchestrator.run_outcome(final) == "synthesis_failed"


# ==============================================================================
# Through the API: the blocking route and the stream
# ==============================================================================


@pytest.fixture
def client():
    return TestClient(api_main.app)


@pytest.fixture
def low_quality_graph(stub_pipeline, monkeypatch):
    """The real topology; the assessor scores in the band and may be degraded."""
    knobs = {"degraded": True}

    async def assessor(state):
        stub_pipeline.calls.append("quality_assessor")
        entry = dict(DEGRADED) if knobs["degraded"] else dict(ASSESSED)
        return {
            "context_quality_score": entry["score"],
            "quality_breakdown": {"relevance": 0.2, "completeness": 0.2, "precision": 0.2},
            "force_refusal": False,
            "agent_trace": [entry],
        }

    async def rewriter(state):
        stub_pipeline.calls.append("query_rewriter")
        return {
            "current_query": state["current_query"],
            "rewrite_iteration": state.get("rewrite_iteration", 0) + 1,
            "agent_trace": [{"agent": "query_rewriter", "degraded": knobs["degraded"]}],
        }

    monkeypatch.setattr(orchestrator, "quality_assessor_node", assessor)
    monkeypatch.setattr(orchestrator, "query_rewriter_node", rewriter)
    compiled = orchestrator.build_graph().compile(checkpointer=MemorySaver())
    monkeypatch.setattr(api_main, "_app_graph", compiled)
    return knobs


def test_blocking_query_reports_the_outage(client, low_quality_graph, stub_pipeline):
    body = client.post("/api/v1/query", json=QUERY).json()

    assert stub_pipeline.calls.count("quality_assessor") == 3
    assert "answer_synthesizer" not in stub_pipeline.calls
    assert body["outcome"] == "synthesis_failed"
    assert body["is_refusal"] is True and body["confidence_score"] == 0.0
    assert REFUSAL_HINT not in body["answer"] and "service problem" in body["answer"]
    terminal = [t for t in body["agent_trace"] if t.get("agent") == "insufficient_context"]
    assert len(terminal) == 1
    # `reason` survives the public allow-list (Decision 96); `error` is
    # redacted for public callers as every trace error is (scrub_public).
    assert terminal[0]["reason"] == "assessor_unavailable"
    assert terminal[0]["error"] == security.scrub_public({"error": "connection"})["error"]
    assert terminal[0]["force_refusal"] is True


def test_blocking_query_still_refuses_when_the_assessor_answered(
    client, low_quality_graph, stub_pipeline
):
    low_quality_graph["degraded"] = False
    body = client.post("/api/v1/query", json=QUERY).json()

    assert body["outcome"] == "refused" and body["is_refusal"] is True
    assert REFUSAL_HINT in body["answer"]


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for frame in text.split("\n\n"):
        name, data = None, None
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if name:
            out.append((name, data))
    return out


@pytest.mark.parametrize("degraded", [True, False])
def test_stream_reports_what_the_blocking_route_does(
    client, low_quality_graph, stub_pipeline, degraded
):
    low_quality_graph["degraded"] = degraded
    response = client.post("/api/v1/query/stream", json=QUERY)
    events = _events(response.text)

    assert events[-1][0] == "result"
    result = events[-1][1]
    stages = [d for name, d in events if name == "stage" and d.get("agent") == "insufficient_context"]
    assert len(stages) == 1
    if degraded:
        assert result["outcome"] == "synthesis_failed"
        assert stages[0]["summary"].startswith("Declined - the language models")
        assert REFUSAL_HINT not in result["answer"]
    else:
        assert result["outcome"] == "refused"
        assert stages[0]["summary"].startswith("Refused - no passage")
        assert REFUSAL_HINT in result["answer"]
