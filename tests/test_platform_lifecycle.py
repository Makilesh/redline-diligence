"""
Request lifecycle, observability and checkpoint retention.

What happens to a pipeline run when its client hangs up, when it runs out of
time, when the process is shutting down; what a finished run reports about
where its time and its LLM calls went; and what is left in the checkpointer
afterwards. The graph is the real topology over stub nodes (see
platform_stubs.py) on an in-memory checkpointer.
"""

import asyncio
import json
import logging

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import api.routes.query as query_routes
import api.security as security
import src.workflow.orchestrator as orchestrator
import platform_stubs
from platform_stubs import ADMIN_KEY, DEMO_DEAL, SECRET_ERROR, AsgiCall, thread_ids
from src.utils import metrics, runtime_status

# Fixtures defined in platform_stubs.py, bound here so pytest finds them.
platform_environment = platform_stubs.platform_environment
stub_pipeline = platform_stubs.stub_pipeline
graph = platform_stubs.graph

QUERY = {"query": "What is the revenue?", "deal_id": DEMO_DEAL}


@pytest.fixture
def client():
    # No `with`: entering the context would run the real lifespan.
    return TestClient(api_main.app)


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


# ==============================================================================
# Client disconnect
# ==============================================================================


@pytest.mark.asyncio
async def test_blocking_query_is_cancelled_when_the_client_disconnects(monkeypatch):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def slow_run(**kwargs):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(query_routes, "run_query", slow_run)
    monkeypatch.setattr(query_routes, "_DISCONNECT_POLL_SECONDS", 0.02)

    call = AsgiCall(api_main.app, "POST", "/api/v1/query", QUERY)
    await asyncio.wait_for(started.wait(), 5)
    assert security.pipelines_in_flight() == 1

    call.disconnect()
    await call.finish()

    assert cancelled.is_set(), "the pipeline kept running with nobody waiting for it"
    assert security.pipelines_in_flight() == 0
    assert metrics.QUERIES.value(transport="blocking", outcome="cancelled") == 1


@pytest.mark.asyncio
async def test_streamed_query_is_cancelled_when_the_client_disconnects(monkeypatch):
    cancelled = asyncio.Event()

    async def slow_stream(**kwargs):
        yield "start", {"session_id": kwargs["session_id"], "planned_stages": []}
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(query_routes, "stream_query", slow_stream)

    call = AsgiCall(api_main.app, "POST", "/api/v1/query/stream", QUERY)
    await call.wait_for_text("event: start")
    assert security.pipelines_in_flight() == 1

    call.disconnect()
    await call.finish()

    assert cancelled.is_set()
    assert security.pipelines_in_flight() == 0
    assert metrics.QUERIES.value(transport="stream", outcome="cancelled") == 1
    assert "event: result" not in call.text and "event: error" not in call.text


@pytest.mark.asyncio
async def test_disconnect_cancels_the_node_in_flight_in_the_real_graph(graph, stub_pipeline):
    # End to end through the real stream_query and LangGraph: the node that is
    # running when the client leaves must be cancelled, and the nodes after it
    # must never start.
    stub_pipeline.synthesis_delay = 60
    call = AsgiCall(api_main.app, "POST", "/api/v1/query/stream", QUERY)
    await call.wait_for_text('"agent": "quality_assessor"')

    call.disconnect()
    await call.finish()

    assert "hallucination_validator" not in stub_pipeline.calls
    assert security.pipelines_in_flight() == 0


# ==============================================================================
# Deadlines
# ==============================================================================


def test_blocking_query_times_out_with_504(client, graph, stub_pipeline, monkeypatch):
    stub_pipeline.synthesis_delay = 60
    monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "0.2")

    response = client.post("/api/v1/query", json=QUERY)

    assert response.status_code == 504
    assert response.headers[security.REQUEST_ID_HEADER] in response.json()["detail"]
    assert "hallucination_validator" not in stub_pipeline.calls
    assert security.pipelines_in_flight() == 0
    assert metrics.QUERIES.value(transport="blocking", outcome="timeout") == 1


def test_streamed_query_times_out_with_a_coded_error_event(
    client, graph, stub_pipeline, monkeypatch
):
    stub_pipeline.synthesis_delay = 60
    monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "0.2")

    response = client.post("/api/v1/query/stream", json=QUERY)
    events = _events(response.text)

    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "timeout"
    assert events[-1][1]["request_id"] == response.headers[security.REQUEST_ID_HEADER]
    assert not any(name == "result" for name, _ in events)
    assert metrics.QUERIES.value(transport="stream", outcome="timeout") == 1


@pytest.mark.asyncio
async def test_node_timeout_is_applied_at_compile_time(stub_pipeline, monkeypatch):
    # Port 9 refuses immediately, so this lands on the in-memory fallback.
    monkeypatch.setenv("NODE_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setenv("CHECKPOINTER_CONNECT_TIMEOUT_SECONDS", "1")
    stub_pipeline.synthesis_delay = 60
    compiled = await orchestrator.get_compiled_graph("postgresql://u:p@127.0.0.1:9/none")
    assert compiled.step_timeout == 0.2

    result = await orchestrator.run_query(compiled, "q", DEMO_DEAL, "s1", timeout_seconds=0)
    assert result["status"] == "error" and result["error_kind"] == "timeout"


def test_default_deadline_covers_a_pinned_measurement_run(monkeypatch):
    assert orchestrator.query_timeout_seconds() == 900
    # The slot reaper must never fire before the run's own deadline does.
    assert security._slot_stale_seconds() >= 900

    monkeypatch.setenv("SYNTHESIS_MODEL_PIN", "gemini/gemini-3.7-flash")
    assert orchestrator.query_timeout_seconds() == 2100
    assert security._slot_stale_seconds() > 2100

    monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "0")
    assert orchestrator.query_timeout_seconds() == 0
    assert security._slot_stale_seconds() == 900


# ==============================================================================
# Stream keep-alive, draining
# ==============================================================================


def test_idle_stream_sends_comment_heartbeats(client, graph, stub_pipeline, monkeypatch):
    stub_pipeline.synthesis_delay = 0.3
    monkeypatch.setenv("SSE_HEARTBEAT_SECONDS", "0.05")

    response = client.post("/api/v1/query/stream", json=QUERY)

    assert ": keepalive\n\n" in response.text
    # Comment frames carry no event and no data, so a client that parses
    # frames by their `data:` line (web/lib/api.ts) never sees them.
    events = _events(response.text)
    assert events[0][0] == "start" and events[-1][0] == "result"


def test_draining_process_refuses_new_runs(client, graph, stub_pipeline):
    runtime_status.set_phase("draining")

    blocking = client.post("/api/v1/query", json=QUERY)
    stream = client.post("/api/v1/query/stream", json=QUERY)

    assert blocking.status_code == 503 and stream.status_code == 503
    assert blocking.headers["Retry-After"] == "30"
    assert stub_pipeline.calls == []
    assert client.get("/health").json()["phase"] == "draining"


# ==============================================================================
# Per-node timings and LLM usage
# ==============================================================================


def test_response_reports_per_node_latency_and_llm_calls(client, graph, stub_pipeline):
    stub_pipeline.synthesis_delay = 0.2
    body = client.post("/api/v1/query", json=QUERY).json()

    rows = {row["agent"]: row for row in body["node_timings"]}
    assert list(rows) == [
        "query_intelligence",
        "retrieval_executor",
        "quality_assessor",
        "answer_synthesizer",
        "hallucination_validator",
    ]
    # Loose bound: Windows timers tick at ~15 ms.
    assert rows["answer_synthesizer"]["duration_ms"] >= 150
    assert rows["retrieval_executor"]["duration_ms"] < 100
    assert rows["answer_synthesizer"]["llm_calls"] == 2
    assert rows["answer_synthesizer"]["models"] == ["gemini/reasoning"]
    assert rows["query_intelligence"]["prompt_tokens"] == 100
    assert rows["retrieval_executor"]["llm_calls"] == 0
    assert body["llm_calls"] == 3
    assert body["request_id"]
    # The same numbers ride on each node's own trace entry.
    assert body["agent_trace"][0]["perf"]["llm_calls"] == 1
    # Existing trace keys are untouched.
    assert body["agent_trace"][0]["model"] == "gemini/lite"


def test_stream_stage_events_carry_the_nodes_own_measurement(client, graph, stub_pipeline):
    response = client.post("/api/v1/query/stream", json=QUERY)
    stages = {data["agent"]: data for name, data in _events(response.text) if name == "stage"}

    assert stages["answer_synthesizer"]["perf"]["llm_calls"] == 2
    assert stages["answer_synthesizer"]["duration_ms"] >= 0  # unchanged field
    result = _events(response.text)[-1][1]
    assert result["llm_calls"] == 3 and len(result["node_timings"]) == 5


def test_llm_usage_record_contract():
    # metrics.py counts LLM calls from this log record instead of a hook in the
    # LLM layer. If the record is renamed or loses a field, this fails — rather
    # than every llm_calls quietly becoming 0.
    import src.llm.litellm_wrapper as llm_wrapper

    assert llm_wrapper.logger.name == metrics.LLM_USAGE_LOGGER
    seen: list[logging.LogRecord] = []

    class Capture(logging.Filter):
        def filter(self, record):
            seen.append(record)
            return True

    capture = Capture()
    llm_wrapper.logger.addFilter(capture)
    try:
        llm_wrapper._log_usage(
            type("R", (), {"usage": {"prompt_tokens": 7, "completion_tokens": 3}})(),
            "gemini/x",
            "answer_synthesizer",
            0.0,
        )
    finally:
        llm_wrapper.logger.removeFilter(capture)

    record = seen[-1]
    assert record.msg == metrics.LLM_USAGE_MESSAGE
    assert (record.model, record.agent) == ("gemini/x", "answer_synthesizer")
    assert (record.prompt_tokens, record.completion_tokens) == (7, 3)
    assert record.latency_ms >= 0


def test_metrics_endpoint_is_admin_only_and_reports_node_latency(
    client, graph, stub_pipeline, monkeypatch
):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    client.post("/api/v1/query", json=QUERY)

    assert client.get("/metrics").status_code == 403

    admin = {security.ADMIN_HEADER: ADMIN_KEY}
    text = client.get("/metrics", headers=admin).text
    assert 'redline_queries_total{outcome="answered",transport="blocking"} 1' in text
    assert 'redline_node_duration_seconds_count{node="answer_synthesizer"} 1' in text
    assert 'redline_llm_calls_total{model="gemini/reasoning",node="answer_synthesizer"} 2' in text
    assert "redline_pipelines_in_flight 0" in text

    as_json = client.get("/metrics?format=json", headers=admin).json()
    nodes = {s["labels"]["node"]: s for s in as_json["redline_node_duration_seconds"]}
    assert nodes["answer_synthesizer"]["count"] == 1
    assert nodes["answer_synthesizer"]["p95_le_seconds"] is not None

    monkeypatch.setenv("METRICS_PUBLIC", "1")
    assert client.get("/metrics").status_code == 200


def test_log_level_is_applied_to_output_but_usage_records_still_flow(monkeypatch):
    from src.utils.logger import setup_logger

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    logger = setup_logger("platform_test_logger")
    assert all(h.level == logging.WARNING for h in logger.handlers)
    # INFO records must still be created, or the LLM usage tap goes blind.
    assert logger.isEnabledFor(logging.INFO)


# ==============================================================================
# Checkpoint retention
# ==============================================================================


@pytest.mark.asyncio
async def test_reused_session_does_not_accumulate_previous_runs(graph, monkeypatch):
    # LangGraph applies a run's input on top of the thread's existing state,
    # and agent_trace has an operator.add reducer: the second question in a
    # session used to return the first question's trace too.
    monkeypatch.setenv("CHECKPOINT_KEEP_COMPLETED", "1")
    first = await orchestrator.run_query(graph, "first question", DEMO_DEAL, "session-1")
    second = await orchestrator.run_query(graph, "second question", DEMO_DEAL, "session-1")

    assert len(first["agent_trace"]) == 5
    assert len(second["agent_trace"]) == 5
    assert "second question" in second["generated_answer"]
    # One run's worth of checkpoints on the thread, not two.
    kept = [c async for c in graph.checkpointer.alist(None)]
    assert await thread_ids(graph) == {f"{DEMO_DEAL}:session-1"}
    assert len(kept) <= 8


@pytest.mark.asyncio
async def test_completed_runs_leave_no_checkpoints_by_default(graph):
    await orchestrator.run_query(graph, "q", DEMO_DEAL, "s1")
    async for _ in orchestrator.stream_query(graph, "q", DEMO_DEAL, "s2"):
        pass

    assert await thread_ids(graph) == set()
    assert metrics.CHECKPOINT_THREADS_DELETED.value(reason="completed") == 2


@pytest.mark.asyncio
async def test_failed_run_keeps_its_checkpoints_until_purge_or_ttl(graph, stub_pipeline):
    stub_pipeline.fail_in = "answer_synthesizer"
    sandbox = "sbx-65000000-" + "ab" * 16
    failed = await orchestrator.run_query(graph, "q", sandbox, "s1")
    await orchestrator.run_query(graph, "q", DEMO_DEAL, "s1")  # also fails
    assert failed["status"] == "error"
    assert await thread_ids(graph) == {f"{sandbox}:s1", f"{DEMO_DEAL}:s1"}

    # Purging the deal removes its threads and nobody else's.
    assert await orchestrator.purge_deal_checkpoints(graph, sandbox) == 1
    assert await thread_ids(graph) == {f"{DEMO_DEAL}:s1"}

    # Not yet expired; then expired.
    assert await orchestrator.sweep_stale_checkpoints(graph, max_age_seconds=3600) == 0
    await asyncio.sleep(0.05)
    assert await orchestrator.sweep_stale_checkpoints(graph, max_age_seconds=0.01) == 1
    assert await thread_ids(graph) == set()


def test_thread_ids_cannot_collide_across_deal_session_pairs():
    # Both ids allow "_" and "-", so "<deal>_<session>" was ambiguous.
    assert orchestrator.thread_id_for("sbx-1_a", "b") != orchestrator.thread_id_for("sbx-1", "a_b")


@pytest.mark.asyncio
async def test_retention_failure_never_fails_the_query(graph, monkeypatch):
    async def broken_delete(thread_id):
        raise ConnectionError(SECRET_ERROR)

    monkeypatch.setattr(graph.checkpointer, "adelete_thread", broken_delete)
    result = await orchestrator.run_query(graph, "q", DEMO_DEAL, "s1")
    assert result["status"] == "completed"


# ==============================================================================
# Checkpointer degradation and recovery
# ==============================================================================


@pytest.mark.asyncio
async def test_checkpointer_fallback_is_recorded_not_just_logged(stub_pipeline):
    compiled = await orchestrator.get_compiled_graph("postgresql://u:p@127.0.0.1:9/none")

    status = runtime_status.describe()
    assert compiled is not None
    assert status["checkpointer"]["backend"] == "memory"
    assert status["checkpointer"]["durable"] is False
    assert "checkpointer_in_memory" in status["degraded"]
    assert metrics.CHECKPOINTER_DURABLE.value() == 0
    # The class name is reported; the message (which holds the URL) is not.
    assert "127.0.0.1" not in json.dumps(status)


@pytest.mark.asyncio
async def test_require_durable_checkpoints_refuses_to_start(stub_pipeline, monkeypatch):
    monkeypatch.setenv("REQUIRE_DURABLE_CHECKPOINTS", "1")
    with pytest.raises(RuntimeError, match="REQUIRE_DURABLE_CHECKPOINTS"):
        await orchestrator.get_compiled_graph("postgresql://u:p@127.0.0.1:9/none")


def test_dead_checkpointer_connection_triggers_a_rebuild(client, graph, monkeypatch):
    # The Postgres saver holds one connection and never reconnects: after a
    # Postgres restart every checkpoint write — so every query — failed until
    # the API was restarted too.
    class DeadConnection:
        closed = True

    broken = type("Graph", (), {"checkpointer": type("Saver", (), {"conn": DeadConnection()})()})()
    monkeypatch.setattr(api_main, "_app_graph", broken)
    rebuilt_with: list[str] = []

    async def fake_compile(postgres_url):
        rebuilt_with.append(postgres_url)
        return graph

    monkeypatch.setattr(api_main, "get_compiled_graph", fake_compile)

    assert orchestrator.checkpointer_connection_broken(broken) is True
    assert orchestrator.checkpointer_connection_broken(graph) is False

    response = client.post("/api/v1/query", json=QUERY)
    assert response.status_code == 200
    assert len(rebuilt_with) == 1
    assert api_main.get_graph() is graph
