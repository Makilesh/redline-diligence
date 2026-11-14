"""
Shared stubs for the platform test files (lifecycle, checkpoints, access).

Not a test module and not conftest: these are helpers only the platform tests
need. Everything runs offline — the lifespan is never entered, the graph is the
real topology compiled over an in-memory checkpointer with stub nodes, and the
"LLM calls" are the wrapper's own usage record emitted by hand.
"""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver

import api.main as api_main
import api.routes.deals as deals
import api.security as security
import src.llm.litellm_wrapper as llm_wrapper
import src.workflow.orchestrator as orchestrator
from src.utils import metrics, runtime_status

DEMO_DEAL = "aurora_vertex_2024"
ADMIN_KEY = "test-admin-key"
SECRET_ERROR = "connection refused at http://10.0.0.7:6333 api_key=sk-live-123"


@pytest.fixture(autouse=True)
def platform_environment(monkeypatch):
    """An anonymous public caller in production mode, with clean process state."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    for name in (
        "ADMIN_API_KEY",
        "TRUST_PROXY_HEADERS",
        "PUBLIC_DEAL_IDS",
        "QUERY_TIMEOUT_SECONDS",
        "NODE_TIMEOUT_SECONDS",
        "SYNTHESIS_MODEL_PIN",
        "CHECKPOINT_KEEP_COMPLETED",
        "REQUIRE_DURABLE_CHECKPOINTS",
        "METRICS_PUBLIC",
        "ALLOW_MULTIPLE_WORKERS",
        "WEB_CONCURRENCY",
        "MAX_REGISTERED_DEALS",
        "MAX_JSON_BODY_KB",
        "MAX_UPLOAD_MB",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "RATE_LIMIT_QUERY_PER_MINUTE",
        "RATE_LIMIT_QUERY_PER_DAY",
        "RATE_LIMIT_DEAL_PER_MINUTE",
        "RATE_LIMIT_DEAL_PER_DAY",
        "RATE_LIMIT_READ_PER_MINUTE",
        "RATE_LIMIT_INGEST_PER_MINUTE",
        "RATE_LIMIT_INGEST_PER_DAY",
        "DAILY_QUERY_CAP",
    ):
        monkeypatch.setenv(name, "1000")
    monkeypatch.setenv("MAX_CONCURRENT_PIPELINES", "2")
    # Tests that compile against an unreachable Postgres should not wait out
    # the production connect timeout.
    monkeypatch.setenv("CHECKPOINTER_CONNECT_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(api_main, "_app_graph", object())

    def clean():
        security.reset_limits()
        metrics.reset_metrics()
        runtime_status.reset_status()
        deals._deals.clear()
        deals._documents.clear()

    clean()
    yield
    clean()


# ==============================================================================
# The real graph topology over stub nodes
# ==============================================================================


class StubPipeline:
    """Knobs for the stub nodes, and a record of what ran."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.synthesis_delay = 0.0
        self.fail_in: str | None = None
        self.context_chars = 2000
        self.quality_score: object = 0.9

    def _llm_call(self, model: str, agent: str) -> None:
        # Exactly what a completed call does in the wrapper: one usage record.
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        )
        llm_wrapper._log_usage(response, model, agent, time.monotonic())

    def _maybe_fail(self, node: str) -> None:
        self.calls.append(node)
        if self.fail_in == node:
            raise RuntimeError(SECRET_ERROR)

    async def query_intelligence(self, state):
        self._maybe_fail("query_intelligence")
        self._llm_call("gemini/lite", "query_intelligence")
        return {
            "query_type": "summary",
            "parsed_intent": {},
            "agent_trace": [{"agent": "query_intelligence", "model": "gemini/lite"}],
        }

    async def retrieval_executor(self, state):
        self._maybe_fail("retrieval_executor")
        return {
            "expanded_context": [{"text": "x" * self.context_chars}],
            "agent_trace": [{"agent": "retrieval_executor", "reranked_count": 1}],
        }

    async def quality_assessor(self, state):
        self._maybe_fail("quality_assessor")
        return {
            "context_quality_score": self.quality_score,
            "quality_breakdown": {"relevance": 0.9, "completeness": 0.9, "precision": 0.9},
            "agent_trace": [{"agent": "quality_assessor"}],
        }

    async def answer_synthesizer(self, state):
        self._maybe_fail("answer_synthesizer")
        await asyncio.sleep(self.synthesis_delay)
        self._llm_call("gemini/reasoning", "answer_synthesizer")
        self._llm_call("gemini/reasoning", "answer_synthesizer")
        return {
            "generated_answer": f"Answer to: {state['original_query']} [1]",
            "agent_trace": [{"agent": "answer_synthesizer"}],
        }

    async def hallucination_validator(self, state):
        self._maybe_fail("hallucination_validator")
        return {
            "validation_status": "passed",
            "validation_attempt": 1,
            "confidence_score": 0.9,
            "agent_trace": [{"agent": "hallucination_validator"}],
        }


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replaces the LLM/retrieval nodes with stubs; the wiring stays real."""
    stub = StubPipeline()
    for name in (
        "query_intelligence",
        "retrieval_executor",
        "quality_assessor",
        "answer_synthesizer",
        "hallucination_validator",
    ):
        monkeypatch.setattr(orchestrator, f"{name}_node", getattr(stub, name))
    return stub


@pytest.fixture
def graph(stub_pipeline, monkeypatch):
    """The compiled graph on an in-memory checkpointer, installed as the app's graph."""
    compiled = orchestrator.build_graph().compile(checkpointer=MemorySaver())
    monkeypatch.setattr(api_main, "_app_graph", compiled)
    return compiled


async def thread_ids(compiled) -> set[str]:
    return {
        item.config["configurable"]["thread_id"]
        async for item in compiled.checkpointer.alist(None)
    }


# ==============================================================================
# Driving the ASGI app with a client that can hang up
# ==============================================================================


class AsgiCall:
    """One in-flight request against the app, with a controllable disconnect."""

    def __init__(self, app, method: str, path: str, body: dict | None = None, headers=None):
        raw = json.dumps(body).encode() if body is not None else b""
        self._body = raw
        self._sent_body = False
        self._gone = asyncio.Event()
        self.messages: list[dict] = []
        scope = {
            "type": "http",
            # 2.3 is what uvicorn 0.49 reports: Starlette then watches the
            # receive channel and cancels the stream on http.disconnect.
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(raw)).encode()),
                *[(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
            ],
            "client": ("203.0.113.9", 4321),
            "server": ("testserver", 80),
        }
        self.task = asyncio.ensure_future(app(scope, self._receive, self._send))

    async def _receive(self) -> dict:
        if not self._sent_body:
            self._sent_body = True
            return {"type": "http.request", "body": self._body, "more_body": False}
        await self._gone.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: dict) -> None:
        self.messages.append(message)

    def disconnect(self) -> None:
        self._gone.set()

    @property
    def status(self) -> int | None:
        for message in self.messages:
            if message["type"] == "http.response.start":
                return message["status"]
        return None

    @property
    def text(self) -> str:
        return b"".join(
            m.get("body", b"") for m in self.messages if m["type"] == "http.response.body"
        ).decode()

    async def wait_for_text(self, needle: str, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while needle not in self.text:
            if time.monotonic() > deadline:
                raise AssertionError(f"{needle!r} never arrived; got {self.text!r}")
            await asyncio.sleep(0.01)

    async def finish(self, timeout: float = 5.0) -> None:
        await asyncio.wait_for(asyncio.shield(self.task), timeout)
