"""
Shared pytest fixtures for M&A Due Diligence Intelligence Engine tests.
"""

import os
from dataclasses import dataclass

import pytest
import pytest_asyncio

# Decision-model (Laya) paths are stubbed in unit tests. Without this default an
# unstubbed path — the API's query guard, say — loads the real 1.7GB checkpoint
# on a developer machine, which is slow and differs from CI. Tests that want the
# real model set LAYA_ENABLED=1 themselves and are marked `models`.
os.environ.setdefault("LAYA_ENABLED", "0")

# The Streamlit UI tests run with the backing API deliberately down. Without a
# default they target localhost:8000 and pass or fail on whether a developer's
# API happens to be running there. Port 9 (discard) is never served.
os.environ.setdefault("API_URL", "http://127.0.0.1:9")

# The API lifespan loads several GB of weights unless told not to. No offline
# test enters the lifespan today; this keeps a future one from doing so by
# accident on a machine that has the weights cached.
os.environ.setdefault("WARM_MODELS", "0")

# `import litellm` calls dotenv.load_dotenv(), which walks up from litellm's own
# install directory. With the venv inside the checkout that finds the
# developer's .env and loads all of it into the test process — measured: 25
# variables, among them real GEMINI_API_KEYS, ADMIN_API_KEY, ENVIRONMENT,
# VERIFY_NLI, VERIFICATION_BACKEND, POSTGRES_URL and QDRANT_URL. A local run then
# tests a different configuration than CI (which has no .env), and any unmocked
# LLM path spends real quota — the thing CI blanks the keys to prevent.
# litellm skips the load outside its "DEV" mode, and reads the variable for
# nothing else. Live tests need the real keys, so they keep the default.
if os.getenv("RUN_LIVE_TESTS", "").strip().lower() not in {"1", "true", "yes"}:
    os.environ.setdefault("LITELLM_MODE", "PRODUCTION")


# Scripts that live in tests/ but are not test modules. test_pipeline_offline.py
# matches the test_*.py glob yet defines no tests — it is a CLI validation run
# (`python tests/test_pipeline_offline.py`) — so collecting it only imports the
# ingestion stack for nothing.
collect_ignore = ["test_pipeline_offline.py"]


def pytest_collection_modifyitems(config, items):
    """
    Skips `@pytest.mark.live` tests unless RUN_LIVE_TESTS=1.

    Live tests need real services or credentials. Without this they would fail
    — not skip — on any machine that lacks them, which trains people to ignore
    red runs. CI additionally deselects them with `-m "not live and not models"`.
    """
    if _live_tests_enabled():
        return
    skip_live = pytest.mark.skip(reason="live test: set RUN_LIVE_TESTS=1 to run")
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(skip_live)


def _live_tests_enabled() -> bool:
    return os.getenv("RUN_LIVE_TESTS", "").strip().lower() in {"1", "true", "yes"}


@pytest.fixture(autouse=True, scope="session")
def _dead_api_refuses_at_once():
    """
    Makes "the API is down" cost nothing instead of two seconds per request.

    The UI tests point the Streamlit app at API_URL with nothing listening, and
    the app degrades each panel on `requests.ConnectionError`. On Linux a
    refused loopback connection fails in microseconds. On Windows the TCP stack
    retransmits the SYN before reporting the refusal: measured 2.04s per
    request, every time. The app makes several requests per rerun, so four UI
    tests took 199s of a 231s suite (`--durations`: 153.6s, 30.6s, 10.2s, 4.6s)
    — all of it waiting for a refusal that was never in doubt.

    Requests to API_URL's host:port now raise the same `requests.ConnectionError`
    without opening a socket. The app sees exactly what it saw before, so the
    tests assert the same degraded behaviour; only the wait is gone. It also
    removes a dependence on the machine's state: if something *was* listening on
    that port, these tests used to talk to it.

    Requests to any other address pass through untouched, and nothing is patched
    when RUN_LIVE_TESTS=1 (live tests want the real API).
    """
    if _live_tests_enabled():
        yield
        return
    try:
        import requests
        from requests.adapters import HTTPAdapter
    except ImportError:  # requests arrives with streamlit; nothing to patch without it
        yield
        return
    from urllib.parse import urlsplit

    real_send = HTTPAdapter.send

    def send(self, request, **kwargs):
        dead = urlsplit(os.environ.get("API_URL", "http://127.0.0.1:9"))
        target = urlsplit(request.url)
        if (target.hostname, target.port) == (dead.hostname, dead.port):
            raise requests.ConnectionError(
                f"connection refused (offline test suite): {request.url}",
                request=request,
            )
        return real_send(self, request, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(HTTPAdapter, "send", send)
        yield


@pytest_asyncio.fixture
async def offline_pipeline(monkeypatch, tmp_path):
    """
    The real compiled graph over an in-memory index with a scripted LLM.

    See tests/fixtures/pipeline_harness.py for what is real, what is replaced
    and where. Yields an OfflinePipeline; `.llm` is the ScriptedLLM.
    """
    from tests.fixtures import pipeline_harness

    pipeline = pipeline_harness.install(monkeypatch, tmp_path)
    await pipeline_harness.start_tracker()
    try:
        yield pipeline
    finally:
        await pipeline.aclose()


@dataclass
class MockScoredPoint:
    """Mock Qdrant ScoredPoint for testing."""
    id: int
    score: float
    payload: dict


def make_scored_point(chunk_id: str, score: float) -> MockScoredPoint:
    """Factory for mock ScoredPoint objects."""
    return MockScoredPoint(
        id=hash(chunk_id) % (2**63),
        score=score,
        payload={"chunk_id": chunk_id, "text": f"Text for {chunk_id}"},
    )


@pytest.fixture
def sample_dense_results():
    """Dense search results — 5 chunks sorted by cosine similarity."""
    return [
        make_scored_point("chunk_a", 0.95),
        make_scored_point("chunk_b", 0.88),
        make_scored_point("chunk_c", 0.82),
        make_scored_point("chunk_d", 0.75),
        make_scored_point("chunk_e", 0.68),
    ]


@pytest.fixture
def sample_sparse_results():
    """Sparse BM25 results — 5 chunks with some overlap with dense."""
    return [
        make_scored_point("chunk_c", 12.5),   # overlap with dense
        make_scored_point("chunk_f", 11.2),
        make_scored_point("chunk_a", 10.8),   # overlap with dense
        make_scored_point("chunk_g", 9.4),
        make_scored_point("chunk_h", 8.1),
    ]


@pytest.fixture
def sample_reranked_chunks():
    """Reranked chunks with scores and metadata."""
    return [
        {
            "chunk_id": "chunk_a",
            "text": "Revenue was $45.2M in FY2023.",
            "reranker_score": 0.92,
            "source_file": "financials_2023.pdf",
            "page_number": 5,
            "document_category": "financial",
        },
        {
            "chunk_id": "chunk_b",
            "text": "EBITDA margin improved to 18.5%.",
            "reranker_score": 0.85,
            "source_file": "financials_2023.pdf",
            "page_number": 7,
            "document_category": "financial",
        },
        {
            "chunk_id": "chunk_c",
            "text": "The merger agreement includes a change of control provision.",
            "reranker_score": 0.78,
            "source_file": "merger_agreement.pdf",
            "page_number": 12,
            "document_category": "legal",
        },
    ]


@pytest.fixture
def sample_agent_state():
    """Minimal valid AgentState for testing."""
    return {
        "original_query": "What was the revenue in FY2023?",
        "current_query": "What was the revenue in FY2023?",
        "query_type": "financial",
        "parsed_intent": {"requires_numerical_precision": True},
        "extracted_filters": {"fiscal_year": "FY2023"},
        "retrieval_config": {},
        "dense_results": [],
        "sparse_results": [],
        "fused_results": [],
        "reranked_results": [],
        "expanded_context": [],
        "context_quality_score": 0.0,
        "quality_breakdown": {},
        "quality_method": "heuristic",
        "missing_aspects": [],
        "rewrite_iteration": 0,
        "rewrite_history": [],
        "agent_trace": [],
        "numerical_registry": {},
        "inconsistencies": [],
        "generated_answer": "",
        "citations": [],
        "numerical_claims": [],
        "confidence_score": 0.0,
        "hallucination_flags": [],
        "validation_status": "passed",
        "validation_attempt": 0,
        "force_refusal": False,
        "deal_id": "test_deal_001",
        "session_id": "test_session_001",
        "total_latency_ms": 0.0,
        "status": "running",
        "error": None,
    }
