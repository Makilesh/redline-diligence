"""
Who may read what, how much anyone may send, and what the probes admit to.

Read access to deals, limits on the routes that had none, request-body caps,
sandbox purge completeness, single-worker enforcement, and the degraded-mode
reporting in /health and /ready. Offline: Qdrant and the pipeline are stubbed.
"""

import io
import zipfile
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import api.routes.deals as deals
import api.routes.ingest as ingest_routes
import api.routes.query as query_routes
import api.security as security
import src.vector_db.qdrant_client as qdrant_module
import platform_stubs
from platform_stubs import ADMIN_KEY, DEMO_DEAL
from src.utils import runtime_status
from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME
from src.workflow import conditional_edges

# Fixture defined in platform_stubs.py, bound here so pytest finds it.
platform_environment = platform_stubs.platform_environment

PRIVATE_DEAL = "project_falcon_2025"
ADMIN = {security.ADMIN_HEADER: ADMIN_KEY}


class FakeQdrant:
    """Both collections, with a switch to fail deletes on either."""

    def __init__(self, deal_ids: list[str]):
        self.points = {COLLECTION_NAME: set(deal_ids), PARENT_COLLECTION_NAME: set(deal_ids)}
        self.fail_delete_in: str | None = None
        self.parent_missing = False
        self.delete_calls: list[str] = []
        self.count_value = 131

    async def facet(self, collection_name, key, facet_filter=None, limit=1000):
        if collection_name == PARENT_COLLECTION_NAME and self.parent_missing:
            raise RuntimeError("Not found: Collection `parents` doesn't exist!")
        if key == "deal_id":
            values = sorted(self.points[collection_name])
        else:
            values = ["a.pdf", "b.pdf"]
        return SimpleNamespace(hits=[SimpleNamespace(value=v, count=1) for v in values])

    async def delete(self, collection_name, points_selector):
        self.delete_calls.append(collection_name)
        if collection_name == PARENT_COLLECTION_NAME and self.parent_missing:
            raise RuntimeError("Not found: Collection `parents` doesn't exist!")
        if self.fail_delete_in == collection_name:
            raise RuntimeError("timeout talking to http://internal-qdrant:6333")
        self.points[collection_name].discard(points_selector.filter.must[0].match.value)

    async def scroll(self, **kwargs):
        return [], None

    async def get_collections(self):
        return SimpleNamespace(collections=[])

    async def count(self, collection_name, count_filter=None, exact=True):
        # 131 chunks, of which `stale_points` lack the current BM25 marker.
        return SimpleNamespace(count=self.stale_points if count_filter else self.count_value)

    stale_points = 0


@pytest.fixture
def qdrant(monkeypatch):
    fake = FakeQdrant([DEMO_DEAL, PRIVATE_DEAL])
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: fake)
    # Installed as the singleton too, the way the eval harness swaps a client
    # in, so get_qdrant_backend() sees it.
    monkeypatch.setattr(qdrant_module, "_qdrant_client", fake)
    return fake


@pytest.fixture
def pipeline_calls(monkeypatch):
    calls: list[dict] = []

    async def fake_run_query(**kwargs):
        calls.append(kwargs)
        return {"generated_answer": "Answer [1].", "status": "completed"}

    monkeypatch.setattr(query_routes, "run_query", fake_run_query)
    return calls


@pytest.fixture
def client():
    return TestClient(api_main.app)


def _sandbox_id() -> str:
    return deals.new_sandbox_id(datetime.now(timezone.utc))


# ==============================================================================
# Read access: public callers reach PUBLIC_DEAL_IDS and sandboxes, nothing else
# ==============================================================================


def test_public_caller_cannot_query_a_private_deal(client, qdrant, pipeline_calls, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    body = {"query": "What is the purchase price?", "deal_id": PRIVATE_DEAL}

    for route in ("/api/v1/query", "/api/v1/query/stream"):
        assert client.post(route, json=body).status_code == 404
    assert pipeline_calls == []  # refused before a slot, a guard call or an LLM call

    assert client.post("/api/v1/query", json=body, headers=ADMIN).status_code == 200
    assert client.post("/api/v1/query", json={**body, "deal_id": DEMO_DEAL}).status_code == 200
    assert client.post("/api/v1/query", json={**body, "deal_id": _sandbox_id()}).status_code == 200


def test_private_deals_are_not_listed_or_readable_publicly(client, qdrant, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)

    assert [d["deal_id"] for d in client.get("/api/v1/deals").json()] == [DEMO_DEAL]
    for path in ("", "/documents", "/risk-signals"):
        assert client.get(f"/api/v1/deals/{PRIVATE_DEAL}{path}").status_code == 404
        assert client.get(f"/api/v1/deals/{DEMO_DEAL}{path}").status_code == 200

    listed = {d["deal_id"] for d in client.get("/api/v1/deals", headers=ADMIN).json()}
    assert listed == {DEMO_DEAL, PRIVATE_DEAL}
    assert client.get(f"/api/v1/deals/{PRIVATE_DEAL}", headers=ADMIN).status_code == 200


def test_public_deal_list_is_configurable(client, qdrant, monkeypatch):
    monkeypatch.setenv("PUBLIC_DEAL_IDS", f"{DEMO_DEAL}, {PRIVATE_DEAL}")
    assert len(client.get("/api/v1/deals").json()) == 2

    monkeypatch.setenv("PUBLIC_DEAL_IDS", "")
    assert client.get("/api/v1/deals").json() == []
    assert client.get(f"/api/v1/deals/{DEMO_DEAL}").status_code == 404


# ==============================================================================
# Limits on the routes that had none
# ==============================================================================


def test_sandbox_creation_is_rate_limited_and_the_registry_is_bounded(client, monkeypatch):
    create = {"deal_name": "Visitor upload", "is_sandbox": True}

    monkeypatch.setenv("RATE_LIMIT_DEAL_PER_MINUTE", "3")
    assert [client.post("/api/v1/deals", json=create).status_code for _ in range(4)] == [
        200,
        200,
        200,
        429,
    ]

    # Many addresses, each under its own limit: the registry cap is what holds.
    monkeypatch.setenv("RATE_LIMIT_DEAL_PER_MINUTE", "1000")
    monkeypatch.setenv("MAX_REGISTERED_DEALS", "5")
    statuses = [client.post("/api/v1/deals", json=create).status_code for _ in range(4)]
    assert statuses == [200, 200, 503, 503]
    assert len(deals._deals) == 5


def test_expired_sandboxes_make_room_before_creation_is_refused(client, monkeypatch):
    monkeypatch.setenv("MAX_REGISTERED_DEALS", "2")
    expired = deals.new_sandbox_id(datetime(2020, 1, 1, tzinfo=timezone.utc))
    deals._deals[expired] = {"deal_id": expired, "is_sandbox": True}
    deals._deals["other"] = {"deal_id": "other", "is_sandbox": False}

    response = client.post("/api/v1/deals", json={"deal_name": "x", "is_sandbox": True})
    assert response.status_code == 200
    assert expired not in deals._deals


def test_read_routes_share_a_rate_limit(client, qdrant, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_READ_PER_MINUTE", "2")
    sandbox = _sandbox_id()
    statuses = [
        client.get("/api/v1/deals").status_code,
        client.get(f"/api/v1/deals/{DEMO_DEAL}/documents").status_code,
        client.get("/api/v1/deals").status_code,
        client.post(f"/api/v1/deals/{sandbox}/purge").status_code,
    ]
    assert statuses == [200, 200, 429, 429]


def test_budget_hides_the_per_key_breakdown_from_public_callers(client, monkeypatch):
    status = {
        "synthesis_primary": {
            "model": "gemini/x",
            "used": 12,
            "limit": 100,
            "keys": [{"key_index": 0, "used": 12, "limit": 20}],
        }
    }

    class Tracker:
        async def get_budget_status(self):
            return status

    async def get_instance(*args, **kwargs):
        return Tracker()

    import src.llm.budget_tracker as budget_tracker

    monkeypatch.setattr(budget_tracker.BudgetTracker, "get_instance", get_instance)
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)

    public = client.get("/api/v1/budget").json()
    assert public == {"synthesis_primary": {"model": "gemini/x", "used": 12, "limit": 100}}
    assert client.get("/api/v1/budget", headers=ADMIN).json() == status


# ==============================================================================
# Where document text goes (Decision 116)
# ==============================================================================

_GEMINI_VARS = ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GEMINI_API_KEY_1", "UPLOAD_NOTICE")


@pytest.fixture
def no_tracker(monkeypatch):
    import src.llm.budget_tracker as budget_tracker

    monkeypatch.setattr(budget_tracker.BudgetTracker, "_instance", None)
    for name in _GEMINI_VARS:
        monkeypatch.delenv(name, raising=False)


def test_with_a_gemini_key_the_upload_notice_says_text_leaves_the_server(client, no_tracker, monkeypatch):
    from src.llm.data_handling import DEFAULT_UPLOAD_NOTICE

    key = "k" * 39
    monkeypatch.setenv("GEMINI_API_KEYS", key)
    body = client.get("/api/v1/data-handling").json()
    assert body == {"hosted_llm": "Google Gemini API", "upload_notice": DEFAULT_UPLOAD_NOTICE}
    assert "free tier" in body["upload_notice"] and "confidential" in body["upload_notice"]
    assert key not in str(body)


def test_with_no_key_nothing_leaves_and_no_notice_is_shown(client, no_tracker, monkeypatch):
    monkeypatch.setenv("UPLOAD_NOTICE", "ignored without a hosted model")
    assert client.get("/api/v1/data-handling").json() == {"hosted_llm": None, "upload_notice": None}


def test_the_owner_sets_the_notices_wording(client, no_tracker, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k" * 39)
    monkeypatch.setenv("UPLOAD_NOTICE", "  Paid tier under our DPA: text is not used for training.  ")
    assert client.get("/api/v1/data-handling").json()["upload_notice"] == (
        "Paid tier under our DPA: text is not used for training."
    )


def test_the_running_trackers_keys_decide(client, no_tracker, monkeypatch):
    import src.llm.budget_tracker as budget_tracker

    monkeypatch.setattr(budget_tracker.BudgetTracker, "_instance", SimpleNamespace(_api_keys=[]))
    monkeypatch.setenv("GEMINI_API_KEYS", "k" * 39)  # read at startup, not now
    assert client.get("/api/v1/data-handling").json()["hosted_llm"] is None


# ==============================================================================
# Request body limits
# ==============================================================================


def test_oversized_json_body_is_refused_before_it_is_read(client, pipeline_calls):
    big = {"query": "q", "deal_id": DEMO_DEAL, "padding": "x" * 100_000}
    response = client.post("/api/v1/query", json=big)

    assert response.status_code == 413
    assert "64 KB" in response.json()["detail"]
    assert response.headers[security.REQUEST_ID_HEADER]
    assert pipeline_calls == []


def test_body_without_a_content_length_is_cut_off_at_the_limit(client, pipeline_calls):
    def chunks():
        yield b'{"query": "q", "deal_id": "' + DEMO_DEAL.encode() + b'", "padding": "'
        for _ in range(200):
            yield b"x" * 1024
        yield b'"}'

    response = client.post(
        "/api/v1/query", content=chunks(), headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
    assert pipeline_calls == []


def test_oversized_upload_is_refused_before_it_is_spooled_to_disk(client, monkeypatch):
    # The route's own cap runs after Starlette has parsed — and spooled to the
    # temp directory — the entire multipart body.
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    parsed = []
    monkeypatch.setattr(
        ingest_routes, "_save_upload", lambda *a, **k: parsed.append(1)
    )
    response = client.post(
        "/api/v1/ingest",
        data={"deal_id": _sandbox_id()},
        files={"file": ("big.txt", b"x" * (3 * 1024 * 1024), "text/plain")},
    )
    assert response.status_code == 413
    assert parsed == []
    assert security.max_body_bytes("/api/v1/ingest") == 2 * 1024 * 1024
    assert security.max_body_bytes("/api/v1/query") == 64 * 1024


def test_office_archive_bomb_is_rejected_without_decompressing(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_EXPANDED_MB", "1")
    bomb = tmp_path / "upload.docx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"\0" * (4 * 1024 * 1024))
    assert bomb.stat().st_size < 64 * 1024  # 4 MB of zeros, a few KB on the wire

    with pytest.raises(ingest_routes.HTTPException) as rejected:
        ingest_routes._reject_archive_bomb(str(bomb), ".docx")
    assert rejected.value.status_code == 413

    small = tmp_path / "ok.docx"
    with zipfile.ZipFile(small, "w") as archive:
        archive.writestr("word/document.xml", b"<w/>")
    ingest_routes._reject_archive_bomb(str(small), ".docx")  # no exception

    not_zip = tmp_path / "fake.xlsx"
    not_zip.write_bytes(io.BytesIO(b"plain text").getvalue())
    with pytest.raises(ingest_routes.HTTPException) as invalid:
        ingest_routes._reject_archive_bomb(str(not_zip), ".xlsx")
    assert invalid.value.status_code == 400
    ingest_routes._reject_archive_bomb(str(not_zip), ".txt")  # only office formats


# ==============================================================================
# Supersession is recorded only when the index applied it
# ==============================================================================


@pytest.fixture
def stub_indexing(monkeypatch):
    async def fake_index_document(path, filename, deal_id, category, **kwargs):
        return {
            "doc_id": "doc-v2",
            "document_category": "legal",
            "chunks_created": 3,
            "parent_chunks_created": 1,
            "previously_indexed": False,
            "warnings": [],
            "risk_signals": [],
        }

    monkeypatch.setattr(ingest_routes, "index_document", fake_index_document)
    retirement = {"result": {"ok": True, "chunks_retired": 4, "error": None}}

    async def fake_mark_superseded(**kwargs):
        return retirement["result"]

    monkeypatch.setattr(ingest_routes, "mark_superseded", fake_mark_superseded)
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    return retirement


def _upload_v2(client):
    return client.post(
        "/api/v1/ingest",
        headers=ADMIN,
        data={"deal_id": DEMO_DEAL, "supersedes_doc_id": "doc-v1"},
        files={"file": ("spa_v2.txt", b"Purchase price is $34.50 per share.", "text/plain")},
    )


def test_applied_supersession_is_recorded(client, stub_indexing):
    response = _upload_v2(client)

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["notices"] == []
    assert response.json()["version_candidates"] == []
    [record] = deals._documents[DEMO_DEAL]
    assert record["supersedes_doc_id"] == "doc-v1"


def test_version_candidates_are_proposed_not_applied(client, stub_indexing, monkeypatch):
    candidate = {"doc_id": "doc-v1", "source_file": "spa_v1.txt", "relation": "newer"}
    original = ingest_routes.index_document

    async def with_candidates(*args, **kwargs):
        return {**(await original(*args, **kwargs)), "version_candidates": [candidate]}

    monkeypatch.setattr(ingest_routes, "index_document", with_candidates)
    retired: list[dict] = []

    async def record_retirement(**kwargs):
        retired.append(kwargs)
        return {"ok": True, "chunks_retired": 1, "error": None}

    monkeypatch.setattr(ingest_routes, "mark_superseded", record_retirement)

    response = client.post(
        "/api/v1/ingest",
        headers=ADMIN,
        data={"deal_id": DEMO_DEAL},
        files={"file": ("spa_v2.txt", b"Purchase price is $34.50 per share.", "text/plain")},
    )

    assert response.json()["version_candidates"] == [candidate]
    assert retired == []  # no supersedes_doc_id sent, so nothing is retired
    assert deals._documents[DEMO_DEAL][0]["supersedes_doc_id"] == ""


def test_partial_ingest_says_why(client, stub_indexing, monkeypatch):
    # The pipeline's warnings make an upload "partial"; the uploader has to be
    # told which sheets or pages were affected, not just "Partly indexed".
    warnings = [
        "Sheet 'Q3' is hidden in the workbook; it was indexed and is marked hidden",
        "Page(s) 4-5 of 9 have no text layer (likely scanned images) and were not indexed",
        "Sheet 'Q3' is hidden in the workbook; it was indexed and is marked hidden",
        None,
    ]
    original = ingest_routes.index_document

    async def with_warnings(*args, **kwargs):
        return {**(await original(*args, **kwargs)), "warnings": warnings}

    monkeypatch.setattr(ingest_routes, "index_document", with_warnings)

    response = client.post(
        "/api/v1/ingest",
        headers=ADMIN,
        data={"deal_id": DEMO_DEAL},
        files={"file": ("model.txt", b"Revenue was $12.0 million.", "text/plain")},
    )

    body = response.json()
    assert body["status"] == "partial"
    assert body["notices"] == warnings[:2]


def test_failed_supersession_is_partial_and_not_recorded(client, stub_indexing):
    deals.register_document(DEMO_DEAL, "doc-v1", "spa_v1.txt", "legal", 4, True, None)
    stub_indexing["result"] = {
        "ok": False,
        "chunks_retired": 0,
        "error": "the superseded document is not indexed in this deal",
    }

    response = _upload_v2(client)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert "doc-v1 is still marked as the current version" in body["notices"][0]
    records = {r["doc_id"]: r for r in deals._documents[DEMO_DEAL]}
    # The registry must not claim a retirement the index never applied.
    assert records["doc-v2"]["supersedes_doc_id"] == ""
    assert records["doc-v1"]["is_current_version"] is True
    assert records["doc-v1"]["superseded_by"] == ""


# ==============================================================================
# Sandbox purge completeness
# ==============================================================================


def test_failed_parent_delete_fails_the_purge_and_leaves_the_deal_sweepable(client, qdrant):
    sandbox = _sandbox_id()
    for collection in qdrant.points.values():
        collection.add(sandbox)
    qdrant.fail_delete_in = PARENT_COLLECTION_NAME

    # It used to report "purged" here with the visitor's parent sections —
    # ~2,000-token slices of the upload — still in Qdrant, and with the child
    # points gone nothing would ever list the deal for a retry.
    assert client.delete(f"/api/v1/deals/{sandbox}").status_code == 500
    assert client.post(f"/api/v1/deals/{sandbox}/purge").json()["status"] == "deferred"
    assert sandbox in qdrant.points[COLLECTION_NAME]

    qdrant.fail_delete_in = None
    assert client.delete(f"/api/v1/deals/{sandbox}").json()["status"] == "purged"
    assert all(sandbox not in points for points in qdrant.points.values())


def test_absent_parent_collection_does_not_block_a_purge(client, qdrant):
    sandbox = _sandbox_id()
    qdrant.points[COLLECTION_NAME].add(sandbox)
    qdrant.parent_missing = True

    assert client.delete(f"/api/v1/deals/{sandbox}").json()["status"] == "purged"
    assert sandbox not in qdrant.points[COLLECTION_NAME]


@pytest.mark.asyncio
async def test_sweeper_finds_sandboxes_that_survive_only_in_the_parent_collection(qdrant):
    orphan = deals.new_sandbox_id(datetime(2020, 1, 1, tzinfo=timezone.utc))
    qdrant.points[PARENT_COLLECTION_NAME].add(orphan)

    assert await deals._sweep_expired_sandboxes() == 1
    assert orphan not in qdrant.points[PARENT_COLLECTION_NAME]


# ==============================================================================
# Single worker
# ==============================================================================


@pytest.mark.parametrize(
    ("argv", "environ", "expected"),
    [
        (["uvicorn", "api.main:app"], {}, 1),
        (["uvicorn", "api.main:app", "--workers", "4"], {}, 4),
        (["uvicorn", "api.main:app", "--workers=3"], {}, 3),
        (["gunicorn", "-w", "2", "api.main:app"], {}, 2),
        (["python", "run_api.py"], {"WEB_CONCURRENCY": "8"}, 8),
        (["python", "run_api.py"], {"WEB_CONCURRENCY": "junk"}, 1),
    ],
)
def test_declared_worker_count(argv, environ, expected):
    assert security.declared_worker_count(argv, environ) == expected


def test_multiple_workers_are_refused_unless_explicitly_allowed(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with pytest.raises(RuntimeError, match="single worker"):
        security.enforce_single_worker()

    monkeypatch.setenv("ALLOW_MULTIPLE_WORKERS", "1")
    assert security.enforce_single_worker() == 4
    assert "multiple_workers" in runtime_status.describe()["degraded"]


# ==============================================================================
# Probes that admit to degradation
# ==============================================================================


def test_ready_detail_is_for_the_operator_only(client, qdrant, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)

    public = client.get("/ready")
    assert public.status_code == 200
    assert set(public.json()) == {"status", "checks"}

    assert "checkpointer" in client.get("/ready", headers=ADMIN).json()
    monkeypatch.setenv("METRICS_PUBLIC", "1")
    assert "checkpointer" in client.get("/ready").json()


def test_health_and_ready_report_degraded_modes(client, qdrant, monkeypatch):
    monkeypatch.setenv("METRICS_PUBLIC", "1")  # the /ready detail, without a key
    assert client.get("/health").json()["degraded"] == []

    runtime_status.set_checkpointer("memory", error="OperationalError")
    runtime_status.set_model("reranker", runtime_status.MODEL_FAILED)
    runtime_status.set_model("nli", runtime_status.MODEL_READY)
    runtime_status.set_phase("ready")

    health = client.get("/health")
    assert health.status_code == 200  # still alive: liveness never fails on this
    assert health.json()["checkpointer"] == "memory"
    assert health.json()["degraded"] == ["checkpointer_in_memory", "model_warmup_failed:reranker"]

    ready = client.get("/ready")
    assert ready.status_code == 200  # degraded, but it can answer a query
    body = ready.json()
    assert body["checkpointer"] == {
        "backend": "memory",
        "durable": False,
        "error": "OperationalError",
    }
    assert body["models"] == {"reranker": "failed", "nli": "ready"}
    assert "in-flight runs do not survive a restart" in (
        body["degraded_detail"]["checkpointer_in_memory"]
    )

    # Recovery clears the flag.
    runtime_status.set_checkpointer("postgres")
    assert client.get("/health").json()["degraded"] == ["model_warmup_failed:reranker"]


def test_ready_reports_the_qdrant_backend_and_stale_sparse_vectors(client, qdrant, monkeypatch):
    monkeypatch.setenv("METRICS_PUBLIC", "1")
    body = client.get("/ready").json()
    # The fake is swapped in for the singleton, so it is reported as such —
    # never as a server it is not.
    assert body["qdrant"] == {"mode": "injected", "location": None}
    assert body["sparse_index"]["total_points"] == 131
    assert body["sparse_index"]["stale_points"] == 0
    assert "sparse_index_stale" not in body["degraded"]

    qdrant.stale_points = 40
    api_main._sparse_status_read_at = float("-inf")  # skip the 60 s cache
    body = client.get("/ready").json()
    assert body["sparse_index"]["stale_points"] == 40
    assert "sparse_index_stale" in body["degraded"]
    assert "--reindex" in body["degraded_detail"]["sparse_index_stale"]
    # Degraded, still ready: dense search answers while the index is rebuilt.
    assert client.get("/ready").status_code == 200


def test_ready_fails_while_draining(client, qdrant):
    runtime_status.set_phase("draining")
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["checks"]["accepting_queries"] is False


# ==============================================================================
# Routing on scores a model returned in the wrong type
# ==============================================================================


@pytest.mark.parametrize("bad", ["0.8", None, "high", [0.9], True])
def test_unreadable_quality_scores_do_not_crash_or_admit(bad):
    state = {
        "query_type": "summary",
        "context_quality_score": bad,
        "quality_breakdown": {"relevance": bad, "completeness": 0.9, "precision": 0.9},
        "rewrite_iteration": 0,
    }
    route = conditional_edges.route_after_quality_check(state)
    # "0.8" is a readable number and passes; everything else counts as no
    # evidence of quality and is rewritten rather than synthesised.
    assert route == ("answer_synthesizer" if bad == "0.8" else "query_rewriter")

    state["rewrite_iteration"] = 2
    if bad != "0.8":
        assert conditional_edges.route_after_quality_check(state) == "insufficient_context"


def test_valid_scores_route_exactly_as_before():
    state = {
        "query_type": "financial",
        "context_quality_score": 0.62,
        "quality_breakdown": {"relevance": 0.55, "completeness": 0.4, "precision": 0.3},
        "rewrite_iteration": 0,
    }
    assert conditional_edges.route_after_quality_check(state) == "answer_synthesizer"
    state["quality_breakdown"]["precision"] = 0.2
    assert conditional_edges.route_after_quality_check(state) == "query_rewriter"
    state["quality_breakdown"]["precision"] = 0.3
    state["answerability_veto"] = True
    assert conditional_edges.route_after_quality_check(state) == "query_rewriter"
