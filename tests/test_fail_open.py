"""
Optional steps fail open to their documented default; safety steps fail closed
(Decision 88, src/utils/fail_open.py).

Each optional step is made to raise, and the request is shown to finish with
the step's default — and, under STRICT_OPTIONAL_STEPS=1 (what the suite runs
with), to raise instead. Each safety step is made to raise, and the request is
shown to fail whatever that switch says. Offline: Qdrant is in memory or
stubbed, the embedder and the cross-encoder are stubbed.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

import src.agents.retrieval_executor as rx
import src.data_processing.ingest_pipeline as ip
from src.agents.financial_verifier import financial_verifier_node
from src.agents.retrieval_strategy import RETRIEVAL_CONFIGS
from src.data_processing.ingest_pipeline import extract_and_chunk
from src.utils import metrics
from src.utils.fail_open import fail_open, fail_open_async, strict_optional_steps
from src.workflow.orchestrator import _attach_perf
from tests.conftest import make_scored_point
from tests.fixtures import ingestion_docs as fx


@pytest.fixture
def lenient(monkeypatch):
    """Production behaviour: a failing optional step falls back to its default."""
    monkeypatch.setenv("STRICT_OPTIONAL_STEPS", "0")


def _failures(step: str) -> float:
    return metrics.OPTIONAL_STEP_FAILURES.value(step=step)


def _boom(*args, **kwargs):
    raise KeyError("spa")


async def _aboom(*args, **kwargs):
    raise KeyError("spa")


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

class TestHelper:
    def test_the_suite_runs_strict(self):
        assert strict_optional_steps()

    def test_strict_reraises_after_logging_and_counting(self, caplog):
        before = _failures("helper_strict")
        with caplog.at_level(logging.ERROR), pytest.raises(KeyError):
            fail_open("helper_strict", _boom, default="fallback")
        assert _failures("helper_strict") == before + 1
        assert [r.step for r in caplog.records if r.name == "src.utils.fail_open"] == ["helper_strict"]

    def test_lenient_returns_the_default_logs_once_and_counts(self, lenient, caplog):
        before = _failures("helper_lenient")
        with caplog.at_level(logging.ERROR):
            assert fail_open("helper_lenient", _boom, 1, x=2, default="fallback") == "fallback"
        records = [r for r in caplog.records if r.name == "src.utils.fail_open"]
        assert len(records) == 1
        assert records[0].levelno == logging.ERROR and records[0].step == "helper_lenient"
        assert records[0].exc_info and records[0].exc_info[0] is KeyError
        assert _failures("helper_lenient") == before + 1

    def test_a_step_that_succeeds_is_untouched(self, lenient, caplog):
        before = _failures("helper_ok")
        with caplog.at_level(logging.ERROR):
            assert fail_open("helper_ok", lambda a, b=0: a + b, 1, b=2, default=-1) == 3
        assert not [r for r in caplog.records if r.name == "src.utils.fail_open"]
        assert _failures("helper_ok") == before

    @pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit, asyncio.CancelledError, GeneratorExit])
    def test_base_exceptions_pass_through(self, lenient, error):
        def stop():
            raise error()

        with pytest.raises(error):
            fail_open("helper_base", stop, default=None)

    @pytest.mark.asyncio
    async def test_async(self, lenient):
        assert await fail_open_async("helper_async", _aboom, default=[]) == []

        async def ok(x):
            return x * 2

        assert await fail_open_async("helper_async", ok, 4, default=None) == 8

        async def cancelled():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await fail_open_async("helper_async", cancelled, default=None)

    @pytest.mark.asyncio
    async def test_async_strict(self):
        with pytest.raises(KeyError):
            await fail_open_async("helper_async_strict", _aboom, default=[])

    def test_the_node_trace_names_the_fallback(self, lenient):
        with metrics.node_scope("retrieval_executor") as scope:
            fail_open("named_document", _boom, default=None)
            fail_open("named_document", _boom, default=None)
        delta = _attach_perf({"agent_trace": [{"agent": "retrieval_executor"}]}, scope, "retrieval_executor")
        assert delta["agent_trace"][-1]["perf"]["fallbacks"] == ["named_document"]
        # Nothing failed: no key at all.
        with metrics.node_scope("retrieval_executor") as clean:
            pass
        delta = _attach_perf({"agent_trace": [{"agent": "retrieval_executor"}]}, clean, "retrieval_executor")
        assert "fallbacks" not in delta["agent_trace"][-1]["perf"]


# ---------------------------------------------------------------------------
# Query path: the retrieval executor's optional steps
# ---------------------------------------------------------------------------

def _point(chunk: str, doc: str, text: str = "text", **extra):
    point = make_scored_point(chunk, 0.5)
    point.payload.update({"chunk_id": chunk, "doc_id": doc, "source_file": f"{doc}.txt", "text": text,
                          "deal_id": "deal-f", **extra})
    return point


class _Retrieval:
    """Stubs every model and store call; records what reaches each step."""

    def __init__(self, monkeypatch):
        self.focus_docs: list[str] = []
        self.reranked: list[str] = []
        ident = {"doc_title": "NON-DISCLOSURE AGREEMENT"}
        self.dense = [_point("a1", "A", **ident, doc_parties=["Harbor Analytics"]),
                      _point("a2", "A", **ident, doc_parties=["Harbor Analytics"]),
                      _point("t1", "T", table_id="T-1", table_representation="narrative", is_table=1),
                      _point("t2", "T", table_id="T-1", table_representation="markdown", is_table=1)]
        self.sparse = [_point("b1", "B", "Gamma Systems and Buyer agree", **ident,
                              doc_parties=["Gamma Systems"])]

        async def embed(texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

        async def search(**kwargs):
            return list(self.dense), list(self.sparse)

        async def dense(**kwargs):
            self.focus_docs.append(kwargs["doc_id"])
            return [_point(f"{kwargs['doc_id']}_clause", kwargs["doc_id"], "the clause")]

        async def rerank(pairs):
            self.reranked = [p for _, p in pairs]
            return np.linspace(0.9, 0.1, len(pairs)).astype(np.float32)

        async def expand(**kwargs):
            return [dict(c, parent_text="parent") for c in kwargs["chunks"]]

        monkeypatch.setattr(rx, "embed_texts_async", embed)
        monkeypatch.setattr(rx, "compute_sparse_bm25", lambda text: "sparse")
        monkeypatch.setattr(rx, "hybrid_search", search)
        monkeypatch.setattr(rx, "dense_search", dense)
        monkeypatch.setattr(rx, "rerank_pairs_async", rerank)
        monkeypatch.setattr(rx, "expand_context", expand)

    @staticmethod
    def state(query="Consider the NDA between Gamma Systems and Buyer; may it be disclosed?") -> dict:
        return {"current_query": query, "query_type": "legal", "parsed_intent": {}, "deal_id": "deal-f",
                "include_pii": False, "rewrite_iteration": 0, "extracted_filters": {}, "sub_questions": [],
                "retrieval_config": {**RETRIEVAL_CONFIGS["legal"], "document_focus_k": 5}}

    async def run(self, query=None) -> dict:
        with metrics.node_scope("retrieval_executor") as scope:
            out = await rx.retrieval_executor_node(self.state(query) if query else self.state())
        out["_fallbacks"] = scope.get("fallbacks", [])
        return out


def _ids(chunks) -> set[str]:
    return {c["chunk_id"] for c in chunks}


class TestRetrievalFallbacks:
    @pytest.mark.asyncio
    async def test_nothing_fails_nothing_changes(self, monkeypatch):
        stubs = _Retrieval(monkeypatch)
        strict = await stubs.run()
        monkeypatch.setenv("STRICT_OPTIONAL_STEPS", "0")
        lenient = await stubs.run()
        for key in ("reranked_results", "expanded_context", "retrieval_passes"):
            assert strict[key] == lenient[key]
        assert strict["_fallbacks"] == lenient["_fallbacks"] == []
        # The question names B: focus searched it, and the cap kept one of T's two.
        assert stubs.focus_docs == ["B", "B"]
        assert len(_ids(strict["reranked_results"]) & {"t1", "t2"}) == 1

    @pytest.mark.asyncio
    async def test_named_document_falls_back_to_the_fused_choice(self, monkeypatch, lenient):
        stubs = _Retrieval(monkeypatch)
        monkeypatch.setattr(rx, "name_document", _boom)
        out = await stubs.run()
        assert stubs.focus_docs == ["A"]                     # the fused leader
        assert "A_clause" in _ids(out["reranked_results"])
        assert out["_fallbacks"] == ["named_document"]

    @pytest.mark.asyncio
    async def test_named_document_strict(self, monkeypatch):
        _Retrieval(monkeypatch)
        monkeypatch.setattr(rx, "name_document", _boom)
        with pytest.raises(KeyError):
            await rx.retrieval_executor_node(_Retrieval.state())

    @pytest.mark.asyncio
    async def test_period_preference_falls_back_to_the_lists_as_retrieved(self, monkeypatch, lenient):
        stubs = _Retrieval(monkeypatch)
        monkeypatch.setattr(rx, "prefer_named_period", _aboom)
        out = await stubs.run("What was FY2023 revenue of Gamma Systems?")
        assert {"a1", "a2", "b1"} <= _ids(out["reranked_results"])
        assert out["_fallbacks"] == ["filing_period"]

    @pytest.mark.asyncio
    async def test_representation_cap_falls_back_to_no_cap(self, monkeypatch, lenient):
        stubs = _Retrieval(monkeypatch)
        monkeypatch.setattr(rx, "cap_table_representations", _boom)
        out = await stubs.run()
        assert {"t1", "t2"} <= _ids(out["reranked_results"])
        assert out["_fallbacks"] == ["representation_cap"]

    @pytest.mark.asyncio
    async def test_document_focus_falls_back_to_no_focus_candidates(self, monkeypatch, lenient):
        stubs = _Retrieval(monkeypatch)
        monkeypatch.setattr(rx, "dense_search", _aboom)
        out = await stubs.run()
        assert not any(cid.endswith("_clause") for cid in _ids(out["reranked_results"]))
        assert {"a1", "b1"} <= _ids(out["reranked_results"])
        assert out["_fallbacks"] == ["document_focus"]

    @pytest.mark.asyncio
    async def test_context_expansion_falls_back_to_the_reranked_chunks(self, monkeypatch, lenient):
        stubs = _Retrieval(monkeypatch)
        monkeypatch.setattr(rx, "expand_context", _aboom)
        out = await stubs.run()
        assert out["expanded_context"] == out["reranked_results"]
        assert out["expanded_context"] is not out["reranked_results"]
        assert not any("parent_text" in c for c in out["expanded_context"])
        assert out["_fallbacks"] == ["context_expansion"]

    @pytest.mark.asyncio
    async def test_a_required_step_still_fails_the_request(self, monkeypatch, lenient):
        # The search applies deal isolation and the version and PII filters:
        # it has no default, so a failure fails the request.
        _Retrieval(monkeypatch)
        monkeypatch.setattr(rx, "hybrid_search", _aboom)
        with pytest.raises(KeyError):
            await rx.retrieval_executor_node(_Retrieval.state())


class TestFinancialVerifierFallback:
    STATE = {"expanded_context": [{"chunk_id": "c1", "source_file": "a.txt",
                                   "text": "                FY2023   FY2022\nRevenue        $452.8   $387.1"}]}

    @pytest.mark.asyncio
    async def test_a_failed_cross_check_is_skipped_and_says_so(self, monkeypatch, lenient):
        monkeypatch.setattr("src.agents.financial_verifier.NumericalRegistry.from_chunks", _boom)
        out = await financial_verifier_node(self.STATE)
        assert out["numerical_registry"] == {} and out["inconsistencies"] == []
        assert out["agent_trace"][0]["skipped"] and out["agent_trace"][0]["reason"] == "failed"

    @pytest.mark.asyncio
    async def test_strict(self, monkeypatch):
        monkeypatch.setattr("src.agents.financial_verifier.NumericalRegistry.from_chunks", _boom)
        with pytest.raises(KeyError):
            await financial_verifier_node(self.STATE)

    @pytest.mark.asyncio
    async def test_unchanged_when_it_works(self):
        out = await financial_verifier_node(self.STATE)
        assert out["agent_trace"][0]["figures"] == 2


# ---------------------------------------------------------------------------
# Ingest path
# ---------------------------------------------------------------------------

CONTRACT = ("MUTUAL NON-DISCLOSURE AGREEMENT\n\nThis Agreement is made by and between Harbor Analytics, Inc., "
            "a Delaware corporation, and Gamma Systems LLC.\n\n"
            + "The Receiving Party shall keep the Confidential Information in confidence. " * 40
            + "\n\nAny change of control of either party terminates this Agreement.\n")


TEN_K = ("UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\nWashington, D.C. 20549\nFORM 10-K\n"
         "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934\n"
         "For the fiscal year ended December 31, 2022\nCommission file number 1-3285\n"
         "3M COMPANY\n(Exact name of registrant as specified in its charter)\n\n"
         + "Revenue grew in every segment during the year. " * 30 + "\n")


@pytest.fixture
def contract(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_ENABLED", "0")
    return fx.write_txt(tmp_path, "nda.txt", CONTRACT)


def _ingest(path):
    return extract_and_chunk(str(path), path.name, "deal-f", "legal")


class TestIngestFallbacks:
    def test_unchanged_when_nothing_fails(self, contract, monkeypatch):
        strict = _ingest(contract)
        monkeypatch.setenv("STRICT_OPTIONAL_STEPS", "0")
        lenient = _ingest(contract)
        assert strict.chunks == lenient.chunks and strict.parents == lenient.parents
        assert strict.chunks[0]["doc_title"] == "MUTUAL NON-DISCLOSURE AGREEMENT"
        assert "change_of_control" in {s for c in strict.chunks for s in c["risk_signals"]}

    def test_identity_falls_back_to_none(self, contract, monkeypatch, lenient):
        monkeypatch.setattr(ip, "extract_document_identity", _boom)
        doc = _ingest(contract)
        assert doc.chunks and not any("doc_title" in c or "doc_parties" in c for c in doc.chunks)

    def test_filing_metadata_falls_back_to_none(self, tmp_path, monkeypatch, lenient):
        path = fx.write_txt(tmp_path, "10k.txt", TEN_K)
        assert _ingest(path).filing["filing_fiscal_years"] == [2022]
        monkeypatch.setattr(ip, "extract_filing_metadata", _boom)
        doc = _ingest(path)
        assert doc.filing == {} and not any(k.startswith("filing_") for c in doc.chunks for k in c)

    def test_risk_signals_fall_back_to_none(self, contract, monkeypatch, lenient):
        monkeypatch.setattr(ip.RiskSignalExtractor, "extract", _boom)
        doc = _ingest(contract)
        assert doc.chunks and all(c["risk_signals"] == [] for c in doc.chunks)
        assert doc.risk_signals == []

    def test_context_header_falls_back_to_bare_text(self, contract, monkeypatch, lenient):
        monkeypatch.setenv("CHUNK_CONTEXT_HEADERS", "1")
        assert all(c.get("context_header") for c in _ingest(contract).chunks)
        monkeypatch.setattr(ip, "build_context_header", _boom)
        assert not any("context_header" in c for c in _ingest(contract).chunks)

    @pytest.mark.parametrize("name", ["extract_document_identity", "extract_filing_metadata"])
    def test_strict(self, contract, monkeypatch, name):
        monkeypatch.setattr(ip, name, _boom)
        with pytest.raises(KeyError):
            _ingest(contract)

    @pytest.mark.asyncio
    async def test_version_candidates_fall_back_to_no_proposal(self, contract, monkeypatch, lenient):
        from tests.test_ingestion_retrieval import _client, _index

        monkeypatch.setattr(ip, "_version_candidates", _aboom)
        result = await _index(await _client(), contract)
        assert result["chunks_created"] > 0 and result["version_candidates"] == []


class TestSafetyStepsFailClosed:
    """No STRICT_OPTIONAL_STEPS setting lets a safety step's failure through."""

    @pytest.mark.parametrize("strict", ["0", "1"])
    def test_pii_detection_failure_fails_the_ingest(self, contract, monkeypatch, strict):
        # Indexing the chunk as PII-free would be failing open.
        monkeypatch.setenv("STRICT_OPTIONAL_STEPS", strict)
        monkeypatch.setattr(ip.PIIDetector, "detect", _boom)
        with pytest.raises(KeyError):
            _ingest(contract)

    @pytest.mark.parametrize("strict", ["0", "1"])
    def test_injection_screen_failure_fails_the_ingest(self, contract, monkeypatch, strict):
        # An unscreened chunk would carry no label however it reads.
        monkeypatch.setenv("STRICT_OPTIONAL_STEPS", strict)
        monkeypatch.setattr(ip, "screen_text", _boom)
        with pytest.raises(KeyError):
            _ingest(contract)

    def test_a_laya_pii_screen_that_cannot_run_indexes_nothing(self, contract, monkeypatch, lenient):
        # With LAYA_PII=1 Laya flags chunks the regex does not; falling back to
        # the regex would index those as PII-free. Found failing open (Decision 88).
        import src.decisions.ingest_signals as sig
        from src.data_processing.errors import PIIScreeningUnavailable
        from src.decisions.laya_client import LayaUnavailable

        def outage(requests):
            raise LayaUnavailable("stubbed outage")

        monkeypatch.setenv("LAYA_ENABLED", "1")
        monkeypatch.setattr(sig, "decide_batch", outage)
        monkeypatch.setenv("LAYA_PII", "1")
        with pytest.raises(PIIScreeningUnavailable) as raised:
            _ingest(contract)
        assert raised.value.status_code == 503
        # Without Laya PII the regex is the configured screen: the outage costs
        # only Laya's risk opinions, as before.
        monkeypatch.setenv("LAYA_PII", "0")
        doc = _ingest(contract)
        assert doc.chunks and all(c["pii_source"] == "regex" for c in doc.chunks)


@pytest.mark.asyncio
async def test_backfill_leaves_a_document_it_cannot_read_as_it_was(monkeypatch, lenient):
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import PointStruct, SparseVector

    import src.vector_db.identity_backfill as bf
    from src.vector_db.collection_manager import setup_collections
    from src.vector_db.constants import COLLECTION_NAME, VECTOR_SIZE

    client = AsyncQdrantClient(location=":memory:")
    await setup_collections(client)
    rows = [{"chunk_id": "x-0", "deal_id": "d", "doc_id": "x", "chunk_index": 0, "contains_pii": 0,
             "text": "CRASH", "doc_title": "KEPT TITLE"},
            {"chunk_id": "y-0", "deal_id": "d", "doc_id": "y", "chunk_index": 0, "contains_pii": 0,
             "text": "MUTUAL NON-DISCLOSURE AGREEMENT\nbetween Harbor Analytics, Inc. and Gamma Systems LLC."}]
    vector = {"dense": [1.0] + [0.0] * (VECTOR_SIZE - 1), "sparse": SparseVector(indices=[1], values=[1.0])}
    await client.upsert(COLLECTION_NAME, points=[PointStruct(id=i, vector=vector, payload=p)
                                                 for i, p in enumerate(rows)])
    real = bf.extract_document_identity

    def flaky(payloads):
        if payloads[0]["text"] == "CRASH":
            raise KeyError("spa")
        return real(payloads)

    monkeypatch.setattr(bf, "extract_document_identity", flaky)
    stats = await bf.backfill_document_identity(client)
    records, _ = await client.scroll(COLLECTION_NAME, limit=10, with_payload=True)
    by_doc = {r.payload["doc_id"]: r.payload for r in records}
    assert stats["failed"] == 1 and stats["updated"] == 1
    assert by_doc["x"]["doc_title"] == "KEPT TITLE"          # not deleted
    assert by_doc["y"]["doc_title"] == "MUTUAL NON-DISCLOSURE AGREEMENT"
