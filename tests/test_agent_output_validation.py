"""
Agents 1 and 6 against the shapes a model can actually return.

JSON mode guarantees syntax, not shape. Each case here is an output that used
to crash the node or turn an LLM guess into a hard retrieval filter.
"""

import pytest

import src.agents.query_intelligence as qi
import src.agents.query_rewriter as rw
from src.llm.schemas import QueryIntelligenceOutput, QueryRewriterOutput


class _Choice:
    model = "gemini/selected"
    api_key = "k"
    key_index = 0


class _Tracker:
    async def get_model_for_agent(self):
        return _Choice()


async def _fake_instance(*args, **kwargs):
    return _Tracker()


def _patch(monkeypatch, module, reply=None, error=None, served_model=None):
    async def fake_call(**kwargs):
        if error is not None:
            raise error
        if served_model and kwargs.get("served") is not None:
            kwargs["served"].model = served_model
        return reply

    monkeypatch.setattr(module, "call_structured_agent", fake_call)
    monkeypatch.setattr(module.BudgetTracker, "get_instance", _fake_instance)


async def _agent1(monkeypatch, reply=None, **kw):
    _patch(monkeypatch, qi, reply, **kw)
    return await qi.query_intelligence_node(
        {"original_query": "What was FY2023 revenue?", "deal_id": "d1"}
    )


_REWRITE_STATE = {
    "original_query": "q", "current_query": "current q", "rewrite_iteration": 0,
    "retrieval_config": {"reranker_top_k": 20},
    "extracted_filters": {"document_category": "financial"},
}


async def _agent6(monkeypatch, reply=None, **kw):
    _patch(monkeypatch, rw, reply, **kw)
    return await rw.query_rewriter_node(dict(_REWRITE_STATE))


class TestQueryIntelligenceSchema:
    @pytest.mark.parametrize("raw", [None, [], "text", 42, ["financial", "legal"]])
    def test_a_non_object_becomes_the_defaults(self, raw):
        parsed = QueryIntelligenceOutput.from_llm(raw)
        assert parsed.query_type == "summary"
        assert parsed.metadata_filters == {} and parsed.sub_questions == []
        assert parsed.reformulated_query is None

    @pytest.mark.parametrize("value, expected", [
        ("financial", "financial"), (" Legal ", "legal"), ("MULTI_HOP", "multi_hop"),
        ("finance", "summary"), (None, "summary"), (7, "summary"), (["legal"], "summary"),
    ])
    def test_query_type_is_always_a_known_type(self, value, expected):
        assert QueryIntelligenceOutput.from_llm({"query_type": value}).query_type == expected

    @pytest.mark.parametrize("value", [None, [], "legal", 3])
    def test_filters_are_always_a_dict(self, value):
        assert QueryIntelligenceOutput.from_llm({"metadata_filters": value}).metadata_filters == {}

    @pytest.mark.parametrize("value, expected", [
        (True, True), (False, False), ("true", True), ("no", False), (None, False), (1, False),
    ])
    def test_flags_are_always_booleans(self, value, expected):
        parsed = QueryIntelligenceOutput.from_llm(
            {"requires_numerical_precision": value, "requires_cross_document": value})
        assert parsed.requires_numerical_precision is expected
        assert parsed.requires_cross_document is expected

    def test_sub_questions_keep_only_real_strings(self):
        parsed = QueryIntelligenceOutput.from_llm(
            {"sub_questions": ["  What is X?  ", "", None, 3, {"q": "y"}, "What is Y?"]})
        assert parsed.sub_questions == ["What is X?", "What is Y?"]
        assert QueryIntelligenceOutput.from_llm({"sub_questions": "What is X?"}).sub_questions == []

    @pytest.mark.parametrize("value", [None, "", "   ", 12, ["q"]])
    def test_a_blank_reformulation_is_none(self, value):
        assert QueryIntelligenceOutput.from_llm({"reformulated_query": value}).reformulated_query is None


class TestAgent1Node:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("reply", [
        {"query_type": "financial", "metadata_filters": None},          # audit W2 (a)
        ["financial", "legal"],                                         # audit W2 (b)
        {"query_type": "financial", "reformulated_query": None},        # audit W2 (c)
        {},
        None,
    ])
    async def test_malformed_shapes_do_not_fail_the_query(self, monkeypatch, reply):
        out = await _agent1(monkeypatch, reply)

        assert out["current_query"] == "What was FY2023 revenue?", "the embedder gets a string"
        assert out["extracted_filters"] == {}
        assert isinstance(out["parsed_intent"], dict)
        assert out["query_type"] in ("financial", "summary")

    @pytest.mark.asyncio
    async def test_an_llm_guess_does_not_become_a_hard_filter(self, monkeypatch):
        """Audit W2 (d): doc_id / is_table / content_type were passed to Qdrant."""
        out = await _agent1(monkeypatch, {
            "query_type": "legal",
            "metadata_filters": {
                "document_category": "legal",
                "doc_id": "merger_agreement",
                "is_table": 1,
                "content_type": "table",
                "include_pii": True,
                "is_current_version": 0,
                "fiscal_year": "FY2023",
            },
        })
        assert out["extracted_filters"] == {"document_category": "legal"}
        assert out["parsed_intent"]["metadata_filters"] == {"document_category": "legal"}, (
            "unvetted proposals must not ride along in state either"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("category", ["made-up", "null", None, ["financial", "legal"], 3])
    async def test_an_unknown_category_is_no_filter(self, monkeypatch, category):
        out = await _agent1(monkeypatch, {"metadata_filters": {"document_category": category}})
        assert out["extracted_filters"] == {}

    @pytest.mark.asyncio
    async def test_downstream_flags_are_booleans(self, monkeypatch):
        """conditional_edges and retrieval_strategy read these with .get()."""
        out = await _agent1(monkeypatch, {
            "query_type": "financial", "requires_numerical_precision": "true",
            "requires_cross_document": None, "primary_intent": "kept for the trace",
        })
        intent = out["parsed_intent"]
        assert intent["requires_numerical_precision"] is True
        assert intent["requires_cross_document"] is False
        assert intent["primary_intent"] == "kept for the trace"

    @pytest.mark.asyncio
    async def test_trace_names_the_model_that_answered(self, monkeypatch):
        out = await _agent1(monkeypatch, {"query_type": "legal"}, served_model="gemini/second")
        assert out["agent_trace"][0]["model"] == "gemini/second"

    @pytest.mark.asyncio
    async def test_trace_keeps_the_selected_model_when_nothing_descended(self, monkeypatch):
        out = await _agent1(monkeypatch, {"query_type": "legal"})
        assert out["agent_trace"][0]["model"] == "gemini/selected"
        assert "degraded" not in out["agent_trace"][0]

    @pytest.mark.asyncio
    async def test_total_llm_failure_degrades_to_defaults(self, monkeypatch):
        """Retrieval, the gate and verification do not need this call."""
        out = await _agent1(monkeypatch, error=RuntimeError("every rung failed"))

        assert out["query_type"] == "summary"
        assert out["current_query"] == "What was FY2023 revenue?"
        assert out["extracted_filters"] == {} and out["sub_questions"] == []
        trace = out["agent_trace"][0]
        assert trace["degraded"] is True and "every rung failed" in trace["error"]


class TestQueryRewriterSchema:
    @pytest.mark.parametrize("raw", [None, [], "text", ["a"]])
    def test_a_non_object_becomes_the_defaults(self, raw):
        parsed = QueryRewriterOutput.from_llm(raw)
        assert parsed.rewritten_query is None
        assert parsed.updated_retrieval_config == {} and parsed.updated_metadata_filters == {}
        assert parsed.rewrite_reasoning == "" and parsed.alternative_formulations == []

    def test_a_null_category_survives_as_a_removal(self):
        """null means 'drop the category filter' and must reach the whitelist."""
        parsed = QueryRewriterOutput.from_llm(
            {"updated_metadata_filters": {"document_category": None}})
        assert parsed.updated_metadata_filters == {"document_category": None}


class TestAgent6Node:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("reply", [
        None, [], {"rewritten_query": None}, {"rewritten_query": "  "},
        {"rewritten_query": 5, "updated_retrieval_config": None,
         "updated_metadata_filters": "legal", "rewrite_reasoning": None,
         "alternative_formulations": None},
    ])
    async def test_malformed_shapes_keep_the_current_query(self, monkeypatch, reply):
        out = await _agent6(monkeypatch, reply)

        assert out["current_query"] == "current q"
        assert out["retrieval_config"] == {"reranker_top_k": 20}
        assert out["extracted_filters"] == {"document_category": "financial"}
        assert out["rewrite_iteration"] == 1
        assert out["rewrite_history"][0]["reasoning"] == ""
        assert out["rewrite_history"][0]["alternatives"] == []

    @pytest.mark.asyncio
    async def test_a_null_category_removes_the_filter(self, monkeypatch):
        out = await _agent6(monkeypatch, {
            "rewritten_query": "broader q",
            "updated_metadata_filters": {"document_category": None},
        })
        assert out["current_query"] == "broader q"
        assert out["extracted_filters"] == {}

    @pytest.mark.asyncio
    async def test_total_llm_failure_still_advances_the_loop(self, monkeypatch):
        """The rewrite loop is bounded by the iteration count; it must move."""
        out = await _agent6(monkeypatch, error=RuntimeError("every rung failed"))

        assert out["current_query"] == "current q"
        assert out["rewrite_iteration"] == 1
        assert out["agent_trace"][0]["degraded"] is True

    @pytest.mark.asyncio
    async def test_trace_names_the_model_that_answered(self, monkeypatch):
        out = await _agent6(monkeypatch, {"rewritten_query": "x"}, served_model="gemini/second")
        assert out["agent_trace"][0]["model"] == "gemini/second"
