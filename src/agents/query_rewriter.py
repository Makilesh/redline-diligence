"""
Agent 6 — Query Rewriter Agent.

Model: selected per call from the agent ladder (src/llm/model_registry.py) | Temp: 0.2 | Tokens: 600
JSON mode: response_format={"type": "json_object"}
Max iterations: 2

CRITICAL: JSON output key → state key mapping.
The rewriter's JSON output uses updated_retrieval_config and updated_metadata_filters,
but AgentState fields are retrieval_config and extracted_filters.
This node MUST map these correctly when returning state.
"""

import json

from src.agents.retrieval_strategy import apply_config_overrides, apply_filter_overrides
from src.llm.failures import classify_failure
from src.llm.litellm_wrapper import ServedModel, call_structured_agent
from src.llm.budget_tracker import BudgetTracker
from src.llm.prompt_templates.query_rewriter import (
    QUERY_REWRITER_SYSTEM_PROMPT,
    QUERY_REWRITER_USER_TEMPLATE,
)
from src.llm.schemas import QueryRewriterOutput
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def query_rewriter_node(state: AgentState) -> dict:
    """
    LangGraph node — rewrites query to improve retrieval quality.

    CRITICAL: The rewriter feeds back into the retrieval executor.
    Edge is: rewriter → executor (explicit, not conditional).
    This creates the self-correction loop.

    Args:
        state: Current AgentState with quality assessment results.

    Returns:
        Partial state dict with rewritten query and updated config.
    """
    iteration = state.get("rewrite_iteration", 0) + 1

    logger.info(
        "Agent 6: Query Rewriter starting",
        extra={"iteration": iteration, "max_iterations": 2},
    )

    tracker = await BudgetTracker.get_instance()
    choice = await tracker.get_model_for_agent()
    model = choice.model

    user_prompt = QUERY_REWRITER_USER_TEMPLATE.format(
        original_query=state["original_query"],
        current_query=state["current_query"],
        missing_aspects=json.dumps(state.get("missing_aspects", []), indent=2),
        quality_score=state.get("context_quality_score", 0.0),
        quality_breakdown=json.dumps(state.get("quality_breakdown", {}), indent=2),
        retrieval_config=json.dumps(state.get("retrieval_config", {}), indent=2),
        iteration=iteration,
    )

    # The call may descend the ladder; `served` names the model that answered.
    served = ServedModel(model)
    degraded: str | None = None
    try:
        result = await call_structured_agent(
            system_prompt=QUERY_REWRITER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            temperature=0.2,  # Slightly higher temp for creative rewrites
            max_tokens=600,
            api_key=choice.api_key,
            agent="query_rewriter",
            served=served,
        )
    except Exception as e:
        # No rung could produce a rewrite. The loop this node sits in is
        # bounded by rewrite_iteration, so carrying the query forward unchanged
        # costs one repeated retrieval pass and then lets the quality gate
        # decide — better than failing a query whose first retrieval already
        # happened and whose answer path does not depend on this call.
        logger.error(
            "Agent 6: rewrite unavailable, keeping the current query",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        result = {}
        # The trace reaches API callers; keep only the failure class (the full
        # error is in the log line above).
        degraded = classify_failure(e).kind.value
    model = served.model

    parsed = QueryRewriterOutput.from_llm(result)
    rewritten_query = parsed.rewritten_query or state["current_query"]

    # Map JSON output keys → AgentState keys. Both are whitelisted and clamped:
    # this is LLM output feeding straight into retrieval cost (top-k values)
    # and into Qdrant filters, so nothing it proposes is taken on trust.
    # include_pii and is_current_version are never accepted (compliance).
    updated_config = apply_config_overrides(
        state.get("retrieval_config", {}),
        parsed.updated_retrieval_config,
    )
    updated_filters = apply_filter_overrides(
        state.get("extracted_filters", {}),
        parsed.updated_metadata_filters,
    )

    logger.info(
        "Agent 6: Query Rewriter complete",
        extra={
            "iteration": iteration,
            "original_query": state["original_query"],
            "rewritten_query": rewritten_query,
        },
    )

    trace = {
        "agent": "query_rewriter",
        "model": model,
        "iteration": iteration,
        "rewritten_query": rewritten_query,
    }
    if degraded:
        trace["degraded"] = True
        trace["error"] = degraded

    return {
        "current_query": rewritten_query,
        "rewrite_iteration": iteration,
        "retrieval_config": updated_config,
        "extracted_filters": updated_filters,
        "rewrite_history": [
            {
                "iteration": iteration,
                "rewritten_query": rewritten_query,
                "reasoning": parsed.rewrite_reasoning,
                "alternatives": parsed.alternative_formulations,
            }
        ],
        "agent_trace": [trace],
    }
