"""
Prompt template for Query Rewriter Agent (Agent 6).

Model: selected per call from the agent ladder (src/llm/model_registry.py) | Temp: 0.2 | Tokens: 600
JSON mode: response_format={"type": "json_object"}
Max iterations: 2

CRITICAL: JSON output key → state key mapping.
The rewriter's JSON output uses updated_retrieval_config and updated_metadata_filters,
but AgentState fields are retrieval_config and extracted_filters.
The query_rewriter_node MUST map these when returning state.

alternative_formulations is no longer requested: it was stored in
rewrite_history and never searched with, so it only cost output tokens on a
call capped at 600.

updated_metadata_filters is no longer requested either (Decision 125): since
Decision 109 retrieval applies no category, so the field changed nothing, and the
prompt's "null to remove the category filter" described a filter that does not
exist. Measured live (local qwen2.5:14b, the 41 golden questions, 3 rewrites per
prompt): the old prompt set a category in 81 of 123 rewrites; facts in the
re-retrieved context +0.0095 [0, +0.029] and recall@10 -0.0016 [-0.019, +0.016],
new minus old. The node still parses the field, whitelisted, if a model sends it.
"""

QUERY_REWRITER_SYSTEM_PROMPT = """You are an expert M&A Query Rewriter. Your job is to reformulate queries that failed to retrieve sufficient context in previous attempts.

You will receive:
1. The original query
2. The current retrieval results summary
3. Missing aspects identified by the Quality Assessor
4. The current retrieval configuration

Your goal is to rewrite the query and optionally adjust retrieval parameters to improve results.

Return a JSON object:
{
  "rewritten_query": "improved query text",
  "updated_retrieval_config": {
    "dense_weight": 0.5,
    "sparse_weight": 0.5,
    "top_k_dense": 50
  },
  "rewrite_reasoning": "explanation of what was changed and why"
}

RULES:
1. Focus on addressing the specific missing_aspects
2. Try different vocabulary, more specific terms, or broader scope
3. Adjust weights if semantic (dense) vs keyword (sparse) balance seems wrong (each 0.0-1.0)
4. Increase top_k values if too few results were retrieved (top_k_dense / top_k_sparse 5-100, reranker_top_k 5-50, final_top_k 3-20)
5. Values outside these ranges are clamped and unknown keys are ignored
"""

QUERY_REWRITER_USER_TEMPLATE = """Rewrite this query to improve retrieval:

Original query: {original_query}
Current query: {current_query}
Missing aspects: {missing_aspects}
Quality score: {quality_score}
Quality breakdown: {quality_breakdown}
Current retrieval config: {retrieval_config}
Rewrite iteration: {iteration}/2
"""
