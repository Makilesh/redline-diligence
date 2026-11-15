"""
Agent 3 — Parallel Retrieval Executor.

No LLM calls. All embedding/reranking calls via run_in_executor (non-blocking).
Executes the retrieval pipeline: embed → hybrid search → RRF fusion → rerank.
"""

import asyncio
import time

from src.vector_db.reranker import embed_texts_async, get_embed_executor, rerank_pairs_async
from src.vector_db.hybrid_search import compute_sparse_bm25, dense_search, hybrid_search
from src.vector_db.filing_period import prefer_named_period
from src.vector_db.representation_cap import cap_table_representations
from src.vector_db.rrf_fusion import leading_document, reciprocal_rank_fusion
from src.vector_db.parent_child_retrieval import expand_context
from src.agents.retrieval_strategy import clamp_retrieval_config, get_retrieval_config
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Smallest share of the final context any one sub-question is guaranteed. The
# guarantee is the whole point of decomposing: without it the dominant facet
# simply crowds the others out again and nothing has been gained.
MIN_CHUNKS_PER_SUB_QUESTION = 2

def _passage_for_reranking(chunk: dict) -> str:
    """The text the cross-encoder scores: the chunk with its document and section."""
    parts = []
    if chunk.get("source_file"):
        parts.append(f"Document: {chunk['source_file']}")
    if chunk.get("section_heading"):
        parts.append(f"Section: {chunk['section_heading']}")
    parts.append(chunk.get("text", ""))
    return " | ".join(parts)


async def _candidates_for_query(
    query: str,
    query_vector: list[float],
    query_sparse,
    config: dict,
    deal_id: str,
    metadata_filters: dict,
) -> list[dict]:
    """
    Hybrid search + RRF for one query: the candidates its rerank will score.

    Payloads are taken from the search responses themselves. Both searches
    already return them, so the scroll that used to re-fetch the fused ids was a
    second round trip to Qdrant for data already in hand.

    Args:
        query: The query or sub-question (logging only).
        query_vector: Its dense embedding.
        query_sparse: Its BM25 sparse vector.
        config: Retrieval config (weights, top-k values).
        deal_id: Deal scope.
        metadata_filters: Filters, already relaxed/augmented by the caller.

    Returns:
        Up to reranker_top_k chunk payloads in fused-rank order, then (with
        document_focus_k) up to document_focus_k more from the leading
        document.
    """
    dense_results, sparse_results = await hybrid_search(
        query_text=query,
        query_vector=query_vector,
        query_sparse=query_sparse,
        deal_id=deal_id,
        metadata_filters=metadata_filters,
        top_k_dense=config.get("top_k_dense", 40),
        top_k_sparse=config.get("top_k_sparse", 40),
    )
    # Another year's filing of the same company is set aside when the question
    # names a period that one of that company's filings covers (filing_period.py;
    # a no-op for documents without filing metadata).
    dense_results, sparse_results, _ = await prefer_named_period(
        query, dense_results, sparse_results,
        query_vector=query_vector, query_sparse=query_sparse, deal_id=deal_id,
        metadata_filters=metadata_filters,
        top_k_dense=config.get("top_k_dense", 40),
        top_k_sparse=config.get("top_k_sparse", 40),
    )

    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=60,
        dense_weight=config.get("dense_weight", 0.6),
        sparse_weight=config.get("sparse_weight", 0.4),
    )

    payloads = {p.payload["chunk_id"]: p.payload for p in (*dense_results, *sparse_results)}
    # One converter table holds one of the reranker's slots (representation_cap.py).
    ranked = cap_table_representations([cid for cid, _ in fused if cid in payloads], payloads)
    candidates = [payloads[cid] for cid in ranked[: config.get("reranker_top_k", 20)]]

    # Document focus: search again inside the document the fused list points
    # at, and let the cross-encoder choose among both sets.
    #
    # On a room of similar contracts the corpus-wide lists find the right
    # agreement and the wrong passage: a deal-point question ("the closing
    # condition on compliance with covenants") matches that clause in every
    # merger agreement, and the party names only match the preamble. Inside
    # the leading document the dense search has only one agreement's clauses
    # to choose from (a BM25 search there added nothing on top of it).
    # Candidates are added, never removed, so a question
    # spanning several documents keeps everything it had; the reranker
    # decides. Measured in Decision 67.
    focus_k = int(config.get("document_focus_k", 0) or 0)
    if focus_k > 0 and candidates:
        doc_id, _share = leading_document(
            fused, {cid: p.get("doc_id") for cid, p in payloads.items()}
        )
        if doc_id:
            in_document = await dense_search(
                query_vector=query_vector,
                deal_id=deal_id,
                metadata_filters=metadata_filters,
                top_k=focus_k,
                doc_id=doc_id,
            )
            seen = {c["chunk_id"] for c in candidates}
            candidates += [p.payload for p in in_document if p.payload["chunk_id"] not in seen]
    return candidates


async def _retrieve_for_queries(
    queries: list[str],
    config: dict,
    deal_id: str,
    filters_per_query: list[dict],
) -> list[list[dict]]:
    """
    Runs one retrieval pass per query and reranks every pass in a single call.

    Embed → BM25 → hybrid search → RRF → rerank, returning for each query the
    chunks scored against *that* query. Reranking against the query that fetched
    them is the point of keeping the passes separate: a passage holding only the
    share price scores poorly against "what is the implied EV/EBITDA multiple",
    but scores highly against "what is the per-share merger consideration".
    Scoring each candidate against the sub-question that went looking for it is
    what makes the facet survive to the final context.

    The passes share their model calls, though. Each used to embed, encode and
    rerank on its own, which made a decomposed question pay the cross-encoder
    once per pass, one after another: measured on the retrieval eval, an
    undecomposed question took ~0.8s in this node and a decomposed one 2.2-4.6s.
    All queries are now embedded in one call and every (query, passage) pair of
    every pass goes to the cross-encoder together.

    Args:
        queries: The parent query followed by any sub-questions.
        config: Retrieval config (weights, top-k values), shared by all passes.
        deal_id: Deal scope.
        filters_per_query: Filters for each query, already relaxed/augmented.

    Returns:
        One list per query, in order: chunks sorted by descending reranker score,
        each carrying reranker_score. No score cut-off is applied here: a fixed
        threshold measurably cost fact coverage (91.6% -> 85.2% on the retrieval
        eval), and irrelevant context is the Quality Assessor's call, not this
        function's.
    """
    query_vectors = await embed_texts_async(queries)

    loop = asyncio.get_running_loop()
    query_sparses = await loop.run_in_executor(
        get_embed_executor(), lambda: [compute_sparse_bm25(q) for q in queries]
    )

    candidates = await asyncio.gather(*[
        _candidates_for_query(
            query, query_vectors[i].tolist(), query_sparses[i], config, deal_id,
            filters_per_query[i],
        )
        for i, query in enumerate(queries)
    ])

    pairs = [
        (query, _passage_for_reranking(chunk))
        for query, chunks in zip(queries, candidates)
        for chunk in chunks
    ]
    scores = await rerank_pairs_async(pairs) if pairs else []

    results: list[list[dict]] = []
    offset = 0
    for chunks in candidates:
        scored = []
        for chunk, score in zip(chunks, scores[offset: offset + len(chunks)]):
            chunk = dict(chunk)
            chunk["reranker_score"] = float(score)
            scored.append(chunk)
        offset += len(chunks)
        scored.sort(key=lambda c: c["reranker_score"], reverse=True)
        results.append(scored)
    return results


def _merge_by_quota(
    per_query_results: list[list[dict]],
    final_top_k: int,
) -> list[dict]:
    """
    Merges per-sub-question results so every facet is represented.

    A plain union followed by a global top-k does not work, and failing to see
    why is how the original multi-hop weakness survived: the facet with the
    strongest passages simply fills all the slots, and the other facet — the one
    decomposition was supposed to rescue — is ranked out again. Measured before
    this change: asked for an EV/EBITDA multiple, retrieval returned EBITDA
    passages and no share price, and the engine correctly reported it could not
    compute the multiple.

    So results are interleaved round-robin instead: position 1 from each
    sub-question, then position 2, and so on. The first slots therefore cover
    every facet by construction, and truncation to the context budget removes
    the tail rather than an entire facet. Duplicates keep their highest score.

    Args:
        per_query_results: One ranked list per query, best first.
        final_top_k: Total chunks to return.

    Returns:
        Merged chunks, de-duplicated by chunk_id.
    """
    merged: list[dict] = []
    seen: dict[str, dict] = {}

    depth = max((len(r) for r in per_query_results), default=0)
    for rank in range(depth):
        for results in per_query_results:
            if rank >= len(results):
                continue
            chunk = results[rank]
            key = chunk.get("chunk_id") or f"{chunk.get('source_file')}:{rank}"

            if key in seen:
                # Same passage found by another sub-question — keep the stronger
                # score so ordering reflects its best evidence of relevance.
                if chunk["reranker_score"] > seen[key]["reranker_score"]:
                    seen[key]["reranker_score"] = chunk["reranker_score"]
                continue

            seen[key] = chunk
            merged.append(chunk)
            if len(merged) >= final_top_k:
                return merged

    return merged


async def retrieval_executor_node(state: AgentState) -> dict:
    """
    LangGraph node — executes full retrieval pipeline.

    Pipeline:
    1. Get or use existing retrieval config
    2. Embed query
    3. Compute BM25 sparse vector
    4. Hybrid search (dense + sparse in parallel)
    5. RRF fusion
    6. Rerank
    7. Context expansion (parent + sibling)

    Args:
        state: Current AgentState with current_query, query_type,
               parsed_intent, extracted_filters, deal_id.

    Returns:
        Partial state dict with retrieval results.
    """
    start = time.monotonic()
    query = state["current_query"]
    deal_id = state["deal_id"]

    logger.info(
        "Agent 3: Retrieval Executor starting",
        extra={"query": query, "deal_id": deal_id},
    )

    # Step 1: Get retrieval config (deterministic — Agent 2)
    if state.get("retrieval_config"):
        config = state["retrieval_config"]
    else:
        config = get_retrieval_config(state["query_type"], state["parsed_intent"])
    # Clamp on read as well: the config in state may have been rewritten by an LLM.
    config = {**config, **clamp_retrieval_config(config)}

    # Step 2: Build filters
    # include_pii is injected here rather than living in extracted_filters, because
    # Agents 1 and 6 deliberately strip that key from anything an LLM produced. The
    # only trusted source is the authenticated request, which lands on the state root.
    include_pii = bool(state.get("include_pii", False))
    proposed = state.get("extracted_filters")
    metadata_filters = {
        **(proposed if isinstance(proposed, dict) else {}),  # a model may return null
        "include_pii": include_pii,
    }

    # Progressive filter relaxation.
    #
    # document_category is Agent 1's *guess* at which kind of document holds the
    # answer, applied as a hard `must` condition. When that guess is wrong the
    # answer is removed from the search space entirely and no amount of query
    # rewriting can recover it — the rewriter only changes text, never filters.
    #
    # Measured on the golden set: legal_01 ("per-share merger consideration") was
    # classified financial, so the merger agreement — the one document containing
    # the answer — was filtered out; fin_05 (credit facility terms) was classified
    # legal and excluded the financials; comp_02 (valuation methodologies) was
    # classified financial and excluded the board deck. All three refused after
    # burning both rewrite attempts against a search space that could never
    # contain the answer.
    #
    # It is also compounding two error sources: an LLM guessing a category that
    # was itself assigned by a heuristic classifier at ingestion time.
    #
    # So: keep the filter on the first attempt, where it buys precision, and drop
    # it once retrieval has already failed and we are explicitly trading precision
    # for recall. deal_id, version and PII filters are NEVER relaxed — those are
    # isolation and compliance constraints, not relevance hints.
    if state.get("rewrite_iteration", 0) >= 1:
        dropped = metadata_filters.pop("document_category", None)
        if dropped:
            logger.info(
                "Relaxing document_category filter after failed retrieval",
                extra={"dropped_category": dropped,
                       "rewrite_iteration": state.get("rewrite_iteration")},
            )
    # Step 3: Plan the retrieval passes.
    #
    # For a pointed question this is a single pass on the query itself. For a
    # multi-hop or comparative question, Agent 1 has decomposed it into atomic
    # sub-questions, and each gets its own pass. The parent query is retained as
    # the first pass so passages relevant to the question as a whole — which no
    # single sub-question asks for — are still reachable.
    sub_questions = [s for s in (state.get("sub_questions") or []) if s.strip()]
    retrieval_queries = [query] + sub_questions

    final_top_k = config.get("final_top_k", 10)

    # Decomposition is ADDITIVE, not a redistribution of a fixed budget.
    #
    # Splitting final_top_k across the passes was the obvious design and it is
    # wrong: the parent query loses most of its slots, so a question whose answer
    # sat in one passage the parent already found can end up losing it. Measured
    # on comp_05 (comparing severance multiples across executives): the single
    # chunk holding the whole table was squeezed out and fact coverage went from
    # 20% to 0% — decomposition made a question worse that it should not have
    # touched at all.
    #
    # So the parent keeps its full budget and each sub-question adds a small
    # fixed allocation on top. Decomposition can then only add evidence, never
    # displace it; the cost is a slightly larger synthesis context on the
    # minority of queries that decompose.
    if sub_questions:
        parent_k = final_top_k
        sub_k = MIN_CHUNKS_PER_SUB_QUESTION
        merged_budget = parent_k + sub_k * len(sub_questions)
        logger.info(
            "Agent 3: decomposed retrieval",
            extra={"sub_questions": len(sub_questions),
                   "parent_k": parent_k, "per_sub_k": sub_k,
                   "context_budget": merged_budget},
        )
    else:
        parent_k = sub_k = final_top_k
        merged_budget = final_top_k

    # Each pass fetches the larger of the two budgets; truncation to the correct
    # per-pass share happens at merge time.
    pass_config = {**config, "final_top_k": max(parent_k, sub_k)}

    # A sub-question must not inherit the parent's document_category.
    #
    # That filter is Agent 1's guess at which document answers the question as a
    # whole. A sub-question exists precisely because it asks for a *different*
    # fact, which usually lives somewhere else — so applying the parent's guess
    # to it re-creates the problem decomposition was meant to solve.
    #
    # Measured: "implied EV/EBITDA multiple" decomposed correctly into share
    # price, share count and FY2023 EBITDA, but every pass inherited one category
    # filter and all three returned chunks from the same document. The engine
    # then reported it could not compute the multiple — decomposition had done
    # its job and the filter had thrown the result away.
    #
    # deal_id, version and PII filters still apply to every pass: those are
    # isolation and compliance constraints, not relevance hints.
    sub_question_filters = {
        k: v for k, v in metadata_filters.items() if k != "document_category"
    }

    # The parent pass drops the category too, once a query has been decomposed.
    #
    # Decomposing is Agent 1 stating that the answer spans several facts. Those
    # facts routinely live in documents of different categories, so constraining
    # the parent search to one category contradicts the judgment that produced
    # the sub-questions in the first place.
    #
    # Measured on "implied EV/EBITDA multiple of the Vertex deal": the aggregate
    # merger consideration of $696 million is stated in the regulatory memo and
    # in the merger agreement. Agent 1 classifies the question `financial` or
    # `legal`, and either filter removes the chunk that states it — retrieval
    # then returns a context full of plausible, high-scoring financial chunks
    # with the one required number absent, and the engine reports that the
    # multiple cannot be computed.
    #
    # Progressive relaxation (see the rewrite path) does not rescue this. It
    # triggers on *low* context quality, and this context scores well — it is
    # incomplete, not weak. Nothing downstream can tell the difference, which is
    # why the filter has to come off before the search rather than after it.
    #
    # deal_id, version and PII filters still apply to every pass: those are
    # isolation and compliance constraints, not relevance hints.
    parent_filters = sub_question_filters if sub_questions else metadata_filters

    def _filters_for(index: int) -> dict:
        return parent_filters if index == 0 else sub_question_filters

    # Passes are independent: their searches run concurrently and their model
    # work is batched (see _retrieve_for_queries).
    per_query_results = await _retrieve_for_queries(
        retrieval_queries,
        pass_config,
        deal_id,
        [_filters_for(i) for i in range(len(retrieval_queries))],
    )

    if len(retrieval_queries) == 1:
        reranked = per_query_results[0][:final_top_k]
    else:
        # Parent first at full budget, then each sub-question's small allocation.
        capped = [per_query_results[0][:parent_k]] + [
            r[:sub_k] for r in per_query_results[1:]
        ]
        reranked = _merge_by_quota(capped, final_top_k=merged_budget)
        # Present the merged context best-first; the round-robin order exists to
        # guarantee coverage during selection, not to dictate reading order.
        reranked.sort(key=lambda c: c["reranker_score"], reverse=True)

    # Step 7: Context expansion
    expanded = await expand_context(
        chunks=reranked,
        include_parents=config.get("use_parent_expansion", True),
        include_siblings=config.get("use_sibling_expansion", True),
        include_pii=include_pii,
        deal_id=deal_id,
    )

    elapsed_ms = (time.monotonic() - start) * 1000

    candidates_per_pass = [len(r) for r in per_query_results]

    # Tables whose sibling lookup failed: the answer is then written without the
    # exact grid, which must be visible in the trace rather than only in a log.
    siblings_missing = sorted(
        {c["table_id"] for c in expanded if c.get("table_siblings_missing")}
    )
    section_tables_missing = sorted(
        {c.get("source_file", "") for c in expanded if c.get("section_tables_missing")}
    )

    logger.info(
        "Agent 3: Retrieval Executor complete",
        extra={
            "retrieval_passes": len(retrieval_queries),
            "candidates_per_pass": candidates_per_pass,
            "reranked_count": len(reranked),
            "expanded_count": len(expanded),
            "sources": len({c.get("source_file") for c in reranked}),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )

    # dense_results / sparse_results / fused_results are no longer meaningful as
    # single lists: with decomposition there are N of each, one per pass. The
    # per-pass candidate counts carry the same diagnostic signal without
    # pretending a single fused ranking still exists.
    return {
        "retrieval_config": config,
        "dense_results": [],
        "sparse_results": [],
        "fused_results": [],
        "retrieval_passes": [
            {"query": q, "candidates": n}
            for q, n in zip(retrieval_queries, candidates_per_pass)
        ],
        "reranked_results": reranked,
        "expanded_context": expanded,
        "agent_trace": [
            {
                "agent": "retrieval_executor",
                "elapsed_ms": round(elapsed_ms, 2),
                "retrieval_passes": len(retrieval_queries),
                "sub_questions": sub_questions,
                "reranked_count": len(reranked),
                "sources": sorted(
                    {c.get("source_file", "") for c in reranked} - {""}
                ),
                **({"table_siblings_missing": siblings_missing} if siblings_missing else {}),
                **({"section_tables_missing": section_tables_missing}
                   if section_tables_missing else {}),
            }
        ],
    }
