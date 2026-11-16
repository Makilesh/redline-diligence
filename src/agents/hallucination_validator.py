"""
Agent 8 — Hallucination Validation Agent.

Deterministic-first claim verification (see src/verification/claim_checker.py):

1. every figure in the answer is grounded against the retrieved context;
2. every claim is scored for entailment by a local NLI cross-encoder;
3. only the claims neither check can decide go to an LLM judge, batched into
   one JSON-mode call — so the common case makes no LLM call at all.

The previous design sent the whole answer and context to a lite model and
reported that model's self-assessed `confidence_score` to users: a weaker model
grading a stronger one, on every query, with a number nobody could reproduce.
Confidence is now computed from the claim checks (definition in
claim_checker.py), and the per-claim results are kept in `claim_checks`.
"""

import time

from src.agents.answer_synthesizer import SYNTHESIS_FAILED_ANSWER
from src.llm.litellm_wrapper import call_verification_agent_with_model
from src.llm.prompt_templates.hallucination_validator import (
    HALLUCINATION_VALIDATOR_SYSTEM_PROMPT,
    HALLUCINATION_VALIDATOR_USER_TEMPLATE,
)
from src.verification.claim_checker import verify_answer
from src.verification.evidence import chunk_evidence_text
from src.verification.nli import nli_enabled
from src.verification.prompt_safety import chunk_attributes, wrap_document
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Per-document character budget inside the judge prompt. The judge sees only
# the documents relevant to the undecided claims, so this bounds cost without
# hiding the evidence the way the old 500-character truncation did.
_JUDGE_DOC_CHARS = 9000


def build_judge_prompt(query: str, claims: list[dict], documents: list[dict]) -> str:
    """
    Builds the judge's user prompt: numbered claims plus delimited documents.

    Args:
        query: The user's original question.
        claims: [{"id", "claim"}] to judge.
        documents: Chunks to judge against.

    Returns:
        User prompt text.
    """
    claim_lines = "\n".join(f"{c['id']}. {c['claim']}" for c in claims)
    doc_parts = []
    for i, chunk in enumerate(documents, 1):
        # Same evidence text the synthesizer was shown (parent section included).
        body = chunk_evidence_text(chunk)
        doc_parts.append(wrap_document(i, body[:_JUDGE_DOC_CHARS], **chunk_attributes(chunk)))
    return HALLUCINATION_VALIDATOR_USER_TEMPLATE.format(
        query=query,
        claims=claim_lines,
        context="\n\n".join(doc_parts),
    )


async def _llm_judge(query: str, claims: list[dict], documents: list[dict]):
    """
    One batched LLM call for the claims the deterministic checks left open.

    Args:
        query: The user's original question.
        claims: [{"id", "claim"}] to judge.
        documents: Chunks to judge against.

    Returns:
        ({claim id: verdict dict}, model that served the call).
    """
    result, model = await call_verification_agent_with_model(
        system_prompt=HALLUCINATION_VALIDATOR_SYSTEM_PROMPT,
        user_prompt=build_judge_prompt(query, claims, documents),
        temperature=0.0,
        max_tokens=1500,
        agent="hallucination_validator",
    )
    verdicts: dict[int, dict] = {}
    for item in result.get("verdicts", []) if isinstance(result, dict) else []:
        try:
            verdicts[int(item.get("id"))] = item
        except (TypeError, ValueError, AttributeError):
            continue
    return verdicts, model


def _synthesis_failed(state: AgentState, answer: str) -> bool:
    """
    True when the "answer" in state is the synthesizer's failure apology.

    The synthesizer marks the failure on its trace entry; the text itself is
    also compared, so a state rebuilt without traces is still recognised.

    Args:
        state: Current AgentState.
        answer: state["generated_answer"].

    Returns:
        Whether the latest synthesis produced no answer.
    """
    if answer.strip() == SYNTHESIS_FAILED_ANSWER:
        return True
    for entry in reversed(state.get("agent_trace") or []):
        if isinstance(entry, dict) and entry.get("agent") == "answer_synthesizer":
            return bool(entry.get("synthesis_failed"))
    return False


# Shown with an answer the synthesizer streamed and the model cut off at its
# output budget (state["answer_truncated"]). What is there can be checked; what
# is missing cannot, so such an answer is never reported as "passed".
TRUNCATED_ANSWER_FLAG = (
    "The answer was cut off at the model's output limit and may be incomplete: "
    "statements after the cut were never written."
)


def _mark_truncation(state: AgentState, status: str, flags: list[str]) -> tuple[str, list[str]]:
    """
    Downgrades a cut-off answer's verdict: "passed" becomes "warning", and the
    flag says why. "warning" and "failed" are kept — a cut answer whose claims
    also failed should still be revised.
    """
    if not state.get("answer_truncated"):
        return status, flags
    return ("warning" if status == "passed" else status), [*flags, TRUNCATED_ANSWER_FLAG]


async def hallucination_validator_node(state: AgentState) -> dict:
    """
    LangGraph node — verifies the answer claim by claim.
    Populates: confidence_score, hallucination_flags, validation_status,
    validation_attempt, claim_checks, agent_trace.

    A verification failure never discards the answer: if any step raises, the
    answer is kept with validation_status "warning" and a "validation
    unavailable" flag. Losing a good answer because a verifier was down is the
    worse outcome for the reviewer.

    Args:
        state: Current AgentState with generated_answer and context.

    Returns:
        Partial state dict with validation results.
    """
    logger.info("Agent 8: Hallucination Validator starting")
    attempt = state.get("validation_attempt", 0) + 1

    answer = state.get("generated_answer", "")
    chunks = state.get("expanded_context") or state.get("reranked_results") or []

    # A synthesis that failed outright returns an apology, not an answer. It used
    # to be verified like one: its two sentences went to the LLM judge — a model
    # call made precisely when the provider is failing — and came back
    # "warning" at confidence 0.5, or "passed" at 1.0 if the judge was generous.
    # Nothing was answered, so nothing is verified and nothing is vouched for.
    if _synthesis_failed(state, answer):
        logger.info("Agent 8: Skipping validation — synthesis produced no answer")
        return {
            "confidence_score": 0.0,
            "hallucination_flags": [
                "No answer was generated: the language model did not return a usable response."
            ],
            "claim_checks": [],
            "validation_status": "warning",
            "validation_attempt": attempt,
            "agent_trace": [
                {
                    "agent": "hallucination_validator",
                    "skipped": True,
                    "reason": "synthesis_failed",
                    "validation_status": "warning",
                    "confidence_score": 0.0,
                    "llm_calls": 0,
                }
            ],
        }

    # Skip validation for forced refusals
    if state.get("force_refusal") or not answer:
        logger.info("Agent 8: Skipping validation — refusal or empty answer")
        return {
            "confidence_score": 0.0,
            "hallucination_flags": [],
            "claim_checks": [],
            "validation_status": "passed",
            "validation_attempt": attempt,
            "agent_trace": [
                {"agent": "hallucination_validator", "skipped": True}
            ],
        }

    start = time.monotonic()
    query = state.get("original_query") or state.get("current_query", "")
    try:
        report = await verify_answer(answer, chunks, query, judge=_llm_judge)
    except Exception as e:
        logger.error(
            "Agent 8: verification failed; keeping the answer with a warning",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        _, flags = _mark_truncation(
            state, "warning", [f"validation unavailable: {type(e).__name__}: {e}"]
        )
        return {
            "confidence_score": 0.0,
            "hallucination_flags": flags,
            "claim_checks": [],
            "validation_status": "warning",
            "validation_attempt": attempt,
            "agent_trace": [
                {
                    "agent": "hallucination_validator",
                    "validation_status": "warning",
                    "validation_unavailable": True,
                    "error": str(e),
                    "elapsed_ms": round((time.monotonic() - start) * 1000, 1),
                }
            ],
        }

    logger.info(
        "Agent 8: Hallucination Validator complete",
        extra={
            "validation_status": report.validation_status,
            "confidence_score": report.confidence,
            "claims": report.status_counts,
            "methods": report.method_counts,
            "numeric": report.numeric_counts,
            "llm_calls": report.llm_calls,
            "elapsed_ms": report.elapsed_ms,
        },
    )

    # The model names reported are the ones that did the work: the NLI model
    # that scored claims and, only if one was needed, the LLM that judged the
    # rest — never a guess at the first rung of the ladder.
    models = [m for m in (report.nli_model, report.llm_model) if m]
    status, flags = _mark_truncation(state, report.validation_status, report.flags)

    return {
        "confidence_score": report.confidence,
        "hallucination_flags": flags,
        "claim_checks": report.claim_checks,
        "validation_status": status,
        "validation_attempt": attempt,
        "agent_trace": [
            {
                "agent": "hallucination_validator",
                "model": " + ".join(models) if models else "deterministic",
                "nli_model": report.nli_model,
                "nli_enabled": nli_enabled(),
                "llm_model": report.llm_model,
                "llm_calls": report.llm_calls,
                "validation_status": status,
                "answer_truncated": bool(state.get("answer_truncated")),
                "confidence_score": report.confidence,
                "flags_count": len(flags),
                "claims_checked": len(report.claim_checks),
                "status_counts": report.status_counts,
                "method_counts": report.method_counts,
                "numeric_counts": report.numeric_counts,
                "notes": report.notes,
                # Figures checked against the documents their claims cite
                # (Decision 78): counts, and each figure cited to a document
                # that does not state it.
                "citation_counts": report.citation_counts,
                "citation_mismatches": [
                    {**m, "claim": m["claim"][:240]} for m in report.citation_mismatches
                ],
                "attempt": attempt,
                "elapsed_ms": report.elapsed_ms,
                # Compact per-claim view for the UI; the full checks are in state.
                "claim_checks": [
                    {
                        "claim": c["claim"][:240],
                        "status": c["status"],
                        "method": c["method"],
                        "score": c["score"],
                        "evidence_source": c["evidence_source"],
                        **({"citation_mismatch": True} if c.get("citation_mismatch") else {}),
                    }
                    for c in report.claim_checks
                ],
            }
        ],
    }
