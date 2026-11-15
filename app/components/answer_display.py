"""
Answer display component — renders answer with confidence badge and validation status.
"""

import html

import streamlit as st

from app.components.verification import answer_verdict, render_claim_checks
from app.styles import escape_currency, pill, stat_row

_VERDICT_PILL = {
    "verified": ("✓", "ok"),
    "partial": ("⚠", "warn"),
    "failed": ("✕", "bad"),
    "no_claims": ("○", "muted"),
    "unknown": ("?", "warn"),
}


def render_answer(
    answer: str,
    confidence_score: float,
    validation_status: str,
    hallucination_flags: list[str] | None = None,
    query_type: str = "",
    latency_ms: float = 0.0,
    rewrite_iterations: int = 0,
    claim_checks: list[dict] | None = None,
    outcome: str = "",
    answer_truncated: bool = False,
) -> None:
    """
    Renders the main answer with confidence badge, validation status, the
    per-claim verification record and the validator's notes.

    Args:
        answer: Generated answer text with citations.
        confidence_score: Answer confidence 0.0–1.0.
        validation_status: "passed", "warning", or "failed" — or anything
            else, which is shown as not verified rather than as a pass.
        hallucination_flags: The validator's free-text notes.
        query_type: Detected query type.
        latency_ms: Total pipeline latency.
        rewrite_iterations: Number of query rewrites performed.
        claim_checks: Per-claim verification results.
        outcome: The run's outcome (answered | answered_unverified).
        answer_truncated: The answer stopped at the model's output limit.
    """
    claim_checks = claim_checks or []
    # Header metrics. A compact strip rather than st.metric cards: these are
    # context for the answer below, not the subject of the page, and four full
    # cards pushed the answer itself below the fold.
    st.markdown(
        stat_row(
            [
                ("Confidence", f"{confidence_score:.0%}"),
                ("Query Type", (query_type or "—").replace("_", " ").title()),
                ("Latency", f"{latency_ms / 1000:.1f}s"),
                ("Rewrites", str(rewrite_iterations)),
            ]
        ),
        unsafe_allow_html=True,
    )

    # Verdict banner. Derived from the status AND the claim list: "passed" is
    # also the server's default for a missing status, and an answer with no
    # checkable statement passes vacuously — neither may read as verified.
    st.write("")
    key, label, detail = answer_verdict(
        {
            "validation_status": validation_status,
            "claim_checks": claim_checks,
            "outcome": outcome,
            "answer": answer,
            "answer_truncated": answer_truncated,
        }
    )
    icon, variant = _VERDICT_PILL[key]
    st.markdown(
        pill(f"{icon} {label}", variant) + f"<span class='dd-cite-meta'>{html.escape(detail)}</span>",
        unsafe_allow_html=True,
    )

    # Answer body — a bordered container rather than raw HTML, so the model's
    # markdown (lists, tables, emphasis) still renders correctly.
    #
    # The empty marker span is how the stylesheet finds this container. Wrapping
    # the answer in an opening <div> and closing it in a later st.markdown call
    # does not work: Streamlit renders every markdown call into its own DOM
    # container, so the parser closes the div immediately and the answer ends up
    # outside it. A marker plus `:has()` targets the real container instead.
    st.write("")
    with st.container(border=True):
        st.markdown("<span class='dd-answer-marker'></span>", unsafe_allow_html=True)
        st.markdown(escape_currency(answer))
        if answer_truncated:
            st.caption("✂ Cut off here: the model reached its output limit, so the answer is incomplete.")

    if claim_checks:
        st.write("")
        st.markdown(pill("CLAIM CHECKS", "info"), unsafe_allow_html=True)
        render_claim_checks(claim_checks)

    if hallucination_flags:
        st.write("")
        # With per-claim checks above, the notes mostly repeat them.
        with st.expander(f"Validator notes ({len(hallucination_flags)})", expanded=not claim_checks):
            for flag in hallucination_flags:
                st.markdown(f"- {escape_currency(html.escape(flag))}")


def render_unavailable(request_id: str = "") -> None:
    """
    Renders the synthesis_failed outcome: no model returned a usable answer.

    Not a refusal — a refusal is a judgment that the data room does not support
    an answer, and this says nothing about the question. The synthesizer's
    apology text is not shown: it is not an answer and was never verified.

    Args:
        request_id: Correlation id for the server log.
    """
    st.markdown(pill("NO ANSWER — MODELS UNAVAILABLE", "muted"), unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "The documents were searched, but every language model the engine tried "
            "failed to return a usable answer, so nothing was written or verified. "
            "This is a service problem, not a finding about the question or the data "
            "room — run it again in a minute."
        )
    if request_id:
        st.caption(f"Request ID: `{request_id}`")


def render_refusal(quality_score: float, rewrite_count: int) -> None:
    """
    Renders a styled refusal message when context is insufficient.

    A refusal is a designed terminal state, not an error: the engine prefers a
    traceable "I don't know" over a confident guess on a deal-critical number.

    Args:
        quality_score: Best achieved quality score.
        rewrite_count: Total search attempts.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best Quality Score", f"{quality_score:.0%}")
    with col2:
        st.metric("Search Attempts", str(rewrite_count + 1))

    st.write("")
    st.markdown(pill("⊘ INSUFFICIENT CONTEXT", "bad"), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            "The engine could not find sufficient relevant information in the data "
            "room to answer this question accurately, so it declined to answer "
            "rather than produce an unsupported one."
        )

    st.write("")
    st.markdown(
        "**Try this:**\n"
        "- Confirm the relevant documents have been uploaded to this deal\n"
        "- Rephrase using the terminology the documents themselves use\n"
        "- Narrow the scope (a specific fiscal year, or a single document type)"
    )
