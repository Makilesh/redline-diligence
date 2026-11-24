"""
Answer display component — renders answer with confidence badge and validation status.
"""

import html

import streamlit as st

from app.components.verification import (
    NOT_FOUND_DETAIL,
    answer_verdict,
    evidence_cut_note,
    injection_note,
    render_claim_checks,
)
from app.markdown_safety import inert_markdown
from app.styles import escape_currency, pill, stat_row

_VERDICT_PILL = {
    "verified": ("✓", "ok"),
    "partial": ("⚠", "warn"),
    "failed": ("✕", "bad"),
    "no_claims": ("○", "muted"),
    "unknown": ("?", "warn"),
    "not_found": ("⌕", "warn"),
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
    injection_review: dict | None = None,
    evidence_cut: dict | None = None,
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
        outcome: The run's outcome (answered | answered_unverified | not_found).
        answer_truncated: The answer stopped at the model's output limit.
        injection_review: QueryResponse.injection_review — retrieved passages
            with instruction-like text, by location (Decision 82).
        evidence_cut: QueryResponse.evidence_cut — what a fitted local context
            left out of the answer's evidence (Decision 118).
    """
    claim_checks = claim_checks or []
    # Header metrics. A compact strip rather than st.metric cards: these are
    # context for the answer below, not the subject of the page, and four full
    # cards pushed the answer itself below the fold.
    st.markdown(
        stat_row(
            [
                # A score on "the documents do not say" reads as confidence that
                # they do not, which nothing measured (Decision 112).
                ("Confidence", "—" if outcome == "not_found" else f"{confidence_score:.0%}"),
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
            "injection_review": injection_review or {},
            "evidence_cut": evidence_cut,
        }
    )
    icon, variant = _VERDICT_PILL[key]
    # html.escape keeps the API's text from opening a tag; inert_markdown keeps
    # it from showing an image, which markdown still renders inside the span.
    st.markdown(
        pill(f"{icon} {label}", variant)
        + f"<span class='dd-cite-meta'>{inert_markdown(html.escape(detail))}</span>",
        unsafe_allow_html=True,
    )
    # Where instruction-like text sits behind this answer. Its flags fold away
    # below the claim list, so the location is stated up front, in one line.
    note = injection_note(injection_review)
    if note:
        text, note_variant = note
        st.markdown(
            pill("INSTRUCTION-LIKE TEXT", note_variant)
            + f"<span class='dd-cite-meta'>{inert_markdown(escape_currency(html.escape(text)))}</span>",
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
        # An image in the answer would load as soon as it is shown: the alt
        # text instead, as in the web console (Decision 76; app/markdown_safety.py).
        st.markdown(inert_markdown(escape_currency(answer)))
        if answer_truncated:
            st.caption("✂ Cut off here: the model reached its output limit, so the answer is incomplete.")
        cut_note = evidence_cut_note(evidence_cut)
        if cut_note:
            st.caption(f"⧉ {cut_note}")

    if claim_checks:
        st.write("")
        st.markdown(pill("CLAIM CHECKS", "info"), unsafe_allow_html=True)
        render_claim_checks(claim_checks)

    if hallucination_flags:
        st.write("")
        # With per-claim checks above, the notes mostly repeat them.
        with st.expander(f"Validator notes ({len(hallucination_flags)})", expanded=not claim_checks):
            for flag in hallucination_flags:
                st.markdown(f"- {inert_markdown(escape_currency(html.escape(flag)))}")


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


def render_not_found(rewrite_count: int) -> None:
    """
    Renders the banner above a "not found" answer (Decision 112).

    The answer said the documents do not contain what was asked. Like a refusal
    it is not an answer, and unlike one the pipeline did find passages: the
    model's explanation follows, with what it did find, under this banner.

    Args:
        rewrite_count: Query rewrites performed (search attempts minus one).
    """
    st.markdown(pill("⌕ NOT FOUND IN THE PASSAGES SEARCHED", "warn"), unsafe_allow_html=True)
    attempts = rewrite_count + 1
    with st.container(border=True):
        st.markdown(
            f"{NOT_FOUND_DETAIL} Searched {attempts} time{'s' if attempts != 1 else ''}."
        )
        st.markdown(
            "**Try this:**\n"
            "- Name the document or schedule you expect the answer to be in\n"
            "- Rephrase using the terminology the documents themselves use\n"
            "- Confirm the relevant documents have been uploaded to this deal"
        )
    st.write("")


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
