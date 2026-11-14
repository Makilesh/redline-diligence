"""
What a query result means, and the per-claim verification record.

The same rules as the web client (web/lib/verification.ts), so the two
consoles cannot disagree about an answer: the view is chosen from `outcome`
(not `is_refusal`, which is also true when no model answered), and "verified"
is only claimed when the server passed the answer AND every claim it returned
is supported AND there was at least one. Anything unrecognised reads as the
less favourable verdict.

Every string that reaches `unsafe_allow_html` here is escaped: claim text,
evidence and file names come from the model and from counterparty documents.
"""

from __future__ import annotations

import html

import streamlit as st

from app.styles import escape_currency, pill

CLAIM_STATUSES = ("supported", "contradicted", "unsupported", "unverified")
OUTCOMES = ("answered", "answered_unverified", "refused", "synthesis_failed")


def result_outcome(result: dict) -> str:
    """
    The run's outcome, derived the server's way when the field is absent.

    Args:
        result: A QueryResponse dict.

    Returns:
        One of OUTCOMES.
    """
    outcome = result.get("outcome")
    if outcome in OUTCOMES:
        return outcome
    trace = [t for t in result.get("agent_trace") or [] if isinstance(t, dict)]
    if any(
        t.get("agent") == "synthesis_unavailable"
        or (t.get("agent") == "answer_synthesizer" and t.get("synthesis_failed"))
        for t in trace
    ):
        return "synthesis_failed"
    if result.get("is_refusal") or not (result.get("answer") or "").strip():
        return "refused"
    if result.get("validation_status") == "failed":
        return "answered_unverified"
    return "answered"


def claim_counts(checks: list[dict]) -> dict[str, int]:
    """Counts claim checks by status; an unknown status counts as unverified."""
    counts = dict.fromkeys(CLAIM_STATUSES, 0)
    for c in checks:
        status = c.get("status")
        counts[status if status in counts else "unverified"] += 1
    return counts


def answer_verdict(result: dict) -> tuple[str, str, str]:
    """
    The verdict for an answer, never more favourable than the evidence.

    Returns:
        (key, label, detail) — key is one of verified | partial | failed |
        no_claims | unknown.
    """
    checks = result.get("claim_checks") or []
    counts = claim_counts(checks)
    total = len(checks)
    problems = counts["contradicted"] + counts["unsupported"]
    not_confirmed = problems + counts["unverified"]
    status = result.get("validation_status")

    if result_outcome(result) == "answered_unverified" or status == "failed":
        return (
            "failed",
            "FAILED VERIFICATION",
            f"{problems} statement(s) could not be matched to the documents, even after a "
            "revision. Treat the marked statements as wrong until checked."
            if problems
            else "The answer did not pass verification, even after a revision.",
        )
    if status == "warning" or (status == "passed" and not_confirmed):
        return (
            "partial",
            "PARTIALLY VERIFIED",
            f"{counts['supported']} of {total} statement(s) matched to the documents; "
            f"{not_confirmed} could not be confirmed."
            if total
            else "Some statements could not be matched to a source.",
        )
    if status == "passed":
        if not total:
            return (
                "no_claims",
                "NOTHING TO VERIFY",
                "The answer makes no statement that could be checked against the documents.",
            )
        return ("verified", "VERIFIED", f"All {total} statement(s) matched to the retrieved documents.")
    return (
        "unknown",
        "NOT VERIFIED",
        "The engine did not report a recognised verification result for this answer.",
    )


_METHOD = {"numeric": "figure match", "nli": "entailment model", "llm": "LLM judge"}


def claim_label(check: dict) -> tuple[str, str]:
    """(label, pill variant) for one claim's verdict."""
    status = check.get("status")
    if status == "supported":
        return "SUPPORTED", "ok"
    if status == "contradicted":
        return "CONTRADICTED", "bad"
    if status == "unsupported":
        return "UNSUPPORTED", "bad" if check.get("method") == "numeric" else "warn"
    if (check.get("reason") or "").lower().startswith("not checked"):
        return "NOT CHECKED", "muted"
    if check.get("absence"):
        return "UNCONFIRMED ABSENCE", "warn"
    return "UNVERIFIED", "warn"


def _figure_note(fig: dict) -> str:
    if fig.get("status") == "unsupported":
        return "not in documents"
    if fig.get("status") == "derived_unverified":
        return "calculation not reproduced"
    if fig.get("binding") == "conflict":
        return "different row or period"
    if fig.get("binding") == "unbound":
        return "found elsewhere"
    if fig.get("status") == "derived":
        return "computed"
    return "found"


def _rank(check: dict) -> int:
    label, _ = claim_label(check)
    return {"CONTRADICTED": 0, "UNSUPPORTED": 1, "NOT CHECKED": 3, "SUPPORTED": 4}.get(label, 2)


def render_claim_checks(checks: list[dict]) -> None:
    """
    Lists every checked statement with its verdict, method, figures and evidence.

    Problems first; supported statements fold into an expander so the one weak
    claim in a long answer is not buried.

    Args:
        checks: QueryResponse.claim_checks.
    """
    if not checks:
        return
    counts = claim_counts(checks)
    summary = " · ".join(f"{n} {k}" for k, n in counts.items() if n)
    st.markdown(
        f"<span class='dd-cite-meta'>{len(checks)} statement(s) checked — {html.escape(summary)}</span>",
        unsafe_allow_html=True,
    )

    ordered = sorted(checks, key=_rank)
    attention = [c for c in ordered if c.get("status") != "supported"]
    supported = [c for c in ordered if c.get("status") == "supported"]

    for c in attention:
        _render_claim(c)
    if supported:
        with st.expander(f"{len(supported)} supported statement(s) and their evidence"):
            for c in supported:
                _render_claim(c)


def _render_claim(check: dict) -> None:
    label, variant = claim_label(check)
    method = _METHOD.get(check.get("method") or "", "")
    header = pill(label, variant)
    if method:
        header += f"<span class='dd-cite-meta'>via {method}</span>"
    st.markdown(header, unsafe_allow_html=True)
    st.markdown(escape_currency(html.escape(check.get("claim") or "")))

    reason = check.get("reason") or ""
    if reason and check.get("status") != "supported":
        st.caption(escape_currency(html.escape(reason)))

    figures = [f for f in check.get("numbers") or [] if isinstance(f, dict) and f.get("raw")]
    if figures:
        st.caption(
            " · ".join(
                escape_currency(f"{html.escape(str(f['raw']))} ({_figure_note(f)})") for f in figures
            )
        )

    source = check.get("evidence_source") or ""
    page = check.get("evidence_page")
    section = check.get("evidence_section") or ""
    location = " · ".join(
        p for p in [f"{source}, p. {page}" if source and page else source, section] if p
    )
    if check.get("evidence"):
        passage = " ".join(str(check["evidence"]).split())
        st.markdown("> " + escape_currency(html.escape(passage)))
    if location:
        st.caption(f"Source: {html.escape(location)}")
