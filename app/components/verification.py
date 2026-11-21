"""
What a query result means, and the per-claim verification record.

The same rules as the web client (web/lib/verification.ts), so the two
consoles cannot disagree about an answer: the view is chosen from `outcome`
(not `is_refusal`, which is also true when no model answered), and "verified"
is only claimed when the server passed the answer AND every claim it returned
is supported AND there was at least one AND the answer was not cut off.
Anything unrecognised reads as the less favourable verdict.

Every string that reaches `unsafe_allow_html` here is escaped: claim text,
evidence and file names come from the model and from counterparty documents.
And every one of them is made inert before it is shown (app/markdown_safety.py):
markdown renders images, which load with no click (Decision 96).
"""

from __future__ import annotations

import html

import streamlit as st

from app.markdown_safety import inert_markdown
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
        # Refused only because the assessor reached no model: an outage too.
        or (t.get("agent") == "insufficient_context" and t.get("reason") == "assessor_unavailable")
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


# Ends a verdict sentence for an answer cut off at the model's output limit.
_CUT_OFF = "the answer was cut off before it finished."

# Beside a claim whose figures all checked out but whose wording the local
# checker doubted (Decision 64). The verdict is still the judge's.
DOUBTED_WORDING = "wording doubted (its figures match; the passages do not clearly say this)"

# Beside a claim with a figure that no document it cites states (Decision 78).
# The figure is in the documents; the citation points elsewhere.
MISCITED = "cited source lacks figure"


def is_miscited(check: dict) -> bool:
    """True when a figure of the claim is stated only outside the documents it cites."""
    return any(isinstance(m, dict) and m.get("figure") for m in check.get("citation_mismatch") or [])


# Beside a claim with a figure that only instruction-like document text states
# (Decision 76: the figure is "in the documents", so it can still read as supported).
INJECTION_SUSPECT = "figure only in instruction-like text"

# The ingest screen's signals (src/data_processing/injection_screen.py), in words.
SIGNAL_LABELS = {
    "instruction_override": "an instruction to ignore earlier instructions",
    "model_address": "text addressed to an AI model",
    "role_marker": "chat-role labels such as SYSTEM:",
    "forged_marker": "imitation citation or calculation markers",
    "supersession_claim": "a claim that other documents are out of date",
    "invisible_text": "invisible characters",
}


def signal_labels(signals) -> str:
    """The known signals in words, comma-separated; unknown names are left out."""
    return ", ".join(SIGNAL_LABELS[s] for s in SIGNAL_LABELS if s in (signals or []))


def passage_location(loc: dict, with_file: bool = True) -> str:
    """'memo.pdf, p. 3 · Fees' — where a flagged passage is; "" when nothing is known."""
    if not isinstance(loc, dict):
        return ""
    source = str(loc.get("source_file") or "") if with_file else ""
    page = loc.get("page_number")
    page_text = f"p. {page}" if isinstance(page, int) and not isinstance(page, bool) else ""
    head = ", ".join(p for p in (source, page_text) if p)
    return " · ".join(p for p in (head, str(loc.get("section_heading") or "")) if p)


def _review(result: dict) -> dict:
    review = result.get("injection_review")
    return review if isinstance(review, dict) else {}


def _tainted(review: dict) -> list[str]:
    return [f for f in review.get("tainted_figures") or [] if isinstance(f, str) and f]


def _cited_flagged(review: dict) -> int:
    n = review.get("cited_flagged_passages")
    return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else 0


def injection_note(review: dict | None) -> tuple[str, str] | None:
    """
    One line on instruction-like text behind an answer (Decision 82), or None.

    The flags say the same thing but fold away behind the claim list, and a
    flagged passage the answer does not cite is otherwise only in the trace.
    Locations only — the API never sends the flagged text.

    Returns:
        (text, variant) — "warn" when the answer cites such a passage or takes a
        figure only from one, "muted" when it was retrieved and not used.
    """
    review = review if isinstance(review, dict) else {}
    passages = [p for p in review.get("flagged_passages") or [] if isinstance(p, dict)]
    if not passages:
        return None
    where = "; ".join(loc for loc in map(passage_location, passages) if loc)
    where = where or "a retrieved document"
    labels = signal_labels({s for p in passages for s in p.get("signals") or []})
    labelled = f" ({labels})" if labels else ""
    tainted = _tainted(review)
    if tainted or _cited_flagged(review):
        extra = (
            f" — {', '.join(tainted)} {'is' if len(tainted) == 1 else 'are'} stated nowhere else"
            if tainted else ""
        )
        return (
            f"This answer relies on text that reads like instructions to an AI model{labelled} "
            f"in {where}{extra}. Check those statements against the document itself.",
            "warn",
        )
    verb = "contains" if len(passages) == 1 else "contain"
    return (
        f"Retrieved but not cited: {where} {verb} text that reads like instructions to an AI "
        f"model{labelled}.",
        "muted",
    )


def citation_note(check: dict) -> str:
    """Where the claim's miscited figures are stated, in a sentence; "" when none."""
    items = [m for m in check.get("citation_mismatch") or [] if isinstance(m, dict) and m.get("figure")]
    if not items:
        return ""
    cited = [d for d in check.get("cited_documents") or [] if isinstance(d, str) and d]
    parts = []
    for m in items:
        found = ", ".join(f for f in m.get("found_in") or [] if isinstance(f, str) and f)
        parts.append(f"{m['figure']} is stated in {found or 'another document'}")
    where = "; ".join(parts)
    return where + (
        f", not in the cited {', '.join(cited)}." if cited
        else ", not in any retrieved document its citation names."
    )


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
    miscited = sum(1 for c in checks if is_miscited(c))
    status = result.get("validation_status")
    cut = result.get("answer_truncated") is True
    tainted = _tainted(_review(result))
    cites_flagged = _cited_flagged(_review(result)) > 0

    if result_outcome(result) == "answered_unverified" or status == "failed":
        return (
            "failed",
            "FAILED VERIFICATION",
            (
                f"{problems} statement(s) could not be matched to the documents, even after a "
                "revision. Treat the marked statements as wrong until checked."
                if problems
                else "The answer did not pass verification, even after a revision."
            ) + (f" Also, {_CUT_OFF}" if cut else ""),
        )
    injected = bool(tainted) or cites_flagged
    if status == "warning" or (
        status == "passed" and (not_confirmed or cut or miscited or injected)
    ):
        # A cut answer is downgraded to "warning" with every claim possibly
        # supported; "0 could not be confirmed" would hide the real reason.
        if cut and not not_confirmed:
            detail = (
                f"All {total} statement(s) matched to the documents, but {_CUT_OFF}"
                if total
                else f"Nothing in it could be checked, and {_CUT_OFF}"
            )
        elif total and (miscited or injected) and not not_confirmed:
            # Downgraded for a citation that does not hold its figure
            # (Decision 78), or for leaning on instruction-like text
            # (Decision 76) — every statement checked out, so say which.
            reasons = []
            if miscited:
                reasons.append(f"{miscited} cite(s) a document that does not state the figure")
            if tainted:
                reasons.append(
                    f"{', '.join(tainted)} {'rests' if len(tainted) == 1 else 'rest'} only on "
                    "text that reads like instructions to an AI model"
                )
            elif cites_flagged:
                reasons.append("it cites a passage that reads like instructions to an AI model")
            detail = (
                f"All {total} statement(s) matched to the documents, but "
                + ", and ".join(reasons) + "."
            )
        else:
            detail = (
                f"{counts['supported']} of {total} statement(s) matched to the documents; "
                f"{not_confirmed} could not be confirmed."
                if total
                else "Some statements could not be matched to a source."
            ) + (f" Also, {_CUT_OFF}" if cut else "")
        return ("partial", "PARTIALLY VERIFIED", detail)
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


def needs_review(check: dict) -> bool:
    """Whether a claim is listed before the fold rather than with the supported ones."""
    return (
        check.get("status") != "supported"
        or is_miscited(check)
        or check.get("injection_suspect") is True
    )


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
    # A supported claim whose citation does not hold its figure is not folded
    # away: the reader would open the wrong source (Decision 78). Nor is one
    # whose figure only instruction-like text states (Decision 76).
    attention = [c for c in ordered if needs_review(c)]
    supported = [c for c in ordered if not needs_review(c)]

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
    if check.get("suspected_misstatement") is True:
        header += f"<span class='dd-cite-meta'>· {DOUBTED_WORDING}</span>"
    if is_miscited(check):
        header += f"<span class='dd-cite-meta'>· {MISCITED}</span>"
    if check.get("injection_suspect") is True:
        header += f"<span class='dd-cite-meta'>· {INJECTION_SUSPECT}</span>"
    st.markdown(header, unsafe_allow_html=True)
    st.markdown(inert_markdown(escape_currency(html.escape(check.get("claim") or ""))))
    note = citation_note(check)
    if note:
        st.caption(inert_markdown(escape_currency(html.escape(note))))

    reason = check.get("reason") or ""
    if reason and check.get("status") != "supported":
        st.caption(inert_markdown(escape_currency(html.escape(reason))))

    figures = [f for f in check.get("numbers") or [] if isinstance(f, dict) and f.get("raw")]
    if figures:
        st.caption(
            " · ".join(
                inert_markdown(escape_currency(f"{html.escape(str(f['raw']))} ({_figure_note(f)})"))
                for f in figures
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
        st.markdown("> " + inert_markdown(escape_currency(html.escape(passage))))
    if location:
        st.caption(f"Source: {inert_markdown(html.escape(location))}")
