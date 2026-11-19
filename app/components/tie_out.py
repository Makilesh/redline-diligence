"""
Figure tie-out panel — the deal-level report of GET /deals/{id}/tie-out
(Decision 92): every labelled figure two or more documents state, conflicts
first.

Labels, figures, file names and sections are document text from the data room.
They are shown in tables (rendered as text) or escaped before any markdown, and
never through unsafe_allow_html.
"""

import re

import streamlit as st

# Markdown that a file or section name could carry; "$" starts LaTeX in
# Streamlit's markdown and would swallow figures (see app.styles.escape_currency).
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|$<>~])")

_REASONS = {
    "chunk_cap": "the deal is larger than one report reads",
    "time_budget": "the report ran out of time",
}


def _plain(text) -> str:
    """Document text made literal for st.markdown, st.warning and st.caption."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", str(text or ""))


def _where(value: dict) -> str:
    page = value.get("page_number")
    if isinstance(page, int) and page > 0:
        return f"p. {page}"
    return str(value.get("section_heading") or "")


def _rows(figures: list, with_gap: bool) -> list[dict]:
    """One table row per document statement of each figure."""
    rows = []
    for figure in figures:
        if not isinstance(figure, dict):
            continue
        for value in figure.get("values") or []:
            if not isinstance(value, dict):
                continue
            row = {
                "Figure": str(figure.get("metric") or ""),
                "Period": str(figure.get("period") or ""),
            }
            if with_gap:
                row["Gap"] = f"{float(figure.get('max_deviation_pct') or 0):.1f}%"
            row.update({
                "Document": str(value.get("source_file") or ""),
                "Where": _where(value),
                "As printed": str(value.get("as_printed") or ""),
            })
            rows.append(row)
    return rows


def render_tie_out(report: dict | None) -> None:
    """
    Renders the deal's tie-out report.

    Optional for the deal view (Decision 88): when the report could not be
    fetched the panel says so and nothing else on the page changes.

    Args:
        report: Parsed GET /deals/{id}/tie-out response, or None.
    """
    if not isinstance(report, dict):
        st.info("The figure tie-out report could not be loaded. The rest of the deal view is unaffected.")
        return

    conflicts = [f for f in report.get("conflicts") or [] if isinstance(f, dict)]
    agreements = [
        f for f in report.get("agreements") or [] if isinstance(f, dict) and f.get("status") == "agree"
    ]
    read, total = int(report.get("documents_read") or 0), int(report.get("documents_total") or 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Conflicts", int(report.get("conflicts_total") or len(conflicts)))
    with col2:
        st.metric("Agreements", int(report.get("agreements_total") or len(agreements)))
    with col3:
        st.metric("Documents read", f"{read}/{total}")

    # A report that does not say it read everything is not taken to have.
    if report.get("truncated") is not False:
        reason = _REASONS.get(str(report.get("truncation_reason") or ""), "the read was cut short")
        unread = [str(d) for d in report.get("documents_not_read") or []]
        message = f"Partial: read {read} of {total} documents — {reason}."
        if unread:
            message += " Not read: " + ", ".join(unread) + "."
        st.warning(_plain(message))

    if conflicts:
        st.markdown("**Conflicting figures** — largest gap first")
        st.dataframe(_rows(conflicts, with_gap=True), hide_index=True, width="stretch")
        shown, conflicts_total = len(conflicts), int(report.get("conflicts_total") or 0)
        if conflicts_total > shown:
            st.caption(f"Showing the {shown} largest of {conflicts_total} conflicts.")
    elif agreements:
        st.success("Every figure two documents both state agrees.")
    else:
        st.info("No figure is stated by two documents of this deal.")

    if agreements:
        with st.expander(f"Agreeing figures ({int(report.get('agreements_total') or len(agreements))})"):
            st.dataframe(_rows(agreements, with_gap=False), hide_index=True, width="stretch")

    not_compared = [
        (int(report.get("single_source_figures") or 0), "only one document states"),
        (int(report.get("generic_figures") or 0), "under labels like Total or Other"),
        (int(report.get("ambiguous_figures") or 0), "a document states twice with different values"),
        (int(report.get("figures_without_period") or 0), "with no readable period"),
    ]
    parts = [f"{count:,} {'figure' if count == 1 else 'figures'} {why}" for count, why in not_compared if count]
    if parts:
        st.caption("Not compared: " + "; ".join(parts) + ".")
