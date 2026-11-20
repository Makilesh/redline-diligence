"""
Checklist review grid — GET /deals/{id}/checklist (Decision 95): the deal's
contracts against a diligence checklist, each cell the passage of that contract
that best answers the item and a status (found / unclear / not found).

File names, sections and passages are document text from the data room. They
are shown in tables (rendered as text) or escaped before any markdown, and
never through unsafe_allow_html.
"""

import re

import streamlit as st

# Markdown that document text could carry; "$" starts LaTeX in Streamlit's
# markdown and would swallow figures (see app.styles.escape_currency).
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|$<>~])")

_REASONS = {
    "document_cap": "the deal has more contracts than one grid checks",
    "time_budget": "the grid ran out of time",
}
_MARKS = {"found": "● found", "unclear": "? unclear", "not_found": "· not found"}
_VERDICTS = {
    "found": "Found",
    "unclear": "Unclear — read the passage",
    "not_found": "Not found — the closest passage is shown",
}


def _plain(text) -> str:
    """Document text made literal for st.markdown, st.warning and st.caption, on one line."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", " ".join(str(text or "").split()))


def _status(cell) -> str:
    """A cell's status; anything unrecognised is unclear, never found or absent."""
    status = cell.get("status") if isinstance(cell, dict) else None
    return status if status in _MARKS else "unclear"


def _where(passage: dict) -> str:
    parts = []
    page = passage.get("page_number")
    if isinstance(page, int) and page > 0:
        parts.append(f"p. {page}")
    if passage.get("section_heading"):
        parts.append(str(passage["section_heading"]))
    return " · ".join(parts)


def _cells_by_item(row: dict) -> dict:
    return {c.get("item_id"): c for c in row.get("cells") or [] if isinstance(c, dict)}


def grid_rows(items: list[dict], rows: list[dict]) -> list[dict]:
    """One table row per contract: its name, then each item's status."""
    table = []
    for row in rows:
        cells = _cells_by_item(row)
        line = {"Contract": str(row.get("source_file") or "")}
        for item in items:
            line[str(item.get("label") or item.get("id") or "")] = _MARKS[_status(cells.get(item.get("id")))]
        table.append(line)
    return table


def render_checklist(report: dict | None) -> None:
    """
    Renders the deal's checklist grid.

    Optional for the deal view (Decision 88): when the grid could not be
    fetched the panel says so and nothing else on the page changes.

    Args:
        report: Parsed GET /deals/{id}/checklist response, or None.
    """
    if not isinstance(report, dict):
        st.info("The checklist grid could not be loaded. The rest of the deal view is unaffected.")
        return

    items = [i for i in report.get("items") or [] if isinstance(i, dict) and i.get("id")]
    rows = [r for r in report.get("rows") or [] if isinstance(r, dict)]
    statuses = [_status(c) for r in rows for c in _cells_by_item(r).values()]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Contracts checked", len(rows))
    with col2:
        st.metric("Found", statuses.count("found"))
    with col3:
        st.metric("Unclear", statuses.count("unclear"))
    with col4:
        st.metric("Not found", statuses.count("not_found"))

    # A grid that does not say it checked every contract is not taken to have.
    if report.get("truncated") is not False:
        reason = _REASONS.get(str(report.get("truncation_reason") or ""), "the grid was cut short")
        unchecked = [str(d) for d in report.get("documents_not_checked") or []]
        message = f"Partial: checked {len(rows)} of {len(rows) + len(unchecked)} contracts — {reason}."
        if unchecked:
            message += " Not checked: " + ", ".join(unchecked) + "."
        st.warning(_plain(message))

    if not rows:
        if int(report.get("documents_total") or 0):
            st.info("No document of this deal reads as a contract.")
        else:
            st.info("Nothing is indexed for this deal yet.")
    else:
        st.dataframe(grid_rows(items, rows), hide_index=True, width="stretch")

        names = [str(r.get("source_file") or "") for r in rows]
        labels = [str(i.get("label") or i.get("id")) for i in items]
        contract = st.selectbox("Contract", range(len(rows)), format_func=lambda n: names[n],
                                key="checklist_contract")
        item = st.selectbox("Checklist item", range(len(items)), format_func=lambda n: labels[n],
                            key="checklist_item")
        if contract is not None and item is not None and items:
            _render_cell(items[item], _cells_by_item(rows[contract]).get(items[item]["id"]))

    excluded = [d for d in report.get("documents_excluded") or [] if isinstance(d, dict)]
    total = int(report.get("excluded_total") or len(excluded))
    if total:
        listed = "; ".join(
            f"{d.get('source_file') or ''} ({d.get('document_category') or 'other'})" for d in excluded
        )
        noun = "document that is not a contract" if total == 1 else "documents that are not contracts"
        st.caption(_plain(f"Not in the grid: {total} {noun}" + (f": {listed}" if listed else "") + "."))


def _render_cell(item: dict, cell) -> None:
    """One cell: the item's question, the verdict, the passage verbatim and where it is."""
    st.caption(_plain(str(item.get("question") or "")))
    status = _status(cell)
    st.markdown(f"**{_VERDICTS[status]}**")
    passage = cell.get("passage") if isinstance(cell, dict) else None
    if not isinstance(passage, dict) or not passage.get("text"):
        st.caption("No passage of this contract was retrieved.")
        return
    st.markdown("> " + _plain(passage.get("text")))
    location = _where(passage)
    if location:
        st.caption(_plain(location))
    alternates = [a for a in cell.get("alternates") or [] if isinstance(a, dict) and a.get("text")]
    if alternates:
        with st.expander(f"Other passages ({len(alternates)})"):
            for alternate in alternates:
                st.markdown("> " + _plain(alternate.get("text")))
                where = _where(alternate)
                if where:
                    st.caption(_plain(where))
