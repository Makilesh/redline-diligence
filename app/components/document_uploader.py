"""
Document uploader component — file upload with category override and version controls.
"""

import re

import streamlit as st
import requests

from app.components.verification import passage_location, signal_labels
from app.markdown_safety import inert_markdown
from src.utils.admin_client import admin_headers


SUPPORTED_TYPES = ["pdf", "docx", "pptx", "xlsx", "txt"]
CATEGORIES = [
    "auto-detect",
    "financial",
    "legal",
    "board",
    "audit",
    "regulatory",
    "operational",
    "other",
]


_RELATION = {
    "newer": "a newer version of",
    "older": "an older version of",
    "same_document": "the same document as",
}


def _md_escape(text: str) -> str:
    """Backslash-escapes ASCII punctuation so document text renders as plain text."""
    return re.sub(r"([!-/:-@\[-`{-~])", r"\\\1", text)


def _describe_injection_screen(screen) -> str:
    """
    A markdown line for passages the ingest screen flagged (Decision 82); "" if none.

    Says how many passages, what kind of wording and where, and that the file
    was indexed as written. Locations only — the API never sends the text —
    and escaped, since a section heading is the document's own words.
    """
    if not isinstance(screen, dict):
        return ""
    count = screen.get("flagged_passages")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return ""
    where = "; ".join(
        _md_escape(loc)
        for loc in (passage_location(x, with_file=False) for x in screen.get("locations") or [])
        if loc
    )
    labels = signal_labels(screen.get("signals"))
    return (
        f"  - ⚠️ {count} passage(s) read like instructions to an AI model"
        + (f" ({labels})" if labels else "")
        + (f": {where}" if where else "")
        + ". The file was indexed as written; an answer that cites these passages, or takes "
        "a figure only from them, is never shown as verified."
    )


def _describe_ingest(filename: str, result: dict) -> list[str]:
    """
    Markdown lines reporting one ingestion: partial status, the server's notices,
    and documents this one looks like a version of. The version match is the
    server's proposal only — nothing was superseded — and is worded that way.
    """
    partial = result.get("status") == "partial"
    lines = [
        f"{'⚠️' if partial else '✅'} **{filename}**: {result.get('chunks_created', 0)} chunks "
        f"({result.get('document_category', '')})" + (" — partly indexed" if partial else "")
    ]
    lines += [f"  - {n}" for n in result.get("notices") or [] if isinstance(n, str)]
    screen_line = _describe_injection_screen(result.get("injection_screen"))
    if screen_line:
        lines.append(screen_line)
    for c in result.get("version_candidates") or []:
        relation = _RELATION.get(c.get("relation", "")) if isinstance(c, dict) else None
        if relation:
            lines.append(
                f"  - Looks like {relation} `{c.get('source_file') or c.get('doc_id')}` "
                f"(doc_id `{c.get('doc_id', '')}`). Nothing was replaced; to retire the old "
                "version, upload again with *Supersedes Document ID* set."
            )
    return lines


def _error_detail(resp) -> str:
    """The API's `detail` when it sent one, else the status code."""
    try:
        detail = resp.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, str) and detail:
        return detail
    return f"upload failed ({resp.status_code})"


DATA_HANDLING_UNKNOWN = (
    "Could not confirm whether this deployment sends document text to an outside AI "
    "service. Assume it does: do not upload confidential, client or personal documents."
)


def data_handling_notice(api_url: str) -> str | None:
    """
    The notice to show before an upload (Decision 116): the server's wording when a
    hosted model is configured, a cautious one when the server could not be asked,
    None when text stays on the server.
    """
    try:
        resp = requests.get(f"{api_url}/data-handling", timeout=5)
        body = resp.json() if resp.status_code == 200 else None
    except Exception:  # noqa: BLE001 - any failure means "could not be asked"
        body = None
    if not isinstance(body, dict):
        return DATA_HANDLING_UNKNOWN
    notice = body.get("upload_notice")
    return notice.strip() if isinstance(notice, str) and notice.strip() else None


def render_document_uploader(api_url: str, deal_id: str) -> None:
    """
    Renders document upload widget with category override,
    version controls, and progress tracking.

    Args:
        api_url: Base API URL.
        deal_id: Current deal identifier.
    """
    st.subheader("📄 Document Upload")

    notice = data_handling_notice(api_url)
    if notice:
        st.warning(inert_markdown(notice), icon="🌐")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, PPTX, XLSX",
    )

    if not uploaded_files:
        return

    # Upload options
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Document Category", CATEGORIES)
    with col2:
        is_current = st.checkbox("Current Version", value=True)

    supersedes_id = st.text_input(
        "Supersedes Document ID (optional)",
        placeholder="Enter doc_id of the version this replaces",
    )

    if st.button("📤 Upload All", type="primary"):
        progress = st.progress(0.0)
        results = []

        for i, file in enumerate(uploaded_files):
            with st.spinner(f"Ingesting {inert_markdown(file.name)}..."):
                try:
                    data = {"deal_id": deal_id, "is_current_version": str(is_current)}
                    if category != "auto-detect":
                        data["document_category"] = category
                    if supersedes_id:
                        data["supersedes_doc_id"] = supersedes_id

                    resp = requests.post(
                        f"{api_url}/ingest",
                        files={"file": (file.name, file.getvalue())},
                        data=data,
                        headers=admin_headers(),
                        timeout=120,
                    )

                    if resp.status_code == 200:
                        results.extend(_describe_ingest(file.name, resp.json()))
                    else:
                        results.append(f"❌ **{file.name}**: {_error_detail(resp)}")

                except Exception as e:
                    results.append(f"❌ **{file.name}**: {str(e)}")

            progress.progress((i + 1) / len(uploaded_files))

        render_upload_results(results)


def render_upload_results(lines: list[str]) -> None:
    """
    Shows the upload report, one markdown line per item.

    A line carries the file name, the server's notices (a sheet name is the
    workbook's own words) and candidate file names, so each is made inert:
    no image from them can load (Decision 96).

    Args:
        lines: _describe_ingest's lines and the error lines, in order.
    """
    st.markdown("### Upload Results")
    for line in lines:
        st.markdown(inert_markdown(line))
