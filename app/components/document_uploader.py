"""
Document uploader component — file upload with category override and version controls.
"""

import streamlit as st
import requests

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


def render_document_uploader(api_url: str, deal_id: str) -> None:
    """
    Renders document upload widget with category override,
    version controls, and progress tracking.

    Args:
        api_url: Base API URL.
        deal_id: Current deal identifier.
    """
    st.subheader("📄 Document Upload")

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
            with st.spinner(f"Ingesting {file.name}..."):
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

        # Show results
        st.markdown("### Upload Results")
        for r in results:
            st.markdown(r)
