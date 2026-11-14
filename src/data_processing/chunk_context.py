"""
Contextual chunk headers — what a chunk says about itself to the retriever.

A child chunk is embedded on its own. "The cap shall not exceed $174 million"
does not say which agreement it belongs to or that it sits under
Indemnification, so a question phrased in those terms has nothing to match in
the dense or the BM25 vector. The header puts the document and its section path
in front of the text that is embedded:

    Document: merger agreement v2 final — AGREEMENT AND PLAN OF MERGER
    Section: ARTICLE VIII — INDEMNIFICATION > Section 8.2 — Indemnification Cap

It is deterministic (no LLM call per chunk, unlike Anthropic's Contextual
Retrieval, which this approximates) and it is NOT part of the chunk text: the
payload's `text` stays the exact source text, so the synthesizer, the
numeric-grounding validator and the eval's fact matching never see a word the
document did not contain. The header is stored separately as `context_header`
and joined to the text only to build the two index vectors.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Longest header component kept. A heading the parser mislabelled (a sentence
# in capitals) must not crowd out the chunk it describes.
MAX_COMPONENT_CHARS = 160

_FILENAME_NOISE = re.compile(r"[_\-.]+")
_WHITESPACE = re.compile(r"\s+")


def context_headers_enabled() -> bool:
    """
    CHUNK_CONTEXT_HEADERS=1 indexes header + text. Off by default: indexing
    bare chunk text is the measured behaviour, and the header changes every
    vector in the index, so it is switched on only with a retrieval-eval
    number behind it (and needs a re-index when it is).
    """
    return os.getenv("CHUNK_CONTEXT_HEADERS", "0").strip().lower() in {"1", "true", "on", "yes"}


def _clean(value: str | None) -> str:
    """Collapses whitespace and truncates one header component."""
    text = _WHITESPACE.sub(" ", str(value or "")).strip()
    if len(text) > MAX_COMPONENT_CHARS:
        text = text[:MAX_COMPONENT_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def humanize_filename(filename: str) -> str:
    """
    "merger_agreement_v2_final.txt" -> "merger agreement v2 final".

    File names in a data room are written by people and usually name the
    document better than its first line does (a deck's first heading is
    "SLIDE 1").

    Args:
        filename: Display filename, with or without an extension.

    Returns:
        Stem with separators turned into spaces.
    """
    return _FILENAME_NOISE.sub(" ", Path(filename).stem).strip()


def document_title(sections: list[dict]) -> str:
    """
    The document's own title: the outermost heading of its first section.

    Only the first section is consulted. A title, when a document has one, is
    at the top; a heading found further down is a section name, and presenting
    it as the title of every chunk would mislabel the whole document.

    Args:
        sections: Sections in document order, as the processors emit them.

    Returns:
        Title text, or "" when the document does not open with a heading.
    """
    if not sections:
        return ""
    first = sections[0]
    path = first.get("heading_path") or []
    return _clean(path[0] if path else first.get("section_heading"))


def build_context_header(
    filename: str,
    title: str = "",
    heading_path: list[str] | None = None,
    locator: str = "",
) -> str:
    """
    Builds the header for one chunk.

    Args:
        filename: Source filename.
        title: Document title from document_title() (may be empty).
        heading_path: Section headings, outermost first.
        locator: Format-specific position ("Sheet: P&L", "Slide 4").

    Returns:
        One or two lines: "Document: ..." and, when known, "Section: ...".
    """
    name = humanize_filename(filename)
    title = _clean(title)
    document = name
    if title and title.lower() != name.lower():
        document = f"{name} — {title}" if name else title

    path = [_clean(h) for h in (heading_path or []) if _clean(h)]
    # The title is already on the Document line; repeating it as the first
    # path element adds tokens and no information.
    if path and title and path[0].lower() == title.lower():
        path = path[1:]

    lines = [f"Document: {document}"] if document else []
    section = " > ".join(path)
    if locator and section:
        lines.append(f"Section: {_clean(locator)} > {section}")
    elif locator:
        lines.append(f"Section: {_clean(locator)}")
    elif section:
        lines.append(f"Section: {section}")
    return "\n".join(lines)


def embedding_text(payload: dict) -> str:
    """
    The text a chunk is indexed under: its context header, then its text.

    Args:
        payload: Child chunk payload (reads `context_header` and `text`).

    Returns:
        Header and text joined by a blank line, or the bare text when the
        chunk has no header.
    """
    header = payload.get("context_header") or ""
    text = payload.get("text") or ""
    return f"{header}\n\n{text}" if header else text
