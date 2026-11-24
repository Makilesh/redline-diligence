"""
Ingestion pipeline — file → sections → chunks → embeddings → Qdrant.

This is the single code path behind POST /ingest and the one an evaluation
harness calls directly, so what gets measured is what gets served:

    extract_and_chunk()   parse, chunk, PII/risk/category (Laya when enabled; none
                          with LAYA_ENABLED=0) — testable alone
    index_document()      extract_and_chunk + embed + BM25 + upsert (async)

Identity is content-derived. doc_id = uuid5(deal_id, sha256(file bytes)) and
every point id = uuid5(chunk_id), so re-uploading the same file into the same
deal overwrites its own points instead of duplicating them (the old ids were
uuid4 doc_ids and `hash(chunk_id)`, which Python salts per process).

Payload fields written per child chunk (see _child_payload):
    chunk_id, deal_id, doc_id, text, source_file, file_type, document_category,
    section_heading, page_number (None when the format has no pages: .txt,
    .docx, .xlsx), clause_id, is_table, content_type, table_id (tables only),
    table_representation (converter tables only), parent_chunk_id (prose only),
    chunk_index, token_count, is_current_version, contains_pii, risk_signals,
    risk_decisions, pii_source, pii_confidence, supersedes_doc_id,
    superseded_by, is_redline, context_header, plus format extras (sheet_name,
    currency, scale_factor, scale_label, slide_number, page_range, metrics,
    redline_base_doc_id), plus, for a public filing only, filing_form,
    filing_period_end / filing_date, filing_fiscal_years, filing_fiscal_quarter,
    filing_entity_keys and filing_company (filing_metadata.py; read from the
    cover page), plus, when the opening states them, doc_title and doc_parties
    (document_identity.py; the same on every child of a document), plus, only
    on a chunk whose text reads like instructions to a model,
    suspected_injection=1 and injection_signals (injection_screen.py; parents
    are screened too). index_document adds sparse_encoding (default BM25 only)
    and ingest_version (children and parents) to what it stores.

`text` is always the exact source text. `context_header` ("Document: ... /
Section: ...", see chunk_context.py; written only when CHUNK_CONTEXT_HEADERS
is on) is joined to it only to compute the dense and BM25 vectors, so nothing
downstream can quote or ground a figure on words the document did not contain.

risk_signals stays a list of type strings (what filters and the dashboard
read). risk_decisions records how each was decided — {type: {"source":
"regex" | "laya" | "regex+laya", "confidence": Laya P or None, "accepted":
bool}} — including regex matches Laya overruled (accepted=False), so a
dropped signal is auditable rather than silently gone.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from src.data_processing.chunk_context import (
    build_context_header,
    context_headers_enabled,
    document_title,
    embedding_text,
)
from src.data_processing.document_classifier import DocumentClassifier
from src.data_processing.document_identity import extract_document_identity
from src.data_processing.errors import (  # noqa: F401  (re-exported)
    DocumentExtractionError,
    IngestionError,
    NoExtractableContentError,
    PIIScreeningUnavailable,
    UnsafeDocumentError,
    UnsupportedFileTypeError,
)
from src.data_processing.excel_normalizer import ExcelNormalizer
from src.data_processing.filing_metadata import extract_filing_metadata
from src.data_processing.financial_table_converter import (
    FinancialTableConverter,
    representation_content_type,
    table_from_rows,
)
from src.data_processing.injection_screen import screen_text
from src.data_processing.pii_detector import PIIDetector
from src.data_processing.risk_signal_extractor import RiskSignalExtractor, RiskSignalResult
from src.data_processing.semantic_chunker import (
    DEFAULT_PARENT_TARGET_TOKENS,
    SemanticChunker,
)
from src.data_processing.structural_chunker import StructuralChunker
from src.data_processing.text_processor import decode_text_file, parse_text_sections
from src.decisions import ingest_signals
from src.decisions.laya_client import LayaUnavailable, laya_enabled
from src.vector_db.constants import (
    COLLECTION_NAME,
    PARENT_COLLECTION_NAME,
    QDRANT_BATCH_SIZE,
    STAGED_DEAL_FIELD,
)
from src.utils.fail_open import fail_open, fail_open_async
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# .xls is deliberately absent: openpyxl cannot read the legacy binary format.
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".xlsx", ".txt"})
_OFFICE_ZIP_EXTENSIONS = frozenset({".docx", ".pptx", ".xlsx"})

# Zip-bomb guards for Office files (which are zip archives). The limits are
# checked against the central directory before any parser opens the file;
# zipfile then refuses to inflate an entry beyond its declared size, so a
# forged directory cannot smuggle more data past this check.
MAX_ARCHIVE_ENTRIES = 10_000
MAX_ENTRY_COMPRESSION_RATIO = 200
_RATIO_CHECK_MIN_BYTES = 10 * 1024 * 1024

# Fixed namespace for every ingestion id. Changing it changes every doc_id and
# point id, i.e. forces a full re-index.
INGEST_NAMESPACE = uuid.UUID("6f1c2a52-5d0e-4f8e-9a57-3c0b8e2d9a11")

# Which extraction and chunking code wrote a point, stamped on every child and
# parent as `ingest_version` (Decision 97). Point ids come from positions, not
# content, so nothing else tells a live index's documents written before an
# extraction change from those written after it. Bump it with any change that
# alters the chunks or payloads some file type produces, and set that type's
# entry in CURRENT_SINCE to the new value.
INGEST_VERSION = 1

# file type -> the lowest ingest_version whose chunks for that type are what
# the current code writes. A document of a listed type stamped below it, or not
# stamped at all (indexed before Decision 97; counted as 0), needs re-ingesting
# from its file; a type not listed has not changed since before the stamp.
# PDF: a box around text is not a table (Decision 89), and statements keep
# their period headers, titles and pages (Decision 93).
CURRENT_SINCE = {"pdf": 1}

_CONTENT_SAMPLE_CHARS = 2000

# Non-core metadata copied from a chunk into its payload when present.
_PASSTHROUGH_KEYS = (
    "sheet_name",
    "currency",
    "scale_factor",
    "scale_label",
    "slide_number",
    "page_range",
    "table_representation",
    "metrics",
    "redline_base_doc_id",
    # Row-by-row table chunks only: the machine-readable rows the Financial
    # Verifier's registry reads (shape in financial_table_converter.
    # structured_rows). They were built and then dropped here, so the
    # cross-source check saw no figure from any PDF/DOCX/PPTX/XLSX table.
    "structured_rows",
    "table_row_range",
    "table_row_count",
    # "hidden" / "veryHidden" for a chunk from a hidden worksheet.
    "sheet_state",
)

EmbedFn = Callable[[list[str]], Awaitable[Any]]
SparseFn = Callable[[list[str]], list]


# ==============================================================================
# Result container
# ==============================================================================


@dataclass
class ExtractedDocument:
    """Everything extract_and_chunk() produces, before any embedding."""
    doc_id: str
    deal_id: str
    filename: str
    extension: str
    document_category: str
    content_sha256: str
    chunks: list[dict] = field(default_factory=list)    # child payloads
    parents: list[dict] = field(default_factory=list)   # parent payloads
    risk_signals: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_redline: bool = False
    category_source: str = "rules"  # "override" | "laya" | "rules"
    category_confidence: float | None = None
    # filing_* fields of a public filing (filing_metadata.py); {} for anything else.
    filing: dict = field(default_factory=dict)
    # Child chunks whose text reads like instructions to a model
    # (injection_screen.py): {chunk_id, page_number, section_heading, signals}.
    suspected_injection: list[dict] = field(default_factory=list)


# ==============================================================================
# Identity and input validation
# ==============================================================================


def sanitize_filename(name: str | None) -> str:
    """
    Reduces a client-supplied filename to a safe display basename.

    The result is used only as metadata (source_file, classification); the
    upload is written to disk under a generated name, so this is not the only
    thing standing between a client and the filesystem.

    Args:
        name: Raw filename from the multipart upload.

    Returns:
        Basename with path components, control characters and Windows-reserved
        characters removed.

    Raises:
        UnsupportedFileTypeError: If nothing usable remains, or it is a dotfile.
    """
    if not name:
        raise UnsupportedFileTypeError("A filename is required")

    # Split on both separators: Path() only understands the host's own.
    base = re.split(r"[\\/]", name)[-1]
    base = "".join(ch for ch in base if ch.isprintable())
    base = re.sub(r'[<>:"|?*]', "_", base).strip()

    if not base or base in {".", ".."} or base.startswith("."):
        raise UnsupportedFileTypeError("Invalid filename")

    if len(base) > 200:
        stem, ext = os.path.splitext(base)
        # The extension is kept (it selects the parser) unless it alone fills
        # the budget: "a." followed by 300 characters became a 301-character
        # dotfile, which the checks above exist to refuse (Decision 88).
        if len(ext) >= 200:
            stem, ext = base, ""
        base = (stem[: 200 - len(ext)] + ext).rstrip()
    return base


def compute_content_hash(file_path: str) -> str:
    """
    SHA-256 of a file's bytes, streamed.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_doc_id(deal_id: str, content_sha256: str) -> str:
    """
    Deterministic document id: same bytes in the same deal → same id.

    Args:
        deal_id: Deal the document belongs to.
        content_sha256: SHA-256 hex digest of the file bytes.

    Returns:
        UUID string.
    """
    return str(uuid.uuid5(INGEST_NAMESPACE, f"doc:{deal_id}:{content_sha256}"))


def point_id_for(chunk_id: str) -> str:
    """
    Deterministic Qdrant point id for a chunk_id (stable across processes).

    Args:
        chunk_id: Chunk identifier.

    Returns:
        UUID string accepted by Qdrant as a point id.
    """
    return str(uuid.uuid5(INGEST_NAMESPACE, f"point:{chunk_id}"))


def _max_uncompressed_bytes() -> int:
    """Total uncompressed-size cap for Office archives (env MAX_UNCOMPRESSED_MB)."""
    return int(os.getenv("MAX_UNCOMPRESSED_MB", "200")) * 1024 * 1024


def validate_office_archive(file_path: str, extension: str) -> None:
    """
    Rejects .docx/.pptx/.xlsx files that would decompress to an unsafe size.

    Args:
        file_path: Path to the uploaded file.
        extension: Lower-case extension including the dot.

    Raises:
        DocumentExtractionError: If the file is not a zip archive at all.
        UnsafeDocumentError: If entry count, total size or ratio exceed limits.
    """
    if extension not in _OFFICE_ZIP_EXTENSIONS:
        return

    try:
        archive = zipfile.ZipFile(file_path)
    except zipfile.BadZipFile as e:
        with open(file_path, "rb") as f:
            magic = f.read(8)
        # An OLE compound file: what Office writes for a password-protected
        # document, and what a legacy .doc/.xls/.ppt renamed to the new
        # extension is. Both are common in data rooms and both are fixable by
        # the uploader, so say which.
        if magic == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            raise DocumentExtractionError(
                f"The {extension} file is password-protected or in the legacy binary Office "
                f"format; upload an unprotected {extension} copy"
            ) from e
        raise DocumentExtractionError(
            f"The file is not a valid {extension} document"
        ) from e

    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_ENTRIES:
            raise UnsafeDocumentError("Document archive has too many entries")

        total = sum(info.file_size for info in infos)
        if total > _max_uncompressed_bytes():
            raise UnsafeDocumentError(
                "Document expands beyond the allowed uncompressed size"
            )

        for info in infos:
            if (
                info.file_size > _RATIO_CHECK_MIN_BYTES
                and info.compress_size > 0
                and info.file_size / info.compress_size > MAX_ENTRY_COMPRESSION_RATIO
            ):
                raise UnsafeDocumentError("Document has an abnormal compression ratio")


# ==============================================================================
# Extraction
# ==============================================================================


def _content_sample(file_path: str, extension: str, doc_id: str = "") -> str:
    """
    First ~2000 characters of text, for content-based classification.

    Best effort: classification falls back to filename/extension heuristics,
    so a parser's error returns "". A PDF, DOCX or PPTX is read in the parser
    child (Decision 121).

    Args:
        file_path: Path to the file.
        extension: Lower-case extension including the dot.
        doc_id: Document identifier, for the child's logging.

    Returns:
        Text sample (possibly empty).

    Raises:
        DocumentExtractionError: The parser process died or timed out on the file.
    """
    try:
        if extension == ".txt":
            with open(file_path, "rb") as f:
                data = f.read(_CONTENT_SAMPLE_CHARS * 4)
            # Cut at a line end so a multi-byte character is never split.
            if b"\n" in data[:-1]:
                data = data[: data.rindex(b"\n") + 1]
            try:
                return decode_text_file(data)[0][:_CONTENT_SAMPLE_CHARS]
            except (ValueError, UnicodeDecodeError):
                return data[:_CONTENT_SAMPLE_CHARS].decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Content sample extraction failed", extra={"error": str(e)})
        return ""
    if extension in (".pdf", ".docx", ".pptx"):
        # Read in the parser child (Decision 121): the sample opens the file with
        # the parsers the parse uses. A parser that dies or hangs on the file fails
        # the upload (DocumentExtractionError), as the parse would; its errors give "".
        from src.data_processing.parse_isolation import content_sample

        return content_sample(file_path, extension, doc_id)
    return ""


def _table_sections(
    rows: list[list[Any]] | None,
    fallback_text: str,
    heading: str,
    page_number: int | None,
    table_id: str,
    extra: dict | None = None,
    caption: str = "",
    notes: list[str] | None = None,
) -> list[dict]:
    """
    Turns one table into sections sharing a table_id.

    When the rows parse as a numeric table (header + labelled rows), the
    FinancialTableConverter's representations are emitted — narrative,
    row_by_row, markdown, and metrics_summary when a metric is computable.
    Otherwise the table is indexed once, verbatim, as content_type table_text.

    Args:
        rows: Header row followed by data rows, or None if unavailable.
        fallback_text: Verbatim table text for the non-numeric case.
        heading: Section heading the table sits under.
        page_number: Page (or slide) number, None when the format has none.
        table_id: Identifier shared by every representation.
        extra: Additional section keys (slide_number, page_range, ...).
        caption: Text printed next to the table but outside its grid — a
            statement's "(in thousands, except per share data)" line. Read for
            the unit and kept at the top of every representation.
        notes: Footnotes belonging to the table, kept under the grid.

    Returns:
        List of section dicts.
    """
    extra = dict(extra or {})
    base = {
        "section_heading": heading,
        "page_number": page_number,
        "section_type": "table",
        "is_table": 1,
        "table_id": table_id,
        **extra,
    }
    prefix = "\n".join(part for part in (heading, caption) if part)
    prefix = f"{prefix}\n" if prefix else ""

    table = table_from_rows(rows[0], rows[1:]) if rows and len(rows) >= 2 else None
    if table is None:
        if not fallback_text.strip():
            return []
        text = fallback_text if not caption or caption in fallback_text else f"{caption}\n{fallback_text}"
        return [{**base, "text": text, "content_type": "table_text"}]

    table.notes = list(notes or [])
    meta = ExcelNormalizer().detect_scale([str(c) for c in rows[0]] + [heading, caption])
    representations = FinancialTableConverter().generate_all_representations(
        df=table, meta=meta, table_id=table_id, source_metadata={},
    )

    sections = []
    for rep in representations:
        rep_type = rep["table_representation"]
        if rep_type == "metrics_summary" and not rep.get("metrics"):
            continue  # nothing computable — the summary would be a bare title
        section = {
            **base,
            "text": prefix + rep["text"],
            "table_representation": rep_type,
            "content_type": representation_content_type(rep_type),
            "currency": meta.currency,
            "scale_factor": meta.scale_factor,
            "scale_label": meta.scale_label,
            "table_row_range": rep["table_row_range"],
            "table_row_count": rep["table_row_count"],
        }
        for key in ("metrics", "structured_rows"):
            if rep.get(key):
                section[key] = rep[key]
        sections.append(section)
    return sections


def extract_sections(
    file_path: str,
    extension: str,
    doc_id: str,
    deal_id: str,
    document_category: str,
) -> tuple[list[dict], list[str], bool]:
    """
    Runs the format-specific processor and normalises its output to sections.

    Args:
        file_path: Path to the file on disk.
        extension: Lower-case extension including the dot.
        doc_id: Document id (for processor logging and redline linkage).
        deal_id: Deal id (table_ids are "{deal_id}_{doc_id}_tNNN").
        document_category: Category (legal PDFs use clause segmentation).

    Returns:
        Tuple of (sections, warnings, has_redline).
    """
    sections: list[dict] = []
    warnings: list[str] = []
    has_redline = False
    table_seq = 0

    def next_table_id() -> str:
        nonlocal table_seq
        table_id = f"{deal_id}_{doc_id}_t{table_seq:03d}"
        table_seq += 1
        return table_id

    if extension == ".pdf":
        # In a child process: the parse holds the GIL, and on a thread of the API's
        # process it stalled every other request (parse_isolation, Decision 110).
        from src.data_processing.parse_isolation import parse_pdf
        pdf_sections, pdf_warnings = parse_pdf(
            file_path, doc_id, legal_mode=document_category == "legal")
        for s in pdf_sections:
            if s.is_table:
                sections.extend(_table_sections(
                    rows=s.table_rows,
                    fallback_text=s.text,
                    heading=s.section_heading,
                    page_number=s.page_number,
                    table_id=next_table_id(),
                    extra={"page_range": s.page_range} if s.page_range else None,
                    caption=s.caption,
                    notes=s.notes,
                ))
            else:
                sections.append({
                    "text": s.text,
                    "section_heading": s.section_heading,
                    "page_number": s.page_number,
                    "section_type": s.section_type,
                    "clause_id": s.clause_id,
                })
        warnings.extend(pdf_warnings)

    elif extension == ".docx":
        # The office parsers run in the child too (Decision 114).
        from src.data_processing.parse_isolation import parse_office
        clean_chunks, redline_chunks = parse_office(".docx", file_path, doc_id)
        for chunk in clean_chunks:
            if chunk.metadata.get("table_rows"):
                sections.extend(_table_sections(
                    rows=chunk.metadata["table_rows"],
                    fallback_text=chunk.text,
                    heading=chunk.section_heading,
                    page_number=None,
                    table_id=next_table_id(),
                ))
            else:
                sections.append({
                    "text": chunk.text,
                    "section_heading": chunk.section_heading,
                    "page_number": None,
                    "section_type": "text",
                })
        # Redlines are indexed alongside the clean text (is_redline=1) for the
        # paragraphs that actually carry tracked changes.
        for chunk in redline_chunks:
            has_redline = True
            sections.append({
                "text": chunk.text,
                "section_heading": chunk.section_heading,
                "page_number": None,
                "section_type": "text",
                "content_type": "redline",
                "is_redline": 1,
                "redline_base_doc_id": chunk.redline_base_doc_id,
            })

    elif extension == ".pptx":
        from src.data_processing.parse_isolation import parse_office
        for s in parse_office(".pptx", file_path, doc_id):
            if s.get("is_table"):
                sections.extend(_table_sections(
                    rows=s.get("table_rows"),
                    fallback_text=s["text"],
                    heading=s.get("section_heading", ""),
                    page_number=s.get("page_number"),
                    table_id=next_table_id(),
                    extra={"slide_number": s.get("slide_number")},
                ))
            else:
                sections.append(s)

    elif extension == ".xlsx":
        from src.data_processing.parse_isolation import parse_office
        sections, failed_sheets, sheet_warnings = parse_office(
            ".xlsx", file_path, doc_id, table_id_prefix=f"{deal_id}_{doc_id}")
        warnings.extend(
            f"Sheet '{f['sheet_name']}' could not be processed"
            for f in failed_sheets
        )
        warnings.extend(sheet_warnings)

    elif extension == ".txt":
        with open(file_path, "rb") as f:
            data = f.read()
        try:
            text, encoding = decode_text_file(data)
        except ValueError:
            raise DocumentExtractionError(
                "The .txt file is not text (it looks like a binary file with a .txt extension)"
            ) from None
        if encoding not in ("utf-8", "utf-8-sig"):
            logger.info("Text file decoded", extra={"doc_id": doc_id, "encoding": encoding})
        for s in parse_text_sections(text):
            if s["is_table"]:
                sections.extend(_table_sections(
                    rows=None,  # whitespace-aligned text tables stay verbatim
                    fallback_text=s["text"],
                    heading=s["section_heading"],
                    page_number=None,
                    table_id=next_table_id(),
                    extra={"heading_path": s["heading_path"]},
                ))
            else:
                sections.append(s)

    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {extension}")

    return sections, warnings, has_redline


def _child_payload(
    chunk,
    chunk_id: str,
    index: int,
    doc: ExtractedDocument,
    is_current_version: bool,
    supersedes_doc_id: str | None,
    contains_pii: int,
    risk_signals: list[str],
    parent_chunk_id: str | None,
    risk_decisions: dict[str, dict] | None = None,
    pii_source: str = "regex",
    pii_confidence: float | None = None,
    title: str = "",
) -> dict:
    """
    Builds the Qdrant payload for one child chunk.

    Args:
        chunk: SemanticChunk.
        chunk_id: Stable chunk id.
        index: Position of the chunk in the document.
        doc: Document being built (for ids, filename, category).
        is_current_version: Version flag.
        supersedes_doc_id: Document this one replaces, if any.
        contains_pii: 0/1 PII flag.
        risk_signals: Risk signal types detected in this chunk.
        parent_chunk_id: Parent chunk id, or None for tables/redlines.
        risk_decisions: How each risk signal was decided (see module docstring).
        pii_source: "regex" or "regex+laya".
        pii_confidence: Laya's P(PII) when it was consulted.
        title: Document title for the context header.

    Returns:
        Payload dict.
    """
    meta = chunk.metadata
    payload = {
        "chunk_id": chunk_id,
        "deal_id": doc.deal_id,
        "doc_id": doc.doc_id,
        "text": chunk.text,
        "source_file": doc.filename,
        "file_type": doc.extension.lstrip("."),
        "document_category": doc.document_category,
        "section_heading": chunk.section_heading or "",
        "page_number": chunk.page_number,
        "clause_id": chunk.clause_id,
        "is_table": int(meta.get("is_table", 0)),
        "content_type": meta.get("content_type", "text"),
        "chunk_index": index,
        "token_count": chunk.token_count,
        "is_current_version": 1 if is_current_version else 0,
        "contains_pii": contains_pii,
        "risk_signals": risk_signals,
        "risk_decisions": risk_decisions or {},
        "pii_source": pii_source,
        "pii_confidence": pii_confidence,
        "supersedes_doc_id": supersedes_doc_id or "",
        "superseded_by": "",  # stamped later if a newer version replaces this doc
        "is_redline": int(meta.get("is_redline", 0)),
    }
    if meta.get("table_id"):
        payload["table_id"] = meta["table_id"]
    if parent_chunk_id:
        payload["parent_chunk_id"] = parent_chunk_id
    for key in _PASSTHROUGH_KEYS:
        if meta.get(key) is not None:
            payload[key] = meta[key]
    # Document-level, optional: an index written before these fields existed
    # simply has none, and filing_period.py then leaves its results alone.
    payload.update(doc.filing)
    if context_headers_enabled():
        slide = meta.get("slide_number")
        locator = f"Slide {slide}" if slide else ""
        if meta.get("sheet_state"):
            locator = "hidden sheet"
        # Optional (src/utils/fail_open.py): a chunk whose header cannot be
        # built is indexed under its bare text, as with headers off.
        header = fail_open(
            "context_header", build_context_header,
            doc.filename,
            title=title,
            heading_path=meta.get("heading_path") or [chunk.section_heading or ""],
            locator=locator,
            default=None,
        )
        if header is not None:
            payload["context_header"] = header
    return payload


def _mark_injection(payload: dict, doc: ExtractedDocument) -> None:
    """
    Screens one payload's text for instruction-like wording (injection_screen).

    The fields are written only when something is found, so the payload of an
    ordinary chunk is unchanged; an index written before the screen existed is
    re-screened at query time anyway (src/verification/injection_check.py).

    Args:
        payload: Child or parent payload; gains suspected_injection=1 and
            injection_signals when flagged.
        doc: Document being built; flagged children are listed on it.
    """
    signals = screen_text(payload.get("text") or "")
    if not signals:
        return
    payload["suspected_injection"] = 1
    payload["injection_signals"] = signals
    if "child_chunk_ids" not in payload:
        doc.suspected_injection.append({
            "chunk_id": payload["chunk_id"],
            "page_number": payload.get("page_number"),
            "section_heading": payload.get("section_heading") or "",
            "signals": signals,
        })


def _laya_chunk_scores(
    texts: list[str],
    regex_risk: list,
    filename: str,
) -> tuple[list[ingest_signals.ChunkScores] | None, bool]:
    """
    Runs the enabled Laya ingest questions over every chunk of one document.

    One request per chunk carries all of that chunk's questions (risk and PII
    together). In "confirm" risk mode a chunk is asked only about the categories
    its regex matched, so a chunk with no match and PII off costs nothing.

    Args:
        texts: Child chunk texts.
        regex_risk: RiskSignalResult per chunk from the regex pass.
        filename: For logging.

    Returns:
        (scores, risk_used): scores is None when Laya is off or unavailable —
        the caller keeps the regex results; risk_used says whether the scores
        carry risk judgements to apply.
    """
    use_risk = ingest_signals.laya_risk_enabled()
    use_pii = ingest_signals.laya_pii_enabled()
    if not texts or not laya_enabled() or not (use_risk or use_pii):
        return None, False

    risk_categories = None
    mode = None
    if use_risk:
        mode = ingest_signals.risk_mode()
        risk_categories = (
            [None] * len(texts) if mode == "union" else [r.signals for r in regex_risk]
        )
    try:
        scores = ingest_signals.score_chunks(texts, risk_categories=risk_categories, pii=use_pii)
    except LayaUnavailable as e:
        if use_pii:
            # LAYA_PII=1 flags chunks the regex does not, so the regex alone
            # would index as PII-free a chunk the configured screen excludes:
            # PII detection fails closed, and nothing of the document is
            # indexed (Decision 88). Risk and category are optional and do
            # fall back, as below, when PII is not Laya's to judge.
            logger.error(
                "Laya unavailable at ingest with LAYA_PII=1; refusing to index unscreened",
                extra={"file_name": filename, "error": str(e)},
            )
            raise PIIScreeningUnavailable(
                "The document could not be screened for personal data (the screening model is "
                "unavailable), so nothing was indexed. Retry the upload later."
            ) from e
        logger.warning(
            "Laya unavailable at ingest; keeping regex risk and PII results",
            extra={"file_name": filename, "error": str(e)},
        )
        return None, False
    logger.info(
        "Laya ingest decisions applied",
        extra={"file_name": filename, "chunks": len(texts), "risk_mode": mode, "pii": use_pii},
    )
    return scores, use_risk


def extract_and_chunk(
    file_path: str,
    filename: str,
    deal_id: str,
    document_category: str | None = None,
    *,
    is_current_version: bool = True,
    supersedes_doc_id: str | None = None,
    doc_id: str | None = None,
) -> ExtractedDocument:
    """
    Extracts, chunks and annotates a document — everything short of embedding.

    Synchronous and CPU-bound (parsing, tokenising, PII and risk regexes, and
    the Laya decision model when enabled): call it through asyncio.to_thread
    from async code. With LAYA_ENABLED=0 no model is loaded and no network is
    used, so it is directly unit-testable.

    Args:
        file_path: Path to the file on disk (any name; the format is taken
                   from `filename`).
        filename: Original (sanitised) filename — stored as source_file.
        deal_id: Deal the document belongs to.
        document_category: Category override; auto-classified when None.
        is_current_version: Version flag written to every payload.
        supersedes_doc_id: Document this one replaces, if any.
        doc_id: Explicit doc id; by default derived from deal_id + content hash.

    Returns:
        ExtractedDocument with child and parent payloads (no vectors).

    Raises:
        UnsupportedFileTypeError: Unknown extension.
        UnsafeDocumentError: Office archive over the safety limits.
        DocumentExtractionError: The file could not be parsed.
        NoExtractableContentError: Parsing produced nothing to index.
    """
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {extension or '(none)'}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    validate_office_archive(file_path, extension)

    content_sha256 = compute_content_hash(file_path)
    doc_id = doc_id or compute_doc_id(deal_id, content_sha256)

    category_source, category_confidence = "override", None
    if document_category is None:
        document_category, category_source, category_confidence = (
            DocumentClassifier().classify_with_source(
                file_name=filename,
                file_type=extension.lstrip("."),
                content_sample=_content_sample(file_path, extension, doc_id),
            )
        )

    try:
        sections, warnings, has_redline = extract_sections(
            file_path, extension, doc_id, deal_id, document_category
        )
    except IngestionError:
        raise
    except Exception as e:
        logger.error(
            "Document extraction failed",
            extra={"doc_id": doc_id, "file_name": filename, "error": str(e)},
            exc_info=True,
        )
        raise DocumentExtractionError(
            f"The {extension} file could not be parsed"
        ) from e

    for warning in warnings:
        logger.error("Partial extraction failure", extra={"doc_id": doc_id, "warning": warning})

    doc = ExtractedDocument(
        doc_id=doc_id,
        deal_id=deal_id,
        filename=filename,
        extension=extension,
        document_category=document_category,
        content_sha256=content_sha256,
        warnings=warnings,
        has_redline=has_redline,
        category_source=category_source,
        category_confidence=category_confidence,
        # Optional (src/utils/fail_open.py, Decision 88): a filing whose cover
        # cannot be read is indexed without filing_* fields, which retrieval
        # already treats as "not a filing".
        filing=fail_open("filing_metadata", extract_filing_metadata, sections, default={}),
    )

    if not sections:
        raise NoExtractableContentError("No text could be extracted from the document")

    # Used only by context headers; "" is a document that opens with no heading.
    title = fail_open("context_header", document_title, sections, default="")
    structural_chunks = StructuralChunker().chunk(sections)
    children, parents = SemanticChunker().chunk_with_parents(
        structural_chunks, parent_target_tokens=DEFAULT_PARENT_TARGET_TOKENS
    )
    if not children:
        raise NoExtractableContentError("No text could be extracted from the document")

    # PII and risk signals start as regex scans. When enabled, Laya then judges
    # every chunk in one batched pass (see _laya_chunk_scores) and the regex
    # results are combined with its probabilities; if Laya is unavailable the
    # regex results stand. Signals are written per chunk AND aggregated per
    # document for the risk dashboard.
    pii_detector = PIIDetector()
    risk_extractor = RiskSignalExtractor()
    aggregated_risk: dict[str, dict] = {}
    parent_ids = [f"{deal_id}_{doc_id}_p{j:04d}" for j in range(len(parents))]
    parent_pii = [0] * len(parents)
    parent_children: list[list[str]] = [[] for _ in parents]

    # Optional: a chunk whose risk scan fails gets no regex signals. Risk
    # signals are read by the dashboard and the trace, never by a retrieval
    # filter.
    regex_risk = [
        fail_open(
            "risk_signals", risk_extractor.extract,
            c.text, file_name=filename, document_category=document_category,
            default=RiskSignalResult(signals=[], signal_details=[], signal_count=0),
        )
        for c in children
    ]
    laya_scores, laya_risk_used = _laya_chunk_scores(
        [c.text for c in children], regex_risk, filename
    )

    for i, chunk in enumerate(children):
        chunk_id = f"{deal_id}_{doc_id}_{i:04d}"
        scores = laya_scores[i] if laya_scores else None
        pii = pii_detector.detect(
            chunk.text, laya_probability=scores.pii_probability if scores else None
        )
        contains_pii = pii.contains_pii

        risk = regex_risk[i]
        if scores is not None and laya_risk_used:
            risk = risk_extractor.apply_laya(risk, scores.risk)
        risk_decisions = {
            d["signal_type"]: {"source": d["source"], "confidence": d["confidence"], "accepted": True}
            for d in risk.signal_details
        }
        for r in risk.rejected:
            risk_decisions[r["signal_type"]] = {
                "source": "regex", "confidence": r["confidence"], "accepted": False,
            }

        for detail in risk.signal_details:
            entry = aggregated_risk.setdefault(
                detail["signal_type"],
                {
                    "signal_type": detail["signal_type"],
                    "match_count": 0,
                    "sample_matches": [],
                    "page_number": chunk.page_number,
                    "source": "regex",
                    "confidence": None,
                },
            )
            entry["match_count"] += detail["match_count"]
            for sample in detail["sample_matches"]:
                if sample and len(entry["sample_matches"]) < 3:
                    entry["sample_matches"].append(sample)
            # Document level: "laya" once Laya flagged or confirmed the type in
            # any chunk; confidence is the highest Laya probability seen.
            if "laya" in detail["source"]:
                entry["source"] = "laya"
            if detail["confidence"] is not None:
                entry["confidence"] = max(entry["confidence"] or 0.0, detail["confidence"])

        parent_index = chunk.metadata.get("parent_index")
        parent_chunk_id = None
        if parent_index is not None:
            parent_chunk_id = parent_ids[parent_index]
            parent_pii[parent_index] = max(parent_pii[parent_index], contains_pii)
            parent_children[parent_index].append(chunk_id)

        payload = _child_payload(
            chunk, chunk_id, i, doc,
            is_current_version=is_current_version,
            supersedes_doc_id=supersedes_doc_id,
            contains_pii=contains_pii,
            risk_signals=risk.signals,
            parent_chunk_id=parent_chunk_id,
            risk_decisions=risk_decisions,
            pii_source=pii.source,
            pii_confidence=pii.laya_probability,
            title=title,
        )
        _mark_injection(payload, doc)
        doc.chunks.append(payload)

    # Which agreement this is and who its parties are, read from the opening
    # chunks just built and written on every one of them, so a question that
    # names the document can be matched to it (document_identity.py; read by
    # named_document.py, Decision 86). `python -m src.vector_db.identity_backfill`
    # computes the same fields from an existing index's payloads. Optional: a
    # document whose identity cannot be read gets none, and the query step keeps
    # the fused-score choice for it.
    identity = fail_open("document_identity", extract_document_identity, doc.chunks, default={})
    for payload in doc.chunks:
        payload.update(identity)

    for j, parent in enumerate(parents):
        parent_payload = {
            "chunk_id": parent_ids[j],
            "deal_id": deal_id,
            "doc_id": doc_id,
            "text": parent.text,
            "source_file": filename,
            "document_category": document_category,
            "section_heading": parent.section_heading,
            "page_number": parent.page_number,
            "token_count": parent.token_count,
            "child_chunk_ids": parent_children[j],
            "is_current_version": 1 if is_current_version else 0,
            # A parent is shown with any of its children, so it is PII if any
            # child is — otherwise parent expansion would leak what the child
            # filter withheld.
            "contains_pii": parent_pii[j],
        }
        _mark_injection(parent_payload, doc)
        doc.parents.append(parent_payload)

    doc.risk_signals = list(aggregated_risk.values())
    if doc.suspected_injection:
        # Labelled, never dropped: the text stays indexed exactly as written,
        # and the query path tells the reviewer when an answer leans on it.
        logger.warning(
            "Document contains text that reads like instructions to an AI model",
            extra={
                "doc_id": doc_id,
                "file_name": filename,
                "chunks": len(doc.suspected_injection),
                "signals": sorted({s for f in doc.suspected_injection for s in f["signals"]}),
            },
        )

    logger.info(
        "Document extracted and chunked",
        extra={
            "doc_id": doc_id,
            "file_name": filename,
            "category": document_category,
            "sections": len(sections),
            "chunks": len(doc.chunks),
            "parents": len(doc.parents),
            "tables": len({c["table_id"] for c in doc.chunks if c.get("table_id")}),
        },
    )
    return doc


# ==============================================================================
# Indexing
# ==============================================================================


def _doc_filter(deal_id: str, doc_id: str, keep_ids: list[str] | None = None,
                only_ids: list[str] | None = None):
    """Filter for one document's points, optionally excluding, or limited to, the given ids."""
    from qdrant_client.models import FieldCondition, Filter, HasIdCondition, MatchValue

    must = [
        FieldCondition(key="deal_id", match=MatchValue(value=deal_id)),
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
    ]
    if only_ids:
        must.append(HasIdCondition(has_id=only_ids))
    return Filter(
        must=must,
        must_not=[HasIdCondition(has_id=keep_ids)] if keep_ids else None,
    )


def _staged_filter(deal_id: str, doc_id: str, keep_ids: list[str] | None = None):
    """
    Filter for one document's staged points: written by a first-time ingest
    that has not committed them (no deal_id yet), optionally excluding ids.
    """
    from qdrant_client.models import (
        FieldCondition, Filter, HasIdCondition, IsEmptyCondition, MatchValue, PayloadField,
    )

    return Filter(
        must=[
            FieldCondition(key=STAGED_DEAL_FIELD, match=MatchValue(value=deal_id)),
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            IsEmptyCondition(is_empty=PayloadField(key="deal_id")),
        ],
        must_not=[HasIdCondition(has_id=keep_ids)] if keep_ids else None,
    )


def _staged_payload(payload: dict, deal_id: str) -> dict:
    """A point's payload as a first-time ingest writes it: deal_id held back."""
    staged = {k: v for k, v in payload.items() if k != "deal_id"}
    staged[STAGED_DEAL_FIELD] = deal_id
    return staged


async def _commit_staged(client, deal_id: str, written: list[tuple[str, list]]) -> None:
    """
    Makes a first-time ingest's staged points readable: sets deal_id on them.

    Args:
        client: AsyncQdrantClient.
        deal_id: Deal.
        written: (collection, points written) pairs, parents first: a child
            must never be readable while its parent section is not.

    Raises:
        Whatever the client raises; the caller rolls the document back.
    """
    # One write per collection, naming exactly this run's points: once the
    # server has the request it applies all of it (and replays it from its
    # write-ahead log after a crash of its own), so a client that dies here
    # leaves a collection committed or still staged — never the half-written
    # document a crash in the batched upsert used to leave searchable. It is
    # not isolated from readers: a search that runs while the server applies
    # it can see part of the children, for that moment. Points an earlier,
    # cut-short upload of this document staged under other ids are not named,
    # so they stay staged (see _drop_staging_leftovers).
    for name, points in written:
        await client.set_payload(
            collection_name=name,
            payload={"deal_id": deal_id},
            points=[p.id for p in points],
            wait=True,
        )


async def _drop_staging_leftovers(client, deal_id: str, doc_id: str, written: list[tuple[str, list]]) -> None:
    """
    After a commit: removes the staging marker from this run's points, and
    deletes what an earlier, cut-short upload of the same document staged
    under ids this run did not write (it chunked the document differently).

    Neither changes what a reader sees: the marker is read by nothing once
    deal_id is set (it is removed so a committed point stores what it stored
    before staging existed), and the leftovers carry no deal_id.
    """
    for name, points in written:
        if points:
            # Only from points that carry deal_id: one another process staged
            # again since the commit (an upload of the same bytes outside this
            # process's lock) keeps the marker purges and the sweeper find it by.
            await client.delete_payload(
                collection_name=name, keys=[STAGED_DEAL_FIELD],
                points=_doc_filter(deal_id, doc_id, keep_ids=None, only_ids=[p.id for p in points]),
                wait=True,
            )
    for name, points in written:
        await client.delete(
            collection_name=name,
            points_selector=_staged_filter(deal_id, doc_id, [p.id for p in points]),
            wait=True,
        )


async def _version_candidates(
    client,
    collection_name: str,
    deal_id: str,
    doc_id: str,
    filename: str,
    explicit: str | None = None,
) -> list[dict]:
    """
    Current documents of the deal that the upload looks like another version
    of, judged by file name (DocumentVersionResolver.relation).

    A proposal, never an action: version state changes only through an
    explicit supersedes_doc_id, because retiring a document on a file-name
    guess removes it from every answer without anyone having decided that.
    Without this, uploading merger_agreement_v2.pdf next to v1 left both
    current and nothing said so (audit I10). Best effort: a failure to look
    returns [] — the document is already indexed.

    Args:
        client: AsyncQdrantClient.
        collection_name: Child collection.
        deal_id: Deal.
        doc_id: The document just indexed (excluded).
        filename: Its file name.
        explicit: supersedes_doc_id the caller already gave (excluded).

    Returns:
        [{"doc_id", "source_file", "relation": "newer"|"older"|"same_document"}],
        "newer" meaning the upload looks like the later version.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from src.data_processing.document_version_resolver import DocumentVersionResolver

    try:
        points, _ = await client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="deal_id", match=MatchValue(value=deal_id)),
                FieldCondition(key="chunk_index", match=MatchValue(value=0)),
                FieldCondition(key="is_current_version", match=MatchValue(value=1)),
            ]),
            limit=10_000,
            with_payload=["doc_id", "source_file"],
            with_vectors=False,
        )
    except Exception as e:
        logger.warning("Version candidate lookup failed", extra={"doc_id": doc_id, "error": str(e)})
        return []

    resolver = DocumentVersionResolver()
    candidates = []
    for point in points:
        payload = point.payload or {}
        other = payload.get("doc_id")
        if not other or other in (doc_id, explicit):
            continue
        relation = resolver.relation(filename, str(payload.get("source_file") or ""))
        if relation:
            candidates.append({
                "doc_id": other, "source_file": payload.get("source_file"), "relation": relation,
            })
    if candidates:
        logger.info(
            "Upload looks like another version of an indexed document",
            extra={"doc_id": doc_id, "file_name": filename, "candidates": candidates},
        )
    return candidates


async def _retired_by(client, collection_name: str, deal_id: str, doc_id: str) -> str:
    """The doc_id that superseded this document, or "" when it is current."""
    points, _ = await client.scroll(
        collection_name=collection_name,
        scroll_filter=_doc_filter(deal_id, doc_id),
        limit=1,
        with_payload=["is_current_version", "superseded_by"],
        with_vectors=False,
    )
    if not points:
        return ""
    payload = points[0].payload or {}
    if payload.get("is_current_version") == 0 and payload.get("superseded_by"):
        return str(payload["superseded_by"])
    return ""


async def _upsert_batches(client, collection_name: str, points: list) -> None:
    """Upserts points in QDRANT_BATCH_SIZE batches, waiting for each."""
    for start in range(0, len(points), QDRANT_BATCH_SIZE):
        await client.upsert(
            collection_name=collection_name,
            points=points[start:start + QDRANT_BATCH_SIZE],
            wait=True,
        )


_DOCUMENT_LOCKS: dict[tuple[str, str], list] = {}


@contextlib.asynccontextmanager
async def _document_lock(deal_id: str, doc_id: str):
    """One write phase at a time per (deal_id, doc_id) in this process."""
    key = (deal_id, doc_id)
    entry = _DOCUMENT_LOCKS.setdefault(key, [asyncio.Lock(), 0])
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if not entry[1]:
            _DOCUMENT_LOCKS.pop(key, None)


async def index_document(
    file_path: str,
    filename: str,
    deal_id: str,
    document_category: str | None = None,
    *,
    is_current_version: bool = True,
    supersedes_doc_id: str | None = None,
    client=None,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
    embed_fn: EmbedFn | None = None,
    sparse_fn: SparseFn | None = None,
    ensure_collections: bool = False,
) -> dict:
    """
    Extracts, chunks, embeds and upserts one document. Idempotent per content.

    Steps: extract_and_chunk (worker thread) → dense embeddings → batched BM25
    (worker thread) → upsert children and parents → delete this document's
    stale points (ids no longer produced, e.g. after a chunking change).
    Re-indexing identical bytes therefore replaces rather than duplicates.

    A first-time ingest writes its points staged — deal_id held back in
    STAGED_DEAL_FIELD — and commits them (sets deal_id) only once every point
    is written, parents before children. Every reader filters on deal_id, so
    a run cut short where no handler runs (a crash, an OOM kill, a cancelled
    request) leaves points nothing reads; purging the deal deletes them, and
    indexing the file again commits it. If an upsert or the commit fails, every point for this
    (deal_id, doc_id), staged or committed, is deleted from both collections
    before the error propagates, so a new document is either fully indexed or
    absent. On a
    re-index nothing is deleted: when the chunking is unchanged the copy that
    was serving stays as it was; after an extraction or chunking change the
    document holds old and new chunks until it is indexed again (its points
    then carry two ingest_version stamps, which reingest_report flags), and
    indexing the same file again completes it.

    A document that a newer version has superseded stays superseded when it is
    re-indexed (is_current_version=0, superseded_by kept, and a warning).

    Args:
        file_path: Path to the file on disk.
        filename: Original filename (stored as source_file; sets the format).
        deal_id: Deal the document belongs to.
        document_category: Category override; auto-classified when None.
        is_current_version: Version flag for every point.
        supersedes_doc_id: Document this one replaces (the caller retires it
                           with mark_superseded()).
        client: AsyncQdrantClient; defaults to the application singleton.
        collection_name: Child collection.
        parent_collection_name: Parent collection.
        embed_fn: async (texts) -> array (n, dim); defaults to bge-m3.
        sparse_fn: (texts) -> list[SparseVector]; defaults to batched BM25.
        ensure_collections: Create collections/indexes first if missing.

    Returns:
        Dict with doc_id, deal_id, filename, document_category, chunks_created,
        parent_chunks_created, table_count, risk_signals, has_redline, warnings,
        suspected_injection (child chunks with instruction-like text), content_sha256, previously_indexed, superseded_by ("" unless this
        document had been retired by a newer version) and version_candidates
        (current documents of the deal this upload looks like another version
        of — a proposal; see _version_candidates).

    Raises:
        IngestionError subclasses for client-attributable failures; other
        exceptions (Qdrant, embedding) propagate after rollback.
    """
    if client is None:
        from src.vector_db.qdrant_client import get_qdrant_client
        client = get_qdrant_client()
    if embed_fn is None:
        from src.vector_db.reranker import embed_texts_async
        embed_fn = embed_texts_async
    # Stamped on every child point so vectors written by an older BM25 text
    # normalisation can be found (see hybrid_search.SPARSE_ENCODING_VERSION).
    # A caller-supplied sparse_fn has no known encoding, so it is not stamped.
    sparse_encoding = None
    if sparse_fn is None:
        from src.vector_db.hybrid_search import (
            SPARSE_ENCODING_VERSION,
            compute_sparse_bm25_batch,
        )
        sparse_fn = compute_sparse_bm25_batch
        sparse_encoding = SPARSE_ENCODING_VERSION
    if ensure_collections:
        from src.vector_db.collection_manager import setup_collections
        await setup_collections(client, collection_name, parent_collection_name)

    # Parsing, chunking, PII and risk scans (and Laya's forward passes) are
    # CPU/GPU-bound; running them on the event loop froze every other request
    # on the single-worker API.
    doc = await asyncio.to_thread(
        extract_and_chunk,
        file_path,
        filename,
        deal_id,
        document_category,
        is_current_version=is_current_version,
        supersedes_doc_id=supersedes_doc_id,
    )

    # Concurrent uploads of the same bytes write the same point ids. Staged,
    # the later one's upserts would hide the earlier one's committed points
    # again, and the earlier one's cleanup would strip the later one's marker;
    # one at a time, the later one finds the document and re-indexes in place.
    # The API runs a single worker (enforce_single_worker), so this serialises
    # every upload of the API; a separate process (a CLI loader) is not covered.
    async with _document_lock(deal_id, doc.doc_id):
        return await _write_document(
            doc, deal_id, filename, supersedes_doc_id,
            client=client, collection_name=collection_name,
            parent_collection_name=parent_collection_name,
            embed_fn=embed_fn, sparse_fn=sparse_fn, sparse_encoding=sparse_encoding,
        )


async def _write_document(
    doc,
    deal_id: str,
    filename: str,
    supersedes_doc_id: str | None,
    *,
    client,
    collection_name: str,
    parent_collection_name: str,
    embed_fn: EmbedFn,
    sparse_fn: SparseFn,
    sparse_encoding: int | None,
) -> dict:
    """index_document's write phase, run under the document's lock: see there."""
    from qdrant_client.models import PointStruct

    existing = await client.count(
        collection_name=collection_name,
        count_filter=_doc_filter(deal_id, doc.doc_id),
        exact=True,
    )
    # Parents with no child left (a rollback whose parent delete failed) are
    # this document's too. They decide nothing below but the stale delete,
    # which removes those the new chunking does not overwrite.
    leftover_parents = 0 if existing.count else (await client.count(
        collection_name=parent_collection_name,
        count_filter=_doc_filter(deal_id, doc.doc_id),
        exact=True,
    )).count
    retired_by = await _retired_by(client, collection_name, deal_id, doc.doc_id) \
        if existing.count else ""
    if retired_by:
        # Re-indexing a retired version (a folder re-ingest, a bulk loader, the
        # old file uploaded again by mistake) used to rewrite its payloads as
        # current and both versions were then retrieved side by side. The
        # retirement stands; only superseding the newer version undoes it.
        for payload in [*doc.chunks, *doc.parents]:
            payload["is_current_version"] = 0
            payload["superseded_by"] = retired_by
        doc.warnings.append(
            f"This document was superseded by {retired_by}; it was re-indexed as a superseded "
            "version and is not searched by default"
        )

    # Vectors are computed over header + text (see chunk_context.py); the
    # payload keeps the source text alone.
    texts = [embedding_text(c) for c in doc.chunks]
    dense = np.asarray(await embed_fn(texts), dtype=np.float32)
    sparse = await asyncio.to_thread(sparse_fn, texts)

    # Every stored point says which extraction and chunking code wrote it
    # (INGEST_VERSION), so a re-index cut short, or a document indexed before an
    # extraction change, can be told apart (src/vector_db/reingest_report.py).
    stamp = {"ingest_version": INGEST_VERSION}
    if sparse_encoding is not None:
        stamp["sparse_encoding"] = sparse_encoding

    # A re-index writes in place (the copy that was serving stays readable
    # throughout, audit I9); a first-time ingest stages until committed.
    staged = not existing.count

    def stored(payload: dict) -> dict:
        return _staged_payload(payload, deal_id) if staged else payload

    child_points = [
        PointStruct(
            id=point_id_for(payload["chunk_id"]),
            vector={"dense": dense[i].tolist(), "sparse": sparse[i]},
            payload=stored({**payload, **stamp}),
        )
        for i, payload in enumerate(doc.chunks)
    ]

    # The parent collection requires a dense vector, but parents are fetched by
    # id, never searched — the normalised mean of their children's vectors is a
    # meaningful placeholder that costs no extra embedding pass.
    child_index = {c["chunk_id"]: i for i, c in enumerate(doc.chunks)}
    parent_points = []
    for parent in doc.parents:
        rows = [child_index[cid] for cid in parent["child_chunk_ids"]]
        vector = dense[rows].mean(axis=0)
        norm = float(np.linalg.norm(vector)) or 1.0
        parent_points.append(PointStruct(
            id=point_id_for(parent["chunk_id"]),
            vector={"dense": (vector / norm).tolist()},
            payload=stored({**parent, "ingest_version": INGEST_VERSION}),
        ))

    try:
        await _upsert_batches(client, collection_name, child_points)
        await _upsert_batches(client, parent_collection_name, parent_points)
        if staged:
            await _commit_staged(client, deal_id, [
                (parent_collection_name, parent_points), (collection_name, child_points),
            ])
    except Exception:
        if existing.count:
            # A re-index of a document that was already serving. Deleting every
            # point of the doc_id here removed the good copy along with the
            # failed attempt (audit I9), so nothing is deleted. Point ids come
            # from positions, not content: with the chunking unchanged, what is
            # there now is the old copy with some chunks rewritten to the same
            # content. After an extraction or chunking change it is old chunks
            # and new ones, with parents that may not hold their children, until
            # the same file is indexed again (that run upserts every new point
            # and deletes the rest). The two ingest_version stamps the document
            # now carries are what reingest_report flags (Decision 97).
            logger.error(
                "Re-index upsert failed; the document holds the chunks of both runs until it is "
                "indexed again",
                extra={"doc_id": doc.doc_id, "deal_id": deal_id},
                exc_info=True,
            )
            raise
        logger.error(
            "Upsert failed; rolling back document",
            extra={"doc_id": doc.doc_id, "deal_id": deal_id},
            exc_info=True,
        )
        # Committed points too: the commit writes parents first, so a failure
        # writing the children's can leave this document's parents committed.
        for name in (collection_name, parent_collection_name):
            for selector in (_doc_filter(deal_id, doc.doc_id), _staged_filter(deal_id, doc.doc_id)):
                try:
                    await client.delete(
                        collection_name=name,
                        points_selector=selector,
                        wait=True,
                    )
                except Exception as rollback_error:
                    logger.error(
                        "Rollback delete failed",
                        extra={"doc_id": doc.doc_id, "collection": name,
                               "error": str(rollback_error)},
                    )
        raise

    if staged:
        # Optional (Decision 88): the document is committed and whole by now.
        await fail_open_async(
            "staging_cleanup", _drop_staging_leftovers, client, deal_id, doc.doc_id,
            [(parent_collection_name, parent_points), (collection_name, child_points)],
            default=None,
        )

    async def delete_stale() -> None:
        for name, points in (
            (collection_name, child_points),
            (parent_collection_name, parent_points),
        ):
            await client.delete(
                collection_name=name,
                points_selector=_doc_filter(deal_id, doc.doc_id, [p.id for p in points]),
                wait=True,
            )

    # Remove points this document no longer produces (only relevant on re-index).
    if existing.count:
        await delete_stale()
    elif leftover_parents:
        # A first ingest that finds parents an earlier failed rollback left
        # behind. The document is already whole and searchable, so failing to
        # remove them must not turn the upload into an error with no registry
        # or audit entry: optional (Decision 88), and a parent no child names
        # is reported as orphan_parent by reingest_report.
        await fail_open_async("leftover_parent_delete", delete_stale, default=None)

    result = {
        "doc_id": doc.doc_id,
        "deal_id": deal_id,
        "filename": filename,
        "document_category": doc.document_category,
        "chunks_created": len(child_points),
        "parent_chunks_created": len(parent_points),
        "table_count": len({c["table_id"] for c in doc.chunks if c.get("table_id")}),
        "risk_signals": doc.risk_signals,
        "category_source": doc.category_source,
        "category_confidence": doc.category_confidence,
        "has_redline": doc.has_redline,
        "warnings": doc.warnings,
        "suspected_injection": doc.suspected_injection,
        "content_sha256": doc.content_sha256,
        "previously_indexed": bool(existing.count),
        "superseded_by": retired_by,
        # Optional, and after the upsert: a failure here must not turn an
        # indexed document into an error response. No proposal, then.
        "version_candidates": await fail_open_async(
            "version_candidates", _version_candidates,
            client, collection_name, deal_id, doc.doc_id, filename,
            explicit=supersedes_doc_id,
            default=[],
        ),
    }
    logger.info(
        "Document indexed",
        extra={
            # "filename" is a reserved LogRecord attribute
            "file_name": filename,
            **{k: v for k, v in result.items()
               if k not in ("filename", "risk_signals", "warnings", "suspected_injection")},
        },
    )
    return result


async def delete_document(
    deal_id: str,
    doc_id: str,
    *,
    client=None,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
    unless: Callable[[], bool] | None = None,
) -> bool:
    """
    Deletes every point of one document: committed and staged, children and
    parents, the same selectors a failed ingest rolls back with.

    Used when an upload must not stand although it was written: its deal was
    purged while it was being indexed (api/routes/ingest.py). Held under the
    document's lock, so it does not interleave with a write of the same bytes.

    Args:
        unless: Checked under the lock; when it returns True nothing is
            deleted (another upload of the same bytes owns the points now).

    Returns:
        True if the delete ran, False if `unless` stopped it.

    Raises:
        Exception: the first delete that failed, after every delete was tried.
    """
    if client is None:
        from src.vector_db.qdrant_client import get_qdrant_client
        client = get_qdrant_client()
    failed: Exception | None = None
    async with _document_lock(deal_id, doc_id):
        if unless is not None and unless():
            return False
        for name in (parent_collection_name, collection_name):
            for selector in (_doc_filter(deal_id, doc_id), _staged_filter(deal_id, doc_id)):
                try:
                    await client.delete(collection_name=name, points_selector=selector, wait=True)
                except Exception as e:
                    logger.error(
                        "Document delete failed",
                        extra={"doc_id": doc_id, "deal_id": deal_id, "collection": name,
                               "error": str(e)},
                    )
                    failed = failed or e
    if failed is not None:
        raise failed
    return True


async def mark_superseded(
    deal_id: str,
    superseded_doc_id: str,
    superseded_by: str,
    client=None,
    collection_name: str = COLLECTION_NAME,
    parent_collection_name: str = PARENT_COLLECTION_NAME,
) -> dict:
    """
    Flips every chunk (and parent) of a superseded document to is_current_version=0.

    Without this, uploading a replacement document leaves both versions marked
    current, and retrieval's is_current_version=1 filter happily returns stale
    terms alongside the ones that replaced them.

    Failures are not raised — the new document is already indexed and usable —
    but they are reported: the result says whether the retirement happened, so
    the caller can answer "partial" instead of recording a supersession that
    Qdrant never applied (audit I8). A superseded_doc_id with no points in the
    deal (a typo, another deal's id) is a failure, not a silent no-op.

    Args:
        deal_id: Deal scope, so one deal can never retire another deal's docs.
        superseded_doc_id: Document being retired.
        superseded_by: Document ID replacing it.
        client: AsyncQdrantClient; defaults to the application singleton.
        collection_name: Child collection.
        parent_collection_name: Parent collection.

    Returns:
        {"ok": bool, "chunks_retired": int, "error": str | None}. ok is True
        only when the document existed and both collections were updated.
    """
    if superseded_doc_id == superseded_by:
        # Re-uploading identical bytes yields the same doc_id; retiring it
        # would retire the document that was just indexed.
        logger.warning(
            "Ignoring self-supersession", extra={"doc_id": superseded_doc_id}
        )
        return {"ok": False, "chunks_retired": 0, "error": "a document cannot supersede itself"}

    if client is None:
        from src.vector_db.qdrant_client import get_qdrant_client
        client = get_qdrant_client()

    try:
        found = await client.count(
            collection_name=collection_name,
            count_filter=_doc_filter(deal_id, superseded_doc_id),
            exact=True,
        )
        if not found.count:
            logger.error(
                "Superseded document not found in the deal",
                extra={"deal_id": deal_id, "superseded_doc_id": superseded_doc_id},
            )
            return {"ok": False, "chunks_retired": 0,
                    "error": "the superseded document is not indexed in this deal"}
        for name in (collection_name, parent_collection_name):
            await client.set_payload(
                collection_name=name,
                payload={"is_current_version": 0, "superseded_by": superseded_by},
                points=_doc_filter(deal_id, superseded_doc_id),
                wait=True,
            )
        logger.info(
            "Superseded document retired",
            extra={
                "deal_id": deal_id,
                "superseded_doc_id": superseded_doc_id,
                "superseded_by": superseded_by,
                "chunks_retired": found.count,
            },
        )
        return {"ok": True, "chunks_retired": found.count, "error": None}
    except Exception as e:
        logger.error(
            "Failed to retire superseded document",
            extra={
                "deal_id": deal_id,
                "superseded_doc_id": superseded_doc_id,
                "error": str(e),
            },
        )
        return {"ok": False, "chunks_retired": 0, "error": "the version store could not be updated"}
