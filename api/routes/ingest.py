"""
Document ingestion routes.

The route owns the HTTP concerns — upload size cap, filename handling, error
mapping. Everything from parsing to upsert lives in
src/data_processing/ingest_pipeline.py so the evaluation harness indexes
through exactly the same code.
"""

import asyncio
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from api.models.request_models import DEAL_ID_PATTERN
from api.models.response_models import IngestResponse, InjectionScreen, PassageLocation
from api.routes.deals import (
    IngestTicket,
    register_document,
    uploaded_again_since_purge,
    wait_for_purges,
    watch_for_purge,
)
from api.security import (
    ClientContext,
    acquire_pipeline_slot,
    ingest_rate_limit,
    max_upload_bytes,
    public_error_detail,
    require_sandbox_deal,
)
from src.data_processing.ingest_pipeline import (
    SUPPORTED_EXTENSIONS,
    IngestionError,
    compute_content_hash,
    compute_doc_id,
    delete_document,
    index_document,
    mark_superseded,
    sanitize_filename,
)
from src.data_processing.injection_screen import SIGNALS, flagged_locations
from src.utils import audit_log, metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

_READ_CHUNK_BYTES = 1024 * 1024


def _max_upload_bytes() -> int:
    """Upload size cap in bytes (env MAX_UPLOAD_MB, default 25)."""
    return max_upload_bytes()


# Office formats are zip archives, and a zip's size on the wire says nothing
# about its size once opened: a few hundred KB of repeated bytes expands to
# gigabytes. The upload cap bounds what is received; this bounds what the
# parsers are then asked to inflate into memory on a host with no room for it.
_ZIP_EXTENSIONS = {".docx", ".xlsx", ".pptx"}
_MAX_ARCHIVE_MEMBERS = 5000


def _max_expanded_bytes() -> int:
    """Ceiling on an office file's total uncompressed size (env MAX_EXPANDED_MB, default 300)."""
    try:
        return max(1, int(os.getenv("MAX_EXPANDED_MB", "300"))) * 1024 * 1024
    except ValueError:
        return 300 * 1024 * 1024


def _reject_archive_bomb(path: str, extension: str) -> None:
    """
    Refuses an office document whose declared uncompressed size is implausible.

    Reads only the archive's central directory — no member is decompressed.

    Args:
        path: The saved upload.
        extension: Its lower-cased extension.

    Raises:
        HTTPException: 400 for a file that is not a valid archive, 413 for one
            that would expand past the limit or holds too many members.
    """
    if extension not in _ZIP_EXTENSIONS:
        return
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=400, detail=f"The file is not a valid {extension} document."
        ) from None
    expanded = sum(member.file_size for member in members)
    if len(members) > _MAX_ARCHIVE_MEMBERS or expanded > _max_expanded_bytes():
        logger.warning(
            "Rejected archive bomb",
            extra={"members": len(members), "expanded_bytes": expanded},
        )
        raise HTTPException(
            status_code=413,
            detail="The document expands to more data than this service will process.",
        )


def _injection_screen(result: dict) -> InjectionScreen:
    """
    The ingest screen's findings for the uploader (Decision 82).

    index_document labels passages whose text reads like instructions to an AI
    model and indexes them unchanged (Decision 76); the person who uploaded a
    seller-supplied file is the one best placed to open it and look, so the
    response says how many passages, which signals, and where. Locations only:
    no document text is echoed.

    Args:
        result: index_document's result.

    Returns:
        The screen summary; empty when nothing was flagged.
    """
    flagged = [f for f in result.get("suspected_injection") or [] if isinstance(f, dict)]
    if not flagged:
        return InjectionScreen()
    locations = flagged_locations(flagged)
    return InjectionScreen(
        flagged_passages=len(flagged),
        signals=[s for s in SIGNALS if any(s in (f.get("signals") or []) for f in flagged)],
        locations=[
            PassageLocation(**{k: v for k, v in loc.items() if k != "source_file"})
            for loc in locations
        ],
    )


async def _save_upload(file: UploadFile, dest_path: str, max_bytes: int) -> int:
    """
    Streams an upload to disk, enforcing the size cap while reading.

    Reading in fixed-size chunks keeps memory flat regardless of what the
    client sends, and the cap is checked before each write rather than after
    the whole body is buffered.

    Args:
        file: The multipart upload.
        dest_path: Where to write it.
        max_bytes: Maximum accepted size.

    Returns:
        Number of bytes written.

    Raises:
        HTTPException: 413 when over the cap, 400 when empty.
    """
    too_large = HTTPException(
        status_code=413,
        detail=f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit",
    )
    if file.size is not None and file.size > max_bytes:
        raise too_large

    written = 0
    with open(dest_path, "wb") as out:
        while True:
            block = await file.read(_READ_CHUNK_BYTES)
            if not block:
                break
            written += len(block)
            if written > max_bytes:
                raise too_large
            out.write(block)

    if written == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return written


async def _discard_purged_upload(
    deal_id: str, doc_id: str, filename: str, request_id: str, ticket: IngestTicket
) -> None:
    """
    Deletes an upload whose deal was purged (or asked to be) while it was being
    indexed, and refuses it. Registering it would undo the purge: the points
    written after the purge's delete, and the commit, would leave it readable.

    The delete waits for the purge to finish, and is skipped when the same
    bytes were uploaded to the deal again since: those points are that upload's
    now (same ids). A delete that fails is reported as the failure it is (500,
    with the request id): the points it could not delete are still readable.
    """
    await wait_for_purges(deal_id)
    try:
        deleted = await delete_document(
            deal_id, doc_id, unless=lambda: uploaded_again_since_purge(deal_id, doc_id, ticket)
        )
    except Exception as e:
        metrics.INGESTS.inc(outcome="error")
        logger.error(
            "Upload of a purged deal could not be deleted",
            extra={"deal_id": deal_id, "doc_id": doc_id, "request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        audit_log.record(
            "document_discard_failed",
            request_id=request_id, doc_id=doc_id, deal_id=deal_id, file_name=filename,
            reason="deal_purged_during_ingest",
        )
        raise HTTPException(
            status_code=500, detail=public_error_detail("Ingestion", request_id)
        ) from e
    metrics.INGESTS.inc(outcome="rejected")
    logger.warning(
        "Upload discarded: its deal was purged while it was being indexed",
        extra={"deal_id": deal_id, "doc_id": doc_id, "request_id": request_id,
               "points_deleted": deleted},
    )
    audit_log.record(
        "document_discarded",
        request_id=request_id, doc_id=doc_id, deal_id=deal_id, file_name=filename,
        reason="deal_purged_during_ingest",
        # False: the same file was uploaded again since, and its points are kept.
        points_deleted=deleted,
    )
    raise HTTPException(
        status_code=409,
        detail=(
            "A delete of this deal was requested while the file was being indexed, "
            "so the upload was discarded."
        ),
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    http_request: Request,
    file: UploadFile = File(...),
    deal_id: str = Form(..., pattern=DEAL_ID_PATTERN),
    document_category: str | None = Form(None),
    is_current_version: bool = Form(True),
    supersedes_doc_id: str | None = Form(None, max_length=64),
    client: ClientContext = Depends(ingest_rate_limit),
):
    """
    Ingests a document into the RAG pipeline.

    Pipeline:
    0. Access control: non-admin callers may only write into a live sandbox
       deal; per-client rate limit and a pipeline slot bound CPU use
    1. Validate filename and file type
    2. Stream to a temp file under a generated name (size-capped)
    3. index_document(): classify → process → chunk → embed → upsert
       (idempotent: identical bytes in the same deal keep the same doc_id)
    4. Retire the superseded document, if any
    5. Return ingestion summary

    Args:
        file: Uploaded file.
        deal_id: Deal identifier for data isolation.
        document_category: Override category (auto-detected if not provided).
        is_current_version: Whether this is the current version.
        supersedes_doc_id: Doc ID this version supersedes.
        http_request: Incoming request (admin check for the sandbox guard).
        client: Caller context from the ingest rate-limit dependency.

    Returns:
        IngestResponse with doc_id and chunk count.
    """
    await require_sandbox_deal(deal_id, http_request)

    try:
        filename = sanitize_filename(file.filename)
    except IngestionError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from None

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension or '(none)'}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    logger.info(
        "Document ingestion started",
        extra={"deal_id": deal_id, "file_name": filename, "extension": extension},
    )

    # The client's filename never becomes part of a filesystem path; it is kept
    # only as metadata. The generated name preserves the extension, which is
    # all the processors need.
    temp_dir = tempfile.mkdtemp(prefix="manda_ingest_")
    temp_path = os.path.join(temp_dir, f"upload{extension}")

    # Open until the document is registered: a purge of the deal meanwhile
    # marks the ticket, and the upload is then discarded, not registered.
    with watch_for_purge(deal_id) as ticket:
        # Ingestion spends CPU, not Gemini quota: it holds a slot but does not
        # count toward the daily query cap.
        slot = acquire_pipeline_slot(client, count_toward_daily_cap=False)
        try:
            await _save_upload(file, temp_path, _max_upload_bytes())
            _reject_archive_bomb(temp_path, extension)
            # The id index_document will derive (same bytes, same deal), on the
            # ticket before indexing, so a discard of an earlier upload of these
            # bytes can tell that this one is in flight.
            ticket.doc_id = compute_doc_id(
                deal_id, await asyncio.to_thread(compute_content_hash, temp_path)
            )
            result = await index_document(
                temp_path,
                filename,
                deal_id,
                document_category,
                is_current_version=is_current_version,
                supersedes_doc_id=supersedes_doc_id,
            )
        except HTTPException:
            metrics.INGESTS.inc(outcome="rejected")
            raise
        except IngestionError as e:
            metrics.INGESTS.inc(outcome="rejected")
            logger.warning(
                "Document rejected",
                extra={"deal_id": deal_id, "file_name": filename, "reason": str(e)},
            )
            raise HTTPException(status_code=e.status_code, detail=str(e)) from None
        except Exception as e:
            metrics.INGESTS.inc(outcome="error")
            # Internal detail (paths, Qdrant errors) stays in the log.
            logger.error(
                "Document ingestion failed",
                extra={
                    "deal_id": deal_id,
                    "file_name": filename,
                    "error": str(e),
                    "request_id": client.request_id,
                },
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=public_error_detail("Ingestion", client.request_id),
            ) from e
        finally:
            slot.release()
            shutil.rmtree(temp_dir, ignore_errors=True)

        doc_id = result["doc_id"]
        # The pipeline's warnings are what make an upload "partial" (failed or
        # hidden sheets, pages without a text layer, tables not extracted, a
        # superseded re-upload). They are written for the uploader and name only
        # parts of the uploaded file, so they are shown as the reason; without
        # them the consoles could only say "Partly indexed".
        notices: list[str] = list(
            dict.fromkeys(w for w in result.get("warnings") or [] if isinstance(w, str) and w)
        )

        # Retire the superseded document's chunks so retrieval's is_current_version=1
        # filter stops returning them, and any citation that does surface them
        # carries the pointer forward.
        retired = True
        if supersedes_doc_id:
            retirement = await mark_superseded(
                deal_id=deal_id,
                superseded_doc_id=supersedes_doc_id,
                superseded_by=doc_id,
            )
            # The pipeline reports {"ok", "chunks_retired", "error"}. A bare None is
            # the older best-effort contract, which reported nothing either way.
            retired = retirement is None or bool(retirement.get("ok"))
            if not retired:
                reason = retirement.get("error") or "the version store could not be updated"
                notices.append(
                    f"The new document is indexed, but {supersedes_doc_id} is still marked as "
                    f"the current version: {reason}."
                )
                logger.warning(
                    "Supersession not applied",
                    extra={
                        "deal_id": deal_id,
                        "doc_id": doc_id,
                        "supersedes_doc_id": supersedes_doc_id,
                        "reason": reason,
                    },
                )

        # A label, not a failure: it never makes the upload "partial".
        screen = _injection_screen(result)

        # No await between this check and register_document, so a purge cannot
        # slip in after it.
        if ticket.purged:
            await _discard_purged_upload(deal_id, doc_id, filename, client.request_id, ticket)

        # The registry records a supersession only when the index applied it.
        # Recording one that did not happen would make the version browser show the
        # older document as retired while retrieval still serves it as current.
        register_document(
            deal_id=deal_id,
            doc_id=doc_id,
            filename=filename,
            document_category=result["document_category"],
            chunks_created=result["chunks_created"],
            is_current_version=is_current_version,
            supersedes_doc_id=supersedes_doc_id if retired else None,
            risk_signals=result["risk_signals"],
            injection_screen={"flagged_passages": screen.flagged_passages, "signals": screen.signals},
        )

        partial = bool(result["warnings"]) or not retired
        metrics.INGESTS.inc(outcome="partial" if partial else "indexed")

        audit_log.record(
            "document_ingested",
            **{
                "request_id": client.request_id,
                "doc_id": doc_id,
                "deal_id": deal_id,
                "file_name": filename,
                "category": result["document_category"],
                "chunks_created": result["chunks_created"],
                "parent_chunks_created": result["parent_chunks_created"],
                "previously_indexed": result["previously_indexed"],
                "warnings": result["warnings"],
                "risk_signal_types": [s["signal_type"] for s in result["risk_signals"]],
                "suspected_injection_passages": screen.flagged_passages,
                "injection_signals": screen.signals,
                "supersedes_doc_id": supersedes_doc_id or "",
                "supersession_applied": retired if supersedes_doc_id else None,
            },
        )

        return IngestResponse(
            doc_id=doc_id,
            deal_id=deal_id,
            document_category=result["document_category"],
            chunks_created=result["chunks_created"],
            # Some sheets failed, or the older version could not be retired — say
            # so rather than "success".
            status="partial" if partial else "success",
            notices=notices,
            # Passed through, never acted on: superseding stays an explicit choice.
            version_candidates=result.get("version_candidates") or [],
            injection_screen=screen,
        )
