"""
Ingestion errors whose messages are safe to return to an API client.

They live apart from ingest_pipeline so the format processors can raise a
specific, client-readable reason ("the PDF is password-protected") instead of
letting a library exception surface as "could not be parsed". ingest_pipeline
re-exports every name, so existing imports keep working.
"""


class IngestionError(Exception):
    """An ingestion failure whose message is safe to show a client."""

    status_code = 422


class UnsupportedFileTypeError(IngestionError):
    """The file extension is not one the pipeline can process."""

    status_code = 400


class UnsafeDocumentError(IngestionError):
    """The file exceeds a safety limit (e.g. a decompression bomb)."""

    status_code = 413


class DocumentExtractionError(IngestionError):
    """The file could not be parsed as the format its extension claims."""

    status_code = 422


class NoExtractableContentError(IngestionError):
    """Parsing succeeded but produced nothing to index."""

    status_code = 422
