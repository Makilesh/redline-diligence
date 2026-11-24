"""
The opening text of a PDF, DOCX or PPTX, for content-based classification.

Run in the parser child (parse_isolation.content_sample, Decision 121): reading the
sample opens the whole file with the same parsers as the parse itself — python-docx
builds the full package, PyMuPDF opens the PDF — so a file that hangs or crashes a
parser did so in the API's process before the isolated parse began. This module
imports only the parsers, so the child stays light (ingest_pipeline takes ~6 s to
import).
"""

from __future__ import annotations

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

SAMPLE_CHARS = 2000


def sample(file_path: str, extension: str, chars: int = SAMPLE_CHARS) -> str:
    """
    First ~chars characters of text of a PDF, DOCX or PPTX.

    Best effort: classification falls back to filename and extension heuristics,
    so a parser's error returns "". (A parser that kills or hangs the child is
    handled by the caller: the upload fails, as for the parse.)

    Args:
        file_path: Path to the file.
        extension: ".pdf", ".docx" or ".pptx"; anything else returns "".
        chars: Characters to return at most.

    Returns:
        Text sample (possibly empty).
    """
    try:
        if extension == ".pdf":
            import fitz

            with fitz.open(file_path) as pdf:
                text = ""
                for page in pdf:
                    text += page.get_text("text")
                    if len(text) >= chars:
                        break
                return text[:chars]
        if extension == ".docx":
            import docx

            document = docx.Document(file_path)
            return "\n".join(p.text for p in document.paragraphs[:80])[:chars]
        if extension == ".pptx":
            from pptx import Presentation

            parts = []
            for slide in list(Presentation(file_path).slides)[:5]:
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        parts.append(shape.text_frame.text)
            return "\n".join(parts)[:chars]
    except Exception as e:  # noqa: BLE001 - best effort, as documented
        logger.warning("Content sample extraction failed", extra={"error": str(e)})
    return ""
