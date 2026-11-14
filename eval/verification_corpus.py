"""
Evidence contexts for the verification benchmark — built offline, no models.

The verifier is measured on what it is given in production: ~10 retrieved child
chunks, each carrying its ~2k-token parent section as `parent_text`. Running the
real retriever here would put bge-m3 and the reranker (and a GPU) between a
verification change and its measurement, so the context is assembled
deterministically instead:

1. `data/sample_deal/*.txt` goes through the production chunker
   (`extract_and_chunk`, the same call `POST /ingest` makes, with the Laya
   decision model off so nothing is loaded) and each child gets its parent text
   attached exactly as `expand_context` does.
2. For one benchmark item, the chunk holding the item's evidence excerpt is
   placed first (retrieval found it), and the rest of the context is filled
   with the chunks sharing the most vocabulary with the claim — on-topic
   distractors, which is what a reranker returns alongside the right passage.
   Unsupported items have no evidence chunk and get distractors only.

Ranking is a plain token-overlap count with ties broken by chunk id, so a
context never changes between runs or machines.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "sample_deal"
DEAL_ID = "aurora_vertex_2024"
CONTEXT_SIZE = 10

# Mirrors CATEGORY_OVERRIDES in eval/corpus.py and run_demo.py, so the chunks
# are the ones the retrieval harness and the demo index.
CATEGORY_OVERRIDES = {
    "board_deck_strategic_review_mar2024.txt": "board",
    "regulatory_and_data_privacy_memo.txt": "regulatory",
    "employment_and_retention_agreements.txt": "legal",
}

_WORD = re.compile(r"[a-z][a-z\-]{2,}|\d+(?:\.\d+)?")
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "has",
    "had", "have", "its", "which", "into", "than", "their", "also", "been", "not",
    "company", "aurora",
}


def normalise(text: str) -> str:
    """Collapses whitespace, so an excerpt matches across hard-wrapped lines."""
    return " ".join((text or "").split())


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall((text or "").lower().replace(",", "")) if t not in _STOP}


@functools.lru_cache(maxsize=1)
def corpus_chunks() -> tuple[dict, ...]:
    """
    Every child chunk of the sample data room, production-shaped.

    Returns:
        Chunk payloads in (file, chunk index) order, `parent_text` attached
        where the chunk has a parent.
    """
    # Chunking calls the risk/PII annotators; the Laya model would add minutes
    # and a GPU for fields verification never reads.
    previous = os.environ.get("LAYA_ENABLED")
    os.environ["LAYA_ENABLED"] = "0"
    try:
        from src.data_processing.ingest_pipeline import extract_and_chunk

        chunks: list[dict] = []
        for path in sorted(CORPUS_DIR.glob("*.txt")):
            doc = extract_and_chunk(
                str(path), path.name, DEAL_ID, CATEGORY_OVERRIDES.get(path.name)
            )
            parents = {p["chunk_id"]: p["text"] for p in doc.parents}
            for chunk in doc.chunks:
                chunk = dict(chunk)
                parent = parents.get(chunk.get("parent_chunk_id") or "")
                if parent:
                    chunk["parent_text"] = parent
                chunks.append(chunk)
        return tuple(chunks)
    finally:
        if previous is None:
            os.environ.pop("LAYA_ENABLED", None)
        else:
            os.environ["LAYA_ENABLED"] = previous


def corpus_text(source: str) -> str:
    """Whitespace-normalised text of one corpus file."""
    return normalise((CORPUS_DIR / source).read_text(encoding="utf-8"))


def evidence_chunks(source: str, evidence: str) -> list[dict]:
    """
    Chunks of `source` that contain the evidence excerpt.

    A child whose own text holds the excerpt is preferred; failing that, any
    child whose parent section does (an excerpt can straddle two children).

    Args:
        source: Corpus file name.
        evidence: Verbatim excerpt (whitespace-insensitive).

    Returns:
        Matching chunks, own-text matches first.
    """
    needle = normalise(evidence)
    own, via_parent = [], []
    for chunk in corpus_chunks():
        if chunk.get("source_file") != source:
            continue
        if needle in normalise(chunk.get("text", "")):
            own.append(chunk)
        elif needle in normalise(chunk.get("parent_text", "")):
            via_parent.append(chunk)
    return own or via_parent[:1]


def build_context(
    claim: str,
    evidence: list[tuple[str, str]] | None = None,
    size: int = CONTEXT_SIZE,
) -> list[dict]:
    """
    The retrieved context a claim is verified against.

    Args:
        claim: Claim text (drives the choice of distractors).
        evidence: (source file, verbatim excerpt) pairs that must be present.
        size: Number of chunks, as production's final_top_k.

    Returns:
        Chunk dicts, evidence chunks first, then distractors by vocabulary
        overlap with the claim.

    Raises:
        ValueError: An evidence excerpt is not in the named file's chunks.
    """
    context: list[dict] = []
    seen: set[str] = set()
    for source, excerpt in evidence or []:
        found = evidence_chunks(source, excerpt)
        if not found:
            raise ValueError(f"evidence not found in chunks of {source}: {excerpt[:80]!r}")
        chunk = found[0]
        if chunk["chunk_id"] not in seen:
            seen.add(chunk["chunk_id"])
            context.append(chunk)

    claim_tokens = _tokens(claim)
    ranked = sorted(
        (c for c in corpus_chunks() if c["chunk_id"] not in seen),
        key=lambda c: (-len(claim_tokens & _tokens(c.get("text", ""))), c["chunk_id"]),
    )
    context.extend(ranked[: max(0, size - len(context))])
    return context


def file_context(sources: list[str]) -> list[dict]:
    """Every chunk of the named files — the context for a recorded answer when
    its real retrieval context was not saved (see run_verification_eval)."""
    wanted = set(sources)
    return [c for c in corpus_chunks() if c.get("source_file") in wanted]
