"""
Builds the evaluation index: the sample data room in an in-memory Qdrant.

Indexing goes through src.data_processing.ingest_pipeline.index_document with
its default models (bge-m3 dense, FastEmbed BM25 sparse) — the same path as
POST /ingest — so what is measured is what is served. Nothing touches the Docker
Qdrant or qdrant_local_db.

Retrieval code reaches Qdrant through get_qdrant_client(), a module-level
singleton. use_client() points that singleton at the in-memory client for the
duration of a run; no production code changes are needed.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from src.data_processing.ingest_pipeline import index_document
from src.vector_db import qdrant_client as qdrant_client_module
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "sample_deal"
GOLDEN_SET_PATH = PROJECT_ROOT / "tests" / "golden_qa_set.json"
DEAL_ID = "aurora_vertex_2024"

# Mirrors CATEGORY_OVERRIDES in run_demo.py: the three documents whose
# auto-classification is ambiguous are pinned; the rest are classified at
# ingestion exactly as an upload would be.
CATEGORY_OVERRIDES = {
    "board_deck_strategic_review_mar2024.txt": "board",
    "regulatory_and_data_privacy_memo.txt": "regulatory",
    "employment_and_retention_agreements.txt": "legal",
}


# ------------------------------------------------------------------------------
# Distractor documents (--distractors N)
#
# The sample data room is 9 documents and ~110 chunks, so every ranking stage is
# choosing among a handful of candidates and Qdrant never has to discriminate.
# The distractor mode adds real commercial contracts to the SAME deal, so the
# golden questions have to be answered out of a few thousand chunks of
# plausible-looking legal text. The contracts are CUAD v1 (510 contracts, The
# Atticus Project, CC BY 4.0), pinned to one dataset revision, downloaded on
# first use into a cache outside the repository and never committed.
# ------------------------------------------------------------------------------
DISTRACTOR_REPO = "theatticusproject/cuad"
DISTRACTOR_REVISION = "a3c393f5d103fd0c516374e4fdff676c8176dcb1"
DISTRACTOR_SUBDIR = "CUAD_v1/full_contract_txt"
DISTRACTOR_LICENCE = "CC BY 4.0 (The Atticus Project, CUAD v1)"
DISTRACTOR_PREFIX = "distractor_"
DISTRACTOR_CACHE_ENV = "REDLINE_EVAL_CACHE"


def is_distractor(chunk: dict) -> bool:
    """True for a chunk that came from a distractor document."""
    return str(chunk.get("source_file") or "").startswith(DISTRACTOR_PREFIX)


def distractor_cache_dir() -> Path:
    """Where downloaded distractor documents are kept (never inside the repo)."""
    import os

    configured = os.getenv(DISTRACTOR_CACHE_ENV)
    return Path(configured) if configured else Path.home() / ".cache" / "redline-eval"


def fetch_distractors(count: int, cache_dir: Path | None = None) -> list[Path]:
    """
    Downloads the first `count` CUAD contracts (by file name) and returns their paths.

    Deterministic: the dataset revision is pinned and files are taken in sorted
    order, so `--distractors 100` is the same 100 contracts on every machine.
    Files already in the cache are not downloaded again.

    Args:
        count: Number of contracts.
        cache_dir: Download cache; defaults to distractor_cache_dir().

    Returns:
        Paths of the contract text files, in sorted order.

    Raises:
        RuntimeError: If fewer than `count` contracts are available.
    """
    from huggingface_hub import HfApi, hf_hub_download

    cache_dir = cache_dir or distractor_cache_dir()
    names = sorted(
        name for name in HfApi().list_repo_files(
            DISTRACTOR_REPO, repo_type="dataset", revision=DISTRACTOR_REVISION)
        if name.startswith(DISTRACTOR_SUBDIR + "/") and name.lower().endswith(".txt")
    )
    if len(names) < count:
        raise RuntimeError(f"only {len(names)} distractor documents available, {count} requested")
    return [
        Path(hf_hub_download(
            DISTRACTOR_REPO, name, repo_type="dataset", revision=DISTRACTOR_REVISION,
            local_dir=cache_dir / "cuad"))
        for name in names[:count]
    ]


async def index_distractors(client: AsyncQdrantClient, paths: list[Path]) -> list[dict]:
    """
    Indexes distractor documents into the evaluation deal through index_document.

    They go into the same deal as the sample data room on purpose: a distractor
    in another deal is removed by the deal filter before ranking and would only
    test isolation. The Laya ingest decisions (risk signals, PII second opinion)
    are skipped for these documents: they do not affect what is retrieved and
    would add minutes per run. Rule-based PII flagging still applies, so a
    flagged distractor chunk is unreachable exactly as in production.

    Args:
        client: The evaluation Qdrant client.
        paths: Distractor text files.

    Returns:
        Per-document index_document results.
    """
    import os

    previous = os.environ.get("LAYA_ENABLED")
    os.environ["LAYA_ENABLED"] = "0"
    try:
        return [
            await index_document(
                str(path), f"{DISTRACTOR_PREFIX}{number:04d}.txt", DEAL_ID, "legal", client=client)
            for number, path in enumerate(paths, start=1)
        ]
    finally:
        if previous is None:
            os.environ.pop("LAYA_ENABLED", None)
        else:
            os.environ["LAYA_ENABLED"] = previous


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> dict:
    """Loads tests/golden_qa_set.json."""
    return json.loads(path.read_text(encoding="utf-8"))


async def build_index(corpus_dir: Path = CORPUS_DIR) -> tuple[AsyncQdrantClient, list[dict]]:
    """
    Indexes every .txt in the corpus into a fresh in-memory Qdrant.

    Args:
        corpus_dir: Directory holding the sample data room.

    Returns:
        (client, per-document index_document results).
    """
    client = AsyncQdrantClient(location=":memory:")
    await setup_collections(client, COLLECTION_NAME, PARENT_COLLECTION_NAME)
    documents = []
    for path in sorted(corpus_dir.glob("*.txt")):
        documents.append(await index_document(
            str(path),
            path.name,
            DEAL_ID,
            CATEGORY_OVERRIDES.get(path.name),
            client=client,
        ))
    return client, documents


async def all_chunks(client: AsyncQdrantClient) -> list[dict]:
    """Every child-chunk payload in the index."""
    payloads: list[dict] = []
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend(p.payload for p in points)
        if offset is None:
            return payloads


def is_retrievable(chunk: dict) -> bool:
    """
    Whether production retrieval can ever return this chunk.

    Mirrors the non-negotiable conditions of hybrid_search._build_filter with
    include_pii=False: the deal, the current version, and no PII flag. A chunk
    failing these is unreachable by policy, so it is left out of the relevance
    labels rather than counted as a retrieval miss.

    Distractor chunks are retrievable too; callers that build relevance labels
    or look for facts exclude them with is_distractor().
    """
    return (
        chunk.get("deal_id") == DEAL_ID
        and chunk.get("is_current_version") == 1
        and chunk.get("contains_pii") == 0
    )


@contextlib.contextmanager
def use_client(client: AsyncQdrantClient):
    """Points the get_qdrant_client() singleton at `client`, restoring it after."""
    previous = qdrant_client_module._qdrant_client
    qdrant_client_module._qdrant_client = client
    try:
        yield client
    finally:
        qdrant_client_module._qdrant_client = previous
