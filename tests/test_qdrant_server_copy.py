"""eval/run_qdrant_server_latency.copy_store moves every point, vectors and payload, between clients."""

import asyncio

from qdrant_client import AsyncQdrantClient, models

from eval.run_qdrant_server_latency import copy_store
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME


async def _copy_round_trip():
    src = AsyncQdrantClient(location=":memory:")
    dst = AsyncQdrantClient(location=":memory:")
    await setup_collections(src, COLLECTION_NAME, PARENT_COLLECTION_NAME)
    info = await src.get_collection(COLLECTION_NAME)
    dense_name, dense = next(iter(info.config.params.vectors.items()))
    sparse_name = next(iter(info.config.params.sparse_vectors))
    points = [
        models.PointStruct(
            id=i,
            vector={dense_name: [float(i + 1)] + [0.0] * (dense.size - 1),
                    sparse_name: models.SparseVector(indices=[i, 100 + i], values=[0.5, 1.5])},
            payload={"chunk_id": f"c{i}", "deal_id": "d"},
        )
        for i in range(600)  # more than one scroll page
    ]
    await src.upsert(COLLECTION_NAME, points=points, wait=True)
    counts = await copy_store(src, dst)
    got = {}
    offset = None
    while True:
        page, offset = await dst.scroll(COLLECTION_NAME, limit=1000, offset=offset,
                                        with_payload=True, with_vectors=True)
        got.update({p.id: p for p in page})
        if offset is None:
            break
    return counts, got, dense_name, sparse_name


def test_copy_store_keeps_every_point_with_vectors_and_payload():
    counts, got, dense_name, sparse_name = asyncio.run(_copy_round_trip())
    assert counts[COLLECTION_NAME] == 600 and len(got) == 600
    p = got[7]
    assert p.payload == {"chunk_id": "c7", "deal_id": "d"}
    assert p.vector[sparse_name].indices == [7, 107]
    assert p.vector[sparse_name].values == [0.5, 1.5]
    assert len(p.vector[dense_name]) > 1
