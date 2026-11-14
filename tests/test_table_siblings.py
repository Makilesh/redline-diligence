"""
Table sibling expansion: the exact grid reaches the synthesizer, whole and in order.

A converted table is indexed as several representations that share a table_id,
and a large one splits each representation across chunks: a 90-row, four-period
schedule is 8 narrative + 6 row_by_row + 6 markdown points. The lookup used to be
one scroll capped at 8 points in point-id (random) order with its failures
swallowed. These tests pin what replaced it, against in-memory Qdrant:

- every grid chunk is attached, in table order, right after the chunk that hit;
- the selection does not depend on point ids;
- an oversized grid keeps its header chunk and the part nearest the hit, and says so;
- a failed lookup is marked on the chunk and reported in Agent 3's trace;
- deal, version and PII conditions hold for siblings exactly as for search.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

import src.vector_db.sibling_retrieval as sr
from src.vector_db.collection_manager import setup_collections
from src.vector_db.constants import COLLECTION_NAME, VECTOR_SIZE
from src.vector_db.parent_child_retrieval import expand_context

DEAL = "deal-tables"
TABLE = f"{DEAL}_doc_t001"
ALL_REPRESENTATIONS = ["narrative", "row_by_row", "metrics_summary", "markdown"]


def _chunk(index: int, representation: str | None, table_id: str = TABLE, **overrides) -> dict:
    payload = {
        "chunk_id": f"{table_id}_{index:04d}",
        "deal_id": DEAL,
        "doc_id": "doc",
        "text": f"{representation or 'verbatim'} part {index}",
        "source_file": "schedule.xlsx",
        "is_table": 1,
        "table_id": table_id,
        "chunk_index": index,
        "is_current_version": 1,
        "contains_pii": 0,
    }
    if representation:
        payload["table_representation"] = representation
    payload.update(overrides)
    return payload


def _table(narrative: int, row_by_row: int, markdown: int, metrics: int = 0,
           table_id: str = TABLE) -> list[dict]:
    """Chunks in the order ingestion writes them: one representation after another."""
    layout = [("narrative", narrative), ("row_by_row", row_by_row),
              ("metrics_summary", metrics), ("markdown", markdown)]
    chunks, index = [], 0
    for representation, count in layout:
        for _ in range(count):
            chunks.append(_chunk(index, representation, table_id))
            index += 1
    return chunks


async def _client_with(payloads: list[dict]) -> AsyncQdrantClient:
    client = AsyncQdrantClient(location=":memory:")
    await setup_collections(client)
    unit = [1.0] + [0.0] * (VECTOR_SIZE - 1)
    await client.upsert(COLLECTION_NAME, points=[
        # Random ids: Qdrant returns points in id order, which is what made the
        # old selection arbitrary.
        PointStruct(id=str(uuid.uuid4()), payload=p, vector={"dense": unit}) for p in payloads
    ])
    return client


def _prose(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "deal_id": DEAL, "text": "prose", "is_table": 0}


def _by_representation(chunks: list[dict], representation: str) -> list[int]:
    return [c["chunk_index"] for c in chunks if c.get("table_representation") == representation]


class TestLargeTable:
    @pytest.mark.asyncio
    async def test_the_whole_grid_follows_the_hit_in_table_order(self):
        table = _table(narrative=8, row_by_row=6, markdown=6)
        client = await _client_with(table)
        hit = dict(table[3])  # a narrative chunk

        out = await expand_context(
            [_prose("p1"), hit, _prose("p2")], client=client, deal_id=DEAL, include_parents=False)

        ids = [c["chunk_id"] for c in out]
        assert ids[:2] == ["p1", hit["chunk_id"]]
        assert ids[-1] == "p2", "siblings belong next to their table, not at the end"
        siblings = out[2:-1]
        markdown = [c for c in table if c.get("table_representation") == "markdown"]
        assert [c["chunk_id"] for c in siblings[:6]] == [c["chunk_id"] for c in markdown]
        # 6 grid chunks leave room for two more under the per-table cap.
        assert len(siblings) == sr.MAX_SIBLINGS_PER_TABLE
        assert _by_representation(siblings[6:], "row_by_row") == [8, 9]

    @pytest.mark.asyncio
    async def test_selection_does_not_depend_on_point_ids(self):
        table = _table(narrative=8, row_by_row=6, markdown=6)
        runs = []
        for _ in range(3):
            client = await _client_with(table)
            out = await expand_context([dict(table[0])], client=client, deal_id=DEAL,
                                       include_parents=False)
            runs.append([c["chunk_id"] for c in out])

        assert runs[0] == runs[1] == runs[2]

    @pytest.mark.asyncio
    async def test_two_hits_in_one_table_share_one_set_of_siblings(self):
        table = _table(narrative=2, row_by_row=2, markdown=2, metrics=1)
        client = await _client_with(table)

        out = await expand_context([dict(table[0]), _prose("p1"), dict(table[5])],
                                   client=client, deal_id=DEAL, include_parents=False)

        ids = [c["chunk_id"] for c in out]
        assert len(ids) == len(set(ids))
        assert {c["chunk_id"] for c in table} <= set(ids)
        # Grid first after the first hit; the second hit is itself a grid chunk.
        assert out[1]["table_representation"] == "markdown"
        assert out[2]["table_representation"] == "metrics_summary"

    @pytest.mark.asyncio
    async def test_verbatim_table_parts_are_restored_in_order(self):
        """A table that was not converted has no representations, only parts."""
        parts = [_chunk(i, None) for i in range(5)]
        client = await _client_with(parts)

        out = await expand_context([dict(parts[3])], client=client, deal_id=DEAL,
                                   include_parents=False)

        assert [c["chunk_index"] for c in out] == [3, 0, 1, 2, 4]


class TestOversizedGrid:
    def test_header_chunk_and_the_part_nearest_the_hit_are_kept(self):
        table = _table(narrative=8, row_by_row=0, markdown=30)
        hit = table[4]  # 5th of 8 narrative chunks: ~4/7 of the way down the table
        grid = [c for c in table if c.get("table_representation") == "markdown"]

        selected, omitted = sr.select_siblings(table, [hit], ALL_REPRESENTATIONS)

        positions = [grid.index(c) for c in selected if c.get("table_representation") == "markdown"]
        assert len(positions) == sr.MAX_GRID_CHUNKS_PER_TABLE
        assert omitted == 30 - sr.MAX_GRID_CHUNKS_PER_TABLE
        assert positions == sorted(positions)
        assert positions[0] == 0, "only the first grid chunk carries the header row"
        target = round(4 / 7 * 29)
        assert target in positions
        assert max(abs(p - target) for p in positions[1:]) <= sr.MAX_GRID_CHUNKS_PER_TABLE // 2 + 1
        assert all(c.get("table_representation") == "markdown" for c in selected)

    @pytest.mark.asyncio
    async def test_omission_is_recorded_on_the_hit(self):
        table = _table(narrative=1, row_by_row=0, markdown=20)
        client = await _client_with(table)

        out = await expand_context([dict(table[0])], client=client, deal_id=DEAL,
                                   include_parents=False)

        assert out[0]["table_grid_chunks_omitted"] == 20 - sr.MAX_GRID_CHUNKS_PER_TABLE
        assert len(out) == 1 + sr.MAX_GRID_CHUNKS_PER_TABLE


class TestRequiredRepresentations:
    def test_only_requested_representations_are_added(self):
        """The argument used to be logged and otherwise ignored."""
        table = _table(narrative=2, row_by_row=2, markdown=2, metrics=1)

        selected, _ = sr.select_siblings(table, [table[0]], ["markdown", "metrics_summary"])

        assert {c["table_representation"] for c in selected} == {"markdown", "metrics_summary"}

    @pytest.mark.asyncio
    async def test_small_table_still_gets_every_representation(self):
        table = _table(narrative=1, row_by_row=1, markdown=1, metrics=1)
        client = await _client_with(table)

        out = await expand_context([dict(table[0])], client=client, deal_id=DEAL,
                                   include_parents=False)

        assert [c["table_representation"] for c in out] == [
            "narrative", "markdown", "metrics_summary", "row_by_row"]


class TestFailureIsVisible:
    @pytest.mark.asyncio
    async def test_failed_lookup_marks_the_chunk_and_keeps_the_context(self, monkeypatch):
        monkeypatch.setattr(sr, "QDRANT_BASE_DELAY_S", 0.0)
        table = _table(narrative=1, row_by_row=1, markdown=1)
        calls = []

        class Down:
            async def scroll(self, **kwargs):
                calls.append(kwargs)
                raise ConnectionError("qdrant down")

        chunks = [_prose("p1"), dict(table[0])]
        out = await sr.fetch_table_siblings(chunks, ALL_REPRESENTATIONS, Down(), deal_id=DEAL)

        assert [c["chunk_id"] for c in out] == ["p1", table[0]["chunk_id"]]
        assert out[1]["table_siblings_missing"] is True
        assert "table_siblings_missing" not in out[0]
        assert len(calls) == sr.QDRANT_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_agent_3_reports_it_in_the_trace(self, monkeypatch):
        import numpy as np

        import src.agents.retrieval_executor as rx

        async def embed(texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

        async def search(**kwargs):
            return [], []

        async def expand(**kwargs):
            return [{"chunk_id": "c1", "table_id": "t9", "table_siblings_missing": True}]

        monkeypatch.setattr(rx, "embed_texts_async", embed)
        monkeypatch.setattr(rx, "compute_sparse_bm25", lambda text: None)
        monkeypatch.setattr(rx, "hybrid_search", search)
        monkeypatch.setattr(rx, "expand_context", expand)

        out = await rx.retrieval_executor_node({
            "current_query": "q", "query_type": "financial", "parsed_intent": {},
            "deal_id": DEAL, "include_pii": False, "rewrite_iteration": 0,
            "retrieval_config": {}, "extracted_filters": {},
        })

        assert out["agent_trace"][0]["table_siblings_missing"] == ["t9"]


class TestIsolation:
    @pytest.mark.asyncio
    async def test_siblings_obey_deal_version_and_pii_filters(self):
        table = _table(narrative=1, row_by_row=0, markdown=1)
        outsiders = [
            _chunk(10, "markdown", deal_id="another-deal", chunk_id="other-deal"),
            _chunk(11, "markdown", is_current_version=0, chunk_id="superseded"),
            _chunk(12, "markdown", contains_pii=1, chunk_id="pii"),
        ]
        client = await _client_with(table + outsiders)
        hit = dict(table[0])

        default = await expand_context([dict(hit)], client=client, deal_id=DEAL,
                                       include_parents=False)
        with_pii = await expand_context([dict(hit)], client=client, deal_id=DEAL,
                                        include_parents=False, include_pii=True)

        assert [c["chunk_id"] for c in default] == [hit["chunk_id"], table[1]["chunk_id"]]
        assert {c["chunk_id"] for c in with_pii} == {hit["chunk_id"], table[1]["chunk_id"], "pii"}


# ==============================================================================
# Section tables: a retrieved section brings the table printed inside it
# ==============================================================================

SECTION = "HART-SCOTT-RODINO (UNITED STATES)"


def _prose_in(section: str, index: int, doc_id: str = "doc", **overrides) -> dict:
    payload = {
        "chunk_id": f"{doc_id}_prose_{index:04d}", "deal_id": DEAL, "doc_id": doc_id,
        "text": f"prose {index}", "is_table": 0, "section_heading": section,
        "chunk_index": index, "is_current_version": 1, "contains_pii": 0,
        "parent_chunk_id": f"{doc_id}_p0",
    }
    payload.update(overrides)
    return payload


def _section_table(section: str, index: int, representation: str | None = None,
                   table_id: str = "t-sec", doc_id: str = "doc", **overrides) -> dict:
    chunk = _chunk(index, representation, table_id=table_id, section_heading=section,
                   doc_id=doc_id)
    chunk["chunk_id"] = f"{doc_id}_{table_id}_{index:04d}"
    chunk.update(overrides)
    return chunk


class TestSectionTables:
    @pytest.mark.asyncio
    async def test_the_table_inside_a_retrieved_section_follows_its_prose(self):
        """
        The case from the eval: the HSR paragraph ranks, the fee / waiting-period
        table printed under it does not, and the parent section holds prose only.
        """
        prose = _prose_in(SECTION, 1)
        table = _section_table(SECTION, 2)
        elsewhere = _section_table("FOREIGN MERGER CONTROL", 4, table_id="t-other")
        client = await _client_with([prose, table, elsewhere])

        out = await expand_context([_prose("p1"), dict(prose), _prose("p2")],
                                   client=client, deal_id=DEAL)

        assert [c["chunk_id"] for c in out] == ["p1", prose["chunk_id"], table["chunk_id"], "p2"]

    @pytest.mark.asyncio
    async def test_only_the_grid_nearest_the_prose_and_never_twice(self):
        prose = _prose_in(SECTION, 10)
        grid = [_section_table(SECTION, i, "markdown") for i in (11, 12, 13, 14, 15)]
        narrative = _section_table(SECTION, 16, "narrative")
        client = await _client_with([prose, *grid, narrative])
        already_retrieved = dict(grid[1])

        out = await expand_context([dict(prose), already_retrieved], client=client, deal_id=DEAL,
                                   include_siblings=False)

        ids = [c["chunk_id"] for c in out]
        assert len(ids) == len(set(ids))
        attached = out[1:-1]
        assert len(attached) == sr.MAX_SECTION_TABLE_CHUNKS
        assert [c["chunk_index"] for c in attached] == [11, 13, 14]  # 12 is already there
        assert all(c["table_representation"] == "markdown" for c in attached)

    @pytest.mark.asyncio
    async def test_an_attached_table_does_not_pull_in_its_other_representations(self):
        prose = _prose_in(SECTION, 1)
        table = [_section_table(SECTION, 2, "narrative"), _section_table(SECTION, 3, "row_by_row"),
                 _section_table(SECTION, 4, "markdown")]
        client = await _client_with([prose, *table])

        out = await expand_context([dict(prose)], client=client, deal_id=DEAL)

        assert [c.get("table_representation") for c in out[1:]] == ["markdown"]

    @pytest.mark.asyncio
    async def test_same_heading_in_another_document_is_not_the_same_section(self):
        prose = _prose_in(SECTION, 1)
        foreign = _section_table(SECTION, 2, doc_id="another-doc")
        client = await _client_with([prose, foreign])

        out = await expand_context([dict(prose)], client=client, deal_id=DEAL)

        assert [c["chunk_id"] for c in out] == [prose["chunk_id"]]

    @pytest.mark.asyncio
    async def test_section_tables_obey_deal_version_and_pii_filters(self):
        prose = _prose_in(SECTION, 1)
        outsiders = [
            _section_table(SECTION, 2, table_id="t-deal", deal_id="another-deal"),
            _section_table(SECTION, 3, table_id="t-old", is_current_version=0),
            _section_table(SECTION, 4, table_id="t-pii", contains_pii=1),
        ]
        client = await _client_with([prose, *outsiders])

        default = await expand_context([dict(prose)], client=client, deal_id=DEAL)
        with_pii = await expand_context([dict(prose)], client=client, deal_id=DEAL,
                                        include_pii=True)

        assert [c["chunk_id"] for c in default] == [prose["chunk_id"]]
        assert [c["chunk_id"] for c in with_pii] == [prose["chunk_id"], outsiders[2]["chunk_id"]]

    @pytest.mark.asyncio
    async def test_off_when_parent_expansion_is_off(self):
        prose = _prose_in(SECTION, 1)
        client = await _client_with([prose, _section_table(SECTION, 2)])

        out = await expand_context([dict(prose)], client=client, deal_id=DEAL,
                                   include_parents=False)

        assert len(out) == 1

    @pytest.mark.asyncio
    async def test_failed_lookup_is_marked_not_raised(self, monkeypatch):
        monkeypatch.setattr(sr, "QDRANT_BASE_DELAY_S", 0.0)

        class Down:
            async def scroll(self, **kwargs):
                raise ConnectionError("qdrant down")

        chunks = [_prose_in(SECTION, 1), _prose("p1")]
        out = await sr.fetch_section_tables(chunks, Down(), deal_id=DEAL)

        assert [c["chunk_id"] for c in out] == [chunks[0]["chunk_id"], "p1"]
        assert out[0]["section_tables_missing"] is True
