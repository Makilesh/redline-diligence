"""
A table row's cells are read in the table's left-to-right order, whatever order the index
returned its `values` in (Decision 105).

`values` is a JSON object, and a Qdrant server reached over gRPC — the app's client whenever no
API key is set — does not keep a JSON object's key order (protobuf maps carry none). In the
re-ingestion rehearsal every FinanceBench chunk came back with its columns shuffled, and the
registry, which reads a figure's period from neighbouring columns, moved the tie-out by 45
figures without a period and 42 single-source ones. Local-mode stores, where every benchmark
runs, keep the order, so nothing measured it. Rows now carry `columns`, a list.

The fixture is one chunk of JPMorgan's 2023 Q2 10-Q (FinanceBench) as the server returned it
in natural order, chosen because shuffling its columns changes what the registry records.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from src.utils.numerical_registry import NumericalRegistry, row_cells
from tests.test_table_conversion import MIXED_UNIT_ROWS, _structured

FIXTURE = Path(__file__).parent / "fixtures" / "table_column_order_chunk.json"


def _chunk() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _entries(chunk: dict) -> list[tuple]:
    registry, _ = NumericalRegistry.from_documents([copy.deepcopy(chunk)])
    return sorted((metric, period, round(e.normalized_value, 4))
                  for (metric, period), entries in registry._entries.items() for e in entries)


def _shuffled(chunk: dict, *, keep_columns: bool, seed: int = 7) -> dict:
    """The chunk as a gRPC round trip can return it: every row's values in another order."""
    out = copy.deepcopy(chunk)
    rng = random.Random(seed)
    for row in out["structured_rows"]:
        items = list((row.get("values") or {}).items())
        natural = [k for k, _ in items]
        while len(items) > 1 and [k for k, _ in items] == natural:
            rng.shuffle(items)
        row["values"] = dict(items)
        if keep_columns:
            row["columns"] = natural
        else:
            row.pop("columns", None)
    return out


def _renamed(chunk: dict) -> dict:
    """Header names that sort in no useful order, so only `columns` can say the order."""
    out = copy.deepcopy(chunk)
    for row in out["structured_rows"]:
        names = {k: f"{k.replace('Column ', 'Col-')}-h" for k in (row.get("values") or {})}
        row["values"] = {names[k]: v for k, v in (row.get("values") or {}).items()}
    return out


def test_the_fixture_depends_on_column_order(monkeypatch):
    import src.utils.numerical_registry as nr

    natural = _chunk()
    # Read the old way (cells in stored order), a shuffled row records other figures.
    monkeypatch.setattr(nr, "row_cells", lambda row: list((row.get("values") or {}).items()))
    assert _entries(_shuffled(natural, keep_columns=False)) != _entries(natural)
    monkeypatch.undo()
    # Header names carry no order of their own: without `columns`, shuffling still changes them.
    renamed = _renamed(natural)
    assert _entries(_shuffled(renamed, keep_columns=False)) != _entries(renamed)


def test_a_shuffled_row_with_its_columns_reads_as_the_natural_one():
    natural = _renamed(_chunk())
    for seed in range(5):
        assert _entries(_shuffled(natural, keep_columns=True, seed=seed)) == _entries(natural)


def test_placeholder_columns_without_an_order_sort_by_number():
    natural = _chunk()  # its columns are "Column 2" .. "Column 7", written before `columns`
    for row in natural["structured_rows"]:
        row.pop("columns", None)
    for seed in range(5):
        assert _entries(_shuffled(natural, keep_columns=False, seed=seed)) == _entries(natural)


@pytest.mark.parametrize("row, expected", [
    ({"values": {"b": 1, "a": 2}, "columns": ["a", "b"]}, ["a", "b"]),
    ({"values": {"Column 10": 1, "Column 2": 2, "Column 3": 3}}, ["Column 2", "Column 3", "Column 10"]),
    ({"values": {"FY2023": 1, "FY2022": 2}}, ["FY2023", "FY2022"]),  # header names: stored order
    ({"values": {"b": 1, "a": 2}, "columns": ["a"]}, ["b", "a"]),  # a stale order is not trusted
    ({"values": {"Column 3": 1, "Total": 2, "Column 2": 3}}, ["Column 3", "Total", "Column 2"]),  # mixed
    ({}, []),
])
def test_row_cells(row, expected):
    assert [column for column, _ in row_cells(row)] == expected


def test_the_converter_writes_each_rows_column_order():
    rows = _structured(MIXED_UNIT_ROWS)
    assert rows
    for row in rows.values():
        assert row["columns"] == list(row["values"])
