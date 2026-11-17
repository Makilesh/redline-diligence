"""
The held-out LegalBench-RAG set of Decision 80: the full benchmark's documents
that no mini query uses, split by document (stratified, seeded) into fresh-dev
and a sealed fresh-test, with a seeded per-document query sample.
"""

import asyncio
import json
import unicodedata
import zipfile
from pathlib import Path

import pytest

from eval import realdocs_data as data


def _tests(subset: str, n_docs: int, per_doc: int, extra_snippet: str | None = None) -> list[dict]:
    out = []
    for d in range(n_docs):
        for i in range(per_doc):
            snippets = [{"file_path": f"{subset}/doc{d:02d}.txt", "span": [0, 3], "answer": "abc"}]
            if extra_snippet and d == 0 and i == 0:
                snippets.append({"file_path": extra_snippet, "span": [0, 3], "answer": "abc"})
            out.append({"query": f"{subset} q{d}-{i}", "snippets": snippets})
    return out


def _queries(per_subset: dict[str, int], per_doc: int = 4) -> list[data.SpanQuery]:
    tests = {s: _tests(s, n, per_doc) for s, n in per_subset.items()}
    return data.legalbench_full_queries(tests)


# ── full benchmark ────────────────────────────────────────────────────────────


def test_full_query_ids_follow_the_benchmark_file_order():
    queries = _queries({"maud": 2, "cuad": 1}, per_doc=3)
    assert [q.id for q in queries if q.subset == "cuad"] == ["cuad_f0000", "cuad_f0001", "cuad_f0002"]
    # Sub-corpora in LEGALBENCH_SUBSETS order, whatever the dict order.
    assert [q.subset for q in queries][:1] == ["maud"]
    assert len(queries) == 9


def test_a_decomposed_unicode_file_name_resolves_to_the_corpus_file():
    # cuad.json spells one file name decomposed (E + U+0301); the zip has it composed.
    composed = unicodedata.normalize("NFC", "cuad/LECLANCHÉ S.A.txt")
    decomposed = "cuad/LECLANCHÉ S.A.txt"
    assert composed != decomposed
    tests = {"cuad": [{"query": "q", "snippets": [{"file_path": decomposed, "span": [1, 2]}]}]}
    (query,) = data.legalbench_full_queries(tests, [composed, "cuad/other.txt"])
    assert query.document == composed


# ── fresh documents ───────────────────────────────────────────────────────────


def test_fresh_documents_exclude_every_file_a_mini_query_touches():
    full = _queries({"maud": 6, "cuad": 6})
    mini = [q for q in full if q.document in {"maud/doc00.txt", "cuad/doc01.txt"}]
    # A mini query whose second snippet reaches into another file makes that file used too.
    mini[0].snippets.append(("maud/doc05.txt", 0, 1))
    fresh = data.fresh_documents(full, mini)
    assert fresh["maud"] == ["maud/doc01.txt", "maud/doc02.txt", "maud/doc03.txt", "maud/doc04.txt"]
    assert "cuad/doc01.txt" not in fresh["cuad"] and len(fresh["cuad"]) == 5


def test_a_subset_mini_uses_entirely_has_no_fresh_documents():
    full = _queries({"privacy_qa": 3, "maud": 2})
    mini = [q for q in full if q.subset == "privacy_qa"]
    assert set(data.fresh_documents(full, mini)) == {"maud"}


# ── split ─────────────────────────────────────────────────────────────────────


def test_fresh_split_is_by_document_stratified_and_half_each():
    docs = {"cuad": [f"cuad/d{i:03d}.txt" for i in range(433)],
            "maud": [f"maud/d{i:03d}.txt" for i in range(132)],
            "contractnli": [f"contractnli/d{i:02d}.txt" for i in range(77)]}
    split = data.fresh_split(docs)
    assert set(split) == {d for v in docs.values() for d in v}
    for names in docs.values():
        dev = [d for d in names if split[d] == "dev"]
        assert len(dev) == (len(names) + 1) // 2  # the odd one to dev


def test_fresh_split_is_seeded_and_independent_of_input_order():
    docs = {"maud": [f"maud/d{i}.txt" for i in range(20)], "cuad": [f"cuad/d{i}.txt" for i in range(9)]}
    a = data.fresh_split(docs, seed=5)
    shuffled = {k: list(reversed(v)) for k, v in reversed(list(docs.items()))}
    assert a == data.fresh_split(shuffled, seed=5)
    assert a != data.fresh_split(docs, seed=6)


def test_the_recorded_fresh_split_has_not_drifted():
    # The pre-registered seed (Decision 80). Changing the procedure or the seed
    # would silently move sealed fresh-test documents into dev.
    assert data.FRESH_SPLIT_SEED == 20261116
    split = data.fresh_split({"maud": [f"maud/doc_{i:02d}.txt" for i in range(6)]})
    assert sorted(d for d, s in split.items() if s == "dev") == RECORDED_MAUD6_DEV


RECORDED_MAUD6_DEV = ["maud/doc_01.txt", "maud/doc_02.txt", "maud/doc_04.txt"]


# ── per-document sample ───────────────────────────────────────────────────────


def test_sample_keeps_at_most_n_per_document_and_does_not_depend_on_other_documents():
    queries = _queries({"maud": 5}, per_doc=7) + _queries({"cuad": 2}, per_doc=2)
    sample = data.sample_queries_per_document(queries, 3, seed=1)
    per_doc: dict = {}
    for q in sample:
        per_doc[q.document] = per_doc.get(q.document, 0) + 1
    assert per_doc == {**{f"maud/doc{d:02d}.txt": 3 for d in range(5)}, "cuad/doc00.txt": 2, "cuad/doc01.txt": 2}
    # Same draw for a document when the rest of the set changes.
    alone = data.sample_queries_per_document([q for q in queries if q.document == "maud/doc03.txt"], 3, seed=1)
    assert [q.id for q in alone] == [q.id for q in sample if q.document == "maud/doc03.txt"]
    # ... and when the input order changes.
    assert sorted(q.id for q in sample) == sorted(
        q.id for q in data.sample_queries_per_document(list(reversed(queries)), 3, seed=1))


# ── loader, end to end on a small archive ─────────────────────────────────────


def _write_archive(root: Path) -> None:
    tests = {
        "privacy_qa": _tests("privacy_qa", 2, 3),
        "contractnli": _tests("contractnli", 70, 3),
        # The first maud query also cites a second file: if mini keeps it, that file is not fresh.
        "maud": _tests("maud", 50, 5, extra_snippet="maud/doc07.txt"),
        "cuad": _tests("cuad", 100, 2),
    }
    path = data.legalbench_zip_path(root)
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as z:
        for subset, items in tests.items():
            z.writestr(f"benchmarks/{subset}.json", json.dumps({"tests": items}))
            for name in sorted({s["file_path"] for t in items for s in t["snippets"]}):
                z.writestr(f"corpus/{name}", "abc\r\ndef")


def test_loader_returns_disjoint_halves_without_mini_documents_and_every_file(tmp_path):
    _write_archive(tmp_path)
    dev = data.load_legalbench_fresh("dev", root=tmp_path, verify=False, per_document=2)
    test = data.load_legalbench_fresh("test", root=tmp_path, verify=False, per_document=2)
    assert dev.split == test.split
    # Every corpus file is indexed, with universal newlines.
    assert len(dev.documents) == 2 + 70 + 50 + 100 and dev.documents["maud/doc00.txt"] == "abc\ndef"
    with zipfile.ZipFile(data.legalbench_zip_path(tmp_path)) as z:
        tests = {s: json.loads(z.read(f"benchmarks/{s}.json"))["tests"] for s in data.LEGALBENCH_SUBSETS}
    mini_files = {p for q in data.select_legalbench_mini(tests) for p, _, _ in q.snippets}
    assert not set(dev.split) & mini_files
    assert not any(d.startswith("privacy_qa/") for d in dev.split)  # mini uses all of them
    dev_docs = {q.document for q in dev.queries}
    test_docs = {q.document for q in test.queries}
    assert dev_docs and test_docs and not dev_docs & test_docs
    assert all(dev.split[d] == "dev" for d in dev_docs)
    assert max(sum(1 for q in dev.queries if q.document == d) for d in dev_docs) == 2
    with pytest.raises(ValueError):
        data.load_legalbench_fresh("all", root=tmp_path, verify=False)


# ── harness ───────────────────────────────────────────────────────────────────


def test_cli_seals_fresh_test_and_needs_a_split():
    from eval.run_realdocs_eval import parse_args

    args = parse_args(["legalbench", "--corpus", "fresh", "--split", "dev"])
    assert args.split_seed == data.FRESH_SPLIT_SEED and args.queries_per_document == 3
    assert parse_args(["legalbench", "--split", "dev"]).split_seed == 20261115  # mini unchanged
    assert parse_args(["legalbench", "--corpus", "fresh", "--ingest-only", "--store", "x"]).split is None
    with pytest.raises(SystemExit):
        parse_args(["legalbench", "--corpus", "fresh"])
    with pytest.raises(SystemExit):
        parse_args(["legalbench", "--corpus", "fresh", "--split", "test"])
    assert parse_args(["legalbench", "--corpus", "fresh", "--split", "test", "--unseal-fresh-test"]).split == "test"


def test_legalbench_store_ingests_in_batches(tmp_path, monkeypatch):
    from eval import run_realdocs_eval as R

    documents = {f"cuad/doc{i}.txt": f"text {i}" for i in range(5)}
    monkeypatch.setattr(data, "load_legalbench_fresh", lambda split, per_document=None, seed=None:
                        data.LegalBenchFresh(queries=[], documents=documents, split={}))
    indexed: list[list[str]] = []

    class Client:
        async def close(self):
            pass

    async def open_store(path):
        log = Path(path) / "ingestion.json"
        return Client(), (json.loads(log.read_text(encoding="utf-8")) if log.exists() else [])

    async def index_files(client, files, deal_id):
        indexed.append([name for _, name in files])
        return [{"filename": name, "ok": True, "seconds": 0.0} for _, name in files]

    monkeypatch.setattr(R, "open_store", open_store)
    monkeypatch.setattr(R, "index_files", index_files)
    store = tmp_path / "store"
    store.mkdir()
    argv = ["legalbench", "--corpus", "fresh", "--store", str(store), "--ingest-only", "--max-new-documents", "2"]
    first = asyncio.run(R.run_legalbench(R.parse_args(argv)))
    second = asyncio.run(R.run_legalbench(R.parse_args(argv)))
    third = asyncio.run(R.run_legalbench(R.parse_args(argv)))
    assert [len(batch) for batch in indexed] == [2, 2, 1]
    assert len({n for batch in indexed for n in batch}) == 5
    assert (len(first["remaining"]), len(second["remaining"]), third["remaining"]) == (3, 1, [])
    assert len(json.loads((store / "ingestion.json").read_text(encoding="utf-8"))) == 5
