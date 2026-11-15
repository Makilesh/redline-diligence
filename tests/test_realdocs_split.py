"""The LegalBench-RAG dev/test split: by document, stratified, seeded, stable."""

from eval import realdocs_data as data
from eval.run_realdocs_eval import split_legalbench_documents


def _queries(per_subset: dict[str, int], per_doc: int = 3) -> list[data.SpanQuery]:
    out = []
    for subset, n_docs in per_subset.items():
        for d in range(n_docs):
            for k in range(per_doc):
                doc = f"{subset}/doc_{d:02d}.txt"
                out.append(data.SpanQuery(id=f"{subset}-{d}-{k}", subset=subset, query="q",
                                          snippets=[(doc, 0, 1)]))
    return out


def test_split_is_by_document_and_stratified_by_subset():
    queries = _queries({"maud": 18, "cuad": 29, "privacy_qa": 7})
    split = split_legalbench_documents(queries, seed=20261115)
    assert set(split) == {q.document for q in queries}
    for subset, n in {"maud": 18, "cuad": 29, "privacy_qa": 7}.items():
        dev = [d for d, s in split.items() if d.startswith(subset) and s == "dev"]
        assert len(dev) == (n + 1) // 2


def test_split_is_seeded_and_independent_of_query_order():
    queries = _queries({"maud": 18, "contractnli": 18})
    a = split_legalbench_documents(queries, seed=7)
    assert a == split_legalbench_documents(list(reversed(queries)), seed=7)
    assert a != split_legalbench_documents(queries, seed=8)


def test_the_recorded_split_has_not_drifted():
    # The pre-registered tuning split (Decision 67): changing the function would
    # silently move held-out documents into dev.
    queries = _queries({"maud": 4})
    split = split_legalbench_documents(queries, seed=20261115)
    assert sorted(d for d, s in split.items() if s == "dev") == RECORDED_MAUD4_DEV


RECORDED_MAUD4_DEV = ["maud/doc_01.txt", "maud/doc_03.txt"]
