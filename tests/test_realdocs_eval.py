"""
The real-document harness (eval/run_realdocs_eval.py) on synthetic inputs:
span alignment and the LegalBench-RAG metrics, the mini selection, the
FinanceBench page and token scoring, and the context mirror. No download,
no model.
"""

import random

import pytest

from eval import realdocs_data as data
from eval import span_alignment as sa
from eval.run_realdocs_eval import (
    chunk_pages,
    context_bodies,
    evidence_tokens,
    score_pages,
    score_spans,
    token_recall,
)

SOURCE = (
    "ARTICLE I\n\n"
    "1.1    Definitions.   The following terms have the meanings below.\n"
    "=====\n"
    "  \"Affiliate\" means any Person that controls another Person.\n\n"
    "1.2    Interpretation.  Headings are for convenience only.\n"
)


# ── alignment ────────────────────────────────────────────────────────────────


def test_a_chunk_whose_whitespace_changed_aligns_to_its_exact_offsets():
    source = sa.NormalisedText(SOURCE)
    chunk = "1.1 Definitions. The following terms have the meanings below."
    alignment = sa.align(chunk, source)
    assert alignment.coverage == 1.0
    (start, end), = alignment.spans
    assert SOURCE[start:end].split() == chunk.split()


def test_a_dropped_rule_line_is_aligned_piecewise_and_counted():
    source = sa.NormalisedText(SOURCE)
    # The sectioner drops "=====" rule lines: the chunk skips them.
    chunk = ("1.1 Definitions. The following terms have the meanings below. "
             "\"Affiliate\" means any Person that controls another Person.")
    alignment = sa.align(chunk, source)
    assert 0.8 < alignment.coverage <= 1.0
    assert alignment.spans
    covered = "".join(SOURCE[s:e] for s, e in alignment.spans)
    assert "Definitions" in covered and "controls another" in covered


def test_text_that_is_not_in_the_source_is_not_placed():
    alignment = sa.align("A narrative the converter generated about revenue growth in FY2023.",
                         sa.NormalisedText(SOURCE))
    assert alignment.spans == [] and alignment.coverage == 0.0


def test_the_hint_selects_the_later_of_two_identical_passages():
    text = "Notices shall be in writing. " * 2
    source = sa.NormalisedText(text)
    late = sa.align("Notices shall be in writing.", source, hint=10)
    assert late.spans[0][0] > 10


# ── LegalBench-RAG metrics ───────────────────────────────────────────────────


def test_char_precision_and_recall_follow_the_benchmark_definition():
    truth = {"a.txt": [(100, 200)]}
    precision, recall = sa.char_precision_recall({"a.txt": [(150, 250)]}, truth)
    assert precision == pytest.approx(0.5) and recall == pytest.approx(0.5)
    # Overlapping retrieved spans are merged: a character retrieved twice counts once.
    precision, recall = sa.char_precision_recall({"a.txt": [(100, 200), (120, 180)]}, truth)
    assert precision == 1.0 and recall == 1.0
    assert sa.char_precision_recall({"b.txt": [(100, 200)]}, truth) == (0.0, 0.0)
    assert sa.char_precision_recall({}, truth) == (0.0, 0.0)


def test_score_spans_reports_rank_and_context_metrics():
    query = data.SpanQuery(id="q", subset="cuad", query="?", snippets=[("a.txt", 100, 200)])
    ranking = [("b.txt", [(0, 500)]), ("a.txt", [(90, 300)])]
    metrics = score_spans(ranking, query, [1, 5], context=[("a.txt", [(0, 1000)])])
    assert metrics["hit@1"] == 0.0 and metrics["hit@5"] == 1.0
    assert metrics["mrr@10"] == pytest.approx(0.5)
    assert metrics["doc_hit@1"] == 0.0 and metrics["doc_hit@5"] == 1.0
    assert metrics["char_recall@5"] == 1.0
    assert metrics["char_precision@5"] == pytest.approx(100 / 710)
    assert metrics["snippet_full@context"] == 1.0 and metrics["char_recall@context"] == 1.0


def test_universal_newlines_matches_python_text_mode():
    assert data.universal_newlines("a\r\nb\rc\n") == "a\nb\nc\n"


def test_mini_selection_reproduces_the_authors_ordering():
    def tests_for(n_docs, per_doc):
        return [{"query": f"q{d}-{i}", "snippets": [{"file_path": f"maud/doc{d}.txt", "span": [0, 1]}]}
                for d in range(n_docs) for i in range(per_doc)]

    tests = {"maud": tests_for(60, 10), "privacy_qa": tests_for(3, 5)}
    queries = data.select_legalbench_mini(tests, per_subset=194)
    assert [q.subset for q in queries].count("privacy_qa") == 15  # under the cap: all kept
    maud = [q for q in queries if q.subset == "maud"]
    assert len(maud) == 194
    # benchmark.py: sorted by random.seed(first snippet's path); random.random().
    expected = sorted(tests["maud"], key=lambda t: (random.seed(t["snippets"][0]["file_path"]),
                                                     random.random())[1])[:194]
    assert [q.query for q in maud] == [t["query"] for t in expected]
    # Tests about one document stay together, so the cut keeps few documents.
    assert len({q.document for q in maud}) == 20


# ── FinanceBench scoring ─────────────────────────────────────────────────────


def test_evidence_tokens_survive_a_different_parser():
    tokens = evidence_tokens("Purchases of property, plant and\nequipment (PP&E)\n(1,577)\n$\n1,373")
    assert {"purchases", "property", "1577", "1373"} <= tokens
    assert token_recall(tokens, "Purchases of property, plant and equipment (PP&E)  (1,577)  (1,373)") == 1.0
    assert token_recall(set(), "anything") is None


def test_chunk_pages_covers_a_stitched_tables_range():
    assert chunk_pages({"page_number": 41}) == {41}
    assert chunk_pages({"page_number": 41, "page_range": [41, 43]}) == {41, 42, 43}
    assert chunk_pages({"page_number": None}) == set()


def test_page_hits_need_the_right_filing_and_page():
    question = data.FinanceQuestion(
        id="fb1", question="?", answer="1", question_type="metrics-generated", question_reasoning="",
        doc_name="ACME_2022_10K", evidence=[data.EvidencePage("ACME_2022_10K", 60, "capital expenditure")])
    filename_of = {"ACME_2022_10K": "ACME_2022_10K.pdf"}
    run = {"ranking": [{"source_file": "OTHER_2022_10K.pdf", "page_number": 60, "text": ""},
                       {"source_file": "ACME_2022_10K.pdf", "page_number": 59, "page_range": [59, 60],
                        "text": "capital expenditure"}]}
    metrics = score_pages(run, question, filename_of, [1, 5], {})
    assert metrics["page_hit@1"] == 0.0 and metrics["page_hit@5"] == 1.0
    assert metrics["doc_hit@1"] == 0.0 and metrics["mrr@10"] == pytest.approx(0.5)


def test_financebench_document_sample_is_seeded_and_keeps_whole_filings():
    questions = [data.FinanceQuestion(id=str(i), question="?", answer="", question_type="",
                                      question_reasoning="", doc_name=f"D{i % 9}", evidence=[])
                 for i in range(30)]
    a = data.select_financebench_documents(questions, 4, seed=7)
    assert a == data.select_financebench_documents(questions, 4, seed=7)
    assert len(a) == 4 and a == sorted(a)
    assert data.select_financebench_documents(questions, None) == [f"D{i}" for i in range(9)]


def test_git_blob_id_matches_git(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello\n")
    # `git hash-object` of "hello\n".
    assert data.git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"


# ── context mirror ───────────────────────────────────────────────────────────


def test_context_bodies_mirror_the_synthesizer():
    from src.agents.answer_synthesizer import _format_context_for_synthesis

    parent = "Parent section text. " * 20
    chunks = [
        {"chunk_id": "c1", "text": "Parent section text.", "parent_text": parent,
         "parent_chunk_id": "p1", "source_file": "a.txt"},
        {"chunk_id": "c2", "text": "Parent section text.", "parent_text": parent,
         "parent_chunk_id": "p1", "source_file": "a.txt"},   # same parent: shown once
        {"chunk_id": "c3", "text": "A table row 1,234.", "source_file": "b.pdf", "table_id": "t1"},
    ]
    bodies = context_bodies(chunks)
    assert [c["chunk_id"] for c, _ in bodies] == ["c1", "c3"]
    assert bodies[0][1] == parent
    assert _format_context_for_synthesis(chunks).count("<document") == len(bodies)
