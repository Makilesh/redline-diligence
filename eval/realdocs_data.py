"""
Real-document benchmarks: pinned downloads, checksums and loaders.

    python -m eval.realdocs_data                      # download both, verify checksums
    python -m eval.realdocs_data --only legalbench
    python -m eval.realdocs_data --only financebench

Two public benchmarks, fetched into a cache OUTSIDE the repository
(`REDLINE_EVAL_CACHE`, default ~/.cache/redline-eval) and never committed —
their licences allow evaluation use, not redistribution inside this repo:

LegalBench-RAG (Pipitone & Houir Alami, arXiv 2408.10343; code
github.com/ZeroEntropy-AI/legalbenchrag, MIT). 6,889 retrieval queries over
714 real contracts with character-span ground truth, in four sub-corpora that
keep their own licences: CUAD (The Atticus Project, CC BY 4.0), MAUD (The
Atticus Project, CC BY 4.0), ContractNLI (Koreeda & Manning, CC BY 4.0) and
PrivacyQA (Ravichander et al., MIT per its repository). The data is one zip
published on Dropbox; Dropbox links are not versioned, so the zip's SHA-256
is pinned and a different file is refused. "Mini" is the subset the authors'
own benchmark.py selects (MAX_TESTS_PER_BENCHMARK = 194, SORT_BY_DOCUMENT):
194 queries per sub-corpus over the fewest documents, 776 queries on 72
documents (8.7M characters). select_legalbench_mini() reproduces it exactly.

FinanceBench (Islam et al., arXiv 2311.11944; github.com/patronus-ai/
financebench; the Hugging Face card PatronusAI/financebench states CC BY-NC
4.0 — non-commercial). The open 150-question sample with answers and
evidence pages, over 84 real 10-K / 10-Q / 8-K / earnings PDFs. Files are
taken from one pinned commit; every PDF is checked against its git blob id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


# ------------------------------------------------------------------------------
# LegalBench-RAG
# ------------------------------------------------------------------------------
LEGALBENCH_URL = (
    "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4"
    "?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=1"
)
LEGALBENCH_ZIP = "LegalBench-RAG.zip"
# The file served on 2026-11-15 (90,591,976 bytes; benchmark files dated
# 2025-11-01 inside the archive).
LEGALBENCH_SHA256 = "27431be37db9b1db23f8ab790a42d076adb1f72d7f9e7562e36a10573405f88d"
LEGALBENCH_CODE_REVISION = "431bc8f2488a81569ab7259fa633dcc50ab77f9a"  # ZeroEntropy-AI/legalbenchrag
LEGALBENCH_SUBSETS = ("privacy_qa", "contractnli", "maud", "cuad")
LEGALBENCH_LICENCES = {
    "cuad": "CC BY 4.0 (The Atticus Project, CUAD v1)",
    "maud": "CC BY 4.0 (The Atticus Project, MAUD)",
    "contractnli": "CC BY 4.0 (Koreeda & Manning, ContractNLI)",
    "privacy_qa": "MIT (Ravichander et al., PrivacyQA repository)",
}
# benchmark.py: MAX_TESTS_PER_BENCHMARK with SORT_BY_DOCUMENT = True.
LEGALBENCH_MINI_PER_SUBSET = 194

# ------------------------------------------------------------------------------
# FinanceBench
# ------------------------------------------------------------------------------
FINANCEBENCH_REPO = "patronus-ai/financebench"
FINANCEBENCH_REVISION = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
FINANCEBENCH_LICENCE = "CC BY-NC 4.0 (Patronus AI; per the PatronusAI/financebench dataset card)"
FINANCEBENCH_RAW = f"https://raw.githubusercontent.com/{FINANCEBENCH_REPO}/{FINANCEBENCH_REVISION}"
FINANCEBENCH_TREE = (
    f"https://api.github.com/repos/{FINANCEBENCH_REPO}/git/trees/{FINANCEBENCH_REVISION}?recursive=1"
)
FINANCEBENCH_QUESTIONS = "data/financebench_open_source.jsonl"
FINANCEBENCH_DOC_INFO = "data/financebench_document_information.jsonl"


def cache_root() -> Path:
    """
    The download cache shared with the distractor mode (eval/corpus.py):
    REDLINE_EVAL_CACHE, default ~/.cache/redline-eval. Kept import-free so
    downloading does not load the ingestion stack.
    """
    configured = os.getenv("REDLINE_EVAL_CACHE")
    return Path(configured) if configured else Path.home() / ".cache" / "redline-eval"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """The id git gives a file's content — what a pinned GitHub tree lists per file."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "redline-eval"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    tmp.replace(dest)


# ==============================================================================
# LegalBench-RAG
# ==============================================================================


@dataclass
class SpanQuery:
    """One LegalBench-RAG query with its ground-truth character spans."""
    id: str
    subset: str
    query: str
    # (corpus file path, start, end) — end exclusive, offsets into the decoded text.
    snippets: list[tuple[str, int, int]]

    @property
    def document(self) -> str:
        """The (single) document the query is about; mini has no multi-file query."""
        return self.snippets[0][0]


@dataclass
class LegalBenchMini:
    queries: list[SpanQuery]
    documents: dict[str, str] = field(default_factory=dict)  # corpus path -> text


def legalbench_zip_path(root: Path | None = None) -> Path:
    return (root or cache_root()) / "legalbenchrag" / LEGALBENCH_ZIP


def fetch_legalbench(root: Path | None = None, verify: bool = True) -> Path:
    """Downloads the LegalBench-RAG zip once and checks its SHA-256."""
    path = legalbench_zip_path(root)
    if not path.exists():
        print(f"[realdocs] downloading LegalBench-RAG (~91 MB) to {path}", file=sys.stderr)
        _download(LEGALBENCH_URL, path)
    if verify:
        digest = sha256_file(path)
        if digest != LEGALBENCH_SHA256:
            raise RuntimeError(
                f"{path} has SHA-256 {digest}, expected {LEGALBENCH_SHA256}: the Dropbox file "
                "changed (or the download is corrupt). Results would not be comparable; "
                "re-pin LEGALBENCH_SHA256 deliberately if the change is intended.")
    return path


def _mini_order(test: dict) -> float:
    """benchmark.py's sort key: random.seed(first snippet's file path); random.random()."""
    rng = random.Random()
    rng.seed(test["snippets"][0]["file_path"])
    return rng.random()


def select_legalbench_mini(tests_by_subset: dict[str, list[dict]],
                           per_subset: int = LEGALBENCH_MINI_PER_SUBSET) -> list[SpanQuery]:
    """
    The authors' mini subset, reproduced from their benchmark.py.

    For a sub-corpus with more than `per_subset` tests, tests are ordered by a
    random number seeded with their first snippet's file path — so tests about
    one document stay together and the cut keeps few documents — and the
    first `per_subset` are kept. Python's str seeding is deterministic across
    processes and versions (version-2 seeding hashes the string with SHA-512).

    Args:
        tests_by_subset: Sub-corpus name -> the "tests" list of its benchmark JSON.
        per_subset: Tests kept per sub-corpus.

    Returns:
        SpanQuery list, sub-corpora in LEGALBENCH_SUBSETS order.
    """
    queries: list[SpanQuery] = []
    for subset in LEGALBENCH_SUBSETS:
        tests = list(tests_by_subset.get(subset) or [])
        if len(tests) > per_subset:
            tests = sorted(tests, key=_mini_order)[:per_subset]
        for n, test in enumerate(tests):
            queries.append(SpanQuery(
                id=f"{subset}_{n:03d}",
                subset=subset,
                query=test["query"],
                snippets=[(s["file_path"], int(s["span"][0]), int(s["span"][1]))
                          for s in test["snippets"]],
            ))
    return queries


def universal_newlines(text: str) -> str:
    """
    The text as the benchmark indexed it. Its spans are offsets into files read
    with Python's open() in text mode, which turns "\\r\\n" and "\\r" into "\\n";
    sliced from the raw bytes, 306 of the mini snippets miss their text.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_legalbench_mini(root: Path | None = None, verify: bool = True,
                         subsets: tuple[str, ...] = LEGALBENCH_SUBSETS) -> LegalBenchMini:
    """
    The mini queries and the text of every document they use, read from the zip.

    Read straight from the archive: several file names contain characters a
    Windows file system rejects (":" and "||"), so extracting it is not an option.
    """
    path = fetch_legalbench(root, verify=verify)
    with zipfile.ZipFile(path) as archive:
        tests = {
            subset: json.loads(archive.read(f"benchmarks/{subset}.json"))["tests"]
            for subset in LEGALBENCH_SUBSETS
        }
        queries = [q for q in select_legalbench_mini(tests) if q.subset in subsets]
        documents = {
            name: universal_newlines(archive.read(f"corpus/{name}").decode("utf-8"))
            for name in sorted({s[0] for q in queries for s in q.snippets})
        }
    return LegalBenchMini(queries=queries, documents=documents)


# ------------------------------------------------------------------------------
# The full benchmark, and a held-out set nothing was tuned on (Decision 80)
# ------------------------------------------------------------------------------
# Every legal-retrieval setting since Decision 67 was chosen and tested on
# mini's 72 documents. The full benchmark has 6,889 queries over 714 files;
# the 642 files no mini query touches are "fresh". They are split by
# document, per sub-corpus, with a seed fixed before anything was measured on
# them: fresh-dev is for measuring and tuning, fresh-test stays sealed for one
# later look. PrivacyQA has no fresh document (mini uses all 7).
FRESH_SPLIT_SEED = 20261116
# Queries kept per fresh document. The full benchmark averages 9.6 queries a
# document, and a query costs seconds on a 49K-chunk local-mode store; the
# intervals resample documents, so documents, not queries per document, are
# what buys precision.
FRESH_QUERIES_PER_DOCUMENT = 3


def canonical_corpus_path(path: str, corpus: dict[str, str]) -> str:
    """
    The corpus name a benchmark snippet's file_path refers to. One CUAD file
    name is NFD in cuad.json ("LECLANCHE\\u0301") and NFC in the zip; matching
    on NFC keeps its eight queries.

    Args:
        path: file_path as the benchmark JSON gives it.
        corpus: NFC-normalised name -> corpus name, for every file in the zip.
    """
    return corpus.get(unicodedata.normalize("NFC", path), path)


def legalbench_full_queries(tests_by_subset: dict[str, list[dict]],
                            corpus_names: list[str] | None = None) -> list[SpanQuery]:
    """
    Every benchmark query, id "<subset>_f<index in its benchmark file>" so an
    id never changes with the sample drawn from them.

    Args:
        tests_by_subset: Sub-corpus name -> the "tests" list of its benchmark JSON.
        corpus_names: Every file under corpus/ (to canonicalise snippet paths).

    Returns:
        SpanQuery list, sub-corpora in LEGALBENCH_SUBSETS order.
    """
    corpus = {unicodedata.normalize("NFC", n): n for n in corpus_names or []}
    queries: list[SpanQuery] = []
    for subset in LEGALBENCH_SUBSETS:
        for n, test in enumerate(tests_by_subset.get(subset) or []):
            queries.append(SpanQuery(
                id=f"{subset}_f{n:04d}",
                subset=subset,
                query=test["query"],
                snippets=[(canonical_corpus_path(s["file_path"], corpus), int(s["span"][0]), int(s["span"][1]))
                          for s in test["snippets"]],
            ))
    return queries


def fresh_documents(full: list[SpanQuery], mini: list[SpanQuery]) -> dict[str, list[str]]:
    """
    Sub-corpus -> sorted documents of the full benchmark that no mini query
    uses (any snippet, not only the first), so nothing tuned on mini is in them.
    """
    used = {p for q in mini for p, _, _ in q.snippets}
    out: dict[str, set[str]] = {}
    for q in full:
        for path, _, _ in q.snippets:
            if path not in used:
                out.setdefault(q.subset, set()).add(path)
    return {subset: sorted(docs) for subset, docs in sorted(out.items())}


def fresh_split(documents_by_subset: dict[str, list[str]], seed: int = FRESH_SPLIT_SEED) -> dict[str, str]:
    """
    Document -> "dev" or "test": per sub-corpus, the sorted names shuffled with
    Random("<seed>:<subset>") and the first half (the odd one) to dev — the
    procedure Decision 67 used on mini (split_legalbench_documents), applied to
    the fresh documents with their own seed.
    """
    split: dict[str, str] = {}
    for subset in sorted(documents_by_subset):
        docs = sorted(documents_by_subset[subset])
        random.Random(f"{seed}:{subset}").shuffle(docs)
        n_dev = (len(docs) + 1) // 2
        split.update({d: ("dev" if i < n_dev else "test") for i, d in enumerate(docs)})
    return split


def sample_queries_per_document(queries: list[SpanQuery], per_document: int,
                                seed: int = FRESH_SPLIT_SEED) -> list[SpanQuery]:
    """
    At most `per_document` queries about each document, drawn with
    Random("<seed>:<document>") from its queries in id order — so a document's
    sample does not depend on which other documents are in the set. Returned
    in id order within the input's sub-corpus order.
    """
    by_doc: dict[str, list[SpanQuery]] = {}
    for q in queries:
        by_doc.setdefault(q.document, []).append(q)
    kept: set[str] = set()
    for doc, items in by_doc.items():
        items = sorted(items, key=lambda q: q.id)
        if len(items) > per_document:
            items = random.Random(f"{seed}:{doc}").sample(items, per_document)
        kept.update(q.id for q in items)
    return [q for q in queries if q.id in kept]


@dataclass
class LegalBenchFresh:
    queries: list[SpanQuery]                                # the split's sampled queries
    documents: dict[str, str] = field(default_factory=dict)  # EVERY corpus file -> text
    split: dict[str, str] = field(default_factory=dict)      # fresh document -> "dev" / "test"


def load_legalbench_fresh(split: str, root: Path | None = None, verify: bool = True,
                          per_document: int | None = FRESH_QUERIES_PER_DOCUMENT,
                          seed: int = FRESH_SPLIT_SEED) -> LegalBenchFresh:
    """
    The sampled queries of one half of the fresh documents, and the text of
    every corpus file (all 714 are indexed: a held-out document's passages
    compete with mini's and with the other half's for every rank).

    Args:
        split: "dev" or "test".
        per_document: Queries kept per document (None keeps all).
        seed: Split and sampling seed.
    """
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', not {split!r}")
    path = fetch_legalbench(root, verify=verify)
    with zipfile.ZipFile(path) as archive:
        tests = {
            subset: json.loads(archive.read(f"benchmarks/{subset}.json"))["tests"]
            for subset in LEGALBENCH_SUBSETS
        }
        names = sorted(n[len("corpus/"):] for n in archive.namelist()
                       if n.startswith("corpus/") and not n.endswith("/"))
        documents = {name: universal_newlines(archive.read(f"corpus/{name}").decode("utf-8"))
                     for name in names}
    full = legalbench_full_queries(tests, names)
    corpus = {unicodedata.normalize("NFC", n): n for n in names}
    mini = [SpanQuery(id=q.id, subset=q.subset, query=q.query,
                      snippets=[(canonical_corpus_path(p, corpus), s, e) for p, s, e in q.snippets])
            for q in select_legalbench_mini(tests)]
    halves = fresh_split(fresh_documents(full, mini), seed)
    # Single-document queries only (the full benchmark has no other kind; a
    # query reaching into a mini document would not be fresh).
    chosen = [q for q in full if halves.get(q.document) == split
              and all(halves.get(p) == split for p, _, _ in q.snippets)]
    if per_document is not None:
        chosen = sample_queries_per_document(chosen, per_document, seed)
    return LegalBenchFresh(queries=chosen, documents=documents, split=halves)


# ==============================================================================
# FinanceBench
# ==============================================================================


@dataclass
class EvidencePage:
    doc_name: str
    page: int          # 1-based, as the ingestion pipeline numbers pages
    text: str          # the annotator's evidence extract


@dataclass
class FinanceQuestion:
    id: str
    question: str
    answer: str
    question_type: str
    question_reasoning: str
    doc_name: str
    evidence: list[EvidencePage]
    company: str = ""  # FinanceBench's company label: the unit the dev/test split keeps together


def financebench_dir(root: Path | None = None) -> Path:
    return (root or cache_root()) / "financebench"


def _financebench_tree(root: Path | None = None) -> dict[str, dict]:
    """{repo path: {"sha", "size"}} of the pinned commit, cached next to the files."""
    path = financebench_dir(root) / f"tree_{FINANCEBENCH_REVISION[:12]}.json"
    if not path.exists():
        _download(FINANCEBENCH_TREE, path)
    tree = json.loads(path.read_text(encoding="utf-8"))
    return {e["path"]: {"sha": e["sha"], "size": e.get("size")} for e in tree["tree"]
            if e.get("type") == "blob"}


def _fetch_financebench_file(repo_path: str, tree: dict, root: Path | None = None) -> Path:
    dest = financebench_dir(root) / repo_path
    expected = tree[repo_path]["sha"]
    if dest.exists() and git_blob_sha1(dest) == expected:
        return dest
    _download(f"{FINANCEBENCH_RAW}/{urllib.request.quote(repo_path)}", dest)
    actual = git_blob_sha1(dest)
    if actual != expected:
        raise RuntimeError(f"{repo_path}: git blob {actual}, expected {expected} at "
                           f"{FINANCEBENCH_REVISION[:12]}")
    return dest


def load_financebench(root: Path | None = None) -> list[FinanceQuestion]:
    """The 150 open questions (questions file fetched and verified on first use)."""
    tree = _financebench_tree(root)
    path = _fetch_financebench_file(FINANCEBENCH_QUESTIONS, tree, root)
    _fetch_financebench_file(FINANCEBENCH_DOC_INFO, tree, root)
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(FinanceQuestion(
            id=row["financebench_id"],
            question=row["question"],
            answer=str(row["answer"]),
            question_type=row["question_type"],
            question_reasoning=str(row.get("question_reasoning") or ""),
            doc_name=row["doc_name"],
            evidence=[
                EvidencePage(doc_name=e["doc_name"], page=int(e["evidence_page_num"]) + 1,
                             text=e.get("evidence_text") or "")
                for e in row["evidence"]
            ],
            company=str(row.get("company") or ""),
        ))
    return questions


def financebench_pdf(doc_name: str, root: Path | None = None) -> Path:
    """Local path of one filing, downloaded and verified on first use."""
    tree = _financebench_tree(root)
    return _fetch_financebench_file(f"pdfs/{doc_name}.pdf", tree, root)


def select_financebench_documents(questions: list[FinanceQuestion], max_documents: int | None,
                                  seed: int = 20261115) -> list[str]:
    """
    The filings to index, and so the questions to ask.

    All 84 by default. With `max_documents`, a seeded random sample of the
    filings (not of the questions), so every question about a sampled filing
    is kept and none is asked of a store that lacks its document.

    Args:
        questions: All questions.
        max_documents: Cap on filings, or None for all.
        seed: Sampling seed (reported with the results).

    Returns:
        Sorted doc_names.
    """
    names = sorted({q.doc_name for q in questions} | {e.doc_name for q in questions for e in q.evidence})
    if max_documents is None or max_documents >= len(names):
        return names
    return sorted(random.Random(seed).sample(names, max_documents))


# The tuning split used for Decision 68. Fixed before any setting was chosen.
FINANCEBENCH_SPLIT_SEED = 20261115


def financebench_company_split(questions: list[FinanceQuestion],
                               seed: int = FINANCEBENCH_SPLIT_SEED) -> dict[str, list[str]]:
    """
    Companies for a dev/test split that keeps each company's filings together.

    Splitting by question would put one year's 10-K in dev and the next year's
    in test, and the main failure on this benchmark is confusing exactly those
    two. So the sorted company labels are shuffled with a fixed seed and the
    first half is dev.

    Args:
        questions: All questions.
        seed: Shuffle seed.

    Returns:
        {"dev": [...], "test": [...]} sorted company labels.
    """
    companies = sorted({q.company for q in questions})
    random.Random(seed).shuffle(companies)
    half = len(companies) // 2
    return {"dev": sorted(companies[:half]), "test": sorted(companies[half:])}


def select_financebench_split(questions: list[FinanceQuestion], split: str,
                              seed: int = FINANCEBENCH_SPLIT_SEED) -> list[str]:
    """Sorted doc_names of every filing a question of the split's companies uses."""
    companies = set(financebench_company_split(questions, seed)[split])
    chosen = [q for q in questions if q.company in companies]
    return sorted({q.doc_name for q in chosen} | {e.doc_name for q in chosen for e in q.evidence})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.realdocs_data", description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=("legalbench", "financebench"))
    args = parser.parse_args(argv)
    total = 0
    if args.only in (None, "legalbench"):
        mini = load_legalbench_mini()
        size = legalbench_zip_path().stat().st_size
        total += size
        print(f"LegalBench-RAG: {len(mini.queries)} mini queries, {len(mini.documents)} documents, "
              f"{sum(len(t) for t in mini.documents.values()):,} characters; zip {size / 1e6:.1f} MB")
    if args.only in (None, "financebench"):
        questions = load_financebench()
        names = select_financebench_documents(questions, None)
        size = 0
        for name in names:
            size += financebench_pdf(name).stat().st_size
        total += size
        print(f"FinanceBench: {len(questions)} questions, {len(names)} PDFs, {size / 1e6:.1f} MB")
    print(f"downloaded/verified {total / 1e6:.1f} MB under {cache_root()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
