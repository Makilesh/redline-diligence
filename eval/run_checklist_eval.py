"""
The checklist review grid (Decision 95) measured on CUAD contracts.

    python -m eval.run_checklist_eval prepare --cuad-zip CUAD_v1.zip --out LABELS.json [--unseen]
    python -m eval.run_checklist_eval sweep   --store DIR --labels LABELS.json --half tune --out SWEEP.jsonl [--laya]
    python -m eval.run_checklist_eval analyse --sweep SWEEP.jsonl [--out RESULT.json]
    python -m eval.run_checklist_eval grid    --store DIR --labels LABELS.json --half confirm --out GRID.json
    python -m eval.run_checklist_eval latency --store DIR --labels LABELS.json --half tune --sizes 10 25 50 --out LAT.json

CUAD v1 (The Atticus Project, CC BY 4.0; Hendrycks et al., arXiv 2103.06268)
labels every one of its 510 contracts for 41 clause categories, absent ones
included, with character offsets. LegalBench-RAG's 462 CUAD files are CUAD's
own text release (identical after universal newlines), so a label's offsets
are offsets into the indexed file. Neither dataset is committed: the zips are
read where they are and labels are written outside the repository.

Documents. Only LegalBench-RAG's DEV contracts are labelled for tuning and
confirmation: mini-dev (Decision 67's split) and fresh-dev (Decision 80's).
Their labels are read; every other CUAD entry is skipped before its labels
are touched — fresh-test and mini-test are those benchmarks' held-out
queries. The dev contracts are halved, per group, by a seeded shuffle
(CHECKLIST_SPLIT_SEED): TUNE chooses every setting, CONFIRM checks them. The
48 CUAD contracts LegalBench-RAG does not ship are UNSEEN: no round indexed
them; `prepare --unseen` reads their labels, once the settings are frozen.

Cells and metrics (a cell is a contract and a checklist item; it is present
when CUAD marks at least one span of the item's category in that contract):
the shown window is aligned to the contract with span_alignment (whitespace-
collapsed) and hits when it shares a character with a gold span; hit@3 counts
the two alternates too. FA = present cells marked not_found; NF = absent cells
marked not_found; FP = found cells that are present with a hit; DC = present
cells found with a hit plus absent cells not_found, over all cells.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

from eval import realdocs_data as data
from eval import span_alignment as sa

CHECKLIST_SPLIT_SEED = "grid11:20261120"
DEV_GROUPS = ("mini-dev", "fresh-dev")
MINI_SPLIT_SEED = 20261115  # Decision 67's mini split (run_realdocs_eval's default)
DEAL_ID = "realdocs_legalbench"
DEPTHS = (1, 2, 3, 5)
FORMS = ("document", "section")
BOOTSTRAP_SEED = 95
RERANKERS = {
    "bge": {"RERANKER_MODEL": "BAAI/bge-reranker-v2-m3"},
    # The CPU host's reranker, in full precision as it runs there.
    "minilm": {"RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2", "RERANKER_FP16": "0"},
}


# ==============================================================================
# Labels
# ==============================================================================


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def legalbench_cuad_splits(archive: zipfile.ZipFile) -> dict[str, str]:
    """
    LegalBench-RAG CUAD file -> "mini-dev" | "mini-test" | "fresh-dev" | "fresh-test",
    by the procedures of Decisions 67 and 80 (run_realdocs_eval, realdocs_data).
    """
    from eval.run_realdocs_eval import split_legalbench_documents

    tests = {subset: json.loads(archive.read(f"benchmarks/{subset}.json"))["tests"]
             for subset in data.LEGALBENCH_SUBSETS}
    names = sorted(n[len("corpus/"):] for n in archive.namelist()
                   if n.startswith("corpus/") and not n.endswith("/"))
    corpus = {_nfc(n): n for n in names}
    full = data.legalbench_full_queries(tests, names)
    mini = [data.SpanQuery(id=q.id, subset=q.subset, query=q.query,
                           snippets=[(data.canonical_corpus_path(p, corpus), s, e) for p, s, e in q.snippets])
            for q in data.select_legalbench_mini(tests)]
    out = {p: f"mini-{s}" for p, s in split_legalbench_documents(mini, MINI_SPLIT_SEED).items()}
    for p, s in data.fresh_split(data.fresh_documents(full, mini), data.FRESH_SPLIT_SEED).items():
        out.setdefault(p, f"fresh-{s}")
    return {p: s for p, s in out.items() if p.startswith("cuad/")}


def checklist_halves(groups: dict[str, str], seed: str = CHECKLIST_SPLIT_SEED) -> dict[str, str]:
    """
    Dev document -> "tune" | "confirm": per group, the sorted names shuffled with
    Random("<seed>:<group>"), the first half (the odd one) to tune.
    """
    halves: dict[str, str] = {}
    for group in DEV_GROUPS:
        docs = sorted(d for d, g in groups.items() if g == group)
        random.Random(f"{seed}:{group}").shuffle(docs)
        n_tune = (len(docs) + 1) // 2
        halves.update({d: ("tune" if i < n_tune else "confirm") for i, d in enumerate(docs)})
    return halves


def display_names(corpus_names: list[str]) -> dict[str, str]:
    """Corpus path -> the source_file write_corpus gives it (the store's names)."""
    from src.data_processing.ingest_pipeline import sanitize_filename

    taken: set[str] = set()
    out: dict[str, str] = {}
    for n, path in enumerate(sorted(corpus_names)):
        name = sanitize_filename(path.split("/")[-1])
        if not name.lower().endswith(".txt"):
            name += ".txt"
        if name in taken:
            name = f"{n:03d}_{name}"
        taken.add(name)
        out[path] = name
    return out


def _spans(context: str, text: str, answers: list[dict]) -> list[list[int]] | None:
    """Gold spans as offsets into `text`; None when an answer is not where it says."""
    spans = []
    for answer in answers:
        start, span = int(answer["answer_start"]), answer["text"]
        if context[start:start + len(span)] != span:
            return None
        if context != text:
            start = text.find(span)
            if start < 0:
                return None
        spans.append([start, start + len(span)])
    return spans


def prepare(args) -> dict:
    """Writes the labels of the dev contracts (and, with --unseen, of the unseen ones)."""
    from src.reports.checklist import CHECKLIST

    categories = {item.cuad_category for item in CHECKLIST}
    with zipfile.ZipFile(data.fetch_legalbench()) as archive:
        groups = {p: g for p, g in legalbench_cuad_splits(archive).items() if g in DEV_GROUPS}
        names = sorted(n[len("corpus/"):] for n in archive.namelist()
                       if n.startswith("corpus/") and not n.endswith("/"))
        lb_cuad = [n for n in names if n.startswith("cuad/")]
        texts = {p: data.universal_newlines(archive.read(f"corpus/{p}").decode("utf-8")) for p in groups}
    shown = display_names(names)
    halves = checklist_halves(groups)
    documents: dict[str, dict] = {
        p: {"corpus": "legalbench", "path": p, "source_file": shown[p], "group": groups[p], "half": halves[p]}
        for p in groups
    }
    with zipfile.ZipFile(args.cuad_zip) as archive:
        members = sorted(n for n in archive.namelist()
                         if n.startswith("CUAD_v1/full_contract_txt/") and n.endswith(".txt"))
        stem_of_member = {m: _nfc(m.rsplit("/", 1)[-1][:-len(".txt")]) for m in members}
        # Each LegalBench-RAG CUAD file is the CUAD text of the same name, or —
        # one file — of the one name that starts with its own ("..._Development
        # Agreement" ships CUAD's "..._Development Agreement_Option Agreement").
        # Matched on file names, so CUAD_v1.json is then looked up exactly and
        # no other contract's labels can be taken for a dev contract's.
        member_of: dict[str, str] = {}
        for p in lb_cuad:
            stem = _nfc(p[len("cuad/"):-len(".txt")])
            exact = [m for m in members if stem_of_member[m] == stem]
            longer = [m for m in members if stem_of_member[m].startswith(stem)]
            if exact or len(longer) == 1:
                member_of[p] = (exact or longer)[0]
        if len(member_of) != len(lb_cuad):
            raise SystemExit(f"{len(lb_cuad) - len(member_of)} LegalBench-RAG CUAD files have no CUAD text")
        mapped = set(member_of.values())
        unseen = [m for m in members if m not in mapped]
        wanted = {stem_of_member[member_of[p]]: p for p in groups}
        if args.unseen:
            shown_unseen = display_names(unseen)  # as write_corpus names them when they are indexed
            for m in unseen:
                documents[m] = {"corpus": "cuad", "path": m, "source_file": shown_unseen[m],
                                "group": "unseen", "half": "unseen"}
                texts[m] = data.universal_newlines(archive.read(m).decode("utf-8"))
                wanted[stem_of_member[m]] = m
        entries = json.loads(archive.read("CUAD_v1/CUAD_v1.json"))["data"]

    labelled, problems = 0, []
    for entry in entries:
        key = wanted.get(_nfc(entry["title"]))
        if key is None:
            continue  # not a document of these halves: its labels are never read
        (paragraph,) = entry["paragraphs"]
        cats: dict[str, list] = {}
        for qa in paragraph["qas"]:
            category = qa["id"].split("__", 1)[1]
            if category not in categories:
                continue
            spans = _spans(paragraph["context"], texts[key], qa["answers"])
            if spans is None:
                problems.append(f"{key}: {category}")
                continue
            cats[category] = spans
        documents[key]["categories"] = cats
        documents[key]["chars"] = len(texts[key])
        labelled += 1
    missing = sorted(k for k, v in documents.items() if "categories" not in v)
    out = {"split_seed": CHECKLIST_SPLIT_SEED, "documents": documents, "unseen_names": unseen}
    Path(args.out).write_text(json.dumps(out, indent=0), encoding="utf-8")
    summary = {
        "documents": len(documents), "labelled": labelled, "missing": missing, "problems": problems,
        "halves": {h: sum(v["half"] == h for v in documents.values()) for h in ("tune", "confirm", "unseen")},
        "unseen_in_cuad": len(unseen),
    }
    print(json.dumps(summary, indent=1))
    return summary


def document_texts(labels: dict, keys: list[str], cuad_zip: str | None) -> dict[str, str]:
    """The contract texts the labels index into."""
    texts: dict[str, str] = {}
    lb = [k for k in keys if labels["documents"][k]["corpus"] == "legalbench"]
    if lb:
        with zipfile.ZipFile(data.fetch_legalbench()) as archive:
            for k in lb:
                texts[k] = data.universal_newlines(archive.read(f"corpus/{k}").decode("utf-8"))
    cuad = [k for k in keys if labels["documents"][k]["corpus"] == "cuad"]
    if cuad:
        with zipfile.ZipFile(cuad_zip) as archive:
            for k in cuad:
                texts[k] = data.universal_newlines(archive.read(k).decode("utf-8"))
    return texts


class Aligner:
    """Window text -> spans in its contract, near its chunk."""

    def __init__(self, texts: dict[str, str]):
        self.normalised = {k: sa.NormalisedText(t) for k, t in texts.items()}
        self._chunk_start: dict[tuple[str, str], int] = {}

    def spans(self, key: str, chunk_id: str, chunk_text: str, text: str) -> list[list[int]]:
        source = self.normalised[key]
        if (key, chunk_id) not in self._chunk_start:
            self._chunk_start[(key, chunk_id)] = sa.align(chunk_text, source).norm_start or 0
        hint = self._chunk_start[(key, chunk_id)]
        return [list(s) for s in sa.align(text.rstrip("…"), source, hint).spans]


def _hits(spans: list, gold: list) -> bool:
    return any(sa.touches([tuple(s) for s in spans], tuple(g)) for g in gold)


# ==============================================================================
# Store
# ==============================================================================


def _configure(args) -> None:
    """Model environment, before anything imports the reranker module."""
    for name, value in RERANKERS[args.reranker].items():
        os.environ[name] = value
    if getattr(args, "device", "gpu") == "cpu":
        # "-1", not "": on Windows an empty value deletes the variable (the C
        # runtime's putenv), which shows the GPU again.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(args.threads)
    os.environ.setdefault("WARM_MODELS", "0")


async def _open(store: str):
    from qdrant_client import AsyncQdrantClient

    return AsyncQdrantClient(path=str(Path(store) / "qdrant"))


async def _rows(client, labels: dict, half: str, deal: str,
                limit: int | None = None) -> tuple[list[str], dict[str, str]]:
    """The half's documents as store doc ids (the first `limit` by name), and doc id -> label key."""
    from src.reports.checklist import deal_documents
    from src.vector_db.constants import COLLECTION_NAME

    documents, _ = await deal_documents(client, COLLECTION_NAME, deal)
    doc_of = {d.source_file: d.doc_id for d in documents}
    keys = sorted(k for k, v in labels["documents"].items() if v["half"] == half)
    if limit:
        keys = keys[:limit]
    missing = [k for k in keys if labels["documents"][k]["source_file"] not in doc_of]
    if missing:
        raise SystemExit(f"{len(missing)} documents of {half} are not in the store, e.g. {missing[:2]}")
    key_of = {doc_of[labels["documents"][k]["source_file"]]: k for k in keys}
    return [doc_of[labels["documents"][k]["source_file"]] for k in keys], key_of


# ==============================================================================
# Sweep (TUNE): every window of the top chunks, scored in both forms
# ==============================================================================


async def sweep(args) -> None:
    import torch

    from src.decisions.reranker_calibration import active_profile
    from src.reports import checklist
    from src.vector_db.constants import COLLECTION_NAME

    torch.set_num_threads(args.threads)
    items = checklist.resolve_items(args.items)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    client = await _open(args.store)
    rows, key_of = await _rows(client, labels, args.half, args.deal, args.limit)
    started = time.perf_counter()
    candidates = await checklist.search_candidates(
        client, COLLECTION_NAME, args.deal, items, rows, depth=max(DEPTHS))
    search_s = time.perf_counter() - started
    texts = document_texts(labels, sorted(key_of.values()), args.cuad_zip)
    aligner = Aligner(texts)
    out = open(args.out, "w", encoding="utf-8")
    rerank_s = 0.0
    for b in range(0, len(rows), 8):
        batch = set(rows[b:b + 8])
        cells = {k: v for k, v in candidates.items() if k[0] in batch}
        scored = {}
        for form in FORMS:
            mark = time.perf_counter()
            scored[form] = await checklist.score_windows(cells, items, form)
            rerank_s += time.perf_counter() - mark
        for doc_id in [d for d in rows if d in batch]:
            key = key_of[doc_id]
            for item in items:
                gold = labels["documents"][key]["categories"].get(item.cuad_category) or []
                record = {"key": key, "doc_id": doc_id, "item": item.id, "category": item.cuad_category,
                          "present": bool(gold), "gold": gold, "windows": [], "chunks": {}}
                by_form = {form: {(w.payload["chunk_id"], w.start, w.end): w
                                  for w in scored[form].get((doc_id, item.id), [])} for form in FORMS}
                for (chunk_id, start, end), w in by_form[FORMS[0]].items():
                    chunk_text = str(w.payload.get("text") or "")
                    if chunk_id not in record["chunks"]:
                        record["chunks"][chunk_id] = aligner.spans(key, chunk_id, chunk_text, chunk_text)
                    record["windows"].append({
                        "chunk": chunk_id, "rank": w.chunk_rank, "start": start, "end": end,
                        "spans": aligner.spans(key, chunk_id, chunk_text, w.text),
                        "raw": {form: by_form[form][(chunk_id, start, end)].raw for form in FORMS},
                        "text": w.text,
                    })
                out.write(json.dumps(record) + "\n")
        print(f"[checklist] swept {min(b + 8, len(rows))}/{len(rows)} documents", file=sys.stderr)
    out.close()
    print(json.dumps({"documents": len(rows), "cells": len(rows) * len(items),
                      "profile": active_profile().name, "search_s": round(search_s, 1),
                      "rerank_s": round(rerank_s, 1)}))
    if args.laya:
        await add_answerability(args.out, active_profile())


def best_window(windows: list[dict], form: str, depth: int, calibrate) -> tuple[dict | None, list[dict]]:
    """A cell's evidence and its alternates under one setting, as the report picks them."""
    pool = sorted((w for w in windows if w["rank"] < depth),
                  key=lambda w: (-calibrate(w["raw"][form]), w["rank"], w["start"]))
    if not pool:
        return None, []
    chosen: list[dict] = []
    for w in pool[1:]:
        if len(chosen) == 2:
            break
        if any(o["chunk"] == w["chunk"] and w["start"] < o["end"] and o["start"] < w["end"]
               for o in [pool[0], *chosen]):
            continue
        chosen.append(w)
    return pool[0], chosen


async def add_answerability(path: str, profile) -> None:
    """Laya's P(stated) for every window that is a cell's evidence under some (form, depth)."""
    from src.reports.checklist import ITEMS_BY_ID, read_answerability

    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    todo: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        for form in FORMS:
            for depth in DEPTHS:
                w, _ = best_window(r["windows"], form, depth, profile.calibrate)
                if w is not None:
                    todo[(ITEMS_BY_ID[r["item"]].question, w["text"])].append(w)
    pairs = list(todo)
    started = time.perf_counter()
    for b in range(0, len(pairs), 64):
        values = await read_answerability(pairs[b:b + 64])
        for pair, value in zip(pairs[b:b + 64], values):
            for w in todo[pair]:
                w["laya"] = value
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"laya_pairs": len(pairs), "laya_s": round(time.perf_counter() - started, 1)}))


# ==============================================================================
# Metrics
# ==============================================================================


def cell_metrics(cells: list[dict]) -> dict:
    """
    FA, NF, FP, DC and the rest over scored cells. A cell: {"present", "status",
    "hit1", "hit3", "key", "category"}.
    """
    present = [c for c in cells if c["present"]]
    absent = [c for c in cells if not c["present"]]
    found = [c for c in cells if c["status"] == "found"]
    not_found = [c for c in cells if c["status"] == "not_found"]

    def share(part, whole):
        return None if not whole else part / len(whole)

    return {
        "cells": len(cells), "present": len(present), "absent": len(absent),
        "hit1": share(sum(c["hit1"] for c in present), present),
        "hit3": share(sum(c["hit3"] for c in present), present),
        "FA": share(sum(c["status"] == "not_found" for c in present), present),
        "NF": share(sum(c["status"] == "not_found" for c in absent), absent),
        "found_rate_present": share(sum(c["status"] == "found" for c in present), present),
        "FP": share(sum(c["present"] and c["hit1"] for c in found), found),
        "FP_lenient": share(sum(c["present"] for c in found), found),
        "DC": share(sum(c["present"] and c["hit1"] for c in found) + sum(not c["present"] for c in not_found), cells),
        "decided": share(len(found) + len(not_found), cells),
        "unclear": share(sum(c["status"] == "unclear" for c in cells), cells),
        "found": len(found), "not_found": len(not_found),
    }


def with_intervals(cells: list[dict]) -> dict:
    """cell_metrics with bootstrap intervals that resample documents."""
    from eval.decisions import stats

    point = cell_metrics(cells)
    out = {"point": point}
    for name in ("hit1", "hit3", "FA", "NF", "FP", "DC", "decided"):
        out[name] = stats.bootstrap_ci(cells, lambda rows, n=name: cell_metrics(rows)[n],
                                       seed=BOOTSTRAP_SEED, cluster=lambda c: c["key"])
    return out


def per_category(cells: list[dict]) -> dict:
    from eval.decisions import stats

    out = {}
    for category in sorted({c["category"] for c in cells}):
        mine = [c for c in cells if c["category"] == category]
        m = cell_metrics(mine)
        present = [c for c in mine if c["present"]]
        absent = [c for c in mine if not c["present"]]
        decided_right = sum(c["present"] and c["hit1"] and c["status"] == "found" for c in mine) + sum(
            c["status"] == "not_found" for c in absent)
        m["FA_ci"] = stats.wilson_interval(sum(c["status"] == "not_found" for c in present), len(present))
        m["NF_ci"] = stats.wilson_interval(sum(c["status"] == "not_found" for c in absent), len(absent))
        m["DC_ci"] = stats.wilson_interval(decided_right, len(mine))
        out[category] = m
    return out


GRID = [round(i / 1000, 3) for i in range(0, 1001)]


def absent_threshold(p_present: list[float], cells: list[dict], fa_max: float = 0.02) -> float:
    """S3: the largest grid value with at most fa_max of the present cells below it."""
    present = sorted(p for p, c in zip(p_present, cells) if c["present"])
    t_absent = 0.0
    for t in GRID:
        if sum(p < t for p in present) <= fa_max * len(present):
            t_absent = t
        else:
            break
    return t_absent


def found_threshold(p_correct: list[float], cells: list[dict], fp_min: float = 0.90) -> float | None:
    """S3: the smallest grid value at which >= fp_min of the cells at or above it are present with a hit."""
    for t in GRID:
        above = [c for p, c in zip(p_correct, cells) if p >= t]
        if above and sum(c["present"] and c["hit1"] for c in above) >= fp_min * len(above):
            return t
    return None


def statuses(p_present: list[float], p_correct: list[float], t_absent: list[float | None],
             t_found: float | None) -> list[str]:
    """checklist.cell_status's rule: found first, then not_found (per-cell bar; None: never), else unclear."""
    out = []
    for present, correct, bar in zip(p_present, p_correct, t_absent):
        if t_found is not None and correct >= t_found:
            out.append("found")
        elif bar is not None and present < bar:
            out.append("not_found")
        else:
            out.append("unclear")
    return out


# A3: an item's own not-found bar needs at least this many present cells — the
# fewest for which 2% is a whole clause.
MIN_PRESENT_FOR_ABSENCE = 50


def item_absent_bars(p_present: list[float], cells: list[dict], pooled: float,
                     fa_max: float = 0.02) -> dict[str, float]:
    """A3: per item with >= MIN_PRESENT_FOR_ABSENCE present cells, its S3 bar on its own present cells, capped."""
    bars = {}
    for item in sorted({c["item"] for c in cells}):
        present = [p for p, c in zip(p_present, cells) if c["item"] == item and c["present"]]
        if len(present) < MIN_PRESENT_FOR_ABSENCE:
            continue
        bars[item] = min(pooled, absent_threshold(present, [{"present": True}] * len(present), fa_max))
    return bars


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fit_status_model(cells: list[dict], laya: bool) -> dict:
    """
    A1: P_present and P_correct by logistic regression on [logit s, (logit P_laya),
    item one-hot] — one shared slope, an offset per item (C=1.0) — as
    checklist.Calibration parameters, with S3's thresholds on them.
    """
    from sklearn.linear_model import LogisticRegression

    items = sorted({c["item"] for c in cells})

    def features(c: dict) -> list[float]:
        row = [_logit(c["score"])] + ([_logit(c["laya"])] if laya else [])
        return row + [1.0 if c["item"] == i else 0.0 for i in items]

    x = [features(c) for c in cells]
    out = {}
    for name, target in (("present", lambda c: c["present"]), ("correct", lambda c: c["present"] and c["hit1"])):
        y = [int(target(c)) for c in cells]
        model = LogisticRegression(C=1.0, max_iter=5000).fit(x, y)
        coef = [float(v) for v in model.coef_[0]]
        out[name] = {
            "intercept": float(model.intercept_[0]),
            "score_weight": coef[0],
            "answerability_weight": coef[1] if laya else 0.0,
            "item_offsets": dict(zip(items, coef[2 if laya else 1:])),
        }
    return out


def calibrated(params: dict, c: dict) -> float:
    """checklist.Calibration.probability for one analysed cell."""
    z = params["intercept"] + params["score_weight"] * _logit(c["score"]) + params["item_offsets"].get(c["item"], 0.0)
    if params["answerability_weight"]:
        z += params["answerability_weight"] * _logit(c["laya"])
    return 1 / (1 + math.exp(-max(min(z, 60.0), -60.0)))


def apply_status_model(model: dict, cells: list[dict]) -> list[dict]:
    """Cells with their status under a fitted model (no candidate: unclear)."""
    usable = [c for c in cells if c["score"] is not None]
    p_present = [calibrated(model["present"], c) for c in usable]
    p_correct = [calibrated(model["correct"], c) for c in usable]
    bars = [model["absent_below"].get(c["item"]) for c in usable]
    out = [dict(c, status=s, p_present=p, p_correct=q) for c, s, p, q in zip(
        usable, statuses(p_present, p_correct, bars, model["found_from"]), p_present, p_correct)]
    return out + [dict(c, status="unclear") for c in cells if c["score"] is None]


def status_model(cells: list[dict], laya: bool) -> dict:
    """fit_status_model plus S3's thresholds (A3: the not-found bar per item) on the same cells."""
    usable = [c for c in cells if c["score"] is not None]
    model = fit_status_model(usable, laya)
    p_present = [calibrated(model["present"], c) for c in usable]
    model["pooled_absent_below"] = absent_threshold(p_present, usable)
    model["absent_below"] = item_absent_bars(p_present, usable, model["pooled_absent_below"])
    model["found_from"] = found_threshold([calibrated(model["correct"], c) for c in usable], usable)
    return model


def cross_validated(cells: list[dict], laya: bool, folds: int = 5) -> dict:
    """Informational: models and thresholds fit on 4/5 of TUNE's documents, applied to the fifth."""
    keys = sorted({c["key"] for c in cells})
    random.Random(f"{CHECKLIST_SPLIT_SEED}:cv").shuffle(keys)
    fold_of = {k: i % folds for i, k in enumerate(keys)}
    held = []
    for f in range(folds):
        train = [c for c in cells if fold_of[c["key"]] != f]
        model = status_model(train, laya)
        held += apply_status_model(model, [c for c in cells if fold_of[c["key"]] == f])
    return cell_metrics(held)


def analysed_cells(records: list[dict], form: str, depth: int, calibrate) -> list[dict]:
    cells = []
    for r in records:
        w, alternates = best_window(r["windows"], form, depth, calibrate)
        hit1 = bool(w) and _hits(w["spans"], r["gold"])
        cells.append({
            "key": r["key"], "item": r["item"], "category": r["category"], "present": r["present"],
            "hit1": hit1, "hit3": hit1 or any(_hits(a["spans"], r["gold"]) for a in alternates),
            "chunk_hit1": bool(w) and _hits(r["chunks"].get(w["chunk"], []), r["gold"]),
            "score": calibrate(w["raw"][form]) if w else None, "laya": (w or {}).get("laya"),
        })
    return cells


def _hit1(cells: list[dict]) -> float:
    present = [c for c in cells if c["present"]]
    return sum(c["hit1"] for c in present) / len(present) if present else 0.0


def analyse(args) -> dict:
    """S1, S2, A2 and A1 (with the Laya rule) on TUNE sweeps."""
    from eval.decisions import stats
    from src.decisions.reranker_calibration import PROFILES

    records = [json.loads(line) for line in open(args.sweep, encoding="utf-8")]
    profile = next(p for p in PROFILES if p.name == args.profile)
    calibrate = profile.calibrate

    def table_of(recs):
        table = {}
        for form in FORMS:
            for depth in DEPTHS:
                cells = analysed_cells(recs, form, depth, calibrate)
                scored = [c for c in cells if c["score"] is not None]
                present = [c for c in cells if c["present"]]
                table[(form, depth)] = {
                    "cells": cells, "hit1": _hit1(cells),
                    "hit3": sum(c["hit3"] for c in present) / len(present),
                    "chunk_hit1": sum(c["chunk_hit1"] for c in present) / len(present),
                    "auc": stats.auc([c["score"] for c in scored if c["present"]],
                                     [c["score"] for c in scored if not c["present"]]),
                    "no_candidate": sum(c["score"] is None for c in cells),
                }
        # S1: on the production form, the smallest depth within 1.0 point of
        # the best hit@1. S2: the section form only when it beats it by > 1.0.
        best = max(table[("document", d)]["hit1"] for d in DEPTHS)
        depth = min(d for d in DEPTHS if table[("document", d)]["hit1"] >= best - 0.01)
        form = "section" if (table[("section", depth)]["hit1"]
                             > table[("document", depth)]["hit1"] + 0.01) else "document"
        return table, form, depth

    table, form, depth = table_of(records)
    revisions = {}
    if args.revised:
        # A2: a revised question replaces the original only where its TUNE
        # hit@1, at the chosen depth and form, is higher.
        revised = [json.loads(line) for line in open(args.revised, encoding="utf-8")]
        for item in sorted({r["item"] for r in revised}):
            old = _hit1(analysed_cells([r for r in records if r["item"] == item], form, depth, calibrate))
            new = _hit1(analysed_cells([r for r in revised if r["item"] == item], form, depth, calibrate))
            revisions[item] = {"original_hit1": old, "revised_hit1": new, "kept": new > old}
        kept = {i for i, v in revisions.items() if v["kept"]}
        records = [r for r in records if r["item"] not in kept] + [r for r in revised if r["item"] in kept]
        table, form, depth = table_of(records)

    cells = table[(form, depth)]["cells"]
    pooled = 0
    for path in args.pool or []:
        # The pre-registered fallback: refit on TUNE plus a confirmation half's
        # cells as the production grid recorded them (score and Laya P of the
        # shown window). Depth and form stay TUNE's.
        grid_cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
        cells = cells + [dict(c, laya=c.get("answerability")) for c in grid_cells]
        pooled += len(grid_cells)
    result = {
        "profile": profile.name, "sweep": args.sweep, "revised": args.revised, "revisions": revisions,
        "pool": args.pool, "pooled_cells": pooled,
        "table": {f"{f}@{d}": {k: v for k, v in row.items() if k != "cells"} for (f, d), row in table.items()},
        "chosen": {"form": form, "depth": depth},
    }
    laya = all(c["laya"] is not None for c in cells if c["score"] is not None)
    for name, with_laya in (("reranker", False), ("with_laya", True)):
        if with_laya and not laya:
            continue
        model = status_model(cells, with_laya)
        judged = apply_status_model(model, cells)
        result[name] = {
            "model": model, "metrics": with_intervals(judged), "per_category": per_category(judged),
            "cross_validated": cross_validated(cells, with_laya),
        }
    if "with_laya" in result:
        result["laya_dc_gain"] = (result["with_laya"]["metrics"]["point"]["DC"]
                                  - result["reranker"]["metrics"]["point"]["DC"])
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    brief = {k: v for k, v in result.items() if k not in ("reranker", "with_laya")}
    for name in ("reranker", "with_laya"):
        if name in result:
            brief[name] = {"metrics": result[name]["metrics"]["point"], "cv": result[name]["cross_validated"],
                           "thresholds": [result[name]["model"]["absent_below"], result[name]["model"]["found_from"]]}
    print(json.dumps(brief, indent=1, default=str))
    return result


# ==============================================================================
# The production grid on a half (CONFIRM, UNSEEN), and its latency
# ==============================================================================


async def _chunk_texts(client, chunk_ids: list[str]) -> dict[str, str]:
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    from src.vector_db.constants import COLLECTION_NAME

    out: dict[str, str] = {}
    for b in range(0, len(chunk_ids), 256):
        part = chunk_ids[b:b + 256]
        points, _ = await client.scroll(
            collection_name=COLLECTION_NAME, limit=len(part), with_vectors=False,
            with_payload=["chunk_id", "text"],
            scroll_filter=Filter(must=[FieldCondition(key="chunk_id", match=MatchAny(any=part))]))
        out.update({p.payload["chunk_id"]: p.payload.get("text") or "" for p in points})
    return out


def score_report(report, labels: dict, key_of: dict[str, str], texts: dict[str, str],
                 chunk_texts: dict[str, str]) -> list[dict]:
    """One scored cell per (row, item) of a ChecklistReport."""
    from src.reports.checklist import ITEMS_BY_ID

    aligner = Aligner(texts)
    cells = []
    for row in report.rows:
        key = key_of[row.doc_id]
        for cell in row.cells:
            category = ITEMS_BY_ID[cell.item_id].cuad_category
            gold = labels["documents"][key]["categories"].get(category) or []
            shown = [cell.passage, *cell.alternates] if cell.passage else []
            hits = [_hits(aligner.spans(key, p.chunk_id, chunk_texts.get(p.chunk_id, ""), p.text), gold)
                    for p in shown]
            cells.append({"key": key, "category": category, "item": cell.item_id, "present": bool(gold),
                          "status": cell.status, "score": cell.score, "answerability": cell.answerability,
                          "hit1": bool(hits) and hits[0], "hit3": any(hits),
                          # Kept for reading the failures by hand.
                          "passage": cell.passage.text if cell.passage else "",
                          "section": cell.passage.section_heading if cell.passage else ""})
    return cells


async def grid(args) -> dict:
    import torch

    from src.reports.checklist import build_checklist

    torch.set_num_threads(args.threads)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    client = await _open(args.store)
    rows, key_of = await _rows(client, labels, args.half, args.deal, args.limit)
    report = await build_checklist(args.deal, client=client, doc_ids=rows, document_cap=10**6, budget_s=1e9)
    texts = document_texts(labels, sorted(key_of.values()), args.cuad_zip)
    chunk_ids = sorted({p.chunk_id for row in report.rows for c in row.cells
                        for p in ([c.passage] if c.passage else []) + c.alternates})
    cells = score_report(report, labels, key_of, texts, await _chunk_texts(client, chunk_ids))
    result = {
        "half": args.half, "reranker": report.reranker, "answerability": report.answerability,
        "documents": report.documents_checked, "timings_ms": report.timings_ms, "elapsed_ms": report.elapsed_ms,
        "metrics": with_intervals(cells), "per_category": per_category(cells), "cells": cells,
    }
    if report.answerability:
        # The Laya rule (b): the same cells' status from the reranker alone,
        # by the shipped reranker-only model, against the grid's.
        from eval.decisions import stats
        from src.reports.checklist import STATUS_MODELS, cell_status

        alone = [dict(c, status=cell_status(c["item"], c["score"], STATUS_MODELS[report.reranker]))
                 for c in cells]
        result["reranker_only"] = {"metrics": with_intervals(alone), "per_category": per_category(alone)}
        paired = [{"key": a["key"], "with": w, "without": a} for w, a in zip(cells, alone)]
        result["laya_dc_paired"] = stats.paired_bootstrap(
            paired, lambda rows: cell_metrics([r["without"] for r in rows])["DC"],
            lambda rows: cell_metrics([r["with"] for r in rows])["DC"],
            seed=BOOTSTRAP_SEED, cluster=lambda r: r["key"])
    Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cells", "per_category")}, indent=1, default=str))
    return result


async def latency(args) -> dict:
    import torch

    from src.reports.checklist import build_checklist

    if args.device == "cpu" and torch.cuda.is_available():
        # A CPU host measured on the GPU is not measured (it happened once here).
        raise SystemExit("A GPU is visible: refusing to time the CPU host (set CUDA_VISIBLE_DEVICES=-1).")
    torch.set_num_threads(args.threads)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    client = await _open(args.store)
    rows, _ = await _rows(client, labels, args.half, args.deal, args.limit)
    # Warm the models and the question vectors (a running API has both).
    await build_checklist(args.deal, client=client, doc_ids=sorted(rows)[:1], budget_s=1e9)
    runs = []
    for size in args.sizes:
        chosen = random.Random(f"{CHECKLIST_SPLIT_SEED}:latency:{size}").sample(sorted(rows), size)
        for n in range(args.runs):
            started = time.perf_counter()
            report = await build_checklist(args.deal, client=client, doc_ids=chosen, document_cap=10**6,
                                           budget_s=1e9)
            wall = time.perf_counter() - started
            cells = report.documents_checked * len(report.items)
            runs.append({"documents": size, "run": n, "cells": cells, "wall_s": round(wall, 2),
                         "timings_ms": report.timings_ms, "answerability": report.answerability,
                         "reranker": report.reranker,
                         # The reranker's own rule (src/vector_db/reranker.py).
                         "device": "cuda" if torch.cuda.is_available() else "cpu",
                         "model_ms_per_cell": round((report.timings_ms.get("rerank", 0)
                                                     + report.timings_ms.get("answerability", 0)) / cells, 1)})
            print(json.dumps(runs[-1]), file=sys.stderr)
    Path(args.out).write_text(json.dumps(runs, indent=1), encoding="utf-8")
    return {"runs": runs}


# ==============================================================================
# CLI
# ==============================================================================


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval.run_checklist_eval", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cuad-zip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--unseen", action="store_true", help="also read the labels of the 48 unseen contracts")
    for name in ("sweep", "grid", "latency"):
        p = sub.add_parser(name)
        p.add_argument("--store", required=True, help="a local-mode store directory (holding qdrant/)")
        p.add_argument("--labels", required=True)
        p.add_argument("--half", choices=("tune", "confirm", "unseen"), required=True)
        p.add_argument("--deal", default=DEAL_ID)
        p.add_argument("--reranker", choices=sorted(RERANKERS), default="bge")
        p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--cuad-zip")
        p.add_argument("--limit", type=int, help="the first N documents of the half, by name (smoke runs)")
        p.add_argument("--out", required=True)
        if name == "sweep":
            p.add_argument("--laya", action="store_true")
            p.add_argument("--items", nargs="+", help="checklist item ids (default: all)")
        if name == "latency":
            p.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50])
            p.add_argument("--runs", type=int, default=3)
    p = sub.add_parser("analyse")
    p.add_argument("--sweep", required=True)
    p.add_argument("--revised", help="a sweep of revised questions (A2), kept per item where hit@1 rises")
    p.add_argument("--pool", nargs="+", help="grid results whose cells join the fit (the fallback refit)")
    p.add_argument("--profile", default="bge-reranker-v2-m3")
    p.add_argument("--out")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "analyse":
        analyse(args)
    else:
        _configure(args)
        asyncio.run({"sweep": sweep, "grid": grid, "latency": latency}[args.command](args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
