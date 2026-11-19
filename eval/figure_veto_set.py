"""
New data for the GPU figure veto (Decision 90): claims built from the tables the
index really holds on FinanceBench's DEV-company evidence pages.

    python -m eval.figure_veto_set extract --store D:/redline_data/int9/fb_int9 --out D:/redline_data/veto10
    python -m eval.figure_veto_set build --out D:/redline_data/veto10
    python -m eval.figure_veto_set build --kind two-row --source D:/redline_data/veto10 \
        --out D:/redline_data/veto10/two_row

The verification benchmark's test half has been read (Decisions 63-64), so a
veto chosen on its dev half needs data no round has seen. FinanceBench is
CC BY-NC 4.0: everything this writes stays under --out, never in the repository.

extract   reads a store written by the production ingestion straight from its
          local-mode sqlite files (read-only), keeps every child chunk of the
          dev companies' filings with its parent text attached, as
          expand_context attaches it. Decision 68's company split
          (seed 20261115); of the test companies only the labels are read,
          because the split shuffles all of them.
build     turns every converter table on a dev evidence page into claims:
          for sampled (row, year) cells, the figure as printed with its line
          item and period, and wrong claims made from it by construction —
          another year's figure, the neighbouring row's figure, the direction
          or the sign flipped, another company named. Each claim's context is
          every chunk the index holds for that page. Tables whose columns the
          PDF processor could not name ("Column 2") yield no claims and are
          counted. A claim whose figure numeric grounding cannot find at all
          is a generation slip and is dropped (counted), whatever its label.
          --kind two-row builds claims joining two rows of one table for the
          same year ("In fiscal 2017, goodwill was $X and other assets was
          $Y."), and wrong ones in which one of the two figures is another
          year's or the neighbouring row's, or another company is named —
          the claims the single-figure set did not cover.

The plan, written before anything was scored, is in Decision 90.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261119
CELLS_PER_TABLE = 16
PAIRS_PER_TABLE = 12
UNITS = ("currency", "per_share", "percent")
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_GENERIC = re.compile(r"^Column \d+$")
_FOOTNOTE = re.compile(r"\(\s*(?:\d{1,2}|[a-z])\s*\)|\[\s*\w{1,2}\s*\]|[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§]")
_LEADER = re.compile(r"\s*(?:\.\s*){2,}")
# "Net income (loss)": the printed sign says which side of the pair holds.
_SIGN_PAIRS = (("income", "loss"), ("income", "expense"), ("gain", "loss"), ("increase", "decrease"))


# ==============================================================================
# extract
# ==============================================================================


def _rows(db: Path):
    """Payloads of a local-mode collection, read-only. The store was written by
    this repository's own ingestion, and qdrant_client's local mode unpickles
    exactly these rows when it opens the store."""
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        for (blob,) in con.execute("SELECT point FROM points"):
            yield pickle.loads(blob).payload or {}
    finally:
        con.close()


def extract(store: Path, out: Path) -> dict:
    """Dev-company chunks and evidence pages, from the store's sqlite files."""
    from eval import realdocs_data as data

    questions = data.load_financebench()
    dev = set(data.financebench_company_split(questions)["dev"])
    dev_q = [q for q in questions if q.company in dev]
    del questions  # nothing of the test companies is kept
    filings = sorted({q.doc_name for q in dev_q} | {e.doc_name for q in dev_q for e in q.evidence})
    company = {}
    for q in dev_q:
        company[q.doc_name] = q.company
        for e in q.evidence:
            company.setdefault(e.doc_name, q.company)
    meta = {
        "dev_companies": sorted(dev),
        "filings": filings,
        "company_of": company,
        "evidence_pages": sorted({(e.doc_name, e.page) for q in dev_q for e in q.evidence}),
    }
    wanted = {f"{name}.pdf" for name in filings}
    collection = store / "qdrant" / "collection"
    children = [p for p in _rows(collection / "manda_due_diligence" / "storage.sqlite")
                if p.get("source_file") in wanted]
    parent_ids = {c["parent_chunk_id"] for c in children if c.get("parent_chunk_id")}
    parents = {p["chunk_id"]: p.get("text", "")
               for p in _rows(collection / "manda_parent_chunks" / "storage.sqlite")
               if p.get("chunk_id") in parent_ids}
    for c in children:
        if c.get("parent_chunk_id") in parents:
            c["parent_text"] = parents[c["parent_chunk_id"]]
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "fb_dev_chunks.jsonl.gz", "wt", encoding="utf-8") as f:
        for c in children:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out / "fb_dev_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"filings": len(filings), "chunks": len(children), "parents": len(parents),
            "evidence_pages": len(meta["evidence_pages"])}


# ==============================================================================
# build
# ==============================================================================


def chunk_pages(chunk: dict) -> set[int]:
    """The pages a chunk is cited with: its page, or a stitched table's range."""
    pages = {int(chunk["page_number"])} if chunk.get("page_number") else set()
    page_range = chunk.get("page_range") or []
    if len(page_range) == 2:
        pages |= set(range(int(page_range[0]), int(page_range[1]) + 1))
    return pages


def clean_label(label: str) -> str:
    """A row label as a sentence can use it: no dot leaders, footnotes or colon."""
    text = _LEADER.sub(" ", _FOOTNOTE.sub(" ", label))
    text = " ".join(text.split()).strip(" :;,")
    first, _, rest = text.partition(" ")
    if first[:1].isupper() and first[1:].islower():
        first = first.lower()
    return f"{first} {rest}".strip()


def render(cell: dict, unit: str) -> str:
    """
    A cell's figure as printed, the way a claim states it: "$" before an
    amount or a per-share figure, "%" after a percentage, "-" for a value the
    statement prints negative ("(1,488.4)"). No scale word: the claim states
    the figure the table prints.
    """
    digits = re.sub(r"[^\d.,]", "", cell["raw"]).strip(".,")
    sign = "-" if cell["raw_value"] < 0 else ""
    if unit == "percent":
        return f"{sign}{digits}%"
    return f"{sign}${digits}"


def column_years(columns: list[str]) -> dict[str, int]:
    """Columns that name exactly one year, which no other column names."""
    from src.data_processing.financial_table_converter import registry_period

    years = {}
    for col in columns:
        found = set(_YEAR.findall(col))
        if len(found) == 1 and registry_period(col) is not None:
            years[col] = int(found.pop())
    counts = Counter(years.values())
    return {col: year for col, year in years.items() if counts[year] == 1}


def sign_sides(label: str) -> tuple[str, str] | None:
    """("net income", "net loss") for "Net income (loss)" — (positive, negative)."""
    for pos, neg in _SIGN_PAIRS:
        for pattern in (rf"\b{pos}\s*\(\s*{neg}\s*\)", rf"\(\s*{neg}\s*\)\s*{pos}\b"):
            m = re.search(pattern, label, re.IGNORECASE)
            if m:
                head, tail = label[: m.start()], label[m.end():]
                return (" ".join(f"{head}{pos}{tail}".split()), " ".join(f"{head}{neg}{tail}".split()))
    return None


def _table_rows(chunks: list[dict]) -> tuple[list[str], list[dict]]:
    """A table's columns and structured rows, from its row_by_row windows in order."""
    windows = sorted(
        (c for c in chunks if c.get("table_representation") == "row_by_row"),
        key=lambda c: (c.get("table_row_range") or [0])[0],
    )
    rows = [r for c in windows for r in c.get("structured_rows") or []]
    columns: list[str] = []
    for r in rows:
        for col in r.get("values", {}):
            if col not in columns:
                columns.append(col)
    return columns, rows


def _eligible(row: dict) -> bool:
    """A row a claim can name: a figure kind the claims state, a label of words."""
    label = clean_label(row.get("line_item") or "")
    return (row.get("unit") in UNITS and bool(re.search(r"[A-Za-z]", label))
            and not re.search(r"\d", label))


def _numeric(row: dict, col: str) -> dict | None:
    cell = row.get("values", {}).get(col)
    if cell and cell.get("raw_value") is not None and cell.get("period") is not None \
            and re.search(r"\d", cell.get("raw", "")):
        return cell
    return None


def claims_for_table(table_id: str, rows: list[dict], columns: list[str], company: str,
                     others: list[str]) -> list[dict]:
    """Every claim one table yields (see the module docstring)."""
    years = column_years(columns)
    cells = [(i, col) for i, row in enumerate(rows) if _eligible(row)
             for col in years if _numeric(row, col)]
    rng = random.Random(f"{SEED}:{table_id}")
    chosen = sorted(rng.sample(cells, min(CELLS_PER_TABLE, len(cells))))
    out = []
    for i, col in chosen:
        row = rows[i]
        cell = _numeric(row, col)
        unit = row["unit"]
        label, year = clean_label(row["line_item"]), years[col]
        figure = render(cell, unit)
        printed_in_row = {c.get("raw") for c in row["values"].values()}
        base = {"table_id": table_id, "row": i, "column": col, "line_item": row["line_item"],
                "year": year, "figure": cell["raw"]}

        def add(label_: str, kind: str, text: str, base: dict = base, **extra) -> None:
            out.append({**base, "label": label_, "kind": kind, "claim": text, **extra})

        add("supported", "plain", f"In fiscal {year}, {label} was {figure}.")
        add("supported", "company", f"{company} reported {label} of {figure} in fiscal {year}.")
        swap = random.Random(f"{SEED}:{table_id}:{i}:{col}:company").choice(others)
        add("contradicted", "company", f"{swap} reported {label} of {figure} in fiscal {year}.",
            named=swap)

        # Another year's figure of the same row, attached to this year.
        for other in sorted((c for c in years if c != col and _numeric(row, c)),
                            key=lambda c: (abs(years[c] - year), years[c] > year)):
            alt = _numeric(row, other)
            if alt["raw"] != cell["raw"]:
                add("contradicted", "period", f"In fiscal {year}, {label} was {render(alt, unit)}.",
                    source_column=other)
                break

        # The neighbouring row's figure in the same column, attached to this row.
        order = [k for k in range(i + 1, len(rows))] + [k for k in range(i - 1, -1, -1)]
        for k in order:
            other_row = rows[k]
            alt = _numeric(other_row, col) if _eligible(other_row) else None
            if alt and other_row["unit"] == unit and alt["raw"] not in printed_in_row:
                add("contradicted", "neighbour", f"In fiscal {year}, {label} was {render(alt, unit)}.",
                    source_row=k)
                break

        # Direction against the prior year of the same row.
        prior = next((c for c, y in years.items() if y == year - 1), None)
        prev = _numeric(row, prior) if prior else None
        if prev and cell["raw_value"] > 0 and prev["raw_value"] > 0 and prev["raw_value"] != cell["raw_value"]:
            true, false = ("rose", "fell") if cell["raw_value"] > prev["raw_value"] else ("fell", "rose")
            add("supported", "direction", f"{label[:1].upper()}{label[1:]} {true} to {figure} in fiscal {year}.")
            add("contradicted", "direction", f"{label[:1].upper()}{label[1:]} {false} to {figure} in fiscal {year}.")

        # The side of an "income (loss)" pair the printed sign selects.
        sides = sign_sides(label)
        if sides and cell["raw_value"] != 0:
            magnitude = render({**cell, "raw_value": abs(cell["raw_value"])}, unit)
            true, false = (sides[0], sides[1]) if cell["raw_value"] > 0 else (sides[1], sides[0])
            add("supported", "sign", f"In fiscal {year}, {true} was {magnitude}.")
            add("contradicted", "sign", f"In fiscal {year}, {false} was {magnitude}.")
    return out


def _other_year(row: dict, col: str, years: dict[str, int]) -> dict | None:
    """The row's figure for the nearest other year (earlier first), printed differently."""
    cell = _numeric(row, col)
    for other in sorted((c for c in years if c != col and _numeric(row, c)),
                        key=lambda c: (abs(years[c] - years[col]), years[c] > years[col])):
        alt = _numeric(row, other)
        if alt["raw"] != cell["raw"]:
            return alt
    return None


def _neighbour(rows: list[dict], k: int, col: str, avoid: set[int]) -> dict | None:
    """The same column's figure in the nearest eligible row below (else above)
    row k, of the same unit, printed nowhere in row k; rows in `avoid` skipped."""
    row = rows[k]
    printed = {c.get("raw") for c in row["values"].values()}
    for m in [*range(k + 1, len(rows)), *range(k - 1, -1, -1)]:
        if m in avoid:
            continue
        other = rows[m]
        alt = _numeric(other, col) if _eligible(other) else None
        if alt and other["unit"] == row["unit"] and alt["raw"] not in printed:
            return alt
    return None


def two_row_claims_for_table(table_id: str, rows: list[dict], columns: list[str], company: str,
                             others: list[str]) -> list[dict]:
    """
    Claims joining two rows of one table for one year (--kind two-row).

    For up to PAIRS_PER_TABLE seeded (row, row, year) triples whose figures
    are printed differently and whose labels differ:
        supported/plain      "In fiscal Y, A was Fa and B was Fb."
        supported/company    "C reported A of Fa and B of Fb in fiscal Y."
        contradicted/company the same with another dev company named
        contradicted/period  one of the two figures (seeded) replaced by its
                             row's figure for another year
        contradicted/neighbour  the same figure replaced by the neighbouring
                             row's figure in that column
    """
    years = column_years(columns)
    labels = [clean_label(r.get("line_item") or "") for r in rows]
    triples = [
        (i, j, col)
        for col in years for i in range(len(rows)) for j in range(i + 1, len(rows))
        if _eligible(rows[i]) and _eligible(rows[j]) and labels[i] != labels[j]
        and _numeric(rows[i], col) and _numeric(rows[j], col)
        and _numeric(rows[i], col)["raw"] != _numeric(rows[j], col)["raw"]
    ]
    rng = random.Random(f"{SEED}:pairs:{table_id}")
    out = []
    for i, j, col in sorted(rng.sample(triples, min(PAIRS_PER_TABLE, len(triples)))):
        year = years[col]
        cells = {i: _numeric(rows[i], col), j: _numeric(rows[j], col)}
        figs = {k: render(cells[k], rows[k]["unit"]) for k in (i, j)}
        base = {"table_id": table_id, "row": i, "row2": j, "column": col, "year": year,
                "line_item": rows[i]["line_item"], "line_item2": rows[j]["line_item"],
                "figure": cells[i]["raw"], "figure2": cells[j]["raw"]}
        key = f"{SEED}:{table_id}:{i}:{j}:{col}"
        swap = random.Random(f"{key}:company").choice(others)
        made = [
            ("supported", "plain", _two_rows(year, labels[i], figs[i], labels[j], figs[j]), {}),
            ("supported", "company", _two_rows(year, labels[i], figs[i], labels[j], figs[j], company), {}),
            ("contradicted", "company", _two_rows(year, labels[i], figs[i], labels[j], figs[j], swap), {}),
        ]
        target = random.Random(f"{key}:which").choice((i, j))
        swapped = {"period": _other_year(rows[target], col, years),
                   "neighbour": _neighbour(rows, target, col, avoid={i, j})}
        for kind, alt in swapped.items():
            if alt is None:
                continue
            wrong = dict(figs)
            wrong[target] = render(alt, rows[target]["unit"])
            made.append(("contradicted", kind, _two_rows(year, labels[i], wrong[i], labels[j], wrong[j]),
                         {"replaced_row": target}))
        for label_, kind, text, extra in made:
            out.append({**base, "label": label_, "kind": kind, "claim": text, **extra})
    return out


def _two_rows(year: int, label_a: str, fig_a: str, label_b: str, fig_b: str, company: str = "") -> str:
    """One claim stating two rows' figures for one year, naming a company or not."""
    if company:
        return f"{company} reported {label_a} of {fig_a} and {label_b} of {fig_b} in fiscal {year}."
    return f"In fiscal {year}, {label_a} was {fig_a} and {label_b} was {fig_b}."


def page_contexts(chunks: list[dict], pages: list[tuple[str, int]]) -> dict[tuple[str, int], list[dict]]:
    """Every chunk the index holds for each page, in document order."""
    wanted = set(pages)
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for c in sorted(chunks, key=lambda c: (c["source_file"], c.get("chunk_index", 0))):
        doc = c["source_file"][:-4] if c["source_file"].lower().endswith(".pdf") else c["source_file"]
        for p in chunk_pages(c):
            if (doc, p) in wanted:
                out[(doc, p)].append(c)
    return out


def build(out: Path, kind: str = "single", source: Path | None = None) -> dict:
    """
    claims.jsonl and pages.jsonl.gz under `out`, and a summary.

    Args:
        out: Where the set is written.
        kind: "single" (one figure per claim) or "two-row" (claims joining two
            rows of one table).
        source: Where extract wrote the chunks (default: `out`).
    """
    import logging

    from src.verification.claims import split_claims
    from src.verification.numeric_grounding import ContextIndex, ground_texts

    logging.disable(logging.WARNING)
    source = source or out
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "fb_dev_meta.json").read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in gzip.open(source / "fb_dev_chunks.jsonl.gz", "rt", encoding="utf-8")]
    make = two_row_claims_for_table if kind == "two-row" else claims_for_table
    pages = [tuple(p) for p in meta["evidence_pages"]]
    contexts = page_contexts(chunks, pages)
    companies = meta["dev_companies"]

    stats: Counter = Counter()
    generic_tables: list[dict] = []
    claims: list[dict] = []
    seen_tables: set[str] = set()
    for doc, page in pages:
        context = contexts.get((doc, page), [])
        by_table: dict[str, list[dict]] = defaultdict(list)
        for c in context:
            if c.get("table_id") and c.get("table_representation"):
                by_table[c["table_id"]].append(c)
        for table_id, table_chunks in by_table.items():
            columns, rows = _table_rows(table_chunks)
            if not rows:
                continue
            stats["tables"] += 1
            if any(_GENERIC.match(col) for col in columns):
                eligible = sum(1 for r in rows if _eligible(r) and any(_numeric(r, c) for c in columns))
                generic_tables.append({"doc": doc, "page": page, "table_id": table_id,
                                       "columns": columns, "eligible_rows": eligible})
                stats["generic_tables"] += 1
                stats["generic_eligible_rows"] += eligible
                continue
            if table_id in seen_tables:  # a table on two evidence pages: claims once
                stats["tables_repeated"] += 1
                continue
            seen_tables.add(table_id)
            company = meta["company_of"].get(doc, doc)
            others = [c for c in companies if c != company]
            for claim in make(table_id, rows, columns, company, others):
                claims.append({**claim, "doc": doc, "page": page, "company": company})

    # Contexts, shuffled per claim; generation slips (a figure grounding
    # cannot find at all) dropped.
    kept = []
    for claim in claims:
        rows_key = f"r{claim['row']}" + (f"+r{claim['row2']}" if "row2" in claim else "")
        claim["id"] = (f"{claim['doc']}|p{claim['page']}|{claim['table_id'][-4:]}|{rows_key}|"
                       f"{claim['year']}|{claim['label'][:3]}-{claim['kind']}")
        context = list(contexts[(claim["doc"], claim["page"])])
        random.Random(f"{SEED}:{claim['id']}").shuffle(context)
        claim["context_ids"] = [c["chunk_id"] for c in context]
        claim["answer"] = f"{claim['claim']} [📄 {claim['doc']}.pdf | p. {claim['page']}]"
        pieces = split_claims(claim["answer"])
        checks = ground_texts([p.text for p in pieces], ContextIndex(context))
        if any(n_.status == "unsupported" for group in checks for n_ in group):
            stats[f"dropped_unfound_{claim['label']}"] += 1
            continue
        stats[f"claims_{claim['label']}"] += 1
        stats[f"kind_{claim['label']}_{claim['kind']}"] += 1
        kept.append(claim)

    with open(out / "claims.jsonl", "w", encoding="utf-8") as f:
        for claim in kept:
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    used = {(c["doc"], c["page"]) for c in kept}
    with gzip.open(out / "pages.jsonl.gz", "wt", encoding="utf-8") as f:
        for (doc, page) in sorted(used):
            f.write(json.dumps({"doc": doc, "page": page, "chunks": contexts[(doc, page)]},
                               ensure_ascii=False) + "\n")
    summary = {"stats": dict(sorted(stats.items())), "generic_tables": generic_tables,
               "filings_with_claims": len({c["doc"] for c in kept}), "pages_with_claims": len(used)}
    (out / "build_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def load_cases(out: Path) -> list[dict]:
    """The built claims, each with its context chunks in its own order."""
    pages = {}
    with gzip.open(out / "pages.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pages[(rec["doc"], rec["page"])] = {c["chunk_id"]: c for c in rec["chunks"]}
    cases = []
    for line in open(out / "claims.jsonl", encoding="utf-8"):
        claim = json.loads(line)
        by_id = pages[(claim["doc"], claim["page"])]
        cases.append({**claim, "chunks": [by_id[i] for i in claim["context_ids"]]})
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.figure_veto_set", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("build")
    p.add_argument("--out", required=True)
    p.add_argument("--kind", choices=("single", "two-row"), default="single")
    p.add_argument("--source", help="extract's output directory (default: --out)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.cmd == "extract":
        print(json.dumps(extract(Path(args.store), Path(args.out)), indent=1))
    else:
        summary = build(Path(args.out), args.kind, Path(args.source) if args.source else None)
        print(json.dumps(summary["stats"], indent=1))
        print("generic-header tables:", len(summary["generic_tables"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
