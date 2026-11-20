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
          --kind grounding (Decision 94) builds claims for numeric grounding
          from the converter tables of the dev filings that touch none of
          Decision 90's evidence pages, so none of the tables Decision 90's
          sets were built from. Per sampled (row, fiscal year) cell: the
          figure as printed, and — in a currency row of a table whose
          converter states a scale — the same with its scale word ("$1,488.4
          million"); wrong by construction: another year's figure, the
          neighbouring row's figure (each in both forms), the figure with
          another scale word, and a per-share figure stated in the table's
          scale. Only full fiscal-year columns are used (the converter's own
          registry_period), and no claim is dropped for what grounding does
          with it: grounding is what this set measures.
          --kind grounding-cds builds, on the same tables and cells, the
          kinds grounding cannot see, as Decision 90 built them: the filing's
          own company or another dev company named, the true or the opposite
          direction against the row's prior year, and the side of an "income
          (loss)" pair the printed sign selects or the other.

    python -m eval.figure_veto_set build --kind grounding --source D:/redline_data/veto10 \
        --out D:/redline_data/ground11/set
    python -m eval.figure_veto_set build --kind grounding-cds --source D:/redline_data/veto10 \
        --out D:/redline_data/ground11/set_cds

The plans, written before anything was scored, are in Decisions 90 and 94.
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
# --kind grounding (Decision 94): its own seed, and fewer cells from each of
# many more tables.
GROUNDING_SEED = 20261120
GROUNDING_CELLS_PER_TABLE = 4
_SCALE_WORDS = {"thousands": "thousand", "millions": "million", "billions": "billion"}
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


_PRINTED_FIGURE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def fiscal_year_columns(columns: list[str]) -> dict[str, int]:
    """column_years, kept to the columns the converter reads as one full
    fiscal year of actuals (registry_period "FYyyyy"): a quarter's or a
    budget's figure is not "fiscal 2019"'s."""
    from src.data_processing.financial_table_converter import registry_period

    return {col: year for col, year in column_years(columns).items()
            if registry_period(col) == f"FY{year}"}


def _printed(row: dict, col: str) -> dict | None:
    """_numeric, and the cell prints one figure of its row's kind."""
    cell = _numeric(row, col)
    if cell is None:
        return None
    digits = re.sub(r"[^\d.,]", "", cell["raw"]).strip(".,")
    kind = "percent" if row.get("unit") == "percent" else "currency"
    if cell.get("kind") != kind or not _PRINTED_FIGURE.fullmatch(digits):
        return None
    return cell


def grounding_claims_for_table(table_id: str, rows: list[dict], columns: list[str],
                               scale_label: str) -> list[dict]:
    """
    Claims for numeric grounding from one table (--kind grounding).

    For up to GROUNDING_CELLS_PER_TABLE seeded (row, fiscal year) cells:
        supported/plain           "In fiscal Y, L was F."  (as printed)
        supported/scaled          "In fiscal Y, L was F million."  (currency
                                  row of a table stated in millions; the
                                  table's own scale word)
        contradicted/period       another year's figure of the row
        contradicted/period_scaled    the same with the scale word
        contradicted/neighbour    the neighbouring row's figure, same column
        contradicted/neighbour_scaled the same with the scale word
        contradicted/scale        the figure with another scale word
        contradicted/unit         a per-share row's figure stated in the
                                  table's scale
    """
    years = fiscal_year_columns(columns)
    word = _SCALE_WORDS.get(scale_label, "")
    cells = [(i, col) for i, row in enumerate(rows) if _eligible(row)
             for col in years if _printed(row, col)]
    rng = random.Random(f"{GROUNDING_SEED}:{table_id}")
    out = []
    for i, col in sorted(rng.sample(cells, min(GROUNDING_CELLS_PER_TABLE, len(cells)))):
        row, cell = rows[i], _printed(rows[i], col)
        unit, label, year = row["unit"], clean_label(row["line_item"]), years[col]
        printed_in_row = {c.get("raw") for c in row["values"].values()}
        base = {"table_id": table_id, "row": i, "column": col, "line_item": row["line_item"],
                "year": year, "figure": cell["raw"], "unit": unit, "scale_label": scale_label}
        scaled = bool(word) and unit == "currency"

        def add(label_: str, kind: str, figure: str, base: dict = base, line: str = label, **extra) -> None:
            out.append({**base, "label": label_, "kind": kind,
                        "claim": f"In fiscal {base['year']}, {line} was {figure}.", **extra})

        figure = render(cell, unit)
        add("supported", "plain", figure)
        if scaled:
            add("supported", "scaled", f"{figure} {word}")
            others = [w for w in _SCALE_WORDS.values() if w != word]
            wrong = random.Random(f"{GROUNDING_SEED}:{table_id}:{i}:{col}:scale").choice(others)
            add("contradicted", "scale", f"{figure} {wrong}", stated_scale=wrong)
        if word and unit == "per_share":
            add("contradicted", "unit", f"{figure} {word}")

        # Another year's figure of the same row, attached to this year.
        for other in sorted((c for c in years if c != col and _printed(row, c)),
                            key=lambda c: (abs(years[c] - year), years[c] > year)):
            alt = _printed(row, other)
            if alt["raw"] != cell["raw"]:
                add("contradicted", "period", render(alt, unit), source_column=other)
                if scaled:
                    add("contradicted", "period_scaled", f"{render(alt, unit)} {word}", source_column=other)
                break

        # The neighbouring row's figure in the same column, attached to this row.
        for k in [*range(i + 1, len(rows)), *range(i - 1, -1, -1)]:
            other_row = rows[k]
            alt = _printed(other_row, col) if _eligible(other_row) else None
            if alt and other_row["unit"] == unit and alt["raw"] not in printed_in_row:
                add("contradicted", "neighbour", render(alt, unit), source_row=k)
                if scaled:
                    add("contradicted", "neighbour_scaled", f"{render(alt, unit)} {word}", source_row=k)
                break
    return out


def cds_claims_for_table(table_id: str, rows: list[dict], columns: list[str], company: str,
                         others: list[str]) -> list[dict]:
    """
    Claims of the kinds grounding cannot see (--kind grounding-cds), in
    Decision 90's construction:
        supported/company, contradicted/company     the filing's own company, or
                                                    another dev company, named
        supported/direction, contradicted/direction  the true or the opposite
                                                    verb against the row's prior
                                                    year (both printed positive)
        supported/sign, contradicted/sign           the side of an "income
                                                    (loss)" pair the printed
                                                    sign selects, or the other
    Company and direction claims are made on the seeded cells of
    grounding_claims_for_table; sign claims on every signed cell of the table,
    since the seeded cells hold few (11 pairs on the set's 337 tables, 52 with
    every cell; decided before anything was scored).
    """
    years = fiscal_year_columns(columns)
    cells = [(i, col) for i, row in enumerate(rows) if _eligible(row)
             for col in years if _printed(row, col)]
    rng = random.Random(f"{GROUNDING_SEED}:{table_id}")
    seeded = set(rng.sample(cells, min(GROUNDING_CELLS_PER_TABLE, len(cells))))
    signed = {(i, col) for i, col in cells if sign_sides(clean_label(rows[i]["line_item"]))}
    out = []
    for i, col in sorted(seeded | signed):
        row, cell = rows[i], _printed(rows[i], col)
        unit, label, year = row["unit"], clean_label(row["line_item"]), years[col]
        base = {"table_id": table_id, "row": i, "column": col, "line_item": row["line_item"],
                "year": year, "figure": cell["raw"], "unit": unit}

        def add(label_: str, kind: str, text: str, base: dict = base, **extra) -> None:
            out.append({**base, "label": label_, "kind": kind, "claim": text, **extra})

        figure = render(cell, unit)
        if (i, col) in seeded:
            add("supported", "company", f"{company} reported {label} of {figure} in fiscal {year}.")
            swap = random.Random(f"{GROUNDING_SEED}:{table_id}:{i}:{col}:company").choice(others)
            add("contradicted", "company", f"{swap} reported {label} of {figure} in fiscal {year}.", named=swap)

        prior = next((c for c, y in years.items() if y == year - 1), None)
        prev = _printed(row, prior) if prior else None
        if (i, col) in seeded and prev and cell["raw_value"] > 0 and prev["raw_value"] > 0 \
                and prev["raw_value"] != cell["raw_value"]:
            true, false = ("rose", "fell") if cell["raw_value"] > prev["raw_value"] else ("fell", "rose")
            head = f"{label[:1].upper()}{label[1:]}"
            add("supported", "direction", f"{head} {true} to {figure} in fiscal {year}.", prior=prev["raw"])
            add("contradicted", "direction", f"{head} {false} to {figure} in fiscal {year}.", prior=prev["raw"])

        sides = sign_sides(label)
        if sides and cell["raw_value"] != 0:
            magnitude = render({**cell, "raw_value": abs(cell["raw_value"])}, unit)
            true, false = (sides[0], sides[1]) if cell["raw_value"] > 0 else (sides[1], sides[0])
            add("supported", "sign", f"In fiscal {year}, {true} was {magnitude}.")
            add("contradicted", "sign", f"In fiscal {year}, {false} was {magnitude}.")
    return out


def build_grounding(out: Path, source: Path, kind: str = "grounding") -> dict:
    """
    --kind grounding (or grounding-cds): claims.jsonl, pages.jsonl.gz and
    build_summary.json under `out`, from the tables of the dev filings that
    touch none of Decision 90's evidence pages. No grounding is run: what
    grounding does with a claim is what the set measures, so none is dropped
    for it.
    """
    from src.verification.claims import split_claims

    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "fb_dev_meta.json").read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in gzip.open(source / "fb_dev_chunks.jsonl.gz", "rt", encoding="utf-8")]
    excluded = {tuple(p) for p in meta["evidence_pages"]}

    def doc_of(chunk: dict) -> str:
        name = chunk["source_file"]
        return name[:-4] if name.lower().endswith(".pdf") else name

    tables: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if c.get("table_id") and c.get("table_representation"):
            tables[c["table_id"]].append(c)

    stats: Counter = Counter()
    claims: list[dict] = []
    wanted_pages: set[tuple[str, int]] = set()
    for table_id in sorted(tables):
        table_chunks = tables[table_id]
        doc = doc_of(table_chunks[0])
        pages = set().union(*(chunk_pages(c) for c in table_chunks))
        columns, rows = _table_rows(table_chunks)
        if not rows or not pages:
            continue
        stats["tables"] += 1
        if any((doc, p) in excluded for p in pages):
            stats["tables_on_decision90_pages"] += 1
            continue
        if any(_GENERIC.match(col) for col in columns):
            stats["tables_generic_columns"] += 1
            continue
        if len(fiscal_year_columns(columns)) < 2:
            stats["tables_without_two_fiscal_years"] += 1
            continue
        scale_label = next((c.get("scale_label") for c in table_chunks
                            if c.get("table_representation") == "row_by_row"), "") or ""
        company = meta["company_of"].get(doc, doc)
        if kind == "grounding-cds":
            others = [c for c in meta["dev_companies"] if c != company]
            made = cds_claims_for_table(table_id, rows, columns, company, others)
        else:
            made = grounding_claims_for_table(table_id, rows, columns, scale_label)
        if not made:
            stats["tables_without_eligible_cells"] += 1
            continue
        stats["tables_used"] += 1
        stats[f"tables_scale_{scale_label or 'none'}"] += 1
        page = min(pages)
        wanted_pages.add((doc, page))
        for claim in made:
            claims.append({**claim, "doc": doc, "page": page, "company": company})

    contexts = page_contexts(chunks, sorted(wanted_pages))
    kept = []
    for claim in claims:
        claim["id"] = (f"{claim['doc']}|p{claim['page']}|{claim['table_id'][-4:]}|r{claim['row']}|"
                       f"{claim['year']}|{claim['label'][:3]}-{claim['kind']}")
        context = list(contexts[(claim["doc"], claim["page"])])
        random.Random(f"{GROUNDING_SEED}:{claim['id']}").shuffle(context)
        claim["context_ids"] = [c["chunk_id"] for c in context]
        claim["answer"] = f"{claim['claim']} [📄 {claim['doc']}.pdf | p. {claim['page']}]"
        if len(split_claims(claim["answer"])) != 1:  # a generation slip, by construction
            stats["dropped_not_one_claim"] += 1
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
    summary = {"stats": dict(sorted(stats.items())), "filings_with_claims": len({c["doc"] for c in kept}),
               "pages_with_claims": len(used), "tables_with_claims": len({c["table_id"] for c in kept}),
               "excluded_pages": len(excluded)}
    (out / "build_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


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
    p.add_argument("--kind", choices=("single", "two-row", "grounding", "grounding-cds"), default="single")
    p.add_argument("--source", help="extract's output directory (default: --out)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.cmd == "extract":
        print(json.dumps(extract(Path(args.store), Path(args.out)), indent=1))
    elif args.kind in ("grounding", "grounding-cds"):
        summary = build_grounding(Path(args.out), Path(args.source) if args.source else Path(args.out), args.kind)
        print(json.dumps(summary, indent=1))
    else:
        summary = build(Path(args.out), args.kind, Path(args.source) if args.source else None)
        print(json.dumps(summary["stats"], indent=1))
        print("generic-header tables:", len(summary["generic_tables"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
