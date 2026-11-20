"use client";

import { useCallback, useEffect, useState } from "react";
import { ListChecks, TriangleAlert } from "lucide-react";

import { getChecklist } from "@/lib/api";
import { prettyFilename } from "@/lib/citations";
import type {
  ChecklistCell,
  ChecklistColumn,
  ChecklistPassage,
  ChecklistReport,
  ChecklistRow,
  ChecklistStatus,
} from "@/lib/types";

/*
 * The checklist review grid (Decision 95): the deal's contracts against a
 * diligence checklist. Each cell is the passage of that contract that best
 * answers the item and a status; selecting a cell shows the passage. File
 * names, sections and passages are document text from the data room, so they
 * are rendered as text only and were clipped on arrival (lib/normalize.ts).
 */

type LoadState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "error" }
  | { phase: "ready"; report: ChecklistReport };

interface ChecklistPanelProps {
  dealId: string | null;
  /** Fetch only once the panel is shown: the grid reranks every contract. */
  active?: boolean;
  /** Replaces the API call; for tests. */
  load?: (dealId: string) => Promise<ChecklistReport | null>;
}

const REASONS: Record<string, string> = {
  document_cap: "the deal has more contracts than one grid checks",
  time_budget: "the grid ran out of time",
};

// Column headings: the grid is narrow, the item's full name is its title.
const SHORT: Record<string, string> = {
  change_of_control: "CoC",
  anti_assignment: "Asg",
  non_compete: "NC",
  exclusivity: "Exc",
  no_solicit_of_customers: "NSC",
  most_favoured_nation: "MFN",
  right_of_first_refusal: "RFR",
  termination_for_convenience: "TfC",
  cap_on_liability: "Cap",
  uncapped_liability: "Unc",
  liquidated_damages: "LD",
  insurance: "Ins",
  ip_ownership_assignment: "IP",
  governing_law: "Law",
  audit_rights: "Aud",
};

const STATUS_TEXT: Record<ChecklistStatus, string> = {
  found: "found",
  unclear: "unclear",
  not_found: "not found",
};

const STATUS_STYLE: Record<ChecklistStatus, string> = {
  found: "bg-ok/80 text-ink-950",
  unclear: "bg-warn/25 text-warn",
  not_found: "bg-ink-750 text-ash-600",
};

const STATUS_MARK: Record<ChecklistStatus, string> = { found: "●", unclear: "?", not_found: "·" };

function shortLabel(item: ChecklistColumn): string {
  return (
    SHORT[item.id] ??
    item.label
      .split(/[\s-]+/)
      .map((w) => w.charAt(0).toUpperCase())
      .join("")
      .slice(0, 4)
  );
}

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;
}

function where(passage: ChecklistPassage): string {
  const parts: string[] = [];
  if (passage.page_number !== null) parts.push(`p. ${passage.page_number}`);
  if (passage.section_heading) parts.push(passage.section_heading);
  return parts.join(" · ");
}

function Passage({ passage, label }: { passage: ChecklistPassage; label: string }) {
  const location = where(passage);
  return (
    <figure className="mt-2">
      <blockquote
        aria-label={label}
        className="whitespace-pre-wrap border-l-2 border-gold-700 pl-2.5 text-[0.74rem] leading-relaxed text-ash-50"
      >
        {passage.text}
      </blockquote>
      {location && <figcaption className="mt-1 text-[0.66rem] text-ash-500">{location}</figcaption>}
    </figure>
  );
}

function CellDetail({ row, item, cell }: { row: ChecklistRow; item: ChecklistColumn; cell: ChecklistCell }) {
  const verdict =
    cell.status === "found"
      ? "Found"
      : cell.status === "unclear"
        ? "Unclear — read the passage"
        : "Not found — the closest passage is shown";
  return (
    <section aria-label="Selected cell" className="border-t border-line px-4 py-3">
      <p className="text-[0.75rem] text-ash-300">
        <span className="font-medium text-ash-50">{item.label}</span> ·{" "}
        <span title={row.source_file}>{prettyFilename(row.source_file)}</span>
      </p>
      <p className="mt-0.5 text-[0.68rem] leading-snug text-ash-500">{item.question}</p>
      <p
        className={`mt-1.5 text-[0.7rem] font-semibold uppercase tracking-wider ${
          cell.status === "found" ? "text-ok" : cell.status === "unclear" ? "text-warn" : "text-ash-500"
        }`}
      >
        {verdict}
      </p>
      {cell.passage ? (
        <Passage passage={cell.passage} label="Evidence passage" />
      ) : (
        <p className="mt-1.5 text-[0.72rem] text-ash-500">No passage of this contract was retrieved.</p>
      )}
      {cell.alternates.length > 0 && (
        <details className="mt-2 text-[0.7rem] text-ash-300">
          <summary className="cursor-pointer">
            {plural(cell.alternates.length, "other passage")}
          </summary>
          {cell.alternates.map((alternate, i) => (
            <Passage key={`${alternate.chunk_id}-${i}`} passage={alternate} label="Other passage" />
          ))}
        </details>
      )}
    </section>
  );
}

function Grid({ report }: { report: ChecklistReport }) {
  const [selected, setSelected] = useState<{ row: number; item: number } | null>(null);
  const totals = report.items.reduce(
    (acc, item) => ({
      found: acc.found + item.found,
      unclear: acc.unclear + item.unclear,
      not_found: acc.not_found + item.not_found,
    }),
    { found: 0, unclear: 0, not_found: 0 },
  );
  const pick = selected && report.rows[selected.row];

  return (
    <div>
      <div className="border-b border-line px-4 py-3 text-[0.75rem] leading-relaxed text-ash-300">
        {plural(report.rows.length, "contract")} × {plural(report.items.length, "item")}
        {" · "}
        <span className="text-ok">{totals.found} found</span> · <span className="text-warn">{totals.unclear} unclear</span>{" "}
        · {totals.not_found} not found
        {report.truncated && (
          <div role="status" className="mt-1.5 flex items-start gap-1 text-[0.7rem] text-warn">
            <TriangleAlert size={11} className="mt-0.5 shrink-0" aria-hidden />
            <span>
              Partial: checked {report.documents_checked} of{" "}
              {report.documents_checked + report.documents_not_checked.length} contracts —{" "}
              {REASONS[report.truncation_reason] ?? "the grid was cut short"}.
              {report.documents_not_checked.length > 0 &&
                ` Not checked: ${report.documents_not_checked.map(prettyFilename).join(", ")}.`}
            </span>
          </div>
        )}
      </div>

      {report.rows.length === 0 ? (
        <p className="px-4 py-4 text-[0.78rem] leading-relaxed text-ash-500">
          {report.documents_total
            ? "No document of this deal reads as a contract."
            : "Nothing is indexed for this deal yet."}
        </p>
      ) : (
        <div className="overflow-x-auto px-2 py-2">
          <table className="border-separate border-spacing-0.5 text-[0.62rem]">
            <caption className="sr-only">Checklist: contracts by item</caption>
            <thead>
              <tr>
                <th scope="col" className="sr-only">
                  Contract
                </th>
                {report.items.map((item) => (
                  <th key={item.id} scope="col" className="px-0.5 font-mono font-normal text-ash-500">
                    <abbr title={item.label} className="no-underline">
                      {shortLabel(item)}
                    </abbr>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {report.rows.map((row, r) => (
                <tr key={row.doc_id || `${row.source_file}-${r}`}>
                  <th
                    scope="row"
                    title={row.source_file}
                    className="max-w-[8rem] truncate pr-1.5 text-left text-[0.68rem] font-normal text-ash-300"
                  >
                    {prettyFilename(row.source_file)}
                  </th>
                  {row.cells.map((cell, i) => {
                    const item = report.items[i];
                    const isSelected = selected?.row === r && selected?.item === i;
                    return (
                      <td key={cell.item_id}>
                        <button
                          type="button"
                          onClick={() => setSelected(isSelected ? null : { row: r, item: i })}
                          aria-pressed={isSelected}
                          aria-label={`${item.label} in ${prettyFilename(row.source_file)}: ${STATUS_TEXT[cell.status]}`}
                          title={`${item.label}: ${STATUS_TEXT[cell.status]}`}
                          className={`h-5 w-6 rounded text-[0.65rem] leading-none ${STATUS_STYLE[cell.status]} ${
                            isSelected ? "ring-1 ring-gold-400" : ""
                          }`}
                        >
                          {STATUS_MARK[cell.status]}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1.5 px-1 text-[0.62rem] text-ash-500">
            ● found · ? unclear · · not found. Select a cell for its passage.
          </p>
        </div>
      )}

      {pick && selected && (
        <CellDetail row={pick} item={report.items[selected.item]} cell={pick.cells[selected.item]} />
      )}

      {report.excluded_total > 0 && (
        <p className="border-t border-line px-4 py-3 text-[0.68rem] leading-relaxed text-ash-500">
          Not in the grid: {plural(report.excluded_total, "document")} that{" "}
          {report.excluded_total === 1 ? "is not a contract" : "are not contracts"}
          {report.documents_excluded.length > 0 &&
            ` (${report.documents_excluded
              .map((d) => `${prettyFilename(d.source_file)}${d.document_category ? `, ${d.document_category}` : ""}`)
              .join("; ")})`}
          .
        </p>
      )}
    </div>
  );
}

export default function ChecklistPanel({ dealId, active = true, load = getChecklist }: ChecklistPanelProps) {
  const [state, setState] = useState<LoadState>({ phase: "idle" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!active || !dealId) return;
    // A response for a deal the reader has left is dropped.
    let current = true;
    setState({ phase: "loading" });
    load(dealId)
      .then((report) => {
        if (!current) return;
        setState(report ? { phase: "ready", report } : { phase: "error" });
      })
      .catch(() => current && setState({ phase: "error" }));
    return () => {
      current = false;
    };
  }, [active, dealId, load, attempt]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  if (!dealId) {
    return <p className="px-4 py-5 text-[0.8rem] text-ash-500">Select a deal to check its contracts.</p>;
  }
  if (state.phase === "ready") return <Grid report={state.report} />;
  if (state.phase === "error") {
    return (
      <div role="alert" className="px-4 py-5 text-[0.8rem] leading-relaxed text-ash-300">
        The checklist grid could not be loaded.{" "}
        <button
          type="button"
          onClick={retry}
          className="font-medium text-gold-400 underline-offset-2 hover:underline"
        >
          Try again
        </button>
      </div>
    );
  }
  return (
    <p className="flex items-center gap-2 px-4 py-5 text-[0.8rem] text-ash-500" role="status">
      <ListChecks size={12} aria-hidden />
      Checking every contract against the checklist…
    </p>
  );
}
