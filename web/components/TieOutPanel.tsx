"use client";

import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, ChevronDown, Scale, TriangleAlert } from "lucide-react";

import { getTieOut } from "@/lib/api";
import { prettyFilename } from "@/lib/citations";
import type { TieOutFigure, TieOutReport, TieOutValue } from "@/lib/types";

/*
 * The deal-level figure tie-out (Decision 92): every labelled figure two or
 * more documents state, conflicts first. Everything shown here — labels,
 * figures, file names, sections — is document text from the data room, so it
 * is rendered as text only and was clipped on arrival (lib/normalize.ts).
 */

type LoadState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "error" }
  | { phase: "ready"; report: TieOutReport };

interface TieOutPanelProps {
  dealId: string | null;
  /** Fetch only once the panel is shown: the report reads the whole deal. */
  active?: boolean;
  /** Replaces the API call; for tests. */
  load?: (dealId: string) => Promise<TieOutReport | null>;
}

const REASONS: Record<string, string> = {
  chunk_cap: "the deal is larger than one report reads",
  time_budget: "the report ran out of time",
};

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;
}

function where(value: TieOutValue): string {
  if (value.page_number !== null) return `p. ${value.page_number}`;
  return value.section_heading;
}

function ValueRow({ value }: { value: TieOutValue }) {
  const location = where(value);
  return (
    <li className="flex items-baseline justify-between gap-3 py-0.5">
      <span className="min-w-0 truncate text-ash-300" title={value.source_file}>
        {prettyFilename(value.source_file)}
        {location && <span className="text-ash-500"> · {location}</span>}
      </span>
      <span
        className="shrink-0 font-mono tabular-nums text-ash-50"
        title={value.scale_known ? undefined : "No unit stated: compared at any power of 1,000"}
      >
        {value.as_printed}
      </span>
    </li>
  );
}

function FigureItem({ figure }: { figure: TieOutFigure }) {
  const conflict = figure.status === "conflict";
  return (
    <li className="px-4 py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <span className="text-[0.8rem] font-medium leading-snug text-ash-50">{figure.metric}</span>
          <span className="ml-2 rounded bg-ink-750 px-1.5 py-0.5 font-mono text-[0.62rem] text-ash-300">
            {figure.period}
          </span>
        </div>
        {conflict ? (
          <span className="shrink-0 rounded bg-bad/10 px-1.5 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wider text-bad">
            {figure.max_deviation_pct.toFixed(1)}% apart
          </span>
        ) : (
          <CheckCircle2 size={12} className="mt-0.5 shrink-0 text-ok" aria-label="agrees" />
        )}
      </div>
      <ul className="mt-1.5 text-[0.7rem]">
        {figure.values.map((value, i) => (
          <ValueRow key={`${value.source_file}-${i}`} value={value} />
        ))}
      </ul>
      {figure.more_documents > 0 && (
        <p className="mt-0.5 text-[0.66rem] text-ash-500">
          and {plural(figure.more_documents, "more document")}
        </p>
      )}
    </li>
  );
}

function Report({ report }: { report: TieOutReport }) {
  const [showAgreements, setShowAgreements] = useState(false);
  const notCompared = [
    report.single_source_figures > 0 &&
      `${plural(report.single_source_figures, "figure")} only one document states`,
    report.generic_figures > 0 &&
      `${plural(report.generic_figures, "figure")} under labels like "Total" or "Other"`,
    report.ambiguous_figures > 0 &&
      `${plural(report.ambiguous_figures, "figure")} a document states twice with different values`,
    report.figures_without_period > 0 &&
      `${plural(report.figures_without_period, "figure")} with no readable period`,
  ].filter(Boolean);

  return (
    <div>
      <div className="border-b border-line px-4 py-3 text-[0.75rem] leading-relaxed text-ash-300">
        <span className={report.conflicts_total ? "font-medium text-bad" : "text-ash-300"}>
          {plural(report.conflicts_total, "conflict")}
        </span>
        {" · "}
        {plural(report.agreements_total, "agreement")} across{" "}
        {plural(report.documents_read, "document")}
        {report.truncated && (
          <div role="status" className="mt-1.5 flex items-start gap-1 text-[0.7rem] text-warn">
            <TriangleAlert size={11} className="mt-0.5 shrink-0" aria-hidden />
            <span>
              Partial: read {report.documents_read} of {report.documents_total} documents —{" "}
              {REASONS[report.truncation_reason] ?? "the read was cut short"}.
              {report.documents_not_read.length > 0 &&
                ` Not read: ${report.documents_not_read.map(prettyFilename).join(", ")}.`}
            </span>
          </div>
        )}
      </div>

      {report.conflicts.length > 0 ? (
        <ul className="divide-y divide-line" aria-label="Conflicting figures">
          {report.conflicts.map((figure, i) => (
            <FigureItem key={`c-${figure.metric}-${figure.period}-${i}`} figure={figure} />
          ))}
        </ul>
      ) : (
        <p className="px-4 py-4 text-[0.78rem] leading-relaxed text-ash-500">
          {report.agreements_total
            ? "Every figure two documents both state agrees."
            : "No figure is stated by two documents of this deal."}
        </p>
      )}
      {report.conflicts_total > report.conflicts.length && (
        <p className="px-4 pb-2 text-[0.66rem] text-ash-500">
          Showing the {report.conflicts.length} largest of {report.conflicts_total} conflicts.
        </p>
      )}

      {report.agreements.length > 0 && (
        <div className="border-t border-line">
          <button
            type="button"
            onClick={() => setShowAgreements((v) => !v)}
            aria-expanded={showAgreements}
            className="flex w-full items-center justify-between px-4 py-2.5 text-[0.75rem] text-ash-300 hover:text-ash-50"
          >
            <span>
              {showAgreements ? "Hide" : "Show"} {plural(report.agreements_total, "agreement")}
            </span>
            <ChevronDown
              size={12}
              className={`transition-transform ${showAgreements ? "rotate-180" : ""}`}
              aria-hidden
            />
          </button>
          {showAgreements && (
            <ul className="divide-y divide-line" aria-label="Agreeing figures">
              {report.agreements.map((figure, i) => (
                <FigureItem key={`a-${figure.metric}-${figure.period}-${i}`} figure={figure} />
              ))}
            </ul>
          )}
        </div>
      )}

      {notCompared.length > 0 && (
        <p className="border-t border-line px-4 py-3 text-[0.68rem] leading-relaxed text-ash-500">
          Not compared: {notCompared.join("; ")}.
        </p>
      )}
    </div>
  );
}

export default function TieOutPanel({ dealId, active = true, load = getTieOut }: TieOutPanelProps) {
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
    return <p className="px-4 py-5 text-[0.8rem] text-ash-500">Select a deal to tie out its figures.</p>;
  }
  if (state.phase === "ready") return <Report report={state.report} />;
  if (state.phase === "error") {
    return (
      <div role="alert" className="px-4 py-5 text-[0.8rem] leading-relaxed text-ash-300">
        The tie-out report could not be loaded.{" "}
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
      <Scale size={12} aria-hidden />
      Tying out the figures across the data room…
    </p>
  );
}
