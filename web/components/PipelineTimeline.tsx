"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Brain,
  Check,
  ChevronDown,
  Calculator,
  CloudOff,
  FileSearch,
  Gauge,
  Hand,
  Minus,
  PenLine,
  RefreshCw,
  ShieldCheck,
  X,
} from "lucide-react";

import type { PipelineStage } from "@/lib/types";

const AGENT_ICONS: Record<string, React.ElementType> = {
  query_intelligence: Brain,
  retrieval_executor: FileSearch,
  financial_verifier: Calculator,
  quality_assessor: Gauge,
  query_rewriter: RefreshCw,
  answer_synthesizer: PenLine,
  retry_synthesis: PenLine,
  hallucination_validator: ShieldCheck,
  insufficient_context: Hand,
  synthesis_unavailable: CloudOff,
};

/** "12 supported, 2 unverified" from the validator's status counts. */
function countsLine(value: unknown): string {
  if (!value || typeof value !== "object") return "";
  return Object.entries(value as Record<string, unknown>)
    .filter(([, n]) => typeof n === "number" && n > 0)
    .map(([k, n]) => `${n} ${k}`)
    .join(", ");
}

/** Renders the structured `detail` bag a stage carries, when it has anything worth showing. */
function StageDetail({ stage }: { stage: PipelineStage }) {
  const d = stage.detail ?? {};
  const rows: [string, string][] = [];

  const subQuestions = d.sub_questions as string[] | undefined;
  const sources = d.sources as string[] | undefined;
  const missing = d.missing_aspects as string[] | undefined;

  if (typeof d.model === "string" && d.model) rows.push(["Model", d.model]);
  if (typeof d.query_type === "string") rows.push(["Query type", d.query_type]);
  if (typeof d.reranked_count === "number")
    rows.push(["Chunks after rerank", String(d.reranked_count)]);
  if (typeof d.retrieval_passes === "number")
    rows.push(["Retrieval passes", String(d.retrieval_passes)]);
  if (typeof d.figures === "number") rows.push(["Figures normalised", String(d.figures)]);
  if (typeof d.score === "number") rows.push(["Quality score", d.score.toFixed(2)]);
  if (typeof d.method === "string") rows.push(["Scoring method", d.method]);
  if (typeof d.citation_count === "number")
    rows.push(["Citations", String(d.citation_count)]);
  if (typeof d.confidence_score === "number")
    rows.push(["Confidence", `${Math.round(d.confidence_score * 100)}%`]);
  if (typeof d.rewritten_query === "string" && d.rewritten_query)
    rows.push(["Rewritten as", d.rewritten_query]);
  const claims = countsLine(d.status_counts);
  if (claims) rows.push(["Statements", claims]);
  const methods = countsLine(d.method_counts);
  if (methods) rows.push(["Decided by", methods]);

  // The node's own measurement, when the server sent one.
  const perf = stage.perf;
  if (perf && perf.llm_calls > 0) {
    rows.push(["LLM calls", String(perf.llm_calls)]);
    const tokens = perf.prompt_tokens + perf.completion_tokens;
    if (tokens > 0) rows.push(["Tokens", tokens.toLocaleString()]);
    if (perf.models.length && !rows.some(([k]) => k === "Model"))
      rows.push(["Model", perf.models.join(", ")]);
  }

  const hasLists =
    (subQuestions?.length ?? 0) > 0 ||
    (sources?.length ?? 0) > 0 ||
    (missing?.length ?? 0) > 0;

  if (!rows.length && !hasLists) {
    return (
      <p className="text-[0.78rem] text-ash-500">
        This step reported no additional detail.
      </p>
    );
  }

  return (
    <div className="space-y-3.5">
      {rows.length > 0 && (
        <dl className="grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
          {rows.map(([k, v]) => (
            <div key={k} className="flex items-baseline justify-between gap-3">
              <dt className="text-[0.72rem] uppercase tracking-wider text-ash-500">
                {k}
              </dt>
              <dd className="truncate text-right text-[0.8rem] font-medium tabular-nums text-ash-50">
                {v}
              </dd>
            </div>
          ))}
        </dl>
      )}

      {subQuestions && subQuestions.length > 0 && (
        <div>
          <div className="mb-1.5 text-[0.68rem] font-semibold uppercase tracking-widest text-ash-500">
            Sub-questions retrieved separately
          </div>
          <ol className="space-y-1">
            {subQuestions.map((q, i) => (
              <li
                key={i}
                className="flex gap-2 text-[0.8rem] leading-snug text-ash-300"
              >
                <span className="shrink-0 font-mono text-[0.7rem] text-gold-600">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span>{q}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {sources && sources.length > 0 && (
        <div>
          <div className="mb-1.5 text-[0.68rem] font-semibold uppercase tracking-widest text-ash-500">
            Documents reached
          </div>
          <div className="flex flex-wrap gap-1.5">
            {sources.map((s) => (
              <span
                key={s}
                className="rounded border border-line bg-ink-800 px-2 py-0.5 font-mono text-[0.68rem] text-ash-300"
              >
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {missing && missing.length > 0 && (
        <div>
          <div className="mb-1.5 text-[0.68rem] font-semibold uppercase tracking-widest text-warn">
            Aspects the context did not cover
          </div>
          <ul className="space-y-1">
            {missing.map((m, i) => (
              <li key={i} className="text-[0.8rem] leading-snug text-ash-300">
                {m}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function StageRow({ stage, isLast }: { stage: PipelineStage; isLast: boolean }) {
  const [open, setOpen] = useState(false);
  const Icon = AGENT_ICONS[stage.agent] ?? Brain;

  const done = stage.status === "done";
  const running = stage.status === "running";
  // A stage the run never reached, or the one in flight when it broke. Neither
  // gets a check mark: a check means the stage ran.
  const skipped = stage.status === "skipped";
  const interrupted = stage.status === "interrupted";

  return (
    <li className="relative">
      {/* Connector to the next row. Stops short of the last marker so the
          timeline terminates cleanly instead of trailing into nothing. */}
      {!isLast && (
        <span
          className={`absolute left-[15px] top-8 bottom-0 w-px ${
            done ? "bg-gold-700/45" : "bg-line"
          }`}
          aria-hidden
        />
      )}

      <div
        className={`relative flex gap-3.5 rounded-lg px-2 py-2 transition-colors ${
          running ? "is-running" : ""
        }`}
      >
        <div
          className={`relative z-10 mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border transition-colors ${
            done
              ? "border-gold-600/45 bg-gold-500/12 text-gold-400"
              : running
                ? "pulse-ring border-gold-500 bg-gold-500/18 text-gold-300"
                : interrupted
                  ? "border-bad/45 bg-bad/10 text-bad"
                  : "border-line bg-ink-850 text-ash-600"
          }`}
          aria-label={skipped ? "Not reached" : interrupted ? "Interrupted" : undefined}
        >
          {done ? (
            <Check size={14} />
          ) : interrupted ? (
            <X size={14} />
          ) : skipped ? (
            <Minus size={14} />
          ) : (
            <Icon size={14} />
          )}
        </div>

        <div className="min-w-0 flex-1 pt-0.5">
          <div className="flex items-baseline justify-between gap-3">
            <span
              className={`text-[0.83rem] font-semibold ${
                done || running ? "text-ash-50" : "text-ash-600"
              }`}
            >
              {stage.label}
            </span>
            {done && (
              <span className="shrink-0 font-mono text-[0.68rem] tabular-nums text-ash-500">
                {(stage.duration_ms / 1000).toFixed(1)}s
              </span>
            )}
          </div>

          <p
            className={`mt-0.5 text-[0.79rem] leading-snug ${
              done ? "text-ash-300" : "text-ash-600"
            }`}
          >
            {done
              ? stage.summary
              : skipped
                ? "Not reached — the run ended before this step"
                : interrupted
                  ? "Interrupted — the run stopped during this step"
                  : stage.description}
          </p>

          {done && (
            <>
              <button
                onClick={() => setOpen((v) => !v)}
                className="mt-1.5 inline-flex items-center gap-1 text-[0.7rem] font-medium text-ash-500 transition-colors hover:text-gold-400"
                aria-expanded={open}
              >
                <ChevronDown
                  size={11}
                  className={`transition-transform ${open ? "rotate-180" : ""}`}
                />
                {open ? "Hide" : "Detail"}
              </button>

              <AnimatePresence initial={false}>
                {open && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2, ease: "easeOut" }}
                    className="overflow-hidden"
                  >
                    <div className="mt-2.5 rounded-lg border border-line bg-ink-850/70 p-3">
                      <StageDetail stage={stage} />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </>
          )}
        </div>
      </div>
    </li>
  );
}

interface PipelineTimelineProps {
  stages: PipelineStage[];
  running: boolean;
  elapsedMs: number;
}

/**
 * The live agent timeline.
 *
 * This is the answer to a real problem rather than decoration: the pipeline
 * takes tens of seconds, and a spinner for that long reads as a hang. Showing
 * each agent's finding as it lands turns the wait into the most interesting
 * part of the product — the reader watches the question get classified,
 * decomposed, retrieved against, scored and validated.
 */
export default function PipelineTimeline({
  stages,
  running,
  elapsedMs,
}: PipelineTimelineProps) {
  const doneCount = stages.filter((s) => s.status === "done").length;

  return (
    <div className="panel overflow-hidden">
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="text-[0.72rem] font-semibold uppercase tracking-widest text-ash-500">
            Pipeline
          </span>
          {running && (
            <span className="inline-flex h-1.5 w-1.5 rounded-full bg-gold-500 pulse-ring" />
          )}
        </div>
        <span className="font-mono text-[0.7rem] tabular-nums text-ash-500">
          {doneCount}/{stages.length} · {(elapsedMs / 1000).toFixed(1)}s
        </span>
      </div>

      <ol className="space-y-0.5 p-3">
        {stages.map((stage, i) => (
          <StageRow
            key={`${stage.agent}-${stage.seq}-${i}`}
            stage={stage}
            isLast={i === stages.length - 1}
          />
        ))}
      </ol>
    </div>
  );
}
