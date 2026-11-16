"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  BadgeCheck,
  CircleDashed,
  Clock,
  Gauge,
  Loader2,
  RefreshCw,
  Scissors,
  ShieldAlert,
  ShieldQuestion,
  ShieldX,
} from "lucide-react";

import ClaimChecks from "./ClaimChecks";
import CitationPill from "./CitationPill";
import { parseAnswer, type ParsedCitation } from "@/lib/citations";
import type { QueryResponse } from "@/lib/types";
import { answerVerdict, type AnswerVerdict } from "@/lib/verification";

interface AnswerPanelProps {
  /** The validated answer. Null while the answer is still being written. */
  result: QueryResponse | null;
  /**
   * Streamed answer text, shown only until `result` arrives. It has not been
   * verified yet, so it renders as a draft with no verdict or confidence.
   */
  draft?: string;
  onSelectCitation?: (index: number) => void;
}

const NO_CITATIONS: QueryResponse["citations"] = [];

/** Header treatment while the answer is still streaming in, unverified. */
const DRAFT_VERDICT = {
  icon: Loader2,
  label: "Drafting…",
  detail: "Unverified draft — replaced by the checked answer when validation finishes",
  className: "text-ash-300 bg-ink-850/70 border-line",
} as const;

/**
 * Visual treatment per verdict. The wording comes from `answerVerdict`, which
 * never reads a missing or unrecognised status as a pass.
 */
const VERDICT_STYLE: Record<AnswerVerdict["key"], { icon: React.ElementType; className: string }> = {
  verified: { icon: BadgeCheck, className: "text-ok bg-ok/12 border-ok/30" },
  partial: { icon: ShieldAlert, className: "text-warn bg-warn/12 border-warn/30" },
  failed: { icon: ShieldX, className: "text-bad bg-bad/12 border-bad/30" },
  no_claims: { icon: CircleDashed, className: "text-ash-300 bg-ink-850/70 border-line" },
  unknown: { icon: ShieldQuestion, className: "text-warn bg-warn/12 border-warn/30" },
};

function Metric({
  icon: Icon,
  label,
  value,
  tone = "default",
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  tone?: "default" | "good" | "warn" | "bad";
}) {
  const toneClass = {
    default: "text-ash-50",
    good: "text-ok",
    warn: "text-warn",
    bad: "text-bad",
  }[tone];

  return (
    <div className="flex items-center gap-2">
      <Icon size={13} className="shrink-0 text-ash-500" />
      <span className="text-[0.7rem] uppercase tracking-wider text-ash-500">
        {label}
      </span>
      <span className={`text-[0.82rem] font-semibold tabular-nums ${toneClass}`}>
        {value}
      </span>
    </div>
  );
}

export default function AnswerPanel({
  result,
  draft = "",
  onSelectCitation,
}: AnswerPanelProps) {
  const isDraft = !result;
  const body = result ? result.answer : draft;
  const records = result ? result.citations : NO_CITATIONS;

  // Parsing rewrites every bracket citation into a numbered link, so it must
  // run before the markdown is handed to the renderer — and only when the
  // answer changes, since it walks the whole body. While drafting that is once
  // per streamed token; a half-written bracket simply stays literal until it
  // closes.
  const { markdown, citations } = useMemo(
    () => parseAnswer(body, records),
    [body, records],
  );

  const byIndex = useMemo(() => {
    const map = new Map<number, ParsedCitation>();
    citations.forEach((c) => map.set(c.index, c));
    return map;
  }, [citations]);

  const verdict = result
    ? (() => {
        const v = answerVerdict(result);
        return { ...VERDICT_STYLE[v.key], key: v.key, label: v.label, detail: v.detail };
      })()
    : { ...DRAFT_VERDICT, key: "draft" as const };
  const VerdictIcon = verdict.icon;

  const confidence = Math.round((result?.confidence_score ?? 0) * 100);
  const confidenceTone =
    confidence >= 75 ? "good" : confidence >= 45 ? "warn" : "bad";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
      className="panel overflow-hidden"
    >
      {/* Gold hairline along the top edge — groups the answer as the primary
          surface without adding another border weight. */}
      <div className="h-px bg-gradient-to-r from-gold-500 via-gold-500/25 to-transparent" />

      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 border-b border-line px-6 py-3.5">
        <div
          className={`inline-flex items-center gap-2 rounded-lg border px-2.5 py-1.5 ${verdict.className}`}
          title={verdict.detail}
        >
          <VerdictIcon size={14} className={isDraft ? "animate-spin" : undefined} />
          <span className="text-[0.74rem] font-semibold tracking-wide">
            {verdict.label}
          </span>
        </div>

        {isDraft ? (
          <span className="text-[0.74rem] text-ash-500">{verdict.detail}</span>
        ) : result && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <Metric
            icon={Gauge}
            label="Confidence"
            value={`${confidence}%`}
            tone={confidenceTone}
          />
          <Metric
            icon={Clock}
            label="Latency"
            value={`${(result.total_latency_ms / 1000).toFixed(1)}s`}
          />
          {result.rewrite_iterations > 0 && (
            <Metric
              icon={RefreshCw}
              label="Rewrites"
              value={String(result.rewrite_iterations)}
              tone="warn"
            />
          )}
        </div>
        )}
      </div>

      {/* The badge's tooltip is not enough for a failed or partial answer:
          the reader must see, before reading the text, that it is not all
          backed by the documents. */}
      {(verdict.key === "failed" || verdict.key === "partial" || verdict.key === "unknown") && (
        <p
          role="note"
          className={`border-b border-line px-6 py-2.5 text-[0.8rem] leading-snug ${
            verdict.key === "failed" ? "bg-bad/[0.06] text-bad" : "bg-warn/[0.06] text-warn"
          }`}
        >
          {verdict.detail}
        </p>
      )}

      <div className={`answer-body px-6 py-6 ${isDraft ? "opacity-80" : ""}`}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // Citations arrive as `[n](#cite-n)` links from parseAnswer. Any
            // other link is a genuine link the model wrote and is left alone.
            a({ href, children, ...props }) {
              const match = /^#cite-(\d+)$/.exec(href ?? "");
              if (match) {
                const citation = byIndex.get(Number(match[1]));
                if (citation) {
                  return (
                    <CitationPill
                      citation={citation}
                      onSelect={onSelectCitation}
                    />
                  );
                }
              }
              // The answer is model output built from counterparty documents,
              // so a link in it is untrusted. react-markdown already blanks
              // `javascript:` and `data:` URLs; anything that is not plainly
              // web or mail is rendered as text rather than as a link to "".
              if (!href || !/^(https?:|mailto:)/i.test(href)) {
                return <span className="text-ash-300">{children}</span>;
              }
              return (
                <a
                  href={href}
                  title={href}
                  className="text-gold-400 underline underline-offset-2 hover:text-gold-300"
                  target="_blank"
                  rel="noreferrer noopener nofollow"
                  {...props}
                >
                  {children}
                </a>
              );
            },
            // An image would load with no click, so `![](https://host/?q=…)`
            // in an answer steered by a planted document could send data out
            // (Decision 76). Answers never need images: show the alt text.
            img({ alt }) {
              return alt ? <span className="text-ash-300">{alt}</span> : null;
            },
            // Wide financial tables must scroll inside their own container —
            // otherwise a ten-column comparison scrolls the whole page sideways.
            table({ children, ...props }) {
              return (
                <div className="table-scroll">
                  <table {...props}>{children}</table>
                </div>
              );
            },
          }}
        >
          {markdown}
        </ReactMarkdown>
      </div>

      {/* Directly under the text, where it stops: the model hit its output
          limit, so the last sentence may be unfinished and later points
          missing. Outside .answer-body so the prose styles do not apply. */}
      {result?.answer_truncated && (
        <p
          role="note"
          className="-mt-2 flex items-center gap-2 px-6 pb-4 text-[0.78rem] leading-snug text-warn"
        >
          <Scissors size={12} className="shrink-0" aria-hidden />
          Cut off here: the model reached its output limit, so the answer is incomplete.
        </p>
      )}

      {result && <ClaimChecks checks={result.claim_checks} />}

      {/* The flags are the validator's free-text summary. With per-claim
          checks above they mostly repeat them, so they fold away; without
          checks they are the only record and stay open. */}
      {result && result.hallucination_flags.length > 0 && (
        <details
          open={result.claim_checks.length === 0}
          className="group border-t border-line bg-bad/[0.04] px-6 py-4"
        >
          <summary className="flex cursor-pointer list-none items-center gap-2 text-[0.72rem] font-semibold uppercase tracking-wider text-bad">
            <ShieldAlert size={13} />
            {result.hallucination_flags.length} validator note
            {result.hallucination_flags.length === 1 ? "" : "s"}
          </summary>
          <ul className="mt-2.5 space-y-1.5">
            {result.hallucination_flags.map((flag, i) => (
              <li
                key={i}
                className="border-l-2 border-bad/40 pl-3 text-[0.82rem] leading-relaxed text-ash-300"
              >
                {flag}
              </li>
            ))}
          </ul>
        </details>
      )}
    </motion.div>
  );
}
