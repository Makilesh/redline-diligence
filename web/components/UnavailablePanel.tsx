"use client";

import { motion } from "framer-motion";
import { CloudOff, RotateCcw } from "lucide-react";

import type { QueryResponse } from "@/lib/types";

interface UnavailablePanelProps {
  result: QueryResponse;
  onRetry?: () => void;
}

/**
 * The `synthesis_failed` outcome: retrieval ran, but no language model returned
 * a usable answer.
 *
 * Deliberately not the refusal surface. A refusal is a judgment that the data
 * room does not support an answer; this is an outage that says nothing about
 * the question. Presenting it as a refusal would teach the reader that the
 * documents lack something they may well contain. The synthesizer's apology
 * text is not shown — it is not an answer and was never verified.
 */
export default function UnavailablePanel({ result, onRetry }: UnavailablePanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
      className="panel overflow-hidden"
    >
      <div className="h-px bg-gradient-to-r from-ash-500 via-ash-500/25 to-transparent" />

      <div className="flex items-start gap-4 px-6 py-6">
        <div className="mt-0.5 shrink-0 rounded-xl border border-line bg-ink-800 p-2.5 text-ash-300">
          <CloudOff size={18} />
        </div>

        <div className="min-w-0 flex-1">
          <h2 className="text-[1.02rem] font-semibold tracking-tight text-ash-50">
            No answer — the language models did not respond
          </h2>
          <p className="mt-2 max-w-[62ch] text-[0.9rem] leading-relaxed text-ash-300">
            The documents were searched, but every model the engine tried failed to
            return a usable answer, so nothing was written and nothing was verified.
            This is a service problem, not a finding about your question or the data
            room. Asking again in a minute usually works.
          </p>

          <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2">
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-ink-800 px-3 py-1.5 text-[0.78rem] font-medium text-ash-50 transition-colors hover:border-gold-600/50 hover:text-gold-300"
              >
                <RotateCcw size={12} />
                Ask again
              </button>
            )}
            {result.request_id && (
              <span className="text-[0.72rem] text-ash-500">
                Request ID{" "}
                <code className="select-all font-mono text-ash-300">{result.request_id}</code>
              </span>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
