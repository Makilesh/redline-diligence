"use client";

import { motion } from "framer-motion";
import { RotateCcw, TriangleAlert } from "lucide-react";

import { failureTitle, isRetryable, type ApiError } from "@/lib/errors";

interface ErrorPanelProps {
  error: ApiError;
  /** Re-asks the same question. Omitted when asking again cannot help. */
  onRetry?: () => void;
}

/**
 * A run that ended without an answer.
 *
 * Nothing is shown in place of the answer — in particular not the streamed
 * draft, which never reached verification. The message is the server's own
 * where it sent one; the request id is what lets the owner find the run in
 * the server log, so it is shown whenever there is one.
 */
export default function ErrorPanel({ error, onRetry }: ErrorPanelProps) {
  const retryable = isRetryable(error.kind) && onRetry;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28 }}
      role="alert"
      className="panel border-bad/30 px-5 py-4"
    >
      <div className="flex items-center gap-2 text-[0.78rem] font-semibold uppercase tracking-widest text-bad">
        <TriangleAlert size={13} />
        {failureTitle(error.kind)}
      </div>
      <p className="mt-1.5 max-w-[68ch] text-[0.86rem] leading-relaxed text-ash-300">
        {error.message}
      </p>

      {(retryable || error.requestId) && (
        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2">
          {retryable && (
            <button
              type="button"
              onClick={onRetry}
              className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-ink-800 px-3 py-1.5 text-[0.78rem] font-medium text-ash-50 transition-colors hover:border-gold-600/50 hover:text-gold-300"
            >
              <RotateCcw size={12} />
              Ask again
            </button>
          )}
          {error.requestId && (
            <span className="text-[0.72rem] text-ash-500">
              Request ID{" "}
              <code className="select-all font-mono text-ash-300">{error.requestId}</code>
            </span>
          )}
        </div>
      )}
    </motion.div>
  );
}
