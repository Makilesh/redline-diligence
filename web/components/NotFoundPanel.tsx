"use client";

import { motion } from "framer-motion";
import { Lightbulb, SearchX } from "lucide-react";

import type { QueryResponse } from "@/lib/types";
import { NOT_FOUND_DETAIL } from "@/lib/verification";

interface NotFoundPanelProps {
  result: QueryResponse;
}

/**
 * The banner above a "not found" answer (Decision 112).
 *
 * The model said the documents do not contain what was asked. That is not an
 * answer, and it is weaker evidence than it reads: the model saw only the
 * passages retrieved for this question, and in round 16 a retrieval filter hid
 * the very schedule that stated the figure while the answer said "the provided
 * documents do not contain" it. So it is styled with the refusal, not with the
 * answers, and says plainly what was and was not searched. The model's
 * explanation, and what it did find, follow in the answer panel.
 */
export default function NotFoundPanel({ result }: NotFoundPanelProps) {
  const attempts = result.rewrite_iterations + 1;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
      className="panel overflow-hidden"
    >
      <div className="h-px bg-gradient-to-r from-warn via-warn/25 to-transparent" />

      <div className="px-6 py-6">
        <div className="flex items-start gap-4">
          <div className="mt-0.5 shrink-0 rounded-xl border border-warn/25 bg-warn/10 p-2.5 text-warn">
            <SearchX size={18} />
          </div>

          <div className="min-w-0 flex-1">
            <h2 className="text-[1.02rem] font-semibold tracking-tight text-ash-50">
              Not found in the passages searched
            </h2>
            <p className="mt-2 max-w-[62ch] text-[0.9rem] leading-relaxed text-ash-300">
              {NOT_FOUND_DETAIL} Retrieval ran {attempts} time{attempts === 1 ? "" : "s"}.
            </p>
          </div>
        </div>
      </div>

      <div className="border-t border-line bg-ink-850/60 px-6 py-4">
        <div className="mb-2 flex items-center gap-2 text-[0.7rem] font-semibold uppercase tracking-widest text-ash-500">
          <Lightbulb size={12} />
          Before concluding it is not there
        </div>
        <ul className="grid gap-1.5 text-[0.84rem] leading-relaxed text-ash-300 sm:grid-cols-2">
          <li>Name the document or schedule you expect the answer to be in.</li>
          <li>Use the language of the filing — &ldquo;customer concentration&rdquo;, not &ldquo;biggest clients&rdquo;.</li>
          <li>Check that the relevant document has been indexed for this deal.</li>
          <li>Ask for one fact at a time rather than a broad summary.</li>
        </ul>
      </div>
    </motion.div>
  );
}
