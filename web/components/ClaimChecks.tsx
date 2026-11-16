"use client";

import { useState } from "react";
import { ChevronDown, FileText, ListChecks, TriangleAlert } from "lucide-react";

import type { ClaimCheck } from "@/lib/types";
import {
  citationMarker,
  claimMarker,
  claimView,
  countClaims,
  evidenceLocation,
  figureView,
  needsReview,
  sortClaimsForReview,
  type ClaimMarker,
  type Tone,
} from "@/lib/verification";

const TONE_TEXT: Record<Tone, string> = {
  ok: "text-ok",
  warn: "text-warn",
  bad: "text-bad",
  neutral: "text-ash-500",
};

const TONE_CHIP: Record<Tone, string> = {
  ok: "text-ok bg-ok/10 border-ok/25",
  warn: "text-warn bg-warn/10 border-warn/25",
  bad: "text-bad bg-bad/10 border-bad/25",
  neutral: "text-ash-300 bg-ink-800 border-line",
};

const TONE_RULE: Record<Tone, string> = {
  ok: "border-ok/35",
  warn: "border-warn/45",
  bad: "border-bad/50",
  neutral: "border-line-strong",
};

function ClaimRow({ check }: { check: ClaimCheck }) {
  const view = claimView(check);
  const markers = [claimMarker(check), citationMarker(check)].filter(
    (m): m is ClaimMarker => m !== null,
  );
  const location = evidenceLocation(check);

  return (
    <li className={`border-l-2 pl-3.5 ${TONE_RULE[view.tone]}`} data-status={check.status}>
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
        <span
          className={`rounded border px-1.5 py-px text-[0.64rem] font-semibold uppercase tracking-wider ${TONE_CHIP[view.tone]}`}
        >
          {view.label}
        </span>
        {view.method && (
          <span className="text-[0.68rem] text-ash-500">via {view.method}</span>
        )}
        {/* Beside the verdict, not instead of it: the judge may still have
            confirmed the statement. */}
        {markers.map((marker) => (
          <span
            key={marker.label}
            title={marker.title}
            className="inline-flex items-center gap-1 text-[0.68rem] text-warn"
          >
            <TriangleAlert size={11} className="shrink-0" aria-hidden />
            {marker.label}
          </span>
        ))}
      </div>

      <p className="mt-1.5 text-[0.84rem] leading-relaxed text-ash-50">{check.claim}</p>

      <p className={`mt-1 text-[0.76rem] leading-snug ${TONE_TEXT[view.tone]}`}>
        {view.meaning}
        {/* The checker's own reason is more specific than the generic meaning
            whenever it says something beyond "every figure appears". */}
        {check.reason && check.status !== "supported" && (
          <span className="text-ash-500"> — {check.reason}</span>
        )}
      </p>

      {check.numbers.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {check.numbers.map((figure, i) => {
            const fv = figureView(figure);
            return (
              <span
                key={`${figure.raw}-${i}`}
                title={fv.title}
                className={`inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 text-[0.7rem] ${TONE_CHIP[fv.tone]}`}
              >
                <span className="font-mono tabular-nums">{figure.raw}</span>
                <span className="opacity-80">{fv.label}</span>
              </span>
            );
          })}
        </div>
      )}

      {check.evidence ? (
        <figure className="mt-2 rounded-md border border-line bg-ink-850/70 px-3 py-2">
          <blockquote className="text-[0.78rem] italic leading-relaxed text-ash-300">
            &ldquo;{check.evidence}&rdquo;
          </blockquote>
          {location && (
            <figcaption className="mt-1.5 flex items-center gap-1.5 font-mono text-[0.66rem] text-ash-500">
              <FileText size={10} className="shrink-0" />
              <span className="truncate">{location}</span>
            </figcaption>
          )}
        </figure>
      ) : (
        // A judge's verdict names its document without quoting a passage.
        location && (
          <p className="mt-1.5 flex items-center gap-1.5 font-mono text-[0.66rem] text-ash-500">
            <FileText size={10} className="shrink-0" />
            <span className="truncate">{location}</span>
          </p>
        )
      )}
    </li>
  );
}

interface ClaimChecksProps {
  checks: ClaimCheck[];
}

/**
 * The per-statement verification record.
 *
 * The badge in the answer header compresses this into one word; this is the
 * evidence behind it — every statement the validator extracted, what it
 * decided, how, and the passage the decision rests on. Statements that need
 * attention are listed first and always visible; the supported ones fold away
 * behind a count so a long, clean answer does not bury the one weak claim.
 */
export default function ClaimChecks({ checks }: ClaimChecksProps) {
  const [showSupported, setShowSupported] = useState(false);
  if (checks.length === 0) return null;

  const counts = countClaims(checks);
  const ordered = sortClaimsForReview(checks);
  // A supported claim whose citation does not hold its figure is not folded
  // away with the rest: the reader would open the wrong source (Decision 78).
  const attention = ordered.filter(needsReview);
  const supported = ordered.filter((c) => !needsReview(c));

  const summary: [number, string, Tone][] = [
    [counts.supported, "supported", "ok"],
    [counts.contradicted, "contradicted", "bad"],
    [counts.unsupported, "unsupported", "bad"],
    [counts.unverified, "unverified", "warn"],
  ];

  return (
    <section className="border-t border-line px-6 py-4" aria-label="Claim checks">
      <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="inline-flex items-center gap-2 text-[0.72rem] font-semibold uppercase tracking-wider text-ash-500">
          <ListChecks size={13} />
          {counts.total} statement{counts.total === 1 ? "" : "s"} checked
        </span>
        {summary
          .filter(([n]) => n > 0)
          .map(([n, label, tone]) => (
            <span key={label} className={`text-[0.72rem] tabular-nums ${TONE_TEXT[tone]}`}>
              {n} {label}
            </span>
          ))}
      </div>

      {attention.length > 0 && (
        <ul className="space-y-4">
          {attention.map((c, i) => (
            <ClaimRow key={`a-${i}`} check={c} />
          ))}
        </ul>
      )}

      {supported.length > 0 && (
        <div className={attention.length > 0 ? "mt-4" : ""}>
          <button
            type="button"
            onClick={() => setShowSupported((v) => !v)}
            aria-expanded={showSupported}
            className="inline-flex items-center gap-1 text-[0.74rem] font-medium text-ash-500 transition-colors hover:text-gold-400"
          >
            <ChevronDown
              size={12}
              className={`transition-transform ${showSupported ? "rotate-180" : ""}`}
            />
            {showSupported ? "Hide" : "Show"} {supported.length} supported statement
            {supported.length === 1 ? "" : "s"} and their evidence
          </button>
          {showSupported && (
            <ul className="mt-3 space-y-4">
              {supported.map((c, i) => (
                <ClaimRow key={`s-${i}`} check={c} />
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
