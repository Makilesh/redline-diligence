/**
 * What a verification result means, in words a reader can act on.
 *
 * Pure functions over the normalised response, kept apart from the components
 * so the rules are testable and stated once. The governing rule: the page may
 * understate how well an answer was checked, never overstate it. Anything the
 * client does not recognise — a missing status, a new outcome, a claim list
 * that disagrees with the summary — lands on the less favourable reading.
 */

import type {
  ClaimCheck,
  FigureCheck,
  InjectionReview,
  InjectionScreen,
  InjectionSignal,
  PassageLocation,
  QueryResponse,
} from "./types";

export type Tone = "ok" | "warn" | "bad" | "neutral";

/** Which surface the page shows for a finished run. */
export type ResultView =
  /** An answer, verified or not — rendered with its verdict and claim checks. */
  | "answer"
  /** A considered refusal: the data room did not support an answer. */
  | "refusal"
  /**
   * The answer says the documents searched do not contain what was asked
   * (Decision 112): shown as not-an-answer, with the model's explanation.
   */
  | "not_found"
  /** An outage: no model produced an answer, so there is nothing to show or verify. */
  | "unavailable";

/**
 * Picks the surface from `outcome` — not `is_refusal`, which is also true for
 * a synthesis failure and would present an outage as a judgment about the
 * question.
 */
export function resultView(result: QueryResponse): ResultView {
  switch (result.outcome) {
    case "refused":
      return "refusal";
    case "not_found":
      return "not_found";
    case "synthesis_failed":
      return "unavailable";
    default:
      return "answer";
  }
}

export interface ClaimCounts {
  total: number;
  supported: number;
  contradicted: number;
  unsupported: number;
  unverified: number;
}

export function countClaims(checks: ClaimCheck[]): ClaimCounts {
  const counts: ClaimCounts = {
    total: checks.length,
    supported: 0,
    contradicted: 0,
    unsupported: 0,
    unverified: 0,
  };
  for (const c of checks) counts[c.status] += 1;
  return counts;
}

export interface AnswerVerdict {
  key: "verified" | "partial" | "failed" | "not_found" | "no_claims" | "unknown";
  label: string;
  /** One sentence stating exactly what was and was not established. */
  detail: string;
  tone: Tone;
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/**
 * What a reader must know before trusting a "not found" (Decision 112): the
 * absence was checked against the passages retrieved, not the data room.
 */
export const NOT_FOUND_DETAIL =
  "The answer says the documents do not contain this. That was checked against the passages retrieved for this question, not against every document in the data room, so the information may still be there.";

/** Ends a verdict sentence for an answer cut off at the model's output limit. */
const CUT_OFF = "the answer was cut off before it finished.";

/**
 * The verdict badge for an answer.
 *
 * "Verified" is only shown when the server said `passed` AND every claim it
 * returned is supported AND there was at least one claim AND the answer was
 * not cut off. Each of those can fail independently: `passed` is also the
 * server's default for a missing status, and an answer with no checkable
 * statement passes vacuously.
 */
export function answerVerdict(result: QueryResponse): AnswerVerdict {
  const counts = countClaims(result.claim_checks);
  const problems = counts.contradicted + counts.unsupported;
  const notConfirmed = problems + counts.unverified;
  const miscited = result.claim_checks.filter(isMiscited).length;
  const tainted = result.injection_review.tainted_figures;
  const citesFlagged = result.injection_review.cited_flagged_passages > 0;
  const injected = tainted.length > 0 || citesFlagged;

  if (result.outcome === "answered_unverified" || result.validation_status === "failed") {
    return {
      key: "failed",
      label: "Failed verification",
      detail:
        (problems > 0
          ? `${plural(problems, "statement")} could not be matched to the documents, even after a revision. Treat the marked statements as wrong until checked.`
          : "The answer did not pass verification, even after a revision. Treat it as unconfirmed.") +
        (result.answer_truncated ? ` Also, ${CUT_OFF}` : ""),
      tone: "bad",
    };
  }

  // Never "verified": its absence statements hold for the passages searched,
  // not necessarily for the data room (Decision 112).
  if (result.outcome === "not_found") {
    return {
      key: "not_found",
      label: "Not found",
      detail: NOT_FOUND_DETAIL + (result.answer_truncated ? ` Also, ${CUT_OFF}` : ""),
      tone: "warn",
    };
  }

  // The server turns "passed" into "warning" for an answer cut off at the
  // output limit, so its claims can all be supported. Saying "0 could not be
  // confirmed" would leave the reader looking for a problem that is not there.
  if (
    result.validation_status === "warning" ||
    (result.validation_status === "passed" &&
      (notConfirmed > 0 || result.answer_truncated || miscited > 0 || injected))
  ) {
    let detail: string;
    if (result.answer_truncated && notConfirmed === 0) {
      detail =
        counts.total > 0
          ? `All ${plural(counts.total, "statement")} matched to the documents, but ${CUT_OFF}`
          : `Nothing in it could be checked, and ${CUT_OFF}`;
    } else if (notConfirmed === 0 && counts.total > 0 && (miscited > 0 || injected)) {
      // Downgraded for a citation that does not hold its figure (Decision 78)
      // or for leaning on instruction-like text (Decision 76): every statement
      // checked out, so say what is wrong instead.
      const reasons: string[] = [];
      if (miscited > 0) {
        reasons.push(
          `${miscited} ${
            miscited === 1 ? "cites a document that does not state its figures" : "cite documents that do not state their figures"
          }`,
        );
      }
      if (tainted.length > 0) {
        reasons.push(
          `${tainted.join(", ")} ${tainted.length === 1 ? "rests" : "rest"} only on text that reads like instructions to an AI model`,
        );
      } else if (citesFlagged) {
        reasons.push("it cites a passage that reads like instructions to an AI model");
      }
      detail = `All ${plural(counts.total, "statement")} matched to the documents, but ${reasons.join(", and ")}.`;
    } else {
      detail =
        (counts.total > 0
          ? `${counts.supported} of ${plural(counts.total, "statement")} matched to the documents; ${notConfirmed} could not be confirmed.`
          : "Some statements could not be matched to a source.") +
        (result.answer_truncated ? ` Also, ${CUT_OFF}` : "");
    }
    return { key: "partial", label: "Partially verified", detail, tone: "warn" };
  }

  if (result.validation_status === "passed") {
    if (counts.total === 0) {
      return {
        key: "no_claims",
        label: "Nothing to verify",
        detail: "The answer makes no statement that could be checked against the documents.",
        tone: "neutral",
      };
    }
    return {
      key: "verified",
      label: "Verified",
      detail: `All ${plural(counts.total, "statement")} matched to the retrieved documents.`,
      tone: "ok",
    };
  }

  return {
    key: "unknown",
    label: "Not verified",
    detail: "The engine did not report a recognised verification result for this answer.",
    tone: "warn",
  };
}

export interface ClaimView {
  label: string;
  tone: Tone;
  /** What the verdict means for this claim, in a sentence. */
  meaning: string;
  /** How the verdict was reached; empty when nothing decided it. */
  method: string;
}

const METHOD_LABEL: Record<ClaimCheck["method"], string> = {
  numeric: "Figure match",
  nli: "Entailment model",
  llm: "LLM judge",
  none: "",
};

/** Whether the claim was left unchecked because the checking budget ran out. */
export function isNotChecked(c: ClaimCheck): boolean {
  return c.status === "unverified" && /^not checked\b/i.test(c.reason);
}

/** Describes one claim's verdict. */
export function claimView(c: ClaimCheck): ClaimView {
  const method = METHOD_LABEL[c.method];
  switch (c.status) {
    case "supported":
      return {
        label: "Supported",
        tone: "ok",
        method,
        meaning:
          c.method === "numeric"
            ? "Every figure in it was found in, or computed from, the documents."
            : c.method === "nli"
              ? "A retrieved passage states it."
              : "An LLM judge confirmed it against the documents.",
      };
    case "contradicted":
      return {
        label: "Contradicted",
        tone: "bad",
        method,
        meaning: "A document says otherwise.",
      };
    case "unsupported":
      return {
        label: "Unsupported",
        // A figure that is not in the documents fails the answer; a statement
        // no passage backs is a warning.
        tone: c.method === "numeric" ? "bad" : "warn",
        method,
        meaning:
          c.method === "numeric"
            ? "A figure in it does not appear in the retrieved documents."
            : "No retrieved passage supports it.",
      };
    default:
      if (isNotChecked(c)) {
        return {
          label: "Not checked",
          tone: "neutral",
          method: "",
          meaning: "Beyond the number of statements checked per answer — neither confirmed nor refuted.",
        };
      }
      return {
        label: c.absence ? "Unconfirmed absence" : "Unverified",
        tone: "warn",
        method,
        meaning: c.absence
          ? "It says the documents do not contain something; that could not be confirmed."
          : "No check could confirm or refute it.",
      };
  }
}

export interface ClaimMarker {
  label: string;
  /** Longer explanation, for a title attribute. */
  title: string;
}

/**
 * A note shown beside a claim's verdict when the local checker doubted the
 * statement's wording although every figure in it checked out. The verdict
 * itself is the judge's (it may still be "Supported"); the marker only says
 * the statement was doubted on the way, so a reader can look twice.
 */
export function claimMarker(c: ClaimCheck): ClaimMarker | null {
  if (!c.suspected_misstatement) return null;
  return {
    label: "Wording doubted",
    title:
      "Every figure in it is in the documents, but the passages holding them did not " +
      "clearly support the statement as worded (another party, a reversed condition), " +
      "so it was not confirmed on its figures alone and was left to the LLM judge.",
  };
}

/** Whether a figure of the claim is stated only outside the documents it cites. */
export function isMiscited(c: ClaimCheck): boolean {
  return (c.citation_mismatch?.length ?? 0) > 0;
}

/**
 * A note beside a claim whose citation does not hold its figures (Decision
 * 78). The figures are in the documents, so the verdict may be "Supported";
 * the note tells the reader where to look instead of the cited source.
 */
export function citationMarker(c: ClaimCheck): ClaimMarker | null {
  if (!isMiscited(c)) return null;
  const items = c.citation_mismatch!;
  const cited = c.cited_documents?.length
    ? `the cited ${c.cited_documents.join(", ")}`
    : "any retrieved document its citation names";
  return {
    label: "Cited source lacks figure",
    title:
      items
        .map((m) => `${m.figure} is stated in ${m.found_in.join(", ") || "another document"}`)
        .join("; ") + `, not in ${cited}.`,
  };
}

/**
 * A note beside a claim with a figure that only instruction-like document
 * text states (Decision 76). The figure is in the documents, so the verdict
 * may be "Supported"; the note says where it comes from.
 */
export function injectionMarker(c: ClaimCheck): ClaimMarker | null {
  if (!c.injection_suspect) return null;
  return {
    label: "Figure from instruction-like text",
    title:
      "A figure in it is stated only by document text that reads like an instruction to an " +
      "AI model (possible prompt injection). Check it against the document itself.",
  };
}

/** Whether a claim belongs in the list a reader sees first, not the folded one. */
export function needsReview(c: ClaimCheck): boolean {
  return c.status !== "supported" || isMiscited(c) || c.injection_suspect === true;
}

/** The ingest screen's signals, in words. */
const SIGNAL_LABEL: Record<InjectionSignal, string> = {
  instruction_override: "an instruction to ignore earlier instructions",
  model_address: "text addressed to an AI model",
  role_marker: "chat-role labels such as SYSTEM:",
  forged_marker: "imitation citation or calculation markers",
  supersession_claim: "a claim that other documents are out of date",
  invisible_text: "invisible characters",
};

/** The known signals in words, comma-separated; an unknown name is left out. */
export function signalLabels(signals: Iterable<string>): string {
  const present = new Set(signals);
  return (Object.keys(SIGNAL_LABEL) as InjectionSignal[])
    .filter((s) => present.has(s))
    .map((s) => SIGNAL_LABEL[s])
    .join(", ");
}

/** "memo.pdf, p. 3 · Fees" — where a flagged passage is; "" when nothing is known. */
export function passageLocation(loc: PassageLocation & { source_file?: string }): string {
  const head = [loc.source_file ?? "", loc.page_number !== null ? `p. ${loc.page_number}` : ""]
    .filter(Boolean)
    .join(", ");
  return [head, loc.section_heading].filter(Boolean).join(" · ");
}

export interface Note {
  text: string;
  tone: Tone;
}

/**
 * One line on instruction-like text behind an answer (Decision 82), or null.
 *
 * The validator's flags say what it found, but they fold away behind the claim
 * list, and a flagged passage the answer does not cite was only in the trace.
 * Warn when the answer cites such a passage or takes a figure only from one;
 * neutral when the passage was retrieved and not used.
 */
export function injectionNote(review: InjectionReview): Note | null {
  const passages = review.flagged_passages;
  if (passages.length === 0) return null;
  const where = passages.map(passageLocation).filter(Boolean).join("; ") || "a retrieved document";
  const labels = signalLabels(passages.flatMap((p) => p.signals));
  const labelled = labels ? ` (${labels})` : "";
  const tainted = review.tainted_figures;
  if (tainted.length > 0 || review.cited_flagged_passages > 0) {
    const extra =
      tainted.length > 0
        ? ` — ${tainted.join(", ")} ${tainted.length === 1 ? "is" : "are"} stated nowhere else`
        : "";
    return {
      text:
        `This answer relies on text that reads like instructions to an AI model${labelled} ` +
        `in ${where}${extra}. Check those statements against the document itself.`,
      tone: "warn",
    };
  }
  return {
    text:
      `Retrieved but not cited: ${where} ${passages.length === 1 ? "contains" : "contain"} ` +
      `text that reads like instructions to an AI model${labelled}.`,
    tone: "neutral",
  };
}

/**
 * What the ingest screen found in an upload, in two sentences, or null when
 * nothing was flagged. Factual, not alarming: the file was indexed as written,
 * and the consequence for answers is stated as it is.
 */
export function uploadScreenNote(screen: InjectionScreen): { finding: string; consequence: string } | null {
  const n = screen.flagged_passages;
  if (n <= 0) return null;
  const labels = signalLabels(screen.signals);
  const where = screen.locations.map(passageLocation).filter(Boolean).join("; ");
  return {
    finding:
      `${plural(n, "passage")} ${n === 1 ? "reads" : "read"} like instructions to an AI model` +
      (labels ? ` (${labels})` : "") +
      (where ? `: ${where}.` : "."),
    consequence:
      "The file was indexed as written. An answer that cites these passages, or takes a figure " +
      "only from them, is never shown as Verified.",
  };
}

/** Problems first, so the statements that need a reader's attention lead. */
export function sortClaimsForReview(checks: ClaimCheck[]): ClaimCheck[] {
  const rank = (c: ClaimCheck) =>
    c.status === "contradicted"
      ? 0
      : c.status === "unsupported"
        ? 1
        : c.status === "unverified"
          ? isNotChecked(c)
            ? 3
            : 2
          : 4;
  return checks
    .map((c, i) => [c, i] as const)
    .sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1])
    .map(([c]) => c);
}

export interface FigureView {
  label: string;
  tone: Tone;
  /** Longer explanation, for a title attribute. */
  title: string;
}

/** Describes one figure inside a claim. */
export function figureView(f: FigureCheck): FigureView {
  if (f.status === "unsupported") {
    return { label: "not in documents", tone: "bad", title: "This figure was not found in the retrieved documents." };
  }
  if (f.status === "derived_unverified") {
    return {
      label: "calculation not reproduced",
      tone: "warn",
      title: "Worded as a calculation, but it could not be reproduced from figures in the documents.",
    };
  }
  if (f.binding === "conflict") {
    return {
      label: "different row or period",
      tone: "bad",
      title: f.binding_note || "The documents give this value for a different line item or period.",
    };
  }
  if (f.binding === "unbound") {
    return {
      label: "found elsewhere",
      tone: "warn",
      title: f.binding_note || "The value appears in the documents, but not where this statement places it.",
    };
  }
  if (f.status === "derived") {
    return {
      label: "computed",
      tone: "ok",
      title: f.formula ? `Reproduced as ${f.formula}` : "Reproduced from figures in the documents.",
    };
  }
  return { label: "found", tone: "ok", title: "Found in the documents where the statement places it." };
}

/** "source.pdf, p. 4 · Section" — the location an evidence passage came from. */
export function evidenceLocation(c: ClaimCheck): string {
  const parts: string[] = [];
  if (c.evidence_source) {
    parts.push(c.evidence_page !== null ? `${c.evidence_source}, p. ${c.evidence_page}` : c.evidence_source);
  }
  if (c.evidence_section) parts.push(c.evidence_section);
  return parts.join(" · ");
}
