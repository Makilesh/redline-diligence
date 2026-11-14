/**
 * Runtime normalisation of payloads from the API.
 *
 * TypeScript's view of the wire is a promise nobody checks. These functions
 * are the check: each takes `unknown` and returns the typed shape with safe
 * defaults, so a missing list is an empty list in every component rather than
 * a `.map` of undefined that blanks the page mid-answer. They coerce shape
 * only — no verdict, score or status is ever invented or upgraded here; an
 * unrecognised value always lands on the less favourable reading.
 */

import type {
  Citation,
  ClaimCheck,
  ClaimMethod,
  ClaimStatus,
  FigureBinding,
  FigureCheck,
  FigureStatus,
  IngestResult,
  NodePerf,
  NodeTiming,
  Outcome,
  PlannedStage,
  QueryResponse,
  StageEvent,
  ValidationStatus,
  VersionCandidate,
} from "./types";

type Bag = Record<string, unknown>;

const isBag = (v: unknown): v is Bag => typeof v === "object" && v !== null && !Array.isArray(v);
const str = (v: unknown, fallback = ""): string => (typeof v === "string" ? v : fallback);
const num = (v: unknown, fallback = 0): number =>
  typeof v === "number" && Number.isFinite(v) ? v : fallback;
const numOrNull = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const oneOf = <T extends string>(v: unknown, allowed: readonly T[]): T | undefined =>
  allowed.includes(v as T) ? (v as T) : undefined;

const CLAIM_STATUSES: readonly ClaimStatus[] = [
  "supported",
  "contradicted",
  "unsupported",
  "unverified",
];
const CLAIM_METHODS: readonly ClaimMethod[] = ["numeric", "nli", "llm", "none"];
const FIGURE_STATUSES: readonly FigureStatus[] = [
  "grounded",
  "derived",
  "derived_unverified",
  "unsupported",
];
const FIGURE_BINDINGS: readonly FigureBinding[] = ["bound", "unbound", "conflict"];
const OUTCOMES: readonly Outcome[] = [
  "answered",
  "answered_unverified",
  "refused",
  "synthesis_failed",
];
const RELATIONS: readonly VersionCandidate["relation"][] = ["newer", "older", "same_document"];

function normalizeCitation(raw: unknown): Citation | null {
  if (!isBag(raw)) return null;
  return {
    chunk_id: str(raw.chunk_id),
    source_file: str(raw.source_file),
    page_number: numOrNull(raw.page_number),
    section_heading: str(raw.section_heading),
    // Absent means the server did not say; "current" is its own default.
    is_current_version: raw.is_current_version !== false,
    content_type: str(raw.content_type, "text"),
    is_redline: raw.is_redline === true,
    superseded_by: str(raw.superseded_by),
  };
}

function normalizeFigure(raw: unknown): FigureCheck | null {
  if (!isBag(raw) || !str(raw.raw)) return null;
  return {
    raw: str(raw.raw),
    // A figure whose status the client does not recognise is not shown as
    // grounded: "unsupported" is the conservative reading.
    status: oneOf(raw.status, FIGURE_STATUSES) ?? "unsupported",
    formula: str(raw.formula) || undefined,
    evidence_source: str(raw.evidence_source) || undefined,
    binding: oneOf(raw.binding, FIGURE_BINDINGS),
    binding_note: str(raw.binding_note) || undefined,
  };
}

function normalizeClaim(raw: unknown): ClaimCheck | null {
  if (!isBag(raw) || !str(raw.claim)) return null;
  let reason = str(raw.reason);
  // Never sent by the current checker (a claim beyond the budget is
  // "unverified" with a "not checked: …" reason), but named in the contract
  // notes; read it as that rather than as an unknown verdict.
  if (raw.status === "not_checked" && !reason) reason = "not checked";
  // The judge is reported as "llm"; accept the descriptive name too.
  const method = raw.method === "judge" ? "llm" : raw.method;
  return {
    claim: str(raw.claim),
    // Same rule as figures: an unrecognised verdict is never read as support.
    status: oneOf(raw.status, CLAIM_STATUSES) ?? "unverified",
    method: oneOf(method, CLAIM_METHODS) ?? "none",
    score: numOrNull(raw.score),
    numbers: list(raw.numbers)
      .map(normalizeFigure)
      .filter((f): f is FigureCheck => f !== null),
    cited: raw.cited === true,
    reason,
    evidence: str(raw.evidence),
    evidence_source: str(raw.evidence_source),
    evidence_page: numOrNull(raw.evidence_page),
    evidence_section: str(raw.evidence_section),
    nli_contradiction: numOrNull(raw.nli_contradiction) ?? undefined,
    suspected_contradiction: raw.suspected_contradiction === true || undefined,
    suspected_misattribution: raw.suspected_misattribution === true || undefined,
    absence: raw.absence === true || undefined,
  };
}

/** Normalises a `perf` block; undefined when there is none. */
export function normalizePerf(raw: unknown): NodePerf | undefined {
  if (!isBag(raw)) return undefined;
  return {
    duration_ms: num(raw.duration_ms),
    llm_calls: num(raw.llm_calls),
    prompt_tokens: num(raw.prompt_tokens),
    completion_tokens: num(raw.completion_tokens),
    models: list(raw.models).filter((m): m is string => typeof m === "string" && m.length > 0),
  };
}

function normalizeTiming(raw: unknown): NodeTiming | null {
  if (!isBag(raw) || !str(raw.agent)) return null;
  return { agent: str(raw.agent), ...normalizePerf(raw)! };
}

/**
 * Reduces a response that predates `outcome` (or carries a value this client
 * does not know) to one of the four outcomes, the way the server's own
 * `run_outcome` does: synthesis failure first, then refusal, then a failed
 * verification.
 */
function deriveOutcome(raw: Bag, trace: Bag[]): Outcome {
  const synthesisFailed = trace.some(
    (t) =>
      t.agent === "synthesis_unavailable" ||
      (t.agent === "answer_synthesizer" && t.synthesis_failed === true),
  );
  if (synthesisFailed) return "synthesis_failed";
  if (raw.is_refusal === true || !str(raw.answer).trim()) return "refused";
  if (raw.validation_status === "failed") return "answered_unverified";
  return "answered";
}

/**
 * Normalises a `QueryResponse`.
 *
 * @returns The typed response, or null when the payload is not a response at
 *   all (no `answer` string) — the caller treats that as a failed stream
 *   rather than rendering an empty "answer".
 */
export function normalizeQueryResponse(raw: unknown): QueryResponse | null {
  if (!isBag(raw) || typeof raw.answer !== "string") return null;
  const trace = list(raw.agent_trace).filter(isBag);
  const timings = list(raw.node_timings)
    .map(normalizeTiming)
    .filter((t): t is NodeTiming => t !== null);
  return {
    answer: raw.answer,
    query_type: str(raw.query_type, "summary"),
    confidence_score: Math.min(1, Math.max(0, num(raw.confidence_score))),
    // Passed through as sent. An unexpected value is handled where the verdict
    // is derived (it reads as "not verified"), not rewritten to a known one.
    validation_status: str(raw.validation_status) as ValidationStatus,
    citations: list(raw.citations)
      .map(normalizeCitation)
      .filter((c): c is Citation => c !== null),
    hallucination_flags: list(raw.hallucination_flags).filter(
      (f): f is string => typeof f === "string" && f.length > 0,
    ),
    claim_checks: list(raw.claim_checks)
      .map(normalizeClaim)
      .filter((c): c is ClaimCheck => c !== null),
    total_latency_ms: num(raw.total_latency_ms),
    session_id: str(raw.session_id),
    rewrite_iterations: num(raw.rewrite_iterations),
    agent_trace: trace,
    is_refusal: raw.is_refusal === true,
    context_quality_score: num(raw.context_quality_score),
    outcome: oneOf(raw.outcome, OUTCOMES) ?? deriveOutcome(raw, trace),
    node_timings: timings,
    llm_calls: num(raw.llm_calls, timings.reduce((n, t) => n + t.llm_calls, 0)),
    request_id: str(raw.request_id),
  };
}

/** Normalises a `POST /ingest` response; null when it names no document. */
export function normalizeIngestResult(raw: unknown): IngestResult | null {
  if (!isBag(raw) || !str(raw.doc_id)) return null;
  return {
    doc_id: str(raw.doc_id),
    deal_id: str(raw.deal_id),
    document_category: str(raw.document_category),
    chunks_created: num(raw.chunks_created),
    status: str(raw.status, "success"),
    notices: list(raw.notices).filter((n): n is string => typeof n === "string" && n.length > 0),
    version_candidates: list(raw.version_candidates)
      .filter(isBag)
      .map((c) => ({
        doc_id: str(c.doc_id),
        source_file: str(c.source_file),
        relation: oneOf(c.relation, RELATIONS),
      }))
      // A candidate the client cannot describe is dropped rather than guessed at.
      .filter((c): c is VersionCandidate => Boolean(c.relation && (c.source_file || c.doc_id))),
  };
}

/** Normalises the `planned_stages` list of a `start` event. */
export function normalizePlannedStages(raw: unknown): PlannedStage[] {
  return list(raw)
    .filter(isBag)
    .filter((p) => str(p.agent))
    .map((p) => ({
      agent: str(p.agent),
      label: str(p.label) || str(p.agent),
      description: str(p.description),
    }));
}

/** Normalises a `stage` event; null when it names no agent. */
export function normalizeStageEvent(raw: unknown): StageEvent | null {
  if (!isBag(raw) || !str(raw.agent)) return null;
  return {
    seq: num(raw.seq),
    agent: str(raw.agent),
    label: str(raw.label) || str(raw.agent),
    description: str(raw.description),
    summary: str(raw.summary),
    detail: isBag(raw.detail) ? raw.detail : {},
    duration_ms: num(raw.duration_ms),
    elapsed_ms: num(raw.elapsed_ms),
    perf: normalizePerf(raw.perf),
  };
}

/** Reads a list of strings out of an untyped detail bag. */
export function stringList(value: unknown): string[] {
  return list(value).filter((v): v is string => typeof v === "string" && v.length > 0);
}
