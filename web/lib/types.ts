/**
 * Wire types for the FastAPI backend.
 *
 * These mirror `api/models/response_models.py` and the SSE payloads built in
 * `src/workflow/orchestrator.py` / `api/routes/query.py`. Nothing enforces the
 * correspondence at build time, so every payload that crosses the network is
 * also normalised at runtime (`lib/normalize.ts`) — a field the server stops
 * sending degrades to an empty default instead of throwing inside the render
 * tree, and an unrecognised verdict is never read as a favourable one.
 */

export interface Citation {
  chunk_id: string;
  source_file: string;
  page_number: number | null;
  section_heading: string;
  is_current_version: boolean;
  content_type: string;
  is_redline: boolean;
  superseded_by: string;
}

/**
 * The validator's verdict. Typed as the three values the server emits, but the
 * UI never assumes that list is closed — an unrecognised status renders as
 * "not verified", not as a pass (see `lib/verification.ts`).
 */
export type ValidationStatus = "passed" | "warning" | "failed";

/**
 * A claim's verdict (`src/verification/claim_checker.py`). "contradicted" is
 * only ever an LLM judge's verdict; "unverified" covers both a claim no check
 * could settle and one beyond the checking budget ("not checked").
 */
export type ClaimStatus = "supported" | "contradicted" | "unsupported" | "unverified";

/** How a claim's verdict was reached. */
export type ClaimMethod = "numeric" | "nli" | "llm" | "none";

export type FigureStatus =
  | "grounded"
  | "derived"
  | "derived_unverified"
  | "unsupported";

/**
 * Whether a grounded figure sits where the claim says it does: "bound" (the
 * row or sentence the claim names), "unbound" (the value occurs, but somewhere
 * the claim does not name), "conflict" (the claim's row or period gives a
 * different value). Absent for figures computed from the claim's own figures.
 */
export type FigureBinding = "bound" | "unbound" | "conflict";

/** One figure inside a claim, as located (or not) in the retrieved text. */
export interface FigureCheck {
  raw: string;
  status: FigureStatus;
  /** Present when the figure was reproduced from other grounded figures. */
  formula?: string;
  evidence_source?: string;
  binding?: FigureBinding;
  /** Why the binding is what it is, e.g. "that is the FY2022 column". */
  binding_note?: string;
}

/** One entry of `QueryResponse.claim_checks`. */
export interface ClaimCheck {
  claim: string;
  status: ClaimStatus;
  method: ClaimMethod;
  /** NLI entailment (or suspected-contradiction) score, when one was computed. */
  score: number | null;
  numbers: FigureCheck[];
  /** Whether the claim carried a citation that resolved to a retrieved chunk. */
  cited: boolean;
  reason: string;
  /** The passage the verdict rests on. */
  evidence: string;
  evidence_source: string;
  evidence_page: number | null;
  evidence_section: string;
  /** Set when NLI disagreed with a figure-grounded "supported" verdict. */
  nli_contradiction?: number;
  /** The NLI model raised a contradiction the judge was asked to confirm. */
  suspected_contradiction?: boolean;
  /** A figure was found under a different row or period than the claim names. */
  suspected_misattribution?: boolean;
  /** The claim says the documents do not contain something. */
  absence?: boolean;
}

/**
 * How a run ended (`run_outcome` in `src/workflow/orchestrator.py`). This, not
 * `is_refusal`, decides what the page shows: `is_refusal` is also true for
 * "synthesis_failed", which is an outage, not a considered refusal.
 *
 *   answered            — verified, or verified with warnings
 *   answered_unverified — failed verification after the one re-synthesis;
 *                         shipped with its flags
 *   refused             — no usable context; the pipeline declined
 *   synthesis_failed    — no model returned a usable answer; nothing verified
 */
export type Outcome =
  | "answered"
  | "answered_unverified"
  | "refused"
  | "synthesis_failed";

/** A node's own measurement (`perf` on trace entries and stage events). */
export interface NodePerf {
  duration_ms: number;
  llm_calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  models: string[];
}

/** One row of `QueryResponse.node_timings`, one per node execution. */
export interface NodeTiming extends NodePerf {
  agent: string;
}

export interface QueryResponse {
  answer: string;
  query_type: string;
  confidence_score: number;
  validation_status: ValidationStatus;
  citations: Citation[];
  hallucination_flags: string[];
  claim_checks: ClaimCheck[];
  total_latency_ms: number;
  session_id: string;
  rewrite_iterations: number;
  agent_trace: Record<string, unknown>[];
  is_refusal: boolean;
  context_quality_score: number;
  outcome: Outcome;
  node_timings: NodeTiming[];
  /** Completed LLM calls across all nodes. */
  llm_calls: number;
  /** Correlation id, also sent as X-Request-ID; quoted when reporting a problem. */
  request_id: string;
}

export interface Deal {
  deal_id: string;
  deal_name: string;
  description: string;
  document_count: number;
  status: string;
  is_sandbox: boolean;
  expires_at: string;
}

export interface DocumentRecord {
  doc_id: string;
  filename: string;
  document_category: string;
  chunks_created: number;
  version_label: string;
  upload_date: string;
  is_current_version: boolean;
  supersedes_doc_id: string;
  superseded_by: string;
  has_redline: boolean;
}

export type Severity = "high" | "medium" | "low";

export interface RiskSignal {
  signal_type: string;
  severity: Severity;
  source_file: string;
  description: string;
  page_number: number | null;
}

/**
 * An indexed document the upload looks related to. A proposal only: nothing
 * has been superseded, and the UI must not present it as if it had.
 */
export interface VersionCandidate {
  doc_id: string;
  source_file: string;
  /** How the NEW upload relates to this document. */
  relation: "newer" | "older" | "same_document";
}

/** `POST /ingest` response. `status` is "partial" when part of the file failed. */
export interface IngestResult {
  doc_id: string;
  deal_id: string;
  document_category: string;
  chunks_created: number;
  status: string;
  /** Why a partial ingestion was partial, when known. */
  notices: string[];
  version_candidates: VersionCandidate[];
}

export interface BudgetEntry {
  used: number;
  limit: number;
  remaining: number;
}

export type BudgetStatus = Record<string, BudgetEntry>;

/**
 * One row of the live pipeline timeline.
 *
 * `status` is client-side only. The backend emits a stage once it has
 * completed, so "running" is inferred as "the next row after the last one that
 * reported". "skipped" is a planned stage the run ended without reaching (a
 * refusal stops before synthesis); "interrupted" is the stage that was in
 * flight when the run failed or was stopped. Neither is ever shown as done.
 */
export interface PipelineStage {
  seq: number;
  agent: string;
  label: string;
  description: string;
  summary: string;
  detail: Record<string, unknown>;
  duration_ms: number;
  elapsed_ms: number;
  /** The node's own measurement, when the server sent one. */
  perf?: NodePerf;
  status: "pending" | "running" | "done" | "skipped" | "interrupted";
}

/** A `stage` SSE payload — a PipelineStage before the client assigns a status. */
export type StageEvent = Omit<PipelineStage, "status">;

export interface PlannedStage {
  agent: string;
  label: string;
  description: string;
}
