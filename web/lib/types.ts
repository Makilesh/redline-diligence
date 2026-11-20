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
  /**
   * Every figure checked out, but the local checker doubted the statement as
   * worded (another party, a flipped negation) and sent it to the LLM judge.
   * The verdict in `status` is the judge's; this records that it was doubted.
   */
  suspected_misstatement?: boolean;
  /** The claim says the documents do not contain something. */
  absence?: boolean;
  /**
   * Figures of the claim that no document it cites states, and where they are
   * stated instead (Decision 78). The verdict in `status` is unaffected: the
   * figure is in the documents, the citation points elsewhere.
   */
  citation_mismatch?: CitationMismatch[];
  /** The retrieved documents the claim's citations name (set with a mismatch). */
  cited_documents?: string[];
  /**
   * A figure of the claim is stated only by document text that reads like an
   * instruction to an AI model (Decision 76). The verdict in `status` may still
   * be "supported": the figure is in the documents.
   */
  injection_suspect?: boolean;
}

/**
 * The ingest screen's signals (`src/data_processing/injection_screen.py`).
 * Typed as plain strings on the wire: a signal this client does not know is
 * kept, and simply not described.
 */
export type InjectionSignal =
  | "instruction_override"
  | "model_address"
  | "role_marker"
  | "forged_marker"
  | "supersession_claim"
  | "invisible_text";

/** Where a passage with instruction-like text sits. Never what it says. */
export interface PassageLocation {
  page_number: number | null;
  section_heading: string;
  signals: string[];
}

/** A retrieved passage with instruction-like text, by document and place. */
export interface FlaggedPassage extends PassageLocation {
  source_file: string;
}

/**
 * `QueryResponse.injection_review` (Decision 82): retrieved passages whose text
 * reads like instructions to an AI model, how many the answer cites, and the
 * figures only they state. Empty lists and 0 when there were none.
 */
export interface InjectionReview {
  flagged_passages: FlaggedPassage[];
  cited_flagged_passages: number;
  tainted_figures: string[];
}

/**
 * `IngestResult.injection_screen` (Decision 82): passages of the upload whose
 * text reads like instructions to an AI model. A label only — the file was
 * indexed as written, and `status` does not change for it.
 */
export interface InjectionScreen {
  flagged_passages: number;
  signals: string[];
  /** Distinct pages/sections, at most ten. */
  locations: PassageLocation[];
}

/** One figure cited to a document that does not state it. */
export interface CitationMismatch {
  figure: string;
  found_in: string[];
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
  /**
   * The answer stopped at the model's output limit rather than where the model
   * ended it (streamed route only). The server never reports such an answer as
   * "passed"; the page also says plainly that the text is incomplete.
   */
  answer_truncated: boolean;
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
  /** Instruction-like text behind the answer; empty when there was none. */
  injection_review: InjectionReview;
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
  /** Passages whose text reads like instructions to an AI model (Decision 82). Absent from older servers. */
  instruction_like_passages?: number;
  instruction_signals?: string[];
}

export type Severity = "high" | "medium" | "low";

export interface RiskSignal {
  signal_type: string;
  severity: Severity;
  source_file: string;
  description: string;
  page_number: number | null;
}

/** One document's statement of a figure (`GET /deals/{id}/tie-out`). */
export interface TieOutValue {
  source_file: string;
  page_number: number | null;
  section_heading: string;
  /** The figure as the document prints it. Document text: rendered as text. */
  as_printed: string;
  /** Absolute value after unit and scale normalisation. */
  value: number;
  /** False when the document states no unit; compared at any power of 1,000. */
  scale_known: boolean;
}

/** A labelled figure two or more documents state (Decision 92). */
export interface TieOutFigure {
  metric: string;
  period: string;
  kind: "amount" | "percent" | "multiple";
  status: "conflict" | "agree";
  max_deviation_pct: number;
  values: TieOutValue[];
  more_documents: number;
}

/**
 * The deal-level figure tie-out: conflicts first, then agreements, then what
 * was not compared. Deterministic; no model wrote any of it.
 */
export interface TieOutReport {
  deal_id: string;
  conflicts: TieOutFigure[];
  agreements: TieOutFigure[];
  conflicts_total: number;
  agreements_total: number;
  single_source_figures: number;
  ambiguous_figures: number;
  generic_figures: number;
  figures_without_period: number;
  documents_total: number;
  documents_read: number;
  documents_not_read: string[];
  chunks_read: number;
  truncated: boolean;
  truncation_reason: string;
  elapsed_ms: number;
}

/** A window of a contract, verbatim (`GET /deals/{id}/checklist`). Document text: rendered as text. */
export interface ChecklistPassage {
  text: string;
  page_number: number | null;
  section_heading: string;
  chunk_id: string;
  /** Reranker score on the common scale, 0-1. */
  score: number;
}

export type ChecklistStatus = "found" | "not_found" | "unclear";

/** One contract's answer to one checklist item (Decision 95). */
export interface ChecklistCell {
  item_id: string;
  status: ChecklistStatus;
  score: number | null;
  /** Laya's P(the passage states it), when the grid read it. */
  answerability: number | null;
  /** The best evidence in the contract; null when nothing was retrieved. */
  passage: ChecklistPassage | null;
  alternates: ChecklistPassage[];
}

/** A contract and its cells, in the order of the report's items. */
export interface ChecklistRow {
  doc_id: string;
  source_file: string;
  document_category: string;
  cells: ChecklistCell[];
}

/** A checklist item and how many contracts carry it. */
export interface ChecklistColumn {
  id: string;
  label: string;
  question: string;
  found: number;
  unclear: number;
  not_found: number;
}

/**
 * The checklist review grid: the deal's contracts against a diligence
 * checklist, each cell the best evidence passage and a status. Deterministic;
 * no model wrote any of it.
 */
export interface ChecklistReport {
  deal_id: string;
  items: ChecklistColumn[];
  rows: ChecklistRow[];
  documents_total: number;
  documents_checked: number;
  documents_not_checked: string[];
  documents_excluded: { source_file: string; document_category: string }[];
  excluded_total: number;
  truncated: boolean;
  truncation_reason: string;
  reranker: string;
  answerability: boolean;
  elapsed_ms: number;
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
  injection_screen: InjectionScreen;
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
