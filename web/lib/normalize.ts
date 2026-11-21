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
  ChecklistCell,
  ChecklistColumn,
  ChecklistPassage,
  ChecklistReport,
  ChecklistRow,
  ChecklistStatus,
  Citation,
  CitationMismatch,
  ClaimCheck,
  ClaimMethod,
  ClaimStatus,
  FigureBinding,
  FigureCheck,
  FigureStatus,
  IngestResult,
  InjectionReview,
  InjectionScreen,
  NodePerf,
  NodeTiming,
  Outcome,
  PassageLocation,
  PlannedStage,
  QueryResponse,
  StageEvent,
  TieOutFigure,
  TieOutReport,
  TieOutValue,
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

function normalizeMismatch(raw: unknown): CitationMismatch | null {
  if (!isBag(raw) || !str(raw.figure)) return null;
  return { figure: str(raw.figure), found_in: stringList(raw.found_in) };
}

function normalizeClaim(raw: unknown): ClaimCheck | null {
  if (!isBag(raw) || !str(raw.claim)) return null;
  const mismatch = list(raw.citation_mismatch)
    .map(normalizeMismatch)
    .filter((m): m is CitationMismatch => m !== null);
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
    suspected_misstatement: raw.suspected_misstatement === true || undefined,
    absence: raw.absence === true || undefined,
    citation_mismatch: mismatch.length > 0 ? mismatch : undefined,
    cited_documents: mismatch.length > 0 ? stringList(raw.cited_documents) : undefined,
    injection_suspect: raw.injection_suspect === true || undefined,
  };
}

function normalizeLocation(raw: Bag): PassageLocation {
  const page = numOrNull(raw.page_number);
  return {
    page_number: page !== null && page >= 1 ? page : null,
    section_heading: str(raw.section_heading),
    signals: stringList(raw.signals),
  };
}

/** Normalises `injection_review`; absent (an older server) means none found. */
function normalizeInjectionReview(raw: unknown): InjectionReview {
  if (!isBag(raw)) return { flagged_passages: [], cited_flagged_passages: 0, tainted_figures: [] };
  return {
    flagged_passages: list(raw.flagged_passages)
      .filter(isBag)
      .map((p) => ({ source_file: str(p.source_file), ...normalizeLocation(p) })),
    cited_flagged_passages: Math.max(0, Math.floor(num(raw.cited_flagged_passages))),
    tainted_figures: stringList(raw.tainted_figures),
  };
}

/** Normalises `injection_screen`; absent (an older server) means nothing flagged. */
function normalizeInjectionScreen(raw: unknown): InjectionScreen {
  if (!isBag(raw)) return { flagged_passages: 0, signals: [], locations: [] };
  return {
    flagged_passages: Math.max(0, Math.floor(num(raw.flagged_passages))),
    signals: stringList(raw.signals),
    locations: list(raw.locations).filter(isBag).map(normalizeLocation),
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
      (t.agent === "answer_synthesizer" && t.synthesis_failed === true) ||
      // Refused only because the assessor reached no model: an outage too.
      (t.agent === "insufficient_context" && t.reason === "assessor_unavailable"),
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
    // Only an explicit true marks an answer as cut off. A server that predates
    // the field cannot say either way; its verdict still stands on its own.
    answer_truncated: raw.answer_truncated === true,
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
    injection_review: normalizeInjectionReview(raw.injection_review),
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
    injection_screen: normalizeInjectionScreen(raw.injection_screen),
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

// The server caps these too (src/reports/tie_out.py); the client does not rely
// on it, because every one of them is document text.
export const TIE_OUT_LIMITS = {
  figures: 200,
  values: 10,
  label: 120,
  printed: 40,
  source: 200,
  section: 120,
  period: 40,
  documents: 50,
} as const;

const clip = (v: unknown, limit: number): string => {
  const flat = str(v).replace(/\s+/g, " ").trim();
  return flat.length <= limit ? flat : `${flat.slice(0, limit - 1)}…`;
};

function normalizeTieOutValue(raw: unknown): TieOutValue | null {
  if (!isBag(raw) || !str(raw.source_file)) return null;
  return {
    source_file: clip(raw.source_file, TIE_OUT_LIMITS.source),
    page_number: numOrNull(raw.page_number),
    section_heading: clip(raw.section_heading, TIE_OUT_LIMITS.section),
    as_printed: clip(raw.as_printed, TIE_OUT_LIMITS.printed),
    value: num(raw.value),
    scale_known: raw.scale_known !== false,
  };
}

function normalizeTieOutFigure(raw: unknown): TieOutFigure | null {
  if (!isBag(raw) || !str(raw.metric)) return null;
  // Only an explicit "agree" is an agreement: anything else is shown as the
  // less favourable reading, a conflict to look at.
  const status = raw.status === "agree" ? "agree" : "conflict";
  const values = list(raw.values)
    .slice(0, TIE_OUT_LIMITS.values)
    .map(normalizeTieOutValue)
    .filter((v): v is TieOutValue => v !== null);
  return {
    metric: clip(raw.metric, TIE_OUT_LIMITS.label),
    period: clip(raw.period, TIE_OUT_LIMITS.period),
    kind: oneOf(raw.kind, ["amount", "percent", "multiple"] as const) ?? "amount",
    status,
    max_deviation_pct: num(raw.max_deviation_pct),
    values,
    more_documents: num(raw.more_documents),
  };
}

/**
 * Normalises a `GET /deals/{id}/tie-out` report.
 *
 * @returns The typed report, or null when the payload is not one (no deal_id).
 */
export function normalizeTieOut(raw: unknown): TieOutReport | null {
  if (!isBag(raw) || typeof raw.deal_id !== "string") return null;
  const figures = (value: unknown) =>
    list(value)
      .slice(0, TIE_OUT_LIMITS.figures)
      .map(normalizeTieOutFigure)
      .filter((f): f is TieOutFigure => f !== null);
  const conflicts = figures(raw.conflicts).map((f) => ({ ...f, status: "conflict" as const }));
  const agreements = figures(raw.agreements).filter((f) => f.status === "agree");
  return {
    deal_id: raw.deal_id,
    conflicts,
    agreements,
    conflicts_total: Math.max(num(raw.conflicts_total), conflicts.length),
    agreements_total: Math.max(num(raw.agreements_total), agreements.length),
    single_source_figures: num(raw.single_source_figures),
    ambiguous_figures: num(raw.ambiguous_figures),
    generic_figures: num(raw.generic_figures),
    figures_without_period: num(raw.figures_without_period),
    documents_total: num(raw.documents_total),
    documents_read: num(raw.documents_read),
    documents_not_read: stringList(raw.documents_not_read)
      .slice(0, TIE_OUT_LIMITS.documents)
      .map((d) => clip(d, TIE_OUT_LIMITS.source)),
    chunks_read: num(raw.chunks_read),
    // A report that does not say it read everything is not taken to have.
    truncated: raw.truncated !== false,
    truncation_reason: str(raw.truncation_reason),
    elapsed_ms: num(raw.elapsed_ms),
  };
}

// The server caps these too (src/reports/checklist.py); the client does not
// rely on it, because the passages and names are document text.
export const CHECKLIST_LIMITS = {
  items: 30,
  rows: 200,
  passage: 800,
  alternate: 240,
  alternates: 2,
  source: 200,
  section: 120,
  label: 80,
  question: 400,
  documents: 50,
} as const;

const CHECKLIST_STATUSES: readonly ChecklistStatus[] = ["found", "not_found", "unclear"];

function normalizeChecklistPassage(raw: unknown, limit: number): ChecklistPassage | null {
  if (!isBag(raw) || !str(raw.text).trim()) return null;
  return {
    text: clip(raw.text, limit),
    page_number: numOrNull(raw.page_number),
    section_heading: clip(raw.section_heading, CHECKLIST_LIMITS.section),
    chunk_id: str(raw.chunk_id),
    score: num(raw.score),
  };
}

function normalizeChecklistCell(raw: unknown, itemId: string): ChecklistCell {
  const bag = isBag(raw) ? raw : {};
  return {
    item_id: itemId,
    // An unrecognised status is neither found nor absent: it is read as unclear.
    status: oneOf(bag.status, CHECKLIST_STATUSES) ?? "unclear",
    score: numOrNull(bag.score),
    answerability: numOrNull(bag.answerability),
    passage: normalizeChecklistPassage(bag.passage, CHECKLIST_LIMITS.passage),
    alternates: list(bag.alternates)
      .slice(0, CHECKLIST_LIMITS.alternates)
      .map((a) => normalizeChecklistPassage(a, CHECKLIST_LIMITS.alternate))
      .filter((a): a is ChecklistPassage => a !== null),
  };
}

function normalizeChecklistColumn(raw: unknown): ChecklistColumn | null {
  if (!isBag(raw) || !str(raw.id)) return null;
  return {
    id: clip(raw.id, 64),
    label: clip(raw.label, CHECKLIST_LIMITS.label) || clip(raw.id, 64),
    question: clip(raw.question, CHECKLIST_LIMITS.question),
    found: num(raw.found),
    unclear: num(raw.unclear),
    not_found: num(raw.not_found),
  };
}

/**
 * Normalises a `GET /deals/{id}/checklist` grid. Each row's cells are put in
 * the order of the items; a cell the row lacks is unclear, never absent.
 *
 * @returns The typed grid, or null when the payload is not one (no deal_id).
 */
export function normalizeChecklist(raw: unknown): ChecklistReport | null {
  if (!isBag(raw) || typeof raw.deal_id !== "string") return null;
  const items = list(raw.items)
    .slice(0, CHECKLIST_LIMITS.items)
    .map(normalizeChecklistColumn)
    .filter((c): c is ChecklistColumn => c !== null);
  const rows: ChecklistRow[] = list(raw.rows)
    .slice(0, CHECKLIST_LIMITS.rows)
    .filter(isBag)
    .map((row) => {
      const byItem = new Map<string, unknown>();
      for (const cell of list(row.cells)) {
        if (isBag(cell) && typeof cell.item_id === "string" && !byItem.has(cell.item_id)) {
          byItem.set(cell.item_id, cell);
        }
      }
      return {
        doc_id: str(row.doc_id),
        source_file: clip(row.source_file, CHECKLIST_LIMITS.source),
        document_category: clip(row.document_category, 40),
        cells: items.map((item) => normalizeChecklistCell(byItem.get(item.id), item.id)),
      };
    });
  return {
    deal_id: raw.deal_id,
    items,
    rows,
    documents_total: num(raw.documents_total),
    // What is shown is what was checked.
    documents_checked: rows.length,
    documents_not_checked: stringList(raw.documents_not_checked)
      .slice(0, CHECKLIST_LIMITS.documents)
      .map((d) => clip(d, CHECKLIST_LIMITS.source)),
    documents_excluded: list(raw.documents_excluded)
      .filter(isBag)
      .slice(0, CHECKLIST_LIMITS.documents)
      .map((d) => ({
        source_file: clip(d.source_file, CHECKLIST_LIMITS.source),
        document_category: clip(d.document_category, 40),
      })),
    excluded_total: num(raw.excluded_total),
    // A grid that does not say it checked every contract is not taken to have.
    truncated: raw.truncated !== false,
    truncation_reason: str(raw.truncation_reason),
    reranker: clip(raw.reranker, 64),
    answerability: raw.answerability === true,
    elapsed_ms: num(raw.elapsed_ms),
  };
}
