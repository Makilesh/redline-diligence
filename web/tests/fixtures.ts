/**
 * Wire payloads shaped like the integration API's (api/routes/query.py
 * `_build_response`, src/verification/claim_checker.py) for tests.
 */

import { normalizeQueryResponse } from "@/lib/normalize";
import type { QueryResponse } from "@/lib/types";

export const supportedClaim = {
  claim: "Revenue was $452.8 million in FY2023.",
  status: "supported",
  method: "numeric",
  score: 0.41,
  numbers: [{ raw: "$452.8 million", status: "grounded", binding: "bound", evidence_source: "10-K.pdf" }],
  cited: true,
  reason: "every figure appears in, or is computed from, the retrieved documents",
  evidence: "Total revenue | 452.8 | 387.1",
  evidence_source: "10-K.pdf",
  evidence_page: null,
  evidence_section: "",
};

export const contradictedClaim = {
  claim: "The termination fee is $20 million.",
  status: "contradicted",
  method: "llm",
  score: 0.97,
  numbers: [],
  cited: true,
  reason: "The merger agreement sets the fee at $13.92 million.",
  evidence: "",
  evidence_source: "Merger Agreement.pdf",
  evidence_page: 41,
  evidence_section: "Section 8.3 Termination Fee",
};

export const notCheckedClaim = {
  claim: "The company has 12 subsidiaries.",
  status: "unverified",
  method: "none",
  score: null,
  numbers: [],
  cited: false,
  reason: "not checked: beyond the first 40 statements that needed checking against the documents' text",
  evidence: "",
  evidence_source: "",
  evidence_page: null,
  evidence_section: "",
};

export function wireResponse(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    answer: "Revenue was $452.8 million in FY2023 [10-K.pdf, p. 4].",
    query_type: "financial",
    confidence_score: 0.92,
    validation_status: "passed",
    citations: [{ chunk_id: "c1", source_file: "10-K.pdf", page_number: 4 }],
    hallucination_flags: [],
    claim_checks: [supportedClaim],
    total_latency_ms: 18234.5,
    session_id: "s-1",
    rewrite_iterations: 0,
    agent_trace: [{ agent: "answer_synthesizer", perf: { duration_ms: 900, llm_calls: 1 } }],
    is_refusal: false,
    context_quality_score: 0.81,
    outcome: "answered",
    node_timings: [
      { agent: "answer_synthesizer", duration_ms: 900, llm_calls: 1, prompt_tokens: 3000, completion_tokens: 200, models: ["m"] },
    ],
    llm_calls: 1,
    request_id: "req-123",
    ...overrides,
  };
}

export function response(overrides: Record<string, unknown> = {}): QueryResponse {
  const r = normalizeQueryResponse(wireResponse(overrides));
  if (!r) throw new Error("fixture did not normalise");
  return r;
}
