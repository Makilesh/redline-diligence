import { describe, expect, it } from "vitest";

import {
  normalizeIngestResult,
  normalizeQueryResponse,
  normalizeStageEvent,
} from "@/lib/normalize";
import { wireResponse } from "./fixtures";

describe("normalizeQueryResponse", () => {
  it("carries the integration fields through", () => {
    const r = normalizeQueryResponse(wireResponse())!;
    expect(r.outcome).toBe("answered");
    expect(r.request_id).toBe("req-123");
    expect(r.llm_calls).toBe(1);
    expect(r.node_timings[0]).toMatchObject({ agent: "answer_synthesizer", prompt_tokens: 3000 });
    expect(r.claim_checks[0].numbers[0]).toMatchObject({ raw: "$452.8 million", binding: "bound" });
  });

  it("returns null for something that is not a response", () => {
    expect(normalizeQueryResponse(null)).toBeNull();
    expect(normalizeQueryResponse({ detail: "x" })).toBeNull();
  });

  it("defaults every missing list instead of leaving undefined", () => {
    const r = normalizeQueryResponse({ answer: "x" })!;
    expect(r.citations).toEqual([]);
    expect(r.claim_checks).toEqual([]);
    expect(r.hallucination_flags).toEqual([]);
    expect(r.node_timings).toEqual([]);
    expect(r.agent_trace).toEqual([]);
  });

  describe("outcome for a response that predates it", () => {
    const legacy = (o: Record<string, unknown>) => {
      const w = wireResponse(o);
      delete w.outcome;
      return normalizeQueryResponse(w)!.outcome;
    };

    it("reads a synthesis failure from the trace, not as a refusal", () => {
      expect(
        legacy({ is_refusal: true, agent_trace: [{ agent: "synthesis_unavailable", force_refusal: true }] }),
      ).toBe("synthesis_failed");
      expect(legacy({ agent_trace: [{ agent: "answer_synthesizer", synthesis_failed: true }] })).toBe(
        "synthesis_failed",
      );
    });

    it("reads a refusal and a failed verification", () => {
      expect(legacy({ is_refusal: true })).toBe("refused");
      expect(legacy({ validation_status: "failed" })).toBe("answered_unverified");
      expect(legacy({})).toBe("answered");
    });

    it("derives rather than trusts an outcome it does not know", () => {
      expect(normalizeQueryResponse(wireResponse({ outcome: "new_thing", is_refusal: true }))!.outcome).toBe(
        "refused",
      );
    });
  });

  it("never upgrades an unknown claim or figure status", () => {
    const r = normalizeQueryResponse(
      wireResponse({
        claim_checks: [
          { claim: "A", status: "probably_fine", method: "vibes", numbers: [{ raw: "$1", status: "close_enough" }] },
          { claim: "B", status: "not_checked", method: "judge" },
        ],
      }),
    )!;
    expect(r.claim_checks[0]).toMatchObject({ status: "unverified", method: "none" });
    expect(r.claim_checks[0].numbers[0].status).toBe("unsupported");
    expect(r.claim_checks[1]).toMatchObject({ status: "unverified", method: "llm", reason: "not checked" });
  });

  it("keeps an unknown validation status as sent, for the verdict to treat as unknown", () => {
    expect(normalizeQueryResponse(wireResponse({ validation_status: undefined }))!.validation_status).toBe("");
  });
});

describe("normalizeStageEvent", () => {
  it("keeps the node's perf block", () => {
    const s = normalizeStageEvent({
      seq: 3,
      agent: "hallucination_validator",
      summary: "12 claims",
      perf: { duration_ms: 812, llm_calls: 1, prompt_tokens: 10, completion_tokens: 5, models: ["judge-model"] },
    })!;
    expect(s.perf).toEqual({
      duration_ms: 812,
      llm_calls: 1,
      prompt_tokens: 10,
      completion_tokens: 5,
      models: ["judge-model"],
    });
    expect(s.label).toBe("hallucination_validator");
  });
});

describe("normalizeIngestResult", () => {
  it("keeps notices and describable version candidates only", () => {
    const r = normalizeIngestResult({
      doc_id: "d2",
      deal_id: "sbx",
      document_category: "financial",
      chunks_created: 40,
      status: "partial",
      notices: ["Sheet 'Q4' could not be read."],
      version_candidates: [
        { doc_id: "d1", source_file: "SPA v1.docx", relation: "newer" },
        { doc_id: "d0", source_file: "x", relation: "cousin" },
      ],
    })!;
    expect(r.status).toBe("partial");
    expect(r.notices).toEqual(["Sheet 'Q4' could not be read."]);
    expect(r.version_candidates).toEqual([{ doc_id: "d1", source_file: "SPA v1.docx", relation: "newer" }]);
  });

  it("defaults the lists an older server does not send", () => {
    const r = normalizeIngestResult({ doc_id: "d", deal_id: "x", chunks_created: 3 })!;
    expect(r.notices).toEqual([]);
    expect(r.version_candidates).toEqual([]);
    expect(r.status).toBe("success");
  });
});
