import { describe, expect, it } from "vitest";

import { applyStage, initialStages, settleStages } from "@/lib/pipeline";
import type { StageEvent } from "@/lib/types";

const planned = ["query_intelligence", "retrieval_executor", "quality_assessor", "answer_synthesizer", "hallucination_validator"].map(
  (agent) => ({ agent, label: agent, description: "" }),
);

const stage = (agent: string, seq: number): StageEvent => ({
  seq,
  agent,
  label: agent,
  description: "",
  summary: `${agent} done`,
  detail: {},
  duration_ms: 10,
  elapsed_ms: 10 * seq,
});

const statuses = (s: ReturnType<typeof initialStages>) => s.map((x) => `${x.agent}:${x.status}`);

describe("pipeline timeline", () => {
  it("marks the first planned stage as running", () => {
    expect(initialStages(planned).map((s) => s.status)).toEqual([
      "running",
      "pending",
      "pending",
      "pending",
      "pending",
    ]);
  });

  it("completes stages in order and moves the running marker", () => {
    let s = initialStages(planned);
    s = applyStage(s, stage("query_intelligence", 1));
    expect(s.map((x) => x.status).slice(0, 2)).toEqual(["done", "running"]);
  });

  it("inserts a conditional stage after the last completed one", () => {
    let s = initialStages(planned);
    s = applyStage(s, stage("query_intelligence", 1));
    s = applyStage(s, stage("retrieval_executor", 2));
    s = applyStage(s, stage("financial_verifier", 3));
    expect(s[2].agent).toBe("financial_verifier");
    expect(s[2].status).toBe("done");
    expect(s[3]).toMatchObject({ agent: "quality_assessor", status: "running" });
  });

  it("records a second pass through the rewrite loop as its own row", () => {
    let s = initialStages(planned);
    for (const [i, a] of ["query_intelligence", "retrieval_executor", "quality_assessor", "query_rewriter", "retrieval_executor"].entries()) {
      s = applyStage(s, stage(a, i + 1));
    }
    expect(s.filter((x) => x.agent === "retrieval_executor" && x.status === "done")).toHaveLength(2);
  });

  it("never marks a stage the run did not reach as done", () => {
    // A refusal: the graph ends at insufficient_context, before synthesis.
    let s = initialStages(planned);
    for (const [i, a] of ["query_intelligence", "retrieval_executor", "quality_assessor", "insufficient_context"].entries()) {
      s = applyStage(s, stage(a, i + 1));
    }
    s = settleStages(s, "result");
    expect(statuses(s)).toEqual([
      "query_intelligence:done",
      "retrieval_executor:done",
      "quality_assessor:done",
      "insufficient_context:done",
      "answer_synthesizer:skipped",
      "hallucination_validator:skipped",
    ]);
  });

  it("marks the stage in flight as interrupted when the run breaks", () => {
    let s = initialStages(planned);
    s = applyStage(s, stage("query_intelligence", 1));
    s = settleStages(s, "failed");
    expect(s.map((x) => x.status)).toEqual(["done", "interrupted", "skipped", "skipped", "skipped"]);
  });
});
