import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import PipelineTimeline from "@/components/PipelineTimeline";
import type { PipelineStage } from "@/lib/types";

/*
 * Decision 96: a public caller gets every file name that reads like an
 * instruction as the same placeholder, so a stage's sources can repeat.
 */

const WITHHELD = "(withheld: instruction-like text; signals: model_address)";

const stage: PipelineStage = {
  seq: 2,
  agent: "retrieval_executor",
  label: "Retrieval Executor",
  description: "",
  summary: "3 chunks reranked from 1 retrieval pass across 3 documents",
  detail: { reranked_count: 3, retrieval_passes: 1, sources: [WITHHELD, WITHHELD, "clean.txt"] },
  duration_ms: 10,
  elapsed_ms: 20,
  status: "done",
};

afterEach(() => vi.restoreAllMocks());

describe("pipeline timeline", () => {
  it("shows repeated sources once each, without a duplicate React key", () => {
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    render(<PipelineTimeline stages={[stage]} running={false} elapsedMs={20} />);
    fireEvent.click(screen.getByRole("button", { name: /detail/i }));

    expect(screen.getAllByText(WITHHELD)).toHaveLength(2);
    expect(screen.getByText("clean.txt")).toBeInTheDocument();
    const keyWarnings = errors.mock.calls.filter((call) => String(call[0]).includes("same key"));
    expect(keyWarnings).toEqual([]);
  });
});
