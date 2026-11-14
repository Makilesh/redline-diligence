/**
 * Folding streamed stage events into the pipeline timeline.
 *
 * The server emits a stage once it has completed; everything else on a row —
 * running, skipped, interrupted — is the client's inference, made here.
 */

import type { PipelineStage, PlannedStage, StageEvent } from "./types";

/** The timeline as painted from the `start` event, before any stage reports. */
export function initialStages(planned: PlannedStage[]): PipelineStage[] {
  return planned.map((p, i) => ({
    ...p,
    seq: 0,
    summary: "",
    detail: {},
    duration_ms: 0,
    elapsed_ms: 0,
    // The first agent is already running by the time `start` arrives — the
    // server emits it immediately before invoking the graph.
    status: i === 0 ? "running" : "pending",
  }));
}

/**
 * Folds one completed stage into the timeline.
 *
 * Not a simple index bump, because the graph's shape is decided at runtime:
 * the financial verifier only runs for financial queries, and the rewriter
 * re-enters retrieval on a loop, so a stage can arrive that the planned list
 * never mentioned, or arrive twice.
 */
export function applyStage(prev: PipelineStage[], incoming: StageEvent): PipelineStage[] {
  const next = prev.map((s) => (s.status === "running" ? { ...s, status: "pending" as const } : s));

  // A planned row still waiting for this agent — the ordinary case.
  const slot = next.findIndex((s) => s.agent === incoming.agent && s.status === "pending");

  if (slot !== -1) {
    next[slot] = { ...next[slot], ...incoming, status: "done" };
  } else {
    // A conditional node or a second pass through the rewrite loop. Insert it
    // after the last completed row so the timeline reads in the order things
    // actually happened.
    let lastDone = -1;
    next.forEach((s, i) => {
      if (s.status === "done") lastDone = i;
    });
    next.splice(lastDone + 1, 0, { ...incoming, status: "done" });
  }

  // Whatever is next in line is now the one running.
  const upcoming = next.findIndex((s) => s.status === "pending");
  if (upcoming !== -1) next[upcoming] = { ...next[upcoming], status: "running" };

  return next;
}

/**
 * Settles the rows that never reported once the run is over.
 *
 * @param how "result" — the run finished: rows that never reported were not
 *   reached (a refusal stops before synthesis) and are "skipped". "failed" —
 *   the run broke: the row in flight is "interrupted", the rest "skipped".
 *   Neither is ever marked done; a check mark means the stage ran.
 */
export function settleStages(prev: PipelineStage[], how: "result" | "failed"): PipelineStage[] {
  return prev.map((s) => {
    if (s.status === "done") return s;
    if (how === "failed" && s.status === "running") return { ...s, status: "interrupted" };
    return { ...s, status: "skipped" };
  });
}
