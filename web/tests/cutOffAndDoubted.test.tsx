import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import AnswerPanel from "@/components/AnswerPanel";
import ClaimChecks from "@/components/ClaimChecks";
import { streamQuery, type StreamEvent } from "@/lib/api";
import { normalizeQueryResponse } from "@/lib/normalize";
import { answerVerdict, claimMarker } from "@/lib/verification";
import { contradictedClaim, response, supportedClaim, wireResponse } from "./fixtures";

/*
 * Two signals the API sends that the page used to drop on the floor:
 *   answer_truncated               — the answer stopped at the model's output
 *                                    limit (Decision 66, streamed route);
 *   claim_checks[].suspected_misstatement — every figure checked out, but the
 *                                    local checker doubted the wording and left
 *                                    the claim to the judge (Decision 64).
 */

// A cut answer the server downgraded from "passed" to "warning" (Decision 66).
const CUT = {
  answer: "Revenue was $452.8 million in FY2023 [10-K.pdf, p. 4], while Adjusted EBITDA",
  answer_truncated: true,
  validation_status: "warning",
  hallucination_flags: ["The answer was cut off at the model's output limit"],
};

// The figure is right; the judge confirmed the wording after the local doubt.
const doubtedClaim = {
  ...supportedClaim,
  claim: "The buyer pays the termination fee of $13.92 million.",
  method: "llm",
  suspected_misstatement: true,
  reason: "The agreement has the buyer paying the fee.",
};

describe("normalising the two signals", () => {
  it("reads answer_truncated only when it is explicitly true", () => {
    expect(normalizeQueryResponse(wireResponse(CUT))!.answer_truncated).toBe(true);
    expect(normalizeQueryResponse(wireResponse())!.answer_truncated).toBe(false);
    expect(normalizeQueryResponse(wireResponse({ answer_truncated: "yes" }))!.answer_truncated).toBe(false);
  });

  it("keeps suspected_misstatement on a claim, and leaves it off when absent", () => {
    const r = normalizeQueryResponse(wireResponse({ claim_checks: [doubtedClaim, supportedClaim] }))!;
    expect(r.claim_checks[0].suspected_misstatement).toBe(true);
    expect(r.claim_checks[1].suspected_misstatement).toBeUndefined();
  });
});

describe("the verdict of a cut answer", () => {
  it("is never Verified, and says why when every statement checked out", () => {
    // Even a server that still said "passed" for a cut answer is not believed.
    for (const status of ["warning", "passed"]) {
      const v = answerVerdict(response({ ...CUT, validation_status: status }));
      expect(v.key).toBe("partial");
      expect(v.detail).toBe("All 1 statement matched to the documents, but the answer was cut off before it finished.");
      expect(v.detail).not.toMatch(/0 could not be confirmed/);
    }
  });

  it("adds the cut-off to the existing reasons when statements are also unconfirmed", () => {
    const v = answerVerdict(response({ ...CUT, claim_checks: [supportedClaim, contradictedClaim] }));
    expect(v.detail).toBe(
      "1 of 2 statements matched to the documents; 1 could not be confirmed. Also, the answer was cut off before it finished.",
    );
    const failed = answerVerdict(response({ ...CUT, validation_status: "failed", outcome: "answered_unverified" }));
    expect(failed.key).toBe("failed");
    expect(failed.detail).toMatch(/Also, the answer was cut off before it finished\.$/);
  });

  it("is unchanged for an answer that was not cut off", () => {
    expect(answerVerdict(response()).key).toBe("verified");
    expect(answerVerdict(response({ validation_status: "warning" })).detail).toBe(
      "1 of 1 statement matched to the documents; 0 could not be confirmed.",
    );
  });
});

describe("AnswerPanel", () => {
  it("marks where a cut answer stops", () => {
    render(<AnswerPanel result={response(CUT)} />);
    const notes = screen.getAllByRole("note").map((n) => n.textContent);
    expect(notes).toContain("Cut off here: the model reached its output limit, so the answer is incomplete.");
  });

  it("shows no cut-off notice on a complete answer or on a streaming draft", () => {
    const { unmount } = render(<AnswerPanel result={response()} />);
    expect(screen.queryByText(/Cut off here/)).toBeNull();
    unmount();
    render(<AnswerPanel result={null} draft="Revenue was" />);
    expect(screen.queryByText(/Cut off here/)).toBeNull();
  });
});

describe("ClaimChecks", () => {
  it("marks a doubted claim beside its verdict, without changing the verdict", async () => {
    const r = response({ claim_checks: [doubtedClaim, supportedClaim] });
    render(<ClaimChecks checks={r.claim_checks} />);
    // Both are supported, so they start folded away.
    await userEvent.click(screen.getByRole("button", { name: /Show 2 supported statements/ }));

    const marker = screen.getByText("Wording doubted");
    expect(marker.closest("[title]")?.getAttribute("title")).toMatch(/left to the LLM judge/);
    const row = marker.closest("li")!;
    expect(row.getAttribute("data-status")).toBe("supported");
    expect(row.textContent).toContain("Supported");
    expect(screen.getAllByText("Wording doubted")).toHaveLength(1);
  });

  it("has a marker only for doubted claims", () => {
    expect(claimMarker(response({ claim_checks: [doubtedClaim] }).claim_checks[0])?.label).toBe("Wording doubted");
    expect(claimMarker(response().claim_checks[0])).toBeNull();
  });
});

describe("the streamed result frame", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("carries answer_truncated and suspected_misstatement through to the page", async () => {
    const payload = wireResponse({ ...CUT, claim_checks: [doubtedClaim] });
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(`event: result\ndata: ${JSON.stringify(payload)}\n\n`));
        controller.close();
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } })),
    );

    const events: StreamEvent[] = [];
    await streamQuery({ query: "q", dealId: "deal", includePii: false, onEvent: (e) => events.push(e) });

    const result = events.at(-1)!;
    expect(result.event).toBe("result");
    if (result.event !== "result") return;
    expect(result.data.answer_truncated).toBe(true);
    expect(result.data.claim_checks[0].suspected_misstatement).toBe(true);
  });
});
