import { describe, expect, it } from "vitest";

import { normalizeQueryResponse } from "@/lib/normalize";
import {
  answerVerdict,
  claimView,
  figureView,
  resultView,
  sortClaimsForReview,
} from "@/lib/verification";
import { contradictedClaim, notCheckedClaim, response, supportedClaim } from "./fixtures";

describe("resultView", () => {
  it("shows a synthesis failure as an outage even though is_refusal is true", () => {
    expect(resultView(response({ outcome: "synthesis_failed", is_refusal: true }))).toBe("unavailable");
  });

  it("shows a refusal as a refusal and every answered outcome as an answer", () => {
    expect(resultView(response({ outcome: "refused", is_refusal: true }))).toBe("refusal");
    expect(resultView(response({ outcome: "answered" }))).toBe("answer");
    expect(resultView(response({ outcome: "answered_unverified", validation_status: "failed" }))).toBe("answer");
  });

  it("shows a not-found as its own surface (Decision 112)", () => {
    expect(resultView(response({ outcome: "not_found" }))).toBe("not_found");
  });
});

describe("answerVerdict for a not-found (Decision 112)", () => {
  it("is never Verified, even when the server passed it and every claim is supported", () => {
    const v = answerVerdict(response({ outcome: "not_found" }));
    expect(v.key).toBe("not_found");
    expect(v.tone).toBe("warn");
    expect(v.detail).toMatch(/not against every document in the data room/);
  });

  it("still reads as failed when the validation failed", () => {
    const v = answerVerdict(
      response({ outcome: "not_found", validation_status: "failed", claim_checks: [contradictedClaim] }),
    );
    expect(v.key).toBe("failed");
  });

  it("is kept by the normaliser", () => {
    expect(normalizeQueryResponse({ ...response(), outcome: "not_found" })!.outcome).toBe("not_found");
  });
});

describe("answerVerdict", () => {
  it("says Verified only when the server passed it and every claim is supported", () => {
    const v = answerVerdict(response());
    expect(v.key).toBe("verified");
    expect(v.detail).toBe("All 1 statement matched to the retrieved documents.");
  });

  it("does not pass an answer with no checkable statements", () => {
    expect(answerVerdict(response({ claim_checks: [] })).key).toBe("no_claims");
  });

  it("does not trust 'passed' when a claim says otherwise", () => {
    const v = answerVerdict(response({ claim_checks: [supportedClaim, notCheckedClaim] }));
    expect(v.key).toBe("partial");
    expect(v.detail).toContain("1 of 2 statements");
  });

  it("treats a missing or unrecognised status as not verified, never as passed", () => {
    expect(answerVerdict(response({ validation_status: undefined })).key).toBe("unknown");
    expect(answerVerdict(response({ validation_status: "great" })).key).toBe("unknown");
  });

  it("fails an answer that failed verification after the retry", () => {
    const v = answerVerdict(
      response({
        outcome: "answered_unverified",
        validation_status: "failed",
        claim_checks: [supportedClaim, contradictedClaim],
      }),
    );
    expect(v.key).toBe("failed");
    expect(v.detail).toMatch(/^1 statement could not be matched/);
  });
});

describe("claimView", () => {
  const claim = (raw: Record<string, unknown>) =>
    normalizeQueryResponse({ answer: "x", claim_checks: [raw] })!.claim_checks[0];

  it("says how a supported claim was established", () => {
    expect(claimView(claim(supportedClaim))).toMatchObject({ label: "Supported", method: "Figure match" });
    expect(claimView(claim({ ...supportedClaim, method: "nli" })).meaning).toBe("A retrieved passage states it.");
  });

  it("distinguishes a budget skip from an unresolved check", () => {
    expect(claimView(claim(notCheckedClaim))).toMatchObject({ label: "Not checked", tone: "neutral" });
    expect(claimView(claim({ ...notCheckedClaim, reason: "" }))).toMatchObject({ label: "Unverified", tone: "warn" });
    expect(claimView(claim({ ...notCheckedClaim, reason: "", absence: true })).label).toBe("Unconfirmed absence");
  });

  it("marks a missing figure as worse than an unbacked statement", () => {
    expect(claimView(claim({ ...supportedClaim, status: "unsupported", method: "numeric" })).tone).toBe("bad");
    expect(claimView(claim({ ...supportedClaim, status: "unsupported", method: "nli" })).tone).toBe("warn");
  });

  it("orders problems first", () => {
    const sorted = sortClaimsForReview(
      response({ claim_checks: [supportedClaim, notCheckedClaim, contradictedClaim] }).claim_checks,
    );
    expect(sorted.map((c) => c.status)).toEqual(["contradicted", "unverified", "supported"]);
  });
});

describe("figureView", () => {
  it("never calls a figure found in the wrong row 'found'", () => {
    expect(figureView({ raw: "$387.1M", status: "grounded", binding: "conflict" }).tone).toBe("bad");
    expect(figureView({ raw: "$387.1M", status: "grounded", binding: "unbound" }).label).toBe("found elsewhere");
    expect(figureView({ raw: "$387.1M", status: "grounded", binding: "bound" }).label).toBe("found");
    expect(figureView({ raw: "12%", status: "derived", formula: "a / b" }).title).toBe("Reproduced as a / b");
  });
});
