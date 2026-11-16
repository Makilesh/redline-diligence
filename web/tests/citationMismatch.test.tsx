import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ClaimChecks from "@/components/ClaimChecks";
import { normalizeQueryResponse } from "@/lib/normalize";
import { answerVerdict, citationMarker, needsReview } from "@/lib/verification";
import { response, supportedClaim, wireResponse } from "./fixtures";

/*
 * claim_checks[].citation_mismatch (Decision 78): a figure of the claim is in
 * the documents, but none of the documents the claim cites states it. The
 * server keeps the claim's verdict and turns "passed" into "warning".
 */

const miscited = {
  ...supportedClaim,
  claim: "The $58.00 price is near the top of Vertex's $55–$60 indication.",
  citation_mismatch: [{ figure: "$58.00", found_in: ["merger_agreement_v2_final.txt"] }],
  cited_documents: ["board_deck_strategic_review_mar2024.txt"],
};

describe("normalising citation_mismatch", () => {
  it("keeps well-formed entries and drops the rest", () => {
    const r = normalizeQueryResponse(
      wireResponse({
        claim_checks: [
          { ...miscited, citation_mismatch: [...miscited.citation_mismatch, { found_in: ["x"] }, "junk"] },
          supportedClaim,
        ],
      }),
    )!;
    expect(r.claim_checks[0].citation_mismatch).toEqual(miscited.citation_mismatch);
    expect(r.claim_checks[0].cited_documents).toEqual(miscited.cited_documents);
    expect(r.claim_checks[1].citation_mismatch).toBeUndefined();
    expect(r.claim_checks[1].cited_documents).toBeUndefined();
  });
});

describe("the verdict of a miscited answer", () => {
  it("says what is wrong when every statement checked out", () => {
    const v = answerVerdict(response({ validation_status: "warning", claim_checks: [miscited, supportedClaim] }));
    expect(v.key).toBe("partial");
    expect(v.detail).toBe(
      "All 2 statements matched to the documents, but 1 cites a document that does not state its figures.",
    );
  });

  it("is never Verified, even if a server still said passed", () => {
    expect(answerVerdict(response({ claim_checks: [miscited] })).key).toBe("partial");
  });

  it("leaves a clean answer Verified", () => {
    expect(answerVerdict(response()).key).toBe("verified");
  });
});

describe("ClaimChecks", () => {
  it("lists a miscited supported claim up front, marked, with its verdict kept", () => {
    const r = response({ validation_status: "warning", claim_checks: [supportedClaim, miscited] });
    render(<ClaimChecks checks={r.claim_checks} />);
    const marker = screen.getByText("Cited source lacks figure");
    expect(marker.closest("[title]")?.getAttribute("title")).toBe(
      "$58.00 is stated in merger_agreement_v2_final.txt, not in the cited board_deck_strategic_review_mar2024.txt.",
    );
    const row = marker.closest("li")!;
    expect(row.getAttribute("data-status")).toBe("supported");
    // Visible without unfolding; the clean claim stays folded.
    expect(screen.getByRole("button", { name: /Show 1 supported statement/ })).toBeInTheDocument();
  });

  it("has no marker for a claim whose citation holds its figures", () => {
    const r = response();
    expect(citationMarker(r.claim_checks[0])).toBeNull();
    expect(needsReview(r.claim_checks[0])).toBe(false);
  });
});
