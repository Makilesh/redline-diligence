import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import AnswerPanel from "@/components/AnswerPanel";
import ClaimChecks from "@/components/ClaimChecks";
import UploadPanel from "@/components/UploadPanel";
import { normalizeIngestResult, normalizeQueryResponse } from "@/lib/normalize";
import {
  answerVerdict,
  injectionMarker,
  injectionNote,
  needsReview,
  signalLabels,
  uploadScreenNote,
} from "@/lib/verification";
import { response, supportedClaim, wireResponse } from "./fixtures";

/*
 * Decision 82: what the ingest screen and the validator's injection review
 * found reaches the person who should act on it — the uploader at upload
 * time, the reader beside the answer. Locations only: the API sends no
 * flagged text, so there is none to render.
 */

const MEMO = "northwind_information_memorandum.txt";

const taintedClaim = {
  ...supportedClaim,
  claim: "Total revenue was $900.0 million.",
  numbers: [{ raw: "$900.0 million", status: "grounded", binding: "bound" }],
  injection_suspect: true,
};

const review = {
  flagged_passages: [{ source_file: MEMO, page_number: null, section_heading: "Financial Overview", signals: ["instruction_override"] }],
  cited_flagged_passages: 1,
  tainted_figures: ["$900.0 million"],
};

const flaggedUpload = {
  doc_id: "d1",
  deal_id: "sbx",
  document_category: "financial",
  chunks_created: 12,
  status: "success",
  notices: [],
  version_candidates: [],
  injection_screen: {
    flagged_passages: 2,
    signals: ["model_address", "role_marker"],
    locations: [
      { page_number: 3, section_heading: "Fees", signals: ["model_address"] },
      { page_number: 7, section_heading: "", signals: ["role_marker"] },
    ],
  },
};

describe("normalising the new fields", () => {
  it("defaults both summaries when an older server does not send them", () => {
    const r = normalizeQueryResponse(wireResponse())!;
    expect(r.injection_review).toEqual({ flagged_passages: [], cited_flagged_passages: 0, tainted_figures: [] });
    expect(r.claim_checks[0].injection_suspect).toBeUndefined();
    const u = normalizeIngestResult({ doc_id: "d", deal_id: "x", chunks_created: 3 })!;
    expect(u.injection_screen).toEqual({ flagged_passages: 0, signals: [], locations: [] });
  });

  it("coerces shape without inventing findings", () => {
    const r = normalizeQueryResponse(
      wireResponse({
        injection_review: {
          flagged_passages: [{ source_file: MEMO, page_number: 0, signals: ["model_address", 4] }, "junk"],
          cited_flagged_passages: -2,
          tainted_figures: ["$1", "", null],
        },
        claim_checks: [{ ...taintedClaim, injection_suspect: "yes" }],
      }),
    )!;
    expect(r.injection_review).toEqual({
      flagged_passages: [{ source_file: MEMO, page_number: null, section_heading: "", signals: ["model_address"] }],
      cited_flagged_passages: 0,
      tainted_figures: ["$1"],
    });
    // Only an explicit true marks a claim.
    expect(r.claim_checks[0].injection_suspect).toBeUndefined();
  });
});

describe("the answer", () => {
  it("names the reason when every statement checked out", () => {
    const r = response({ validation_status: "warning", claim_checks: [taintedClaim], injection_review: review });
    const v = answerVerdict(r);
    expect(v.key).toBe("partial");
    expect(v.detail).toBe(
      "All 1 statement matched to the documents, but $900.0 million rests only on text that reads like instructions to an AI model.",
    );
  });

  it("says when the answer only cites a flagged passage", () => {
    const r = response({
      validation_status: "warning",
      injection_review: { ...review, tainted_figures: [] },
    });
    expect(answerVerdict(r).detail).toBe(
      "All 1 statement matched to the documents, but it cites a passage that reads like instructions to an AI model.",
    );
  });

  it("is never Verified when the review found reliance, even if a server said passed", () => {
    expect(answerVerdict(response({ injection_review: review })).key).toBe("partial");
  });

  it("stays Verified when a flagged passage was retrieved but not used", () => {
    const quiet = { ...review, cited_flagged_passages: 0, tainted_figures: [] };
    const r = response({ injection_review: quiet });
    expect(answerVerdict(r).key).toBe("verified");
    expect(injectionNote(r.injection_review)).toEqual({
      text: `Retrieved but not cited: ${MEMO} · Financial Overview contains text that reads like instructions to an AI model (an instruction to ignore earlier instructions).`,
      tone: "neutral",
    });
  });

  it("shows where the instruction-like text is, above the answer", () => {
    render(<AnswerPanel result={response({ validation_status: "warning", claim_checks: [taintedClaim], injection_review: review })} />);
    const note = screen.getByText(/This answer relies on text that reads like instructions to an AI model/);
    expect(note.textContent).toBe(
      `This answer relies on text that reads like instructions to an AI model (an instruction to ignore earlier instructions) in ${MEMO} · Financial Overview — $900.0 million is stated nowhere else. Check those statements against the document itself.`,
    );
    expect(note.closest("[role=note]")?.className).toContain("text-warn");
  });

  it("shows nothing extra for an ordinary answer", () => {
    render(<AnswerPanel result={response()} />);
    expect(screen.queryByText(/instructions to an AI model/)).toBeNull();
  });
});

describe("the claim", () => {
  it("is marked and listed before the fold, with its verdict kept", () => {
    const r = response({ validation_status: "warning", claim_checks: [supportedClaim, taintedClaim], injection_review: review });
    render(<ClaimChecks checks={r.claim_checks} />);
    const marker = screen.getByText("Figure from instruction-like text");
    expect(marker.closest("li")!.getAttribute("data-status")).toBe("supported");
    expect(screen.getByRole("button", { name: /Show 1 supported statement/ })).toBeInTheDocument();
  });

  it("has no marker otherwise", () => {
    const c = response().claim_checks[0];
    expect(injectionMarker(c)).toBeNull();
    expect(needsReview(c)).toBe(false);
  });
});

describe("the upload", () => {
  it("tells the uploader what was found and where, as a label", () => {
    const result = normalizeIngestResult(flaggedUpload)!;
    expect(uploadScreenNote(result.injection_screen)).toEqual({
      finding:
        "2 passages read like instructions to an AI model (text addressed to an AI model, chat-role labels such as SYSTEM:): p. 3 · Fees; p. 7.",
      consequence:
        "The file was indexed as written. An answer that cites these passages, or takes a figure only from them, is never shown as Verified.",
    });
    render(
      <UploadPanel
        sandboxDeal={null}
        uploading={false}
        error={null}
        lastUpload={{ filename: "seller_memo.pdf", result }}
        onUpload={() => {}}
        onDiscard={() => {}}
      />,
    );
    expect(screen.getByText(/2 passages read like instructions to an AI model/)).toBeInTheDocument();
    // Indexed, not failed: the status line is unchanged.
    expect(screen.getByText(/Indexed:/)).toBeInTheDocument();
  });

  it("says nothing when nothing was flagged", () => {
    const result = normalizeIngestResult({ ...flaggedUpload, injection_screen: undefined })!;
    expect(uploadScreenNote(result.injection_screen)).toBeNull();
    render(
      <UploadPanel
        sandboxDeal={null}
        uploading={false}
        error={null}
        lastUpload={{ filename: "clean.pdf", result }}
        onUpload={() => {}}
        onDiscard={() => {}}
      />,
    );
    expect(screen.queryByText(/instructions to an AI model/)).toBeNull();
  });

  it("describes known signals only", () => {
    expect(signalLabels(["invisible_text", "something_new"])).toBe("invisible characters");
  });
});
