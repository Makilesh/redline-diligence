import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import AnswerPanel from "@/components/AnswerPanel";
import { normalizeQueryResponse } from "@/lib/normalize";
import { answerVerdict, evidenceCutNote } from "@/lib/verification";
import { response, wireResponse } from "./fixtures";

/* An answer written from part of the retrieved evidence says so (Decision 118). */

const CUT = { passages_used: 6, passages_available: 14, chars_used: 30_000, chars_available: 64_000 };

describe("evidence_cut", () => {
  it("is kept by the normaliser only when something was left out", () => {
    expect(response({ evidence_cut: CUT }).evidence_cut).toEqual(CUT);
    expect(response().evidence_cut).toBeNull(); // a server that predates the field
    expect(response({ evidence_cut: { ...CUT, passages_used: 14, chars_used: 64_000 } }).evidence_cut).toBeNull();
    expect(response({ evidence_cut: "6 of 14" }).evidence_cut).toBeNull();
    // Fewer characters of the same passages is still less evidence.
    expect(response({ evidence_cut: { ...CUT, passages_used: 14 } }).evidence_cut).not.toBeNull();
  });

  it("is said under the answer, with how much and what it means", () => {
    render(<AnswerPanel result={response({ evidence_cut: CUT })} />);
    expect(screen.getByRole("note", { name: /evidence left out/i })).toHaveTextContent(
      "Written from 6 of the 14 passages retrieved (about 47% of their text): the local model's window could not hold them all. A detail found only in the others is not in this answer.",
    );
  });

  it("is not said for an answer from the full context", () => {
    render(<AnswerPanel result={response()} />);
    expect(screen.queryByRole("note", { name: /evidence left out/i })).toBeNull();
    expect(evidenceCutNote(null)).toBeNull();
  });

  it("weakens a not-found further", () => {
    const v = answerVerdict(response({ outcome: "not_found", evidence_cut: CUT }));
    expect(v.key).toBe("not_found");
    expect(v.detail).toMatch(/judged on only 6 of the 14 passages retrieved, so the rest may hold it\.$/);
    expect(answerVerdict(response({ outcome: "not_found" })).detail).not.toMatch(/judged on only/);
  });

  it("survives the wire", () => {
    expect(normalizeQueryResponse(wireResponse({ evidence_cut: CUT }))!.evidence_cut).toEqual(CUT);
  });
});
