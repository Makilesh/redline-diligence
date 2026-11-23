import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import Home from "@/app/page";
import type { StreamEvent, StreamQueryArgs } from "@/lib/api";
import { ApiError } from "@/lib/errors";
import { contradictedClaim, response, supportedClaim } from "./fixtures";

/*
 * The page against a scripted engine. `engine.script` decides what one
 * query's stream does: emit some events, then resolve or throw.
 */
const engine = vi.hoisted(() => ({
  script: (async () => {}) as (emit: (e: StreamEvent) => void) => Promise<void>,
}));

vi.mock("@/lib/api", () => ({
  API_BASE: "http://engine.test",
  MAX_QUERY_LENGTH: 2000,
  listDeals: vi.fn(async () => [
    {
      deal_id: "demo",
      deal_name: "Demo deal",
      description: "",
      document_count: 3,
      status: "active",
      is_sandbox: false,
      expires_at: "",
    },
  ]),
  listDocuments: vi.fn(async () => []),
  listRiskSignals: vi.fn(async () => []),
  getTieOut: vi.fn(async () => null),
  getBudget: vi.fn(async () => null),
  createDeal: vi.fn(),
  ingestDocument: vi.fn(),
  purgeDeal: vi.fn(),
  streamQuery: vi.fn((args: StreamQueryArgs) => engine.script(args.onEvent)),
}));

const planned = {
  event: "start" as const,
  data: {
    session_id: "s",
    planned_stages: [
      { agent: "query_intelligence", label: "Query Intelligence", description: "" },
      { agent: "answer_synthesizer", label: "Answer Synthesizer", description: "" },
    ],
  },
};

async function ask(question = "What was FY2023 revenue?") {
  const user = userEvent.setup();
  render(<Home />);
  const box = await screen.findByRole("textbox", { name: /ask a question/i });
  await waitFor(() => expect(box).not.toBeDisabled());
  await user.type(box, question);
  await user.click(screen.getByRole("button", { name: /run the query/i }));
}

beforeEach(() => {
  engine.script =async () => {};
  // jsdom has no layout; the page scrolls citations into view.
  Element.prototype.scrollIntoView = vi.fn();
});

describe("Home", () => {
  it("drops the unverified draft when the stream ends without a result (F1)", async () => {
    let fail!: (e: unknown) => void;
    engine.script =(emit) =>
      new Promise<void>((_, reject) => {
        emit(planned);
        emit({ event: "token", data: { text: "Revenue was $999 million." } });
        fail = reject;
      });

    await ask();
    // While running, the draft is shown, and labelled as one.
    expect(await screen.findByText(/Revenue was \$999 million/)).toBeInTheDocument();
    expect(screen.getByText(/Drafting/)).toBeInTheDocument();

    await act(async () => {
      fail(
        new ApiError({
          kind: "interrupted",
          message: "The connection to the engine closed before a verified answer arrived.",
        }),
      );
    });

    expect(await screen.findByRole("alert")).toHaveTextContent(/Connection lost/);
    expect(screen.queryByText(/Revenue was \$999 million/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /ask again/i })).toBeInTheDocument();
    // The stage in flight is not shown as completed.
    expect(screen.getByText(/Interrupted — the run stopped during this step/)).toBeInTheDocument();
  });

  it("shows the request id from an in-band error", async () => {
    engine.script =async (emit) => {
      emit(planned);
      throw new ApiError({ kind: "timeout", message: "The query exceeded the time limit.", requestId: "r-42" });
    };
    await ask();
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Timed out");
    expect(alert).toHaveTextContent("r-42");
  });

  it("presents a synthesis failure as an outage, not as a refusal", async () => {
    engine.script =async (emit) => {
      emit(planned);
      emit({
        event: "result",
        data: response({
          outcome: "synthesis_failed",
          is_refusal: true,
          answer: "I could not generate an answer.",
          claim_checks: [],
          confidence_score: 0,
        }),
      });
    };
    await ask();
    expect(
      await screen.findByRole("heading", { name: /the language models did not respond/i }),
    ).toBeInTheDocument();
    // Also announced to assistive technology, once, politely.
    expect(document.querySelector("[aria-live=polite]")).toHaveTextContent(
      "No answer: the language models did not respond.",
    );
    expect(screen.queryByText(/Declined/)).not.toBeInTheDocument();
    expect(screen.queryByText(/I could not generate an answer/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Verified/)).not.toBeInTheDocument();
  });

  it("shows a considered refusal with the refusal surface", async () => {
    engine.script =async (emit) => {
      emit(planned);
      emit({ event: "result", data: response({ outcome: "refused", is_refusal: true, answer: "", claim_checks: [] }) });
    };
    await ask();
    expect(await screen.findByText(/Declined — the data room does not support an answer/)).toBeInTheDocument();
  });

  it("shows a not-found under its own banner, with the model's explanation and no confidence", async () => {
    engine.script =async (emit) => {
      emit(planned);
      emit({
        event: "result",
        data: response({
          outcome: "not_found",
          confidence_score: 0.93,
          answer: "The provided documents do not contain the revenue share of the top ten customers.",
          claim_checks: [],
        }),
      });
    };
    await ask();
    expect(await screen.findByText("Not found in the passages searched")).toBeInTheDocument();
    expect(screen.getByText(/do not contain the revenue share of the top ten customers/)).toBeInTheDocument();
    expect(screen.getByText("Not found: the documents searched do not contain the answer.")).toBeInTheDocument();
    expect(screen.queryByText("Verified")).not.toBeInTheDocument();
    expect(screen.queryByText("93%")).not.toBeInTheDocument();
    expect(screen.queryByText(/Declined/)).not.toBeInTheDocument();
  });

  it("marks a failed answer and shows which statement failed, with its source", async () => {
    engine.script =async (emit) => {
      emit(planned);
      emit({
        event: "result",
        data: response({
          outcome: "answered_unverified",
          validation_status: "failed",
          answer: "Revenue was $452.8 million. The termination fee is $20 million.",
          claim_checks: [supportedClaim, contradictedClaim],
          hallucination_flags: ["contradicted by Merger Agreement.pdf: The termination fee is $20 million."],
        }),
      });
    };
    await ask();
    expect(await screen.findByText("Failed verification")).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent(/1 statement could not be matched/);

    const checks = screen.getByRole("region", { name: /claim checks/i });
    expect(checks).toHaveTextContent("Contradicted");
    expect(checks).toHaveTextContent("The termination fee is $20 million.");
    expect(checks).toHaveTextContent("Merger Agreement.pdf, p. 41 · Section 8.3 Termination Fee");
    // Supported statements are folded behind a count, not hidden.
    expect(checks).toHaveTextContent(/Show 1 supported statement/);
  });

  it("shows Verified only for a fully supported answer", async () => {
    engine.script =async (emit) => {
      emit(planned);
      emit({ event: "result", data: response() });
    };
    await ask();
    expect(await screen.findByText("Verified")).toBeInTheDocument();
    expect(document.querySelector("[aria-live=polite]")).toHaveTextContent("Answer ready. Verified.");
  });
});
