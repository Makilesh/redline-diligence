import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import ChecklistPanel from "@/components/ChecklistPanel";
import DataRoomPanel from "@/components/DataRoomPanel";
import { CHECKLIST_LIMITS, normalizeChecklist } from "@/lib/normalize";
import type { ChecklistReport } from "@/lib/types";

/*
 * Decision 95: the checklist review grid. One row per contract, one column per
 * checklist item; selecting a cell shows its passage, verbatim, with its page
 * or section. Everything shown is document text, so it is rendered as text and
 * clipped on arrival.
 */

const getChecklist = vi.hoisted(() =>
  vi.fn<(dealId: string) => Promise<ChecklistReport | null>>(async () => null),
);
const getTieOut = vi.hoisted(() => vi.fn(async () => null));
vi.mock("@/lib/api", () => ({ getChecklist, getTieOut }));

const ASSIGNMENT =
  "Neither party may assign this Agreement without the prior written consent of the other party.";

const wire = {
  deal_id: "aurora_vertex_2024",
  items: [
    { id: "change_of_control", label: "Change of control", question: "Does a party have the right to terminate on a change of control?", found: 1, unclear: 0, not_found: 1 },
    { id: "anti_assignment", label: "Anti-assignment", question: "Is consent required to assign the contract?", found: 1, unclear: 1, not_found: 0 },
  ],
  rows: [
    {
      doc_id: "d1",
      source_file: "merger_agreement_v2_final.txt",
      document_category: "legal",
      cells: [
        {
          item_id: "change_of_control",
          status: "found",
          score: 0.91,
          answerability: null,
          passage: { text: "Upon a Change of Control, the Company may terminate.", page_number: null, section_heading: "Section 7.2 Termination", chunk_id: "c1", score: 0.91 },
          alternates: [{ text: "A Change of Control means a merger.", page_number: null, section_heading: "Definitions", chunk_id: "c0", score: 0.4 }],
        },
        {
          item_id: "anti_assignment",
          status: "unclear",
          score: 0.2,
          answerability: null,
          passage: { text: "This Agreement binds successors.", page_number: 12, section_heading: "", chunk_id: "c4", score: 0.2 },
          alternates: [],
        },
      ],
    },
    {
      doc_id: "d2",
      source_file: "credit_agreement_summary.pdf",
      document_category: "financial",
      cells: [
        { item_id: "change_of_control", status: "not_found", score: 0.003, answerability: null, passage: { text: "Interest accrues monthly.", page_number: 3, section_heading: "", chunk_id: "c9", score: 0.003 }, alternates: [] },
        { item_id: "anti_assignment", status: "found", score: 0.95, answerability: 0.9, passage: { text: ASSIGNMENT, page_number: 8, section_heading: "Assignment", chunk_id: "c8", score: 0.95 }, alternates: [] },
      ],
    },
  ],
  documents_total: 9,
  documents_checked: 2,
  documents_not_checked: [],
  documents_excluded: [{ source_file: "aurora_financials_fy2023.txt", document_category: "financial" }],
  excluded_total: 1,
  truncated: false,
  truncation_reason: "",
  reranker: "bge-reranker-v2-m3",
  answerability: false,
  timings_ms: { rerank: 812.4 },
  elapsed_ms: 1312.0,
};

const report = (overrides: Record<string, unknown> = {}): ChecklistReport => {
  const r = normalizeChecklist({ ...wire, ...overrides });
  if (!r) throw new Error("fixture did not normalise");
  return r;
};

describe("normalizeChecklist", () => {
  it("returns null for something that is not a grid", () => {
    expect(normalizeChecklist(null)).toBeNull();
    expect(normalizeChecklist({ detail: "Deal not found." })).toBeNull();
  });

  it("clips document text and bounds every list", () => {
    const long = "L".repeat(5000);
    const cell = { ...wire.rows[0].cells[0], passage: { ...wire.rows[0].cells[0].passage, text: long, section_heading: long } };
    const r = normalizeChecklist({
      ...wire,
      rows: Array.from({ length: 400 }, (_, i) => ({ ...wire.rows[0], source_file: `${long}${i}`, cells: [cell] })),
      documents_not_checked: Array.from({ length: 400 }, () => long),
    })!;
    expect(r.rows).toHaveLength(CHECKLIST_LIMITS.rows);
    expect(r.rows[0].source_file.length).toBeLessThanOrEqual(CHECKLIST_LIMITS.source);
    expect(r.rows[0].cells[0].passage!.text.length).toBeLessThanOrEqual(CHECKLIST_LIMITS.passage);
    expect(r.rows[0].cells[0].passage!.section_heading.length).toBeLessThanOrEqual(CHECKLIST_LIMITS.section);
    expect(r.documents_not_checked).toHaveLength(CHECKLIST_LIMITS.documents);
  });

  it("never upgrades a cell or a partial grid, and fills a missing cell as unclear", () => {
    const r = normalizeChecklist({
      ...wire,
      rows: [{ ...wire.rows[0], cells: [{ ...wire.rows[0].cells[0], status: "probably" }] }],
      truncated: undefined,
    })!;
    expect(r.rows[0].cells.map((c) => c.status)).toEqual(["unclear", "unclear"]);
    expect(r.rows[0].cells[1].item_id).toBe("anti_assignment");
    expect(r.rows[0].cells[1].passage).toBeNull();
    expect(r.truncated).toBe(true);
  });

  it("puts a row's cells in the order of the items", () => {
    const row = { ...wire.rows[0], cells: [...wire.rows[0].cells].reverse() };
    const r = normalizeChecklist({ ...wire, rows: [row] })!;
    expect(r.rows[0].cells.map((c) => c.item_id)).toEqual(["change_of_control", "anti_assignment"]);
  });
});

describe("ChecklistPanel", () => {
  it("shows a row per contract and a status per item", async () => {
    render(<ChecklistPanel dealId="aurora_vertex_2024" load={async () => report()} />);
    const grid = await screen.findByRole("table");
    expect(within(grid).getByRole("rowheader", { name: "merger agreement v2 final" })).toBeInTheDocument();
    expect(
      within(grid).getByRole("button", { name: "Change of control in merger agreement v2 final: found" }),
    ).toBeInTheDocument();
    expect(
      within(grid).getByRole("button", { name: "Anti-assignment in merger agreement v2 final: unclear" }),
    ).toBeInTheDocument();
    expect(
      within(grid).getByRole("button", { name: "Change of control in credit agreement summary: not found" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/2 contracts × 2 items/)).toBeInTheDocument();
    // What the grid has no row for is named, with its category.
    expect(screen.getByText(/Not in the grid: 1 document that is not a contract/)).toHaveTextContent(
      /aurora financials fy2023, financial/,
    );
  });

  it("shows a cell's passage, verbatim, with its page or section", async () => {
    const user = userEvent.setup();
    render(<ChecklistPanel dealId="d" load={async () => report()} />);
    await user.click(
      await screen.findByRole("button", { name: "Anti-assignment in credit agreement summary: found" }),
    );
    const detail = screen.getByRole("region", { name: "Selected cell" });
    expect(within(detail).getByLabelText("Evidence passage")).toHaveTextContent(ASSIGNMENT);
    expect(within(detail).getByText("p. 8 · Assignment")).toBeInTheDocument();
    expect(within(detail).getByText("Found")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Change of control in merger agreement v2 final: found" }));
    const next = screen.getByRole("region", { name: "Selected cell" });
    expect(within(next).getByText("Section 7.2 Termination")).toBeInTheDocument();
    await user.click(within(next).getByText("1 other passage"));
    expect(within(next).getByLabelText("Other passage")).toHaveTextContent("A Change of Control means a merger.");
  });

  it("says what a not-found or unclear cell is", async () => {
    const user = userEvent.setup();
    render(<ChecklistPanel dealId="d" load={async () => report()} />);
    await user.click(
      await screen.findByRole("button", { name: "Change of control in credit agreement summary: not found" }),
    );
    expect(screen.getByText(/Not found — the closest passage is shown/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Anti-assignment in merger agreement v2 final: unclear" }));
    expect(screen.getByText(/Unclear — read the passage/)).toBeInTheDocument();
  });

  it("renders document text as text, never as markup", async () => {
    const user = userEvent.setup();
    const hostile = '<img src=x onerror="alert(1)"><b>Assignment</b>';
    const { container } = render(
      <ChecklistPanel
        dealId="d"
        load={async () =>
          report({
            rows: [
              {
                ...wire.rows[1],
                source_file: "<b>evil</b>.pdf",
                cells: [wire.rows[1].cells[0], { ...wire.rows[1].cells[1], passage: { ...wire.rows[1].cells[1].passage, text: hostile } }],
              },
            ],
          })
        }
      />,
    );
    await user.click(await screen.findByRole("button", { name: /Anti-assignment in .*evil.*: found/ }));
    expect(screen.getByLabelText("Evidence passage")).toHaveTextContent(hostile);
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
  });

  it("says when the grid is partial and what it did not check", async () => {
    render(
      <ChecklistPanel
        dealId="d"
        load={async () =>
          report({ truncated: true, truncation_reason: "time_budget", documents_not_checked: ["SUPPLY_AGREEMENT.pdf"] })
        }
      />,
    );
    const notice = (await screen.findByText(/Partial: checked 2 of 3 contracts/)).closest("div")!;
    expect(notice).toHaveAttribute("role", "status");
    expect(notice).toHaveTextContent(/ran out of time/);
    expect(notice).toHaveTextContent(/SUPPLY AGREEMENT/);
  });

  it("says plainly when the deal has no contract", async () => {
    render(<ChecklistPanel dealId="d" load={async () => report({ rows: [] })} />);
    expect(await screen.findByText(/No document of this deal reads as a contract/)).toBeInTheDocument();
  });

  it("fails open: an unavailable grid is said to be so, and can be retried", async () => {
    const user = userEvent.setup();
    const load = vi.fn<(dealId: string) => Promise<ChecklistReport | null>>(async () => null);
    render(<ChecklistPanel dealId="d" load={load} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not be loaded/);
    load.mockResolvedValueOnce(report());
    await user.click(screen.getByRole("button", { name: /try again/i }));
    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("does not fetch until it is shown", () => {
    const load = vi.fn(async () => report());
    render(<ChecklistPanel dealId="d" active={false} load={load} />);
    expect(load).not.toHaveBeenCalled();
  });
});

describe("the data room panel's checklist tab", () => {
  it("fetches the grid only when the tab is opened", async () => {
    const user = userEvent.setup();
    getChecklist.mockResolvedValue(report());
    render(<DataRoomPanel documents={[]} risks={[]} dealId="aurora_vertex_2024" />);
    expect(getChecklist).not.toHaveBeenCalled();
    await user.click(screen.getByRole("tab", { name: /checklist/i }));
    await waitFor(() => expect(getChecklist).toHaveBeenCalledWith("aurora_vertex_2024"));
    expect(await screen.findByRole("table")).toBeInTheDocument();
  });
});
