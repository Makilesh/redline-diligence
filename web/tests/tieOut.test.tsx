import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import DataRoomPanel from "@/components/DataRoomPanel";
import TieOutPanel from "@/components/TieOutPanel";
import { TIE_OUT_LIMITS, normalizeTieOut } from "@/lib/normalize";
import type { TieOutReport } from "@/lib/types";

/*
 * Decision 92: the deal-level figure tie-out. Conflicts come first and each
 * value names its document and page; everything shown is document text, so it
 * is rendered as text and clipped on arrival.
 */

const getTieOut = vi.hoisted(() =>
  vi.fn<(dealId: string) => Promise<TieOutReport | null>>(async () => null),
);
vi.mock("@/lib/api", () => ({ getTieOut }));

const wire = {
  deal_id: "aurora_vertex_2024",
  conflicts: [
    {
      metric: "Adjusted EBITDA",
      period: "FY2023",
      kind: "amount",
      status: "conflict",
      max_deviation_pct: 1.72,
      values: [
        { source_file: "aurora_financials_fy2023.txt", page_number: null, section_heading: "KEY FINANCIAL METRICS", as_printed: "$97.3", value: 97.3e6, scale_known: true },
        { source_file: "quality_of_earnings_report_fy2023.pdf", page_number: 4, section_heading: "", as_printed: "$99.0", value: 99.0e6, scale_known: true },
      ],
      more_documents: 0,
    },
  ],
  agreements: [
    {
      metric: "Net Income",
      period: "FY2023",
      kind: "amount",
      status: "agree",
      max_deviation_pct: 0,
      values: [
        { source_file: "a.pdf", page_number: 2, section_heading: "", as_printed: "$46.4", value: 46.4e6, scale_known: true },
        { source_file: "b.pdf", page_number: 9, section_heading: "", as_printed: "46.4", value: 46.4, scale_known: false },
      ],
      more_documents: 0,
    },
  ],
  conflicts_total: 1,
  agreements_total: 1,
  single_source_figures: 412,
  ambiguous_figures: 3,
  generic_figures: 7,
  figures_without_period: 90,
  documents_total: 9,
  documents_read: 9,
  documents_not_read: [],
  chunks_read: 111,
  truncated: false,
  truncation_reason: "",
  elapsed_ms: 41.2,
};

const report = (overrides: Record<string, unknown> = {}): TieOutReport => {
  const r = normalizeTieOut({ ...wire, ...overrides });
  if (!r) throw new Error("fixture did not normalise");
  return r;
};

describe("normalizeTieOut", () => {
  it("returns null for something that is not a report", () => {
    expect(normalizeTieOut(null)).toBeNull();
    expect(normalizeTieOut({ detail: "Deal not found." })).toBeNull();
  });

  it("clips document text and bounds every list", () => {
    const long = "L".repeat(5000);
    const many = Array.from({ length: 500 }, (_, i) => ({ ...wire.conflicts[0], metric: `${long}${i}` }));
    const r = normalizeTieOut({
      ...wire,
      conflicts: many,
      documents_not_read: Array.from({ length: 400 }, () => long),
    })!;
    expect(r.conflicts).toHaveLength(TIE_OUT_LIMITS.figures);
    expect(r.conflicts[0].metric.length).toBeLessThanOrEqual(TIE_OUT_LIMITS.label);
    expect(r.documents_not_read).toHaveLength(TIE_OUT_LIMITS.documents);
    expect(r.documents_not_read[0].length).toBeLessThanOrEqual(TIE_OUT_LIMITS.source);
  });

  it("never upgrades a figure to an agreement or a partial report to a whole one", () => {
    const r = normalizeTieOut({
      ...wire,
      conflicts: [{ ...wire.conflicts[0], status: "agree" }],
      agreements: [{ ...wire.agreements[0], status: "maybe" }],
      truncated: undefined,
    })!;
    expect(r.conflicts[0].status).toBe("conflict");
    expect(r.agreements).toEqual([]);
    expect(r.truncated).toBe(true);
  });
});

describe("TieOutPanel", () => {
  it("lists conflicts first, each value with its document and page", async () => {
    render(<TieOutPanel dealId="aurora_vertex_2024" load={async () => report()} />);
    const conflicts = await screen.findByRole("list", { name: "Conflicting figures" });
    const item = within(conflicts).getByText("Adjusted EBITDA").closest("li")!;
    expect(within(item).getByText("1.7% apart")).toBeInTheDocument();
    expect(within(item).getByText("$97.3")).toBeInTheDocument();
    expect(within(item).getByText("$99.0")).toBeInTheDocument();
    expect(within(item).getByText(/quality of earnings report fy2023/)).toBeInTheDocument();
    expect(within(item).getByText(/p\. 4/)).toBeInTheDocument();
    // A format without pages is located by its section instead.
    expect(within(item).getByText(/KEY FINANCIAL METRICS/)).toBeInTheDocument();
    // Agreements wait behind a toggle; the counts of what was not compared are stated.
    expect(screen.queryByRole("list", { name: "Agreeing figures" })).not.toBeInTheDocument();
    expect(screen.getByText(/412 figures only one document states/)).toBeInTheDocument();
    expect(screen.getByText(/7 figures under labels like "Total" or "Other"/)).toBeInTheDocument();
  });

  it("shows agreements on request", async () => {
    const user = userEvent.setup();
    render(<TieOutPanel dealId="d" load={async () => report()} />);
    await user.click(await screen.findByRole("button", { name: /show 1 agreement/i }));
    const agreements = screen.getByRole("list", { name: "Agreeing figures" });
    expect(within(agreements).getByText("Net Income")).toBeInTheDocument();
    expect(within(agreements).getByText("46.4")).toBeInTheDocument();
  });

  it("renders document text as text, never as markup", async () => {
    const hostile = '<img src=x onerror="alert(1)">Revenue';
    const { container } = render(
      <TieOutPanel
        dealId="d"
        load={async () =>
          report({ conflicts: [{ ...wire.conflicts[0], metric: hostile, values: [{ ...wire.conflicts[0].values[0], as_printed: "<b>9</b>" }] }] })
        }
      />,
    );
    expect(await screen.findByText(hostile)).toBeInTheDocument();
    expect(screen.getByText("<b>9</b>")).toBeInTheDocument();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
  });

  it("says when the report is partial and what it did not read", async () => {
    render(
      <TieOutPanel
        dealId="d"
        load={async () =>
          report({ truncated: true, truncation_reason: "chunk_cap", documents_read: 2, documents_not_read: ["JPMORGAN_2022_10K.pdf"] })
        }
      />,
    );
    const notice = (await screen.findByText(/Partial: read 2 of 9 documents/)).closest("div")!;
    expect(notice).toHaveAttribute("role", "status");
    expect(notice).toHaveTextContent(/larger than one report reads/);
    expect(notice).toHaveTextContent(/JPMORGAN 2022 10K/);
  });

  it("says plainly when nothing is stated twice", async () => {
    render(<TieOutPanel dealId="d" load={async () => report({ conflicts: [], agreements: [], conflicts_total: 0, agreements_total: 0 })} />);
    expect(await screen.findByText(/No figure is stated by two documents/)).toBeInTheDocument();
  });

  it("fails open: an unavailable report is said to be so, and can be retried", async () => {
    const user = userEvent.setup();
    const load = vi.fn<(dealId: string) => Promise<TieOutReport | null>>(async () => null);
    render(<TieOutPanel dealId="d" load={load} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not be loaded/);
    load.mockResolvedValueOnce(report());
    await user.click(screen.getByRole("button", { name: /try again/i }));
    expect(await screen.findByText("Adjusted EBITDA")).toBeInTheDocument();
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("does not fetch until it is shown", () => {
    const load = vi.fn(async () => report());
    render(<TieOutPanel dealId="d" active={false} load={load} />);
    expect(load).not.toHaveBeenCalled();
  });
});

describe("the data room panel's tie-out tab", () => {
  it("fetches the report only when the tab is opened", async () => {
    const user = userEvent.setup();
    getTieOut.mockResolvedValue(report());
    render(<DataRoomPanel documents={[]} risks={[]} dealId="aurora_vertex_2024" />);
    expect(getTieOut).not.toHaveBeenCalled();
    await user.click(screen.getByRole("tab", { name: /tie-out/i }));
    await waitFor(() => expect(getTieOut).toHaveBeenCalledWith("aurora_vertex_2024"));
    expect(await screen.findByText("Adjusted EBITDA")).toBeInTheDocument();
  });
});
