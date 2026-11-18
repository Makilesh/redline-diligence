import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import DataRoomPanel from "@/components/DataRoomPanel";
import type { DocumentRecord } from "@/lib/types";

/*
 * Decision 82: the document list keeps the ingest screen's label, so a file
 * flagged at upload is still marked when someone opens the deal later.
 */

const base: DocumentRecord = {
  doc_id: "doc-1",
  filename: "seller_memo.pdf",
  document_category: "legal",
  chunks_created: 9,
  version_label: "v1",
  upload_date: "",
  is_current_version: true,
  supersedes_doc_id: "",
  superseded_by: "",
  has_redline: false,
};

describe("document list", () => {
  it("labels a flagged document with what the screen found", () => {
    render(
      <DataRoomPanel
        documents={[{ ...base, instruction_like_passages: 1, instruction_signals: ["role_marker"] }]}
        risks={[]}
      />,
    );
    expect(
      screen.getByText(/1 passage reads like instructions to an AI model \(chat-role labels such as SYSTEM:\)/),
    ).toBeInTheDocument();
  });

  it("shows nothing for ordinary documents or older servers", () => {
    render(<DataRoomPanel documents={[base, { ...base, doc_id: "doc-2", instruction_like_passages: 0 }]} risks={[]} />);
    expect(screen.queryByText(/like instructions to an AI model/)).not.toBeInTheDocument();
  });
});
