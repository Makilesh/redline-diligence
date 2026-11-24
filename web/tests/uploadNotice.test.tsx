import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import UploadPanel, { DATA_HANDLING_UNKNOWN } from "@/components/UploadPanel";
import { getDataHandling } from "@/lib/api";
import type { DataHandling } from "@/lib/types";

/* Where uploaded text goes, said before the upload (Decision 116). */

function panel(dataHandling: DataHandling | null | undefined) {
  return render(
    <UploadPanel
      sandboxDeal={null}
      uploading={false}
      error={null}
      dataHandling={dataHandling}
      onUpload={() => {}}
      onDiscard={() => {}}
    />,
  );
}

describe("the upload panel's data notice", () => {
  it("shows the server's wording when a hosted model is configured", () => {
    panel({ hosted_llm: "Google Gemini API", upload_notice: "Passages go to Google's Gemini API." });
    expect(screen.getByRole("note", { name: /where uploaded text goes/i })).toHaveTextContent(
      "Passages go to Google's Gemini API.",
    );
  });

  it("assumes the text leaves when the server could not be asked", () => {
    panel(null);
    expect(screen.getByRole("note", { name: /where uploaded text goes/i })).toHaveTextContent(
      DATA_HANDLING_UNKNOWN,
    );
  });

  it("says nothing while loading, or when text stays on the server", () => {
    panel(undefined);
    expect(screen.queryByRole("note", { name: /where uploaded text goes/i })).toBeNull();
    panel({ hosted_llm: null, upload_notice: null });
    expect(screen.queryByRole("note", { name: /where uploaded text goes/i })).toBeNull();
  });
});

describe("getDataHandling", () => {
  afterEach(() => vi.unstubAllGlobals());

  const serve = (body: unknown, ok = true) =>
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok, json: async () => body })));

  it("keeps the two fields and drops anything that is not text", async () => {
    serve({ hosted_llm: "Google Gemini API", upload_notice: "  ", extra: 1 });
    expect(await getDataHandling()).toEqual({ hosted_llm: "Google Gemini API", upload_notice: null });
  });

  it("is null when the server cannot say", async () => {
    serve({}, false);
    expect(await getDataHandling()).toBeNull();
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("offline"))));
    expect(await getDataHandling()).toBeNull();
  });
});
