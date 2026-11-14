import { describe, expect, it } from "vitest";

import {
  ApiError,
  errorFromResponse,
  formatWait,
  isRetryable,
  parseRetryAfter,
  toApiError,
} from "@/lib/errors";

function json(status: number, body: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

describe("errorFromResponse", () => {
  it("keeps the daily-cap message and states the wait in hours", async () => {
    const err = await errorFromResponse(
      json(429, { detail: "The demo's daily query budget is spent. It resets at 00:00 UTC." }, {
        "Retry-After": "43211",
      }),
      "Query",
    );
    expect(err.kind).toBe("rate_limited");
    expect(err.message).toContain("resets at 00:00 UTC");
    expect(err.message).toContain("about 12 hours");
    expect(err.message).not.toContain("43211");
  });

  it("still explains a 429 when Retry-After is not readable cross-origin", async () => {
    const err = await errorFromResponse(
      json(429, { detail: "Rate limit exceeded for this demo. Please wait and try again." }),
      "Query",
    );
    expect(err.kind).toBe("rate_limited");
    expect(err.message).toBe("Rate limit exceeded for this demo. Please wait and try again.");
  });

  it.each([
    "The engine is busy with other questions. Please retry shortly.",
    "The engine is restarting. Please retry shortly.",
    "Too many temporary workspaces are open right now. Please retry later.",
  ])("reports a 503 with a reason as busy, verbatim: %s", async (detail) => {
    const err = await errorFromResponse(json(503, { detail }), "Query");
    expect(err.kind).toBe("busy");
    expect(err.message).toBe(detail);
  });

  it("reports a cold start (503 not initialised) as unavailable", async () => {
    const err = await errorFromResponse(
      json(503, { detail: "Application not fully initialized" }),
      "Query",
    );
    expect(err.kind).toBe("unavailable");
    expect(err.message).toMatch(/starting up/);
  });

  it("reports a proxy's HTML 502/503 as unavailable", async () => {
    for (const status of [502, 503]) {
      const err = await errorFromResponse(new Response("<html>Bad gateway</html>", { status }), "Query");
      expect(err.kind).toBe("unavailable");
    }
  });

  it("reports the pipeline deadline (504 with detail) as a timeout", async () => {
    const detail = "The query exceeded the time limit and was stopped (request id: abc123).";
    const err = await errorFromResponse(json(504, { detail }, { "X-Request-ID": "abc123" }), "Query");
    expect(err.kind).toBe("timeout");
    expect(err.message).toBe(detail);
    expect(err.requestId).toBe("abc123");
  });

  it("reports a gateway 504 without a body as unavailable", async () => {
    const err = await errorFromResponse(new Response("", { status: 504 }), "Query");
    expect(err.kind).toBe("unavailable");
  });

  it("maps 404 to a deal that is not available, and does not offer a retry", async () => {
    const err = await errorFromResponse(json(404, { detail: "Deal not found." }), "Query");
    expect(err.kind).toBe("not_found");
    expect(err.message).toMatch(/not available/);
    expect(isRetryable(err.kind)).toBe(false);
  });

  it("tells a guard block (text detail) from a validation error (field list) on 422", async () => {
    const blocked = await errorFromResponse(
      json(422, { detail: "This demo answers due-diligence questions about the documents: x" }),
      "Query",
    );
    expect(blocked.kind).toBe("blocked");
    expect(blocked.message).toContain("due-diligence questions");

    const invalid = await errorFromResponse(
      json(422, {
        detail: [{ loc: ["body", "query"], msg: "String should have at most 2000 characters" }],
      }),
      "Query",
    );
    expect(invalid.kind).toBe("invalid");
    expect(invalid.message).toBe("query: String should have at most 2000 characters");
  });

  it("passes the 413 limit through", async () => {
    const err = await errorFromResponse(
      json(413, { detail: "File exceeds the 25 MB upload limit" }),
      "Indexing the document",
    );
    expect(err.kind).toBe("invalid");
    expect(err.message).toBe("File exceeds the 25 MB upload limit");
  });

  it("treats 499 as a lost connection", async () => {
    const err = await errorFromResponse(new Response(null, { status: 499 }), "Query");
    expect(err.kind).toBe("interrupted");
  });

  it("uses the generic 500 detail and keeps the request id", async () => {
    const err = await errorFromResponse(
      json(500, { detail: "Query failed. Reference: r-9" }, { "X-Request-ID": "r-9" }),
      "Query",
    );
    expect(err.kind).toBe("server");
    expect(err.message).toBe("Query failed. Reference: r-9");
    expect(err.requestId).toBe("r-9");
  });
});

describe("helpers", () => {
  it("parses Retry-After seconds and dates", () => {
    expect(parseRetryAfter("30")).toBe(30);
    expect(parseRetryAfter(null)).toBeUndefined();
    expect(parseRetryAfter("soon")).toBeUndefined();
    const now = Date.parse("2026-01-01T00:00:00Z");
    expect(parseRetryAfter("Thu, 01 Jan 2026 00:01:00 GMT", now)).toBe(60);
  });

  it("formats waits in human units", () => {
    expect(formatWait(1)).toBe("1 second");
    expect(formatWait(15)).toBe("15 seconds");
    expect(formatWait(300)).toBe("5 minutes");
    expect(formatWait(7200)).toBe("2 hours");
  });

  it("turns a fetch TypeError into a network error naming the host", () => {
    const err = toApiError(new TypeError("Failed to fetch"), "https://api.example.test");
    expect(err).toBeInstanceOf(ApiError);
    expect(err.kind).toBe("network");
    expect(err.message).toContain("api.example.test");
  });
});
