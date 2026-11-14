import { afterEach, describe, expect, it, vi } from "vitest";

import { streamQuery, type StreamEvent } from "@/lib/api";
import { ApiError } from "@/lib/errors";
import { wireResponse } from "./fixtures";

const encoder = new TextEncoder();

/**
 * A fetch stub whose body delivers `chunks` (strings or raw bytes) and then
 * either closes, stays open forever (`hang`), or errors (`drop`). Like a real
 * fetch body, it errors when the request's signal aborts.
 */
function stubFetch(
  chunks: (string | Uint8Array)[],
  end: "close" | "hang" | "drop" = "close",
  init: ResponseInit = { status: 200 },
) {
  const fetchMock = vi.fn(async (_url: string, opts: RequestInit) => {
    const signal = opts.signal!;
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        signal.addEventListener("abort", () =>
          controller.error(new DOMException("aborted", "AbortError")),
        );
        for (const c of chunks) controller.enqueue(typeof c === "string" ? encoder.encode(c) : c);
        if (end === "close") controller.close();
        if (end === "drop") controller.error(new TypeError("network error"));
      },
    });
    return new Response(body, { headers: { "Content-Type": "text/event-stream" }, ...init });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const frame = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

async function run(idleTimeoutMs = 2_000, signal?: AbortSignal) {
  const events: StreamEvent[] = [];
  const outcome = await streamQuery({
    query: "q",
    dealId: "deal",
    includePii: false,
    onEvent: (e) => events.push(e),
    idleTimeoutMs,
    signal,
  }).then(
    () => null,
    (e: unknown) => e,
  );
  return { events, error: outcome };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("streamQuery", () => {
  it("delivers progress and resolves on the result, ignoring keep-alives", async () => {
    stubFetch([
      frame("start", { session_id: "s", planned_stages: [{ agent: "query_intelligence", label: "QI" }] }),
      ": keepalive\n\n",
      frame("stage", { seq: 1, agent: "query_intelligence", summary: "ok", perf: { llm_calls: 1 } }),
      frame("token", { text: "Revenue " }),
      frame("token", { text: "was $452.8M" }),
      frame("result", wireResponse()),
    ]);
    const { events, error } = await run();
    expect(error).toBeNull();
    expect(events.map((e) => e.event)).toEqual(["start", "stage", "token", "token", "result"]);
    const result = events.at(-1)!;
    expect(result.event === "result" && result.data.outcome).toBe("answered");
  });

  it("decodes a multi-byte character split across network chunks", async () => {
    const bytes = encoder.encode(frame("result", wireResponse({ answer: "Fee: €13.92 million" })));
    const euro = bytes.indexOf(0xe2);
    stubFetch([bytes.slice(0, euro + 1), bytes.slice(euro + 1)]);
    const { events } = await run();
    const result = events.at(-1)!;
    expect(result.event === "result" && result.data.answer).toBe("Fee: €13.92 million");
  });

  it("rejects when the stream closes after the draft but before a result (F1)", async () => {
    stubFetch([frame("start", { planned_stages: [] }), frame("token", { text: "Revenue was $9" })]);
    const { events, error } = await run();
    expect(events.map((e) => e.event)).toEqual(["start", "token"]);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).kind).toBe("interrupted");
    expect((error as ApiError).message).toMatch(/unverified draft/);
  });

  it("rejects when the connection drops mid-stream", async () => {
    stubFetch([frame("token", { text: "x" })], "drop");
    const { error } = await run();
    expect((error as ApiError).kind).toBe("interrupted");
  });

  it("rejects a half-received result frame rather than rendering it", async () => {
    stubFetch([`event: result\ndata: {"answer": "Rev`]);
    const { events, error } = await run();
    expect(events).toEqual([]);
    expect((error as ApiError).kind).toBe("interrupted");
  });

  it("surfaces the error frame's code, detail and request id", async () => {
    stubFetch([
      frame("start", { planned_stages: [] }),
      frame("error", {
        detail: "The query exceeded the time limit and was stopped (request id: r-42).",
        code: "timeout",
        session_id: "s",
        request_id: "r-42",
      }),
    ]);
    const { error } = await run();
    expect(error).toMatchObject({ kind: "timeout", requestId: "r-42" });
    expect((error as ApiError).message).toContain("time limit");
  });

  it("reports a pipeline_error frame as a pipeline failure", async () => {
    stubFetch([frame("error", { detail: "Query failed. Reference: r-1", code: "pipeline_error", request_id: "r-1" })]);
    const { error } = await run();
    expect(error).toMatchObject({ kind: "pipeline", requestId: "r-1" });
  });

  it("gives up on a stream that goes silent", async () => {
    stubFetch([frame("start", { planned_stages: [] })], "hang");
    const { error } = await run(50);
    expect((error as ApiError).kind).toBe("timeout");
  });

  it("propagates the caller's abort as an AbortError, not an ApiError", async () => {
    stubFetch([frame("start", { planned_stages: [] })], "hang");
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 20);
    const { error } = await run(5_000, controller.signal);
    expect(error).toBeInstanceOf(DOMException);
    expect((error as DOMException).name).toBe("AbortError");
  });

  it("maps an HTTP error before the stream starts", async () => {
    stubFetch([JSON.stringify({ detail: "Rate limit exceeded for this demo. Please wait and try again." })], "close", {
      status: 429,
      headers: { "Content-Type": "application/json" },
    });
    const { error } = await run();
    expect((error as ApiError).kind).toBe("rate_limited");
  });

  it("lets an exception in the caller's handler surface", async () => {
    stubFetch([frame("start", { planned_stages: [] })]);
    const result = streamQuery({
      query: "q",
      dealId: "d",
      includePii: false,
      onEvent: () => {
        throw new Error("handler bug");
      },
    });
    await expect(result).rejects.toThrow("handler bug");
  });
});
