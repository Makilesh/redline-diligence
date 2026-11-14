import { describe, expect, it } from "vitest";

import { SseParser } from "@/lib/sse";

/** Feeds the chunks in order and returns every message produced. */
function parse(...chunks: string[]) {
  const parser = new SseParser();
  return chunks.flatMap((c) => parser.feed(c));
}

describe("SseParser", () => {
  it("parses LF-framed events with names and JSON data", () => {
    expect(parse('event: stage\ndata: {"a":1}\n\nevent: result\ndata: {}\n\n')).toEqual([
      { event: "stage", data: '{"a":1}' },
      { event: "result", data: "{}" },
    ]);
  });

  it("accepts CRLF and bare-CR line endings", () => {
    // The trailing comment line resolves the final CR: a CR that is the last
    // byte received is held until the next byte shows whether an LF follows.
    expect(parse("event: a\r\ndata: 1\r\n\r\nevent: b\rdata: 2\r\r: next\n")).toEqual([
      { event: "a", data: "1" },
      { event: "b", data: "2" },
    ]);
  });

  it("does not dispatch early when a CRLF is split across chunks", () => {
    // Treating the trailing CR as a line end would turn the LF that follows
    // into a blank line and dispatch a frame with no event name.
    const messages = parse("event: token\r", "\ndata: x\r", "\n\r", "\n");
    expect(messages).toEqual([{ event: "token", data: "x" }]);
  });

  it("reassembles a frame split mid-line and mid-field", () => {
    expect(parse("ev", "ent: res", "ult\nda", 'ta: {"answ', 'er":"ok"}\n', "\n")).toEqual([
      { event: "result", data: '{"answer":"ok"}' },
    ]);
  });

  it("ignores keep-alive comments, including between frames of one event", () => {
    expect(parse(": keepalive\n\n", "event: stage\n: still here\ndata: 1\n\n")).toEqual([
      { event: "stage", data: "1" },
    ]);
  });

  it("joins multiple data lines with newlines", () => {
    expect(parse("data: one\ndata: two\n\n")).toEqual([{ event: "message", data: "one\ntwo" }]);
  });

  it("does not dispatch a frame that carries no data", () => {
    expect(parse("event: start\n\n")).toEqual([]);
  });

  it("strips exactly one leading space from a value", () => {
    expect(parse("data:  two spaces\n\n")).toEqual([{ event: "message", data: " two spaces" }]);
  });

  it("never dispatches an unterminated frame", () => {
    expect(parse('event: result\ndata: {"answer":"half"}\n')).toEqual([]);
  });

  it("drops a leading byte-order mark", () => {
    expect(parse("﻿data: x\n\n")).toEqual([{ event: "message", data: "x" }]);
  });
});
