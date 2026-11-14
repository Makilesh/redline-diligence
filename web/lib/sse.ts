/**
 * Incremental Server-Sent Events parser.
 *
 * `/query/stream` is read with `fetch` rather than `EventSource` (it is a POST
 * and must be abortable), so the framing has to be parsed here. This follows
 * the WHATWG event-stream rules rather than splitting on "\n\n", because the
 * bytes pass through proxies this project does not control (Caddy, the Hugging
 * Face edge, Vercel rewrites):
 *
 *   - a line ends with LF, CRLF or a bare CR;
 *   - a network chunk can end anywhere — mid-line, mid-frame, or between the
 *     CR and the LF of one line ending;
 *   - a line starting with ":" is a comment (the keep-alive a proxy or the
 *     server may send while a slow stage runs) and carries no event;
 *   - several `data:` lines in one frame are joined with "\n";
 *   - a frame with no data is not dispatched;
 *   - whatever is still buffered when the stream ends is an incomplete frame
 *     and is discarded, never dispatched half-read.
 */

export interface SseMessage {
  /** The `event:` field, or "message" when the frame did not name one. */
  event: string;
  /** The frame's `data:` lines, joined with newlines. */
  data: string;
}

export class SseParser {
  private buffer = "";
  private eventName = "";
  private dataLines: string[] = [];
  private atStart = true;

  /**
   * Feeds one decoded chunk and returns every frame it completed.
   *
   * @param chunk Text from the stream, in arrival order.
   * @returns Completed messages, possibly none.
   */
  feed(chunk: string): SseMessage[] {
    if (this.atStart && chunk) {
      // A UTF-8 byte-order mark is legal at the very start of the stream.
      if (chunk.charCodeAt(0) === 0xfeff) chunk = chunk.slice(1);
      this.atStart = false;
    }
    this.buffer += chunk;

    const messages: SseMessage[] = [];
    let cursor = 0;

    while (cursor < this.buffer.length) {
      const lf = this.buffer.indexOf("\n", cursor);
      const cr = this.buffer.indexOf("\r", cursor);
      if (lf === -1 && cr === -1) break;

      let lineEnd: number;
      let next: number;
      if (cr !== -1 && (lf === -1 || cr < lf)) {
        // A CR as the buffer's last character may be the first half of a CRLF
        // whose LF is still on the wire. Wait for the next chunk to find out;
        // treating it as a line end now would turn that LF into a blank line
        // and dispatch the frame early.
        if (cr === this.buffer.length - 1) break;
        lineEnd = cr;
        next = this.buffer[cr + 1] === "\n" ? cr + 2 : cr + 1;
      } else {
        lineEnd = lf;
        next = lf + 1;
      }

      const message = this.consumeLine(this.buffer.slice(cursor, lineEnd));
      if (message) messages.push(message);
      cursor = next;
    }

    this.buffer = this.buffer.slice(cursor);
    return messages;
  }

  private consumeLine(line: string): SseMessage | null {
    if (line === "") {
      // Blank line: the frame is complete.
      const complete =
        this.dataLines.length > 0
          ? { event: this.eventName || "message", data: this.dataLines.join("\n") }
          : null;
      this.eventName = "";
      this.dataLines = [];
      return complete;
    }

    if (line.startsWith(":")) return null;

    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    // Exactly one leading space is part of the framing, not of the value.
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") this.eventName = value;
    else if (field === "data") this.dataLines.push(value);
    // `id` and `retry` are ignored: the endpoint is not resumable, so there is
    // no Last-Event-ID to send and no reconnection to schedule.
    return null;
  }
}
