/**
 * HTTP client for the M&A Due Diligence API.
 *
 * Every function here is a thin transport wrapper. No pipeline decision is made
 * on this side — refusal, confidence and validation status all arrive from the
 * server already decided, and the UI only renders them.
 *
 * What this module does own is failure. Each call either returns a payload that
 * has been shape-checked (`lib/normalize.ts`) or throws an `ApiError` that says
 * what went wrong in terms a visitor can act on (`lib/errors.ts`). Nothing
 * resolves "successfully" with half an answer.
 */

import { ApiError, errorFromResponse, toApiError } from "./errors";
import {
  normalizeIngestResult,
  normalizePlannedStages,
  normalizeQueryResponse,
  normalizeStageEvent,
  normalizeTieOut,
} from "./normalize";
import { SseParser } from "./sse";
import type {
  BudgetStatus,
  Deal,
  DocumentRecord,
  IngestResult,
  PlannedStage,
  QueryResponse,
  RiskSignal,
  StageEvent,
  TieOutReport,
} from "./types";

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

const API = `${API_BASE}/api/v1`;

/** The server's `QueryRequest.query` max_length; longer questions are a 422. */
export const MAX_QUERY_LENGTH = 2000;

/**
 * How long an open stream may stay completely silent before it is abandoned.
 *
 * The server emits nothing while a stage runs, and a slow stage is legitimate:
 * several retrieval passes on a CPU host, or a synthesis model consuming its
 * whole deadline before the ladder moves on, can each run past a minute. So
 * this is a ceiling on a dead connection (no FIN ever arrives, the reader
 * would otherwise wait forever), not a latency target.
 */
export const STREAM_IDLE_TIMEOUT_MS = 180_000;

/**
 * GETs a JSON endpoint, returning null on any failure.
 *
 * Panels are independent: the sources rail must still render when the risk
 * endpoint is down, so transport errors do not throw into the render tree.
 * They return null rather than an empty value, though, because "could not
 * load" and "there is nothing" are different facts and the panels say which.
 */
async function getJson<T>(path: string, timeoutMs = 10_000): Promise<T | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API}${path}`, { signal: controller.signal });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Lists the queryable deals.
 *
 * Unlike the panel fetches this one throws: with no deal there is nothing to
 * ask, so the page needs to know whether the list is empty or the engine is
 * unreachable (a cold start looks exactly like "no deals" otherwise).
 *
 * @throws ApiError
 */
export async function listDeals(timeoutMs = 10_000): Promise<Deal[]> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API}/deals`, { signal: controller.signal });
    if (!res.ok) throw await errorFromResponse(res, "Loading deals");
    const body: unknown = await res.json();
    return Array.isArray(body) ? (body as Deal[]) : [];
  } catch (e) {
    // An abort here is this function's own timeout, which is a network
    // failure from the caller's point of view.
    if (e instanceof DOMException && e.name === "AbortError") {
      throw toApiError(new TypeError("timeout"), API_BASE);
    }
    throw toApiError(e, API_BASE);
  } finally {
    clearTimeout(timer);
  }
}

export const listDocuments = (dealId: string) =>
  getJson<DocumentRecord[]>(`/deals/${encodeURIComponent(dealId)}/documents`);

export const listRiskSignals = (dealId: string) =>
  getJson<RiskSignal[]>(`/deals/${encodeURIComponent(dealId)}/risk-signals`);

export const getBudget = () => getJson<BudgetStatus>("/budget");

/**
 * The deal's figure tie-out report (Decision 92), or null when it cannot be
 * had. Optional for the page like every panel: a failure leaves the rest of
 * the deal view as it is, and the panel says the report could not be loaded.
 * The server reads the whole data room within a 20 s budget, hence the
 * longer timeout.
 */
export async function getTieOut(dealId: string): Promise<TieOutReport | null> {
  const raw = await getJson<unknown>(`/deals/${encodeURIComponent(dealId)}/tie-out`, 45_000);
  return raw === null ? null : normalizeTieOut(raw);
}

/**
 * Creates a deal.
 *
 * @param dealName Human-readable name.
 * @param isSandbox Mark the deal ephemeral — purged on tab close, swept on TTL.
 * @throws ApiError
 */
export async function createDeal(
  dealName: string,
  isSandbox = false,
): Promise<Deal> {
  try {
    const res = await fetch(`${API}/deals`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        deal_name: dealName,
        description: isSandbox ? "Temporary sandbox deal" : "",
        is_sandbox: isSandbox,
      }),
    });
    if (!res.ok) throw await errorFromResponse(res, "Creating the temporary deal");
    return (await res.json()) as Deal;
  } catch (e) {
    throw toApiError(e, API_BASE);
  }
}

/**
 * Uploads and ingests one document into a deal.
 *
 * @param dealId Target deal.
 * @param file The document.
 * @returns The ingest result, including `notices` (why a "partial" ingestion
 *   was partial) and `version_candidates` — documents this one looks like a
 *   version of. Those are a proposal; nothing has been superseded.
 * @throws ApiError — 400/413 carry the server's reason (unsupported type,
 *   over the size limit, empty file), 410 means the sandbox has expired.
 */
export async function ingestDocument(dealId: string, file: File): Promise<IngestResult> {
  const form = new FormData();
  form.append("deal_id", dealId);
  form.append("file", file);

  try {
    const res = await fetch(`${API}/ingest`, { method: "POST", body: form });
    if (!res.ok) throw await errorFromResponse(res, "Indexing the document");
    const result = normalizeIngestResult(await res.json());
    if (!result) {
      throw new ApiError({
        kind: "server",
        status: res.status,
        message: "The engine accepted the file but returned a response this client could not read.",
      });
    }
    return result;
  } catch (e) {
    throw toApiError(e, API_BASE);
  }
}

/**
 * Deletes a sandbox deal and everything indexed under it.
 *
 * Uses `sendBeacon` when the page is unloading, because a normal `fetch` is
 * cancelled the moment the browser tears the document down — which is precisely
 * when this needs to run. The beacon is queued by the browser and delivered
 * independently of the page's lifetime. It can only issue POST, hence the
 * dedicated `/purge` alias on the server.
 *
 * @param dealId Deal to delete.
 * @param unloading True when called from a `pagehide` handler.
 */
export function purgeDeal(dealId: string, unloading = false): void {
  const url = `${API}/deals/${encodeURIComponent(dealId)}/purge`;

  if (unloading && typeof navigator !== "undefined" && navigator.sendBeacon) {
    navigator.sendBeacon(url, new Blob([], { type: "text/plain" }));
    return;
  }

  // Not unloading: a normal request is fine, and `keepalive` covers the case
  // where the tab closes mid-flight anyway.
  void fetch(url, { method: "POST", keepalive: true }).catch(() => {
    // Deliberately silent. The TTL sweeper is the guarantee; this is the
    // fast path, and a user should never see an error for cleanup they did
    // not ask for.
  });
}

/**
 * Progress events delivered while a query streams.
 *
 * These are the server's `start`, `stage`, `token`, `answer_reset` and
 * `result` frames after shape-checking. `token` frames carry the answer as the
 * model writes it — an unverified draft; `answer_reset` tells the client to
 * discard that draft (the model stream broke and another model is taking over,
 * or a failed validation is being revised). The checked answer always arrives
 * in `result`. The server's `error` frame is not delivered as an event: it
 * ends the stream, so `streamQuery` throws it.
 */
export type StreamEvent =
  | { event: "start"; data: { session_id: string; planned_stages: PlannedStage[] } }
  | { event: "stage"; data: StageEvent }
  | { event: "token"; data: { text: string } }
  | { event: "answer_reset"; data: { reason: string } }
  | { event: "result"; data: QueryResponse };

export interface StreamQueryArgs {
  query: string;
  dealId: string;
  includePii: boolean;
  signal?: AbortSignal;
  onEvent: (event: StreamEvent) => void;
  /** Overrides STREAM_IDLE_TIMEOUT_MS; for tests. */
  idleTimeoutMs?: number;
}

const INTERRUPTED_MESSAGE =
  "The connection to the engine closed before a verified answer arrived. " +
  "Anything shown while it was writing was an unverified draft and has been discarded.";

function parseFrame(event: string, data: string): StreamEvent | ApiError | null {
  let payload: unknown;
  try {
    payload = JSON.parse(data);
  } catch {
    // One unreadable progress frame is not worth abandoning the run for. If it
    // was the result frame, the stream ends without one and that is reported.
    return null;
  }
  const bag = (payload && typeof payload === "object" ? payload : {}) as Record<string, unknown>;

  switch (event) {
    case "start":
      return {
        event: "start",
        data: {
          session_id: typeof bag.session_id === "string" ? bag.session_id : "",
          planned_stages: normalizePlannedStages(bag.planned_stages),
        },
      };
    case "stage": {
      const stage = normalizeStageEvent(payload);
      return stage ? { event: "stage", data: stage } : null;
    }
    case "token":
      return typeof bag.text === "string" && bag.text
        ? { event: "token", data: { text: bag.text } }
        : null;
    case "answer_reset":
      return {
        event: "answer_reset",
        data: { reason: typeof bag.reason === "string" ? bag.reason : "" },
      };
    case "result": {
      const result = normalizeQueryResponse(payload);
      return result ? { event: "result", data: result } : null;
    }
    case "error":
      // Always {detail, code, session_id, request_id}. `detail` is generic,
      // visitor-safe text; `code` is "timeout" or "pipeline_error".
      return new ApiError({
        kind: bag.code === "timeout" ? "timeout" : "pipeline",
        message:
          typeof bag.detail === "string" && bag.detail
            ? bag.detail
            : "The engine reported an internal error while answering.",
        requestId:
          typeof bag.request_id === "string" && bag.request_id ? bag.request_id : undefined,
      });
    default:
      // An event this client does not know — a newer server. Ignore it.
      return null;
  }
}

/**
 * Runs a query against the streaming endpoint, invoking `onEvent` per frame.
 *
 * Uses `fetch` + a ReadableStream reader rather than `EventSource`, for two
 * reasons: EventSource is GET-only (the query would have to go in the URL) and
 * cannot be aborted cleanly mid-flight.
 *
 * The contract with the caller is deliberately narrow: this resolves only
 * after a `result` event has been delivered. Every other ending throws —
 * a non-2xx response, an in-band `error` frame, a connection that drops, a
 * stream that closes without a result, or one that goes silent. There is no
 * automatic retry or reconnect: the endpoint is not resumable, and a second
 * request would spend another unit of the visitor's rate limit and of the
 * engine's model quota on a question that may already have been answered
 * server-side.
 *
 * @param args Query parameters and the per-event callback.
 * @throws ApiError, or the caller's own AbortError when `signal` fires.
 */
export async function streamQuery({
  query,
  dealId,
  includePii,
  signal,
  onEvent,
  idleTimeoutMs = STREAM_IDLE_TIMEOUT_MS,
}: StreamQueryArgs): Promise<void> {
  // The caller's signal and the idle watchdog both need to cancel the same
  // request, so they are funnelled through one controller.
  const controller = new AbortController();
  const onCallerAbort = () => controller.abort();
  if (signal?.aborted) controller.abort();
  else signal?.addEventListener("abort", onCallerAbort, { once: true });

  let timedOut = false;
  let watchdog: ReturnType<typeof setTimeout> | undefined;
  const armWatchdog = () => {
    clearTimeout(watchdog);
    watchdog = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, idleTimeoutMs);
  };

  const callerAborted = () => signal?.aborted === true;
  const abortError = () => new DOMException("The query was cancelled.", "AbortError");

  try {
    armWatchdog();

    let res: Response;
    try {
      res = await fetch(`${API}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({
          query,
          deal_id: dealId,
          include_pii: includePii,
        }),
        signal: controller.signal,
      });
    } catch (e) {
      if (callerAborted()) throw abortError();
      if (timedOut) throw idleTimeout();
      throw toApiError(e, API_BASE);
    }

    if (!res.ok) throw await errorFromResponse(res, "Query");
    if (!res.body) {
      throw new ApiError({ kind: "interrupted", message: INTERRUPTED_MESSAGE });
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    const parser = new SseParser();

    while (true) {
      let chunk: ReadableStreamReadResult<Uint8Array>;
      try {
        chunk = await reader.read();
      } catch {
        if (callerAborted()) throw abortError();
        if (timedOut) throw idleTimeout();
        // The socket dropped mid-answer (network change, API restart, proxy
        // idle timeout).
        throw new ApiError({ kind: "interrupted", message: INTERRUPTED_MESSAGE });
      }
      if (chunk.done) break;
      armWatchdog();

      for (const message of parser.feed(decoder.decode(chunk.value, { stream: true }))) {
        const parsed = parseFrame(message.event, message.data);
        if (parsed === null) continue;
        if (parsed instanceof ApiError) {
          void reader.cancel().catch(() => {});
          throw parsed;
        }
        // Outside any try/catch on purpose: an exception thrown by the
        // caller's handler is the caller's bug and must surface, not be
        // mistaken for a malformed frame and swallowed.
        onEvent(parsed);
        if (parsed.event === "result") {
          // Anything after the result is not part of the contract.
          void reader.cancel().catch(() => {});
          return;
        }
      }
    }

    // The stream closed cleanly but never said how the run ended.
    throw new ApiError({ kind: "interrupted", message: INTERRUPTED_MESSAGE });
  } finally {
    clearTimeout(watchdog);
    signal?.removeEventListener("abort", onCallerAbort);
  }
}

function idleTimeout(): ApiError {
  return new ApiError({
    kind: "timeout",
    message:
      "The engine stopped responding before a verified answer arrived. " +
      "The question may still be running on the server; nothing shown here was verified.",
  });
}

/**
 * Non-streaming query. Not used by the page (the stream is the product), kept
 * as the one-call path for scripts and tests against `/query`.
 *
 * @throws ApiError
 */
export async function runQuery(
  query: string,
  dealId: string,
  includePii: boolean,
): Promise<QueryResponse> {
  try {
    const res = await fetch(`${API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, deal_id: dealId, include_pii: includePii }),
    });
    if (!res.ok) throw await errorFromResponse(res, "Query");
    const result = normalizeQueryResponse(await res.json());
    if (!result) {
      throw new ApiError({
        kind: "server",
        status: res.status,
        message: "The engine returned a response this client could not read.",
      });
    }
    return result;
  } catch (e) {
    throw toApiError(e, API_BASE);
  }
}
