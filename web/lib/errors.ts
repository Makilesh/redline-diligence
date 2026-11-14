/**
 * One error type for everything that can go wrong between the browser and the
 * engine, classified by what the visitor can do about it.
 *
 * The server's own `detail` strings are used verbatim wherever it sends one —
 * they are written for visitors ("The demo's daily query budget is spent. It
 * resets at 00:00 UTC.", "File exceeds the 25 MB upload limit") and are more
 * specific than anything the client could infer from a status code. Every
 * message here is plain text and is rendered as a text node, never as markup.
 *
 * `Retry-After` is read when the browser lets it be read. Cross-origin, that
 * needs the API to list it in `Access-Control-Expose-Headers`, which it does
 * not do today (only `X-Request-ID` is exposed) — so nothing here depends on
 * it: the server's `detail` already says what to do, and a known wait is a
 * bonus, not the message.
 */

export type FailureKind =
  /** 429 — this visitor's rate limit, or the demo's global daily cap. */
  | "rate_limited"
  /** 503 with a reason — all pipeline slots taken, restarting, workspaces full. */
  | "busy"
  /** 502/503/504 from a proxy, or a cold start — the engine is not up yet. */
  | "unavailable"
  /** 422 with a text reason — the public query guard declined the question. */
  | "blocked"
  /** 400/413/422 — the request itself is not acceptable. */
  | "invalid"
  /** 403/410 — the target deal may not be used (not a sandbox, or expired). */
  | "forbidden"
  /** 404 — the deal does not exist, or this visitor may not read it. */
  | "not_found"
  /** `fetch` rejected before any response: offline, DNS, CORS, connection refused. */
  | "network"
  /** The stream ended, or the connection dropped, before a verified result. */
  | "interrupted"
  /** The run passed its deadline (504 / `code: "timeout"`), or the stream went silent. */
  | "timeout"
  /** The pipeline reported its own failure in an `error` frame. */
  | "pipeline"
  /** Any other non-2xx. */
  | "server";

export interface ApiErrorInit {
  kind: FailureKind;
  message: string;
  status?: number;
  retryAfterSeconds?: number;
  requestId?: string;
}

export class ApiError extends Error {
  readonly kind: FailureKind;
  readonly status?: number;
  /** Seconds the server asked the client to wait, when it said and CORS exposed it. */
  readonly retryAfterSeconds?: number;
  /** Correlation id for the server log (`X-Request-ID` / the error frame's `request_id`). */
  readonly requestId?: string;

  constructor({ kind, message, status, retryAfterSeconds, requestId }: ApiErrorInit) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.retryAfterSeconds = retryAfterSeconds;
    this.requestId = requestId;
  }
}

/** True for failures where asking the same thing again later can succeed. */
export function isRetryable(kind: FailureKind): boolean {
  return (
    kind !== "blocked" && kind !== "invalid" && kind !== "forbidden" && kind !== "not_found"
  );
}

/** A short heading for an error panel, by kind. */
export function failureTitle(kind: FailureKind): string {
  switch (kind) {
    case "rate_limited":
      return "Query limit reached";
    case "busy":
      return "Engine busy";
    case "unavailable":
      return "Engine unavailable";
    case "blocked":
      return "Question not run";
    case "invalid":
      return "Request not accepted";
    case "forbidden":
      return "Not permitted";
    case "not_found":
      return "Deal not available";
    case "network":
      return "Engine unreachable";
    case "interrupted":
      return "Connection lost";
    case "timeout":
      return "Timed out";
    case "pipeline":
    case "server":
      return "The engine hit an error";
  }
}

/**
 * Parses a `Retry-After` header: delta-seconds or an HTTP date.
 *
 * @returns Whole seconds to wait, or undefined when absent or unparseable.
 */
export function parseRetryAfter(value: string | null, now = Date.now()): number | undefined {
  if (!value) return undefined;
  const trimmed = value.trim();
  if (/^\d+$/.test(trimmed)) return Number(trimmed);
  const date = Date.parse(trimmed);
  if (Number.isNaN(date)) return undefined;
  return Math.max(0, Math.ceil((date - now) / 1000));
}

/**
 * Renders a wait in the unit a person would use.
 *
 * The daily cap's Retry-After runs to tens of thousands of seconds; "in about
 * 43211 seconds" is technically an answer and practically useless.
 */
export function formatWait(seconds: number): string {
  const s = Math.max(1, Math.ceil(seconds));
  if (s < 90) return `${s} second${s === 1 ? "" : "s"}`;
  const minutes = Math.round(s / 60);
  if (minutes < 90) return `${minutes} minutes`;
  const hours = Math.round(s / 3600);
  return `${hours} hour${hours === 1 ? "" : "s"}`;
}

/**
 * Extracts a human-readable reason from a FastAPI error body.
 *
 * `detail` is a string for HTTPException and a list of `{loc, msg}` objects
 * for request-validation failures; both shapes occur on 422.
 */
export function detailFromBody(body: unknown): string {
  if (!body || typeof body !== "object") return "";
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail.trim();
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (!item || typeof item !== "object") return "";
        const { loc, msg } = item as { loc?: unknown; msg?: unknown };
        const field = Array.isArray(loc) ? loc.filter((p) => p !== "body").join(".") : "";
        const text = typeof msg === "string" ? msg : "";
        return field && text ? `${field}: ${text}` : text;
      })
      .filter(Boolean)
      .join("; ");
  }
  return "";
}

function withWait(message: string, retryAfterSeconds: number | undefined): string {
  if (retryAfterSeconds === undefined || retryAfterSeconds <= 0) return message;
  return `${message} You can try again in about ${formatWait(retryAfterSeconds)}.`;
}

const COLD_START =
  "The engine is starting up or temporarily unavailable. " +
  "A cold start loads several models and can take a minute or two — try again shortly.";

/**
 * Turns a non-2xx response into an ApiError a visitor can act on.
 *
 * @param res The failed response. Its body is consumed.
 * @param operation What was being attempted, for the generic fallback ("Query").
 */
export async function errorFromResponse(res: Response, operation: string): Promise<ApiError> {
  let detail = "";
  let detailIsFieldList = false;
  try {
    const body: unknown = await res.json();
    detail = detailFromBody(body);
    detailIsFieldList = Array.isArray((body as { detail?: unknown } | null)?.detail);
  } catch {
    // Not JSON (a proxy's HTML error page, an empty body) — status is all there is.
  }

  const retryAfterSeconds = parseRetryAfter(res.headers.get("Retry-After"));
  const requestId = res.headers.get("X-Request-ID") ?? undefined;
  const base = { status: res.status, retryAfterSeconds, requestId };

  switch (res.status) {
    case 429:
      return new ApiError({
        ...base,
        kind: "rate_limited",
        message: withWait(
          detail || "You have reached the query limit for this demo.",
          retryAfterSeconds,
        ),
      });
    case 503:
      // Three of the API's 503s carry a visitor-facing reason (every pipeline
      // slot taken, the process draining for a restart, the sandbox registry
      // full). The fourth — "Application not fully initialized" — is a cold
      // start, and a proxy's 503 has no JSON body at all.
      if (detail && !/not fully initiali[sz]ed/i.test(detail)) {
        return new ApiError({ ...base, kind: "busy", message: withWait(detail, retryAfterSeconds) });
      }
      return new ApiError({ ...base, kind: "unavailable", message: COLD_START });
    case 504:
      // The API's own 504 is the pipeline deadline, and its detail names the
      // request id. A gateway's 504 has no JSON body.
      if (detail) return new ApiError({ ...base, kind: "timeout", message: detail });
      return new ApiError({ ...base, kind: "unavailable", message: COLD_START });
    case 502:
      return new ApiError({ ...base, kind: "unavailable", message: COLD_START });
    case 499:
      // Sent when the server saw this client disconnect. If the page aborted
      // on purpose the caller never gets here; anything else is a lost link.
      return new ApiError({
        ...base,
        kind: "interrupted",
        message: "The connection closed before the engine finished. Nothing was answered.",
      });
    case 404:
      return new ApiError({
        ...base,
        kind: "not_found",
        message:
          "This deal is not available — it may have expired or been removed. " +
          "Choose another deal from the list.",
      });
    case 422:
      // A text reason is the query guard explaining itself; a field list is a
      // malformed request (an over-long question, for instance).
      return new ApiError({
        ...base,
        kind: detail && !detailIsFieldList ? "blocked" : "invalid",
        message: detail || `${operation} was rejected as invalid.`,
      });
    case 400:
    case 413:
      // The 413 detail names the limit ("File exceeds the 25 MB upload limit").
      return new ApiError({
        ...base,
        kind: "invalid",
        message:
          detail ||
          (res.status === 413
            ? `${operation} was rejected: the file is larger than the server accepts.`
            : `${operation} was rejected (400).`),
      });
    case 403:
    case 410:
      return new ApiError({
        ...base,
        kind: "forbidden",
        message: detail || `${operation} is not permitted (${res.status}).`,
      });
    default:
      return new ApiError({
        ...base,
        kind: "server",
        message: detail || `${operation} failed (${res.status}).`,
      });
  }
}

/** The error for a `fetch` that never produced a response. */
export function networkError(apiBase: string): ApiError {
  let host = apiBase;
  try {
    host = new URL(apiBase).host;
  } catch {
    // Relative or malformed base — show it as configured.
  }
  return new ApiError({
    kind: "network",
    message:
      `Could not reach the engine at ${host}. ` +
      "It may be starting up, or the connection may be down — try again in a moment.",
  });
}

/** Normalises anything thrown into an ApiError. */
export function toApiError(error: unknown, apiBase: string): ApiError {
  if (error instanceof ApiError) return error;
  // `fetch` rejects with a TypeError for every transport-level failure.
  if (error instanceof TypeError) return networkError(apiBase);
  return new ApiError({
    kind: "server",
    message: error instanceof Error && error.message ? error.message : "Unexpected error.",
  });
}
