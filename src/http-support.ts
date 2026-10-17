import { createHash, timingSafeEqual } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

/** Largest request body accepted (8 MiB). Larger bodies are answered with 413. */
export const MAX_BODY_BYTES = 8 * 1024 * 1024;

/** `error.type` used for malformed or unsupported requests. */
export const INVALID_REQUEST_ERROR = "invalid_request_error";

/** HTTP status for a body that exceeds {@link MAX_BODY_BYTES}. */
const STATUS_PAYLOAD_TOO_LARGE = 413;

/** HTTP status for a body that is not a JSON object. */
const STATUS_BAD_REQUEST = 400;

/**
 * An error that maps directly to an OpenAI style error response.
 *
 * Thrown by request validation and routing; the server turns it into
 * `{ error: { message, type, param: null, code: null } }` with `status`.
 */
export class HttpError extends Error {
  /**
   * @param status - HTTP status code of the response.
   * @param message - Message shown to the client.
   * @param type - OpenAI `error.type`. Defaults to `invalid_request_error`.
   */
  public constructor(
    public readonly status: number,
    message: string,
    public readonly type: string = INVALID_REQUEST_ERROR,
  ) {
    super(message);
  }
}

/** One choice of a non-streaming chat completion. */
export interface ChatCompletionChoice {
  readonly index: number;
  readonly message: { readonly role: "assistant"; readonly content: string };
  readonly finish_reason: string;
}

/** An OpenAI `chat.completion` response body. */
export interface ChatCompletion {
  readonly id: string;
  readonly object: "chat.completion";
  readonly created: number;
  readonly model: string;
  readonly choices: readonly ChatCompletionChoice[];
  readonly usage: {
    readonly prompt_tokens: number;
    readonly completion_tokens: number;
    readonly total_tokens: number;
  };
}

/**
 * Renders a completion as a server-sent event stream.
 *
 * The upstream call is not streaming, so a stream request gets the whole
 * answer as one content chunk, a closing chunk carrying the finish reason, and
 * the `[DONE]` sentinel.
 *
 * @param completion - The finished completion.
 * @returns The full text of the event stream.
 * @example
 * ```ts
 * response.writeHead(200, { "content-type": "text/event-stream" }).end(toSseChunks(completion));
 * ```
 */
export function toSseChunks(completion: ChatCompletion): string {
  const choice = completion.choices[0]!;
  const base = {
    id: completion.id,
    object: "chat.completion.chunk",
    created: completion.created,
    model: completion.model,
  };
  const first = {
    ...base,
    choices: [{ index: 0, delta: { role: "assistant", content: choice.message.content }, finish_reason: null }],
  };
  const last = {
    ...base,
    choices: [{ index: 0, delta: {}, finish_reason: choice.finish_reason }],
  };
  return `data: ${JSON.stringify(first)}\n\ndata: ${JSON.stringify(last)}\n\ndata: [DONE]\n\n`;
}

/**
 * Writes a JSON response and ends it.
 *
 * @param response - Response to write.
 * @param status - HTTP status code.
 * @param body - Value to serialize.
 */
export function sendJson(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { "content-type": "application/json" }).end(JSON.stringify(body));
}

/**
 * Writes an OpenAI shaped error envelope.
 *
 * @param response - Response to write.
 * @param status - HTTP status code.
 * @param message - Message shown to the client.
 * @param type - OpenAI `error.type`.
 */
export function sendError(response: ServerResponse, status: number, message: string, type: string): void {
  sendJson(response, status, { error: { message, type, param: null, code: null } });
}

/**
 * Reads and parses a JSON object request body.
 *
 * @param request - Incoming request.
 * @returns The parsed object.
 * @throws {HttpError} 413 when the body exceeds {@link MAX_BODY_BYTES}; 400 when
 *   it is not valid JSON or not a JSON object (arrays and null are rejected).
 */
export async function readJson(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    size += (chunk as Buffer).length;
    if (size > MAX_BODY_BYTES) {
      throw new HttpError(STATUS_PAYLOAD_TOO_LARGE, "request body too large");
    }
    chunks.push(chunk as Buffer);
  }
  try {
    const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("not an object");
    }
    return parsed as Record<string, unknown>;
  } catch {
    throw new HttpError(STATUS_BAD_REQUEST, "request body must be a JSON object");
  }
}

/**
 * Checks an `Authorization` header against the configured API key.
 *
 * Both sides are hashed first so the comparison takes the same time whatever
 * the lengths are, which avoids leaking the key length.
 *
 * @param header - Value of the `Authorization` header, if any.
 * @param apiKey - Configured API key.
 * @returns True when the header is exactly `Bearer <apiKey>`.
 */
export function isApiKeyValid(header: string | undefined, apiKey: string): boolean {
  const digest = (value: string): Buffer => createHash("sha256").update(value).digest();
  return timingSafeEqual(digest(header ?? ""), digest(`Bearer ${apiKey}`));
}
