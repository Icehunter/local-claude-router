import { createHash, timingSafeEqual } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

/** Largest request body accepted (8 MiB). Larger bodies are answered with 413. */
export const MAX_BODY_BYTES = 8 * 1024 * 1024;

/** Maps upstream stop reasons to OpenAI finish reasons. Unknown reasons become `unknown`. */
const FINISH_REASONS: Readonly<Record<string, string>> = {
  max_tokens: "length",
  end_turn: "stop",
  stop_sequence: "stop",
  refusal: "content_filter",
  tool_use: "tool_calls",
  pause_turn: "incomplete",
};

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

/** One entry of an OpenAI `tool_calls` array. */
export interface ChatCompletionToolCall {
  readonly id: string;
  readonly type: "function";
  readonly function: { readonly name: string; readonly arguments: string };
}

/** One choice of a non-streaming chat completion. */
export interface ChatCompletionChoice {
  readonly index: number;
  readonly message: { readonly role: "assistant"; readonly content: string; readonly tool_calls?: readonly ChatCompletionToolCall[] };
  readonly finish_reason: string;
}

/** An OpenAI `chat.completion` response body. */
export interface ChatCompletion {
  readonly id: string;
  readonly object: "chat.completion";
  readonly created: number;
  readonly model: string;
  readonly choices: readonly ChatCompletionChoice[];
  readonly upstream?: { readonly request_id: string | null; readonly stop_reason: string | null; readonly block_types: string[]; readonly usage: unknown };
  readonly usage: {
    readonly prompt_tokens: number;
    readonly completion_tokens: number;
    readonly total_tokens: number;
  } | null;
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
    choices: [{
      index: 0,
      delta: {
        role: "assistant",
        content: choice.message.content === "" ? null : choice.message.content,
        ...(choice.message.tool_calls ? { tool_calls: choice.message.tool_calls } : {}),
      },
      finish_reason: null,
    }],
  };
  const last = {
    ...base,
    usage: completion.usage,
    ...(completion.upstream ? {upstream: completion.upstream} : {}),
    choices: [{ index: 0, delta: {}, finish_reason: choice.finish_reason }],
  };
  return `data: ${JSON.stringify(first)}\n\ndata: ${JSON.stringify(last)}\n\ndata: [DONE]\n\n`;
}

/**
 * Translates upstream Anthropic SSE events into OpenAI `chat.completion.chunk`
 * stream lines.
 *
 * The generator yields `data: <json>\n\n` lines in this order: a role chunk on
 * `message_start`, one content chunk per text delta, one `tool_calls` delta
 * per stopped tool_use block (with a distinct `index` per block, so parallel
 * calls stay separate), a finish-reason chunk on `message_delta`, a usage
 * chunk, and finally the `[DONE]` sentinel. Events the translator does not
 * recognise are skipped.
 *
 * @param upstreamEvents - An async iterable of `{ event, data }` pairs.
 * @param completionId - OpenAI id echoed on every chunk.
 * @param model - Model id echoed on every chunk.
 * @param created - Creation time in epoch seconds, echoed on every chunk.
 * @returns An async generator of SSE lines to write to the response.
 */
export async function* translateSSE(
  upstreamEvents: AsyncIterable<{ event: string; data: string }>,
  completionId: string,
  model: string,
  created: number,
): AsyncGenerator<string> {
  const base = {
    id: completionId,
    object: "chat.completion.chunk",
    created,
    model,
  };

  const emit = (choices: unknown[], usage?: unknown): string =>
    `data: ${JSON.stringify({ ...base, choices, ...(usage ? { usage } : {}) })}\n\n`;

  let toolId = "";
  let toolName = "";
  let toolArgs = "";
  let toolCallIndex = -1;
  let stopReason: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;

  const parse = (data: string): Record<string, unknown> => {
    try {
      return JSON.parse(data) as Record<string, unknown>;
    } catch {
      return {};
    }
  };

  for await (const { event, data } of upstreamEvents) {
    const parsed = parse(data);
    const delta = parsed.delta as Record<string, unknown> | undefined;

    if (event === "message_start") {
      const usage = parsed.usage as { input_tokens?: number; output_tokens?: number } | undefined;
      if (usage) {
        inputTokens = usage.input_tokens ?? 0;
        outputTokens = usage.output_tokens ?? 0;
      }
      yield emit([{ index: 0, delta: { role: "assistant" }, finish_reason: null }]);
    } else if (event === "content_block_start") {
      const block = parsed.content_block as { type?: string; id?: string; name?: string } | undefined;
      if (block?.type === "tool_use") {
        toolId = block.id ?? "";
        toolName = block.name ?? "";
        toolArgs = "";
        toolCallIndex += 1;
      }
    } else if (event === "content_block_delta") {
      if (delta?.type === "text_delta") {
        yield emit([{ index: 0, delta: { content: String(delta.text ?? "") }, finish_reason: null }]);
      } else if (delta?.type === "input_json_delta") {
        toolArgs += String(delta.partial_json ?? "");
      }
    } else if (event === "content_block_stop") {
      if (toolId !== "") {
        yield emit([{ index: 0, delta: { tool_calls: [{ index: toolCallIndex, id: toolId, type: "function", function: { name: toolName, arguments: toolArgs } }] }, finish_reason: null }]);
        toolId = "";
        toolName = "";
        toolArgs = "";
      }
    } else if (event === "message_delta") {
      const d = (delta?.delta ?? delta) as Record<string, unknown>;
      stopReason = d.stop_reason as string | null;
      const usage = d.usage as { input_tokens?: number; output_tokens?: number } | undefined;
      if (usage) {
        inputTokens = usage.input_tokens ?? inputTokens;
        outputTokens = usage.output_tokens ?? outputTokens;
      }
      if (stopReason !== null) {
        yield emit([{ index: 0, delta: {}, finish_reason: FINISH_REASONS[stopReason] ?? "unknown" }], {
          prompt_tokens: inputTokens,
          completion_tokens: outputTokens,
          total_tokens: inputTokens + outputTokens,
        });
      }
    }
    // Other events are ignored.
  }
  yield "data: [DONE]\n\n";
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
 * Writes an OpenAI-style SSE error event and ends the response.
 *
 * A streaming response has already sent its 200 headers, so it can no longer
 * return an HTTP error status. The error is delivered as an `event: error`
 * line (matching how OpenAI surfaces stream errors) instead of an empty body.
 *
 * @param response - Streaming response that has already started.
 * @param message - Message shown to the client.
 * @param type - OpenAI `error.type`.
 */
export function sendSseError(response: ServerResponse, message: string, type: string): void {
  response.write(`event: error\ndata: ${JSON.stringify({ error: { message, type, param: null, code: null } })}\n\n`);
  response.end();
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
