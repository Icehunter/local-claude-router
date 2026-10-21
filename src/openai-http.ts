#!/usr/bin/env node
import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { responseTool } from "./response-tool.js";
import { realpathSync } from "node:fs";
import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import { fileURLToPath } from "node:url";
import { pino } from "pino";
import type { Logger } from "pino";
import pretty from "pino-pretty";
import { APIStatusError } from "./anthropic/client.js";
import type { Message, MessageRequest, MessageResult, TextBlock } from "./anthropic/client.js";
import { isHaikuModel } from "./anthropic/headers.js";
import { depsFromEnv, MAX_MAX_TOKENS } from "./deps.js";
import type { Deps } from "./deps.js";
import {
  HttpError,
  isApiKeyValid,
  readJson,
  sendError,
  sendJson,
  sendSseError,
  translateSSE,
} from "./http-support.js";
import type { ChatCompletion } from "./http-support.js";

export { HttpError } from "./http-support.js";
export type { ChatCompletion } from "./http-support.js";

/** Port used when `CLAUDE_WIRE_PORT` is not set. */
const DEFAULT_PORT = 8787;

/** Highest valid TCP port. */
const MAX_PORT = 65535;

/** Address used when `CLAUDE_WIRE_HOST` is not set. */
const DEFAULT_HOST = "127.0.0.1";

/** Log level used when `CLAUDE_WIRE_LOG_LEVEL` is not set. */
const DEFAULT_LOG_LEVEL = "info";

/** Value of `CLAUDE_WIRE_THINKING` used when it is not set. */
const DEFAULT_THINKING = "off";

/** Route that serves chat completions. */
const CHAT_COMPLETIONS_PATH = "/v1/chat/completions";

/** Route that lists models. */
const MODELS_PATH = "/v1/models";

/** Route for liveness checks. It is answered before authentication. */
const HEALTH_PATH = "/health";

/** Status at and above which a response is logged as a client error. */
const CLIENT_ERROR_STATUS = 400;

/** Status at and above which a response is logged as a server error. */
const SERVER_ERROR_STATUS = 500;

/** Highest status code treated as a valid HTTP error status when relaying an upstream error. */
const MAX_ERROR_STATUS = 599;

/** Status used when the upstream failed in a way that has no usable status code. */
const BAD_GATEWAY_STATUS = 502;

/** Hosts that are only reachable from this machine. */
const LOOPBACK_HOSTS: ReadonlySet<string> = new Set(["127.0.0.1", "::1", "localhost"]);

/** Accepted values of `CLAUDE_WIRE_EFFORT`, lowest to highest. */
const EFFORTS = ["low", "medium", "high", "xhigh", "max"] as const;

/** Effort level accepted by the upstream API. */
type Effort = (typeof EFFORTS)[number];

/** Maps upstream stop reasons to OpenAI finish reasons. Unknown reasons become `stop`. */
const FINISH_REASONS: Readonly<Record<string, string>> = {
  max_tokens: "length",
  end_turn: "stop",
  stop_sequence: "stop",
  refusal: "content_filter",
  tool_use: "tool_calls",
  pause_turn: "incomplete",
};

/** Finish reason used for stop reasons that have no entry in {@link FINISH_REASONS}. */
const DEFAULT_FINISH_REASON = "unknown";

/** Dependencies and settings of the HTTP server. */
export interface OpenAIHttpDeps {
  /** Upstream client that sends the translated request. */
  readonly client: Pick<Deps["client"], "createMessage" | "createMessageStream">;
  /** Model used when the request names a non-Claude model. Also listed by `/v1/models`. */
  readonly defaultModel: string;
  /** `max_tokens` used when the request names none. */
  readonly defaultMaxTokens: number;
  /** When set, requests must carry `Authorization: Bearer <apiKey>`. */
  readonly apiKey?: string;
  /** Fallback thinking mode when the request supplies no override. */
  readonly thinking?: MessageRequest["thinking"];
  /** Effort level added to every upstream request. */
  readonly effort?: MessageRequest["effort"];
  /** Logger. Defaults to a logger that discards everything. */
  readonly log?: Pick<Logger, "info" | "warn" | "error">;
  /** Clock in epoch seconds, used for the `created` field. */
  readonly now?: () => number;
}

/** Message content as the OpenAI API allows it: text, text parts, or absent. */
type OpenAIContent = string | Array<{ type?: string; text?: string }> | null | undefined;

/** Per-request fields written to the log line. Prompts and replies are never included. */
type RequestLogDetail = Record<string, unknown>;

/** A logger that discards everything. */
const SILENT_LOG: Pick<Logger, "info" | "warn" | "error"> = {
  info: () => {},
  warn: () => {},
  error: () => {},
};

/**
 * Flattens OpenAI message content to plain text.
 *
 * @param content - A string, an array of `{ type: "text", text }` parts, or nothing.
 * @param where - Location used in error messages, such as `messages[2]`.
 * @returns The text; parts are concatenated without a separator.
 * @throws {HttpError} 400 when the content is not text or text parts.
 */
function cacheControl(value: unknown): { type: "ephemeral"; ttl?: "5m" } | undefined {
  if (value === undefined) return undefined;
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new HttpError(400, "unsupported cache_control");
  const c = value as Record<string, unknown>;
  if (c.type !== "ephemeral" || c.ttl !== undefined && c.ttl !== "5m" || Object.keys(c).some(k => k !== "type" && k !== "ttl")) throw new HttpError(400, "unsupported cache_control");
  return { type: "ephemeral", ...(c.ttl === "5m" ? { ttl: "5m" } : {}) };
}
function contentText(content: OpenAIContent, where: string): string | TextBlock[] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) throw new HttpError(400, `${where}: content must be text`);
  const parts: TextBlock[] = content.map(part => {
    if (part?.type !== "text" || typeof part.text !== "string") throw new HttpError(400, `${where}: only text content parts are supported`);
    const cache = cacheControl((part as unknown as Record<string, unknown>).cache_control);
    return { type: "text", text: part.text, ...(cache ? { cache_control: cache } : {}) };
  });
  return parts.some(part => part.cache_control !== undefined) ? parts : parts.map(part => part.text).join("");
}
function blocks(content: string | TextBlock[]): TextBlock[] { return typeof content === "string" ? [{type:"text",text:content}] : content; }

/**
 * Rejects fields this server cannot honor.
 *
 * @param body - Parsed request body.
 * @throws {HttpError} 400 for tools, functions, `tool_choice`, or `n` other than 1.
 */
function assertSupportedFeatures(body: Record<string, unknown>): void {
  const allowed = new Set(["messages", "model", "max_tokens", "max_completion_tokens", "stream", "n", "thinking", "effort", "reasoning_effort", "cache_control", "response_format"]);
  for (const key of Object.keys(body)) if (!allowed.has(key)) throw new HttpError(400, `unsupported request field: ${key}`);
  if (body.stream !== undefined && typeof body.stream !== "boolean") throw new HttpError(400, "stream must be boolean");
  if (body.tools !== undefined || body.functions !== undefined || body.tool_choice !== undefined) {
    throw new HttpError(400, "tools and function calling are not supported");
  }
  if (body.n !== undefined && body.n !== 1) {
    throw new HttpError(400, "only n=1 is supported");
  }
}

/** System text and conversation turns split out of an OpenAI `messages` array. */
interface SplitMessages {
  readonly system: TextBlock[];
  readonly messages: Message[];
}

/**
 * Splits OpenAI messages into system text and alternating turns.
 *
 * `system` and `developer` messages are lifted out. Consecutive messages with
 * the same role are merged with a blank line between them, because the upstream
 * API requires roles to alternate.
 *
 * @param rawMessages - The request's `messages` array.
 * @returns The system texts and the merged turns.
 * @throws {HttpError} 400 for non-text content or an unsupported role.
 */
function splitMessages(rawMessages: unknown[]): SplitMessages {
  const system: TextBlock[] = [];
  const messages: Message[] = [];
  rawMessages.forEach((raw, index) => {
    const message = raw as { role?: string; content?: OpenAIContent };
    const where = `messages[${index}]`;
    const text = contentText(message?.content, where);
    if (message.role === "system" || message.role === "developer") {
      system.push(...blocks(text));
      return;
    }
    if (message.role !== "user" && message.role !== "assistant") {
      throw new HttpError(400, `${where}: unsupported role "${String(message?.role)}"`);
    }
    const last = messages[messages.length - 1];
    if (last && last.role === message.role) {
      last.content = typeof last.content === "string" && typeof text === "string" ? `${last.content}\n\n${text}` : [...blocks(last.content as string | TextBlock[]), {type:"text",text:"\n\n"}, ...blocks(text)];
      return;
    }
    messages.push({ role: message.role, content: text });
  });
  return { system, messages };
}

/**
 * Checks that the conversation starts and ends with a user turn.
 *
 * @param messages - Turns after merging.
 * @throws {HttpError} 400 when the first or last turn is not from the user.
 */
function assertUserBookends(messages: Message[]): void {
  if (messages.length === 0 || messages[0]!.role !== "user") {
    throw new HttpError(400, "the first non-system message must have role \"user\"");
  }
  if (messages[messages.length - 1]!.role !== "user") {
    throw new HttpError(400, "the last message must have role \"user\"");
  }
}

/**
 * Picks and validates the output token limit.
 *
 * Precedence is `max_completion_tokens`, then `max_tokens`, then the default.
 *
 * @param body - Parsed request body.
 * @param defaultMaxTokens - Limit used when the request names none.
 * @returns A positive integer no larger than {@link MAX_MAX_TOKENS}.
 * @throws {HttpError} 400 when the value is not an integer in range.
 */
function resolveMaxTokens(body: Record<string, unknown>, defaultMaxTokens: number): number {
  if (body.max_completion_tokens !== undefined && body.max_tokens !== undefined && body.max_completion_tokens !== body.max_tokens) throw new HttpError(400, "conflicting output token limits");
  const raw = body.max_completion_tokens ?? body.max_tokens ?? defaultMaxTokens;
  const isValid = typeof raw === "number" && Number.isInteger(raw) && raw >= 1 && raw <= MAX_MAX_TOKENS;
  if (!isValid) {
    throw new HttpError(400, `max_tokens must be an integer between 1 and ${MAX_MAX_TOKENS}`);
  }
  return raw;
}

/**
 * Picks the upstream model.
 *
 * Any non-empty model string is passed through to the upstream API so clients
 * can target any Claude model (e.g. "claude-haiku-4-5") instead of only the
 * configured default. The upstream API is the source of truth for model
 * validity; unknown names surface as an error response.
 *
 * @param requested - The request's `model` field.
 * @param defaultModel - Model used when none is supplied.
 * @returns The model id to send upstream.
 */
function resolveModel(requested: unknown, defaultModel: string): string {
  if (typeof requested === "string" && requested.trim().length > 0) {
    return requested;
  }
  if (requested !== undefined) throw new HttpError(400, "model must be a non-empty string");
  return defaultModel;
}

/**
 * Translates an OpenAI chat completion request into an upstream message request.
 *
 * Notes on fields:
 *
 * - Unsupported behavioral fields, including temperature, are rejected.
 * - Explicit models must match the configured model (optionally with [1m]).
 * - `system` and `developer` messages become the system prompt, joined by a blank line.
 *
 * @param body - Parsed JSON body of `POST /v1/chat/completions`.
 * @param defaults - Default model and token limit.
 * @returns The upstream request, without thinking or effort settings.
 * @throws {HttpError} 400 when the body is invalid or uses an unsupported feature.
 * @example
 * ```ts
 * const request = translateRequest(
 *   { model: "gpt-4o", messages: [{ role: "user", content: "Hi" }] },
 *   { defaultModel: "claude-sonnet-5-5", defaultMaxTokens: 4096 },
 * );
 * // request.model === "claude-sonnet-5-5"
 * ```
 */
export function translateRequest(
  body: Record<string, unknown>,
  defaults: Pick<OpenAIHttpDeps, "defaultModel" | "defaultMaxTokens">,
): MessageRequest {
  assertSupportedFeatures(body);
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    throw new HttpError(400, "`messages` is required and must be a non-empty array");
  }

  const { system, messages } = splitMessages(body.messages);
  assertUserBookends(messages);
  const maxTokens = resolveMaxTokens(body, defaults.defaultMaxTokens);

  const request: MessageRequest = {
    model: resolveModel(body.model, defaults.defaultModel),
    maxTokens,
    messages,
    ...(body.cache_control !== undefined ? { cacheControl: cacheControl(body.cache_control) } : {}),
  };
  if (system.length > 0) {
    request.system = system.some(block => block.cache_control !== undefined) ? system : system.map(block => block.text).join("\n\n");
  }
  const schema = responseTool(body.response_format);
  if (schema) {
    request.tools = [{name:"respond",description:"Return the requested JSON result.",input_schema:schema.schema}];
    request.toolChoice = {type:"auto"};
    request.responseTool = "respond";
    request.responseEnvelope = schema.envelope;
    const instruction = schema.envelope
      ? 'Use the respond tool exactly once. Put the requested JSON object in its required result property. Do not return the result as plain text.'
      : 'Use the respond tool exactly once to return the requested JSON result. Its input is the result. Do not return the result as plain text.';
    request.system = request.system === undefined ? instruction : typeof request.system === "string" ? request.system + "\n\n" + instruction : [...request.system,{type:"text",text:instruction}];
  }
  return request;
}

/**
 * Concatenates the text blocks of an upstream response.
 *
 * @param result - Upstream result.
 * @returns The text, with non-text blocks skipped.
 */
function responseText(result: MessageResult): string {
  return result.response.content
    .filter((block) => block.type === "text" && typeof block.text === "string")
    .map((block) => block.text)
    .join("");
}

/**
 * Converts an upstream message response to an OpenAI `chat.completion`.
 *
 * `stop_reason` maps to `finish_reason` through {@link FINISH_REASONS}; anything
 * unknown remains unknown. Missing usage remains null.
 *
 * @param result - Upstream result.
 * @param created - Creation time in epoch seconds.
 * @returns The completion body.
 * @example
 * ```ts
 * const completion = toChatCompletion(await client.createMessage(request), Math.floor(Date.now() / 1000));
 * ```
 */
export function toChatCompletion(result: MessageResult, created: number, expectedTool?: string, envelope=false): ChatCompletion {
  const { response } = result;
  const rawUsage = response.usage;
  const valid = (n: unknown): n is number => typeof n === "number" && Number.isSafeInteger(n) && n >= 0;
  const cacheCreated = rawUsage?.cache_creation_input_tokens ?? 0;
  const cacheRead = rawUsage?.cache_read_input_tokens ?? 0;
  const usage = rawUsage && valid(rawUsage.input_tokens) && valid(rawUsage.output_tokens) && valid(cacheCreated) && valid(cacheRead)
    ? { prompt_tokens: rawUsage.input_tokens + cacheCreated + cacheRead, completion_tokens: rawUsage.output_tokens, total_tokens: rawUsage.input_tokens + cacheCreated + cacheRead + rawUsage.output_tokens } : null;
  const blockTypes = response.content.map(block => block.type);
  const toolBlocks = response.content.filter(block => block.type === "tool_use");
  const tool = toolBlocks.length === 1 && toolBlocks[0]!.name === expectedTool && typeof toolBlocks[0]!.input === "object" && toolBlocks[0]!.input !== null && !Array.isArray(toolBlocks[0]!.input) ? toolBlocks[0] : undefined;
  const envelopeValid = !envelope || tool !== undefined && Object.keys(tool.input as object).length === 1 && Object.hasOwn(tool.input as object,"result") && typeof (tool.input as Record<string,unknown>).result === "object" && (tool.input as Record<string,unknown>).result !== null && !Array.isArray((tool.input as Record<string,unknown>).result);
  const validTool = envelopeValid && tool !== undefined && response.stop_reason === "tool_use" && blockTypes.every(type=>["text","thinking","redacted_thinking","tool_use"].includes(type));
  const unexpected = blockTypes.some(type => !["text", "thinking", "redacted_thinking"].includes(type));
  return {
    id: `chatcmpl-${response.id}`,
    object: "chat.completion",
    created,
    model: response.model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: expectedTool ? validTool ? JSON.stringify(envelope ? (tool!.input as Record<string,unknown>).result : tool!.input) : "" : responseText(result) },
        finish_reason: expectedTool ? validTool ? "stop" : "unsupported_content" : unexpected ? "unsupported_content" : FINISH_REASONS[response.stop_reason ?? ""] ?? DEFAULT_FINISH_REASON,
      },
    ],
    usage,
    upstream: { request_id: result.requestId, stop_reason: response.stop_reason, block_types: blockTypes, usage: rawUsage ?? null },
  };
}

/**
 * Refuses to bind beyond loopback unless an API key is set.
 *
 * This server spends the signed-in subscription, so it must not be reachable
 * from the network without authentication.
 *
 * @param host - Address the server will bind to.
 * @param apiKey - Configured API key, if any.
 * @throws {Error} When `host` is not a loopback address and `apiKey` is empty.
 * @example
 * ```ts
 * assertSafeBind("0.0.0.0", undefined); // throws
 * assertSafeBind("0.0.0.0", "secret"); // ok
 * ```
 */
export function assertSafeBind(host: string, apiKey: string | undefined): void {
  if (!LOOPBACK_HOSTS.has(host) && !apiKey) {
    throw new Error(`CLAUDE_WIRE_HOST=${host} exposes your subscription beyond this machine; set CLAUDE_WIRE_API_KEY first`);
  }
}

/**
 * Extracts the path from a request target.
 *
 * @param target - Raw `incomingRequest.url`.
 * @returns The path, or `/` when the target is missing or malformed. A
 *   malformed target therefore falls through to the 404 route.
 */
function pathOf(target: string | undefined): string {
  try {
    return new URL(target ?? "/", "http://localhost").pathname;
  } catch {
    return "/";
  }
}

/**
 * Chooses the log level for a finished response.
 *
 * @param status - HTTP status code.
 * @returns `error` for 5xx, `warn` for 4xx, `info` otherwise.
 */
function logLevelFor(status: number): "error" | "warn" | "info" {
  if (status >= SERVER_ERROR_STATUS) {
    return "error";
  }
  if (status >= CLIENT_ERROR_STATUS) {
    return "warn";
  }
  return "info";
}

/**
 * Maps an upstream error to the status sent to the client.
 *
 * @param err - Error from the upstream client.
 * @returns The upstream status when it is a valid error status, otherwise 502.
 */
function relayedStatus(err: APIStatusError): number {
  const isErrorStatus = err.status >= CLIENT_ERROR_STATUS && err.status <= MAX_ERROR_STATUS;
  if (isErrorStatus) {
    return err.status;
  }
  return BAD_GATEWAY_STATUS;
}

/**
 * Answers a request that failed, if the response has not started.
 *
 * If headers were already sent the connection is simply ended.
 *
 * @param response - Response to write.
 * @param err - Whatever was thrown while handling the request.
 */
function sendFailure(response: ServerResponse, err: unknown): void {
  if (response.headersSent) {
    const type = err instanceof APIStatusError ? err.errorType ?? "api_error" : "api_error";
    sendSseError(response, (err as Error)?.message ?? "upstream request failed", type);
    return;
  }
  if (err instanceof HttpError) {
    sendError(response, err.status, err.message, err.type);
    return;
  }
  if (err instanceof APIStatusError) {
    sendError(response, relayedStatus(err), err.message, err.errorType ?? "api_error");
    return;
  }
  sendError(response, BAD_GATEWAY_STATUS, (err as Error).message, "api_error");
}

/**
 * Builds the upstream request for a chat completion call.
 *
 * @param body - Parsed request body.
 * @param deps - Server settings.
 * @returns The translated request with validated thinking and effort overrides, falling back to server settings.
 * @throws {HttpError} 400 when the body is invalid.
 */
function buildUpstreamRequest(body: Record<string, unknown>, deps: OpenAIHttpDeps): MessageRequest {
  const request = translateRequest(body, deps);
  if (body.thinking !== undefined) {
    const thinking = body.thinking;
    if (typeof thinking !== "object" || thinking === null || Array.isArray(thinking)
      || Object.keys(thinking).length !== 1 || !("type" in thinking)
      || (thinking.type !== "adaptive" && thinking.type !== "between_tools")) {
      throw new HttpError(400, "thinking must be an object with only type: adaptive or between_tools");
    }
    request.thinking = { type: thinking.type };
  } else if (deps.thinking && !isHaikuModel(request.model)) {
    request.thinking = deps.thinking;
  }
  for (const field of ["effort", "reasoning_effort"] as const) {
    const value = body[field];
    if (value !== undefined && (typeof value !== "string" || !isEffort(value))) {
      throw new HttpError(400, `${field} must be one of ${EFFORTS.join(", ")}`);
    }
  }
  if (body.effort !== undefined && body.reasoning_effort !== undefined && body.effort !== body.reasoning_effort) {
    throw new HttpError(400, "effort and reasoning_effort must agree when both are supplied");
  }
  const effort = body.effort ?? body.reasoning_effort ?? deps.effort;
  if (effort !== undefined) {
    request.effort = effort as Effort;
  }
  return request;
}

/**
 * Streams a chat completion as OpenAI `chat.completion.chunk` SSE.
 *
 * The upstream Anthropic SSE events are translated live by {@link translateSSE}
 * and written to the response. The shared abort signal is aborted when the
 * client disconnects, which cancels the upstream call.
 *
 * @param request - Upstream request, already carrying `stream: true`.
 * @param response - Response to write.
 * @param deps - Server settings.
 * @param abort - Shared abort controller, aborted on client disconnect.
 * @param now - Clock in epoch seconds.
 * @param detail - Log fields, mutated in place.
 */
async function handleStream(
  request: MessageRequest,
  response: ServerResponse,
  deps: OpenAIHttpDeps,
  abort: AbortController,
  now: () => number,
  detail: RequestLogDetail,
): Promise<void> {
  detail.upstreamAttempts = 1;
  response.setHeader("x-claude-wire-upstream-attempts", "1");
  response.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    "connection": "keep-alive",
  });
  const completionId = randomUUID();
  try {
    for await (const line of translateSSE(
      deps.client.createMessageStream(request, abort.signal),
      completionId,
      request.model,
      now(),
    )) {
      response.write(line);
      if (!response.write("")) {
        await once(response, "drain");
      }
    }
  } catch (err) {
    sendFailure(response, err);
  } finally {
    if (!response.writableEnded) {
      response.end();
    }
  }
}

/**
 * Handles `POST /v1/chat/completions`.
 *
 * The upstream call is cancelled if the client disconnects before the response
 * is finished. Metadata is added to `detail` for the request log.
 *
 * @param incomingRequest - Incoming request.
 * @param response - Response to write.
 * @param deps - Server settings.
 * @param now - Clock in epoch seconds.
 * @param detail - Log fields, mutated in place.
 */
async function handleChatCompletion(
  incomingRequest: IncomingMessage,
  response: ServerResponse,
  deps: OpenAIHttpDeps,
  now: () => number,
  detail: RequestLogDetail,
): Promise<void> {
  const body = await readJson(incomingRequest);
  const request = buildUpstreamRequest(body, deps);
  const isStream = body.stream === true;
  detail.model = request.model;
  detail.stream = isStream;
  detail.thinking = request.thinking?.type ?? "default";
  detail.effort = request.effort ?? "default";
  detail.messages = request.messages.length;

  const abort = new AbortController();
  response.on("close", () => {
    if (!response.writableEnded) {
      abort.abort();
    }
  });

  if (isStream) {
    return handleStream(request, response, deps, abort, now, detail);
  }

  const singleAttempt = incomingRequest.headers["x-pinkie-single-attempt"] === "1";
  let upstreamAttempts = 0;
  const result = singleAttempt
    ? await deps.client.createMessage(request, abort.signal, { singleAttempt: true, onAttempt: () => {
        upstreamAttempts++;
        detail.upstreamAttempts = upstreamAttempts;
        response.setHeader("x-claude-wire-upstream-attempts", String(upstreamAttempts));
      } })
    : await deps.client.createMessage(request, abort.signal);
  const completion = toChatCompletion(result, now(), request.responseTool, request.responseEnvelope);

  detail.finish = completion.choices[0]!.finish_reason;
  detail.tokens = completion.usage ? `${completion.usage.prompt_tokens} in / ${completion.usage.completion_tokens} out` : "unknown";
  sendJson(response, 200, completion);
}

/**
 * Handles `GET /v1/models` with the single configured model.
 *
 * @param response - Response to write.
 * @param deps - Server settings.
 */
function handleModels(response: ServerResponse, deps: OpenAIHttpDeps): void {
  sendJson(response, 200, {
    object: "list",
    data: [{ id: deps.defaultModel, object: "model", created: 0, owned_by: "anthropic" }],
  });
}

/**
 * Routes one request. Any thrown error is turned into a response by the caller.
 *
 * Order matters: `/health` is answered before the API key check so liveness
 * probes need no credentials.
 *
 * @param incomingRequest - Incoming request.
 * @param response - Response to write.
 * @param path - Path of the request.
 * @param deps - Server settings.
 * @param now - Clock in epoch seconds.
 * @param detail - Log fields, mutated in place.
 * @throws {HttpError} 401 for a bad API key, 404 for an unknown route, 405 for a wrong method.
 */
async function route(
  incomingRequest: IncomingMessage,
  response: ServerResponse,
  path: string,
  deps: OpenAIHttpDeps,
  now: () => number,
  detail: RequestLogDetail,
): Promise<void> {
  if (incomingRequest.method === "GET" && path === HEALTH_PATH) {
    sendJson(response, 200, { status: "ok" });
    return;
  }
  if (deps.apiKey && !isApiKeyValid(incomingRequest.headers.authorization, deps.apiKey)) {
    throw new HttpError(401, "invalid or missing API key", "authentication_error");
  }
  if (incomingRequest.method === "GET" && path === MODELS_PATH) {
    handleModels(response, deps);
    return;
  }
  if (path !== CHAT_COMPLETIONS_PATH) {
    throw new HttpError(404, `no route for ${incomingRequest.method} ${path}`, "not_found_error");
  }
  if (incomingRequest.method !== "POST") {
    throw new HttpError(405, "use POST");
  }
  await handleChatCompletion(incomingRequest, response, deps, now, detail);
}

/**
 * Creates the OpenAI compatible HTTP server. It is not listening yet.
 *
 * Routes: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`. Every
 * response is logged once, when it finishes, with method, path, status, model,
 * token counts and duration. Prompts and replies are never logged.
 *
 * @param deps - See {@link OpenAIHttpDeps}.
 * @returns A Node HTTP server; call `listen` on it.
 * @example
 * ```ts
 * const server = createOpenAIHttpServer({
 *   client,
 *   defaultModel: "claude-sonnet-5-5",
 *   defaultMaxTokens: 4096,
 *   apiKey: "secret",
 * });
 * server.listen(8787, "127.0.0.1");
 * ```
 */
export function createOpenAIHttpServer(deps: OpenAIHttpDeps): Server {
  const now = deps.now ?? (() => Math.floor(Date.now() / 1000));
  const log = deps.log ?? SILENT_LOG;

  return createServer(async (incomingRequest, response) => {
    const started = Date.now();
    const method = incomingRequest.method ?? "?";
    const path = pathOf(incomingRequest.url);
    // Request metadata only: prompts and replies are never logged.
    const detail: RequestLogDetail = {};
    response.on("finish", () => {
      const status = response.statusCode;
      log[logLevelFor(status)]({ ...detail, ms: Date.now() - started }, `${method} ${path} ${status}`);
    });
    try {
      await route(incomingRequest, response, path, deps, now, detail);
    } catch (err) {
      detail.error = (err as Error).message;
      sendFailure(response, err);
    }
  });
}

/**
 * Reads `CLAUDE_WIRE_THINKING`.
 *
 * - `off` (the default) sends `thinking: { type: "between_tools" }`. Sonnet 5.5
 *   rejects `thinking: disabled`; `between_tools` is its "off", and it needs an
 *   effort of `high` or lower.
 * - `adaptive` sends `thinking: { type: "adaptive" }`.
 * - `default` sends nothing and lets the model decide.
 *
 * @param raw - Raw environment value. Case and surrounding space are ignored.
 * @returns The thinking setting, or undefined for `default`.
 * @throws {Error} When the value is not one of the accepted names.
 * @example
 * ```ts
 * thinkingFromEnv(undefined); // { type: "between_tools" }
 * thinkingFromEnv("default"); // undefined
 * ```
 */
export function thinkingFromEnv(raw: string | undefined): MessageRequest["thinking"] {
  const thinkingValue = raw?.trim().toLowerCase() || DEFAULT_THINKING;
  if (thinkingValue === "off") {
    return { type: "between_tools" };
  }
  if (thinkingValue === "adaptive") {
    return { type: "adaptive" };
  }
  if (thinkingValue === "default") {
    return undefined;
  }
  throw new Error(`CLAUDE_WIRE_THINKING must be off, adaptive or default, got: "${raw}"`);
}

/**
 * Checks whether a string is an accepted effort level.
 *
 * @param candidate - Candidate value.
 * @returns True when `candidate` is one of {@link EFFORTS}.
 */
function isEffort(candidate: string): candidate is Effort {
  return (EFFORTS as readonly string[]).includes(candidate);
}

/**
 * Reads `CLAUDE_WIRE_EFFORT`.
 *
 * @param raw - Raw environment value. Case and surrounding space are ignored.
 * @returns The effort level, or undefined when the variable is unset or blank.
 * @throws {Error} When the value is not `low`, `medium`, `high`, `xhigh` or `max`.
 * @example
 * ```ts
 * effortFromEnv("HIGH"); // "high"
 * effortFromEnv(undefined); // undefined
 * ```
 */
export function effortFromEnv(raw: string | undefined): MessageRequest["effort"] {
  const effortValue = raw?.trim().toLowerCase();
  if (!effortValue) {
    return undefined;
  }
  if (!isEffort(effortValue)) {
    throw new Error(`CLAUDE_WIRE_EFFORT must be one of ${EFFORTS.join(", ")}, got: "${raw}"`);
  }
  return effortValue;
}

/**
 * Reads `CLAUDE_WIRE_PORT`.
 *
 * @param raw - Raw environment value.
 * @returns The port, or {@link DEFAULT_PORT} when unset or blank. Port 0 picks a free port.
 * @throws {Error} When the value is not an integer from 0 to 65535.
 */
function portFromEnv(raw: string | undefined): number {
  const trimmed = raw?.trim();
  if (!trimmed) {
    return DEFAULT_PORT;
  }
  const port = Number(trimmed);
  if (!Number.isInteger(port) || port < 0 || port > MAX_PORT) {
    throw new Error(`CLAUDE_WIRE_PORT must be a port number, got: "${trimmed}"`);
  }
  return port;
}

/**
 * Reads `CLAUDE_WIRE_API_KEY`.
 *
 * @param env - Environment to read from.
 * @returns The trimmed key, or undefined when unset or blank.
 */
function apiKeyFromEnv(env: NodeJS.ProcessEnv): string | undefined {
  return env.CLAUDE_WIRE_API_KEY?.trim() || undefined;
}

/**
 * Creates the console logger used by the executable.
 *
 * @param env - Environment supplying `CLAUDE_WIRE_LOG_LEVEL` and `NO_COLOR`.
 * @returns A pino logger writing human readable lines.
 */
function createLogger(env: NodeJS.ProcessEnv): Logger {
  return pino(
    { level: env.CLAUDE_WIRE_LOG_LEVEL?.trim() || DEFAULT_LOG_LEVEL },
    pretty({ colorize: env.NO_COLOR === undefined, translateTime: "HH:MM:ss", ignore: "pid,hostname", sync: true }),
  );
}

/**
 * Starts the server and installs signal handlers for a clean shutdown.
 *
 * Validation order is: port, credentials and defaults, thinking and effort, then
 * the bind safety check. The server binds to loopback by default because it
 * spends the signed-in subscription and must not be reachable from the LAN.
 *
 * @throws {Error} When any setting is invalid or the port cannot be bound.
 */
async function main(): Promise<void> {
  const env = process.env;
  const port = portFromEnv(env.CLAUDE_WIRE_PORT);
  const base = depsFromEnv(env);
  const log = createLogger(env);
  const server = createOpenAIHttpServer({
    log,
    client: base.client,
    defaultModel: base.defaultModel,
    defaultMaxTokens: base.defaultMaxTokens,
    apiKey: apiKeyFromEnv(env),
    thinking: thinkingFromEnv(env.CLAUDE_WIRE_THINKING),
    effort: effortFromEnv(env.CLAUDE_WIRE_EFFORT),
  });
  const host = env.CLAUDE_WIRE_HOST?.trim() || DEFAULT_HOST;
  assertSafeBind(host, apiKeyFromEnv(env));
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, resolve);
  });
  const shutdown = (signal: string): void => {
    log.info(`${signal} received, shutting down`);
    server.close(() => process.exit(0));
    server.closeAllConnections();
  };
  process.once("SIGINT", () => shutdown("SIGINT"));
  process.once("SIGTERM", () => shutdown("SIGTERM"));
  log.info(`OpenAI-compatible server on http://${host}:${port}/v1 (default model ${base.defaultModel})`);
}

/**
 * Tells whether this module is the process entry point.
 *
 * Symlinks are resolved on both sides, so running through an npm bin link works.
 *
 * @returns True when started directly, false when imported (for example by tests).
 */
function isDirectRun(): boolean {
  const entry = process.argv[1];
  if (!entry) {
    return false;
  }
  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (isDirectRun()) {
  main().catch((err) => {
    process.stderr.write(`[local-claude-router] fatal: ${(err as Error).message}\n`);
    process.exit(1);
  });
}
