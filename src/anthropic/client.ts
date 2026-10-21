import { randomUUID } from "node:crypto";
import type { CredentialManager } from "./credentials.js";
import { dynamicBillingBlock } from "./billing.js";
import { buildHeaders } from "./headers.js";
import type { FetchLike } from "./tokens.js";
import { BASE_API_URL, IDENTITY_SYSTEM } from "./wire.js";

/** A text content block. */
export interface TextBlock {
  [key: string]: unknown;
  type: "text";
  text: string;
  cache_control?: { type: "ephemeral"; ttl?: "5m" };
}

/** A content block of any type. Only `text` blocks are read by this server. */
export interface ContentBlock {
  type: string;
  text?: string;
  [k: string]: unknown;
}

/** One conversation turn. */
export interface Message {
  role: "user" | "assistant";
  content: string | ContentBlock[];
}

/** A request for one assistant message, in the shape this client accepts. */
export interface MessageRequest {
  responseTool?: string;
  responseEnvelope?: boolean;
  tools?: {name: string; description: string; input_schema: Record<string, unknown>}[];
  toolChoice?: {type: "auto"};
  cacheControl?: { type: "ephemeral"; ttl?: "5m" };
  model: string;
  maxTokens: number;
  messages: Message[];
  /** System prompt text or blocks. Sent after the billing and identity blocks. */
  system?: string | TextBlock[];
  /**
   * Sampling temperature. Forwarded when set, but current models reject it as
   * deprecated (HTTP 400), so the HTTP layer never sets it.
   */
  temperature?: number;
  /**
   * Thinking mode. `between_tools` is the "off" setting for Sonnet 5.5, which
   * rejects `thinking: disabled`.
   */
  thinking?: { type: "adaptive" | "between_tools" };
  /** Effort level, sent as `output_config.effort`. */
  effort?: "low" | "medium" | "high" | "xhigh" | "max";
  /** When true, the upstream call is a streaming request. */
  stream?: boolean;
}

/** The subset of the upstream message response that this server reads. */
export interface MessageResponse {
  id: string;
  type: string;
  role: string;
  model: string;
  content: ContentBlock[];
  stop_reason: string | null;
  usage?: { input_tokens: number; output_tokens: number; [k: string]: unknown };
}

/** A decoded response and the upstream request id, for logs and support. */
export interface MessageResult {
  response: MessageResponse;
  requestId: string | null;
}

/** One decoded server-sent event from a streaming upstream response. */
export interface SSEEvent {
  readonly event: string;
  readonly data: string;
}

/** Non-2xx answer from the API. */
export class APIStatusError extends Error {
  /**
   * @param message - Description including status and upstream detail.
   * @param status - HTTP status code.
   * @param errorType - Upstream `error.type`, when the body carried one.
   * @param retryAfterMs - Server suggested wait in milliseconds, or 0 when none.
   * @param requestId - Upstream `request-id` header, when present.
   */
  public constructor(
    message: string,
    public readonly status: number,
    public readonly errorType: string | undefined,
    public readonly retryAfterMs: number,
    public readonly requestId: string | null,
  ) {
    super(message);
    this.name = "APIStatusError";
  }
}

/** The part of {@link CredentialManager} that the client needs. */
type Credentials = Pick<CredentialManager, "get" | "refreshAfter401">;

/** Credentials as returned by {@link CredentialManager.get}. */
type ResolvedCreds = Awaited<ReturnType<Credentials["get"]>>;

/** Options for {@link AnthropicClient}. Optional fields exist so tests can inject fakes. */
export interface ClientOptions {
  readonly credentials: Credentials;
  readonly fetch?: FetchLike;
  /** API origin. Defaults to the first-party API. Trailing slashes are removed. */
  readonly baseUrl?: string;
  /** Session id sent with every request. */
  readonly sessionId?: string;
  /** Waits for the given milliseconds between retries. */
  readonly sleep?: (ms: number) => Promise<void>;
  /** Random source in [0, 1) for retry jitter. */
  readonly random?: () => number;
  /** Retries after the first attempt. Defaults to 5. */
  readonly maxRetries?: number;
  /** Per request timeout. Defaults to five minutes. */
  readonly requestTimeoutMs?: number;
}

/** Backoff base delay. */
const RETRY_BASE_MS = 1_000;

/** Upper bound of the exponential part of the backoff. */
const RETRY_MAX_MS = 32_000;

/** Longest wait accepted between retries; a longer suggestion fails the call instead. */
const MAX_RETRY_WAIT_MS = 120_000;

/** Largest delay `setTimeout` accepts (2^31 - 1), so larger server hints are ignored. */
const MAX_RETRY_AFTER_MS = 2_147_483_647;

/** Relative jitter applied to the backoff: plus or minus 25 percent. */
const RETRY_JITTER_FRACTION = 0.5;

/** Retries after the first attempt, when not configured. */
const DEFAULT_MAX_RETRIES = 5;

/** Per request timeout, when not configured (five minutes). */
const DEFAULT_TIMEOUT_MS = 5 * 60_000;

/** Most bytes of an error body that are read (1 MiB). */
const MAX_ERROR_BODY_CHARS = 1 << 20;

/** Model id suffix that selects the 1M context variant; the API itself does not know it. */
const CONTEXT_1M_SUFFIX = "[1m]";

/** Milliseconds per second. */
const MS_PER_SECOND = 1000;

/**
 * Reads how long the server asks the client to wait before retrying.
 *
 * Checks `retry-after-ms`, then `retry-after` as seconds, then `retry-after` as
 * an HTTP date. Values that are not positive, or that exceed what `setTimeout`
 * accepts, are ignored.
 *
 * @param headers - Response headers.
 * @param nowMs - Current time in epoch milliseconds, for HTTP date values.
 * @returns The wait in milliseconds, or 0 when there is no usable hint.
 */
export function retryAfterMs(headers: Headers, nowMs: number = Date.now()): number {
  const ms = Number.parseFloat((headers.get("retry-after-ms") ?? "").trim());
  if (Number.isFinite(ms) && ms > 0 && ms <= MAX_RETRY_AFTER_MS) {
    return ms;
  }
  const raw = (headers.get("retry-after") ?? "").trim();
  if (raw === "") {
    return 0;
  }
  const secs = Number(raw);
  if (Number.isFinite(secs) && secs > 0) {
    const fromSeconds = secs * MS_PER_SECOND;
    if (fromSeconds <= MAX_RETRY_AFTER_MS) {
      return fromSeconds;
    }
    return 0;
  }
  const date = Date.parse(raw);
  if (Number.isFinite(date) && date > nowMs) {
    return date - nowMs;
  }
  return 0;
}

/**
 * Decides whether a failed response should be retried.
 *
 * An explicit `x-should-retry` header wins. Otherwise timeouts (408), conflicts
 * (409), rate limits (429) and server errors (5xx) are retried.
 *
 * @param status - HTTP status code.
 * @param header - Value of the `x-should-retry` header, if any.
 * @returns True when the request should be sent again.
 */
export function shouldRetryStatus(status: number, header: string | null): boolean {
  const normalizedHeader = (header ?? "").trim().toLowerCase();
  if (normalizedHeader === "true") {
    return true;
  }
  if (normalizedHeader === "false") {
    return false;
  }
  return status === 408 || status === 409 || status === 429 || status >= 500;
}

/**
 * Computes the wait before the next attempt.
 *
 * A server hint is used as is. Otherwise the delay doubles per attempt up to
 * {@link RETRY_MAX_MS}, with plus or minus 25 percent jitter.
 *
 * @param attempt - Zero based index of the attempt that just failed.
 * @param hintMs - Server suggested wait, or 0.
 * @param random - Random source in [0, 1).
 * @returns The wait in milliseconds, never negative.
 */
export function backoffMs(attempt: number, hintMs: number, random: () => number = Math.random): number {
  if (hintMs > 0) {
    return hintMs;
  }
  const base = Math.min(RETRY_BASE_MS * 2 ** Math.max(attempt, 0), RETRY_MAX_MS);
  const jitter = (random() * RETRY_JITTER_FRACTION - RETRY_JITTER_FRACTION / 2) * base;
  return Math.max(base + jitter, 0);
}

/**
 * Extracts the text of the first user message, which seeds the billing suffix.
 *
 * @param messages - Conversation turns.
 * @returns The concatenated text blocks, or an empty string when there is no user turn.
 */
function firstUserText(messages: Message[]): string {
  const first = messages.find((message) => message.role === "user");
  if (!first) {
    return "";
  }
  if (typeof first.content === "string") {
    return first.content;
  }
  return first.content
    .filter((block) => block.type === "text" && typeof block.text === "string")
    .map((block) => block.text)
    .join("");
}

/**
 * Normalizes a system prompt to a list of text blocks.
 *
 * @param system - Prompt text, blocks, or nothing.
 * @returns Zero or more blocks.
 */
function systemBlocks(system: MessageRequest["system"]): TextBlock[] {
  if (system === undefined || system === "") {
    return [];
  }
  if (typeof system === "string") {
    return [{ type: "text", text: system }];
  }
  return system;
}

/**
 * Removes the `[1m]` marker from a model id.
 *
 * The suffix is how callers ask for the 1M context beta; the wire model id
 * does not include it.
 *
 * @param model - Model id as given by the caller.
 * @returns The id to send in the request body.
 */
function toWireModel(model: string): string {
  if (model.toLowerCase().endsWith(CONTEXT_1M_SUFFIX)) {
    return model.slice(0, -CONTEXT_1M_SUFFIX.length);
  }
  return model;
}

/**
 * Serializes a request into the JSON body of `/v1/messages`.
 *
 * The system prompt is the billing block, then the identity line, then the
 * caller's system text. `temperature`, `thinking` and `effort` are included
 * only when set.
 *
 * @param request - The request.
 * @param promptId - UUID for this prompt, embedded in the billing block.
 * @returns The JSON text.
 */
function buildRequestBody(request: MessageRequest, promptId: string): string {
  const body: Record<string, unknown> = {
    model: toWireModel(request.model),
    max_tokens: request.maxTokens,
    system: [
      dynamicBillingBlock(firstUserText(request.messages), { promptId }),
      { type: "text", text: IDENTITY_SYSTEM },
      ...systemBlocks(request.system),
    ],
    messages: request.messages,
    ...(request.stream ? { stream: true } : {}),
    ...(request.tools ? {tools:request.tools,tool_choice:request.toolChoice} : {}),
    ...(request.cacheControl ? { cache_control: request.cacheControl } : {}),
  };
  if (request.temperature !== undefined) {
    body.temperature = request.temperature;
  }
  if (request.thinking) {
    body.thinking = request.thinking;
  }
  if (request.effort) {
    body.output_config = { effort: request.effort };
  }
  return JSON.stringify(body);
}

/**
 * Converts a failed response into an {@link APIStatusError}.
 *
 * Reads at most 1 MiB of the body. When it is the standard error envelope the
 * `type` and `message` are surfaced; otherwise the raw text is kept.
 *
 * @param httpResponse - A non-2xx response. Its body is consumed.
 * @returns The error, carrying the retry hint and request id.
 */
async function toStatusError(httpResponse: Response): Promise<APIStatusError> {
  const raw = (await httpResponse.text().catch(() => "")).slice(0, MAX_ERROR_BODY_CHARS);
  let type: string | undefined;
  let detail = raw.trim();
  try {
    const envelope = JSON.parse(raw) as { error?: { type?: string; message?: string } };
    if (envelope.error?.type) {
      type = envelope.error.type;
      detail = `${envelope.error.type}: ${envelope.error.message ?? ""}`;
    }
  } catch {
    // body was not JSON; keep the raw text
  }
  return new APIStatusError(
    `api: ${httpResponse.status} ${httpResponse.statusText}: ${detail}`,
    httpResponse.status,
    type,
    retryAfterMs(httpResponse.headers),
    httpResponse.headers.get("request-id"),
  );
}

/**
 * Decodes a response body into server-sent events.
 *
 * Each event is a block of `event:` and `data:` lines separated by a blank
 * line. `data:` lines are yielded as-is; multiple `data:` lines in one event
 * are not expected by the upstream API and are concatenated with a newline.
 *
 * @param body - A readable stream of UTF-8 text, or a string.
 * @returns An async iterable of parsed events.
 */
async function* parseSSEStream(
  body: ReadableStream<Uint8Array> | string,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const decoder = new TextDecoder();
  let buffer = "";

  const feed = (chunk: Uint8Array): void => {
    buffer += decoder.decode(chunk, { stream: true });
  };

  const parseBlock = (block: string): SSEEvent => {
    let event = "";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) {
        event = line.slice("event: ".length);
      } else if (line.startsWith("data: ")) {
        data = data === "" ? line.slice("data: ".length) : `${data}\n${line.slice("data: ".length)}`;
      }
    }
    return { event, data };
  };

  const emit = async function* (): AsyncGenerator<SSEEvent> {
    while (true) {
      const separatorIndex = buffer.indexOf("\n\n");
      if (separatorIndex === -1) break;
      yield parseBlock(buffer.slice(0, separatorIndex));
      buffer = buffer.slice(separatorIndex + 2);
    }
    const remaining = buffer.trim();
    buffer = "";
    if (remaining !== "") {
      yield parseBlock(remaining);
    }
  };

  if (typeof body === "string") {
    buffer = body;
    yield* emit();
    return;
  }

  const reader = body.getReader();
  try {
    for (;;) {
      const read = reader.read();
      const aborted = signal
        ? new Promise<{ value?: Uint8Array; done: boolean }>((_, reject) => {
            if (signal.aborted) return reject(new Error("request cancelled by the caller"));
            signal.addEventListener("abort", () => reject(new Error("request cancelled by the caller")), { once: true });
          })
        : undefined;
      const result = aborted ? await Promise.race([read, aborted]) : await read;
      if (result.done) break;
      if (result.value) feed(result.value);
      yield* emit();
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Decides whether to authenticate with the API key rather than the access token.
 *
 * @param credentials - Resolved credentials.
 * @returns True for console accounts, or when there is no access token.
 */
function shouldUseApiKey(credentials: ResolvedCreds): boolean {
  return credentials.apiKey !== undefined && (credentials.accountKind === "anthropic-console" || !credentials.accessToken);
}

/**
 * Combines the caller's abort signal with the timeout.
 *
 * @param signal - Caller's signal, if any.
 * @param timeoutMs - Timeout in milliseconds.
 * @returns A signal that aborts on whichever fires first.
 */
function withTimeout(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs);
  if (!signal) {
    return timeout;
  }
  return AbortSignal.any([signal, timeout]);
}

/**
 * Minimal client for the Anthropic Messages API, shaped like Claude Code.
 *
 * It signs requests with the credentials from a {@link CredentialManager},
 * retries transient failures with backoff, and retries once with refreshed
 * credentials after a 401.
 */
export interface ClientCallOptions {
  readonly singleAttempt?: boolean;
  readonly onAttempt?: () => void;
}

export class AnthropicClient {
  private readonly fetchImpl: FetchLike;
  private readonly baseUrl: string;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly random: () => number;
  private readonly maxRetries: number;
  private readonly timeoutMs: number;

  /**
   * @param options - See {@link ClientOptions}.
   */
  public constructor(private readonly options: ClientOptions) {
    this.fetchImpl = options.fetch ?? fetch;
    this.baseUrl = (options.baseUrl ?? BASE_API_URL).replace(/\/+$/, "");
    this.sleep = options.sleep ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    this.random = options.random ?? Math.random;
    this.maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
    this.timeoutMs = options.requestTimeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  /**
   * Sends a message request and returns the decoded response.
   *
   * Transient failures (see {@link shouldRetryStatus}) are retried. A 401 is
   * answered by refreshing credentials and sending the request once more.
   *
   * @param request - The request.
   * @param signal - Aborts the call; an abort surfaces as `request cancelled by the caller`.
   * @returns The response and the upstream request id.
   * @throws {APIStatusError} When the API answers with a non-2xx status after retries.
   * @throws {Error} On network failure, cancellation, or an undecodable body.
   * @example
   * ```ts
   * const { response } = await client.createMessage({
   *   model: "claude-sonnet-5-5",
   *   maxTokens: 256,
   *   messages: [{ role: "user", content: "Hello" }],
   * });
   * ```
   */
  public async createMessage(request: MessageRequest, signal?: AbortSignal, callOptions: ClientCallOptions = {}): Promise<MessageResult> {
    const promptId = randomUUID();
    const body = buildRequestBody(request, promptId);

    let credentials = await this.options.credentials.get();
    let httpResponse = await this.sendWithRetry(request.model, body, credentials, promptId, signal, callOptions);
    if (httpResponse.status === 401 && !callOptions.singleAttempt) {
      await httpResponse.body?.cancel();
      credentials = await this.options.credentials.refreshAfter401(credentials.accessToken || credentials.apiKey || "");
      httpResponse = await this.sendWithRetry(request.model, body, credentials, promptId, signal, callOptions);
    }
    if (!httpResponse.ok) {
      throw await toStatusError(httpResponse);
    }

    let response: MessageResponse;
    try {
      response = (await httpResponse.json()) as MessageResponse;
    } catch (err) {
      throw new Error(`api: decode response: ${(err as Error).message}`);
    }
    return { response, requestId: httpResponse.headers.get("request-id") };
  }

  /**
   * Sends a message request as a stream and yields the upstream SSE events.
   *
   * Streaming calls are not retried: a failure would lose the stream, so the
   * first response is final. A 401 is answered by refreshing credentials and
   * sending the request once more. The caller must iterate the returned
   * generator to completion so the connection is released.
   *
   * @param request - The request, with `stream` set to `true`.
   * @param signal - Aborts the call; an abort surfaces as `request cancelled`.
   * @returns An async iterable of upstream SSE events.
   * @throws {APIStatusError} When the API answers with a non-2xx status.
   */
  public async *createMessageStream(
    request: MessageRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<SSEEvent> {
    if (signal?.aborted) {
      throw new Error("request cancelled by the caller");
    }
    const promptId = randomUUID();
    const body = buildRequestBody({ ...request, stream: true }, promptId);

    let credentials = await this.options.credentials.get();
    let httpResponse = await this.sendOnce(request.model, body, credentials, promptId, signal);
    if (httpResponse.status === 401) {
      await httpResponse.body?.cancel();
      credentials = await this.options.credentials.refreshAfter401(credentials.accessToken || credentials.apiKey || "");
      httpResponse = await this.sendOnce(request.model, body, credentials, promptId, signal);
    }
    if (!httpResponse.ok) {
      throw await toStatusError(httpResponse);
    }
    if (!httpResponse.body) {
      throw new Error("api: streaming response has no body");
    }
    yield* parseSSEStream(httpResponse.body, signal);
  }

  /**
   * Sends one HTTP request with no retry.
   *
   * @param model - Model id, used to pick beta flags.
   * @param body - Serialized request body.
   * @param credentials - Credentials to sign with.
   * @param promptId - UUID for the gateway hint headers.
   * @param signal - Caller's abort signal.
   * @returns The raw response, whatever its status.
   */
  private async sendOnce(
    model: string,
    body: string,
    credentials: ResolvedCreds,
    promptId: string,
    signal?: AbortSignal,
  ): Promise<Response> {
    let accessToken: string | undefined = credentials.accessToken;
    let apiKey: string | undefined;
    if (shouldUseApiKey(credentials)) {
      accessToken = undefined;
      apiKey = credentials.apiKey;
    }
    const headers = buildHeaders({
      baseUrl: this.baseUrl,
      model,
      accessToken,
      apiKey,
      sessionId: this.options.sessionId,
      hints: { requestClass: "main", promptId },
    });
    return this.fetchImpl(`${this.baseUrl}/v1/messages?beta=true`, {
      method: "POST",
      headers,
      body,
      signal: withTimeout(signal, this.timeoutMs),
    });
  }

  /**
   * Sends a request, retrying transient failures with backoff.
   *
   * A 401 is returned to the caller untouched so it can refresh credentials. A
   * wait longer than {@link MAX_RETRY_WAIT_MS} gives up instead of sleeping.
   *
   * @param model - Model id, used to pick beta flags.
   * @param body - Serialized request body.
   * @param credentials - Credentials to sign with.
   * @param promptId - UUID for the gateway hint headers.
   * @param signal - Caller's abort signal.
   * @returns The first response that is successful, final, or a 401.
   * @throws {Error} The last failure once retries are exhausted or the caller aborts.
   */
  private async sendWithRetry(
    model: string,
    body: string,
    credentials: ResolvedCreds,
    promptId: string,
    signal?: AbortSignal,
    callOptions: ClientCallOptions = {},
  ): Promise<Response> {
    for (let attempt = 0; ; attempt++) {
      let failure: Error;
      let hintMs = 0;
      try {
        if (signal?.aborted) throw new Error("request cancelled by the caller");
        callOptions.onAttempt?.();
        const httpResponse = await this.sendOnce(model, body, credentials, promptId, signal);
        const isFinal = httpResponse.status < 400 || httpResponse.status === 401 || !shouldRetryStatus(httpResponse.status, httpResponse.headers.get("x-should-retry"));
        if (isFinal) {
          return httpResponse;
        }
        const statusError = await toStatusError(httpResponse);
        failure = statusError;
        hintMs = statusError.retryAfterMs;
      } catch (err) {
        if (signal?.aborted) {
          throw new Error("request cancelled by the caller");
        }
        if (err instanceof APIStatusError) {
          failure = err;
        } else {
          failure = new Error(`api: send: ${(err as Error).message}`);
        }
      }
      if (attempt >= (callOptions.singleAttempt ? 0 : this.maxRetries)) {
        throw failure;
      }
      const delay = backoffMs(attempt, hintMs, this.random);
      if (delay > MAX_RETRY_WAIT_MS) {
        throw failure;
      }
      await this.sleep(delay);
    }
  }
}
