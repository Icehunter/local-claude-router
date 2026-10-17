import { randomUUID } from "node:crypto";
import { isCanonicalUuid } from "./billing.js";
import {
  ANTHROPIC_VERSION,
  BETA_HEADERS,
  CLAUDE_CODE_VERSION,
  DEFAULT_ENTRYPOINT,
  SDK_PACKAGE_VERSION,
  STAINLESS_RUNTIME_VERSION,
} from "./wire.js";

/** Beta flag that selects OAuth bearer authentication. Dropped for API key requests. */
const OAUTH_BETA = "oauth-2025-04-20";

/** Beta flag for the 1M token context window. Kept only for models addressed with the `[1m]` suffix. */
const CONTEXT_1M_BETA = "context-1m-2025-08-07";

/** Beta flag for interleaved thinking. Haiku models do not support it. */
const INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14";

/** Model name prefixes (after any provider path) that identify Haiku models. */
const HAIKU_PREFIXES = ["claude-haiku", "claude-3-haiku", "claude-3-5-haiku"] as const;

/** Suffix that requests the 1M context variant of a model. */
const CONTEXT_1M_SUFFIX = "[1m]";

/** Values of `CLAUDE_CODE_GATEWAY_HINT_HEADERS` that force hint headers on. */
const HINT_ENABLED_VALUES = ["1", "true", "yes", "on"] as const;

/** Values of `CLAUDE_CODE_GATEWAY_HINT_HEADERS` that force hint headers off. */
const HINT_DISABLED_VALUES = ["0", "false", "no", "off"] as const;

/** Host of the first-party API, the only host that gets hint headers by default. */
const FIRST_PARTY_HOST = "api.anthropic.com";

/** Percent sign byte, which must itself be escaped in hint values. */
const PERCENT_BYTE = 0x25;

/** Lowest printable ASCII byte. Anything below is escaped. */
const FIRST_PRINTABLE_BYTE = 0x20;

/** Highest printable ASCII byte. Anything above is escaped. */
const LAST_PRINTABLE_BYTE = 0x7e;

/** `X-Stainless-Timeout` value the official SDK sends, in seconds. */
const STAINLESS_TIMEOUT_SECONDS = "600";

/**
 * Maps a Node `process.platform` value to the OS name the Stainless SDK reports.
 *
 * @param platform - Node platform identifier. Defaults to the current platform.
 * @returns The `X-Stainless-OS` header value; unknown platforms become `Other:<platform>`.
 */
export function stainlessOs(platform: string = process.platform): string {
  switch (platform) {
    case "darwin":
      return "MacOS";
    case "linux":
      return "Linux";
    case "win32":
      return "Windows";
    case "freebsd":
      return "FreeBSD";
    case "openbsd":
      return "OpenBSD";
    default:
      return `Other:${platform}`;
  }
}

/**
 * Maps a Node `process.arch` value to the architecture name the Stainless SDK reports.
 *
 * @param arch - Node architecture identifier. Defaults to the current architecture.
 * @returns The `X-Stainless-Arch` header value; unknown values become `other:<arch>`.
 */
export function stainlessArch(arch: string = process.arch): string {
  switch (arch) {
    case "x64":
      return "x64";
    case "ia32":
      return "x32";
    case "arm":
      return "arm";
    case "arm64":
      return "arm64";
    default:
      return `other:${arch}`;
  }
}

/**
 * Lists beta flags a model does not accept.
 *
 * @param model - Model id, optionally prefixed with a provider path (`a/b/model`).
 * @returns The set of denied flags, or null when the model has no model-specific denials.
 */
export function betaDenylistForModel(model: string): Set<string> | null {
  const lower = model.slice(model.lastIndexOf("/") + 1).toLowerCase();
  const isHaiku = HAIKU_PREFIXES.some((prefix) => lower.startsWith(prefix));
  if (isHaiku) {
    return new Set([CONTEXT_1M_BETA, INTERLEAVED_THINKING_BETA]);
  }
  return null;
}

/**
 * Removes beta flags that the target model does not support.
 *
 * The 1M context flag is always removed unless the model id ends in `[1m]`.
 *
 * @param betas - Candidate beta flags.
 * @param model - Model id the request is for.
 * @returns A new array with unsupported flags removed.
 */
export function filterBetasForModel(betas: readonly string[], model: string): string[] {
  const deny = new Set(betaDenylistForModel(model) ?? []);
  const isContext1m = model.toLowerCase().endsWith(CONTEXT_1M_SUFFIX);
  if (!isContext1m) {
    deny.add(CONTEXT_1M_BETA);
  }
  return betas.filter((beta) => !deny.has(beta));
}

/**
 * Checks whether a base URL points at the first-party Anthropic API.
 *
 * @param baseUrl - API base URL. An empty string means the default, first-party URL.
 * @returns True for an empty string or a URL whose host is `api.anthropic.com`;
 *   false for other hosts and for strings that are not URLs.
 */
export function isFirstPartyBaseUrl(baseUrl: string): boolean {
  if (baseUrl === "") {
    return true;
  }
  try {
    return new URL(baseUrl).host === FIRST_PARTY_HOST;
  } catch {
    return false;
  }
}

/** Optional hints that identify the kind of request to a gateway. */
export interface WireHints {
  /** Request class, for example `main`. */
  readonly requestClass?: string;
  /** Agent type. Percent-escaped on the wire. */
  readonly agentType?: string;
  /** Canonical UUID of the prompt. Dropped when malformed. */
  readonly promptId?: string;
}

/**
 * Decides whether gateway hint headers are sent.
 *
 * `CLAUDE_CODE_GATEWAY_HINT_HEADERS` wins when set to a recognized value;
 * otherwise hints go only to the first-party API.
 *
 * @param baseUrl - API base URL of the request.
 * @returns True when hint headers should be added.
 */
function isHintHeadersEnabled(baseUrl: string): boolean {
  const override = (process.env.CLAUDE_CODE_GATEWAY_HINT_HEADERS ?? "").trim().toLowerCase();
  if ((HINT_ENABLED_VALUES as readonly string[]).includes(override)) {
    return true;
  }
  if ((HINT_DISABLED_VALUES as readonly string[]).includes(override)) {
    return false;
  }
  return isFirstPartyBaseUrl(baseUrl);
}

/**
 * Percent-escapes a hint value so it is safe in an HTTP header.
 *
 * Works on UTF-8 bytes: `%`, control characters and every non-ASCII byte become `%XX`.
 *
 * @param rawValue - Raw value.
 * @returns The escaped value.
 */
function escapeHintValue(rawValue: string): string {
  let out = "";
  for (const byte of Buffer.from(rawValue, "utf8")) {
    const isUnsafe = byte === PERCENT_BYTE || byte < FIRST_PRINTABLE_BYTE || byte > LAST_PRINTABLE_BYTE;
    if (isUnsafe) {
      out += `%${byte.toString(16).toUpperCase().padStart(2, "0")}`;
      continue;
    }
    out += String.fromCharCode(byte);
  }
  return out;
}

/**
 * Adds the `x-claude-code-*` hint headers to `headers` in place.
 *
 * Does nothing when `hints` is undefined or hint headers are disabled for `baseUrl`.
 *
 * @param headers - Header map to mutate.
 * @param hints - Hints to apply.
 * @param baseUrl - API base URL, used for the first-party check.
 */
export function applyWireHints(
  headers: Record<string, string>,
  hints: WireHints | undefined,
  baseUrl: string,
): void {
  if (!hints || !isHintHeadersEnabled(baseUrl)) {
    return;
  }
  if (hints.requestClass) {
    headers["x-claude-code-request-class"] = hints.requestClass;
  }
  if (hints.agentType) {
    headers["x-claude-code-agent-type"] = escapeHintValue(hints.agentType);
  }
  if (hints.promptId && isCanonicalUuid(hints.promptId)) {
    headers["x-claude-code-prompt-id"] = hints.promptId;
  }
}

/** Inputs for {@link buildHeaders}. */
export interface BuildHeadersOptions {
  /** API base URL, used to decide whether hint headers apply. */
  readonly baseUrl: string;
  /** Target model id, used to filter beta flags. */
  readonly model: string;
  /** OAuth access token. When set, bearer authentication is used. */
  readonly accessToken?: string;
  /** API key, used only when no access token is given. */
  readonly apiKey?: string;
  /** Session id sent as `X-Claude-Code-Session-Id`. */
  readonly sessionId?: string;
  /** Overrides the generated `User-Agent`. */
  readonly userAgent?: string;
  /** Overrides the entrypoint name. Defaults to `CLAUDE_CODE_ENTRYPOINT`, then `sdk-cli`. */
  readonly entrypoint?: string;
  /** Gateway hints. */
  readonly hints?: WireHints;
  /** Overrides the generated `x-client-request-id`. */
  readonly requestId?: string;
  /** Overrides the detected platform. */
  readonly platform?: string;
  /** Overrides the detected architecture. */
  readonly arch?: string;
}

/**
 * Resolves the entrypoint name: explicit option, then environment, then default.
 *
 * @param explicit - Entrypoint passed by the caller, if any.
 * @returns The entrypoint name.
 */
function resolveEntrypoint(explicit: string | undefined): string {
  if (explicit !== undefined) {
    return explicit;
  }
  return process.env.CLAUDE_CODE_ENTRYPOINT || DEFAULT_ENTRYPOINT;
}

/**
 * Builds the request headers for one call to `/v1/messages`.
 *
 * The set mirrors what Claude Code sends: client identity, Stainless SDK
 * fingerprint, model-filtered beta flags, and authentication. The OAuth beta
 * flag is sent only with a bearer token, since API key requests reject it.
 *
 * @param options - See {@link BuildHeadersOptions}.
 * @returns A header map ready to pass to `fetch`.
 * @example
 * ```ts
 * const headers = buildHeaders({
 *   baseUrl: "https://api.anthropic.com",
 *   model: "claude-sonnet-5-5",
 *   accessToken,
 * });
 * ```
 */
export function buildHeaders(options: BuildHeadersOptions): Record<string, string> {
  const entrypoint = resolveEntrypoint(options.entrypoint);
  const isBearer = !!options.accessToken;
  const headers: Record<string, string> = {
    Accept: "application/json",
    "Content-Type": "application/json",
    "User-Agent": options.userAgent ?? `claude-cli/${CLAUDE_CODE_VERSION} (external, ${entrypoint})`,
    "anthropic-version": ANTHROPIC_VERSION,
    "x-app": "cli",
  };
  if (options.sessionId) {
    headers["X-Claude-Code-Session-Id"] = options.sessionId;
  }

  let betas = filterBetasForModel(BETA_HEADERS, options.model);
  if (!isBearer) {
    betas = betas.filter((beta) => beta !== OAUTH_BETA);
  }
  if (betas.length > 0) {
    headers["anthropic-beta"] = betas.join(",");
  }

  headers["X-Stainless-Lang"] = "js";
  headers["X-Stainless-Package-Version"] = SDK_PACKAGE_VERSION;
  headers["X-Stainless-OS"] = stainlessOs(options.platform);
  headers["X-Stainless-Arch"] = stainlessArch(options.arch);
  headers["X-Stainless-Runtime"] = "node";
  headers["X-Stainless-Runtime-Version"] = STAINLESS_RUNTIME_VERSION;
  headers["x-client-request-id"] = options.requestId ?? randomUUID();
  headers["anthropic-dangerous-direct-browser-access"] = "true";
  headers["X-Stainless-Retry-Count"] = "0";
  headers["X-Stainless-Timeout"] = STAINLESS_TIMEOUT_SECONDS;

  if (isBearer) {
    headers.Authorization = `Bearer ${options.accessToken}`;
  } else if (options.apiKey) {
    headers["x-api-key"] = options.apiKey;
  }

  applyWireHints(headers, options.hints, options.baseUrl);
  return headers;
}
