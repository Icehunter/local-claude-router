import { randomUUID } from "node:crypto";
import { AnthropicClient } from "./anthropic/client.js";
import { CredentialManager } from "./anthropic/credentials.js";
import { LoginController } from "./anthropic/login.js";

/** Model used when `ANTHROPIC_DEFAULT_MODEL` is not set. */
const DEFAULT_MODEL = "claude-sonnet-5-5";

/** `max_tokens` used when `ANTHROPIC_MAX_TOKENS` is not set. */
const DEFAULT_MAX_TOKENS = 4096;

/** Largest `max_tokens` accepted, for both the environment default and per-request values. */
export const MAX_MAX_TOKENS = 128_000;

/** The wired-up services shared by the CLI and the HTTP server. */
export interface Deps {
  readonly credentials: CredentialManager;
  readonly client: Pick<AnthropicClient, "createMessage" | "createMessageStream">;
  readonly login: Pick<LoginController, "start" | "state" | "settled">;
  /** Model used when a request does not name a Claude model. */
  readonly defaultModel: string;
  /** `max_tokens` used when a request does not set one. */
  readonly defaultMaxTokens: number;
}

/**
 * Reads `ANTHROPIC_MAX_TOKENS`.
 *
 * @param raw - Raw environment value.
 * @returns The configured limit, or {@link DEFAULT_MAX_TOKENS} when unset or blank.
 * @throws {Error} When the value is not an integer from 1 to {@link MAX_MAX_TOKENS}.
 */
function maxTokensFromEnv(raw: string | undefined): number {
  const trimmed = raw?.trim();
  if (!trimmed) {
    return DEFAULT_MAX_TOKENS;
  }
  const parsedLimit = Number(trimmed);
  if (!Number.isInteger(parsedLimit) || parsedLimit < 1 || parsedLimit > MAX_MAX_TOKENS) {
    throw new Error(`ANTHROPIC_MAX_TOKENS must be an integer between 1 and ${MAX_MAX_TOKENS}, got: "${trimmed}"`);
  }
  return parsedLimit;
}

/**
 * Builds the shared services from environment variables.
 *
 * Reads `ANTHROPIC_MAX_TOKENS`, `ANTHROPIC_DEFAULT_MODEL`, and the variables
 * understood by {@link CredentialManager} (such as `ANTHROPIC_CREDENTIALS_FILE`).
 * Each call creates a fresh session id for the upstream client.
 *
 * @param env - Environment to read. Defaults to `process.env`.
 * @returns The wired services and defaults.
 * @throws {Error} When `ANTHROPIC_MAX_TOKENS` is invalid.
 * @example
 * ```ts
 * const deps = depsFromEnv();
 * const status = await deps.credentials.inspect();
 * ```
 */
export function depsFromEnv(env: NodeJS.ProcessEnv = process.env): Deps {
  const credentials = new CredentialManager({ env });
  const defaultMaxTokens = maxTokensFromEnv(env.ANTHROPIC_MAX_TOKENS);
  return {
    credentials,
    client: new AnthropicClient({ credentials, sessionId: randomUUID() }),
    login: new LoginController({ credentials }),
    defaultModel: env.ANTHROPIC_DEFAULT_MODEL?.trim() || DEFAULT_MODEL,
    defaultMaxTokens,
  };
}
