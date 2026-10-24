/**
 * Constants that make requests look like the Claude Code CLI to the upstream API.
 *
 * Subscription OAuth tokens are only accepted for requests shaped like Claude
 * Code, so versions, salts, scopes and beta flags here mirror that client. They
 * are plain data: changing one changes what is sent on the wire.
 *
 * @module
 */

/** Claude Code CLI version reported in the `User-Agent` header and the billing block. */
export const CLAUDE_CODE_VERSION = "2.1.309";

/** `@anthropic-ai/sdk` version reported in `X-Stainless-Package-Version`. */
export const SDK_PACKAGE_VERSION = "0.128.0";

/** Node runtime version reported in `X-Stainless-Runtime-Version`. */
export const STAINLESS_RUNTIME_VERSION = "v26.3.0";

/** Value of the `anthropic-version` header. */
export const ANTHROPIC_VERSION = "2023-06-01";

/** Salt mixed into the per-request billing suffix hash. See `computeBillingSuffix`. */
export const BILLING_SALT = "59cf53e54c78";

/** Constant `cch` field of the billing header text. */
export const BILLING_CCH = "00000";

/** Entrypoint name used when `CLAUDE_CODE_ENTRYPOINT` is not set. */
export const DEFAULT_ENTRYPOINT = "sdk-cli";

/** Identity line sent as the second system block, ahead of any caller-supplied system text. */
export const IDENTITY_SYSTEM = "You are a Claude agent, built on Anthropic's Claude Agent SDK.";

/** First-party API origin used when no base URL override is given. */
export const BASE_API_URL = "https://api.anthropic.com";

/** OAuth client identity and endpoints used for sign-in and token refresh. */
export const OAUTH_CONFIG = {
  clientId: "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
  claudeAiAuthorizeUrl: "https://claude.ai/oauth/authorize",
  consoleAuthorizeUrl: "https://platform.claude.com/oauth/authorize",
  tokenUrl: "https://platform.claude.com/v1/oauth/token",
  manualRedirectUrl: "https://platform.claude.com/oauth/code/callback",
  claudeAiSuccessUrl: "https://platform.claude.com/oauth/code/success?app=claude-code",
  consoleSuccessUrl:
    "https://platform.claude.com/buy_credits?returnUrl=/oauth/code/success%3Fapp%3Dclaude-code",
} as const;

/** Scopes requested when refreshing a claude.ai subscription token. */
export const SCOPES_CLAUDE_AI: readonly string[] = [
  "user:profile",
  "user:inference",
  "user:sessions:claude_code",
  "user:mcp_servers",
  "user:file_upload",
];

/** Scopes requested by a full interactive sign-in, including API key creation. */
export const SCOPES_ALL: readonly string[] = [
  "org:create_api_key",
  "user:profile",
  "user:inference",
  "user:sessions:claude_code",
  "user:mcp_servers",
  "user:file_upload",
];

/** Minimal scope set for tokens that only need to call the inference API. */
export const SCOPES_INFERENCE_ONLY: readonly string[] = ["user:inference"];

/** Beta flags sent in `anthropic-beta`, before per-model and per-credential filtering. */
export const BETA_HEADERS: readonly string[] = [
  "claude-code-20250219",
  "oauth-2025-04-20",
  "interleaved-thinking-2025-05-14",
  "thinking-token-count-2026-05-13",
  "context-management-2025-06-27",
  "prompt-caching-scope-2026-01-05",
  "advisor-tool-2026-03-01",
  "advanced-tool-use-2025-11-20",
  "effort-2025-11-24",
  "extended-cache-ttl-2025-04-11",
  "cache-diagnosis-2026-04-07",
];
