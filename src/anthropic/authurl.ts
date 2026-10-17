import { OAUTH_CONFIG, SCOPES_ALL, SCOPES_INFERENCE_ONLY } from "./wire.js";

/** Inputs for {@link buildAuthUrl}. */
export interface BuildAuthUrlParams {
  /** PKCE `S256` challenge for the verifier kept by the caller. */
  readonly codeChallenge: string;
  /** Opaque value echoed back on the callback to bind it to this login. */
  readonly state: string;
  /** Loopback port of the callback listener. Required unless `isManual` is true. */
  readonly port?: number;
  /** Use the hosted manual redirect page instead of a loopback callback. */
  readonly isManual?: boolean;
  /** Authorize through claude.ai (subscription) rather than the console. */
  readonly loginWithClaudeAi?: boolean;
  /** Request only the `user:inference` scope. */
  readonly inferenceOnly?: boolean;
  /** Pre-select an organization. */
  readonly orgUuid?: string;
  /** Pre-fill the sign-in form with an email address. */
  readonly loginHint?: string;
  /** Pre-select a sign-in method. */
  readonly loginMethod?: string;
}

/**
 * Returns the redirect URI that matches the chosen flow.
 *
 * @param isManual - Whether the manual (copy the code) flow is in use.
 * @param port - Loopback listener port for the automatic flow.
 * @returns The redirect URI to send to the authorize endpoint.
 */
function redirectUriFor(isManual: boolean | undefined, port: number | undefined): string {
  if (isManual) {
    return OAUTH_CONFIG.manualRedirectUrl;
  }
  return `http://localhost:${port}/callback`;
}

/**
 * Builds the OAuth authorize URL for a PKCE sign-in.
 *
 * @param params - See {@link BuildAuthUrlParams}.
 * @returns The full authorize URL to open in a browser.
 * @throws {Error} When `codeChallenge` or `state` is empty, or when the
 *   automatic flow is requested without a positive `port`.
 * @example
 * ```ts
 * const url = buildAuthUrl({
 *   codeChallenge: s256Challenge(verifier),
 *   state,
 *   port: listener.port,
 *   loginWithClaudeAi: true,
 * });
 * ```
 */
export function buildAuthUrl(params: BuildAuthUrlParams): string {
  if (!params.codeChallenge) {
    throw new Error("buildAuthUrl: missing code_challenge");
  }
  if (!params.state) {
    throw new Error("buildAuthUrl: missing state");
  }
  const isPortMissing = params.port === undefined || params.port <= 0;
  if (!params.isManual && isPortMissing) {
    throw new Error("buildAuthUrl: automatic flow requires port > 0");
  }

  let base: string = OAUTH_CONFIG.consoleAuthorizeUrl;
  if (params.loginWithClaudeAi) {
    base = OAUTH_CONFIG.claudeAiAuthorizeUrl;
  }
  let scopes = SCOPES_ALL;
  if (params.inferenceOnly) {
    scopes = SCOPES_INFERENCE_ONLY;
  }

  const searchParams = new URLSearchParams();
  searchParams.append("code", "true");
  searchParams.append("client_id", OAUTH_CONFIG.clientId);
  searchParams.append("response_type", "code");
  searchParams.append("redirect_uri", redirectUriFor(params.isManual, params.port));
  searchParams.append("scope", scopes.join(" "));
  searchParams.append("code_challenge", params.codeChallenge);
  searchParams.append("code_challenge_method", "S256");
  searchParams.append("state", params.state);
  if (params.orgUuid) {
    searchParams.append("orgUUID", params.orgUuid);
  }
  if (params.loginHint) {
    searchParams.append("login_hint", params.loginHint);
  }
  if (params.loginMethod) {
    searchParams.append("login_method", params.loginMethod);
  }
  return `${base}?${searchParams.toString()}`;
}
