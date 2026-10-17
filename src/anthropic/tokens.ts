import { OAUTH_CONFIG, SCOPES_CLAUDE_AI } from "./wire.js";

/** Signature of `fetch`, injectable so tests can stub the network. */
export type FetchLike = typeof fetch;

/** Normalized result of a successful token exchange or refresh. */
export interface Tokens {
  readonly accessToken: string;
  readonly refreshToken: string;
  /** Lifetime in seconds as reported by the server, or 0 when absent. */
  readonly expiresIn: number;
  readonly tokenType: string;
  readonly scopes: string[];
  /** Account email, or an empty string when the response carried none. */
  readonly email: string;
}

/** Failure talking to the OAuth token endpoint. */
export class TokenError extends Error {
  /**
   * @param message - Human readable description.
   * @param status - HTTP status of the token endpoint, when a response was received.
   */
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "TokenError";
  }
}

/** Inputs for {@link exchangeCode}. */
export interface ExchangeParams {
  /** Authorization code from the callback. */
  readonly code: string;
  /** State value of the login, sent back to the server. */
  readonly state: string;
  /** PKCE verifier that matches the challenge in the authorize URL. */
  readonly codeVerifier: string;
  /** Loopback port used to build the redirect URI. */
  readonly port: number;
  /** Use the hosted manual redirect URI instead of the loopback one. */
  readonly useManual?: boolean;
  /** Requested token lifetime in seconds. Omitted from the request unless positive. */
  readonly expiresIn?: number;
}

/** Optional settings for {@link refreshTokens}. */
export interface RefreshOptions {
  /** Scopes to request. Falls back to {@link SCOPES_CLAUDE_AI} when empty or absent. */
  readonly scopes?: readonly string[];
  /** Requested token lifetime in seconds. Omitted from the request unless positive. */
  readonly expiresIn?: number;
  /** OAuth client id override. */
  readonly clientId?: string;
}

/** Timeout for one call to the token endpoint. */
const REQUEST_TIMEOUT_MS = 30_000;

/** Raw token endpoint response body. Every field is optional until validated. */
interface TokenResponse {
  access_token?: string;
  refresh_token?: string;
  expires_in?: number;
  token_type?: string;
  scope?: string;
  account?: { uuid?: string; email_address?: string };
}

/**
 * Returns the redirect URI that was used in the authorize request.
 *
 * @param params - Exchange parameters.
 * @returns The manual redirect URL, or the loopback callback URL.
 */
function redirectUriFor(params: ExchangeParams): string {
  if (params.useManual) {
    return OAUTH_CONFIG.manualRedirectUrl;
  }
  return `http://localhost:${params.port}/callback`;
}

/**
 * POSTs a JSON body to the token endpoint with a timeout.
 *
 * @param body - Request fields for the grant.
 * @param fetchImpl - Fetch implementation to use.
 * @returns The raw response, which may be a non-2xx status.
 * @throws {TokenError} When the request itself fails (network error or timeout).
 */
async function postToken(body: Record<string, unknown>, fetchImpl: FetchLike): Promise<Response> {
  try {
    return await fetchImpl(OAUTH_CONFIG.tokenUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (err) {
    throw new TokenError(`token endpoint request: ${(err as Error).message}`);
  }
}

/**
 * Validates a successful token response and converts it to {@link Tokens}.
 *
 * @param response - A 2xx response from the token endpoint.
 * @param fallbackRefresh - Refresh token to keep when the server does not rotate it.
 * @returns The normalized tokens.
 * @throws {TokenError} When the body is not JSON, lacks an access token, or the
 *   token type is not bearer.
 */
async function finalizeTokens(response: Response, fallbackRefresh: string): Promise<Tokens> {
  let tokenResponse: TokenResponse;
  try {
    tokenResponse = (await response.json()) as TokenResponse;
  } catch (err) {
    throw new TokenError(`decode token response: ${(err as Error).message}`);
  }
  if (!tokenResponse.access_token) {
    throw new TokenError("token endpoint response missing access_token");
  }
  if (tokenResponse.token_type && tokenResponse.token_type.toLowerCase() !== "bearer") {
    throw new TokenError(`unsupported token_type "${tokenResponse.token_type}" (want bearer)`);
  }
  return {
    accessToken: tokenResponse.access_token,
    refreshToken: tokenResponse.refresh_token || fallbackRefresh,
    expiresIn: tokenResponse.expires_in ?? 0,
    tokenType: tokenResponse.token_type ?? "",
    scopes: (tokenResponse.scope ?? "").split(/\s+/).filter(Boolean),
    email: tokenResponse.account?.email_address ?? "",
  };
}

/**
 * Exchanges an authorization code for tokens (the final step of the PKCE flow).
 *
 * @param params - See {@link ExchangeParams}.
 * @param fetchImpl - Fetch implementation. Defaults to the global `fetch`.
 * @returns The issued tokens. The refresh token is empty if the server sent none.
 * @throws {TokenError} When a required field is missing, the request fails, the
 *   server answers with a non-2xx status, or the response is malformed.
 * @example
 * ```ts
 * const tokens = await exchangeCode({ code, state, codeVerifier: verifier, port: listener.port });
 * ```
 */
export async function exchangeCode(params: ExchangeParams, fetchImpl: FetchLike = fetch): Promise<Tokens> {
  if (!params.code || !params.state || !params.codeVerifier) {
    throw new TokenError("exchangeCode: code, state, and code_verifier are required");
  }
  const body: Record<string, unknown> = {
    grant_type: "authorization_code",
    code: params.code,
    redirect_uri: redirectUriFor(params),
    client_id: OAUTH_CONFIG.clientId,
    code_verifier: params.codeVerifier,
    state: params.state,
  };
  if (params.expiresIn && params.expiresIn > 0) {
    body.expires_in = params.expiresIn;
  }

  const response = await postToken(body, fetchImpl);
  if (response.status === 401) {
    throw new TokenError("Authentication failed: Invalid authorization code", 401);
  }
  if (!response.ok) {
    throw new TokenError(`Token exchange failed (${response.status}): ${response.statusText}`, response.status);
  }
  return finalizeTokens(response, "");
}

/**
 * Trades a refresh token for a new access token.
 *
 * Refresh tokens rotate: a successful call consumes the one presented, and the
 * new one in the response must be stored. If the server omits a new refresh
 * token, the presented one is returned unchanged.
 *
 * @param refreshToken - The current refresh token.
 * @param options - See {@link RefreshOptions}.
 * @param fetchImpl - Fetch implementation. Defaults to the global `fetch`.
 * @returns The refreshed tokens.
 * @throws {TokenError} When `refreshToken` is empty, the request fails, the
 *   server answers with a non-2xx status, or the response is malformed.
 */
export async function refreshTokens(
  refreshToken: string,
  options: RefreshOptions = {},
  fetchImpl: FetchLike = fetch,
): Promise<Tokens> {
  if (!refreshToken) {
    throw new TokenError("refreshTokens: refresh_token required");
  }
  let scopes: readonly string[] = SCOPES_CLAUDE_AI;
  if (options.scopes && options.scopes.length > 0) {
    scopes = options.scopes;
  }
  const body: Record<string, unknown> = {
    grant_type: "refresh_token",
    refresh_token: refreshToken,
    client_id: options.clientId || OAUTH_CONFIG.clientId,
    scope: scopes.join(" "),
  };
  if (options.expiresIn && options.expiresIn > 0) {
    body.expires_in = options.expiresIn;
  }

  const response = await postToken(body, fetchImpl);
  if (!response.ok) {
    throw new TokenError(`Token refresh failed: ${response.status} ${response.statusText}`, response.status);
  }
  return finalizeTokens(response, refreshToken);
}
