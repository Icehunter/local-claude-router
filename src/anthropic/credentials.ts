import { refreshTokens } from "./tokens.js";
import type { FetchLike, Tokens } from "./tokens.js";
import {
  ownCredentialsPath,
  readConduitCredentials,
  readOwnCredentials,
  writeOwnCredentials,
} from "./store.js";
import type { KeychainReader, StoredCredentials } from "./store.js";

/** Where credentials were found: this server's own file, or conduit's store (read-only). */
export type CredentialSource = "own" | "conduit";

/** Stored credentials tagged with the source they came from. */
export interface ResolvedCredentials extends StoredCredentials {
  readonly source: CredentialSource;
}

/** Thrown when no credentials exist in either source. */
export class NotLoggedInError extends Error {
  public constructor() {
    super(
      "No Claude credentials found. Run `npm run login` to sign in, " +
        "or sign in with conduit so its keychain entry can be read.",
    );
    this.name = "NotLoggedInError";
  }
}

/** Message for an expired token that came from conduit's store. */
const CONDUIT_EXPIRED_MESSAGE =
  "The conduit access token has expired. This server never refreshes conduit's token " +
  "(refresh tokens rotate and a second refresher could revoke conduit's session). " +
  "Run conduit once to renew it, or run `npm run login` to give this server its own login.";

/** Message for an expired token of this server's own login that has no refresh token. */
const OWN_EXPIRED_MESSAGE = "The stored access token has expired and cannot be refreshed. Run `npm run login` again.";

/**
 * Picks the user facing message for an expired credential.
 *
 * @param source - Where the expired credential came from.
 * @returns The message, with source specific remediation steps.
 */
function expiredMessageFor(source: CredentialSource): string {
  if (source === "conduit") {
    return CONDUIT_EXPIRED_MESSAGE;
  }
  return OWN_EXPIRED_MESSAGE;
}

/** Thrown when a token has expired and cannot be renewed from this process. */
export class CredentialsExpiredError extends Error {
  /**
   * @param source - Where the expired credential came from. Conduit's token is
   *   never refreshed here, because refresh tokens rotate and a second refresher
   *   could revoke conduit's session.
   */
  public constructor(source: CredentialSource) {
    super(expiredMessageFor(source));
    this.name = "CredentialsExpiredError";
  }
}

/** Options for {@link CredentialManager}. Every field exists so tests can inject fakes. */
export interface CredentialManagerOptions {
  /** Fetch implementation used for token refresh. */
  readonly fetch?: FetchLike;
  /** Clock in epoch milliseconds. */
  readonly now?: () => number;
  /** Path of this server's credentials file. */
  readonly ownPath?: string;
  /** Keychain reader used for conduit's store. */
  readonly keychain?: KeychainReader;
  /** Environment used to locate files. */
  readonly env?: NodeJS.ProcessEnv;
  /** Path of conduit's file-based credential cache. */
  readonly conduitFilePath?: string;
  /** Safety margin: a token expiring within this many milliseconds counts as expired. */
  readonly skewMs?: number;
}

/** Default safety margin before expiry (one minute). */
const DEFAULT_SKEW_MS = 60_000;

/** Milliseconds per second, for converting `expires_in`. */
const MS_PER_SECOND = 1000;

/**
 * Resolves, refreshes and stores the credentials used to call the API.
 *
 * Two sources are consulted, in order:
 *
 * 1. This server's own file, which it may refresh and rewrite.
 * 2. Conduit's store, which is read-only here.
 *
 * Refresh tokens rotate: presenting one consumes it. Concurrent refreshes in
 * this process are therefore collapsed into one call, and the file is re-read
 * before each refresh in case another process already rotated the token.
 */
export class CredentialManager {
  private inflight: Promise<ResolvedCredentials> | null = null;
  private readonly fetchImpl: FetchLike;
  private readonly now: () => number;
  private readonly ownPath: string;
  private readonly skewMs: number;

  /**
   * @param options - See {@link CredentialManagerOptions}. All fields are optional.
   */
  public constructor(private readonly options: CredentialManagerOptions = {}) {
    this.fetchImpl = options.fetch ?? fetch;
    this.now = options.now ?? Date.now;
    this.ownPath = options.ownPath ?? ownCredentialsPath(options.env);
    this.skewMs = options.skewMs ?? DEFAULT_SKEW_MS;
  }

  /**
   * Returns credentials that are valid for a request, refreshing if needed.
   *
   * Own credentials are used first and refreshed when stale. Conduit credentials
   * are returned only while still fresh; they are never refreshed here.
   *
   * @returns Credentials with their source.
   * @throws {NotLoggedInError} When neither source has credentials.
   * @throws {CredentialsExpiredError} When the token is stale and cannot be renewed.
   * @throws {Error} When the refresh request fails.
   * @example
   * ```ts
   * const creds = await manager.get();
   * const token = creds.accessToken;
   * ```
   */
  public async get(): Promise<ResolvedCredentials> {
    const own = this.loadOwn();
    if (own) {
      if (this.isFresh(own)) {
        return own;
      }
      return this.refreshOwn(own.accessToken);
    }
    const conduit = await this.loadConduit();
    if (!conduit) {
      throw new NotLoggedInError();
    }
    if (!this.isFresh(conduit)) {
      throw new CredentialsExpiredError("conduit");
    }
    return conduit;
  }

  /**
   * Read-only view for status reporting: never refreshes and never throws on expiry.
   *
   * @returns The credentials found plus an `expired` flag, or null when there are none.
   * @example
   * ```ts
   * const found = await manager.inspect();
   * console.log(found?.source, found?.expired);
   * ```
   */
  public async inspect(): Promise<(ResolvedCredentials & { expired: boolean }) | null> {
    const found = this.loadOwn() ?? (await this.loadConduit());
    if (!found) {
      return null;
    }
    return { ...found, expired: !this.isFresh(found) };
  }

  /**
   * Called after the API rejected `rejectedToken` with 401.
   *
   * Own credentials are refreshed (unless another caller already replaced the
   * rejected token). Conduit credentials are returned only if conduit has since
   * renewed its token.
   *
   * @param rejectedToken - The access token (or API key) that got the 401.
   * @returns Credentials that differ from the rejected one.
   * @throws {NotLoggedInError} When neither source has credentials.
   * @throws {CredentialsExpiredError} When conduit's token is unchanged or stale.
   * @throws {Error} When the refresh request fails.
   * @example
   * ```ts
   * const next = await manager.refreshAfter401(creds.accessToken);
   * ```
   */
  public async refreshAfter401(rejectedToken: string): Promise<ResolvedCredentials> {
    const own = this.loadOwn();
    if (own) {
      return this.refreshOwn(rejectedToken);
    }
    const conduit = await this.loadConduit();
    if (!conduit) {
      throw new NotLoggedInError();
    }
    const isRenewed = conduit.accessToken !== rejectedToken && this.isFresh(conduit);
    if (isRenewed) {
      return conduit;
    }
    throw new CredentialsExpiredError("conduit");
  }

  /**
   * Persists a completed login as the own credential source.
   *
   * @param tokens - Tokens from the code exchange.
   * @returns The stored credentials, tagged as `own`.
   * @example
   * ```ts
   * manager.saveLogin(await exchangeCode(params));
   * ```
   */
  public saveLogin(tokens: Tokens): ResolvedCredentials {
    const stored: StoredCredentials = {
      accessToken: tokens.accessToken,
      refreshToken: tokens.refreshToken,
      expiresAt: this.now() + tokens.expiresIn * MS_PER_SECOND,
      scopes: tokens.scopes,
      email: tokens.email,
    };
    writeOwnCredentials(stored, this.ownPath);
    return { ...stored, source: "own" };
  }

  /**
   * Checks whether credentials can be used now.
   *
   * @param credentials - Credentials to check.
   * @returns True for API keys (which do not expire) and for tokens that expire
   *   later than the skew margin from now.
   */
  private isFresh(credentials: StoredCredentials): boolean {
    return credentials.apiKey !== undefined || this.now() + this.skewMs < credentials.expiresAt;
  }

  /**
   * Reads conduit's credentials.
   *
   * @returns The credentials tagged `conduit`, or null when absent.
   */
  private async loadConduit(): Promise<ResolvedCredentials | null> {
    const credentials = await readConduitCredentials({
      keychain: this.options.keychain,
      env: this.options.env,
      fileStoragePath: this.options.conduitFilePath,
    });
    if (!credentials) {
      return null;
    }
    return { ...credentials, source: "conduit" };
  }

  /**
   * Reads this server's own credentials file.
   *
   * @returns The credentials tagged `own`, or null when absent.
   */
  private loadOwn(): ResolvedCredentials | null {
    const credentials = readOwnCredentials(this.ownPath);
    if (!credentials) {
      return null;
    }
    return { ...credentials, source: "own" };
  }

  /**
   * Refreshes the own credentials, sharing one in-flight call among concurrent callers.
   *
   * @param staleToken - The access token that is known to be stale or rejected.
   * @returns The refreshed credentials.
   */
  private refreshOwn(staleToken: string): Promise<ResolvedCredentials> {
    if (!this.inflight) {
      this.inflight = this.doRefreshOwn(staleToken).finally(() => {
        this.inflight = null;
      });
    }
    return this.inflight;
  }

  /**
   * Performs one refresh and writes the result to disk.
   *
   * Another process (or an earlier call) may already have rotated the token;
   * presenting a consumed refresh token would revoke the session, so the file
   * is re-read first and a newer fresh token is returned as is.
   *
   * @param staleToken - The access token that is known to be stale or rejected.
   * @returns The refreshed credentials.
   * @throws {NotLoggedInError} When the credentials file disappeared.
   * @throws {CredentialsExpiredError} When there is no refresh token to use.
   * @throws {Error} When the refresh request fails; the message suggests logging in again.
   */
  private async doRefreshOwn(staleToken: string): Promise<ResolvedCredentials> {
    const latest = this.loadOwn();
    if (!latest) {
      throw new NotLoggedInError();
    }
    const isAlreadyRotated = latest.accessToken !== staleToken && this.isFresh(latest);
    if (isAlreadyRotated) {
      return latest;
    }
    if (!latest.refreshToken) {
      throw new CredentialsExpiredError("own");
    }

    let refreshed: Tokens;
    try {
      refreshed = await refreshTokens(latest.refreshToken, { scopes: latest.scopes }, this.fetchImpl);
    } catch (err) {
      throw new Error(`${(err as Error).message}. If the refresh token was revoked or consumed, run \`npm run login\`.`, { cause: err });
    }
    let scopes = latest.scopes;
    if (refreshed.scopes.length > 0) {
      scopes = refreshed.scopes;
    }
    const next: StoredCredentials = {
      accessToken: refreshed.accessToken,
      refreshToken: refreshed.refreshToken || latest.refreshToken,
      expiresAt: this.now() + refreshed.expiresIn * MS_PER_SECOND,
      scopes,
      email: refreshed.email || latest.email,
      apiKey: latest.apiKey,
      accountKind: latest.accountKind,
    };
    writeOwnCredentials(next, this.ownPath);
    return { ...next, source: "own" };
  }
}
