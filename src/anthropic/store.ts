import { execFile } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/** Keychain service name under which conduit stores its OAuth tokens. */
export const CONDUIT_KEYCHAIN_SERVICE = "conduit-credentials";

/** Service name conduit uses as a key prefix in its file-based credential cache. */
export const CONDUIT_FILE_SERVICE = "com.icehunter.conduit";

/** Absolute path of the macOS `security` tool, used instead of a PATH lookup. */
const SECURITY_BIN = "/usr/bin/security";

/** Directory mode for the credentials folder (owner only). */
const CREDENTIALS_DIR_MODE = 0o700;

/** File mode for the credentials file (owner read and write only). */
const CREDENTIALS_FILE_MODE = 0o600;

/** Prefix of keyring values that hold base64 text. */
const KEYRING_BASE64_PREFIX = "go-keyring-base64:";

/** Prefix of keyring values that hold hex text. */
const KEYRING_HEX_PREFIX = "go-keyring-encoded:";

/** Account kind assumed when conduit does not record one. */
const DEFAULT_ACCOUNT_KIND = "claude-ai";

/** Account kinds that conduit can embed as a `kind:email` prefix. */
const ACCOUNT_KINDS: readonly string[] = [DEFAULT_ACCOUNT_KIND, "anthropic-console"];

/** Credentials in the in-memory shape used by this server. */
export interface StoredCredentials {
  readonly accessToken: string;
  readonly refreshToken: string;
  /** Expiry as epoch milliseconds, or 0 when unknown. */
  readonly expiresAt: number;
  readonly scopes: string[];
  readonly email: string;
  /** Console API key, present for console accounts. */
  readonly apiKey?: string;
  /** Account kind such as `claude-ai` or `anthropic-console`. */
  readonly accountKind?: string;
}

/**
 * Resolves where this server keeps its own credentials.
 *
 * @param env - Environment to read `ANTHROPIC_CREDENTIALS_FILE` from.
 * @returns The override path, or `~/.local-claude-router/anthropic-credentials.json`.
 */
export function ownCredentialsPath(env: NodeJS.ProcessEnv = process.env): string {
  if (env.ANTHROPIC_CREDENTIALS_FILE) {
    return env.ANTHROPIC_CREDENTIALS_FILE;
  }
  return join(homedir(), ".local-claude-router", "anthropic-credentials.json");
}

/** Credentials as they appear on disk and in conduit's store (snake case). */
interface WireCredentials {
  access_token?: string;
  refresh_token?: string;
  expires_at?: string;
  scopes?: string[] | null;
  email?: string;
  api_key?: string;
  account_kind?: string;
}

/**
 * Parses an ISO timestamp into epoch milliseconds.
 *
 * @param iso - Timestamp text, possibly absent.
 * @returns Epoch milliseconds, or 0 when absent or unparseable.
 */
function parseExpiry(iso: string | undefined): number {
  if (!iso) {
    return 0;
  }
  const parsed = Date.parse(iso);
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return parsed;
}

/**
 * Converts the on-disk shape to {@link StoredCredentials}.
 *
 * @param wireCredentials - Parsed credentials JSON.
 * @returns The credentials, or null when neither an access token nor an API key is present.
 */
function fromWire(wireCredentials: WireCredentials): StoredCredentials | null {
  if (!wireCredentials.access_token && !wireCredentials.api_key) {
    return null;
  }
  return {
    accessToken: wireCredentials.access_token ?? "",
    refreshToken: wireCredentials.refresh_token ?? "",
    expiresAt: parseExpiry(wireCredentials.expires_at),
    scopes: wireCredentials.scopes ?? [],
    email: wireCredentials.email ?? "",
    apiKey: wireCredentials.api_key || undefined,
    accountKind: wireCredentials.account_kind || undefined,
  };
}

/**
 * Reads this server's own credentials file.
 *
 * @param path - File to read. Defaults to {@link ownCredentialsPath}.
 * @returns The credentials, or null when the file is missing or holds no usable token.
 * @throws {Error} When the file exists but cannot be read or parsed.
 */
export function readOwnCredentials(path: string = ownCredentialsPath()): StoredCredentials | null {
  if (!existsSync(path)) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(path, "utf8"));
  } catch (err) {
    throw new Error(`Failed to read ${path}: ${(err as Error).message}`);
  }
  return fromWire(parsed as WireCredentials);
}

/**
 * Converts {@link StoredCredentials} to the on-disk shape.
 *
 * @param credentials - Credentials to serialize.
 * @returns The snake case record. Optional fields are omitted when empty.
 */
function toWire(credentials: StoredCredentials): WireCredentials {
  const wire: WireCredentials = {
    access_token: credentials.accessToken,
    refresh_token: credentials.refreshToken,
    expires_at: new Date(credentials.expiresAt).toISOString(),
    scopes: credentials.scopes,
    email: credentials.email,
  };
  if (credentials.apiKey) {
    wire.api_key = credentials.apiKey;
  }
  if (credentials.accountKind) {
    wire.account_kind = credentials.accountKind;
  }
  return wire;
}

/**
 * Writes this server's own credentials file.
 *
 * The write goes to a temp file with mode 0600 and is renamed into place, so a
 * crash never leaves a partial file and the token is never world readable.
 * Writing matters because refresh tokens rotate: losing the new one after a
 * refresh would force a fresh login.
 *
 * @param credentials - Credentials to store.
 * @param path - Destination file. Defaults to {@link ownCredentialsPath}.
 */
export function writeOwnCredentials(credentials: StoredCredentials, path: string = ownCredentialsPath()): void {
  mkdirSync(dirname(path), { recursive: true, mode: CREDENTIALS_DIR_MODE });
  const temporaryPath = `${path}.${process.pid}.temporaryPath`;
  writeFileSync(temporaryPath, JSON.stringify(toWire(credentials)), { mode: CREDENTIALS_FILE_MODE });
  chmodSync(temporaryPath, CREDENTIALS_FILE_MODE);
  renameSync(temporaryPath, path);
}

/** Returns the stored secret string for a keychain account name, or null when absent. */
export type KeychainReader = (service: string, account: string) => Promise<string | null>;

/**
 * Decodes a value written by the Go `keyring` library.
 *
 * Values may carry a `go-keyring-base64:` or `go-keyring-encoded:` (hex) prefix;
 * anything else is returned trimmed.
 *
 * @param raw - Raw value from the keychain.
 * @returns The plain text value.
 */
export function decodeKeyringValue(raw: string): string {
  const trimmedValue = raw.trim();
  if (trimmedValue.startsWith(KEYRING_BASE64_PREFIX)) {
    return Buffer.from(trimmedValue.slice(KEYRING_BASE64_PREFIX.length), "base64").toString("utf8");
  }
  if (trimmedValue.startsWith(KEYRING_HEX_PREFIX)) {
    return Buffer.from(trimmedValue.slice(KEYRING_HEX_PREFIX.length), "hex").toString("utf8");
  }
  return trimmedValue;
}

/**
 * Keychain reader backed by the macOS `security` tool.
 *
 * Returns null on other platforms and on any lookup failure (for example a
 * missing item or a denied prompt).
 */
export const macKeychainReader: KeychainReader = async (service, account) => {
  if (process.platform !== "darwin") {
    return null;
  }
  try {
    const { stdout } = await execFileAsync(SECURITY_BIN, ["find-generic-password", "-s", service, "-wa", account]);
    return decodeKeyringValue(stdout);
  } catch {
    return null;
  }
};

/**
 * Resolves conduit's config directory.
 *
 * @param env - Environment to read `CONDUIT_CONFIG_DIR` from.
 * @returns The override, or `~/.conduit`.
 */
export function conduitDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.CONDUIT_CONFIG_DIR || join(homedir(), ".conduit");
}

/**
 * Resolves conduit's file-based credential cache, used when the keychain has no entry.
 *
 * @returns `~/.claude/.conduit-credentials.json`.
 */
export function conduitFileStoragePath(): string {
  return join(homedir(), ".claude", ".conduit-credentials.json");
}

/** One account entry in conduit's settings. */
interface AccountEntryJson {
  email?: string;
  kind?: string;
}

/** The `accounts` section of conduit's settings. */
interface AccountStoreJson {
  active?: string;
  accounts?: Record<string, AccountEntryJson>;
}

/**
 * Extracts the account kind from an `active` id shaped like `kind:email`.
 *
 * @param active - The active account id.
 * @returns The kind prefix, or undefined when the id has no recognized prefix.
 */
function kindPrefixOf(active: string): string | undefined {
  const colon = active.indexOf(":");
  if (colon <= 0) {
    return undefined;
  }
  const prefix = active.slice(0, colon);
  if (ACCOUNT_KINDS.includes(prefix)) {
    return prefix;
  }
  return undefined;
}

/**
 * Lists candidate keychain account names for the active conduit account, most specific first.
 *
 * Conduit has written the active account id in several shapes over time: a
 * `kind:email` string, a bare email, or a key into the `accounts` map. All are
 * resolved to a `kind` and `email`, and both the kind-qualified and the bare
 * key names are tried.
 *
 * @param store - The `accounts` section of conduit's settings.
 * @returns Keychain account names to try in order, empty when no account is active.
 */
export function conduitKeyCandidates(store: AccountStoreJson | undefined): string[] {
  const active = store?.active;
  if (!active) {
    return [];
  }
  const accounts = store?.accounts ?? {};
  const entry = accounts[active];
  const prefixKind = kindPrefixOf(active);
  let kind = DEFAULT_ACCOUNT_KIND;
  let email = active;

  if (entry?.email) {
    email = entry.email;
    kind = entry.kind || prefixKind || DEFAULT_ACCOUNT_KIND;
  } else if (prefixKind !== undefined) {
    kind = prefixKind;
    email = active.slice(active.indexOf(":") + 1);
  } else {
    const byEmail = Object.values(accounts).find((account) => account.email === active);
    if (byEmail?.kind) {
      kind = byEmail.kind;
    }
  }
  return [`oauth-tokens-${kind}:${email}`, `oauth-tokens-${email}`];
}

/** Options for {@link readConduitCredentials}. */
export interface ConduitReadOptions {
  /** Keychain reader. Defaults to {@link macKeychainReader}. */
  readonly keychain?: KeychainReader;
  /** Environment used to locate conduit's config directory. */
  readonly env?: NodeJS.ProcessEnv;
  /** Path of the file-based credential cache. Defaults to {@link conduitFileStoragePath}. */
  readonly fileStoragePath?: string;
}

/**
 * Parses a credential blob stored by conduit.
 *
 * @param raw - JSON text.
 * @returns The credentials, or null when the blob holds no usable token.
 */
function parseConduitBlob(raw: string): StoredCredentials | null {
  return fromWire(JSON.parse(raw) as WireCredentials);
}

/**
 * Reads the active conduit account's credentials without modifying anything.
 *
 * Looks in the keychain first, then in conduit's file cache. This server never
 * writes to conduit's store.
 *
 * @param options - See {@link ConduitReadOptions}.
 * @returns The credentials, or null when conduit is not set up or has no entry.
 * @throws {Error} When conduit's settings file exists but cannot be parsed.
 */
export async function readConduitCredentials(options: ConduitReadOptions = {}): Promise<StoredCredentials | null> {
  const keychain = options.keychain ?? macKeychainReader;
  const settingsPath = join(conduitDir(options.env), "conduit.json");
  if (!existsSync(settingsPath)) {
    return null;
  }

  let store: AccountStoreJson | undefined;
  try {
    store = (JSON.parse(readFileSync(settingsPath, "utf8")) as { accounts?: AccountStoreJson }).accounts;
  } catch (err) {
    throw new Error(`Failed to read ${settingsPath}: ${(err as Error).message}`);
  }

  const keys = conduitKeyCandidates(store);
  if (keys.length === 0) {
    return null;
  }

  for (const key of keys) {
    const raw = await keychain(CONDUIT_KEYCHAIN_SERVICE, key);
    if (raw) {
      return parseConduitBlob(raw);
    }
  }

  const filePath = options.fileStoragePath ?? conduitFileStoragePath();
  if (!existsSync(filePath)) {
    return null;
  }
  const cache = JSON.parse(readFileSync(filePath, "utf8")) as Record<string, string>;
  for (const key of keys) {
    const raw = cache[`${CONDUIT_FILE_SERVICE}\u0000${key}`];
    if (raw) {
      return parseConduitBlob(raw);
    }
  }
  return null;
}
