#!/usr/bin/env node
import { depsFromEnv } from "./deps.js";
import type { Deps } from "./deps.js";
import { ownCredentialsPath } from "./anthropic/store.js";

/** Message shown when the command is missing or unknown. */
const USAGE = "usage: cli.js <login|status>";

/** Commands the CLI understands. */
const COMMANDS = ["login", "status"] as const;

/** One of {@link COMMANDS}. */
type Command = (typeof COMMANDS)[number];

/**
 * Checks whether an argument names a known command.
 *
 * @param arg - First command line argument.
 * @returns True for `login` and `status`.
 */
function isCommand(arg: string | undefined): arg is Command {
  return arg !== undefined && (COMMANDS as readonly string[]).includes(arg);
}

/**
 * Formats the ", expires ..." suffix of the status line.
 *
 * @param expiresAt - Expiry in epoch milliseconds, or 0 when unknown.
 * @returns The suffix, or an empty string when the expiry is unknown.
 */
function expirySuffix(expiresAt: number): string {
  if (expiresAt > 0) {
    return `, expires ${new Date(expiresAt).toISOString()}`;
  }
  return "";
}

/**
 * Formats the optional " (email)" part of the status line.
 *
 * @param email - Account email, possibly empty.
 * @returns The parenthesized email, or an empty string.
 */
function emailSuffix(email: string): string {
  if (email) {
    return ` (${email})`;
  }
  return "";
}

/**
 * Describes whether the stored credentials can be refreshed automatically.
 *
 * @param source - Where the credentials were found.
 * @param refreshToken - Stored refresh token, possibly empty.
 * @returns Text for the `Auto-refresh:` line.
 */
function autoRefreshText(source: string, refreshToken: string): string {
  const canRefresh = source === "own" && refreshToken !== "";
  if (canRefresh) {
    return "yes (refresh token stored)";
  }
  return "no";
}

/**
 * Prints which credentials are in use, without refreshing anything.
 *
 * @param deps - Shared services.
 */
async function status(deps: Deps): Promise<void> {
  const credentials = await deps.credentials.inspect();
  if (!credentials) {
    console.log("Credentials: none. Run `npm run login`, or sign in with conduit.");
    console.log(`Own credentials file: ${ownCredentialsPath()}`);
    return;
  }
  let readOnly = "";
  if (credentials.source === "conduit") {
    readOnly = ", read-only (never refreshed here)";
  }
  let validity = "valid";
  if (credentials.expired) {
    validity = "EXPIRED";
  }
  console.log(`Credentials: ${credentials.source}${emailSuffix(credentials.email)}, ${validity}${expirySuffix(credentials.expiresAt)}${readOnly}`);
  console.log(`Auto-refresh: ${autoRefreshText(credentials.source, credentials.refreshToken)}`);
  console.log(`Own credentials file: ${ownCredentialsPath()}`);
}

/**
 * Prints what to do when the browser could not be opened automatically.
 *
 * @param browserError - Reason reported by the launcher, if any.
 */
function printBrowserHint(browserError: string | undefined): void {
  if (browserError) {
    console.log(`Could not open the browser (${browserError}); open the URL manually.`);
    return;
  }
  console.log("Open the URL in a browser.");
}

/**
 * Runs the interactive browser sign-in and waits for it to finish.
 *
 * @param deps - Shared services.
 * @throws {Error} When the login fails or ends in an unexpected state.
 */
async function login(deps: Deps): Promise<void> {
  const started = await deps.login.start(true);
  console.log(`Sign in at:\n${started.url}\n`);
  if (!started.browserOpened) {
    printBrowserHint(started.browserError);
  }
  console.log("Waiting for the browser sign-in (5 minute timeout)...");
  await deps.login.settled();
  const state = deps.login.state;
  if (state.status === "complete") {
    console.log(`Signed in as ${state.email || "(unknown account)"}.`);
    return;
  }
  if (state.status === "failed") {
    throw new Error(state.error);
  }
  throw new Error(`login ended in state "${state.status}"`);
}

/**
 * Entry point: dispatches `login` or `status`.
 *
 * @throws {Error} With the usage text when the command is missing or unknown.
 */
async function main(): Promise<void> {
  const command = process.argv[2];
  if (!isCommand(command)) {
    throw new Error(USAGE);
  }
  const deps = depsFromEnv();
  if (command === "status") {
    await status(deps);
    return;
  }
  await login(deps);
}

main().catch((err) => {
  process.stderr.write(`${(err as Error).message}\n`);
  process.exit(1);
});
