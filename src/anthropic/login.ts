import { execFile } from "node:child_process";
import { buildAuthUrl } from "./authurl.js";
import type { CredentialManager } from "./credentials.js";
import { startCallbackListener } from "./listener.js";
import type { CallbackListener } from "./listener.js";
import { generateState, generateVerifier, s256Challenge } from "./pkce.js";
import { exchangeCode } from "./tokens.js";
import type { FetchLike } from "./tokens.js";
import { OAUTH_CONFIG } from "./wire.js";

/** Progress of the browser sign-in, as seen by callers polling {@link LoginController.state}. */
export type LoginState =
  | { status: "idle" }
  | { status: "pending"; url: string; startedAt: string }
  | { status: "complete"; email: string; finishedAt: string }
  | { status: "failed"; error: string; finishedAt: string };

/** Result of {@link LoginController.start}. */
export interface LoginStartResult {
  /** Authorize URL the user must visit. */
  readonly url: string;
  /** Whether a browser was launched. */
  readonly browserOpened: boolean;
  /** Reason the browser could not be launched, when it failed. */
  readonly browserError?: string;
}

/** Options for {@link LoginController}. Optional fields exist so tests can inject fakes. */
export interface LoginControllerOptions {
  /** Store that receives the tokens when the login completes. */
  readonly credentials: Pick<CredentialManager, "saveLogin">;
  /** Fetch implementation for the code exchange. */
  readonly fetch?: FetchLike;
  /** Opens a URL in the user's browser. Defaults to {@link systemOpenUrl}. */
  readonly openUrl?: (url: string) => Promise<void>;
  /** How long to wait for the browser callback. Defaults to five minutes. */
  readonly timeoutMs?: number;
  /** Clock in epoch milliseconds. */
  readonly now?: () => number;
}

/** Default time to wait for the user to finish signing in (five minutes). */
const DEFAULT_TIMEOUT_MS = 5 * 60_000;

/** A program name and its arguments. */
interface OpenCommand {
  readonly cmd: string;
  readonly args: string[];
}

/**
 * Chooses the platform command that opens a URL in the default browser.
 *
 * @param platform - Node platform identifier.
 * @param url - URL to open.
 * @returns The command and arguments to run.
 */
function openCommandFor(platform: string, url: string): OpenCommand {
  if (platform === "darwin") {
    return { cmd: "open", args: [url] };
  }
  if (platform === "win32") {
    // The empty string is the window title argument that `start` requires.
    return { cmd: "cmd", args: ["/c", "start", "", url] };
  }
  return { cmd: "xdg-open", args: [url] };
}

/**
 * Opens a URL in the default browser using the platform's launcher.
 *
 * @param url - URL to open.
 * @returns A promise that resolves once the launcher exits successfully.
 * @throws {Error} Rejects when the launcher is missing or exits with an error.
 */
export function systemOpenUrl(url: string): Promise<void> {
  const { cmd, args } = openCommandFor(process.platform, url);
  return new Promise((resolve, reject) => {
    execFile(cmd, args, (err) => {
      if (err) {
        reject(err);
        return;
      }
      resolve();
    });
  });
}

/**
 * Runs the browser based OAuth sign-in and saves the resulting tokens.
 *
 * {@link LoginController.start} returns the authorize URL straight away. The
 * rest of the flow (wait for the callback, exchange the code, save the tokens)
 * runs in the background; observe it through {@link LoginController.state} or
 * await {@link LoginController.settled}. Starting a new login cancels the
 * previous one.
 */
export class LoginController {
  private current: LoginState = { status: "idle" };
  private run: Promise<void> = Promise.resolve();
  private abort: AbortController | null = null;
  private readonly timeoutMs: number;
  private readonly now: () => number;

  /**
   * @param options - See {@link LoginControllerOptions}.
   */
  public constructor(private readonly options: LoginControllerOptions) {
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.now = options.now ?? Date.now;
  }

  /**
   * Current progress of the login.
   *
   * @returns `idle`, `pending`, `complete` or `failed`. A login cancelled by a
   *   newer one returns to `idle`.
   */
  public get state(): LoginState {
    return this.current;
  }

  /**
   * Resolves when the in-flight login (if any) has finished, successfully or not.
   *
   * @returns A promise that never rejects; check {@link LoginController.state} for the outcome.
   * @example
   * ```ts
   * await controller.start();
   * await controller.settled();
   * console.log(controller.state.status);
   * ```
   */
  public settled(): Promise<void> {
    return this.run;
  }

  /**
   * Begins a login and returns the authorize URL immediately; completion happens in the background.
   *
   * Any login already in progress is cancelled first.
   *
   * @param openBrowser - Launch the system browser at the URL. Defaults to true.
   * @returns The URL, and whether the browser was opened (with the reason if it was not).
   * @throws {Error} When the local callback server cannot start.
   * @example
   * ```ts
   * const started = await controller.start(true);
   * console.log(`Sign in at ${started.url}`);
   * ```
   */
  public async start(openBrowser = true): Promise<LoginStartResult> {
    this.abort?.abort();
    await this.run;

    const verifier = generateVerifier();
    const state = generateState();
    const listener = await startCallbackListener(state);
    const url = buildAuthUrl({
      codeChallenge: s256Challenge(verifier),
      state,
      port: listener.port,
      loginWithClaudeAi: true,
    });
    const abortController = new AbortController();
    this.abort = abortController;
    this.current = { status: "pending", url, startedAt: this.timestamp() };
    this.run = this.complete(listener, verifier, state, abortController);

    if (!openBrowser) {
      return { url, browserOpened: false };
    }
    try {
      await (this.options.openUrl ?? systemOpenUrl)(url);
      return { url, browserOpened: true };
    } catch (err) {
      return { url, browserOpened: false, browserError: (err as Error).message };
    }
  }

  /**
   * Current time as an ISO string, from the injected clock.
   *
   * @returns An ISO 8601 timestamp.
   */
  private timestamp(): string {
    return new Date(this.now()).toISOString();
  }

  /**
   * Background half of the login: waits for the callback, exchanges the code and saves the tokens.
   *
   * The browser is sent to the success page once the exchange has finished,
   * whether or not it succeeded. A cancellation (by a newer login) resets the
   * state to `idle` instead of reporting a failure. The listener is always closed.
   *
   * @param listener - Callback server for this login.
   * @param verifier - PKCE verifier matching the challenge in the URL.
   * @param state - State value the callback must echo.
   * @param abortController - Controller whose abort cancels this login.
   */
  private async complete(
    listener: CallbackListener,
    verifier: string,
    state: string,
    abortController: AbortController,
  ): Promise<void> {
    try {
      const signal = AbortSignal.any([abortController.signal, AbortSignal.timeout(this.timeoutMs)]);
      const code = await listener.wait(signal);
      try {
        const tokens = await exchangeCode(
          { code, state, codeVerifier: verifier, port: listener.port },
          this.options.fetch,
        );
        this.options.credentials.saveLogin(tokens);
        this.current = { status: "complete", email: tokens.email, finishedAt: this.timestamp() };
      } finally {
        listener.redirect(OAUTH_CONFIG.claudeAiSuccessUrl);
      }
    } catch (err) {
      if (abortController.signal.aborted) {
        this.current = { status: "idle" };
      } else {
        this.current = { status: "failed", error: (err as Error).message, finishedAt: this.timestamp() };
      }
    } finally {
      await listener.close();
    }
  }
}
