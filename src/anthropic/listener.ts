import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";

/** Loopback address the callback server binds to. */
const LOOPBACK_HOST = "127.0.0.1";

/** Port value that asks the OS for any free port. */
const ANY_FREE_PORT = 0;

/** A one-shot local HTTP server that receives the OAuth redirect. */
export interface CallbackListener {
  /** Port the server is listening on. */
  readonly port: number;

  /**
   * Waits for the authorization code.
   *
   * @param signal - Optional abort signal that cancels the wait.
   * @returns The authorization code from a callback with a matching state.
   * @throws {Error} When the callback has no code, the state does not match,
   *   or `signal` aborts first.
   */
  wait(signal?: AbortSignal): Promise<string>;

  /**
   * Answers the browser request that delivered the code, sending it on to `url`.
   * Does nothing if no request is waiting for an answer.
   *
   * @param url - Page to send the browser to.
   */
  redirect(url: string): void;

  /**
   * Releases the browser request if one is still waiting, then stops the server
   * and drops open connections.
   */
  close(): Promise<void>;
}

/**
 * Settles the callback promise at most once and remembers that it did.
 */
interface CallbackOutcome {
  readonly result: Promise<string>;
  isSettled(): boolean;
  resolve(code: string): void;
  reject(err: Error): void;
}

/**
 * Creates the promise that the HTTP handler settles when the callback arrives.
 *
 * @returns The promise plus resolve, reject and settled-state accessors.
 */
function createCallbackOutcome(): CallbackOutcome {
  let settled = false;
  let resolveCode!: (code: string) => void;
  let rejectCode!: (err: Error) => void;
  const result = new Promise<string>((resolvePromise, rejectPromise) => {
    resolveCode = (code) => {
      settled = true;
      resolvePromise(code);
    };
    rejectCode = (err) => {
      settled = true;
      rejectPromise(err);
    };
  });
  // wait() may never be called (for example when the login is abandoned);
  // do not surface that as an unhandled rejection.
  result.catch(() => {});
  return {
    result,
    isSettled: () => settled,
    resolve: resolveCode,
    reject: rejectCode,
  };
}

/**
 * Wraps a callback promise so a wait can be cancelled without cancelling the
 * callback itself.
 *
 * @param result - The callback promise.
 * @param signal - Optional abort signal.
 * @returns `result` itself when there is no signal, otherwise a promise that
 *   rejects with `callback wait cancelled` on abort.
 */
function waitWithSignal(result: Promise<string>, signal: AbortSignal | undefined): Promise<string> {
  if (!signal) {
    return result;
  }
  return new Promise<string>((resolve, reject) => {
    const onAbort = (): void => reject(new Error("callback wait cancelled"));
    if (signal.aborted) {
      onAbort();
      return;
    }
    signal.addEventListener("abort", onAbort, { once: true });
    result.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
  });
}

/**
 * Starts a loopback HTTP server that waits for the OAuth redirect.
 *
 * The server binds to `127.0.0.1` on a free port. A request is accepted only on
 * `callbackPath` with a `code` and a `state` equal to `expectedState`. The
 * browser response for the accepted request is held open until {@link
 * CallbackListener.redirect} or {@link CallbackListener.close} is called, so the
 * caller can send the user to a success page after the token exchange finishes.
 *
 * @param expectedState - State value the callback must echo back.
 * @param callbackPath - Path that receives the redirect. Defaults to `/callback`.
 * @returns A listener that is already accepting connections.
 * @throws {Error} When `expectedState` is empty or the server cannot bind.
 * @example
 * ```ts
 * const listener = await startCallbackListener(state);
 * const url = buildAuthUrl({ codeChallenge, state, port: listener.port });
 * const code = await listener.wait();
 * listener.redirect("https://example.com/success");
 * await listener.close();
 * ```
 */
export async function startCallbackListener(
  expectedState: string,
  callbackPath = "/callback",
): Promise<CallbackListener> {
  if (!expectedState) {
    throw new Error("startCallbackListener: empty state");
  }

  const outcome = createCallbackOutcome();
  let parked: ServerResponse | null = null;

  /**
   * Handles one request: validates it and, if it carries the code, parks the
   * response until the caller redirects it.
   */
  const handleRequest = (request: IncomingMessage, response: ServerResponse): void => {
    const url = new URL(request.url ?? "/", "http://localhost");
    if (url.pathname !== callbackPath) {
      response.writeHead(404).end("Not found");
      return;
    }
    const code = url.searchParams.get("code");
    const state = url.searchParams.get("state");
    if (!code) {
      response.writeHead(400).end("Authorization code not found");
      if (!outcome.isSettled()) {
        outcome.reject(new Error("callback received without code"));
      }
      return;
    }
    if (state !== expectedState) {
      response.writeHead(400).end("Invalid state parameter");
      if (!outcome.isSettled()) {
        outcome.reject(new Error("callback state mismatch"));
      }
      return;
    }
    if (outcome.isSettled()) {
      response.writeHead(200).end("ok");
      return;
    }
    parked = response;
    outcome.resolve(code);
  };

  const server: Server = createServer(handleRequest);

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(ANY_FREE_PORT, LOOPBACK_HOST, () => {
      server.off("error", reject);
      resolve();
    });
  });
  const port = (server.address() as AddressInfo).port;

  /** Finishes the parked browser request: a redirect when `location` is given, otherwise 204. */
  const completeParked = (location?: string): void => {
    const response = parked;
    parked = null;
    if (!response) {
      return;
    }
    if (location) {
      response.writeHead(302, { Location: location }).end();
      return;
    }
    response.writeHead(204).end();
  };

  return {
    port,
    wait: (signal) => waitWithSignal(outcome.result, signal),
    redirect: (url) => completeParked(url),
    async close() {
      completeParked();
      await new Promise<void>((resolve) => {
        server.close(() => resolve());
        server.closeAllConnections();
      });
    },
  };
}
