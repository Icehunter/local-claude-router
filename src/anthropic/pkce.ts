import { createHash, randomBytes } from "node:crypto";

/** Number of random bytes behind both the PKCE verifier and the OAuth state value. */
const RANDOM_TOKEN_BYTES = 32;

/**
 * Creates a PKCE code verifier (RFC 7636).
 *
 * @returns A 43 character base64url string made from 32 random bytes.
 * @example
 * ```ts
 * const verifier = generateVerifier();
 * const challenge = s256Challenge(verifier);
 * ```
 */
export function generateVerifier(): string {
  return randomBytes(RANDOM_TOKEN_BYTES).toString("base64url");
}

/**
 * Creates the OAuth `state` value that ties a browser callback to the login that started it.
 *
 * @returns A 43 character base64url string made from 32 random bytes.
 */
export function generateState(): string {
  return randomBytes(RANDOM_TOKEN_BYTES).toString("base64url");
}

/**
 * Derives the `S256` PKCE code challenge for a verifier.
 *
 * @param verifier - The verifier returned by {@link generateVerifier}.
 * @returns The base64url encoded SHA-256 digest of the verifier.
 */
export function s256Challenge(verifier: string): string {
  return createHash("sha256").update(verifier, "utf8").digest("base64url");
}
