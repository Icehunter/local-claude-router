import { createHash } from "node:crypto";
import { BILLING_CCH, BILLING_SALT, CLAUDE_CODE_VERSION, DEFAULT_ENTRYPOINT } from "./wire.js";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const PREV_REQUEST_ID_RE = /^req_[A-Za-z0-9_-]{1,36}$/;

/** Byte offsets of the first user message that feed the billing suffix hash. */
const BILLING_SAMPLE_OFFSETS = [4, 7, 20] as const;

/** Byte used in place of a sampled offset that lies past the end of the message ("0"). */
const BILLING_PAD_BYTE = 0x30;

/** Number of hex characters of the digest kept as the billing suffix. */
const BILLING_SUFFIX_LENGTH = 3;

/**
 * Checks whether a string is a UUID in canonical 8-4-4-4-12 hex form.
 *
 * @param candidate - Candidate string.
 * @returns True when `candidate` is a canonical UUID, case-insensitive.
 */
export function isCanonicalUuid(candidate: string): boolean {
  return UUID_RE.test(candidate);
}

/**
 * Computes the three character suffix of the `cc_version` field in the billing header.
 *
 * The hash input is the salt, three bytes sampled from the first user message,
 * and the client version. Offsets index UTF-8 bytes, not UTF-16 units, to match
 * the Go implementation this was ported from. A missing byte is padded with `"0"`.
 *
 * @param firstUserMessage - Text of the first user message.
 * @param version - Client version to hash in. Defaults to {@link CLAUDE_CODE_VERSION}.
 * @returns The first three hex characters of the SHA-256 digest.
 * @example
 * ```ts
 * computeBillingSuffix("hello world, this is a prompt"); // three hex characters
 * ```
 */
export function computeBillingSuffix(firstUserMessage: string, version: string = CLAUDE_CODE_VERSION): string {
  const bytes = Buffer.from(firstUserMessage, "utf8");
  const picked = Buffer.alloc(BILLING_SAMPLE_OFFSETS.length);
  BILLING_SAMPLE_OFFSETS.forEach((offset, index) => {
    const isInRange = offset < bytes.length;
    if (isInRange) {
      picked[index] = bytes[offset]!;
      return;
    }
    picked[index] = BILLING_PAD_BYTE;
  });
  const digest = createHash("sha256")
    .update(Buffer.concat([Buffer.from(BILLING_SALT, "utf8"), picked, Buffer.from(version, "utf8")]))
    .digest("hex");
  return digest.slice(0, BILLING_SUFFIX_LENGTH);
}

/** Optional per-request fields appended to the billing header text. */
export interface BillingContext {
  /** Canonical UUID identifying this prompt. Ignored when malformed. */
  readonly promptId?: string;
  /** Upstream id (`req_...`) of the previous request. Ignored when malformed. */
  readonly prevRequestId?: string;
  /** Marks the request as issued by a sub-agent. */
  readonly isSubAgent?: boolean;
}

/**
 * Builds the first system block, a text block that carries the billing header.
 *
 * Malformed `promptId` and `prevRequestId` values are dropped instead of sent,
 * so a bad id never reaches the wire.
 *
 * @param firstUserMessage - Text of the first user message, used for the version suffix.
 * @param billingContext - Optional billing context.
 * @returns A text block whose text ends with a newline.
 * @example
 * ```ts
 * const block = dynamicBillingBlock("hi", { promptId: randomUUID() });
 * ```
 */
export function dynamicBillingBlock(
  firstUserMessage: string,
  billingContext: BillingContext = {},
): { type: "text"; text: string } {
  const entrypoint = process.env.CLAUDE_CODE_ENTRYPOINT || DEFAULT_ENTRYPOINT;
  let text =
    `x-anthropic-billing-header: cc_version=${CLAUDE_CODE_VERSION}.${computeBillingSuffix(firstUserMessage)}; ` +
    `cc_entrypoint=${entrypoint}; cch=${BILLING_CCH};`;
  if (billingContext.isSubAgent) {
    text += " cc_is_subagent=true;";
  }
  const hasValidPrevRequestId = billingContext.prevRequestId !== undefined && PREV_REQUEST_ID_RE.test(billingContext.prevRequestId);
  if (hasValidPrevRequestId) {
    text += ` cc_prev_req=${billingContext.prevRequestId};`;
  }
  const hasValidPromptId = billingContext.promptId !== undefined && isCanonicalUuid(billingContext.promptId);
  if (hasValidPromptId) {
    text += ` cc_prompt_id=${billingContext.promptId};`;
  }
  return { type: "text", text: text + "\n" };
}
