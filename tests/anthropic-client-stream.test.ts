import { afterEach, describe, expect, it } from "vitest";
import type { FetchLike } from "../src/anthropic/tokens.js";
import { AnthropicClient } from "../src/anthropic/client.js";

/** A fake credential store that returns a stable access token. */
const credentials = {
  async get() {
    return { accountKind: "anthropic-console", accessToken: "tok" } as never;
  },
  async refreshAfter401() {
    return { accountKind: "anthropic-console", accessToken: "tok" } as never;
  },
};

/** Splits raw SSE text into { event, data } pairs the same way the client does. */
function parseSse(raw: string): Array<{ event: string; data: string }> {
  const events: Array<{ event: string; data: string }> = [];
  for (const block of raw.split("\n\n")) {
    const trimmed = block.trimEnd();
    if (trimmed === "") continue;
    let event = "";
    const lines = trimmed.split("\n");
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        event = line.slice("event: ".length).trim();
      } else if (line.startsWith("data: ")) {
        events.push({ event, data: line.slice("data: ".length) });
      }
    }
  }
  return events;
}

describe("AnthropicClient.createMessageStream", () => {
  it("streams parsed SSE events as an async iterable", async () => {
    const raw = [
      'event: message_start',
      "data: {\"type\":\"message\",\"id\":\"m\"}",
      "",
      'event: content_block_start',
      "data: {\"type\":\"content_block_start\",\"index\":0}",
      "",
      'event: content_block_delta',
      "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"he\"}}",
      "",
      'event: content_block_delta',
      "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"llo\"}}",
      "",
      'event: content_block_stop',
      "data: {\"type\":\"content_block_stop\",\"index\":0}",
      "",
      'event: message_delta',
      "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"usage\":{\"output_tokens\":2}}}",
      "",
      "event: message_stop",
      "data: {\"type\":\"message_stop\"}",
      "",
    ].join("\n");

    const fetchImpl: FetchLike = async (url, init) => {
      expect(String(url)).toBe("https://api.anthropic.com/v1/messages?beta=true");
      const body = JSON.parse(init!.body as string);
      expect(body.stream).toBe(true);
      return new Response(raw, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
    };

    const client = new AnthropicClient({ credentials, fetch: fetchImpl });
    const collected: Array<{ event: string; data: string }> = [];
    for await (const event of client.createMessageStream({
      model: "claude-sonnet-5-5",
      maxTokens: 256,
      messages: [{ role: "user", content: "hi" }],
    })) {
      collected.push(event);
    }
    expect(collected).toEqual(parseSse(raw));
  });

  it("does not retry a streaming call", async () => {
    let calls = 0;
    const fetchImpl: FetchLike = async () => {
      calls++;
      return new Response("event: message_start\ndata: {}\n\n", {
        status: 500,
        headers: { "content-type": "text/event-stream" },
      });
    };
    const client = new AnthropicClient({
      credentials,
      fetch: fetchImpl,
      maxRetries: 5,
    });
    await expect(
      (async () => {
        for await (const _ of client.createMessageStream({
          model: "claude-sonnet-5-5",
          maxTokens: 256,
          messages: [{ role: "user", content: "hi" }],
        })) {
          // drain
        }
      })(),
    ).rejects.toThrow(/api: 500/);
    expect(calls).toBe(1);
  });

  it("aborts when the caller signal fires", async () => {
    const abort = new AbortController();
    const fetchImpl: FetchLike = async (_url, init) => {
      init!.signal!.addEventListener("abort", () => {
        throw new Error("request cancelled by the caller");
      });
      // A slow stream the abort interrupts.
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode("event: message_start\ndata: {}\n\n"));
          },
        }),
        { status: 200, headers: { "content-type": "text/event-stream" } },
      );
    };
    const client = new AnthropicClient({ credentials, fetch: fetchImpl });
    const iterator = client.createMessageStream({
      model: "claude-sonnet-5-5",
      maxTokens: 256,
      messages: [{ role: "user", content: "hi" }],
    }, abort.signal);
    abort.abort();
    await expect(async () => {
      for await (const _ of iterator) {
        // drain
      }
    }).rejects.toThrow(/cancelled/);
  });

  it("aborts mid-stream when the caller signal fires", async () => {
    const abort = new AbortController();
    const fetchImpl: FetchLike = async (_url, init) => {
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode("event: message_start\ndata: {}\n\n"));
            // Never closes; the abort interrupts the read.
          },
        }),
        { status: 200, headers: { "content-type": "text/event-stream" } },
      );
    };
    const client = new AnthropicClient({ credentials, fetch: fetchImpl });
    const iterator = client.createMessageStream({
      model: "claude-sonnet-5-5",
      maxTokens: 256,
      messages: [{ role: "user", content: "hi" }],
    }, abort.signal);
    // First event is available, then the read blocks until abort.
    const first = await iterator.next();
    expect(first.done).toBe(false);
    abort.abort();
    await expect(iterator.next()).rejects.toThrow(/cancelled/);
  });
});
