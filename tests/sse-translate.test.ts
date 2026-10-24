import { describe, expect, it } from "vitest";
import { translateSSE } from "../src/http-support.js";

/** Event stream as consumed by the translator. */
function events(...pairs: Array<{ event: string; data: string }>): AsyncGenerator<{ event: string; data: string }> {
  return (async function* () {
    for (const pair of pairs) yield pair;
  })();
}

/** A parsed `chat.completion.chunk` response. */
interface Chunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{ index: number; delta: Record<string, unknown>; finish_reason: string | null }>;
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

/** Collects translated SSE lines into full parsed chunks (usage preserved). */
async function collect(lines: AsyncGenerator<string>): Promise<Chunk[]> {
  const chunks: Chunk[] = [];
  for await (const line of lines) {
    const payload = line.slice("data: ".length).trim();
    if (payload === "[DONE]") continue;
    chunks.push(JSON.parse(payload) as Chunk);
  }
  return chunks;
}

describe("translateSSE", () => {
  it("emits a role chunk, content chunks, finish reason and usage", async () => {
    const upstream = events(
      { event: "message_start", data: '{"type":"message","id":"m","usage":{"input_tokens":5}}' },
      { event: "content_block_start", data: '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}' },
      { event: "content_block_delta", data: '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"he"}}' },
      { event: "content_block_delta", data: '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"llo"}}' },
      { event: "content_block_stop", data: '{"type":"content_block_stop","index":0}' },
      { event: "message_delta", data: '{"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"output_tokens":3}}}' },
      { event: "message_stop", data: '{"type":"message_stop"}' },
    );

    const chunks = await collect(
      translateSSE(upstream, "chatcmpl-1", "claude-sonnet-5-5", 42),
    );

    expect(chunks[0]).toMatchObject({ choices: [{ delta: { role: "assistant" }, finish_reason: null }] });
    expect(chunks[1]).toMatchObject({ choices: [{ delta: { content: "he" }, finish_reason: null }] });
    expect(chunks[2]).toMatchObject({ choices: [{ delta: { content: "llo" }, finish_reason: null }] });
    const last = chunks[chunks.length - 1]!;
    expect(last.choices[0]).toEqual({ index: 0, delta: {}, finish_reason: "stop" });
    expect(last.usage).toEqual({ prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 });
  });

  it("carries the completion id, object and model on every chunk", async () => {
    const upstream = events(
      { event: "message_start", data: '{"type":"message","id":"m","usage":{}}' },
      { event: "message_delta", data: '{"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"output_tokens":1}}}' },
    );
    const chunks = await collect(translateSSE(upstream, "chatcmpl-abc", "claude-opus-5-5", 7));
    for (const chunk of chunks) {
      expect(chunk.id).toBe("chatcmpl-abc");
      expect(chunk.object).toBe("chat.completion.chunk");
      expect(chunk.created).toBe(7);
      expect(chunk.model).toBe("claude-opus-5-5");
    }
  });

  it("translates tool_use blocks into a tool_calls delta", async () => {
    const ev = (event: string, data: unknown) => ({ event, data: JSON.stringify(data) });
    const upstream = events(
      ev("message_start", { type: "message", id: "m", usage: {} }),
      ev("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "call_1", name: "search" } }),
      ev("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: '{"q":' } }),
      ev("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: '"hi"}' } }),
      ev("content_block_stop", { type: "content_block_stop", index: 0, content_block: { type: "tool_use", id: "call_1", name: "search" } }),
      ev("message_delta", { type: "message_delta", delta: { stop_reason: "tool_use", usage: { output_tokens: 2 } } }),
    );
    const chunks = await collect(translateSSE(upstream, "chatcmpl-1", "claude-sonnet-5-5", 1));
    const toolChunk = chunks.find((c) => c.choices?.[0]?.delta?.tool_calls);
    const toolChoice = toolChunk!.choices.find((c) => c.delta?.tool_calls) as unknown as { delta: { tool_calls: Array<Record<string, unknown>> } };
    expect(toolChoice).toBeDefined();
    expect(toolChoice.delta.tool_calls[0]).toEqual({
      index: 0,
      id: "call_1",
      type: "function",
      function: { name: "search", arguments: '{"q":"hi"}' },
    });
    const finish = chunks.find((c) => c.choices[0]?.finish_reason);
    expect(finish!.choices[0]!.finish_reason).toBe("tool_calls");
  });

  it("gives parallel tool_use blocks distinct tool_calls indices", async () => {
    const ev = (event: string, data: unknown) => ({ event, data: JSON.stringify(data) });
    const upstream = events(
      ev("message_start", { type: "message", id: "m", usage: {} }),
      ev("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "toolu_a", name: "search" } }),
      ev("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: '{"q":"a"}' } }),
      ev("content_block_stop", { type: "content_block_stop", index: 0 }),
      ev("content_block_start", { type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "toolu_b", name: "fetch" } }),
      ev("content_block_delta", { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: '{"u":"b"}' } }),
      ev("content_block_stop", { type: "content_block_stop", index: 1 }),
      ev("message_delta", { type: "message_delta", delta: { stop_reason: "tool_use", usage: { output_tokens: 2 } } }),
    );
    const chunks = await collect(translateSSE(upstream, "chatcmpl-1", "claude-sonnet-5-5", 1));
    const toolDeltas = chunks.map((c) => c.choices[0]?.delta.tool_calls).filter((d) => d !== undefined) as Array<Array<Record<string, unknown>>>;
    expect(toolDeltas).toEqual([
      [{ index: 0, id: "toolu_a", type: "function", function: { name: "search", arguments: '{"q":"a"}' } }],
      [{ index: 1, id: "toolu_b", type: "function", function: { name: "fetch", arguments: '{"u":"b"}' } }],
    ]);
  });

  it("maps stop reasons through the shared mapping", async () => {
    for (const [stop, expected] of [["max_tokens", "length"], ["stop_sequence", "stop"], ["refusal", "content_filter"]] as const) {
      const upstream = events(
        { event: "message_start", data: '{"type":"message","id":"m","usage":{}}' },
        { event: "message_delta", data: `{"type":"message_delta","delta":{"stop_reason":"${stop}","usage":{"output_tokens":1}}}` },
      );
      const chunks = await collect(translateSSE(upstream, "c", "m", 1));
      expect(chunks[chunks.length - 1]!.choices[0]!.finish_reason).toBe(expected);
    }
  });

  it("emits the [DONE] sentinel", async () => {
    const upstream = events(
      { event: "message_start", data: '{"type":"message","id":"m","usage":{}}' },
      { event: "message_delta", data: '{"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"output_tokens":1}}}' },
    );
    let lastLine = "";
    for await (const line of translateSSE(upstream, "c", "m", 1)) {
      lastLine = line.trim();
    }
    expect(lastLine).toBe("data: [DONE]");
  });

  it("ignores events it does not understand", async () => {
    const upstream = events(
      { event: "message_start", data: '{"type":"message","id":"m","usage":{}}' },
      { event: "ping", data: '{"type":"ping"}' },
      { event: "message_delta", data: '{"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"output_tokens":1}}}' },
    );
    const chunks = await collect(translateSSE(upstream, "c", "m", 1));
    expect(chunks.length).toBe(2);
  });
});
