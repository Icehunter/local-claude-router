import { describe, expect, it } from "vitest";
import { createOpenAIHttpServer } from "../src/openai-http.js";
import { AnthropicClient } from "../src/anthropic/client.js";

describe("diagnostic physical attempts", () => {
  it("disables transient retries for one request without changing the client default", async () => {
    let requests = 0;
    const client = new AnthropicClient({ credentials: { get: async () => ({ accessToken: "test-token", accountKind: "claude-pro" } as never), refreshAfter401: async () => { throw new Error("not expected"); } }, fetch: async () => { requests++; return new Response(JSON.stringify({ error: { message: "busy" } }), { status: 503 }); }, sleep: async () => {}, maxRetries: 2 });
    let attempts = 0;
    await expect(client.createMessage({ model: "claude-sonnet-5-5", maxTokens: 10, messages: [{ role: "user", content: "test" }] }, undefined, { singleAttempt: true, onAttempt: () => attempts++ })).rejects.toThrow();
    expect(requests).toBe(1); expect(attempts).toBe(1);
    requests = 0;
    await expect(client.createMessage({ model: "claude-sonnet-5-5", maxTokens: 10, messages: [{ role: "user", content: "test" }] })).rejects.toThrow();
    expect(requests).toBe(3);
  });
});

it("returns the actual upstream-attempt count for a diagnostic request", async () => {
  const server = createOpenAIHttpServer({ defaultModel: "claude-sonnet-5-5", defaultMaxTokens: 10, client: { async createMessage(_request, _signal, opts) { opts?.onAttempt?.(); return { requestId: null, response: { id: "test", type: "message", role: "assistant", model: "claude-sonnet-5-5", content: [{ type: "text", text: "ok" }], stop_reason: "end_turn" } }; }, async *createMessageStream() {} } });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const port = (server.address() as import("node:net").AddressInfo).port;
    const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: "POST", headers: { "content-type": "application/json", "x-pinkie-single-attempt": "1" }, body: JSON.stringify({ messages: [{ role: "user", content: "test" }] }) });
    expect(response.status).toBe(200); expect(response.headers.get("x-claude-wire-upstream-attempts")).toBe("1");
  } finally { await new Promise<void>((resolve) => server.close(() => resolve())); }
});
