import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import type { MessageRequest } from "../src/anthropic/client.js";
import { assertSafeBind, createOpenAIHttpServer, effortFromEnv, thinkingFromEnv, toChatCompletion, translateRequest } from "../src/openai-http.js";

const defaults = { defaultModel: "claude-sonnet-5-5", defaultMaxTokens: 4096 };

type HttpServer = ReturnType<typeof createOpenAIHttpServer>;

/** Builds an upstream response whose only content is one text block. */
function textResponse(model: string, text: string, usage?: { input_tokens: number; output_tokens: number }) {
  return {
    requestId: null,
    response: {
      id: "m",
      type: "message",
      role: "assistant",
      model,
      content: [{ type: "text", text }],
      stop_reason: "end_turn",
      usage,
    },
  };
}

/** Starts listening on a free loopback port and returns the base URL. */
async function listen(server: HttpServer): Promise<string> {
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

describe("translateRequest", () => {
  it("lifts system messages and merges consecutive same-role turns", () => {
    const translated = translateRequest(
      { messages: [{ role: "system", content: "be brief" }, { role: "user", content: "a" }, { role: "user", content: [{ type: "text", text: "b" }] }] },
      defaults,
    );
    expect(translated.system).toBe("be brief");
    expect(translated.messages).toEqual([{ role: "user", content: "a\n\nb" }]);
  });

  it("maps non-Claude model names to the default and drops temperature", () => {
    const translated = translateRequest({ model: "gpt-4o", temperature: 1.8, max_tokens: 50, messages: [{ role: "user", content: "x" }] }, defaults);
    expect(translated).toMatchObject({ model: "claude-sonnet-5-5", maxTokens: 50 });
    expect(translated).not.toHaveProperty("temperature");
  });

  it("keeps claude model names", () => {
    expect(translateRequest({ model: "claude-opus-5-5", messages: [{ role: "user", content: "x" }] }, defaults).model).toBe("claude-opus-5-5");
  });

  it.each([
    [{ messages: [{ role: "user", content: "x" }], tools: [] }],
    [{ messages: [] }],
    [{ messages: [{ role: "assistant", content: "x" }] }],
    [{ messages: [{ role: "user", content: [{ type: "image_url" }] }] }],
    [{ messages: [{ role: "user", content: "x" }], max_tokens: 0 }],
  ])("rejects %j", (body) => {
    expect(() => translateRequest(body, defaults)).toThrow();
  });
});

describe("toChatCompletion", () => {
  it("maps text, finish reason and usage", () => {
    const out = toChatCompletion(
      {
        requestId: null,
        response: {
          id: "msg_1", type: "message", role: "assistant", model: "claude-sonnet-5-5",
          content: [{ type: "text", text: "hi" }], stop_reason: "max_tokens", usage: { input_tokens: 3, output_tokens: 2 },
        },
      },
      100,
    );
    expect(out.choices[0]).toEqual({ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "length" });
    expect(out.usage).toEqual({ prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 });
  });
});

describe("http server", () => {
  const servers: HttpServer[] = [];
  afterEach(() => {
    servers.splice(0).forEach((server) => server.close());
  });

  async function start(apiKey?: string) {
    const seen: MessageRequest[] = [];
    const server = createOpenAIHttpServer({
      ...defaults,
      apiKey,
      now: () => 1,
      client: {
        async createMessage(request) {
          seen.push(request);
          return textResponse(request.model, "pong", { input_tokens: 1, output_tokens: 1 });
        },
      },
    });
    servers.push(server);
    return { url: await listen(server), seen };
  }

  const post = (url: string, body: unknown, headers: Record<string, string> = {}) =>
    fetch(`${url}/v1/chat/completions`, { method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify(body) });

  it("serves a chat completion", async () => {
    const { url, seen } = await start();
    const response = await post(url, { model: "gpt-4o", messages: [{ role: "user", content: "ping" }] });
    expect(response.status).toBe(200);
    const json = (await response.json()) as { choices: Array<{ message: { content: string } }> };
    expect(json.choices[0]!.message.content).toBe("pong");
    expect(seen[0]!.model).toBe("claude-sonnet-5-5");
  });

  it("streams a single chunk then [DONE]", async () => {
    const { url } = await start();
    const response = await post(url, { stream: true, messages: [{ role: "user", content: "ping" }] });
    expect(response.headers.get("content-type")).toBe("text/event-stream");
    const text = await response.text();
    expect(text).toContain('"content":"pong"');
    expect(text.trimEnd().endsWith("data: [DONE]")).toBe(true);
  });

  it("enforces the API key and returns OpenAI-shaped errors", async () => {
    const { url } = await start("secret");
    const denied = await post(url, { messages: [{ role: "user", content: "x" }] });
    expect(denied.status).toBe(401);
    const bad = await post(url, { messages: [] }, { authorization: "Bearer secret" });
    expect(bad.status).toBe(400);
    expect(((await bad.json()) as { error: { type: string } }).error.type).toBe("invalid_request_error");
  });

  it("lists models", async () => {
    const { url } = await start();
    const json = (await (await fetch(`${url}/v1/models`)).json()) as { data: Array<{ id: string }> };
    expect(json.data[0]!.id).toBe("claude-sonnet-5-5");
  });

  it("answers /health without credentials checks failing open", async () => {
    const { url } = await start("secret");
    expect((await fetch(`${url}/health`)).status).toBe(200);
    expect((await fetch(`${url}/v1/models`, { headers: { authorization: "Bearer secre" } })).status).toBe(401);
    expect((await fetch(`${url}/v1/models`, { headers: { authorization: "Bearer secret" } })).status).toBe(200);
  });

  it("returns 404 for unknown routes and 400 for bad JSON", async () => {
    const { url } = await start();
    expect((await fetch(`${url}/nope`)).status).toBe(404);
    const bad = await fetch(`${url}/v1/chat/completions`, { method: "POST", body: "{not json" });
    expect(bad.status).toBe(400);
    expect((await fetch(`${url}/v1/chat/completions`)).status).toBe(405);
  });

  it("forwards thinking and effort to the client", async () => {
    const seen: MessageRequest[] = [];
    const server = createOpenAIHttpServer({
      ...defaults,
      thinking: { type: "between_tools" },
      effort: "low",
      client: {
        async createMessage(request) {
          seen.push(request);
          return textResponse(request.model, "x");
        },
      },
    });
    servers.push(server);
    const url = await listen(server);
    await post(url, { messages: [{ role: "user", content: "x" }] });
    expect(seen[0]).toMatchObject({ thinking: { type: "between_tools" }, effort: "low" });
  });
});

describe("config helpers", () => {
  it("maps thinking env values", () => {
    expect(thinkingFromEnv(undefined)).toEqual({ type: "between_tools" });
    expect(thinkingFromEnv("adaptive")).toEqual({ type: "adaptive" });
    expect(thinkingFromEnv("default")).toBeUndefined();
    expect(() => thinkingFromEnv("maybe")).toThrow(/CLAUDE_WIRE_THINKING/);
  });

  it("validates effort", () => {
    expect(effortFromEnv(undefined)).toBeUndefined();
    expect(effortFromEnv("LOW")).toBe("low");
    expect(() => effortFromEnv("extreme")).toThrow(/CLAUDE_WIRE_EFFORT/);
  });

  it("refuses a non-loopback bind without an API key", () => {
    expect(() => assertSafeBind("0.0.0.0", undefined)).toThrow(/CLAUDE_WIRE_API_KEY/);
    expect(() => assertSafeBind("0.0.0.0", "k")).not.toThrow();
    expect(() => assertSafeBind("127.0.0.1", undefined)).not.toThrow();
  });
});
