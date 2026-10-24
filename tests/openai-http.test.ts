import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import type { MessageRequest } from "../src/anthropic/client.js";
import { APIStatusError } from "../src/anthropic/client.js";
import { isHaikuModel } from "../src/anthropic/headers.js";
import { assertSafeBind, createOpenAIHttpServer, effortFromEnv, thinkingFromEnv, toChatCompletion, translateRequest } from "../src/openai-http.js";

const defaults = { defaultModel: "claude-sonnet-5-5", defaultMaxTokens: 4096 };

describe("isHaikuModel", () => {
  it.each(["claude-haiku-4-5", "claude-haiku-5-5", "claude-3-haiku-20240307", "claude-3-5-haiku-20241022", "anthropic/claude-haiku-4-5"])("flags haiku %s as haiku", (model) => {
    expect(isHaikuModel(model)).toBe(true);
  });

  it.each(["claude-sonnet-5-5", "claude-opus-5-5", "claude-pro"])("flags non-haiku %s as not haiku", (model) => {
    expect(isHaikuModel(model)).toBe(false);
  });
});

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

  it("forwards any explicit model unchanged and still rejects unsupported fields", () => {
    expect(translateRequest({model:"claude-haiku-4-5",messages:[{role:"user",content:"x"}]},defaults).model).toBe("claude-haiku-4-5");
    expect(translateRequest({model:"anthropic/claude-haiku-4-5[1m]",messages:[{role:"user",content:"x"}]},defaults).model).toBe("anthropic/claude-haiku-4-5[1m]");
    expect(() => translateRequest({stop:["x"],messages:[{role:"user",content:"x"}]},defaults)).toThrow("unsupported request field: stop");
    expect(translateRequest({max_tokens:50,messages:[{role:"user",content:"x"}]},defaults)).toMatchObject({model:"claude-sonnet-5-5",maxTokens:50});
  });

  it("falls back to the configured model for empty or non-string models", () => {
    expect(translateRequest({messages:[{role:"user",content:"x"}]},defaults).model).toBe(defaults.defaultModel);
    for (const model of [defaults.defaultModel, `${defaults.defaultModel}[1m]`]) expect(translateRequest({model,messages:[{role:"user",content:"x"}]},defaults).model).toBe(model);
    expect(translateRequest({model:"claude-opus-5-5",messages:[{role:"user",content:"x"}]},defaults).model).toBe("claude-opus-5-5");
    expect(() => translateRequest({model:"",messages:[{role:"user",content:"x"}]},defaults)).toThrow("model must be a non-empty string");
    expect(() => translateRequest({model:42,messages:[{role:"user",content:"x"}]},defaults)).toThrow("model must be a non-empty string");
  });

  it.each([
    [{ messages: [] }],
    [{ messages: [{ role: "assistant", content: "x" }] }],
    [{ messages: [{ role: "user", content: [{ type: "image_url" }] }] }],
    [{ messages: [{ role: "user", content: "x" }], max_tokens: 0 }],
    [{ messages: [{ role: "user", content: "x" }], tools: "search" }],
    [{ messages: [{ role: "user", content: "x" }], tools: [{ type: "custom", function: { name: "a" } }] }],
    [{ messages: [{ role: "user", content: "x" }], tools: [{ type: "function", function: { name: "a b" } }] }],
    [{ messages: [{ role: "user", content: "x" }], tools: [{ type: "function", function: { name: "a", description: 3 } }] }],
    [{ messages: [{ role: "user", content: "x" }], tools: [{ type: "function", function: { name: "a", parameters: [] } }] }],
    [{ messages: [{ role: "user", content: "x" }], tools: [{ type: "function", function: { name: "a" } }, { type: "function", function: { name: "a" } }] }],
    [{ messages: [{ role: "user", content: "x" }], tool_choice: "sometimes" }],
    [{ messages: [{ role: "user", content: "x" }], tool_choice: { type: "function", function: { name: "a" } } }],
    [{ messages: [{ role: "user", content: "x" }], tools: [{ type: "function", function: { name: "a" } }], tool_choice: { type: "function", function: { name: "b" } } }],
    [{ messages: [{ role: "user", content: "x" }], tool_choice: "required" }],
    [{ messages: [{ role: "user", content: "x" }], parallel_tool_calls: false }],
    [{ messages: [{ role: "user", content: "x" }], tools: [{ type: "function", function: { name: "a" } }], response_format: { type: "json_schema", json_schema: { name: "out", schema: { type: "object" } } } }],
  ])("rejects %j", (body) => {
    expect(() => translateRequest(body, defaults)).toThrow();
  });

  it("accepts an empty tools array and sends no tools upstream", () => {
    const translated = translateRequest({ messages: [{ role: "user", content: "x" }], tools: [] }, defaults);
    expect(translated.tools).toBeUndefined();
    expect(translated.toolChoice).toBeUndefined();
  });

  it("translates tools and tool_choice to the upstream shape", () => {
    const translated = translateRequest(
      {
        messages: [{ role: "user", content: "x" }],
        tools: [{ type: "function", function: { name: "search", description: "find things", parameters: { type: "object", properties: { q: { type: "string" } } } } }],
        tool_choice: "required",
      },
      defaults,
    );
    expect(translated.tools).toEqual([{ name: "search", description: "find things", input_schema: { type: "object", properties: { q: { type: "string" } } } }]);
    expect(translated.toolChoice).toEqual({ type: "any" });
  });

  it("defaults missing parameters to an empty object schema and omits tool_choice for auto", () => {
    const translated = translateRequest(
      { messages: [{ role: "user", content: "x" }], tools: [{ type: "function", function: { name: "ping" } }], tool_choice: "auto" },
      defaults,
    );
    expect(translated.tools).toEqual([{ name: "ping", input_schema: { type: "object", properties: {} } }]);
    expect(translated.toolChoice).toEqual({ type: "auto" });
  });

  it("maps a named tool_choice and drops tools for tool_choice none", () => {
    const named = translateRequest(
      { messages: [{ role: "user", content: "x" }], tools: [{ type: "function", function: { name: "ping" } }], tool_choice: { type: "function", function: { name: "ping" } } },
      defaults,
    );
    expect(named.toolChoice).toEqual({ type: "tool", name: "ping" });
    const none = translateRequest(
      { messages: [{ role: "user", content: "x" }], tools: [{ type: "function", function: { name: "ping" } }], tool_choice: "none" },
      defaults,
    );
    expect(none.tools).toBeUndefined();
    expect(none.toolChoice).toBeUndefined();
  });

  it("turns assistant tool_calls and tool messages into tool_use and tool_result blocks", () => {
    const translated = translateRequest(
      {
        messages: [
          { role: "user", content: "what is 2+2?" },
          { role: "assistant", content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "calc", arguments: '{"a":2,"b":2}' } }] },
          { role: "tool", tool_call_id: "call_1", content: "4" },
          { role: "user", content: "thanks" },
        ],
        tools: [{ type: "function", function: { name: "calc" } }],
      },
      defaults,
    );
    expect(translated.messages).toEqual([
      { role: "user", content: "what is 2+2?" },
      { role: "assistant", content: [{ type: "tool_use", id: "call_1", name: "calc", input: { a: 2, b: 2 } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "call_1", content: "4" }, { type: "text", text: "thanks" }] },
    ]);
  });

  it("merges parallel tool calls and their results, keeping text alongside tool_use", () => {
    const translated = translateRequest(
      {
        messages: [
          { role: "user", content: "go" },
          { role: "assistant", content: "checking both", tool_calls: [
            { id: "call_a", type: "function", function: { name: "calc", arguments: "{}" } },
            { id: "call_b", type: "function", function: { name: "calc" } },
          ] },
          { role: "tool", tool_call_id: "call_b", content: "second" },
          { role: "tool", tool_call_id: "call_a", content: "first" },
        ],
        tools: [{ type: "function", function: { name: "calc" } }],
      },
      defaults,
    );
    expect(translated.messages).toEqual([
      { role: "user", content: "go" },
      { role: "assistant", content: [{ type: "text", text: "checking both" }, { type: "tool_use", id: "call_a", name: "calc", input: {} }, { type: "tool_use", id: "call_b", name: "calc", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "call_b", content: "second" }, { type: "tool_result", tool_use_id: "call_a", content: "first" }] },
    ]);
  });

  it.each([
    [// user text before the tool result
      [{ role: "user", content: "go" }, { role: "assistant", content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "calc", arguments: "{}" } }] }, { role: "user", content: "never mind" }]],
    [// result for an unknown call
      [{ role: "user", content: "go" }, { role: "assistant", content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "calc", arguments: "{}" } }] }, { role: "tool", tool_call_id: "call_2", content: "x" }]],
    [// missing result at the end
      [{ role: "user", content: "go" }, { role: "assistant", content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "calc", arguments: "{}" } }] }, { role: "tool", tool_call_id: "call_1", content: "ok" }, { role: "assistant", content: null, tool_calls: [{ id: "call_2", type: "function", function: { name: "calc", arguments: "{}" } }] }]],
    [// duplicate call id
      [{ role: "user", content: "go" }, { role: "assistant", content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "calc", arguments: "{}" } }, { id: "call_1", type: "function", function: { name: "calc", arguments: "{}" } }] }, { role: "tool", tool_call_id: "call_1", content: "a" }, { role: "tool", tool_call_id: "call_1", content: "b" }]],
    [// invalid arguments
      [{ role: "user", content: "go" }, { role: "assistant", content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "calc", arguments: "{nope" } }] }, { role: "tool", tool_call_id: "call_1", content: "ok" }]],
    [// tool message without tool_call_id
      [{ role: "user", content: "go" }, { role: "assistant", content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "calc", arguments: "{}" } }] }, { role: "tool", content: "x" }]],
  ])("rejects broken tool history %j", (messages) => {
    expect(() => translateRequest({ messages, tools: [{ type: "function", function: { name: "calc" } }] }, defaults)).toThrow();
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

  async function start(apiKey?: string, effort?: MessageRequest["effort"], thinking?: MessageRequest["thinking"], defaultModel?: string) {
    const seen: MessageRequest[] = [];
    const server = createOpenAIHttpServer({
      ...defaults,
      ...(defaultModel ? { defaultModel } : {}),
      apiKey,
      effort,
      thinking,
      now: () => 1,
      client: {
        async createMessage(request) {
          seen.push(request);
          return textResponse(request.model, "pong", { input_tokens: 1, output_tokens: 1 });
        },
        async *createMessageStream(request) {
          seen.push(request);
          yield { event: "message_start", data: '{"type":"message","id":"m","usage":{"input_tokens":1}}' };
          yield { event: "content_block_delta", data: '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"pong"}}' };
          yield { event: "message_delta", data: '{"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"output_tokens":1}}}' };
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
    const response = await post(url, { model: defaults.defaultModel, messages: [{ role: "user", content: "ping" }] });
    expect(response.status).toBe(200);
    const json = (await response.json()) as { choices: Array<{ message: { content: string } }> };
    expect(json.choices[0]!.message.content).toBe("pong");
    expect(seen[0]!.model).toBe("claude-sonnet-5-5");
  });

  it("passes an explicit non-default model through to the upstream request", async () => {
    const { url, seen } = await start();
    const response = await post(url, { model: "claude-haiku-4-5", messages: [{ role: "user", content: "ping" }] });
    expect(response.status).toBe(200);
    expect(seen[0]!.model).toBe("claude-haiku-4-5");
  });

  it.each(["adaptive", "between_tools"] as const)("forwards request thinking %s without changing the startup default", async (type) => {
    const { url, seen } = await start(undefined, "medium", { type: "between_tools" });
    expect((await post(url, { thinking: { type }, messages: [{ role: "user", content: "ping" }] })).status).toBe(200);
    expect(seen[0]!.thinking).toEqual({ type });
    expect((await post(url, { messages: [{ role: "user", content: "ping" }] })).status).toBe(200);
    expect(seen[1]!.thinking).toEqual({ type: "between_tools" });
  });

  it.each([null, "adaptive", [], {}, { type: "enabled" }, { type: "disabled" }, { type: "adaptive", budget_tokens: 1000 }, { type: 1 }])("rejects malformed thinking without upstream dispatch: %j", async (thinking) => {
    const { url, seen } = await start(undefined, "medium", { type: "between_tools" });
    expect((await post(url, { thinking, messages: [{ role: "user", content: "ping" }] })).status).toBe(400);
    expect(seen).toHaveLength(0);
  });

  it("does not force the default thinking override on haiku models", async () => {
    const { url, seen } = await start(undefined, "medium", { type: "between_tools" }, "claude-haiku-4-5");
    expect((await post(url, { messages: [{ role: "user", content: "ping" }] })).status).toBe(200);
    expect(seen[0]!.thinking).toBeUndefined();
  });

  it("still applies the default thinking override to non-haiku models", async () => {
    const { url, seen } = await start(undefined, "medium", { type: "between_tools" });
    await post(url, { model: "claude-sonnet-5-5", messages: [{ role: "user", content: "ping" }] });
    expect(seen[0]!.thinking).toEqual({ type: "between_tools" });
  });

  it.each(["effort", "reasoning_effort"])("forwards request %s over the startup fallback", async (field) => {
    const { url, seen } = await start(undefined, "high");
    const response = await post(url, { [field]: "medium", messages: [{ role: "user", content: "ping" }] });
    expect(response.status).toBe(200);
    expect(seen).toHaveLength(1);
    expect(seen[0]!.effort).toBe("medium");
  });

  it("retains startup effort when the request has no override", async () => {
    const { url, seen } = await start(undefined, "high");
    expect((await post(url, { messages: [{ role: "user", content: "ping" }] })).status).toBe(200);
    expect(seen[0]!.effort).toBe("high");
  });

  it.each(["low", "medium", "high", "xhigh", "max"])("accepts matching effort aliases for %s", async (effort) => {
    const { url, seen } = await start();
    expect((await post(url, { effort, reasoning_effort: effort, messages: [{ role: "user", content: "ping" }] })).status).toBe(200);
    expect(seen[0]!.effort).toBe(effort);
  });

  it.each([
    { effort: "extreme" }, { reasoning_effort: 3 }, { effort: null }, { effort: {} },
    { effort: "" }, { effort: "MEDIUM" }, { effort: "medium", reasoning_effort: "high" },
    { effort: "medium", reasoning_effort: null },
  ])("rejects malformed or conflicting effort without an upstream call: %j", async (fields) => {
    const { url, seen } = await start(undefined, "high");
    const response = await post(url, { ...fields, messages: [{ role: "user", content: "ping" }] });
    expect(response.status).toBe(400);
    expect(seen).toHaveLength(0);
  });

  it("forwards output_config.effort over the startup fallback", async () => {
    const { url, seen } = await start(undefined, "high");
    const response = await post(url, { output_config: { effort: "medium" }, messages: [{ role: "user", content: "ping" }] });
    expect(response.status).toBe(200);
    expect(seen).toHaveLength(1);
    expect(seen[0]!.effort).toBe("medium");
  });

  it.each(["low", "medium", "high", "xhigh", "max"])("accepts all effort aliases agreeing for %s", async (effort) => {
    const { url, seen } = await start();
    const body = { effort, reasoning_effort: effort, output_config: { effort }, messages: [{ role: "user", content: "ping" }] };
    expect((await post(url, body)).status).toBe(200);
    expect(seen[0]!.effort).toBe(effort);
  });

  it("prefers top-level effort over output_config.effort when they agree", async () => {
    const { url, seen } = await start();
    expect((await post(url, { effort: "low", output_config: { effort: "low" }, messages: [{ role: "user", content: "ping" }] })).status).toBe(200);
    expect(seen[0]!.effort).toBe("low");
  });

  it("retains startup effort when output_config has no effort key", async () => {
    const { url, seen } = await start(undefined, "high");
    expect((await post(url, { output_config: {}, messages: [{ role: "user", content: "ping" }] })).status).toBe(200);
    expect(seen[0]!.effort).toBe("high");
  });

  it.each([
    { output_config: { effort: "extreme" } }, { output_config: { effort: null } }, { output_config: { effort: {} } },
    { output_config: { effort: "" } }, { output_config: { effort: "MEDIUM" } },
    { output_config: { temperature: 1 } }, { output_config: "medium" }, { output_config: null }, { output_config: [] },
    { effort: "medium", output_config: { effort: "high" } },
    { effort: "medium", reasoning_effort: "medium", output_config: { effort: "high" } },
  ])("rejects malformed or conflicting output_config without an upstream call: %j", async (fields) => {
    const { url, seen } = await start(undefined, "high");
    const response = await post(url, { ...fields, messages: [{ role: "user", content: "ping" }] });
    expect(response.status).toBe(400);
    expect(seen).toHaveLength(0);
  });

  it("streams real SSE chunks then [DONE]", async () => {
    const { url } = await start();
    const response = await post(url, { stream: true, messages: [{ role: "user", content: "ping" }] });
    expect(response.headers.get("content-type")).toBe("text/event-stream");
    const text = await response.text();
    const lines = text.trimEnd().split("\n\n").filter(Boolean);
    expect(lines[0]).toContain('"delta":{"role":"assistant"}');
    expect(lines[1]).toContain('"content":"pong"');
    expect(lines[lines.length - 1]).toBe("data: [DONE]");
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
        async *createMessageStream() {
          // unused by this non-streaming test
        },
      },
    });
    servers.push(server);
    const url = await listen(server);
    await post(url, { messages: [{ role: "user", content: "x" }] });
    expect(seen[0]).toMatchObject({ thinking: { type: "between_tools" }, effort: "low" });
  });
});

describe("streaming server", () => {
  const servers: HttpServer[] = [];
  afterEach(() => {
    servers.splice(0).forEach((server) => server.close());
  });

  async function start(streamEvents: AsyncGenerator<{ event: string; data: string }>) {
    const seen: MessageRequest[] = [];
    const signalRef: { current: AbortSignal | undefined } = { current: undefined };
    const server = createOpenAIHttpServer({
      ...defaults,
      now: () => 1,
      client: {
        async createMessage(request) {
          seen.push(request);
          return textResponse(request.model, "pong", { input_tokens: 1, output_tokens: 1 });
        },
        async *createMessageStream(request, s) {
          signalRef.current = s;
          seen.push(request);
          yield* streamEvents;
        },
      },
    });
    servers.push(server);
    return { url: await listen(server), seen, signalRef };
  }

  const post = (url: string, body: unknown) =>
    fetch(`${url}/v1/chat/completions`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

  const textStream = async function* () {
    yield { event: "message_start", data: '{"type":"message","id":"m","usage":{"input_tokens":2}}' };
    yield { event: "content_block_delta", data: '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"he"}}' };
    yield { event: "content_block_delta", data: '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"llo"}}' };
    yield { event: "message_delta", data: '{"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"output_tokens":3}}}' };
  };

  const slowStream = async function* () {
    yield { event: "message_start", data: '{"type":"message","id":"m","usage":{}}' };
    for (;;) {
      yield { event: "content_block_delta", data: '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}' };
      await new Promise((r) => setTimeout(r, 5));
    }
  };

  it("translates upstream SSE into OpenAI chunks", async () => {
    const { url } = await start(textStream());
    const response = await post(url, { stream: true, messages: [{ role: "user", content: "ping" }] });
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toBe("text/event-stream");
    const text = await response.text();
    const chunks = text.trimEnd().split("\n\n").filter((line) => line !== "data: [DONE]").map((line) => JSON.parse(line.slice(6)));
    expect(chunks[0].choices[0].delta).toEqual({ role: "assistant" });
    expect(chunks[1].choices[0].delta).toEqual({ content: "he" });
    expect(chunks[2].choices[0].delta).toEqual({ content: "llo" });
    expect(chunks[chunks.length - 1].choices[0].finish_reason).toBe("stop");
    expect(chunks[chunks.length - 1].usage).toEqual({ prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 });
  });

  it("uses createMessage for non-streaming requests", async () => {
    const { url, seen } = await start(textStream());
    const response = await post(url, { messages: [{ role: "user", content: "ping" }] });
    expect(response.status).toBe(200);
    const json = (await response.json()) as { choices: Array<{ message: { content: string } }> };
    expect(json.choices[0].message.content).toBe("pong");
    expect(seen).toHaveLength(1);
  });

  it("aborts the upstream stream when the client disconnects", async () => {
    const { url, signalRef } = await start(slowStream());
    const response = await post(url, { stream: true, messages: [{ role: "user", content: "ping" }] });
    // Consume a couple of chunks, then cancel the response body.
    const reader = response.body!.getReader();
    await reader.read();
    await reader.read();
    reader.releaseLock();
    await response.body!.cancel();
    await new Promise((r) => setTimeout(r, 20));
    expect(signalRef.current?.aborted).toBe(true);
  });

  it("surfaces an upstream error as an SSE error event, not an empty stream", async () => {
    async function* failing() {
      throw new APIStatusError("api: 404 Not Found: model: claude-haiku-5-5", 404, "not_found_error", 0, null);
    }
    const { url } = await start(failing());
    const response = await post(url, { stream: true, messages: [{ role: "user", content: "ping" }] });
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toBe("text/event-stream");
    const text = await response.text();
    expect(text.startsWith("event: error\n")).toBe(true);
    const payload = JSON.parse(text.slice("event: error\n".length).split("\n\n")[0].slice(6));
    expect(payload.error.type).toBe("not_found_error");
    expect(payload.error.message).toContain("404");
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
