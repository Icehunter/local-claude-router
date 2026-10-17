# claude-wire

An OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/models`, `/health`) that sends your requests to Claude as a Claude Code-shaped request, authenticated with your subscription OAuth login.

## Setup

```
npm ci
npm run login      # browser sign-in; or reuse conduit's keychain entry (read-only)
npm run status     # which credentials are in use, account, expiry
```

## Run the HTTP server

```
npm start          # build, then serve on http://127.0.0.1:8787/v1
npm run dev        # tsx watch: restarts on every source change
```

Point the app at `http://127.0.0.1:8787/v1`. The API key is ignored unless `CLAUDE_WIRE_API_KEY` is set. Model names that do not start with `claude` map to the default model.

| Variable | Default | Meaning |
|---|---|---|
| `CLAUDE_WIRE_PORT` | `8787` | Listen port |
| `CLAUDE_WIRE_HOST` | `127.0.0.1` | Bind address. A non-loopback host refuses to start without `CLAUDE_WIRE_API_KEY` |
| `CLAUDE_WIRE_API_KEY` | unset | Require `Authorization: Bearer <key>` |
| `CLAUDE_WIRE_THINKING` | `off` | `off` sends `thinking: {type: "between_tools"}`, `adaptive`, or `default` (send nothing; Sonnet 5.5 then thinks adaptively) |
| `CLAUDE_WIRE_EFFORT` | unset | `low`, `medium`, `high`, `xhigh`, `max`. `between_tools` needs `high` or lower |
| `CLAUDE_WIRE_LOG_LEVEL` | `info` | pino level. `NO_COLOR=1` disables colors |
| `ANTHROPIC_DEFAULT_MODEL` | `claude-sonnet-5-5` | Default model |
| `ANTHROPIC_MAX_TOKENS` | `4096` | Default `max_tokens` |

Each request logs method, path, status, model, token counts and duration. Prompts and replies are never logged.

## What it does not do

- No tools or function calling, no image parts, `n` must be 1 (400 otherwise).
- `stream: true` returns the full answer as one chunk, then `[DONE]`; the upstream call is not streaming.
- `temperature` is accepted and dropped: the upstream rejects it for current models.
- `response_format` is ignored, so JSON replies may arrive in a code fence.

## Development

```
npm run typecheck && npm test && npm run build
```

TypeScript 7, ESM with `.js` import extensions, Vitest in `tests/`. TypeScript 7 does not auto-include `@types/*`; add any new one to `types` in `tsconfig.json`.
