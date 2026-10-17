# claude-wire

OpenAI-compatible HTTP server over an Anthropic OAuth client. See README.md for usage.

## Conventions

- ESM only, TypeScript strict (TypeScript 7), `.js` extensions in imports.
- Source in `src/`, tests in `tests/` (Vitest). Name every `@types/*` package in `tsconfig.json` `types`.
- Never log prompts or model output.
- The upstream rejects `temperature` for current models and `thinking: {type: "disabled"}` on Sonnet 5.5; do not forward the former, use `between_tools` for the latter.
- The HTTP server must stay loopback-only unless `CLAUDE_WIRE_API_KEY` is set.

## After source changes

`dist/` is gitignored. `npm run dev` restarts the server on source changes; `npm start` builds and runs.
