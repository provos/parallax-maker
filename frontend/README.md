# Parallax Maker frontend (Svelte 5)

Svelte 5 + TypeScript + Vite frontend for Parallax Maker. It is being built
incrementally alongside the existing (frozen) Dash UI; see
[`../docs/svelte-migration/ARCHITECTURE.md`](../docs/svelte-migration/ARCHITECTURE.md)
for the target layout and HTTP contract.

This is its own npm package (own `package.json` / lockfile), separate from
the root `package.json`, which still builds the Dash UI's Tailwind v3
stylesheet. Run all commands below from this `frontend/` directory (or use
the `npm --prefix frontend run <script>` wrappers added to the root
`package.json`).

## Prerequisites

Node.js >= 20.19 (or >= 22.12). Install dependencies once with:

```sh
npm install
```

## Development

The backend serves the API at `/api/v1` and the (test-only) `/__e2e__`
endpoints. Start it first, from the repository root, on port 8050:

```sh
.venv/bin/python -m parallax_maker.server
# or, for the deterministic e2e backend:
.venv/bin/python -m parallax_maker.e2e_server --host 127.0.0.1 --port 8050
```

Then, in `frontend/`, start the Vite dev server, which proxies `/api` and
`/__e2e__` to `http://127.0.0.1:8050`:

```sh
npm run dev
```

## Build

Builds the production bundle into `../parallax_maker/static/next` (served by
the backend under `/next/`):

```sh
npm run build
```

## Tests and type checking

```sh
npm run test    # vitest (unit/component tests)
npm run check   # svelte-check + tsc, no emit
```

## Generating API types

`src/lib/api/generated.ts` is generated from the backend's pydantic models
(`parallax_maker/api/schemas.py`); `src/lib/api/types.ts` re-exports it
under client-side names. After changing the schemas, regenerate with:

```sh
npm run gen:api-types
```

CI fails if the committed types are stale.
