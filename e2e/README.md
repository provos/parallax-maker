# Browser characterization tests

These Playwright tests exercise the real Svelte UI, the HTTP API, persistence,
canvas, image composition, and exporters. The test-only server replaces only
expensive model inference with deterministic images and masks.

Run the suite with:

```sh
npm ci
npm --prefix frontend ci
npx playwright install chromium
npm run test:e2e
```

Run the command from the repository root; the server process deliberately uses
the caller's working directory so Python package and virtualenv paths resolve.
`pretest:e2e` builds the Svelte frontend first (`npm run build:frontend`) so
the server has something to serve at `/`.

The default server command is:

```sh
python -m parallax_maker.e2e_server --host 127.0.0.1 --port 8050
```

The harness can target another interpreter, command, or already-running server:

```sh
E2E_SERVER_COMMAND='.venv/bin/python -m parallax_maker.e2e_server --host 127.0.0.1 --port 8050' npm run test:e2e
E2E_BASE_URL='http://127.0.0.1:9000' E2E_SKIP_SERVER=1 npm run test:e2e
```

The suite deliberately uses one worker because the current application has a
process-global `AppState.cache` and filesystem-backed state. Every restore gets
a fresh deterministic state directory. The harness uses a fixed Chromium
viewport and device scale factor, records a trace/video/screenshot on failure,
and fails on uncaught page exceptions, unexpected console errors, failed
same-origin requests, or same-origin HTTP error responses.

The scenarios cover:

- upload, deterministic depth, thresholds, and three generated slices;
- point segmentation, including positive and negative points;
- a real canvas stroke, mask persistence, three exact checkerboard candidates,
  candidate selection, apply, and undo;
- slice editing (create/delete/copy/paste/add-mask/remove-mask/balance/depth
  reorder/replace-image) and mask tools (invert/feather/checkerboard);
- project lifecycle, configuration and export/render (camera, displacement,
  dark mode, model selection, connection/API-key probes, glTF/animation/
  upscale/download);
- restoration of images, slices, prompts, camera controls, model, and theme;
- downloaded glTF structure and the current server-side animation behavior;
- zoom/pan/reset of the main image and queued multi-point markers.

The animation test intentionally asserts that no browser download occurs. The
current implementation renders numbered PNG files server-side and logs
completion but never triggers a browser download. Change this assertion when
that product behavior is fixed.

## Driver layout

The scenario files (`parallax-maker.spec.ts`, `slice-editing.spec.ts`,
`project-export.spec.ts`, `ux-parity.spec.ts`, `layout.spec.ts`) are
frontend-neutral: they only call methods on a `UiDriver` (see
`drivers/types.ts`) and the helper modules,
never a CSS selector or `/__e2e__` URL directly:

- `drivers/types.ts` defines the `UiDriver` interface — the frontend-neutral
  vocabulary of actions ("click this pixel", "open this tab", "select this
  slice") the frontend implements.
- `drivers/svelte.ts` (`SvelteDriver`) is the adapter, and implements every
  workflow. It holds every Svelte selector and gesture (`data-testid`
  attributes, slider steps, the checkerboard candidate classes, etc.) so the
  rest of the suite never needs to know about them.
- `helpers/image.ts` has pixel utilities (`imagePixel`, `imageHash`,
  `imageContainsRGB`, ...) that work against any `Locator` for an `img`
  element.
- `helpers/oracle.ts` talks to the test-only backend oracle
  (`/__e2e__/state`, `/__e2e__/artifact(s)`, `/__e2e__/fixture/*`) through
  `page.request` only, keyed by project ID (the `appstate-*` directory name
  returned by `UiDriver.restoreFixtureState()`).
- `helpers/layout.ts` resizes the viewport and measures page-level overflow
  from the document itself (no frontend selectors).

### Selecting a frontend: `uiTarget`

Which driver the `ui` fixture provides is controlled by the Playwright test
option `uiTarget` (only `'svelte'` is implemented; the type keeps this explicit
rather than hard-coding a single driver everywhere). It defaults to `'svelte'`
and is set per Playwright project in `playwright.config.ts`:

```ts
projects: [
  { name: 'svelte', use: { uiTarget: 'svelte' } },
],
```

Each scenario declares which `Workflow` groups it exercises via
`requireWorkflow(ui, workflow)`, which skips the test when the active driver's
`supports(workflow)` returns false. `SvelteDriver.supports()` always returns
`true`, since every workflow has been migrated (see
`docs/svelte-migration/PARITY.md`); the check is kept so a future
partially-implemented driver (e.g. while building a redesign) can return
`false` for workflows it hasn't built yet, and those scenarios skip instead
of failing.

Most selectors inside `SvelteDriver` use `data-testid` attributes, visible
labels, or accessible roles. Do not add test IDs to every control; semantic
roles and labels are preferred where they are stable enough.
