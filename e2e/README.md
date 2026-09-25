# Browser characterization tests

These Playwright tests exercise the real Dash UI, callbacks, persistence, canvas,
image composition, and exporters. The test-only server replaces only expensive
model inference with deterministic images and masks.

Run the suite with:

```sh
npm ci
npx playwright install chromium
npm run test:e2e
```

Run the command from the repository root; the server process deliberately uses
the caller's working directory so Python package and virtualenv paths resolve.

The default server command is:

```sh
python -m parallax_maker.e2e_server --host 127.0.0.1 --port 8050
```

The harness can target another interpreter, command, or already-running server:

```sh
E2E_SERVER_COMMAND='poetry run python -m parallax_maker.e2e_server --host 127.0.0.1 --port 8050' npm run test:e2e
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
- restoration of images, slices, prompts, camera controls, model, and theme;
- downloaded glTF structure and the current server-side animation behavior.

The animation test intentionally asserts that no browser download occurs. The
current callback renders numbered PNG files server-side and logs completion but
does not populate `download-animation`. Change this assertion when that product
behavior is fixed.

## Driver layout

The scenario file, `parallax-maker.spec.ts`, is frontend-neutral: it only calls
methods on a `UiDriver` (see `drivers/types.ts`) and two helper modules, never
a CSS selector, Dash ID, or `/__e2e__` URL directly:

- `drivers/types.ts` defines the `UiDriver` interface — the frontend-neutral
  vocabulary of actions ("click this pixel", "open this tab", "select this
  slice") that every frontend implements the same way.
- `drivers/dash.ts` (`DashDriver`) is the current, and so far only, adapter.
  It holds every Dash-specific selector and gesture (`#image`, `#log`,
  slider IDs, the checkerboard candidate classes, etc.) so the rest of the
  suite never needs to know about them.
- `helpers/image.ts` has frontend-neutral pixel utilities (`imagePixel`,
  `imageHash`, `imageContainsRGB`, ...) that work against any `Locator` an
  `img` element, regardless of which driver produced it.
- `helpers/oracle.ts` talks to the test-only backend oracle
  (`/__e2e__/state`, `/__e2e__/artifact(s)`, `/__e2e__/fixture/*`) through
  `page.request` only. It is shared unchanged across frontends because the
  oracle and artifact endpoints are keyed by project ID (the `appstate-*`
  directory name returned by `UiDriver.restoreFixtureState()`), not by UI.

A future Svelte adapter implements `UiDriver` in `drivers/svelte.ts` with its
own selectors and accessible names, and the test-only fixture/state API stays
the same, so `parallax-maker.spec.ts` does not change.

### Selecting a frontend: `uiTarget`

Which driver the `ui` fixture provides is controlled by the Playwright test
option `uiTarget` (`'dash'` today; `'svelte'` is recognized by the type but
not implemented yet and throws a clear error if selected). It defaults to
`'dash'` and is set per Playwright project in `playwright.config.ts`:

```ts
projects: [
  { name: 'dash', use: { uiTarget: 'dash' } },
],
```

Each scenario declares which `Workflow` groups it exercises via
`requireWorkflow(ui, workflow)`, which skips the test when the active driver's
`supports(workflow)` returns false. `DashDriver.supports()` currently returns
`true` for every workflow; a partially-implemented Svelte driver can return
`false` for the workflows it hasn't built yet so those scenarios skip instead
of failing.

Most selectors inside `DashDriver` use visible labels, button text, or stable
IDs. The few image-workflow elements that would benefit from explicit test IDs
in a future Svelte UI are:

- `input-image`, `paint-canvas`, and `preview-canvas`;
- `depth-map-image`;
- `slice-image-{index}` and `slice-overlay-{index}`;
- `inpainting-candidate-{index}`;
- `slice-undo-{index}` and `slice-redo-{index}`.

Do not add test IDs to every control; semantic roles and labels are preferred.
