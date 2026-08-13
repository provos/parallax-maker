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

## Selector contract for a future Svelte UI

Most selectors use visible labels, button text, or existing stable IDs. A Svelte
replacement should preserve accessible names for the user-facing contract. The
framework-specific selectors are centralized in `helpers/app.ts`; treat that file
as the current Dash UI adapter. A Svelte adapter should also implement the same
test-only fixture/state API so the behavioral scenario files stay unchanged. The
few image-workflow elements that would benefit from explicit test IDs are:

- `input-image`, `paint-canvas`, and `preview-canvas`;
- `depth-map-image`;
- `slice-image-{index}` and `slice-overlay-{index}`;
- `inpainting-candidate-{index}`;
- `slice-undo-{index}` and `slice-redo-{index}`.

Do not add test IDs to every control; semantic roles and labels are preferred.
