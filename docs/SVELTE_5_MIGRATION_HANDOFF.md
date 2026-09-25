# Svelte 5 migration handoff

Prepared 2026-09-24 from local `main` at
`ee6f4ea58d9f6bb6667c086d07becb86d96cd15f`.

## Objective and working agreement

Migrate the whole Parallax Maker browser application from Dash to Svelte 5,
retaining Python for inference, image processing, project persistence, and export.
Preserve the existing workflows and image fidelity. The user previously deferred
Svelte while we extracted services and established browser characterization;
that preparation is now merged. This document is the continuation plan, not a
claim that a Svelte frontend or production HTTP API already exists.

The user favors autonomous implementation, agent supervision, meaningful tests,
and incremental PRs. Before moving each workflow, determine what the existing
Playwright tests actually prove, add characterization for uncovered behavior,
then extract and migrate it. Keep expensive model calls deterministic in routine
tests while exercising real composition, masks, persistence, and exports.
Use separate implementation/test/review lanes if delegation is authorized for the
continuing task. Prior work used independent review and fixed actionable findings
before handoff. PR creation, CI/Copilot review, and squash merging have been done
on explicit requests; this document does not authorize a future merge or paid
model execution.

## Starting point

The working tree was clean before this document was added. Local history contains:

| Commit / PR | Delivered |
| --- | --- |
| `a6a99a8`, [#4](https://github.com/provos/parallax-maker/pull/4) | Deterministic offline Playwright harness and first characterization suite. |
| `83b31d0`, [#5](https://github.com/provos/parallax-maker/pull/5) | Core upload/depth/threshold/slice-generation services. |
| `036f917`, [#7](https://github.com/provos/parallax-maker/pull/7) | Point segmentation and multipoint services, validation, expanded tests. |
| `ee6f4ea`, [#8](https://github.com/provos/parallax-maker/pull/8) | Inpainting/mask/version services, adapter tests, browser coverage, review fixes. |

PR #8 is already squash-merged into this checkout. Do not restart from the old
`codex/extract-inpainting-services` branch. Refresh remote status before starting
implementation and create an appropriately scoped `codex/` branch.

Last recorded verification before #8 merged: **258 Python tests and 11 Playwright
scenarios passed**, and GitHub CI was green. These are historical results, not a
fresh test run on the date of this document. The handoff was checked against the
current source; no application tests were rerun for this documentation-only edit.

## Architecture and source map

| Source | Role / migration relevance |
| --- | --- |
| `parallax_maker/webui.py` | Dash app, layout/stores, callback registration, entry point, image serving, and remaining orchestration. Creates three service instances. |
| `parallax_maker/components.py` | UI construction and several nested callback groups; still contains business operations. |
| `parallax_maker/clientside.py`, `parallax_maker/assets/scripts/utility.js` | Dash/browser wiring, canvas painting, image geometry, zoom/pan, point markers and mask preview. Must be replaced with owned frontend state and event handling. |
| `parallax_maker/constants.py` | Existing element IDs and UI constants; useful for parity inventory and selectors. |
| `parallax_maker/controller.py` | `AppState`, process-global cache, filesystem persistence, composition, camera and upscaler lifecycle. Still mixes domain and presentation methods. |
| `parallax_maker/slice.py` | `ImageSlice`, filenames, image versions, undo/redo. |
| `parallax_maker/workflow_services.py` | Core workflow commands/results and repository abstraction. |
| `parallax_maker/segmentation_services.py` | Point/multipoint commands, mask combination, validation and model lifecycle. |
| `parallax_maker/inpainting_services.py` | Masks, prompts/model settings, candidates, apply/erase, versions, artifact repository. |
| `parallax_maker/segmentation.py`, `camera.py`, `gltf.py`, `gltf_cli.py` | Image/slice/render/export algorithms and independent CLI support. |
| `parallax_maker/inpainting.py`, `depth.py`, `instance.py`, provider modules | Real model/provider integrations. Keep algorithms and provider functionality in Python. |
| `e2e/`, `parallax_maker/e2e_server.py`, `parallax_maker/e2e_support/` | Existing behavior oracle and deterministic test-only composition root. |
| `pyproject.toml`, `package.json`, `.github/workflows/python-app.yml` | Python packaging, current Tailwind/Playwright tooling, CI. No Svelte dependencies today. |

The normal entry point is `parallax-maker = parallax_maker.webui:main`.
The three service modules contain no Dash imports or callback concepts, but they
still operate on `AppState`, PIL images and NumPy arrays. They are reusable Python
boundaries, **not JSON APIs**. An HTTP presenter must encode results and assets;
do not serialize service objects or the whole `AppState` indiscriminately.

## Completed service contracts to preserve

### Core workflow

`WorkflowService` implements `upload_image`, `generate_depth`,
`configure_thresholds`, `update_threshold_values`, and `generate_slices`.
Use its existing command/result dataclasses and dependency injection.

- Repository saves take explicit `(state_id, state, StateSaveOptions)`; the command
  identity, rather than a mutable filename field, determines the destination.
- Upload validates PIL input before creating a state. It caches the image but
  does not perform a full project save; existing presentation code lazily writes
  an input image when serving it.
- Depth generation resets thresholds, reuses a matching model, and saves JSON
  plus depth only: slices/input flags are false.
- Threshold operations mutate cached state without saving. Boundary count must
  be `num_slices + 1`; fallback boundaries include 0 and 255. Missing/invalid
  readiness raises `WorkflowNotReady`; no change raises `WorkflowUnchanged`.
- Slice generation requires image/depth/valid boundary count, uses expansion 5,
  assigns slices and performs a full save.
- The service returns image payloads where needed; Dash owns URLs and rendering.

### Segmentation

`SegmentationService` implements `select_depth_point`, `select_instance_point`,
`commit_multi_point`, and `set_multi_point_mode`.

- Plain/Shift/Ctrl clicks map to replace/add/subtract; Shift wins when both
  modifiers are present. Single Ctrl still sends a positive point to the model,
  then subtracts its resulting mask. Multipoint Ctrl instead queues a negative
  model prompt. These are different semantics.
- Queueing does not change the committed mask or run inference. Commit partitions
  positive/negative points, applies the result, and retains queued points. Toggling
  multipoint mode clears the queue. A valid queued commit does not require a
  current depth map afterward.
- A selected slice supplies `slice_image_composed(..., CompositeMode.NONE)` as
  inference input. Otherwise the source image is used. Preserve preview semantics.
- Commands validate coordinates, source/threshold readiness, masks, and selected
  indices. New model instances are cached only after successful interface,
  inference and mask validation; a failed first attempt must be retryable.
- These interactions do not save the project. Invert/feather/checkerboard actions
  still live in component callbacks and have not been extracted.

### Inpainting, masks and versions

`InpaintingService` implements `save_mask`, `load_mask`, `delete_mask`,
`update_prompts`, `update_model`, `generate_candidates`, `select_candidate`,
`clear_selection`, `apply_candidate`, `erase`, and `move_slice_version`.

- Canvas alpha becomes an L mask resized BICUBIC to source dimensions. Mask
  save/delete changes its artifact, without a project JSON save. The adapter
  renders loaded mask pixels as RGBA `(r, 0, 0, r)`.
- Paint uses the saved mask; Fill uses inverse slice alpha. Both patch a **copy**
  of source pixels and generate three candidates with real blur/composition.
  Generating candidates must not mutate the slice before Apply.
- Enhance generates two candidates, resizes LANCZOS to original dimensions, and
  restores the original alpha. It still delegates upscaling through `AppState`.
- Generation persists normalized prompts before model work; they survive model
  failure. Old candidate selection survives failed generation and clears only
  when the complete replacement set succeeds.
- Selecting the same candidate toggles selection off. Slice/model changes clear
  stale selection/candidates. Apply requires a valid selection; the Dash adapter
  additionally checks that the corresponding candidate is visibly selected.
- Candidate pixels currently live in browser data URLs. Apply decodes them and
  passes a sequence of PIL images to the service. Candidate storage by server ID
  would be a new transport design, not an existing facility.
- Apply/Erase write an image version and then save JSON/file mapping only. Undo
  and Redo use the same JSON-only save semantics. Erase now logs once.
- Generation currently passes `crop=True` regardless of the ROI checkbox; the
  checkbox controls the bounding-box preview. Decide separately whether to change
  this product behavior, with explicit regression tests.
- Cache reuse uses transient `AppState.inpainting_pipeline_cache_identity`, with
  both pipeline object identity and model/server/API/workflow configuration.
  Workflow digest is relevant only to ComfyUI. Do not attach cache fields to
  injected pipeline objects or compare mutable loaded pipeline dimensions.
- Replacing a pipeline invalidates its upscaler. Texture export still calls the
  older `create_inpainting_pipeline` helper, so it can replace the shared pipeline;
  the identity check deliberately detects that replacement.
- The adapters now log generation/apply/mask-save domain and malformed-transport
  errors. They decode workflow uploads only for ComfyUI. Preserve useful visible
  error feedback when replacing `PreventUpdate` with API errors.

Version writes and JSON mapping saves are not a cross-file transaction. Global
cache/progress and mutation during callbacks also remain concurrency limitations.

## Remaining application scope

Inventory every user-facing action, not just the 11 passing browser scenarios.
These are the main remaining callback groups to extract or deliberately place in
the frontend:

| Workflow | Current locations | Required continuation |
| --- | --- | --- |
| Slice editing | `webui.delete_slice_request`, `copy_to_clipboard`, `paste_clipboard_request`, `remove_mask_slice_request`, `add_mask_slice_request`, `create_single_slice_request`, `balance_slices_request`, `record_depth_input`, `slice_upload` | Characterize alpha/depth/order/selection, clipboard semantics, upload composition, versioning and exact saves; extract Python commands. |
| Mask tools and display | `components.make_segmentation_callbacks`, `webui.display_slice`, `update_slices`, `update_depth_map_callback` | Extract remaining mutations; separate raw images from checkerboard/overlay presentation. Keep selection invalidation explicit. |
| Project lifecycle | `webui.restore_state`, `save_state`, all `restore_*`, `controller.from_json/fill_from_files/to_file` | Create project service and a public state projection; test old projects and missing/corrupt artifacts. |
| Export/render | `webui.export_state_as_gltf`, `gltf_create`, `gltf_export`, `export_animation`, `upscale_texture`, `download_image` | Move orchestration into export jobs/services; characterize displacement, DOF, upscaled assets and downloads. Keep `parallax-gltf-cli` working. |
| Configuration | `components.make_configuration_callbacks`, `webui.remember_*`, external-server/API/workflow restore callbacks | Extract model/server/credential/workflow/camera persistence and connection probes; retain provider-specific behavior. There are two functions named `remember_camera_parameters`; inspect their callback decorators. |
| Browser interactions | `clientside.py`, `utility.js`, navigation/tools/tab/theme callbacks | Port canvas/brush/eraser/load/clear, point overlays, crop box, zoom/pan/reset, tabs, progress/logs and responsive layout into Svelte. |
| Delivery | Python entry points/package data, Dockerfile, workflows, README | Bundle compiled frontend, serve it reliably, verify installed-package/container behavior and document startup. |

Known baseline issues that must be triaged rather than silently copied or
accidentally hidden by migration:

- `AppState.balance_slices_depths` contains `for i in len(self.image_slices)` and
  divides by `len - 1`; characterize then fix its normal/empty/single-slice cases.
- State upload is JSON referencing server-side files; it is not a self-contained
  browser project upload. Restore requires the corresponding `appstate-*`
  directory and images on the server. Keep compatibility; a portable archive is
  a separate format/behavior decision.
- Animation writes `rendered_image_%03d.png` server-side; no download currently
  fires, and the browser test intentionally records that behavior.
- Depth-point behavior at the first threshold has a preserved wraparound/empty
  mask case. Inspect its service test before changing the algorithm.
- The production main-image URL uses second-resolution cache busting. E2E patches
  this to a monotonic version, so passing tests do not prove production refresh
  correctness. Use explicit revisioned assets in the new transport.
- Existing Dockerfile still references `/app/webui.py`, while current code is
  packaged under `parallax_maker`; verify/fix packaging during delivery work.

## Recommended target and sequence

The following is a proposed design, not an implemented or user-mandated stack
choice beyond Svelte 5 and the retained Python backend. A Svelte 5 + TypeScript +
Vite client and a small Python HTTP layer are a suitable starting point. Reusing
the existing Flask infrastructure is the lowest-change API option; a new framework
should have a concrete benefit. Verify current official framework documentation
and compatible versions when scaffolding, and commit the dependency lockfile.

1. **Complete the parity inventory and baseline.** Run the current suite, enumerate
   all controls/callbacks, and capture representative UI screenshots. Record
   uncovered actions and intentional fixes. The existing tests are behavioral
   image assertions, not a complete visual/layout acceptance suite.
2. **Extract remaining backend workflows incrementally.** Start with slice editing
   and remaining mask operations, then project/configuration and export. Add the
   missing browser characterizations before extraction; add service and adapter
   contracts alongside it. Keep working Dash adapters during this phase.
3. **Introduce an explicit application composition root and HTTP contracts.** Wire
   repositories, real/fake providers and per-project execution there, independently
   of `webui` imports. Start with upload → depth → thresholds → generated slices
   as the first usable frontend/API vertical slice.
4. **Migrate successive Svelte workflows.** Add selection and point segmentation,
   slice editing, canvas/masks and inpainting, project/settings restoration, then
   previews/export. Preserve loading, disabled controls, logs and retry behavior
   along with successful images. Establish an API test fixture interface before
   retargeting browser tests.
5. **Run Dash and Svelte parity suites while both exist.** Use UI adapters for
   framework-specific selectors/events, sharing behavioral assertions and fake
   providers. Keep both targets green as each supported workflow migrates.
6. **Cut over and package.** Build assets into the Python distribution, select the
   Svelte UI as the normal entry point, document migration and compatibility, and
   remove Dash/runtime callback JS only after every required workflow has parity
   and the replacement has standalone startup tests.

Suggested API design constraints:

- Use explicit state/project IDs, slice IDs or validated indices, and state
  revisions. Expose a deliberate JSON projection: metadata, settings, selections,
  availability of undo/redo, busy state and revisioned asset references.
- Keep PIL/NumPy/model objects and credentials out of public state responses.
  Resolve asset IDs within a project's directory, with proper path containment;
  do not turn the legacy raw filename route into an unrestricted file API.
- Use binary uploads/assets or bounded data URLs as an initial transport. If
  candidates become server assets, bind each set to project, slice, source
  revision and generation ID; reject applying a stale set to a different slice.
- Map domain errors to documented HTTP error codes/messages; keep no-op results
  distinct from failures. Retain detailed internal diagnostics without exposing
  credentials or provider payloads to browser logs.
- Real inference is slow. Define jobs and progress (polling is sufficient at first),
  serialize conflicting mutations per project, and prevent stale responses from
  overwriting newer selections/images. Do not assume the process-global cache
  works across multiple worker processes. Start with an explicit supported
  single-process execution model, then expand deliberately.
- Svelte owns transient UI state and coordinate transforms. Persisted project
  state has one authoritative backend representation. Avoid reproducing Dash's
  trigger stores and callback cycles as frontend reactive loops.

## Tests: existing oracle and gaps

`e2e/parallax-maker.spec.ts` currently has 11 scenarios:

1. Upload → deterministic depth → three slices with pixel signatures.
2. Instance point replace/Shift union/Ctrl subtraction with exact mask regions.
3. Positive/negative multipoint queue/commit and toggle/reset behavior.
4. Depth click coordinates, depth, log and mask.
5. Selected-slice segmentation input, observed via a test-only source tag.
6. Painted mask → three candidates, inside/outside composition, prompts,
   Apply/version change and Undo.
7. Fill candidates favor transparent regions, fill their alpha, and retain opaque
   sample alpha; the test checks relative color preservation, not exact opaque RGB.
8. Enhance creates two same-size candidates while retaining alpha.
9. Erase removes painted alpha and supports Undo/Redo.
10. Restore checks exact input/depth signatures and selected controls/settings.
11. glTF download checks three meshes/textures and inline PNGs; animation checks
    four valid 320×240 frames with distinct hashes and no browser download.

Relevant Python contracts are in `test_workflow_services.py`,
`test_workflow_adapters.py`, `test_segmentation_services.py`,
`test_segmentation_adapters.py`, `test_segmentation_component_adapters.py`,
`test_inpainting_services.py`, `test_inpainting_component_adapters.py`, and
`test_inpainting_webui_adapters.py`. Keep backend contracts after retiring Dash;
replace Dash adapter contracts with API/frontend contracts.

Missing or incomplete browser coverage includes general slice editing/clipboard,
depth-mode modifier combinations, invert/feather, canvas clear/load and zoom/pan,
candidate deselection and failed-generation recovery, provider configuration
failures, project save→restore round trips, displaced/DOF/upscaled export, actual
3D viewer rendering, responsiveness and keyboard accessibility. Some of these
already have service/adapter unit tests; do not describe them all as untested.

The current suite is **not frontend-neutral**: `helpers/app.ts` knows Dash JSON IDs,
dropdowns and sliders, and scenario bodies also use Dash-specific selectors,
classes and raw `/__e2e__/` requests. A base-URL change alone cannot retarget it.
Move those interactions behind an adapter and preserve image/state oracles.
Use real clicks and pointer gestures, not forced clicks or synthetic dispatch
that bypasses hit testing. Select the visible tab before using a slice thumbnail.

Pixel pitfalls: thumbnails contain a checkerboard composite whereas raw slice
artifacts/candidates may contain transparency. Compare like representations.
Real feathering requires the small existing tolerance for inside-mask colors;
outside-mask preservation is exact. Do not weaken these checks to image counts
or “something changed.”

## Deterministic harness details

- `python -m parallax_maker.e2e_server` is an explicit test-only entry point. It
  creates an isolated temporary working directory and cleans it up on SIGTERM.
- `/__e2e__/ready` reports palettes/check size; `/__e2e__/fixture/input.png` serves
  the fixture; each GET of `/__e2e__/fixture/state.json` creates a fresh project
  directory/cache key. `/__e2e__/state?filename=...` exposes mask/selection/version
  metadata; `/__e2e__/artifacts?filename=...` lists files and
  `/__e2e__/artifact/<filename>/<path>` serves a contained artifact.
- These are test observation/setup endpoints, not the production API. Keep them
  absent from normal startup. State fixture images already exist server-side
  when its JSON is uploaded through the real restore UI.
- `install_fakes()` currently imports `webui`, rebuilds its depth service with a
  fake factory, uses runtime segmentation/inpainting factories, patches upscaling
  and provider probes, blocks `requests` networking, and disables external UI
  scripts/viewer loading. Factor shared fake dependencies into the new composition
  root; merely patching the old Dash module will not fake a new API service.
- Preserve real mask handling, patching, composition, version files and exporters.
  Fake only expensive provider/inference boundaries and required external probes.
  This does not validate real model quality, downloads, GPU memory or credentials.
- One Playwright worker, fresh server, fixed viewport/DPR and unique restore states
  prevent cross-test pollution. Keep diagnostics for console errors, failed
  requests, unexpected HTTP errors, screenshots, video and traces.
- Wait on observable state/images; Dash's periodic progress polling prevents
  reliable `networkidle` waits. Existing canvas save happens on `mouseout`, not
  `mouseup`; `drawCanvasStroke` explicitly exits the canvas. A new save-on-pointerup
  contract can be an intentional improvement, but must test pending-save ordering
  before Generate and slice changes.
- The old JS keeps `currentSlice` asynchronously and clears canvases on image
  changes. Give the new canvas an explicit lifecycle so selection, resize or
  inference completion cannot silently discard an unsaved stroke.

## Setup and validation commands

Run from repository root. Prefer a working Python 3.12 environment; project
metadata permits Python >=3.10,<3.14, while CI currently uses Python 3.10 and
Node 24. An old local `.venv` symlink has broken after Python upgrades before;
check it before diagnosing application failures. No special interpreter path
from an earlier agent session should be assumed to exist.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
npm ci
npx playwright install chromium
.venv/bin/python -m pytest -o addopts='' -q
E2E_SERVER_COMMAND='.venv/bin/python -m parallax_maker.e2e_server --host 127.0.0.1 --port 8050' npm run test:e2e
git diff --check
```

Use the environment setup only when needed. Normal CI runs `pytest` with configured
coverage; `-o addopts=''` above avoids coverage overhead/shared coverage-file races
for local verification. CI also runs fatal flake8 checks, `npm ci`, Chromium
installation, and Playwright. Browser diagnostics are configured under
`e2e/test-results/`. Local browser/server tests may require sandbox approval to
bind localhost; distinguish that from application test failures.

For the new frontend, add type checking, a production build, API contracts and
installed-asset startup checks to CI. Retain `package-lock.json`; do not rely on
unrecorded global Node tooling. Inspect Python package-data patterns when adding
nested Vite assets; a source checkout passing is insufficient evidence that a
wheel contains the frontend. Build/verify Docker and the independent glTF CLI
before declaring distribution parity.

## Completion gate and first next step

Migration is complete when every inventoried workflow works through Svelte,
the shared deterministic browser suite and new gap tests pass, backend contracts
remain green, errors/progress/retries are usable, existing saved projects restore,
and a built distribution runs the full UI without Dash. Include visual review at
representative viewport sizes, real viewer verification, and a bounded real-model
smoke test when execution resources/credentials are authorized. Mocked tests alone
cannot establish real-provider compatibility.

The next agent should first baseline the merged `main`, create a feature/parity
checklist from the remaining callback inventory, and characterize **slice editing
plus remaining mask tools**. That is the next bounded extraction. Also define
the project/state/asset API contract that will let the
already-extracted upload/depth/slices workflow become the first Svelte vertical
slice. Keep each PR independently runnable and reviewable; report actual test
results, intentional behavior changes and remaining gaps at each handoff.

## Migration complete

The cutover landed: every inventoried workflow reached parity (see
`docs/svelte-migration/PARITY.md`, kept as the historical per-callback
record), the shared `e2e/*.spec.ts` scenarios pass on the Svelte UI, and Dash
has been removed entirely from the codebase and its dependencies.

Final state:

- **Server**: `parallax_maker/server.py`'s `create_server(runtime)` builds a
  plain Flask app - no Dash import anywhere - registering the `/api/v1`
  blueprint and serving the built Svelte app (`parallax_maker/static/app/`,
  renamed from the migration-era `static/next/`) at `/`, with a permanent
  redirect from the old `/next/...` bookmarked path. `main()` is the
  `parallax-maker` console-script entry point; it runs single-process by
  design (the in-memory `ProjectRegistry`/`JobManager` and `AppState.cache`
  are not shared across worker processes - see `server.py`'s own docstring).
- **Removed**: `webui.py`, `components.py`, `clientside.py`,
  `parallax_maker/assets/` (Dash CSS/JS), Dash-only helpers in `utils.py`
  (`find_pixel_from_click`/`find_pixel_from_event`, `get_gltf_iframe`,
  `get_no_gltf_available`, `highlight_selected_element`), the root Tailwind
  v3 files (`tailwind.css`, `tailwind.config.js`, `postcss.config.js`) and
  their npm scripts, `constants.py` (every constant in it was a Dash element
  ID), and every Dash-adapter test module (`test_webui.py`,
  `test_components.py`, `test_workflow_adapters.py`,
  `test_segmentation_adapters.py`, `test_segmentation_component_adapters.py`,
  `test_inpainting_component_adapters.py`,
  `test_inpainting_webui_adapters.py`, `test_constants.py`). None of these
  covered algorithm/service behavior not already covered by
  `*_services.py`'s own tests or `test_api_*.py`, so nothing needed porting.
  `dash`/`dash_extensions` were dropped from `pyproject.toml` and
  `requirements.txt` (both now list `flask`/`pydantic` explicitly, which Dash
  previously pulled in transitively). The stale, never-used `poetry.lock`
  was deleted; pip with `pyproject.toml`/`requirements.txt` is the only
  supported toolchain.
- **`gltf_cli.py`**: no longer imports `export_state_as_gltf` from `webui`;
  it now builds an `ExportService` (`export_services.py`) with a small
  `_PreloadedStateRepository` adapter so the CLI's own `-i`/`-o` semantics
  (load an arbitrary on-disk project, write the scene to an arbitrary output
  directory) map onto that service's `state_id`-keyed contract. Covered by
  the new `test_gltf_cli.py`.
- **`e2e_server.py`/`e2e_support/fakes.py`**: `install_fakes()` no longer
  imports or patches `webui`/`components` at all; every fake is now a direct
  module-attribute patch on the provider/model modules the API and services
  actually resolve at call time (`automatic1111`, `comfyui`, `stabilityai`,
  `falai`, `controller`, `inpainting`), most of which
  `configuration_services.py` was already documented to expect. `/__e2e__/*`
  routes are registered directly on the plain Flask app (`@app.get(...)`
  instead of the old `@app.server.get(...)`).
- **`e2e/`**: `DashDriver` (`drivers/dash.ts`) and the `dash` Playwright
  project are gone; `SvelteDriver.goto()` now navigates to `/` instead of
  `/next/`. The `UiDriver` abstraction and its frontend-neutral scenario
  files are unchanged in shape. The `test.fail(ui.target === 'dash', ...)`
  gates for historical Dash-only bugs (no drag-to-pan, no zoom reset, the
  mismatched-aspect slice-upload collapse, the broken Balance button) were
  removed along with their Dash-specific comments; the underlying
  assertions were kept as ordinary passing checks against Svelte.
- **Docker**: multi-stage build - a Node stage (`npm ci` + `vite build` in
  `frontend/`) followed by a Python stage that copies the built
  `parallax_maker/static/app/` into place before `pip install .`, so the
  wheel's package-data step picks it up; `CMD` runs
  `parallax-maker --host 0.0.0.0 --port 8050` (the old image's `CMD ["python",
  "/app/webui.py"]` had already been broken by the earlier move to the
  `parallax_maker` package before this PR).
- **CI** (`.github/workflows/python-app.yml`): unchanged pytest/flake8/
  frontend-check/test/build steps, now followed by installing the built wheel
  into the job's own environment (`pip install --force-reinstall --no-deps`)
  and running `scripts/smoke_installed.sh` against it before the Playwright
  (`svelte`-only) step.
- **Packaging**: the build output directory (and therefore
  `[tool.setuptools.package-data]`, `.gitignore`, `frontend/vite.config.ts`'s
  `base`/`outDir`) was renamed from `static/next` to `static/app` as part of
  cutover, matching its new role as *the* UI rather than a migration-era
  side-by-side preview.

Verified before this note was written: 377 Python tests pass (see
`test_gltf_cli.py`, new), `flake8 --select=E9,F63,F7,F82` is clean,
`npm run check:frontend`/`test:frontend`/`build:frontend` all pass (140
frontend unit tests), `npm run check:e2e` type-checks cleanly, and the full
Playwright suite (`svelte` project, 38 scenarios) passed three consecutive
runs. `scripts/smoke_installed.sh` passed against both an editable install
and a real built wheel installed into a fresh location. `python -m pip wheel
. --no-deps` produces a wheel containing
`parallax_maker/static/app/index.html` and its hashed `assets/`. A full
`docker build .` and a Node-only `docker build --target frontend-build .`
were both exercised locally; see this PR's own final report for their actual
output, since a heavy multi-stage build (torch, diffusers, etc.) is
environment/time-dependent in a way this static document shouldn't assert a
permanent result for.

Real-model smoke test (run locally against the real server, no paid APIs):

- DINOv2 depth -> slices -> depth click -> glTF export -> animation, through
  both the HTTP API and the Svelte UI in a browser.
- MiDaS and ZoeDepth: these failed to load under `timm` 0.6.12 on Python
  3.11+ (a pre-existing bug CI never saw, since it uses fakes on 3.10);
  pinning `timm==0.6.13` fixed both.
- SAM (single click and multi-point with a negative point) and SD-XL
  inpainting (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`): generate
  returned 3 candidates, apply created a new slice version that changed the
  masked region, and undo restored the previous version.

Remaining optional follow-up: a **visual/UX redesign** was deliberately
deferred throughout the migration (see "Decisions" in
`docs/svelte-migration/ARCHITECTURE.md`); the Svelte UI intentionally kept
Dash's look and feel via shared CSS tokens in `frontend/src/app.css`. Now
that Dash is gone, a redesign is unconstrained by parity.
