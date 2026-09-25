# Dash → Svelte parity checklist

> **Historical record.** Dash reached full parity and was removed from the
> codebase in the cutover PR (`webui.py`, `components.py`, `clientside.py`,
> `parallax_maker/assets/`, and their adapter tests are gone; see the
> "Migration complete" section of `../SVELTE_5_MIGRATION_HANDOFF.md`). This
> document is kept as-is - including its "Dash" language and file:line
> references to the now-deleted modules - as the record of what was checked
> and how, not as a description of the current codebase.

This document tracks parity between the (now-removed) Dash UI and the Svelte 5
replacement, one row per Dash callback / clientside callback / browser event
handler discovered in `parallax_maker/webui.py`, `parallax_maker/components.py`,
`parallax_maker/clientside.py`, and `parallax_maker/assets/scripts/utility.js`.
A row's "Svelte" checkbox is only ever checked when the Svelte UI implements the
same behavior **and** a shared e2e scenario (`e2e/parallax-maker.spec.ts`, once
frontend-neutral) or an equivalent API contract test exercises it on both UIs —
not merely when Svelte code exists for it.

## Counts

- `parallax_maker/webui.py`: **48** `@app.callback` registrations (WEB-01..WEB-48).
- `parallax_maker/components.py`: **25** callback-decorator statements, producing
  **26** live registrations at runtime (`make_tabs_callback` is invoked twice, for
  `"viewer"` and `"main"`) plus **1 defined-but-never-registered** callback
  (`make_label_container_callback` is never called from `webui.py`) — **CMP-01..CMP-26**.
- Clientside callbacks: **7** registered in `parallax_maker/clientside.py` +
  **5** registered inside `components.make_canvas_callbacks` = **12** total
  (CLI-01..CLI-12).
- Pure browser event handlers in `utility.js` with no Dash `Input`/`Output` at all
  (plain `addEventListener`/`ResizeObserver` wiring): **6** (JS-01..JS-06). (A
  further 2 `addEventListener` calls exist but are implementation detail of
  CLI-01/CLI-02 and are folded into those rows rather than double-counted.)
- **Total Dash-reachable behaviors enumerated: 92** (48 WEB + 26 CMP + 12 CLI + 6 JS).

## E2E scenario legend

Numbers below refer to `e2e/parallax-maker.spec.ts`, in file order:

1. Upload → deterministic depth → three real slices.
2. Instance-point replace/Shift-union/Ctrl-subtract exact mask regions.
3. Multipoint queue/commit and toggle/reset behavior.
4. Default depth-map click records pixel/depth/log/mask.
5. Selected-slice segmentation input (composed slice sent to model).
6. Painted mask → 3 candidates, prompts, Apply + version change + Undo.
7. Fill candidates weighted toward transparent slice pixels.
8. Enhance → 2 same-size candidates, alpha preserved.
9. Erase removes painted alpha; Undo/Redo.
10. Restore: images/controls/prompts/camera/theme.
11. glTF download (3 meshes/textures) + animation export (4 frames, no download).

Scenarios 2–11 all start from `restoreFixtureState()`; only scenario 1 starts
from `uploadInputImage()`. Scenarios 1–11 all call `clickMainTab()` at least once.

### `e2e/slice-editing.spec.ts` legend ("SE-N")

All 15 scenarios start from `restoreFixtureState()` (three slices at depths
`[85, 170, 255]`, thresholds `[0, 85, 170, 255]`, 320x240 input). "SE-N" in
the tables above refers to this list, in file order:

1. Create slice from an instance-segmentation mask: appended, selected,
   depth taken from the click, alpha matches the mask (inside > 200, outside
   exactly 0).
2. Create slice with no mask: empty/transparent, appended at depth 127,
   deterministic index 1.
3. Delete the selected slice: count/order/depths of the remainder, selection
   cleared, main image reverts to the raw input image.
4. Delete with nothing selected: logged no-op, slice count unchanged.
5. Add mask then remove mask on the selected slice: alpha in/out of the mask
   region, filename version bumps to `_v2`/`_v3`, undo restores `_v1`. Also
   pins that Add's own mask does not survive the operation (see Known
   quirks) and that a fresh click is needed before Remove does anything.
6. Add/remove mask with no selection or no mask: logged no-ops, no version
   bump.
7. Copy without a mask is a logged no-op; copy with a selected slice + mask,
   then paste: alpha becomes opaque under the mask and is unchanged outside
   it, filename bumps to `_v2`.
8. Paste with no slice selected (mask copied from the full image while
   nothing is selected): logged no-op.
9. Set slice depth: reordering an *unselected* slice clears the selection
   even when a *different* slice is the one selected; a non-reordering edit
   leaves the selection alone. (Clicking a *selected* slice's own depth
   badge is not exercised — see Known quirks.)
10. Upload a matching-aspect-ratio replacement image onto a slice thumbnail:
    no resize, version bump, exact pixel/alpha match, thumbnail hash changes.
11. Upload a mismatched-aspect-ratio replacement image: pins the literal
    (non-interpolated) log message and the resulting collapsed dimensions —
    see Known quirks.
12. Invert with no mask creates an all-zero mask first; inverting a real
    mask flips every oracle sample and its nonzero count exactly
    (`total - before`).
13. Feather: interior stays at the original maximum (255), nonzero count
    strictly grows, and the tight bounding box does not shrink on any side —
    proof of an intermediate-valued blurred edge without depending on exact
    kernel output.
14. Checkerboard toggles `use_checkerboard` and no-ops the main image without
    a selection; with a selection it recomposes, and toggling back reproduces
    the exact same image hash.
15. Balance: `test.fail()`-pinned — the Dash button currently 500s (see Known
    quirks); the fixed `balance_slices_depths()` behavior itself is proven by
    `test_controller.py::TestBalanceSlicesDepths`, not by this scenario.

### `e2e/project-export.spec.ts` legend

All 9 scenarios start from `restoreFixtureState()`, whose fixture also sets
`dark_mode=true`, camera distance/focalLength/maxDistance=125/475/140,
mesh_displacement=15, depth model `dinov2` and inpainting model SD XL 1.0
(`parallax_maker/e2e_support/fixtures.py`), in file order:

1. Save State then restore: the actual on-disk `appstate.json` (fetched via
   the `/__e2e__/artifact/...` oracle after clicking Save State) is fed back
   through Load State on the *same* project, and every oracle field plus the
   main/depth image hashes match exactly before vs. after.
2. Camera distance/focal length/max distance/displacement slider changes
   persist immediately (before any explicit Save) and again survive a full
   save/restore round trip - both the oracle's `camera`/`mesh_displacement`
   and the sliders' own `aria-valuenow`.
3. Dark mode (fixture starts `true`) survives a save/restore round trip.
4. Depth model (Mode tab) and inpainting model (Configuration tab)
   selections both survive a save/restore round trip.
5. External-server connection test (Automatic1111) highlights success when
   the fake `make_models_request` probe succeeds; asserts the neutral state
   beforehand too.
6. glTF export with the displacement slider set to a positive value produces
   a subdivided (`501x501`-vertex) mesh instead of a flat 4-vertex quad.
7. glTF export with DOF enabled switches every material from `BLEND` to
   `MASK` alpha mode (`alphaCutoff=0.5`); the same project's `dof: false`
   export is `BLEND` for comparison.
8. Upscale Textures then glTF export embeds the upscaled (`FakeUpscaler`'s
   exact 2x) image dimensions, not the original slice size.
9. Clicking a slice's download icon downloads bytes byte-for-byte identical
   to the on-disk slice PNG (`/__e2e__/artifact/...`).

Discovered while writing scenario 2: `UiDriver.setSlider` (`drivers/dash.ts`)
had two real, previously-latent bugs of its own, both fixed as part of this
slice since every prior scenario that used it only ever set a slider to `0`
or used a step-1 slider — see "Known quirks" for the real Dash 500 the second
one avoids re-triggering:
- It trusted the slider handle's `aria-valuestep` attribute to compute a
  press count, but rc-slider does not reliably render that attribute (e.g.
  the displacement slider's real `step=5`, `components.py:1436`) - a
  press-and-re-read loop replaces the pre-computed count.
- It pressed `Home` before stepping, unconditionally resetting to each
  slider's own minimum (`0` for every camera/displacement slider) even when
  only that one slider's value needed to change - now steps directly from
  the slider's current value instead, which also happens to be required to
  avoid the WEB-30 crash below.

---

## Upload/Depth/Slices

| ID | Function (file:line) | Trigger(s) | Effect | Backend service | E2E coverage | Svelte |
| --- | --- | --- | --- | --- | --- | --- |
| WEB-04 | `update_threshold_values` webui.py:331 | Input `{threshold-slider,ALL}.value` (also its own Output); State `SLIDER_NUM_SLICES`, filename | Pushes edited threshold values through `WorkflowService.update_threshold_values`; re-renders main image preview if changed | WorkflowService | none (no test drags a threshold slider) | [ ] |
| WEB-05 | `update_thresholds_html` webui.py:361 | Input `STORE_UPDATE_THRESHOLD_CONTAINER.data` | Rebuilds the threshold `dcc.Slider` list from `state.imgThresholds` | inline (AppState read) | 1, 10 | [ ] |
| WEB-06 | `update_thresholds` webui.py:397 | Input `CTR_DEPTH_MAP.children`, `SLIDER_NUM_SLICES.value` | Calls `WorkflowService.configure_thresholds`; logs missing-depth/threshold info; triggers WEB-05 | WorkflowService | 1, 10 | [ ] |
| WEB-07 | `update_input_image` webui.py:428 | Input `UPLOAD_IMAGE.contents`; gated on active tab classnames | Decodes upload, calls `WorkflowService.upload_image`; serves input image, resets depth-map placeholder, triggers depth generation | WorkflowService | 1 | [ ] |
| WEB-09 | `generate_depth_map_from_button` webui.py:556 | Input `BTN_GENERATE_DEPTHMAP.n_clicks`; `running=` disables button | Sets `STORE_TRIGGER_GEN_DEPTHMAP` to kick off depth generation | inline (trigger only) | none (button never clicked in tests; upload auto-triggers depth instead) | [ ] |
| WEB-10 | `generate_depth_map_callback` webui.py:570 | Input `STORE_TRIGGER_GEN_DEPTHMAP.data` | Calls `WorkflowService.generate_depth` (resets thresholds, caches model) | WorkflowService | 1 | [ ] |
| WEB-20 | `generate_slices_request` webui.py:899 | Input `BTN_GENERATE_SLICE.n_clicks` (`#generate-slice-button`) | Forwards click count into `STORE_GENERATE_SLICE` to trigger WEB-25 | inline (trigger only) | 1 | [ ] |
| WEB-25 | `generate_slices` webui.py:1123 | Input `STORE_GENERATE_SLICE.data` | Calls `WorkflowService.generate_slices` (expand=5, full save); triggers WEB-21 | WorkflowService | 1 | [ ] |

## Segmentation

| ID | Function (file:line) | Trigger(s) | Effect | Backend service | E2E coverage | Svelte |
| --- | --- | --- | --- | --- | --- | --- |
| WEB-08 | `click_event` webui.py:473 | Input `SEG_MULTI_COMMIT.n_clicks`, `el.n_events` (image click); State rect data, mode, filename | Routes plain/Shift/Ctrl clicks to `SegmentationService.select_depth_point` / `select_instance_point`, or commits queued multipoint via `commit_multi_point`; renders resulting mask preview and appends log lines | SegmentationService | 2, 3, 4, 5 (also exercised over HTTP by `POST .../segmentation/click` and `.../segmentation/commit`, `test_api_segmentation.py`, PR 3) | [x] Svelte: `InputImagePanel.svelte`'s `onImageClick`/`onImageContextMenu` + `workflow.clickSegmentation`/`commitMultiPoint` (`POST .../segmentation/click`, `.../segmentation/commit`); depth/instance replace/Shift-union/Ctrl-subtract and the commit log line are asserted on both UIs by e2e scenarios 2, 3, 4, and 5 |
| CMP-21 | `toggle_segmentation_buttons` components.py:1663 | Input `DROPDOWN_MODE_SELECTOR.value`, `STORE_APPSTATE_FILENAME.data` | Enables/disables Multi and Commit buttons when mode == "segment" and a project is loaded | inline | 1–11 (fires on every filename set); mode explicitly switched in 2, 3, 5; also `e2e/ux-parity.spec.ts`'s "queued multi-point markers appear at the clicked points" (switches to Instance Segmentation before toggling Multi) | [x] Svelte: `InputImagePanel.svelte`'s `segmentationToolsActive` (`uiStore.segmentationMode === 'segment' && !!projectStore.view`) gates the Multi/Commit buttons' `disabled` attribute exactly like Dash's `value == "segment" and filename is not None` - same two conditions, same precedence. Asserted on both UIs by scenario 3 and `ux-parity.spec.ts` |
| CMP-22 | `toggle_multi_point` components.py:1676 | Input `SEG_MULTI_POINT.n_clicks` (`#multi-point`) | Calls `SegmentationService.set_multi_point_mode`; toggles selected/unselected class; clears point-preview via `STORE_CLEAR_PREVIEW` (CLI-06) | SegmentationService | 3 (also exercised over HTTP by `PUT .../segmentation/multi-point`, `test_api_segmentation.py`, PR 3) | [x] Svelte: `InputImagePanel.svelte`'s Multi button + `workflow.setMultiPointMode` (`PUT .../segmentation/multi-point`); toggle/aria-pressed and queue-reset-on-toggle asserted on both UIs by e2e scenario 3. The point-preview canvas dot itself is CLI-05/CLI-06's own row (now also ported, see below) |
| CLI-05 | `visualize_point` clientside.py:39 | Input `STORE_CLICKED_POINT.data` (written only by WEB-08's queued-point branch) | Draws a green/red dot on the preview canvas for a queued multipoint click | inline (JS) | 3; `e2e/ux-parity.spec.ts`'s "queued multi-point markers appear at the clicked points" | [x] Svelte: `components/canvas/PreviewOverlay.svelte` renders one marker per `ProjectView.segmentation.queuedPoints` entry (green for `negative: false`, red for `negative: true` - same colors as Dash's `rgba(0,255,0,1)`/`rgba(255,0,0,1)`), positioned in source-image-pixel percentage coordinates inside the same zoom/pan-transformed `.image-stack` box the image/mask canvas live in, so markers track zoom/pan exactly (counter-scaled by `1/viewportStore.scale` for a constant on-screen dot size - a deliberate, documented deviation from Dash's own literal behavior, which bakes the dot into the same CSS-scaled canvas and so lets it visually grow with zoom too; see PreviewOverlay.svelte's own doc comment). DashDriver asserts the same behavior by sampling `#preview-canvas`'s own pixels for the expected colors (there is no DOM node to query). Asserted on both UIs by `ux-parity.spec.ts` |
| CLI-06 | `preview_canvas_clear` clientside.py:46 | Input `STORE_CLEAR_PREVIEW.data` (written only by CMP-22) | Clears the preview canvas overlay when multipoint mode is toggled | inline (JS) | 3 | [x] Svelte: no separate clear step needed - `workflow.setMultiPointMode` always clears the server-held queue (see CMP-22's own row), and `PreviewOverlay.svelte`'s markers are a plain `{#each ProjectView.segmentation.queuedPoints}`, so an empty queue already renders no markers at all. Exercised the same way CMP-22 is (scenario 3's toggle/reset assertions) |

## Slice editing

Backend logic for every row below (except WEB-22, a pure UI reveal, and
WEB-24, which already used `InpaintingService`) is now extracted into
`SliceEditingService` (`parallax_maker/slice_editing_services.py`,
characterized by `test_slice_editing_services.py`) and exposed over HTTP by
`parallax_maker/api/slice_editing.py` (`test_api_slice_editing.py`). Dash
itself is unchanged and still runs its own inline `AppState` mutations; the
"Backend service" column reflects what the service/API layer now uses, not
a Dash rewire. See the "SE-N" legend below `e2e/parallax-maker.spec.ts`'s
own legend for what each new `e2e/slice-editing.spec.ts` scenario pins down,
and "Known quirks" for several surprising discoveries made while
characterizing these rows.

| ID | Function (file:line) | Trigger(s) | Effect | Backend service | E2E coverage | Svelte |
| --- | --- | --- | --- | --- | --- | --- |
| WEB-12 | `delete_slice_request` webui.py:623 | Input `BTN_DELETE_SLICE.n_clicks` | Deletes `state.selected_slice`; JSON-only save; re-renders main image | SliceEditingService (`delete_slice`) | SE-3, SE-4 (also `DELETE .../slices/{index}`, `test_api_slice_editing.py`) | [x] Svelte: `SegmentationTab.svelte`'s Delete button + `workflow.deleteSlice` (`DELETE .../slices/{index}`); a client-side "No slice selected" no-op when nothing is selected (matching Dash's own precedence, since the API route requires an index) otherwise the selected index; asserted on both UIs by SE-3, SE-4 |
| WEB-13 | `copy_to_clipboard` webui.py:654 | Input `BTN_COPY_SLICE.n_clicks` | Copies composed slice (or full image) + current mask into `state.clipboard_image` (in-memory only, not persisted) | SliceEditingService (`copy_to_clipboard`) | SE-7 (also `POST .../clipboard/copy`, `test_api_slice_editing.py`) | [x] Svelte: `SegmentationTab.svelte`'s Copy button + `workflow.copySlice` (`POST .../clipboard/copy`); "No mask selected" no-op without a mask matches Dash's own check; asserted on both UIs by SE-7 |
| WEB-14 | `paste_clipboard_request` webui.py:688 | Input `BTN_PASTE_SLICE.n_clicks` | Blends clipboard image into selected slice **in place** (mutates slice array directly), bumps version, JSON-only save | SliceEditingService (`paste_clipboard`) | SE-7, SE-8 (also `POST .../clipboard/paste`, `test_api_slice_editing.py`) | [x] Svelte: `SegmentationTab.svelte`'s Paste button + `workflow.pasteSlice` (`POST .../clipboard/paste`); reproduces Dash's exact precedence (clipboard checked before selection: "Nothing in the clipboard" vs. "No slice selected"); asserted on both UIs by SE-7, SE-8 |
| WEB-15 | `remove_mask_slice_request` webui.py:726 | Input `BTN_REMOVE_SLICE.n_clicks` | Subtracts current mask from selected slice's alpha in place, bumps version, JSON-only save | SliceEditingService (`remove_mask_from_slice`) | SE-5, SE-6 (also `POST .../slices/{index}/remove-mask`, `test_api_slice_editing.py`) | [x] Svelte: `SegmentationTab.svelte`'s Remove button + `workflow.removeMaskFromSlice` (`POST .../slices/{index}/remove-mask`); reproduces Dash's exact precedence (mask checked before selection); asserted on both UIs by SE-5, SE-6 |
| WEB-16 | `add_mask_slice_request` webui.py:766 | Input `BTN_ADD_SLICE.n_clicks`; `running=` disables button | Creates a masked crop of `state.imgData` and blends it into the selected slice in place, bumps version, JSON-only save | SliceEditingService (`add_mask_to_slice`) | SE-5, SE-6 (also `POST .../slices/{index}/add-mask`, `test_api_slice_editing.py`) | [x] Svelte: `SegmentationTab.svelte`'s Add button + `workflow.addMaskToSlice` (`POST .../slices/{index}/add-mask`); same mask-then-selection precedence as Remove; asserted on both UIs by SE-5, SE-6, including the "mask does not survive the mutation" chained side effect |
| WEB-18 | `create_single_slice_request` webui.py:832 | Input `BTN_CREATE_SLICE.n_clicks`; `running=` disables button | Creates a brand-new `ImageSlice` from the current mask (or an empty transparent slice if no mask), appends and selects it, saves image + JSON | SliceEditingService (`create_slice`) | SE-1, SE-2 (also `POST .../slices/create`, `test_api_slice_editing.py`) | [x] Svelte: `SegmentationTab.svelte`'s Create button + `workflow.createSlice` (`POST .../slices/create`); asserted on both UIs by SE-1, SE-2 |
| WEB-19 | `balance_slices_request` webui.py:875 | Input `BTN_BALANCE_SLICE.n_clicks` | Calls `AppState.balance_slices_depths()`, JSON-only save | SliceEditingService (`balance_slices`) | SE-15 pins the *current, broken* Dash behavior (`test.fail()`); the fixed method itself is proven by `test_controller.py::TestBalanceSlicesDepths` and `POST .../slices/balance`, `test_api_slice_editing.py` — see Known quirks | [x] Svelte: `SegmentationTab.svelte`'s Balance button + `workflow.balanceSlices` (`POST .../slices/balance`). SE-15 is `test.fail()`-pinned **on Dash only** (`webui.py:883`'s `state.image_depths` bug — see "Known quirks"); on Svelte the same scenario genuinely passes end-to-end against the fixed `balance_slices_depths()`, since the Svelte control goes through the API, not the frozen Dash callback |
| WEB-22 | `display_depth_input` webui.py:1037 | Input `{slice-depth-display,MATCH}.n_clicks` | Un-hides the numeric depth `<input>` on a slice thumbnail | inline | SE-9 (indirectly, via `setSliceDepth`); see Known quirks for a real hit-testing limitation this row has when its slice is selected | [x] Svelte: `SegmentationTab.svelte`'s depth badge (`slice-depth-display`) toggles to a numeric input (`slice-depth-input`) on click, styled with `pointer-events: auto` and no competing z-index against `.slice-overlay` (which stays `pointer-events: none`) — so, unlike Dash, the badge stays clickable on a *selected* slice too, rather than reproducing that hit-testing bug; committed on Enter or blur via `workflow.setSliceDepth`; asserted (on an unselected slice, to stay comparable with Dash) on both UIs by SE-9 |
| WEB-23 | `record_depth_input` webui.py:1053 | Input `{slice-depth-input,ALL}.value`/`.n_submit` | Calls `state.change_slice_depth`, possibly reorders/deselects, JSON-only save; also flips `STORE_INPAINTING`, which always clears `state.selected_inpainting` (CMP-07) regardless of whether the edited slice was selected | SliceEditingService (`set_slice_depth`) | SE-9 (also `PUT .../slices/{index}/depth`, `test_api_slice_editing.py`) | [x] Svelte: see WEB-22 — same control commits via `PUT .../slices/{index}/depth`; reorder-clears-selection is asserted on both UIs by SE-9 |
| WEB-24 | `undo_slice` webui.py:1083 | Input `{slice-undo-backwards/forwards,ALL}.n_clicks` | Calls `InpaintingService.move_slice_version` (FORWARD/BACKWARD) to step a slice's saved image-version history; triggers WEB-21 re-render | InpaintingService | 6, 9, SE-5 (also `POST .../slices/{index}/undo`/`redo`, `test_api_slice_editing.py`, new in this PR) | [x] Svelte: `SegmentationTab.svelte`'s per-slice undo/redo carets (`slice-undo`/`slice-redo`, `title="Undo last change"`/`"Redo last change"`, disabled per `SliceView.canUndo`/`canRedo`) + `workflow.undoSlice`/`redoSlice` (`POST .../slices/{index}/undo`/`redo`); asserted on both UIs by SE-5 (scenarios 6/9 are inpainting-only and remain skipped on Svelte, since undo/redo there is exercised through candidates the Inpainting tab doesn't have yet) |
| WEB-34 | `slice_upload` webui.py:1428 | Input `{UPLOAD_SLICE,ALL}.contents` (per-thumbnail drag/drop) | Decodes dropped image, fixes aspect ratio, writes a new slice version, JSON-only save, and re-composes `state.imgData` from all slices | SliceEditingService (`replace_slice_image`) | SE-10, SE-11 (also `PUT .../slices/{index}/image`, `test_api_slice_editing.py`); see Known quirks for two real bugs this row pins as-is | [x] Svelte: `SegmentationTab.svelte`'s per-thumbnail drop target + hidden `slice-upload-input` file input + `workflow.uploadSliceImage` (`PUT .../slices/{index}/image`). SE-10 (matching-aspect upload) is asserted identically on both UIs. SE-11 (mismatched-aspect) is now `test.fail()`-pinned **on Svelte only**: `SliceEditingService.replace_slice_image` intentionally fixes Dash's collapse-to-near-1px bug by resizing to the slice canvas instead (see `test_replace_slice_image_is_resized_to_the_slice_canvas`), and its log line is not a copy of Dash's non-interpolated literal — Dash keeps genuinely passing SE-11 as-is (unaffected, still pinning the real bug), Svelte cannot match either assertion by design and is `test.fail()`-pinned instead |

## Mask tools

| ID | Function (file:line) | Trigger(s) | Effect | Backend service | E2E coverage | Svelte |
| --- | --- | --- | --- | --- | --- | --- |
| WEB-11 | `update_depth_map_callback` webui.py:591 | Input `STORE_TRIGGER_UPDATE_DEPTHMAP.data` | Encodes `state.depthMapData` as a PNG `<img>` for `#depthmap-image` | inline (AppState) | 1, 10 | [ ] |
| WEB-21 | `update_slices` webui.py:913 | Input `STORE_UPDATE_SLICE.data` | Rebuilds the slice-thumbnail strip (undo/redo carets, depth badges, upload targets); re-renders the composed main image for the selected slice and clears `slice_pixel`/`slice_mask` | inline (AppState) | 1, 6, 9, 10 (the selected-slice display/clear half is also exercised over HTTP by `POST .../slices`, `test_api_jobs.py`/`test_api_mutations.py`, PR 3) | [x] Svelte: `SegmentationTab.svelte`'s thumbnail strip renders the undo/redo carets, depth badge and per-thumbnail upload target for every `ProjectView.slices[i]`; the "recompose preview and clear the mask when a slice is selected" half is the shared `refresh_selection_preview()` every slice-editing/mask-tool route applies server-side (see ARCHITECTURE.md), reflected client-side by `projectStore.applyView`. Asserted on both UIs by scenario 1 (thumbnail rendering) and every SE-N scenario that follows an operation with a selected slice (SE-5 most directly, via the "mask does not survive Add" chain) |
| WEB-26 | `display_slice` webui.py:1150 | Input `{slice,ALL}.n_clicks`, `{slice-overlay,ALL}.n_clicks` | Selects/deselects a slice, composes checkerboard/grayscale preview, loads its saved prompts, calls `InpaintingService.clear_selection` | InpaintingService (selection) + inline (AppState) | 5, 6, 7, 8, 9, 10 (also exercised over HTTP by `PUT .../selection`, `test_api_segmentation.py`, PR 3 — prompt loading stays a Svelte-side concern, since `ProjectView.slices[i]` already carries the prompts) | [x] Svelte: `SegmentationTab.svelte`'s thumbnail click (`onSliceClick`) + `workflow.selectSlice` (`PUT .../selection`, sending `slice: null` to deselect instead of Dash's click-to-toggle); selection and the composed-slice segmentation source are asserted on both UIs by e2e scenario 5 (prompt loading into the Inpainting tab is not yet ported) |
| CMP-19 | `invert_mask` components.py:1615 | Input `SEG_INVERT_MASK.n_clicks` ("Invert") | Inverts `state.slice_mask` (creating an all-zero mask first if none exists) and re-renders masked preview | SliceEditingService (`invert_mask`) | SE-12 (also `POST .../mask/invert`, `test_api_slice_editing.py`) | [x] Svelte: `InputImagePanel.svelte`'s Invert button + `workflow.invertMask` (`POST .../mask/invert`); asserted on both UIs by SE-12 |
| CMP-20 | `blur_mask` components.py:1639 | Input `SEG_FEATHER_MASK.n_clicks` ("Feather") | `cv2.blur`s `state.slice_mask` by a fixed 10px kernel and re-renders masked preview | SliceEditingService (`feather_mask`) | SE-13 (also `POST .../mask/feather`, `test_api_slice_editing.py`) | [x] Svelte: `InputImagePanel.svelte`'s Feather button + `workflow.featherMask` (`POST .../mask/feather`); a client-side "No mask to feather" no-op without a mask matches Dash's own `blur_mask` message; asserted on both UIs by SE-13 |
| CMP-23 | `toggle_checkerboard` components.py:1706 | Input `SEG_TOGGLE_CHECKERBOARD.n_clicks` | Toggles `state.use_checkerboard`, re-composes the selected slice preview in the new mode | SliceEditingService (`set_checkerboard`) | SE-14 (also `PUT .../display`, `test_api_slice_editing.py`) | [x] Svelte: `InputImagePanel.svelte`'s checkerboard toggle (`toggle-checkerboard`, `aria-pressed`) + `workflow.toggleCheckerboard` (`PUT .../display`); no-op main image without a selection, recompose with one, matching Dash; asserted on both UIs by SE-14 |

## Canvas/Inpainting

| ID | Function (file:line) | Trigger(s) | Effect | Backend service | E2E coverage | Svelte |
| --- | --- | --- | --- | --- | --- | --- |
| WEB-17 | `update_prompt_text` webui.py:804 | Input `TEXT_POSITIVE_PROMPT.value`, `TEXT_NEGATIVE_PROMPT.value` (`#positive-prompt`/`#negative-prompt`) | Persists normalized prompts via `InpaintingService.update_prompts` (survives model failure); no Output | InpaintingService | 6 (also exercised over HTTP by `PUT .../slices/{index}/prompts`, `test_api_inpainting.py`, PR 5) | [x] Svelte: `InpaintingTab.svelte`'s Positive/Negative Prompt textareas (`positive-prompt`/`negative-prompt`), loaded from `SliceView.positivePrompt`/`negativePrompt` on selection and committed on `onchange` via `workflow.updateInpaintingPrompts` (`PUT .../slices/{index}/prompts`); the commit registers with `state/canvas.svelte.ts`'s `canvasSaveStore` so it can never race Generate/a slice-selection change into a `409 busy` the way an un-awaited fire-and-forget PUT did during implementation. Loading/persisting and the final restored values (via a slice reselect) are asserted on both UIs by scenario 6 (`fillPrompts`/`expectPrompts`) |
| CMP-02 | `enable_apply_inpainting_button` components.py:684 | Input `CTR_INPAINTING_DISPLAY.children`, `{inpainting-image,ALL}.className` | Enables `#apply-inpainting-button` only when `state.selected_inpainting` is a valid, visibly-selected candidate index | inline (AppState read) | 6 | [x] Svelte: `InpaintingTab.svelte`'s `canApply` (`!!candidates && selectedCandidate !== null && !isBusy()`) gates `apply-inpainting`; asserted on both UIs by scenario 6's `selectCandidate`/`applyCandidate` |
| CMP-03 | `erase_inpainting` components.py:707 | Input `BTN_ERASE_INPAINTING.n_clicks` (`#erase-inpainting-button`); `running=` disables button | Calls `InpaintingService.erase` (inverse-slice-alpha patch, new version, JSON/file-mapping save); logs slice index; triggers WEB-21 | InpaintingService | 9 (also exercised over HTTP by `POST .../slices/{index}/inpainting/erase`, `test_api_inpainting.py`, PR 5) | [x] Svelte: `InpaintingTab.svelte`'s Erase button (`erase-inpainting`) + `workflow.eraseInpainting` (`POST .../slices/{index}/inpainting/erase`, after flushing any pending mask/prompt save); asserted on both UIs by scenario 9 |
| CMP-04 | `update_inpainting_image_display` components.py:745 | Input `BTN_GENERATE_INPAINTING`/`BTN_FILL_INPAINTING`/`BTN_ENHANCE` `.n_clicks` (`#generate-inpainting-button`/`#fill-inpainting-button`/`#enhance-button`); `running=` disables all three buttons | Decodes ComfyUI workflow if needed, calls `InpaintingService.generate_candidates` in PAINT/FILL/ENHANCE mode, renders returned candidate images as data URLs | InpaintingService | 6 (PAINT), 7 (FILL), 8 (ENHANCE) (also exercised over HTTP by `POST .../slices/{index}/inpainting/generate`, which stores the result server-side as a candidate set instead of browser data URLs - see ARCHITECTURE.md's "Candidates" section - `test_api_inpainting.py`, PR 5) | [x] Svelte: `InpaintingTab.svelte`'s Generate/Fill/Enhance buttons (`generate-inpainting`/`fill-inpainting`/`enhance-inpainting`) + `workflow.generateInpainting('paint'\|'fill'\|'enhance', ...)` (`POST .../slices/{index}/inpainting/generate`, a job, polled to completion); a failed job leaves the previous candidate set untouched (unit-tested in `workflow.inpainting.test.ts`) instead of clearing it eagerly. Asserted on both UIs by scenarios 6/7/8 |
| CMP-05 | `select_inpainting_image` components.py:823 | Input `{inpainting-image,ALL}.n_clicks` | Calls `InpaintingService.select_candidate`; toggles the clicked candidate's highlight class off if re-clicked (deselect); previews selected candidate or falls back to composed slice image | InpaintingService | 6 (also exercised over HTTP by `PUT .../inpainting/selection`, `test_api_inpainting.py`, PR 5; the API now also swaps the main image to the selected candidate and back to the slice composite on deselect, `test_selected_candidate_is_previewed_in_the_main_image`) | [x] Svelte: `InpaintingTab.svelte`'s candidate strip (`candidate-image`, `aria-selected`) + `workflow.selectInpaintingCandidate` (`PUT .../inpainting/selection`, always sending the clicked index - the toggle-off-on-re-click is `InpaintingService.select_candidate`'s own contract, not reimplemented client-side); asserted on both UIs by scenario 6 (`selectCandidate`) and by `InpaintingTab.test.ts`'s toggle/apply-enablement tests. The main-image preview swap on selection is not ported |
| CMP-06 | `apply_inpainting` components.py:875 | Input `BTN_APPLY_INPAINTING.n_clicks` (`#apply-inpainting-button`); `running=` disables button | Decodes candidate data URLs to PIL images, calls `InpaintingService.apply_candidate` (writes new image version + JSON/file-mapping save); triggers WEB-21 and CMP-07 | InpaintingService | 6 (also exercised over HTTP by `POST .../slices/{index}/inpainting/apply`, which additionally rejects a stale `generationId`/slice/version with `409 stale_revision`, `test_api_inpainting.py`, PR 5) | [x] Svelte: `InpaintingTab.svelte`'s Apply button (`apply-inpainting`) + `workflow.applyInpaintingCandidate` (`POST .../slices/{index}/inpainting/apply`, after flushing any pending mask/prompt save); asserted on both UIs by scenario 6 (log text + version change via Undo) |
| CMP-07 | `react_selected_slice_change` components.py:916 | Input `STORE_INPAINTING.data` | Enables/disables all inpainting controls based on whether a slice is selected; calls `InpaintingService.clear_selection`; writes `STORE_SELECTED_SLICE` (feeds CLI-04) | InpaintingService | 5, 6, 7, 8, 9, 10 (the `clear_selection` call is also exercised over HTTP by `PUT .../selection`, `test_api_segmentation.py`, PR 3, which now also drops the API's server-held candidate set the same way, `test_api_inpainting.py`, PR 5; control enable/disable stays a Svelte-side concern) | [x] Svelte: `InpaintingTab.svelte`'s `hasSlice`/`canGenerate`/`canApply` derived state disables Generate/Fill/Enhance/Erase/Apply whenever `ProjectView.selectedSlice` is `null`; the server-side `clear_selection` half is exercised by `PUT .../selection` as noted above. Asserted on both UIs by scenarios 5-9 (a selected slice's controls behave correctly) and by `InpaintingTab.test.ts`'s "disables ... with no project loaded" test |
| CMP-24 | `save_slice_mask` components.py:1748 | Input `CANVAS_DATA.data` (from CLI-09's mouseout save) | Empty string → `InpaintingService.delete_mask`; else decodes canvas PNG → `InpaintingService.save_mask` (alpha resized BICUBIC, padding, ROI-crop flag); returns bounding box for CLI-07 | InpaintingService | 6, 9 (also exercised over HTTP by `PUT`/`DELETE .../slices/{index}/mask`, `test_api_inpainting.py`, PR 5). **Updated in the UX-parity slice**: `PUT .../slices/{index}/mask` now accepts an additional multipart field `cropToRegion` (default `"true"`, matching Dash's own `CHECKLIST_REGION_OF_INTEREST` default-checked state) and threads it through to `show_crop_region`; the response gains an additive `boundingBox: [x0,y0,x1,y1] \| null` field (not part of the generated `ProjectView` schema - see `api/inpainting.py`'s `save_inpainting_mask`) for PreviewOverlay.svelte's CLI-07-equivalent preview. `test_api_inpainting.py::test_mask_save_defaults_to_cropping_region_of_interest_and_returns_bounding_box`/`test_mask_save_with_crop_to_region_false_omits_bounding_box` | [x] Svelte: `MaskCanvas.svelte`'s Clear button (`canvas-clear`) + `workflow.deleteMask` (`DELETE .../mask`); painting itself (`PUT .../mask` on pointerup) is CLI-09's row, now sending `uiStore.cropToRoi` as `cropToRegion` and feeding a returned `boundingBox` into `canvasPreviewStore.showRoiBox` (see CLI-07). Save/delete asserted on both UIs by scenarios 6 and 9 |
| CMP-25 | `load_canvas_mask` components.py:1799 | Input `BTN_LOAD_CANVAS.n_clicks` ("Load") | Calls `InpaintingService.load_mask`; re-renders it as RGBA `(r,0,0,r)` and feeds CLI-08 to paint it back onto the canvas | InpaintingService | none (handoff-listed gap: canvas load) (the RGBA `(r,0,0,r)` rendering itself is also exercised over HTTP by `GET .../assets/mask-{index}`, `test_api_inpainting.py`, PR 5, served directly from disk rather than through `InpaintingService.load_mask`, which is selection-scoped) | [ ] Svelte: `MaskCanvas.svelte`'s Load button (`canvas-load`) fetches `SliceView.mask.url` (already rendered server-side as RGBA `(r,0,0,r)`, exactly as this row describes) and draws it onto the canvas via `drawImage`; not checked here because no shared e2e scenario exercises the Load button (same handoff-listed gap as Dash) |
| CLI-04 | `record_selected_slice` clientside.py:30 | Input `STORE_SELECTED_SLICE.data` (written by CMP-07) | Sets JS `currentSlice`, which gates whether CLI-09/JS-01 record paint strokes and drives JS-06's contextual help text | inline (JS) | 5, 6, 7, 8, 9, 10 | [x] Svelte: `MaskCanvas.svelte`'s slice-transition `$effect` (`lastSelected`/`loadToken`) is the equivalent gate - it flushes any pending save, then clears/reloads the canvas only on a real `ProjectView.selectedSlice` change, so a stroke is always attributed to the slice it was drawn on. The contextual-help-text half (JS-06) is not ported. Exercised on both UIs by scenarios 6 and 9 (paint only persists/affects the selected slice) |
| CLI-07 | `show_bounding_box` clientside.py:55 | Input `STORE_BOUNDING_BOX.data` (written by CMP-24) | Draws a 2s ROI-preview rectangle on the preview canvas | inline (JS) | none (no *e2e* test enables the ROI checkbox on either UI, so `bounding_box` stays untested end-to-end there; pytest covers the Svelte/API side directly - see CMP-24) | [ ] Svelte: implemented - `MaskCanvas.svelte`'s `saveCurrentCanvas` passes `uiStore.cropToRoi` (default checked, matching Dash) as the mask-save request's `cropToRegion` field and feeds the response's `boundingBox` into `canvasPreviewStore.showRoiBox`, which auto-clears after 2000ms (`setTimeout`, matching Dash's own `previewRect`'s timing exactly); `PreviewOverlay.svelte` renders it as an orange (`rgb(255,128,0)`, matching Dash's `rgba(255,128,0,1)`) rectangle, counter-scaled border width for a constant on-screen line thickness (see CLI-05's row for the same zoom/pan-tracking technique). Not ticked because no *shared e2e* scenario exercises it (same handoff-listed gap as Dash itself) - verified instead by `test_api_inpainting.py`'s bounding-box tests (CMP-24) and by manual screenshot inspection (`ux-inpainting-*.png`) |
| CLI-08 | `canvas_load` components.py:1818 | Input `CANVAS_MASK_DATA.data` (written by CMP-25) | Sets up the main canvas if needed and draws the loaded mask image onto it | inline (JS) | none | [ ] Svelte: see CMP-25's `drawMaskImage`; not checked for the same reason (no shared scenario exercises Load) |
| CLI-09 | `canvas_draw` components.py:1825 | Input `CANVAS_PAINT.event` (mousedown/mouseup/**mouseout**/mouseenter via `EventListener`) | Sets up canvas on first use; starts/stops drawing; **saves on `mouseout`, not `mouseup`** — returns `canvas.toDataURL()` into `CANVAS_DATA` only if a stroke was drawn since last save | inline (JS) | 6, 9 | [x] Svelte: `MaskCanvas.svelte`'s `pointerdown`/`pointermove`/`pointerup` handlers, with pointer capture - a deliberate improvement over Dash's `mouseout`-triggered save (see the migration handoff's "Deterministic harness details"): the canvas saves directly on `pointerup` via `workflow.saveMask`, registered with `canvasSaveStore` so Generate/Apply/Erase/a slice-selection change all await it first (`state/canvas.svelte.ts`, unit-tested in `canvas.svelte.test.ts` and `workflow.inpainting.test.ts`). The backing store is sized to the source image's own pixel dimensions rather than the CSS-rendered size Dash uses, so the saved mask maps 1:1 with no resampling and a window resize never touches the canvas's pixel content. Asserted on both UIs by scenarios 6 and 9 (`drawMaskStroke`, which waits for the `Saved mask for slice` log line on both) |
| CLI-10 | `canvas_clear` (Clear button) components.py:1832 | Input `BTN_CLEAR_CANVAS.n_clicks` | Clears canvas pixels and resets cached 2D context | inline (JS) | none (handoff-listed gap) | [ ] Svelte: implemented (`MaskCanvas.svelte`'s `canvas-clear` button, `onClear` - clears pixels and calls `DELETE .../mask`); not checked because no shared scenario clicks it (same handoff-listed gap as Dash) |
| CLI-11 | `canvas_clear` (auto-clear) components.py:1838 | Input `IMAGE.src` (**every** main-image change) | Clears canvas pixels whenever the main image updates — see Known quirks (`# XXX - this will kill the canvas during inpainting - bad`) | inline (JS) | 1–11 (fires on virtually every image update) | [ ] Deliberately **not** reproduced in Svelte - this row *is* the bug the handoff's "give the new canvas an explicit lifecycle" guidance calls out by name. `MaskCanvas.svelte` never watches `mainImage`/`ProjectView` updates in general; it only clears/reloads on an actual `selectedSlice` transition (see CLI-04's row), so a Generate/Fill/Enhance/Apply/Erase in progress - all of which repaint the main image - can never silently wipe an unsaved stroke the way Dash's `IMAGE.src`-triggered `canvas_clear` can |
| CLI-12 | `canvas_toggle_erase` components.py:1845 | Input `BTN_ERASE_MODE.n_clicks` ("Erase" — the canvas eraser-brush toggle, distinct from `#erase-inpainting-button`) | Flips `isErasing`, switches `globalCompositeOperation` between `source-over`/`destination-out` and stroke color/width | inline (JS) | none | [ ] Svelte: implemented (`MaskCanvas.svelte`'s `canvas-erase-mode` button, `toggleErase` - flips `isErasing`, switches `globalCompositeOperation`/brush width exactly like this row); not checked because no shared scenario exercises the paint-canvas eraser toggle (same handoff-listed gap as Dash - distinct from `#erase-inpainting-button`/CMP-03, which *is* covered) |
| JS-01 | canvas `mousemove` → `draw`/`previewBrush` utility.js:288,152,58 | Native `mousemove` on `#canvas` | Live brush-size preview circle when idle; paints the red stroke and records `canvasLastDrawnTime` while `isDrawing` | inline (JS) | 6, 9 (via `drawCanvasStroke` helper) | [x] Svelte: `MaskCanvas.svelte`'s `pointermove`/`moveStroke` paints the stroke (see CLI-09). **Updated in the UX-parity slice**: the idle live brush-size preview circle is now also ported - `updateBrushPreview` (called from `pointermove` while idle, and momentarily on `pointerup`) sets `canvasPreviewStore.brush` (source-image-pixel position + diameter, converted from the CSS `drawWidth`/`eraseWidth` via the same `scaleFactor()` the real stroke's `lineWidth` uses), rendered by `PreviewOverlay.svelte` as a circle that follows the pointer, colored red (drawing) or black-ish (erasing) like Dash's `previewBrush`; cleared on `pointerleave` and on leaving the Inpainting tab. Its diameter is deliberately a constant on-screen size regardless of zoom (Dash's own preview, baked onto the same CSS-scaled canvas as the real stroke, visually grows with zoom instead - a documented simplification, not a bug fix, since `scaleFactor()` already compensates for the current zoom when computing the natural-pixel diameter). The paint half is asserted on both UIs by scenarios 6 and 9; the preview circle itself has no shared e2e scenario (not independently testable without pixel-sampling the canvas - see JS-02 for what *is* covered) but is unit-exercised indirectly via `MaskCanvas.svelte`'s `scaleFactor`/`canvasPoint` logic and visually verified in the `ux-inpainting-*.png` screenshots |
| JS-02 | canvas `contextmenu` suppression utility.js:289 | Native `contextmenu` on `#canvas` | `preventDefault()`s the native menu so Alt+Right-drag can resize the brush instead | inline (JS) | none | [x] Svelte: `MaskCanvas.svelte`'s `oncontextmenu={(event) => event.preventDefault()}` suppresses the native menu the same way. **Updated in the UX-parity slice**: Alt+Right-drag brush resize is now also ported - `beginStroke`'s `event.button === 2 && event.altKey` branch (mirroring Dash's `startDrawing`'s identical check) starts a resize instead of a paint stroke; `moveStroke`'s `adjustBrushSize` applies the exact same formula and `[5, 100]` clamp as Dash's `adjustBrushSize` (1 unit per 15px of horizontal drag). Not checked because no shared e2e scenario exercises either the menu suppression or the resize gesture (same handoff-listed gap as Dash itself) |

## Project lifecycle

| ID | Function (file:line) | Trigger(s) | Effect | Backend service | E2E coverage | Svelte |
| --- | --- | --- | --- | --- | --- | --- |
| WEB-36 | `restore_inpainting` webui.py:1514 | Input `STORE_RESTORE_STATE.data` | Sets `STORE_INPAINTING=True` to re-run CMP-07 after a restore | inline (trigger only) | 2–11 | [x] Svelte: no separate trigger needed - `InpaintingTab.svelte`'s controls already derive their enabled/disabled state straight from `ProjectView.selectedSlice`/`.inpainting`, which a restore already refreshes via `projectStore.applyView` |
| WEB-37 | `remember_camera_parameters` (2nd def, restore variant) webui.py:1527 | Input `STORE_RESTORE_STATE.data` | Populates camera-distance/focal-length/max-distance/displacement sliders from `state.camera`/`state.mesh_displacement` | inline (AppState read) | 2–11 (asserted in 10); camera/displacement round trip also pinned by `project-export.spec.ts` and `ProjectService.restore_legacy_state`/`update_settings` (`test_project_services.py`, `test_api_project_export.py`) | [x] Svelte: `ExportTab.svelte`'s camera-distance/max-distance/focal-length/displacement sliders read straight from `ProjectView.settings.camera`/`.meshDisplacement` (re-synced by a `$effect` on every view change, restore included - see its own doc comment on why that resync is gated on no local commit being in flight); asserted on both UIs by `project-export.spec.ts`'s "camera and displacement slider changes persist..." |
| WEB-38 | `restore_dark_mode` webui.py:1547 | Input `STORE_RESTORE_STATE.data` | Sets `BTN_DARK_MODE.n_clicks` parity to reflect `state.dark_mode`, indirectly re-running WEB-01's class toggle | inline (AppState read) | 2–11 (asserted in 10); dark-mode round trip also pinned by `project-export.spec.ts` and `ProjectService`/`test_api_project_export.py` | [x] Svelte: `App.svelte`'s `$effect` follows `ProjectView.settings.darkMode` into `uiStore.theme` (and thus the root `dark` class) on every view change, restore included; asserted on both UIs by `project-export.spec.ts`'s "dark mode persists..." and by `App.test.ts`'s "restoring a project applies its persisted dark-mode setting to the theme" |
| WEB-39 | `restore_api_key` webui.py:1561 | Input `STORE_RESTORE_STATE.data` | Restores `INPUT_API_KEY.value` from `state.api_key` | inline (AppState read) | 2–11 | [ ] Deliberately **not** reproduced in Svelte: the API's `apiKey` is write-only and never echoed back by any response (see ARCHITECTURE.md's Configuration endpoints table), so `ConfigurationTab.svelte`'s API-key field is never pre-filled from a restore - an intentional parity deviation (avoiding a real credential-echoing pattern Dash itself has), not a gap; `ConfigurationTab.test.ts`'s "the API key field never displays a previously-saved key back" pins this |
| WEB-40 | `restore_workflow` webui.py:1579 | Input `STORE_RESTORE_STATE.data` | Re-uploads a saved ComfyUI workflow file's bytes as a data URL if one exists on disk | inline (AppState read) | 2–11 | [ ] Svelte: `ProjectView.inpainting.hasWorkflow` (shown as a "Workflow uploaded" hint in `ConfigurationTab.svelte`) already reflects a restored workflow's presence without needing to re-upload its bytes into the browser at all - no shared scenario restores a project with a saved ComfyUI workflow to exercise either UI's version of this row |
| WEB-41 | `restore_models` webui.py:1597 | Input `STORE_RESTORE_STATE.data` | Restores depth/inpainting model dropdown selections | inline (AppState read) | 2–11 (asserted in 10); depth/inpainting model round trip also pinned by `project-export.spec.ts` (note: `ProjectView.settings.depthModel`/the restored dropdown, not the unrelated top-level `ProjectView.depthModel` field - see ARCHITECTURE.md) | [x] Svelte: `ModeTab.svelte`'s depth-model select follows `ProjectView.settings.depthModel` (falling back to the top-level `depthModel` only before that has ever been persisted - see its own doc comment); `ConfigurationTab.svelte`'s inpainting-model select already reads `ProjectView.inpainting.model` directly. Asserted on both UIs by `project-export.spec.ts`'s "depth and inpainting model selections persist..." |
| WEB-42 | `update_external_server_address` webui.py:1617 | Input `STORE_RESTORE_STATE.data` | Restores `INPUT_EXTERNAL_SERVER.value` | inline (AppState read) | 2–11 | [x] Svelte: `ConfigurationTab.svelte`'s server-address field reads straight from `ProjectView.inpainting.externalServer`, which a restore already refreshes; not independently asserted by a shared scenario (the fixture's server address matches the field's own default), but exercised implicitly by every `project-export.spec.ts` scenario that restores the fixture project |
| WEB-43 | `update_model_viewer` webui.py:1631 | Input `STORE_RESTORE_STATE.data` | Rebuilds the glTF `<iframe>` `srcDoc` from any previously exported model file | inline (AppState read) | 2–11 | [x] Svelte: `Model3DViewer.svelte` reads `ProjectView.exports.gltf` directly (content-versioned, so a restored project's already-exported scene renders immediately once the 3D tab is opened - see its own lazy-load doc comment); not independently asserted by a shared scenario (no scenario restores a project with a pre-existing export and then opens the 3D tab), but its "no scene yet" / "scene exists" branches are covered by `Model3DViewer.test.ts` |
| WEB-44 | `restore_state_slices` webui.py:1648 | Input `STORE_RESTORE_STATE.data` | Triggers WEB-21 if the restored project has slices | inline (trigger only) | 2–11 | [x] Svelte: no separate trigger needed - `SegmentationTab.svelte`'s thumbnail strip is a plain `{#each ProjectView.slices}`, already covered by `restoreFixtureState()`'s own 3-slice wait in every scenario |
| WEB-45 | `restore_state_depthmap` webui.py:1665 | Input `STORE_RESTORE_STATE.data` | Triggers WEB-11 to redraw the depth map | inline (trigger only) | 2–11 | [x] Svelte: no separate trigger needed - `ModeTab.svelte`'s depth `<img>` binds `ProjectView.assets.depth.url` directly, already covered by `restoreFixtureState()`'s own depth-image wait in every scenario |
| WEB-46 | `restore_state` webui.py:1681 | Input `UPLOAD_STATE.contents` | Decodes uploaded project JSON via `AppState.from_json` + `fill_from_files`, seeds the process-global cache, serves the restored input image, sets `STORE_RESTORE_STATE` to fan out to WEB-36..45 | `ProjectService.restore_legacy_state` (`project_services.py`) now also backs `POST /api/v1/projects/restore` itself (`api/projects.py`'s route delegates instead of duplicating the containment/validation logic) | 2–11; `test_project_services.py`, `test_api_restore.py`, `test_api_project_export.py` | [x] Svelte: `ConfigurationTab.svelte`'s "Load State" file input + `workflow.restoreProject` (`POST /api/v1/projects/restore`); asserted on both UIs by every scenario that calls `restoreFixtureState()`/`restoreStateFromBytes()` |
| WEB-47 | `save_state` webui.py:1710 | Input `BTN_SAVE_STATE.n_clicks` | Calls `AppState.to_file` (full save: image slices, depth map, input image) | `ProjectService.save_project` (`project_services.py`) | `project-export.spec.ts` ("save state then restore round trips...") exercises the Save button for the first time; also `test_project_services.py`, `POST .../projects/{id}/save`/`GET .../state-file`, `test_api_project_export.py` | [x] Svelte: `ConfigurationTab.svelte`'s "Save State" button (`save-state`) + `workflow.saveProject` (`POST .../projects/{id}/save`); matches Dash exactly - no browser download either (see ARCHITECTURE.md's `GET .../state-file` note and `ConfigurationTab.test.ts`'s "Save State calls POST /save and does not trigger any browser download"). Asserted on both UIs by every `project-export.spec.ts` scenario that calls `saveState()` |

## Configuration

| ID | Function (file:line) | Trigger(s) | Effect | Backend service | E2E coverage | Svelte |
| --- | --- | --- | --- | --- | --- | --- |
| WEB-29 | `remember_depth_model` webui.py:1247 | Input `DROPDOWN_DEPTH_MODEL.value` | Persists `state.depth_model_name` if changed, JSON-only save (contains a harmless duplicated `if` check) | `ProjectService.update_settings` (`project_services.py`, `depth_model` field) | `project-export.spec.ts` ("depth and inpainting model selections persist..."); `test_project_services.py`, `PUT .../projects/{id}/settings`, `test_api_project_export.py` | [x] Svelte: `ModeTab.svelte`'s depth-model select (`depth-model`) + `workflow.updateSettings({depthModel})`, persisted on every change regardless of whether Regenerate is ever clicked; asserted on both UIs by `project-export.spec.ts`'s "depth and inpainting model selections persist..." |
| WEB-30 | `remember_camera_parameters` (1st def, persist variant) webui.py:1272 | Input camera-distance/focal-length/max-distance/displacement sliders | Persists `state.camera`/`state.mesh_displacement` if changed, JSON-only save — **same function name as WEB-37**, see Known quirks (including a newly-discovered real 500 crash, not merely the name collision) | `ProjectService.update_settings` (`project_services.py`, `camera`/`meshDisplacement` fields; does not reproduce the swapped-constructor crash - see Known quirks and ARCHITECTURE.md) | 11 (displacement slider set before export); `project-export.spec.ts` ("camera and displacement slider changes persist..."); `test_project_services.py`, `PUT .../projects/{id}/settings`, `test_api_project_export.py` | [x] Svelte: `ExportTab.svelte`'s four camera/displacement sliders (`camera-distance`/`max-distance`/`focal-length`/`displacement`) + `workflow.updateSettings`, always sent together as one `PUT .../settings` (matching Dash's own single callback reading all four); commits are coalesced (never sent concurrently - see its own doc comment) so a fast keyboard-driven sequence can never hit `409 busy` the way an earlier, uncoordinated-per-keystroke version did during implementation. Asserted on both UIs by `project-export.spec.ts`'s "camera and displacement slider changes persist..." and `ExportTab.test.ts`'s "commits all four camera/displacement fields together on slider release" |
| WEB-31 | `remember_inpaint_model` webui.py:1301 | Input `DROPDOWN_INPAINT_MODEL.value` | Calls `InpaintingService.update_model`; clears `CTR_INPAINTING_DISPLAY` only when the model actually changed (not on `InpaintingUnchanged`) | InpaintingService | none (also exercised over HTTP by `PUT .../inpainting/settings`'s `model` field, including the candidate-clearing behavior, `test_api_inpainting.py`, PR 5); model round trip also pinned by `project-export.spec.ts` via `ProjectView.settings.depthModel`/the restored dropdown - see WEB-41 | [x] Svelte: `ConfigurationTab.svelte`'s inpainting-model select (`inpainting-model`, pre-existing) + `workflow.updateInpaintingSettings({model})`; model round trip pinned the same way as Dash by `project-export.spec.ts`'s "depth and inpainting model selections persist..." |
| CMP-08 | `validate_workflow` components.py:955 | Input `UPLOAD_COMFYUI_WORKFLOW.contents` | Validates a dropped ComfyUI workflow JSON via `patch_inpainting_workflow`; persists it to `state.workflow_path()` if valid | inline (`patch_inpainting_workflow` + AppState) | none | [ ] Svelte: `ConfigurationTab.svelte`'s ComfyUI Workflow upload (`comfyui-workflow-input`, shown only for the `comfyui` model - CMP-10) + `workflow.uploadInpaintingWorkflow` (`PUT .../inpainting/workflow`); not checked here because no shared scenario uploads a workflow file (same handoff-listed gap as Dash) |
| CMP-09 | `toggle_blur_slider` components.py:1000 | Input `DROPDOWN_INPAINT_MODEL.value` | Disables the guidance/blur slider for `stabilityai` (no mask-blur support) | inline | none | [ ] Svelte: `ConfigurationTab.svelte`'s Mask Blur slider (`mask-blur`) disables for `stabilityai` the same way (`blurDisabled`); not checked because no shared scenario exercises it (same handoff-listed gap as Dash) |
| CMP-10 | `toggle_automatic_config` components.py:1012 | Input `DROPDOWN_INPAINT_MODEL.value` | Shows/hides the Automatic1111/ComfyUI server config and ComfyUI workflow-upload panels | inline | none | [x] Svelte: `ConfigurationTab.svelte`'s `showServerPanel`/`showWorkflowUpload` derived state (`external-server-panel`/`comfyui-workflow-panel`); visibility asserted by `ConfigurationTab.test.ts`'s "shows the Automatic1111/ComfyUI server panel..."/"shows the ComfyUI workflow upload only for comfyui..." |
| CMP-11 | `reset_external_server_address` components.py:1029 | Input `INPUT_EXTERNAL_SERVER.value` | Persists `state.server_address`; clears the success/failure highlight class | inline (AppState) | none | [x] Svelte: `ConfigurationTab.svelte`'s server-address field (`external-server-address`) + `workflow.updateInpaintingSettings({externalServer})`, resetting `uiStore.externalConnectionStatus` to `'none'` on every edit; the reset-on-edit half is asserted by `ConfigurationTab.test.ts`'s "Test Connection highlights success and resets to none when the server address is edited again" |
| CMP-12 | `test_external_connection` components.py:1054 | Input `BTN_EXTERNAL_TEST_CONNECTION.n_clicks` | Probes Automatic1111 (`make_models_request`) or ComfyUI (`get_history`) and highlights success/failure | `configuration_services.probe_server` | `project-export.spec.ts` ("external server connection test highlights success..."); `test_configuration_services.py`, `POST /api/v1/config/probe-server`, `test_api_project_export.py` | [x] Svelte: `ConfigurationTab.svelte`'s "Test Connection" button (`external-test-connection`) + `workflow.probeExternalServer` (`POST /api/v1/config/probe-server`), highlighting `external-server-address` via `data-status`; asserted on both UIs by `project-export.spec.ts`'s "external server connection test highlights success..." and the failure case by `ConfigurationTab.test.ts`'s "Test Connection highlights failure when the probe fails" |
| CMP-13 | `toggle_stabilityai_config` components.py:1084 | Input `DROPDOWN_INPAINT_MODEL.value` | Shows/hides the API-key panel for StabilityAI and `falai-*` models | inline | none | [x] Svelte: `ConfigurationTab.svelte`'s `showApiKeyPanel` derived state (`api-key-panel`); asserted by `ConfigurationTab.test.ts`'s "shows the ComfyUI workflow upload only for comfyui, and the API key panel only for stabilityai/fal.ai" |
| CMP-14 | `reset_external_api_key` components.py:1098 | Input `INPUT_API_KEY.value` | Persists `state.api_key`; clears success/failure highlight | inline (AppState) | none | [x] Svelte: `ConfigurationTab.svelte`'s API-key field (`api-key`) + `workflow.updateInpaintingSettings({apiKey})`, resetting `uiStore.apiKeyStatus` to `'none'` on every edit the same way CMP-11 does for the server field; not independently asserted by a shared scenario (same handoff-listed gap as Dash's own API-key highlight - see CMP-15), but the field's write-only "never echoed back" contract is pinned by `ConfigurationTab.test.ts` |
| CMP-15 | `test_api_key` components.py:1123 | Input `BTN_VALIDATE_API_KEY.n_clicks` | Validates a StabilityAI or fal.ai key against the live provider; highlights success/failure | `configuration_services.validate_api_key` | none in the browser suite (no scenario checks the API-key highlight - only the server-connection probe, per this slice's scope); `test_configuration_services.py`, `POST /api/v1/config/validate-key`, `test_api_project_export.py` | [ ] Svelte: `ConfigurationTab.svelte`'s "Test API Key" button (`validate-api-key`) + `workflow.probeApiKey` (`POST /api/v1/config/validate-key`), highlighting `api-key` via `data-status`; not checked here for the same reason as Dash (no shared scenario covers it) |

## Export/Render

| ID | Function (file:line) | Trigger(s) | Effect | Backend service | E2E coverage | Svelte |
| --- | --- | --- | --- | --- | --- | --- |
| WEB-27 | `upscale_texture` webui.py:1203 | Input `BTN_UPSCALE_TEXTURES.n_clicks`; `running=` disables button | Builds an inpainting pipeline via `create_inpainting_pipeline`, calls `state.upscale_slices()` | `ExportService.upscale_textures` (`export_services.py`) | `project-export.spec.ts` ("upscale textures then glTF export embeds the upscaled (2x) images") closes the handoff-listed gap; `test_export_services.py`, `POST .../export/upscale`, `test_api_project_export.py` | [x] Svelte: `ExportTab.svelte`'s "Upscale Textures" button (`upscale-textures`) + `workflow.startUpscaleExport` (`POST .../export/upscale`, a job, polled to completion); the log line (`api/export.py`) was changed to drop its slice count so it matches Dash's exact wording (`"Upscaled textures for slices"`) - see PARITY.md's own top-level note on how a Svelte row's checkbox requires a *shared* scenario, not just Svelte code, and `test_export_services.py`'s `result.slice_count` still pins the count itself. Asserted on both UIs by `project-export.spec.ts`'s "upscale textures then glTF export..." |
| WEB-28 | `gltf_export` webui.py:1229 | Input `BTN_GLTF_EXPORT.n_clicks` (`#gltf-export`); `running=` disables button | Calls module-level `export_state_as_gltf()` (webui.py:1346, not itself a callback — generates per-slice depth maps if displacement>0, prefers upscaled slice files, calls `segmentation.export_gltf`), sends the `.gltf` file for download | `ExportService.export_gltf` (`export_services.py`) - the *one* implementation both WEB-28 and WEB-32's duplicated call sites now map to, see ARCHITECTURE.md | 11; displacement>0 (subdivided mesh) and DOF (MASK alpha mode) also pinned by `project-export.spec.ts`; `test_export_services.py`, `POST`/`GET .../export/gltf`, `test_api_project_export.py` | [x] Svelte: `ExportTab.svelte`'s "Export glTF Scene" button (`gltf-export`) + `workflow.startGltfExport` (`POST .../export/gltf`, a job, polled), then a real `<a download>` click against `ProjectView.exports.gltf.url` (`GET .../export/gltf`) - not `window.open`, per this task's own instruction; asserted on both UIs by scenario 11 and every glTF-export `project-export.spec.ts` scenario, and by `ExportTab.test.ts`'s "Export glTF Scene starts the export job and then triggers a real \<a download\> click, not window.open" |
| WEB-32 | `gltf_create` webui.py:1328 | Input `BTN_GLTF_CREATE.n_clicks`; `running=` disables button | Same `export_state_as_gltf()` helper as WEB-28 (duplicated call site, see Known quirks) but renders the in-page `<iframe>` model viewer instead of downloading | `ExportService.export_gltf` (`export_services.py`) - same service as WEB-28; the API has one `POST .../export/gltf` job + `GET .../export/gltf` download, no separate "create for viewer" route (`ProjectView.exports.gltf` already exposes the download URL) | none (no scenario clicks the in-page viewer button specifically - see WEB-28 for the shared underlying export coverage) | [x] Svelte: `ExportTab.svelte`'s "Create glTF Scene" button (`gltf-create`) + the same `workflow.startGltfExport` job as WEB-28 (no download); the exported scene then renders in `Model3DViewer.svelte` (the left "3D" viewer tab) once opened, reading `ProjectView.exports.gltf` directly rather than a second dedicated "create" route. Not independently asserted by a shared scenario (same as Dash), but `Model3DViewer.test.ts` covers rendering `<model-viewer>` once `exports.gltf` is set |
| WEB-33 | `download_image` webui.py:1403 | Input `{slice-info,ALL}.n_clicks` | Sends the raw slice PNG file for download | `ExportService.slice_download_path` (`export_services.py`) | `project-export.spec.ts` ("clicking a slice download icon downloads the exact raw slice PNG") closes the handoff-listed gap; `test_export_services.py`, `GET .../slices/{index}/download`, `test_api_project_export.py` | [x] Svelte: `SegmentationTab.svelte`'s slice-label button (`slice-download`) + a real `<a download>` click against `GET .../slices/{index}/download` (`download.ts`'s `triggerDownload`, same technique as WEB-28); asserted on both UIs by `project-export.spec.ts`'s "clicking a slice download icon downloads the exact raw slice PNG" |
| WEB-35 | `export_animation` webui.py:1481 | Input `BTN_EXPORT_ANIMATION.n_clicks` (`#animation-export`); `running=` disables button | Calls `segmentation.render_image_sequence`, writing `rendered_image_%03d.png` **server-side only** — no `dcc.Download` fires despite `ANIMATION_OUTPUT` existing | `ExportService.render_animation` (`export_services.py`) | 11; `test_export_services.py`, `POST .../export/animation` (still no download field in the response, matching Dash exactly), `test_api_project_export.py` | [x] Svelte: `ExportTab.svelte`'s "Export Animation" button (`animation-export`) + Number of Frames slider (`number-of-frames`, a purely local draft - not a persisted setting, matching Dash) + `workflow.startAnimationExport` (`POST .../export/animation`, a job); deliberately never triggers a download, matching Dash exactly. Asserted on both UIs by scenario 11 (`"Exported 4 frames to animation"`, no download) and `ExportTab.test.ts`'s "Export Animation never triggers a download..." |

## Navigation/Layout

| ID | Function (file:line) | Trigger(s) | Effect | Backend service | E2E coverage | Svelte |
| --- | --- | --- | --- | --- | --- | --- |
| WEB-01 | `toggle_dark_mode` webui.py:275 | Input `BTN_DARK_MODE.n_clicks` | Flips dark-mode class on `#app-container`/icon; persists `state.dark_mode` (JSON-only save) if a project is loaded | inline (AppState) | none (button itself unclicked; class asserted only after restore, via WEB-38) | [x] Svelte: `Header.svelte`'s theme toggle (`theme-toggle`) + `workflow.toggleDarkMode` - flips `uiStore.theme`/the root `dark` class immediately, and persists via `PUT .../settings` only once a project is loaded, exactly like Dash's own `filename is not None` guard. No shared scenario clicks it either, but the click itself and its persist-only-with-a-project behavior are pinned by `App.test.ts`'s "the theme toggle switches the root dark class..." and "toggling the theme persists it via PUT settings once a project is loaded" |
| WEB-48 | `update_current_tab` webui.py:1727 | Input `{tab-content-main,ALL}.className` | Writes the active main-tab name into `STORE_CURRENT_TAB` (feeds CLI-03, used by JS-06's help text) | inline | 1–11 | [ ] |
| CMP-01 | `update_events` components.py:286 | Input `{tab-content-main,ALL}.className` | Sets canvas vs. image z-index and shows/hides the segmentation/inpainting tool bars based on active main tab | inline | 1–11 | [ ] |
| CMP-16 | `toggle_depth_map` components.py:1527 (**dead — never registered**) | Would be Input `{label}-label.n_clicks` | Would show/hide a labeled container; `make_label_container_callback` is defined but `webui.py` never calls it, and its only real consumer (`make_configuration_container`) is used only in `test_components.py`, not the live layout | inline | none (unreachable) | [ ] |
| CMP-17 | `toggle_tab_container` ("viewer" instance) components.py:1588 | Input `{tab-label-viewer,ALL}.n_clicks` (2D/3D tabs) | Switches the 2D/3D viewer tab, underlines the active label | inline | none (no scenario clicks the 2D/3D viewer tabs) | [x] Svelte: `ViewerTabs.svelte`'s 2D/3D `role="tab"` strip + `uiStore.setViewerTab`; no shared scenario clicks it (same gap as Dash), but the switch and each panel's visibility are pinned by `ViewerTabs.test.ts` |
| CMP-18 | `toggle_tab_container` ("main" instance) components.py:1588 | Input `{tab-label-main,ALL}.n_clicks` (Mode/Segmentation/Inpainting/Export/Configuration) | Same logic as CMP-17, bound to the main tab strip; this is what `clickMainTab()` drives | inline | 1–11 | [ ] |
| CMP-26 | `navigate_image` components.py:1869 | Input `NAV_RESET`/`NAV_UP`/`NAV_DOWN`/`NAV_LEFT`/`NAV_RIGHT`/`NAV_ZOOM_IN`/`NAV_ZOOM_OUT` `.n_clicks` | Moves `state.camera.camera_position`, re-renders the composed 3D-ish preview via `segmentation.render_view`, deselects any selected slice | inline (AppState + `segmentation.render_view`) | none (handoff-listed gap: pan/zoom via UI buttons) | [ ] **Not to be confused with the UX-parity slice's new zoom/pan controls** (`InputImagePanel.svelte`'s `−`/`⟳`/`+` buttons, `state/viewport.svelte.ts`) - despite living in the same Dash toolbar container (`make_inpainting_tools_container`) as this row's `NAV_*` buttons and both being colloquially "zoom", they are unrelated features: this row is a server-side 3D-camera dolly over the composited slice cards (deselects the current slice and re-renders `IMAGE.src` from scratch on every click), while the new controls are a client-only CSS `transform` over the existing image/canvas (JS-03's row). This row remains untouched/out of scope |
| CLI-01 | `store_rect_coords` clientside.py:9 | Input `IMAGE.src`, `evScroll.n_events` | Resolves `#image`'s bounding rect + rendered size (waiting for the `load` event if needed) into `STORE_RECT_DATA`, used by WEB-08 to map click coordinates to pixels | inline (JS) | 2, 3, 4, 5 (via `clickImagePixel`); zoom/pan correctness specifically: `e2e/ux-parity.spec.ts`'s "after zooming in and panning, a depth-mode click..." | [x] Svelte: no caching step exists or is needed - `InputImagePanel.svelte`'s `onImageClick` reads `event.currentTarget.getBoundingClientRect()` *live*, at click time, which already reflects any current CSS `transform` (zoom/pan or otherwise) on an ancestor; `lib/geometry.ts`'s `findPixelFromClick` is unmodified for zoom/pan (its plain ratio math is transform-agnostic by construction - see its own doc comment and the new `transformedRect` helper/tests in `geometry.test.ts`, which prove a click at a known source pixel round-trips exactly under an arbitrary zoom+pan state). Exact-pixel-under-zoom-and-pan is asserted on both UIs by `ux-parity.spec.ts` |
| CLI-02 | `suppress_contextmenu` clientside.py:16 | Input `CTR_INPUT_IMAGE.id` (fires once on load) | Calls `setupHelper()` (wires JS-06) and adds a `contextmenu` listener that turns a Ctrl+right-click into a synthetic left-click dispatch | inline (JS) | none | [ ] |
| CLI-03 | `store_current_tab` clientside.py:24 | Input `STORE_CURRENT_TAB.data` (self-referential Output==Input; always returns `no_update`) | Records JS `currentTab` for JS-06's help text; resets cached canvas context | inline (JS) | 1–11 | [ ] |
| JS-03 | canvas `wheel` → `handleWheel` utility.js:292,203 | Native `wheel` on `#canvas` | Zooms image/canvas/preview via CSS `transform: scale()` + `transform-origin`, clamped to 0.125×–8× | inline (JS) | none (handoff-listed gap: zoom/pan) → closed: `e2e/ux-parity.spec.ts`'s zoom/pan scenarios | [x] Svelte: `state/viewport.svelte.ts`'s `zoomAt` (same `ZOOM_FACTOR = 1.1` and `[0.125, 8]` clamp as Dash, same "zoom towards the cursor" math), wired to `InputImagePanel.svelte`'s `onWheel` on the same sign check (`deltaY < 0` zooms in) and applied as `transform: translate(panX, panY) scale(scale)` on `.image-stack` (image + mask canvas + `PreviewOverlay.svelte` together, so every overlay tracks it identically). **Empirically discovered while writing this scenario** (see "Known quirks"): Dash's own wheel-zoom only works while the Inpainting tab is active and only after the canvas has been hovered/pressed at least once - a real, confirmed gap in Dash itself, not reproduced here (Svelte's zoom works on every tab, always). Svelte also adds drag-to-pan (`onDropZonePointerDown`/`Move`/`Up`, middle-button always, primary-button outside the Inpainting tab past a 4px threshold so a real click is never affected) and Zoom In/Out/Reset buttons - all deliberate, documented improvements over Dash, which has neither (see viewport.svelte.ts's own doc comment). Unit-tested in `state/viewport.svelte.test.ts` and `lib/geometry.test.ts`; asserted end-to-end on both UIs by `e2e/ux-parity.spec.ts` (zoom on both; pan and reset `test.fail()`-pinned on Dash only, per the confirmed gaps above) |
| JS-04 | `window resize` → `resetContext` utility.js:294,39 | Native `resize` on `window` | Drops cached canvas 2D contexts/rect so they're rebuilt at the new size | inline (JS) | none | [ ] Svelte has no equivalent cache to invalidate (see CLI-01's row - rects are always read live), so there is nothing for this row to port; not ticked since it describes Dash-internal cache-invalidation plumbing, not user-visible behavior |
| JS-05 | `ResizeObserver(#canvas)` utility.js:34-37,296 | Native canvas resize | Keeps the cached `gRect` bounding-rect in sync with layout changes | inline (JS) | none (passive infra; not independently asserted) | [ ] Same as JS-04 - Svelte's live `getBoundingClientRect()` reads make a cached/observed rect unnecessary |
| JS-06 | help-window tooltip system utility.js:543-649 | Native `mousemove`/`mousedown`/`keypress`/`mouseout`/`mouseenter` on the input-image container (wired by CLI-02) | Shows a randomized, tab-contextual help tip after a 3s idle delay | inline (JS) | none | [ ] |

## Progress/Logs

| ID | Function (file:line) | Trigger(s) | Effect | Backend service | E2E coverage | Svelte |
| --- | --- | --- | --- | --- | --- | --- |
| WEB-02 | `update_logs` webui.py:301 | Input `LOGS_DATA.data` | Renders the last 3 log entries into `#log` | inline | 3, 4, 5, 6, 9 (scenarios that assert `#log` text) | [ ] |
| WEB-03 | `update_progress` webui.py:315 | Input `PROGRESS_INTERVAL.n_intervals` (polled every 500ms) | Renders `current_progress`/`total_progress` (module globals updated by `progress_callback`) as a progress-bar width; disables the interval once done | inline (module-global state) | none (handoff notes this polling defeats reliable `networkidle` waits; no test asserts bar width) | [ ] |

---

## Known quirks to preserve or fix deliberately

- **`AppState.balance_slices_depths` was broken; now fixed (intentional
  behavior change).** `parallax_maker/controller.py:249` used to read
  `for i in len(self.image_slices):` — `len(...)` returns an `int`, which is
  not iterable — and divided by `len(...) - 1` (`ZeroDivisionError` for one
  slice). Fixed to: zero slices is a no-op; one slice is set to depth 0; two
  or more slices are spread evenly with `int(i * 255 / (count - 1))` in
  existing list order. Covered directly by
  `test_controller.py::TestBalanceSlicesDepths` (0/1/2/5-slice cases) and by
  `SliceEditingService.balance_slices` (`test_slice_editing_services.py`,
  `test_api_slice_editing.py::test_balance_evenly_redistributes_depths`).
- **The "Balance" button is *still* completely unreachable through the live
  Dash UI — a separate, pre-existing bug, discovered while adding e2e
  coverage for the fix above.** `webui.py:883`'s `balance_slices_request`
  reads `state.image_depths`, an attribute `AppState` has never defined
  (only `image_slices`); every click raises
  `AttributeError: 'AppState' object has no attribute 'image_depths'` and
  500s *before* `balance_slices_depths()` is ever called — independent of
  and not fixed by the controller.py change above, since `webui.py` is
  frozen and out of scope for this extraction. `e2e/slice-editing.spec.ts`'s
  SE-15 pins this exact current (broken) behavior with `test.fail()` rather
  than asserting the unreachable "fixed" one; `SliceEditingService.
  balance_slices` and its `POST /api/v1/projects/{id}/slices/balance` route
  implement the correct, intended behavior for the API/Svelte side instead.
- **Two functions are both named `remember_camera_parameters`.** WEB-30
  (webui.py:1272, `Input` = the four camera/displacement sliders, *persists*
  `state.camera`/`state.mesh_displacement`) and WEB-37 (webui.py:1527, `Input`
  = `STORE_RESTORE_STATE.data`, *restores* the same sliders from `state`) do
  opposite things but shadow each other in the `webui` module namespace after
  import. Both callbacks are still registered correctly because Dash captures
  the function object at the `@app.callback` decoration call site, not by
  later name lookup — but any future code that imports
  `webui.remember_camera_parameters` will only ever get the second (restore)
  definition. Rename one for the Svelte-era service/API layer.
- **WEB-30's `remember_camera_parameters` (persist variant, webui.py:1272-1290)
  can 500 a real, reachable Dash request, not just misfire its own
  unchanged-value check.** It builds `Camera(camera_distance, focal_length,
  max_distance)` positionally against a constructor whose actual signature is
  `Camera.__init__(self, distance, max_distance, focal_length, ...)` - the
  live *max-distance* slider value lands in the constructor's `focal_length`
  parameter, whose setter rejects non-positive numbers. Dragging (or
  keyboard-`Home`-ing) the max-distance slider `(0..1000)` down to exactly
  `0` therefore raises `ValueError: focal_length must be a positive number`
  inside the callback and Dash returns `500`. Camera distance and focal
  length transiting `0` are unaffected (their bound parameters only require
  `>= 0`). Discovered while writing `e2e/project-export.spec.ts`'s camera/
  displacement persistence scenario; `e2e/drivers/dash.ts`'s `setSlider` no
  longer resets to `Home` before stepping (it steps directly from the
  slider's current value instead) specifically to avoid retriggering this for
  every future scenario that sets more than one camera slider in the same
  test. Not fixed (Dash is frozen) and not reproduced by
  `project_services.ProjectService.update_settings`, which assigns each
  field directly with no positional `Camera(...)` construction at all.
- **`make_label_container_callback` (components.py:1519) is dead code.** It is
  defined but never invoked from `webui.py`; its only consumer,
  `make_configuration_container` (components.py:1172), is used only from
  `test_components.py`, not the live layout (`webui.py` uses
  `make_configuration_div()` directly). There is no reachable
  `configuration-label`/`configuration-container` pair in the running app for
  this callback to attach to. Do not port a "collapsible configuration label"
  feature that doesn't actually exist in production today.
- **`make_tabs_callback` is registered twice** (webui.py:263-264, for `"viewer"`
  and `"main"`) from one function definition (`toggle_tab_container`,
  components.py:1588) — two independent Dash callbacks with the same Python
  name and pattern-matched `ALL` ids scoped by `tab_id`. Only the `"main"` tab
  strip (Mode/Segmentation/Inpainting/Export/Configuration) is exercised by any
  e2e scenario; the `"viewer"` 2D/3D tab strip (CMP-17) has zero test coverage.
- **Self-referential clientside callbacks.** `store_current_tab` (CLI-03,
  clientside.py:24-28) and WEB-04 `update_threshold_values` (webui.py:323-326,
  `{threshold-slider,ALL}.value` is *both* the `Input` and `Output`) each list
  the same store/property as both their own trigger and their own target.
  Both are safe today only because they either always return `no_update`
  (CLI-03) or raise `PreventUpdate`/return unchanged values on no-op
  (WEB-04 via `WorkflowUnchanged`) — a naive re-implementation that ever
  returns a genuinely new value here would self-trigger indefinitely.
- **Canvas auto-clear on every main-image change is explicitly flagged as
  wrong in the source.** CLI-11 (components.py:1838-1843) carries the comment
  `# XXX - this will kill the canvas during inpainting - bad`; it clears the
  paint canvas on *every* `IMAGE.src` update, which includes updates that
  happen mid-inpainting-workflow, not just slice/tab changes. The handoff's
  "give the new canvas an explicit lifecycle" guidance is about this exact
  callback.
- **Canvas save fires on `mouseout`, not `mouseup`.** CLI-09 `canvas_draw`
  (components.py:1825, `switch (event.type)` in utility.js:452-475) only
  calls `canvas_get()`/saves on `mouseout` (`bShouldSave = canvasLastDrawnTime
  > canvasLastSavedTime`), not on `mouseup`. A pointer that lifts and re-enters
  without leaving the canvas element never triggers a save until it finally
  exits. The handoff explicitly calls out testing pending-save ordering before
  Generate/slice changes if this is changed for Svelte.
- **`BTN_ERASE_MODE` ("Erase" canvas-brush toggle, CLI-12) is a completely
  different control from `BTN_ERASE_INPAINTING` (`#erase-inpainting-button`,
  CMP-03, "Erase" in the Inpainting action row).** The names are easy to
  conflate; only the latter (erasing already-applied inpainting content via
  `InpaintingService.erase`) has e2e coverage (scenario 9). The paint-canvas
  eraser toggle has none.
- **`WEB-14`/`WEB-15`/`WEB-16`/CMP-19/CMP-20 mutate `state.slice_mask`/
  `state.image_slices[i].image` in place** (e.g.
  `blend_with_alpha(state.image_slices[...].image, image)` at
  webui.py:706/787, `state.image_slices[...].image[:, :, 3] = final_mask` at
  webui.py:745) rather than going through the same copy-on-write
  version/service pattern `InpaintingService` uses for paint/fill/enhance.
  **Now extracted** into `SliceEditingService` (`parallax_maker/
  slice_editing_services.py`), which reproduces this same in-place mutation
  style on purpose (it mirrors `ImageSlice.new_version()`'s own semantics),
  not a copy-on-write rewrite; Dash's own callbacks are unchanged and still
  perform the mutation inline themselves.
- **A slice's own mask does not survive the mutation that used it — a
  surprising, real chained side effect discovered while characterizing
  WEB-16/WEB-18 on Dash.** WEB-16 `add_mask_slice_request` (and WEB-18
  `create_single_slice_request`, WEB-14 `paste_clipboard_request`, WEB-19
  `balance_slices_request`, WEB-23 `record_depth_input`) only ever sets
  `STORE_UPDATE_SLICE.data = True`; they never touch `IMAGE.src` or
  `slice_mask` themselves. That store write chains into WEB-21
  `update_slices` (webui.py:913), which — *whenever a slice remains
  selected* — recomposes the preview **and unconditionally clears
  `state.slice_pixel`/`slice_pixel_depth`/`slice_mask`**
  (webui.py:1024-1026) as a side effect of that re-render. Concretely: click
  Add, and the mask you just used is gone by the time the button's own
  network round trip finishes — Remove immediately afterward is a silent
  no-op ("No mask selected") until a fresh mask is generated.
  `e2e/slice-editing.spec.ts`'s SE-5 pins this exactly (asserts
  `slice_mask.present === false` right after Add, then re-clicks before
  Remove). `SliceEditingService` reproduces the *effective* chain as a
  single command via the shared `refresh_selection_preview()` helper (also
  used by the API's new undo/redo routes, which trigger the identical chain)
  so a Svelte/API caller sees one atomic result instead of Dash's two-step
  callback dance.
- **A selected slice's own depth badge cannot be clicked through normal
  hit-testing — a real Dash UI limitation, also discovered while
  characterizing WEB-22/WEB-23.** The depth-number display
  (`{"type":"depth-display","index":i}`) and its `.overlay` highlight
  sibling (`{"type":"slicer-overlay","index":i}`) are both
  absolutely-positioned children of the same `position: relative` thumbnail
  container (components.py, `update_slices`); `.overlay` (`tailwind.css:179`,
  `absolute inset-0`) has no explicit `z-index` but comes later in DOM
  order, so once a slice is selected its overlay paints on top and swallows
  clicks meant for the depth badge underneath. `e2e/slice-editing.spec.ts`'s
  SE-9 works around this by editing a *different*, unselected slice's depth
  instead (which is also how it discovered the next quirk); a Svelte
  redesign should give the depth editor its own non-overlapping hit target.
- **`record_depth_input` clears `state.selected_inpainting` unconditionally
  on every call, not only when the edited slice was selected.** WEB-23 also
  sets `STORE_INPAINTING.data = True`, which chains into CMP-07
  `react_selected_slice_change` → `InpaintingService.clear_selection()`
  regardless of which slice's depth changed or whether it reordered.
  Changing an unrelated slice's depth silently drops the current inpainting
  candidate selection. Pinned by
  `test_slice_editing_services.py::test_set_slice_depth_always_clears_inpainting_selection`.
- **`export_state_as_gltf` (webui.py:1346) is not a callback** — it's a plain
  helper called from both WEB-28 `gltf_export` and WEB-32 `gltf_create`
  (webui.py:1235, 1334), each independently regenerating per-slice depth maps
  and re-reading upscaled-vs-original slice files from disk. The existing
  `# XXX - this and the callback above can be chained to avoid code
  duplication` comment (webui.py:1314) still applies; an export job/service
  should own this once, not duplicate it per entry point.
- **`allow_duplicate=True` is pervasive** on `IMAGE.src`, `LOGS_DATA.data`,
  `STORE_UPDATE_SLICE.data`, and several others, across more than a dozen
  callbacks in both files. Dash gives no ordering guarantee between
  same-cycle duplicate-output callbacks; combined with the process-global
  `AppState.cache` and progress globals the handoff already flags, this is a
  real concurrency hazard for anything beyond today's single-user,
  single-process assumption — not merely a style note.
- **`remember_depth_model` (webui.py:1247-1256) has a duplicated `if`**
  (`if state.depth_model_name == value: raise PreventUpdate()` appears twice
  in a row) — harmless but worth cleaning up rather than porting verbatim.
- **`CLI-07` (`show_bounding_box`) is effectively untested end-to-end** because
  `SaveInpaintingMask`'s `show_crop_region` is only `True` when the
  `CHECKLIST_REGION_OF_INTEREST` ("crop") box is checked, and no e2e scenario
  checks it — matching the handoff's note that "the checkbox controls the
  bounding-box preview" as a distinct, currently-unexercised code path from
  mask generation itself (which always passes `crop=True` regardless).
- **`slice_upload`'s "fixing aspect ratio" log message is not an f-string**
  (webui.py:1451-1453): `logs.append("Fixing aspect ratio from
  {image.size[0] / image.size[1]} to {aspect_ratio}")` is missing the `f`
  prefix, so Dash literally logs that placeholder text verbatim, never the
  actual numbers. `e2e/slice-editing.spec.ts`'s SE-11 pins the exact literal
  string Dash produces today. `SliceEditingService.replace_slice_image`
  returns `source_aspect_ratio`/`target_aspect_ratio` on its result instead,
  so an adapter that wants a correctly-interpolated message can build one
  without reproducing the bug (which only affects a log string).
- **`slice_upload`'s mismatched-aspect-ratio resize is a real, more
  consequential bug**, also pinned as-is per this task's instructions:
  `image = image.resize((int(aspect_ratio * image.size[1]), image.size[1]))`
  (webui.py:1454) resizes to the *uploaded* image's own height, not the
  existing slice canvas's — it does not fit/crop the upload into the slice's
  dimensions at all. A small, wrong-aspect-ratio upload can collapse a slice
  to a tiny fraction of the canvas (e.g. a 2x1 upload against a 320x240
  slice collapses to 1x1; `e2e/slice-editing.spec.ts`'s SE-11 and
  `test_slice_editing_services.py::test_replace_slice_image_mismatched_aspect_collapses_dimensions`
  pin this). Worse, `slice_upload` then unconditionally recomposes
  `state.imgData` by `blend_with_alpha`-ing every slice together starting
  from `image_slices[0]` (webui.py:1461-1464): if the *first* slice (or any
  slice not shape-broadcastable against the collapsed one, e.g. not reduced
  to a literal 1x1) was the one resized, that recompose loop raises a raw
  NumPy `ValueError` and the request 500s.
  `test_slice_editing_services.py::test_replace_slice_image_mismatched_aspect_can_crash_the_recompose`
  reproduces this exact crash; SE-11's own upload happens to collapse to a
  broadcastable 1x1, which is why it doesn't 500 in the browser suite. This
  was not fixed (per this task's "reproduce Dash semantics exactly, bugs and
  all" instruction for extraction), only characterized; it is a strong
  candidate for a deliberate behavior fix (fit/crop into the canvas instead)
  before this endpoint is used from a real Svelte upload control.
- **Dash's wheel-zoom (JS-03) only works while the Inpainting tab is
  active, and only after the canvas has been hovered/clicked at least
  once - a real, confirmed Dash limitation, discovered empirically while
  writing `e2e/ux-parity.spec.ts`.** `setupMainCanvas` (which attaches
  `canvas.addEventListener('wheel', handleWheel)`) is only ever called
  lazily, from `canvas_draw`'s `mouseenter`/`mousedown` cases - never
  unconditionally at page load. Even once attached, a wheel event only
  reaches that listener while `#canvas` is the topmost element under the
  cursor, which `update_events`/CMP-01 only makes true while the Inpainting
  tab is active (every other tab puts `#image` on top instead, with no
  wheel listener of its own). Net effect, verified directly against the
  live Dash app: scrolling over the Input Image box on the Mode/
  Segmentation/Export/Configuration tabs does nothing at all, ever;
  scrolling on the Inpainting tab works, but only from the moment the
  pointer first enters/presses the canvas onward. `state/viewport.svelte.ts`
  does not reproduce this - Svelte's zoom works on every tab, immediately.
  Dash also has **no drag-to-pan mechanism at all** for the image/canvas
  (only wheel-zoom; the similarly-named `NAV_*` buttons are an unrelated
  3D-camera-dolly feature - see CMP-26's own row) and **no reset control**
  for the zoom it does have. `e2e/ux-parity.spec.ts`'s pan and reset
  scenarios `test.fail(ui.target === 'dash', ...)`-pin these two gaps
  directly against a real drag/repeated-zoom-out gesture (not a synthetic
  shortcut), and its zoom scenario visits the Inpainting tab first
  specifically so Dash's own zoom has a fair chance to actually engage.

## Remaining gaps

Updated by the **UX-parity slice** (zoom/pan, queued-point markers, ROI-box
preview, brush preview + Alt+drag resize, help tooltips, slider value
labels, CMP-21, keyboard/roving-tabindex, responsive layout), which closed
CMP-21, CLI-05, CLI-06, CLI-01, JS-03, JS-01 (the preview-circle half) and
JS-02 (the resize-gesture half) from the previous version of this section,
and added the CMP-24/CLI-07 backend field described below. Everything below
is what is still genuinely missing after that slice, grouped by reason.

**Implemented in this slice but not tickable per this document's own rule**
(Svelte code + Dash-matching behavior exist, but no *shared* e2e scenario
exercises the specific row - each row's own text says so):

- **CLI-07** (`show_bounding_box`, the ROI-box preview) - the mask-save
  request now carries a real `cropToRegion` flag and the response a real
  `boundingBox` (see CMP-24's row and `api/inpainting.py`), and
  `PreviewOverlay.svelte` renders it exactly like Dash's `previewRect`
  (same color, same 2s auto-clear). Covered by `test_api_inpainting.py`
  and manually verified via the `ux-inpainting-*.png` screenshots, but no
  *e2e* scenario enables the checkbox and asserts the rendered box on
  either UI - an inherited gap (Dash's own row says the same: "no test
  enables the ROI checkbox"), not a new one.
- **JS-01**'s brush-preview circle and **JS-02**'s Alt+drag resize
  gesture - both implemented in `MaskCanvas.svelte`/`PreviewOverlay.svelte`
  (see their own rows), not independently checked by a shared scenario
  (canvas-content pixel-sampling would be the only way to assert a preview
  circle's presence, which felt like more test fragility than the payoff
  justified); visually verified in the `ux-inpainting-*.png` screenshots.

**Deliberate redesigns, not straight ports** (own reasoning in each area):

- **JS-06** (the idle-timer help-tooltip popup) - replaced with
  `components/shared/HelpTooltip.svelte`, an on-demand, accessible "?"
  button per relevant tab (Segmentation/Inpainting/Export/Configuration)
  instead of a floating, randomized, 3-second-idle popup with no ARIA
  semantics or keyboard path at all. Same texts (`lib/helpTexts.ts`, copied
  verbatim from `utility.js`'s `helpTexts`), different (and, unlike Dash's,
  keyboard/screen-reader-reachable) interaction model - see
  `HelpTooltip.svelte`'s own doc comment. Unit-tested in `HelpTooltip.test.ts`;
  not "ticked" against JS-06 since it is not the same behavior, by design.
- **CMP-18**/tab-strip keyboard support - Dash's tab strips
  (`MainTabs.svelte`/`ViewerTabs.svelte`'s equivalents) have no keyboard
  affordance at all (plain `<label>` click targets). Both Svelte tab strips
  now use roving tabindex (`lib/a11y/rovingTabindex.ts`: only the active tab
  is in the Tab order, Left/Right/Home/End move and activate) - a deliberate
  accessibility improvement with no Dash equivalent to match, so CMP-18
  itself stays unticked (its own row is about click-based tab switching,
  which is unchanged) rather than being (mis)credited with a Dash behavior
  that doesn't exist. Unit-tested in `rovingTabindex.test.ts`.
- **CMP-26**'s `NAV_*` buttons remain completely untouched (see that row's
  own note distinguishing them from the new zoom/pan controls) - out of
  scope; a different feature entirely.

**Pre-existing gaps from earlier PRs' slices, untouched by this one** (not
this task's scope: zoom/pan, markers, tooltips, sliders, keyboard nav,
responsiveness):

- Upload/Depth/Slices: WEB-04, WEB-05, WEB-06, WEB-07, WEB-09, WEB-10,
  WEB-20, WEB-25 - inline `AppState`/trigger-only rows with no Svelte column
  entry at all yet; the actual upload/depth/slice-generation *behavior* they
  describe is implemented and covered (scenario 1 and others), but these
  specific Dash-internal plumbing rows were never individually annotated.
- Mask tools: WEB-11 - depth-map `<img>` rendering; implemented
  (`ModeTab.svelte`'s `depth-image`) but not individually annotated.
- Canvas/Inpainting: CMP-25/CLI-08/CLI-10/CLI-12 - canvas Load/eraser-toggle
  rows, each already documented inline as implemented-but-unchecked (no
  shared scenario) or a deliberately-dropped Dash bug (CLI-11, already `[ ]`
  with its own "Deliberately not reproduced" note).
- Navigation/Layout: WEB-48, CMP-01, CMP-16 (dead code in Dash itself -
  unreachable, nothing to port), CLI-02/03 (Ctrl+right-click-as-click
  bookkeeping / current-tab bookkeeping - CLI-02's actual behavior is
  implemented, see `InputImagePanel.svelte`'s `onImageContextMenu`, but its
  row was never individually annotated - inherited, not new), JS-04/JS-05
  (Dash-internal rect-caching plumbing that Svelte's always-live
  `getBoundingClientRect()` reads make structurally unnecessary - see their
  own rows).
- Progress/Logs: WEB-02, WEB-03 - the log pane and progress bar are already
  implemented generically (`LogPanel.svelte`, and every tab's own
  `*-progress` bar in `ModeTab`/`InpaintingTab`/`ExportTab.svelte`) and
  exercised indirectly by nearly every scenario, but were never individually
  annotated by the PR that added them.
- Configuration: CMP-08, CMP-09, CMP-15, WEB-39 (deliberately not
  reproduced - see its own row), WEB-40 - all implemented, just not ticked
  because no shared e2e scenario exercises them (identically true of Dash
  for the same rows).
