import { test, expect, disableAnimations, requireWorkflow } from './fixtures';
import { imageHash, sourceImagePixel, imageBufferMetadata } from './helpers/image';
import { fetchArtifact, fetchFixture, rawArtifactDataUrl, readE2EState } from './helpers/oracle';

// Fixture geometry (see parallax_maker/e2e_support/fixtures.py): a 320x240
// input image, thresholds [0, 85, 170, 255], and three generated slices whose
// depths are the upper edge of their threshold bucket: [85, 170, 255].
const IMAGE_WIDTH = 320;
const IMAGE_HEIGHT = 240;
const TOTAL_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

test.beforeEach(async ({ ui, page }) => {
  await ui.goto();
  await disableAnimations(page);
});

// --- Create slice ------------------------------------------------------------

test('create slice from a mask appends and selects a new slice matching the mask', async ({ page, ui }) => {
  requireWorkflow(ui, 'slice-editing');
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.setSegmentationMode('Instance Segmentation');
  await ui.openTab('Segmentation');

  // No slice selected: the mask is generated from the full input image.
  await ui.clickImagePixel(80, 96);
  await expect.poll(async () => (await readE2EState(page, projectId)).slice_mask.present).toBe(true);
  const mask = (await readE2EState(page, projectId)).slice_mask;
  expect(mask.inside).not.toBeNull();
  expect(mask.outside).not.toBeNull();
  const clickedDepth = (await readE2EState(page, projectId)).slice_pixel_depth;
  expect(clickedDepth).not.toBeNull();

  await ui.createSlice();
  await expect(ui.sliceImages()).toHaveCount(4);

  const after = await readE2EState(page, projectId);
  expect(after.slice_count).toBe(4);
  expect(after.selected_slice).not.toBeNull();
  const newIndex = after.selected_slice as number;
  // The new slice is inserted in depth order; add_slice() bumps by one on an
  // exact depth collision, so allow that single-step fallback.
  expect([clickedDepth, (clickedDepth as number) + 1]).toContain(after.slice_depths[newIndex]);
  // add_slice() names the new file from the pre-insert slice count (3),
  // independent of where it ends up in the (depth-sorted) list.
  expect(after.slice_filenames).toContain('image_slice_3.png');

  const rawSrc = await rawArtifactDataUrl(page, projectId, 'image_slice_3.png');
  const insideAlpha = (await sourceImagePixel(page, rawSrc, ...(mask.inside as [number, number])))[3];
  const outsideAlpha = (await sourceImagePixel(page, rawSrc, ...(mask.outside as [number, number])))[3];
  expect(insideAlpha).toBeGreaterThan(200);
  expect(outsideAlpha).toBe(0);
});

test('create slice with no mask appends an empty transparent slice at depth 127', async ({ page, ui }) => {
  requireWorkflow(ui, 'slice-editing');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');
  expect((await readE2EState(page, projectId)).slice_mask.present).toBe(false);

  await ui.createSlice();
  await expect(ui.sliceImages()).toHaveCount(4);

  const after = await readE2EState(page, projectId);
  expect(after.slice_count).toBe(4);
  // depth 127 falls strictly between the fixture's 85 and 170, deterministically at index 1.
  expect(after.selected_slice).toBe(1);
  expect(after.slice_depths).toEqual([85, 127, 170, 255]);
  expect(after.slice_filenames[1]).toBe('image_slice_3.png');

  const rawSrc = await rawArtifactDataUrl(page, projectId, 'image_slice_3.png');
  expect((await sourceImagePixel(page, rawSrc, 10, 10))[3]).toBe(0);
  expect((await sourceImagePixel(page, rawSrc, 160, 120))[3]).toBe(0);
});

// --- Delete slice --------------------------------------------------------------

test('delete removes the selected slice, clears selection, and restores the main image', async ({ page, ui }) => {
  requireWorkflow(ui, 'slice-editing');
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  const originalMainHash = await imageHash(ui.mainImage());

  await ui.openTab('Segmentation');
  await ui.selectSlice(projectId, 1);
  expect(await imageHash(ui.mainImage())).not.toBe(originalMainHash);

  await ui.deleteSlice();
  await expect(ui.sliceImages()).toHaveCount(2);

  const after = await readE2EState(page, projectId);
  expect(after.slice_count).toBe(2);
  expect(after.selected_slice).toBeNull();
  expect(after.slice_depths).toEqual([85, 255]);
  expect(after.slice_filenames).toEqual(['image_slice_0.png', 'image_slice_2.png']);
  await expect.poll(() => imageHash(ui.mainImage())).toBe(originalMainHash);
});

test('deleting with no slice selected is a no-op logged to the console', async ({ page, ui }) => {
  requireWorkflow(ui, 'slice-editing');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');

  await ui.deleteSlice();
  await expect(ui.log()).toContainText('No slice selected');
  expect((await readE2EState(page, projectId)).slice_count).toBe(3);
});

// --- Add mask / remove mask (in-place mutation, versioned) ---------------------

test('add mask and remove mask mutate the slice alpha in place with versioning and undo', async ({ page, ui }) => {
  requireWorkflow(ui, 'slice-editing');
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');
  await ui.selectSlice(projectId, 1);

  await ui.clickImagePixel(80, 96);
  await expect.poll(async () => (await readE2EState(page, projectId)).slice_mask.present).toBe(true);
  const mask = (await readE2EState(page, projectId)).slice_mask;
  const inside = mask.inside as [number, number];
  const outside = mask.outside as [number, number];

  const v1Src = await rawArtifactDataUrl(page, projectId, 'image_slice_1.png');
  const beforeOutsideAlpha = (await sourceImagePixel(page, v1Src, ...outside))[3];

  await ui.addMaskToSlice();
  await expect.poll(async () => (await readE2EState(page, projectId)).slice_filenames[1]).toBe(
    'image_slice_1_v2.png',
  );
  expect((await readE2EState(page, projectId)).slice_versions[1]).toBe(2);

  const v2Src = await rawArtifactDataUrl(page, projectId, 'image_slice_1_v2.png');
  // create_slice_from_mask() feathers by EXPAND_MASK=5px before blending, so
  // even the mask's own "most interior" sample lands just short of 255.
  expect((await sourceImagePixel(page, v2Src, ...inside))[3]).toBeGreaterThan(200);
  expect((await sourceImagePixel(page, v2Src, ...outside))[3]).toBe(beforeOutsideAlpha);

  // Surprising but real: Add's own STORE_UPDATE_SLICE=True output triggers
  // Dash's update_slices (webui.py:913), which -- because a slice is still
  // selected -- unconditionally clears state.slice_mask/slice_pixel(_depth)
  // as a side effect of re-rendering the preview (webui.py:1024-1026). The
  // mask used to create it does not survive the operation, so chaining
  // Remove straight after Add would silently no-op with today's Dash UI.
  expect((await readE2EState(page, projectId)).slice_mask.present).toBe(false);

  // Regenerate an equivalent mask (the click is deterministic) before Remove.
  await ui.clickImagePixel(80, 96);
  await expect.poll(async () => (await readE2EState(page, projectId)).slice_mask.present).toBe(true);

  await ui.removeMaskFromSlice();
  await expect.poll(async () => (await readE2EState(page, projectId)).slice_filenames[1]).toBe(
    'image_slice_1_v3.png',
  );
  expect((await readE2EState(page, projectId)).slice_versions[1]).toBe(3);

  const v3Src = await rawArtifactDataUrl(page, projectId, 'image_slice_1_v3.png');
  expect((await sourceImagePixel(page, v3Src, ...inside))[3]).toBe(0);
  expect((await sourceImagePixel(page, v3Src, ...outside))[3]).toBe(beforeOutsideAlpha);

  // Undo twice restores the pre-mutation (v1) file.
  await ui.undoButton(1).click();
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_filenames[1])
    .toBe('image_slice_1_v2.png');
  await ui.undoButton(1).click();
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_filenames[1])
    .toBe('image_slice_1.png');
  expect((await readE2EState(page, projectId)).slice_versions[1]).toBe(1);
});

test('add mask and remove mask with no selection or no mask are logged no-ops', async ({ page, ui }) => {
  requireWorkflow(ui, 'slice-editing');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');

  await ui.addMaskToSlice();
  await expect(ui.log()).toContainText('No mask selected');
  expect((await readE2EState(page, projectId)).slice_versions).toEqual([1, 1, 1]);

  await ui.removeMaskFromSlice();
  await expect(ui.log()).toContainText('No mask selected');
  expect((await readE2EState(page, projectId)).slice_versions).toEqual([1, 1, 1]);
});

// --- Copy / paste clipboard ------------------------------------------------------

test('copy requires a mask and paste blends the clipboard into the selected slice', async ({ page, ui }) => {
  requireWorkflow(ui, 'slice-editing');
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');

  // No mask yet: copy is a logged no-op and does not populate the clipboard.
  await ui.copySlice();
  await expect(ui.log()).toContainText('No mask selected');
  expect((await readE2EState(page, projectId)).clipboard_present).toBe(false);

  await ui.selectSlice(projectId, 1);
  await ui.clickImagePixel(80, 96);
  await expect.poll(async () => (await readE2EState(page, projectId)).slice_mask.present).toBe(true);
  const mask = (await readE2EState(page, projectId)).slice_mask;
  const inside = mask.inside as [number, number];
  const outside = mask.outside as [number, number];

  const v1Src = await rawArtifactDataUrl(page, projectId, 'image_slice_1.png');
  const beforeOutsideAlpha = (await sourceImagePixel(page, v1Src, ...outside))[3];

  await ui.copySlice();
  await expect(ui.log()).toContainText('Copied mask to clipboard');
  expect((await readE2EState(page, projectId)).clipboard_present).toBe(true);

  await ui.pasteSlice();
  await expect(ui.log()).toContainText('Pasted clipboard to slice 1');
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_filenames[1])
    .toBe('image_slice_1_v2.png');

  const v2Src = await rawArtifactDataUrl(page, projectId, 'image_slice_1_v2.png');
  // Paste blends by the clipboard's alpha (the mask), so the target alpha
  // becomes max(original, mask): opaque under the mask, unchanged outside it.
  expect((await sourceImagePixel(page, v2Src, ...inside))[3]).toBe(255);
  expect((await sourceImagePixel(page, v2Src, ...outside))[3]).toBe(beforeOutsideAlpha);
});

test('paste without a selected slice is a logged no-op', async ({ page, ui }) => {
  requireWorkflow(ui, 'slice-editing');
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');

  // Depth-mode clicks don't require a selected slice (the mask comes from
  // the full input image); copy it with nothing selected, per copy_to_clipboard's
  // `else: image = state.imgData` branch.
  await ui.clickImagePixel(80, 96);
  await expect.poll(async () => (await readE2EState(page, projectId)).slice_mask.present).toBe(true);
  await ui.copySlice();
  expect((await readE2EState(page, projectId)).clipboard_present).toBe(true);
  expect((await readE2EState(page, projectId)).selected_slice).toBeNull();

  await ui.pasteSlice();
  await expect(ui.log()).toContainText('No slice selected');
  expect((await readE2EState(page, projectId)).slice_versions).toEqual([1, 1, 1]);
});

// --- Slice depth reordering ------------------------------------------------------

test('setting a slice depth reorders slices and clears selection whenever any slice moves', async ({
  page,
  ui,
}) => {
  requireWorkflow(ui, 'slice-editing');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');

  // Note: a slice's own depth-number badge is covered by its ".overlay" once
  // selected (both are absolutely-positioned siblings and the overlay paints
  // on top), so it cannot be clicked through normal hit-testing while that
  // slice is selected -- a real, reproducible Dash UI limitation. This test
  // instead edits a *different*, unselected slice's depth, which both sides
  // of record_depth_input's "if index != new_index: state.selected_slice =
  // None" (webui.py:1066-1068) allow: the clause clears the selection
  // unconditionally on any reorder, not only when the selected slice itself
  // is the one that moved.
  await ui.selectSlice(projectId, 1); // depth 170, filename image_slice_1.png

  // Change the (unselected) first slice's depth so it reorders past the
  // second and third slices.
  await ui.setSliceDepth(0, 200);
  let state = await readE2EState(page, projectId);
  expect(state.slice_depths).toEqual([170, 200, 255]);
  expect(state.slice_filenames).toEqual([
    'image_slice_1.png',
    'image_slice_0.png',
    'image_slice_2.png',
  ]);
  // The edited slice's index changed (0 -> 1); Dash clears the selection even
  // though the *selected* slice (image_slice_1.png) never moved.
  expect(state.selected_slice).toBeNull();

  // Re-select the slice now at index 0 (depth 170, unaffected by the next
  // edit) and nudge a *different* slice's depth without crossing a neighbor:
  // no reorder occurs, so the selection survives.
  await ui.selectSlice(projectId, 0);
  await ui.setSliceDepth(2, 254); // depth 255 -> 254, stays last
  state = await readE2EState(page, projectId);
  expect(state.slice_depths).toEqual([170, 200, 254]);
  expect(state.selected_slice).toBe(0);
});

// --- Slice image upload ------------------------------------------------------------

test('uploading a matching-aspect image replaces the slice content and bumps its version', async ({ page, ui }) => {
  requireWorkflow(ui, 'slice-editing');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');
  const thumbnailBefore = await imageHash(ui.sliceImages().nth(1));

  // The 320x240 input fixture shares the slice canvas's exact aspect ratio
  // (every slice image is full-canvas size, only alpha differs), so this
  // upload takes the "no resize" path.
  const fixture = await fetchFixture(page, 'input.png');
  expect(fixture.ok()).toBeTruthy();
  await ui.uploadSliceImage(1, {
    name: 'replacement.png',
    mimeType: 'image/png',
    buffer: await fixture.body(),
  });

  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_filenames[1])
    .toBe('image_slice_1_v2.png');
  expect((await readE2EState(page, projectId)).slice_versions[1]).toBe(2);
  await expect.poll(() => imageHash(ui.sliceImages().nth(1))).not.toBe(thumbnailBefore);

  // The fixture draws a solid red rectangle at (24,24)-(104,96); (30, 30) is
  // inside it. No resize occurred, so the uploaded pixel lands unchanged and
  // fully opaque (input.png has no alpha channel).
  const rawSrc = await rawArtifactDataUrl(page, projectId, 'image_slice_1_v2.png');
  expect(await sourceImagePixel(page, rawSrc, 30, 30)).toEqual([240, 50, 45, 255]);
});

test('uploading a mismatched-aspect image collapses the slice to the resized dimensions', async ({
  page,
  ui,
}) => {
  requireWorkflow(ui, 'slice-editing');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');

  // A 2x1 solid-color PNG: aspect ratio 2.0 versus the slice's 320/240
  // (~1.333), so slice_upload's mismatched-aspect branch resizes it to
  // (int(1.333 * 1), 1) = (1, 1) -- collapsing the slice to a single pixel.
  // This also pins a real bug: the log message is a plain (non-f) string
  // literal, so it is never interpolated -- see the written report.
  const twoByOnePng = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAD0lEQVR4nGPkEpFjYGAAAAEmAD5j+GBZAAAAAElFTkSuQmCC',
    'base64',
  );
  await ui.uploadSliceImage(2, {
    name: 'mismatched.png',
    mimeType: 'image/png',
    buffer: twoByOnePng,
  });

  await expect(ui.log()).toContainText(
    'Fixing aspect ratio from {image.size[0] / image.size[1]} to {aspect_ratio}',
  );
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_filenames[2])
    .toBe('image_slice_2_v2.png');

  const response = await fetchArtifact(page, projectId, 'image_slice_2_v2.png');
  expect(response.ok()).toBeTruthy();
  const metadata = await imageBufferMetadata(page, await response.body());
  expect(metadata.width).toBe(1);
  expect(metadata.height).toBe(1);
});

// --- Invert / feather mask ------------------------------------------------------------

test('invert flips every mask sample and starts from an all-zero mask when none exists', async ({ page, ui }) => {
  requireWorkflow(ui, 'mask-tools');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');

  // No mask yet: invert creates an all-zero mask first, then flips it to
  // all-255 (state.slice_mask = 255 - zeros).
  await ui.invertMask();
  let state = await readE2EState(page, projectId);
  expect(state.slice_mask.present).toBe(true);
  expect(state.slice_mask.nonzero).toBe(TOTAL_PIXELS);
  expect(state.slice_mask.max).toBe(255);

  // Inverting a real binary mask flips every sample and its nonzero count.
  // (The mode dropdown already defaults to "Depth Map".) A plain click always
  // REPLACEs the mask, but it was already present (all-255) from the previous
  // step, so poll for the click's own effect (a shrunken mask) rather than
  // mere presence, which was already satisfied.
  await ui.clickImagePixel(80, 96);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_mask.nonzero)
    .toBeLessThan(TOTAL_PIXELS);
  const before = (await readE2EState(page, projectId)).slice_mask;

  await ui.invertMask();
  const after = (await readE2EState(page, projectId)).slice_mask;
  expect(after.nonzero).toBe(TOTAL_PIXELS - before.nonzero);
  for (const key of Object.keys(before.samples)) {
    expect(after.samples[key]).toBe(255 - before.samples[key]);
  }
});

test('feather blurs the mask edge to intermediate values while capping interior at 255', async ({ page, ui }) => {
  requireWorkflow(ui, 'mask-tools');
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.setSegmentationMode('Depth Map');
  await ui.openTab('Segmentation');

  await ui.clickImagePixel(80, 96);
  await expect.poll(async () => (await readE2EState(page, projectId)).slice_mask.present).toBe(true);
  const before = (await readE2EState(page, projectId)).slice_mask;
  expect(before.bounds).not.toBeNull();

  await ui.featherMask();

  const after = (await readE2EState(page, projectId)).slice_mask;
  // A 10px box blur cannot raise the mask above its original binary maximum,
  // so the interior (far from any edge) keeps full opacity...
  expect(after.max).toBe(255);
  // ...but every previously-zero pixel within blur range of the boundary now
  // carries a nonzero (partial, intermediate) contribution: a box blur can
  // only touch pixels strictly outside a real mask edge with values strictly
  // between 0 and 255 (reaching exactly 255 there would require the kernel's
  // entire footprint to already be inside the mask, i.e. not be an edge
  // pixel at all), so the nonzero count strictly grows and the tight bounding
  // box does not shrink on any side.
  expect(after.nonzero).toBeGreaterThan(before.nonzero);
  const beforeBounds = before.bounds as [number, number, number, number];
  const afterBounds = after.bounds as [number, number, number, number];
  expect(afterBounds[0]).toBeLessThanOrEqual(beforeBounds[0]);
  expect(afterBounds[1]).toBeLessThanOrEqual(beforeBounds[1]);
  expect(afterBounds[2]).toBeGreaterThanOrEqual(beforeBounds[2]);
  expect(afterBounds[3]).toBeGreaterThanOrEqual(beforeBounds[3]);
});

// --- Checkerboard toggle ------------------------------------------------------------

test('checkerboard toggles the selected-slice preview and no-ops without a selection', async ({ page, ui }) => {
  requireWorkflow(ui, 'mask-tools');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');

  const hashBeforeAnySelection = await imageHash(ui.mainImage());
  await ui.toggleCheckerboard();
  expect((await readE2EState(page, projectId)).use_checkerboard).toBe(true);
  // No slice is selected, so the main image is left untouched (Dash's
  // toggle_checkerboard only recomposes when state.selected_slice is set).
  expect(await imageHash(ui.mainImage())).toBe(hashBeforeAnySelection);

  await ui.selectSlice(projectId, 1);
  const checkerboardHash = await imageHash(ui.mainImage());

  await ui.toggleCheckerboard();
  expect((await readE2EState(page, projectId)).use_checkerboard).toBe(false);
  const grayscaleHash = await imageHash(ui.mainImage());
  expect(grayscaleHash).not.toBe(checkerboardHash);

  await ui.toggleCheckerboard();
  expect((await readE2EState(page, projectId)).use_checkerboard).toBe(true);
  await expect.poll(() => imageHash(ui.mainImage())).toBe(checkerboardHash);
});

// --- Balance slice depths (controller.py bug fix) ------------------------------------
//
// AppState.balance_slices_depths() itself is fixed in parallax_maker/controller.py
// (see parallax_maker/test_controller.py::TestBalanceSlicesDepths for direct,
// thorough characterization: 0/1/2/5-slice cases). But the Dash "Balance"
// button cannot actually reach that method today: webui.py:883's
// `balance_slices_request` reads `state.image_depths`, an attribute AppState
// has never defined (only `image_slices`). This is a separate, pre-existing
// bug in the frozen webui.py, independent of and not fixed by the
// controller.py change, so every click raises AttributeError -> HTTP 500
// before `balance_slices_depths()` ever runs. The scenario asserts the
// correct outcome and is an expected failure on Dash only.

test('balance evenly distributes slice depths', async ({ page, ui }) => {
  requireWorkflow(ui, 'slice-editing');
  test.fail(
    ui.target === 'dash',
    'webui.py:883 reads state.image_depths, which AppState never defines; every ' +
      'Balance click 500s before reaching the (now-fixed) AppState.balance_slices_depths(). ' +
      'Dash is frozen; see PARITY.md "Known quirks".',
  );

  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');
  await ui.balanceSlices();
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_depths)
    .toEqual([0, 127, 255]);
});
