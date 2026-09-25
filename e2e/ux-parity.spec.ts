/**
 * Shared UX-parity scenarios for behaviors Dash also has: zoom/pan of the
 * main image + mask canvas (Dash's JS-03 wheel-only zoom; Svelte's
 * state/viewport.svelte.ts adds drag-to-pan and a Reset button -- see that
 * module's own doc comment for exactly what Dash lacks and why) and queued
 * multi-point markers (Dash's CLI-05 `visualize_point`; Svelte's
 * PreviewOverlay.svelte).
 *
 * Every scenario runs on both drivers (`e2e/drivers/dash.ts`/`svelte.ts`);
 * where Dash's own behavior is a confirmed gap rather than a difference in
 * approach, the assertion is still made, `test.fail(ui.target === 'dash',
 * reason)`-pinned with the empirical evidence for why (see each test's own
 * comment and docs/svelte-migration/PARITY.md's "Known quirks").
 */
import { disableAnimations, expect, requireWorkflow, test } from './fixtures';
import { readE2EState } from './helpers/oracle';

test.beforeEach(async ({ ui, page }) => {
  await ui.goto();
  await disableAnimations(page);
});

test('after zooming in and panning, a depth-mode click at a known source pixel logs the exact expected pixel coordinates', async ({
  page,
  ui,
}) => {
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.expectSegmentationMode('Depth Map');

  // Visit the Inpainting tab first: on Dash, the wheel listener is only
  // ever attached lazily (the mouse must enter/press `#canvas` once -
  // utility.js's `setupMainCanvas`, called from `canvas_draw`'s
  // `mouseenter` case), and a wheel event only reaches that listener while
  // `#canvas` is the topmost element under the cursor, which is only true
  // on the Inpainting tab (`update_events`/CMP-01's z-index swap) -
  // confirmed empirically while writing this scenario. A real user
  // zooming in before painting a precise mask is a completely natural
  // flow, so this is not a workaround specific to the test, just the tab
  // that makes the gesture meaningful on Dash. Segmentation clicks on the
  // main image are not tab-gated on either UI (Dash's `click_event`/
  // Svelte's `onImageClick` both attach unconditionally), so no tab switch
  // is needed afterwards.
  await ui.openTab('Inpainting');

  const beforeZoom = await ui.mainImage().boundingBox();
  if (!beforeZoom) throw new Error('Main image has no bounding box');

  // Each real wheel gesture is one zoom "tick" (ZOOM_FACTOR = 1.1, matching
  // Dash's own utility.js exactly); five ticks comfortably clears a 30%
  // growth threshold (1.1^5 ≈ 1.61x) with margin for the two UIs' slightly
  // different rendered-box rounding.
  for (let i = 0; i < 5; i += 1) await ui.zoomIn();

  const afterZoom = await ui.mainImage().boundingBox();
  if (!afterZoom) throw new Error('Main image has no bounding box after zoom');
  // Zooming in must visibly enlarge the rendered box on both UIs (Dash's
  // own CSS `transform: scale()`, Svelte's `state/viewport.svelte.ts`).
  expect(afterZoom.width, 'image width grows after zooming in').toBeGreaterThan(
    beforeZoom.width * 1.3,
  );

  await ui.panBy(35, -20);
  const afterPan = await ui.mainImage().boundingBox();
  if (!afterPan) throw new Error('Main image has no bounding box after pan');

  // Dash has no drag-to-pan mechanism at all for the main image/canvas
  // (utility.js has no `mousedown`-driven pan; only wheel-zoom, brush
  // painting, and the unrelated NAV_* 3D-camera-dolly buttons - see
  // PARITY.md's Navigation/Layout section and viewport.svelte.ts's own doc
  // comment). `ui.panBy` performs a real drag gesture on both UIs; on Dash
  // it is a real, confirmed no-op.
  test.fail(
    ui.target === 'dash',
    'Dash has no drag-to-pan for the main image/canvas at all (only CSS wheel-zoom, JS-03); ' +
      'a real drag gesture over the image has no effect there, confirmed empirically.',
  );
  expect(
    Math.abs(afterPan.x - afterZoom.x) + Math.abs(afterPan.y - afterZoom.y),
    'panning moves the rendered image box',
  ).toBeGreaterThan(5);

  // Leave the Inpainting tab before clicking: on both UIs, the mask canvas
  // is the topmost (interactive) element over the image while that tab is
  // active (Dash's `update_events`/CMP-01; Svelte's `MaskCanvas.svelte`
  // `interactive` class), so a click there would hit the canvas's own
  // paint-stroke handling instead of segmentation - not a zoom/pan concern,
  // just normal tab-based hit-testing on both UIs. The zoom/pan transform
  // itself is unaffected by the tab switch (proven by `afterZoom`/`afterPan`
  // being measured before this point, and re-confirmed by `finalBox` below).
  await ui.openTab('Segmentation');
  const finalBox = await ui.mainImage().boundingBox();
  if (!finalBox) throw new Error('Main image has no bounding box after switching tabs');
  expect(finalBox.width, 'zoom survives switching tabs').toBeCloseTo(afterZoom.width, 0);

  // Whatever the current zoom/pan transform actually is, a click at a known
  // source pixel must still resolve to (within one pixel of) that exact
  // pixel: both drivers' `clickImagePixel` recompute the click position
  // from the *live* (post-transform) bounding rect at click time, so this
  // holds regardless of whether the pan above actually took effect -
  // proving the pixel mapping itself (lib/geometry.ts's
  // `findPixelFromClick`/Dash's `find_pixel_from_click`) stays exact under
  // zoom, independent of the pan gap just pinned above. The `<= 1`
  // tolerance (not exact equality) is the same real sub-pixel truncation
  // quirk `e2e/parallax-maker.spec.ts`'s own "default depth click" scenario
  // pins for a *plain, unzoomed* click (a requested (16, 16) truncates to
  // (15, 15)) - a fractional zoom scale only makes that rounding harder to
  // predict exactly by hand, not any less exact in what the app itself does.
  await ui.clickImagePixel(160, 120);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_pixel)
    .not.toBeNull();
  const { slice_pixel: clicked } = await readE2EState(page, projectId);
  expect(clicked, 'clicked pixel is within 1px of the requested (160, 120)').not.toBeNull();
  const [px, py] = clicked as [number, number];
  expect(Math.abs(px - 160)).toBeLessThanOrEqual(1);
  expect(Math.abs(py - 120)).toBeLessThanOrEqual(1);
  await expect(ui.log()).toContainText(`Click event at pixel coordinates (${px}, ${py})`);
});

test('queued multi-point markers appear at the clicked points', async ({ ui }) => {
  requireWorkflow(ui, 'segmentation');
  await ui.restoreFixtureState();
  await ui.setSegmentationMode('Instance Segmentation');
  await ui.openTab('Segmentation');

  await ui.toggleMultiPoint();
  await ui.expectMultiPointEnabled(true);

  // A plain click queues a positive (green) point; a Ctrl-click queues a
  // negative (red) one - same modifier Dash's `click_event`/CLI-05 key off.
  await ui.clickImagePixel(90, 96);
  await ui.clickImagePixel(150, 110, ['Control']);

  await ui.expectQueuedPointMarkers([
    { x: 90, y: 96, negative: false },
    { x: 150, y: 110, negative: true },
  ]);
});

test('a reset returns the view to its original, unzoomed state', async ({ ui }) => {
  requireWorkflow(ui, 'segmentation');
  await ui.restoreFixtureState();
  await ui.openTab('Inpainting'); // see the first test's comment on why Dash needs this to zoom at all.

  const original = await ui.mainImage().boundingBox();
  if (!original) throw new Error('Main image has no bounding box');

  for (let i = 0; i < 5; i += 1) await ui.zoomIn();
  const zoomed = await ui.mainImage().boundingBox();
  if (!zoomed) throw new Error('Main image has no bounding box after zoom');
  expect(zoomed.width).toBeGreaterThan(original.width * 1.3);

  // Svelte has an explicit Reset button (state/viewport.svelte.ts); Dash
  // has no reset control for its CSS zoom at all, so `resetZoom` there is
  // only a documented best-effort (repeated zoom-out), not an exact reset -
  // see DashDriver.resetZoom's own doc comment.
  test.fail(
    ui.target === 'dash',
    'Dash has no reset control for its CSS zoom; repeated zoom-out only approaches, ' +
      "but is not guaranteed to exactly reach, the image's original unzoomed size.",
  );
  await ui.resetZoom();
  const reset = await ui.mainImage().boundingBox();
  if (!reset) throw new Error('Main image has no bounding box after reset');
  expect(reset.width).toBeCloseTo(original.width, 0);
  expect(reset.x).toBeCloseTo(original.x, 0);
  expect(reset.y).toBeCloseTo(original.y, 0);
});
