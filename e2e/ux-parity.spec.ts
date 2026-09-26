/**
 * Shared UX-parity scenarios: zoom/pan of the main image + mask canvas
 * (state/viewport.svelte.ts's wheel-zoom, drag-to-pan and Reset button) and
 * queued multi-point markers (PreviewOverlay.svelte).
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
  await ui.setSegmentationMode('Depth Map');

  // A real user zooming in before painting a precise mask is a completely
  // natural flow, so visit the Inpainting step first (the click below
  // switches back to the Segment tool; zoom/pan survives that).
  await ui.openTab('Inpainting');

  const beforeZoom = await ui.canvasImage().boundingBox();
  if (!beforeZoom) throw new Error('Main image has no bounding box');

  // Each real wheel gesture is one zoom "tick" (ZOOM_FACTOR = 1.1); five
  // ticks comfortably clears a 30% growth threshold (1.1^5 ≈ 1.61x) with
  // margin for rendered-box rounding.
  for (let i = 0; i < 5; i += 1) await ui.zoomIn();

  const afterZoom = await ui.canvasImage().boundingBox();
  if (!afterZoom) throw new Error('Main image has no bounding box after zoom');
  // Zooming in must visibly enlarge the rendered box (state/viewport.svelte.ts).
  expect(afterZoom.width, 'image width grows after zooming in').toBeGreaterThan(
    beforeZoom.width * 1.3,
  );

  await ui.panBy(35, -20);
  const afterPan = await ui.canvasImage().boundingBox();
  if (!afterPan) throw new Error('Main image has no bounding box after pan');

  expect(
    Math.abs(afterPan.x - afterZoom.x) + Math.abs(afterPan.y - afterZoom.y),
    'panning moves the rendered image box',
  ).toBeGreaterThan(5);

  // Leave the Inpainting step before clicking: its Brush tool makes the mask
  // canvas the topmost (interactive) element over the image, so a click
  // there would paint instead of segmenting. The zoom/pan transform itself
  // is unaffected by the switch (re-confirmed by `finalBox` below).
  await ui.openTab('Segmentation');
  const finalBox = await ui.canvasImage().boundingBox();
  if (!finalBox) throw new Error('Main image has no bounding box after switching tabs');
  expect(finalBox.width, 'zoom survives switching tabs').toBeCloseTo(afterZoom.width, 0);

  // Whatever the current zoom/pan transform actually is, a click at a known
  // source pixel must still resolve to (within one pixel of) that exact
  // pixel: `clickImagePixel` recomputes the click position from the *live*
  // (post-transform) bounding rect at click time, so this holds regardless
  // of whether the pan above actually took effect - proving the pixel
  // mapping itself (lib/geometry.ts's `findPixelFromClick`) stays exact
  // under zoom, independent of the pan gap just pinned above. The `<= 1`
  // tolerance allows for the fractional zoom scale's sub-pixel rounding.
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
  // negative (red) one.
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
  await ui.openTab('Inpainting');

  const original = await ui.canvasImage().boundingBox();
  if (!original) throw new Error('Main image has no bounding box');

  for (let i = 0; i < 5; i += 1) await ui.zoomIn();
  const zoomed = await ui.canvasImage().boundingBox();
  if (!zoomed) throw new Error('Main image has no bounding box after zoom');
  expect(zoomed.width).toBeGreaterThan(original.width * 1.3);

  // The Reset button (state/viewport.svelte.ts) returns the view to its
  // original, unzoomed size exactly.
  await ui.resetZoom();
  const reset = await ui.canvasImage().boundingBox();
  if (!reset) throw new Error('Main image has no bounding box after reset');
  expect(reset.width).toBeCloseTo(original.width, 0);
  expect(reset.x).toBeCloseTo(original.x, 0);
  expect(reset.y).toBeCloseTo(original.y, 0);
});
