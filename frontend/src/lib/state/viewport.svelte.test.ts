import { describe, expect, it } from 'vitest';
import { MAX_SCALE, MIN_SCALE, ZOOM_FACTOR, viewportStore } from './viewport.svelte';

describe('viewportStore', () => {
  it('starts at 1x scale with no pan', () => {
    viewportStore.reset();
    expect(viewportStore.scale).toBe(1);
    expect(viewportStore.panX).toBe(0);
    expect(viewportStore.panY).toBe(0);
  });

  it('zoomInAt multiplies scale by ZOOM_FACTOR', () => {
    viewportStore.reset();
    viewportStore.zoomInAt(50, 50);
    expect(viewportStore.scale).toBeCloseTo(ZOOM_FACTOR, 10);
  });

  it('zoomOutAt divides scale by ZOOM_FACTOR', () => {
    viewportStore.reset();
    viewportStore.zoomOutAt(50, 50);
    expect(viewportStore.scale).toBeCloseTo(1 / ZOOM_FACTOR, 10);
  });

  it('clamps scale to MAX_SCALE when zooming in repeatedly', () => {
    viewportStore.reset();
    for (let i = 0; i < 200; i += 1) viewportStore.zoomInAt(0, 0);
    expect(viewportStore.scale).toBe(MAX_SCALE);
  });

  it('clamps scale to MIN_SCALE when zooming out repeatedly', () => {
    viewportStore.reset();
    for (let i = 0; i < 200; i += 1) viewportStore.zoomOutAt(0, 0);
    expect(viewportStore.scale).toBe(MIN_SCALE);
  });

  it('zooming towards a point keeps that point fixed on screen', () => {
    viewportStore.reset();
    const localX = 40;
    const localY = 30;
    viewportStore.zoomInAt(localX, localY);
    // content point under the cursor before zooming: (localX - panX0) / scale0 = (40 - 0) / 1 = 40
    // after zooming, the same content point (40) must render back at localX:
    // panX + 40 * scale === localX
    expect(viewportStore.panX + 40 * viewportStore.scale).toBeCloseTo(localX, 8);
    expect(viewportStore.panY + 30 * viewportStore.scale).toBeCloseTo(localY, 8);
  });

  it('panBy adds a scale-independent screen-pixel offset', () => {
    viewportStore.reset();
    viewportStore.zoomInAt(0, 0);
    viewportStore.zoomInAt(0, 0);
    const scaleBefore = viewportStore.scale;
    viewportStore.panBy(12, -7);
    expect(viewportStore.panX).toBeCloseTo(12, 8);
    expect(viewportStore.panY).toBeCloseTo(-7, 8);
    expect(viewportStore.scale).toBe(scaleBefore); // panning never changes scale
  });

  it('reset restores 1x scale and clears pan, regardless of prior state', () => {
    viewportStore.zoomInAt(10, 10);
    viewportStore.zoomInAt(20, 5);
    viewportStore.panBy(100, -50);
    viewportStore.reset();
    expect(viewportStore.scale).toBe(1);
    expect(viewportStore.panX).toBe(0);
    expect(viewportStore.panY).toBe(0);
  });

  it('repeated zoom-in then zoom-out at the same point returns to (approximately) the original scale', () => {
    viewportStore.reset();
    viewportStore.zoomInAt(64, 48);
    viewportStore.zoomInAt(64, 48);
    viewportStore.zoomInAt(64, 48);
    viewportStore.zoomOutAt(64, 48);
    viewportStore.zoomOutAt(64, 48);
    viewportStore.zoomOutAt(64, 48);
    expect(viewportStore.scale).toBeCloseTo(1, 8);
    expect(viewportStore.panX).toBeCloseTo(0, 6);
    expect(viewportStore.panY).toBeCloseTo(0, 6);
  });
});
