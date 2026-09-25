import { describe, expect, it } from 'vitest';
import { findPixelFromClick, screenToImage, transformedRect, type Rect } from './geometry';

describe('screenToImage', () => {
  it('maps the box center to the image center for a square image in a square box', () => {
    const rect: Rect = { left: 0, top: 0, width: 100, height: 100 };
    const point = screenToImage(50, 50, rect, 200, 200);
    expect(point).toEqual({ x: 100, y: 100 });
  });

  it('maps the top-left corner to image origin', () => {
    const rect: Rect = { left: 0, top: 0, width: 100, height: 100 };
    const point = screenToImage(0, 0, rect, 200, 200);
    expect(point).toEqual({ x: 0, y: 0 });
  });

  it('accounts for horizontal letterboxing when the box is wider than the image aspect ratio', () => {
    // Box is 200x100 (2:1); image is 100x100 (1:1) -> rendered at 100x100,
    // centered with 50px of empty space on each side.
    const rect: Rect = { left: 0, top: 0, width: 200, height: 100 };
    const center = screenToImage(100, 50, rect, 100, 100);
    expect(center).toEqual({ x: 50, y: 50 });

    // A click in the left letterbox padding is outside the rendered image.
    expect(screenToImage(25, 50, rect, 100, 100)).toBeNull();
    // A click in the right letterbox padding is outside the rendered image.
    expect(screenToImage(175, 50, rect, 100, 100)).toBeNull();

    // Just inside the left edge of the rendered image (x=50 in box coords).
    const nearLeftEdge = screenToImage(51, 50, rect, 100, 100);
    expect(nearLeftEdge).not.toBeNull();
    expect(nearLeftEdge!.x).toBeCloseTo(1, 5);
  });

  it('accounts for vertical letterboxing when the box is taller than the image aspect ratio', () => {
    // Box is 100x200 (1:2); image is 100x100 (1:1) -> rendered at 100x100,
    // centered with 50px of empty space on top and bottom.
    const rect: Rect = { left: 0, top: 0, width: 100, height: 200 };
    const center = screenToImage(50, 100, rect, 100, 100);
    expect(center).toEqual({ x: 50, y: 50 });

    expect(screenToImage(50, 25, rect, 100, 100)).toBeNull();
    expect(screenToImage(50, 175, rect, 100, 100)).toBeNull();
  });

  it('handles a wide (16:9) image inside a square box', () => {
    const rect: Rect = { left: 0, top: 0, width: 300, height: 300 };
    // scale = min(300/1600, 300/900) = 0.1875 -> rendered 300x168.75,
    // vertical offset = (300 - 168.75) / 2 = 65.625
    const top = screenToImage(0, 65.625, rect, 1600, 900);
    expect(top).not.toBeNull();
    expect(top!.x).toBeCloseTo(0, 5);
    expect(top!.y).toBeCloseTo(0, 5);

    expect(screenToImage(0, 0, rect, 1600, 900)).toBeNull();
  });

  it('offsets by the rect origin, not just the viewport origin', () => {
    const rect: Rect = { left: 40, top: 20, width: 100, height: 100 };
    const point = screenToImage(90, 70, rect, 200, 200);
    expect(point).toEqual({ x: 100, y: 100 });
  });

  it('returns null for points outside the rect entirely', () => {
    const rect: Rect = { left: 0, top: 0, width: 100, height: 100 };
    expect(screenToImage(-10, 50, rect, 100, 100)).toBeNull();
    expect(screenToImage(50, -10, rect, 100, 100)).toBeNull();
    expect(screenToImage(150, 50, rect, 100, 100)).toBeNull();
    expect(screenToImage(50, 150, rect, 100, 100)).toBeNull();
  });

  it('returns null for degenerate rects or images', () => {
    expect(screenToImage(0, 0, { left: 0, top: 0, width: 0, height: 100 }, 100, 100)).toBeNull();
    expect(screenToImage(0, 0, { left: 0, top: 0, width: 100, height: 100 }, 0, 100)).toBeNull();
  });
});

describe('findPixelFromClick', () => {
  it('maps a click through the exact ratio Dash uses (no letterboxing)', () => {
    // rect is a clean half-scale rendering of a 320x240 image.
    const rect: Rect = { left: 0, top: 0, width: 160, height: 120 };
    expect(findPixelFromClick(80, 60, rect, 320, 240)).toEqual({ x: 160, y: 120 });
    expect(findPixelFromClick(0, 0, rect, 320, 240)).toEqual({ x: 0, y: 0 });
  });

  it('offsets by the rect origin, not just the viewport origin', () => {
    const rect: Rect = { left: 40, top: 20, width: 160, height: 120 };
    expect(findPixelFromClick(120, 80, rect, 320, 240)).toEqual({ x: 160, y: 120 });
  });

  it('truncates towards zero, matching a sub-pixel rendered box', () => {
    // A real browser's measured box width is rarely an exact integer; a
    // slightly-larger-than-nominal rect shifts the scaled-back pixel down by
    // one via truncation, exactly like the real click-through-Playwright
    // scenarios in e2e/parallax-maker.spec.ts (e.g. requested (90, 96)
    // arrives as (89, 95)).
    const rect: Rect = { left: 0, top: 0, width: 160.01, height: 120.0075 };
    const requestedX = 90 * (rect.width / 320);
    const requestedY = 96 * (rect.height / 240);
    expect(findPixelFromClick(requestedX, requestedY, rect, 320, 240)).toEqual({ x: 89, y: 95 });
  });

  it('returns null for a pixel outside the image (e.g. the far edge from rounding)', () => {
    const rect: Rect = { left: 0, top: 0, width: 160, height: 120 };
    expect(findPixelFromClick(160, 60, rect, 320, 240)).toBeNull(); // x truncates to 320
    expect(findPixelFromClick(-1, 60, rect, 320, 240)).toBeNull();
    expect(findPixelFromClick(80, -1, rect, 320, 240)).toBeNull();
  });

  it('returns null for degenerate rects or images', () => {
    expect(findPixelFromClick(0, 0, { left: 0, top: 0, width: 0, height: 100 }, 100, 100)).toBeNull();
    expect(findPixelFromClick(0, 0, { left: 0, top: 0, width: 100, height: 100 }, 0, 100)).toBeNull();
  });
});

describe('findPixelFromClick under zoom and pan (InputImagePanel.svelte + state/viewport.svelte.ts)', () => {
  // A real browser's `getBoundingClientRect()` already reflects any CSS
  // `transform` applied to an element (or one of its ancestors), so zoomed/
  // panned clicks need no special-casing in `findPixelFromClick` itself --
  // only the *rect* passed in changes. `transformedRect` computes exactly
  // that rect for a given `viewport.svelte.ts` state, so these tests double
  // as a proof that "click a known source pixel after zooming in and
  // panning" resolves to the exact same pixel a plain, unzoomed click would.
  const naturalWidth = 320;
  const naturalHeight = 240;
  // The base (zoom=1, pan=0) rect: a clean half-scale rendering, same as
  // the plain findPixelFromClick tests above.
  const baseRect: Rect = { left: 0, top: 0, width: 160, height: 120 };

  it('zooming in 2x about the origin doubles the rendered box and halves the per-pixel screen distance', () => {
    const rect = transformedRect(baseRect, { scale: 2, panX: 0, panY: 0 });
    expect(rect).toEqual({ left: 0, top: 0, width: 320, height: 240 });
    // Source pixel (160, 120) -- the exact center of the 320x240 image --
    // is now at screen (160, 120) rather than the unzoomed (80, 60).
    expect(findPixelFromClick(160, 120, rect, naturalWidth, naturalHeight)).toEqual({ x: 160, y: 120 });
  });

  it('a pan offset shifts every mapped pixel by exactly the pan amount', () => {
    const rect = transformedRect(baseRect, { scale: 1, panX: 37, panY: -11 });
    expect(rect).toEqual({ left: 37, top: -11, width: 160, height: 120 });
    // Unpanned, source pixel (80, 60) is at screen (40, 30) (half-scale).
    // Panned by (37, -11), it moves to screen (77, 19).
    expect(findPixelFromClick(77, 19, rect, naturalWidth, naturalHeight)).toEqual({ x: 80, y: 60 });
  });

  it('combined zoom + pan (as produced by zooming to a cursor position) still resolves the exact clicked pixel', () => {
    // Mirrors state/viewport.svelte.ts's `zoomAt`: zoom 3x about local point
    // (40, 30) (the unzoomed screen position of source pixel (80, 60)), which
    // keeps that same content point fixed under the cursor.
    const scale = 3;
    const localX = 40;
    const localY = 30;
    const contentX = (localX - 0) / 1; // pan was 0 before this zoom
    const contentY = (localY - 0) / 1;
    const panX = localX - contentX * scale;
    const panY = localY - contentY * scale;

    const rect = transformedRect(baseRect, { scale, panX, panY });
    // The point under the cursor should still be source pixel (80, 60).
    expect(findPixelFromClick(localX, localY, rect, naturalWidth, naturalHeight)).toEqual({ x: 80, y: 60 });

    // A different, previously off-screen pixel is now reachable too: source
    // pixel (0, 0) (the image's top-left corner) maps to screen (panX, panY).
    expect(findPixelFromClick(panX, panY, rect, naturalWidth, naturalHeight)).toEqual({ x: 0, y: 0 });
  });

  it('zooming out below 1x still maps correctly (clamped scale range is state/viewport.svelte.ts\'s concern, not geometry.ts\'s)', () => {
    const rect = transformedRect(baseRect, { scale: 0.5, panX: 5, panY: 5 });
    expect(rect).toEqual({ left: 5, top: 5, width: 80, height: 60 });
    expect(findPixelFromClick(5, 5, rect, naturalWidth, naturalHeight)).toEqual({ x: 0, y: 0 });
  });
});
