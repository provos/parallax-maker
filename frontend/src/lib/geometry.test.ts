import { describe, expect, it } from 'vitest';
import { screenToImage, type Rect } from './geometry';

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
