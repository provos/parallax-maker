/** Minimal rect shape; a subset of DOMRect so callers can pass a plain object in tests. */
export type Rect = {
  left: number;
  top: number;
  width: number;
  height: number;
};

export type ImagePoint = {
  x: number;
  y: number;
};

/**
 * Convert a point in viewport (client) coordinates to a point in the
 * natural pixel space of an image displayed with CSS `object-fit: contain`
 * inside `rect`.
 *
 * `object-fit: contain` scales the image to fit inside its box while
 * preserving aspect ratio, centering it and letterboxing the remaining
 * space on either the horizontal or vertical axis (whichever doesn't
 * exactly fill the box). This mirrors that layout math so pointer events
 * on the displayed <img> can be translated back into image pixels.
 *
 * Returns `null` when the point falls outside the rendered image (i.e. in
 * the letterboxed padding, or outside `rect` entirely).
 */
export function screenToImage(
  clientX: number,
  clientY: number,
  rect: Rect,
  naturalWidth: number,
  naturalHeight: number,
): ImagePoint | null {
  if (
    rect.width <= 0 ||
    rect.height <= 0 ||
    naturalWidth <= 0 ||
    naturalHeight <= 0
  ) {
    return null;
  }

  const scale = Math.min(rect.width / naturalWidth, rect.height / naturalHeight);
  const renderedWidth = naturalWidth * scale;
  const renderedHeight = naturalHeight * scale;

  // object-fit: contain centers the scaled image within the box.
  const offsetX = (rect.width - renderedWidth) / 2;
  const offsetY = (rect.height - renderedHeight) / 2;

  const localX = clientX - rect.left - offsetX;
  const localY = clientY - rect.top - offsetY;

  if (localX < 0 || localY < 0 || localX > renderedWidth || localY > renderedHeight) {
    return null;
  }

  const imageX = localX / scale;
  const imageY = localY / scale;

  // Clamp to guard against floating point overshoot exactly on the edge.
  return {
    x: Math.min(Math.max(imageX, 0), naturalWidth),
    y: Math.min(Math.max(imageY, 0), naturalHeight),
  };
}
