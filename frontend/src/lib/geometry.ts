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

/**
 * Convert a click's viewport (client) coordinates to integer source-image
 * pixel coordinates, exactly like Dash's `find_pixel_from_click`/
 * `find_pixel_from_event` (`parallax_maker/utils.py`): truncate
 * `(clientX - rect.left) * naturalWidth / rect.width` (and the same for Y
 * using height) -- no letterbox/aspect-ratio compensation.
 *
 * This assumes the element's box has the image's own aspect ratio (see
 * InputImagePanel.svelte, which renders the main image at `width: 100%;
 * height: auto` for exactly this reason); with letterboxing this formula
 * would not agree with `screenToImage` above.
 *
 * Returns `null` when the resulting pixel falls outside the image bounds
 * (e.g. a click landing exactly on the box's far edge due to rounding), so
 * callers can ignore it the same way the backend would reject it with 400.
 */
export function findPixelFromClick(
  clientX: number,
  clientY: number,
  rect: Rect,
  naturalWidth: number,
  naturalHeight: number,
): ImagePoint | null {
  if (rect.width <= 0 || rect.height <= 0 || naturalWidth <= 0 || naturalHeight <= 0) {
    return null;
  }

  const x = Math.trunc(((clientX - rect.left) * naturalWidth) / rect.width);
  const y = Math.trunc(((clientY - rect.top) * naturalHeight) / rect.height);

  if (x < 0 || y < 0 || x >= naturalWidth || y >= naturalHeight) {
    return null;
  }

  return { x, y };
}
