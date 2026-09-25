/**
 * Zoom/pan state for the main image + mask canvas + preview overlay stack
 * (InputImagePanel.svelte's `.image-stack`), applied as one CSS
 * `transform: translate(panX, panY) scale(scale); transform-origin: 0 0`.
 *
 * Mirrors Dash's own zoom (utility.js's `handleWheel`/JS-03: mouse-wheel
 * only, clamped to 0.125x-8x, zoom-factor 1.1 per wheel tick, scaling about
 * the cursor via `transform-origin`) with the same clamp/factor constants,
 * so a given number of wheel ticks lands on the same scale on both UIs.
 *
 * Dash has no drag-to-pan and no reset control for this transform at all
 * (only the wheel handler exists; the `NAV_*` buttons near it operate a
 * completely different feature -- a server-side 3D camera dolly over the
 * composited slice cards, `components.py`'s `navigate_image`/CMP-26, not a
 * CSS transform -- see docs/svelte-migration/PARITY.md's Navigation/Layout
 * section). Once zoomed in, a Dash user has no way to see the rest of the
 * image again short of zooming back out by hand: a real, if minor,
 * usability gap. `panBy`/`reset` below are a deliberate, documented
 * improvement over Dash for exactly that gap (see PARITY.md's "Remaining
 * gaps"/"Known quirks").
 *
 * Because the transform is `translate() scale()` (in that order), `panX`/
 * `panY` are plain, scale-independent CSS pixels in the *untransformed*
 * parent's coordinate space -- panning by a screen-pixel drag delta is
 * always just `panX += dx; panY += dy`, at any zoom level.
 */

export const MIN_SCALE = 0.125;
export const MAX_SCALE = 8;
export const ZOOM_FACTOR = 1.1;

function clampScale(scale: number): number {
  return Math.min(Math.max(MIN_SCALE, scale), MAX_SCALE);
}

function createViewportStore() {
  let scale = $state(1);
  let panX = $state(0);
  let panY = $state(0);

  /**
   * Zooms in (`zoomIn = true`) or out about the point `(localX, localY)`,
   * expressed in the *untransformed* container's own CSS-pixel coordinate
   * space (i.e. `event.clientX - container.getBoundingClientRect().left`,
   * ignoring any current transform) -- exactly what Dash's `handleWheel`
   * does with `e.offsetX`/`e.offsetY` and `transform-origin`, just expressed
   * as an explicit translate instead of a moving transform-origin so it
   * composes cleanly with panning.
   *
   * The content point under `(localX, localY)` stays fixed on screen: this
   * is what makes "zoom towards the cursor" feel natural instead of
   * re-centering on every wheel tick.
   */
  function zoomAt(localX: number, localY: number, zoomIn: boolean): void {
    const factor = zoomIn ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
    const next = clampScale(scale * factor);
    if (next === scale) return;
    const contentX = (localX - panX) / scale;
    const contentY = (localY - panY) / scale;
    panX = localX - contentX * next;
    panY = localY - contentY * next;
    scale = next;
  }

  return {
    get scale(): number {
      return scale;
    },
    get panX(): number {
      return panX;
    },
    get panY(): number {
      return panY;
    },

    zoomAt,
    zoomInAt(localX: number, localY: number): void {
      zoomAt(localX, localY, true);
    },
    zoomOutAt(localX: number, localY: number): void {
      zoomAt(localX, localY, false);
    },

    /** Pans by a screen-pixel delta (e.g. a pointer drag's movementX/movementY). */
    panBy(dx: number, dy: number): void {
      panX += dx;
      panY += dy;
    },

    /** Restores 1x scale, no pan -- see the module doc for why Dash has no equivalent. */
    reset(): void {
      scale = 1;
      panX = 0;
      panY = 0;
    },
  };
}

export const viewportStore = createViewportStore();
export type ViewportStore = ReturnType<typeof createViewportStore>;
