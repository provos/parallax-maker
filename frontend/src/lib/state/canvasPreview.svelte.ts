/**
 * Transient, purely-visual overlay state drawn by
 * `components/canvas/PreviewOverlay.svelte` on top of the main image/mask
 * canvas: the brush-size preview circle that follows the pointer (Dash's
 * JS-01 `previewBrush`) and the region-of-interest bounding-box preview
 * shown for ~2s after a mask save with "Crop to region of interest"
 * checked (Dash's CLI-07 `show_bounding_box`/CMP-24). Both are in *source
 * image pixel* coordinates (the same space as `ProjectView.image`), so
 * `PreviewOverlay.svelte` can position them purely with CSS percentages
 * inside the same zoom/pan-transformed `.image-stack` box the image and
 * mask canvas live in -- see viewport.svelte.ts.
 *
 * Owned by `MaskCanvas.svelte` (the only thing that tracks pointer position
 * over the canvas and calls the mask-save endpoint); `PreviewOverlay.svelte`
 * only reads it.
 */

export type BrushPreview = {
  /** Source-image pixel coordinates of the pointer. */
  x: number;
  y: number;
  /** Brush diameter, in source-image pixels at the current zoom (see MaskCanvas.svelte's `scaleFactor`). */
  diameter: number;
  erasing: boolean;
};

export type RoiBox = readonly [number, number, number, number];

function createCanvasPreviewStore() {
  let brush = $state<BrushPreview | null>(null);
  let roiBox = $state<RoiBox | null>(null);
  let roiTimer: ReturnType<typeof setTimeout> | undefined;

  return {
    get brush(): BrushPreview | null {
      return brush;
    },
    setBrush(next: BrushPreview | null): void {
      brush = next;
    },
    clearBrush(): void {
      brush = null;
    },

    get roiBox(): RoiBox | null {
      return roiBox;
    },
    /** Shows `box` and auto-clears it after 2s, matching Dash's `previewRect`'s own `setTimeout`. */
    showRoiBox(box: RoiBox): void {
      roiBox = box;
      if (roiTimer) clearTimeout(roiTimer);
      roiTimer = setTimeout(() => {
        roiBox = null;
        roiTimer = undefined;
      }, 2000);
    },

    /** Test-only: restores default (empty) state. */
    reset(): void {
      brush = null;
      roiBox = null;
      if (roiTimer) clearTimeout(roiTimer);
      roiTimer = undefined;
    },
  };
}

export const canvasPreviewStore = createCanvasPreviewStore();
export type CanvasPreviewStore = ReturnType<typeof createCanvasPreviewStore>;
