/**
 * Paint-canvas tool state shared between the canvas itself
 * (MaskCanvas.svelte, which lives inside the zoomed image box) and its
 * toolbar (MaskToolbar.svelte, which lives in the tool rows under the
 * image). Clear/Load need the canvas's own pixels, so the canvas registers
 * those two actions here while it is mounted.
 */

export const BRUSH_MIN = 5;
export const BRUSH_MAX = 100;

type CanvasActions = { clear: () => Promise<void>; load: () => Promise<void> };

function createMaskToolsStore() {
  let erasing = $state(false);
  let drawWidth = $state(40);
  let eraseWidth = $state(60);
  let actions: CanvasActions | null = null;

  return {
    get erasing(): boolean {
      return erasing;
    },
    /** Width of the active brush (the eraser has its own width). */
    get brushWidth(): number {
      return erasing ? eraseWidth : drawWidth;
    },
    setBrushWidth(value: number): void {
      const clamped = Math.min(BRUSH_MAX, Math.max(BRUSH_MIN, value));
      if (erasing) eraseWidth = clamped;
      else drawWidth = clamped;
    },
    toggleErasing(): void {
      erasing = !erasing;
    },
    bindCanvas(next: CanvasActions): () => void {
      actions = next;
      return () => {
        if (actions === next) actions = null;
      };
    },
    clear(): Promise<void> {
      return actions?.clear() ?? Promise.resolve();
    },
    load(): Promise<void> {
      return actions?.load() ?? Promise.resolve();
    },
    reset(): void {
      erasing = false;
      drawWidth = 40;
      eraseWidth = 60;
    },
  };
}

export const maskToolsStore = createMaskToolsStore();
