<script lang="ts">
  /**
   * The paint-mask canvas overlaid on the main image (InputImagePanel.svelte
   * renders this as a sibling of the `<img>`, inside a shared wrapper so both
   * occupy exactly the same box). Painting is only interactive on the
   * Inpainting tab (mirrors Dash's `update_events`/CMP-01, which raises the
   * canvas above the image only while that tab is active), but the element
   * stays mounted the rest of the time so its pixel content survives tab
   * switches.
   *
   * Unlike Dash (utility.js's `setupCanvasCtx`), the backing store is sized
   * to the *source image's own pixel dimensions*, not the CSS-rendered size:
   * the saved mask then maps to the source 1:1 with no resampling, and a
   * window resize never needs to touch the canvas's pixel content at all
   * (only its CSS box changes) -- see state/canvas.svelte.ts's module doc.
   *
   * Saving happens on pointerup (not Dash's `mouseout`), and every save is
   * registered with `canvasSaveStore` so Generate/Fill/Enhance/Erase and a
   * slice-selection change can all await it first -- see workflow.ts and
   * state/canvas.svelte.ts for the other half of that contract.
   */
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import { logStore } from '../../state/logs.svelte';
  import { canvasSaveStore } from '../../state/canvas.svelte';
  import * as workflow from '../../workflow';

  let canvasEl: HTMLCanvasElement | undefined;
  let ctx2d: CanvasRenderingContext2D | null = null;

  let isErasing = $state(false);
  let drawWidth = $state(40);
  let eraseWidth = $state(60);

  let isDrawing = false;
  let strokeDirty = false;
  let lastPoint: { x: number; y: number } | null = null;

  // Bumped on every slice-selection transition so a slow, now-superseded
  // mask fetch (Load, or the auto-load below) can recognize it is stale and
  // avoid painting onto the wrong slice's canvas.
  let loadToken = 0;
  let lastSelected: number | null | undefined = undefined;

  const interactiveNow = $derived(uiStore.mainTab === 'Inpainting');

  function getCtx(): CanvasRenderingContext2D | null {
    if (!canvasEl) return null;
    if (!ctx2d) ctx2d = canvasEl.getContext('2d');
    return ctx2d;
  }

  // Keep the backing store's pixel dimensions equal to the source image's,
  // so the saved PNG's alpha channel maps onto the mask 1:1 (the API resizes
  // BICUBIC to source dimensions regardless, but starting there avoids any
  // resampling loss). Guarded so this never fires -- and clears the canvas
  // -- unless the size actually changed.
  $effect(() => {
    const size = projectStore.view?.image;
    if (!canvasEl || !size) return;
    if (canvasEl.width !== size.width || canvasEl.height !== size.height) {
      canvasEl.width = size.width;
      canvasEl.height = size.height;
    }
  });

  // Explicit canvas lifecycle for slice-selection changes: flush any pending
  // save for the *previous* slice before doing anything else, then load the
  // newly-selected slice's own saved mask (or clear, if it has none). A
  // `loadToken` guards against a stale async load finishing after another
  // transition has already started.
  $effect(() => {
    const view = projectStore.view;
    const index = view?.selectedSlice ?? null;
    if (index === lastSelected) return;
    lastSelected = index;
    const slice = index !== null ? view?.slices.find((s) => s.index === index) : undefined;
    const maskUrl = slice?.mask?.url ?? null;
    const token = ++loadToken;

    void (async () => {
      await canvasSaveStore.flush();
      if (token !== loadToken) return;
      clearCanvasPixels();
      if (maskUrl) await drawMaskImage(maskUrl, token);
    })();
  });

  function clearCanvasPixels(): void {
    const ctx = getCtx();
    if (!ctx || !canvasEl) return;
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    ctx.restore();
  }

  async function drawMaskImage(url: string, token: number): Promise<void> {
    const ctx = getCtx();
    if (!ctx || !canvasEl) return;
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      if (token !== loadToken || !canvasEl) return;
      ctx.save();
      ctx.globalCompositeOperation = 'source-over';
      ctx.drawImage(img, 0, 0, canvasEl.width, canvasEl.height);
      ctx.restore();
    } catch {
      // Best-effort: a failed mask fetch/decode should not crash the canvas.
    }
  }

  /** Canvas-pixel-space coordinates for a pointer event (see the module doc). */
  function canvasPoint(event: PointerEvent): { x: number; y: number } | null {
    if (!canvasEl) return null;
    const rect = canvasEl.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvasEl.width,
      y: ((event.clientY - rect.top) / rect.height) * canvasEl.height,
    };
  }

  /** Converts a "CSS pixel" brush size into canvas-space units for the current display scale. */
  function scaleFactor(): number {
    if (!canvasEl) return 1;
    const rect = canvasEl.getBoundingClientRect();
    if (rect.width <= 0) return 1;
    return canvasEl.width / rect.width;
  }

  function beginStroke(event: PointerEvent): void {
    if (!interactiveNow || isBusy()) return;
    const view = projectStore.view;
    if (!view || view.selectedSlice === null) return;
    const ctx = getCtx();
    const point = canvasPoint(event);
    if (!ctx || !canvasEl || !point) return;

    // Guarded: not every test/browser environment implements pointer
    // capture (e.g. jsdom), and losing it is not fatal here since drawing
    // itself only depends on the pointermove/pointerup listeners below.
    canvasEl.setPointerCapture?.(event.pointerId);
    isDrawing = true;
    strokeDirty = false;
    lastPoint = point;

    ctx.globalCompositeOperation = isErasing ? 'destination-out' : 'source-over';
    ctx.strokeStyle = 'rgba(255, 0, 0, 1)';
    ctx.lineWidth = (isErasing ? eraseWidth : drawWidth) * scaleFactor();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // A plain click (no move) should still leave a dot.
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    ctx.lineTo(point.x + 0.01, point.y + 0.01);
    ctx.stroke();
    strokeDirty = true;
  }

  function moveStroke(event: PointerEvent): void {
    if (!isDrawing) return;
    const ctx = getCtx();
    const point = canvasPoint(event);
    if (!ctx || !point || !lastPoint) return;
    ctx.beginPath();
    ctx.moveTo(lastPoint.x, lastPoint.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    lastPoint = point;
    strokeDirty = true;
  }

  function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob | null> {
    return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), 'image/png'));
  }

  async function saveCurrentCanvas(): Promise<void> {
    const view = projectStore.view;
    if (!view || view.selectedSlice === null || !canvasEl) return;
    const index = view.selectedSlice as number;
    const blob = await canvasToBlob(canvasEl);
    if (!blob) return;
    const savePromise = workflow.saveMask(index, blob);
    canvasSaveStore.register(savePromise);
    await savePromise;
  }

  async function endStroke(event: PointerEvent): Promise<void> {
    if (!isDrawing) return;
    isDrawing = false;
    try {
      canvasEl?.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer capture may already have been released (e.g. pointercancel).
    }
    lastPoint = null;
    if (!strokeDirty) return;
    strokeDirty = false;
    await saveCurrentCanvas();
  }

  async function onClear(): Promise<void> {
    if (isBusy()) return;
    const view = projectStore.view;
    clearCanvasPixels();
    if (!view || view.selectedSlice === null) return;
    const index = view.selectedSlice as number;
    await canvasSaveStore.flush();
    const clearPromise = workflow.deleteMask(index);
    canvasSaveStore.register(clearPromise);
    await clearPromise;
  }

  async function onLoad(): Promise<void> {
    if (isBusy()) return;
    const view = projectStore.view;
    if (!view || view.selectedSlice === null) return;
    const slice = view.slices.find((s) => s.index === view.selectedSlice);
    if (!slice?.mask) {
      logStore.pushClient('No mask to load', 'info');
      return;
    }
    await canvasSaveStore.flush();
    await drawMaskImage(slice.mask.url, loadToken);
  }

  function toggleErase(): void {
    isErasing = !isErasing;
  }
</script>

<canvas
  bind:this={canvasEl}
  data-testid="mask-canvas"
  class="mask-canvas"
  class:interactive={interactiveNow}
  onpointerdown={beginStroke}
  onpointermove={moveStroke}
  onpointerup={(event) => void endStroke(event)}
  onpointercancel={(event) => void endStroke(event)}
  oncontextmenu={(event) => event.preventDefault()}
></canvas>

{#if interactiveNow}
  <div class="canvas-tools" data-testid="canvas-tools">
    <button type="button" class="tool-btn" data-testid="canvas-clear" disabled={isBusy()} onclick={onClear}>
      Clear
    </button>
    <button
      type="button"
      class="tool-btn"
      class:tool-btn-selected={isErasing}
      data-testid="canvas-erase-mode"
      aria-pressed={isErasing}
      disabled={isBusy()}
      onclick={toggleErase}
    >
      Erase
    </button>
    <button type="button" class="tool-btn" data-testid="canvas-load" disabled={isBusy()} onclick={onLoad}>
      Load
    </button>
    <label class="brush-size">
      Brush
      <input
        type="range"
        min="5"
        max="100"
        step="1"
        data-testid="brush-size"
        value={isErasing ? eraseWidth : drawWidth}
        oninput={(event) => {
          const value = Number((event.currentTarget as HTMLInputElement).value);
          if (isErasing) eraseWidth = value;
          else drawWidth = value;
        }}
      />
    </label>
  </div>
{/if}

<style>
  .mask-canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    opacity: 0.5;
    pointer-events: none;
    touch-action: none;
    z-index: 5;
  }

  .mask-canvas.interactive {
    pointer-events: auto;
    cursor: crosshair;
  }

  .canvas-tools {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) 0 0;
  }

  .tool-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background-color: var(--color-accent);
    color: var(--color-accent-text);
    border: none;
    border-radius: var(--radius-md);
    padding: var(--space-2);
    font: inherit;
    cursor: pointer;
  }

  .tool-btn:hover:not(:disabled) {
    background-color: var(--color-accent-hover);
  }

  .tool-btn:disabled {
    background-color: var(--color-disabled-bg);
    color: var(--color-disabled-text);
    cursor: default;
  }

  .tool-btn-selected:not(:disabled) {
    background-color: var(--color-success);
    color: var(--color-success-text);
  }

  .brush-size {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }
</style>
