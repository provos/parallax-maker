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
  import { canvasPreviewStore } from '../../state/canvasPreview.svelte';
  import { maskToolsStore } from '../../state/maskTools.svelte';
  import * as workflow from '../../workflow';

  let canvasEl: HTMLCanvasElement | undefined;
  let ctx2d: CanvasRenderingContext2D | null = null;

  // Tool state lives in maskToolsStore so the toolbar can sit in the tool
  // rows under the image (MaskToolbar.svelte) instead of inside the zoomed
  // image box this canvas overlays.
  const isErasing = $derived(maskToolsStore.erasing);
  const brushWidth = $derived(maskToolsStore.brushWidth);

  let isDrawing = false;
  let strokeDirty = false;
  let lastPoint: { x: number; y: number } | null = null;

  // Alt+Right-drag brush resize (Dash's `startDrawing`/`adjustBrushSize`,
  // utility.js:126-149): holding Alt while dragging with the *right* mouse
  // button adjusts the active brush's width instead of painting, by 1 unit
  // per 15 CSS px of horizontal drag, clamped to [5, 100] -- same formula
  // and same clamp range as Dash, so a given drag distance resizes the
  // brush by the same amount on both UIs.
  let isResizingBrush = false;
  let resizeStartClientX = 0;
  let resizeStartWidth = 0;

  // Bumped on every slice-selection transition so a slow, now-superseded
  // mask fetch (Load, or the auto-load below) can recognize it is stale and
  // avoid painting onto the wrong slice's canvas.
  let loadToken = 0;
  let lastLoadedKey: string | null | undefined = undefined;

  const interactiveNow = $derived(uiStore.mainTab === 'Inpainting');

  // The brush preview is only ever meaningful while this tab is actually
  // interactive; drop it immediately on a tab switch away rather than
  // leaving a stale circle floating over the (now non-interactive) canvas.
  $effect(() => {
    if (!interactiveNow) canvasPreviewStore.clearBrush();
  });

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

  // Explicit canvas lifecycle for slice-selection and slice-version changes:
  // flush any pending save first, then load the selected slice's own saved
  // mask (or clear, if it has none). A new version (Apply, Erase, Undo, Redo,
  // mask add/remove, paste) carries its own mask - usually none after Apply -
  // so the canvas must follow it or it would show a stroke the backend no
  // longer has. Saving a stroke changes only the mask URL, not the version,
  // so painting never reloads the canvas under the user. A `loadToken`
  // guards against a stale async load finishing after another transition
  // has already started.
  $effect(() => {
    const view = projectStore.view;
    const index = view?.selectedSlice ?? null;
    const slice = index !== null ? view?.slices.find((s) => s.index === index) : undefined;
    const key = index === null ? null : `${index}:${slice?.version ?? ''}`;
    if (key === lastLoadedKey) return;
    lastLoadedKey = key;
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

    // Alt+Right-drag brush resize takes precedence over painting, exactly
    // like Dash's `startDrawing`'s own `if (e.button === 2 && e.altKey)`
    // early return -- a plain (non-Alt) right-click still paints, matching
    // Dash's real (unguarded) behavior for that case.
    if (event.button === 2 && event.altKey) {
      isResizingBrush = true;
      resizeStartClientX = event.clientX;
      resizeStartWidth = brushWidth;
      canvasEl?.setPointerCapture?.(event.pointerId);
      return;
    }

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
    canvasPreviewStore.clearBrush();

    ctx.globalCompositeOperation = isErasing ? 'destination-out' : 'source-over';
    ctx.strokeStyle = 'rgba(255, 0, 0, 1)';
    ctx.lineWidth = brushWidth * scaleFactor();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // A plain click (no move) should still leave a dot.
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    ctx.lineTo(point.x + 0.01, point.y + 0.01);
    ctx.stroke();
    strokeDirty = true;
  }

  /** Clamped exactly like Dash's `adjustBrushSize` (utility.js:139-149): `[5, 100]`, 1 unit per 15px of drag. */
  function adjustBrushSize(deltaX: number): void {
    maskToolsStore.setBrushWidth(resizeStartWidth + deltaX / 15);
  }

  function moveStroke(event: PointerEvent): void {
    if (isResizingBrush) {
      adjustBrushSize(event.clientX - resizeStartClientX);
      updateBrushPreview(event);
      return;
    }
    if (isDrawing) {
      const ctx = getCtx();
      const point = canvasPoint(event);
      if (!ctx || !point || !lastPoint) return;
      ctx.beginPath();
      ctx.moveTo(lastPoint.x, lastPoint.y);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
      lastPoint = point;
      strokeDirty = true;
      return;
    }
    // Idle: show the brush-size preview circle (Dash's JS-01 `previewBrush`).
    updateBrushPreview(event);
  }

  /** Live brush-size preview circle that follows the pointer while idle (Dash's JS-01 `previewBrush`). */
  function updateBrushPreview(event: PointerEvent): void {
    if (!interactiveNow) {
      canvasPreviewStore.clearBrush();
      return;
    }
    const point = canvasPoint(event);
    if (!point) return;
    canvasPreviewStore.setBrush({
      x: point.x,
      y: point.y,
      diameter: brushWidth * scaleFactor(),
      erasing: isErasing,
    });
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
    const cropToRegion = uiStore.cropToRoi;
    const savePromise = workflow.saveMask(index, blob, cropToRegion).then((boundingBox) => {
      if (boundingBox) canvasPreviewStore.showRoiBox(boundingBox);
    });
    canvasSaveStore.register(savePromise);
    await savePromise;
  }

  async function endStroke(event: PointerEvent): Promise<void> {
    if (isResizingBrush) {
      isResizingBrush = false;
      try {
        canvasEl?.releasePointerCapture(event.pointerId);
      } catch {
        // Pointer capture may already have been released (e.g. pointercancel).
      }
      updateBrushPreview(event);
      return;
    }
    if (!isDrawing) return;
    isDrawing = false;
    try {
      canvasEl?.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer capture may already have been released (e.g. pointercancel).
    }
    lastPoint = null;
    updateBrushPreview(event);
    if (!strokeDirty) return;
    strokeDirty = false;
    await saveCurrentCanvas();
  }

  function onPointerLeave(): void {
    canvasPreviewStore.clearBrush();
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

  // Clear/Load operate on this canvas's pixels; expose them to the toolbar.
  $effect(() => {
    const unbind = maskToolsStore.bindCanvas({ clear: onClear, load: onLoad });
    return unbind; // unregister on unmount
  });
</script>

<!-- A stroke ends in a `click`; keep it from reaching the drop zone behind
     the image, whose click opens the upload file chooser. The canvas only
     receives clicks while it is interactive (pointer-events). -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<canvas
  bind:this={canvasEl}
  data-testid="mask-canvas"
  class="mask-canvas"
  class:interactive={interactiveNow}
  onclick={(event) => event.stopPropagation()}
  onpointerdown={beginStroke}
  onpointermove={moveStroke}
  onpointerup={(event) => void endStroke(event)}
  onpointercancel={(event) => void endStroke(event)}
  onpointerleave={onPointerLeave}
  oncontextmenu={(event) => event.preventDefault()}
></canvas>


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
</style>
