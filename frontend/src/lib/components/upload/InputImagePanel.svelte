<script lang="ts">
  /**
   * The canvas stage (docs/redesign/HANDOFF.md §3): the image for the
   * current view with its overlays (mask canvas, points and ROI box,
   * horizon line), zoom/pan, and image upload by drop or, before any image
   * is loaded, from the empty state.
   *
   * Views: Input and Parallax 2D show the server's display image
   * (`ProjectView.mainImage`: the input, a segmentation preview, the
   * selected slice's highlight, or the latest camera render). Depth, Slice
   * and Composite are drawn here from the project's assets.
   */
  import ImageUp from '@lucide/svelte/icons/image-up';
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import { findPixelFromClick } from '../../geometry';
  import { viewportStore } from '../../state/viewport.svelte';
  import * as workflow from '../../workflow';
  import MaskCanvas from '../canvas/MaskCanvas.svelte';
  import PreviewOverlay from '../canvas/PreviewOverlay.svelte';
  import HorizonOverlay from '../canvas/HorizonOverlay.svelte';

  let fileInput: HTMLInputElement | undefined;
  let dragging = $state(false);
  let dropZoneEl: HTMLDivElement | undefined;
  let fitEl: HTMLDivElement | undefined;

  export function pickFile(): void {
    if (isBusy()) return;
    fileInput?.click();
  }

  function handleFile(file: File | undefined | null): void {
    if (!file || isBusy()) return;
    void workflow.uploadImage(file, uiStore.depthModel);
  }

  function onInputChange(event: Event): void {
    const target = event.currentTarget as HTMLInputElement;
    handleFile(target.files?.[0]);
    target.value = '';
  }

  function onDrop(event: DragEvent): void {
    event.preventDefault();
    dragging = false;
    handleFile(event.dataTransfer?.files?.[0]);
  }

  function onDragOver(event: DragEvent): void {
    event.preventDefault();
    dragging = true;
  }

  function onDragLeave(): void {
    dragging = false;
  }

  // The display image: a segmentation preview, a selected slice's
  // highlight, a camera render, or (before any of those) the input image
  // itself (see ProjectView.mainImage in ARCHITECTURE.md), falling back to
  // the raw input asset only if the server hasn't reported a main image yet.
  const mainImageUrl = $derived(
    projectStore.view?.mainImage?.url ?? projectStore.view?.assets.input?.url,
  );
  const hasInputImage = $derived(!!projectStore.view?.assets.input);
  // Source aspect ratio; `.image-fit` uses it to fit the whole image inside
  // the drop-zone (never taller or wider than the available space).
  const aspectRatio = $derived.by(() => {
    const size = projectStore.view?.image;
    return size && size.width > 0 && size.height > 0 ? size.width / size.height : null;
  });

  const selectedSlice = $derived(
    projectStore.view?.slices.find((s) => s.index === projectStore.view?.selectedSlice) ?? null,
  );
  // The reference render's order: the ground first, then far to near.
  const compositeLayers = $derived(
    [...(projectStore.view?.slices ?? [])].sort(
      (a, b) => Number(!!b.isGround) - Number(!!a.isGround) || a.depth - b.depth || a.index - b.index,
    ),
  );
  const showsMain = $derived(uiStore.view === 'input' || uiStore.view === 'parallax');

  /**
   * A click on the display image with the Segment tool selects by object
   * or by depth band at that pixel (webui.py's `click_event`). A click
   * landing outside the image (possible at the exact far edge due to
   * rounding) is ignored, matching the backend's own bounds check.
   *
   * Reads `event.ctrlKey` (not `metaKey`): Playwright's `Control` modifier
   * sets `ctrlKey` on click events in Chromium on both macOS and Linux.
   *
   * `rect` is read live from `getBoundingClientRect()` at click time, which
   * already reflects the current zoom/pan CSS transform applied to
   * `.image-stack` below, so `findPixelFromClick`'s ratio math stays
   * pixel-exact under zoom/pan (see geometry.ts's `transformedRect`).
   */
  function onImageClick(event: MouseEvent): void {
    if (!hasInputImage) return;
    event.stopPropagation();
    if (suppressNextClick) {
      // A real drag-to-pan just ended on this same pointer sequence; treat
      // it as a pan, not a segmentation click (see onDropZonePointerMove).
      suppressNextClick = false;
      return;
    }
    if (isBusy() || uiStore.tool !== 'segment' || uiStore.view !== 'input') return;

    const size = projectStore.view?.image;
    if (!size) return;
    // Use the project's image size rather than the element's natural size:
    // the latter reads 0 while a new display image is loading.
    const rect = (event.currentTarget as HTMLImageElement).getBoundingClientRect();
    const point = findPixelFromClick(event.clientX, event.clientY, rect, size.width, size.height);
    if (!point) return;

    const mode = uiStore.segmentationMode === 'segment' ? 'instance' : 'depth';
    void workflow.clickSegmentation(point.x, point.y, mode, event.shiftKey, event.ctrlKey);
  }

  /**
   * Chromium on macOS never fires a `click` event for a Ctrl+left-click --
   * it fires `contextmenu` instead (mousedown -> contextmenu -> mouseup, no
   * click). Treat a Ctrl-held `contextmenu` on the image as the click it
   * would have been everywhere else, and let an un-modified right-click
   * open the browser's context menu as normal.
   */
  function onImageContextMenu(event: MouseEvent): void {
    if (!event.ctrlKey) return;
    event.preventDefault();
    onImageClick(event);
  }

  // -- Zoom/pan for the image and its overlays (state/viewport.svelte.ts).

  /**
   * Coordinates relative to `.image-fit` -- the untransformed box the image
   * is fitted into, i.e. the zoom/pan transform's own origin -- ignoring any
   * current zoom/pan transform; see viewportStore.zoomAt's own doc.
   */
  function localPoint(clientX: number, clientY: number): { x: number; y: number } | null {
    if (!fitEl) return null;
    const rect = fitEl.getBoundingClientRect();
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  function onWheel(event: WheelEvent): void {
    if (!hasInputImage) return;
    const point = localPoint(event.clientX, event.clientY);
    if (!point) return;
    event.preventDefault();
    // deltaY < 0 (scroll up) zooms in.
    viewportStore.zoomAt(point.x, point.y, event.deltaY < 0);
  }

  // Drag-to-pan: the middle mouse button always pans; the primary button
  // pans too, except with the Brush (it paints on MaskCanvas.svelte) or
  // the Horizon tool (it drags the line) -- and only once the drag has
  // moved far enough to stop looking like a plain click, so a zero-movement
  // `.click()` is never affected. A drag that *does* cross that threshold
  // suppresses the `click` event that would otherwise still fire afterward.
  const PAN_THRESHOLD_PX = 4;
  let panPointerId: number | null = null;
  let panStartX = 0;
  let panStartY = 0;
  let panEngaged = false;
  let panButton = 0;
  let suppressNextClick = false;

  function canLeftDragPan(): boolean {
    return uiStore.tool !== 'brush' && uiStore.tool !== 'horizon';
  }

  function onDropZonePointerDown(event: PointerEvent): void {
    if (!hasInputImage) return;
    const isMiddle = event.button === 1;
    const isPrimary = event.button === 0;
    if (!isMiddle && !(isPrimary && canLeftDragPan())) return;

    panPointerId = event.pointerId;
    panStartX = event.clientX;
    panStartY = event.clientY;
    panEngaged = false;
    panButton = event.button;
    if (isMiddle) {
      // A middle-button press never generates a `click` DOM event at all
      // (only `auxclick`), so capturing immediately here can never hijack
      // the image's segmentation-click handler the way capturing on a
      // primary-button press would -- see the deferred capture in
      // onDropZonePointerMove below for why that one waits.
      dropZoneEl?.setPointerCapture?.(event.pointerId);
      event.preventDefault(); // suppress the OS's middle-click autoscroll cursor
    }
  }

  function onDropZonePointerMove(event: PointerEvent): void {
    if (panPointerId === null || event.pointerId !== panPointerId) return;
    const dx = event.clientX - panStartX;
    const dy = event.clientY - panStartY;
    if (!panEngaged) {
      if (Math.hypot(dx, dy) < PAN_THRESHOLD_PX) return;
      panEngaged = true;
      // Only a *primary*-button drag has a `click` to suppress afterward
      // (a middle-button press never generates one at all).
      if (panButton === 0) suppressNextClick = true;
      // Deferred until a real drag is detected (not on pointerdown): a
      // capturing element intercepts the browser's compatibility mouse
      // events too, which would steal the `click` a zero-movement press
      // should still deliver to the <img> beneath.
      dropZoneEl?.setPointerCapture?.(event.pointerId);
    }
    viewportStore.panBy(event.movementX, event.movementY);
  }

  function onDropZonePointerUp(event: PointerEvent): void {
    if (panPointerId === null || event.pointerId !== panPointerId) return;
    try {
      dropZoneEl?.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer capture may already have been released.
    }
    panPointerId = null;
    panEngaged = false;
  }

  /** The visible stage's center, in `.image-fit`-relative coordinates. */
  function stageCenter(): { x: number; y: number } {
    const rect = dropZoneEl?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return localPoint(rect.left + rect.width / 2, rect.top + rect.height / 2) ?? { x: 0, y: 0 };
  }

  export function zoomIn(): void {
    const { x, y } = stageCenter();
    viewportStore.zoomInAt(x, y);
  }

  export function zoomOut(): void {
    const { x, y } = stageCenter();
    viewportStore.zoomOutAt(x, y);
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  bind:this={dropZoneEl}
  class="stage"
  class:dragging
  class:pannable={hasInputImage && uiStore.tool === 'pan'}
  data-testid="input-image-panel"
  ondrop={onDrop}
  ondragover={onDragOver}
  ondragleave={onDragLeave}
  onwheel={onWheel}
  onpointerdown={onDropZonePointerDown}
  onpointermove={onDropZonePointerMove}
  onpointerup={onDropZonePointerUp}
  onpointercancel={onDropZonePointerUp}
>
  <!-- `.image-fit` is the untransformed box the image is fitted into
       (aspect ratio preserved, as large as the stage allows); zoom/pan
       coordinates are measured relative to it (see localPoint). The
       `.image-stack` inside carries the zoom/pan transform, which every
       layer and overlay inherits (and each one's own
       `getBoundingClientRect()` reflects), so lib/geometry.ts's ratio-based
       pixel math stays correct. -->
  <div
    bind:this={fitEl}
    class="image-fit"
    class:fitted={aspectRatio !== null}
    class:hidden={!hasInputImage}
    style={aspectRatio !== null ? `--image-aspect: ${aspectRatio};` : undefined}
  >
    <div
      class="image-stack"
      data-testid="canvas-image"
      class:checker={uiStore.view === 'slice' || uiStore.view === 'composite'}
      style={`transform: translate(${viewportStore.panX}px, ${viewportStore.panY}px) scale(${viewportStore.scale}); transform-origin: 0 0;`}
    >
      <!-- Segmentation click target; mouse-only (no keyboard equivalent
           for "pick a pixel"). -->
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <img
        class="layer"
        class:hidden={!showsMain}
        class:segmenting={uiStore.tool === 'segment'}
        data-testid="main-image"
        alt=""
        src={mainImageUrl}
        draggable="false"
        onclick={onImageClick}
        oncontextmenu={onImageContextMenu}
        ondragstart={(event) => event.preventDefault()}
      />
      {#if uiStore.view === 'depth' && projectStore.view?.assets.depth}
        <img class="layer" data-testid="view-depth-image" alt="Depth map" src={projectStore.view.assets.depth.url} draggable="false" />
      {:else if uiStore.view === 'slice'}
        {#if selectedSlice}
          <img class="layer" data-testid="view-slice-image" alt={`image_slice_${selectedSlice.index}`} src={selectedSlice.image.url} draggable="false" />
        {:else}
          <div class="layer placeholder" data-testid="view-slice-empty">Select a layer to see it on its own.</div>
        {/if}
      {:else if uiStore.view === 'composite'}
        <div class="layer" data-testid="view-composite">
          {#each compositeLayers as slice (slice.index)}
            <img class="layer" alt="" src={slice.image.url} draggable="false" />
          {/each}
        </div>
      {/if}
      <MaskCanvas />
      <PreviewOverlay />
      {#if uiStore.tool === 'horizon'}
        <HorizonOverlay />
      {/if}
    </div>
  </div>

  {#if !hasInputImage}
    <div class="empty-state" data-testid="empty-state">
      <ImageUp size={32} strokeWidth={1.4} />
      <div class="empty-title">Drop an image here</div>
      <div class="empty-text">A photo with clear foreground and background works best.</div>
      <button type="button" class="btn btn-primary" data-testid="choose-image" disabled={isBusy()} onclick={pickFile}>
        Choose image…
      </button>
      <button type="button" class="btn btn-ghost btn-sm" data-testid="empty-load-project" onclick={() => uiStore.setMainTab('Configuration')}>
        or load a saved project…
      </button>
    </div>
  {/if}

  <input
    bind:this={fileInput}
    type="file"
    accept="image/*"
    class="sr-only"
    data-testid="upload-image-input"
    onchange={onInputChange}
  />
</div>

<style>
  /* Fills the canvas area; the image is fitted inside it, so a tall image
     can never push the page past the viewport. */
  .stage {
    position: relative;
    height: 100%;
    min-height: 8rem;
    /* Size container: `.image-fit` below sizes itself in cqw/cqh. */
    container-type: size;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    /* The wheel/drag-to-pan gestures above handle zoom/pan themselves;
       without this, touch input would also try to scroll/zoom the page. */
    touch-action: none;
    background: var(--color-canvas);
  }

  .stage.pannable {
    cursor: grab;
  }

  .stage.dragging {
    outline: 2px dashed var(--color-selection);
    outline-offset: -8px;
  }

  .image-fit {
    position: relative;
    width: 100%;
    height: 100%;
  }

  /* Contain-fit without letterboxing inside the element itself: the box
     takes on the image's own aspect ratio at the largest size that fits
     the stage, so pixel-click math (lib/geometry.ts's findPixelFromClick)
     can use the plain ratio formula without compensating for empty space
     on an axis. */
  .image-fit.fitted {
    width: min(100cqw, calc(100cqh * var(--image-aspect)));
    height: auto;
    aspect-ratio: var(--image-aspect);
    flex: none;
  }

  .image-stack {
    position: relative;
    width: 100%;
    height: 100%;
  }

  .image-stack.checker {
    background-color: var(--color-checker-1);
    background-image:
      linear-gradient(45deg, var(--color-checker-2) 25%, transparent 25%, transparent 75%, var(--color-checker-2) 75%),
      linear-gradient(45deg, var(--color-checker-2) 25%, transparent 25%, transparent 75%, var(--color-checker-2) 75%);
    background-size: 16px 16px;
    background-position:
      0 0,
      8px 8px;
  }

  .layer {
    position: absolute;
    inset: 0;
    display: block;
    width: 100%;
    height: 100%;
  }

  img.segmenting {
    cursor: crosshair;
  }

  /* Chromium renders a "broken image" glyph for an <img> with layout space
     and no loaded resource, even with no `src` attribute at all. */
  .image-stack img:not([src]) {
    visibility: hidden;
  }

  .placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-text-secondary);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-3);
    color: var(--color-text-secondary);
    text-align: center;
  }

  .empty-state :global(svg) {
    color: var(--color-text-muted);
  }

  .empty-title {
    font-size: var(--text-title);
    font-weight: 600;
    color: var(--color-text);
  }
</style>
