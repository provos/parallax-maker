<script lang="ts">
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import { findPixelFromClick } from '../../geometry';
  import { viewportStore } from '../../state/viewport.svelte';
  import * as workflow from '../../workflow';
  import MaskCanvas from '../canvas/MaskCanvas.svelte';
  import PreviewOverlay from '../canvas/PreviewOverlay.svelte';

  let fileInput: HTMLInputElement | undefined;
  let dragging = $state(false);
  let dropZoneEl: HTMLDivElement | undefined;

  function pickFile(): void {
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

  function onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      pickFile();
    }
  }

  // The main <img> shows the current display image: a segmentation preview,
  // a selected slice's composite, or (before any of those) the input image
  // itself (see ProjectView.mainImage in ARCHITECTURE.md), falling back to
  // the raw input asset only if the server hasn't reported a main image yet.
  const mainImageUrl = $derived(
    projectStore.view?.mainImage?.url ?? projectStore.view?.assets.input?.url,
  );
  const hasInputImage = $derived(!!projectStore.view?.assets.input);

  /**
   * Handles a real click on the main image for segmentation (webui.py's
   * `click_event`). Only runs when a project with an input image exists and
   * nothing else is in flight; a click landing outside the image (which
   * shouldn't happen given the image fills its own box, but can at the
   * exact far edge due to rounding) is ignored, matching the backend's own
   * bounds check.
   *
   * Reads `event.ctrlKey` (not `metaKey`): Playwright's `Control` modifier
   * sets `ctrlKey` on click events in Chromium on both macOS and Linux, and
   * this must match Dash's `click_event`, which also keys off `ctrlKey`.
   *
   * `rect` is read live from `getBoundingClientRect()` at click time, which
   * already reflects the current zoom/pan CSS transform applied to
   * `.image-stack` below -- `findPixelFromClick`'s plain ratio math needs no
   * changes at all to stay pixel-exact under zoom/pan (see geometry.ts's
   * `transformedRect` doc comment and its own zoom/pan unit tests).
   */
  function onImageClick(event: MouseEvent): void {
    if (!hasInputImage) return; // let the click bubble to the drop-zone's pickFile
    event.stopPropagation();
    if (suppressNextClick) {
      // A real drag-to-pan just ended on this same pointer sequence; treat
      // it as a pan, not a segmentation click (see onDropZonePointerMove).
      suppressNextClick = false;
      return;
    }
    if (isBusy()) return;

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
   * click), which is how real trackpad users open a context menu with one
   * mouse button. Dash works around this the same way
   * (utility.js's `suppress_contextmenu`): treat a Ctrl-held `contextmenu`
   * on the image as the click it would have been everywhere else, and let
   * an un-modified right-click open the browser's context menu as normal.
   */
  function onImageContextMenu(event: MouseEvent): void {
    if (!event.ctrlKey) return;
    event.preventDefault();
    onImageClick(event);
  }

  // -- Zoom/pan for the main image + mask canvas (see state/viewport.svelte.ts
  // for why Dash only has wheel-zoom, and why drag-to-pan/reset buttons here
  // are a documented improvement rather than a strict parity port).

  /** `.drop-zone`-relative coordinates, ignoring any current zoom/pan transform -- see viewportStore.zoomAt's own doc. */
  function localPoint(clientX: number, clientY: number): { x: number; y: number } | null {
    if (!dropZoneEl) return null;
    const rect = dropZoneEl.getBoundingClientRect();
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  function onWheel(event: WheelEvent): void {
    if (!hasInputImage) return;
    const point = localPoint(event.clientX, event.clientY);
    if (!point) return;
    event.preventDefault();
    // Matches Dash's handleWheel exactly: deltaY < 0 (scroll up) zooms in.
    viewportStore.zoomAt(point.x, point.y, event.deltaY < 0);
  }

  // Drag-to-pan: the middle mouse button always pans (works on every tab,
  // never ambiguous with anything else); the primary button also pans, but
  // only outside the Inpainting tab, where it instead paints on
  // MaskCanvas.svelte -- and only once the drag has moved far enough to
  // stop looking like a plain click, so a real Playwright `.click()` (zero
  // movement) is never affected. A drag that *does* cross that threshold
  // suppresses the `click` event that would otherwise still fire afterward.
  const PAN_THRESHOLD_PX = 4;
  let panPointerId: number | null = null;
  let panStartX = 0;
  let panStartY = 0;
  let panEngaged = false;
  let panButton = 0;
  let suppressNextClick = false;

  function canLeftDragPan(): boolean {
    return uiStore.mainTab !== 'Inpainting';
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
      // (a middle-button press never generates one at all -- only
      // `auxclick`); setting this unconditionally would leave a stale
      // `true` around forever after a middle-drag pan, silently swallowing
      // the *next*, unrelated real click.
      if (panButton === 0) suppressNextClick = true;
      // Deferred until a real drag is detected (not on pointerdown): a
      // capturing element intercepts the browser's compatibility mouse
      // events too (not just pointer events), which would silently steal
      // the `click` a zero-movement press should still deliver to the
      // <img> beneath -- capturing only once we know this is really a
      // drag keeps a plain click completely unaffected.
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

  function zoomButtonCenter(): { x: number; y: number } {
    const rect = dropZoneEl?.getBoundingClientRect();
    return rect ? { x: rect.width / 2, y: rect.height / 2 } : { x: 0, y: 0 };
  }

  function zoomIn(): void {
    const { x, y } = zoomButtonCenter();
    viewportStore.zoomInAt(x, y);
  }

  function zoomOut(): void {
    const { x, y } = zoomButtonCenter();
    viewportStore.zoomOutAt(x, y);
  }

  function resetZoom(): void {
    viewportStore.reset();
  }

  const segmentation = $derived(projectStore.view?.segmentation);
  // Multi/Commit are only meaningful in Instance Segmentation mode with a
  // project loaded, matching Dash's toggle_segmentation_buttons.
  const segmentationToolsActive = $derived(
    uiStore.segmentationMode === 'segment' && !!projectStore.view,
  );
  const multiPointEnabled = $derived(segmentation?.multiPointMode ?? false);
  const canCommit = $derived(
    segmentationToolsActive &&
      multiPointEnabled &&
      (segmentation?.queuedPoints.length ?? 0) > 0 &&
      !isBusy(),
  );

  function toggleMultiPoint(): void {
    if (!segmentationToolsActive || isBusy()) return;
    void workflow.setMultiPointMode(!multiPointEnabled);
  }

  function commit(): void {
    if (!canCommit) return;
    void workflow.commitMultiPoint();
  }

  // Checkerboard/Invert/Feather are mask tools (components.py's
  // make_segmentation_tools_container); like the rest of that row they are
  // only gated on "a project is loaded and nothing else is in flight" -- Dash
  // never disables these three buttons based on selection/mask state either,
  // it just no-ops with a log message (see workflow.ts's slice-editing
  // section, which owns the exact precondition/wording for Invert/Feather).
  const maskToolsActive = $derived(!!projectStore.view);
  const checkerboardOn = $derived(projectStore.view?.useCheckerboard ?? false);

  function toggleCheckerboard(): void {
    if (!maskToolsActive || isBusy()) return;
    void workflow.toggleCheckerboard();
  }

  function invertMask(): void {
    if (!maskToolsActive || isBusy()) return;
    void workflow.invertMask();
  }

  function featherMask(): void {
    if (!maskToolsActive || isBusy()) return;
    void workflow.featherMask();
  }
</script>

<div class="input-image-outer panel">
  <span class="panel-label">Input Image</span>
  <div
    bind:this={dropZoneEl}
    class="drop-zone panel"
    class:dragging
    role="button"
    tabindex="0"
    aria-label="Upload input image"
    data-testid="input-image-panel"
    onclick={pickFile}
    onkeydown={onKeydown}
    ondrop={onDrop}
    ondragover={onDragOver}
    ondragleave={onDragLeave}
    onwheel={onWheel}
    onpointerdown={onDropZonePointerDown}
    onpointermove={onDropZonePointerMove}
    onpointerup={onDropZonePointerUp}
    onpointercancel={onDropZonePointerUp}
  >
    <!-- Shared box for the main image and the mask canvas overlay (same
         size, `MaskCanvas.svelte`'s own doc comment explains why); the
         wrapper takes on the image's rendered size exactly like the `<img>`
         did on its own before, so lib/geometry.ts's ratio-based pixel math
         above stays correct. `transform` implements zoom/pan
         (state/viewport.svelte.ts); every descendant (image, mask canvas,
         preview overlay) inherits it, and each one's own
         `getBoundingClientRect()` automatically reflects it. -->
    <div
      class="image-stack"
      style={`transform: translate(${viewportStore.panX}px, ${viewportStore.panY}px) scale(${viewportStore.scale}); transform-origin: 0 0;`}
    >
      <!-- Segmentation click target: mirrors Dash's EventListener-wrapped
           <img id="image">, which is likewise mouse-only (no keyboard
           equivalent for "pick a pixel"). -->
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <img
        data-testid="main-image"
        alt=""
        src={mainImageUrl}
        draggable="false"
        onclick={onImageClick}
        oncontextmenu={onImageContextMenu}
        ondragstart={(event) => event.preventDefault()}
      />
      <MaskCanvas />
      <PreviewOverlay />
    </div>
    <input
      bind:this={fileInput}
      type="file"
      accept="image/*"
      class="sr-only"
      data-testid="upload-image-input"
      onchange={onInputChange}
    />
  </div>

  <!-- Zoom/pan controls (state/viewport.svelte.ts): buttons zoom about the
       viewport center; the wheel and drag-to-pan gestures live on the
       drop-zone above. See viewport.svelte.ts's doc comment for why Dash
       has no equivalent buttons/pan/reset of its own. -->
  <div class="viewport-controls" data-testid="viewport-controls">
    <button
      type="button"
      class="tool-btn tool-btn-icon"
      data-testid="zoom-out"
      aria-label="Zoom out"
      title="Zoom out"
      disabled={!hasInputImage}
      onclick={zoomOut}
    >
      &minus;
    </button>
    <button
      type="button"
      class="tool-btn tool-btn-icon"
      data-testid="zoom-reset"
      aria-label="Reset zoom and pan"
      title="Reset zoom and pan"
      disabled={!hasInputImage}
      onclick={resetZoom}
    >
      &#x27F3;
    </button>
    <button
      type="button"
      class="tool-btn tool-btn-icon"
      data-testid="zoom-in"
      aria-label="Zoom in"
      title="Zoom in"
      disabled={!hasInputImage}
      onclick={zoomIn}
    >
      &plus;
    </button>
    <span class="zoom-level" data-testid="zoom-level">{Math.round(viewportStore.scale * 100)}%</span>
  </div>

  <!-- Tool row under the Input Image panel, same order as Dash's
       make_segmentation_tools_container: checkerboard, Invert, Feather,
       Multi, Commit. -->
  <div class="tool-row">
    <button
      type="button"
      class="tool-btn tool-btn-icon"
      class:tool-btn-selected={checkerboardOn}
      data-testid="toggle-checkerboard"
      aria-label="Toggle checkerboard background"
      aria-pressed={checkerboardOn}
      title="Toggle checkerboard background"
      disabled={!maskToolsActive || isBusy()}
      onclick={toggleCheckerboard}
    >
      &#x25A6;
    </button>
    <button
      type="button"
      class="tool-btn"
      data-testid="invert-mask"
      title="Invert the current mask"
      disabled={!maskToolsActive || isBusy()}
      onclick={invertMask}
    >
      Invert
    </button>
    <button
      type="button"
      class="tool-btn"
      data-testid="feather-mask"
      title="Feather the current mask"
      disabled={!maskToolsActive || isBusy()}
      onclick={featherMask}
    >
      Feather
    </button>
    <button
      type="button"
      class="tool-btn"
      class:tool-btn-selected={multiPointEnabled}
      data-testid="multi-point"
      aria-pressed={multiPointEnabled}
      disabled={!segmentationToolsActive || isBusy()}
      onclick={toggleMultiPoint}
    >
      Multi
    </button>
    <button
      type="button"
      class="tool-btn"
      data-testid="multi-commit"
      disabled={!canCommit}
      onclick={commit}
    >
      Commit
    </button>
  </div>
</div>

<style>
  .input-image-outer {
    display: flex;
    flex-direction: column;
    min-height: 30rem;
  }

  .drop-zone {
    position: relative;
    flex: 1;
    min-height: 24rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    overflow: hidden;
    /* The wheel/drag-to-pan gestures above handle zoom/pan themselves;
       without this, touch input would also try to scroll/zoom the page. */
    touch-action: none;
  }

  .drop-zone.dragging {
    border-color: var(--color-accent);
  }

  .image-stack {
    position: relative;
    width: 100%;
  }

  /* No letterboxing: the box takes on the image's own aspect ratio, so
     pixel-click math (lib/geometry.ts's findPixelFromClick) can use Dash's
     plain ratio formula without compensating for empty space on an axis. */
  .image-stack img {
    display: block;
    width: 100%;
    height: auto;
  }

  /* Chromium renders a "broken image" glyph for an <img> with layout space
     and no loaded resource, even with no `src` attribute at all. Hide it
     until there is something to show, matching Dash's empty panel look. */
  .image-stack img:not([src]) {
    visibility: hidden;
  }

  .tool-row {
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

  .tool-btn-icon {
    padding-left: var(--space-3);
    padding-right: var(--space-3);
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

  .viewport-controls {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) 0 0;
  }

  .zoom-level {
    font-size: 0.75rem;
    color: var(--color-text-muted);
    min-width: 3rem;
  }
</style>
