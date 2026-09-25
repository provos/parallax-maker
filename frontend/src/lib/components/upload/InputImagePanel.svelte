<script lang="ts">
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import { findPixelFromClick } from '../../geometry';
  import * as workflow from '../../workflow';

  let fileInput: HTMLInputElement | undefined;
  let dragging = $state(false);

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
   */
  function onImageClick(event: MouseEvent): void {
    if (!hasInputImage) return; // let the click bubble to the drop-zone's pickFile
    event.stopPropagation();
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
</script>

<div class="input-image-outer panel">
  <span class="panel-label">Input Image</span>
  <div
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
      onclick={onImageClick}
      oncontextmenu={onImageContextMenu}
    />
    <input
      bind:this={fileInput}
      type="file"
      accept="image/*"
      class="sr-only"
      data-testid="upload-image-input"
      onchange={onInputChange}
    />
  </div>

  <!-- Tool row under the Input Image panel, same order as Dash's
       make_segmentation_tools_container: checkerboard, Invert, Feather,
       Multi, Commit. Checkerboard/Invert/Feather are mask tools that land
       in a later PR, so they stay disabled here. -->
  <div class="tool-row">
    <button
      type="button"
      class="tool-btn tool-btn-icon"
      data-testid="toggle-checkerboard"
      aria-label="Toggle checkerboard background"
      title="Not available yet"
      disabled
    >
      &#x25A6;
    </button>
    <button type="button" class="tool-btn" data-testid="invert-mask" title="Not available yet" disabled>
      Invert
    </button>
    <button type="button" class="tool-btn" data-testid="feather-mask" title="Not available yet" disabled>
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
  }

  .drop-zone.dragging {
    border-color: var(--color-accent);
  }

  /* No letterboxing: the box takes on the image's own aspect ratio, so
     pixel-click math (lib/geometry.ts's findPixelFromClick) can use Dash's
     plain ratio formula without compensating for empty space on an axis. */
  .drop-zone img {
    width: 100%;
    height: auto;
  }

  /* Chromium renders a "broken image" glyph for an <img> with layout space
     and no loaded resource, even with no `src` attribute at all. Hide it
     until there is something to show, matching Dash's empty panel look. */
  .drop-zone img:not([src]) {
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
</style>
