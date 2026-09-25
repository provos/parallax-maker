<script lang="ts">
  /**
   * Non-interactive overlay drawn on top of the main image + mask canvas,
   * inside the same zoom/pan-transformed `.image-stack` box
   * (InputImagePanel.svelte) so every marker tracks zoom/pan exactly like
   * the image and mask canvas beneath it (see state/viewport.svelte.ts).
   *
   * Three things live here, all positioned in source-image-pixel percentage
   * coordinates so they need no re-computation on resize:
   *  - Queued multi-point segmentation markers (Dash's CLI-05
   *    `visualize_point`: green for a plain/Shift point, red for a
   *    Ctrl/negative one), driven directly by
   *    `ProjectView.segmentation.queuedPoints`.
   *  - The brush-size preview circle that follows the pointer while painting
   *    is idle (Dash's JS-01 `previewBrush`), driven by
   *    `canvasPreviewStore.brush` (set by MaskCanvas.svelte).
   *  - The region-of-interest bounding-box preview shown for ~2s after a
   *    mask save with "Crop to region of interest" checked (Dash's CLI-07
   *    `show_bounding_box`), driven by `canvasPreviewStore.roiBox`.
   *
   * Every marker is counter-scaled by `1 / viewportStore.scale` so its
   * *visual* size on screen stays constant across zoom levels -- Dash's
   * literal behavior instead bakes these onto a canvas that itself gets the
   * same CSS `transform: scale()` as the image (so they visually grow with
   * zoom too), but a marker that grows to dozens of CSS pixels at 8x zoom is
   * actively unhelpful; keeping a constant on-screen size is a deliberate,
   * documented improvement (see PARITY.md).
   */
  import { projectStore } from '../../state/project.svelte';
  import { viewportStore } from '../../state/viewport.svelte';
  import { canvasPreviewStore } from '../../state/canvasPreview.svelte';

  const size = $derived(projectStore.view?.image);
  const queuedPoints = $derived(projectStore.view?.segmentation.queuedPoints ?? []);
  const brush = $derived(canvasPreviewStore.brush);
  const roiBox = $derived(canvasPreviewStore.roiBox);
  const inverseScale = $derived(1 / viewportStore.scale);

  function pct(value: number, extent: number): number {
    return extent > 0 ? (value / extent) * 100 : 0;
  }
</script>

{#if size}
  <div class="preview-overlay" data-testid="preview-overlay" aria-hidden="true">
    {#if roiBox}
      {@const [x0, y0, x1, y1] = roiBox}
      <div
        class="roi-box"
        data-testid="roi-box"
        style={`left:${pct(x0, size.width)}%; top:${pct(y0, size.height)}%;
                width:${pct(x1 - x0, size.width)}%; height:${pct(y1 - y0, size.height)}%;
                border-width:${2 * inverseScale}px;`}
      ></div>
    {/if}

    {#each queuedPoints as point, index (index)}
      <div
        class="marker"
        class:negative={point.negative}
        data-testid="queued-point-marker"
        style={`left:${pct(point.x, size.width)}%; top:${pct(point.y, size.height)}%;
                transform: translate(-50%, -50%) scale(${inverseScale});`}
      ></div>
    {/each}

    {#if brush}
      <div
        class="brush-preview"
        class:erasing={brush.erasing}
        data-testid="brush-preview"
        style={`left:${pct(brush.x, size.width)}%; top:${pct(brush.y, size.height)}%;
                width:${pct(brush.diameter, size.width)}%; height:${pct(brush.diameter, size.height)}%;
                transform: translate(-50%, -50%);`}
      ></div>
    {/if}
  </div>
{/if}

<style>
  .preview-overlay {
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: 15;
  }

  .marker {
    position: absolute;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    /* Positive point: green, matches Dash's `rgba(0, 255, 0, 1)`. */
    background-color: rgb(0, 200, 0);
    box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.6);
  }

  /* Negative (Ctrl-click) point: red, matches Dash's `rgba(255, 0, 0, 1)`. */
  .marker.negative {
    background-color: rgb(220, 0, 0);
  }

  .roi-box {
    position: absolute;
    border-style: solid;
    border-color: rgb(255, 128, 0);
    box-sizing: border-box;
  }

  .brush-preview {
    position: absolute;
    border-radius: 50%;
    border: 2px solid rgba(255, 0, 0, 0.6);
    box-sizing: border-box;
  }

  .brush-preview.erasing {
    border-color: rgba(0, 0, 0, 0.6);
  }
</style>
