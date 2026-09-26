<script lang="ts">
  /**
   * The Brush tool's options (Clear, eraser toggle, Load, brush size), shown
   * in the tool options bar while the Brush tool is active -- the mask
   * canvas is only interactive then (MaskCanvas.svelte).
   */
  import { isBusy } from '../../state/busy.svelte';
  import { maskToolsStore, BRUSH_MAX, BRUSH_MIN } from '../../state/maskTools.svelte';
</script>

<div class="mask-toolbar" data-testid="canvas-tools">
  <button
    type="button"
    class="tool-btn"
    data-testid="canvas-clear"
    title="Clear the painted mask"
    disabled={isBusy()}
    onclick={() => void maskToolsStore.clear()}
  >
    Clear
  </button>
  <button
    type="button"
    class="tool-btn"
    class:tool-btn-selected={maskToolsStore.erasing}
    data-testid="canvas-erase-mode"
    title="Toggle the eraser brush"
    aria-pressed={maskToolsStore.erasing}
    disabled={isBusy()}
    onclick={() => maskToolsStore.toggleErasing()}
  >
    Erase
  </button>
  <button
    type="button"
    class="tool-btn"
    data-testid="canvas-load"
    title="Load the slice's saved mask onto the canvas"
    disabled={isBusy()}
    onclick={() => void maskToolsStore.load()}
  >
    Load
  </button>
  <label class="brush-size">
    Brush
    <input
      type="range"
      min={BRUSH_MIN}
      max={BRUSH_MAX}
      step="1"
      data-testid="brush-size"
      value={maskToolsStore.brushWidth}
      oninput={(event) => maskToolsStore.setBrushWidth(Number((event.currentTarget as HTMLInputElement).value))}
    />
  </label>
</div>

<style>
  .mask-toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }

  .brush-size {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }
</style>
