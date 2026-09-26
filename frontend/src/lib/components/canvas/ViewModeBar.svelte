<script lang="ts">
  /**
   * Above the canvas (docs/redesign/HANDOFF.md §3): what the canvas shows
   * (Input, Depth, Slice, Composite, Parallax 2D, 3D), the checkerboard
   * toggle, and zoom.
   */
  import Minus from '@lucide/svelte/icons/minus';
  import Plus from '@lucide/svelte/icons/plus';
  import Grid2x2 from '@lucide/svelte/icons/grid-2x2';
  import { uiStore, CANVAS_VIEWS } from '../../state/ui.svelte';
  import { projectStore } from '../../state/project.svelte';
  import { viewportStore } from '../../state/viewport.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';

  let { onZoomIn, onZoomOut }: { onZoomIn: () => void; onZoomOut: () => void } = $props();

  const view = $derived(projectStore.view);
  const hasImage = $derived(!!view?.assets.input);
  const hasSlices = $derived((view?.slices.length ?? 0) > 0);
  const available = $derived.by(() => ({
    input: hasImage,
    depth: !!view?.assets.depth,
    slice: view?.selectedSlice != null,
    composite: hasSlices,
    parallax: hasSlices,
    '3d': hasSlices,
  }));
  const checkerboardOn = $derived(view?.useCheckerboard ?? false);
</script>

<div class="view-mode-bar" data-testid="view-mode-bar">
  <div class="seg" role="tablist" aria-label="View mode">
    {#each CANVAS_VIEWS as { view: mode, label, key } (mode)}
      <button
        type="button"
        role="tab"
        data-testid={`view-${mode}`}
        aria-selected={uiStore.view === mode}
        title={`${label} (${key})`}
        disabled={!available[mode]}
        onclick={() => uiStore.setView(mode)}
      >
        {label}
      </button>
    {/each}
  </div>
  <div class="controls">
    <button
      type="button"
      class="btn btn-ghost btn-sm checker-toggle"
      class:btn-selected={checkerboardOn}
      data-testid="toggle-checkerboard"
      aria-pressed={checkerboardOn}
      title="Show the selected slice over a checkerboard"
      disabled={!view || isBusy()}
      onclick={() => void workflow.toggleCheckerboard()}
    >
      <Grid2x2 size={14} strokeWidth={1.6} />
      <span class="checker-label">Checkerboard</span>
    </button>
    <button
      type="button"
      class="btn btn-ghost btn-sm btn-icon"
      data-testid="zoom-out"
      aria-label="Zoom out"
      title="Zoom out"
      disabled={!hasImage}
      onclick={onZoomOut}
    >
      <Minus size={14} strokeWidth={1.6} />
    </button>
    <span class="zoom-level mono" data-testid="zoom-level">{Math.round(viewportStore.scale * 100)}%</span>
    <button
      type="button"
      class="btn btn-ghost btn-sm btn-icon"
      data-testid="zoom-in"
      aria-label="Zoom in"
      title="Zoom in"
      disabled={!hasImage}
      onclick={onZoomIn}
    >
      <Plus size={14} strokeWidth={1.6} />
    </button>
    <button
      type="button"
      class="btn btn-ghost btn-sm"
      data-testid="zoom-reset"
      aria-label="Reset zoom and pan"
      title="Fit the image"
      disabled={!hasImage}
      onclick={() => viewportStore.reset()}
    >
      Fit
    </button>
  </div>
</div>

<style>
  .view-mode-bar {
    height: var(--viewbar-h);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    padding: 0 10px;
    background: var(--color-surface);
    border-bottom: 1px solid var(--color-border);
    min-width: 0;
  }

  .seg {
    display: inline-flex;
    background: var(--color-surface-raised);
    border-radius: 7px;
    padding: 2px;
    gap: 2px;
    border: 1px solid var(--color-border);
    min-width: 0;
    overflow: hidden;
  }

  .seg button {
    height: var(--control-h-sm);
    padding: 0 10px;
    border: none;
    border-radius: 5px;
    background: transparent;
    color: var(--color-text-secondary);
    font: inherit;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
  }

  .seg button[aria-selected='true'] {
    background: var(--color-surface-hover);
    color: var(--color-text);
    box-shadow: 0 0 0 1px var(--color-border-strong);
  }

  .seg button:disabled {
    opacity: 0.42;
    cursor: default;
  }

  .seg button:focus-visible {
    outline: 2px solid var(--color-selection);
    outline-offset: 1px;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    color: var(--color-text-secondary);
    flex-shrink: 0;
  }

  .checker-toggle {
    margin-right: var(--space-2);
  }

  .zoom-level {
    min-width: 40px;
    text-align: center;
    font-size: 12px;
  }

  @media (max-width: 1360px) {
    .checker-label {
      display: none;
    }
  }
</style>
