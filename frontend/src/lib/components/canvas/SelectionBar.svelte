<script lang="ts">
  /**
   * Floats under the image while a Segment Anything selection exists
   * (docs/redesign/HANDOFF.md §5): segment queued points, add the selection
   * to the selected slice or remove it from there, or make it a new slice.
   */
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';

  const view = $derived(projectStore.view);
  const segmentation = $derived(view?.segmentation);
  const hasMask = $derived(segmentation?.hasMask ?? false);
  const queued = $derived(segmentation?.multiPointMode ? (segmentation?.queuedPoints.length ?? 0) : 0);
  const selected = $derived(view?.slices.find((s) => s.index === view.selectedSlice) ?? null);
  const selectedName = $derived(selected ? `image_slice_${selected.index}` : null);
</script>

{#if hasMask || queued > 0}
  <div class="selection-bar" role="toolbar" aria-label="Selection actions" data-testid="selection-bar">
    <span class="summary">{hasMask ? 'Selection' : `${queued} ${queued === 1 ? 'point' : 'points'}`}</span>
    {#if queued > 0}
      <span class="divider" aria-hidden="true"></span>
      <button
        type="button"
        class="btn btn-sm btn-selected"
        data-testid="selection-commit"
        disabled={isBusy()}
        onclick={() => void workflow.commitMultiPoint()}
      >
        Segment {queued} {queued === 1 ? 'point' : 'points'}
      </button>
    {/if}
    {#if hasMask}
      <span class="divider" aria-hidden="true"></span>
      <button
        type="button"
        class="btn btn-sm"
        data-testid="selection-add"
        title="Add the selection to the selected slice"
        disabled={!selected || isBusy()}
        onclick={() => void workflow.addMaskToSlice()}
      >
        Add to {selectedName ?? 'slice'}
      </button>
      <button
        type="button"
        class="btn btn-sm"
        data-testid="selection-remove"
        title="Remove the selection from the selected slice"
        disabled={!selected || isBusy()}
        onclick={() => void workflow.removeMaskFromSlice()}
      >
        Remove from {selectedName ?? 'slice'}
      </button>
      <button
        type="button"
        class="btn btn-sm btn-primary"
        data-testid="selection-new-slice"
        disabled={isBusy()}
        onclick={() => void workflow.createSlice()}
      >
        New slice
      </button>
    {/if}
  </div>
{/if}

<style>
  .selection-bar {
    position: absolute;
    bottom: var(--space-4);
    left: 50%;
    transform: translateX(-50%);
    z-index: 6;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 6px 6px var(--space-3);
    background: var(--color-float);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-float);
    white-space: nowrap;
  }

  .summary {
    font-weight: 600;
  }

  .divider {
    width: 1px;
    height: 22px;
    margin: 0 var(--space-1);
    background: var(--color-border-strong);
  }
</style>
