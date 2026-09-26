<script lang="ts">
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import HelpTooltip from '../shared/HelpTooltip.svelte';
  import { SEGMENTATION_HELP_TEXTS } from '../../helpTexts';

  type ActionButton = { label: string; testId: string; onClick: () => void };

  // Slice-generation actions (Copy, Paste and Delete are in the layer
  // panel's header). None of them are gated on selection or mask state: each
  // one runs and logs a no-op message when a precondition is missing (see
  // workflow.ts's slice-editing section). The only gating is "is there a
  // project, and is nothing else in flight".
  const actions: ActionButton[] = [
    { label: 'Generate', testId: 'generate-slices', onClick: () => void workflow.generateSlices() },
    { label: 'Balance', testId: 'balance-slices', onClick: () => void workflow.balanceSlices() },
    { label: 'Create', testId: 'create-slice', onClick: () => void workflow.createSlice() },
    { label: 'Add', testId: 'add-mask-to-slice', onClick: () => void workflow.addMaskToSlice() },
    { label: 'Remove', testId: 'remove-mask-from-slice', onClick: () => void workflow.removeMaskFromSlice() },
  ];

  // Ground plane: the selected slice becomes (or stops being) the horizontal
  // ground; Fit places the horizon and ground from the scene.
  const selectedIsGround = $derived(
    projectStore.view?.slices.find((s) => s.index === projectStore.view?.selectedSlice)?.isGround ?? false,
  );
  const hasGround = $derived(projectStore.view?.slices.some((s) => s.isGround) ?? false);

  // Interior threshold boundaries only (length numSlices - 1); the backend
  // rejects any other length (see UpdateThresholdValues.update_threshold_values).
  const interiorThresholds = $derived.by((): number[] => {
    const view = projectStore.view;
    if (!view || view.numSlices < 2 || view.thresholds.length !== view.numSlices + 1) return [];
    return view.thresholds.slice(1, view.numSlices);
  });

  // Local editable copy so dragging feels smooth; re-synced whenever the
  // server's thresholds change (including our own applied response).
  let localValues = $state<number[]>([]);
  $effect(() => {
    localValues = [...interiorThresholds];
  });

  function onSliderInput(index: number, event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    localValues = localValues.map((v, i) => (i === index ? value : v));
  }

  function onSliderChange(): void {
    void workflow.updateThresholds([...localValues]);
  }
</script>

<div class="segmentation-tab" data-testid="tab-segmentation">
  <div class="tab-header">
    <HelpTooltip label="Segmentation" texts={SEGMENTATION_HELP_TEXTS} />
  </div>
  <div class="segmentation-grid">
    <div class="thresholds-column">
      <span class="panel-label">Thresholds</span>
      <div class="panel thresholds-box" data-testid="thresholds-container">
        {#each localValues as value, index (index)}
          <div class="threshold-row">
            <input
              type="range"
              min="0"
              max="255"
              step="1"
              data-testid="threshold-handle"
              aria-label={`Threshold ${index + 1}`}
              value={value}
              disabled={isBusy() || !projectStore.view}
              oninput={(event) => onSliderInput(index, event)}
              onchange={onSliderChange}
            />
            <!-- Always-visible current value, matching Dash's
                 `tooltip={"always_visible": True}` on every threshold slider
                 (webui.py's `update_thresholds_html`). -->
            <span class="slider-value" data-testid="threshold-value">{value}</span>
          </div>
        {/each}
      </div>
    </div>

    <div class="actions-column">
      <span class="panel-label">Actions</span>
      <div class="actions-grid">
        {#each actions as action (action.label)}
          <button
            type="button"
            class="btn"
            data-testid={action.testId}
            disabled={isBusy() || !projectStore.view}
            onclick={action.onClick}
          >
            {action.label}
          </button>
        {/each}
        <button
          type="button"
          class="btn"
          class:btn-selected={selectedIsGround}
          data-testid="ground-toggle"
          title="Make the selected slice the horizontal ground plane"
          aria-pressed={selectedIsGround}
          disabled={isBusy() || !projectStore.view || projectStore.view.selectedSlice === null}
          onclick={() => void workflow.toggleGroundPlane()}
        >
          Ground
        </button>
        <button
          type="button"
          class="btn"
          data-testid="ground-fit"
          title="Put the horizon on the ground's top edge and the ground under the nearest object"
          disabled={isBusy() || !hasGround}
          onclick={() => void workflow.fitGround()}
        >
          Fit ground
        </button>
      </div>
    </div>
  </div>
</div>

<style>
  .tab-header {
    display: flex;
    justify-content: flex-end;
    padding: var(--space-2) var(--space-2) 0;
  }

  .segmentation-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: var(--space-4);
    padding: var(--space-2);
  }

  .thresholds-box {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-height: 2rem;
  }

  .thresholds-box input[type='range'] {
    width: 100%;
  }

  .threshold-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  .threshold-row input[type='range'] {
    flex: 1;
  }

  .slider-value {
    font-size: 0.75rem;
    color: var(--color-text-muted);
    min-width: 2rem;
    text-align: right;
  }

  .actions-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-2);
  }
</style>
