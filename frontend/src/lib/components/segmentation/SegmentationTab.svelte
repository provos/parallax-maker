<script lang="ts">
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';

  type ActionButton = { label: string; testId?: string; onClick?: () => void };

  // Same 8 buttons as components.py's make_slice_generation_container Actions
  // panel; only Generate is wired up in this PR.
  const actions: ActionButton[] = [
    { label: 'Generate', testId: 'generate-slices', onClick: () => void workflow.generateSlices() },
    { label: 'Balance' },
    { label: 'Create' },
    { label: 'Delete' },
    { label: 'Add' },
    { label: 'Remove' },
    { label: 'Copy' },
    { label: 'Paste' },
  ];

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

  // Clicking a slice thumbnail selects it, or deselects (sends `slice:
  // null`) if it is already the selected slice -- Dash's click-to-toggle
  // (display_slice), reproduced here since the API's selection endpoint
  // itself is not a toggle (see ARCHITECTURE.md's segmentation endpoints).
  function onSliceClick(index: number): void {
    if (isBusy() || !projectStore.view) return;
    const alreadySelected = projectStore.view.selectedSlice === index;
    void workflow.selectSlice(alreadySelected ? null : index);
  }
</script>

<div class="segmentation-tab" data-testid="tab-segmentation">
  <div class="segmentation-grid">
    <div class="thresholds-column">
      <span class="panel-label">Thresholds</span>
      <div class="panel thresholds-box" data-testid="thresholds-container">
        {#each localValues as value, index (index)}
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
            title={action.onClick ? undefined : 'Not available yet'}
            disabled={action.onClick ? isBusy() || !projectStore.view : true}
            onclick={action.onClick}
          >
            {action.label}
          </button>
        {/each}
      </div>
    </div>
  </div>

  <div class="panel slice-strip">
    {#each projectStore.view?.slices ?? [] as slice (slice.index)}
      {@const selected = projectStore.view?.selectedSlice === slice.index}
      <div
        class="slice-thumb"
        data-testid="slice-thumbnail-wrapper"
        role="option"
        tabindex="0"
        aria-selected={selected}
        data-selected={selected}
        onclick={() => onSliceClick(slice.index)}
        onkeydown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onSliceClick(slice.index);
          }
        }}
      >
        <span class="depth-number">{slice.depth}</span>
        <img data-testid="slice-thumbnail" alt={`image_slice_${slice.index}`} src={slice.thumbnail.url} />
        <span class="slice-label">{`image_slice_${slice.index}`}</span>
        {#if selected}
          <div class="slice-overlay"></div>
        {/if}
      </div>
    {/each}
  </div>
</div>

<style>
  .segmentation-grid {
    display: grid;
    grid-template-columns: 3fr 2fr;
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

  .actions-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-2);
  }

  .slice-strip {
    margin: var(--space-2);
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--space-1);
    min-height: 2rem;
  }

  .slice-thumb {
    position: relative;
    cursor: pointer;
  }

  .slice-overlay {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background-color: color-mix(in srgb, var(--color-success) 50%, transparent);
  }

  .slice-thumb img {
    width: 100%;
    display: block;
  }

  .depth-number {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 2.5rem;
    color: var(--color-depth-number);
    pointer-events: none;
  }

  .slice-label {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    text-align: center;
    font-size: 0.75rem;
    padding: 2px;
    background-color: color-mix(in srgb, var(--color-bg) 60%, transparent);
    color: var(--color-text);
  }
</style>
