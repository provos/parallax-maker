<script lang="ts">
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';

  // Keep the slider in sync with the project once one exists (including
  // after a restore), the same way Dash's slider reflects state.
  $effect(() => {
    const view = projectStore.view;
    if (view) uiStore.setPendingNumSlices(view.numSlices);
  });

  function onNumSlicesChange(event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    uiStore.setPendingNumSlices(value);
    if (projectStore.view) void workflow.updateSliceCount(value);
  }

  let restoreInput: HTMLInputElement | undefined;

  function onRestoreChange(event: Event): void {
    const target = event.currentTarget as HTMLInputElement;
    const file = target.files?.[0];
    target.value = '';
    if (file) void workflow.restoreProject(file);
  }
</script>

<div class="configuration-tab" data-testid="tab-configuration">
  <div class="field">
    <label class="field-label" for="num-slices">Number of Slices</label>
    <input
      id="num-slices"
      type="range"
      min="2"
      max="10"
      step="1"
      data-testid="num-slices"
      value={uiStore.pendingNumSlices}
      disabled={isBusy()}
      onchange={onNumSlicesChange}
    />
    <div class="range-marks">
      {#each Array.from({ length: 9 }, (_, i) => i + 2) as mark (mark)}
        <span>{mark}</span>
      {/each}
    </div>
  </div>

  <div class="field">
    <span class="field-label">Export/Import State</span>
    <div class="state-actions">
      <label class="btn" for="restore-state-input">
        Load State
        <input
          bind:this={restoreInput}
          id="restore-state-input"
          type="file"
          accept="application/json"
          class="sr-only"
          data-testid="restore-state-input"
          onchange={onRestoreChange}
        />
      </label>
      <button type="button" class="btn" disabled title="Not available yet">Save State</button>
    </div>
  </div>
</div>

<style>
  .field {
    margin-bottom: var(--space-4);
  }

  .range-marks {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }

  input[type='range'] {
    width: 100%;
  }

  .state-actions {
    display: flex;
    gap: var(--space-4);
  }

  .state-actions .btn {
    cursor: pointer;
  }
</style>
