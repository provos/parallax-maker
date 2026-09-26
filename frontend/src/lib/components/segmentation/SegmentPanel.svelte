<script lang="ts">
  /**
   * The Slices step's Inspector panel (docs/redesign/HANDOFF.md §3), Segment
   * Anything first: click objects, refine, and make them slices. Splitting
   * the whole image into depth bands is the secondary path, and the
   * selected slice's depth and ground flag sit at the bottom.
   */
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';

  const view = $derived(projectStore.view);
  const segmentation = $derived(view?.segmentation);
  const hasMask = $derived(segmentation?.hasMask ?? false);
  const queued = $derived(segmentation?.queuedPoints.length ?? 0);
  const selected = $derived(view?.slices.find((s) => s.index === view.selectedSlice) ?? null);
  const disabled = $derived(isBusy() || !view);

  // Sub-step progress: 1 click an object, 2 refine, 3 make it a slice.
  const stepState = (n: 1 | 2 | 3): 'done' | 'now' | 'todo' => {
    const current = hasMask ? 3 : queued > 0 ? 2 : 1;
    return n < current ? 'done' : n === current ? 'now' : 'todo';
  };

  const summary = $derived(
    hasMask
      ? 'Selection ready'
      : queued > 0
        ? `${queued} ${queued === 1 ? 'point' : 'points'} queued`
        : 'Nothing selected yet',
  );

  // -- Split by depth: open until the first slices exist.
  let splitOpen = $state<boolean | null>(null);
  const splitShown = $derived(splitOpen ?? (view?.slices.length ?? 0) === 0);

  // Interior threshold boundaries only (length numSlices - 1); the backend
  // rejects any other length (see UpdateThresholdValues.update_threshold_values).
  const interiorThresholds = $derived.by((): number[] => {
    if (!view || view.numSlices < 2 || view.thresholds.length !== view.numSlices + 1) return [];
    return view.thresholds.slice(1, view.numSlices);
  });

  // Local editable copy so dragging feels smooth; re-synced whenever the
  // server's thresholds change (including our own applied response).
  let localValues = $state<number[]>([]);
  $effect(() => {
    localValues = [...interiorThresholds];
  });

  function onThresholdInput(index: number, event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    localValues = localValues.map((v, i) => (i === index ? value : v));
  }

  // The number of slices a split makes (persisted with the project).
  $effect(() => {
    if (view) uiStore.setPendingNumSlices(view.numSlices);
  });

  function onNumSlicesChange(event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    uiStore.setPendingNumSlices(value);
    if (view) void workflow.updateSliceCount(value);
  }

  // -- Selected slice depth (slider and field commit on change).
  function onDepthChange(event: Event): void {
    if (!selected) return;
    const raw = Number.parseInt((event.currentTarget as HTMLInputElement).value, 10);
    if (Number.isNaN(raw)) return;
    const depth = Math.max(0, Math.min(255, raw));
    if (depth !== selected.depth) void workflow.setSliceDepth(selected.index, depth);
  }
</script>

<div class="segment-panel" data-testid="tab-segmentation">
  <section class="sec">
    <div class="sec-title">
      <h3 class="label accent">Segment objects</h3>
      <span class="faint">Segment Anything</span>
    </div>
    <ol class="substeps">
      <li data-state={stepState(1)}>
        <span class="sub">1</span>
        <span>Click an object on the image. <span class="faint">Shift-click adds a point, Ctrl-click removes one.</span></span>
      </li>
      <li data-state={stepState(2)}>
        <span class="sub">2</span>
        <span>Refine: <span class="faint">Invert or Feather the selection from the options bar.</span></span>
      </li>
      <li data-state={stepState(3)}>
        <span class="sub">3</span>
        <span>Make it a <b>new slice</b>, or add it to the selected one.</span>
      </li>
    </ol>
    <div class="summary">
      <span class="summary-text" data-testid="selection-summary">{summary}</span>
      <button
        type="button"
        class="btn btn-sm"
        class:btn-primary={hasMask}
        data-testid="create-slice"
        title={hasMask ? 'Make the selection a new slice' : 'Add an empty slice'}
        {disabled}
        onclick={() => void workflow.createSlice()}
      >
        New slice
      </button>
    </div>
    <div class="row">
      <button
        type="button"
        class="btn btn-sm grow"
        data-testid="add-mask-to-slice"
        title="Add the selection to the selected slice"
        {disabled}
        onclick={() => void workflow.addMaskToSlice()}
      >
        Add to selected
      </button>
      <button
        type="button"
        class="btn btn-sm grow"
        data-testid="remove-mask-from-slice"
        title="Remove the selection from the selected slice"
        {disabled}
        onclick={() => void workflow.removeMaskFromSlice()}
      >
        Remove from selected
      </button>
    </div>
  </section>

  <section class="sec">
    <button
      type="button"
      class="disclosure"
      aria-expanded={splitShown}
      data-testid="split-toggle"
      onclick={() => (splitOpen = !splitShown)}
    >
      {#if splitShown}<ChevronDown size={14} strokeWidth={1.6} />{:else}<ChevronRight size={14} strokeWidth={1.6} />{/if}
      <span class="label">Split by depth</span>
      <span class="faint grow-right">{view?.numSlices ?? uiStore.pendingNumSlices} bands</span>
    </button>
    <div class="split-body" class:hidden={!splitShown}>
      <p class="faint">Cut the whole image into bands of depth. Good for backgrounds; use Segment for objects.</p>
      <div class="g3">
        <label for="num-slices" class="muted">Slices</label>
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
        <span class="mono value">{uiStore.pendingNumSlices}</span>
      </div>
      <div class="thresholds" data-testid="thresholds-container">
        {#each localValues as value, index (index)}
          <div class="g3">
            <span class="muted">Band {index + 1}</span>
            <input
              type="range"
              min="0"
              max="255"
              step="1"
              data-testid="threshold-handle"
              aria-label={`Threshold ${index + 1}`}
              {value}
              {disabled}
              oninput={(event) => onThresholdInput(index, event)}
              onchange={() => void workflow.updateThresholds([...localValues])}
            />
            <span class="mono value" data-testid="threshold-value">{value}</span>
          </div>
        {/each}
      </div>
      <div class="row">
        <button type="button" class="btn grow" data-testid="generate-slices" {disabled} onclick={() => void workflow.generateSlices()}>
          Split image
        </button>
        <button
          type="button"
          class="btn btn-ghost"
          data-testid="balance-slices"
          title="Spread the slice depths evenly"
          {disabled}
          onclick={() => void workflow.balanceSlices()}
        >
          Balance
        </button>
      </div>
    </div>
  </section>

  {#if selected}
    <section class="sec" data-testid="selected-slice-section">
      <h3 class="label">Selected slice</h3>
      <div class="g3">
        <label for="selected-depth" class="muted">Depth</label>
        <input
          id="selected-depth"
          type="range"
          min="0"
          max="255"
          step="1"
          data-testid="selected-depth-slider"
          value={selected.depth}
          disabled={isBusy()}
          onchange={onDepthChange}
        />
        <input
          type="number"
          min="0"
          max="255"
          class="text-input mono depth-field"
          aria-label="Depth value"
          data-testid="selected-depth-input"
          value={selected.depth}
          disabled={isBusy()}
          onchange={onDepthChange}
        />
      </div>
      <div class="row">
        <button
          type="button"
          class="btn btn-sm grow"
          class:btn-selected={selected.isGround}
          data-testid="ground-toggle"
          title="Make the selected slice the horizontal ground plane"
          aria-pressed={selected.isGround ?? false}
          disabled={isBusy()}
          onclick={() => void workflow.toggleGroundPlane()}
        >
          {selected.isGround ? 'Ground plane' : 'Make ground plane'}
        </button>
      </div>
    </section>
  {/if}
</div>

<style>
  .segment-panel {
    display: flex;
    flex-direction: column;
    margin: calc(-1 * var(--space-3)) -14px;
  }

  .sec {
    padding: var(--space-3) 14px;
    border-bottom: 1px solid var(--color-border);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .sec-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .label {
    margin: 0;
  }

  .accent {
    color: var(--color-primary-soft-text);
  }

  .faint {
    color: var(--color-text-muted);
    font-size: var(--text-small);
  }

  p.faint {
    margin: 0;
  }

  .muted {
    color: var(--color-text-secondary);
  }

  .substeps {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .substeps li {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    line-height: 1.45;
  }

  .sub {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 10.5px;
    font-weight: 600;
    flex-shrink: 0;
    border: 1px solid var(--color-border-strong);
    color: var(--color-text-secondary);
    box-sizing: border-box;
  }

  li[data-state='done'] .sub {
    background: var(--color-success-soft);
    border-color: var(--color-success-line);
    color: var(--color-success);
  }

  li[data-state='now'] .sub {
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: var(--color-primary-text);
  }

  .summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) 10px;
    border-radius: var(--radius-md);
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
  }

  .summary-text {
    flex: 1 1 auto;
    color: var(--color-text-secondary);
  }

  .row {
    display: flex;
    gap: var(--space-2);
  }

  .grow {
    flex: 1 1 0;
  }

  .disclosure {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0;
    height: 22px;
    background: none;
    border: none;
    color: var(--color-text-muted);
    font: inherit;
    cursor: pointer;
  }

  .grow-right {
    margin-left: auto;
  }

  .split-body {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .thresholds {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .g3 {
    display: grid;
    grid-template-columns: 70px minmax(0, 1fr) 64px;
    align-items: center;
    gap: 10px;
  }

  .value {
    text-align: right;
    font-size: 12px;
  }

  .depth-field {
    padding: 3px 6px;
    text-align: right;
  }
</style>
