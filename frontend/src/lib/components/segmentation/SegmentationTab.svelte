<script lang="ts">
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import * as api from '../../api/client';
  import { triggerDownload } from '../../download';
  import HelpTooltip from '../shared/HelpTooltip.svelte';
  import { SEGMENTATION_HELP_TEXTS } from '../../helpTexts';

  type ActionButton = { label: string; testId: string; onClick: () => void };

  // Same 8 buttons as components.py's make_slice_generation_container Actions
  // panel. None of them are gated on selection/mask/clipboard state here --
  // matching Dash exactly, none of webui.py's delete/add-mask/remove-mask/
  // copy/paste/balance button elements are ever `disabled` for that reason
  // either; each one just runs and logs a no-op message when a precondition
  // (selection/mask/clipboard) is missing (see workflow.ts's slice-editing
  // section for the exact per-button precedence). The only gating here is
  // "is there a project, and is nothing else in flight", same as Generate.
  const actions: ActionButton[] = [
    { label: 'Generate', testId: 'generate-slices', onClick: () => void workflow.generateSlices() },
    { label: 'Balance', testId: 'balance-slices', onClick: () => void workflow.balanceSlices() },
    { label: 'Create', testId: 'create-slice', onClick: () => void workflow.createSlice() },
    { label: 'Delete', testId: 'delete-slice', onClick: () => void workflow.deleteSlice() },
    { label: 'Add', testId: 'add-mask-to-slice', onClick: () => void workflow.addMaskToSlice() },
    { label: 'Remove', testId: 'remove-mask-from-slice', onClick: () => void workflow.removeMaskFromSlice() },
    { label: 'Copy', testId: 'copy-slice', onClick: () => void workflow.copySlice() },
    { label: 'Paste', testId: 'paste-slice', onClick: () => void workflow.pasteSlice() },
  ];

  // Ground plane: the selected slice becomes (or stops being) the horizontal
  // ground; Fit places the horizon and ground from the scene.
  const selectedIsGround = $derived(
    projectStore.view?.slices.find((s) => s.index === projectStore.view?.selectedSlice)?.isGround ?? false,
  );
  const hasGround = $derived(projectStore.view?.slices.some((s) => s.isGround) ?? false);

  // -- Per-slice depth editing: click the badge to reveal a numeric input,
  // Enter or blur commits it (webui.py's WEB-22/WEB-23: `display_depth_input`
  // un-hides the input, `record_depth_input` commits on Enter). Unlike Dash,
  // the selection overlay uses `pointer-events: none` (see `.slice-overlay`
  // below) so the badge stays clickable on a selected slice too -- Dash's own
  // depth badge is unreachable through normal hit-testing once its slice is
  // selected (see PARITY.md "Known quirks"); this UI does not reproduce that.
  let editingDepthIndex = $state<number | null>(null);
  let depthDraft = $state('');

  function startEditingDepth(index: number, currentDepth: number, event: Event): void {
    event.stopPropagation();
    if (isBusy()) return;
    editingDepthIndex = index;
    depthDraft = String(currentDepth);
  }

  function commitDepth(index: number): void {
    if (editingDepthIndex !== index) return;
    editingDepthIndex = null;
    // Dash's record_depth_input uses int(value): blank or non-integer input
    // does not commit (an empty string must not become depth 0).
    const draft = depthDraft.trim();
    if (!/^-?\d+$/.test(draft)) return;
    void workflow.setSliceDepth(index, Number.parseInt(draft, 10));
  }

  function onDepthInputKeydown(index: number, event: KeyboardEvent): void {
    event.stopPropagation();
    if (event.key === 'Enter') {
      event.preventDefault();
      (event.currentTarget as HTMLInputElement).blur();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      editingDepthIndex = null;
    }
  }

  function onDepthInputBlur(index: number): void {
    commitDepth(index);
  }

  // -- Undo/redo carets (webui.py's WEB-24 `undo_slice`).

  function onUndo(index: number, event: Event): void {
    event.stopPropagation();
    if (isBusy()) return;
    void workflow.undoSlice(index);
  }

  function onRedo(index: number, event: Event): void {
    event.stopPropagation();
    if (isBusy()) return;
    void workflow.redoSlice(index);
  }

  // -- Per-thumbnail image upload/drop target (webui.py's WEB-34 `slice_upload`).

  function uploadFile(index: number, file: File | undefined | null): void {
    if (!file || isBusy()) return;
    void workflow.uploadSliceImage(index, file);
  }

  function onUploadInputChange(index: number, event: Event): void {
    event.stopPropagation();
    const target = event.currentTarget as HTMLInputElement;
    uploadFile(index, target.files?.[0]);
    target.value = '';
  }

  function onSliceDrop(index: number, event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    uploadFile(index, event.dataTransfer?.files?.[0]);
  }

  function onSliceDragOver(event: DragEvent): void {
    event.preventDefault();
  }

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

  // Slice download (webui.py's WEB-33 `download_image`, triggered by
  // clicking the label the way Dash's `#slice-info` button does): a real
  // `<a download>` click against the raw-slice-PNG endpoint, not
  // `window.open` (see ExportTab.svelte's glTF-export comment for why).
  function onDownloadSlice(index: number, event: Event): void {
    event.stopPropagation();
    const view = projectStore.view;
    if (!view) return;
    triggerDownload(api.getSliceDownloadUrl(view.id, index), `image_slice_${index}.png`);
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
        ondrop={(event) => onSliceDrop(slice.index, event)}
        ondragover={onSliceDragOver}
      >
        {#if editingDepthIndex === slice.index}
          <!-- svelte-ignore a11y_autofocus -->
          <input
            type="number"
            class="depth-input"
            data-testid="slice-depth-input"
            autofocus
            value={depthDraft}
            oninput={(event) => (depthDraft = (event.currentTarget as HTMLInputElement).value)}
            onclick={(event) => event.stopPropagation()}
            onkeydown={(event) => onDepthInputKeydown(slice.index, event)}
            onblur={() => onDepthInputBlur(slice.index)}
          />
        {:else}
          <button
            type="button"
            class="depth-number"
            data-testid="slice-depth-display"
            onclick={(event) => startEditingDepth(slice.index, slice.depth, event)}
          >
            {slice.depth}
          </button>
        {/if}
        <img data-testid="slice-thumbnail" alt={`image_slice_${slice.index}`} src={slice.thumbnail.url} />
        {#if slice.isGround}
          <span class="ground-badge" data-testid="ground-badge">Ground</span>
        {/if}
        <input
          type="file"
          accept="image/*"
          class="sr-only"
          data-testid="slice-upload-input"
          onclick={(event) => event.stopPropagation()}
          onchange={(event) => onUploadInputChange(slice.index, event)}
        />
        <div class="slice-label">
          <button
            type="button"
            class="caret"
            data-testid="slice-undo"
            title="Undo last change"
            disabled={!slice.canUndo || isBusy()}
            onclick={(event) => onUndo(slice.index, event)}
          >
            &#x25C2;
          </button>
          <button
            type="button"
            class="caret"
            data-testid="slice-redo"
            title="Redo last change"
            disabled={!slice.canRedo || isBusy()}
            onclick={(event) => onRedo(slice.index, event)}
          >
            &#x25B8;
          </button>
          <button
            type="button"
            class="slice-info"
            data-testid="slice-download"
            title="Download slice image"
            onclick={(event) => onDownloadSlice(slice.index, event)}
          >
            {`image_slice_${slice.index}`}
          </button>
        </div>
        {#if selected}
          <div class="slice-overlay"></div>
        {/if}
      </div>
    {/each}
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

  /* Center badge/input; distinct from .slice-overlay (which is
     `pointer-events: none` and comes later in DOM order), so this stays
     clickable even on a selected slice -- see the depth-editing comment
     above the `<script>` block for why that matters. */
  .depth-number,
  .depth-input {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1;
  }

  .depth-number {
    font-size: 2.5rem;
    line-height: 1;
    color: var(--color-depth-number);
    background: none;
    border: none;
    padding: 0;
    font-family: inherit;
    cursor: pointer;
  }

  .depth-input {
    width: 3.5rem;
    font-size: 1.25rem;
    text-align: center;
  }

  .ground-badge {
    position: absolute;
    z-index: 1;
    top: var(--space-1);
    left: var(--space-1);
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    font-size: 0.75rem;
    background-color: var(--color-success);
    color: var(--color-success-text);
    pointer-events: none;
  }

  .slice-label {
    position: absolute;
    z-index: 1;
    bottom: 0;
    left: 0;
    right: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-1);
    text-align: center;
    font-size: 0.75rem;
    padding: 2px;
    background-color: color-mix(in srgb, var(--color-bg) 60%, transparent);
    color: var(--color-text);
  }

  .caret {
    background: none;
    border: none;
    padding: 0 2px;
    font-family: inherit;
    font-size: 0.75rem;
    color: var(--color-text);
    cursor: pointer;
  }

  .caret:disabled {
    color: var(--color-disabled-text);
    cursor: default;
  }

  .slice-info {
    background: none;
    border: none;
    padding: 0;
    font-family: inherit;
    font-size: inherit;
    color: inherit;
    cursor: pointer;
    text-decoration: underline dotted;
  }
</style>
