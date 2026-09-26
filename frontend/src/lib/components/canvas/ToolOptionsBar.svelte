<script lang="ts">
  /**
   * Options for the active tool only (docs/redesign/HANDOFF.md §5): what a
   * Segment click selects and the mask tools, the Brush options, the
   * horizon's row and pitch, or the camera pad in the Parallax 2D view.
   */
  import ArrowUp from '@lucide/svelte/icons/arrow-up';
  import ArrowDown from '@lucide/svelte/icons/arrow-down';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import ZoomIn from '@lucide/svelte/icons/zoom-in';
  import ZoomOut from '@lucide/svelte/icons/zoom-out';
  import LocateFixed from '@lucide/svelte/icons/locate-fixed';
  import { uiStore } from '../../state/ui.svelte';
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import type { CameraDirection } from '../../api/client';
  import MaskToolbar from './MaskToolbar.svelte';

  const view = $derived(projectStore.view);
  const segmentation = $derived(view?.segmentation);
  const multiPointEnabled = $derived(segmentation?.multiPointMode ?? false);
  const queued = $derived(segmentation?.queuedPoints.length ?? 0);
  const objectMode = $derived(uiStore.segmentationMode === 'segment');
  const canNavigate = $derived((view?.slices.length ?? 0) > 0);

  const CAMERA_BUTTONS: { direction: CameraDirection; label: string; icon: typeof ArrowUp }[] = [
    { direction: 'left', label: 'Move camera left', icon: ArrowLeft },
    { direction: 'up', label: 'Move camera up', icon: ArrowUp },
    { direction: 'down', label: 'Move camera down', icon: ArrowDown },
    { direction: 'right', label: 'Move camera right', icon: ArrowRight },
    { direction: 'in', label: 'Move camera forward', icon: ZoomIn },
    { direction: 'out', label: 'Move camera back', icon: ZoomOut },
    { direction: 'reset', label: 'Reset camera position', icon: LocateFixed },
  ];

  function navigate(direction: CameraDirection): void {
    if (!canNavigate || isBusy()) return;
    void workflow.navigateCamera(direction);
  }

  function toggleMultiPoint(): void {
    if (!view || !objectMode || isBusy()) return;
    void workflow.setMultiPointMode(!multiPointEnabled);
  }
</script>

<div class="tool-options" data-testid="tool-options">
  {#if uiStore.view === 'parallax'}
    <span class="label">Camera</span>
    <div class="camera-pad" role="group" aria-label="Parallax camera" data-testid="camera-nav">
      {#each CAMERA_BUTTONS as { direction, label, icon: Icon } (direction)}
        <button
          type="button"
          class="btn btn-sm btn-icon"
          data-testid={`camera-${direction}`}
          aria-label={label}
          title={label}
          disabled={!canNavigate || isBusy()}
          onclick={() => navigate(direction)}
        >
          <Icon size={14} strokeWidth={1.6} />
        </button>
      {/each}
    </div>
    <span class="hint">Moves the camera over the slices and renders the view.</span>
  {:else if uiStore.view === '3d'}
    <span class="hint">Drag to orbit · scroll to zoom the 3D scene.</span>
  {:else if uiStore.tool === 'segment'}
    <span class="label">Select by</span>
    <div class="seg" role="radiogroup" aria-label="Select by" data-testid="select-by">
      <button
        type="button"
        role="radio"
        aria-checked={objectMode}
        data-testid="select-by-object"
        onclick={() => uiStore.setSegmentationMode('segment')}
      >
        Object
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={!objectMode}
        data-testid="select-by-depth"
        onclick={() => uiStore.setSegmentationMode('depth')}
      >
        Depth band
      </button>
    </div>
    <button
      type="button"
      class="btn btn-sm"
      class:btn-selected={multiPointEnabled}
      data-testid="multi-point"
      aria-pressed={multiPointEnabled}
      title="Queue several points, then segment them together"
      disabled={!view || !objectMode || isBusy()}
      onclick={toggleMultiPoint}
    >
      Multi-point
    </button>
    {#if multiPointEnabled}
      <button
        type="button"
        class="btn btn-sm btn-primary"
        data-testid="multi-commit"
        disabled={queued === 0 || !objectMode || isBusy()}
        onclick={() => void workflow.commitMultiPoint()}
      >
        Segment {queued} {queued === 1 ? 'point' : 'points'}
      </button>
    {/if}
    <span class="divider" aria-hidden="true"></span>
    <button
      type="button"
      class="btn btn-sm"
      data-testid="invert-mask"
      title="Invert the selection mask"
      disabled={!view || isBusy()}
      onclick={() => void workflow.invertMask()}
    >
      Invert
    </button>
    <button
      type="button"
      class="btn btn-sm"
      data-testid="feather-mask"
      title="Feather the selection mask"
      disabled={!view || isBusy()}
      onclick={() => void workflow.featherMask()}
    >
      Feather
    </button>
    <span class="hint">Shift-click adds · Ctrl-click subtracts</span>
  {:else if uiStore.tool === 'brush'}
    <MaskToolbar />
  {:else if uiStore.tool === 'horizon'}
    <span class="label">Horizon</span>
    {#if view?.settings.camera.horizonRow != null}
      <span class="mono readout" data-testid="horizon-options-readout">
        row {Math.round(view.settings.camera.horizonRow)} · pitch {(view.settings.camera.pitch ?? 0).toFixed(1)}°
      </span>
    {/if}
    <span class="hint">Drag the line to where the ground meets the sky.</span>
  {:else}
    <span class="hint">Drag to pan · scroll to zoom</span>
  {/if}
</div>

<style>
  .tool-options {
    height: var(--toolopts-h);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: 0 var(--space-3);
    background: var(--color-surface);
    border-bottom: 1px solid var(--color-border);
    min-width: 0;
    overflow: hidden;
    white-space: nowrap;
  }

  .label {
    margin: 0;
  }

  .hint {
    color: var(--color-text-muted);
    font-size: var(--text-small);
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .readout {
    font-size: 12px;
  }

  .camera-pad {
    display: flex;
    gap: 2px;
  }

  .divider {
    align-self: stretch;
    width: 1px;
    margin: 8px var(--space-1);
    background: var(--color-border);
  }

  .seg {
    display: inline-flex;
    background: var(--color-surface-raised);
    border-radius: 7px;
    padding: 2px;
    gap: 2px;
    border: 1px solid var(--color-border);
  }

  .seg button {
    height: 24px;
    padding: 0 10px;
    border: none;
    border-radius: 5px;
    background: transparent;
    color: var(--color-text-secondary);
    font: inherit;
    font-weight: 500;
    cursor: pointer;
  }

  .seg button[aria-checked='true'] {
    background: var(--color-surface-hover);
    color: var(--color-text);
    box-shadow: 0 0 0 1px var(--color-border-strong);
  }
</style>
