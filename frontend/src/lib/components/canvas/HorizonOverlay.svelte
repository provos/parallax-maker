<script lang="ts">
  /**
   * Draggable horizon line over the main image (inside the zoom/pan box, so
   * it follows the image). Dragging moves the horizon to another image row;
   * releasing commits it and the server derives the camera pitch from it.
   */
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';

  let lineEl = $state<HTMLDivElement | undefined>();
  let draftRow = $state<number | null>(null);
  let dragPointer: number | null = null;

  const height = $derived(projectStore.view?.image?.height ?? 0);
  const committedRow = $derived(projectStore.view?.settings.camera.horizonRow ?? null);
  const row = $derived(draftRow ?? committedRow);
  const topPercent = $derived(row === null || height === 0 ? 0 : Math.min(100, Math.max(0, (row / height) * 100)));

  function rowAt(clientY: number): number | null {
    const box = lineEl?.parentElement?.getBoundingClientRect();
    if (!box || box.height <= 0 || height === 0) return null;
    return Math.min(height - 1, Math.max(0, ((clientY - box.top) / box.height) * height));
  }

  function onPointerDown(event: PointerEvent): void {
    if (event.button !== 0 || isBusy()) return;
    // Don't let the drop-zone start a pan or the image take a segmentation click.
    event.stopPropagation();
    event.preventDefault();
    dragPointer = event.pointerId;
    lineEl?.setPointerCapture(event.pointerId);
    draftRow = rowAt(event.clientY);
  }

  function onPointerMove(event: PointerEvent): void {
    if (event.pointerId !== dragPointer) return;
    event.stopPropagation();
    draftRow = rowAt(event.clientY);
  }

  /** Arrow keys nudge the horizon by a row (Shift: ten rows). */
  function onKeydown(event: KeyboardEvent): void {
    // Keep keys from reaching the drop-zone (Enter/Space open its file picker).
    if (event.key === 'Enter' || event.key === ' ') event.stopPropagation();
    if ((event.key !== 'ArrowUp' && event.key !== 'ArrowDown') || committedRow === null || isBusy()) return;
    event.preventDefault();
    event.stopPropagation();
    const step = (event.shiftKey ? 10 : 1) * (event.key === 'ArrowUp' ? -1 : 1);
    void workflow.setHorizonRow(Math.min(height - 1, Math.max(0, committedRow + step)));
  }

  async function onPointerUp(event: PointerEvent): Promise<void> {
    if (event.pointerId !== dragPointer) return;
    event.stopPropagation();
    dragPointer = null;
    const target = draftRow;
    if (target !== null && target !== committedRow) await workflow.setHorizonRow(target);
    draftRow = null;
  }
</script>

{#if row !== null && height > 0}
  <div
    bind:this={lineEl}
    class="horizon"
    class:dragging={draftRow !== null}
    style={`top: ${topPercent}%`}
    data-testid="horizon-line"
    data-row={Math.round(row)}
    role="slider"
    tabindex="0"
    aria-label="Horizon"
    aria-valuemin="0"
    aria-valuemax={height}
    aria-valuenow={Math.round(row)}
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={(event) => void onPointerUp(event)}
    onpointercancel={(event) => void onPointerUp(event)}
    onclick={(event) => event.stopPropagation()}
    onkeydown={onKeydown}
  >
    <span class="handle" aria-hidden="true"></span>
    <span class="label" aria-hidden="true">Horizon · drag</span>
  </div>
{/if}

<style>
  /* A tall transparent hit area centred on a dashed line. */
  .horizon {
    position: absolute;
    left: 0;
    right: 0;
    height: 14px;
    transform: translateY(-50%);
    z-index: 7;
    cursor: ns-resize;
    touch-action: none;
  }

  .horizon::before {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    top: 50%;
    border-top: 2px dashed var(--color-horizon, #d85a30);
  }

  .horizon.dragging::before {
    border-top-style: solid;
  }

  .handle {
    position: absolute;
    left: 8px;
    top: 50%;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    transform: translateY(-50%);
    background-color: var(--color-horizon, #d85a30);
  }

  .label {
    position: absolute;
    left: 26px;
    bottom: 100%;
    font-size: 0.75rem;
    padding: 0 var(--space-1);
    border-radius: var(--radius-md);
    background-color: var(--color-bg);
    color: var(--color-text);
    white-space: nowrap;
  }
</style>
