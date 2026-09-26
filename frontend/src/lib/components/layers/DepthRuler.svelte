<script lang="ts" module>
  /** One slice on the ruler: its depth and the vertical center of its row. */
  export type RulerEntry = {
    index: number;
    depth: number;
    rowCenter: number;
    selected: boolean;
    ground: boolean;
  };

  /** Top of the axis and the ruler's width, in px (the list starts right of it). */
  export const RULER_TOP = 24;
  export const RULER_WIDTH = 52;
</script>

<script lang="ts">
  /**
   * Vertical depth axis beside the layer list (docs/redesign/HANDOFF.md §3):
   * 255 (near) at the top, 0 (far) at the bottom, one draggable handle per
   * slice at its depth, and a curve from each handle to its row. Dragging
   * (or arrow keys on a focused handle) previews the new depth locally and
   * commits it with `setSliceDepth` on release.
   */
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';

  let { entries, axisHeight }: { entries: RulerEntry[]; axisHeight: number } = $props();

  // The handle being dragged or nudged, and its uncommitted depth.
  let draft = $state<{ index: number; depth: number } | null>(null);
  let axisEl: HTMLDivElement | undefined = $state();

  const depthOf = (entry: RulerEntry) => (draft?.index === entry.index ? draft.depth : entry.depth);
  const handleY = (depth: number) => RULER_TOP + ((255 - depth) / 255) * axisHeight;

  function depthAt(clientY: number): number {
    const rect = axisEl?.getBoundingClientRect();
    if (!rect || rect.height <= 0) return 0;
    const depth = Math.round(255 - ((clientY - rect.top) / rect.height) * 255);
    return Math.max(0, Math.min(255, depth));
  }

  function commit(): void {
    if (!draft) return;
    const { index, depth } = draft;
    const entry = entries.find((e) => e.index === index);
    draft = null;
    if (entry && entry.depth !== depth) void workflow.setSliceDepth(index, depth);
  }

  function onPointerDown(entry: RulerEntry, event: PointerEvent): void {
    if (isBusy() || event.button !== 0) return;
    event.preventDefault();
    (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
    draft = { index: entry.index, depth: entry.depth };
  }

  function onPointerMove(entry: RulerEntry, event: PointerEvent): void {
    if (draft?.index !== entry.index) return;
    draft = { index: entry.index, depth: depthAt(event.clientY) };
  }

  function onKeydown(entry: RulerEntry, event: KeyboardEvent): void {
    const step = event.shiftKey ? 10 : 1;
    const delta = event.key === 'ArrowUp' ? step : event.key === 'ArrowDown' ? -step : 0;
    if (delta === 0 || isBusy()) return;
    event.preventDefault();
    const current = depthOf(entry);
    draft = { index: entry.index, depth: Math.max(0, Math.min(255, current + delta)) };
  }

  function onKeyup(event: KeyboardEvent): void {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') commit();
  }
</script>

<div class="ruler" style={`height: ${RULER_TOP + axisHeight + 28}px`}>
  <span class="end mono" style="top: 4px">255 near</span>
  <div class="axis" bind:this={axisEl} style={`top: ${RULER_TOP}px; height: ${axisHeight}px`}></div>
  <span class="end mono" style={`top: ${RULER_TOP + axisHeight + 8}px`}>0 far</span>

  <svg class="links" width="60" height={RULER_TOP + axisHeight + 28} aria-hidden="true">
    {#each entries as entry (entry.index)}
      {@const y = handleY(depthOf(entry))}
      <path
        d={`M40 ${y.toFixed(1)} C 50 ${y.toFixed(1)}, 50 ${entry.rowCenter}, 60 ${entry.rowCenter}`}
        class:selected={entry.selected}
        class:ground={entry.ground && !entry.selected}
      />
    {/each}
  </svg>

  {#each entries as entry (entry.index)}
    {@const depth = depthOf(entry)}
    <button
      type="button"
      class="handle"
      class:selected={entry.selected}
      class:ground={entry.ground}
      class:dragging={draft?.index === entry.index}
      style={`top: ${handleY(depth) - (entry.selected ? 6 : 5)}px`}
      role="slider"
      aria-orientation="vertical"
      aria-valuemin="0"
      aria-valuemax="255"
      aria-valuenow={depth}
      aria-label={`Depth of image_slice_${entry.index}${entry.ground ? ', ground' : ''}`}
      title={`Depth ${depth}. Drag to change.`}
      data-testid="depth-handle"
      data-slice-index={entry.index}
      disabled={isBusy()}
      onpointerdown={(event) => onPointerDown(entry, event)}
      onpointermove={(event) => onPointerMove(entry, event)}
      onpointerup={commit}
      onpointercancel={() => (draft = null)}
      onkeydown={(event) => onKeydown(entry, event)}
      onkeyup={onKeyup}
      onblur={commit}
    ></button>
  {/each}
</div>

<style>
  .ruler {
    position: absolute;
    left: 0;
    top: 0;
    width: 52px;
    background: var(--color-bg);
    border-right: 1px solid var(--color-border);
  }

  .end {
    position: absolute;
    left: 0;
    width: 52px;
    text-align: center;
    font-size: 10px;
    color: var(--color-text-muted);
    white-space: nowrap;
  }

  .axis {
    position: absolute;
    left: 24px;
    width: 4px;
    border-radius: 2px;
    background: linear-gradient(180deg, var(--color-text), var(--color-surface-hover));
  }

  .links {
    position: absolute;
    left: 0;
    top: 0;
    overflow: visible;
    pointer-events: none;
  }

  .links path {
    fill: none;
    stroke: var(--color-border-strong);
    stroke-width: 1.3;
  }

  .links path.ground {
    stroke: var(--color-success);
  }

  .links path.selected {
    stroke: var(--color-selection);
  }

  .handle {
    position: absolute;
    left: 12px;
    width: 28px;
    height: 10px;
    padding: 0;
    border: none;
    border-radius: 3px;
    background: var(--color-text-secondary);
    cursor: ns-resize;
    touch-action: none;
  }

  .handle.ground {
    background: var(--color-success-soft);
    box-shadow: inset 0 0 0 2px var(--color-success);
  }

  .handle.selected {
    background: var(--color-selection);
    height: 12px;
    left: 10px;
    width: 32px;
  }

  .handle:focus-visible {
    outline: 2px solid var(--color-selection);
    outline-offset: 2px;
  }

  .handle:disabled {
    cursor: default;
  }
</style>
