<script lang="ts">
  /**
   * The always-visible layer list (docs/redesign/HANDOFF.md §3): slices
   * nearest first with their thumbnail, name, badges and depth, a depth
   * ruler to drag depths, and Copy / Paste / Delete. Clicking a row selects
   * the slice (clicking the selected row deselects it). Each row's depth
   * chip edits that slice's depth in place without selecting it.
   */
  import Copy from '@lucide/svelte/icons/copy';
  import ClipboardPaste from '@lucide/svelte/icons/clipboard-paste';
  import Trash2 from '@lucide/svelte/icons/trash-2';
  import Layers from '@lucide/svelte/icons/layers';
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import DepthRuler, { RULER_TOP, RULER_WIDTH, type RulerEntry } from './DepthRuler.svelte';

  const ROW_PITCH = 68; // 64px row + 4px gap
  const LIST_TOP = 16;

  const view = $derived(projectStore.view);
  // Nearest first: highest depth at the top, like the ruler.
  const rows = $derived([...(view?.slices ?? [])].sort((a, b) => b.depth - a.depth || b.index - a.index));
  const aspect = $derived(view?.image ? view.image.width / view.image.height : 4 / 3);
  const thumbWidth = $derived(Math.round(Math.min(64, Math.max(28, 48 * aspect))));

  let scrollHeight = $state(0);
  const axisHeight = $derived(
    Math.max(rows.length * ROW_PITCH - 24, scrollHeight - RULER_TOP - 40, 160),
  );
  const entries = $derived(
    rows.map(
      (slice, i): RulerEntry => ({
        index: slice.index,
        depth: slice.depth,
        rowCenter: LIST_TOP + i * ROW_PITCH + 32,
        selected: view?.selectedSlice === slice.index,
        ground: !!slice.isGround,
      }),
    ),
  );

  function onRowClick(index: number): void {
    if (isBusy() || !view) return;
    void workflow.selectSlice(view.selectedSlice === index ? null : index);
  }

  function onListKeydown(event: KeyboardEvent): void {
    // Delete removes the selected slice, but only from the focused layer list.
    if ((event.key === 'Delete' || event.key === 'Backspace') && view?.selectedSlice != null && !isBusy()) {
      event.preventDefault();
      void workflow.deleteSlice();
    }
  }

  // -- In-place depth editing from a row's depth chip.
  let editingIndex = $state<number | null>(null);
  let depthDraft = $state('');

  function startEditing(index: number, depth: number, event: Event): void {
    event.stopPropagation();
    if (isBusy()) return;
    editingIndex = index;
    depthDraft = String(depth);
  }

  function commitDepth(index: number): void {
    if (editingIndex !== index) return;
    editingIndex = null;
    // Blank or non-integer input does not commit (it must not become 0);
    // anything else is clamped to the 0-255 depth range.
    const draft = depthDraft.trim();
    if (!/^-?\d+$/.test(draft)) return;
    const depth = Math.max(0, Math.min(255, Number.parseInt(draft, 10)));
    void workflow.setSliceDepth(index, depth);
  }

  function onDepthKeydown(event: KeyboardEvent): void {
    event.stopPropagation();
    if (event.key === 'Enter') {
      event.preventDefault();
      (event.currentTarget as HTMLInputElement).blur();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      editingIndex = null;
    }
  }
</script>

<aside class="layer-panel" aria-label="Layers" data-testid="layer-panel">
  <div class="panel-header">
    <span class="label">Layers · {rows.length}</span>
    <div class="header-actions">
      <button
        type="button"
        class="btn btn-ghost btn-icon"
        data-testid="copy-slice"
        aria-label="Copy the selection mask"
        title="Copy the selection mask"
        disabled={isBusy() || !view}
        onclick={() => void workflow.copySlice()}
      >
        <Copy size={16} strokeWidth={1.6} />
      </button>
      <button
        type="button"
        class="btn btn-ghost btn-icon"
        data-testid="paste-slice"
        aria-label="Paste into the selected slice"
        title="Paste into the selected slice"
        disabled={isBusy() || !view}
        onclick={() => void workflow.pasteSlice()}
      >
        <ClipboardPaste size={16} strokeWidth={1.6} />
      </button>
      <button
        type="button"
        class="btn btn-ghost btn-icon"
        data-testid="delete-slice"
        aria-label="Delete the selected slice"
        title="Delete the selected slice (Del)"
        disabled={isBusy() || !view}
        onclick={() => void workflow.deleteSlice()}
      >
        <Trash2 size={16} strokeWidth={1.6} />
      </button>
    </div>
  </div>

  {#if rows.length === 0}
    <div class="empty" data-testid="layers-empty">
      <Layers size={28} strokeWidth={1.6} />
      <div class="empty-title">No layers yet</div>
      {#if view?.assets.input}
        <div>Click objects on the image to select them, then create a slice. Or split the whole image by depth.</div>
      {:else}
        <div>Layers appear here once you load an image and cut it into slices.</div>
      {/if}
    </div>
  {:else}
    <div class="scroll" bind:clientHeight={scrollHeight}>
      <div class="content" style={`height: ${Math.max(RULER_TOP + axisHeight + 28, LIST_TOP + rows.length * ROW_PITCH + 48)}px`}>
        <DepthRuler {entries} {axisHeight} />
        <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
        <div
          class="list"
          role="listbox"
          aria-label="Slices, nearest first"
          tabindex="-1"
          style={`left: ${RULER_WIDTH + 6}px; top: ${LIST_TOP}px`}
          onkeydown={onListKeydown}
        >
          {#each rows as slice (slice.index)}
            {@const selected = view?.selectedSlice === slice.index}
            <div
              class="row"
              role="option"
              tabindex="0"
              aria-selected={selected}
              data-selected={selected}
              data-testid="slice-thumbnail-wrapper"
              data-slice-index={slice.index}
              onclick={() => onRowClick(slice.index)}
              onkeydown={(event) => {
                if (event.target !== event.currentTarget) return; // e.g. the depth chip
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  onRowClick(slice.index);
                }
              }}
            >
              <img
                class="thumb"
                style={`width: ${thumbWidth}px`}
                data-testid="slice-thumbnail"
                alt={`image_slice_${slice.index}`}
                src={slice.thumbnail.url}
                draggable="false"
              />
              <div class="meta">
                <div class="name mono">image_slice_{slice.index}</div>
                <div class="badges">
                  {#if slice.isGround}
                    <span class="badge-ground" data-testid="ground-badge">GROUND</span>
                  {/if}
                </div>
              </div>
              {#if editingIndex === slice.index}
                <!-- svelte-ignore a11y_autofocus -->
                <input
                  type="number"
                  class="depth-input mono"
                  min="0"
                  max="255"
                  data-testid="slice-depth-input"
                  aria-label={`Depth of image_slice_${slice.index}`}
                  autofocus
                  value={depthDraft}
                  oninput={(event) => (depthDraft = (event.currentTarget as HTMLInputElement).value)}
                  onclick={(event) => event.stopPropagation()}
                  onkeydown={onDepthKeydown}
                  onblur={() => commitDepth(slice.index)}
                />
              {:else}
                <button
                  type="button"
                  class="chip mono"
                  data-testid="slice-depth-display"
                  title="Edit depth"
                  onclick={(event) => startEditing(slice.index, slice.depth, event)}
                >
                  {slice.depth}
                </button>
              {/if}
            </div>
          {/each}
        </div>
        <p class="hint" style={`left: ${RULER_WIDTH + 14}px; top: ${LIST_TOP + rows.length * ROW_PITCH + 10}px`}>
          Drag a handle on the ruler, or click a depth, to change it.
        </p>
      </div>
    </div>
  {/if}
</aside>

<style>
  .layer-panel {
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    background: var(--color-surface);
    border-right: 1px solid var(--color-border);
  }

  .panel-header {
    height: var(--viewbar-h);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 var(--space-2) 0 14px;
    border-bottom: 1px solid var(--color-border);
  }

  .panel-header .label {
    margin: 0;
  }

  .header-actions {
    display: flex;
    gap: 2px;
  }

  .empty {
    flex: 1 1 auto;
    padding: 28px 22px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    color: var(--color-text-secondary);
    line-height: 1.5;
  }

  .empty :global(svg) {
    color: var(--color-text-muted);
  }

  .empty-title {
    font-weight: 600;
    color: var(--color-text);
  }

  .scroll {
    position: relative;
    flex: 1 1 0;
    min-height: 0;
    overflow: auto;
  }

  .content {
    position: relative;
  }

  .list {
    position: absolute;
    right: 6px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    outline: none;
  }

  .row {
    height: 64px;
    box-sizing: border-box;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: 6px;
    border-radius: var(--radius-md);
    cursor: pointer;
  }

  .row:hover {
    background: var(--color-surface-hover);
  }

  .row[aria-selected='true'] {
    background: var(--color-selection-soft);
    box-shadow: inset 0 0 0 1px var(--color-selection);
  }

  .row:focus-visible {
    outline: 2px solid var(--color-selection);
    outline-offset: -2px;
  }

  .thumb {
    height: 48px;
    flex-shrink: 0;
    object-fit: cover;
    border-radius: var(--radius-sm);
    background: var(--color-checker-1);
  }

  .meta {
    flex: 1 1 auto;
    min-width: 0;
  }

  .name {
    font-size: 12px;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .badges {
    display: flex;
    gap: var(--space-1);
    margin-top: 3px;
    min-height: 16px;
  }

  .badge-ground {
    font-size: 10.5px;
    font-weight: 600;
    color: var(--color-success);
    border: 1px solid var(--color-success-line);
    border-radius: var(--radius-sm);
    padding: 0 4px;
    letter-spacing: 0.03em;
  }

  .chip {
    font-size: 12px;
    color: var(--color-text);
    background: var(--color-surface-hover);
    border: none;
    border-radius: var(--radius-sm);
    padding: 2px 6px;
    cursor: text;
  }

  .chip:focus-visible {
    outline: 2px solid var(--color-selection);
  }

  .depth-input {
    width: 52px;
    font-size: 12px;
    padding: 2px 4px;
    background: var(--color-surface-raised);
    color: var(--color-text);
    border: 1px solid var(--color-selection);
    border-radius: var(--radius-sm);
  }

  .hint {
    position: absolute;
    right: 12px;
    margin: 0;
    font-size: var(--text-small);
    color: var(--color-text-muted);
  }
</style>
