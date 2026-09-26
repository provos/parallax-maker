<script lang="ts">
  /**
   * The canvas tools, floating at the canvas's left edge (docs/redesign/
   * HANDOFF.md §5): Pan, Segment, Brush, Extend (planned) and Horizon.
   * Picking a tool also moves the workflow to that tool's step.
   */
  import Hand from '@lucide/svelte/icons/hand';
  import MousePointerClick from '@lucide/svelte/icons/mouse-pointer-click';
  import Paintbrush from '@lucide/svelte/icons/paintbrush';
  import Expand from '@lucide/svelte/icons/expand';
  import MoveHorizontal from '@lucide/svelte/icons/move-horizontal';
  import { uiStore, type CanvasTool } from '../../state/ui.svelte';
  import { projectStore } from '../../state/project.svelte';

  type ToolButton = {
    tool: CanvasTool | 'extend';
    label: string;
    key: string;
    icon: typeof Hand;
  };

  const TOOLS: ToolButton[] = [
    { tool: 'pan', label: 'Pan', key: 'H', icon: Hand },
    { tool: 'segment', label: 'Segment', key: 'S', icon: MousePointerClick },
    { tool: 'brush', label: 'Inpaint brush', key: 'B', icon: Paintbrush },
    { tool: 'extend', label: 'Extend edges (coming soon)', key: 'O', icon: Expand },
    { tool: 'horizon', label: 'Horizon', key: 'G', icon: MoveHorizontal },
  ];

  const hasImage = $derived(!!projectStore.view?.assets.input);
  const hasSelection = $derived(projectStore.view?.selectedSlice != null);

  function enabled(tool: ToolButton['tool']): boolean {
    if (tool === 'extend' || !hasImage) return false;
    if (tool === 'brush') return hasSelection;
    return true;
  }
</script>

<div class="canvas-toolbar" role="toolbar" aria-label="Canvas tools" aria-orientation="vertical" data-testid="canvas-toolbar">
  {#each TOOLS as { tool, label, key, icon: Icon } (tool)}
    <button
      type="button"
      class="btn btn-ghost btn-icon tool"
      data-testid={`tool-${tool}`}
      aria-label={label}
      aria-pressed={uiStore.tool === tool}
      title={`${label} (${key})`}
      disabled={!enabled(tool)}
      onclick={() => tool !== 'extend' && uiStore.setTool(tool)}
    >
      <Icon size={16} strokeWidth={1.6} />
    </button>
  {/each}
</div>

<style>
  .canvas-toolbar {
    position: absolute;
    left: var(--space-3);
    top: var(--space-3);
    z-index: 5;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: var(--space-1);
    background: var(--color-float);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-float);
  }

  .tool[aria-pressed='true'] {
    background: var(--color-selection-soft);
    border-color: var(--color-selection);
    color: var(--color-selection-text);
  }
</style>
