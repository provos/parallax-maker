<script lang="ts">
  /**
   * Logo and project name, the workflow stepper, and the global actions:
   * undo/redo for the selected slice, theme, Settings and Export
   * (docs/redesign/HANDOFF.md §3).
   */
  import Undo2 from '@lucide/svelte/icons/undo-2';
  import Redo2 from '@lucide/svelte/icons/redo-2';
  import Moon from '@lucide/svelte/icons/moon';
  import Sun from '@lucide/svelte/icons/sun';
  import Settings from '@lucide/svelte/icons/settings';
  import Download from '@lucide/svelte/icons/download';
  import { uiStore } from '../../state/ui.svelte';
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import WorkflowStepper from './WorkflowStepper.svelte';

  const view = $derived(projectStore.view);
  const selected = $derived(view?.slices.find((slice) => slice.index === view.selectedSlice) ?? null);
  const hasSlices = $derived((view?.slices.length ?? 0) > 0);
</script>

<header class="app-header">
  <div class="brand">
    <svg class="logo" width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
      <rect x="1" y="7" width="12" height="12" rx="3" fill="var(--color-selection)"></rect>
      <rect x="5" y="4" width="12" height="12" rx="3" fill="none" stroke="currentColor" stroke-width="1.5"></rect>
      <rect x="9" y="1" width="12" height="12" rx="3" fill="var(--color-primary)"></rect>
    </svg>
    <h1 data-testid="app-title">Parallax Maker</h1>
    {#if view}
      <span class="project mono" data-testid="project-name" title={view.id}>/ {view.id}</span>
    {/if}
  </div>

  <WorkflowStepper />

  <div class="actions">
    <button
      type="button"
      class="btn btn-ghost btn-icon"
      data-testid="header-undo"
      aria-label="Undo slice change"
      title="Undo slice change"
      disabled={!selected?.canUndo || isBusy()}
      onclick={() => selected && void workflow.undoSlice(selected.index)}
    >
      <Undo2 size={16} strokeWidth={1.6} />
    </button>
    <button
      type="button"
      class="btn btn-ghost btn-icon"
      data-testid="header-redo"
      aria-label="Redo slice change"
      title="Redo slice change"
      disabled={!selected?.canRedo || isBusy()}
      onclick={() => selected && void workflow.redoSlice(selected.index)}
    >
      <Redo2 size={16} strokeWidth={1.6} />
    </button>
    <button
      type="button"
      class="btn btn-ghost btn-icon"
      data-testid="theme-toggle"
      aria-label="Toggle dark mode"
      aria-pressed={uiStore.theme === 'dark'}
      title="Light / dark"
      onclick={() => void workflow.toggleDarkMode()}
    >
      {#if uiStore.theme === 'dark'}
        <Moon size={16} strokeWidth={1.6} />
      {:else}
        <Sun size={16} strokeWidth={1.6} />
      {/if}
    </button>
    <button
      type="button"
      class="btn btn-ghost btn-icon"
      class:btn-selected={uiStore.mainTab === 'Configuration'}
      data-testid="open-settings"
      aria-label="Settings"
      aria-pressed={uiStore.mainTab === 'Configuration'}
      title="Settings"
      onclick={() =>
        uiStore.mainTab === 'Configuration' ? uiStore.setStep(uiStore.step) : uiStore.setMainTab('Configuration')}
    >
      <Settings size={16} strokeWidth={1.6} />
    </button>
    <button
      type="button"
      class="btn btn-primary export"
      data-testid="open-export"
      disabled={!hasSlices}
      onclick={() => uiStore.setStep('export')}
    >
      <Download size={16} strokeWidth={1.6} />
      Export…
    </button>
  </div>
</header>

<style>
  .app-header {
    height: var(--header-h);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: 0 var(--space-3) 0 var(--space-4);
    background: var(--color-surface);
    border-bottom: 1px solid var(--color-border);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
    flex: 0 1 280px;
  }

  .logo {
    flex-shrink: 0;
  }

  h1 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
  }

  .project {
    color: var(--color-text-secondary);
    font-size: var(--text-small);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .actions {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    flex-shrink: 0;
  }

  .export {
    margin-left: var(--space-2);
  }
</style>
