<script lang="ts">
  /**
   * The bottom bar (docs/redesign/HANDOFF.md §2): the running job with its
   * progress, the latest log message, the image size, and the log toggle.
   */
  import SquareTerminal from '@lucide/svelte/icons/square-terminal';
  import { uiStore } from '../../state/ui.svelte';
  import { logStore } from '../../state/logs.svelte';
  import { projectStore } from '../../state/project.svelte';
  import ActivityIndicator from './ActivityIndicator.svelte';

  const last = $derived(logStore.entries.at(-1) ?? null);
  const size = $derived(projectStore.view?.image ?? null);
</script>

<footer class="status-bar" data-testid="status-bar">
  <ActivityIndicator />
  {#if last}
    <span class="message" class:error={last.level === 'error'} data-testid="status-message">{last.message}</span>
  {/if}
  <span class="spacer"></span>
  {#if size}
    <span class="mono size" data-testid="image-size">{size.width} × {size.height}</span>
  {/if}
  <button
    type="button"
    class="btn btn-ghost btn-sm toggle"
    data-testid="log-toggle"
    aria-pressed={uiStore.logOpen}
    onclick={() => uiStore.toggleLog()}
  >
    <SquareTerminal size={14} strokeWidth={1.6} />
    Log · {logStore.entries.length}
  </button>
  <span class="copyright" data-testid="app-footer">&copy; 2024 Niels Provos</span>
</footer>

<style>
  .status-bar {
    height: var(--statusbar-h);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 var(--space-3);
    background: var(--color-surface);
    border-top: 1px solid var(--color-border);
    font-size: 12px;
    color: var(--color-text-secondary);
    min-width: 0;
  }

  .message {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .message.error {
    color: var(--color-danger);
  }

  .spacer {
    flex: 1 1 auto;
  }

  .size {
    font-size: 11px;
    white-space: nowrap;
  }

  .toggle {
    height: 22px;
    font-size: 12px;
  }

  .copyright {
    color: var(--color-text-muted);
    font-size: 11px;
    white-space: nowrap;
  }
</style>
