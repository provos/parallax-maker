<script lang="ts">
  /**
   * The full project log, newest at the bottom, in a drawer above the
   * status bar (docs/redesign/HANDOFF.md §2). It stays mounted while closed
   * so the log is always readable (the e2e driver reads `data-testid="log"`).
   */
  import X from '@lucide/svelte/icons/x';
  import { uiStore } from '../../state/ui.svelte';
  import { logStore } from '../../state/logs.svelte';

  let listEl: HTMLDivElement | undefined = $state();

  // Keep the newest entry in view as entries arrive.
  $effect(() => {
    void logStore.entries.length;
    if (listEl) listEl.scrollTop = listEl.scrollHeight;
  });
</script>

<section class="log-drawer" class:hidden={!uiStore.logOpen} aria-label="Log" data-testid="log-drawer">
  <div class="log-header">
    <span class="label">Log</span>
    <button
      type="button"
      class="btn btn-ghost btn-sm btn-icon"
      aria-label="Close log"
      onclick={() => uiStore.toggleLog()}
    >
      <X size={14} strokeWidth={1.6} />
    </button>
  </div>
  <div class="log-list mono" role="log" data-testid="log" bind:this={listEl}>
    {#each logStore.entries as entry (entry.seq)}
      <div class="log-entry" class:log-error={entry.level === 'error'}>{entry.message}</div>
    {/each}
  </div>
</section>

<style>
  .log-drawer {
    height: 180px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    background: var(--color-surface);
    border-top: 1px solid var(--color-border);
  }

  .log-header {
    height: 32px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    padding: 0 var(--space-2) 0 14px;
    border-bottom: 1px solid var(--color-border);
  }

  .log-header .label {
    flex: 1 1 auto;
    margin: 0;
  }

  .log-list {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    padding: 6px 14px;
    font-size: 12px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .log-error {
    color: var(--color-danger);
  }
</style>
