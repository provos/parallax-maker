<script lang="ts">
  import { logStore } from '../../state/logs.svelte';
</script>

<div class="log-panel-outer">
  <div class="log-panel panel" data-testid="log">
    {#each logStore.last3 as entry (entry.seq)}
      <div class="log-entry" class:log-error={entry.level === 'error'}>{entry.message}</div>
    {/each}
  </div>
</div>

<style>
  .log-panel-outer {
    padding: var(--space-2);
  }

  .log-panel {
    height: 6rem;
    overflow-y: auto;
  }

  .log-entry {
    font-size: 0.875rem;
    line-height: 1.4;
  }

  .log-error {
    color: var(--color-danger-strong);
  }
</style>
