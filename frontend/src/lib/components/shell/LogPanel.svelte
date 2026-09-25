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

  /* Exactly the three most recent entries (logStore.last3), no spare
     height: every pixel here is taken from the image area. */
  .log-panel {
    font-size: 0.875rem;
    line-height: 1.4;
    height: calc(3 * 1.4em);
    box-sizing: content-box;
    overflow-y: auto;
  }

  .log-error {
    color: var(--color-danger-strong);
  }
</style>
