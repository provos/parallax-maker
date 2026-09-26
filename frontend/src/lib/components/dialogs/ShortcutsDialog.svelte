<script lang="ts">
  /** The keyboard shortcuts (docs/redesign/HANDOFF.md §6), opened with `?`. */
  import Dialog from '../shared/Dialog.svelte';
  import { SHORTCUTS } from '../../shortcuts';
</script>

<Dialog name="shortcuts" title="Keyboard shortcuts" testId="shortcuts-dialog" width={640}>
  <dl class="grid">
    {#each SHORTCUTS as { keys, action } (keys)}
      <div class="item">
        <dt class="muted">{action}</dt>
        <dd><span class="kbd">{keys}</span></dd>
      </div>
    {/each}
  </dl>
</Dialog>

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0 28px;
    margin: 0;
    padding: 12px 18px 18px;
    width: 100%;
    overflow-y: auto;
  }

  .item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    padding: 5px 0;
    border-bottom: 1px solid var(--color-border);
  }

  dt,
  dd {
    margin: 0;
  }

  dd {
    white-space: nowrap;
  }

  .muted {
    color: var(--color-text-secondary);
  }
</style>
