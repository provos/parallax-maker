<script lang="ts">
  /**
   * Toasts, top-right of the canvas (docs/redesign/HANDOFF.md §9). Errors
   * stay until dismissed and offer their actions.
   */
  import X from '@lucide/svelte/icons/x';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import Info from '@lucide/svelte/icons/info';
  import { toastStore } from '../../state/toasts.svelte';
</script>

<div class="toast-stack" data-testid="toast-stack" aria-live="polite">
  {#each toastStore.toasts as toast (toast.id)}
    <div class="toast" data-testid="toast" data-kind={toast.kind} role={toast.kind === 'error' ? 'alert' : 'status'}>
      <span class="icon" aria-hidden="true">
        {#if toast.kind === 'success'}
          <CircleCheck size={16} strokeWidth={1.6} />
        {:else if toast.kind === 'error'}
          <CircleAlert size={16} strokeWidth={1.6} />
        {:else}
          <Info size={16} strokeWidth={1.6} />
        {/if}
      </span>
      <div class="body">
        <span class="title">{toast.title}</span>
        {#if toast.message}<span class="message">{toast.message}</span>{/if}
        {#if toast.actions.length > 0}
          <div class="actions">
            {#each toast.actions as action (action.label)}
              <button
                type="button"
                class="btn btn-sm"
                data-testid="toast-action"
                onclick={() => {
                  toastStore.dismiss(toast.id);
                  action.run();
                }}
              >
                {action.label}
              </button>
            {/each}
          </div>
        {/if}
      </div>
      <button
        type="button"
        class="btn btn-ghost btn-icon dismiss"
        aria-label="Dismiss"
        data-testid="toast-dismiss"
        onclick={() => toastStore.dismiss(toast.id)}
      >
        <X size={14} strokeWidth={1.6} />
      </button>
    </div>
  {/each}
</div>

<style>
  .toast-stack {
    position: absolute;
    top: 12px;
    right: 12px;
    z-index: 20;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    width: min(340px, calc(100% - 24px));
    pointer-events: none;
  }

  .toast {
    pointer-events: auto;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 8px 10px 12px;
    background: var(--color-float);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-float);
  }

  .toast[data-kind='success'] .icon {
    color: var(--color-success);
  }

  .toast[data-kind='error'] {
    border-color: var(--color-danger-line);
  }

  .toast[data-kind='error'] .icon {
    color: var(--color-danger);
  }

  .toast[data-kind='info'] .icon {
    color: var(--color-selection);
  }

  .icon {
    display: inline-flex;
    padding-top: 2px;
  }

  .body {
    flex: 1 1 auto;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .title {
    font-weight: 600;
  }

  .message {
    color: var(--color-text-secondary);
    font-size: var(--text-small);
    overflow-wrap: anywhere;
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 4px;
  }

  .dismiss {
    width: 24px;
    height: 24px;
  }
</style>
