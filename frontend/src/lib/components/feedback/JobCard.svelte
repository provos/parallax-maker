<script lang="ts">
  /**
   * A running job, shown next to the control that started it
   * (docs/redesign/HANDOFF.md §9): label, percent, an optional detail line
   * (e.g. "Loading the depth model") and Cancel where the job can stop
   * safely. After a failure of one of `kinds` it shows an error card with
   * the same actions as the error toast, until the next job starts. Hidden
   * otherwise.
   */
  import { jobStore, type JobKind } from '../../state/jobs.svelte';
  import X from '@lucide/svelte/icons/x';
  import { jobLabel as activityLabel } from '../../state/jobs.svelte';

  let { kinds, testId }: { kinds: JobKind[]; testId?: string } = $props();

  const running = $derived(jobStore.active !== null && kinds.includes(jobStore.active));
  const error = $derived(!running && jobStore.lastError && kinds.includes(jobStore.lastError.kind) ? jobStore.lastError : null);
  const percent = $derived(Math.round(Math.min(1, Math.max(0, jobStore.progress)) * 100));
</script>

<div class="job-card" class:idle={!running && !error} class:failed={!!error} data-testid={testId} role="status" aria-live="polite">
  {#if running && jobStore.active}
    <div class="head">
      <span class="spinner" aria-hidden="true"></span>
      <span class="job-label" data-testid="job-label">{activityLabel(jobStore.active)}</span>
      <span class="mono pct">{percent}%</span>
      {#if jobStore.cancellable || jobStore.cancelling}
        <button
          type="button"
          class="btn btn-ghost btn-sm"
          data-testid="job-cancel"
          disabled={jobStore.cancelling}
          onclick={() => void jobStore.cancel()}
        >
          {jobStore.cancelling ? 'Cancelling…' : 'Cancel'}
        </button>
      {/if}
    </div>
    <div class="progress-bar"><div class="progress-bar-fill" style={`width: ${percent}%`}></div></div>
    {#if jobStore.detail}
      <span class="detail" data-testid="job-detail">{jobStore.detail}</span>
    {/if}
  {:else if error}
    <div class="head">
      <span class="job-label error-title" data-testid="job-error">{error.title}</span>
      <button
        type="button"
        class="btn btn-ghost btn-icon dismiss"
        aria-label="Dismiss"
        data-testid="job-error-dismiss"
        onclick={() => jobStore.clearError()}
      >
        <X size={14} strokeWidth={1.6} />
      </button>
    </div>
    <span class="detail">{error.message}</span>
    <div class="actions">
      {#each error.actions as action (action.label)}
        <button type="button" class="btn btn-sm" data-testid="job-error-action" onclick={() => action.run()}>
          {action.label}
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .job-card {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px;
    border-radius: var(--radius-md);
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
  }

  .job-card.idle {
    display: none;
  }

  .job-card.failed {
    border-color: var(--color-danger-line);
    background: var(--color-danger-soft);
  }

  .error-title {
    color: var(--color-danger);
    font-weight: 600;
  }

  .detail {
    overflow-wrap: anywhere;
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .dismiss {
    width: 24px;
    height: 24px;
  }

  .head {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: var(--control-h-sm);
  }

  .job-label {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .pct {
    font-size: 12px;
    color: var(--color-text-secondary);
  }

  .detail {
    font-size: var(--text-small);
    color: var(--color-text-muted);
  }

  .spinner {
    width: 12px;
    height: 12px;
    flex-shrink: 0;
    border-radius: 50%;
    border: 2px solid var(--color-border-strong);
    border-top-color: var(--color-primary);
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .spinner {
      animation: none;
    }
  }
</style>
