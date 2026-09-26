<script lang="ts" module>
  import { jobLabel } from '../../state/jobs.svelte';

  /** Human-readable label for a local job kind or a server-reported busy kind. */
  export const activityLabel = jobLabel;

  /** Quick operations finish before this, so they never flash the indicator. */
  export const SHOW_DELAY_MS = 250;
</script>

<script lang="ts">
  import { jobStore } from '../../state/jobs.svelte';
  import { projectStore } from '../../state/project.svelte';

  // A locally started job wins; otherwise reflect a job the server reports
  // (e.g. one started from another browser tab).
  const kind = $derived(jobStore.active ?? projectStore.view?.busy?.kind ?? null);
  const progress = $derived(jobStore.active ? jobStore.progress : 0);
  const percent = $derived(Math.round(Math.min(1, Math.max(0, progress)) * 100));

  let visible = $state(false);
  $effect(() => {
    if (kind === null) {
      visible = false;
      return;
    }
    const timer = setTimeout(() => (visible = true), SHOW_DELAY_MS);
    return () => clearTimeout(timer);
  });
</script>

<div class="activity" role="status" aria-live="polite" data-testid="activity-indicator" data-active={visible}>
  {#if visible && kind !== null}
    <span class="spinner" aria-hidden="true"></span>
    <span class="activity-label" data-testid="activity-label">
      {activityLabel(kind)}{percent > 0 ? ` ${percent}%` : '…'}
    </span>
    <span
      class="activity-bar"
      class:indeterminate={percent === 0}
      data-testid="activity-bar"
      role="progressbar"
      aria-label={activityLabel(kind)}
      aria-valuemin="0"
      aria-valuemax="100"
      aria-valuenow={percent > 0 ? percent : undefined}
    >
      <span class="activity-bar-fill" style={percent > 0 ? `width: ${percent}%` : undefined}></span>
    </span>
    {#if jobStore.active && (jobStore.cancellable || jobStore.cancelling)}
      <button
        type="button"
        class="btn btn-ghost btn-sm cancel"
        data-testid="activity-cancel"
        disabled={jobStore.cancelling}
        onclick={() => void jobStore.cancel()}
      >
        {jobStore.cancelling ? 'Cancelling…' : 'Cancel'}
      </button>
    {/if}
  {/if}
</div>

<style>
  .activity {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    white-space: nowrap;
    min-width: 0;
  }

  .activity-label {
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .cancel {
    height: 22px;
    font-size: 12px;
  }

  .spinner {
    width: 12px;
    height: 12px;
    flex: none;
    border-radius: 50%;
    border: 1.6px solid currentColor;
    border-right-color: transparent;
    animation: spin 0.9s linear infinite;
  }

  .activity-bar {
    display: inline-block;
    flex: none;
    width: 120px;
    height: 4px;
    border-radius: 2px;
    overflow: hidden;
    background: var(--color-surface-hover);
  }

  .activity-bar-fill {
    display: block;
    height: 100%;
    background: var(--color-primary);
    border-radius: 2px;
    transition: width 0.2s ease;
  }

  .activity-bar.indeterminate .activity-bar-fill {
    width: 30%;
    animation: slide 1.2s ease-in-out infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  @keyframes slide {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(340%);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .spinner,
    .activity-bar.indeterminate .activity-bar-fill {
      animation-duration: 3s;
    }
  }
</style>
