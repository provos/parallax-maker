<script lang="ts" module>
  import type { JobKind } from '../../state/jobs.svelte';

  const LABELS: Partial<Record<string, string>> = {
    upload: 'Uploading image',
    depth: 'Generating depth map',
    slices: 'Generating slices',
    restore: 'Restoring project',
    segmentation: 'Segmenting',
    'multi-point': 'Segmenting',
    inpainting: 'Generating inpainting candidates',
    'inpainting-mutate': 'Updating slice',
    'slice-editing': 'Updating slices',
    'mask-tools': 'Updating mask',
    save: 'Saving project',
    'export-gltf': 'Exporting glTF scene',
    upscale: 'Upscaling textures',
    animation: 'Rendering animation',
    navigate: 'Rendering view',
    probe: 'Testing connection',
    'validate-key': 'Validating API key',
  };

  /** Human-readable label for a local job kind or a server-reported busy kind. */
  export function activityLabel(kind: JobKind | string): string {
    return LABELS[kind] ?? 'Working';
  }

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
    <span class="label" data-testid="activity-label">
      {activityLabel(kind)}{percent > 0 ? ` ${percent}%` : '…'}
    </span>
  {/if}
</div>

{#if visible && kind !== null}
  <div
    class="activity-bar"
    class:indeterminate={percent === 0}
    data-testid="activity-bar"
    role="progressbar"
    aria-label={activityLabel(kind)}
    aria-valuemin="0"
    aria-valuemax="100"
    aria-valuenow={percent > 0 ? percent : undefined}
  >
    <div class="activity-bar-fill" style={percent > 0 ? `width: ${percent}%` : undefined}></div>
  </div>
{/if}

<style>
  .activity {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
    font-size: 0.875rem;
    white-space: nowrap;
    min-width: 0;
  }

  .label {
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .spinner {
    width: 1rem;
    height: 1rem;
    flex: none;
    border-radius: 50%;
    border: 2px solid currentColor;
    border-right-color: transparent;
    animation: spin 0.8s linear infinite;
  }

  /* Thin bar along the bottom edge of the header (see Header.svelte). */
  .activity-bar {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 4px;
    overflow: hidden;
    background-color: rgb(255 255 255 / 25%);
  }

  .activity-bar-fill {
    height: 100%;
    background-color: var(--color-success);
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
