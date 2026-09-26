<script lang="ts">
  /**
   * Top of the Inspector while a slice is selected (docs/redesign/HANDOFF.md
   * §3): which slice the panel below acts on, with Download and Replace
   * image for it.
   */
  import Download from '@lucide/svelte/icons/download';
  import ImageUp from '@lucide/svelte/icons/image-up';
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import * as api from '../../api/client';
  import { triggerDownload } from '../../download';

  const view = $derived(projectStore.view);
  const slice = $derived(view?.slices.find((s) => s.index === view.selectedSlice) ?? null);

  let fileInput: HTMLInputElement | undefined = $state();

  function download(): void {
    if (!view || !slice) return;
    // A real `<a download>` click against the raw-slice-PNG endpoint.
    triggerDownload(api.getSliceDownloadUrl(view.id, slice.index), `image_slice_${slice.index}.png`);
  }

  function onFileChange(event: Event): void {
    const target = event.currentTarget as HTMLInputElement;
    const file = target.files?.[0];
    target.value = '';
    if (!file || !slice || isBusy()) return;
    void workflow.uploadSliceImage(slice.index, file);
  }
</script>

{#if slice}
  <div class="selected-slice" data-testid="selected-slice-header">
    <img class="thumb" src={slice.thumbnail.url} alt="" />
    <div class="meta">
      <div class="name mono">image_slice_{slice.index}</div>
      <div class="details">
        <span class="chip mono" title="Depth">{slice.depth}</span>
        {#if slice.isGround}<span class="badge-ground">GROUND</span>{/if}
      </div>
    </div>
    <button
      type="button"
      class="btn btn-ghost btn-icon"
      data-testid="slice-download"
      aria-label="Download slice image"
      title="Download slice image"
      onclick={download}
    >
      <Download size={16} strokeWidth={1.6} />
    </button>
    <button
      type="button"
      class="btn btn-ghost btn-icon"
      data-testid="slice-replace"
      aria-label="Replace slice image…"
      title="Replace slice image…"
      disabled={isBusy()}
      onclick={() => fileInput?.click()}
    >
      <ImageUp size={16} strokeWidth={1.6} />
    </button>
    <input
      bind:this={fileInput}
      type="file"
      accept="image/*"
      class="sr-only"
      data-testid="slice-upload-input"
      onchange={onFileChange}
    />
  </div>
{/if}

<style>
  .selected-slice {
    position: relative;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-bottom: 1px solid var(--color-border);
  }

  .thumb {
    height: 40px;
    max-width: 64px;
    object-fit: cover;
    border-radius: var(--radius-sm);
  }

  .meta {
    flex: 1 1 auto;
    min-width: 0;
  }

  .name {
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .details {
    display: flex;
    gap: var(--space-1);
    align-items: center;
    margin-top: 2px;
  }

  .chip {
    font-size: 12px;
    background: var(--color-surface-hover);
    border-radius: var(--radius-sm);
    padding: 1px 6px;
  }

  .badge-ground {
    font-size: 10.5px;
    font-weight: 600;
    color: var(--color-success);
    border: 1px solid var(--color-success-line);
    border-radius: var(--radius-sm);
    padding: 0 4px;
    letter-spacing: 0.03em;
  }
</style>
