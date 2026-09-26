<script lang="ts">
  /**
   * The Ground step's Inspector panel (docs/redesign/HANDOFF.md §3): which
   * slice is the horizontal ground plane, Fit ground, the horizon and camera
   * pitch, the ground distance, and a side view of the scene.
   */
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import { cameraDraftStore } from '../../state/cameraDraft.svelte';
  import * as workflow from '../../workflow';
  import CameraSlider from '../shared/CameraSlider.svelte';
  import SceneSideView from './SceneSideView.svelte';

  const view = $derived(projectStore.view);
  // Nearest first, like the layer panel.
  const slices = $derived([...(view?.slices ?? [])].sort((a, b) => b.depth - a.depth || b.index - a.index));
  const ground = $derived(view?.slices.find((s) => s.isGround) ?? null);

  $effect(() => cameraDraftStore.sync(view?.settings));
</script>

<div class="panel-body" data-testid="ground-panel">
  <section class="sec">
    <div class="sec-title">
      <h3 class="label accent">Ground plane</h3>
    </div>
    <p class="faint">
      A horizontal ground (sea, floor, field) instead of an upright card. Pick the slice that holds it.
    </p>
    <div class="choices" role="radiogroup" aria-label="Ground slice" data-testid="ground-choices">
      <label class="choice">
        <input
          type="radio"
          name="ground-slice"
          data-testid="ground-choice-none"
          checked={ground === null}
          disabled={isBusy() || !view}
          onchange={() => void workflow.setGroundSlice(null)}
        />
        <span>No ground plane</span>
      </label>
      {#each slices as slice (slice.index)}
        <label class="choice">
          <input
            type="radio"
            name="ground-slice"
            data-testid="ground-choice"
            data-slice-index={slice.index}
            checked={!!slice.isGround}
            disabled={isBusy()}
            onchange={() => void workflow.setGroundSlice(slice.index)}
          />
          <img class="thumb" src={slice.thumbnail.url} alt="" />
          <span class="mono">image_slice_{slice.index}</span>
          <span class="faint depth">{slice.depth}</span>
        </label>
      {/each}
    </div>
    <button
      type="button"
      class="btn"
      class:btn-primary={!!ground}
      data-testid="ground-fit"
      title="Put the horizon on the ground's top edge and the ground under the nearest object"
      disabled={isBusy() || !ground}
      onclick={() => void workflow.fitGround()}
    >
      Fit ground
    </button>
  </section>

  <section class="sec">
    <h3 class="label">Horizon &amp; distance</h3>
    <p class="readout" data-testid="horizon-readout">
      {#if view?.settings.camera.horizonRow != null}
        Horizon at row {Math.round(view.settings.camera.horizonRow)}, camera pitch
        {(view.settings.camera.pitch ?? 0).toFixed(1)}°
      {:else}
        No image loaded
      {/if}
    </p>
    <p class="faint">Drag the horizon line on the image (Horizon tool) to tilt the camera.</p>
    <CameraSlider
      field="groundNear"
      label="Ground distance"
      testId="ground-distance"
      min={0}
      max={Math.max(0, cameraDraftStore.draft.maxDistance - 1)}
      disabled={!view || !ground}
    />
    {#if view?.sceneProfile}
      <SceneSideView profile={view.sceneProfile} />
    {/if}
  </section>
</div>

<style>
  .panel-body {
    display: flex;
    flex-direction: column;
    margin: calc(-1 * var(--space-3)) -14px;
  }

  .sec {
    padding: var(--space-3) 14px;
    border-bottom: 1px solid var(--color-border);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .label {
    margin: 0;
  }

  .accent {
    color: var(--color-primary-soft-text);
  }

  .faint {
    margin: 0;
    color: var(--color-text-muted);
    font-size: var(--text-small);
    line-height: 1.45;
  }

  .readout {
    margin: 0;
  }

  .choices {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .choice {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: 4px 6px;
    border-radius: var(--radius-md);
    cursor: pointer;
  }

  .choice:hover {
    background: var(--color-surface-hover);
  }

  .thumb {
    height: 24px;
    width: 32px;
    object-fit: cover;
    border-radius: var(--radius-sm);
  }

  .depth {
    margin-left: auto;
  }
</style>
