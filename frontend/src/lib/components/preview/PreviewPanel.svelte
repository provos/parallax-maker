<script lang="ts">
  /**
   * The Preview step's Inspector panel (docs/redesign/HANDOFF.md §3): the
   * 2D/3D switch and the camera that renders the parallax.
   */
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { cameraDraftStore } from '../../state/cameraDraft.svelte';
  import CameraSlider from '../shared/CameraSlider.svelte';

  const view = $derived(projectStore.view);
  const hasSlices = $derived((view?.slices.length ?? 0) > 0);

  $effect(() => cameraDraftStore.sync(view?.settings));
</script>

<div class="panel-body" data-testid="preview-panel">
  <section class="sec">
    <h3 class="label">Preview</h3>
    <div class="seg" role="radiogroup" aria-label="Preview mode">
      <button
        type="button"
        role="radio"
        aria-checked={uiStore.view !== '3d'}
        data-testid="preview-2d"
        disabled={!hasSlices}
        onclick={() => uiStore.setView('parallax')}
      >
        Parallax 2D
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={uiStore.view === '3d'}
        data-testid="preview-3d"
        disabled={!hasSlices}
        onclick={() => uiStore.setView('3d')}
      >
        3D scene
      </button>
    </div>
    <p class="faint">
      In Parallax 2D, move the camera with the pad above the image. The 3D scene can be orbited with the
      mouse.
    </p>
  </section>

  <section class="sec">
    <h3 class="label">Camera</h3>
    <CameraSlider field="distance" label="Distance" testId="camera-distance" min={0} max={500} disabled={!view} />
    <CameraSlider field="maxDistance" label="Max distance" testId="max-distance" min={0} max={1000} disabled={!view} />
    <CameraSlider field="focalLength" label="Focal length" testId="focal-length" min={1} max={500} disabled={!view} />
    <p class="faint">Distance is how far the camera sits from the image; max distance is how deep the farthest slice lies.</p>
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

  .faint {
    margin: 0;
    color: var(--color-text-muted);
    font-size: var(--text-small);
    line-height: 1.45;
  }

  .seg {
    display: flex;
    background: var(--color-surface-raised);
    border-radius: 7px;
    padding: 2px;
    gap: 2px;
    border: 1px solid var(--color-border);
  }

  .seg button {
    flex: 1 1 0;
    height: var(--control-h-sm);
    border: none;
    border-radius: 5px;
    background: transparent;
    color: var(--color-text-secondary);
    font: inherit;
    font-weight: 500;
    cursor: pointer;
  }

  .seg button[aria-checked='true'] {
    background: var(--color-surface-hover);
    color: var(--color-text);
    box-shadow: 0 0 0 1px var(--color-border-strong);
  }

  .seg button:disabled {
    opacity: 0.42;
    cursor: default;
  }
</style>
