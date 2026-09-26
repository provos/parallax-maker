<script lang="ts">
  /**
   * The Export step's Inspector panel: the glTF scene (with mesh
   * displacement and depth of field), texture upscaling, and the parallax
   * animation. The camera itself is set in the Preview panel.
   */
  import { projectStore } from '../../state/project.svelte';
  import { jobStore } from '../../state/jobs.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import { triggerDownload } from '../../download';
  import { cameraDraftStore } from '../../state/cameraDraft.svelte';
  import CameraSlider from '../shared/CameraSlider.svelte';

  const view = $derived(projectStore.view);

  $effect(() => cameraDraftStore.sync(view?.settings));

  // -- Depth of Field checkbox: purely a parameter of the next export
  // request, not a persisted setting (components.py's CHECKLIST_DOF has no
  // "remember" callback of its own).
  let dofEnabled = $state(false);

  // -- Create glTF Scene (WEB-32): renders into the 3D viewer tab (see
  // Model3DViewer.svelte), same as Dash's own
  // `gltf_create` updating the `#model-viewer` iframe - it does not, by
  // itself, switch the active viewer tab either.
  const creatingGltf = $derived(jobStore.active === 'export-gltf');

  function createGltfScene(): void {
    if (!view || isBusy()) return;
    void workflow.startGltfExport(dofEnabled);
  }

  // -- Export glTF Scene (WEB-28): same job as Create, but triggers a real
  // browser download of the resulting `.gltf` file once it lands - an
  // `<a download>` click (not `window.open`), per the migration task.
  async function exportGltfScene(): Promise<void> {
    if (!view || isBusy()) return;
    await workflow.startGltfExport(dofEnabled);
    const url = projectStore.view?.exports.gltf?.url;
    if (url) triggerDownload(url, 'scene.gltf');
  }

  // -- Upscale Textures (WEB-27).
  const upscaling = $derived(jobStore.active === 'upscale');

  function upscaleTextures(): void {
    if (!view || isBusy()) return;
    void workflow.startUpscaleExport();
  }

  // -- Export Animation + Number of Frames (WEB-35): job only, no download
  // (see PARITY.md "Known quirks" - `ANIMATION_OUTPUT` exists in Dash but no
  // `dcc.Download` ever fires). The frame count itself is not a persisted
  // setting (components.py's SLIDER_NUM_FRAMES has no "remember" callback).
  let numFrames = $state(100);
  const animating = $derived(jobStore.active === 'animation');

  function exportAnimation(): void {
    if (!view || isBusy() || numFrames <= 0) return;
    void workflow.startAnimationExport(numFrames);
  }

</script>

<div class="export-panel" data-testid="tab-export">
  <section class="sec">
    <h3 class="label accent">glTF scene</h3>
    <p class="faint">Cards at their depths, for Blender, Unity or a web viewer.</p>
    <CameraSlider
      field="displacement"
      label="Displacement"
      testId="displacement"
      min={0}
      max={150}
      step={5}
      disabled={!view}
    />
    <label class="checkbox-row">
      <input
        type="checkbox"
        data-testid="toggle-dof"
        checked={dofEnabled}
        onchange={(event) => (dofEnabled = (event.currentTarget as HTMLInputElement).checked)}
      />
      Support depth of field
    </label>
    <div class="row">
      <button
        type="button"
        class="btn grow"
        data-testid="gltf-create"
        title="Build the scene and show it in the 3D view"
        disabled={isBusy() || !view}
        onclick={createGltfScene}
      >
        Create scene
      </button>
      <button
        type="button"
        class="btn btn-primary grow"
        data-testid="gltf-export"
        title="Build the scene and download scene.gltf"
        disabled={isBusy() || !view}
        onclick={() => void exportGltfScene()}
      >
        Download glTF
      </button>
    </div>
    <button
      type="button"
      class="btn"
      data-testid="upscale-textures"
      title="Upscale every slice texture before exporting"
      disabled={isBusy() || !view}
      onclick={upscaleTextures}
    >
      Upscale textures
    </button>
    <div class="progress-bar" data-testid="export-progress" class:idle={!(creatingGltf || upscaling)}>
      <div
        class="progress-bar-fill"
        style={`width: ${(creatingGltf || upscaling) ? Math.round(jobStore.progress * 100) : 0}%`}
      ></div>
    </div>
  </section>

  <section class="sec">
    <h3 class="label">Animation</h3>
    <div class="g3">
      <label class="muted" for="number-of-frames">Frames</label>
      <input
        id="number-of-frames"
        type="range"
        min="0"
        max="300"
        step="1"
        data-testid="number-of-frames"
        value={numFrames}
        disabled={isBusy() || !view}
        oninput={(event) => (numFrames = Number((event.currentTarget as HTMLInputElement).value))}
      />
      <span class="mono value">{numFrames}</span>
    </div>
    <button
      type="button"
      class="btn"
      data-testid="animation-export"
      disabled={isBusy() || !view || numFrames <= 0}
      onclick={exportAnimation}
    >
      Export animation
    </button>
    <div class="progress-bar" data-testid="animation-progress" class:idle={!animating}>
      <div class="progress-bar-fill" style={`width: ${animating ? Math.round(jobStore.progress * 100) : 0}%`}></div>
    </div>
  </section>
</div>

<style>
  .export-panel {
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
  }

  .muted {
    color: var(--color-text-secondary);
  }

  .checkbox-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--color-text-secondary);
  }

  .row {
    display: flex;
    gap: var(--space-2);
  }

  .grow {
    flex: 1 1 0;
  }

  .g3 {
    display: grid;
    grid-template-columns: 96px minmax(0, 1fr) 44px;
    align-items: center;
    gap: 10px;
  }

  .value {
    text-align: right;
    font-size: 12px;
  }

  .progress-bar.idle {
    visibility: hidden;
  }
</style>
