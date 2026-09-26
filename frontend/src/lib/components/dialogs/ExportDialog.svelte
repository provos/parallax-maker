<script lang="ts">
  /**
   * The Export dialog (docs/redesign/HANDOFF.md §3), opened by the Export
   * step, the header's Export button or Ctrl+E: the 3D scene (camera
   * mirrored from Preview, mesh displacement, depth of field, the ground,
   * texture upscaling, Create / Download glTF) and the animation.
   */
  import { projectStore } from '../../state/project.svelte';
  import { jobStore } from '../../state/jobs.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import { triggerDownload } from '../../download';
  import { cameraDraftStore } from '../../state/cameraDraft.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import CameraSlider from '../shared/CameraSlider.svelte';
  import Dialog from '../shared/Dialog.svelte';
  import Box from '@lucide/svelte/icons/box';
  import Clapperboard from '@lucide/svelte/icons/clapperboard';
  import Check from '@lucide/svelte/icons/check';

  type ExportKind = 'gltf' | 'animation';
  let kind = $state<ExportKind>('gltf');

  const view = $derived(projectStore.view);
  const ground = $derived(view?.slices.find((s) => s.isGround) ?? null);
  const gltfReady = $derived(!!view?.exports.gltf);

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
  const seconds = $derived((numFrames / 30).toFixed(1));
  const animating = $derived(jobStore.active === 'animation');

  function exportAnimation(): void {
    if (!view || isBusy() || numFrames <= 0) return;
    void workflow.startAnimationExport(numFrames);
  }

  function viewIn3d(): void {
    uiStore.setView('3d');
    uiStore.closeDialog();
  }

</script>

<Dialog name="export" title="Export" testId="export-dialog" width={680}>
  <div class="body">
    <div class="kinds" role="tablist" aria-orientation="vertical" aria-label="Export type">
      <button
        type="button"
        role="tab"
        class="kind"
        aria-selected={kind === 'gltf'}
        data-testid="export-tab-gltf"
        onclick={() => (kind = 'gltf')}
      >
        <Box size={16} strokeWidth={1.6} />
        <span><b>3D scene</b><br /><span class="faint">glTF for Blender, Unreal</span></span>
      </button>
      <button
        type="button"
        role="tab"
        class="kind"
        aria-selected={kind === 'animation'}
        data-testid="export-tab-animation"
        onclick={() => (kind = 'animation')}
      >
        <Clapperboard size={16} strokeWidth={1.6} />
        <span><b>Animation</b><br /><span class="faint">Parallax camera move</span></span>
      </button>
    </div>

    <div class="pane" role="tabpanel" data-testid="tab-export" hidden={kind !== 'gltf'}>
      <CameraSlider field="distance" label="Camera dist." testId="export-camera-distance" min={0} max={500} disabled={!view} />
      <CameraSlider field="maxDistance" label="Max dist." testId="export-max-distance" min={0} max={1000} disabled={!view} />
      <CameraSlider field="focalLength" label="Focal len." testId="export-focal-length" min={1} max={500} disabled={!view} />
      <CameraSlider field="displacement" label="Displace" testId="displacement" min={0} max={150} step={5} disabled={!view} />
      <label class="checkbox-row">
        <input
          type="checkbox"
          data-testid="toggle-dof"
          checked={dofEnabled}
          onchange={(event) => (dofEnabled = (event.currentTarget as HTMLInputElement).checked)}
        />
        Support depth-of-field effect
      </label>
      <div class="summary" data-testid="export-ground-summary">
        <span class="grow">
          {#if ground}
            Ground plane: <span class="mono">image_slice_{ground.index}</span> · distance
            <span class="mono">{Math.round(view?.settings.camera.groundNear ?? 0)}</span>
          {:else}
            No ground plane
          {/if}
        </span>
        <button type="button" class="btn btn-ghost btn-sm" data-testid="export-edit-ground" onclick={() => uiStore.setStep('ground')}>
          Edit
        </button>
      </div>
      <div class="row">
        <button
          type="button"
          class="btn"
          data-testid="upscale-textures"
          title="Upscale slice textures so you can zoom in further"
          disabled={isBusy() || !view}
          onclick={upscaleTextures}
        >
          Upscale textures
        </button>
        {#if view?.exports.upscaled}<span class="faint">Textures upscaled</span>{/if}
      </div>
      <span class="spacer"></span>
      <div class="progress-bar" data-testid="export-progress" class:idle={!(creatingGltf || upscaling)}>
        <div
          class="progress-bar-fill"
          style={`width: ${(creatingGltf || upscaling) ? Math.round(jobStore.progress * 100) : 0}%`}
        ></div>
      </div>
      <div class="row end">
        {#if gltfReady}
          <span class="ready grow"><Check size={16} strokeWidth={1.6} /> Scene ready</span>
          <button type="button" class="btn" data-testid="export-view-3d" onclick={viewIn3d}>View in 3D</button>
        {/if}
        <button
          type="button"
          class="btn"
          data-testid="gltf-create"
          title="Build the scene without downloading it"
          disabled={isBusy() || !view}
          onclick={createGltfScene}
        >
          Create scene
        </button>
        <button
          type="button"
          class="btn btn-primary"
          data-testid="gltf-export"
          title="Build the scene and download scene.gltf"
          disabled={isBusy() || !view}
          onclick={() => void exportGltfScene()}
        >
          Download glTF
        </button>
      </div>
    </div>

    <div class="pane" role="tabpanel" hidden={kind !== 'animation'}>
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
      <p class="faint">
        Renders a camera move across the scene with the current camera settings. At 30 fps, {numFrames} frames is
        about {seconds} s.
      </p>
      <span class="spacer"></span>
      <div class="progress-bar" data-testid="animation-progress" class:idle={!animating}>
        <div class="progress-bar-fill" style={`width: ${animating ? Math.round(jobStore.progress * 100) : 0}%`}></div>
      </div>
      <div class="row end">
        <button
          type="button"
          class="btn btn-primary"
          data-testid="animation-export"
          disabled={isBusy() || !view || numFrames <= 0}
          onclick={exportAnimation}
        >
          Render animation
        </button>
      </div>
    </div>
  </div>
</Dialog>

<style>
  .body {
    display: flex;
    flex: 1 1 auto;
    min-height: 380px;
    min-width: 0;
  }

  .kinds {
    width: 190px;
    flex-shrink: 0;
    border-right: 1px solid var(--color-border);
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .kind {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px;
    border: none;
    border-radius: var(--radius-md);
    background: transparent;
    color: var(--color-text);
    font: inherit;
    text-align: left;
    cursor: pointer;
  }

  .kind:hover {
    background: var(--color-surface-hover);
  }

  .kind[aria-selected='true'] {
    background: var(--color-selection-soft);
    box-shadow: inset 0 0 0 1px var(--color-selection);
  }

  .kind b {
    font-weight: 600;
  }

  .pane {
    flex: 1 1 auto;
    min-width: 0;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .pane[hidden] {
    display: none;
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

  .summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: 8px 10px;
    border-radius: var(--radius-md);
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
    color: var(--color-text-secondary);
  }

  .row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  .end {
    justify-content: flex-end;
  }

  .grow {
    flex: 1 1 auto;
  }

  .spacer {
    flex: 1 1 auto;
  }

  .ready {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--color-success);
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
