<script lang="ts">
  /**
   * The Export tab: Create/Export glTF Scene, Upscale Textures, the DOF
   * checkbox, camera distance/max distance/focal length/mesh displacement
   * sliders, and Export Animation + Number of Frames - see components.py's
   * `make_3d_export_div`/`make_animation_export_div` and
   * `docs/svelte-migration/reference/dash-1440-export.png` for the Dash
   * layout this mirrors.
   */
  import { projectStore } from '../../state/project.svelte';
  import { jobStore } from '../../state/jobs.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import { triggerDownload } from '../../download';

  const view = $derived(projectStore.view);

  // -- Camera distance / max distance / focal length / mesh displacement.
  // All four are read straight from the persisted settings, with a small
  // local draft for a smooth drag (re-synced whenever the server's settings
  // change, including our own applied response) - same pattern as
  // SegmentationTab.svelte's threshold sliders. `CameraSettingsRequest`
  // requires distance/focalLength/maxDistance together (see
  // `updateSettings`'s doc comment); this always commits all four fields in
  // one `PUT .../settings`, matching Dash's own single `remember_camera_
  // parameters` callback, which reads all four sliders on any one's change.
  type CameraDraft = { distance: number; maxDistance: number; focalLength: number; displacement: number };
  const DEFAULT_CAMERA: CameraDraft = { distance: 100, maxDistance: 200, focalLength: 100, displacement: 0 };

  let draft = $state<CameraDraft>({ ...DEFAULT_CAMERA });

  // `onchange` fires per discrete arrow-key step (see the sliders' own doc
  // comment above), so a fast key-repeat sequence can call `commitCamera`
  // many times before the first `PUT .../settings` round trip lands. Per the
  // architecture doc's "Concurrency" rule ("the UI must never send a request
  // it knows will be rejected with 409 busy"), firing those uncoordinated
  // and concurrently would 409 every commit but the first; instead, coalesce
  // them the same way `logStore.refresh` coalesces overlapping calls
  // (state/logs.svelte.ts): only one `updateSettings` call is ever in
  // flight, later commits just flag a follow-up that reads the *latest*
  // `draft` once the in-flight one resolves, so the final persisted value is
  // always the last one the user actually landed on.
  let commitInFlight = false;
  let commitPending = false;

  // Re-sync `draft` from the persisted settings whenever the project view
  // changes (including a restore) - but never while a local edit is still
  // being committed: `runCommit`'s own `applyView` (inside
  // `workflow.updateSettings`) would otherwise race an in-progress key-repeat
  // sequence, snapping `draft` back to an older, already-superseded value
  // moments after a later key press already advanced it further (observed
  // directly as `setSlider` getting "stuck" partway through a many-step
  // sequence, since each snap-back fights the next arrow-key increment).
  $effect(() => {
    const settings = view?.settings;
    if (!settings || commitInFlight || commitPending) return;
    draft = {
      distance: settings.camera.distance,
      maxDistance: settings.camera.maxDistance,
      focalLength: settings.camera.focalLength,
      displacement: settings.meshDisplacement,
    };
  });

  function onCameraInput(field: keyof CameraDraft, event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    draft = { ...draft, [field]: value };
  }

  async function runCommit(): Promise<void> {
    if (commitInFlight) return;
    commitInFlight = true;
    try {
      while (commitPending) {
        commitPending = false;
        await workflow.updateSettings({
          camera: { distance: draft.distance, maxDistance: draft.maxDistance, focalLength: draft.focalLength },
          meshDisplacement: draft.displacement,
        });
      }
    } finally {
      commitInFlight = false;
    }
  }

  function commitCamera(): void {
    if (!view) return;
    commitPending = true;
    void runCommit();
  }

  // -- Depth of Field checkbox: purely a parameter of the next export
  // request, not a persisted setting (components.py's CHECKLIST_DOF has no
  // "remember" callback of its own).
  let dofEnabled = $state(false);

  // -- Create glTF Scene (WEB-32): renders into the 3D viewer tab (see
  // ViewerTabs.svelte/Model3DViewer.svelte), same as Dash's own
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

<div class="export-tab" data-testid="tab-export">
  <div class="panel gltf-panel">
    <div class="action-row">
      <button
        type="button"
        class="btn"
        data-testid="gltf-create"
        disabled={isBusy() || !view}
        onclick={createGltfScene}
      >
        Create glTF Scene
      </button>
      <button
        type="button"
        class="btn"
        data-testid="gltf-export"
        disabled={isBusy() || !view}
        onclick={() => void exportGltfScene()}
      >
        Export glTF Scene
      </button>
      <button
        type="button"
        class="btn"
        data-testid="upscale-textures"
        disabled={isBusy() || !view}
        onclick={upscaleTextures}
      >
        Upscale Textures
      </button>
    </div>

    <div class="progress-bar" data-testid="export-progress">
      <div
        class="progress-bar-fill"
        style={`width: ${(creatingGltf || upscaling) ? Math.round(jobStore.progress * 100) : 0}%`}
      ></div>
    </div>

    <label class="checkbox-row">
      <input
        type="checkbox"
        data-testid="toggle-dof"
        checked={dofEnabled}
        onchange={(event) => (dofEnabled = (event.currentTarget as HTMLInputElement).checked)}
      />
      Support Depth of Field Effect
    </label>

    <!--
      The four camera/displacement sliders below deliberately do NOT disable
      on `isBusy()` (unlike every other control in this tab): each commits on
      every discrete change (`onchange` fires per arrow-key step, not just on
      pointer release), matching components.py's SLIDER_CAMERA_DISTANCE/etc.,
      which Dash never gates with a `running=` disable list either. Disabling
      mid-flight would make a fast arrow-key sequence (e.g. `setSlider`
      stepping many values in a row) race its own still-in-flight commit.
    -->
    <div class="field">
      <label class="field-label" for="camera-distance">Camera Distance</label>
      <input
        id="camera-distance"
        type="range"
        min="0"
        max="500"
        step="1"
        data-testid="camera-distance"
        value={draft.distance}
        disabled={!view}
        oninput={(event) => onCameraInput('distance', event)}
        onchange={commitCamera}
      />
      <span class="slider-value">{draft.distance}</span>
    </div>

    <div class="field">
      <label class="field-label" for="max-distance">Max Distance</label>
      <input
        id="max-distance"
        type="range"
        min="0"
        max="1000"
        step="1"
        data-testid="max-distance"
        value={draft.maxDistance}
        disabled={!view}
        oninput={(event) => onCameraInput('maxDistance', event)}
        onchange={commitCamera}
      />
      <span class="slider-value">{draft.maxDistance}</span>
    </div>

    <div class="field">
      <label class="field-label" for="focal-length">Focal Length</label>
      <input
        id="focal-length"
        type="range"
        min="1"
        max="500"
        step="1"
        data-testid="focal-length"
        value={draft.focalLength}
        disabled={!view}
        oninput={(event) => onCameraInput('focalLength', event)}
        onchange={commitCamera}
      />
      <span class="slider-value">{draft.focalLength}</span>
    </div>

    <div class="field">
      <label class="field-label" for="displacement">Mesh Displacement</label>
      <input
        id="displacement"
        type="range"
        min="0"
        max="150"
        step="5"
        data-testid="displacement"
        value={draft.displacement}
        disabled={!view}
        oninput={(event) => onCameraInput('displacement', event)}
        onchange={commitCamera}
      />
      <span class="slider-value">{draft.displacement}</span>
    </div>
  </div>

  <div class="panel animation-panel">
    <button
      type="button"
      class="btn"
      data-testid="animation-export"
      disabled={isBusy() || !view || numFrames <= 0}
      onclick={exportAnimation}
    >
      Export Animation
    </button>

    <div class="progress-bar" data-testid="animation-progress">
      <div class="progress-bar-fill" style={`width: ${animating ? Math.round(jobStore.progress * 100) : 0}%`}></div>
    </div>

    <div class="field">
      <label class="field-label" for="number-of-frames">Number of Frames</label>
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
      <span class="slider-value">{numFrames}</span>
    </div>
  </div>
</div>

<style>
  .export-tab {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    padding: var(--space-2);
  }

  .panel {
    padding: var(--space-2);
  }

  .action-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-bottom: var(--space-2);
  }

  .checkbox-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin: var(--space-2) 0;
  }

  .field {
    margin-bottom: var(--space-2);
  }

  input[type='range'] {
    width: 100%;
  }

  .slider-value {
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }
</style>
