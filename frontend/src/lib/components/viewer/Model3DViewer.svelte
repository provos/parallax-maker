<script lang="ts">
  /**
   * The 3D view: `<model-viewer>` showing the project's glTF export.
   *
   * The export must match the current scene, so entering the view while it is
   * missing or stale (`exports.gltfStale`: slices, camera or displacement
   * changed since it was built) rebuilds it first. The camera starts at the
   * reference camera, so the scene first looks like the image, and orbits
   * around the middle of the card stack.
   *
   * `@google/model-viewer` is bundled from npm (no CDN, so this works
   * offline in e2e) and lazy-loaded here - only when the view is actually
   * opened - so the main bundle stays small.
   */
  import { untrack } from 'svelte';
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { jobStore } from '../../state/jobs.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';

  let ready = $state(false);
  let loadFailed = $state(false);

  $effect(() => {
    if (uiStore.view !== '3d' || ready || loadFailed) return;
    let cancelled = false;
    import('@google/model-viewer')
      .then(() => {
        if (!cancelled) ready = true;
      })
      .catch(() => {
        if (!cancelled) loadFailed = true;
      });
    return () => {
      cancelled = true;
    };
  });

  const gltfUrl = $derived(projectStore.view?.exports.gltf?.url ?? null);
  const needsBuild = $derived.by(() => {
    const view = projectStore.view;
    if (!view || view.slices.length === 0) return false;
    return !view.exports.gltf || !!view.exports.gltfStale;
  });
  const building = $derived(jobStore.active === 'export-gltf');
  // The revision a rebuild failed at: no automatic retry until the scene
  // changes again (or the user asks).
  let failedRevision = $state<number | null>(null);
  const failed = $derived(failedRevision !== null && failedRevision === projectStore.view?.revision);
  // Covers the whole export call: the job store is already idle while the
  // call finishes up, before its outcome is known.
  let pending = $state(false);

  function rebuild(): void {
    const revision = projectStore.view?.revision ?? null;
    pending = true;
    void workflow.startGltfExport(uiStore.exportDof, { preview: true }).then((ok) => {
      failedRevision = ok ? null : revision;
      pending = false;
    });
  }

  $effect(() => {
    if (uiStore.view !== '3d' || !needsBuild || failed || pending || isBusy()) return;
    untrack(rebuild);
  });

  function retry(): void {
    if (pending || isBusy()) return;
    failedRevision = null;
    rebuild();
  }

  // glTF is y-up (the scene's y points down) and the reference camera sits at
  // (0, 0, cameraZ) looking along +z, tilted up by the pitch.
  const framing = $derived.by(() => {
    const profile = projectStore.view?.sceneProfile;
    if (!profile) return null;
    const depths = profile.cards.map((card) => card.z);
    const middle = depths.length
      ? (Math.min(...depths) + Math.max(...depths)) / 2
      : (projectStore.view?.settings.camera.maxDistance ?? 0) / 2;
    const pitch = (profile.pitch * Math.PI) / 180;
    const along = Math.max(middle - profile.cameraZ, 1);
    const radius = along / Math.cos(pitch);
    return {
      target: `0m ${along * Math.tan(pitch)}m ${middle}m`,
      orbit: `180deg ${90 + profile.pitch}deg ${radius}m`,
      maxOrbit: `auto auto ${radius * 4}m`,
      fov: `${2 * profile.halfFov}deg`,
    };
  });
</script>

<div class="model3d" data-testid="model-viewer-container">
  {#if loadFailed}
    <p class="placeholder" data-testid="model-viewer-error">Failed to load the 3D viewer.</p>
  {:else if !ready}
    <p class="placeholder" data-testid="model-viewer-loading">Loading 3D viewer…</p>
  {:else if needsBuild && failed && !building && !pending}
    <div class="placeholder" data-testid="model-viewer-failed">
      <p>Couldn't build the 3D scene.</p>
      <button type="button" class="btn" data-testid="model-viewer-retry" disabled={isBusy()} onclick={retry}>
        Try again
      </button>
    </div>
  {:else if needsBuild}
    <p class="placeholder" data-testid="model-viewer-building">
      {building || !isBusy()
        ? 'Building the 3D scene…'
        : 'The 3D scene will update when the current job finishes.'}
    </p>
  {:else if gltfUrl}
    <!-- svelte-ignore element_invalid_self_closing_tag -->
    <model-viewer
      data-testid="model-viewer"
      src={gltfUrl}
      alt="glTF Scene"
      ar
      camera-target={framing?.target ?? 'auto auto auto'}
      camera-orbit={framing?.orbit ?? '180deg 90deg auto'}
      field-of-view={framing?.fov ?? 'auto'}
      min-camera-orbit="auto auto 1%"
      max-camera-orbit={framing?.maxOrbit ?? 'auto auto 100%'}
      min-field-of-view="1deg"
      max-field-of-view="60deg"
      interaction-prompt="none"
      camera-controls
      touch-action="pan-y"
      class="viewer-el"
    ></model-viewer>
  {:else}
    <p class="placeholder" data-testid="model-viewer-empty">No glTF file available.</p>
  {/if}
</div>

<style>
  .model3d {
    min-height: 30rem;
    height: 70vh;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .placeholder {
    color: var(--color-text-muted);
  }

  div.placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-2);
  }

  .viewer-el {
    width: 100%;
    height: 100%;
    display: block;
  }
</style>
