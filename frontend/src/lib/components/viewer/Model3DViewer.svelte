<script lang="ts">
  /**
   * The left-panel "3D" viewer tab: renders `<model-viewer>` pointing at the
   * project's exported glTF asset, exactly like Dash's `get_gltf_iframe`
   * (`parallax_maker/utils.py`) - same camera/AR/controls attributes, just
   * as a real custom element instead of an `<iframe srcDoc>`. Placeholder
   * text (`get_no_gltf_available`'s own wording) when no scene has been
   * exported yet.
   *
   * `@google/model-viewer` is bundled from npm (no CDN, so this works
   * offline in e2e) and lazy-loaded here - only when this tab is actually
   * opened - so the main bundle stays small (see the migration task's
   * bundle-size requirement).
   */
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';

  let ready = $state(false);
  let loadFailed = $state(false);

  $effect(() => {
    if (uiStore.viewerTab !== '3D' || ready || loadFailed) return;
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
</script>

<div class="model3d" data-testid="model-viewer-container">
  {#if loadFailed}
    <p class="placeholder" data-testid="model-viewer-error">Failed to load the 3D viewer.</p>
  {:else if !ready}
    <p class="placeholder" data-testid="model-viewer-loading">Loading 3D viewer…</p>
  {:else if gltfUrl}
    <!-- svelte-ignore element_invalid_self_closing_tag -->
    <model-viewer
      data-testid="model-viewer"
      src={gltfUrl}
      alt="glTF Scene"
      ar
      auto-rotate
      camera-target="0m 0m 0m"
      camera-orbit="3.106650330236851rad 1.5658376358588284rad 50m"
      field-of-view="8"
      min-camera-orbit="auto auto 1%"
      max-camera-orbit="auto auto 100%"
      min-field-of-view="1deg"
      max-field-of-view="60deg"
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

  .viewer-el {
    width: 100%;
    height: 100%;
    display: block;
  }
</style>
