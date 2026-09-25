<script lang="ts">
  import { uiStore, type ViewerTab } from '../../state/ui.svelte';
  import InputImagePanel from './InputImagePanel.svelte';
  import Model3DViewer from '../viewer/Model3DViewer.svelte';

  const tabs: ViewerTab[] = ['2D', '3D'];
</script>

<div class="viewer-column">
  <div class="tab-strip" role="tablist" aria-label="Viewer">
    {#each tabs as tab (tab)}
      <button
        type="button"
        role="tab"
        aria-selected={uiStore.viewerTab === tab}
        class:active={uiStore.viewerTab === tab}
        onclick={() => uiStore.setViewerTab(tab)}
      >
        {tab}
      </button>
    {/each}
  </div>

  <div class:hidden={uiStore.viewerTab !== '2D'} data-testid="viewer-2d">
    <InputImagePanel />
  </div>
  <div class="panel viewer-3d" class:hidden={uiStore.viewerTab !== '3D'} data-testid="viewer-3d">
    <Model3DViewer />
  </div>
</div>

<style>
  .viewer-column {
    display: flex;
    flex-direction: column;
  }

  .viewer-3d {
    min-height: 30rem;
  }
</style>
