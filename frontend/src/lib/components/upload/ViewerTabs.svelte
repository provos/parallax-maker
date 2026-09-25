<script lang="ts">
  import { uiStore, type ViewerTab } from '../../state/ui.svelte';
  import { rovingTabs } from '../../a11y/rovingTabindex';
  import InputImagePanel from './InputImagePanel.svelte';
  import Model3DViewer from '../viewer/Model3DViewer.svelte';

  const tabs: ViewerTab[] = ['2D', '3D'];
</script>

<div class="viewer-column">
  <div
    class="tab-strip"
    role="tablist"
    aria-label="Viewer"
    use:rovingTabs={(index) => uiStore.setViewerTab(tabs[index])}
  >
    {#each tabs as tab (tab)}
      <button
        type="button"
        role="tab"
        aria-selected={uiStore.viewerTab === tab}
        tabindex={uiStore.viewerTab === tab ? 0 : -1}
        class:active={uiStore.viewerTab === tab}
        onclick={() => uiStore.setViewerTab(tab)}
      >
        {tab}
      </button>
    {/each}
  </div>

  <div class="viewer-pane" class:hidden={uiStore.viewerTab !== '2D'} data-testid="viewer-2d">
    <InputImagePanel />
  </div>
  <div class="panel viewer-pane viewer-3d" class:hidden={uiStore.viewerTab !== '3D'} data-testid="viewer-3d">
    <Model3DViewer />
  </div>
</div>

<style>
  .viewer-column {
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  /* Each pane takes the column's remaining height (never more). */
  .viewer-pane {
    flex: 1 1 0;
    min-height: 0;
  }
</style>
