<script lang="ts">
  import { uiStore, MAIN_TABS } from '../../state/ui.svelte';
  import ModeTab from '../depth/ModeTab.svelte';
  import SegmentationTab from '../segmentation/SegmentationTab.svelte';
  import InpaintingTab from '../inpainting/InpaintingTab.svelte';
  import ExportTab from '../export/ExportTab.svelte';
  import ConfigurationTab from '../config/ConfigurationTab.svelte';
</script>

<div class="main-tabs">
  <div class="tab-strip" role="tablist" aria-label="Workflow">
    {#each MAIN_TABS as tab (tab)}
      <button
        type="button"
        role="tab"
        aria-selected={uiStore.mainTab === tab}
        class:active={uiStore.mainTab === tab}
        onclick={() => uiStore.setMainTab(tab)}
      >
        {tab}
      </button>
    {/each}
  </div>

  <!--
    Every tab's content stays mounted (hidden via CSS) rather than being
    conditionally created/destroyed, mirroring Dash's make_tabs(): e2e
    scenarios query threshold sliders and slice thumbnails without first
    switching to the Segmentation tab (see e2e/parallax-maker.spec.ts).
  -->
  <div class="panel tab-content" class:hidden={uiStore.mainTab !== 'Mode'}>
    <ModeTab />
  </div>
  <div class="panel tab-content" class:hidden={uiStore.mainTab !== 'Segmentation'}>
    <SegmentationTab />
  </div>
  <div class="panel tab-content" class:hidden={uiStore.mainTab !== 'Inpainting'}>
    <InpaintingTab />
  </div>
  <div class="panel tab-content" class:hidden={uiStore.mainTab !== 'Export'}>
    <ExportTab />
  </div>
  <div class="panel tab-content" class:hidden={uiStore.mainTab !== 'Configuration'}>
    <ConfigurationTab />
  </div>
</div>

<style>
  .main-tabs {
    display: flex;
    flex-direction: column;
  }

  .tab-content {
    min-height: 20rem;
  }
</style>
