<script lang="ts">
  /**
   * The right-hand column: the panel for the current workflow step (or
   * Settings). Until each step has its own panel (docs/redesign/HANDOFF.md
   * §12), it hosts the pre-redesign tab bodies. All of them stay mounted and
   * only the active one is shown: e2e scenarios read threshold sliders and
   * slice thumbnails without first switching to them.
   */
  import { uiStore, WORKFLOW_STEPS, type MainTab } from '../../state/ui.svelte';
  import ModeTab from '../depth/ModeTab.svelte';
  import SegmentationTab from '../segmentation/SegmentationTab.svelte';
  import InpaintingTab from '../inpainting/InpaintingTab.svelte';
  import ExportTab from '../export/ExportTab.svelte';
  import ConfigurationTab from '../config/ConfigurationTab.svelte';
  import SelectedSliceHeader from '../layers/SelectedSliceHeader.svelte';

  const title = $derived(
    uiStore.mainTab === 'Configuration'
      ? 'Settings'
      : (WORKFLOW_STEPS.find(({ step }) => step === uiStore.step)?.label ?? ''),
  );

  const hidden = (tab: MainTab) => uiStore.mainTab !== tab;
</script>

<aside class="inspector" aria-label="Inspector" data-testid="inspector" data-panel={uiStore.mainTab}>
  <div class="inspector-header">
    <h2 class="label">{title}</h2>
  </div>
  <div class="inspector-body">
    {#if uiStore.mainTab !== 'Configuration'}<SelectedSliceHeader />{/if}
    <div class="section" class:hidden={hidden('Mode')}><ModeTab /></div>
    <div class="section" class:hidden={hidden('Segmentation')}><SegmentationTab /></div>
    <div class="section" class:hidden={hidden('Inpainting')}><InpaintingTab /></div>
    <div class="section" class:hidden={hidden('Export')}><ExportTab /></div>
    <div class="section" class:hidden={hidden('Configuration')}><ConfigurationTab /></div>
  </div>
</aside>

<style>
  .inspector {
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    background: var(--color-surface);
    border-left: 1px solid var(--color-border);
  }

  .inspector-header {
    height: var(--viewbar-h);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    padding: 0 14px;
    border-bottom: 1px solid var(--color-border);
  }

  .inspector-header .label {
    margin: 0;
  }

  .inspector-body {
    flex: 1 1 0;
    min-height: 0;
    overflow-y: auto;
    /* Keeps absolutely positioned descendants (`.sr-only` file inputs)
       inside this scroll container instead of stretching the page. */
    position: relative;
  }

  .section {
    padding: var(--space-3) 14px;
    display: flex;
    flex-direction: column;
  }
</style>
