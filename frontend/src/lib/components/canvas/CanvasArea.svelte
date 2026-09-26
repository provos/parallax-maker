<script lang="ts">
  /**
   * The canvas column (docs/redesign/HANDOFF.md §2): the view-mode bar, the
   * active tool's options, and the stage with the floating tool palette --
   * or the 3D viewer in the 3D view.
   */
  import { uiStore } from '../../state/ui.svelte';
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import InputImagePanel from '../upload/InputImagePanel.svelte';
  import Model3DViewer from '../viewer/Model3DViewer.svelte';
  import ViewModeBar from './ViewModeBar.svelte';
  import ToolOptionsBar from './ToolOptionsBar.svelte';
  import CanvasToolbar from './CanvasToolbar.svelte';

  let stage: ReturnType<typeof InputImagePanel> | undefined = $state();

  const hasImage = $derived(!!projectStore.view?.assets.input);
  const toolsShown = $derived(hasImage && uiStore.view !== 'parallax' && uiStore.view !== '3d');

  // Parallax 2D shows the latest camera render. On entering it while the
  // display image is something else, render the view from the reference
  // camera (which reproduces the image).
  let previousView = uiStore.view;
  $effect(() => {
    const view = uiStore.view;
    const entered = view === 'parallax' && previousView !== 'parallax';
    previousView = view;
    if (!entered) return;
    const project = projectStore.view;
    if (!project || project.slices.length === 0 || isBusy()) return;
    if (project.mainImage?.url !== uiStore.renderedMainUrl) void workflow.navigateCamera('reset');
  });
</script>

<section class="canvas-area" aria-label="Canvas">
  <ViewModeBar onZoomIn={() => stage?.zoomIn()} onZoomOut={() => stage?.zoomOut()} />
  {#if hasImage}
    <ToolOptionsBar />
  {/if}
  <div class="stage-wrap">
    <div class="stage" class:hidden={uiStore.view === '3d'}>
      <InputImagePanel bind:this={stage} />
    </div>
    <div class="viewer-3d" class:hidden={uiStore.view !== '3d'} data-testid="viewer-3d">
      <Model3DViewer />
    </div>
    {#if toolsShown}
      <CanvasToolbar />
    {/if}
  </div>
</section>

<style>
  .canvas-area {
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    background: var(--color-canvas);
  }

  .stage-wrap {
    position: relative;
    flex: 1 1 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  .stage,
  .viewer-3d {
    flex: 1 1 0;
    min-height: 0;
    padding: var(--space-3);
  }

  /* Room for the floating tool palette on the left (kept symmetric so the
     image stays centered). */
  .stage {
    padding-left: 56px;
    padding-right: 56px;
  }
</style>
