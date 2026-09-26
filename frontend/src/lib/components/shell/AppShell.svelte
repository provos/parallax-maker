<script lang="ts">
  /**
   * The app's frame (docs/redesign/HANDOFF.md §2): header, the layer panel,
   * canvas and Inspector columns, the optional log drawer and the status bar. It fills
   * the viewport exactly; columns scroll inside themselves, never the page.
   */
  import { jobStore } from '../../state/jobs.svelte';
  import AppHeader from './AppHeader.svelte';
  import Inspector from './Inspector.svelte';
  import LogDrawer from './LogDrawer.svelte';
  import StatusBar from './StatusBar.svelte';
  import CanvasArea from '../canvas/CanvasArea.svelte';
  import LayerPanel from '../layers/LayerPanel.svelte';
  import ExportDialog from '../dialogs/ExportDialog.svelte';
  import SettingsDialog from '../dialogs/SettingsDialog.svelte';
  import ShortcutsDialog from '../dialogs/ShortcutsDialog.svelte';
  import { handleShortcut } from '../../shortcuts';

  // Keyboard shortcuts are handled on the app root, which takes focus when
  // anything non-focusable inside it is clicked (tabindex -1), so keys reach
  // it without a window-wide listener.
  let root: HTMLDivElement | undefined = $state();
  $effect(() => {
    if (root && (document.activeElement === document.body || !document.activeElement)) root.focus({ preventScroll: true });
  });

  function onKeydown(event: KeyboardEvent): void {
    if (handleShortcut(event)) event.preventDefault();
  }
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_static_element_interactions -->
<div
  id="app-container"
  class="app-shell"
  class:app-busy={jobStore.active !== null}
  tabindex="-1"
  bind:this={root}
  onkeydown={onKeydown}
>
  <AppHeader />
  <main class="app-main">
    <LayerPanel />
    <CanvasArea />
    <Inspector />
  </main>
  <LogDrawer />
  <StatusBar />
  <ExportDialog />
  <SettingsDialog />
  <ShortcutsDialog />
</div>

<style>
  .app-shell {
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
    /* Contains absolutely positioned descendants (e.g. `.sr-only` inputs). */
    position: relative;
    display: flex;
    flex-direction: column;
    background: var(--color-bg);
    color: var(--color-text);
  }

  .app-shell:focus {
    outline: none;
  }

  .app-busy {
    cursor: progress;
  }

  .app-main {
    flex: 1 1 0;
    min-height: 0;
    min-width: 0;
    display: grid;
    grid-template-columns: var(--layers-w) minmax(0, 1fr) var(--inspector-w);
    grid-template-rows: minmax(0, 1fr);
  }

  /* Below the design's 1280px minimum, give the canvas more of the width. */
  @media (max-width: 1100px) {
    .app-main {
      grid-template-columns: 240px minmax(0, 1fr) 320px;
    }
  }
</style>
