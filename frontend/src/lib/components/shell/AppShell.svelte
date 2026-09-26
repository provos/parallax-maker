<script lang="ts">
  /**
   * The app's frame (docs/redesign/HANDOFF.md §2): header, the canvas and
   * Inspector columns, the optional log drawer and the status bar. It fills
   * the viewport exactly; columns scroll inside themselves, never the page.
   * The layer panel column arrives in the next phase.
   */
  import { jobStore } from '../../state/jobs.svelte';
  import AppHeader from './AppHeader.svelte';
  import Inspector from './Inspector.svelte';
  import LogDrawer from './LogDrawer.svelte';
  import StatusBar from './StatusBar.svelte';
  import ViewerTabs from '../upload/ViewerTabs.svelte';
</script>

<div id="app-container" class="app-shell" class:app-busy={jobStore.active !== null}>
  <AppHeader />
  <main class="app-main">
    <section class="canvas-area" aria-label="Canvas">
      <ViewerTabs />
    </section>
    <Inspector />
  </main>
  <LogDrawer />
  <StatusBar />
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

  .app-busy {
    cursor: progress;
  }

  .app-main {
    flex: 1 1 0;
    min-height: 0;
    min-width: 0;
    display: grid;
    /* The Inspector still hosts the pre-redesign tab bodies, which need more
       than the design's --inspector-w; it narrows once they are rebuilt. */
    grid-template-columns: minmax(0, 1fr) 440px;
    grid-template-rows: minmax(0, 1fr);
  }

  .canvas-area {
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    padding: var(--space-2) var(--space-3) 0;
    background: var(--color-bg);
  }

  .canvas-area > :global(*) {
    flex: 1 1 0;
    min-height: 0;
  }

  /* Below the design's 1280px minimum, give the canvas more of the width. */
  @media (max-width: 1100px) {
    .app-main {
      grid-template-columns: minmax(0, 1fr) 360px;
    }
  }
</style>
