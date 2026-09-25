<script lang="ts">
  import { uiStore } from './lib/state/ui.svelte';
  import { projectStore } from './lib/state/project.svelte';
  import Header from './lib/components/shell/Header.svelte';
  import Footer from './lib/components/shell/Footer.svelte';
  import LogPanel from './lib/components/shell/LogPanel.svelte';
  import ViewerTabs from './lib/components/upload/ViewerTabs.svelte';
  import MainTabs from './lib/components/shell/MainTabs.svelte';

  // Theme tokens in app.css switch on `:root.dark`.
  $effect(() => {
    document.documentElement.classList.toggle('dark', uiStore.theme === 'dark');
  });

  // Restore/Load applies the persisted theme (WEB-38's `restore_dark_mode`):
  // whenever the project view changes, follow `settings.darkMode` exactly -
  // this also keeps the theme in sync after `workflow.toggleDarkMode`
  // persists its own change (a harmless no-op re-application).
  $effect(() => {
    const darkMode = projectStore.view?.settings.darkMode;
    if (darkMode !== undefined) uiStore.setTheme(darkMode ? 'dark' : 'light');
  });
</script>

<div id="app-container" class="app-root" class:dark={uiStore.theme === 'dark'}>
  <Header />

  <main class="app-main">
    <ViewerTabs />
    <MainTabs />
  </main>

  <LogPanel />
  <Footer />
</div>

<style>
  /* The app owns the viewport: exactly one screen tall, never a page
     scrollbar. The main area gets whatever the header, log and footer
     leave over, and each column fits (or scrolls) inside it. */
  .app-root {
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
    /* Contains absolutely positioned descendants (e.g. `.sr-only` inputs). */
    position: relative;
    display: flex;
    flex-direction: column;
    background-color: var(--color-bg);
    color: var(--color-text);
  }

  .app-main {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: var(--space-4);
    grid-template-rows: minmax(0, 1fr);
    flex: 1 1 0;
    min-height: 0;
    padding: var(--space-2) var(--space-2) 0;
    /* Prevent a too-narrow column from forcing its content (button rows,
       slider labels, tab strips) to overflow horizontally instead of
       wrapping -- see the two-column -> single-column breakpoint below for
       the actual fix; this only stops the grid track itself from ever
       being narrower than its content demands before that breakpoint
       kicks in. */
    min-width: 0;
  }

  .app-main > :global(*) {
    min-width: 0;
  }

  /* Below this width the two-column layout (Input Image / workflow tabs)
     no longer has room for both columns without overflowing; stack them
     instead. Checked at 1440x1000, 1024x768 and 768x1024 (see
     docs/svelte-migration/PARITY.md's responsive-pass notes). */
  @media (max-width: 900px) {
    .app-main {
      grid-template-columns: 1fr;
      grid-template-rows: minmax(0, 3fr) minmax(0, 2fr);
    }
  }
</style>
