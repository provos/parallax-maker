<script lang="ts">
  import { uiStore } from './lib/state/ui.svelte';
  import Header from './lib/components/shell/Header.svelte';
  import Footer from './lib/components/shell/Footer.svelte';
  import LogPanel from './lib/components/shell/LogPanel.svelte';
  import ViewerTabs from './lib/components/upload/ViewerTabs.svelte';
  import MainTabs from './lib/components/shell/MainTabs.svelte';

  // Theme tokens in app.css switch on `:root.dark`.
  $effect(() => {
    document.documentElement.classList.toggle('dark', uiStore.theme === 'dark');
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
  .app-root {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background-color: var(--color-bg);
    color: var(--color-text);
  }

  .app-main {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: var(--space-4);
    padding: var(--space-2) var(--space-2) 0;
  }
</style>
