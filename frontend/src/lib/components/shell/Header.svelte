<script lang="ts">
  import { uiStore } from '../../state/ui.svelte';
</script>

<header class="app-header">
  <div class="header-side">
    <button
      type="button"
      class="theme-toggle"
      data-testid="theme-toggle"
      aria-label="Toggle dark mode"
      aria-pressed={uiStore.theme === 'dark'}
      onclick={() => uiStore.toggleTheme()}
    >
      <span class="thumb"></span>
    </button>
  </div>
  <h1 data-testid="app-title">Parallax Maker</h1>
  <div class="header-side" aria-hidden="true"></div>
</header>

<style>
  .app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: var(--color-header-bg);
    color: var(--color-header-text);
    padding: var(--space-4);
  }

  .header-side {
    width: 7rem;
    display: flex;
    align-items: center;
  }

  h1 {
    flex: 1;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
  }

  .theme-toggle {
    width: 3rem;
    height: 1.75rem;
    border-radius: 999px;
    border: none;
    background-color: var(--color-toggle-bg);
    cursor: pointer;
    padding: 2px;
    display: flex;
    align-items: center;
  }

  .theme-toggle:hover {
    background-color: var(--color-toggle-bg-hover);
  }

  .thumb {
    width: 1.35rem;
    height: 1.35rem;
    border-radius: 50%;
    background-color: white;
    display: block;
  }

  .theme-toggle[aria-pressed='true'] .thumb {
    margin-left: auto;
  }
</style>
