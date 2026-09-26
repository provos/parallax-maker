<script lang="ts">
  import { uiStore } from './lib/state/ui.svelte';
  import { projectStore } from './lib/state/project.svelte';
  import AppShell from './lib/components/shell/AppShell.svelte';

  // Theme tokens in app.css switch on `data-theme` on <html>.
  $effect(() => {
    document.documentElement.dataset.theme = uiStore.theme;
  });

  // A loaded or restored project brings its persisted theme with it:
  // whenever the project view changes, follow `settings.darkMode` exactly
  // (this also keeps the theme in sync after `workflow.toggleDarkMode`
  // persists its own change, a harmless re-application).
  $effect(() => {
    const darkMode = projectStore.view?.settings.darkMode;
    if (darkMode !== undefined) uiStore.setTheme(darkMode ? 'dark' : 'light');
  });
</script>

<AppShell />
