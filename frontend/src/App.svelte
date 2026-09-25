<script lang="ts">
  import { health, ApiError } from './lib/api/client';

  type Status = 'checking' | 'ok' | 'error';

  let status = $state<Status>('checking');
  let statusDetail = $state<string>('');

  $effect(() => {
    const controller = new AbortController();

    health(controller.signal)
      .then((res) => {
        status = res.ok ? 'ok' : 'error';
        statusDetail = res.ok ? res.version : 'unhealthy';
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        status = 'error';
        statusDetail = err instanceof ApiError ? err.message : 'unreachable';
      });

    return () => controller.abort();
  });
</script>

<div id="app-container" class="min-h-screen flex flex-col">
  <header class="title-header flex items-center justify-between p-4">
    <h1 class="text-2xl font-bold" data-testid="app-title">Parallax Maker</h1>
    <p class="text-sm" data-testid="health-status" data-status={status}>
      {#if status === 'checking'}
        Checking server&hellip;
      {:else if status === 'ok'}
        Server ok{statusDetail ? ` (${statusDetail})` : ''}
      {:else}
        Server error{statusDetail ? `: ${statusDetail}` : ''}
      {/if}
    </p>
  </header>

  <main class="flex flex-1 gap-4 p-4">
    <section
      class="app-panel flex-1"
      data-testid="main-image-area"
      aria-label="Main image area"
    >
      <p class="text-muted">Image workspace placeholder.</p>
    </section>

    <aside class="app-panel w-80" data-testid="tabs-panel" aria-label="Tabs">
      <p class="text-muted">Tabs placeholder.</p>
    </aside>
  </main>

  <footer class="footer p-2 text-center" data-testid="app-footer">
    &copy; 2024 Niels Provos
  </footer>
</div>

<style>
  .title-header {
    background-color: var(--color-header-bg);
    color: var(--color-header-text);
  }

  .footer {
    color: var(--color-text-muted);
  }

  .app-panel {
    background-color: var(--color-bg-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--space-4);
  }

  .text-muted {
    color: var(--color-text-muted);
  }
</style>
