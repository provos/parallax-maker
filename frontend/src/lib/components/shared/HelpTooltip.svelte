<script lang="ts">
  /**
   * An accessible "?" help affordance carrying Dash's own contextual help
   * texts (utility.js's `setupHelper`/JS-06 `helpTexts` object, wired by
   * `suppress_contextmenu`/CLI-02). Dash shows these as a randomized tooltip
   * that pops up near the cursor after 3s of mouse idle time over the input
   * image container -- a timer-driven popup with no `role`, no
   * `aria-describedby`, and no keyboard equivalent at all (a mouse-only
   * user has no way to trigger it on demand, and a keyboard-only user
   * cannot reach it at all).
   *
   * This is a deliberate, documented redesign rather than a literal port:
   * a focusable `button` shows/hides a `role="tooltip"` panel listing every
   * text Dash would have randomly picked from for the given context,
   * `aria-describedby`-linked to the button, openable by click, Enter or
   * Space, and dismissible with Escape or a click outside -- so the same
   * information Dash only reveals after an idle timer is available
   * on-demand to mouse, keyboard and screen-reader users alike (see
   * PARITY.md's "Remaining gaps"/JS-06 row).
   */
  let { label, texts }: { label: string; texts: string[] } = $props();

  let open = $state(false);
  let buttonEl: HTMLButtonElement | undefined;
  const tooltipId = $derived(`help-tooltip-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`);

  function toggle(): void {
    open = !open;
  }

  function close(): void {
    open = false;
  }

  function onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      event.preventDefault();
      close();
      buttonEl?.focus();
    }
  }

  function onWindowClick(event: MouseEvent): void {
    if (!open) return;
    const target = event.target as Node;
    if (buttonEl && !buttonEl.parentElement?.contains(target)) close();
  }
</script>

<svelte:window onclick={onWindowClick} />

<span class="help-tooltip">
  <button
    bind:this={buttonEl}
    type="button"
    class="help-button"
    aria-label={`Help: ${label}`}
    aria-describedby={open ? tooltipId : undefined}
    aria-expanded={open}
    data-testid="help-button"
    onclick={toggle}
    onkeydown={onKeydown}
  >
    ?
  </button>
  {#if open}
    <div class="help-panel" role="tooltip" id={tooltipId} data-testid="help-panel">
      <ul>
        {#each texts as text (text)}
          <li>{text}</li>
        {/each}
      </ul>
    </div>
  {/if}
</span>

<style>
  .help-tooltip {
    position: relative;
    display: inline-block;
  }

  .help-button {
    width: 1.5rem;
    height: 1.5rem;
    border-radius: 50%;
    border: 1px solid var(--color-border);
    background-color: var(--color-bg);
    color: var(--color-text);
    font: inherit;
    font-weight: 700;
    line-height: 1;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .help-button:hover,
  .help-button:focus-visible {
    background-color: var(--color-accent);
    color: var(--color-accent-text);
  }

  .help-panel {
    position: absolute;
    top: calc(100% + var(--space-1));
    right: 0;
    z-index: 30;
    width: 16rem;
    max-width: 70vw;
    background-color: var(--color-bg);
    color: var(--color-text);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-2);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    font-size: 0.8125rem;
  }

  .help-panel ul {
    margin: 0;
    padding-left: 1.1rem;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }
</style>
