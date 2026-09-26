<script lang="ts">
  /**
   * A modal dialog (docs/redesign/HANDOFF.md §3) on the native `<dialog>`:
   * open while `uiStore.dialog` names it, closed by Esc, the × button or a
   * click on the backdrop. It stays mounted while closed, so its controls
   * (and their values) always exist in the page.
   */
  import type { Snippet } from 'svelte';
  import X from '@lucide/svelte/icons/x';
  import { uiStore, type DialogName } from '../../state/ui.svelte';

  let {
    name,
    title,
    testId,
    width = 640,
    height,
    children,
  }: {
    name: DialogName;
    title: string;
    testId: string;
    width?: number;
    height?: number;
    children: Snippet;
  } = $props();

  let dialog: HTMLDialogElement | undefined = $state();
  const open = $derived(uiStore.dialog === name);
  const titleId = $derived(`${testId}-title`);

  $effect(() => {
    if (!dialog) return;
    if (open && !dialog.open) {
      // jsdom has no showModal.
      if (typeof dialog.showModal === 'function') dialog.showModal();
      else dialog.setAttribute('open', '');
    } else if (!open && dialog.open) {
      if (typeof dialog.close === 'function') dialog.close();
      else dialog.removeAttribute('open');
    }
  });

  // Esc (the native `cancel` -> `close`) and form-less closes land here.
  function onClose(): void {
    if (uiStore.dialog === name) uiStore.closeDialog();
  }

  // A click on the ::backdrop targets the dialog element itself.
  function onClick(event: MouseEvent): void {
    if (event.target === dialog) uiStore.closeDialog();
  }
</script>

<dialog
  bind:this={dialog}
  class="dialog"
  data-testid={testId}
  aria-labelledby={titleId}
  style:width={`min(${width}px, calc(100vw - 32px))`}
  style:height={height ? `min(${height}px, calc(100dvh - 32px))` : undefined}
  onclose={onClose}
  onclick={onClick}
>
  <div class="frame">
    <div class="head">
      <h2 id={titleId}>{title}</h2>
      <button
        type="button"
        class="btn btn-ghost btn-icon"
        aria-label={`Close ${title.toLowerCase()}`}
        data-testid={`${testId}-close`}
        onclick={() => uiStore.closeDialog()}
      >
        <X size={16} strokeWidth={1.6} />
      </button>
    </div>
    <div class="content">
      {@render children()}
    </div>
  </div>
</dialog>

<style>
  .dialog {
    /* The app's reset zeroes margins; a modal dialog centers with auto ones. */
    margin: auto;
    padding: 0;
    max-height: calc(100dvh - 32px);
    background: var(--color-float);
    color: var(--color-text);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-float);
    overflow: hidden;
  }

  .dialog::backdrop {
    background: var(--color-scrim);
  }

  .frame {
    display: flex;
    flex-direction: column;
    height: 100%;
    max-height: calc(100dvh - 34px);
  }

  .head {
    display: flex;
    align-items: center;
    padding: 12px 12px 12px 16px;
    border-bottom: 1px solid var(--color-border);
    flex-shrink: 0;
  }

  h2 {
    margin: 0;
    flex: 1 1 auto;
    font-size: var(--text-title);
    font-weight: 600;
  }

  .content {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
  }
</style>
