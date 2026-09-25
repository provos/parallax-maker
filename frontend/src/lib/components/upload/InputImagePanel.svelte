<script lang="ts">
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';

  let fileInput: HTMLInputElement | undefined;
  let dragging = $state(false);

  function pickFile(): void {
    if (isBusy()) return;
    fileInput?.click();
  }

  function handleFile(file: File | undefined | null): void {
    if (!file || isBusy()) return;
    void workflow.uploadImage(file, uiStore.depthModel);
  }

  function onInputChange(event: Event): void {
    const target = event.currentTarget as HTMLInputElement;
    handleFile(target.files?.[0]);
    target.value = '';
  }

  function onDrop(event: DragEvent): void {
    event.preventDefault();
    dragging = false;
    handleFile(event.dataTransfer?.files?.[0]);
  }

  function onDragOver(event: DragEvent): void {
    event.preventDefault();
    dragging = true;
  }

  function onDragLeave(): void {
    dragging = false;
  }

  function onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      pickFile();
    }
  }
</script>

<div class="input-image-outer panel">
  <span class="panel-label">Input Image</span>
  <div
    class="drop-zone panel"
    class:dragging
    role="button"
    tabindex="0"
    data-testid="input-image-panel"
    onclick={pickFile}
    onkeydown={onKeydown}
    ondrop={onDrop}
    ondragover={onDragOver}
    ondragleave={onDragLeave}
  >
    <img data-testid="main-image" alt="" src={projectStore.view?.assets.input?.url} />
    <input
      bind:this={fileInput}
      type="file"
      accept="image/*"
      class="sr-only"
      data-testid="upload-image-input"
      onchange={onInputChange}
    />
  </div>
  <div class="tools-row-placeholder"></div>
</div>

<style>
  .input-image-outer {
    display: flex;
    flex-direction: column;
    min-height: 30rem;
  }

  .drop-zone {
    position: relative;
    flex: 1;
    min-height: 24rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    overflow: hidden;
  }

  .drop-zone.dragging {
    border-color: var(--color-accent);
  }

  .drop-zone img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  /* Chromium renders a "broken image" glyph for an <img> with layout space
     and no loaded resource, even with no `src` attribute at all. Hide it
     until there is something to show, matching Dash's empty panel look. */
  .drop-zone img:not([src]) {
    visibility: hidden;
  }

  .tools-row-placeholder {
    min-height: 2.75rem;
  }
</style>
