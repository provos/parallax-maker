<script lang="ts">
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';

  // Keep the slider in sync with the project once one exists (including
  // after a restore), the same way Dash's slider reflects state.
  $effect(() => {
    const view = projectStore.view;
    if (view) uiStore.setPendingNumSlices(view.numSlices);
  });

  // Same options/values/default as components.py's DROPDOWN_INPAINT_MODEL.
  const inpaintModelOptions: Array<{ value: string; label: string }> = [
    { value: 'kandinsky-community/kandinsky-2-2-decoder-inpaint', label: 'Kadinksy' },
    { value: 'runwayml/stable-diffusion-v1-5', label: 'SD 1.5' },
    { value: 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1', label: 'SD XL 1.0' },
    { value: 'stabilityai/stable-diffusion-3-medium-diffusers', label: 'StableDiffusion3' },
    { value: 'black-forest-labs/FLUX.1-Fill-dev', label: 'FLUX.1 Fill Dev' },
    { value: 'automatic1111', label: 'Automatic1111' },
    { value: 'comfyui', label: 'ComfyUI' },
    { value: 'stabilityai', label: 'StabilityAI' },
    { value: 'falai-foocus', label: 'Fal.ai Foocus' },
    { value: 'falai-flux-general', label: 'Fal.ai Flux General' },
    { value: 'falai-sd', label: 'Fal.ai SD' },
    { value: 'falai-sdxl', label: 'Fal.ai SDXL' },
  ];

  function onInpaintModelChange(event: Event): void {
    const value = (event.currentTarget as HTMLSelectElement).value;
    void workflow.updateInpaintingSettings({ model: value });
  }

  function onPaddingChange(event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    void workflow.updateInpaintingSettings({ padding: value });
  }

  function onBlurChange(event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    void workflow.updateInpaintingSettings({ blur: value });
  }

  function onNumSlicesChange(event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    uiStore.setPendingNumSlices(value);
    if (projectStore.view) void workflow.updateSliceCount(value);
  }

  let restoreInput: HTMLInputElement | undefined;

  function onRestoreChange(event: Event): void {
    const target = event.currentTarget as HTMLInputElement;
    const file = target.files?.[0];
    target.value = '';
    if (file) void workflow.restoreProject(file);
  }
</script>

<div class="configuration-tab" data-testid="tab-configuration">
  <div class="field">
    <label class="field-label" for="num-slices">Number of Slices</label>
    <input
      id="num-slices"
      type="range"
      min="2"
      max="10"
      step="1"
      data-testid="num-slices"
      value={uiStore.pendingNumSlices}
      disabled={isBusy()}
      onchange={onNumSlicesChange}
    />
    <div class="range-marks">
      {#each Array.from({ length: 9 }, (_, i) => i + 2) as mark (mark)}
        <span>{mark}</span>
      {/each}
    </div>
  </div>

  <div class="field">
    <label class="field-label" for="inpainting-model">Inpainting Model</label>
    <select
      id="inpainting-model"
      class="select"
      data-testid="inpainting-model"
      value={projectStore.view?.inpainting.model ?? inpaintModelOptions[2].value}
      disabled={isBusy()}
      onchange={onInpaintModelChange}
    >
      {#each inpaintModelOptions as option (option.value)}
        <option value={option.value}>{option.label}</option>
      {/each}
    </select>
  </div>

  <div class="field">
    <label class="field-label" for="mask-padding">Mask Padding</label>
    <input
      id="mask-padding"
      type="range"
      min="0"
      max="200"
      step="10"
      data-testid="mask-padding"
      value={projectStore.view?.inpainting.padding ?? 50}
      disabled={isBusy()}
      onchange={onPaddingChange}
    />
    <span class="slider-value">{projectStore.view?.inpainting.padding ?? 50}</span>
  </div>

  <div class="field">
    <label class="field-label" for="mask-blur">Mask Blur</label>
    <input
      id="mask-blur"
      type="range"
      min="0"
      max="200"
      step="10"
      data-testid="mask-blur"
      value={projectStore.view?.inpainting.blur ?? 50}
      disabled={isBusy()}
      onchange={onBlurChange}
    />
    <span class="slider-value">{projectStore.view?.inpainting.blur ?? 50}</span>
  </div>

  <div class="field">
    <span class="field-label">Export/Import State</span>
    <div class="state-actions">
      <label class="btn" for="restore-state-input">
        Load State
        <input
          bind:this={restoreInput}
          id="restore-state-input"
          type="file"
          accept="application/json"
          class="sr-only"
          data-testid="restore-state-input"
          onchange={onRestoreChange}
        />
      </label>
      <button type="button" class="btn" disabled title="Not available yet">Save State</button>
    </div>
  </div>
</div>

<style>
  .field {
    margin-bottom: var(--space-4);
  }

  .slider-value {
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }

  .range-marks {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }

  input[type='range'] {
    width: 100%;
  }

  .state-actions {
    display: flex;
    gap: var(--space-4);
  }

  .state-actions .btn {
    cursor: pointer;
  }
</style>
