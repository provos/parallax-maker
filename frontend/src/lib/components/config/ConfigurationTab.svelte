<script lang="ts">
  /**
   * The Configuration tab: number-of-slices/mask padding/blur (already
   * existed), plus the inpainting-model provider panels (Automatic1111/
   * ComfyUI server address + Test Connection + workflow upload; StabilityAI/
   * fal.ai API key + Validate), and Save/Load State - see components.py's
   * `make_configuration_div`/`make_inpainting_container_callbacks` and
   * `docs/svelte-migration/reference/dash-1440-configuration.png` for the
   * Dash layout this mirrors.
   */
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import HelpTooltip from '../shared/HelpTooltip.svelte';
  import { CONFIGURATION_HELP_TEXTS } from '../../helpTexts';

  // Same options/values/default as components.py's DROPDOWN_INPAINT_MODEL.
  const inpaintModelOptions: Array<{ value: string; label: string }> = [
    { value: 'kandinsky-community/kandinsky-2-2-decoder-inpaint', label: 'Kandinsky' },
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

  const currentModel = $derived(projectStore.view?.inpainting.model ?? inpaintModelOptions[2].value);
  // CMP-10/CMP-13: which provider panel(s) are visible for the current model.
  const showServerPanel = $derived(currentModel === 'automatic1111' || currentModel === 'comfyui');
  const showWorkflowUpload = $derived(currentModel === 'comfyui');
  const showApiKeyPanel = $derived(currentModel === 'stabilityai' || currentModel.startsWith('falai-'));
  // CMP-09: no mask-blur support for stabilityai.
  const blurDisabled = $derived(isBusy() || currentModel === 'stabilityai');

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

  let restoreInput: HTMLInputElement | undefined;

  function onRestoreChange(event: Event): void {
    const target = event.currentTarget as HTMLInputElement;
    const file = target.files?.[0];
    target.value = '';
    if (file) void workflow.restoreProject(file);
  }

  function onSaveState(): void {
    void workflow.saveProject();
  }

  // -- Automatic1111 / ComfyUI server address + Test Connection (CMP-11/12).
  // `externalServer` always comes back from the server (not a credential),
  // so - like Mask Padding/Blur above - the field reads straight from the
  // project view; no local draft needed.

  function onServerAddressChange(event: Event): void {
    const value = (event.currentTarget as HTMLInputElement).value;
    uiStore.setExternalConnectionStatus('none');
    void workflow.updateInpaintingSettings({ externalServer: value });
  }

  function onTestConnection(): void {
    const view = projectStore.view;
    if (!view) return;
    void workflow.probeExternalServer(view.inpainting.model, view.inpainting.externalServer);
  }

  function onWorkflowChange(event: Event): void {
    const target = event.currentTarget as HTMLInputElement;
    const file = target.files?.[0];
    target.value = '';
    if (file) void workflow.uploadInpaintingWorkflow(file);
  }

  // -- StabilityAI / fal.ai API key + Validate (CMP-14/15). Write-only: never
  // pre-filled from the server (see `InpaintingView`, which never carries
  // `apiKey`), so this is a purely local draft, unlike the server field above.

  let apiKeyDraft = $state('');

  function onApiKeyChange(): void {
    uiStore.setApiKeyStatus('none');
    void workflow.updateInpaintingSettings({ apiKey: apiKeyDraft });
  }

  function onValidateApiKey(): void {
    const view = projectStore.view;
    if (!view) return;
    void workflow.probeApiKey(view.inpainting.model, apiKeyDraft);
  }
</script>

<div class="configuration-tab" data-testid="tab-configuration">
  <div class="tab-header">
    <HelpTooltip label="Configuration" texts={CONFIGURATION_HELP_TEXTS} />
  </div>
  <div class="field">
    <label class="field-label" for="inpainting-model">Inpainting Model</label>
    <select
      id="inpainting-model"
      class="select"
      data-testid="inpainting-model"
      value={currentModel}
      disabled={isBusy()}
      onchange={onInpaintModelChange}
    >
      {#each inpaintModelOptions as option (option.value)}
        <option value={option.value}>{option.label}</option>
      {/each}
    </select>
  </div>

  {#if showServerPanel}
    <div class="field" data-testid="external-server-panel">
      <label class="field-label" for="external-server-address">A1111/ComfyUI Server Address</label>
      <div class="row">
        <input
          id="external-server-address"
          type="text"
          class="text-input"
          data-testid="external-server-address"
          data-status={uiStore.externalConnectionStatus}
          class:status-success={uiStore.externalConnectionStatus === 'success'}
          class:status-failure={uiStore.externalConnectionStatus === 'failure'}
          value={projectStore.view?.inpainting.externalServer ?? 'localhost:7860'}
          disabled={isBusy()}
          onchange={onServerAddressChange}
        />
        <button
          type="button"
          class="btn"
          data-testid="external-test-connection"
          disabled={isBusy() || !projectStore.view}
          onclick={onTestConnection}
        >
          Test Connection
        </button>
      </div>

      {#if showWorkflowUpload}
        <div class="field" data-testid="comfyui-workflow-panel">
          <label class="field-label" for="comfyui-workflow-input">ComfyUI Workflow</label>
          <label class="btn upload-btn" for="comfyui-workflow-input">
            Drag and Drop or Upload
            <input
              id="comfyui-workflow-input"
              type="file"
              accept="application/json"
              class="sr-only"
              data-testid="comfyui-workflow-input"
              onchange={onWorkflowChange}
            />
          </label>
          {#if projectStore.view?.inpainting.hasWorkflow}
            <span class="hint" data-testid="comfyui-workflow-status">Workflow uploaded</span>
          {/if}
        </div>
      {/if}
    </div>
  {/if}

  {#if showApiKeyPanel}
    <div class="field" data-testid="api-key-panel">
      <label class="field-label" for="api-key">API Key (StabilityAI / Fal.ai)</label>
      <div class="row">
        <input
          id="api-key"
          type="password"
          class="text-input"
          data-testid="api-key"
          data-status={uiStore.apiKeyStatus}
          class:status-success={uiStore.apiKeyStatus === 'success'}
          class:status-failure={uiStore.apiKeyStatus === 'failure'}
          bind:value={apiKeyDraft}
          disabled={isBusy()}
          onchange={onApiKeyChange}
        />
        <button
          type="button"
          class="btn"
          data-testid="validate-api-key"
          disabled={isBusy() || !projectStore.view}
          onclick={onValidateApiKey}
        >
          Test API Key
        </button>
      </div>
    </div>
  {/if}

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
      disabled={blurDisabled}
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
      <button
        type="button"
        class="btn"
        data-testid="save-state"
        disabled={isBusy() || !projectStore.view}
        onclick={onSaveState}
      >
        Save State
      </button>
    </div>
  </div>
</div>

<style>
  .tab-header {
    display: flex;
    justify-content: flex-end;
    margin-bottom: var(--space-2);
  }

  .field {
    margin-bottom: var(--space-4);
  }

  .slider-value {
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }



  input[type='range'] {
    width: 100%;
  }

  .row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  .text-input {
    flex: 1;
    background-color: var(--color-bg);
    color: var(--color-text);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-1) var(--space-2);
  }

  .text-input.status-success {
    border-color: var(--color-success);
    background-color: color-mix(in srgb, var(--color-success) 15%, transparent);
  }

  .text-input.status-failure {
    border-color: var(--color-danger-strong);
    background-color: color-mix(in srgb, var(--color-danger-strong) 15%, transparent);
  }

  .upload-btn {
    display: inline-block;
    cursor: pointer;
  }

  .hint {
    font-size: 0.75rem;
    color: var(--color-text-muted);
    margin-left: var(--space-2);
  }

  .state-actions {
    display: flex;
    gap: var(--space-4);
  }

  .state-actions .btn {
    cursor: pointer;
  }
</style>
