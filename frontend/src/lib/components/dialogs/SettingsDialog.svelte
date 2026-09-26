<script lang="ts">
  /**
   * The Settings dialog (docs/redesign/HANDOFF.md §3): one scrolling column
   * of sections - Inpainting (model, A1111/ComfyUI server + Test, ComfyUI
   * workflow, API key + Test), Depth, Masks, Slicing, Project (load/save
   * state) and Appearance - with a nav that jumps between them.
   */
  import { projectStore } from '../../state/project.svelte';
  import { uiStore, type SettingsSection } from '../../state/ui.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import Dialog from '../shared/Dialog.svelte';

  const SECTIONS: { id: SettingsSection; label: string }[] = [
    { id: 'inpainting', label: 'Inpainting' },
    { id: 'depth', label: 'Depth' },
    { id: 'masks', label: 'Masks' },
    { id: 'slicing', label: 'Slicing' },
    { id: 'project', label: 'Project' },
    { id: 'appearance', label: 'Appearance' },
  ];
  const sectionEls: Partial<Record<SettingsSection, HTMLElement>> = {};

  function showSection(id: SettingsSection): void {
    uiStore.setSettingsSection(id);
    sectionEls[id]?.scrollIntoView?.({ block: 'start' });
  }

  // Opening at a section (e.g. "Model settings" from Inpaint) jumps to it.
  $effect(() => {
    if (uiStore.dialog !== 'settings') return;
    const id = uiStore.settingsSection;
    queueMicrotask(() => sectionEls[id]?.scrollIntoView?.({ block: 'start' }));
  });

  // Same labels/values as ModeTab.svelte's depth model select.
  const depthOptions: Array<{ value: string; label: string }> = [
    { value: 'midas', label: 'MiDaS' },
    { value: 'dinov2', label: 'DINOv2' },
  ];

  function onDepthModelChange(event: Event): void {
    const value = (event.currentTarget as HTMLSelectElement).value;
    uiStore.setDepthModel(value);
    if (projectStore.view) void workflow.updateSettings({ depthModel: value });
  }

  function onNumSlicesChange(event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    uiStore.setPendingNumSlices(value);
    if (projectStore.view) void workflow.updateSliceCount(value);
  }

  function setTheme(theme: 'dark' | 'light'): void {
    if (uiStore.theme !== theme) void workflow.toggleDarkMode();
  }

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

  // A project loaded from here replaces the one behind the dialog, so the
  // dialog closes once it has loaded.
  async function onRestoreChange(event: Event): Promise<void> {
    const target = event.currentTarget as HTMLInputElement;
    const file = target.files?.[0];
    target.value = '';
    if (!file) return;
    const before = projectStore.view;
    await workflow.restoreProject(file);
    if (projectStore.view && projectStore.view !== before && uiStore.dialog === 'settings') uiStore.closeDialog();
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

<Dialog name="settings" title="Settings" testId="settings-dialog" width={720} height={600}>
  <div class="body" data-testid="tab-configuration">
    <nav class="nav" aria-label="Settings sections">
      {#each SECTIONS as section (section.id)}
        <button
          type="button"
          class="nav-link"
          aria-current={uiStore.settingsSection === section.id ? 'true' : undefined}
          data-testid={`settings-nav-${section.id}`}
          onclick={() => showSection(section.id)}
        >
          {section.label}
        </button>
      {/each}
    </nav>
    <div class="sections">
      <section bind:this={sectionEls.inpainting} data-testid="settings-inpainting">
        <h3 class="label">Inpainting</h3>
        <label class="muted" for="inpainting-model">Model</label>
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

        {#if showServerPanel}
          <div class="group" data-testid="external-server-panel">
            <label class="muted" for="external-server-address">A1111 / ComfyUI server address</label>
            <div class="row">
              <input
                id="external-server-address"
                type="text"
                class="text-input mono"
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
                Test connection
              </button>
            </div>

            {#if showWorkflowUpload}
              <div class="group" data-testid="comfyui-workflow-panel">
                <span class="muted">ComfyUI workflow</span>
                <label class="btn upload-btn" for="comfyui-workflow-input">
                  {projectStore.view?.inpainting.hasWorkflow ? 'Replace workflow…' : 'Upload workflow (JSON)…'}
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
                  <span class="faint" data-testid="comfyui-workflow-status">Workflow uploaded</span>
                {/if}
              </div>
            {/if}
          </div>
        {/if}

        {#if showApiKeyPanel}
          <div class="group" data-testid="api-key-panel">
            <label class="muted" for="api-key">API key (StabilityAI / Fal.ai)</label>
            <div class="row">
              <input
                id="api-key"
                type="password"
                class="text-input mono"
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
                Test API key
              </button>
            </div>
          </div>
        {/if}
      </section>

      <section bind:this={sectionEls.depth} data-testid="settings-depth">
        <h3 class="label">Depth</h3>
        <label class="muted" for="settings-depth-model">Depth model</label>
        <select
          id="settings-depth-model"
          class="select"
          data-testid="settings-depth-model"
          value={uiStore.depthModel}
          disabled={isBusy()}
          onchange={onDepthModelChange}
        >
          {#each depthOptions as option (option.value)}
            <option value={option.value}>{option.label}</option>
          {/each}
        </select>
      </section>

      <section bind:this={sectionEls.masks} data-testid="settings-masks">
        <h3 class="label">Masks</h3>
        <div class="g3">
          <label class="muted" for="mask-padding">Padding</label>
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
          <span class="mono value">{projectStore.view?.inpainting.padding ?? 50}</span>
          <label class="muted" for="mask-blur">Blur</label>
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
          <span class="mono value">{projectStore.view?.inpainting.blur ?? 50}</span>
        </div>
      </section>

      <section bind:this={sectionEls.slicing} data-testid="settings-slicing">
        <h3 class="label">Slicing</h3>
        <div class="g3">
          <label class="muted" for="settings-num-slices">Slices</label>
          <input
            id="settings-num-slices"
            type="range"
            min="2"
            max="10"
            step="1"
            data-testid="settings-num-slices"
            value={uiStore.pendingNumSlices}
            disabled={isBusy()}
            onchange={onNumSlicesChange}
          />
          <span class="mono value">{uiStore.pendingNumSlices}</span>
        </div>
        <p class="faint">How many depth bands Slices › Split by depth makes.</p>
      </section>

      <section bind:this={sectionEls.project} data-testid="settings-project">
        <h3 class="label">Project</h3>
        <div class="row">
          <label class="btn upload-btn" for="restore-state-input">
            Load state…
            <input
              bind:this={restoreInput}
              id="restore-state-input"
              type="file"
              accept="application/json"
              class="sr-only"
              data-testid="restore-state-input"
              onchange={(event) => void onRestoreChange(event)}
            />
          </label>
          <button
            type="button"
            class="btn"
            data-testid="save-state"
            disabled={isBusy() || !projectStore.view}
            onclick={onSaveState}
          >
            Save state
          </button>
        </div>
      </section>

      <section bind:this={sectionEls.appearance} data-testid="settings-appearance">
        <h3 class="label">Appearance</h3>
        <div class="seg" role="radiogroup" aria-label="Theme">
          <button
            type="button"
            role="radio"
            aria-checked={uiStore.theme === 'dark'}
            data-testid="settings-theme-dark"
            onclick={() => setTheme('dark')}
          >
            Dark
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={uiStore.theme === 'light'}
            data-testid="settings-theme-light"
            onclick={() => setTheme('light')}
          >
            Light
          </button>
        </div>
      </section>
    </div>
  </div>
</Dialog>

<style>
  .body {
    display: flex;
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
  }

  .nav {
    width: 170px;
    flex-shrink: 0;
    border-right: 1px solid var(--color-border);
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .nav-link {
    height: var(--control-h);
    padding: 0 10px;
    border: none;
    border-radius: var(--radius-md);
    background: transparent;
    color: var(--color-text-secondary);
    font: inherit;
    text-align: left;
    cursor: pointer;
  }

  .nav-link:hover {
    background: var(--color-surface-hover);
    color: var(--color-text);
  }

  .nav-link[aria-current='true'] {
    background: var(--color-surface-hover);
    color: var(--color-text);
  }

  .sections {
    flex: 1 1 auto;
    min-width: 0;
    overflow-y: auto;
    padding: 4px 18px 18px;
  }

  section {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-top: 16px;
  }

  .label {
    margin: 0;
  }

  .group {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .muted {
    color: var(--color-text-secondary);
  }

  .faint {
    margin: 0;
    color: var(--color-text-muted);
    font-size: var(--text-small);
  }

  .row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  .row .text-input {
    flex: 1 1 auto;
  }

  .g3 {
    display: grid;
    grid-template-columns: 80px minmax(0, 1fr) 44px;
    align-items: center;
    gap: 10px;
  }

  .value {
    text-align: right;
    font-size: 12px;
  }

  .text-input.status-success {
    border-color: var(--color-success);
    background-color: color-mix(in srgb, var(--color-success) 15%, transparent);
  }

  .text-input.status-failure {
    border-color: var(--color-danger);
    background-color: color-mix(in srgb, var(--color-danger) 15%, transparent);
  }

  .upload-btn {
    cursor: pointer;
    align-self: flex-start;
  }

  .seg {
    display: inline-flex;
    align-self: flex-start;
    background: var(--color-surface-raised);
    border-radius: 7px;
    padding: 2px;
    gap: 2px;
    border: 1px solid var(--color-border);
  }

  .seg button {
    min-width: 72px;
    height: var(--control-h-sm);
    border: none;
    border-radius: 5px;
    background: transparent;
    color: var(--color-text-secondary);
    font: inherit;
    font-weight: 500;
    cursor: pointer;
  }

  .seg button[aria-checked='true'] {
    background: var(--color-surface-hover);
    color: var(--color-text);
    box-shadow: 0 0 0 1px var(--color-border-strong);
  }
</style>
