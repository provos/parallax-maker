<script lang="ts">
  import { projectStore } from '../../state/project.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import * as workflow from '../../workflow';
  import JobCard from '../feedback/JobCard.svelte';

  // Same labels/values as components.py's DROPDOWN_DEPTH_MODEL.
  const depthOptions: Array<{ value: string; label: string }> = [
    { value: 'midas', label: 'MiDaS' },
    { value: 'dinov2', label: 'DINOv2' },
  ];

  // Keep the dropdown in sync with the project once it reports a model.
  // `settings.depthModel` is the *persisted* dropdown selection
  // (`AppState.depth_model_name`, restored from JSON - WEB-41/WEB-29; see
  // ARCHITECTURE.md's `ProjectSettingsView` doc comment) and takes
  // precedence once it has ever been set; before that (a fresh upload, where
  // `settings.depthModel` is still `""`), fall back to the top-level
  // `depthModel` (the model instance actually used for the last depth
  // generation) so the dropdown still reflects reality immediately after the
  // very first depth job, exactly like before this field existed.
  $effect(() => {
    const view = projectStore.view;
    if (!view) return;
    const model = view.settings.depthModel || view.depthModel;
    if (model) uiStore.setDepthModel(model);
  });

  function onDepthModelChange(event: Event): void {
    const value = (event.currentTarget as HTMLSelectElement).value;
    uiStore.setDepthModel(value);
    // Persists on every change, regardless of whether Regenerate is ever
    // clicked - mirrors Dash's `remember_depth_model` (WEB-29) exactly.
    if (projectStore.view) void workflow.updateSettings({ depthModel: value });
  }

  function regenerate(): void {
    if (isBusy() || !projectStore.view) return;
    void workflow.startDepth(uiStore.depthModel);
  }
</script>

<div class="mode-tab" data-testid="tab-mode">
  <span class="panel-label">Depth Map</span>
  <div class="panel depth-box">
    <img data-testid="depth-image" alt="" src={projectStore.view?.assets.depth?.url} />
  </div>

  <div class="depth-controls">
    <div>
      <label class="field-label" for="depth-model">Depth Module Algorithm</label>
      <select
        id="depth-model"
        class="select"
        data-testid="depth-model"
        value={uiStore.depthModel}
        onchange={onDepthModelChange}
        disabled={isBusy()}
      >
        {#each depthOptions as option (option.value)}
          <option value={option.value}>{option.label}</option>
        {/each}
      </select>
    </div>
    <button
      type="button"
      class="btn"
      data-testid="regenerate-depth"
      disabled={isBusy() || !projectStore.view}
      onclick={regenerate}
    >
      Regenerate Depth Map
    </button>
  </div>

  <JobCard kinds={['upload', 'depth']} testId="depth-progress" />

</div>

<style>
  /* Fills the tab panel's height; the depth box takes what the controls
     below leave over, and the depth map is contain-fitted inside it. */
  .mode-tab {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
    min-height: 0;
  }

  .depth-box {
    position: relative;
    flex: 1 1 0;
    min-height: 8rem;
    overflow: hidden;
  }

  .depth-box img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  /* See InputImagePanel.svelte: hide Chromium's "broken image" glyph until
     there is a depth map to show. */
  .depth-box img:not([src]) {
    visibility: hidden;
  }

  .depth-controls {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: var(--space-2);
    align-items: end;
    padding: var(--space-2) 0;
  }

</style>
