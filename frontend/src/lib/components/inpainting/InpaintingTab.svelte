<script lang="ts">
  /**
   * The Inpainting tab: prompts, Strength/Guidance sliders, a display-only
   * "Crop to region of interest" checkbox, Generate/Fill/Enhance/Erase,
   * the candidate strip and Apply -- see components.py's
   * `make_inpainting_container`/`make_inpainting_container_callbacks` and
   * `docs/svelte-migration/reference/dash-1440-inpainting.png` for the Dash
   * layout this mirrors. The paint canvas itself (and its Clear/Erase/Load
   * tool row) lives in InputImagePanel.svelte/MaskCanvas.svelte, overlaid on
   * the shared main image, not here.
   */
  import { projectStore } from '../../state/project.svelte';
  import { jobStore } from '../../state/jobs.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import { canvasSaveStore } from '../../state/canvas.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import * as workflow from '../../workflow';
  import HelpTooltip from '../shared/HelpTooltip.svelte';
  import { INPAINTING_HELP_TEXTS } from '../../helpTexts';

  const view = $derived(projectStore.view);
  const selectedSlice = $derived(view?.selectedSlice ?? null);
  const hasSlice = $derived(selectedSlice !== null);

  // -- Prompts: loaded from the selected slice's saved prompts whenever the
  // selection changes (not on every unrelated project-view update, so a
  // draft in progress is never clobbered mid-edit) -- mirrors Dash's
  // TEXT_POSITIVE_PROMPT/TEXT_NEGATIVE_PROMPT, which `display_slice`
  // populates from `state.positive_prompts`/`negative_prompts` on selection.

  let positiveDraft = $state('');
  let negativeDraft = $state('');
  let lastSelected: number | null | undefined = undefined;

  $effect(() => {
    const index = selectedSlice;
    if (index === lastSelected) return;
    lastSelected = index;
    const slice = index !== null ? view?.slices.find((s) => s.index === index) : undefined;
    positiveDraft = slice?.positivePrompt ?? '';
    negativeDraft = slice?.negativePrompt ?? '';
  });

  // A prompt PUT and a subsequent Generate/slice-selection change both take
  // the same server-side project lock; register the save with
  // `canvasSaveStore` (the same "pending save" flush every one of those
  // already awaits) so one can never race the other into a `409 busy`, the
  // same way MaskCanvas.svelte's own mask-save registers there.
  function commitPrompts(): void {
    if (selectedSlice === null) return;
    canvasSaveStore.register(workflow.updateInpaintingPrompts(selectedSlice, positiveDraft, negativeDraft));
  }

  // -- Strength / Guidance Scale: project-level settings (not per-slice),
  // sent on release (`onchange`), matching Dash's slider-driven sliders 1:1
  // in range/step/default (components.py's SLIDER_INPAINT_STRENGTH/
  // SLIDER_INPAINT_GUIDANCE).

  function onStrengthChange(event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    void workflow.updateInpaintingSettings({ strength: value });
  }

  function onGuidanceChange(event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    void workflow.updateInpaintingSettings({ guidanceScale: value });
  }

  // "Crop to region of interest" only affects *generation* display-only in
  // Dash too: generation always passes crop=True regardless of the checkbox
  // (see the migration handoff's "Inpainting, masks and versions" section).
  // It also gates MaskCanvas.svelte's ROI-box preview on mask save (Dash's
  // CLI-07/CMP-24), so it lives in `uiStore` rather than as local state here
  // so MaskCanvas.svelte can read the same value.

  // -- Generate / Fill / Enhance / Erase.

  const canGenerate = $derived(hasSlice && !isBusy());
  const generating = $derived(jobStore.active === 'inpainting');
  const progressPercent = $derived(Math.round((generating ? jobStore.progress : 0) * 100));

  function generate(): void {
    if (!canGenerate) return;
    void workflow.generateInpainting('paint', positiveDraft, negativeDraft);
  }

  function fill(): void {
    if (!canGenerate) return;
    void workflow.generateInpainting('fill', positiveDraft, negativeDraft);
  }

  function enhance(): void {
    if (!canGenerate) return;
    void workflow.generateInpainting('enhance', positiveDraft, negativeDraft);
  }

  function erase(): void {
    if (!canGenerate) return;
    void workflow.eraseInpainting();
  }

  // -- Candidates.

  const candidates = $derived(view?.inpainting.candidates ?? null);
  const selectedCandidate = $derived(view?.inpainting.selectedCandidate ?? null);
  const canApply = $derived(!!candidates && selectedCandidate !== null && !isBusy());

  function selectCandidate(index: number): void {
    if (!candidates || isBusy()) return;
    void workflow.selectInpaintingCandidate(candidates.generationId, index);
  }

  function apply(): void {
    if (!canApply) return;
    void workflow.applyInpaintingCandidate();
  }
</script>

<div class="inpainting-tab" data-testid="tab-inpainting">
  <div class="tab-header">
    <HelpTooltip label="Inpainting" texts={INPAINTING_HELP_TEXTS} />
  </div>
  <div class="field">
    <label class="field-label" for="positive-prompt">Positive Prompt</label>
    <textarea
      id="positive-prompt"
      data-testid="positive-prompt"
      placeholder="Enter a positive generative AI prompt..."
      disabled={!hasSlice}
      bind:value={positiveDraft}
      onchange={commitPrompts}
    ></textarea>
  </div>

  <div class="field">
    <label class="field-label" for="negative-prompt">Negative Prompt</label>
    <textarea
      id="negative-prompt"
      data-testid="negative-prompt"
      placeholder="Enter a negative prompt..."
      disabled={!hasSlice}
      bind:value={negativeDraft}
      onchange={commitPrompts}
    ></textarea>
  </div>

  <div class="sliders-row">
    <div class="field">
      <label class="field-label" for="inpaint-strength">Strength</label>
      <input
        id="inpaint-strength"
        type="range"
        min="0"
        max="1"
        step="0.01"
        data-testid="inpaint-strength"
        value={view?.inpainting.strength ?? 0.8}
        onchange={onStrengthChange}
      />
      <span class="slider-value">{(view?.inpainting.strength ?? 0.8).toFixed(2)}</span>
    </div>
    <div class="field">
      <label class="field-label" for="inpaint-guidance">Guidance Scale</label>
      <input
        id="inpaint-guidance"
        type="range"
        min="1"
        max="15"
        step="0.25"
        data-testid="inpaint-guidance"
        value={view?.inpainting.guidanceScale ?? 7.5}
        onchange={onGuidanceChange}
      />
      <span class="slider-value">{(view?.inpainting.guidanceScale ?? 7.5).toFixed(2)}</span>
    </div>
  </div>

  <label class="checkbox-row">
    <input
      type="checkbox"
      data-testid="crop-to-roi"
      checked={uiStore.cropToRoi}
      onchange={(event) => uiStore.setCropToRoi((event.currentTarget as HTMLInputElement).checked)}
    />
    Crop to region of interest
  </label>

  <div class="action-row">
    <button type="button" class="btn" data-testid="generate-inpainting" disabled={!canGenerate} onclick={generate}>
      Generate
    </button>
    <button type="button" class="btn" data-testid="fill-inpainting" disabled={!canGenerate} onclick={fill}>
      Fill
    </button>
    <button type="button" class="btn" data-testid="enhance-inpainting" disabled={!canGenerate} onclick={enhance}>
      Enhance
    </button>
    <button type="button" class="btn" data-testid="erase-inpainting" disabled={!canGenerate} onclick={erase}>
      Erase
    </button>
  </div>

  <div class="progress-bar" data-testid="inpainting-progress">
    <div class="progress-bar-fill" style={`width: ${generating ? progressPercent : 0}%`}></div>
  </div>

  <div class="panel candidate-strip" data-testid="candidate-strip">
    {#each candidates?.images ?? [] as candidate, index (index)}
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
      <img
        data-testid="candidate-image"
        alt={`inpainting candidate ${index}`}
        src={candidate.url}
        role="option"
        tabindex="0"
        aria-selected={selectedCandidate === index}
        class:selected={selectedCandidate === index}
        onclick={() => selectCandidate(index)}
      />
    {/each}
  </div>

  <button type="button" class="btn btn-primary" data-testid="apply-inpainting" disabled={!canApply} onclick={apply}>
    Apply Selected Image
  </button>
</div>

<style>
  .inpainting-tab {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: var(--space-2);
  }

  .tab-header {
    display: flex;
    justify-content: flex-end;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
  }

  textarea {
    width: 100%;
    min-height: 4rem;
    font: inherit;
    background-color: var(--color-bg);
    color: var(--color-text);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-2);
  }

  .sliders-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-4);
  }

  .sliders-row input[type='range'] {
    width: 100%;
  }

  .slider-value {
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }

  .checkbox-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  .action-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }

  .candidate-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--space-2);
    min-height: 2rem;
    padding: var(--space-2);
  }

  .candidate-strip img {
    width: 100%;
    display: block;
    cursor: pointer;
    border: 2px solid transparent;
    border-radius: var(--radius-md);
  }

  .candidate-strip img.selected {
    border-color: var(--color-success);
  }

  .btn-primary {
    align-self: flex-start;
    background-color: var(--color-success);
    color: var(--color-success-text);
  }

  .btn-primary:disabled {
    background-color: var(--color-disabled-bg);
    color: var(--color-disabled-text);
  }
</style>
