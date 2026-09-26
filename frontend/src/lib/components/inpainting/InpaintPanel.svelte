<script lang="ts">
  /**
   * The Inpaint step's Inspector panel (docs/redesign/HANDOFF.md §3): paint
   * the holes on the selected slice (the Brush tool, over the canvas), describe what
   * fills them, generate three candidates, pick one and apply it. "Extend
   * edges" (outpainting) is reserved but not built yet (HANDOFF §7).
   */
  import { untrack } from 'svelte';
  import { projectStore } from '../../state/project.svelte';
  import { isBusy } from '../../state/busy.svelte';
  import { canvasSaveStore } from '../../state/canvas.svelte';
  import { uiStore } from '../../state/ui.svelte';
  import * as workflow from '../../workflow';
  import { registerInpaintActions } from '../../shortcuts';
  import JobCard from '../feedback/JobCard.svelte';

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

  const hasCandidates = $derived((candidates?.images.length ?? 0) > 0);
  const hasMask = $derived(!!view?.slices.find((s) => s.index === selectedSlice)?.mask);

  // Sub-step progress: 1 paint the holes, 2 describe, 3 generate and pick.
  const stepState = (n: 1 | 2 | 3): 'done' | 'now' | 'todo' => {
    const current = hasCandidates ? 3 : hasMask ? 2 : 1;
    return n < current ? 'done' : n === current ? 'now' : 'todo';
  };

  // Entering the Inpaint step with nothing selected selects the farthest
  // slice (HANDOFF §4) - once per entry, so clearing the selection while
  // here is left alone.
  let previousStep: string | null = null;
  $effect(() => {
    const step = uiStore.step;
    const entering = step === 'inpaint' && previousStep !== 'inpaint';
    previousStep = step;
    if (!entering) return;
    untrack(() => {
      const current = projectStore.view;
      if (!current || current.selectedSlice != null || current.slices.length === 0 || isBusy()) return;
      const farthest = current.slices.reduce((a, b) => (b.depth < a.depth ? b : a));
      void workflow.selectSlice(farthest.index);
    });
  });

  function apply(): void {
    if (!canApply) return;
    void workflow.applyInpaintingCandidate();
  }

  // Ctrl+Enter, 1 2 3 and A (lib/shortcuts.ts).
  $effect(() =>
    registerInpaintActions({
      generate,
      pick: (index) => {
        if (index < (candidates?.images.length ?? 0)) selectCandidate(index);
      },
      apply,
    }),
  );
</script>

<div class="inpaint-panel" data-testid="tab-inpainting">
  <section class="sec">
    <div class="sec-title">
      <h3 class="label accent">Inpaint</h3>
      <button type="button" class="btn btn-ghost btn-sm link" onclick={() => uiStore.openSettings('inpainting')}>
        Model settings
      </button>
    </div>
    <div class="seg" role="radiogroup" aria-label="Inpaint mode">
      <button type="button" role="radio" aria-checked="true" data-testid="inpaint-mode-holes">Fill holes</button>
      <button
        type="button"
        role="radio"
        aria-checked="false"
        data-testid="inpaint-mode-extend"
        title="Coming soon: grow the slice beyond the image edges"
        disabled
      >
        Extend edges <span class="badge-plan">PLANNED</span>
      </button>
    </div>
    {#if !hasSlice}
      <p class="faint" data-testid="inpaint-no-slice">Select a layer to inpaint.</p>
    {/if}
    <ol class="substeps">
      <li data-state={stepState(1)}>
        <span class="sub">1</span>
        <span>Paint the holes on the slice with the brush.</span>
      </li>
      <li data-state={stepState(2)}>
        <span class="sub">2</span>
        <span>Describe what should fill them.</span>
      </li>
      <li data-state={stepState(3)}>
        <span class="sub">3</span>
        <span>Generate, pick a candidate, and apply it.</span>
      </li>
    </ol>
  </section>

  <section class="sec">
    <label class="muted" for="positive-prompt">Prompt</label>
    <textarea
      id="positive-prompt"
      class="text-input"
      data-testid="positive-prompt"
      placeholder="What should fill the holes…"
      rows="2"
      disabled={!hasSlice}
      bind:value={positiveDraft}
      onchange={commitPrompts}
    ></textarea>
    <label class="muted" for="negative-prompt">Negative prompt</label>
    <textarea
      id="negative-prompt"
      class="text-input"
      data-testid="negative-prompt"
      placeholder="What to avoid…"
      rows="2"
      disabled={!hasSlice}
      bind:value={negativeDraft}
      onchange={commitPrompts}
    ></textarea>

    <div class="g3">
      <label class="muted" for="inpaint-strength">Strength</label>
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
      <span class="mono value">{(view?.inpainting.strength ?? 0.8).toFixed(2)}</span>
    </div>
    <div class="g3">
      <label class="muted" for="inpaint-guidance">Guidance</label>
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
      <span class="mono value">{(view?.inpainting.guidanceScale ?? 7.5).toFixed(2)}</span>
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

    <button
      type="button"
      class="btn"
      class:btn-primary={!hasCandidates}
      data-testid="generate-inpainting"
      title="Generate three candidates (Ctrl+Enter)"
      disabled={!canGenerate}
      onclick={generate}
    >
      Generate 3 candidates
    </button>
    <div class="row">
      <button type="button" class="btn btn-sm grow" data-testid="fill-inpainting" title="Fill every transparent pixel of the slice" disabled={!canGenerate} onclick={fill}>
        Fill
      </button>
      <button type="button" class="btn btn-sm grow" data-testid="enhance-inpainting" title="Re-render the slice at higher quality" disabled={!canGenerate} onclick={enhance}>
        Enhance
      </button>
      <button type="button" class="btn btn-sm grow" data-testid="erase-inpainting" title="Erase the painted area from the slice" disabled={!canGenerate} onclick={erase}>
        Erase
      </button>
    </div>

    <JobCard kinds={['inpainting', 'inpainting-mutate']} testId="inpainting-progress" />
  </section>

  <section class="sec">
    <h3 class="label">Candidates</h3>
    <div class="candidates" data-testid="candidate-strip" role="listbox" aria-label="Inpainting candidates">
      {#each candidates?.images ?? [] as candidate, index (index)}
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
        <!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
        <img
          data-testid="candidate-image"
          alt={`inpainting candidate ${index + 1}`}
          src={candidate.url}
          role="option"
          tabindex="0"
          aria-selected={selectedCandidate === index}
          class:selected={selectedCandidate === index}
          onclick={() => selectCandidate(index)}
        />
      {:else}
        <p class="faint">Candidates appear here after you generate.</p>
      {/each}
    </div>
    <button
      type="button"
      class="btn"
      class:btn-primary={hasCandidates}
      data-testid="apply-inpainting"
      title="Apply the picked candidate to the slice (A)"
      disabled={!canApply}
      onclick={apply}
    >
      Apply selected candidate
    </button>
  </section>
</div>

<style>
  .inpaint-panel {
    display: flex;
    flex-direction: column;
    margin: calc(-1 * var(--space-3)) -14px;
  }

  .sec {
    padding: var(--space-3) 14px;
    border-bottom: 1px solid var(--color-border);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .sec-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .label {
    margin: 0;
  }

  .accent {
    color: var(--color-primary-soft-text);
  }

  .link {
    color: var(--color-primary-soft-text);
  }

  .faint {
    margin: 0;
    color: var(--color-text-muted);
    font-size: var(--text-small);
  }

  .muted {
    color: var(--color-text-secondary);
  }

  .seg {
    display: flex;
    background: var(--color-surface-raised);
    border-radius: 7px;
    padding: 2px;
    gap: 2px;
    border: 1px solid var(--color-border);
  }

  .seg button {
    flex: 1 1 0;
    height: var(--control-h-sm);
    border: none;
    border-radius: 5px;
    background: transparent;
    color: var(--color-text-secondary);
    font: inherit;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }

  .seg button[aria-checked='true'] {
    background: var(--color-surface-hover);
    color: var(--color-text);
    box-shadow: 0 0 0 1px var(--color-border-strong);
  }

  .seg button:disabled {
    cursor: default;
  }

  .badge-plan {
    font-size: 10px;
    font-weight: 600;
    color: var(--color-text-muted);
    border: 1px dashed var(--color-border-strong);
    border-radius: var(--radius-sm);
    padding: 0 4px;
  }

  .substeps {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .substeps li {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
  }

  .sub {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 10.5px;
    font-weight: 600;
    flex-shrink: 0;
    border: 1px solid var(--color-border-strong);
    color: var(--color-text-secondary);
    box-sizing: border-box;
  }

  li[data-state='done'] .sub {
    background: var(--color-success-soft);
    border-color: var(--color-success-line);
    color: var(--color-success);
  }

  li[data-state='now'] .sub {
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: var(--color-primary-text);
  }

  textarea {
    resize: vertical;
  }

  .g3 {
    display: grid;
    grid-template-columns: 70px minmax(0, 1fr) 44px;
    align-items: center;
    gap: 10px;
  }

  .value {
    text-align: right;
    font-size: 12px;
  }

  .checkbox-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--color-text-secondary);
  }

  .row {
    display: flex;
    gap: var(--space-2);
  }

  .grow {
    flex: 1 1 0;
  }


  .candidates {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: var(--space-2);
  }

  .candidates .faint {
    grid-column: 1 / -1;
  }

  .candidates img {
    width: 100%;
    display: block;
    cursor: pointer;
    border: 1px solid var(--color-border-strong);
    border-radius: var(--radius-md);
    box-sizing: border-box;
  }

  .candidates img.selected {
    border: 2px solid var(--color-selection);
  }
</style>
