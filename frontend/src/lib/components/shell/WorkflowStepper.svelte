<script lang="ts">
  /**
   * Steps 1-7 of the workflow (docs/redesign/HANDOFF.md §3-4). Each step is
   * done, current or todo; clicking one moves the Inspector to it. "Done"
   * comes from ProjectView where it can, and from this session's progress
   * (uiStore) where ProjectView has no flag for it.
   */
  import { uiStore, WORKFLOW_STEPS, type WorkflowStep } from '../../state/ui.svelte';
  import { projectStore } from '../../state/project.svelte';

  const done = $derived.by((): Record<WorkflowStep, boolean> => {
    const view = projectStore.view;
    const slices = view?.slices ?? [];
    return {
      image: !!view?.assets.input,
      depth: !!view?.assets.depth,
      slices: slices.length > 0,
      inpaint: uiStore.inpainted,
      ground: slices.some((slice) => slice.isGround),
      preview: uiStore.previewed,
      export: uiStore.exported,
    };
  });
</script>

<nav class="stepper" aria-label="Workflow steps" data-testid="workflow-stepper">
  {#each WORKFLOW_STEPS as { step, label }, index (step)}
    {#if index > 0}<span class="link" aria-hidden="true"></span>{/if}
    <button
      type="button"
      class="step"
      data-testid={`step-${step}`}
      data-state={uiStore.step === step ? 'current' : done[step] ? 'done' : 'todo'}
      aria-current={uiStore.step === step ? 'step' : undefined}
      onclick={() => uiStore.setStep(step)}
    >
      <span class="number mono" aria-hidden="true">{done[step] && uiStore.step !== step ? '✓' : index + 1}</span>
      {label}
    </button>
  {/each}
</nav>

<style>
  .stepper {
    flex: 1 1 auto;
    min-width: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 2px;
    overflow: hidden;
  }

  .link {
    width: 12px;
    height: 1px;
    flex: 0 1 auto;
    background: var(--color-border-strong);
  }

  .step {
    height: 30px;
    border: none;
    border-radius: var(--radius-md);
    background: transparent;
    color: var(--color-text-muted);
    font: inherit;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0 var(--space-2);
    cursor: pointer;
    white-space: nowrap;
  }

  .step:hover {
    background: var(--color-surface-hover);
  }

  .step:focus-visible {
    outline: 2px solid var(--color-selection);
    outline-offset: 1px;
  }

  .step[data-state='done'] {
    color: var(--color-success);
  }

  .step[data-state='current'] {
    background: var(--color-primary-soft);
    color: var(--color-primary-soft-text);
    box-shadow: inset 0 0 0 1px var(--color-primary);
  }

  .number {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 10.5px;
    font-weight: 600;
    border: 1px solid var(--color-border-strong);
    box-sizing: border-box;
  }

  .step[data-state='done'] .number {
    background: var(--color-success-soft);
    border-color: var(--color-success-line);
  }

  .step[data-state='current'] .number {
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: var(--color-primary-text);
  }

  /* Narrow windows: numbers only. */
  @media (max-width: 1180px) {
    .step {
      font-size: 0;
      gap: 0;
      padding: 0 6px;
    }
    .number {
      font-size: 10.5px;
    }
  }
</style>
