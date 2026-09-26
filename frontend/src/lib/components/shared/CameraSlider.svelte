<script lang="ts">
  /**
   * One camera setting as a range slider (state/cameraDraft.svelte.ts): it
   * edits the draft as it moves and commits all camera fields on change.
   *
   * Deliberately not disabled while busy: each slider commits on every
   * discrete change (`onchange` fires per arrow-key step), and disabling it
   * mid-flight would make a fast arrow-key sequence race its own commit.
   */
  import { cameraDraftStore, type CameraDraft } from '../../state/cameraDraft.svelte';

  let {
    field,
    label,
    testId,
    min,
    max,
    step = 1,
    disabled = false,
    format = (value: number) => String(Math.round(value)),
  }: {
    field: keyof CameraDraft;
    label: string;
    testId: string;
    min: number;
    max: number;
    step?: number;
    disabled?: boolean;
    format?: (value: number) => string;
  } = $props();

  const id = $derived(`camera-slider-${testId}`);
</script>

<div class="g3">
  <label class="muted" for={id}>{label}</label>
  <input
    {id}
    type="range"
    {min}
    {max}
    {step}
    data-testid={testId}
    value={cameraDraftStore.draft[field]}
    {disabled}
    oninput={(event) => cameraDraftStore.set(field, Number((event.currentTarget as HTMLInputElement).value))}
    onchange={() => cameraDraftStore.commit()}
  />
  <span class="mono value" data-testid={`${testId}-value`}>{format(cameraDraftStore.draft[field])}</span>
</div>

<style>
  .g3 {
    display: grid;
    grid-template-columns: 96px minmax(0, 1fr) 44px;
    align-items: center;
    gap: 10px;
  }

  .muted {
    color: var(--color-text-secondary);
  }

  .value {
    text-align: right;
    font-size: 12px;
  }
</style>
