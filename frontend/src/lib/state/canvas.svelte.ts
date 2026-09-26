/**
 * Tracks the most recent in-flight inpainting-related save that shares the
 * backend's per-project mutation lock with Generate/Apply/Erase/slice
 * selection, so none of those can race a not-yet-persisted request into a
 * `409 busy`: `MaskCanvas.svelte` registers a promise here on pointerup
 * (`PUT .../mask`), `InpaintPanel.svelte` registers one when a prompt
 * textarea commits (`PUT .../prompts`), and `workflow.ts`'s `selectSlice`/
 * inpainting mutation functions all call `flush()` before doing anything
 * else. This is the "explicit lifecycle" the
 * migration handoff asks for (see docs/SVELTE_5_MIGRATION_HANDOFF.md,
 * "Deterministic harness details"): unlike Dash's canvas (which only saves
 * on `mouseout` and auto-clears on every main-image change, including
 * mid-inpainting-job updates - see PARITY.md's "Known quirks", CLI-11), nothing
 * here can silently discard a painted-but-unsaved stroke.
 *
 * A window resize never needs to flow through this store at all: the canvas
 * backing store is sized to the source image's own pixel dimensions (not the
 * CSS-rendered size Dash uses), so resizing only rescales the element via
 * CSS - the pixel data itself is untouched.
 */
function createCanvasSaveStore() {
  let pending: Promise<void> | null = null;

  /**
   * Registers `promise` as the current pending save. `MaskCanvas.svelte`
   * itself is responsible for surfacing a save failure (e.g. via
   * `logStore`); this store only needs to know when the attempt is done, so
   * `flush()` never throws on a caller's behalf.
   */
  function register(promise: Promise<void>): void {
    const settled = promise.catch(() => {});
    pending = settled;
    void settled.then(() => {
      // Only clear if nothing newer has been registered meanwhile.
      if (pending === settled) pending = null;
    });
  }

  /** Awaits the current pending save, if any. Safe to call when idle. */
  async function flush(): Promise<void> {
    if (pending) await pending;
  }

  /** Test-only: forgets any pending save without awaiting it. */
  function reset(): void {
    pending = null;
  }

  return { register, flush, reset };
}

export const canvasSaveStore = createCanvasSaveStore();
export type CanvasSaveStore = ReturnType<typeof createCanvasSaveStore>;
