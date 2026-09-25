import type { Download, Locator } from '@playwright/test';

/**
 * Frontend-neutral UI operations used by the shared behavioral scenarios.
 *
 * Scenarios express *what* the user does and observe results through images,
 * the log and the test-only backend oracle (`/__e2e__/state`). The Svelte
 * frontend implements this interface (`drivers/svelte.ts`) with its own
 * selectors and gestures; the interface stayed frontend-neutral throughout
 * the Dash-to-Svelte migration so the historical Dash driver could implement
 * it too. Drivers must use real clicks and pointer events with normal hit
 * testing; no forced clicks or synthetic event dispatch.
 *
 * Project identity: `restoreFixtureState` returns the project ID, which is the
 * `appstate-*` directory name, so the backend oracle and artifact endpoints
 * stay shared.
 */

export type UiTarget = 'svelte';

/** Workflow groups; a scenario is skipped on a driver that does not support it yet. */
export type Workflow =
  | 'upload-depth-slices'
  | 'segmentation'
  | 'slice-editing'
  | 'mask-tools'
  | 'inpainting'
  | 'project-lifecycle'
  | 'configuration'
  | 'export';

export type MainTab = 'Mode' | 'Segmentation' | 'Inpainting' | 'Export' | 'Configuration';
export type SegmentationMode = 'Depth Map' | 'Instance Segmentation';
export type Modifier = 'Alt' | 'Control' | 'Meta' | 'Shift';
export type SliderName =
  | 'num-slices'
  | 'camera-distance'
  | 'max-distance'
  | 'focal-length'
  | 'displacement'
  | 'number-of-frames';

export interface UiDriver {
  readonly target: UiTarget;
  supports(workflow: Workflow): boolean;

  // Navigation
  goto(): Promise<void>;
  openTab(tab: MainTab): Promise<void>;

  // Observable elements
  mainImage(): Locator;
  depthImage(): Locator;
  /** Slice thumbnails as displayed (checkerboard-composited), in slice order. */
  sliceImages(): Locator;
  /** Inpainting candidate images, in candidate order. */
  candidateImages(): Locator;
  /** Threshold handles; there are `num_slices - 1` of them. */
  thresholdHandles(): Locator;
  log(): Locator;

  // Upload / depth / slices
  /** Uploads the server fixture input and waits for the image and depth map. */
  uploadInputImage(): Promise<void>;
  /**
   * Restores a fresh copy of the fixture project; waits for 3 slices. Returns the project ID.
   * This is baseline test setup that every driver must support; scenarios that only use it
   * as setup do not require the 'project-lifecycle' workflow.
   */
  restoreFixtureState(): Promise<string>;
  generateSlices(): Promise<void>;

  // Segmentation
  setSegmentationMode(mode: SegmentationMode): Promise<void>;
  expectSegmentationMode(mode: SegmentationMode): Promise<void>;
  /** Real click at source-image pixel (x, y) on the main image. */
  clickImagePixel(x: number, y: number, modifiers?: Modifier[]): Promise<void>;
  /**
   * Zooms in once on the main image/canvas via a real wheel gesture over
   * its center (mouse wheel, no buttons).
   */
  zoomIn(): Promise<void>;
  /**
   * Pans the main image/canvas by a real drag gesture of `(dx, dy)` CSS
   * pixels.
   */
  panBy(dx: number, dy: number): Promise<void>;
  /**
   * Resets zoom/pan back to the default view via the explicit Reset button
   * (state/viewport.svelte.ts).
   */
  resetZoom(): Promise<void>;
  /** Selects a slice by thumbnail click and waits until the backend reports it selected. */
  selectSlice(projectId: string, index: number): Promise<Locator>;
  toggleMultiPoint(): Promise<void>;
  expectMultiPointEnabled(enabled: boolean): Promise<void>;
  commitMultiPoint(): Promise<void>;
  /**
   * Asserts that a queued-point marker of the right color (green for a
   * plain/Shift point, red for a Ctrl/negative one) is currently visible for
   * each of `points`, in order (`PreviewOverlay.svelte`'s marker elements).
   */
  expectQueuedPointMarkers(points: Array<{ x: number; y: number; negative: boolean }>): Promise<void>;

  // Canvas / inpainting
  /** Paints one stroke on the mask canvas and waits until the mask is persisted. */
  drawMaskStroke(): Promise<void>;
  /** Whether the mask canvas currently shows any painted pixel. */
  maskCanvasPainted(): Promise<boolean>;
  expectGenerateEnabled(): Promise<void>;
  fillPrompts(positive: string, negative: string): Promise<void>;
  expectPrompts(positive: string, negative: string): Promise<void>;
  generateInpainting(): Promise<void>;
  fillInpainting(): Promise<void>;
  enhance(): Promise<void>;
  erase(): Promise<void>;
  /** Selects a candidate and waits until it is visibly selected and Apply is enabled. */
  selectCandidate(index: number): Promise<void>;
  applyCandidate(): Promise<void>;
  /** Undo/redo controls of the slice at `index` in the slice list. */
  undoButton(index: number): Locator;
  redoButton(index: number): Locator;

  // Slice editing / mask tools
  /** Creates a slice from the current mask (or an empty slice with none). */
  createSlice(): Promise<void>;
  /** Deletes the currently selected slice. */
  deleteSlice(): Promise<void>;
  /** Adds the current mask to the selected slice's alpha, in place. */
  addMaskToSlice(): Promise<void>;
  /** Removes the current mask from the selected slice's alpha, in place. */
  removeMaskFromSlice(): Promise<void>;
  /** Copies the composed selected slice (or full image) plus current mask to the clipboard. */
  copySlice(): Promise<void>;
  /** Blends the clipboard image into the selected slice, in place. */
  pasteSlice(): Promise<void>;
  /** Evenly redistributes slice depths. */
  balanceSlices(): Promise<void>;
  /** Sets the depth of the slice currently at `index` via its numeric input. */
  setSliceDepth(index: number, depth: number): Promise<void>;
  /** Drops a replacement image onto slice `index`'s thumbnail, creating a new version. */
  uploadSliceImage(
    index: number,
    file: { name: string; mimeType: string; buffer: Buffer },
  ): Promise<void>;
  /** Inverts the current mask (creating an all-zero mask first if none exists). */
  invertMask(): Promise<void>;
  /** Feathers (blurs) the current mask by a fixed kernel. */
  featherMask(): Promise<void>;
  /** Toggles the checkerboard vs. grayscale background for the selected-slice preview. */
  toggleCheckerboard(): Promise<void>;

  // Project / configuration
  expectDarkTheme(): Promise<void>;
  expectSliderValue(name: SliderName, value: number): Promise<void>;
  setSlider(name: SliderName, value: number): Promise<void>;
  expectDepthModel(label: string): Promise<void>;
  expectInpaintingModel(label: string): Promise<void>;
  /** Selects a depth-module option by its visible label (e.g. "MiDaS", "DINOv2"). */
  selectDepthModel(label: string): Promise<void>;
  /** Selects an inpainting-model option by its visible label (e.g. "Automatic1111"). */
  selectInpaintingModel(label: string): Promise<void>;
  /** Fills the Automatic1111/ComfyUI server-address field and commits the value. */
  setExternalServer(address: string): Promise<void>;
  /** Clicks "Test Connection" for the currently selected inpainting model. */
  testExternalConnection(): Promise<void>;
  /** Asserts the server-address field's success/failure/neutral highlight. */
  expectExternalConnectionStatus(status: 'success' | 'failure' | 'none'): Promise<void>;
  /** Fills the StabilityAI/fal.ai API-key field and commits the value. */
  setApiKey(key: string): Promise<void>;
  /** Clicks "Test API Key" for the currently selected inpainting model. */
  validateApiKey(): Promise<void>;
  /** Asserts the API-key field's success/failure/neutral highlight. */
  expectApiKeyStatus(status: 'success' | 'failure' | 'none'): Promise<void>;

  // Project lifecycle
  /** Clicks "Save State" (writes the project to disk; no browser download). */
  saveState(): Promise<void>;
  /**
   * Restores the given previously-saved project JSON bytes back into the UI
   * (e.g. from `fetchArtifact(page, projectId, 'appstate.json')` after
   * `saveState()`), for a save -> restore round trip against the same project.
   */
  restoreStateFromBytes(buffer: Buffer): Promise<void>;

  // Export
  /** Clicks glTF export and returns the resulting browser download. */
  exportGltf(): Promise<Download>;
  exportAnimation(): Promise<void>;
  /** Toggles the "Support Depth of Field Effect" export checkbox to `enabled`. */
  setDofEnabled(enabled: boolean): Promise<void>;
  /** Clicks "Upscale Textures" and waits for the resulting log line. */
  upscaleTextures(): Promise<void>;
  /** Clicks a slice thumbnail's download icon and returns the resulting browser download. */
  downloadSlice(index: number): Promise<Download>;
}
