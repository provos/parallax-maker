import type { Download, Locator } from '@playwright/test';

/**
 * Frontend-neutral UI operations used by the shared behavioral scenarios.
 *
 * Scenarios express *what* the user does and observe results through images,
 * the log and the test-only backend oracle (`/__e2e__/state`). Each frontend
 * (Dash today, Svelte during the migration) implements this interface with its
 * own selectors and gestures. Drivers must use real clicks and pointer events
 * with normal hit testing; no forced clicks or synthetic event dispatch.
 *
 * Project identity: `restoreFixtureState` returns the project ID, which is the
 * `appstate-*` directory name for every frontend so the backend oracle and
 * artifact endpoints stay shared.
 */

export type UiTarget = 'dash' | 'svelte';

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

export type MainTab = 'Segmentation' | 'Inpainting' | 'Export' | 'Configuration';
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
  /** Selects a slice by thumbnail click and waits until the backend reports it selected. */
  selectSlice(projectId: string, index: number): Promise<Locator>;
  toggleMultiPoint(): Promise<void>;
  expectMultiPointEnabled(enabled: boolean): Promise<void>;
  commitMultiPoint(): Promise<void>;

  // Canvas / inpainting
  /** Paints one stroke on the mask canvas and waits until the mask is persisted. */
  drawMaskStroke(): Promise<void>;
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

  // Project / configuration
  expectDarkTheme(): Promise<void>;
  expectSliderValue(name: SliderName, value: number): Promise<void>;
  setSlider(name: SliderName, value: number): Promise<void>;
  expectDepthModel(label: string): Promise<void>;
  expectInpaintingModel(label: string): Promise<void>;

  // Export
  /** Clicks glTF export and returns the resulting browser download. */
  exportGltf(): Promise<Download>;
  exportAnimation(): Promise<void>;
}
