/**
 * Transient, client-only UI state: active tabs, theme, and controls whose
 * value is not (yet) persisted by the backend. Per the architecture doc
 * ("Frontend layout"), backend state comes only from `ProjectView`; this
 * store never holds anything the server would consider authoritative.
 */

export type ViewerTab = '2D' | '3D';
export type MainTab = 'Mode' | 'Segmentation' | 'Inpainting' | 'Export' | 'Configuration';
/** The workflow stepper's steps (docs/redesign/HANDOFF.md §4). */
export type WorkflowStep = 'image' | 'depth' | 'slices' | 'inpaint' | 'ground' | 'preview' | 'export';
export type SegmentationMode = 'depth' | 'segment';
export type Theme = 'light' | 'dark';
/** Highlight state of a configuration probe (Test Connection / Validate API Key). */
export type ProbeStatus = 'success' | 'failure' | 'none';

export const MAIN_TABS: MainTab[] = ['Mode', 'Segmentation', 'Inpainting', 'Export', 'Configuration'];

export const WORKFLOW_STEPS: { step: WorkflowStep; label: string }[] = [
  { step: 'image', label: 'Image' },
  { step: 'depth', label: 'Depth' },
  { step: 'slices', label: 'Slices' },
  { step: 'inpaint', label: 'Inpaint' },
  { step: 'ground', label: 'Ground' },
  { step: 'preview', label: 'Preview' },
  { step: 'export', label: 'Export' },
];

/**
 * Which of the pre-redesign tab bodies the Inspector shows for a step,
 * until each step gets its own panel (HANDOFF.md §12).
 */
const STEP_PANELS: Record<WorkflowStep, MainTab> = {
  image: 'Mode',
  depth: 'Mode',
  slices: 'Segmentation',
  inpaint: 'Inpainting',
  ground: 'Segmentation',
  preview: 'Export',
  export: 'Export',
};

/** Matches components.py's DROPDOWN_DEPTH_MODEL default. */
const DEFAULT_DEPTH_MODEL = 'dinov2';
/** Matches components.py's SLIDER_NUM_SLICES default. */
const DEFAULT_NUM_SLICES = 3;

function createUiStore() {
  let viewerTab = $state<ViewerTab>('2D');
  let mainTab = $state<MainTab>('Mode');
  let step = $state<WorkflowStep>('image');
  let theme = $state<Theme>('dark');
  let logOpen = $state(false);
  // Progress the stepper can't read from ProjectView (HANDOFF.md §4 "done"
  // rules): tracked for this browser session only.
  let inpainted = $state(false);
  let previewed = $state(false);
  let exported = $state(false);
  let segmentationMode = $state<SegmentationMode>('depth');
  let depthModel = $state<string>(DEFAULT_DEPTH_MODEL);
  let pendingNumSlices = $state<number>(DEFAULT_NUM_SLICES);
  // Configuration tab probe highlights (CMP-11/CMP-12/CMP-13/CMP-14/CMP-15):
  // reset to 'none' whenever the underlying field is edited, set to
  // 'success'/'failure' by the corresponding Test Connection/Validate probe.
  let externalConnectionStatus = $state<ProbeStatus>('none');
  let apiKeyStatus = $state<ProbeStatus>('none');
  // "Crop to region of interest" (components.py's CHECKLIST_REGION_OF_INTEREST,
  // defaults checked): purely client-side UI state in Dash too, read by both
  // InpaintingTab.svelte (display only there -- generation always passes
  // crop=True regardless, see its own comment) and MaskCanvas.svelte (which
  // sends it as the mask-save request's `cropToRegion` flag, gating whether
  // the server computes a bounding box for PreviewOverlay.svelte's ROI-box
  // preview -- Dash's CLI-07/CMP-24).
  let cropToRoi = $state<boolean>(true);

  return {
    get viewerTab(): ViewerTab {
      return viewerTab;
    },
    setViewerTab(tab: ViewerTab): void {
      viewerTab = tab;
      if (tab === '3D') previewed = true;
    },

    /** The pre-redesign tab body the Inspector shows (see STEP_PANELS). */
    get mainTab(): MainTab {
      return mainTab;
    },
    setMainTab(tab: MainTab): void {
      mainTab = tab;
    },

    get step(): WorkflowStep {
      return step;
    },
    /** Moves to a workflow step and shows that step's Inspector panel. */
    setStep(next: WorkflowStep): void {
      step = next;
      mainTab = STEP_PANELS[next];
      if (next === 'preview') previewed = true;
    },

    get logOpen(): boolean {
      return logOpen;
    },
    toggleLog(): void {
      logOpen = !logOpen;
    },

    get inpainted(): boolean {
      return inpainted;
    },
    markInpainted(): void {
      inpainted = true;
    },
    get previewed(): boolean {
      return previewed;
    },
    markPreviewed(): void {
      previewed = true;
    },
    get exported(): boolean {
      return exported;
    },
    markExported(): void {
      exported = true;
    },

    get theme(): Theme {
      return theme;
    },
    toggleTheme(): void {
      theme = theme === 'light' ? 'dark' : 'light';
    },
    setTheme(next: Theme): void {
      theme = next;
    },

    get segmentationMode(): SegmentationMode {
      return segmentationMode;
    },
    setSegmentationMode(mode: SegmentationMode): void {
      segmentationMode = mode;
    },

    get depthModel(): string {
      return depthModel;
    },
    setDepthModel(model: string): void {
      depthModel = model;
    },

    get pendingNumSlices(): number {
      return pendingNumSlices;
    },
    setPendingNumSlices(value: number): void {
      pendingNumSlices = value;
    },

    get externalConnectionStatus(): ProbeStatus {
      return externalConnectionStatus;
    },
    setExternalConnectionStatus(status: ProbeStatus): void {
      externalConnectionStatus = status;
    },

    get apiKeyStatus(): ProbeStatus {
      return apiKeyStatus;
    },
    setApiKeyStatus(status: ProbeStatus): void {
      apiKeyStatus = status;
    },

    get cropToRoi(): boolean {
      return cropToRoi;
    },
    setCropToRoi(value: boolean): void {
      cropToRoi = value;
    },

    /** Test-only: restores default values so stores don't leak between tests. */
    reset(): void {
      viewerTab = '2D';
      mainTab = 'Mode';
      step = 'image';
      theme = 'dark';
      logOpen = false;
      inpainted = false;
      previewed = false;
      exported = false;
      segmentationMode = 'depth';
      depthModel = DEFAULT_DEPTH_MODEL;
      pendingNumSlices = DEFAULT_NUM_SLICES;
      externalConnectionStatus = 'none';
      apiKeyStatus = 'none';
      cropToRoi = true;
    },
  };
}

export const uiStore = createUiStore();
export type UiStore = ReturnType<typeof createUiStore>;
