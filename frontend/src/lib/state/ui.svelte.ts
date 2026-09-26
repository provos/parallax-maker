/**
 * Transient, client-only UI state: active tabs, theme, and controls whose
 * value is not (yet) persisted by the backend. Per the architecture doc
 * ("Frontend layout"), backend state comes only from `ProjectView`; this
 * store never holds anything the server would consider authoritative.
 */

/** What the canvas shows (docs/redesign/HANDOFF.md §3). */
export type CanvasView = 'input' | 'depth' | 'slice' | 'composite' | 'parallax' | '3d';
/** The active canvas tool (HANDOFF.md §5). */
export type CanvasTool = 'pan' | 'segment' | 'brush' | 'horizon';
/** The Inspector panel shown: one per workflow step (image and depth share one). */
export type MainTab = 'Mode' | 'Segmentation' | 'Inpainting' | 'Ground' | 'Preview';
/** The modal dialogs (HANDOFF.md §3). */
export type DialogName = 'export' | 'settings' | 'shortcuts';
/** The Settings dialog's sections, in nav order. */
export type SettingsSection = 'inpainting' | 'depth' | 'masks' | 'slicing' | 'project' | 'appearance';
/** The workflow stepper's steps (docs/redesign/HANDOFF.md §4). */
export type WorkflowStep = 'image' | 'depth' | 'slices' | 'inpaint' | 'ground' | 'preview' | 'export';
export type SegmentationMode = 'depth' | 'segment';
export type Theme = 'light' | 'dark';
/** Highlight state of a configuration probe (Test Connection / Validate API Key). */
export type ProbeStatus = 'success' | 'failure' | 'none';

export const MAIN_TABS: MainTab[] = ['Mode', 'Segmentation', 'Inpainting', 'Ground', 'Preview'];

export const WORKFLOW_STEPS: { step: WorkflowStep; label: string }[] = [
  { step: 'image', label: 'Image' },
  { step: 'depth', label: 'Depth' },
  { step: 'slices', label: 'Slices' },
  { step: 'inpaint', label: 'Inpaint' },
  { step: 'ground', label: 'Ground' },
  { step: 'preview', label: 'Preview' },
  { step: 'export', label: 'Export' },
];

export const CANVAS_VIEWS: { view: CanvasView; label: string; key: string }[] = [
  { view: 'input', label: 'Input', key: 'I' },
  { view: 'depth', label: 'Depth', key: 'M' },
  { view: 'slice', label: 'Slice', key: 'L' },
  { view: 'composite', label: 'Composite', key: 'C' },
  { view: 'parallax', label: 'Parallax 2D', key: 'P' },
  { view: '3d', label: '3D', key: 'Shift+P' },
];

/** Each step's default view and tool (HANDOFF.md §4); Export keeps both. */
const STEP_DEFAULTS: Partial<Record<WorkflowStep, { view: CanvasView; tool: CanvasTool }>> = {
  image: { view: 'input', tool: 'pan' },
  depth: { view: 'depth', tool: 'pan' },
  slices: { view: 'input', tool: 'segment' },
  inpaint: { view: 'slice', tool: 'brush' },
  ground: { view: 'composite', tool: 'horizon' },
  preview: { view: 'parallax', tool: 'pan' },
};

/** The step a tool belongs to (choosing the tool moves the step there). */
const TOOL_STEPS: Partial<Record<CanvasTool, WorkflowStep>> = {
  segment: 'slices',
  brush: 'inpaint',
  horizon: 'ground',
};

/** The views each tool works in; choosing a tool elsewhere switches view. */
const TOOL_VIEWS: Record<CanvasTool, CanvasView[]> = {
  pan: ['input', 'depth', 'slice', 'composite', 'parallax', '3d'],
  segment: ['input'],
  brush: ['slice', 'input'],
  horizon: ['composite', 'input', 'depth'],
};

/**
 * Which Inspector panel each step shows. Export has no panel of its own: the
 * step opens the Export dialog over the step you were on.
 */
const STEP_PANELS: Record<WorkflowStep, MainTab> = {
  image: 'Mode',
  depth: 'Mode',
  slices: 'Segmentation',
  inpaint: 'Inpainting',
  ground: 'Ground',
  preview: 'Preview',
  export: 'Preview',
};

/** Matches components.py's DROPDOWN_DEPTH_MODEL default. */
const DEFAULT_DEPTH_MODEL = 'dinov2';
/** Matches components.py's SLIDER_NUM_SLICES default. */
const DEFAULT_NUM_SLICES = 3;

function createUiStore() {
  let view = $state<CanvasView>('input');
  // The main-image URL of the latest camera render, so Parallax 2D can tell
  // whether the server's display image is a render or something else.
  let renderedMainUrl = $state<string | null>(null);
  let tool = $state<CanvasTool>('pan');
  let mainTab = $state<MainTab>('Mode');
  let step = $state<WorkflowStep>('image');
  let theme = $state<Theme>('dark');
  let logOpen = $state(false);
  let dialog = $state<DialogName | null>(null);
  let settingsSection = $state<SettingsSection>('inpainting');
  // Progress the stepper can't read from ProjectView (HANDOFF.md §4 "done"
  // rules): tracked for this browser session only.
  let inpainted = $state(false);
  let previewed = $state(false);
  let exported = $state(false);
  // Segment Anything (object) selection is the main way to make slices.
  let segmentationMode = $state<SegmentationMode>('segment');
  let depthModel = $state<string>(DEFAULT_DEPTH_MODEL);
  let pendingNumSlices = $state<number>(DEFAULT_NUM_SLICES);
  // Configuration tab probe highlights (CMP-11/CMP-12/CMP-13/CMP-14/CMP-15):
  // reset to 'none' whenever the underlying field is edited, set to
  // 'success'/'failure' by the corresponding Test Connection/Validate probe.
  let externalConnectionStatus = $state<ProbeStatus>('none');
  let apiKeyStatus = $state<ProbeStatus>('none');
  // "Crop to region of interest" (components.py's CHECKLIST_REGION_OF_INTEREST,
  // defaults checked): purely client-side UI state in Dash too, read by both
  // InpaintPanel.svelte (display only there -- generation always passes
  // crop=True regardless, see its own comment) and MaskCanvas.svelte (which
  // sends it as the mask-save request's `cropToRegion` flag, gating whether
  // the server computes a bounding box for PreviewOverlay.svelte's ROI-box
  // preview -- Dash's CLI-07/CMP-24).
  let cropToRoi = $state<boolean>(true);

  /** Makes `next` the current step and shows its panel (view/tool untouched). */
  function moveToStep(next: WorkflowStep): void {
    step = next;
    mainTab = STEP_PANELS[next];
    if (next === 'preview') previewed = true;
  }

  return {
    get view(): CanvasView {
      return view;
    },
    /** Shows a view; a tool that doesn't work there falls back to Pan. */
    setView(next: CanvasView): void {
      view = next;
      if (!TOOL_VIEWS[tool].includes(next)) tool = 'pan';
      if (next === 'parallax' || next === '3d') moveToStep('preview');
    },

    get renderedMainUrl(): string | null {
      return renderedMainUrl;
    },
    setRenderedMainUrl(url: string | null): void {
      renderedMainUrl = url;
    },

    get tool(): CanvasTool {
      return tool;
    },
    /** Picks a tool; its step becomes current and the view one it works in. */
    setTool(next: CanvasTool): void {
      tool = next;
      if (!TOOL_VIEWS[next].includes(view)) view = TOOL_VIEWS[next][0];
      const toolStep = TOOL_STEPS[next];
      if (toolStep) moveToStep(toolStep);
    },

    /** The Inspector panel shown (see STEP_PANELS). */
    get mainTab(): MainTab {
      return mainTab;
    },
    setMainTab(tab: MainTab): void {
      mainTab = tab;
    },

    get step(): WorkflowStep {
      return step;
    },
    /**
     * Moves to a workflow step: its Inspector panel, default view and tool.
     * Export instead opens the Export dialog and leaves the step alone.
     */
    setStep(next: WorkflowStep): void {
      if (next === 'export') {
        dialog = 'export';
        return;
      }
      dialog = null;
      moveToStep(next);
      const defaults = STEP_DEFAULTS[next];
      if (defaults) {
        view = defaults.view;
        tool = defaults.tool;
      }
    },

    /** The open modal dialog, if any. */
    get dialog(): DialogName | null {
      return dialog;
    },
    openDialog(name: DialogName): void {
      dialog = name;
    },
    closeDialog(): void {
      dialog = null;
    },
    get settingsSection(): SettingsSection {
      return settingsSection;
    },
    /** Opens Settings at a section. */
    openSettings(section: SettingsSection = settingsSection): void {
      settingsSection = section;
      dialog = 'settings';
    },
    setSettingsSection(section: SettingsSection): void {
      settingsSection = section;
    },

    get logOpen(): boolean {
      return logOpen;
    },
    toggleLog(): void {
      logOpen = !logOpen;
    },

    /** Forgets this session's progress (a different project is now loaded). */
    resetProgress(): void {
      inpainted = false;
      previewed = false;
      exported = false;
    },
    /** A new project: forget progress and start over at the first step. */
    resetSession(): void {
      inpainted = false;
      previewed = false;
      exported = false;
      step = 'image';
      mainTab = STEP_PANELS.image;
      dialog = null;
      view = 'input';
      tool = 'pan';
      renderedMainUrl = null;
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
      view = 'input';
      tool = 'pan';
      renderedMainUrl = null;
      mainTab = 'Mode';
      step = 'image';
      theme = 'dark';
      logOpen = false;
      dialog = null;
      settingsSection = 'inpainting';
      inpainted = false;
      previewed = false;
      exported = false;
      segmentationMode = 'segment';
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
