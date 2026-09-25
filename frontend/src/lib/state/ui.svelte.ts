/**
 * Transient, client-only UI state: active tabs, theme, and controls whose
 * value is not (yet) persisted by the backend. Per the architecture doc
 * ("Frontend layout"), backend state comes only from `ProjectView`; this
 * store never holds anything the server would consider authoritative.
 */

export type ViewerTab = '2D' | '3D';
export type MainTab = 'Mode' | 'Segmentation' | 'Inpainting' | 'Export' | 'Configuration';
export type SegmentationMode = 'depth' | 'segment';
export type Theme = 'light' | 'dark';
/** Highlight state of a configuration probe (Test Connection / Validate API Key). */
export type ProbeStatus = 'success' | 'failure' | 'none';

export const MAIN_TABS: MainTab[] = ['Mode', 'Segmentation', 'Inpainting', 'Export', 'Configuration'];

/** Matches components.py's DROPDOWN_DEPTH_MODEL default. */
const DEFAULT_DEPTH_MODEL = 'dinov2';
/** Matches components.py's SLIDER_NUM_SLICES default. */
const DEFAULT_NUM_SLICES = 3;

function createUiStore() {
  let viewerTab = $state<ViewerTab>('2D');
  let mainTab = $state<MainTab>('Mode');
  let theme = $state<Theme>('light');
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
    },

    get mainTab(): MainTab {
      return mainTab;
    },
    setMainTab(tab: MainTab): void {
      mainTab = tab;
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
      theme = 'light';
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
