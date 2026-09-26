/**
 * Orchestrates API calls with the reactive stores: applies `ProjectView`s
 * through `projectStore.applyView` (revision-guarded), tracks locally
 * in-flight requests through `jobStore`, and mirrors errors and job
 * completion into `logStore` (client-side entries plus a log re-fetch, per
 * the architecture doc's "Logs" behavior). Components call these functions
 * rather than `lib/api/client.ts` directly, so the busy/log/error handling
 * stays in one place and is unit-testable without rendering a component.
 */

import * as api from './api/client';
import { ApiError } from './api/client';
import type {
  InpaintingGenerateMode,
  InpaintingSettingsRequest,
  ProjectSettingsRequest,
  SegmentationMode,
} from './api/types';
import { projectStore } from './state/project.svelte';
import { jobStore } from './state/jobs.svelte';
import { logStore } from './state/logs.svelte';
import { canvasSaveStore } from './state/canvas.svelte';
import { uiStore } from './state/ui.svelte';

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}

async function refreshLogs(projectId: string): Promise<void> {
  await logStore.refresh(projectId);
}

/**
 * Uploads a new source image, then immediately starts a depth job using
 * `depthModel` (the currently selected value of the Mode tab's dropdown).
 */
export async function uploadImage(file: File, depthModel: string): Promise<void> {
  jobStore.begin('upload');
  let projectId: string | null = null;
  try {
    let view = await api.createProject(file);
    projectId = view.id;
    uiStore.resetSession();
    // A new project keeps the theme the user is already looking at.
    const dark = uiStore.theme === 'dark';
    if (view.settings.darkMode !== dark) view = await api.updateSettings(view.id, { darkMode: dark });
    projectStore.applyView(view);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
    jobStore.end();
    return;
  }
  jobStore.end();
  if (projectId) await refreshLogs(projectId);
  await startDepth(depthModel);
  // The depth map is ready: the next thing to do is cut the image into slices.
  if (projectStore.view?.assets.depth) uiStore.setStep('slices');
}

/** Starts (or restarts) the depth job for the current project and polls it to completion. */
export async function startDepth(model: string): Promise<void> {
  const projectId = projectStore.view?.id;
  if (!projectId) return;

  jobStore.begin('depth');
  try {
    const { job } = await api.startDepth(projectId, model);
    const finished = await api.pollJob(job.id, {
      onProgress: (j) => jobStore.setProgress(j.progress),
    });
    if (finished.project) projectStore.applyView(finished.project);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(projectId);
  }
}

/** Starts the slice-generation job for the current project and polls it to completion. */
export async function generateSlices(): Promise<void> {
  const projectId = projectStore.view?.id;
  if (!projectId) return;

  jobStore.begin('slices');
  try {
    const { job } = await api.startSlices(projectId);
    const finished = await api.pollJob(job.id, {
      onProgress: (j) => jobStore.setProgress(j.progress),
    });
    if (finished.project) projectStore.applyView(finished.project);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(projectId);
  }
}

/**
 * Sends the full interior thresholds `values` array with the current
 * revision as `baseRevision`. On `stale_revision`, refetches the project
 * once and retries with the fresh revision (per the architecture doc's
 * "Concurrency" section).
 */
export async function updateThresholds(values: number[]): Promise<void> {
  const view = projectStore.view;
  if (!view) return;

  jobStore.begin('thresholds');
  try {
    let result: Awaited<ReturnType<typeof api.setThresholds>>;
    try {
      result = await api.setThresholds(view.id, values, view.revision);
    } catch (err) {
      if (err instanceof ApiError && err.code === 'stale_revision') {
        const fresh = await api.getProject(view.id);
        projectStore.applyView(fresh);
        result = await api.setThresholds(view.id, values, fresh.revision);
      } else {
        throw err;
      }
    }
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/** Updates the slice count for the current project (Configuration tab). */
export async function updateSliceCount(numSlices: number): Promise<void> {
  const view = projectStore.view;
  if (!view) return;

  jobStore.begin('slice-count');
  try {
    const result = await api.setSliceCount(view.id, numSlices);
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/**
 * Selects (or, with `slice: null`, deselects) a slice for display and
 * segmentation input (`PUT .../selection`, sync). Matches Dash's
 * `display_slice`: the caller (SegmentationTab.svelte) is responsible for
 * sending `null` when the clicked slice is already selected, mirroring
 * Dash's click-to-toggle.
 */
export async function selectSlice(slice: number | null): Promise<void> {
  const view = projectStore.view;
  if (!view) return;

  // A pending mask save targets the *currently* selected slice; let it land
  // before the selection (and thus which slice a mask PUT would even be
  // valid for) changes out from under it. See state/canvas.svelte.ts.
  await canvasSaveStore.flush();

  jobStore.begin('selection');
  try {
    const result = await api.updateSelection(view.id, slice);
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/**
 * Sends one segmentation click (`POST .../segmentation/click`, a job) and
 * polls it to completion, mirroring Dash's `click_event`. `mode` is the
 * Mode Selector's current value ("depth" or "instance"); `shiftKey`/
 * `ctrlKey` are the browser click event's modifier keys, read as-is (see
 * InputImagePanel.svelte's note on `ctrlKey` vs `metaKey`).
 */
export async function clickSegmentation(
  x: number,
  y: number,
  mode: SegmentationMode,
  shiftKey: boolean,
  ctrlKey: boolean,
): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  const projectId = view.id;

  jobStore.begin('segmentation');
  try {
    const { job } = await api.segmentationClick(projectId, { x, y, mode, shiftKey, ctrlKey });
    const finished = await api.pollJob(job.id, {
      onProgress: (j) => jobStore.setProgress(j.progress),
    });
    if (finished.project) projectStore.applyView(finished.project);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(projectId);
  }
}

/** Commits the queued multi-point selection (`POST .../segmentation/commit`, a job). */
export async function commitMultiPoint(): Promise<void> {
  const projectId = projectStore.view?.id;
  if (!projectId) return;

  jobStore.begin('segmentation');
  try {
    const { job } = await api.segmentationCommit(projectId);
    const finished = await api.pollJob(job.id, {
      onProgress: (j) => jobStore.setProgress(j.progress),
    });
    if (finished.project) projectStore.applyView(finished.project);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(projectId);
  }
}

/** Toggles multi-point mode (`PUT .../segmentation/multi-point`, sync); always clears the queue. */
export async function setMultiPointMode(enabled: boolean): Promise<void> {
  const view = projectStore.view;
  if (!view) return;

  jobStore.begin('multi-point');
  try {
    const result = await api.setMultiPointMode(view.id, enabled);
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

// --- Slice editing / mask tools --------------------------------------------
//
// Dash's own buttons (webui.py's delete/add-mask/remove-mask/copy/paste
// handlers) never disable themselves based on selection/mask/clipboard state;
// instead they run unconditionally and log a plain-text no-op message when a
// precondition is missing (see components.py/webui.py referenced in
// ARCHITECTURE.md's "Slice editing and mask-tool endpoints"). The API layer
// enforces the same preconditions with structured errors instead (409/400),
// so the functions below reproduce Dash's *exact* precedence and wording as
// a client-side check before ever calling the API - both so the log text
// matches the shared e2e scenarios byte-for-byte, and so we don't need a
// slice index to call an endpoint that requires one (add-mask/remove-mask/
// delete all key off the currently selected slice).
//
// Precedence per Dash source:
//   delete:        selection required                       ("No slice selected")
//   add/remove:     mask required, then selection required    ("No mask selected", "No slice selected")
//   copy:           mask required                             ("No mask selected")
//   paste:          clipboard required, then selection required ("Nothing in the clipboard", "No slice selected")

async function runSliceMutation(
  kind: Parameters<typeof jobStore.begin>[0],
  call: (id: string) => Promise<Awaited<ReturnType<typeof api.createSlice>>>,
): Promise<boolean> {
  const view = projectStore.view;
  if (!view) return false;

  jobStore.begin(kind);
  try {
    const result = await call(view.id);
    projectStore.applyView(result);
    return true;
  } catch (err) {
    logStore.pushClient(errorMessage(err));
    return false;
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/**
 * Moves the preview camera over the slice cards and shows the re-rendered
 * parallax view (Dash's `navigate_image` buttons). Deselects any slice.
 */
export async function navigateCamera(direction: api.CameraDirection): Promise<void> {
  if (await runSliceMutation('navigate', (id) => api.navigateCamera(id, direction))) {
    uiStore.markPreviewed();
    uiStore.setRenderedMainUrl(projectStore.view?.mainImage?.url ?? null);
  }
}

/**
 * Marks the selected slice as the scene's ground plane, or unmarks it (at
 * most one slice is the ground). A logged no-op without a selection.
 */
export async function toggleGroundPlane(): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  const selected = view.slices.find((s) => s.index === view.selectedSlice);
  if (!selected) {
    logStore.pushClient('No slice selected', 'info');
    return;
  }
  await runSliceMutation('slice-editing', (id) =>
    api.setGroundPlane(id, selected.index, !selected.isGround),
  );
}

/** Horizon on the ground mask's top edge; ground under the nearest card's foot. */
export async function fitGround(): Promise<void> {
  await runSliceMutation('settings', (id) => api.fitGround(id));
}

/** Moves the horizon to image row `row` (the server derives the camera pitch). */
export async function setHorizonRow(row: number): Promise<void> {
  const camera = projectStore.view?.settings.camera;
  if (!camera) return;
  await updateSettings({
    camera: {
      distance: camera.distance,
      focalLength: camera.focalLength,
      maxDistance: camera.maxDistance,
      horizonRow: row,
    },
  });
}

/** Creates a slice from the current mask, or an empty slice if there is none. */
export async function createSlice(): Promise<void> {
  await runSliceMutation('slice-editing', (id) => api.createSlice(id));
}

/** Deletes the selected slice (`#delete-slice-button` in Dash); a logged no-op with none selected. */
export async function deleteSlice(): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  if (view.selectedSlice === null) {
    logStore.pushClient('No slice selected', 'info');
    return;
  }
  await runSliceMutation('slice-editing', (id) => api.deleteSlice(id, view.selectedSlice as number));
}

/** Adds the current mask to the selected slice's alpha in place; a logged no-op without a mask/selection. */
export async function addMaskToSlice(): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  if (!view.segmentation.hasMask) {
    logStore.pushClient('No mask selected', 'info');
    return;
  }
  if (view.selectedSlice === null) {
    logStore.pushClient('No slice selected', 'info');
    return;
  }
  await runSliceMutation('slice-editing', (id) => api.addMaskToSlice(id, view.selectedSlice as number));
}

/** Removes the current mask from the selected slice's alpha in place; same preconditions as addMaskToSlice. */
export async function removeMaskFromSlice(): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  if (!view.segmentation.hasMask) {
    logStore.pushClient('No mask selected', 'info');
    return;
  }
  if (view.selectedSlice === null) {
    logStore.pushClient('No slice selected', 'info');
    return;
  }
  await runSliceMutation('slice-editing', (id) =>
    api.removeMaskFromSlice(id, view.selectedSlice as number),
  );
}

/** Copies the composed selected slice (or full image) plus the current mask to the clipboard. */
export async function copySlice(): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  if (!view.segmentation.hasMask) {
    logStore.pushClient('No mask selected', 'info');
    return;
  }
  await runSliceMutation('slice-editing', (id) => api.copyToClipboard(id));
}

/** Blends the clipboard image into the selected slice in place. */
export async function pasteSlice(): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  if (!view.clipboard) {
    logStore.pushClient('Nothing in the clipboard', 'info');
    return;
  }
  if (view.selectedSlice === null) {
    logStore.pushClient('No slice selected', 'info');
    return;
  }
  await runSliceMutation('slice-editing', (id) => api.pasteClipboard(id));
}

/** Evenly redistributes slice depths; a no-op (not an error) with zero slices. */
export async function balanceSlices(): Promise<void> {
  await runSliceMutation('slice-editing', (id) => api.balanceSlices(id));
}

/** Sets the depth of the slice currently at `index` (`PUT .../slices/{index}/depth`, sync). */
export async function setSliceDepth(index: number, depth: number): Promise<void> {
  await runSliceMutation('slice-editing', (id) => api.setSliceDepth(id, index, depth));
}

/** Uploads a replacement image for the slice currently at `index` (`PUT .../slices/{index}/image`). */
export async function uploadSliceImage(index: number, file: File): Promise<void> {
  await runSliceMutation('slice-editing', (id) => api.replaceSliceImage(id, index, file));
}

/** Inverts the current mask, creating an all-zero mask first if none exists. */
export async function invertMask(): Promise<void> {
  await runSliceMutation('mask-tools', (id) => api.invertMask(id));
}

/** Feathers (blurs) the current mask by a fixed kernel; requires an existing mask. */
export async function featherMask(): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  if (!view.segmentation.hasMask) {
    logStore.pushClient('No mask to feather', 'info');
    return;
  }
  await runSliceMutation('mask-tools', (id) => api.featherMask(id));
}

/** Toggles the checkerboard vs. grayscale background for the selected-slice preview. */
export async function toggleCheckerboard(): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  await runSliceMutation('mask-tools', (id) => api.setCheckerboard(id, !view.useCheckerboard));
}

/** Undo/redo one step of a slice's saved image-version history. */
export async function undoSlice(index: number): Promise<void> {
  await runSliceMutation('slice-editing', (id) => api.undoSlice(id, index));
}

export async function redoSlice(index: number): Promise<void> {
  await runSliceMutation('slice-editing', (id) => api.redoSlice(id, index));
}

/** Restores a legacy `appstate.json` (Configuration tab's Load State). */
export async function restoreProject(file: File): Promise<void> {
  jobStore.begin('restore');
  try {
    const view = await api.restoreProject(file);
    logStore.reset();
    uiStore.resetProgress();
    projectStore.applyView(view);
    await refreshLogs(view.id);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
  }
}

// --- Canvas masks / inpainting -----------------------------------------
//
// Every function below (other than `saveMask`/`deleteMask`, called directly
// by MaskCanvas.svelte's own save/clear lifecycle) always targets
// `view.selectedSlice`, matching every `InpaintingService` command/API route
// it calls (see ARCHITECTURE.md's "Canvas-mask and inpainting endpoints").
// Generate/apply/erase all flush any pending canvas save first (via
// `canvasSaveStore`), so a just-painted stroke is never silently dropped by
// a mutation that runs before it lands (see state/canvas.svelte.ts).

/**
 * Persists the canvas's current pixels as the selected slice's mask
 * (`PUT .../slices/{index}/mask`, sync). Called by MaskCanvas.svelte on
 * pointerup; its result is what `canvasSaveStore` tracks as the "pending
 * save" other actions must await.
 *
 * `cropToRegion` mirrors Dash's "Crop to region of interest" checkbox
 * (`uiStore.cropToRoi`); the returned bounding box (or `null`) is what
 * MaskCanvas.svelte feeds into `canvasPreviewStore.showRoiBox` for
 * PreviewOverlay.svelte's ~2s ROI-box preview, matching Dash's CLI-07.
 */
export async function saveMask(
  index: number,
  mask: Blob,
  cropToRegion: boolean,
): Promise<import('./api/client').BoundingBox> {
  const view = projectStore.view;
  if (!view) return null;
  try {
    const result = await api.saveInpaintingMask(view.id, index, mask, cropToRegion);
    projectStore.applyView(result);
    return result.boundingBox;
  } catch (err) {
    logStore.pushClient(errorMessage(err));
    return null;
  } finally {
    await refreshLogs(view.id);
  }
}

/** Deletes the selected slice's saved mask (`DELETE .../slices/{index}/mask`, sync). */
export async function deleteMask(index: number): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  try {
    const result = await api.deleteInpaintingMask(view.id, index);
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    await refreshLogs(view.id);
  }
}

/**
 * Persists the positive/negative prompt textareas for the selected slice
 * (`PUT .../slices/{index}/prompts`, sync), mirroring Dash's
 * `update_prompt_text` (WEB-17), which persists on every textarea change
 * regardless of whether Generate is ever clicked.
 */
export async function updateInpaintingPrompts(
  index: number,
  positivePrompt: string,
  negativePrompt: string,
): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  try {
    const result = await api.updateInpaintingPrompts(view.id, index, positivePrompt, negativePrompt);
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    await refreshLogs(view.id);
  }
}

/**
 * Updates one or more inpainting settings (model/strength/guidanceScale/
 * padding/blur) (`PUT /inpainting/settings`, sync). Only fields present on
 * `settings` are sent (see `api.updateInpaintingSettings`); an actual model
 * change drops any stored candidate set server-side, mirroring Dash's
 * `remember_inpaint_model`.
 */
export async function updateInpaintingSettings(settings: InpaintingSettingsRequest): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  jobStore.begin('inpainting-mutate');
  try {
    const result = await api.updateInpaintingSettings(view.id, settings);
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/**
 * Starts a Generate/Fill/Enhance job (`POST .../inpainting/generate`) and
 * polls it to completion. On failure, the previous candidate set (if any) is
 * left exactly as-is - `generate_candidates`'s own contract guarantees the
 * server never replaces it until a full success, and this function never
 * clears `projectStore.view` eagerly, so old candidates simply stay visible.
 */
export async function generateInpainting(
  mode: InpaintingGenerateMode,
  positivePrompt: string,
  negativePrompt: string,
): Promise<void> {
  await canvasSaveStore.flush();
  const view = projectStore.view;
  if (!view || view.selectedSlice === null) return;
  const index = view.selectedSlice as number;

  jobStore.begin('inpainting');
  try {
    const { job } = await api.generateInpaintingCandidates(view.id, index, {
      mode,
      positivePrompt,
      negativePrompt,
    });
    const finished = await api.pollJob(job.id, {
      onProgress: (j) => jobStore.setProgress(j.progress),
    });
    if (finished.project) projectStore.applyView(finished.project);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/**
 * Selects (or, selecting the same index again, toggles off - per
 * `InpaintingService.select_candidate`'s own contract) one candidate from
 * the current generation (`PUT /inpainting/selection`, sync).
 */
export async function selectInpaintingCandidate(generationId: string, candidate: number): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  jobStore.begin('inpainting-mutate');
  try {
    const result = await api.updateInpaintingSelection(view.id, generationId, candidate);
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/**
 * Applies the currently selected candidate to the selected slice
 * (`POST .../inpainting/apply`, sync); requires both a selected slice and a
 * current candidate selection, matching Dash's own `#apply-inpainting-button`
 * enablement (CMP-02).
 */
export async function applyInpaintingCandidate(): Promise<void> {
  await canvasSaveStore.flush();
  const view = projectStore.view;
  if (!view || view.selectedSlice === null) return;
  const candidates = view.inpainting.candidates;
  if (!candidates || view.inpainting.selectedCandidate == null) return;
  const index = view.selectedSlice as number;

  jobStore.begin('inpainting-mutate');
  try {
    const result = await api.applyInpaintingCandidate(view.id, index, candidates.generationId);
    projectStore.applyView(result);
    uiStore.markInpainted();
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/**
 * Erases the selected slice's painted alpha (`POST .../inpainting/erase`,
 * sync). Unlike Apply, any stored candidate set is left alone, mirroring
 * Dash's `erase_inpainting`.
 */
export async function eraseInpainting(): Promise<void> {
  await canvasSaveStore.flush();
  const view = projectStore.view;
  if (!view || view.selectedSlice === null) return;
  const index = view.selectedSlice as number;

  jobStore.begin('inpainting-mutate');
  try {
    const result = await api.eraseInpainting(view.id, index);
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

// --- Project lifecycle / export / configuration -----------------------------
//
// Project/export/configuration settings (PARITY.md's "Project lifecycle",
// "Configuration" and "Export/Render" sections; see ARCHITECTURE.md's
// "Project lifecycle, export/render and configuration endpoints"). Every UI
// control that shows a *persisted* value (camera/displacement sliders, the
// depth-model select, the inpainting-model select, the external-server
// field, dark mode) reads straight from `projectStore.view.settings`/
// `.inpainting` - restoring/loading a project just re-populates that view, so
// no separate "apply restored settings" step is needed beyond the two
// exceptions below that mirror server state into transient `uiStore` fields
// (`theme`, the depth-model select's local draft, kept for the same reason
// ModeTab.svelte already tracks one - see its own comment).

/** POST /api/v1/projects/{id}/save (Configuration tab's "Save State"); no browser download - matches Dash's own save_state exactly (see ARCHITECTURE.md). */
export async function saveProject(): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  jobStore.begin('save');
  try {
    const result = await api.saveProject(view.id);
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/**
 * PUT /api/v1/projects/{id}/settings. Only the fields present on `settings`
 * are applied server-side (see `ProjectSettingsRequest`); callers send
 * whichever subset they own (e.g. just `depthModel`, or `camera` +
 * `meshDisplacement` together - see ExportTab.svelte's comment on why those
 * two are always sent together, mirroring Dash's own single
 * `remember_camera_parameters` callback).
 */
export async function updateSettings(settings: ProjectSettingsRequest): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  jobStore.begin('settings');
  try {
    const result = await api.updateSettings(view.id, settings);
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/**
 * Flips the theme immediately (matching Dash's own immediate class toggle),
 * then persists it (`PUT .../settings`, `darkMode`) when a project is
 * loaded - mirrors `toggle_dark_mode` (WEB-01), which only ever writes to
 * `AppState` "if filename is not None". Without a project, the toggle stays
 * purely local, same as Dash before any image is uploaded.
 */
export async function toggleDarkMode(): Promise<void> {
  const next = uiStore.theme === 'dark' ? 'light' : 'dark';
  uiStore.setTheme(next);
  if (!projectStore.view) return;
  await updateSettings({ darkMode: next === 'dark' });
}

/** Starts a glTF export job (`POST .../export/gltf`) and polls it to completion. */
export async function startGltfExport(dof: boolean): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  jobStore.begin('export-gltf');
  try {
    const { job } = await api.startGltfExport(view.id, dof);
    const finished = await api.pollJob(job.id, {
      onProgress: (j) => jobStore.setProgress(j.progress),
    });
    if (finished.project) projectStore.applyView(finished.project);
    uiStore.markExported();
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/** Starts a texture-upscale job (`POST .../export/upscale`) and polls it to completion. */
export async function startUpscaleExport(): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  jobStore.begin('upscale');
  try {
    const { job } = await api.startUpscaleExport(view.id);
    const finished = await api.pollJob(job.id, {
      onProgress: (j) => jobStore.setProgress(j.progress),
    });
    if (finished.project) projectStore.applyView(finished.project);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/**
 * Starts an animation-render job (`POST .../export/animation`) and polls it
 * to completion. Deliberately never triggers a browser download - matches
 * Dash's `export_animation` exactly (frames are written server-side only;
 * see PARITY.md "Known quirks").
 */
export async function startAnimationExport(frames: number): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  jobStore.begin('animation');
  try {
    const { job } = await api.startAnimationExport(view.id, frames);
    const finished = await api.pollJob(job.id, {
      onProgress: (j) => jobStore.setProgress(j.progress),
    });
    if (finished.project) projectStore.applyView(finished.project);
    uiStore.markExported();
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/** Uploads a ComfyUI workflow JSON file (`PUT .../inpainting/workflow`). */
export async function uploadInpaintingWorkflow(file: File): Promise<void> {
  const view = projectStore.view;
  if (!view) return;
  jobStore.begin('workflow-upload');
  try {
    const result = await api.uploadInpaintingWorkflow(view.id, file);
    projectStore.applyView(result);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
    await refreshLogs(view.id);
  }
}

/**
 * Probes an Automatic1111/ComfyUI server (`POST /api/v1/config/probe-server`,
 * not project-scoped - mirrors `test_external_connection`/CMP-12). Never
 * throws for a failed probe (the route itself never returns non-2xx); the
 * result's `message` is pushed to the log exactly as Dash logs it, since
 * this route has no project to attach a server-side log entry to.
 */
export async function probeExternalServer(model: string, serverAddress: string): Promise<void> {
  jobStore.begin('probe');
  try {
    const result = await api.probeServer(model, serverAddress);
    uiStore.setExternalConnectionStatus(result.ok ? 'success' : 'failure');
    logStore.pushClient(result.message, result.ok ? 'info' : 'error');
  } catch (err) {
    uiStore.setExternalConnectionStatus('failure');
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
  }
}

/**
 * Validates a StabilityAI/fal.ai API key (`POST /api/v1/config/validate-key`,
 * not project-scoped - mirrors `test_api_key`/CMP-15). Same never-throws/
 * client-only-log contract as `probeExternalServer`; `apiKey` is write-only
 * and never sent anywhere else.
 */
export async function probeApiKey(model: string, apiKey: string): Promise<void> {
  jobStore.begin('validate-key');
  try {
    const result = await api.validateApiKey(model, apiKey);
    uiStore.setApiKeyStatus(result.ok ? 'success' : 'failure');
    logStore.pushClient(result.message, result.ok ? 'info' : 'error');
  } catch (err) {
    uiStore.setApiKeyStatus('failure');
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
  }
}
