import type {
  ApiErrorBody,
  HealthResponse,
  InpaintingGenerateMode,
  InpaintingSettingsRequest,
  Job,
  LogsPage,
  ProbeResultView,
  ProjectSettingsRequest,
  ProjectView,
  SegmentationMode,
} from './types';

const API_BASE = '/api/v1';

/**
 * Error raised for any non-2xx API response. `status` is the HTTP status
 * code; `code` and `message` are parsed from the documented error body
 * `{"error": {"code": str, "message": str}}` (see
 * docs/svelte-migration/ARCHITECTURE.md, "HTTP contract"). If the body
 * cannot be parsed as that shape, `code` falls back to `"unknown"` and
 * `message` falls back to the response's status text.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

/** A background job reached status `failed`; carries the terminal job record. */
export class JobFailedError extends Error {
  readonly job: Job;

  constructor(job: Job) {
    super(job.error ?? 'Job failed');
    this.name = 'JobFailedError';
    this.job = job;
  }
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== 'object' || value === null || !('error' in value)) {
    return false;
  }
  const err = (value as { error?: unknown }).error;
  return (
    typeof err === 'object' &&
    err !== null &&
    typeof (err as { code?: unknown }).code === 'string' &&
    typeof (err as { message?: unknown }).message === 'string'
  );
}

async function parseErrorBody(response: Response): Promise<ApiError> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = undefined;
  }
  if (isApiErrorBody(body)) {
    return new ApiError(response.status, body.error.code, body.error.message);
  }
  return new ApiError(response.status, 'unknown', response.statusText || 'Request failed');
}

type RequestOptions = {
  method?: string;
  body?: BodyInit;
  headers?: Record<string, string>;
  signal?: AbortSignal;
};

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: options.method ?? 'GET',
    body: options.body,
    headers: options.headers,
    signal: options.signal,
  });

  if (!response.ok) {
    throw await parseErrorBody(response);
  }

  // Some endpoints (e.g. asset bytes) are not JSON; callers that need that
  // use the asset URL directly rather than this helper.
  return (await response.json()) as T;
}

function requestJson<T>(
  path: string,
  method: string,
  payload: unknown,
  signal?: AbortSignal,
): Promise<T> {
  return request<T>(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
}

/** GET /api/v1/health */
export function health(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/health', { signal });
}

/** POST /api/v1/projects (multipart `image`) */
export function createProject(file: File, signal?: AbortSignal): Promise<ProjectView> {
  const form = new FormData();
  form.append('image', file);
  return request<ProjectView>('/projects', { method: 'POST', body: form, signal });
}

/** POST /api/v1/projects/restore (multipart `state`, a legacy appstate.json) */
export function restoreProject(file: File, signal?: AbortSignal): Promise<ProjectView> {
  const form = new FormData();
  form.append('state', file);
  return request<ProjectView>('/projects/restore', { method: 'POST', body: form, signal });
}

/** GET /api/v1/projects/{id} */
export function getProject(id: string, signal?: AbortSignal): Promise<ProjectView> {
  return request<ProjectView>(`/projects/${encodeURIComponent(id)}`, { signal });
}

/** POST /api/v1/projects/{id}/depth -> 202 {job} */
export function startDepth(
  id: string,
  model: string,
  signal?: AbortSignal,
): Promise<{ job: Job }> {
  return requestJson<{ job: Job }>(
    `/projects/${encodeURIComponent(id)}/depth`,
    'POST',
    { model },
    signal,
  );
}

/** PUT /api/v1/projects/{id}/slice-count */
export function setSliceCount(
  id: string,
  numSlices: number,
  signal?: AbortSignal,
): Promise<ProjectView> {
  return requestJson<ProjectView>(
    `/projects/${encodeURIComponent(id)}/slice-count`,
    'PUT',
    { numSlices },
    signal,
  );
}

/** PUT /api/v1/projects/{id}/thresholds */
export function setThresholds(
  id: string,
  values: number[],
  baseRevision: number,
  signal?: AbortSignal,
): Promise<ProjectView> {
  return requestJson<ProjectView>(
    `/projects/${encodeURIComponent(id)}/thresholds`,
    'PUT',
    { values, baseRevision },
    signal,
  );
}

/** POST /api/v1/projects/{id}/slices -> 202 {job} */
export function startSlices(id: string, signal?: AbortSignal): Promise<{ job: Job }> {
  return requestJson<{ job: Job }>(
    `/projects/${encodeURIComponent(id)}/slices`,
    'POST',
    {},
    signal,
  );
}

/** GET /api/v1/jobs/{jobId} */
export function getJob(jobId: string, signal?: AbortSignal): Promise<Job> {
  return request<Job>(`/jobs/${encodeURIComponent(jobId)}`, { signal });
}

/** GET /api/v1/projects/{id}/logs?after={seq} */
export function getLogs(
  id: string,
  after: number = 0,
  signal?: AbortSignal,
): Promise<LogsPage> {
  return request<LogsPage>(
    `/projects/${encodeURIComponent(id)}/logs?after=${encodeURIComponent(String(after))}`,
    { signal },
  );
}

/** PUT /api/v1/projects/{id}/selection; `slice: null` deselects. */
export function updateSelection(
  id: string,
  slice: number | null,
  signal?: AbortSignal,
): Promise<ProjectView & { changed: boolean }> {
  return requestJson<ProjectView & { changed: boolean }>(
    `/projects/${encodeURIComponent(id)}/selection`,
    'PUT',
    { slice },
    signal,
  );
}

export type SegmentationClickBody = {
  x: number;
  y: number;
  mode: SegmentationMode;
  shiftKey: boolean;
  ctrlKey: boolean;
};

/** POST /api/v1/projects/{id}/segmentation/click -> 202 {job} */
export function segmentationClick(
  id: string,
  body: SegmentationClickBody,
  signal?: AbortSignal,
): Promise<{ job: Job }> {
  return requestJson<{ job: Job }>(
    `/projects/${encodeURIComponent(id)}/segmentation/click`,
    'POST',
    body,
    signal,
  );
}

/** POST /api/v1/projects/{id}/segmentation/commit -> 202 {job} */
export function segmentationCommit(id: string, signal?: AbortSignal): Promise<{ job: Job }> {
  return requestJson<{ job: Job }>(
    `/projects/${encodeURIComponent(id)}/segmentation/commit`,
    'POST',
    {},
    signal,
  );
}

/** PUT /api/v1/projects/{id}/segmentation/multi-point */
export function setMultiPointMode(
  id: string,
  enabled: boolean,
  signal?: AbortSignal,
): Promise<ProjectView & { changed: boolean }> {
  return requestJson<ProjectView & { changed: boolean }>(
    `/projects/${encodeURIComponent(id)}/segmentation/multi-point`,
    'PUT',
    { enabled },
    signal,
  );
}

// --- Slice editing / mask tools --------------------------------------------
//
// Every route below runs synchronously (no job id) and returns the updated
// `ProjectView` plus a `changed` flag, per the architecture doc's "Slice
// editing and mask-tool endpoints" table. None of these send a JSON body
// except `setSliceDepth`/`setCheckerboard` (the backend routes never call
// `_parse_json_body` for the others), so the rest issue a bare POST/DELETE.

type MutationResult = ProjectView & { changed: boolean };

/** POST /api/v1/projects/{id}/slices/create */
export function createSlice(id: string, signal?: AbortSignal): Promise<MutationResult> {
  return request<MutationResult>(`/projects/${encodeURIComponent(id)}/slices/create`, {
    method: 'POST',
    signal,
  });
}

/** DELETE /api/v1/projects/{id}/slices/{index} */
export function deleteSlice(
  id: string,
  index: number,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return request<MutationResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}`,
    { method: 'DELETE', signal },
  );
}

/** POST /api/v1/projects/{id}/slices/{index}/add-mask */
export function addMaskToSlice(
  id: string,
  index: number,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return request<MutationResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/add-mask`,
    { method: 'POST', signal },
  );
}

/** POST /api/v1/projects/{id}/slices/{index}/remove-mask */
export function removeMaskFromSlice(
  id: string,
  index: number,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return request<MutationResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/remove-mask`,
    { method: 'POST', signal },
  );
}

/** POST /api/v1/projects/{id}/clipboard/copy */
export function copyToClipboard(id: string, signal?: AbortSignal): Promise<MutationResult> {
  return request<MutationResult>(`/projects/${encodeURIComponent(id)}/clipboard/copy`, {
    method: 'POST',
    signal,
  });
}

/** POST /api/v1/projects/{id}/clipboard/paste */
export function pasteClipboard(id: string, signal?: AbortSignal): Promise<MutationResult> {
  return request<MutationResult>(`/projects/${encodeURIComponent(id)}/clipboard/paste`, {
    method: 'POST',
    signal,
  });
}

/** POST /api/v1/projects/{id}/slices/balance */
export function balanceSlices(id: string, signal?: AbortSignal): Promise<MutationResult> {
  return request<MutationResult>(`/projects/${encodeURIComponent(id)}/slices/balance`, {
    method: 'POST',
    signal,
  });
}

/** PUT /api/v1/projects/{id}/slices/{index}/depth */
export function setSliceDepth(
  id: string,
  index: number,
  depth: number,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return requestJson<MutationResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/depth`,
    'PUT',
    { depth },
    signal,
  );
}

/** PUT /api/v1/projects/{id}/slices/{index}/image (multipart `image`) */
export function replaceSliceImage(
  id: string,
  index: number,
  file: File | Blob,
  signal?: AbortSignal,
): Promise<MutationResult> {
  const form = new FormData();
  form.append('image', file);
  return request<MutationResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/image`,
    { method: 'PUT', body: form, signal },
  );
}

/** POST /api/v1/projects/{id}/mask/invert */
export function invertMask(id: string, signal?: AbortSignal): Promise<MutationResult> {
  return request<MutationResult>(`/projects/${encodeURIComponent(id)}/mask/invert`, {
    method: 'POST',
    signal,
  });
}

/** POST /api/v1/projects/{id}/mask/feather */
export function featherMask(id: string, signal?: AbortSignal): Promise<MutationResult> {
  return request<MutationResult>(`/projects/${encodeURIComponent(id)}/mask/feather`, {
    method: 'POST',
    signal,
  });
}

/** PUT /api/v1/projects/{id}/display */
export function setCheckerboard(
  id: string,
  useCheckerboard: boolean,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return requestJson<MutationResult>(
    `/projects/${encodeURIComponent(id)}/display`,
    'PUT',
    { useCheckerboard },
    signal,
  );
}

/** POST /api/v1/projects/{id}/slices/{index}/undo */
export function undoSlice(
  id: string,
  index: number,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return request<MutationResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/undo`,
    { method: 'POST', signal },
  );
}

/** POST /api/v1/projects/{id}/slices/{index}/redo */
export function redoSlice(
  id: string,
  index: number,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return request<MutationResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/redo`,
    { method: 'POST', signal },
  );
}

// --- Canvas masks / inpainting ----------------------------------------------
//
// Every route below always acts on `state.selected_slice`; `index` in the
// path is validated against the current selection server-side (a mismatch is
// `409 not_ready`), per the architecture doc's "Canvas-mask and inpainting
// endpoints" table.

/**
 * A `[x0, y0, x1, y1]` bounding box in source-image pixels, or `null` when
 * none was requested/computed (see `boundingBox` below).
 */
export type BoundingBox = [number, number, number, number] | null;

/**
 * `saveInpaintingMask`'s response: the usual `MutationResult`, plus an
 * optional `boundingBox` -- a small, additive field the generated
 * `ProjectView` schema does not carry (it is not part of persisted project
 * state, only this one response), matching Dash's CLI-07/CMP-24 ROI-preview
 * box. See PreviewOverlay.svelte and docs/svelte-migration/ARCHITECTURE.md.
 */
export type MaskSaveResult = MutationResult & { boundingBox: BoundingBox };

/**
 * PUT /api/v1/projects/{id}/slices/{index}/mask (multipart `mask`, the
 * canvas PNG, plus `cropToRegion` -- mirrors Dash's
 * `CHECKLIST_REGION_OF_INTEREST`/CMP-24's `show_crop_region`: when true, the
 * response's `boundingBox` is the mask's own (padded, squared) bounding box;
 * when false, `boundingBox` is `null` and no bounding-box computation runs
 * at all, matching Dash exactly).
 */
export function saveInpaintingMask(
  id: string,
  index: number,
  mask: Blob,
  cropToRegion: boolean,
  signal?: AbortSignal,
): Promise<MaskSaveResult> {
  const form = new FormData();
  form.append('mask', mask, 'mask.png');
  form.append('cropToRegion', cropToRegion ? 'true' : 'false');
  return request<MaskSaveResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/mask`,
    { method: 'PUT', body: form, signal },
  );
}

/** DELETE /api/v1/projects/{id}/slices/{index}/mask */
export function deleteInpaintingMask(
  id: string,
  index: number,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return request<MutationResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/mask`,
    { method: 'DELETE', signal },
  );
}

/** PUT /api/v1/projects/{id}/slices/{index}/prompts */
export function updateInpaintingPrompts(
  id: string,
  index: number,
  positivePrompt: string,
  negativePrompt: string,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return requestJson<MutationResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/prompts`,
    'PUT',
    { positivePrompt, negativePrompt },
    signal,
  );
}

/**
 * PUT /api/v1/projects/{id}/inpainting/settings. Only fields present on
 * `settings` are sent, so a caller updating just one field (e.g. `strength`)
 * doesn't clobber the others - the backend only applies fields actually
 * present in the JSON body (see `InpaintingSettingsRequest`).
 */
export function updateInpaintingSettings(
  id: string,
  settings: InpaintingSettingsRequest,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return requestJson<MutationResult>(
    `/projects/${encodeURIComponent(id)}/inpainting/settings`,
    'PUT',
    settings,
    signal,
  );
}

/** PUT /api/v1/projects/{id}/inpainting/workflow (multipart `workflow`, a ComfyUI JSON file) */
export function uploadInpaintingWorkflow(
  id: string,
  workflow: File | Blob,
  signal?: AbortSignal,
): Promise<MutationResult> {
  const form = new FormData();
  form.append('workflow', workflow);
  return request<MutationResult>(`/projects/${encodeURIComponent(id)}/inpainting/workflow`, {
    method: 'PUT',
    body: form,
    signal,
  });
}

export type InpaintingGenerateBody = {
  mode: InpaintingGenerateMode;
  positivePrompt: string;
  negativePrompt: string;
};

/** POST /api/v1/projects/{id}/slices/{index}/inpainting/generate -> 202 {job} */
export function generateInpaintingCandidates(
  id: string,
  index: number,
  body: InpaintingGenerateBody,
  signal?: AbortSignal,
): Promise<{ job: Job }> {
  return requestJson<{ job: Job }>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/inpainting/generate`,
    'POST',
    body,
    signal,
  );
}

/**
 * PUT /api/v1/projects/{id}/inpainting/selection. `candidate: null` clears
 * the selection; otherwise `InpaintingService.select_candidate`'s own
 * contract toggles the same index off again, so callers always pass the
 * clicked index (never compute the toggle client-side).
 */
export function updateInpaintingSelection(
  id: string,
  generationId: string,
  candidate: number | null,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return requestJson<MutationResult>(
    `/projects/${encodeURIComponent(id)}/inpainting/selection`,
    'PUT',
    { generationId, candidate },
    signal,
  );
}

/** POST /api/v1/projects/{id}/slices/{index}/inpainting/apply */
export function applyInpaintingCandidate(
  id: string,
  index: number,
  generationId: string,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return requestJson<MutationResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/inpainting/apply`,
    'POST',
    { generationId },
    signal,
  );
}

/** POST /api/v1/projects/{id}/slices/{index}/inpainting/erase */
export function eraseInpainting(
  id: string,
  index: number,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return request<MutationResult>(
    `/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/inpainting/erase`,
    { method: 'POST', signal },
  );
}

// --- Project lifecycle / export / configuration -----------------------------
//
// Project lifecycle (save/settings) and configuration-probe routes run
// synchronously; export/upscale/animation routes are background jobs like
// depth/slices/segmentation/inpainting-generate above (see ARCHITECTURE.md's
// "Project lifecycle, export/render and configuration endpoints").

/** POST /api/v1/projects/{id}/save */
export function saveProject(id: string, signal?: AbortSignal): Promise<ProjectView> {
  return request<ProjectView>(`/projects/${encodeURIComponent(id)}/save`, {
    method: 'POST',
    signal,
  });
}

/**
 * GET /api/v1/projects/{id}/state-file: the exact JSON payload
 * `POST /projects/restore` accepts, reflecting unsaved mutations too. Used
 * by e2e's oracle-based round trips; not wired to a UI download (Dash's own
 * Save State never downloads anything either - see ARCHITECTURE.md).
 */
export function getStateFileUrl(id: string): string {
  return `${API_BASE}/projects/${encodeURIComponent(id)}/state-file`;
}

/**
 * PUT /api/v1/projects/{id}/settings. Only fields present on `settings` are
 * applied (and only when different from the project's current value); see
 * `ProjectSettingsRequest`.
 */
export function updateSettings(
  id: string,
  settings: ProjectSettingsRequest,
  signal?: AbortSignal,
): Promise<MutationResult> {
  return requestJson<MutationResult>(
    `/projects/${encodeURIComponent(id)}/settings`,
    'PUT',
    settings,
    signal,
  );
}

/** POST /api/v1/projects/{id}/export/gltf -> 202 {job} */
export function startGltfExport(
  id: string,
  dof: boolean,
  signal?: AbortSignal,
): Promise<{ job: Job }> {
  return requestJson<{ job: Job }>(
    `/projects/${encodeURIComponent(id)}/export/gltf`,
    'POST',
    { dof },
    signal,
  );
}

/**
 * GET /api/v1/projects/{id}/export/gltf: the most recent export's `.gltf`
 * file, `Content-Disposition: attachment; filename="scene.gltf"`. Returns the
 * URL itself (not fetched here) so callers can drive a real browser download
 * via an `<a download>` click.
 */
export function getGltfDownloadUrl(id: string): string {
  return `${API_BASE}/projects/${encodeURIComponent(id)}/export/gltf`;
}

/** POST /api/v1/projects/{id}/export/upscale -> 202 {job} */
export function startUpscaleExport(id: string, signal?: AbortSignal): Promise<{ job: Job }> {
  return requestJson<{ job: Job }>(
    `/projects/${encodeURIComponent(id)}/export/upscale`,
    'POST',
    {},
    signal,
  );
}

/** POST /api/v1/projects/{id}/export/animation -> 202 {job} */
export function startAnimationExport(
  id: string,
  frames: number,
  signal?: AbortSignal,
): Promise<{ job: Job }> {
  return requestJson<{ job: Job }>(
    `/projects/${encodeURIComponent(id)}/export/animation`,
    'POST',
    { frames },
    signal,
  );
}

/**
 * GET /api/v1/projects/{id}/slices/{index}/download: the raw slice PNG file
 * itself. Returns the URL (not fetched) so callers can drive a real browser
 * download via an `<a download>` click, like `getGltfDownloadUrl`.
 */
export function getSliceDownloadUrl(id: string, index: number): string {
  return `${API_BASE}/projects/${encodeURIComponent(id)}/slices/${encodeURIComponent(String(index))}/download`;
}

/** POST /api/v1/config/probe-server (not project-scoped; never 5xx). */
export function probeServer(
  model: string,
  serverAddress: string,
  signal?: AbortSignal,
): Promise<ProbeResultView> {
  return requestJson<ProbeResultView>('/config/probe-server', 'POST', { model, serverAddress }, signal);
}

/** POST /api/v1/config/validate-key (not project-scoped; never 5xx; apiKey is write-only). */
export function validateApiKey(
  model: string,
  apiKey: string,
  signal?: AbortSignal,
): Promise<ProbeResultView> {
  return requestJson<ProbeResultView>('/config/validate-key', 'POST', { model, apiKey }, signal);
}

export type PollJobOptions = {
  /** Polling interval in milliseconds. Defaults to 250ms per the architecture doc. */
  intervalMs?: number;
  signal?: AbortSignal;
  onProgress?: (job: Job) => void;
};

/**
 * Poll `GET /api/v1/jobs/{jobId}` until the job reaches a terminal state
 * (`succeeded` or `failed`), per the "Concurrency" section of the
 * architecture doc. Resolves with the terminal job. Rejects with an
 * `JobFailedError` if the job fails, an `ApiError` if polling itself fails, or with the
 * abort reason if `signal` is aborted.
 */
export function pollJob(jobId: string, options: PollJobOptions = {}): Promise<Job> {
  const intervalMs = options.intervalMs ?? 250;
  const { signal, onProgress } = options;

  return new Promise<Job>((resolve, reject) => {
    let cancelled = false;

    const onAbort = () => {
      cancelled = true;
      reject(signal?.reason ?? new DOMException('Aborted', 'AbortError'));
    };
    signal?.addEventListener('abort', onAbort);

    const cleanup = () => {
      signal?.removeEventListener('abort', onAbort);
    };

    const tick = async () => {
      if (cancelled) return;
      try {
        const job = await getJob(jobId, signal);
        if (cancelled) return;
        onProgress?.(job);

        if (job.status === 'succeeded') {
          cleanup();
          resolve(job);
          return;
        }
        if (job.status === 'failed') {
          cleanup();
          reject(new JobFailedError(job));
          return;
        }
        setTimeout(tick, intervalMs);
      } catch (err) {
        if (cancelled) return;
        cleanup();
        reject(err);
      }
    };

    void tick();
  });
}
