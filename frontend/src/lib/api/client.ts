import type {
  ApiErrorBody,
  HealthResponse,
  Job,
  LogsPage,
  ProjectView,
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
 * `ApiError` (or the job's own error, wrapped) if it fails, or with the
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
          reject(new ApiError(200, 'provider_error', job.error ?? 'Job failed'));
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
