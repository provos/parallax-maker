/**
 * Tracks a locally-initiated, in-flight request/job so controls can be
 * disabled immediately -- before the server's `ProjectView.busy` would
 * reflect it (e.g. the window between `createProject` returning and
 * `startDepth` being submitted), and for the duration of synchronous
 * mutations (thresholds, slice-count) that don't have a job id at all.
 *
 * `busy.svelte.ts` combines this with `ProjectView.busy` to answer "should
 * mutating controls be disabled right now?".
 *
 * For server jobs it also mirrors the polled job (`track`): its id, a detail
 * line such as "Loading the depth model", and whether it can be cancelled
 * right now, which JobCard.svelte shows next to the control that started it.
 */
import * as api from '../api/client';
import type { Job } from '../api/types';
import type { ToastAction } from './toasts.svelte';

/** The last failure, shown inline by the JobCard for its kind until the next job. */
export type JobError = { kind: JobKind; title: string; message: string; actions: ToastAction[] };

export type JobKind =
  | 'upload'
  | 'depth'
  | 'slices'
  | 'thresholds'
  | 'slice-count'
  | 'restore'
  | 'selection'
  | 'segmentation'
  | 'multi-point'
  | 'slice-editing'
  | 'mask-tools'
  | 'inpainting'
  | 'inpainting-mutate'
  | 'save'
  | 'settings'
  | 'export-gltf'
  | 'upscale'
  | 'animation'
  | 'workflow-upload'
  | 'probe'
  | 'validate-key'
  | 'navigate';

const LABELS: Partial<Record<string, string>> = {
  upload: 'Uploading image',
  depth: 'Generating depth map',
  slices: 'Generating slices',
  restore: 'Restoring project',
  segmentation: 'Segmenting',
  'multi-point': 'Segmenting',
  inpainting: 'Generating inpainting candidates',
  'inpainting-mutate': 'Updating slice',
  'slice-editing': 'Updating slices',
  'mask-tools': 'Updating mask',
  save: 'Saving project',
  'export-gltf': 'Exporting glTF scene',
  upscale: 'Upscaling textures',
  animation: 'Rendering animation',
  navigate: 'Rendering view',
  probe: 'Testing connection',
  'validate-key': 'Validating API key',
};

/** Human-readable label for a local job kind or a server-reported busy kind. */
export function jobLabel(kind: JobKind | string): string {
  return LABELS[kind] ?? 'Working';
}

function createJobStore() {
  let active = $state<JobKind | null>(null);
  let progress = $state<number>(0);
  let jobId = $state<string | null>(null);
  let detail = $state<string | null>(null);
  let cancellable = $state(false);
  let cancelling = $state(false);
  let lastError = $state<JobError | null>(null);

  return {
    get active(): JobKind | null {
      return active;
    },
    get progress(): number {
      return progress;
    },
    get jobId(): string | null {
      return jobId;
    },
    get detail(): string | null {
      return detail;
    },
    /** Whether the server can stop the running job now (and no cancel is pending). */
    get cancellable(): boolean {
      return cancellable && !cancelling;
    },
    get cancelling(): boolean {
      return cancelling;
    },
    get lastError(): JobError | null {
      return lastError;
    },
    fail(error: JobError): void {
      lastError = error;
    },
    clearError(): void {
      lastError = null;
    },
    begin(kind: JobKind): void {
      lastError = null;
      active = kind;
      progress = 0;
      jobId = null;
      detail = null;
      cancellable = false;
      cancelling = false;
    },
    setProgress(value: number): void {
      progress = value;
    },
    /** Mirrors a polled server job (progress, detail, cancellable). */
    track(job: Job): void {
      if (active === null) return;
      jobId = job.id;
      progress = job.progress;
      detail = job.detail ?? null;
      cancellable = job.cancellable ?? false;
    },
    /** Asks the server to stop the tracked job; polling then sees it cancelled. */
    async cancel(): Promise<void> {
      if (!jobId || !cancellable || cancelling) return;
      cancelling = true;
      try {
        await api.cancelJob(jobId);
      } catch {
        // Too late (it finished or can no longer stop): let polling settle it.
        cancelling = false;
      }
    },
    end(): void {
      active = null;
      progress = 0;
      jobId = null;
      detail = null;
      cancellable = false;
      cancelling = false;
    },
  };
}

export const jobStore = createJobStore();
export type JobStore = ReturnType<typeof createJobStore>;
