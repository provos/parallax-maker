/**
 * Tracks a locally-initiated, in-flight request/job so controls can be
 * disabled immediately -- before the server's `ProjectView.busy` would
 * reflect it (e.g. the window between `createProject` returning and
 * `startDepth` being submitted), and for the duration of synchronous
 * mutations (thresholds, slice-count) that don't have a job id at all.
 *
 * `busy.svelte.ts` combines this with `ProjectView.busy` to answer "should
 * mutating controls be disabled right now?".
 */

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
  | 'validate-key';

function createJobStore() {
  let active = $state<JobKind | null>(null);
  let progress = $state<number>(0);

  return {
    get active(): JobKind | null {
      return active;
    },
    get progress(): number {
      return progress;
    },
    begin(kind: JobKind): void {
      active = kind;
      progress = 0;
    },
    setProgress(value: number): void {
      progress = value;
    },
    end(): void {
      active = null;
      progress = 0;
    },
  };
}

export const jobStore = createJobStore();
export type JobStore = ReturnType<typeof createJobStore>;
