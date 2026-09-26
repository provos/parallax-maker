import { afterEach, describe, expect, it, vi } from 'vitest';
import { jobStore, jobLabel } from './jobs.svelte';
import * as api from '../api/client';
import type { Job } from '../api/types';

function makeJob(overrides: Partial<Job> = {}): Job {
  return { id: 'job-1', kind: 'depth', status: 'running', progress: 0.5, ...overrides };
}

describe('jobLabel', () => {
  it('returns the human-readable label for a known kind', () => {
    expect(jobLabel('depth')).toBe('Generating depth map');
    expect(jobLabel('inpainting')).toBe('Generating inpainting candidates');
  });

  it('falls back to "Working" for an unknown kind', () => {
    expect(jobLabel('something-unrecognized')).toBe('Working');
  });
});

describe('jobStore', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    jobStore.end();
    jobStore.clearError();
  });

  it('begin sets the active kind, resets progress/job fields and clears lastError', () => {
    jobStore.fail({ kind: 'depth', title: 'Depth map failed', message: 'boom', actions: [] });
    jobStore.begin('depth');

    expect(jobStore.active).toBe('depth');
    expect(jobStore.progress).toBe(0);
    expect(jobStore.jobId).toBeNull();
    expect(jobStore.detail).toBeNull();
    expect(jobStore.cancellable).toBe(false);
    expect(jobStore.cancelling).toBe(false);
    expect(jobStore.lastError).toBeNull();
  });

  it('end clears job fields but keeps lastError', () => {
    jobStore.begin('depth');
    jobStore.track(makeJob({ jobId: 'job-1' } as Partial<Job>));
    jobStore.fail({ kind: 'depth', title: 'Depth map failed', message: 'boom', actions: [] });

    jobStore.end();

    expect(jobStore.active).toBeNull();
    expect(jobStore.progress).toBe(0);
    expect(jobStore.jobId).toBeNull();
    expect(jobStore.detail).toBeNull();
    expect(jobStore.cancellable).toBe(false);
    expect(jobStore.cancelling).toBe(false);
    expect(jobStore.lastError).toMatchObject({ kind: 'depth', message: 'boom' });
  });

  it('track no-ops when no job is active', () => {
    jobStore.track(makeJob());
    expect(jobStore.jobId).toBeNull();
    expect(jobStore.progress).toBe(0);
  });

  it('track mirrors id, progress, detail and cancellable from the polled job', () => {
    jobStore.begin('depth');
    jobStore.track(makeJob({ id: 'job-9', progress: 0.3, detail: 'Loading the depth model', cancellable: true }));

    expect(jobStore.jobId).toBe('job-9');
    expect(jobStore.progress).toBe(0.3);
    expect(jobStore.detail).toBe('Loading the depth model');
    expect(jobStore.cancellable).toBe(true);
  });

  it('track treats a missing detail/cancellable as null/false', () => {
    jobStore.begin('depth');
    jobStore.track(makeJob({ detail: undefined, cancellable: undefined }));
    expect(jobStore.detail).toBeNull();
    expect(jobStore.cancellable).toBe(false);
  });

  it('cancellable getter is false while a cancel is pending even if the job reports cancellable', () => {
    jobStore.begin('depth');
    jobStore.track(makeJob({ cancellable: true }));
    expect(jobStore.cancellable).toBe(true);

    const cancelJobSpy = vi.spyOn(api, 'cancelJob').mockReturnValue(new Promise(() => {}));
    void jobStore.cancel();
    expect(jobStore.cancelling).toBe(true);
    expect(jobStore.cancellable).toBe(false);
    expect(cancelJobSpy).toHaveBeenCalledWith('job-1');
  });

  it('cancel is a no-op without a jobId, without cancellable, or while already cancelling', async () => {
    const cancelJobSpy = vi.spyOn(api, 'cancelJob').mockResolvedValue(makeJob({ status: 'cancelled' }));

    // No job at all.
    await jobStore.cancel();
    expect(cancelJobSpy).not.toHaveBeenCalled();

    // Tracked but not cancellable.
    jobStore.begin('depth');
    jobStore.track(makeJob({ cancellable: false }));
    await jobStore.cancel();
    expect(cancelJobSpy).not.toHaveBeenCalled();
  });

  it('cancel calls api.cancelJob(jobId) when jobId is set and cancellable', async () => {
    const cancelJobSpy = vi.spyOn(api, 'cancelJob').mockResolvedValue(makeJob({ status: 'cancelled' }));
    jobStore.begin('depth');
    jobStore.track(makeJob({ id: 'job-42', cancellable: true }));

    await jobStore.cancel();

    expect(cancelJobSpy).toHaveBeenCalledWith('job-42');
  });

  it('a rejected cancel request resets cancelling back to false', async () => {
    vi.spyOn(api, 'cancelJob').mockRejectedValue(new Error('too late'));
    jobStore.begin('depth');
    jobStore.track(makeJob({ cancellable: true }));

    await jobStore.cancel();

    expect(jobStore.cancelling).toBe(false);
  });

  it('fail/clearError set and clear lastError', () => {
    jobStore.fail({ kind: 'inpainting', title: 'Generation failed', message: 'boom', actions: [] });
    expect(jobStore.lastError).toMatchObject({ kind: 'inpainting', message: 'boom' });
    jobStore.clearError();
    expect(jobStore.lastError).toBeNull();
  });
});
