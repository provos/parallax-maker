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
import { projectStore } from './state/project.svelte';
import { jobStore } from './state/jobs.svelte';
import { logStore } from './state/logs.svelte';

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
    const view = await api.createProject(file);
    projectId = view.id;
    projectStore.applyView(view);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
    jobStore.end();
    return;
  }
  jobStore.end();
  if (projectId) await refreshLogs(projectId);
  await startDepth(depthModel);
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

/** Restores a legacy `appstate.json` (Configuration tab's Load State). */
export async function restoreProject(file: File): Promise<void> {
  jobStore.begin('restore');
  try {
    const view = await api.restoreProject(file);
    logStore.reset();
    projectStore.applyView(view);
    await refreshLogs(view.id);
  } catch (err) {
    logStore.pushClient(errorMessage(err));
  } finally {
    jobStore.end();
  }
}
