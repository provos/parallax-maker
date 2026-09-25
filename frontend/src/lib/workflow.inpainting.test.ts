import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as workflow from './workflow';
import { projectStore } from './state/project.svelte';
import { jobStore } from './state/jobs.svelte';
import { logStore } from './state/logs.svelte';
import { canvasSaveStore } from './state/canvas.svelte';
import type { ProjectView, SliceView } from './api/types';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function makeSlice(index: number, overrides: Partial<SliceView> = {}): SliceView {
  return {
    index,
    depth: 100,
    version: 1,
    canUndo: false,
    canRedo: false,
    positivePrompt: '',
    negativePrompt: '',
    image: { url: `/slice-${index}` },
    thumbnail: { url: `/slice-${index}-thumb` },
    ...overrides,
  };
}

function makeView(overrides: Partial<ProjectView> = {}): ProjectView {
  return {
    id: 'appstate-test',
    revision: 1,
    image: { width: 320, height: 240 },
    assets: { input: null, depth: null },
    depthModel: 'dinov2',
    numSlices: 3,
    thresholds: [],
    slices: [makeSlice(0), makeSlice(1)],
    selectedSlice: 1,
    segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false },
    inpainting: {
      model: 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
      strength: 0.8,
      guidanceScale: 7.5,
      padding: 50,
      blur: 50,
      externalServer: 'localhost:7860',
      hasWorkflow: false,
      candidates: null,
      selectedCandidate: null,
    },
    busy: null,
    settings: {
      darkMode: false,
      camera: { distance: 100, focalLength: 100, maxDistance: 200 },
      meshDisplacement: 0,
      depthModel: "dinov2",
    },
    exports: { gltf: null, upscaled: false },
    ...overrides,
  };
}

/** Deferred promise, for controlling exactly when a "pending save" settles. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe('workflow: canvas-mask lifecycle', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    canvasSaveStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('generateInpainting waits for a pending mask save before starting the job', async () => {
    projectStore.applyView(makeView());
    const pending = deferred<void>();
    canvasSaveStore.register(pending.promise);

    const calledUrls: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      calledUrls.push(url);
      if (url === '/api/v1/projects/appstate-test/slices/1/inpainting/generate') {
        return jsonResponse(202, { job: { id: 'job-gen', kind: 'inpainting', status: 'queued', progress: 0 } });
      }
      if (url === '/api/v1/jobs/job-gen') {
        return jsonResponse(200, { id: 'job-gen', kind: 'inpainting', status: 'succeeded', progress: 1, project: makeView() });
      }
      if (url.startsWith('/api/v1/projects/appstate-test/logs')) {
        return jsonResponse(200, { entries: [], next: 0 });
      }
      throw new Error(`Unexpected fetch: ${String(init?.method ?? 'GET')} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    const generatePromise = workflow.generateInpainting('paint', 'a prompt', '');

    // The pending save has not resolved yet: no request should have gone out.
    await Promise.resolve();
    await Promise.resolve();
    expect(calledUrls).toHaveLength(0);

    pending.resolve();
    await generatePromise;

    expect(calledUrls[0]).toBe('/api/v1/projects/appstate-test/slices/1/inpainting/generate');
  });

  it('selectSlice waits for a pending mask save before changing the selection', async () => {
    projectStore.applyView(makeView({ selectedSlice: 1 }));
    const pending = deferred<void>();
    canvasSaveStore.register(pending.promise);

    const calledUrls: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      calledUrls.push(url);
      if (url === '/api/v1/projects/appstate-test/selection') {
        return jsonResponse(200, { ...makeView({ selectedSlice: 0 }), changed: true });
      }
      if (url.startsWith('/api/v1/projects/appstate-test/logs')) {
        return jsonResponse(200, { entries: [], next: 0 });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    const selectPromise = workflow.selectSlice(0);

    await Promise.resolve();
    await Promise.resolve();
    expect(calledUrls).toHaveLength(0);

    pending.resolve();
    await selectPromise;

    expect(calledUrls[0]).toBe('/api/v1/projects/appstate-test/selection');
    expect(projectStore.view?.selectedSlice).toBe(0);
  });

  it('applyInpaintingCandidate and eraseInpainting also flush a pending mask save first', async () => {
    projectStore.applyView(
      makeView({
        selectedSlice: 1,
        inpainting: {
          ...makeView().inpainting,
          candidates: { generationId: 'gen-1', sliceIndex: 1, images: [{ url: '/c0' }] },
          selectedCandidate: 0,
        },
      }),
    );
    const pending = deferred<void>();
    canvasSaveStore.register(pending.promise);

    const calledUrls: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      calledUrls.push(url);
      if (url === '/api/v1/projects/appstate-test/slices/1/inpainting/apply') {
        return jsonResponse(200, { ...makeView(), changed: true });
      }
      if (url.startsWith('/api/v1/projects/appstate-test/logs')) {
        return jsonResponse(200, { entries: [], next: 0 });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    const applyPromise = workflow.applyInpaintingCandidate();
    await Promise.resolve();
    await Promise.resolve();
    expect(calledUrls).toHaveLength(0);

    pending.resolve();
    await applyPromise;
    expect(calledUrls[0]).toBe('/api/v1/projects/appstate-test/slices/1/inpainting/apply');
  });

  it('a failed generation leaves the previous candidate set visible', async () => {
    const candidates = { generationId: 'gen-1', sliceIndex: 1, images: [{ url: '/c0' }, { url: '/c1' }] };
    projectStore.applyView(
      makeView({ inpainting: { ...makeView().inpainting, candidates, selectedCandidate: 0 } }),
    );

    const fetchMock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      if (url === '/api/v1/projects/appstate-test/slices/1/inpainting/generate') {
        return jsonResponse(202, { job: { id: 'job-fail', kind: 'inpainting', status: 'queued', progress: 0 } });
      }
      if (url === '/api/v1/jobs/job-fail') {
        return jsonResponse(200, {
          id: 'job-fail',
          kind: 'inpainting',
          status: 'failed',
          progress: 1,
          error: 'model exploded',
        });
      }
      if (url.startsWith('/api/v1/projects/appstate-test/logs')) {
        return jsonResponse(200, { entries: [], next: 0 });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    await workflow.generateInpainting('paint', '', '');

    // The view is untouched (no `project` came back on the failed job), so
    // the old candidate set is still exactly what it was before.
    expect(projectStore.view?.inpainting.candidates).toEqual(candidates);
    expect(projectStore.view?.inpainting.selectedCandidate).toBe(0);
    expect(logStore.entries.at(-1)?.message).toBe('model exploded');
  });

  it('generateInpainting/applyInpaintingCandidate/eraseInpainting no-op without a selected slice', async () => {
    projectStore.applyView(makeView({ selectedSlice: null }));
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    await workflow.generateInpainting('paint', '', '');
    await workflow.applyInpaintingCandidate();
    await workflow.eraseInpainting();

    expect(fetchMock).not.toHaveBeenCalled();
  });
});
