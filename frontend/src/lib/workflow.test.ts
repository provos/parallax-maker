import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as workflow from './workflow';
import { projectStore } from './state/project.svelte';
import { jobStore } from './state/jobs.svelte';
import { logStore } from './state/logs.svelte';
import type { ProjectView } from './api/types';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
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
    slices: [],
    selectedSlice: null,
    busy: null,
    ...overrides,
  };
}

describe('workflow', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('uploadImage creates a project and starts a depth job with the given model', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      const method = init?.method ?? 'GET';

      if (url === '/api/v1/projects' && method === 'POST') {
        return jsonResponse(201, makeView());
      }
      if (url.startsWith('/api/v1/projects/appstate-test/logs')) {
        return jsonResponse(200, { entries: [], next: 0 });
      }
      if (url === '/api/v1/projects/appstate-test/depth' && method === 'POST') {
        expect(JSON.parse(init!.body as string)).toEqual({ model: 'midas' });
        return jsonResponse(202, { job: { id: 'job-1', kind: 'depth', status: 'queued', progress: 0 } });
      }
      if (url === '/api/v1/jobs/job-1') {
        return jsonResponse(200, {
          id: 'job-1',
          kind: 'depth',
          status: 'succeeded',
          progress: 1,
          project: makeView({
            assets: { input: { url: '/input' }, depth: { url: '/depth' } },
            thresholds: [0, 85, 170, 255],
          }),
        });
      }
      throw new Error(`Unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    const file = new File(['bytes'], 'input.png', { type: 'image/png' });
    await workflow.uploadImage(file, 'midas');

    expect(jobStore.active).toBeNull();
    expect(projectStore.view?.assets.depth?.url).toBe('/depth');
    expect(projectStore.view?.thresholds).toEqual([0, 85, 170, 255]);
  });

  it('records an ApiError message in the log store when a mutation fails', async () => {
    projectStore.applyView(makeView());

    const fetchMock = vi.fn(async () =>
      jsonResponse(409, { error: { code: 'busy', message: 'the project is busy' } }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await workflow.updateSliceCount(4);

    expect(jobStore.active).toBeNull();
    expect(logStore.entries.at(-1)?.message).toBe('the project is busy');
  });

  it('generateSlices polls the job and applies the resulting slices', async () => {
    projectStore.applyView(makeView());

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      const method = init?.method ?? 'GET';

      if (url === '/api/v1/projects/appstate-test/slices' && method === 'POST') {
        return jsonResponse(202, { job: { id: 'job-2', kind: 'slices', status: 'queued', progress: 0 } });
      }
      if (url === '/api/v1/jobs/job-2') {
        return jsonResponse(200, {
          id: 'job-2',
          kind: 'slices',
          status: 'succeeded',
          progress: 1,
          project: makeView({
            slices: [
              {
                index: 0,
                depth: 85,
                version: 1,
                canUndo: false,
                canRedo: false,
                positivePrompt: '',
                negativePrompt: '',
                image: { url: '/slice-0' },
                thumbnail: { url: '/slice-0-thumb' },
              },
            ],
          }),
        });
      }
      if (url.startsWith('/api/v1/projects/appstate-test/logs')) {
        return jsonResponse(200, { entries: [], next: 0 });
      }
      throw new Error(`Unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    await workflow.generateSlices();

    expect(projectStore.view?.slices).toHaveLength(1);
  });
});
