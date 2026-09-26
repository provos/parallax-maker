import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { waitFor } from '@testing-library/svelte';
import * as workflow from './workflow';
import { projectStore } from './state/project.svelte';
import { jobStore } from './state/jobs.svelte';
import { logStore } from './state/logs.svelte';
import { uiStore } from './state/ui.svelte';
import { toastStore } from './state/toasts.svelte';
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

describe('workflow', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    jobStore.clearError();
    logStore.reset();
    uiStore.reset();
    toastStore.reset();
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
      if (url === '/api/v1/projects/appstate-test/settings' && method === 'PUT') {
        expect(JSON.parse(init!.body as string)).toEqual({ darkMode: true });
        return jsonResponse(200, { ...makeView({ settings: { ...makeView().settings, darkMode: true } }), changed: true });
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
    expect(uiStore.step).toBe('slices');
  });

  it('a new upload starts over: session progress is cleared', async () => {
    uiStore.markPreviewed();
    uiStore.markExported();
    uiStore.setStep('export');
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url === '/api/v1/projects' && init?.method === 'POST') {
          return jsonResponse(201, makeView({ settings: { ...makeView().settings, darkMode: true } }));
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
        // Leave the depth job failing: only the reset at creation matters here.
        return jsonResponse(500, { error: { code: 'internal', message: 'no depth in this test' } });
      }),
    );

    await workflow.uploadImage(new File(['bytes'], 'input.png', { type: 'image/png' }), 'midas');

    expect(uiStore.previewed).toBe(false);
    expect(uiStore.exported).toBe(false);
    expect(uiStore.step).toBe('image');
  });

  it('navigating the camera marks Preview done only when the move succeeds', async () => {
    await workflow.navigateCamera('left');
    expect(uiStore.previewed).toBe(false); // no project

    projectStore.applyView(makeView());
    let fail = true;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
        const url = String(input);
        if (url === '/api/v1/projects/appstate-test/camera/navigate') {
          return fail
            ? jsonResponse(409, { error: { code: 'not_ready', message: 'no slices' } })
            : jsonResponse(200, { ...makeView(), changed: true });
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
        throw new Error(`Unexpected fetch: ${url}`);
      }),
    );
    await workflow.navigateCamera('left');
    expect(uiStore.previewed).toBe(false);

    fail = false;
    await workflow.navigateCamera('left');
    expect(uiStore.previewed).toBe(true);
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

  describe('slice editing / mask tools', () => {
    function makeSlice(index: number, overrides: Partial<import('./api/types').SliceView> = {}) {
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

    // -- Precondition no-ops: Dash's own buttons never disable themselves for
    // missing selection/mask/clipboard state; they just log a plain no-op
    // message and skip the mutation entirely (see workflow.ts's slice-editing
    // section for the exact per-function precedence this pins down). None of
    // these should reach the network.

    it('deleteSlice logs "No slice selected" and does not call the API when nothing is selected', async () => {
      projectStore.applyView(makeView({ selectedSlice: null }));
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      await workflow.deleteSlice();

      expect(fetchMock).not.toHaveBeenCalled();
      expect(logStore.entries.at(-1)?.message).toBe('No slice selected');
    });

    it('addMaskToSlice checks for a mask before a selection, matching Dash\'s precedence', async () => {
      // No mask AND no selection: Dash's add_mask_slice_request checks
      // state.slice_mask before state.selected_slice, so "No mask selected"
      // wins even though neither precondition is met.
      projectStore.applyView(
        makeView({ selectedSlice: null, segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false } }),
      );
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      await workflow.addMaskToSlice();

      expect(fetchMock).not.toHaveBeenCalled();
      expect(logStore.entries.at(-1)?.message).toBe('No mask selected');
    });

    it('addMaskToSlice logs "No slice selected" once a mask exists but nothing is selected', async () => {
      projectStore.applyView(
        makeView({ selectedSlice: null, segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true } }),
      );
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      await workflow.addMaskToSlice();

      expect(fetchMock).not.toHaveBeenCalled();
      expect(logStore.entries.at(-1)?.message).toBe('No slice selected');
    });

    it('copySlice requires a mask only (no selection needed)', async () => {
      projectStore.applyView(
        makeView({ selectedSlice: null, segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false } }),
      );
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      await workflow.copySlice();

      expect(fetchMock).not.toHaveBeenCalled();
      expect(logStore.entries.at(-1)?.message).toBe('No mask selected');
    });

    it('pasteSlice checks the clipboard before the selection', async () => {
      projectStore.applyView(makeView({ selectedSlice: null, clipboard: false }));
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      await workflow.pasteSlice();

      expect(fetchMock).not.toHaveBeenCalled();
      expect(logStore.entries.at(-1)?.message).toBe('Nothing in the clipboard');
    });

    it('pasteSlice logs "No slice selected" once the clipboard is populated but nothing is selected', async () => {
      projectStore.applyView(makeView({ selectedSlice: null, clipboard: true }));
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      await workflow.pasteSlice();

      expect(fetchMock).not.toHaveBeenCalled();
      expect(logStore.entries.at(-1)?.message).toBe('No slice selected');
    });

    it('featherMask logs "No mask to feather" without calling the API when there is no mask', async () => {
      projectStore.applyView(
        makeView({ segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false } }),
      );
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      await workflow.featherMask();

      expect(fetchMock).not.toHaveBeenCalled();
      expect(logStore.entries.at(-1)?.message).toBe('No mask to feather');
    });

    // -- Successful mutations reach the documented endpoint and apply the result.

    it('createSlice POSTs .../slices/create and applies the resulting view', async () => {
      projectStore.applyView(makeView());
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(
        jsonResponse(200, { ...makeView({ revision: 2, slices: [makeSlice(0)] }), changed: true }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      await workflow.createSlice();

      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/slices/create');
      expect(init.method).toBe('POST');
      expect(projectStore.view?.slices).toHaveLength(1);
    });

    it('deleteSlice DELETEs the selected slice index', async () => {
      projectStore.applyView(makeView({ selectedSlice: 1 }));
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(
        jsonResponse(200, { ...makeView({ revision: 2, selectedSlice: null }), changed: true }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      await workflow.deleteSlice();

      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/slices/1');
      expect(init.method).toBe('DELETE');
    });

    it('setSliceDepth PUTs the depth for the given index', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0), makeSlice(1)] }));
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { ...makeView({ revision: 2 }), changed: true }));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      await workflow.setSliceDepth(1, 200);

      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/slices/1/depth');
      expect(init.method).toBe('PUT');
      expect(JSON.parse(init.body as string)).toEqual({ depth: 200 });
    });

    it('uploadSliceImage PUTs a multipart image to the given slice index', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0)] }));
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { ...makeView({ revision: 2 }), changed: true }));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      const file = new File(['bytes'], 'replacement.png', { type: 'image/png' });
      await workflow.uploadSliceImage(0, file);

      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/slices/0/image');
      expect(init.method).toBe('PUT');
      expect(init.body).toBeInstanceOf(FormData);
      expect((init.body as FormData).get('image')).toBe(file);
    });

    it('undoSlice/redoSlice POST to the per-index undo/redo endpoints', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, { canUndo: true })] }));
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { ...makeView({ revision: 2 }), changed: true }));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      await workflow.undoSlice(0);

      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/slices/0/undo');
      expect(init.method).toBe('POST');

      fetchMock.mockResolvedValueOnce(jsonResponse(200, { ...makeView({ revision: 3 }), changed: true }));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      await workflow.redoSlice(0);
      const [redoUrl] = fetchMock.mock.calls[2] as [string, RequestInit];
      expect(redoUrl).toBe('/api/v1/projects/appstate-test/slices/0/redo');
    });

    it('toggleCheckerboard PUTs the inverse of the current useCheckerboard flag', async () => {
      projectStore.applyView(makeView({ useCheckerboard: false }));
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(
        jsonResponse(200, { ...makeView({ revision: 2, useCheckerboard: true }), changed: true }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      await workflow.toggleCheckerboard();

      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/display');
      expect(JSON.parse(init.body as string)).toEqual({ useCheckerboard: true });
      expect(projectStore.view?.useCheckerboard).toBe(true);
    });

    it('invertMask and balanceSlices call their documented endpoints unconditionally', async () => {
      projectStore.applyView(makeView());
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { ...makeView({ revision: 2 }), changed: true }));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);
      await workflow.invertMask();
      expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/projects/appstate-test/mask/invert');

      fetchMock.mockResolvedValueOnce(
        jsonResponse(200, { ...makeView({ revision: 3 }), changed: false }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      await workflow.balanceSlices();
      expect(fetchMock.mock.calls[2][0]).toBe('/api/v1/projects/appstate-test/slices/balance');
    });
  });

  describe('reportError / toasts / cancellation', () => {
    it('a failed depth job shows an error toast with a View log action and records jobStore.lastError', async () => {
      projectStore.applyView(makeView());
      const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        const method = init?.method ?? 'GET';
        if (url === '/api/v1/projects/appstate-test/depth' && method === 'POST') {
          return jsonResponse(202, { job: { id: 'job-e', kind: 'depth', status: 'queued', progress: 0 } });
        }
        if (url === '/api/v1/jobs/job-e') {
          return jsonResponse(200, {
            id: 'job-e',
            kind: 'depth',
            status: 'failed',
            progress: 0.5,
            error: 'model exploded',
          });
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) {
          return jsonResponse(200, { entries: [], next: 0 });
        }
        throw new Error(`Unexpected fetch: ${method} ${url}`);
      });
      vi.stubGlobal('fetch', fetchMock);

      await workflow.startDepth('midas');

      const errorToast = toastStore.toasts.at(-1);
      expect(errorToast).toMatchObject({ kind: 'error', title: 'Depth map failed', message: 'model exploded' });
      // depth is not an inpainting-settings kind and startDepth passes no retry.
      expect(errorToast?.actions.map((a) => a.label)).toEqual(['View log']);
      expect(jobStore.lastError).toMatchObject({
        kind: 'depth',
        title: 'Depth map failed',
        message: 'model exploded',
      });
    });

    it('a failed inpainting generation offers Retry and Open settings; Retry re-issues the generate POST', async () => {
      projectStore.applyView(makeView({ selectedSlice: 0 }));
      let generateCalls = 0;
      const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        const method = init?.method ?? 'GET';
        if (url === '/api/v1/projects/appstate-test/slices/0/inpainting/generate' && method === 'POST') {
          generateCalls += 1;
          const jobId = generateCalls === 1 ? 'job-a' : 'job-b';
          return jsonResponse(202, { job: { id: jobId, kind: 'inpainting', status: 'queued', progress: 0 } });
        }
        if (url === '/api/v1/jobs/job-a') {
          return jsonResponse(200, {
            id: 'job-a',
            kind: 'inpainting',
            status: 'failed',
            progress: 1,
            error: 'model exploded',
          });
        }
        if (url === '/api/v1/jobs/job-b') {
          return jsonResponse(200, {
            id: 'job-b',
            kind: 'inpainting',
            status: 'succeeded',
            progress: 1,
            project: makeView({ selectedSlice: 0 }),
          });
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) {
          return jsonResponse(200, { entries: [], next: 0 });
        }
        throw new Error(`Unexpected fetch: ${method} ${url}`);
      });
      vi.stubGlobal('fetch', fetchMock);

      await workflow.generateInpainting('paint', 'a prompt', '');

      const errorToast = toastStore.toasts.at(-1);
      expect(errorToast).toMatchObject({ kind: 'error', title: 'Generation failed', message: 'model exploded' });
      expect(errorToast?.actions.map((a) => a.label)).toEqual(['Retry', 'Open settings', 'View log']);
      expect(jobStore.lastError?.kind).toBe('inpainting');
      expect(generateCalls).toBe(1);

      const retry = errorToast!.actions.find((a) => a.label === 'Retry')!;
      retry.run();

      await waitFor(() => expect(generateCalls).toBe(2));
      await waitFor(() => expect(jobStore.active).toBeNull());
    });

    it('a cancelled inpainting job shows only an info toast and clears no error', async () => {
      projectStore.applyView(makeView({ selectedSlice: 0 }));
      const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        const method = init?.method ?? 'GET';
        if (url === '/api/v1/projects/appstate-test/slices/0/inpainting/generate' && method === 'POST') {
          return jsonResponse(202, { job: { id: 'job-c', kind: 'inpainting', status: 'queued', progress: 0 } });
        }
        if (url === '/api/v1/jobs/job-c') {
          return jsonResponse(200, {
            id: 'job-c',
            kind: 'inpainting',
            status: 'cancelled',
            progress: 0.5,
            project: makeView({ selectedSlice: 0, revision: 2 }),
          });
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) {
          return jsonResponse(200, { entries: [], next: 0 });
        }
        throw new Error(`Unexpected fetch: ${method} ${url}`);
      });
      vi.stubGlobal('fetch', fetchMock);

      await workflow.generateInpainting('paint', '', '');

      expect(toastStore.toasts).toHaveLength(1);
      expect(toastStore.toasts[0]).toMatchObject({
        kind: 'info',
        title: 'Cancelled',
        message: 'Generating inpainting candidates',
      });
      expect(jobStore.lastError).toBeNull();
      expect(projectStore.view?.revision).toBe(2);
    });

    it('startDepth shows a success toast once the job succeeds', async () => {
      projectStore.applyView(makeView());
      const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        const method = init?.method ?? 'GET';
        if (url === '/api/v1/projects/appstate-test/depth' && method === 'POST') {
          return jsonResponse(202, { job: { id: 'job-d', kind: 'depth', status: 'queued', progress: 0 } });
        }
        if (url === '/api/v1/jobs/job-d') {
          return jsonResponse(200, {
            id: 'job-d',
            kind: 'depth',
            status: 'succeeded',
            progress: 1,
            project: makeView(),
          });
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) {
          return jsonResponse(200, { entries: [], next: 0 });
        }
        throw new Error(`Unexpected fetch: ${method} ${url}`);
      });
      vi.stubGlobal('fetch', fetchMock);

      await workflow.startDepth('midas');

      expect(toastStore.toasts).toHaveLength(1);
      expect(toastStore.toasts[0]).toMatchObject({ kind: 'success', title: 'Depth map ready' });
    });
  });
});
