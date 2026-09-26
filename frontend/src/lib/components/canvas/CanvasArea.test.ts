import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, waitFor } from '@testing-library/svelte';
import { flushSync } from 'svelte';
import CanvasArea from './CanvasArea.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
import { uiStore } from '../../state/ui.svelte';
import { viewportStore } from '../../state/viewport.svelte';
import type { ProjectView, SliceView } from '../../api/types';

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
    assets: { input: { url: '/input' }, depth: null },
    mainImage: { url: '/main' },
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
      depthModel: 'dinov2',
    },
    exports: { gltf: null, upscaled: false },
    ...overrides,
  };
}

/** A fetch stub that answers known URLs and throws on anything unexpected. */
function makeFetchMock(handlers: Record<string, (init?: RequestInit) => Response>) {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
    const handler = handlers[url];
    if (handler) return handler(init);
    throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
  });
}

const slice = { index: 0, depth: 0 } as unknown as SliceView;

describe('CanvasArea', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    uiStore.reset();
    viewportStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('resets the camera on entering Parallax when the main image differs from the last render', async () => {
    projectStore.applyView(makeView({ slices: [slice], mainImage: { url: '/main' } }));
    // uiStore.reset() leaves renderedMainUrl null, which differs from '/main'.
    expect(uiStore.renderedMainUrl).toBeNull();
    const fetchMock = makeFetchMock({
      '/api/v1/projects/appstate-test/camera/navigate': () =>
        jsonResponse(200, {
          ...makeView({ slices: [slice], mainImage: { url: '/main?v=2' } }),
          changed: true,
        }),
    });
    vi.stubGlobal('fetch', fetchMock);

    render(CanvasArea);
    uiStore.setView('parallax');
    flushSync();

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/camera/navigate'));
      expect(call).toBeDefined();
      expect(JSON.parse(call![1]!.body as string)).toEqual({ direction: 'reset' });
    });
  });

  it('does not navigate the camera when the main image already matches the last render', async () => {
    projectStore.applyView(makeView({ slices: [slice], mainImage: { url: '/main' } }));
    uiStore.setRenderedMainUrl('/main');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    render(CanvasArea);
    uiStore.setView('parallax');
    flushSync();

    // Give any (incorrectly) pending async call a chance to land before asserting.
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('does not navigate the camera when there are no slices yet', async () => {
    projectStore.applyView(makeView({ slices: [], mainImage: { url: '/main' } }));
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    render(CanvasArea);
    uiStore.setView('parallax');
    flushSync();

    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
