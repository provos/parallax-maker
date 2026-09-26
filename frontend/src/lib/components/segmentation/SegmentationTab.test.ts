import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import SegmentationTab from './SegmentationTab.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
import type { ProjectView, SliceView } from '../../api/types';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function makeSlice(index: number, depth: number): SliceView {
  return {
    index,
    depth,
    version: 1,
    canUndo: false,
    canRedo: false,
    positivePrompt: '',
    negativePrompt: '',
    image: { url: `/api/v1/projects/appstate-test/assets/slice-${index}` },
    thumbnail: { url: `/api/v1/projects/appstate-test/assets/slice-${index}-thumb` },
  };
}

function makeView(overrides: Partial<ProjectView> = {}): ProjectView {
  return {
    id: 'appstate-test',
    revision: 1,
    image: { width: 320, height: 240 },
    assets: { input: { url: '/input' }, depth: { url: '/depth' } },
    depthModel: 'dinov2',
    numSlices: 3,
    thresholds: [0, 85, 170, 255],
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

describe('SegmentationTab', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('disables the Generate button while a job is in flight, and re-enables it after', async () => {
    projectStore.applyView(makeView());
    render(SegmentationTab);

    expect(screen.getByTestId('generate-slices')).toBeEnabled();

    jobStore.begin('slices');
    await waitFor(() => expect(screen.getByTestId('generate-slices')).toBeDisabled());

    jobStore.end();
    await waitFor(() => expect(screen.getByTestId('generate-slices')).toBeEnabled());
  });

  it('sends baseRevision on threshold change, and retries once on stale_revision', async () => {
    projectStore.applyView(makeView());

    const fetchMock = vi.fn();
    fetchMock.mockResolvedValueOnce(
      jsonResponse(409, { error: { code: 'stale_revision', message: 'stale' } }),
    );
    fetchMock.mockResolvedValueOnce(jsonResponse(200, makeView({ revision: 2 })));
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { ...makeView({ revision: 3 }), changed: true }));
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
    vi.stubGlobal('fetch', fetchMock);

    render(SegmentationTab);
    const [firstHandle] = screen.getAllByTestId('threshold-handle');
    await fireEvent.input(firstHandle, { target: { value: '90' } });
    await fireEvent.change(firstHandle, { target: { value: '90' } });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4));

    const [firstUrl, firstInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(firstUrl).toBe('/api/v1/projects/appstate-test/thresholds');
    expect(firstInit.method).toBe('PUT');
    expect(JSON.parse(firstInit.body as string)).toMatchObject({ baseRevision: 1, values: [90, 170] });

    const [secondUrl] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(secondUrl).toBe('/api/v1/projects/appstate-test');

    const [thirdUrl, thirdInit] = fetchMock.mock.calls[2] as [string, RequestInit];
    expect(thirdUrl).toBe('/api/v1/projects/appstate-test/thresholds');
    expect(JSON.parse(thirdInit.body as string)).toMatchObject({ baseRevision: 2, values: [90, 170] });
  });

  describe('Actions panel enablement', () => {
    // Matches Dash exactly: webui.py never disables Create/Add/Remove/Balance
    // based on selection/mask state -- only "is a project loaded" and "is
    // nothing else in flight" gate them, same as Generate. Copy/Paste/Delete
    // moved to LayerPanel's header (see layers/LayerPanel.test.ts).
    // Preconditions are enforced by workflow.ts at click time.
    const actionTestIds = ['balance-slices', 'create-slice', 'add-mask-to-slice', 'remove-mask-from-slice'];

    it('disables every action button when there is no project', () => {
      render(SegmentationTab);
      for (const testId of actionTestIds) {
        expect(screen.getByTestId(testId)).toBeDisabled();
      }
    });

    it('enables every action button with a project loaded, regardless of selection or mask state', () => {
      projectStore.applyView(
        makeView({ selectedSlice: null, segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false } }),
      );
      render(SegmentationTab);
      for (const testId of actionTestIds) {
        expect(screen.getByTestId(testId)).toBeEnabled();
      }
    });

    it('disables every action button while a job is in flight', async () => {
      projectStore.applyView(makeView());
      render(SegmentationTab);
      jobStore.begin('slice-editing');
      await waitFor(() => {
        for (const testId of actionTestIds) {
          expect(screen.getByTestId(testId)).toBeDisabled();
        }
      });
    });
  });

  describe('ground plane', () => {
    it('marks the selected slice as the ground and fits it', async () => {
      const ground = { ...makeSlice(1, 170), isGround: true };
      projectStore.applyView(
        makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)], selectedSlice: 1 }),
      );
      const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url === '/api/v1/projects/appstate-test/slices/1/ground') {
          return jsonResponse(200, {
            ...makeView({ slices: [makeSlice(0, 85), ground], selectedSlice: 1 }),
            changed: true,
          });
        }
        if (url === '/api/v1/projects/appstate-test/ground/fit') {
          return jsonResponse(200, { ...makeView({ slices: [makeSlice(0, 85), ground] }), changed: true });
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
        throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
      });
      vi.stubGlobal('fetch', fetchMock);
      render(SegmentationTab);

      expect(screen.getByTestId('ground-toggle')).toHaveAttribute('aria-pressed', 'false');
      expect(screen.getByTestId('ground-fit')).toBeDisabled();
      await fireEvent.click(screen.getByTestId('ground-toggle'));

      await waitFor(() => expect(screen.getByTestId('ground-toggle')).toHaveAttribute('aria-pressed', 'true'));
      const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/slices/1/ground'));
      expect(JSON.parse(call![1]!.body as string)).toEqual({ isGround: true });

      await waitFor(() => expect(screen.getByTestId('ground-fit')).toBeEnabled());
      await fireEvent.click(screen.getByTestId('ground-fit'));
      await waitFor(() =>
        expect(fetchMock).toHaveBeenCalledWith(
          '/api/v1/projects/appstate-test/ground/fit',
          expect.objectContaining({ method: 'POST' }),
        ),
      );
    });

    it('needs a selected slice to toggle', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85)] }));
      render(SegmentationTab);
      expect(screen.getByTestId('ground-toggle')).toBeDisabled();
    });
  });
});
