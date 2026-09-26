import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import SelectionBar from './SelectionBar.svelte';
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
      depthModel: 'dinov2',
    },
    exports: { gltf: null, upscaled: false },
    ...overrides,
  };
}

/** A fetch stub that answers known URLs and auto-handles the log refresh; throws otherwise. */
function makeFetchMock(handlers: Record<string, (init?: RequestInit) => Response>) {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
    const handler = handlers[url];
    if (handler) return handler(init);
    throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
  });
}

describe('SelectionBar', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('is not rendered without a mask or queued points', () => {
    projectStore.applyView(makeView());
    render(SelectionBar);
    expect(screen.queryByTestId('selection-bar')).toBeNull();
  });

  it('is not rendered for queued points outside multi-point mode', () => {
    projectStore.applyView(
      makeView({
        segmentation: { multiPointMode: false, queuedPoints: [{ x: 1, y: 2, negative: false }], hasMask: false },
      }),
    );
    render(SelectionBar);
    expect(screen.queryByTestId('selection-bar')).toBeNull();
  });

  describe('multi-point queue', () => {
    it('shows the queued count and a pluralized Segment N points button', () => {
      projectStore.applyView(
        makeView({
          segmentation: {
            multiPointMode: true,
            queuedPoints: [{ x: 1, y: 2, negative: false }, { x: 3, y: 4, negative: false }],
            hasMask: false,
          },
        }),
      );
      render(SelectionBar);
      expect(screen.getByTestId('selection-bar')).toBeInTheDocument();
      expect(screen.getByTestId('selection-bar')).toHaveTextContent('2 points');
      expect(screen.getByTestId('selection-commit')).toHaveTextContent('Segment 2 points');
      expect(screen.queryByTestId('selection-add')).toBeNull();
      expect(screen.queryByTestId('selection-remove')).toBeNull();
      expect(screen.queryByTestId('selection-new-slice')).toBeNull();
    });

    it('uses the singular for exactly one queued point', () => {
      projectStore.applyView(
        makeView({
          segmentation: { multiPointMode: true, queuedPoints: [{ x: 1, y: 2, negative: false }], hasMask: false },
        }),
      );
      render(SelectionBar);
      expect(screen.getByTestId('selection-bar')).toHaveTextContent('1 point');
      expect(screen.getByTestId('selection-commit')).toHaveTextContent('Segment 1 point');
    });

    it('disables Segment N points while busy', async () => {
      projectStore.applyView(
        makeView({
          segmentation: { multiPointMode: true, queuedPoints: [{ x: 1, y: 2, negative: false }], hasMask: false },
        }),
      );
      render(SelectionBar);
      expect(screen.getByTestId('selection-commit')).toBeEnabled();

      jobStore.begin('segmentation');
      await waitFor(() => expect(screen.getByTestId('selection-commit')).toBeDisabled());
    });

    it('commits the queued points via POST .../segmentation/commit', async () => {
      projectStore.applyView(
        makeView({
          segmentation: { multiPointMode: true, queuedPoints: [{ x: 1, y: 2, negative: false }], hasMask: false },
        }),
      );
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(
        jsonResponse(202, { job: { id: 'job-1', kind: 'segmentation', status: 'queued', progress: 0 } }),
      );
      fetchMock.mockResolvedValueOnce(
        jsonResponse(200, {
          id: 'job-1',
          kind: 'segmentation',
          status: 'succeeded',
          progress: 1,
          project: makeView({
            revision: 2,
            segmentation: { multiPointMode: true, queuedPoints: [], hasMask: true },
          }),
        }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      render(SelectionBar);
      await fireEvent.click(screen.getByTestId('selection-commit'));

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/segmentation/commit');
      expect(init.method).toBe('POST');
    });
  });

  describe('with a mask', () => {
    it('shows the Selection summary, and disables Add/Remove without a selected slice', () => {
      projectStore.applyView(
        makeView({
          segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true },
          selectedSlice: null,
        }),
      );
      render(SelectionBar);
      expect(screen.getByTestId('selection-bar')).toHaveTextContent('Selection');
      expect(screen.getByTestId('selection-add')).toHaveTextContent('Add to slice');
      expect(screen.getByTestId('selection-add')).toBeDisabled();
      expect(screen.getByTestId('selection-remove')).toHaveTextContent('Remove from slice');
      expect(screen.getByTestId('selection-remove')).toBeDisabled();
      expect(screen.getByTestId('selection-new-slice')).toBeEnabled();
      expect(screen.queryByTestId('selection-commit')).toBeNull();
    });

    it('labels Add/Remove with the selected slice name, and enables them', () => {
      projectStore.applyView(
        makeView({
          slices: [makeSlice(0, 85), makeSlice(2, 170)],
          selectedSlice: 2,
          segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true },
        }),
      );
      render(SelectionBar);
      expect(screen.getByTestId('selection-add')).toHaveTextContent('Add to image_slice_2');
      expect(screen.getByTestId('selection-add')).toBeEnabled();
      expect(screen.getByTestId('selection-remove')).toHaveTextContent('Remove from image_slice_2');
      expect(screen.getByTestId('selection-remove')).toBeEnabled();
    });

    it('disables Add/Remove/New slice while busy', async () => {
      projectStore.applyView(
        makeView({
          slices: [makeSlice(2, 170)],
          selectedSlice: 2,
          segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true },
        }),
      );
      render(SelectionBar);
      jobStore.begin('slice-editing');
      await waitFor(() => {
        expect(screen.getByTestId('selection-add')).toBeDisabled();
        expect(screen.getByTestId('selection-remove')).toBeDisabled();
        expect(screen.getByTestId('selection-new-slice')).toBeDisabled();
      });
    });

    it('calls addMaskToSlice, removeMaskFromSlice and createSlice', async () => {
      projectStore.applyView(
        makeView({
          slices: [makeSlice(2, 170)],
          selectedSlice: 2,
          segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true },
        }),
      );
      const fetchMock = makeFetchMock({
        '/api/v1/projects/appstate-test/slices/2/add-mask': () =>
          jsonResponse(200, {
            ...makeView({
              slices: [makeSlice(2, 170)],
              selectedSlice: 2,
              segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true },
            }),
            changed: true,
          }),
        '/api/v1/projects/appstate-test/slices/2/remove-mask': () =>
          jsonResponse(200, {
            ...makeView({
              slices: [makeSlice(2, 170)],
              selectedSlice: 2,
              segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true },
            }),
            changed: true,
          }),
        '/api/v1/projects/appstate-test/slices/create': () =>
          jsonResponse(200, {
            ...makeView({
              slices: [makeSlice(2, 170), makeSlice(3, 200)],
              selectedSlice: 2,
              segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true },
            }),
            changed: true,
          }),
      });
      vi.stubGlobal('fetch', fetchMock);
      render(SelectionBar);

      await fireEvent.click(screen.getByTestId('selection-add'));
      await waitFor(() =>
        expect(fetchMock.mock.calls.some(([u]) => String(u).endsWith('/add-mask'))).toBe(true),
      );

      await fireEvent.click(screen.getByTestId('selection-remove'));
      await waitFor(() =>
        expect(fetchMock.mock.calls.some(([u]) => String(u).endsWith('/remove-mask'))).toBe(true),
      );

      await fireEvent.click(screen.getByTestId('selection-new-slice'));
      await waitFor(() =>
        expect(fetchMock.mock.calls.some(([u]) => String(u).endsWith('/slices/create'))).toBe(true),
      );
    });
  });
});
