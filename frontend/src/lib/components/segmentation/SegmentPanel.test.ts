import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import SegmentPanel from './SegmentPanel.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
import { uiStore } from '../../state/ui.svelte';
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

describe('SegmentPanel', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    uiStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('disables the Generate button while a job is in flight, and re-enables it after', async () => {
    projectStore.applyView(makeView());
    render(SegmentPanel);

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

    render(SegmentPanel);
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
      render(SegmentPanel);
      for (const testId of actionTestIds) {
        expect(screen.getByTestId(testId)).toBeDisabled();
      }
    });

    it('enables every action button with a project loaded, regardless of selection or mask state', () => {
      projectStore.applyView(
        makeView({ selectedSlice: null, segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false } }),
      );
      render(SegmentPanel);
      for (const testId of actionTestIds) {
        expect(screen.getByTestId(testId)).toBeEnabled();
      }
    });

    it('disables every action button while a job is in flight', async () => {
      projectStore.applyView(makeView());
      render(SegmentPanel);
      jobStore.begin('slice-editing');
      await waitFor(() => {
        for (const testId of actionTestIds) {
          expect(screen.getByTestId(testId)).toBeDisabled();
        }
      });
    });
  });

  describe('Segment objects sub-steps and summary', () => {
    it('starts at step 1 (click an object) with nothing selected', () => {
      projectStore.applyView(makeView());
      render(SegmentPanel);
      const items = screen.getAllByRole('listitem');
      expect(items[0]).toHaveAttribute('data-state', 'now');
      expect(items[1]).toHaveAttribute('data-state', 'todo');
      expect(items[2]).toHaveAttribute('data-state', 'todo');
      expect(screen.getByTestId('selection-summary')).toHaveTextContent('Nothing selected yet');
    });

    it('moves to step 2 (refine) once points are queued, and pluralizes the summary', () => {
      projectStore.applyView(
        makeView({
          segmentation: {
            multiPointMode: true,
            queuedPoints: [{ x: 1, y: 2, negative: false }, { x: 3, y: 4, negative: false }],
            hasMask: false,
          },
        }),
      );
      render(SegmentPanel);
      const items = screen.getAllByRole('listitem');
      expect(items[0]).toHaveAttribute('data-state', 'done');
      expect(items[1]).toHaveAttribute('data-state', 'now');
      expect(items[2]).toHaveAttribute('data-state', 'todo');
      expect(screen.getByTestId('selection-summary')).toHaveTextContent('2 points queued');
    });

    it('uses the singular for exactly one queued point', () => {
      projectStore.applyView(
        makeView({
          segmentation: { multiPointMode: true, queuedPoints: [{ x: 1, y: 2, negative: false }], hasMask: false },
        }),
      );
      render(SegmentPanel);
      expect(screen.getByTestId('selection-summary')).toHaveTextContent('1 point queued');
    });

    it('moves to step 3 (make it a slice) once a mask exists, regardless of queued points', () => {
      projectStore.applyView(
        makeView({ segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true } }),
      );
      render(SegmentPanel);
      const items = screen.getAllByRole('listitem');
      expect(items[0]).toHaveAttribute('data-state', 'done');
      expect(items[1]).toHaveAttribute('data-state', 'done');
      expect(items[2]).toHaveAttribute('data-state', 'now');
      expect(screen.getByTestId('selection-summary')).toHaveTextContent('Selection ready');
    });
  });

  describe('Split by depth', () => {
    it('is open by default while there are no slices', () => {
      projectStore.applyView(makeView({ slices: [] }));
      render(SegmentPanel);
      expect(screen.getByTestId('split-toggle')).toHaveAttribute('aria-expanded', 'true');
    });

    it('is closed by default once slices exist', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85)] }));
      render(SegmentPanel);
      expect(screen.getByTestId('split-toggle')).toHaveAttribute('aria-expanded', 'false');
    });

    it('toggles open/closed on click, overriding the default', async () => {
      projectStore.applyView(makeView({ slices: [] }));
      const { container } = render(SegmentPanel);
      const toggle = screen.getByTestId('split-toggle');
      expect(toggle).toHaveAttribute('aria-expanded', 'true');
      expect(container.querySelector('.split-body')?.classList.contains('hidden')).toBe(false);

      await fireEvent.click(toggle);
      expect(toggle).toHaveAttribute('aria-expanded', 'false');
      expect(container.querySelector('.split-body')?.classList.contains('hidden')).toBe(true);

      await fireEvent.click(toggle);
      expect(toggle).toHaveAttribute('aria-expanded', 'true');
      expect(container.querySelector('.split-body')?.classList.contains('hidden')).toBe(false);
    });

    it('PUTs the new slice count on change', async () => {
      projectStore.applyView(makeView({ numSlices: 3 }));
      const fetchMock = makeFetchMock({
        '/api/v1/projects/appstate-test/slice-count': () =>
          jsonResponse(200, makeView({ numSlices: 5 })),
      });
      vi.stubGlobal('fetch', fetchMock);

      render(SegmentPanel);
      await fireEvent.change(screen.getByTestId('num-slices'), { target: { value: '5' } });

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/slice-count'));
      expect(call![1]).toMatchObject({ method: 'PUT' });
      expect(JSON.parse(call![1]!.body as string)).toEqual({ numSlices: 5 });
      expect(uiStore.pendingNumSlices).toBe(5);
    });
  });

  describe('Selected slice section', () => {
    it('is not shown without a selected slice', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85)], selectedSlice: null }));
      render(SegmentPanel);
      expect(screen.queryByTestId('selected-slice-section')).toBeNull();
    });

    it('is shown once a slice is selected', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85)], selectedSlice: 0 }));
      render(SegmentPanel);
      expect(screen.getByTestId('selected-slice-section')).toBeInTheDocument();
    });

    it('commits the new depth on change', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85)], selectedSlice: 0 }));
      const fetchMock = makeFetchMock({
        '/api/v1/projects/appstate-test/slices/0/depth': () =>
          jsonResponse(200, { ...makeView({ slices: [makeSlice(0, 120)], selectedSlice: 0 }), changed: true }),
      });
      vi.stubGlobal('fetch', fetchMock);

      render(SegmentPanel);
      await fireEvent.change(screen.getByTestId('selected-depth-slider'), { target: { value: '120' } });

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/depth'));
      expect(call![1]).toMatchObject({ method: 'PUT' });
      expect(JSON.parse(call![1]!.body as string)).toEqual({ depth: 120 });
    });

    it('clamps the committed depth to 0-255 from either the slider or the number field', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85)], selectedSlice: 0 }));
      const fetchMock = makeFetchMock({
        '/api/v1/projects/appstate-test/slices/0/depth': () =>
          jsonResponse(200, { ...makeView({ slices: [makeSlice(0, 255)], selectedSlice: 0 }), changed: true }),
      });
      vi.stubGlobal('fetch', fetchMock);

      render(SegmentPanel);
      await fireEvent.change(screen.getByTestId('selected-depth-input'), { target: { value: '999' } });

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/depth'));
      expect(JSON.parse(call![1]!.body as string)).toEqual({ depth: 255 });
    });

    it('does not commit when the value is unchanged', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85)], selectedSlice: 0 }));
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      render(SegmentPanel);
      await fireEvent.change(screen.getByTestId('selected-depth-slider'), { target: { value: '85' } });

      expect(fetchMock).not.toHaveBeenCalled();
    });

    describe('ground plane', () => {
      it('marks the selected slice as the ground plane (Fit ground now lives in GroundPanel)', async () => {
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
          if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
          throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
        });
        vi.stubGlobal('fetch', fetchMock);
        render(SegmentPanel);

        expect(screen.getByTestId('ground-toggle')).toHaveAttribute('aria-pressed', 'false');
        expect(screen.getByTestId('ground-toggle')).toHaveTextContent('Make ground plane');
        expect(screen.queryByTestId('ground-fit')).toBeNull();
        await fireEvent.click(screen.getByTestId('ground-toggle'));

        await waitFor(() => expect(screen.getByTestId('ground-toggle')).toHaveAttribute('aria-pressed', 'true'));
        expect(screen.getByTestId('ground-toggle')).toHaveTextContent('Ground plane');
        const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/slices/1/ground'));
        expect(JSON.parse(call![1]!.body as string)).toEqual({ isGround: true });
      });
    });
  });
});
