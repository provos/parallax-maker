import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { flushSync } from 'svelte';
import InputImagePanel from './InputImagePanel.svelte';
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

/** Gives the (jsdom) <img> a stable, non-letterboxed 320x240 box for click math. */
function mockImageGeometry(img: HTMLElement): void {
  vi.spyOn(img, 'getBoundingClientRect').mockReturnValue({
    left: 0,
    top: 0,
    width: 320,
    height: 240,
    right: 320,
    bottom: 240,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } as DOMRect);
  Object.defineProperty(img, 'naturalWidth', { value: 320, configurable: true });
  Object.defineProperty(img, 'naturalHeight', { value: 240, configurable: true });
}

/** A job response that resolves the click/commit poll on its first GET. */
function succeededJob(project: ProjectView): unknown {
  return { id: 'job-1', kind: 'segmentation', status: 'succeeded', progress: 1, project };
}

describe('InputImagePanel', () => {
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

  describe('segmentation clicks (Segment tool only)', () => {
    it('sends a depth-mode click when Select by Depth band is chosen', async () => {
      uiStore.setTool('segment');
      uiStore.setSegmentationMode('depth');
      projectStore.applyView(makeView());
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(
        jsonResponse(202, { job: { id: 'job-1', kind: 'segmentation', status: 'queued', progress: 0 } }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, succeededJob(makeView({ revision: 2 }))));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      render(InputImagePanel);
      const img = screen.getByTestId('main-image') as HTMLImageElement;
      mockImageGeometry(img);
      await fireEvent.click(img, { clientX: 160, clientY: 120 });

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/segmentation/click');
      expect(init.method).toBe('POST');
      expect(JSON.parse(init.body as string)).toEqual({
        x: 160,
        y: 120,
        mode: 'depth',
        shiftKey: false,
        ctrlKey: false,
      });
    });

    it('sends instance mode with shiftKey (Select by Object, the default)', async () => {
      uiStore.setTool('segment');
      expect(uiStore.segmentationMode).toBe('segment');
      projectStore.applyView(makeView());
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(
        jsonResponse(202, { job: { id: 'job-1', kind: 'segmentation', status: 'queued', progress: 0 } }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, succeededJob(makeView({ revision: 2 }))));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      render(InputImagePanel);
      const img = screen.getByTestId('main-image') as HTMLImageElement;
      mockImageGeometry(img);
      await fireEvent.click(img, { clientX: 80, clientY: 96, shiftKey: true });

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
      const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(JSON.parse(init.body as string)).toEqual({
        x: 80,
        y: 96,
        mode: 'instance',
        shiftKey: true,
        ctrlKey: false,
      });
    });

    it('sends ctrlKey through untouched (not metaKey)', async () => {
      uiStore.setTool('segment');
      projectStore.applyView(makeView());
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(
        jsonResponse(202, { job: { id: 'job-1', kind: 'segmentation', status: 'queued', progress: 0 } }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, succeededJob(makeView({ revision: 2 }))));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      render(InputImagePanel);
      const img = screen.getByTestId('main-image') as HTMLImageElement;
      mockImageGeometry(img);
      await fireEvent.click(img, { clientX: 80, clientY: 96, ctrlKey: true, metaKey: true });

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
      const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(JSON.parse(init.body as string)).toMatchObject({ shiftKey: false, ctrlKey: true });
    });

    it('does not send a click while busy', async () => {
      uiStore.setTool('segment');
      projectStore.applyView(makeView());
      jobStore.begin('depth');
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      render(InputImagePanel);
      const img = screen.getByTestId('main-image') as HTMLImageElement;
      mockImageGeometry(img);
      await fireEvent.click(img, { clientX: 160, clientY: 120 });

      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('does not send a click when there is no project', async () => {
      uiStore.setTool('segment');
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      render(InputImagePanel);
      const img = screen.getByTestId('main-image') as HTMLImageElement;
      mockImageGeometry(img);
      await fireEvent.click(img, { clientX: 160, clientY: 120 });

      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('ignores a click that truncates to a pixel outside the image', async () => {
      uiStore.setTool('segment');
      projectStore.applyView(makeView());
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      render(InputImagePanel);
      const img = screen.getByTestId('main-image') as HTMLImageElement;
      mockImageGeometry(img);
      // clientX == rect.width truncates to naturalWidth, outside [0, naturalWidth).
      await fireEvent.click(img, { clientX: 320, clientY: 120 });

      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('does nothing with the Pan tool, even with a project loaded', async () => {
      // uiStore.reset() leaves the default tool, Pan.
      expect(uiStore.tool).toBe('pan');
      projectStore.applyView(makeView());
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      render(InputImagePanel);
      const img = screen.getByTestId('main-image') as HTMLImageElement;
      mockImageGeometry(img);
      await fireEvent.click(img, { clientX: 160, clientY: 120 });

      expect(fetchMock).not.toHaveBeenCalled();
    });
  });

  describe('empty state (no input image)', () => {
    it('shows the drop zone and choose-image affordance', () => {
      render(InputImagePanel);
      expect(screen.getByTestId('empty-state')).toBeInTheDocument();
      expect(screen.getByTestId('choose-image')).toBeEnabled();
    });

    it('choose-image opens the hidden file input', async () => {
      render(InputImagePanel);
      const input = screen.getByTestId('upload-image-input') as HTMLInputElement;
      const clickSpy = vi.spyOn(input, 'click');
      await fireEvent.click(screen.getByTestId('choose-image'));
      expect(clickSpy).toHaveBeenCalledOnce();
    });

    it('disables choose-image while busy', () => {
      jobStore.begin('upload');
      render(InputImagePanel);
      expect(screen.getByTestId('choose-image')).toBeDisabled();
    });
  });

  describe('zoom and pan', () => {
    it('zooms in about the cursor on an upward wheel tick', async () => {
      projectStore.applyView(makeView());
      render(InputImagePanel);
      const stage = screen.getByTestId('input-image-panel');
      const fitEl = stage.querySelector('.image-fit') as HTMLElement;
      vi.spyOn(fitEl, 'getBoundingClientRect').mockReturnValue({
        left: 0, top: 0, width: 320, height: 240, right: 320, bottom: 240, x: 0, y: 0, toJSON: () => ({}),
      } as DOMRect);

      expect(viewportStore.scale).toBe(1);
      await fireEvent.wheel(stage, { clientX: 160, clientY: 120, deltaY: -100 });
      expect(viewportStore.scale).toBeGreaterThan(1);
    });

    it('ignores wheel events without an input image', async () => {
      render(InputImagePanel);
      const stage = screen.getByTestId('input-image-panel');
      await fireEvent.wheel(stage, { clientX: 160, clientY: 120, deltaY: -100 });
      expect(viewportStore.scale).toBe(1);
    });

    it('pans on a primary-button drag past the threshold, and suppresses the click that follows', async () => {
      projectStore.applyView(makeView());
      uiStore.setTool('segment');
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      render(InputImagePanel);
      const stage = screen.getByTestId('input-image-panel');
      const img = screen.getByTestId('main-image') as HTMLImageElement;
      mockImageGeometry(img);
      Object.defineProperty(stage, 'setPointerCapture', { value: vi.fn(), configurable: true });
      Object.defineProperty(stage, 'releasePointerCapture', { value: vi.fn(), configurable: true });

      await fireEvent.pointerDown(stage, { pointerId: 1, button: 0, clientX: 100, clientY: 100 });
      // panBy reads movementX/movementY (not the clientX/clientY delta), so
      // the fired event must carry them explicitly -- jsdom never computes
      // mouse-movement deltas between synthetic events on its own. clientX/Y
      // still need to move past PAN_THRESHOLD_PX to engage the drag at all.
      await fireEvent.pointerMove(stage, { pointerId: 1, clientX: 120, clientY: 130, movementX: 20, movementY: 30 });
      expect(viewportStore.panX).toBeCloseTo(20);
      expect(viewportStore.panY).toBeCloseTo(30);
      await fireEvent.pointerUp(stage, { pointerId: 1, clientX: 120, clientY: 130 });

      // The drag suppresses the click the browser still fires on the <img>.
      await fireEvent.click(img, { clientX: 120, clientY: 130 });
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('does not engage drag-to-pan for a plain click (no movement)', async () => {
      projectStore.applyView(makeView());
      render(InputImagePanel);
      const stage = screen.getByTestId('input-image-panel');
      Object.defineProperty(stage, 'setPointerCapture', { value: vi.fn(), configurable: true });

      await fireEvent.pointerDown(stage, { pointerId: 1, button: 0, clientX: 100, clientY: 100 });
      await fireEvent.pointerUp(stage, { pointerId: 1, clientX: 100, clientY: 100 });
      expect(viewportStore.panX).toBe(0);
      expect(viewportStore.panY).toBe(0);
    });
  });

  describe('brush tool - mask canvas', () => {
    it('painting on the canvas never opens the upload file chooser', async () => {
      projectStore.applyView(makeView());
      uiStore.setTool('brush');
      render(InputImagePanel);
      const openChooser = vi.spyOn(screen.getByTestId('upload-image-input') as HTMLInputElement, 'click');

      await fireEvent.click(screen.getByTestId('mask-canvas'));

      expect(openChooser).not.toHaveBeenCalled();
    });
  });

  describe('horizon overlay (Horizon tool only)', () => {
    const withHorizon = (row: number) =>
      makeView({
        settings: {
          darkMode: false,
          camera: { distance: 100, focalLength: 50, maxDistance: 500, pitch: 0, groundNear: 0, horizonRow: row },
          meshDisplacement: 0,
          depthModel: 'dinov2',
        },
      });

    function stubSettings(fetchMock: ReturnType<typeof vi.fn>) {
      fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url === '/api/v1/projects/appstate-test/settings') {
          const body = JSON.parse(init!.body as string);
          return jsonResponse(200, { ...withHorizon(body.camera.horizonRow), changed: true });
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
        throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
      });
    }

    it('is hidden until the Horizon tool is picked, then sits on the horizon row', () => {
      projectStore.applyView(withHorizon(120));
      render(InputImagePanel);
      expect(screen.queryByTestId('horizon-line')).not.toBeInTheDocument();

      uiStore.setTool('horizon');
      flushSync();

      const line = screen.getByTestId('horizon-line');
      expect(line).toHaveAttribute('data-row', '120');
      expect(line.style.top).toBe('50%'); // 120 of 240 rows
    });

    it('dragging commits the new horizon row', async () => {
      projectStore.applyView(withHorizon(120));
      const fetchMock = vi.fn();
      stubSettings(fetchMock);
      vi.stubGlobal('fetch', fetchMock);
      render(InputImagePanel);
      uiStore.setTool('horizon');
      flushSync();

      const line = screen.getByTestId('horizon-line');
      // The image box is 240 CSS px tall for the 240-row image.
      vi.spyOn(line.parentElement!, 'getBoundingClientRect').mockReturnValue({
        left: 0, top: 0, width: 320, height: 240, right: 320, bottom: 240, x: 0, y: 0, toJSON: () => ({}),
      } as DOMRect);
      line.setPointerCapture = () => {};
      await fireEvent.pointerDown(line, { pointerId: 1, button: 0, clientY: 120 });
      await fireEvent.pointerMove(line, { pointerId: 1, clientY: 180 });
      expect(line).toHaveAttribute('data-row', '180');
      await fireEvent.pointerUp(line, { pointerId: 1, clientY: 180 });

      await waitFor(() => expect(fetchMock).toHaveBeenCalled());
      const body = JSON.parse(fetchMock.mock.calls[0][1]!.body as string);
      expect(body.camera).toEqual({ distance: 100, focalLength: 50, maxDistance: 500, horizonRow: 180 });
      await waitFor(() => expect(screen.getByTestId('horizon-line')).toHaveAttribute('data-row', '180'));
    });

    it('arrow keys nudge the horizon', async () => {
      projectStore.applyView(withHorizon(120));
      const fetchMock = vi.fn();
      stubSettings(fetchMock);
      vi.stubGlobal('fetch', fetchMock);
      render(InputImagePanel);
      uiStore.setTool('horizon');
      flushSync();

      await fireEvent.keyDown(screen.getByTestId('horizon-line'), { key: 'ArrowUp', shiftKey: true });
      await waitFor(() => expect(fetchMock).toHaveBeenCalled());
      expect(JSON.parse(fetchMock.mock.calls[0][1]!.body as string).camera.horizonRow).toBe(110);
    });
  });

  describe('inpainting candidate preview', () => {
    const slice = (index: number) =>
      ({ index, depth: 10 * index, image: { url: `/slice-${index}` }, thumbnail: { url: `/t-${index}` } }) as unknown as SliceView;
    const withCandidates = (selectedCandidate: number | null, selectedSlice = 1): ProjectView =>
      makeView({
        slices: [slice(0), slice(1)],
        selectedSlice,
        inpainting: {
          ...makeView().inpainting,
          candidates: {
            generationId: 'gen',
            sliceIndex: 1,
            images: [{ url: '/cand-0' }, { url: '/cand-1' }, { url: '/cand-2' }],
          },
          selectedCandidate,
        },
      });

    beforeEach(() => uiStore.setView('slice'));

    it('shows the picked candidate in place of the slice and hides the mask', () => {
      projectStore.applyView(withCandidates(2));
      render(InputImagePanel);
      expect(screen.getByTestId('view-slice-image')).toHaveAttribute('src', '/cand-2');
      expect(screen.getByTestId('view-slice-image')).toHaveAttribute('data-candidate', '2');
      expect(screen.getByTestId('mask-canvas')).toHaveClass('hidden');
    });

    it('shows the slice and its mask again when no candidate is picked', () => {
      projectStore.applyView(withCandidates(2));
      render(InputImagePanel);
      projectStore.applyView(withCandidates(null));
      flushSync();
      expect(screen.getByTestId('view-slice-image')).toHaveAttribute('src', '/slice-1');
      expect(screen.getByTestId('mask-canvas')).not.toHaveClass('hidden');
    });

    it("ignores candidates of a slice that isn't selected", () => {
      projectStore.applyView(withCandidates(0, 0));
      render(InputImagePanel);
      expect(screen.getByTestId('view-slice-image')).toHaveAttribute('src', '/slice-0');
      expect(screen.getByTestId('mask-canvas')).not.toHaveClass('hidden');
    });

    it('swaps the candidate into the composite view', () => {
      uiStore.setView('composite');
      projectStore.applyView(withCandidates(1));
      render(InputImagePanel);
      const sources = [...screen.getByTestId('view-composite-layers').querySelectorAll('img')].map((img) =>
        img.getAttribute('src'),
      );
      expect(sources).toEqual(['/slice-0', '/cand-1']);
    });
  });
});
