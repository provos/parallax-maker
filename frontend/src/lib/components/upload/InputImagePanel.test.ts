import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import InputImagePanel from './InputImagePanel.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
import { uiStore } from '../../state/ui.svelte';
import type { ProjectView, SliceView } from '../../api/types';
import { maskToolsStore } from '../../state/maskTools.svelte';

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
      depthModel: "dinov2",
    },
    exports: { gltf: null, upscaled: false },
    ...overrides,
  };
}

/** Gives the (jsdom) <img> a stable, non-letterboxed 320x240 box for click math. */
function mockImageGeometry(img: HTMLImageElement): void {
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
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends a depth-mode click with no modifiers by default', async () => {
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

  it('sends instance mode with shiftKey when the Mode Selector is Instance Segmentation', async () => {
    uiStore.setSegmentationMode('segment');
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
    uiStore.setSegmentationMode('segment');
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
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    render(InputImagePanel);
    const img = screen.getByTestId('main-image') as HTMLImageElement;
    mockImageGeometry(img);
    await fireEvent.click(img, { clientX: 160, clientY: 120 });

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('ignores a click that truncates to a pixel outside the image', async () => {
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

  describe('Multi/Commit enablement', () => {
    it('disables Multi in Depth Map mode even with a project loaded', () => {
      projectStore.applyView(makeView());
      render(InputImagePanel);
      expect(screen.getByTestId('multi-point')).toBeDisabled();
    });

    it('enables Multi once Instance Segmentation mode and a project are both present', () => {
      uiStore.setSegmentationMode('segment');
      projectStore.applyView(makeView());
      render(InputImagePanel);
      expect(screen.getByTestId('multi-point')).toBeEnabled();
    });

    it('reflects multiPointMode as aria-pressed', () => {
      uiStore.setSegmentationMode('segment');
      projectStore.applyView(
        makeView({ segmentation: { multiPointMode: true, queuedPoints: [], hasMask: false } }),
      );
      render(InputImagePanel);
      expect(screen.getByTestId('multi-point')).toHaveAttribute('aria-pressed', 'true');
    });

    it('keeps Commit disabled without queued points, even in multi-point instance mode', () => {
      uiStore.setSegmentationMode('segment');
      projectStore.applyView(
        makeView({ segmentation: { multiPointMode: true, queuedPoints: [], hasMask: false } }),
      );
      render(InputImagePanel);
      expect(screen.getByTestId('multi-commit')).toBeDisabled();
    });

    it('enables Commit once points are queued in multi-point instance mode', () => {
      uiStore.setSegmentationMode('segment');
      projectStore.applyView(
        makeView({
          segmentation: {
            multiPointMode: true,
            queuedPoints: [{ x: 1, y: 2, negative: false }],
            hasMask: false,
          },
        }),
      );
      render(InputImagePanel);
      expect(screen.getByTestId('multi-commit')).toBeEnabled();
    });

    it('disables Commit outside Instance Segmentation mode even with queued points', () => {
      projectStore.applyView(
        makeView({
          segmentation: {
            multiPointMode: true,
            queuedPoints: [{ x: 1, y: 2, negative: false }],
            hasMask: false,
          },
        }),
      );
      render(InputImagePanel);
      expect(screen.getByTestId('multi-commit')).toBeDisabled();
    });
  });

  describe('Checkerboard/Invert/Feather mask tools', () => {
    it('disables all three without a project', () => {
      render(InputImagePanel);
      expect(screen.getByTestId('toggle-checkerboard')).toBeDisabled();
      expect(screen.getByTestId('invert-mask')).toBeDisabled();
      expect(screen.getByTestId('feather-mask')).toBeDisabled();
    });

    it('enables all three with a project loaded', () => {
      projectStore.applyView(makeView());
      render(InputImagePanel);
      expect(screen.getByTestId('toggle-checkerboard')).toBeEnabled();
      expect(screen.getByTestId('invert-mask')).toBeEnabled();
      expect(screen.getByTestId('feather-mask')).toBeEnabled();
    });

    it('reflects useCheckerboard as aria-pressed', () => {
      projectStore.applyView(makeView({ useCheckerboard: true }));
      render(InputImagePanel);
      expect(screen.getByTestId('toggle-checkerboard')).toHaveAttribute('aria-pressed', 'true');
    });

    it('toggle-checkerboard PUTs the inverse of the current flag', async () => {
      projectStore.applyView(makeView({ useCheckerboard: false }));
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(
        jsonResponse(200, { ...makeView({ revision: 2, useCheckerboard: true }), changed: true }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      render(InputImagePanel);
      await fireEvent.click(screen.getByTestId('toggle-checkerboard'));

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/display');
      expect(JSON.parse(init.body as string)).toEqual({ useCheckerboard: true });
    });

    it('invert-mask POSTs .../mask/invert', async () => {
      projectStore.applyView(makeView());
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { ...makeView({ revision: 2 }), changed: true }));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      render(InputImagePanel);
      await fireEvent.click(screen.getByTestId('invert-mask'));

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/projects/appstate-test/mask/invert');
    });

    it('feather-mask logs a no-op without calling the API when there is no mask', async () => {
      projectStore.applyView(
        makeView({ segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false } }),
      );
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      render(InputImagePanel);
      await fireEvent.click(screen.getByTestId('feather-mask'));

      expect(fetchMock).not.toHaveBeenCalled();
      expect(logStore.entries.at(-1)?.message).toBe('No mask to feather');
    });
  });

  describe('camera navigation', () => {
    const slice = { index: 0, depth: 0 } as unknown as SliceView;

    it('is disabled until there are slices to navigate', () => {
      projectStore.applyView(makeView());
      render(InputImagePanel);
      expect(screen.getByTestId('camera-up')).toBeDisabled();
      expect(screen.getByTestId('camera-reset')).toBeDisabled();
    });

    it('posts the clicked direction and shows the re-rendered view', async () => {
      projectStore.applyView(makeView({ slices: [slice] }));
      const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url === '/api/v1/projects/appstate-test/camera/navigate') {
          return jsonResponse(200, { ...makeView({ slices: [slice], mainImage: { url: '/main?v=2' } }), changed: true });
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
        throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
      });
      vi.stubGlobal('fetch', fetchMock);
      render(InputImagePanel);

      await fireEvent.click(screen.getByTestId('camera-left'));

      await waitFor(() => expect(screen.getByTestId('main-image')).toHaveAttribute('src', '/main?v=2'));
      const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/camera/navigate'));
      expect(call![1]).toMatchObject({ method: 'POST' });
      expect(JSON.parse(call![1]!.body as string)).toEqual({ direction: 'left' });
    });
  });

  describe('canvas tools', () => {
    afterEach(() => maskToolsStore.reset());

    it('sit in the tool row below the image, only on the Inpainting tab', () => {
      projectStore.applyView(makeView());
      render(InputImagePanel);
      expect(screen.queryByTestId('canvas-tools')).not.toBeInTheDocument();

      uiStore.setMainTab('Inpainting');
      return waitFor(() => {
        const tools = screen.getByTestId('canvas-tools');
        // Outside the zoomed image box, next to the other tool buttons.
        expect(tools.closest('.image-stack')).toBeNull();
        expect(tools.parentElement).toContainElement(screen.getByTestId('invert-mask'));
      });
    });

    it('drive the shared brush state and the canvas actions', async () => {
      projectStore.applyView(makeView());
      uiStore.setMainTab('Inpainting');
      const clear = vi.fn(async () => {});
      const load = vi.fn(async () => {});
      render(InputImagePanel);
      // The real MaskCanvas has bound its own actions; rebind test doubles.
      maskToolsStore.bindCanvas({ clear, load });

      await fireEvent.input(screen.getByTestId('brush-size'), { target: { value: '25' } });
      expect(maskToolsStore.brushWidth).toBe(25);
      await fireEvent.click(screen.getByTestId('canvas-erase-mode'));
      expect(maskToolsStore.erasing).toBe(true);
      expect(screen.getByTestId('canvas-erase-mode')).toHaveAttribute('aria-pressed', 'true');
      // The eraser keeps its own width.
      expect(maskToolsStore.brushWidth).toBe(60);

      await fireEvent.click(screen.getByTestId('canvas-clear'));
      await fireEvent.click(screen.getByTestId('canvas-load'));
      expect(clear).toHaveBeenCalledOnce();
      expect(load).toHaveBeenCalledOnce();
    });
  });

  describe('horizon line', () => {
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

    it('is hidden until toggled, then sits on the horizon row', async () => {
      projectStore.applyView(withHorizon(120));
      render(InputImagePanel);
      expect(screen.queryByTestId('horizon-line')).not.toBeInTheDocument();

      await fireEvent.click(screen.getByTestId('horizon-toggle'));
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
      await fireEvent.click(screen.getByTestId('horizon-toggle'));

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
      await fireEvent.click(screen.getByTestId('horizon-toggle'));

      await fireEvent.keyDown(screen.getByTestId('horizon-line'), { key: 'ArrowUp', shiftKey: true });
      await waitFor(() => expect(fetchMock).toHaveBeenCalled());
      expect(JSON.parse(fetchMock.mock.calls[0][1]!.body as string).camera.horizonRow).toBe(110);
    });
  });
});
