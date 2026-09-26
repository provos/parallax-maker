import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import ToolOptionsBar from './ToolOptionsBar.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
import { uiStore } from '../../state/ui.svelte';
import { maskToolsStore } from '../../state/maskTools.svelte';
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

describe('ToolOptionsBar', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    uiStore.reset();
    maskToolsStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('Segment tool - Select by', () => {
    beforeEach(() => uiStore.setTool('segment'));

    it('defaults to Object, the main way to make slices', () => {
      render(ToolOptionsBar);
      expect(screen.getByTestId('select-by-object')).toHaveAttribute('aria-checked', 'true');
      expect(screen.getByTestId('select-by-depth')).toHaveAttribute('aria-checked', 'false');
    });

    it('switches segmentationMode between Object and Depth band', async () => {
      render(ToolOptionsBar);
      await fireEvent.click(screen.getByTestId('select-by-depth'));
      expect(uiStore.segmentationMode).toBe('depth');
      expect(screen.getByTestId('select-by-depth')).toHaveAttribute('aria-checked', 'true');

      await fireEvent.click(screen.getByTestId('select-by-object'));
      expect(uiStore.segmentationMode).toBe('segment');
      expect(screen.getByTestId('select-by-object')).toHaveAttribute('aria-checked', 'true');
    });
  });

  describe('Segment tool - Multi-point / Commit', () => {
    beforeEach(() => uiStore.setTool('segment'));

    it('disables Multi in Depth band mode even with a project loaded', () => {
      uiStore.setSegmentationMode('depth');
      projectStore.applyView(makeView());
      render(ToolOptionsBar);
      expect(screen.getByTestId('multi-point')).toBeDisabled();
    });

    it('enables Multi once Object mode (the default) and a project are both present', () => {
      projectStore.applyView(makeView());
      render(ToolOptionsBar);
      expect(screen.getByTestId('multi-point')).toBeEnabled();
    });

    it('reflects multiPointMode as aria-pressed', () => {
      projectStore.applyView(
        makeView({ segmentation: { multiPointMode: true, queuedPoints: [], hasMask: false } }),
      );
      render(ToolOptionsBar);
      expect(screen.getByTestId('multi-point')).toHaveAttribute('aria-pressed', 'true');
    });

    it('keeps Commit disabled without queued points, even in multi-point Object mode', () => {
      projectStore.applyView(
        makeView({ segmentation: { multiPointMode: true, queuedPoints: [], hasMask: false } }),
      );
      render(ToolOptionsBar);
      expect(screen.getByTestId('multi-commit')).toBeDisabled();
    });

    it('enables Commit once points are queued in multi-point Object mode', () => {
      projectStore.applyView(
        makeView({
          segmentation: {
            multiPointMode: true,
            queuedPoints: [{ x: 1, y: 2, negative: false }],
            hasMask: false,
          },
        }),
      );
      render(ToolOptionsBar);
      expect(screen.getByTestId('multi-commit')).toBeEnabled();
    });

    it('disables Commit outside Object mode even with queued points', () => {
      uiStore.setSegmentationMode('depth');
      projectStore.applyView(
        makeView({
          segmentation: {
            multiPointMode: true,
            queuedPoints: [{ x: 1, y: 2, negative: false }],
            hasMask: false,
          },
        }),
      );
      render(ToolOptionsBar);
      expect(screen.getByTestId('multi-commit')).toBeDisabled();
    });
  });

  describe('Segment tool - Invert/Feather', () => {
    beforeEach(() => uiStore.setTool('segment'));

    it('disables Invert and Feather without a project', () => {
      render(ToolOptionsBar);
      expect(screen.getByTestId('invert-mask')).toBeDisabled();
      expect(screen.getByTestId('feather-mask')).toBeDisabled();
    });

    it('enables Invert and Feather with a project loaded', () => {
      projectStore.applyView(makeView());
      render(ToolOptionsBar);
      expect(screen.getByTestId('invert-mask')).toBeEnabled();
      expect(screen.getByTestId('feather-mask')).toBeEnabled();
    });

    it('invert-mask POSTs .../mask/invert', async () => {
      projectStore.applyView(makeView());
      const fetchMock = makeFetchMock({
        '/api/v1/projects/appstate-test/mask/invert': () =>
          jsonResponse(200, { ...makeView({ revision: 2 }), changed: true }),
      });
      vi.stubGlobal('fetch', fetchMock);

      render(ToolOptionsBar);
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

      render(ToolOptionsBar);
      await fireEvent.click(screen.getByTestId('feather-mask'));

      expect(fetchMock).not.toHaveBeenCalled();
      expect(logStore.entries.at(-1)?.message).toBe('No mask to feather');
    });
  });

  describe('Brush tool - MaskToolbar', () => {
    it('is not shown for the Segment tool', () => {
      uiStore.setTool('segment');
      render(ToolOptionsBar);
      expect(screen.queryByTestId('canvas-tools')).not.toBeInTheDocument();
    });

    it('is shown for the Brush tool', () => {
      uiStore.setTool('brush');
      render(ToolOptionsBar);
      const tools = screen.getByTestId('canvas-tools');
      expect(tools).toContainElement(screen.getByTestId('brush-size') as HTMLElement | null);
    });

    it('drives the shared brush state and the canvas actions', async () => {
      uiStore.setTool('brush');
      const clear = vi.fn(async () => {});
      const load = vi.fn(async () => {});
      render(ToolOptionsBar);
      // The real MaskCanvas (InputImagePanel.svelte) has bound its own
      // actions in the app; rebind test doubles for this isolated render.
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

  describe('Parallax view - camera pad', () => {
    const slice = { index: 0, depth: 0 } as unknown as SliceView;

    beforeEach(() => uiStore.setView('parallax'));

    it('is disabled until there are slices to navigate', () => {
      projectStore.applyView(makeView());
      render(ToolOptionsBar);
      expect(screen.getByTestId('camera-up')).toBeDisabled();
      expect(screen.getByTestId('camera-reset')).toBeDisabled();
    });

    it('posts the clicked direction and shows the re-rendered view', async () => {
      projectStore.applyView(makeView({ slices: [slice] }));
      const fetchMock = makeFetchMock({
        '/api/v1/projects/appstate-test/camera/navigate': () =>
          jsonResponse(200, {
            ...makeView({ slices: [slice], mainImage: { url: '/main?v=2' } }),
            changed: true,
          }),
      });
      vi.stubGlobal('fetch', fetchMock);
      render(ToolOptionsBar);

      await fireEvent.click(screen.getByTestId('camera-left'));

      // navigateCamera applies the fresh ProjectView, then (after also
      // awaiting the post-mutation log refresh) records the render so
      // CanvasArea/InputImagePanel know the display image is no longer
      // stale -- both updates land together once the whole call settles.
      await waitFor(() => {
        expect(projectStore.view?.mainImage?.url).toBe('/main?v=2');
        expect(uiStore.renderedMainUrl).toBe('/main?v=2');
      });
      const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/camera/navigate'));
      expect(call![1]).toMatchObject({ method: 'POST' });
      expect(JSON.parse(call![1]!.body as string)).toEqual({ direction: 'left' });
    });
  });

  describe('Horizon tool', () => {
    beforeEach(() => uiStore.setTool('horizon'));

    it('offers Fit ground only once a ground slice exists', () => {
      projectStore.applyView(makeView({ slices: [{ index: 0, depth: 0 } as unknown as SliceView] }));
      const { unmount } = render(ToolOptionsBar);
      expect(screen.queryByTestId('options-fit-ground')).toBeNull();
      unmount();

      projectStore.applyView(makeView({ slices: [{ index: 0, depth: 0, isGround: true } as unknown as SliceView] }));
      render(ToolOptionsBar);
      expect(screen.getByTestId('options-fit-ground')).toBeEnabled();
    });
  });
});

