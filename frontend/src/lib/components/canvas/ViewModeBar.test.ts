import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { flushSync } from 'svelte';
import ViewModeBar from './ViewModeBar.svelte';
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

function renderBar(onZoomIn = vi.fn(), onZoomOut = vi.fn()) {
  return { onZoomIn, onZoomOut, ...render(ViewModeBar, { props: { onZoomIn, onZoomOut } }) };
}

describe('ViewModeBar', () => {
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

  describe('checkerboard toggle', () => {
    it('is disabled without a project', () => {
      renderBar();
      expect(screen.getByTestId('toggle-checkerboard')).toBeDisabled();
    });

    it('is enabled with a project loaded', () => {
      projectStore.applyView(makeView());
      renderBar();
      expect(screen.getByTestId('toggle-checkerboard')).toBeEnabled();
    });

    it('reflects useCheckerboard as aria-pressed', () => {
      projectStore.applyView(makeView({ useCheckerboard: true }));
      renderBar();
      expect(screen.getByTestId('toggle-checkerboard')).toHaveAttribute('aria-pressed', 'true');
    });

    it('PUTs the inverse of the current flag', async () => {
      projectStore.applyView(makeView({ useCheckerboard: false }));
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(
        jsonResponse(200, { ...makeView({ revision: 2, useCheckerboard: true }), changed: true }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      renderBar();
      await fireEvent.click(screen.getByTestId('toggle-checkerboard'));

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/display');
      expect(JSON.parse(init.body as string)).toEqual({ useCheckerboard: true });
    });
  });

  describe('keyboard', () => {
    it('arrow keys move to the next available view, skipping disabled ones', async () => {
      // Input and Depth available; Slice (no selection) and the rest (no slices) are not.
      projectStore.applyView(makeView({ assets: { input: { url: '/input' }, depth: { url: '/depth' } } }));
      renderBar();
      const input = screen.getByTestId('view-input');
      expect(input).toHaveAttribute('tabindex', '0');
      expect(screen.getByTestId('view-depth')).toHaveAttribute('tabindex', '-1');

      input.focus();
      await fireEvent.keyDown(input, { key: 'ArrowRight' });
      expect(uiStore.view).toBe('depth');

      await fireEvent.keyDown(screen.getByTestId('view-depth'), { key: 'ArrowRight' });
      expect(uiStore.view).toBe('input'); // wraps past the disabled tabs
    });
  });

  describe('view tab availability', () => {
    const slice = { index: 0, depth: 0 } as unknown as SliceView;

    it('disables every tab without a project', () => {
      renderBar();
      expect(screen.getByTestId('view-input')).toBeDisabled();
      expect(screen.getByTestId('view-depth')).toBeDisabled();
      expect(screen.getByTestId('view-slice')).toBeDisabled();
      expect(screen.getByTestId('view-composite')).toBeDisabled();
      expect(screen.getByTestId('view-parallax')).toBeDisabled();
      expect(screen.getByTestId('view-3d')).toBeDisabled();
    });

    it('enables Input once an image is loaded, Depth once a depth map exists', () => {
      projectStore.applyView(makeView({ assets: { input: { url: '/input' }, depth: { url: '/depth' } } }));
      renderBar();
      expect(screen.getByTestId('view-input')).toBeEnabled();
      expect(screen.getByTestId('view-depth')).toBeEnabled();
    });

    it('keeps Depth disabled without a depth map, even with an image', () => {
      projectStore.applyView(makeView());
      renderBar();
      expect(screen.getByTestId('view-input')).toBeEnabled();
      expect(screen.getByTestId('view-depth')).toBeDisabled();
    });

    it('enables Slice only once a slice is selected', () => {
      projectStore.applyView(makeView({ slices: [slice], selectedSlice: null }));
      renderBar();
      expect(screen.getByTestId('view-slice')).toBeDisabled();

      projectStore.applyView(makeView({ revision: 2, slices: [slice], selectedSlice: 0 }));
      flushSync();
      expect(screen.getByTestId('view-slice')).toBeEnabled();
    });

    it('enables Composite, Parallax and 3D only once slices exist', () => {
      projectStore.applyView(makeView());
      renderBar();
      expect(screen.getByTestId('view-composite')).toBeDisabled();
      expect(screen.getByTestId('view-parallax')).toBeDisabled();
      expect(screen.getByTestId('view-3d')).toBeDisabled();

      projectStore.applyView(makeView({ revision: 2, slices: [slice] }));
      flushSync();
      expect(screen.getByTestId('view-composite')).toBeEnabled();
      expect(screen.getByTestId('view-parallax')).toBeEnabled();
      expect(screen.getByTestId('view-3d')).toBeEnabled();
    });

    it('clicking an enabled tab selects that view', async () => {
      projectStore.applyView(makeView({ slices: [slice] }));
      renderBar();
      expect(screen.getByTestId('view-input')).toHaveAttribute('aria-selected', 'true');

      await fireEvent.click(screen.getByTestId('view-composite'));
      expect(uiStore.view).toBe('composite');
      expect(screen.getByTestId('view-composite')).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByTestId('view-input')).toHaveAttribute('aria-selected', 'false');
    });
  });

  describe('zoom controls', () => {
    it('disables zoom buttons without an image', () => {
      renderBar();
      expect(screen.getByTestId('zoom-out')).toBeDisabled();
      expect(screen.getByTestId('zoom-in')).toBeDisabled();
      expect(screen.getByTestId('zoom-reset')).toBeDisabled();
    });

    it('shows the current zoom level from viewportStore', () => {
      projectStore.applyView(makeView());
      viewportStore.zoomInAt(0, 0);
      renderBar();
      expect(screen.getByTestId('zoom-level')).toHaveTextContent(`${Math.round(viewportStore.scale * 100)}%`);
    });

    it('zoom-in and zoom-out call the callback props, not the viewport store directly', async () => {
      projectStore.applyView(makeView());
      const { onZoomIn, onZoomOut } = renderBar();

      await fireEvent.click(screen.getByTestId('zoom-in'));
      expect(onZoomIn).toHaveBeenCalledOnce();
      await fireEvent.click(screen.getByTestId('zoom-out'));
      expect(onZoomOut).toHaveBeenCalledOnce();
    });

    it('zoom-reset restores 1x scale with no pan', async () => {
      projectStore.applyView(makeView());
      viewportStore.zoomInAt(10, 10);
      viewportStore.panBy(50, 50);
      renderBar();

      await fireEvent.click(screen.getByTestId('zoom-reset'));
      expect(viewportStore.scale).toBe(1);
      expect(viewportStore.panX).toBe(0);
      expect(viewportStore.panY).toBe(0);
    });
  });
});
