import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import GroundPanel from './GroundPanel.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
import { cameraDraftStore } from '../../state/cameraDraft.svelte';
import type { ProjectView, SliceView } from '../../api/types';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function makeSlice(index: number, depth: number, overrides: Partial<SliceView> = {}): SliceView {
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
    ...overrides,
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

/** Same fixture the old ExportTab ground-plane tests used: a ground slice, a
 * horizon/pitch already computed, and a scene profile for the side view. */
function groundView(): ProjectView {
  return makeView({
    slices: [makeSlice(0, 0), makeSlice(1, 60, { isGround: true })],
    settings: {
      darkMode: false,
      camera: { distance: 100, focalLength: 26, maxDistance: 500, pitch: 9.84, groundNear: 78, horizonRow: 516 },
      meshDisplacement: 0,
      depthModel: 'dinov2',
    },
    sceneProfile: {
      cameraZ: -100,
      pitch: 9.84,
      halfFov: 20,
      cards: [{ index: 0, z: 500, top: -300, bottom: 150 }],
      ground: { height: 60, nearZ: 78, farZ: 500, backdropTop: 40 },
    },
  });
}

describe('GroundPanel', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    cameraDraftStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    cameraDraftStore.reset();
  });

  it('shows the horizon, pitch and a side view of cards and ground', () => {
    projectStore.applyView(groundView());
    render(GroundPanel);
    expect(screen.getByTestId('horizon-readout')).toHaveTextContent('Horizon at row 516, camera pitch 9.8°');
    expect(screen.getByTestId('ground-distance')).toHaveValue('78');
    expect(screen.getByTestId('ground-distance')).toBeEnabled();
    expect(screen.getAllByTestId('side-view-card')).toHaveLength(1);
    expect(screen.getByTestId('side-view-ground')).toBeInTheDocument();
  });

  it('commits the ground distance with the camera settings', async () => {
    projectStore.applyView(groundView());
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      if (url === '/api/v1/projects/appstate-test/settings') return jsonResponse(200, { ...groundView(), changed: true });
      if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
      throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(GroundPanel);

    const slider = screen.getByTestId('ground-distance');
    await fireEvent.input(slider, { target: { value: '120' } });
    await fireEvent.change(slider);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const body = JSON.parse(fetchMock.mock.calls[0][1]!.body as string);
    expect(body.camera).toEqual({ distance: 100, maxDistance: 500, focalLength: 26, groundNear: 120 });
  });

  it('disables the ground distance without a ground slice', () => {
    projectStore.applyView(makeView());
    render(GroundPanel);
    expect(screen.getByTestId('ground-distance')).toBeDisabled();
  });

  describe('ground slice radios', () => {
    it('choosing a slice calls setGroundSlice, which PUTs isGround:true via api.setGroundPlane', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0), makeSlice(1, 60)] }));
      const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url === '/api/v1/projects/appstate-test/slices/1/ground') {
          expect(init?.method).toBe('PUT');
          expect(JSON.parse(init!.body as string)).toEqual({ isGround: true });
          return jsonResponse(200, {
            ...makeView({ slices: [makeSlice(0, 0), makeSlice(1, 60, { isGround: true })] }),
            changed: true,
          });
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
        throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
      });
      vi.stubGlobal('fetch', fetchMock);
      render(GroundPanel);

      const choices = screen.getAllByTestId('ground-choice');
      const forSliceOne = choices.find((el) => el.getAttribute('data-slice-index') === '1');
      expect(forSliceOne).toBeDefined();
      await fireEvent.click(forSliceOne!);

      await waitFor(() =>
        expect(fetchMock).toHaveBeenCalledWith(
          '/api/v1/projects/appstate-test/slices/1/ground',
          expect.objectContaining({ method: 'PUT' }),
        ),
      );
    });

    it('starts with "No ground plane" checked when there is no ground slice', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0), makeSlice(1, 60)] }));
      render(GroundPanel);
      expect(screen.getByTestId('ground-choice-none')).toBeChecked();
    });

    it('"No ground plane" unsets the current ground slice', async () => {
      projectStore.applyView(groundView());
      const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url === '/api/v1/projects/appstate-test/slices/1/ground') {
          expect(init?.method).toBe('PUT');
          expect(JSON.parse(init!.body as string)).toEqual({ isGround: false });
          return jsonResponse(200, {
            ...makeView({ slices: [makeSlice(0, 0), makeSlice(1, 60)] }),
            changed: true,
          });
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
        throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
      });
      vi.stubGlobal('fetch', fetchMock);
      render(GroundPanel);

      expect(screen.getByTestId('ground-choice-none')).not.toBeChecked();
      await fireEvent.click(screen.getByTestId('ground-choice-none'));

      await waitFor(() =>
        expect(fetchMock).toHaveBeenCalledWith(
          '/api/v1/projects/appstate-test/slices/1/ground',
          expect.objectContaining({ method: 'PUT' }),
        ),
      );
    });

    it('"No ground plane" is a no-op when there is already no ground slice', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0), makeSlice(1, 60)] }));
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);
      render(GroundPanel);

      await fireEvent.click(screen.getByTestId('ground-choice-none'));

      expect(fetchMock).not.toHaveBeenCalled();
    });
  });

  describe('Fit ground', () => {
    it('disables Fit ground until a ground slice exists', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)] }));
      render(GroundPanel);
      expect(screen.getByTestId('ground-fit')).toBeDisabled();
    });

    it('enables Fit ground once a ground slice exists, and POSTs on click', async () => {
      const ground = makeSlice(1, 170, { isGround: true });
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85), ground] }));
      const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url === '/api/v1/projects/appstate-test/ground/fit') {
          return jsonResponse(200, { ...makeView({ slices: [makeSlice(0, 85), ground] }), changed: true });
        }
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
        throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
      });
      vi.stubGlobal('fetch', fetchMock);
      render(GroundPanel);

      expect(screen.getByTestId('ground-fit')).toBeEnabled();
      await fireEvent.click(screen.getByTestId('ground-fit'));

      await waitFor(() =>
        expect(fetchMock).toHaveBeenCalledWith(
          '/api/v1/projects/appstate-test/ground/fit',
          expect.objectContaining({ method: 'POST' }),
        ),
      );
    });
  });
});
