import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import PreviewPanel from './PreviewPanel.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
import { uiStore } from '../../state/ui.svelte';
import { cameraDraftStore } from '../../state/cameraDraft.svelte';
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
      camera: { distance: 125, focalLength: 475, maxDistance: 140 },
      meshDisplacement: 15,
      depthModel: 'dinov2',
    },
    exports: { gltf: null, upscaled: false },
    ...overrides,
  };
}

describe('PreviewPanel', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    uiStore.reset();
    cameraDraftStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    cameraDraftStore.reset();
  });

  it('reads the camera sliders straight from the persisted settings', () => {
    projectStore.applyView(makeView());
    render(PreviewPanel);

    expect(screen.getByTestId('camera-distance')).toHaveValue('125');
    expect(screen.getByTestId('max-distance')).toHaveValue('140');
    expect(screen.getByTestId('focal-length')).toHaveValue('475');
  });

  it('commits all camera fields together on slider release', async () => {
    const projectId = 'appstate-test';
    projectStore.applyView(makeView());

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === `/api/v1/projects/${projectId}/settings` && method === 'PUT') {
        return jsonResponse(200, { ...makeView({ revision: 2 }), changed: true });
      }
      if (url.startsWith(`/api/v1/projects/${projectId}/logs`)) {
        return jsonResponse(200, { entries: [], next: 0 });
      }
      throw new Error(`Unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(PreviewPanel);
    const distance = screen.getByTestId('camera-distance');
    await fireEvent.input(distance, { target: { value: '200' } });
    await fireEvent.change(distance);

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        `/api/v1/projects/${projectId}/settings`,
        expect.objectContaining({ method: 'PUT' }),
      ),
    );
    const call = fetchMock.mock.calls.find(([reqUrl]) => String(reqUrl) === `/api/v1/projects/${projectId}/settings`);
    expect(JSON.parse(call![1]!.body as string)).toEqual({
      camera: { distance: 200, maxDistance: 140, focalLength: 475, groundNear: 0 },
      meshDisplacement: 15,
    });
  });

  it('focal length slider cannot reach 0 (the API requires focalLength > 0)', () => {
    projectStore.applyView(makeView());
    render(PreviewPanel);

    const focal = screen.getByTestId('focal-length');
    expect(focal).toHaveAttribute('min', '1');
    fireEvent.input(focal, { target: { value: '0' } });
    expect(focal).toHaveValue('1');
  });

  it('disables the camera sliders without a project', () => {
    render(PreviewPanel);
    expect(screen.getByTestId('camera-distance')).toBeDisabled();
    expect(screen.getByTestId('max-distance')).toBeDisabled();
    expect(screen.getByTestId('focal-length')).toBeDisabled();
  });

  describe('2D/3D radios', () => {
    it('disables both radios without any slices', () => {
      projectStore.applyView(makeView({ slices: [] }));
      render(PreviewPanel);
      expect(screen.getByTestId('preview-2d')).toBeDisabled();
      expect(screen.getByTestId('preview-3d')).toBeDisabled();
    });

    it('defaults to Parallax 2D checked', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85)] }));
      render(PreviewPanel);
      expect(screen.getByTestId('preview-2d')).toHaveAttribute('aria-checked', 'true');
      expect(screen.getByTestId('preview-3d')).toHaveAttribute('aria-checked', 'false');
    });

    it('clicking 3D scene calls uiStore.setView and flips the checked radio', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85)] }));
      render(PreviewPanel);

      await fireEvent.click(screen.getByTestId('preview-3d'));

      expect(uiStore.view).toBe('3d');
      expect(screen.getByTestId('preview-3d')).toHaveAttribute('aria-checked', 'true');
      expect(screen.getByTestId('preview-2d')).toHaveAttribute('aria-checked', 'false');
    });

    it('clicking Parallax 2D from 3D switches back', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85)] }));
      uiStore.setView('3d');
      render(PreviewPanel);
      expect(screen.getByTestId('preview-3d')).toHaveAttribute('aria-checked', 'true');

      await fireEvent.click(screen.getByTestId('preview-2d'));

      expect(uiStore.view).toBe('parallax');
      expect(screen.getByTestId('preview-2d')).toHaveAttribute('aria-checked', 'true');
    });
  });
});
