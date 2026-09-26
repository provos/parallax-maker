import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';

vi.mock('../../download', () => ({ triggerDownload: vi.fn() }));
import { triggerDownload } from '../../download';

import ExportTab from './ExportDialog.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
import { cameraDraftStore } from '../../state/cameraDraft.svelte';
import { uiStore } from '../../state/ui.svelte';
import type { ProjectView, SliceView } from '../../api/types';

function makeSlice(index: number, overrides: Partial<SliceView> = {}): SliceView {
  return {
    index,
    depth: index * 50,
    version: 1,
    canUndo: false,
    canRedo: false,
    positivePrompt: '',
    negativePrompt: '',
    isGround: false,
    image: { url: `/api/v1/projects/appstate-test/assets/slice-${index}` },
    thumbnail: { url: `/api/v1/projects/appstate-test/assets/slice-${index}-thumb` },
    ...overrides,
  } as SliceView;
}

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

describe('ExportTab', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    cameraDraftStore.reset();
    uiStore.reset();
    vi.mocked(triggerDownload).mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    cameraDraftStore.reset();
    uiStore.reset();
  });

  describe('tab switching', () => {
    it('starts on the 3D scene tab, with the animation pane hidden', () => {
      projectStore.applyView(makeView());
      render(ExportTab);

      expect(screen.getByTestId('export-tab-gltf')).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByTestId('export-tab-animation')).toHaveAttribute('aria-selected', 'false');
      expect(screen.getByTestId('tab-export')).not.toHaveAttribute('hidden');
      expect(screen.getByTestId('animation-export').closest('[role="tabpanel"]')).toHaveAttribute('hidden');
    });

    it('switches panes when the Animation tab is clicked, and back', async () => {
      projectStore.applyView(makeView());
      render(ExportTab);

      await fireEvent.click(screen.getByTestId('export-tab-animation'));
      expect(screen.getByTestId('export-tab-animation')).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByTestId('export-tab-gltf')).toHaveAttribute('aria-selected', 'false');
      expect(screen.getByTestId('tab-export')).toHaveAttribute('hidden');
      expect(screen.getByTestId('animation-export').closest('[role="tabpanel"]')).not.toHaveAttribute('hidden');

      await fireEvent.click(screen.getByTestId('export-tab-gltf'));
      expect(screen.getByTestId('tab-export')).not.toHaveAttribute('hidden');
      expect(screen.getByTestId('animation-export').closest('[role="tabpanel"]')).toHaveAttribute('hidden');
    });
  });

  describe('mirrored camera sliders', () => {
    it('reads distance, max distance and focal length straight from the shared camera draft', () => {
      projectStore.applyView(makeView());
      render(ExportTab);

      expect(screen.getByTestId('export-camera-distance')).toHaveValue('125');
      expect(screen.getByTestId('export-max-distance')).toHaveValue('140');
      expect(screen.getByTestId('export-focal-length')).toHaveValue('475');
    });

    it('committing the camera distance here sends all camera fields, same as the Preview panel', async () => {
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

      render(ExportTab);
      const distance = screen.getByTestId('export-camera-distance');
      await fireEvent.input(distance, { target: { value: '77' } });
      await fireEvent.change(distance);

      await waitFor(() =>
        expect(fetchMock).toHaveBeenCalledWith(
          `/api/v1/projects/${projectId}/settings`,
          expect.objectContaining({ method: 'PUT' }),
        ),
      );
      const call = fetchMock.mock.calls.find(([reqUrl]) => String(reqUrl) === `/api/v1/projects/${projectId}/settings`);
      expect(JSON.parse(call![1]!.body as string)).toEqual({
        camera: { distance: 77, maxDistance: 140, focalLength: 475, groundNear: 0 },
        meshDisplacement: 15,
      });
      // The draft is shared, so the slider itself now reflects the commit.
      expect(cameraDraftStore.draft.distance).toBe(77);
    });
  });

  describe('ground summary', () => {
    it('shows "No ground plane" without a ground slice', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0)] }));
      render(ExportTab);
      expect(screen.getByTestId('export-ground-summary')).toHaveTextContent('No ground plane');
    });

    it('names the ground slice and its distance once one is set', () => {
      projectStore.applyView(
        makeView({
          slices: [makeSlice(0), makeSlice(1, { isGround: true })],
          settings: {
            darkMode: false,
            camera: { distance: 125, focalLength: 475, maxDistance: 140, groundNear: 42 },
            meshDisplacement: 15,
            depthModel: 'dinov2',
          },
        }),
      );
      render(ExportTab);
      const summary = screen.getByTestId('export-ground-summary');
      expect(summary).toHaveTextContent('image_slice_1');
      expect(summary).toHaveTextContent('42');
    });

    it('Edit moves to the Ground step and closes the dialog', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0), makeSlice(1, { isGround: true })] }));
      uiStore.setStep('slices');
      uiStore.openDialog('export');
      render(ExportTab);

      await fireEvent.click(screen.getByTestId('export-edit-ground'));

      expect(uiStore.step).toBe('ground');
      expect(uiStore.dialog).toBeNull();
    });
  });

  describe('View in 3D', () => {
    it('is absent until the scene has been exported', () => {
      projectStore.applyView(makeView());
      render(ExportTab);
      expect(screen.queryByTestId('export-view-3d')).not.toBeInTheDocument();
    });

    it('sets the view to 3D and closes the dialog once the scene is ready', async () => {
      projectStore.applyView(
        makeView({ exports: { gltf: { url: '/api/v1/projects/appstate-test/export/gltf?v=1' }, upscaled: false } }),
      );
      uiStore.openDialog('export');
      render(ExportTab);

      await fireEvent.click(screen.getByTestId('export-view-3d'));

      expect(uiStore.view).toBe('3d');
      expect(uiStore.dialog).toBeNull();
    });
  });

  it('reads the displacement slider straight from the persisted settings', () => {
    projectStore.applyView(makeView());
    render(ExportTab);

    expect(screen.getByTestId('displacement')).toHaveValue('15');
  });

  it('commits all camera fields together with the new displacement on slider release', async () => {
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

    render(ExportTab);
    const displacement = screen.getByTestId('displacement');
    await fireEvent.input(displacement, { target: { value: '80' } });
    await fireEvent.change(displacement);

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        `/api/v1/projects/${projectId}/settings`,
        expect.objectContaining({ method: 'PUT' }),
      ),
    );
    const call = fetchMock.mock.calls.find(([reqUrl]) => String(reqUrl) === `/api/v1/projects/${projectId}/settings`);
    expect(JSON.parse(call![1]!.body as string)).toEqual({
      camera: { distance: 125, maxDistance: 140, focalLength: 475, groundNear: 0 },
      meshDisplacement: 80,
    });
  });

  it('toggles the DOF checkbox locally without persisting anything', async () => {
    projectStore.applyView(makeView());
    render(ExportTab);
    const checkbox = screen.getByTestId('toggle-dof') as HTMLInputElement;
    expect(checkbox.checked).toBe(false);
    await fireEvent.click(checkbox);
    expect(checkbox.checked).toBe(true);
  });

  it('Export glTF Scene starts the export job and then triggers a real <a download> click, not window.open', async () => {
    const projectId = 'appstate-test';
    projectStore.applyView(makeView());

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === `/api/v1/projects/${projectId}/export/gltf` && method === 'POST') {
        return jsonResponse(202, { job: { id: 'job-1', kind: 'export-gltf', status: 'queued', progress: 0 } });
      }
      if (url === '/api/v1/jobs/job-1') {
        return jsonResponse(200, {
          id: 'job-1',
          kind: 'export-gltf',
          status: 'succeeded',
          progress: 1,
          project: makeView({
            revision: 2,
            exports: { gltf: { url: `/api/v1/projects/${projectId}/export/gltf?v=1` }, upscaled: false },
          }),
        });
      }
      if (url.startsWith(`/api/v1/projects/${projectId}/logs`)) {
        return jsonResponse(200, { entries: [], next: 0 });
      }
      throw new Error(`Unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(ExportTab);
    await fireEvent.click(screen.getByTestId('gltf-export'));

    await waitFor(() =>
      expect(triggerDownload).toHaveBeenCalledWith(`/api/v1/projects/${projectId}/export/gltf?v=1`, 'scene.gltf'),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/v1/projects/${projectId}/export/gltf`,
      expect.objectContaining({ method: 'POST' }),
    );
  });

  it('Export Animation never triggers a download (matches Dash: no dcc.Download fires)', async () => {
    const projectId = 'appstate-test';
    projectStore.applyView(makeView());

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === `/api/v1/projects/${projectId}/export/animation` && method === 'POST') {
        expect(JSON.parse(init!.body as string)).toEqual({ frames: 100 });
        return jsonResponse(202, { job: { id: 'job-2', kind: 'animation', status: 'queued', progress: 0 } });
      }
      if (url === '/api/v1/jobs/job-2') {
        return jsonResponse(200, {
          id: 'job-2',
          kind: 'animation',
          status: 'succeeded',
          progress: 1,
          project: makeView({ revision: 2 }),
        });
      }
      if (url.startsWith(`/api/v1/projects/${projectId}/logs`)) {
        return jsonResponse(200, { entries: [{ seq: 1, level: 'info', message: 'Exported 100 frames to animation' }], next: 1 });
      }
      throw new Error(`Unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(ExportTab);
    await fireEvent.click(screen.getByTestId('animation-export'));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        `/api/v1/projects/${projectId}/export/animation`,
        expect.objectContaining({ method: 'POST' }),
      ),
    );
    expect(triggerDownload).not.toHaveBeenCalled();
  });
});
