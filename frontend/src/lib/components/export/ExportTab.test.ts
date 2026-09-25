import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';

vi.mock('../../download', () => ({ triggerDownload: vi.fn() }));
import { triggerDownload } from '../../download';

import ExportTab from './ExportTab.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
import type { ProjectView } from '../../api/types';

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
    vi.mocked(triggerDownload).mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reads the camera/displacement sliders straight from the persisted settings', () => {
    projectStore.applyView(makeView());
    render(ExportTab);

    expect(screen.getByTestId('camera-distance')).toHaveValue('125');
    expect(screen.getByTestId('max-distance')).toHaveValue('140');
    expect(screen.getByTestId('focal-length')).toHaveValue('475');
    expect(screen.getByTestId('displacement')).toHaveValue('15');
  });

  it('commits all four camera/displacement fields together on slider release', async () => {
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
      camera: { distance: 200, maxDistance: 140, focalLength: 475 },
      meshDisplacement: 15,
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
