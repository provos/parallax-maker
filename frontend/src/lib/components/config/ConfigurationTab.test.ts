import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import ConfigurationTab from './ConfigurationTab.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
import { uiStore } from '../../state/ui.svelte';
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
      model: 'automatic1111',
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

describe('ConfigurationTab', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    uiStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('shows the Automatic1111/ComfyUI server panel (and its workflow upload) only for those models', () => {
    projectStore.applyView(makeView({ inpainting: { ...makeView().inpainting, model: 'automatic1111' } }));
    render(ConfigurationTab);
    expect(screen.getByTestId('external-server-panel')).toBeInTheDocument();
    expect(screen.queryByTestId('comfyui-workflow-panel')).not.toBeInTheDocument();
    expect(screen.queryByTestId('api-key-panel')).not.toBeInTheDocument();
  });

  it('shows the ComfyUI workflow upload only for comfyui, and the API key panel only for stabilityai/fal.ai', () => {
    projectStore.applyView(makeView({ inpainting: { ...makeView().inpainting, model: 'comfyui' } }));
    const { unmount } = render(ConfigurationTab);
    expect(screen.getByTestId('external-server-panel')).toBeInTheDocument();
    expect(screen.getByTestId('comfyui-workflow-panel')).toBeInTheDocument();
    unmount();

    projectStore.applyView(makeView({ revision: 2, inpainting: { ...makeView().inpainting, model: 'stabilityai' } }));
    render(ConfigurationTab);
    expect(screen.queryByTestId('external-server-panel')).not.toBeInTheDocument();
    expect(screen.getByTestId('api-key-panel')).toBeInTheDocument();
  });

  it('Test Connection highlights success and resets to none when the server address is edited again', async () => {
    const projectId = 'appstate-test';
    projectStore.applyView(makeView());

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === '/api/v1/config/probe-server' && method === 'POST') {
        expect(JSON.parse(init!.body as string)).toEqual({ model: 'automatic1111', serverAddress: 'localhost:7860' });
        return jsonResponse(200, { ok: true, message: 'Connection to automatic1111 successful: []' });
      }
      if (url === `/api/v1/projects/${projectId}/inpainting/settings` && method === 'PUT') {
        return jsonResponse(200, { ...makeView({ revision: 2 }), changed: true });
      }
      if (url.startsWith(`/api/v1/projects/${projectId}/logs`)) {
        return jsonResponse(200, { entries: [], next: 0 });
      }
      throw new Error(`Unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(ConfigurationTab);
    expect(screen.getByTestId('external-server-address')).toHaveAttribute('data-status', 'none');

    await fireEvent.click(screen.getByTestId('external-test-connection'));
    await waitFor(() => expect(screen.getByTestId('external-server-address')).toHaveAttribute('data-status', 'success'));

    await fireEvent.change(screen.getByTestId('external-server-address'), { target: { value: 'localhost:9999' } });
    await waitFor(() => expect(screen.getByTestId('external-server-address')).toHaveAttribute('data-status', 'none'));
  });

  it('Test Connection highlights failure when the probe fails', async () => {
    projectStore.applyView(makeView());
    const fetchMock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      if (url === '/api/v1/config/probe-server') {
        return jsonResponse(200, { ok: false, message: 'Connection to automatic1111 failed' });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(ConfigurationTab);
    await fireEvent.click(screen.getByTestId('external-test-connection'));
    await waitFor(() => expect(screen.getByTestId('external-server-address')).toHaveAttribute('data-status', 'failure'));
  });

  it('the API key field never displays a previously-saved key back', () => {
    projectStore.applyView(makeView({ inpainting: { ...makeView().inpainting, model: 'stabilityai' } }));
    render(ConfigurationTab);
    expect(screen.getByTestId('api-key')).toHaveValue('');
  });

  it('Save State calls POST /save and does not trigger any browser download', async () => {
    const projectId = 'appstate-test';
    projectStore.applyView(makeView());
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url === `/api/v1/projects/${projectId}/save` && method === 'POST') {
        return jsonResponse(200, makeView({ revision: 2 }));
      }
      if (url.startsWith(`/api/v1/projects/${projectId}/logs`)) {
        return jsonResponse(200, { entries: [{ seq: 1, level: 'info', message: `Saved state to ${projectId}` }], next: 1 });
      }
      throw new Error(`Unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(ConfigurationTab);
    await fireEvent.click(screen.getByTestId('save-state'));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(`/api/v1/projects/${projectId}/save`, expect.objectContaining({ method: 'POST' })),
    );
  });
});
