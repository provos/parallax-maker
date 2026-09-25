import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import App from './App.svelte';
import { projectStore } from './lib/state/project.svelte';
import { jobStore } from './lib/state/jobs.svelte';
import { logStore } from './lib/state/logs.svelte';
import { uiStore } from './lib/state/ui.svelte';
import type { ProjectView } from './lib/api/types';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function makeView(overrides: Partial<ProjectView> = {}): ProjectView {
  return {
    id: 'appstate-e2e-test',
    revision: 1,
    image: { width: 320, height: 240 },
    assets: { input: null, depth: null },
    depthModel: 'dinov2',
    numSlices: 3,
    thresholds: [],
    slices: [],
    selectedSlice: null,
    busy: null,
    ...overrides,
  };
}

describe('App', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    uiStore.reset();
  });

  it('renders the header, viewer and workflow tabs, and footer', () => {
    render(App);

    expect(screen.getByTestId('app-title')).toHaveTextContent('Parallax Maker');
    expect(screen.getByTestId('theme-toggle')).toBeInTheDocument();
    expect(screen.getByTestId('input-image-panel')).toBeInTheDocument();
    expect(screen.getByTestId('main-image')).toBeInTheDocument();
    expect(screen.getByTestId('depth-image')).toBeInTheDocument();
    expect(screen.getByTestId('log')).toBeInTheDocument();
    expect(screen.getByTestId('app-footer')).toHaveTextContent('2024 Niels Provos');
  });

  it('uploading an image creates a project, starts a depth job, and renders both images', async () => {
    const projectId = 'appstate-e2e-test';

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input);
      const method = init?.method ?? 'GET';

      if (url === '/api/v1/projects' && method === 'POST') {
        return jsonResponse(
          201,
          makeView({ assets: { input: { url: '/api/v1/projects/appstate-e2e-test/assets/input' }, depth: null } }),
        );
      }
      if (url.startsWith(`/api/v1/projects/${projectId}/logs`)) {
        return jsonResponse(200, { entries: [], next: 0 });
      }
      if (url === `/api/v1/projects/${projectId}/depth` && method === 'POST') {
        return jsonResponse(202, { job: { id: 'job-1', kind: 'depth', status: 'queued', progress: 0 } });
      }
      if (url === '/api/v1/jobs/job-1') {
        return jsonResponse(200, {
          id: 'job-1',
          kind: 'depth',
          status: 'succeeded',
          progress: 1,
          project: makeView({
            assets: {
              input: { url: '/api/v1/projects/appstate-e2e-test/assets/input' },
              depth: { url: '/api/v1/projects/appstate-e2e-test/assets/depth' },
            },
            thresholds: [0, 85, 170, 255],
          }),
        });
      }
      throw new Error(`Unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(App);

    const file = new File(['fake-png-bytes'], 'input.png', { type: 'image/png' });
    const input = screen.getByTestId('upload-image-input') as HTMLInputElement;
    await fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() =>
      expect(screen.getByTestId('main-image')).toHaveAttribute(
        'src',
        '/api/v1/projects/appstate-e2e-test/assets/input',
      ),
    );
    await waitFor(() =>
      expect(screen.getByTestId('depth-image')).toHaveAttribute(
        'src',
        '/api/v1/projects/appstate-e2e-test/assets/depth',
      ),
    );

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/projects', expect.objectContaining({ method: 'POST' }));
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/v1/projects/${projectId}/depth`,
      expect.objectContaining({ method: 'POST' }),
    );
  });
});
