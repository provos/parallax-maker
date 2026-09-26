import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import AppShell from './AppShell.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
import { uiStore } from '../../state/ui.svelte';
import type { ProjectView, SliceView } from '../../api/types';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function slice(index: number, overrides: Partial<SliceView> = {}): SliceView {
  return {
    index,
    depth: index * 50,
    version: 0,
    canUndo: false,
    canRedo: false,
    positivePrompt: '',
    negativePrompt: '',
    isGround: false,
    image: { url: `/slice-${index}` },
    thumbnail: { url: `/thumb-${index}` },
    mask: null,
    ...overrides,
  } as SliceView;
}

function makeView(overrides: Partial<ProjectView> = {}): ProjectView {
  return {
    id: 'appstate-shell',
    revision: 1,
    image: { width: 640, height: 480 },
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
      darkMode: true,
      camera: { distance: 100, focalLength: 100, maxDistance: 200 },
      meshDisplacement: 0,
      depthModel: 'dinov2',
    },
    exports: { gltf: null, upscaled: false },
    ...overrides,
  } as ProjectView;
}

describe('app shell', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    uiStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('workflow stepper', () => {
    it('starts on Image with nothing done and the Mode panel showing', () => {
      render(AppShell);
      expect(screen.getByTestId('step-image')).toHaveAttribute('aria-current', 'step');
      expect(screen.getByTestId('step-depth')).toHaveAttribute('data-state', 'todo');
      expect(screen.getByTestId('inspector')).toHaveAttribute('data-panel', 'Mode');
    });

    it('marks steps done from the project and from this session', () => {
      projectStore.applyView(
        makeView({
          assets: { input: { url: '/input' }, depth: { url: '/depth' } },
          slices: [slice(0), slice(1, { isGround: true })],
        }),
      );
      render(AppShell);
      expect(screen.getByTestId('step-depth')).toHaveAttribute('data-state', 'done');
      expect(screen.getByTestId('step-slices')).toHaveAttribute('data-state', 'done');
      expect(screen.getByTestId('step-ground')).toHaveAttribute('data-state', 'done');
      expect(screen.getByTestId('step-inpaint')).toHaveAttribute('data-state', 'todo');
      expect(screen.getByTestId('step-export')).toHaveAttribute('data-state', 'todo');
    });

    it.each([
      ['step-depth', 'Mode'],
      ['step-slices', 'Segmentation'],
      ['step-inpaint', 'Inpainting'],
      ['step-ground', 'Ground'],
      ['step-preview', 'Preview'],
      ['step-export', 'Export'],
    ])('%s shows the %s panel', async (testId, panel) => {
      render(AppShell);
      await fireEvent.click(screen.getByTestId(testId));
      expect(screen.getByTestId(testId)).toHaveAttribute('aria-current', 'step');
      expect(screen.getByTestId('inspector')).toHaveAttribute('data-panel', panel);
    });

    it('visiting Preview counts as done once another step is current', async () => {
      render(AppShell);
      await fireEvent.click(screen.getByTestId('step-preview'));
      await fireEvent.click(screen.getByTestId('step-image'));
      expect(screen.getByTestId('step-preview')).toHaveAttribute('data-state', 'done');
    });
  });

  describe('header', () => {
    it('shows the project name once a project is loaded', () => {
      projectStore.applyView(makeView());
      render(AppShell);
      expect(screen.getByTestId('project-name')).toHaveTextContent('appstate-shell');
    });

    it('Settings toggles the Settings panel and back to the current step', async () => {
      render(AppShell);
      await fireEvent.click(screen.getByTestId('step-inpaint'));
      await fireEvent.click(screen.getByTestId('open-settings'));
      expect(screen.getByTestId('inspector')).toHaveAttribute('data-panel', 'Configuration');
      expect(screen.getByTestId('open-settings')).toHaveAttribute('aria-pressed', 'true');

      await fireEvent.click(screen.getByTestId('open-settings'));
      expect(screen.getByTestId('inspector')).toHaveAttribute('data-panel', 'Inpainting');
    });

    it('Export is disabled until there are slices, then opens the Export step', async () => {
      projectStore.applyView(makeView());
      const { unmount } = render(AppShell);
      expect(screen.getByTestId('open-export')).toBeDisabled();
      unmount();

      projectStore.applyView(makeView({ slices: [slice(0)] }));
      render(AppShell);
      await fireEvent.click(screen.getByTestId('open-export'));
      expect(screen.getByTestId('step-export')).toHaveAttribute('aria-current', 'step');
    });

    it('undo and redo act on the selected slice and follow its history', async () => {
      projectStore.applyView(
        makeView({ slices: [slice(0), slice(1, { canUndo: true })], selectedSlice: 1 }),
      );
      const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url === '/api/v1/projects/appstate-shell/slices/1/undo' && init?.method === 'POST') {
          return jsonResponse(200, { ...makeView({ slices: [slice(0), slice(1)] }), changed: true });
        }
        if (url.startsWith('/api/v1/projects/appstate-shell/logs')) return jsonResponse(200, { entries: [], next: 0 });
        throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
      });
      vi.stubGlobal('fetch', fetchMock);
      render(AppShell);

      expect(screen.getByTestId('header-redo')).toBeDisabled();
      await fireEvent.click(screen.getByTestId('header-undo'));
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/v1/projects/appstate-shell/slices/1/undo',
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });

  describe('status bar and log', () => {
    it('shows the latest message, the image size and the log count', () => {
      projectStore.applyView(makeView());
      logStore.pushClient('first', 'info');
      logStore.pushClient('Generation failed');
      render(AppShell);
      expect(screen.getByTestId('status-message')).toHaveTextContent('Generation failed');
      expect(screen.getByTestId('status-message')).toHaveClass('error');
      expect(screen.getByTestId('image-size')).toHaveTextContent('640 × 480');
      expect(screen.getByTestId('log-toggle')).toHaveTextContent('Log · 2');
    });

    it('the log drawer holds every entry and opens from the status bar', async () => {
      logStore.pushClient('one', 'info');
      logStore.pushClient('two', 'info');
      logStore.pushClient('three', 'info');
      logStore.pushClient('four', 'info');
      render(AppShell);
      expect(screen.getByTestId('log')).toHaveTextContent(/one.*two.*three.*four/);
      expect(screen.getByTestId('log-drawer')).toHaveClass('hidden');

      await fireEvent.click(screen.getByTestId('log-toggle'));
      expect(screen.getByTestId('log-drawer')).not.toHaveClass('hidden');
      expect(screen.getByTestId('log-toggle')).toHaveAttribute('aria-pressed', 'true');
    });
  });
});
