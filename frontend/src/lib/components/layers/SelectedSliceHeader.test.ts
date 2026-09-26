import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';

vi.mock('../../download', () => ({ triggerDownload: vi.fn() }));
import { triggerDownload } from '../../download';

import SelectedSliceHeader from './SelectedSliceHeader.svelte';
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

describe('SelectedSliceHeader', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    uiStore.reset();
    vi.mocked(triggerDownload).mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders nothing without a selected slice', () => {
    projectStore.applyView(
      makeView({ slices: [makeSlice(0, 85)], selectedSlice: null }),
    );
    render(SelectedSliceHeader);
    expect(screen.queryByTestId('selected-slice-header')).not.toBeInTheDocument();
  });

  it('renders the selected slice once one is selected', () => {
    projectStore.applyView(
      makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)], selectedSlice: 1 }),
    );
    render(SelectedSliceHeader);
    const header = screen.getByTestId('selected-slice-header');
    expect(header).toHaveTextContent('image_slice_1');
  });

  it('Download triggers a real <a download> click against the raw-slice endpoint, not window.open', async () => {
    projectStore.applyView(
      makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)], selectedSlice: 1 }),
    );
    render(SelectedSliceHeader);

    await fireEvent.click(screen.getByTestId('slice-download'));

    expect(triggerDownload).toHaveBeenCalledWith(
      '/api/v1/projects/appstate-test/slices/1/download',
      'image_slice_1.png',
    );
  });

  describe('replace image', () => {
    it('clicking Replace opens the hidden file chooser', async () => {
      projectStore.applyView(
        makeView({ slices: [makeSlice(0, 85)], selectedSlice: 0 }),
      );
      render(SelectedSliceHeader);

      const input = screen.getByTestId('slice-upload-input') as HTMLInputElement;
      const clickSpy = vi.spyOn(input, 'click');
      await fireEvent.click(screen.getByTestId('slice-replace'));

      expect(clickSpy).toHaveBeenCalled();
    });

    it('uploads the chosen file to the selected slice index', async () => {
      projectStore.applyView(
        makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)], selectedSlice: 1 }),
      );
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { ...makeView({ revision: 2 }), changed: true }));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      render(SelectedSliceHeader);
      const input = screen.getByTestId('slice-upload-input') as HTMLInputElement;
      const file = new File(['bytes'], 'replacement.png', { type: 'image/png' });
      await fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/slices/1/image');
      expect(init.method).toBe('PUT');
      expect((init.body as FormData).get('image')).toBe(file);
    });
  });
});
