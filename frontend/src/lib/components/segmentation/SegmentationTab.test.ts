import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import SegmentationTab from './SegmentationTab.svelte';
import { projectStore } from '../../state/project.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { logStore } from '../../state/logs.svelte';
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
      camera: { distance: 100, focalLength: 100, maxDistance: 200 },
      meshDisplacement: 0,
      depthModel: "dinov2",
    },
    exports: { gltf: null, upscaled: false },
    ...overrides,
  };
}

describe('SegmentationTab', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders one thumbnail per slice', () => {
    projectStore.applyView(
      makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170), makeSlice(2, 255)] }),
    );
    render(SegmentationTab);

    const thumbnails = screen.getAllByTestId('slice-thumbnail');
    expect(thumbnails).toHaveLength(3);
    expect(thumbnails.map((img) => img.getAttribute('alt'))).toEqual([
      'image_slice_0',
      'image_slice_1',
      'image_slice_2',
    ]);
  });

  it('disables the Generate button while a job is in flight, and re-enables it after', async () => {
    projectStore.applyView(makeView());
    render(SegmentationTab);

    expect(screen.getByTestId('generate-slices')).toBeEnabled();

    jobStore.begin('slices');
    await waitFor(() => expect(screen.getByTestId('generate-slices')).toBeDisabled());

    jobStore.end();
    await waitFor(() => expect(screen.getByTestId('generate-slices')).toBeEnabled());
  });

  it('sends baseRevision on threshold change, and retries once on stale_revision', async () => {
    projectStore.applyView(makeView());

    const fetchMock = vi.fn();
    fetchMock.mockResolvedValueOnce(
      jsonResponse(409, { error: { code: 'stale_revision', message: 'stale' } }),
    );
    fetchMock.mockResolvedValueOnce(jsonResponse(200, makeView({ revision: 2 })));
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { ...makeView({ revision: 3 }), changed: true }));
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
    vi.stubGlobal('fetch', fetchMock);

    render(SegmentationTab);
    const [firstHandle] = screen.getAllByTestId('threshold-handle');
    await fireEvent.input(firstHandle, { target: { value: '90' } });
    await fireEvent.change(firstHandle, { target: { value: '90' } });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4));

    const [firstUrl, firstInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(firstUrl).toBe('/api/v1/projects/appstate-test/thresholds');
    expect(firstInit.method).toBe('PUT');
    expect(JSON.parse(firstInit.body as string)).toMatchObject({ baseRevision: 1, values: [90, 170] });

    const [secondUrl] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(secondUrl).toBe('/api/v1/projects/appstate-test');

    const [thirdUrl, thirdInit] = fetchMock.mock.calls[2] as [string, RequestInit];
    expect(thirdUrl).toBe('/api/v1/projects/appstate-test/thresholds');
    expect(JSON.parse(thirdInit.body as string)).toMatchObject({ baseRevision: 2, values: [90, 170] });
  });

  describe('slice selection', () => {
    it('sends {slice: index} on the first click, then {slice: null} to deselect the same slice', async () => {
      const view = makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170), makeSlice(2, 255)] });
      projectStore.applyView(view);

      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(
        jsonResponse(200, { ...view, revision: 2, selectedSlice: 1, changed: true }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      render(SegmentationTab);
      const wrappers = screen.getAllByTestId('slice-thumbnail-wrapper');
      await fireEvent.click(wrappers[1]);

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      const [firstUrl, firstInit] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(firstUrl).toBe('/api/v1/projects/appstate-test/selection');
      expect(firstInit.method).toBe('PUT');
      expect(JSON.parse(firstInit.body as string)).toEqual({ slice: 1 });

      // The applied view now reports slice 1 as selected; clicking it again
      // must send `{slice: null}` (Dash's click-to-toggle), not `{slice: 1}`.
      fetchMock.mockResolvedValueOnce(
        jsonResponse(200, { ...view, revision: 3, selectedSlice: null, changed: true }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      await fireEvent.click(wrappers[1]);

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4));
      const [secondUrl, secondInit] = fetchMock.mock.calls[2] as [string, RequestInit];
      expect(secondUrl).toBe('/api/v1/projects/appstate-test/selection');
      expect(JSON.parse(secondInit.body as string)).toEqual({ slice: null });
    });

    it('marks the selected thumbnail wrapper with aria-selected and data-selected', () => {
      const view = makeView({
        slices: [makeSlice(0, 85), makeSlice(1, 170)],
        selectedSlice: 1,
      });
      projectStore.applyView(view);
      render(SegmentationTab);

      const wrappers = screen.getAllByTestId('slice-thumbnail-wrapper');
      expect(wrappers[0]).toHaveAttribute('aria-selected', 'false');
      expect(wrappers[0]).toHaveAttribute('data-selected', 'false');
      expect(wrappers[1]).toHaveAttribute('aria-selected', 'true');
      expect(wrappers[1]).toHaveAttribute('data-selected', 'true');
    });
  });

  describe('Actions panel enablement', () => {
    // Matches Dash exactly: webui.py never disables Create/Delete/Add/Remove/
    // Copy/Paste/Balance based on selection/mask/clipboard state -- only "is
    // a project loaded" and "is nothing else in flight" gate them, same as
    // Generate. Preconditions are enforced by workflow.ts at click time.
    const actionTestIds = [
      'balance-slices',
      'create-slice',
      'delete-slice',
      'add-mask-to-slice',
      'remove-mask-from-slice',
      'copy-slice',
      'paste-slice',
    ];

    it('disables every action button when there is no project', () => {
      render(SegmentationTab);
      for (const testId of actionTestIds) {
        expect(screen.getByTestId(testId)).toBeDisabled();
      }
    });

    it('enables every action button with a project loaded, regardless of selection or mask state', () => {
      projectStore.applyView(
        makeView({ selectedSlice: null, segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false } }),
      );
      render(SegmentationTab);
      for (const testId of actionTestIds) {
        expect(screen.getByTestId(testId)).toBeEnabled();
      }
    });

    it('disables every action button while a job is in flight', async () => {
      projectStore.applyView(makeView());
      render(SegmentationTab);
      jobStore.begin('slice-editing');
      await waitFor(() => {
        for (const testId of actionTestIds) {
          expect(screen.getByTestId(testId)).toBeDisabled();
        }
      });
    });
  });

  describe('depth badge editing', () => {
    it('reveals a numeric input on click and commits the new depth on Enter', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)] }));
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(
        jsonResponse(200, { ...makeView({ revision: 2, slices: [makeSlice(0, 200), makeSlice(1, 170)] }), changed: true }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      render(SegmentationTab);
      const badges = screen.getAllByTestId('slice-depth-display');
      expect(screen.queryByTestId('slice-depth-input')).toBeNull();

      await fireEvent.click(badges[0]);
      const input = screen.getByTestId('slice-depth-input') as HTMLInputElement;
      expect(input).toBeInTheDocument();

      await fireEvent.input(input, { target: { value: '200' } });
      await fireEvent.keyDown(input, { key: 'Enter' });
      await fireEvent.blur(input);

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/slices/0/depth');
      expect(JSON.parse(init.body as string)).toEqual({ depth: 200 });
      // Enter/blur is a single commit, not two.
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });

    it.each(['', '  ', '12.5', 'abc'])('does not commit a blank or non-integer depth (%j)', async (draft) => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)] }));
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      render(SegmentationTab);
      await fireEvent.click(screen.getAllByTestId('slice-depth-display')[0]);
      const input = screen.getByTestId('slice-depth-input') as HTMLInputElement;
      await fireEvent.input(input, { target: { value: draft } });
      await fireEvent.keyDown(input, { key: 'Enter' });
      await fireEvent.blur(input);

      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('clicking a depth badge does not also (de)select its slice', async () => {
      const view = makeView({ slices: [makeSlice(0, 85)], selectedSlice: null });
      projectStore.applyView(view);
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);

      render(SegmentationTab);
      await fireEvent.click(screen.getByTestId('slice-depth-display'));

      expect(fetchMock).not.toHaveBeenCalled();
    });
  });

  describe('undo/redo carets', () => {
    it('disables undo/redo per slice according to canUndo/canRedo', () => {
      projectStore.applyView(
        makeView({
          slices: [
            { ...makeSlice(0, 85), canUndo: false, canRedo: true },
            { ...makeSlice(1, 170), canUndo: true, canRedo: false },
          ],
        }),
      );
      render(SegmentationTab);

      const undoButtons = screen.getAllByTestId('slice-undo');
      const redoButtons = screen.getAllByTestId('slice-redo');
      expect(undoButtons[0]).toBeDisabled();
      expect(redoButtons[0]).toBeEnabled();
      expect(undoButtons[1]).toBeEnabled();
      expect(redoButtons[1]).toBeDisabled();
    });

    it('clicking Undo calls the per-index undo endpoint without selecting the slice', async () => {
      projectStore.applyView(
        makeView({ slices: [{ ...makeSlice(0, 85), canUndo: true }], selectedSlice: null }),
      );
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { ...makeView({ revision: 2 }), changed: true }));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      render(SegmentationTab);
      await fireEvent.click(screen.getByTestId('slice-undo'));

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/slices/0/undo');
      expect(init.method).toBe('POST');
    });
  });

  describe('per-thumbnail image upload', () => {
    it('uploads the chosen file to the matching slice index', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)] }));
      const fetchMock = vi.fn();
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { ...makeView({ revision: 2 }), changed: true }));
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      vi.stubGlobal('fetch', fetchMock);

      render(SegmentationTab);
      const inputs = document.querySelectorAll<HTMLInputElement>('[data-testid="slice-upload-input"]');
      expect(inputs).toHaveLength(2);
      const file = new File(['bytes'], 'replacement.png', { type: 'image/png' });
      await fireEvent.change(inputs[1], { target: { files: [file] } });

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/projects/appstate-test/slices/1/image');
      expect(init.method).toBe('PUT');
      expect((init.body as FormData).get('image')).toBe(file);
    });
  });
});
