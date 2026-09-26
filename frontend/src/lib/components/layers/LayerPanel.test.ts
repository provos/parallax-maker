import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import LayerPanel from './LayerPanel.svelte';
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

/** A fetch stub that answers known URLs and throws on anything unexpected. */
function makeFetchMock(handlers: Record<string, (init?: RequestInit) => Response>) {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
    const handler = handlers[url];
    if (handler) return handler(init);
    throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
  });
}

describe('LayerPanel', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
    uiStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders one thumbnail per slice', () => {
    projectStore.applyView(
      makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170), makeSlice(2, 255)] }),
    );
    render(LayerPanel);

    const thumbnails = screen.getAllByTestId('slice-thumbnail');
    expect(thumbnails).toHaveLength(3);
    expect(thumbnails.map((img) => img.getAttribute('alt'))).toEqual(
      expect.arrayContaining(['image_slice_0', 'image_slice_1', 'image_slice_2']),
    );
  });

  it('shows the empty state with no slices', () => {
    projectStore.applyView(makeView({ slices: [] }));
    render(LayerPanel);

    expect(screen.getByTestId('layers-empty')).toBeInTheDocument();
    expect(screen.queryByTestId('slice-thumbnail-wrapper')).not.toBeInTheDocument();
  });

  it('orders rows nearest first (depth descending)', () => {
    projectStore.applyView(
      makeView({ slices: [makeSlice(0, 10), makeSlice(1, 250), makeSlice(2, 130)] }),
    );
    const { container } = render(LayerPanel);

    const wrappers = Array.from(container.querySelectorAll('[data-testid="slice-thumbnail-wrapper"]'));
    expect(wrappers.map((el) => el.getAttribute('data-slice-index'))).toEqual(['1', '2', '0']);
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

      render(LayerPanel);
      const wrapperFor1 = document.querySelector('[data-testid="slice-thumbnail-wrapper"][data-slice-index="1"]') as HTMLElement;
      await fireEvent.click(wrapperFor1);

      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
      const [firstUrl, firstInit] = fetchMock.mock.calls[0] as [string, RequestInit];
      expect(firstUrl).toBe('/api/v1/projects/appstate-test/selection');
      expect(firstInit.method).toBe('PUT');
      expect(JSON.parse(firstInit.body as string)).toEqual({ slice: 1 });

      // The applied view now reports slice 1 as selected; clicking it again
      // must send `{slice: null}` (click-to-toggle), not `{slice: 1}`.
      fetchMock.mockResolvedValueOnce(
        jsonResponse(200, { ...view, revision: 3, selectedSlice: null, changed: true }),
      );
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { entries: [], next: 0 }));
      await fireEvent.click(wrapperFor1);

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
      render(LayerPanel);

      const wrapper0 = document.querySelector('[data-testid="slice-thumbnail-wrapper"][data-slice-index="0"]') as HTMLElement;
      const wrapper1 = document.querySelector('[data-testid="slice-thumbnail-wrapper"][data-slice-index="1"]') as HTMLElement;
      expect(wrapper0).toHaveAttribute('aria-selected', 'false');
      expect(wrapper0).toHaveAttribute('data-selected', 'false');
      expect(wrapper1).toHaveAttribute('aria-selected', 'true');
      expect(wrapper1).toHaveAttribute('data-selected', 'true');
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

      render(LayerPanel);
      const wrapper0 = document.querySelector('[data-testid="slice-thumbnail-wrapper"][data-slice-index="0"]') as HTMLElement;
      const badge = wrapper0.querySelector('[data-testid="slice-depth-display"]') as HTMLElement;
      expect(screen.queryByTestId('slice-depth-input')).toBeNull();

      await fireEvent.click(badge);
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

      render(LayerPanel);
      const wrapper0 = document.querySelector('[data-testid="slice-thumbnail-wrapper"][data-slice-index="0"]') as HTMLElement;
      await fireEvent.click(wrapper0.querySelector('[data-testid="slice-depth-display"]') as HTMLElement);
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

      render(LayerPanel);
      await fireEvent.click(screen.getByTestId('slice-depth-display'));

      expect(fetchMock).not.toHaveBeenCalled();
      const wrapper0 = document.querySelector('[data-testid="slice-thumbnail-wrapper"][data-slice-index="0"]') as HTMLElement;
      expect(wrapper0).toHaveAttribute('data-selected', 'false');
    });
  });

  describe('ground badge', () => {
    it('shows GROUND only on the ground slice', () => {
      projectStore.applyView(
        makeView({
          slices: [makeSlice(0, 85), makeSlice(1, 170, { isGround: true })],
        }),
      );
      render(LayerPanel);

      const wrapper0 = document.querySelector('[data-testid="slice-thumbnail-wrapper"][data-slice-index="0"]') as HTMLElement;
      const wrapper1 = document.querySelector('[data-testid="slice-thumbnail-wrapper"][data-slice-index="1"]') as HTMLElement;
      expect(wrapper0.querySelector('[data-testid="ground-badge"]')).toBeNull();
      expect(wrapper1.querySelector('[data-testid="ground-badge"]')).not.toBeNull();
    });
  });

  describe('header actions (Copy/Paste/Delete)', () => {
    // Matches Dash exactly: these never disable themselves based on
    // selection/mask/clipboard state -- only "is a project loaded" and "is
    // nothing else in flight" gate them. Preconditions are enforced by
    // workflow.ts at click time.
    const actionTestIds = ['copy-slice', 'paste-slice', 'delete-slice'];

    it('disables every header action when there is no project', () => {
      render(LayerPanel);
      for (const testId of actionTestIds) {
        expect(screen.getByTestId(testId)).toBeDisabled();
      }
    });

    it('enables every header action with a project loaded, regardless of selection or mask state', () => {
      projectStore.applyView(
        makeView({ selectedSlice: null, segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false } }),
      );
      render(LayerPanel);
      for (const testId of actionTestIds) {
        expect(screen.getByTestId(testId)).toBeEnabled();
      }
    });

    it('disables every header action while a job is in flight', async () => {
      projectStore.applyView(makeView());
      render(LayerPanel);
      jobStore.begin('slice-editing');
      await waitFor(() => {
        for (const testId of actionTestIds) {
          expect(screen.getByTestId(testId)).toBeDisabled();
        }
      });
    });
  });

  describe('keyboard delete', () => {
    it('Delete on the focused list deletes the selected slice', async () => {
      projectStore.applyView(
        makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)], selectedSlice: 1 }),
      );
      const fetchMock = makeFetchMock({
        '/api/v1/projects/appstate-test/slices/1': () =>
          jsonResponse(200, { ...makeView({ slices: [makeSlice(0, 85)] }), changed: true }),
      });
      vi.stubGlobal('fetch', fetchMock);

      render(LayerPanel);
      const list = screen.getByRole('listbox');
      await fireEvent.keyDown(list, { key: 'Delete' });

      await waitFor(() =>
        expect(fetchMock).toHaveBeenCalledWith(
          '/api/v1/projects/appstate-test/slices/1',
          expect.objectContaining({ method: 'DELETE' }),
        ),
      );
    });
  });

  describe('depth ruler', () => {
    it('ArrowUp x3 then keyup commits setSliceDepth with depth+3', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)] }));
      const fetchMock = makeFetchMock({
        '/api/v1/projects/appstate-test/slices/1/depth': () =>
          jsonResponse(200, { ...makeView({ slices: [makeSlice(0, 85), makeSlice(1, 173)] }), changed: true }),
      });
      vi.stubGlobal('fetch', fetchMock);

      render(LayerPanel);
      const handle = document.querySelector('[data-testid="depth-handle"][data-slice-index="1"]') as HTMLElement;
      handle.focus();
      await fireEvent.keyDown(handle, { key: 'ArrowUp' });
      await fireEvent.keyDown(handle, { key: 'ArrowUp' });
      await fireEvent.keyDown(handle, { key: 'ArrowUp' });
      expect(fetchMock).not.toHaveBeenCalled();
      await fireEvent.keyUp(handle, { key: 'ArrowUp' });

      await waitFor(() =>
        expect(fetchMock).toHaveBeenCalledWith(
          '/api/v1/projects/appstate-test/slices/1/depth',
          expect.objectContaining({ method: 'PUT' }),
        ),
      );
      const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/slices/1/depth'));
      expect(JSON.parse(call![1]!.body as string)).toEqual({ depth: 173 });
    });

    it('Shift+ArrowDown moves the depth by 10', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)] }));
      const fetchMock = makeFetchMock({
        '/api/v1/projects/appstate-test/slices/1/depth': () =>
          jsonResponse(200, { ...makeView({ slices: [makeSlice(0, 85), makeSlice(1, 160)] }), changed: true }),
      });
      vi.stubGlobal('fetch', fetchMock);

      render(LayerPanel);
      const handle = document.querySelector('[data-testid="depth-handle"][data-slice-index="1"]') as HTMLElement;
      handle.focus();
      await fireEvent.keyDown(handle, { key: 'ArrowDown', shiftKey: true });
      await fireEvent.keyUp(handle, { key: 'ArrowDown', shiftKey: true });

      await waitFor(() =>
        expect(fetchMock).toHaveBeenCalledWith(
          '/api/v1/projects/appstate-test/slices/1/depth',
          expect.objectContaining({ method: 'PUT' }),
        ),
      );
      const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/slices/1/depth'));
      expect(JSON.parse(call![1]!.body as string)).toEqual({ depth: 160 });
    });

    it('pointer drag commits the depth under the pointer', async () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 85), makeSlice(1, 170)] }));
      const fetchMock = makeFetchMock({
        '/api/v1/projects/appstate-test/slices/1/depth': () =>
          jsonResponse(200, { ...makeView({ slices: [makeSlice(0, 85), makeSlice(1, 180)] }), changed: true }),
      });
      vi.stubGlobal('fetch', fetchMock);

      const { container } = render(LayerPanel);
      const axis = container.querySelector('.axis') as HTMLElement;
      // jsdom has no layout: give the ruler axis a fixed 0-255px box so
      // `depthAt(clientY)` maps 1:1 to a depth (255 - clientY).
      vi.spyOn(axis, 'getBoundingClientRect').mockReturnValue({
        left: 0,
        top: 0,
        width: 4,
        height: 255,
        right: 4,
        bottom: 255,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      } as DOMRect);

      const handle = document.querySelector('[data-testid="depth-handle"][data-slice-index="1"]') as HTMLElement;
      await fireEvent.pointerDown(handle, { pointerId: 1, button: 0, clientY: 255 - 170 });
      await fireEvent.pointerMove(handle, { pointerId: 1, clientY: 255 - 180 });
      expect(handle).toHaveAttribute('aria-valuenow', '180');
      expect(fetchMock).not.toHaveBeenCalled();
      await fireEvent.pointerUp(handle, { pointerId: 1, clientY: 255 - 180 });

      await waitFor(() =>
        expect(fetchMock).toHaveBeenCalledWith(
          '/api/v1/projects/appstate-test/slices/1/depth',
          expect.objectContaining({ method: 'PUT' }),
        ),
      );
      const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/slices/1/depth'));
      expect(JSON.parse(call![1]!.body as string)).toEqual({ depth: 180 });
    });
  });
});
