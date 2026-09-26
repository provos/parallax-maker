import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import InpaintPanel from './InpaintPanel.svelte';
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

function makeSlice(index: number, overrides: Partial<SliceView> = {}): SliceView {
  return {
    index,
    depth: 100,
    version: 1,
    canUndo: false,
    canRedo: false,
    positivePrompt: '',
    negativePrompt: '',
    image: { url: `/slice-${index}` },
    thumbnail: { url: `/slice-${index}-thumb` },
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
    slices: [makeSlice(0), makeSlice(1)],
    selectedSlice: 1,
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

describe('InpaintPanel', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('disables Generate/Fill/Enhance/Erase and Apply with no project loaded', () => {
    render(InpaintPanel);
    expect(screen.getByTestId('generate-inpainting')).toBeDisabled();
    expect(screen.getByTestId('fill-inpainting')).toBeDisabled();
    expect(screen.getByTestId('enhance-inpainting')).toBeDisabled();
    expect(screen.getByTestId('erase-inpainting')).toBeDisabled();
    expect(screen.getByTestId('apply-inpainting')).toBeDisabled();
  });

  it('enables Generate once a slice is selected', () => {
    projectStore.applyView(makeView({ selectedSlice: 1 }));
    render(InpaintPanel);
    expect(screen.getByTestId('generate-inpainting')).toBeEnabled();
  });

  it('loads the selected slice\'s saved prompts into the textareas', async () => {
    projectStore.applyView(
      makeView({
        selectedSlice: 1,
        slices: [makeSlice(0, { positivePrompt: 'zero', negativePrompt: 'not zero' }), makeSlice(1, { positivePrompt: 'one', negativePrompt: 'not one' })],
      }),
    );
    render(InpaintPanel);

    expect(screen.getByTestId('positive-prompt')).toHaveValue('one');
    expect(screen.getByTestId('negative-prompt')).toHaveValue('not one');
  });

  it('reloads prompts from the newly selected slice when the selection changes', async () => {
    projectStore.applyView(
      makeView({
        selectedSlice: 1,
        slices: [makeSlice(0, { positivePrompt: 'zero', negativePrompt: '' }), makeSlice(1, { positivePrompt: 'one', negativePrompt: '' })],
      }),
    );
    render(InpaintPanel);
    expect(screen.getByTestId('positive-prompt')).toHaveValue('one');

    projectStore.applyView(
      makeView({
        revision: 2,
        selectedSlice: 0,
        slices: [makeSlice(0, { positivePrompt: 'zero', negativePrompt: '' }), makeSlice(1, { positivePrompt: 'one', negativePrompt: '' })],
      }),
    );
    await waitFor(() => expect(screen.getByTestId('positive-prompt')).toHaveValue('zero'));
  });

  it('does not clobber an in-progress prompt draft on an unrelated project update', async () => {
    projectStore.applyView(makeView({ selectedSlice: 1 }));
    render(InpaintPanel);

    const positive = screen.getByTestId('positive-prompt');
    await fireEvent.input(positive, { target: { value: 'a work in progress draft' } });
    expect(positive).toHaveValue('a work in progress draft');

    // Same selectedSlice, just a newer revision (e.g. a log refresh's side
    // effect) -- must not reset the draft the user is still typing.
    projectStore.applyView(makeView({ revision: 5, selectedSlice: 1 }));
    expect(positive).toHaveValue('a work in progress draft');
  });

  it('selecting a candidate marks it aria-selected and enables Apply; selecting it again toggles it off', async () => {
    const candidates = { generationId: 'gen-1', sliceIndex: 1, images: [{ url: '/c0' }, { url: '/c1' }] };
    projectStore.applyView(
      makeView({ inpainting: { ...makeView().inpainting, candidates, selectedCandidate: null } }),
    );
    const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit): Promise<Response> => {
      const url = String(input);
      if (url === '/api/v1/projects/appstate-test/inpainting/selection') {
        return jsonResponse(200, {
          ...makeView({ inpainting: { ...makeView().inpainting, candidates, selectedCandidate: 0 } }),
          changed: true,
        });
      }
      if (url.startsWith('/api/v1/projects/appstate-test/logs')) {
        return jsonResponse(200, { entries: [], next: 0 });
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(InpaintPanel);
    const images = screen.getAllByTestId('candidate-image');
    expect(images).toHaveLength(2);
    expect(screen.getByTestId('apply-inpainting')).toBeDisabled();

    await fireEvent.click(images[0]);

    await waitFor(() => expect(screen.getByTestId('apply-inpainting')).toBeEnabled());
    expect(screen.getAllByTestId('candidate-image')[0]).toHaveAttribute('aria-selected', 'true');
    expect(JSON.parse(fetchMock.mock.calls[0][1]!.body as string)).toEqual({
      generationId: 'gen-1',
      candidate: 0,
    });
  });

  it('Apply stays disabled when there are candidates but nothing is selected', () => {
    const candidates = { generationId: 'gen-1', sliceIndex: 1, images: [{ url: '/c0' }] };
    projectStore.applyView(
      makeView({ inpainting: { ...makeView().inpainting, candidates, selectedCandidate: null } }),
    );
    render(InpaintPanel);
    expect(screen.getByTestId('apply-inpainting')).toBeDisabled();
  });

  describe('mode switch', () => {
    it('Fill holes is checked; Extend edges is disabled and marked PLANNED', () => {
      render(InpaintPanel);
      expect(screen.getByTestId('inpaint-mode-holes')).toHaveAttribute('aria-checked', 'true');
      expect(screen.getByTestId('inpaint-mode-extend')).toHaveAttribute('aria-checked', 'false');
      expect(screen.getByTestId('inpaint-mode-extend')).toBeDisabled();
      expect(screen.getByTestId('inpaint-mode-extend')).toHaveTextContent('PLANNED');
    });
  });

  describe('sub-step state', () => {
    it('starts at step 1 (paint the holes) with no mask and no candidates', () => {
      projectStore.applyView(makeView({ selectedSlice: 1, slices: [makeSlice(0), makeSlice(1, { mask: null })] }));
      render(InpaintPanel);
      const items = screen.getAllByRole('listitem');
      expect(items[0]).toHaveAttribute('data-state', 'now');
      expect(items[1]).toHaveAttribute('data-state', 'todo');
      expect(items[2]).toHaveAttribute('data-state', 'todo');
    });

    it('moves to step 2 (describe) once the selected slice has a mask', () => {
      projectStore.applyView(
        makeView({ selectedSlice: 1, slices: [makeSlice(0), makeSlice(1, { mask: { url: '/mask-1' } })] }),
      );
      render(InpaintPanel);
      const items = screen.getAllByRole('listitem');
      expect(items[0]).toHaveAttribute('data-state', 'done');
      expect(items[1]).toHaveAttribute('data-state', 'now');
      expect(items[2]).toHaveAttribute('data-state', 'todo');
    });

    it('moves to step 3 (generate and pick) once candidates exist', () => {
      const candidates = { generationId: 'gen-1', sliceIndex: 1, images: [{ url: '/c0' }] };
      projectStore.applyView(
        makeView({
          selectedSlice: 1,
          slices: [makeSlice(0), makeSlice(1, { mask: { url: '/mask-1' } })],
          inpainting: { ...makeView().inpainting, candidates, selectedCandidate: null },
        }),
      );
      render(InpaintPanel);
      const items = screen.getAllByRole('listitem');
      expect(items[0]).toHaveAttribute('data-state', 'done');
      expect(items[1]).toHaveAttribute('data-state', 'done');
      expect(items[2]).toHaveAttribute('data-state', 'now');
    });
  });

  describe('primary-button switch', () => {
    it('Generate is primary before candidates exist; Apply is not', () => {
      projectStore.applyView(makeView({ selectedSlice: 1 }));
      render(InpaintPanel);
      expect(screen.getByTestId('generate-inpainting')).toHaveClass('btn-primary');
      expect(screen.getByTestId('apply-inpainting')).not.toHaveClass('btn-primary');
    });

    it('Apply becomes primary once candidates exist; Generate no longer is', () => {
      const candidates = { generationId: 'gen-1', sliceIndex: 1, images: [{ url: '/c0' }] };
      projectStore.applyView(
        makeView({
          selectedSlice: 1,
          inpainting: { ...makeView().inpainting, candidates, selectedCandidate: null },
        }),
      );
      render(InpaintPanel);
      expect(screen.getByTestId('apply-inpainting')).toHaveClass('btn-primary');
      expect(screen.getByTestId('generate-inpainting')).not.toHaveClass('btn-primary');
    });
  });

  describe('entering the Inpaint step', () => {
    afterEach(() => uiStore.reset());

    function stubSelection() {
      const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const url = String(input);
        if (url.startsWith('/api/v1/projects/appstate-test/logs')) return jsonResponse(200, { entries: [], next: 0 });
        if (url === '/api/v1/projects/appstate-test/selection') {
          const { slice } = JSON.parse(init!.body as string);
          return jsonResponse(200, { ...makeView({ selectedSlice: slice }), changed: true });
        }
        throw new Error(`Unexpected fetch: ${init?.method ?? 'GET'} ${url}`);
      });
      vi.stubGlobal('fetch', fetchMock);
      return fetchMock;
    }

    it('selects the farthest slice when nothing is selected', async () => {
      uiStore.reset();
      projectStore.applyView(
        makeView({ selectedSlice: null, slices: [makeSlice(0, { depth: 40 }), makeSlice(1, { depth: 5 })] }),
      );
      const fetchMock = stubSelection();
      render(InpaintPanel);

      uiStore.setStep('inpaint');

      await waitFor(() => expect(projectStore.view?.selectedSlice).toBe(1));
      const call = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/selection'));
      expect(JSON.parse(call![1]!.body as string)).toEqual({ slice: 1 });
    });

    it('keeps an existing selection', async () => {
      uiStore.reset();
      projectStore.applyView(makeView({ selectedSlice: 0 }));
      const fetchMock = stubSelection();
      render(InpaintPanel);

      uiStore.setStep('inpaint');
      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/selection'))).toBe(false);
    });
  });
});
