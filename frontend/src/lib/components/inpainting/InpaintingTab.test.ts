import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import InpaintingTab from './InpaintingTab.svelte';
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

describe('InpaintingTab', () => {
  beforeEach(() => {
    projectStore.reset();
    jobStore.end();
    logStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('disables Generate/Fill/Enhance/Erase and Apply with no project loaded', () => {
    render(InpaintingTab);
    expect(screen.getByTestId('generate-inpainting')).toBeDisabled();
    expect(screen.getByTestId('fill-inpainting')).toBeDisabled();
    expect(screen.getByTestId('enhance-inpainting')).toBeDisabled();
    expect(screen.getByTestId('erase-inpainting')).toBeDisabled();
    expect(screen.getByTestId('apply-inpainting')).toBeDisabled();
  });

  it('enables Generate once a slice is selected', () => {
    projectStore.applyView(makeView({ selectedSlice: 1 }));
    render(InpaintingTab);
    expect(screen.getByTestId('generate-inpainting')).toBeEnabled();
  });

  it('loads the selected slice\'s saved prompts into the textareas', async () => {
    projectStore.applyView(
      makeView({
        selectedSlice: 1,
        slices: [makeSlice(0, { positivePrompt: 'zero', negativePrompt: 'not zero' }), makeSlice(1, { positivePrompt: 'one', negativePrompt: 'not one' })],
      }),
    );
    render(InpaintingTab);

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
    render(InpaintingTab);
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
    render(InpaintingTab);

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

    render(InpaintingTab);
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
    render(InpaintingTab);
    expect(screen.getByTestId('apply-inpainting')).toBeDisabled();
  });
});
