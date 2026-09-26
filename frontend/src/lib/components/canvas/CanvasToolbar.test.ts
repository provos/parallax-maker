import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import CanvasToolbar from './CanvasToolbar.svelte';
import { projectStore } from '../../state/project.svelte';
import { uiStore } from '../../state/ui.svelte';
import type { ProjectView } from '../../api/types';

function makeView(overrides: Partial<ProjectView> = {}): ProjectView {
  return {
    id: 'appstate-test',
    revision: 1,
    image: { width: 320, height: 240 },
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
      darkMode: false,
      camera: { distance: 100, focalLength: 100, maxDistance: 200 },
      meshDisplacement: 0,
      depthModel: 'dinov2',
    },
    exports: { gltf: null, upscaled: false },
    ...overrides,
  };
}

describe('CanvasToolbar', () => {
  beforeEach(() => {
    projectStore.reset();
    uiStore.reset();
  });

  it('disables every tool without an input image', () => {
    render(CanvasToolbar);
    expect(screen.getByTestId('tool-pan')).toBeDisabled();
    expect(screen.getByTestId('tool-segment')).toBeDisabled();
    expect(screen.getByTestId('tool-brush')).toBeDisabled();
    expect(screen.getByTestId('tool-extend')).toBeDisabled();
    expect(screen.getByTestId('tool-horizon')).toBeDisabled();
  });

  it('enables Pan, Segment and Horizon once an image is loaded, but not Brush without a selection', () => {
    projectStore.applyView(makeView());
    render(CanvasToolbar);
    expect(screen.getByTestId('tool-pan')).toBeEnabled();
    expect(screen.getByTestId('tool-segment')).toBeEnabled();
    expect(screen.getByTestId('tool-horizon')).toBeEnabled();
    expect(screen.getByTestId('tool-brush')).toBeDisabled();
  });

  it('enables Brush only once a slice is selected', () => {
    projectStore.applyView(makeView({ selectedSlice: 0 }));
    render(CanvasToolbar);
    expect(screen.getByTestId('tool-brush')).toBeEnabled();
  });

  it('keeps Extend disabled always (planned)', () => {
    projectStore.applyView(makeView({ selectedSlice: 0 }));
    render(CanvasToolbar);
    expect(screen.getByTestId('tool-extend')).toBeDisabled();
  });

  it('clicking a tool sets uiStore.tool and marks it aria-pressed', async () => {
    projectStore.applyView(makeView());
    render(CanvasToolbar);
    expect(screen.getByTestId('tool-pan')).toHaveAttribute('aria-pressed', 'true');

    await fireEvent.click(screen.getByTestId('tool-segment'));
    expect(uiStore.tool).toBe('segment');
    expect(screen.getByTestId('tool-segment')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTestId('tool-pan')).toHaveAttribute('aria-pressed', 'false');
  });

  it('clicking Horizon sets uiStore.tool to horizon', async () => {
    projectStore.applyView(makeView());
    render(CanvasToolbar);
    await fireEvent.click(screen.getByTestId('tool-horizon'));
    expect(uiStore.tool).toBe('horizon');
  });
});
