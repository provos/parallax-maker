import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/svelte';

// `@google/model-viewer` registers a custom element with real WebGL/Three.js
// internals that jsdom cannot run; mock it so this file only asserts *when*
// it gets loaded (via `loadSpy`), not what it does once loaded.
const { loadSpy } = vi.hoisted(() => ({ loadSpy: vi.fn() }));
vi.mock('@google/model-viewer', () => {
  loadSpy();
  return {};
});

import Model3DViewer from './Model3DViewer.svelte';
import { projectStore } from '../../state/project.svelte';
import { uiStore } from '../../state/ui.svelte';
import type { ProjectView } from '../../api/types';

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

describe('Model3DViewer', () => {
  beforeEach(() => {
    projectStore.reset();
    uiStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('does not load the model-viewer module while the 2D tab is active', () => {
    render(Model3DViewer);
    expect(uiStore.viewerTab).toBe('2D');
    expect(screen.getByTestId('model-viewer-loading')).toBeInTheDocument();
    expect(loadSpy).not.toHaveBeenCalled();
  });

  it('lazy-loads the model-viewer module once the 3D tab is opened', async () => {
    render(Model3DViewer);
    uiStore.setViewerTab('3D');
    await waitFor(() => expect(loadSpy).toHaveBeenCalled());
  });

  it('shows a placeholder (matching Dash\'s "No glTF file available.") before any scene has been exported', async () => {
    uiStore.setViewerTab('3D');
    projectStore.applyView(makeView());
    render(Model3DViewer);
    await waitFor(() =>
      expect(screen.getByTestId('model-viewer-empty')).toHaveTextContent('No glTF file available.'),
    );
  });

  it('renders <model-viewer> pointing at the exported glTF asset once a scene exists', async () => {
    uiStore.setViewerTab('3D');
    projectStore.applyView(
      makeView({ exports: { gltf: { url: '/api/v1/projects/appstate-test/export/gltf?v=1' }, upscaled: false } }),
    );
    render(Model3DViewer);
    await waitFor(() =>
      expect(screen.getByTestId('model-viewer')).toHaveAttribute(
        'src',
        '/api/v1/projects/appstate-test/export/gltf?v=1',
      ),
    );
  });
});
