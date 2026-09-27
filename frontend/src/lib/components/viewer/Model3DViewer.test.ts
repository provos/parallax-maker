import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';

// `@google/model-viewer` registers a custom element with real WebGL/Three.js
// internals that jsdom cannot run; mock it so this file only asserts *when*
// it gets loaded (via `loadSpy`), not what it does once loaded.
const { loadSpy } = vi.hoisted(() => ({ loadSpy: vi.fn() }));
vi.mock('@google/model-viewer', () => {
  loadSpy();
  return {};
});

vi.mock('../../workflow', () => ({ startGltfExport: vi.fn() }));

import Model3DViewer from './Model3DViewer.svelte';
import * as workflow from '../../workflow';
import { projectStore } from '../../state/project.svelte';
import { uiStore } from '../../state/ui.svelte';
import { jobStore } from '../../state/jobs.svelte';
import type { ProjectView, SliceView, SceneProfileView } from '../../api/types';

function makeSlice(index: number, overrides: Partial<SliceView> = {}): SliceView {
  return {
    index,
    depth: index * 50,
    version: 1,
    canUndo: false,
    canRedo: false,
    positivePrompt: '',
    negativePrompt: '',
    isGround: false,
    image: { url: `/api/v1/projects/appstate-test/assets/slice-${index}` },
    thumbnail: { url: `/api/v1/projects/appstate-test/assets/slice-${index}-thumb` },
    ...overrides,
  } as SliceView;
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
    exports: { gltf: null, gltfStale: false, upscaled: false },
    ...overrides,
  };
}

/** Mirrors Model3DViewer.svelte's own `framing` derivation exactly, so the
 * expected `<model-viewer>` attribute strings are computed with the same
 * floating-point math instead of brittle hard-coded decimals. */
function expectedFraming(profile: SceneProfileView, maxDistance = 0) {
  const depths = profile.cards.map((card) => card.z);
  const middle = depths.length ? (Math.min(...depths) + Math.max(...depths)) / 2 : maxDistance / 2;
  const pitch = (profile.pitch * Math.PI) / 180;
  const along = Math.max(middle - profile.cameraZ, 1);
  const radius = along / Math.cos(pitch);
  return {
    target: `0m ${along * Math.tan(pitch)}m ${middle}m`,
    orbit: `180deg ${90 + profile.pitch}deg ${radius}m`,
    maxOrbit: `auto auto ${radius * 4}m`,
    fov: `${2 * profile.halfFov}deg`,
  };
}

/** Flushes the microtask queue so chained `await`s inside `startGltfExport`'s
 * mocked promise (and the component's own `.then`) get a chance to run. */
function flush(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Mocks `workflow.startGltfExport` the way the *real* implementation
 * behaves with respect to `jobStore`: it marks the job busy (`jobStore.begin`)
 * for as long as the call is in flight, and only clears it (`jobStore.end`)
 * once the caller resolves it. This matters for these tests: the component's
 * own auto-rebuild `$effect` re-reads `isBusy()`, and a naive mock that
 * resolves instantly (without ever touching `jobStore`) lets that effect
 * race and re-fire before the real implementation ever would (it stays busy
 * across its own network/poll round trip).
 */
function mockStartGltfExport(): { resolve: (ok: boolean) => void } {
  let resolveFn!: (ok: boolean) => void;
  vi.mocked(workflow.startGltfExport).mockImplementation(() => {
    jobStore.begin('export-gltf');
    return new Promise<boolean>((resolve) => {
      resolveFn = (ok: boolean) => {
        jobStore.end();
        resolve(ok);
      };
    });
  });
  return {
    resolve: (ok: boolean) => resolveFn(ok),
  };
}

describe('Model3DViewer', () => {
  beforeEach(() => {
    projectStore.reset();
    uiStore.reset();
    jobStore.end();
    jobStore.clearError();
    vi.mocked(workflow.startGltfExport).mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('does not load the model-viewer module while the 2D tab is active', () => {
    render(Model3DViewer);
    expect(uiStore.view).toBe('input');
    expect(screen.getByTestId('model-viewer-loading')).toBeInTheDocument();
    expect(loadSpy).not.toHaveBeenCalled();
  });

  it('lazy-loads the model-viewer module once the 3D tab is opened', async () => {
    render(Model3DViewer);
    uiStore.setView('3d');
    await waitFor(() => expect(loadSpy).toHaveBeenCalled());
  });

  it('shows a placeholder (matching Dash\'s "No glTF file available.") before any scene has been exported', async () => {
    uiStore.setView('3d');
    projectStore.applyView(makeView());
    render(Model3DViewer);
    await waitFor(() =>
      expect(screen.getByTestId('model-viewer-empty')).toHaveTextContent('No glTF file available.'),
    );
  });

  it('renders <model-viewer> pointing at the exported glTF asset once a current scene exists', async () => {
    uiStore.setView('3d');
    projectStore.applyView(
      makeView({
        exports: { gltf: { url: '/api/v1/projects/appstate-test/export/gltf?v=1' }, gltfStale: false, upscaled: false },
      }),
    );
    render(Model3DViewer);
    await waitFor(() =>
      expect(screen.getByTestId('model-viewer')).toHaveAttribute(
        'src',
        '/api/v1/projects/appstate-test/export/gltf?v=1',
      ),
    );
  });

  describe('automatic rebuild of a stale/missing export', () => {
    function staleView(overrides: Partial<ProjectView> = {}): ProjectView {
      return makeView({
        slices: [makeSlice(0)],
        exports: { gltf: { url: '/api/v1/projects/appstate-test/export/gltf?v=1' }, gltfStale: true, upscaled: false },
        ...overrides,
      });
    }

    it('fires with { preview: true } and the current uiStore.exportDof value', async () => {
      uiStore.setExportDof(true);
      mockStartGltfExport();
      uiStore.setView('3d');
      projectStore.applyView(staleView());
      render(Model3DViewer);

      await waitFor(() =>
        expect(workflow.startGltfExport).toHaveBeenCalledWith(true, { preview: true }),
      );
      expect(workflow.startGltfExport).toHaveBeenCalledTimes(1);
    });

    it('does not fire when the view is not 3d', async () => {
      vi.mocked(workflow.startGltfExport).mockResolvedValue(true);
      projectStore.applyView(staleView());
      render(Model3DViewer);

      await flush();
      expect(workflow.startGltfExport).not.toHaveBeenCalled();
    });

    it('does not fire when the project has no slices', async () => {
      vi.mocked(workflow.startGltfExport).mockResolvedValue(true);
      uiStore.setView('3d');
      projectStore.applyView(staleView({ slices: [] }));
      render(Model3DViewer);

      await flush();
      expect(workflow.startGltfExport).not.toHaveBeenCalled();
    });

    it('does not fire when the glTF is current (not stale)', async () => {
      vi.mocked(workflow.startGltfExport).mockResolvedValue(true);
      uiStore.setView('3d');
      projectStore.applyView(
        staleView({
          exports: { gltf: { url: '/api/v1/projects/appstate-test/export/gltf?v=1' }, gltfStale: false, upscaled: false },
        }),
      );
      render(Model3DViewer);

      await flush();
      expect(workflow.startGltfExport).not.toHaveBeenCalled();
    });

    it('does not fire while another job is busy', async () => {
      vi.mocked(workflow.startGltfExport).mockResolvedValue(true);
      uiStore.setView('3d');
      projectStore.applyView(staleView());
      jobStore.begin('inpainting');
      render(Model3DViewer);

      await flush();
      expect(workflow.startGltfExport).not.toHaveBeenCalled();
    });

    /**
     * Fails the way the real `startGltfExport` does: the job store goes idle
     * first, and the promise settles only after a further round trip (its
     * trailing log refresh).
     */
    function mockFailingExport(): void {
      vi.mocked(workflow.startGltfExport).mockImplementation(async () => {
        jobStore.begin('export-gltf');
        await flush();
        jobStore.end();
        await flush();
        return false;
      });
    }

    it('shows the retry placeholder after a failed rebuild, without retrying on its own', async () => {
      mockFailingExport();
      uiStore.setView('3d');
      projectStore.applyView(staleView());
      render(Model3DViewer);

      await waitFor(() => expect(screen.getByTestId('model-viewer-failed')).toBeInTheDocument());
      expect(screen.getByTestId('model-viewer-retry')).toBeInTheDocument();
      await flush();
      await flush();
      expect(workflow.startGltfExport).toHaveBeenCalledTimes(1);
    });

    it('retry calls startGltfExport again', async () => {
      mockFailingExport();
      uiStore.setView('3d');
      projectStore.applyView(staleView());
      render(Model3DViewer);

      await waitFor(() => expect(screen.getByTestId('model-viewer-retry')).toBeInTheDocument());
      expect(workflow.startGltfExport).toHaveBeenCalledTimes(1);

      await fireEvent.click(screen.getByTestId('model-viewer-retry'));

      await waitFor(() => expect(workflow.startGltfExport).toHaveBeenCalledTimes(2));
      await waitFor(() => expect(screen.getByTestId('model-viewer-retry')).toBeInTheDocument());
      expect(workflow.startGltfExport).toHaveBeenCalledTimes(2);
    });

    it('shows a building placeholder while another job keeps the scene busy', async () => {
      uiStore.setView('3d');
      projectStore.applyView(staleView());
      jobStore.begin('export-gltf');
      render(Model3DViewer);

      // isBusy() is already true, so the auto-rebuild effect never even
      // calls startGltfExport - the placeholder reflects jobStore state alone.
      await waitFor(() =>
        expect(screen.getByTestId('model-viewer-building')).toHaveTextContent('Building the 3D scene…'),
      );
      expect(workflow.startGltfExport).not.toHaveBeenCalled();
    });
  });

  describe('framing attributes from sceneProfile', () => {
    it('computes camera-target, camera-orbit, field-of-view and max-camera-orbit from the scene profile', async () => {
      const profile: SceneProfileView = {
        cameraZ: 0,
        pitch: 10,
        halfFov: 20,
        cards: [
          { index: 0, z: 100, top: 0, bottom: 10 },
          { index: 1, z: 300, top: 0, bottom: 10 },
        ],
        ground: null,
      };
      uiStore.setView('3d');
      projectStore.applyView(
        makeView({
          exports: { gltf: { url: '/api/v1/projects/appstate-test/export/gltf?v=1' }, gltfStale: false, upscaled: false },
          sceneProfile: profile,
        }),
      );
      render(Model3DViewer);

      const expected = expectedFraming(profile);
      await waitFor(() => {
        const el = screen.getByTestId('model-viewer');
        expect(el).toHaveAttribute('camera-target', expected.target);
        expect(el).toHaveAttribute('camera-orbit', expected.orbit);
        expect(el).toHaveAttribute('max-camera-orbit', expected.maxOrbit);
        expect(el).toHaveAttribute('field-of-view', expected.fov);
      });
    });

    it('falls back to auto framing without a sceneProfile', async () => {
      uiStore.setView('3d');
      projectStore.applyView(
        makeView({
          exports: { gltf: { url: '/api/v1/projects/appstate-test/export/gltf?v=1' }, gltfStale: false, upscaled: false },
        }),
      );
      render(Model3DViewer);

      await waitFor(() => {
        const el = screen.getByTestId('model-viewer');
        expect(el).toHaveAttribute('camera-target', 'auto auto auto');
        expect(el).toHaveAttribute('camera-orbit', '180deg 90deg auto');
        expect(el).toHaveAttribute('field-of-view', 'auto');
      });
    });
  });
});
