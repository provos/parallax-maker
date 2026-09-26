import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../workflow', () => ({ updateSettings: vi.fn() }));

import * as workflow from '../workflow';
import { cameraDraftStore } from './cameraDraft.svelte';
import { projectStore } from './project.svelte';
import type { ProjectView } from '../api/types';

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

/** Flushes the microtask queue (and then some), so chained `await`s inside
 * the store's own async commit loop get a chance to run. */
function flush(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function makeSettings(overrides: Partial<ProjectView['settings']> = {}): ProjectView['settings'] {
  return {
    darkMode: false,
    camera: { distance: 100, focalLength: 200, maxDistance: 300, groundNear: 50 },
    meshDisplacement: 10,
    depthModel: 'dinov2',
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
    settings: makeSettings(),
    exports: { gltf: null, upscaled: false },
    ...overrides,
  };
}

describe('cameraDraftStore', () => {
  beforeEach(() => {
    cameraDraftStore.reset();
    projectStore.reset();
    vi.mocked(workflow.updateSettings).mockReset();
    vi.mocked(workflow.updateSettings).mockResolvedValue(undefined);
  });

  afterEach(() => {
    cameraDraftStore.reset();
    projectStore.reset();
  });

  it('sync loads the draft from the persisted settings', () => {
    cameraDraftStore.sync(makeSettings());
    expect(cameraDraftStore.draft).toEqual({
      distance: 100,
      maxDistance: 300,
      focalLength: 200,
      displacement: 10,
      groundNear: 50,
    });
  });

  it('sync defaults groundNear to 0 when the settings do not carry one', () => {
    cameraDraftStore.sync(makeSettings({ camera: { distance: 1, focalLength: 2, maxDistance: 3 } }));
    expect(cameraDraftStore.draft.groundNear).toBe(0);
  });

  it('commit is a no-op without a project view', async () => {
    projectStore.reset();
    cameraDraftStore.set('distance', 250);
    cameraDraftStore.commit();
    await flush();
    expect(workflow.updateSettings).not.toHaveBeenCalled();
  });

  it('sync is ignored while a commit is in flight or another is queued behind it', async () => {
    projectStore.applyView(makeView());
    const { promise, resolve } = deferred<void>();
    vi.mocked(workflow.updateSettings).mockReturnValueOnce(promise);

    cameraDraftStore.set('distance', 250);
    cameraDraftStore.commit();
    // `commit()` runs the async commit loop synchronously up to its first
    // `await`, so it is already in flight by the time `commit()` returns.
    expect(workflow.updateSettings).toHaveBeenCalledTimes(1);

    // A sync landing mid-flight (e.g. the applied response of an unrelated
    // request) must not clobber the still-uncommitted edit.
    cameraDraftStore.sync(makeSettings({ camera: { distance: 999, focalLength: 1, maxDistance: 2 } }));
    expect(cameraDraftStore.draft.distance).toBe(250);

    resolve();
    await flush();
  });

  it('coalesces commits made while one is in flight: only the latest draft is sent in a follow-up', async () => {
    projectStore.applyView(makeView());
    const first = deferred<void>();
    vi.mocked(workflow.updateSettings).mockReturnValueOnce(first.promise).mockResolvedValueOnce(undefined);

    cameraDraftStore.set('distance', 200);
    cameraDraftStore.commit();
    expect(workflow.updateSettings).toHaveBeenCalledTimes(1);

    // Two more commits land while the first request is still in flight.
    cameraDraftStore.set('distance', 300);
    cameraDraftStore.commit();
    cameraDraftStore.set('distance', 400);
    cameraDraftStore.commit();
    // Coalesced: no second request fired yet, only the first is in flight.
    expect(workflow.updateSettings).toHaveBeenCalledTimes(1);

    first.resolve();
    await flush();

    // Exactly one follow-up request, carrying the *latest* draft.
    expect(workflow.updateSettings).toHaveBeenCalledTimes(2);
    const calls = vi.mocked(workflow.updateSettings).mock.calls;
    expect(calls[0][0].camera?.distance).toBe(200);
    expect(calls[1][0].camera?.distance).toBe(400);
  });

  it('clamps groundNear to maxDistance - 1 when the max distance shrinks below it', async () => {
    projectStore.applyView(
      makeView({
        settings: makeSettings({ camera: { distance: 100, focalLength: 26, maxDistance: 500, groundNear: 78 } }),
      }),
    );
    cameraDraftStore.sync(projectStore.view!.settings);

    cameraDraftStore.set('maxDistance', 60);
    cameraDraftStore.commit();
    await flush();

    expect(workflow.updateSettings).toHaveBeenCalledWith({
      camera: { distance: 100, maxDistance: 60, focalLength: 26, groundNear: 59 },
      meshDisplacement: 10,
    });
    expect(cameraDraftStore.draft.groundNear).toBe(59);
  });

  it('does not clamp groundNear when it is already within the new max distance', async () => {
    projectStore.applyView(
      makeView({
        settings: makeSettings({ camera: { distance: 100, focalLength: 26, maxDistance: 500, groundNear: 78 } }),
      }),
    );
    cameraDraftStore.sync(projectStore.view!.settings);

    cameraDraftStore.set('maxDistance', 200);
    cameraDraftStore.commit();
    await flush();

    expect(workflow.updateSettings).toHaveBeenCalledWith({
      camera: { distance: 100, maxDistance: 200, focalLength: 26, groundNear: 78 },
      meshDisplacement: 10,
    });
  });
});
