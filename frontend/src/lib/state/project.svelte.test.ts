import { describe, expect, it } from 'vitest';
import { projectStore } from './project.svelte';
import type { ProjectView } from '../api/types';

function makeView(overrides: Partial<ProjectView> = {}): ProjectView {
  return {
    id: 'appstate-test',
    revision: 0,
    image: null,
    assets: { input: null, depth: null },
    depthModel: 'midas',
    numSlices: 0,
    thresholds: [],
    slices: [],
    selectedSlice: null,
    segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false },
    busy: null,
    ...overrides,
  };
}

describe('projectStore.applyView', () => {
  it('applies the first view unconditionally', () => {
    projectStore.reset();
    const view = makeView({ revision: 3 });
    expect(projectStore.applyView(view)).toBe(true);
    expect(projectStore.view).toEqual(view);
  });

  it('applies a view with a newer revision', () => {
    projectStore.reset();
    projectStore.applyView(makeView({ revision: 1 }));
    const newer = makeView({ revision: 2, numSlices: 5 });
    expect(projectStore.applyView(newer)).toBe(true);
    expect(projectStore.view).toEqual(newer);
  });

  it('applies a view with an equal revision (idempotent re-fetch)', () => {
    projectStore.reset();
    projectStore.applyView(makeView({ revision: 2 }));
    const same = makeView({ revision: 2, depthModel: 'zoedepth' });
    expect(projectStore.applyView(same)).toBe(true);
    expect(projectStore.view?.depthModel).toBe('zoedepth');
  });

  it('ignores a stale view with an older revision than the one currently held', () => {
    projectStore.reset();
    projectStore.applyView(makeView({ revision: 5, numSlices: 3 }));
    const stale = makeView({ revision: 4, numSlices: 999 });
    expect(projectStore.applyView(stale)).toBe(false);
    // The store keeps the newer view untouched.
    expect(projectStore.view?.revision).toBe(5);
    expect(projectStore.view?.numSlices).toBe(3);
  });

  it('reset clears the view so a subsequent apply is unconditional again', () => {
    projectStore.reset();
    projectStore.applyView(makeView({ revision: 10 }));
    projectStore.reset();
    expect(projectStore.view).toBeNull();
    expect(projectStore.applyView(makeView({ revision: 1 }))).toBe(true);
  });
});
