import type { ProjectView } from '../api/types';

/**
 * Holds the current `ProjectView` as Svelte 5 `$state`.
 *
 * Per the architecture doc's "Concurrency" section: every response carries a
 * `revision`, and the client must ignore views older than the one it
 * already holds (responses can arrive out of order, e.g. a slow GET racing
 * a subsequent mutation). `applyView` is the single place that enforces
 * that rule; callers should always route incoming `ProjectView`s through it
 * rather than assigning to project state directly.
 */
function createProjectStore() {
  let view = $state<ProjectView | null>(null);

  /**
   * Apply a `ProjectView` received from the API, ignoring it if it is
   * older (by `revision`) than the view currently held. Returns `true` if
   * the view was applied, `false` if it was ignored as stale.
   */
  function applyView(next: ProjectView): boolean {
    if (view !== null && next.revision < view.revision) {
      return false;
    }
    view = next;
    return true;
  }

  function reset(): void {
    view = null;
  }

  return {
    get view(): ProjectView | null {
      return view;
    },
    applyView,
    reset,
  };
}

export const projectStore = createProjectStore();
export type ProjectStore = ReturnType<typeof createProjectStore>;
