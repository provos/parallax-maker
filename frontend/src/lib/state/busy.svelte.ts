import { projectStore } from './project.svelte';
import { jobStore } from './jobs.svelte';

/**
 * Whether mutating controls should be disabled: either a locally-initiated
 * request/job is in flight (`jobStore.active`), or the server itself
 * reports the project as busy (`ProjectView.busy`, e.g. a job started from
 * another tab). Per the architecture doc's "Concurrency" section, the UI
 * must never send a request it knows will be rejected with `409 busy`.
 */
export function isBusy(): boolean {
  return jobStore.active !== null || projectStore.view?.busy != null;
}
