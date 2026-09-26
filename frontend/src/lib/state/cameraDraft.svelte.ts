/**
 * The camera settings being edited: distance, max distance and focal length
 * (Preview panel), ground distance (Ground panel) and mesh displacement
 * (Export). The API takes the camera fields together (`CameraSettingsRequest`
 * requires distance/focalLength/maxDistance), so every commit sends all of
 * them plus the displacement in one `PUT .../settings`.
 *
 * Sliders update the draft as they move and commit on change. `onchange`
 * fires per arrow-key step, so a fast key-repeat sequence can commit many
 * times before the first round trip lands; firing those concurrently would
 * 409 every commit but the first. Instead only one `updateSettings` call is
 * ever in flight, and later commits flag a follow-up that sends the *latest*
 * draft once it resolves, so the persisted value is the last one the user
 * landed on.
 */
import { projectStore } from './project.svelte';
import * as workflow from '../workflow';
import type { ProjectView } from '../api/types';

export type CameraDraft = {
  distance: number;
  maxDistance: number;
  focalLength: number;
  displacement: number;
  groundNear: number;
};

const DEFAULT_CAMERA: CameraDraft = {
  distance: 100,
  maxDistance: 200,
  focalLength: 100,
  displacement: 0,
  groundNear: 0,
};

function createCameraDraftStore() {
  let draft = $state<CameraDraft>({ ...DEFAULT_CAMERA });
  let commitInFlight = false;
  let commitPending = false;

  async function runCommit(): Promise<void> {
    if (commitInFlight) return;
    commitInFlight = true;
    try {
      while (commitPending) {
        commitPending = false;
        // The ground must start before the max distance; keep it there when
        // the max distance shrinks below it.
        draft.groundNear = Math.min(draft.groundNear, Math.max(0, draft.maxDistance - 1));
        await workflow.updateSettings({
          camera: {
            distance: draft.distance,
            maxDistance: draft.maxDistance,
            focalLength: draft.focalLength,
            groundNear: draft.groundNear,
          },
          meshDisplacement: draft.displacement,
        });
      }
    } finally {
      commitInFlight = false;
    }
  }

  return {
    get draft(): CameraDraft {
      return draft;
    },

    /**
     * Follows the persisted settings (after any change, including a
     * restore) -- but never while a local edit is still being committed: the
     * commit's own applied response would otherwise snap the draft back to
     * an older value moments after a later key press already advanced it.
     */
    sync(settings: ProjectView['settings'] | undefined): void {
      if (!settings || commitInFlight || commitPending) return;
      draft = {
        distance: settings.camera.distance,
        maxDistance: settings.camera.maxDistance,
        focalLength: settings.camera.focalLength,
        displacement: settings.meshDisplacement,
        groundNear: settings.camera.groundNear ?? 0,
      };
    },

    set(field: keyof CameraDraft, value: number): void {
      draft = { ...draft, [field]: value };
    },

    commit(): void {
      if (!projectStore.view) return;
      commitPending = true;
      void runCommit();
    },

    /** Test-only: restores defaults so state doesn't leak between tests. */
    reset(): void {
      draft = { ...DEFAULT_CAMERA };
      commitInFlight = false;
      commitPending = false;
    },
  };
}

export const cameraDraftStore = createCameraDraftStore();
