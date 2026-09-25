import { expect, type APIResponse, type Page } from '@playwright/test';

/**
 * The test-only backend oracle. Every helper here talks to `/__e2e__/*`
 * through `page.request` only — no DOM, no selectors — so it is shared
 * unchanged by every frontend driver and by the scenario file.
 *
 * Project identity: the `projectId` accepted here is the `appstate-*`
 * directory name returned by `UiDriver.restoreFixtureState()`, which is
 * shared across frontends (see `drivers/types.ts`).
 */

export type E2EMaskStats = {
  present: boolean;
  nonzero: number;
  bounds: [number, number, number, number] | null;
  samples: Record<string, number>;
  inside?: [number, number] | null;
  outside?: [number, number] | null;
  max?: number;
};

export type E2EState = {
  selected_slice: number | null;
  selected_inpainting: number | null;
  positive_prompts: string[];
  negative_prompts: string[];
  slice_filenames: string[];
  thresholds: number[];
  slice_count: number;
  mesh_displacement: number;
  slice_mask: E2EMaskStats;
  slice_pixel: [number, number] | null;
  slice_pixel_depth: number | null;
  multi_point_mode: boolean;
  points_selected: Array<{ point: [number, number]; negative: boolean }>;
  segmentation_input: { calls: number; source: string } | null;
  selected_mask_file: E2EMaskStats;
  [key: string]: unknown;
};

export type ArtifactListing = {
  filename: string;
  files: Array<{ path: string; size: number }>;
};

export async function readE2EState(page: Page, projectId: string): Promise<E2EState> {
  const response = await page.request.get(
    `/__e2e__/state?filename=${encodeURIComponent(projectId)}`,
  );
  expect(response.ok(), 'GET /__e2e__/state').toBeTruthy();
  return response.json();
}

/** Fetches one generated artifact (e.g. a rendered slice or animation frame). */
export async function fetchArtifact(page: Page, projectId: string, path: string): Promise<APIResponse> {
  return page.request.get(
    `/__e2e__/artifact/${encodeURIComponent(projectId)}/${path}`,
  );
}

/** Fetches an artifact and returns it as a `data:image/png;base64,` URL for in-page decoding. */
export async function rawArtifactDataUrl(page: Page, projectId: string, path: string): Promise<string> {
  const response = await fetchArtifact(page, projectId, path);
  expect(response.ok(), `download ${path}`).toBeTruthy();
  const buffer = await response.body();
  return `data:image/png;base64,${buffer.toString('base64')}`;
}

/** Lists every file the backend has generated for a project, so exports can be verified. */
export async function listArtifacts(page: Page, projectId: string): Promise<ArtifactListing> {
  const response = await page.request.get(
    `/__e2e__/artifacts?filename=${encodeURIComponent(projectId)}`,
  );
  expect(response.ok(), 'GET /__e2e__/artifacts').toBeTruthy();
  return response.json();
}

/** Fetches a static test fixture, e.g. `input.png` or `state.json`. */
export async function fetchFixture(page: Page, path: string): Promise<APIResponse> {
  return page.request.get(`/__e2e__/fixture/${path}`);
}
