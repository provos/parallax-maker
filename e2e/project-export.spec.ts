import { readFile } from 'node:fs/promises';
import { test, expect, disableAnimations, requireWorkflow } from './fixtures';
import { imageHash, imageBufferMetadata } from './helpers/image';
import { fetchArtifact, listArtifacts, readE2EState } from './helpers/oracle';

/**
 * Characterizes project lifecycle, configuration and export/render behavior
 * on the frozen Dash reference UI (see docs/svelte-migration/PARITY.md's
 * "Project lifecycle", "Configuration" and "Export/Render" sections), before
 * it is extracted into framework-neutral services/API routes. Every scenario
 * starts from `restoreFixtureState()` (three slices at depths [85, 170, 255],
 * thresholds [0, 85, 170, 255], 320x240 input, dark_mode=true, camera
 * distance/focalLength/maxDistance=125/475/140, mesh_displacement=15 - see
 * `parallax_maker/e2e_support/fixtures.py`).
 */

test.beforeEach(async ({ ui, page }) => {
  await ui.goto();
  await disableAnimations(page);
});

// --- Project lifecycle: Save State -> restore round trip -----------------------

test('save state then restore round trips images, slices and settings exactly', async ({
  page,
  ui,
}) => {
  requireWorkflow(ui, 'project-lifecycle');
  requireWorkflow(ui, 'configuration');
  const projectId = await ui.restoreFixtureState();

  const before = await readE2EState(page, projectId);
  const mainHashBefore = await imageHash(ui.mainImage());
  const depthHashBefore = await imageHash(ui.depthImage());

  // Save State lives under the Configuration tab (make_configuration_div).
  await ui.openTab('Configuration');
  await ui.saveState();
  await expect(ui.log()).toContainText(`Saved state to ${projectId}`);

  const savedFile = await fetchArtifact(page, projectId, 'appstate.json');
  expect(savedFile.ok(), 'download appstate.json').toBeTruthy();
  const savedBytes = await savedFile.body();

  // #depthmap-image only renders under the Mode tab (make_depth_map_container);
  // restoreStateFromBytes waits for it to be visible.
  await ui.openTab('Mode');
  await ui.restoreStateFromBytes(savedBytes);

  const after = await readE2EState(page, projectId);
  expect(after.thresholds).toEqual(before.thresholds);
  expect(after.slice_count).toBe(before.slice_count);
  expect(after.slice_depths).toEqual(before.slice_depths);
  expect(after.positive_prompts).toEqual(before.positive_prompts);
  expect(after.negative_prompts).toEqual(before.negative_prompts);
  expect(after.slice_filenames).toEqual(before.slice_filenames);
  expect(after.dark_mode).toBe(before.dark_mode);
  expect(after.camera).toEqual(before.camera);
  expect(after.mesh_displacement).toBe(before.mesh_displacement);
  expect(await imageHash(ui.mainImage())).toBe(mainHashBefore);
  expect(await imageHash(ui.depthImage())).toBe(depthHashBefore);
});

// --- Configuration: camera/displacement slider persistence ---------------------

test('camera and displacement slider changes persist to JSON and survive a restore', async ({
  page,
  ui,
}) => {
  requireWorkflow(ui, 'project-lifecycle');
  requireWorkflow(ui, 'export');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Export');

  await ui.setSlider('camera-distance', 200);
  await ui.setSlider('focal-length', 300);
  await ui.setSlider('max-distance', 250);
  await ui.setSlider('displacement', 40);

  // remember_camera_parameters (WEB-30) saves on every slider change, before
  // any explicit "Save State" click.
  await expect
    .poll(async () => (await readE2EState(page, projectId)).camera)
    .toEqual({ distance: 200, focal_length: 300, max_distance: 250 });
  await expect.poll(async () => (await readE2EState(page, projectId)).mesh_displacement).toBe(40);

  // Save State lives under the Configuration tab (make_configuration_div).
  await ui.openTab('Configuration');
  await ui.saveState();
  const savedFile = await fetchArtifact(page, projectId, 'appstate.json');
  const savedBytes = await savedFile.body();
  await ui.openTab('Mode');
  await ui.restoreStateFromBytes(savedBytes);

  await ui.openTab('Export');
  await ui.expectSliderValue('camera-distance', 200);
  await ui.expectSliderValue('focal-length', 300);
  await ui.expectSliderValue('max-distance', 250);
  await ui.expectSliderValue('displacement', 40);
  expect((await readE2EState(page, projectId)).camera).toEqual({
    distance: 200,
    focal_length: 300,
    max_distance: 250,
  });
  expect((await readE2EState(page, projectId)).mesh_displacement).toBe(40);
});

// --- Configuration: dark mode persistence ---------------------------------------

test('dark mode persists across a save/restore round trip', async ({ page, ui }) => {
  requireWorkflow(ui, 'project-lifecycle');
  const projectId = await ui.restoreFixtureState();

  // The fixture starts dark_mode=true (see fixtures.py); toggling the button
  // once flips it to light and persists that (toggle_dark_mode, WEB-01).
  await ui.expectDarkTheme();
  await expect
    .poll(async () => (await readE2EState(page, projectId)).dark_mode)
    .toBe(true);

  // Save State lives under the Configuration tab (make_configuration_div).
  await ui.openTab('Configuration');
  await ui.saveState();
  const savedFile = await fetchArtifact(page, projectId, 'appstate.json');
  const savedBytes = await savedFile.body();
  await ui.openTab('Mode');
  await ui.restoreStateFromBytes(savedBytes);

  await ui.expectDarkTheme();
  expect((await readE2EState(page, projectId)).dark_mode).toBe(true);
});

// --- Configuration: depth/inpainting model persistence --------------------------

test('depth and inpainting model selections persist across a restore', async ({ page, ui }) => {
  requireWorkflow(ui, 'project-lifecycle');
  requireWorkflow(ui, 'configuration');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Configuration');

  // The fixture starts with DINOv2 / SD XL 1.0 (fixtures.py).
  await ui.expectDepthModel('DINOv2');
  await ui.expectInpaintingModel('SD XL 1.0');

  // The depth-model dropdown lives under the Mode tab (make_depth_map_container),
  // not Configuration - only its resulting *value* is also readable (but not
  // clickable) from there, which is why expectDepthModel above works regardless.
  await ui.openTab('Mode');
  await ui.selectDepthModel('MiDaS');
  await ui.openTab('Configuration');
  await ui.selectInpaintingModel('Automatic1111');
  await ui.expectDepthModel('MiDaS');
  await ui.expectInpaintingModel('Automatic1111');

  await ui.saveState();
  const savedFile = await fetchArtifact(page, projectId, 'appstate.json');
  const savedBytes = await savedFile.body();
  await ui.openTab('Mode');
  await ui.restoreStateFromBytes(savedBytes);

  await ui.openTab('Configuration');
  await ui.expectDepthModel('MiDaS');
  await ui.expectInpaintingModel('Automatic1111');
});

// --- Configuration: external-server connection probe ----------------------------

test('external server connection test highlights success when the fake probe succeeds', async ({
  ui,
}) => {
  requireWorkflow(ui, 'configuration');
  await ui.restoreFixtureState();
  await ui.openTab('Configuration');

  await ui.selectInpaintingModel('Automatic1111');
  await ui.expectExternalConnectionStatus('none');
  await ui.setExternalServer('localhost:7860');
  await ui.testExternalConnection();

  await ui.expectExternalConnectionStatus('success');
});

// --- Export: glTF with displacement > 0 -----------------------------------------

test('glTF export with displacement produces a subdivided, non-flat mesh', async ({ ui }) => {
  requireWorkflow(ui, 'export');
  await ui.restoreFixtureState();
  await ui.openTab('Export');
  // The fixture already sets mesh_displacement=15 (fixtures.py); confirm it,
  // then drive the slider explicitly so the scenario doesn't depend on it.
  await ui.setSlider('displacement', 20);

  const download = await ui.exportGltf();
  const path = await download.path();
  expect(path).not.toBeNull();
  const scene = JSON.parse(await readFile(path!, 'utf8')) as {
    meshes?: unknown[];
    accessors?: Array<{ count: number }>;
  };

  expect(scene.meshes?.length).toBe(3);
  // accessors[1] is the first mesh's POSITION accessor (create_card appends
  // the texCoord accessor before the vertex accessor); a flat, undisplaced
  // card is exactly 4 corner vertices (see the sibling displacement=0
  // scenario in parallax-maker.spec.ts's glTF test) - subdividing for
  // displacement produces a (subdivisions + 1)^2 = 501x501 grid instead.
  expect(scene.accessors?.[1]?.count).toBeGreaterThan(4);
  expect(scene.accessors?.[1]?.count).toBe(501 * 501);
});

// --- Export: glTF with DOF enabled ----------------------------------------------

test('glTF export with DOF enabled switches every material to MASK alpha mode', async ({
  ui,
}) => {
  requireWorkflow(ui, 'export');
  await ui.restoreFixtureState();
  await ui.openTab('Export');
  await ui.setSlider('displacement', 0);

  const flatDownload = await ui.exportGltf();
  const flatPath = await flatDownload.path();
  const flatScene = JSON.parse(await readFile(flatPath!, 'utf8')) as {
    materials?: Array<{ alphaMode?: string }>;
  };
  expect(flatScene.materials?.every((material) => material.alphaMode === 'BLEND')).toBe(true);

  await ui.setDofEnabled(true);
  const dofDownload = await ui.exportGltf();
  const dofPath = await dofDownload.path();
  const dofScene = JSON.parse(await readFile(dofPath!, 'utf8')) as {
    materials?: Array<{ alphaMode?: string; alphaCutoff?: number }>;
  };

  expect(dofScene.materials?.length).toBe(3);
  expect(dofScene.materials?.every((material) => material.alphaMode === 'MASK')).toBe(true);
  expect(dofScene.materials?.every((material) => material.alphaCutoff === 0.5)).toBe(true);
});

// --- Export: Upscale Textures then glTF export uses the upscaled textures -----

test('upscale textures then glTF export embeds the upscaled (2x) images', async ({
  page,
  ui,
}) => {
  requireWorkflow(ui, 'export');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Export');
  await ui.setSlider('displacement', 0);

  const originalRaw = await fetchArtifact(page, projectId, 'image_slice_0.png');
  expect(originalRaw.ok()).toBeTruthy();
  const originalMetadata = await imageBufferMetadata(page, await originalRaw.body());

  await ui.upscaleTextures();
  await expect(ui.log()).toContainText(/Upscaled textures for slices/);

  const artifactsAfterUpscale = await listArtifacts(page, projectId);
  expect(
    artifactsAfterUpscale.files.some((file) => file.path === 'image_slice_0_upscaled.png'),
  ).toBe(true);

  const download = await ui.exportGltf();
  const path = await download.path();
  const scene = JSON.parse(await readFile(path!, 'utf8')) as {
    images?: Array<{ uri?: string }>;
  };
  expect(scene.images).toHaveLength(3);
  const firstImage = scene.images?.[0]?.uri ?? '';
  expect(firstImage.startsWith('data:image/png;base64,')).toBe(true);
  const decoded = Buffer.from(firstImage.split(',')[1], 'base64');
  const exportedMetadata = await imageBufferMetadata(page, decoded);

  // e2e_support.fakes.FakeUpscaler doubles both dimensions.
  expect(exportedMetadata.width).toBe(originalMetadata.width * 2);
  expect(exportedMetadata.height).toBe(originalMetadata.height * 2);
});

// --- Export: raw slice download --------------------------------------------------

test('clicking a slice download icon downloads the exact raw slice PNG', async ({
  page,
  ui,
}) => {
  requireWorkflow(ui, 'slice-editing');
  requireWorkflow(ui, 'export');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');

  const download = await ui.downloadSlice(1);
  const path = await download.path();
  expect(path).not.toBeNull();
  const downloaded = await readFile(path!);

  const onDisk = await fetchArtifact(page, projectId, 'image_slice_1.png');
  expect(onDisk.ok()).toBeTruthy();
  const onDiskBytes = await onDisk.body();

  expect(downloaded.equals(onDiskBytes)).toBe(true);
});
