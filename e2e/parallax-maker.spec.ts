import { readFile } from 'node:fs/promises';
import { test, expect, disableAnimations, requireWorkflow } from './fixtures';
import {
  imageContainsRGB,
  imageDimensions,
  imageHash,
  imageBufferMetadata,
  imagePixel,
  imageSignature,
  sourceImagePixel,
  sourceImageAlphaPoint,
} from './helpers/image';
import { fetchArtifact, listArtifacts, rawArtifactDataUrl, readE2EState } from './helpers/oracle';

test.beforeEach(async ({ ui, page }) => {
  await ui.goto();
  await disableAnimations(page);
});

test('upload generates deterministic depth and three real slices', async ({ ui }) => {
  requireWorkflow(ui, 'upload-depth-slices');
  await ui.uploadInputImage();

  const input = ui.mainImage();
  const depth = ui.depthImage();
  expect(await imageDimensions(input)).toEqual({ width: 320, height: 240 });
  expect(await imageDimensions(depth)).toEqual({ width: 320, height: 240 });

  const left = await imagePixel(depth, 16, 16);
  const right = await imagePixel(depth, 304, 223);
  expect(left).toEqual([2, 2, 2, 255]);
  expect(right).toEqual([252, 252, 252, 255]);

  await expect(ui.thresholdHandles()).toHaveCount(2);
  await ui.openTab('Segmentation');
  await ui.generateSlices();
  await expect(ui.sliceImages()).toHaveCount(3);
  const slices = await ui.sliceImages().all();
  for (const image of slices) {
    expect(await imageDimensions(image)).toEqual({ width: 320, height: 240 });
  }
  const samplePoints: [number, number][] = [[16, 16], [160, 120], [304, 223]];
  expect(await imageSignature(slices[0], samplePoints)).toEqual([
    [30, 21, 70, 255], [200, 200, 200, 255], [75, 75, 75, 255],
  ]);
  expect(await imageSignature(slices[1], samplePoints)).toEqual([
    [200, 200, 200, 255], [120, 95, 70, 255], [75, 75, 75, 255],
  ]);
  expect(await imageSignature(slices[2], samplePoints)).toEqual([
    [200, 200, 200, 255], [200, 200, 200, 255], [210, 168, 70, 255],
  ]);
});

test('point mask modifiers replace, union, and subtract exact regions', async ({ page, ui }) => {
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.setSegmentationMode('Instance Segmentation');
  await ui.openTab('Segmentation');

  const image = ui.mainImage();
  const originalHash = await imageHash(image);

  await ui.clickImagePixel(80, 96);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_mask.samples['80,96'])
    .toBe(255);
  const first = (await readE2EState(page, projectId)).slice_mask;
  expect(first.samples).toMatchObject({ '8,8': 0, '80,96': 255, '200,96': 0 });
  expect(first.nonzero).toBeGreaterThan(4_000);
  await expect.poll(() => imageHash(image)).not.toBe(originalHash);
  const firstHash = await imageHash(image);

  await ui.clickImagePixel(200, 96, ['Shift']);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_mask.nonzero)
    .toBe(first.nonzero * 2);
  const union = (await readE2EState(page, projectId)).slice_mask;
  expect(union.samples).toMatchObject({ '8,8': 0, '80,96': 255, '200,96': 255 });
  await expect.poll(() => imageHash(image)).not.toBe(firstHash);
  const unionHash = await imageHash(image);

  await ui.clickImagePixel(80, 96, ['Control']);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_mask.samples['80,96'])
    .toBe(0);
  const subtracted = (await readE2EState(page, projectId)).slice_mask;
  expect(subtracted.samples).toMatchObject({ '8,8': 0, '80,96': 0, '200,96': 255 });
  expect(subtracted.nonzero).toBe(first.nonzero);
  await expect.poll(() => imageHash(image)).not.toBe(unionHash);

  await ui.clickImagePixel(80, 96);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_mask.samples['200,96'])
    .toBe(0);
  const replaced = (await readE2EState(page, projectId)).slice_mask;
  expect(replaced.samples).toMatchObject({ '8,8': 0, '80,96': 255, '200,96': 0 });
  expect(replaced.nonzero).toBe(first.nonzero);
  await expect.poll(() => imageHash(image)).toBe(firstHash);
});

test('point segmentation honors positive and negative points through the UI', async ({ page, ui }) => {
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.setSegmentationMode('Instance Segmentation');
  await ui.openTab('Segmentation');

  const image = ui.mainImage();
  const before = await imageSignature(image, [
    [128, 96],
    [8, 8],
  ]);
  await ui.clickImagePixel(128, 96);
  await expect(ui.log()).toContainText(/Segment Anything/);
  const positiveMask = (await readE2EState(page, projectId)).slice_mask;
  expect(positiveMask.samples).toMatchObject({ '8,8': 0, '128,96': 255 });
  expect(positiveMask.nonzero).toBeGreaterThan(4_000);
  const afterPositive = await imageSignature(image, [
    [128, 96],
    [8, 8],
  ]);
  expect(afterPositive[0]).not.toEqual(before[0]);
  expect(afterPositive[1]).not.toEqual(before[1]);
  expect(afterPositive[0]).not.toEqual(afterPositive[1]);
  const committedImageHash = await imageHash(image);
  expect((await readE2EState(page, projectId)).segmentation_input).toEqual({
    calls: 1,
    source: 'state-image',
  });

  await ui.toggleMultiPoint();
  await ui.expectMultiPointEnabled(true);
  await expect
    .poll(async () => {
      const state = await readE2EState(page, projectId);
      return { enabled: state.multi_point_mode, points: state.points_selected };
    })
    .toEqual({ enabled: true, points: [] });

  await ui.clickImagePixel(90, 96);
  // The current browser-to-image transform scales the rendered click and then
  // truncates it, so these requested positions arrive one pixel lower on each axis.
  await expect
    .poll(async () => (await readE2EState(page, projectId)).points_selected)
    .toEqual([{ point: [89, 95], negative: false }]);
  expect((await readE2EState(page, projectId)).slice_mask).toEqual(positiveMask);
  expect(await imageHash(image)).toBe(committedImageHash);

  await ui.clickImagePixel(128, 96, ['Control']);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).points_selected)
    .toEqual([
      { point: [89, 95], negative: false },
      { point: [127, 95], negative: true },
    ]);
  expect((await readE2EState(page, projectId)).slice_mask).toEqual(positiveMask);
  expect(await imageHash(image)).toBe(committedImageHash);

  await ui.toggleMultiPoint();
  await ui.expectMultiPointEnabled(false);
  await expect
    .poll(async () => {
      const state = await readE2EState(page, projectId);
      return { enabled: state.multi_point_mode, points: state.points_selected };
    })
    .toEqual({ enabled: false, points: [] });
  expect((await readE2EState(page, projectId)).slice_mask).toEqual(positiveMask);
  expect(await imageHash(image)).toBe(committedImageHash);

  await ui.toggleMultiPoint();
  await ui.clickImagePixel(90, 96);
  await ui.clickImagePixel(128, 96, ['Control']);
  await ui.commitMultiPoint();
  const afterNegative = await imagePixel(image, 128, 96);
  expect(afterNegative).not.toEqual(afterPositive[0]);
  const negativeMask = (await readE2EState(page, projectId)).slice_mask;
  expect(negativeMask.samples).toMatchObject({ '8,8': 0, '90,96': 255, '128,96': 0 });
  expect(negativeMask.nonzero).toBeGreaterThan(0);
});

test('default depth click records its pixel, depth, log, and mask', async ({ page, ui }) => {
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.expectSegmentationMode('Depth Map');
  await ui.openTab('Segmentation');

  await ui.clickImagePixel(16, 16);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_pixel)
    .not.toBeNull();
  const state = await readE2EState(page, projectId);
  // Lock in the same rendered-coordinate truncation exercised by real clicks.
  expect(state.slice_pixel).toEqual([15, 15]);
  expect(state.slice_pixel_depth).toBe(1);
  expect(state.slice_mask.samples).toMatchObject({ '16,16': 255, '160,96': 0 });
  await expect(ui.log()).toContainText(
    'Click event at pixel coordinates (15, 15) at depth 1',
  );
});

test('selected-slice segmentation sends the composed slice to the model', async ({ page, ui }) => {
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.setSegmentationMode('Instance Segmentation');
  await ui.openTab('Segmentation');
  await ui.selectSlice(projectId, 1);

  await ui.clickImagePixel(128, 96);
  await expect(ui.log()).toContainText(/Segment Anything/);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).segmentation_input)
    .toEqual({ calls: 1, source: 'slice:1:NONE' });
  const state = await readE2EState(page, projectId);
  expect(state.selected_slice).toBe(1);
  expect(state.slice_mask.samples['128,96']).toBe(255);
});

test('painted mask drives three checkerboard candidates, apply, and undo', async ({ page, ui }) => {
  requireWorkflow(ui, 'inpainting');
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');
  const selectedSlice = await ui.selectSlice(projectId, 1);
  const originalHash = await imageHash(selectedSlice);
  const rawSliceSrc = await rawArtifactDataUrl(page, projectId, 'image_slice_1.png');
  await ui.openTab('Inpainting');
  await ui.expectGenerateEnabled();
  await ui.drawMaskStroke();
  const persistedMask = (await readE2EState(page, projectId)).selected_mask_file;
  expect(persistedMask.present).toBe(true);
  expect(persistedMask.nonzero).toBeGreaterThan(0);
  expect(persistedMask.bounds).not.toBeNull();
  expect(persistedMask.max).toBe(255);
  expect(persistedMask.inside).not.toBeNull();
  expect(persistedMask.outside).not.toBeNull();

  await ui.fillPrompts('deterministic browser test', 'deterministic exclusion');
  await ui.generateInpainting();
  await expect(ui.candidateImages()).toHaveCount(3);
  await expect
    .poll(async () => {
      const state = await readE2EState(page, projectId);
      return [state.positive_prompts[1], state.negative_prompts[1]];
    })
    .toEqual(['deterministic browser test', 'deterministic exclusion']);
  await expect.poll(() => imageContainsRGB(ui.candidateImages().nth(0), [0, 255, 255])).toBe(true);
  await expect.poll(() => imageContainsRGB(ui.candidateImages().nth(0), [255, 0, 255])).toBe(true);
  await expect.poll(() => imageContainsRGB(ui.candidateImages().nth(1), [255, 128, 0])).toBe(true);
  await expect.poll(() => imageContainsRGB(ui.candidateImages().nth(1), [0, 64, 255])).toBe(true);
  const inside = persistedMask.inside!;
  const outside = persistedMask.outside!;
  const candidateInside = await imagePixel(ui.candidateImages().nth(1), ...inside);
  const candidateOutside = await imagePixel(ui.candidateImages().nth(1), ...outside);
  const originalOutside = await sourceImagePixel(page, rawSliceSrc, ...outside);
  const candidateRGB = candidateInside.slice(0, 3);
  const distanceFromPalette = ([red, green, blue]: number[]) =>
    Math.max(
      Math.abs(candidateRGB[0] - red),
      Math.abs(candidateRGB[1] - green),
      Math.abs(candidateRGB[2] - blue),
    );
  // The real mask blur/composition intentionally leaves a tiny contribution
  // from the source even at the selected maximum-mask pixel.
  expect(
    Math.min(distanceFromPalette([255, 128, 0]), distanceFromPalette([0, 64, 255])),
  ).toBeLessThanOrEqual(5);
  expect(candidateOutside).toEqual(originalOutside);

  await ui.selectCandidate(1);
  await ui.applyCandidate();
  await expect(ui.log()).toContainText(/Inpainting applied to slice 1/);

  await ui.openTab('Segmentation');
  const undo = ui.undoButton(1);
  await expect(undo).toBeEnabled();
  expect(await imageHash(ui.sliceImages().nth(1))).not.toBe(originalHash);
  await undo.click();
  await expect.poll(() => imageHash(ui.sliceImages().nth(1))).toBe(originalHash);

  await ui.selectSlice(projectId, 0);
  await ui.selectSlice(projectId, 1);
  await ui.openTab('Inpainting');
  await ui.expectPrompts('deterministic browser test', 'deterministic exclusion');
});

test('fill generates three checkerboards weighted toward transparent slice pixels', async ({ page, ui }) => {
  requireWorkflow(ui, 'segmentation');
  requireWorkflow(ui, 'inpainting');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');
  await ui.selectSlice(projectId, 1);
  const rawSliceSrc = await rawArtifactDataUrl(page, projectId, 'image_slice_1.png');
  const transparent = await sourceImageAlphaPoint(page, rawSliceSrc, 0);
  const opaque: [number, number] = [160, 120];

  await ui.openTab('Inpainting');
  await ui.fillInpainting();
  await expect(ui.candidateImages()).toHaveCount(3);
  expect(await imageDimensions(ui.candidateImages().nth(0))).toEqual({ width: 320, height: 240 });

  const filledPixel = await imagePixel(ui.candidateImages().nth(0), ...transparent);
  const originalOpaque = await sourceImagePixel(page, rawSliceSrc, ...opaque);
  expect(originalOpaque[3]).toBe(255);
  const candidateOpaque = await imagePixel(ui.candidateImages().nth(0), ...opaque);
  const distanceFromPalette = ([red, green, blue]: number[]) =>
    Math.max(
      Math.abs(filledPixel[0] - red),
      Math.abs(filledPixel[1] - green),
      Math.abs(filledPixel[2] - blue),
    );
  const filledDistance = Math.min(
    distanceFromPalette([0, 255, 255]),
    distanceFromPalette([255, 0, 255]),
  );
  const opaqueDistance = Math.min(
    Math.max(
      Math.abs(candidateOpaque[0] - 0),
      Math.abs(candidateOpaque[1] - 255),
      Math.abs(candidateOpaque[2] - 255),
    ),
    Math.max(
      Math.abs(candidateOpaque[0] - 255),
      Math.abs(candidateOpaque[1] - 0),
      Math.abs(candidateOpaque[2] - 255),
    ),
  );
  expect(filledDistance).toBeLessThanOrEqual(5);
  expect(filledPixel[3]).toBe(255);
  expect(candidateOpaque[3]).toBe(originalOpaque[3]);
  expect(opaqueDistance).toBeGreaterThan(filledDistance + 50);
});

test('enhance returns two same-size candidates while preserving slice alpha', async ({ page, ui }) => {
  requireWorkflow(ui, 'segmentation');
  requireWorkflow(ui, 'inpainting');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');
  await ui.selectSlice(projectId, 1);
  const rawSliceSrc = await rawArtifactDataUrl(page, projectId, 'image_slice_1.png');

  await ui.openTab('Inpainting');
  await ui.enhance();
  await expect(ui.candidateImages()).toHaveCount(2);
  for (const candidate of await ui.candidateImages().all()) {
    expect(await imageDimensions(candidate)).toEqual({ width: 320, height: 240 });
    const transparent: [number, number] = [0, 0];
    const opaqueBorder: [number, number] = [160, 0];
    const originalTransparent = await sourceImagePixel(page, rawSliceSrc, ...transparent);
    const originalOpaqueBorder = await sourceImagePixel(page, rawSliceSrc, ...opaqueBorder);
    expect(originalTransparent[3]).toBe(0);
    expect(originalOpaqueBorder[3]).toBe(255);
    expect((await imagePixel(candidate, ...transparent))[3]).toBe(originalTransparent[3]);
    const enhancedBorder = await imagePixel(candidate, ...opaqueBorder);
    expect(enhancedBorder[0]).toBeGreaterThan(200);
    expect(enhancedBorder[1]).toBeGreaterThan(200);
    expect(enhancedBorder[2]).toBeLessThan(50);
    expect(enhancedBorder[3]).toBe(originalOpaqueBorder[3]);
  }
});

test('erase removes painted alpha and supports undo and redo', async ({ page, ui }) => {
  requireWorkflow(ui, 'segmentation');
  requireWorkflow(ui, 'inpainting');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');
  const selectedSlice = await ui.selectSlice(projectId, 1);
  const originalHash = await imageHash(selectedSlice);
  const rawSliceSrc = await rawArtifactDataUrl(page, projectId, 'image_slice_1.png');

  await ui.openTab('Inpainting');
  await ui.drawMaskStroke();
  const mask = (await readE2EState(page, projectId)).selected_mask_file;
  expect(mask.inside).not.toBeNull();
  expect(mask.outside).not.toBeNull();
  const inside = mask.inside!;
  const outside = mask.outside!;
  const originalInside = await sourceImagePixel(page, rawSliceSrc, ...inside);
  const originalOutside = await sourceImagePixel(page, rawSliceSrc, ...outside);

  await ui.erase();
  await expect(ui.log()).toContainText(`Inpainting erased for slice 1`);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_filenames[1])
    .toBe('image_slice_1_v2.png');
  const erasedSrc = await rawArtifactDataUrl(page, projectId, 'image_slice_1_v2.png');
  expect((await sourceImagePixel(page, erasedSrc, ...inside))[3]).toBe(0);
  expect(await sourceImagePixel(page, erasedSrc, ...outside)).toEqual(originalOutside);
  expect(originalInside[3]).toBeGreaterThan(0);

  await ui.openTab('Segmentation');
  const erasedHash = await imageHash(ui.sliceImages().nth(1));
  expect(erasedHash).not.toBe(originalHash);
  const undo = ui.undoButton(1);
  await expect(undo).toBeEnabled();
  await undo.click();
  await expect.poll(() => imageHash(ui.sliceImages().nth(1))).toBe(originalHash);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_filenames[1])
    .toBe('image_slice_1.png');

  const redo = ui.redoButton(1);
  await expect(redo).toBeEnabled();
  await redo.click();
  await expect.poll(() => imageHash(ui.sliceImages().nth(1))).toBe(erasedHash);
  await expect
    .poll(async () => (await readE2EState(page, projectId)).slice_filenames[1])
    .toBe('image_slice_1_v2.png');
});

test('saved state restores images, controls, prompts, camera, and theme', async ({ page, ui }) => {
  requireWorkflow(ui, 'project-lifecycle');
  requireWorkflow(ui, 'configuration');
  requireWorkflow(ui, 'segmentation');
  requireWorkflow(ui, 'inpainting');
  const projectId = await ui.restoreFixtureState();

  await ui.expectDarkTheme();
  expect(await imageHash(ui.mainImage())).toBe('528b56cf');
  expect(await imageHash(ui.depthImage())).toBe('c6c301c5');
  expect(await imagePixel(ui.mainImage(), 30, 30)).toEqual([240, 50, 45, 255]);
  expect(await imagePixel(ui.mainImage(), 248, 96)).toEqual([35, 210, 90, 255]);
  expect(await imagePixel(ui.depthImage(), 0, 0)).toEqual([0, 0, 0, 255]);
  expect(await imagePixel(ui.depthImage(), 319, 239)).toEqual([255, 255, 255, 255]);
  await ui.expectSliderValue('num-slices', 3);
  await expect(ui.thresholdHandles()).toHaveCount(2);
  await ui.expectSliderValue('camera-distance', 125);
  await ui.expectSliderValue('max-distance', 140);
  await ui.expectSliderValue('focal-length', 475);
  await ui.expectSliderValue('displacement', 15);
  expect((await readE2EState(page, projectId)).thresholds).toEqual([0, 85, 170, 255]);

  await ui.openTab('Segmentation');
  await ui.selectSlice(projectId, 1);
  await ui.openTab('Inpainting');
  await ui.expectPrompts('fixture foreground 1', 'fixture exclusion 1');

  await ui.openTab('Configuration');
  await ui.expectDepthModel('DINOv2');
  await ui.expectInpaintingModel('SD XL 1.0');
});

test('glTF downloads as a valid scene and animation renders four frames without a download', async ({ page, ui }) => {
  requireWorkflow(ui, 'export');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Export');
  await ui.setSlider('displacement', 0);

  const download = await ui.exportGltf();
  expect(download.suggestedFilename()).toBe('scene.gltf');
  const path = await download.path();
  expect(path).not.toBeNull();
  const scene = JSON.parse(await readFile(path!, 'utf8')) as {
    asset?: { version?: string };
    scenes?: unknown[];
    nodes?: unknown[];
    meshes?: unknown[];
    images?: Array<{ uri?: string }>;
  };
  expect(scene.asset?.version).toBe('2.0');
  expect(scene.scenes?.length).toBeGreaterThan(0);
  expect(scene.nodes?.length).toBe(4);
  expect(scene.meshes?.length).toBe(3);
  expect(scene.images).toHaveLength(3);
  expect(scene.images?.every((image) => image.uri?.startsWith('data:image/png;base64,'))).toBe(true);

  await ui.setSlider('number-of-frames', 4);
  const downloads: string[] = [];
  page.on('download', (event) => downloads.push(event.suggestedFilename()));
  await ui.exportAnimation();
  await expect(ui.log()).toContainText('Exported 4 frames to animation');
  // This locks in the current contract: animation writes server-side frames but the
  // otherwise-present dcc.Download is not populated by the callback.
  expect(downloads).toEqual([]);

  const artifacts = await listArtifacts(page, projectId);
  const frames = artifacts.files
    .filter((file) => /^rendered_image_\d{3}\.png$/.test(file.path))
    .sort((left, right) => left.path.localeCompare(right.path));
  expect(frames).toEqual([
    { path: 'rendered_image_000.png', size: expect.any(Number) },
    { path: 'rendered_image_001.png', size: expect.any(Number) },
    { path: 'rendered_image_002.png', size: expect.any(Number) },
    { path: 'rendered_image_003.png', size: expect.any(Number) },
  ]);
  expect(frames.every((frame) => frame.size > 100)).toBe(true);
  const frameMetadata = await Promise.all(
    frames.map(async (frame) => {
      const response = await fetchArtifact(page, projectId, frame.path);
      expect(response.ok(), `download ${frame.path}`).toBeTruthy();
      expect(response.headers()['content-type']).toContain('image/png');
      return imageBufferMetadata(page, await response.body());
    }),
  );
  expect(frameMetadata.every((frame) => frame.width === 320 && frame.height === 240)).toBe(true);
  expect(new Set(frameMetadata.map((frame) => frame.hash)).size).toBeGreaterThan(1);
});
