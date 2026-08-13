import { readFile } from 'node:fs/promises';
import { test, expect, disableAnimations } from './fixtures';
import {
  clickMainTab,
  drawCanvasStroke,
  gotoApp,
  imageContainsRGB,
  imageDimensions,
  imageHash,
  imageBufferMetadata,
  imagePixel,
  imageSignature,
  sourceImagePixel,
  inpaintingImages,
  restoreFixtureState,
  readE2EState,
  selectDashOption,
  selectSlice,
  setDashSlider,
  sliceImages,
  clickImagePixel,
  uploadInputImage,
} from './helpers/app';

test.beforeEach(async ({ page }) => {
  await gotoApp(page);
  await disableAnimations(page);
});

test('upload generates deterministic depth and three real slices', async ({ page }) => {
  await uploadInputImage(page);

  const input = page.locator('#image');
  const depth = page.locator('#depthmap-image');
  expect(await imageDimensions(input)).toEqual({ width: 320, height: 240 });
  expect(await imageDimensions(depth)).toEqual({ width: 320, height: 240 });

  const left = await imagePixel(depth, 16, 16);
  const right = await imagePixel(depth, 304, 223);
  expect(left).toEqual([2, 2, 2, 255]);
  expect(right).toEqual([252, 252, 252, 255]);

  await expect(page.locator('#thresholds-container [role=slider]')).toHaveCount(2);
  await clickMainTab(page, 'Segmentation');
  await page.locator('#generate-slice-button').click();
  await expect(sliceImages(page)).toHaveCount(3);
  const slices = await sliceImages(page).all();
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

test('point mask modifiers replace, union, and subtract exact regions', async ({ page }) => {
  const filename = await restoreFixtureState(page);
  await selectDashOption(page, 'mode-selector', 'Instance Segmentation');
  await clickMainTab(page, 'Segmentation');

  const image = page.locator('#image');
  const originalHash = await imageHash(image);

  await clickImagePixel(image, 80, 96);
  await expect
    .poll(async () => (await readE2EState(page, filename)).slice_mask.samples['80,96'])
    .toBe(255);
  const first = (await readE2EState(page, filename)).slice_mask;
  expect(first.samples).toMatchObject({ '8,8': 0, '80,96': 255, '200,96': 0 });
  expect(first.nonzero).toBeGreaterThan(4_000);
  await expect.poll(() => imageHash(image)).not.toBe(originalHash);
  const firstHash = await imageHash(image);

  await clickImagePixel(image, 200, 96, ['Shift']);
  await expect
    .poll(async () => (await readE2EState(page, filename)).slice_mask.nonzero)
    .toBe(first.nonzero * 2);
  const union = (await readE2EState(page, filename)).slice_mask;
  expect(union.samples).toMatchObject({ '8,8': 0, '80,96': 255, '200,96': 255 });
  await expect.poll(() => imageHash(image)).not.toBe(firstHash);
  const unionHash = await imageHash(image);

  await clickImagePixel(image, 80, 96, ['Control']);
  await expect
    .poll(async () => (await readE2EState(page, filename)).slice_mask.samples['80,96'])
    .toBe(0);
  const subtracted = (await readE2EState(page, filename)).slice_mask;
  expect(subtracted.samples).toMatchObject({ '8,8': 0, '80,96': 0, '200,96': 255 });
  expect(subtracted.nonzero).toBe(first.nonzero);
  await expect.poll(() => imageHash(image)).not.toBe(unionHash);

  await clickImagePixel(image, 80, 96);
  await expect
    .poll(async () => (await readE2EState(page, filename)).slice_mask.samples['200,96'])
    .toBe(0);
  const replaced = (await readE2EState(page, filename)).slice_mask;
  expect(replaced.samples).toMatchObject({ '8,8': 0, '80,96': 255, '200,96': 0 });
  expect(replaced.nonzero).toBe(first.nonzero);
  await expect.poll(() => imageHash(image)).toBe(firstHash);
});

test('point segmentation honors positive and negative points through the UI', async ({ page }) => {
  const filename = await restoreFixtureState(page);
  await selectDashOption(page, 'mode-selector', 'Instance Segmentation');
  await clickMainTab(page, 'Segmentation');

  const image = page.locator('#image');
  const before = await imageSignature(image, [
    [128, 96],
    [8, 8],
  ]);
  await clickImagePixel(image, 128, 96);
  await expect(page.locator('#log')).toContainText(/Segment Anything/);
  const positiveMask = (await readE2EState(page, filename)).slice_mask;
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
  expect((await readE2EState(page, filename)).segmentation_input).toEqual({
    calls: 1,
    source: 'state-image',
  });

  const multiPoint = page.locator('#multi-point');
  await multiPoint.click();
  await expect(multiPoint).toHaveClass(/color-is-selected/);
  await expect
    .poll(async () => {
      const state = await readE2EState(page, filename);
      return { enabled: state.multi_point_mode, points: state.points_selected };
    })
    .toEqual({ enabled: true, points: [] });

  await clickImagePixel(image, 90, 96);
  // The current browser-to-image transform scales the rendered click and then
  // truncates it, so these requested positions arrive one pixel lower on each axis.
  await expect
    .poll(async () => (await readE2EState(page, filename)).points_selected)
    .toEqual([{ point: [89, 95], negative: false }]);
  expect((await readE2EState(page, filename)).slice_mask).toEqual(positiveMask);
  expect(await imageHash(image)).toBe(committedImageHash);

  await clickImagePixel(image, 128, 96, ['Control']);
  await expect
    .poll(async () => (await readE2EState(page, filename)).points_selected)
    .toEqual([
      { point: [89, 95], negative: false },
      { point: [127, 95], negative: true },
    ]);
  expect((await readE2EState(page, filename)).slice_mask).toEqual(positiveMask);
  expect(await imageHash(image)).toBe(committedImageHash);

  await multiPoint.click();
  await expect(multiPoint).toHaveClass(/color-not-selected/);
  await expect
    .poll(async () => {
      const state = await readE2EState(page, filename);
      return { enabled: state.multi_point_mode, points: state.points_selected };
    })
    .toEqual({ enabled: false, points: [] });
  expect((await readE2EState(page, filename)).slice_mask).toEqual(positiveMask);
  expect(await imageHash(image)).toBe(committedImageHash);

  await multiPoint.click();
  await clickImagePixel(image, 90, 96);
  await clickImagePixel(image, 128, 96, ['Control']);
  await page.locator('#multi-commit').click();
  await expect(page.locator('#log')).toContainText(/Committed points/);
  const afterNegative = await imagePixel(image, 128, 96);
  expect(afterNegative).not.toEqual(afterPositive[0]);
  const negativeMask = (await readE2EState(page, filename)).slice_mask;
  expect(negativeMask.samples).toMatchObject({ '8,8': 0, '90,96': 255, '128,96': 0 });
  expect(negativeMask.nonzero).toBeGreaterThan(0);
});

test('default depth click records its pixel, depth, log, and mask', async ({ page }) => {
  const filename = await restoreFixtureState(page);
  await expect(page.locator('#mode-selector')).toContainText('Depth Map');
  await clickMainTab(page, 'Segmentation');

  await clickImagePixel(page.locator('#image'), 16, 16);
  await expect
    .poll(async () => (await readE2EState(page, filename)).slice_pixel)
    .not.toBeNull();
  const state = await readE2EState(page, filename);
  // Lock in the same rendered-coordinate truncation exercised by real clicks.
  expect(state.slice_pixel).toEqual([15, 15]);
  expect(state.slice_pixel_depth).toBe(1);
  expect(state.slice_mask.samples).toMatchObject({ '16,16': 255, '160,96': 0 });
  await expect(page.locator('#log')).toContainText(
    'Click event at pixel coordinates (15, 15) at depth 1',
  );
});

test('selected-slice segmentation sends the composed slice to the model', async ({ page }) => {
  const filename = await restoreFixtureState(page);
  await selectDashOption(page, 'mode-selector', 'Instance Segmentation');
  await clickMainTab(page, 'Segmentation');
  await selectSlice(page, filename, 1);

  await clickImagePixel(page.locator('#image'), 128, 96);
  await expect(page.locator('#log')).toContainText(/Segment Anything/);
  await expect
    .poll(async () => (await readE2EState(page, filename)).segmentation_input)
    .toEqual({ calls: 1, source: 'slice:1:NONE' });
  const state = await readE2EState(page, filename);
  expect(state.selected_slice).toBe(1);
  expect(state.slice_mask.samples['128,96']).toBe(255);
});

test('painted mask drives three checkerboard candidates, apply, and undo', async ({ page }) => {
  const filename = await restoreFixtureState(page);
  await clickMainTab(page, 'Segmentation');
  const selectedSlice = await selectSlice(page, filename, 1);
  const originalHash = await imageHash(selectedSlice);
  const rawSliceResponse = await page.request.get(
    `/__e2e__/artifact/${encodeURIComponent(filename)}/image_slice_1.png`,
  );
  expect(rawSliceResponse.ok(), 'download raw slice 1').toBeTruthy();
  const rawSliceSrc = `data:image/png;base64,${(await rawSliceResponse.body()).toString('base64')}`;
  await clickMainTab(page, 'Inpainting');
  await expect(page.locator('#generate-inpainting-button')).toBeEnabled();
  await drawCanvasStroke(page);
  const persistedMask = (await readE2EState(page, filename)).selected_mask_file;
  expect(persistedMask.present).toBe(true);
  expect(persistedMask.nonzero).toBeGreaterThan(0);
  expect(persistedMask.bounds).not.toBeNull();
  expect(persistedMask.max).toBe(255);
  expect(persistedMask.inside).not.toBeNull();
  expect(persistedMask.outside).not.toBeNull();

  await page.locator('#positive-prompt').fill('deterministic browser test');
  await page.locator('#generate-inpainting-button').click();
  await expect(inpaintingImages(page)).toHaveCount(3);
  await expect.poll(() => imageContainsRGB(inpaintingImages(page).nth(0), [0, 255, 255])).toBe(true);
  await expect.poll(() => imageContainsRGB(inpaintingImages(page).nth(0), [255, 0, 255])).toBe(true);
  await expect.poll(() => imageContainsRGB(inpaintingImages(page).nth(1), [255, 128, 0])).toBe(true);
  await expect.poll(() => imageContainsRGB(inpaintingImages(page).nth(1), [0, 64, 255])).toBe(true);
  const inside = persistedMask.inside!;
  const outside = persistedMask.outside!;
  const candidateInside = await imagePixel(inpaintingImages(page).nth(1), ...inside);
  const candidateOutside = await imagePixel(inpaintingImages(page).nth(1), ...outside);
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

  await inpaintingImages(page).nth(1).click();
  await expect(inpaintingImages(page).nth(1)).toHaveClass(/color-is-selected-light/);
  await expect(page.locator('#apply-inpainting-button')).toBeEnabled();
  await page.locator('#apply-inpainting-button').click();
  await expect(page.locator('#log')).toContainText(/Inpainting applied to slice 1/);

  await clickMainTab(page, 'Segmentation');
  const undo = page.locator('[title="Undo last change"]').nth(1);
  await expect(undo).toBeEnabled();
  expect(await imageHash(sliceImages(page).nth(1))).not.toBe(originalHash);
  await undo.click();
  await expect.poll(() => imageHash(sliceImages(page).nth(1))).toBe(originalHash);
});

test('saved state restores images, controls, prompts, camera, and theme', async ({ page }) => {
  const filename = await restoreFixtureState(page);

  await expect(page.locator('#app-container')).toHaveClass(/\bdark\b/);
  expect(await imageHash(page.locator('#image'))).toBe('528b56cf');
  expect(await imageHash(page.locator('#depthmap-image'))).toBe('c6c301c5');
  expect(await imagePixel(page.locator('#image'), 30, 30)).toEqual([240, 50, 45, 255]);
  expect(await imagePixel(page.locator('#image'), 248, 96)).toEqual([35, 210, 90, 255]);
  expect(await imagePixel(page.locator('#depthmap-image'), 0, 0)).toEqual([0, 0, 0, 255]);
  expect(await imagePixel(page.locator('#depthmap-image'), 319, 239)).toEqual([255, 255, 255, 255]);
  await expect(page.locator('#num-slices-slider [role=slider]')).toHaveAttribute('aria-valuenow', '3');
  await expect(page.locator('#thresholds-container [role=slider]')).toHaveCount(2);
  await expect(page.locator('#camera-distance-slider [role=slider]')).toHaveAttribute('aria-valuenow', '125');
  await expect(page.locator('#max-distance-slider [role=slider]')).toHaveAttribute('aria-valuenow', '140');
  await expect(page.locator('#focal-length-slider [role=slider]')).toHaveAttribute('aria-valuenow', '475');
  await expect(page.locator('#displacement-slider [role=slider]')).toHaveAttribute('aria-valuenow', '15');
  expect((await readE2EState(page, filename)).thresholds).toEqual([0, 85, 170, 255]);

  await clickMainTab(page, 'Segmentation');
  await selectSlice(page, filename, 1);
  await clickMainTab(page, 'Inpainting');
  await expect(page.locator('#positive-prompt')).toHaveValue('fixture foreground 1');
  await expect(page.locator('#negative-prompt')).toHaveValue('fixture exclusion 1');

  await clickMainTab(page, 'Configuration');
  await expect(page.locator('#depth-model-dropdown')).toContainText('DINOv2');
  await expect(page.locator('#inpainting-model-dropdown')).toContainText('SD XL 1.0');
});

test('glTF downloads as a valid scene and animation renders four frames without a download', async ({ page }) => {
  const filename = await restoreFixtureState(page);
  await clickMainTab(page, 'Export');
  await setDashSlider(page, 'displacement-slider', 0);

  const downloadPromise = page.waitForEvent('download');
  await page.locator('#gltf-export').click();
  const download = await downloadPromise;
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

  await setDashSlider(page, 'number-of-frames-slider', 4);
  const downloads: string[] = [];
  page.on('download', (event) => downloads.push(event.suggestedFilename()));
  await page.locator('#animation-export').click();
  await expect(page.locator('#log')).toContainText('Exported 4 frames to animation');
  // This locks in the current contract: animation writes server-side frames but the
  // otherwise-present dcc.Download is not populated by the callback.
  expect(downloads).toEqual([]);

  const artifacts = await page.request.get(
    `/__e2e__/artifacts?filename=${encodeURIComponent(filename)}`,
  );
  expect(artifacts.ok()).toBeTruthy();
  const body = (await artifacts.json()) as { files: Array<{ path: string; size: number }> };
  const frames = body.files
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
      const response = await page.request.get(
        `/__e2e__/artifact/${encodeURIComponent(filename)}/${frame.path}`,
      );
      expect(response.ok(), `download ${frame.path}`).toBeTruthy();
      expect(response.headers()['content-type']).toContain('image/png');
      return imageBufferMetadata(page, await response.body());
    }),
  );
  expect(frameMetadata.every((frame) => frame.width === 320 && frame.height === 240)).toBe(true);
  expect(new Set(frameMetadata.map((frame) => frame.hash)).size).toBeGreaterThan(1);
});
