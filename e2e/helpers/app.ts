import { expect, type Locator, type Page } from '@playwright/test';

export type RGB = readonly [number, number, number];
export type RGBA = readonly [number, number, number, number];

export type E2EState = {
  selected_slice: number | null;
  selected_inpainting: number | null;
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

export type E2EMaskStats = {
  present: boolean;
  nonzero: number;
  bounds: [number, number, number, number] | null;
  samples: Record<string, number>;
  inside?: [number, number] | null;
  outside?: [number, number] | null;
  max?: number;
};

export type ImageMetadata = { width: number; height: number; hash: string };

export const sliceImages = (page: Page): Locator =>
  page.locator('img[id*=\'"type":"slice"\']');

export const inpaintingImages = (page: Page): Locator =>
  page.locator('img[id*=\'"type":"inpainting-image"\']');

export async function gotoApp(page: Page): Promise<void> {
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { name: 'Parallax Maker' })).toBeVisible();
}

export async function uploadServerFixture(
  page: Page,
  inputSelector: string,
  endpoint: string,
  name: string,
  mimeType: string,
): Promise<void> {
  const response = await page.request.get(endpoint);
  expect(response.ok(), `GET ${endpoint}`).toBeTruthy();
  await page.locator(inputSelector).setInputFiles({
    name,
    mimeType,
    buffer: await response.body(),
  });
}

export async function uploadInputImage(page: Page): Promise<void> {
  await uploadServerFixture(
    page,
    '#upload-image input[type=file]',
    '/__e2e__/fixture/input.png',
    'e2e-input.png',
    'image/png',
  );
  await waitForImage(page.locator('#image'));
  await waitForImage(page.locator('#depthmap-image'));
}

export async function restoreFixtureState(page: Page): Promise<string> {
  const response = await page.request.get('/__e2e__/fixture/state.json');
  expect(response.ok(), 'GET /__e2e__/fixture/state.json').toBeTruthy();
  const buffer = await response.body();
  const fixture = JSON.parse(buffer.toString('utf8')) as { filename?: unknown };
  expect(typeof fixture.filename).toBe('string');
  await page.locator('#upload-state input[type=file]').setInputFiles({
    name: 'appstate.json',
    mimeType: 'application/json',
    buffer,
  });
  await waitForImage(page.locator('#image'));
  await waitForImage(page.locator('#depthmap-image'));
  await expect(sliceImages(page)).toHaveCount(3);
  return fixture.filename as string;
}

export async function readE2EState(page: Page, filename: string): Promise<E2EState> {
  const response = await page.request.get(
    `/__e2e__/state?filename=${encodeURIComponent(filename)}`,
  );
  expect(response.ok(), 'GET /__e2e__/state').toBeTruthy();
  return response.json();
}

export async function waitForImage(image: Locator): Promise<void> {
  await expect(image).toBeVisible();
  await expect
    .poll(() =>
      image.evaluate((element: HTMLImageElement) => ({
        complete: element.complete,
        width: element.naturalWidth,
        height: element.naturalHeight,
      })),
    )
    .toMatchObject({ complete: true });
  await expect.poll(() => image.evaluate((element: HTMLImageElement) => element.naturalWidth)).toBeGreaterThan(0);
}

export async function imageDimensions(image: Locator): Promise<{ width: number; height: number }> {
  await waitForImage(image);
  return image.evaluate((element: HTMLImageElement) => ({
    width: element.naturalWidth,
    height: element.naturalHeight,
  }));
}

export async function imagePixel(image: Locator, x: number, y: number): Promise<RGBA> {
  await waitForImage(image);
  return image.evaluate(
    (element: HTMLImageElement, point): RGBA => {
      const canvas = document.createElement('canvas');
      canvas.width = element.naturalWidth;
      canvas.height = element.naturalHeight;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) throw new Error('Unable to create 2D context for pixel assertion');
      context.drawImage(element, 0, 0);
      return Array.from(context.getImageData(point.x, point.y, 1, 1).data) as unknown as RGBA;
    },
    { x, y },
  );
}

export async function imageContainsRGB(image: Locator, expected: RGB): Promise<boolean> {
  await waitForImage(image);
  return image.evaluate(
    (element: HTMLImageElement, color): boolean => {
      const canvas = document.createElement('canvas');
      canvas.width = element.naturalWidth;
      canvas.height = element.naturalHeight;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) throw new Error('Unable to create 2D context for pixel assertion');
      context.drawImage(element, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      for (let index = 0; index < pixels.length; index += 4) {
        if (
          pixels[index] === color[0] &&
          pixels[index + 1] === color[1] &&
          pixels[index + 2] === color[2]
        ) {
          return true;
        }
      }
      return false;
    },
    expected,
  );
}

export async function imageSignature(image: Locator, points: readonly [number, number][]): Promise<RGBA[]> {
  return Promise.all(points.map(([x, y]) => imagePixel(image, x, y)));
}

export async function imageHash(image: Locator): Promise<string> {
  await waitForImage(image);
  return image.evaluate((element: HTMLImageElement): string => {
    const canvas = document.createElement('canvas');
    canvas.width = element.naturalWidth;
    canvas.height = element.naturalHeight;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('Unable to create 2D context for image hash');
    context.drawImage(element, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let hash = 0x811c9dc5;
    for (const value of pixels) {
      hash ^= value;
      hash = Math.imul(hash, 0x01000193);
    }
    return (hash >>> 0).toString(16).padStart(8, '0');
  });
}

export async function imageBufferMetadata(page: Page, buffer: Buffer): Promise<ImageMetadata> {
  return page.evaluate(async (base64): Promise<ImageMetadata> => {
    const image = new Image();
    image.src = `data:image/png;base64,${base64}`;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('Unable to create 2D context for artifact image');
    context.drawImage(image, 0, 0);
    let hash = 0x811c9dc5;
    for (const value of context.getImageData(0, 0, canvas.width, canvas.height).data) {
      hash ^= value;
      hash = Math.imul(hash, 0x01000193);
    }
    return {
      width: image.naturalWidth,
      height: image.naturalHeight,
      hash: (hash >>> 0).toString(16).padStart(8, '0'),
    };
  }, buffer.toString('base64'));
}

export async function sourceImagePixel(page: Page, src: string, x: number, y: number): Promise<RGBA> {
  return page.evaluate(
    async ({ src, x, y }): Promise<RGBA> => {
      const image = new Image();
      image.src = src;
      await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) throw new Error('Unable to create 2D context for source image');
      context.drawImage(image, 0, 0);
      return Array.from(context.getImageData(x, y, 1, 1).data) as unknown as RGBA;
    },
    { src, x, y },
  );
}

export async function clickImagePixel(
  image: Locator,
  x: number,
  y: number,
  modifiers: ('Alt' | 'Control' | 'Meta' | 'Shift')[] = [],
): Promise<void> {
  const position = await image.evaluate(
    (element: HTMLImageElement, point) => {
      const rect = element.getBoundingClientRect();
      const scale = Math.min(rect.width / element.naturalWidth, rect.height / element.naturalHeight);
      return { x: point.x * scale, y: point.y * scale };
    },
    { x, y },
  );
  await image.click({ position, modifiers });
}

export async function clickMainTab(page: Page, name: string): Promise<void> {
  const label = page.locator('label').filter({ hasText: new RegExp(`^${name}$`) });
  await expect(label).toHaveCount(1);
  await label.click();
  await expect(label).toHaveClass(/underline/);
}

export async function selectDashOption(page: Page, id: string, name: string): Promise<void> {
  const dropdown = page.locator(`#${id}`);
  await dropdown.click();
  const option = page.getByText(name, { exact: true }).last();
  await expect(option).toBeVisible();
  await option.click();
}

export async function selectSlice(page: Page, filename: string, index: number): Promise<Locator> {
  const image = sliceImages(page).nth(index);
  await expect(image).toBeVisible();
  const bounds = await image.boundingBox();
  if (!bounds) throw new Error(`Slice ${index} has no clickable bounds`);
  // The depth-number overlay covers the center and the label covers the bottom.
  // Click the unobstructed upper-left area with normal browser hit-testing.
  await image.click({ position: { x: bounds.width * 0.1, y: bounds.height * 0.15 } });
  const overlay = page.locator('div[id*=\'"type":"slicer-overlay"\']').nth(index);
  await expect(overlay).toHaveClass(/overlay/);
  await expect.poll(async () => (await readE2EState(page, filename)).selected_slice).toBe(index);
  return image;
}

export async function drawCanvasStroke(page: Page): Promise<void> {
  const canvas = page.locator('#canvas');
  await expect(canvas).toBeVisible();
  const box = await canvas.boundingBox();
  if (!box) throw new Error('Canvas has no bounding box');

  await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.45);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.55, { steps: 10 });
  await page.mouse.up();
  // The client intentionally serializes the mask on mouseout, not mouseup.
  await page.mouse.move(box.x + box.width + 10, box.y + box.height / 2);
  await expect(page.locator('#log')).toContainText(/Saved mask for slice/);
}

export async function setDashSlider(page: Page, id: string, value: number): Promise<void> {
  const slider = page.locator(`#${id} [role=slider]`);
  await slider.focus();
  const min = Number(await slider.getAttribute('aria-valuemin'));
  const max = Number(await slider.getAttribute('aria-valuemax'));
  const step = Number(await slider.getAttribute('aria-valuestep')) || 1;
  if (value < min || value > max || (value - min) % step !== 0) {
    throw new Error(`Value ${value} is not valid for ${id} (${min}..${max}, step ${step})`);
  }
  await slider.press('Home');
  for (let current = min; current < value; current += step) {
    await slider.press('ArrowRight');
  }
  await expect(slider).toHaveAttribute('aria-valuenow', String(value));
}
