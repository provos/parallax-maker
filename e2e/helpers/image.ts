import { expect, type Locator, type Page } from '@playwright/test';

/**
 * Frontend-neutral pixel utilities.
 *
 * These operate on rendered `<img>` elements (via a `Locator`) or raw image
 * bytes decoded in the page, and never depend on any particular frontend's
 * DOM structure or selectors. Any frontend's driver can hand its own
 * `mainImage()`/`depthImage()`/etc. locators to these helpers.
 */

export type RGB = readonly [number, number, number];
export type RGBA = readonly [number, number, number, number];
export type ImageMetadata = { width: number; height: number; hash: string };

/**
 * Waits until the image element has a fully decoded image. It need not be
 * on screen: a UI may keep it in a view or panel that isn't showing (the
 * redesigned UI shows the depth map only in its Depth view).
 */
export async function waitForImage(image: Locator): Promise<void> {
  await expect(image).toBeAttached();
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
    async (element: HTMLImageElement, point): Promise<RGBA> => {
      // `waitForImage` polls in a separate round trip from this evaluate;
      // a `src` reassigned to a fresh revisioned URL between the two (e.g.
      // a thumbnail reloading right after a mutation) can transiently leave
      // `naturalWidth` at 0 again. `decode()` waits for whatever the
      // *current* src resolves to, closing that gap without weakening the
      // assertion itself.
      await element.decode().catch(() => {});
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
    async (element: HTMLImageElement, color): Promise<boolean> => {
      // See imagePixel's comment on `decode()` above.
      await element.decode().catch(() => {});
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
  return image.evaluate(async (element: HTMLImageElement): Promise<string> => {
    // See imagePixel's comment on `decode()` above.
    await element.decode().catch(() => {});
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

export async function sourceImageAlphaPoint(
  page: Page,
  src: string,
  expectedAlpha: number,
): Promise<[number, number]> {
  return page.evaluate(
    async ({ src, expectedAlpha }): Promise<[number, number]> => {
      const image = new Image();
      image.src = src;
      await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) throw new Error('Unable to create 2D context for alpha-point assertion');
      context.drawImage(image, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      for (let offset = 3; offset < pixels.length; offset += 4) {
        if (pixels[offset] === expectedAlpha) {
          const pixelIndex = (offset - 3) / 4;
          return [pixelIndex % canvas.width, Math.floor(pixelIndex / canvas.width)];
        }
      }
      throw new Error(`No pixel with alpha ${expectedAlpha} found`);
    },
    { src, expectedAlpha },
  );
}
