/**
 * Single-page layout: the app owns the viewport. Whatever the image's
 * aspect ratio and whichever workflow tab is open, the page itself never
 * scrolls, and the whole main image stays on screen (fitted, not cropped)
 * with its pixel mapping intact.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, requireWorkflow, test } from './fixtures';
import type { MainTab } from './drivers/types';
import { pageOverflow, setViewport } from './helpers/layout';

const TABS: MainTab[] = ['Mode', 'Segmentation', 'Inpainting', 'Export', 'Configuration'];
const IMAGES = [
  { name: 'tall', file: 'example/input.png', size: [480, 856] },
  { name: 'wide', file: 'example/thumb.png', size: [1920, 1080] },
] as const;
const VIEWPORTS = [
  { width: 1280, height: 720 },
  { width: 1440, height: 900 },
  { width: 768, height: 1024 },
];

for (const image of IMAGES) {
  for (const viewport of VIEWPORTS) {
    test(`a ${image.name} image fits a ${viewport.width}x${viewport.height} window on every tab without page scrolling`, async ({
      page,
      ui,
    }) => {
      requireWorkflow(ui, 'upload-depth-slices');
      await setViewport(page, viewport);
      await ui.goto();
      await ui.uploadImageFile({
        name: `${image.name}.png`,
        mimeType: 'image/png',
        // Paths are relative to the repo root, where `npm run test:e2e` runs.
        buffer: readFileSync(resolve(process.cwd(), image.file)),
      });

      for (const tab of TABS) {
        await ui.openTab(tab);
        expect(await pageOverflow(page), `page overflow on ${tab}`).toEqual({ vertical: 0, horizontal: 0 });
        const box = await ui.mainImage().boundingBox();
        if (!box) throw new Error('Main image has no bounding box');
        expect(box.y, `main image top on ${tab}`).toBeGreaterThanOrEqual(0);
        expect(box.y + box.height, `main image bottom on ${tab}`).toBeLessThanOrEqual(viewport.height);
        expect(box.x + box.width, `main image right edge on ${tab}`).toBeLessThanOrEqual(viewport.width);
        // Fitted, not distorted: the rendered box keeps the source aspect ratio.
        expect(box.width / box.height).toBeCloseTo(image.size[0] / image.size[1], 1);
      }

      // Clicks on the fitted image still map to the source pixel under the
      // cursor. The image is shown smaller than its source here, so one CSS
      // pixel spans several source pixels and a real click (whole CSS
      // pixels) can only land within that span.
      const box = await ui.mainImage().boundingBox();
      if (!box) throw new Error('Main image has no bounding box');
      const sourcePerCssPixel = image.size[0] / box.width;
      const [x, y] = [Math.floor(image.size[0] * 0.25), Math.floor(image.size[1] * 0.8)];
      await ui.clickImagePixel(x, y);
      await expect(ui.log()).toContainText('Click event at pixel coordinates');
      const match = /Click event at pixel coordinates \((\d+), (\d+)\)/.exec(await ui.log().innerText());
      if (!match) throw new Error('No click log line');
      expect(Math.abs(Number(match[1]) - x)).toBeLessThanOrEqual(Math.ceil(sourcePerCssPixel));
      expect(Math.abs(Number(match[2]) - y)).toBeLessThanOrEqual(Math.ceil(sourcePerCssPixel));
    });
  }
}
