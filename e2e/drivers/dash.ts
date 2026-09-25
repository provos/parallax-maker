import { expect, type Download, type Locator, type Page } from '@playwright/test';
import { waitForImage } from '../helpers/image';
import { fetchFixture, readE2EState } from '../helpers/oracle';
import type {
  MainTab,
  Modifier,
  SegmentationMode,
  SliderName,
  UiDriver,
  UiTarget,
  Workflow,
} from './types';

const SLIDER_IDS: Record<SliderName, string> = {
  'num-slices': 'num-slices-slider',
  'camera-distance': 'camera-distance-slider',
  'max-distance': 'max-distance-slider',
  'focal-length': 'focal-length-slider',
  displacement: 'displacement-slider',
  'number-of-frames': 'number-of-frames-slider',
};

/**
 * Dash-specific implementation of `UiDriver`. Every Dash selector and
 * gesture used by the shared behavioral scenarios lives here; the scenario
 * file itself never mentions a Dash ID or class.
 */
export class DashDriver implements UiDriver {
  readonly target: UiTarget = 'dash';

  constructor(private readonly page: Page) {}

  supports(_workflow: Workflow): boolean {
    return true;
  }

  // Navigation

  async goto(): Promise<void> {
    await this.page.goto('/', { waitUntil: 'domcontentloaded' });
    await expect(this.page.getByRole('heading', { name: 'Parallax Maker' })).toBeVisible();
  }

  async openTab(tab: MainTab): Promise<void> {
    const label = this.page.locator('label').filter({ hasText: new RegExp(`^${tab}$`) });
    await expect(label).toHaveCount(1);
    await label.click();
    await expect(label).toHaveClass(/underline/);
  }

  // Observable elements

  mainImage(): Locator {
    return this.page.locator('#image');
  }

  depthImage(): Locator {
    return this.page.locator('#depthmap-image');
  }

  sliceImages(): Locator {
    return this.page.locator('img[id*=\'"type":"slice"\']');
  }

  candidateImages(): Locator {
    return this.page.locator('img[id*=\'"type":"inpainting-image"\']');
  }

  thresholdHandles(): Locator {
    return this.page.locator('#thresholds-container [role=slider]');
  }

  log(): Locator {
    return this.page.locator('#log');
  }

  // Upload / depth / slices

  async uploadInputImage(): Promise<void> {
    const response = await fetchFixture(this.page, 'input.png');
    expect(response.ok(), 'GET /__e2e__/fixture/input.png').toBeTruthy();
    await this.page.locator('#upload-image input[type=file]').setInputFiles({
      name: 'e2e-input.png',
      mimeType: 'image/png',
      buffer: await response.body(),
    });
    await waitForImage(this.mainImage());
    await waitForImage(this.depthImage());
  }

  async restoreFixtureState(): Promise<string> {
    const response = await fetchFixture(this.page, 'state.json');
    expect(response.ok(), 'GET /__e2e__/fixture/state.json').toBeTruthy();
    const buffer = await response.body();
    const fixture = JSON.parse(buffer.toString('utf8')) as { filename?: unknown };
    expect(typeof fixture.filename).toBe('string');
    await this.page.locator('#upload-state input[type=file]').setInputFiles({
      name: 'appstate.json',
      mimeType: 'application/json',
      buffer,
    });
    await waitForImage(this.mainImage());
    await waitForImage(this.depthImage());
    await expect(this.sliceImages()).toHaveCount(3);
    return fixture.filename as string;
  }

  async generateSlices(): Promise<void> {
    await this.page.locator('#generate-slice-button').click();
  }

  // Segmentation

  async setSegmentationMode(mode: SegmentationMode): Promise<void> {
    const dropdown = this.page.locator('#mode-selector');
    await dropdown.click();
    const option = this.page.getByText(mode, { exact: true }).last();
    await expect(option).toBeVisible();
    await option.click();
  }

  async expectSegmentationMode(mode: SegmentationMode): Promise<void> {
    await expect(this.page.locator('#mode-selector')).toContainText(mode);
  }

  async clickImagePixel(x: number, y: number, modifiers: Modifier[] = []): Promise<void> {
    const image = this.mainImage();
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

  async selectSlice(projectId: string, index: number): Promise<Locator> {
    const image = this.sliceImages().nth(index);
    await expect(image).toBeVisible();
    const bounds = await image.boundingBox();
    if (!bounds) throw new Error(`Slice ${index} has no clickable bounds`);
    // The depth-number overlay covers the center and the label covers the bottom.
    // Click the unobstructed upper-left area with normal browser hit-testing.
    await image.click({ position: { x: bounds.width * 0.1, y: bounds.height * 0.15 } });
    const overlay = this.page.locator('div[id*=\'"type":"slicer-overlay"\']').nth(index);
    await expect(overlay).toHaveClass(/overlay/);
    await expect.poll(async () => (await readE2EState(this.page, projectId)).selected_slice).toBe(index);
    return image;
  }

  async toggleMultiPoint(): Promise<void> {
    await this.page.locator('#multi-point').click();
  }

  async expectMultiPointEnabled(enabled: boolean): Promise<void> {
    const multiPoint = this.page.locator('#multi-point');
    await expect(multiPoint).toHaveClass(enabled ? /color-is-selected/ : /color-not-selected/);
  }

  async commitMultiPoint(): Promise<void> {
    await this.page.locator('#multi-commit').click();
    await expect(this.log()).toContainText(/Committed points/);
  }

  // Canvas / inpainting

  async drawMaskStroke(): Promise<void> {
    const canvas = this.page.locator('#canvas');
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas has no bounding box');

    await this.page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.45);
    await this.page.mouse.down();
    await this.page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.55, { steps: 10 });
    await this.page.mouse.up();
    // The client intentionally serializes the mask on mouseout, not mouseup.
    await this.page.mouse.move(box.x + box.width + 10, box.y + box.height / 2);
    await expect(this.log()).toContainText(/Saved mask for slice/);
  }

  async expectGenerateEnabled(): Promise<void> {
    await expect(this.page.locator('#generate-inpainting-button')).toBeEnabled();
  }

  async fillPrompts(positive: string, negative: string): Promise<void> {
    await this.page.locator('#positive-prompt').fill(positive);
    await this.page.locator('#negative-prompt').fill(negative);
  }

  async expectPrompts(positive: string, negative: string): Promise<void> {
    await expect(this.page.locator('#positive-prompt')).toHaveValue(positive);
    await expect(this.page.locator('#negative-prompt')).toHaveValue(negative);
  }

  async generateInpainting(): Promise<void> {
    // Deliberately does not wait for candidates: scenarios assert counts themselves.
    await this.page.locator('#generate-inpainting-button').click();
  }

  async fillInpainting(): Promise<void> {
    await this.page.locator('#fill-inpainting-button').click();
  }

  async enhance(): Promise<void> {
    await this.page.locator('#enhance-button').click();
  }

  async erase(): Promise<void> {
    await this.page.locator('#erase-inpainting-button').click();
  }

  async selectCandidate(index: number): Promise<void> {
    const candidate = this.candidateImages().nth(index);
    await candidate.click();
    await expect(candidate).toHaveClass(/color-is-selected-light/);
    await expect(this.page.locator('#apply-inpainting-button')).toBeEnabled();
  }

  async applyCandidate(): Promise<void> {
    await this.page.locator('#apply-inpainting-button').click();
  }

  undoButton(index: number): Locator {
    return this.page.locator('[title="Undo last change"]').nth(index);
  }

  redoButton(index: number): Locator {
    return this.page.locator('[title="Redo last change"]').nth(index);
  }

  // Project / configuration

  async expectDarkTheme(): Promise<void> {
    await expect(this.page.locator('#app-container')).toHaveClass(/\bdark\b/);
  }

  async expectSliderValue(name: SliderName, value: number): Promise<void> {
    const slider = this.page.locator(`#${SLIDER_IDS[name]} [role=slider]`);
    await expect(slider).toHaveAttribute('aria-valuenow', String(value));
  }

  async setSlider(name: SliderName, value: number): Promise<void> {
    const id = SLIDER_IDS[name];
    const slider = this.page.locator(`#${id} [role=slider]`);
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

  async expectDepthModel(label: string): Promise<void> {
    await expect(this.page.locator('#depth-model-dropdown')).toContainText(label);
  }

  async expectInpaintingModel(label: string): Promise<void> {
    await expect(this.page.locator('#inpainting-model-dropdown')).toContainText(label);
  }

  // Export

  async exportGltf(): Promise<Download> {
    const downloadPromise = this.page.waitForEvent('download');
    await this.page.locator('#gltf-export').click();
    return downloadPromise;
  }

  async exportAnimation(): Promise<void> {
    await this.page.locator('#animation-export').click();
  }
}
