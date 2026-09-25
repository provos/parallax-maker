import { expect, type Download, type Locator, type Page } from '@playwright/test';
import { waitForImage } from '../helpers/image';
import { fetchFixture } from '../helpers/oracle';
import type {
  MainTab,
  Modifier,
  SegmentationMode,
  SliderName,
  UiDriver,
  UiTarget,
  Workflow,
} from './types';

/**
 * Svelte-specific implementation of `UiDriver`, built during the migration's
 * first vertical slice (upload -> depth -> thresholds -> slices, plus legacy
 * state restore and the log pane). Every Svelte selector and gesture used by
 * the shared behavioral scenarios lives here; the scenario file itself never
 * mentions a Svelte `data-testid`.
 *
 * Only `upload-depth-slices` is supported so far: every other workflow
 * (segmentation, inpainting, export, ...) is not yet implemented in the new
 * UI (see docs/svelte-migration/PARITY.md), so those methods throw.
 */
export class SvelteDriver implements UiDriver {
  readonly target: UiTarget = 'svelte';

  constructor(private readonly page: Page) {}

  supports(workflow: Workflow): boolean {
    return workflow === 'upload-depth-slices';
  }

  // Navigation

  async goto(): Promise<void> {
    await this.page.goto('/next/', { waitUntil: 'domcontentloaded' });
    await expect(this.page.getByRole('heading', { name: 'Parallax Maker' })).toBeVisible();
  }

  async openTab(tab: MainTab): Promise<void> {
    const button = this.page.getByRole('tab', { name: tab, exact: true });
    await expect(button).toHaveCount(1);
    await button.click();
    await expect(button).toHaveAttribute('aria-selected', 'true');
  }

  // Observable elements

  mainImage(): Locator {
    return this.page.getByTestId('main-image');
  }

  depthImage(): Locator {
    return this.page.getByTestId('depth-image');
  }

  sliceImages(): Locator {
    return this.page.getByTestId('slice-thumbnail');
  }

  candidateImages(): Locator {
    return this.page.getByTestId('candidate-image');
  }

  thresholdHandles(): Locator {
    return this.page.getByTestId('threshold-handle');
  }

  log(): Locator {
    return this.page.getByTestId('log');
  }

  // Upload / depth / slices

  async uploadInputImage(): Promise<void> {
    const response = await fetchFixture(this.page, 'input.png');
    expect(response.ok(), 'GET /__e2e__/fixture/input.png').toBeTruthy();
    await this.page.getByTestId('upload-image-input').setInputFiles({
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

    let input = this.page.getByTestId('restore-state-input');
    if ((await input.count()) === 0) {
      await this.openTab('Configuration');
      input = this.page.getByTestId('restore-state-input');
    }
    await input.setInputFiles({
      name: 'appstate.json',
      mimeType: 'application/json',
      buffer,
    });

    await waitForImage(this.mainImage());
    await waitForImage(this.depthImage());

    // Slice thumbnails are only rendered in the Segmentation tab; check the
    // same way DashDriver does (no tab switch needed there), falling back to
    // opening it here and returning to whichever tab was active before.
    const slices = this.sliceImages();
    if ((await slices.count()) < 3) {
      const previousTab = await this.activeMainTab();
      await this.openTab('Segmentation');
      await expect(slices).toHaveCount(3);
      if (previousTab && previousTab !== 'Segmentation') {
        await this.openTab(previousTab);
      }
    } else {
      await expect(slices).toHaveCount(3);
    }

    return fixture.filename as string;
  }

  async generateSlices(): Promise<void> {
    await this.page.getByTestId('generate-slices').click();
  }

  private async activeMainTab(): Promise<MainTab | null> {
    // Scoped to the workflow tablist specifically: the viewer (2D/3D) tabs
    // also use role="tab" and are also selected by default, so a
    // page-wide `getByRole('tab', { selected: true })` would be ambiguous.
    const active = this.page.locator('[role="tablist"][aria-label="Workflow"] [role="tab"][aria-selected="true"]');
    const count = await active.count();
    if (count === 0) return null;
    const name = await active.first().textContent();
    const trimmed = name?.trim();
    const known: MainTab[] = ['Segmentation', 'Inpainting', 'Export', 'Configuration'];
    return (known.find((tab) => tab === trimmed) as MainTab | undefined) ?? null;
  }

  // Segmentation

  async setSegmentationMode(_mode: SegmentationMode): Promise<void> {
    throw new Error('SvelteDriver: setSegmentationMode not implemented yet');
  }

  async expectSegmentationMode(_mode: SegmentationMode): Promise<void> {
    throw new Error('SvelteDriver: expectSegmentationMode not implemented yet');
  }

  async clickImagePixel(_x: number, _y: number, _modifiers?: Modifier[]): Promise<void> {
    throw new Error('SvelteDriver: clickImagePixel not implemented yet');
  }

  async selectSlice(_projectId: string, _index: number): Promise<Locator> {
    throw new Error('SvelteDriver: selectSlice not implemented yet');
  }

  async toggleMultiPoint(): Promise<void> {
    throw new Error('SvelteDriver: toggleMultiPoint not implemented yet');
  }

  async expectMultiPointEnabled(_enabled: boolean): Promise<void> {
    throw new Error('SvelteDriver: expectMultiPointEnabled not implemented yet');
  }

  async commitMultiPoint(): Promise<void> {
    throw new Error('SvelteDriver: commitMultiPoint not implemented yet');
  }

  // Canvas / inpainting

  async drawMaskStroke(): Promise<void> {
    throw new Error('SvelteDriver: drawMaskStroke not implemented yet');
  }

  async expectGenerateEnabled(): Promise<void> {
    throw new Error('SvelteDriver: expectGenerateEnabled not implemented yet');
  }

  async fillPrompts(_positive: string, _negative: string): Promise<void> {
    throw new Error('SvelteDriver: fillPrompts not implemented yet');
  }

  async expectPrompts(_positive: string, _negative: string): Promise<void> {
    throw new Error('SvelteDriver: expectPrompts not implemented yet');
  }

  async generateInpainting(): Promise<void> {
    throw new Error('SvelteDriver: generateInpainting not implemented yet');
  }

  async fillInpainting(): Promise<void> {
    throw new Error('SvelteDriver: fillInpainting not implemented yet');
  }

  async enhance(): Promise<void> {
    throw new Error('SvelteDriver: enhance not implemented yet');
  }

  async erase(): Promise<void> {
    throw new Error('SvelteDriver: erase not implemented yet');
  }

  async selectCandidate(_index: number): Promise<void> {
    throw new Error('SvelteDriver: selectCandidate not implemented yet');
  }

  async applyCandidate(): Promise<void> {
    throw new Error('SvelteDriver: applyCandidate not implemented yet');
  }

  undoButton(_index: number): Locator {
    throw new Error('SvelteDriver: undoButton not implemented yet');
  }

  redoButton(_index: number): Locator {
    throw new Error('SvelteDriver: redoButton not implemented yet');
  }

  // Project / configuration

  async expectDarkTheme(): Promise<void> {
    throw new Error('SvelteDriver: expectDarkTheme not implemented yet');
  }

  async expectSliderValue(_name: SliderName, _value: number): Promise<void> {
    throw new Error('SvelteDriver: expectSliderValue not implemented yet');
  }

  async setSlider(_name: SliderName, _value: number): Promise<void> {
    throw new Error('SvelteDriver: setSlider not implemented yet');
  }

  async expectDepthModel(_label: string): Promise<void> {
    throw new Error('SvelteDriver: expectDepthModel not implemented yet');
  }

  async expectInpaintingModel(_label: string): Promise<void> {
    throw new Error('SvelteDriver: expectInpaintingModel not implemented yet');
  }

  // Export

  async exportGltf(): Promise<Download> {
    throw new Error('SvelteDriver: exportGltf not implemented yet');
  }

  async exportAnimation(): Promise<void> {
    throw new Error('SvelteDriver: exportAnimation not implemented yet');
  }
}
