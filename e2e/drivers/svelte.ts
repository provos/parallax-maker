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

/**
 * Svelte-specific implementation of `UiDriver`, built during the migration's
 * first vertical slice (upload -> depth -> thresholds -> slices, plus legacy
 * state restore and the log pane). Every Svelte selector and gesture used by
 * the shared behavioral scenarios lives here; the scenario file itself never
 * mentions a Svelte `data-testid`.
 *
 * `upload-depth-slices` and `segmentation` are supported so far: every other
 * workflow (inpainting, export, ...) is not yet implemented in the new UI
 * (see docs/svelte-migration/PARITY.md), so those methods throw.
 */
export class SvelteDriver implements UiDriver {
  readonly target: UiTarget = 'svelte';

  constructor(private readonly page: Page) {}

  supports(workflow: Workflow): boolean {
    return workflow === 'upload-depth-slices' || workflow === 'segmentation';
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
    const known: MainTab[] = ['Mode', 'Segmentation', 'Inpainting', 'Export', 'Configuration'];
    return (known.find((tab) => tab === trimmed) as MainTab | undefined) ?? null;
  }

  /**
   * The Mode Selector lives in the Svelte-only "Mode" workflow tab (not
   * part of the shared `MainTab` type -- Dash keeps its Mode Selector
   * inline, outside any tab). Runs `fn` with that tab visible, using real
   * clicks (no forced actions on a hidden `<select>`), then restores
   * whichever workflow tab was active before.
   */
  private async withModeTabVisible<T>(fn: () => Promise<T>): Promise<T> {
    const tablist = '[role="tablist"][aria-label="Workflow"] [role="tab"]';
    const active = this.page.locator(`${tablist}[aria-selected="true"]`);
    const previousTab = (await active.count()) > 0 ? (await active.first().textContent())?.trim() : null;

    if (previousTab !== 'Mode') {
      const modeButton = this.page.getByRole('tab', { name: 'Mode', exact: true });
      await modeButton.click();
      await expect(modeButton).toHaveAttribute('aria-selected', 'true');
    }
    try {
      return await fn();
    } finally {
      if (previousTab && previousTab !== 'Mode') {
        const button = this.page.getByRole('tab', { name: previousTab, exact: true });
        await button.click();
        await expect(button).toHaveAttribute('aria-selected', 'true');
      }
    }
  }

  // Segmentation

  async setSegmentationMode(mode: SegmentationMode): Promise<void> {
    await this.withModeTabVisible(async () => {
      await this.page.getByTestId('mode-selector').selectOption({ label: mode });
    });
  }

  async expectSegmentationMode(mode: SegmentationMode): Promise<void> {
    await this.withModeTabVisible(async () => {
      const selected = this.page.getByTestId('mode-selector').locator('option:checked');
      await expect(selected).toHaveText(mode);
    });
  }

  async clickImagePixel(x: number, y: number, modifiers: Modifier[] = []): Promise<void> {
    // Same position computation as DashDriver.clickImagePixel: the main
    // image renders at `width: 100%; height: auto` (see
    // InputImagePanel.svelte), so this scale is the exact inverse of the
    // backend's `find_pixel_from_click` ratio, and real click coordinates'
    // sub-pixel rounding truncates the resulting pixel the same way on both
    // UIs (see lib/geometry.ts's `findPixelFromClick`).
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
    const wrapper = this.page.getByTestId('slice-thumbnail-wrapper').nth(index);
    await expect(wrapper).toHaveAttribute('aria-selected', 'true');
    await expect.poll(async () => (await readE2EState(this.page, projectId)).selected_slice).toBe(index);
    return image;
  }

  async toggleMultiPoint(): Promise<void> {
    await this.page.getByTestId('multi-point').click();
  }

  async expectMultiPointEnabled(enabled: boolean): Promise<void> {
    await expect(this.page.getByTestId('multi-point')).toHaveAttribute('aria-pressed', String(enabled));
  }

  async commitMultiPoint(): Promise<void> {
    await this.page.getByTestId('multi-commit').click();
    await expect(this.log()).toContainText(/Committed points/);
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

  // Slice editing / mask tools

  async createSlice(): Promise<void> {
    throw new Error('SvelteDriver: createSlice not implemented yet');
  }

  async deleteSlice(): Promise<void> {
    throw new Error('SvelteDriver: deleteSlice not implemented yet');
  }

  async addMaskToSlice(): Promise<void> {
    throw new Error('SvelteDriver: addMaskToSlice not implemented yet');
  }

  async removeMaskFromSlice(): Promise<void> {
    throw new Error('SvelteDriver: removeMaskFromSlice not implemented yet');
  }

  async copySlice(): Promise<void> {
    throw new Error('SvelteDriver: copySlice not implemented yet');
  }

  async pasteSlice(): Promise<void> {
    throw new Error('SvelteDriver: pasteSlice not implemented yet');
  }

  async balanceSlices(): Promise<void> {
    throw new Error('SvelteDriver: balanceSlices not implemented yet');
  }

  async setSliceDepth(_index: number, _depth: number): Promise<void> {
    throw new Error('SvelteDriver: setSliceDepth not implemented yet');
  }

  async uploadSliceImage(
    _index: number,
    _file: { name: string; mimeType: string; buffer: Buffer },
  ): Promise<void> {
    throw new Error('SvelteDriver: uploadSliceImage not implemented yet');
  }

  async invertMask(): Promise<void> {
    throw new Error('SvelteDriver: invertMask not implemented yet');
  }

  async featherMask(): Promise<void> {
    throw new Error('SvelteDriver: featherMask not implemented yet');
  }

  async toggleCheckerboard(): Promise<void> {
    throw new Error('SvelteDriver: toggleCheckerboard not implemented yet');
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
