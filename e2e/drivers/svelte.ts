import { expect, type Download, type Locator, type Page } from '@playwright/test';
import { waitForImage } from '../helpers/image';
import { fetchFixture, readE2EState } from '../helpers/oracle';
import type {
  CameraDirection,
  MainTab,
  Modifier,
  SegmentationMode,
  SliderName,
  UiDriver,
  UiTarget,
  Workflow,
} from './types';

// Every slider's `data-testid` equals its `SliderName` directly (see
// ConfigurationTab.svelte/ExportTab.svelte) - unlike DashDriver, no lookup
// table of distinct DOM ids is needed.

/**
 * Svelte-specific implementation of `UiDriver`. Every Svelte selector and
 * gesture used by the shared behavioral scenarios lives here; the scenario
 * file itself never mentions a Svelte `data-testid`. All workflows are
 * implemented (see docs/svelte-migration/PARITY.md); `supports()` always
 * returns `true`.
 */
/** The workflow step whose Inspector panel is each former main tab. */
const STEP_FOR_TAB: Record<Exclude<MainTab, 'Configuration'>, string> = {
  Mode: 'depth',
  Segmentation: 'slices',
  Inpainting: 'inpaint',
  Export: 'export',
};

export class SvelteDriver implements UiDriver {
  readonly target: UiTarget = 'svelte';

  constructor(private readonly page: Page) {}

  supports(_workflow: Workflow): boolean {
    return true;
  }

  // Navigation

  async goto(): Promise<void> {
    await this.page.goto('/', { waitUntil: 'domcontentloaded' });
    await expect(this.page.getByRole('heading', { name: 'Parallax Maker' })).toBeVisible();
  }

  /**
   * The redesigned UI has no workflow tabs: each former tab is a workflow
   * step's Inspector panel, and Configuration is the Settings panel.
   */
  async openTab(tab: MainTab): Promise<void> {
    const inspector = this.page.getByTestId('inspector');
    if (tab === 'Configuration') {
      if ((await inspector.getAttribute('data-panel')) !== 'Configuration') {
        await this.page.getByTestId('open-settings').click();
      }
    } else {
      await this.page.getByTestId(`step-${STEP_FOR_TAB[tab]}`).click();
    }
    await expect(inspector).toHaveAttribute('data-panel', tab);
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

  async uploadImageFile(file: { name: string; mimeType: string; buffer: Buffer }): Promise<void> {
    await this.page.getByTestId('upload-image-input').setInputFiles(file);
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
    const panel = await this.page.getByTestId('inspector').getAttribute('data-panel');
    const known: MainTab[] = ['Mode', 'Segmentation', 'Inpainting', 'Export', 'Configuration'];
    return (known.find((tab) => tab === panel) as MainTab | undefined) ?? null;
  }

  /**
   * The Mode Selector lives in the Depth step's Inspector panel (Dash kept
   * it inline, outside any tab). Runs `fn` with that panel visible, using
   * real clicks (no forced actions on a hidden `<select>`), then restores
   * whichever panel was showing before.
   */
  private async withModeTabVisible<T>(fn: () => Promise<T>): Promise<T> {
    const previousTab = await this.activeMainTab();
    if (previousTab !== 'Mode') await this.openTab('Mode');
    try {
      return await fn();
    } finally {
      if (previousTab && previousTab !== 'Mode') await this.openTab(previousTab);
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

  /**
   * Zooms in once via a real wheel gesture over the main image's center
   * (`InputImagePanel.svelte`'s `onWheel`/`state/viewport.svelte.ts`), the
   * same gesture Dash's own JS-03 `handleWheel` responds to.
   */
  async zoomIn(): Promise<void> {
    const box = await this.page.getByTestId('input-image-panel').boundingBox();
    if (!box) throw new Error('Input image panel has no bounding box');
    await this.page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await this.page.mouse.wheel(0, -400); // negative deltaY zooms in, same sign Dash checks.
  }

  /**
   * Pans by a real middle-mouse drag over the main image
   * (`InputImagePanel.svelte`'s `onDropZonePointerDown`/`Move`/`Up`), which
   * always pans regardless of the active tab (see viewport.svelte.ts).
   */
  async panBy(dx: number, dy: number): Promise<void> {
    const box = await this.page.getByTestId('input-image-panel').boundingBox();
    if (!box) throw new Error('Input image panel has no bounding box');
    const startX = box.x + box.width / 2;
    const startY = box.y + box.height / 2;
    await this.page.mouse.move(startX, startY);
    await this.page.mouse.down({ button: 'middle' });
    await this.page.mouse.move(startX + dx, startY + dy, { steps: 10 });
    await this.page.mouse.up({ button: 'middle' });
  }

  async resetZoom(): Promise<void> {
    await this.page.getByTestId('zoom-reset').click();
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

  async expectQueuedPointMarkers(
    points: Array<{ x: number; y: number; negative: boolean }>,
  ): Promise<void> {
    const markers = this.page.getByTestId('queued-point-marker');
    await expect(markers).toHaveCount(points.length);
    for (let i = 0; i < points.length; i += 1) {
      const marker = markers.nth(i);
      const isNegative = await marker.evaluate((el) => el.classList.contains('negative'));
      expect(isNegative, `marker ${i} color`).toBe(points[i].negative);
    }
  }

  // Canvas / inpainting

  /**
   * Paints one stroke on the mask canvas (`data-testid="mask-canvas"`,
   * MaskCanvas.svelte). Same relative stroke geometry as DashDriver's own
   * `drawMaskStroke` (a diagonal drag from ~40%/45% to ~60%/55% of the
   * canvas box), but the Svelte canvas saves on pointerup directly -- no
   * need for DashDriver's trailing "move the mouse off the canvas" step,
   * since the client here does not wait for a `mouseout` to persist (see
   * MaskCanvas.svelte's doc comment and the migration handoff's
   * "Deterministic harness details"). `page.mouse` dispatches real mouse
   * input, which Chromium also synthesizes into the `pointerdown`/
   * `pointermove`/`pointerup` events the canvas actually listens for.
   */
  async drawMaskStroke(): Promise<void> {
    const canvas = this.page.getByTestId('mask-canvas');
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas has no bounding box');

    // A stroke must never reach the upload drop zone behind the canvas
    // (headless Chromium opens its file chooser silently).
    let fileChooserOpened = false;
    const onFileChooser = () => (fileChooserOpened = true);
    this.page.on('filechooser', onFileChooser);
    try {
      await this.page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.45);
      await this.page.mouse.down();
      await this.page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.55, { steps: 10 });
      await this.page.mouse.up();
      await expect(this.log()).toContainText(/Saved mask for slice/);
    } finally {
      this.page.off('filechooser', onFileChooser);
    }
    expect(fileChooserOpened, 'painting opened the upload file chooser').toBe(false);
  }

  async maskCanvasPainted(): Promise<boolean> {
    return this.page.getByTestId('mask-canvas').evaluate((canvas: HTMLCanvasElement) => {
      const ctx = canvas.getContext('2d');
      if (!ctx || canvas.width === 0 || canvas.height === 0) return false;
      const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
      for (let i = 3; i < data.length; i += 4) if (data[i] > 0) return true;
      return false;
    });
  }

  async expectGenerateEnabled(): Promise<void> {
    await expect(this.page.getByTestId('generate-inpainting')).toBeEnabled();
  }

  async fillPrompts(positive: string, negative: string): Promise<void> {
    await this.page.getByTestId('positive-prompt').fill(positive);
    await this.page.getByTestId('negative-prompt').fill(negative);
  }

  async expectPrompts(positive: string, negative: string): Promise<void> {
    await expect(this.page.getByTestId('positive-prompt')).toHaveValue(positive);
    await expect(this.page.getByTestId('negative-prompt')).toHaveValue(negative);
  }

  async generateInpainting(): Promise<void> {
    // Deliberately does not wait for candidates: scenarios assert counts themselves.
    await this.page.getByTestId('generate-inpainting').click();
  }

  async fillInpainting(): Promise<void> {
    await this.page.getByTestId('fill-inpainting').click();
  }

  async enhance(): Promise<void> {
    await this.page.getByTestId('enhance-inpainting').click();
  }

  async erase(): Promise<void> {
    await this.page.getByTestId('erase-inpainting').click();
  }

  async selectCandidate(index: number): Promise<void> {
    const candidate = this.candidateImages().nth(index);
    await candidate.click();
    await expect(candidate).toHaveAttribute('aria-selected', 'true');
    await expect(this.page.getByTestId('apply-inpainting')).toBeEnabled();
  }

  async applyCandidate(): Promise<void> {
    await this.page.getByTestId('apply-inpainting').click();
  }

  undoButton(index: number): Locator {
    return this.page.getByTestId('slice-undo').nth(index);
  }

  redoButton(index: number): Locator {
    return this.page.getByTestId('slice-redo').nth(index);
  }

  // Slice editing / mask tools

  /**
   * Clicks `locator` and waits until the log's text differs from its value
   * beforehand. Mirrors DashDriver.clickAndWaitForLogChange: every
   * slice-editing/mask-tool action produces exactly one log line on every
   * path (a client-side no-op message or a server-side success message), so
   * this generically proves the click was processed; scenarios assert the
   * resulting text themselves.
   */
  private async clickAndWaitForLogChange(locator: Locator): Promise<void> {
    const before = await this.log().innerText();
    await locator.click();
    await expect.poll(() => this.log().innerText()).not.toBe(before);
  }

  async createSlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.getByTestId('create-slice'));
  }

  async deleteSlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.getByTestId('delete-slice'));
  }

  async addMaskToSlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.getByTestId('add-mask-to-slice'));
  }

  async removeMaskFromSlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.getByTestId('remove-mask-from-slice'));
  }

  async copySlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.getByTestId('copy-slice'));
  }

  async pasteSlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.getByTestId('paste-slice'));
  }

  async balanceSlices(): Promise<void> {
    await this.page.getByTestId('balance-slices').click();
  }

  async setSliceDepth(index: number, depth: number): Promise<void> {
    const display = this.page.getByTestId('slice-depth-display').nth(index);
    await expect(display).toBeVisible();
    await display.click();
    const input = this.page.getByTestId('slice-depth-input');
    await expect(input).toBeVisible();
    await input.fill(String(depth));
    await input.press('Enter');
    // The whole thumbnail strip is rebuilt (and may reorder) once the new
    // depth is committed; wait for a depth badge to show the committed value
    // anywhere in the (possibly reordered) strip rather than trusting
    // `index` to still point at the same slice.
    await expect(
      this.page.getByTestId('slice-depth-display').filter({ hasText: new RegExp(`^${depth}$`) }),
    ).not.toHaveCount(0);
  }

  async uploadSliceImage(
    index: number,
    file: { name: string; mimeType: string; buffer: Buffer },
  ): Promise<void> {
    const before = await this.log().innerText();
    const input = this.page.getByTestId('slice-upload-input').nth(index);
    await input.setInputFiles(file);
    await expect.poll(() => this.log().innerText()).not.toBe(before);
  }

  async invertMask(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.getByTestId('invert-mask'));
  }

  async featherMask(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.getByTestId('feather-mask'));
  }

  async toggleCheckerboard(): Promise<void> {
    const button = this.page.getByTestId('toggle-checkerboard');
    const wasSelected = (await button.getAttribute('aria-pressed')) === 'true';
    await button.click();
    await expect(button).toHaveAttribute('aria-pressed', String(!wasSelected));
  }

  // Project / configuration

  async expectDarkTheme(): Promise<void> {
    await expect(this.page.locator('html')).toHaveAttribute('data-theme', 'dark');
  }

  async expectSliderValue(name: SliderName, value: number): Promise<void> {
    const input = this.page.getByTestId(name);
    await expect(input).toHaveValue(String(value));
  }

  /**
   * Native `<input type=range>` elements (unlike Dash's rc-slider) have no
   * `aria-valuestep` quirk to work around, but the same "step directly from
   * the current value, never via Home" rationale from DashDriver.setSlider
   * still applies here (ExportTab.svelte's camera/displacement sliders are
   * committed together - see its own doc comment), so this mirrors that
   * driver's technique with real keyboard events (`ArrowRight`/`ArrowLeft`)
   * rather than any synthetic value assignment.
   */
  async setSlider(name: SliderName, value: number): Promise<void> {
    const input = this.page.getByTestId(name);
    await input.focus();
    const min = Number(await input.getAttribute('min'));
    const max = Number(await input.getAttribute('max'));
    if (value < min || value > max) {
      throw new Error(`Value ${value} is out of range for ${name} (${min}..${max})`);
    }

    let current = Number(await input.inputValue());
    let guard = 0;
    while (current !== value && guard < 2000) {
      await input.press(current < value ? 'ArrowRight' : 'ArrowLeft');
      const next = Number(await input.inputValue());
      if (next === current) break;
      current = next;
      guard += 1;
    }
    if (current !== value) {
      throw new Error(`Could not step ${name} to ${value}; stuck at ${current}`);
    }
  }

  async toggleGroundPlane(): Promise<void> {
    const toggle = this.page.getByTestId('ground-toggle');
    const before = await toggle.getAttribute('aria-pressed');
    await toggle.click();
    await expect(toggle).not.toHaveAttribute('aria-pressed', before ?? 'false');
  }

  async expectGroundSlice(index: number, isGround: boolean): Promise<void> {
    const badge = this.page.getByTestId('slice-thumbnail-wrapper').nth(index).getByTestId('ground-badge');
    await expect(badge).toHaveCount(isGround ? 1 : 0);
  }

  async fitGround(): Promise<void> {
    await this.page.getByTestId('ground-fit').click();
    await expect(this.log()).toContainText('Fitted the ground plane');
  }

  async dragHorizonTo(row: number): Promise<void> {
    const toggle = this.page.getByTestId('horizon-toggle');
    if ((await toggle.getAttribute('aria-pressed')) !== 'true') await toggle.click();
    const line = this.page.getByTestId('horizon-line');
    await expect(line).toBeVisible();
    const lineBox = await line.boundingBox();
    const image = this.mainImage();
    const target = await image.evaluate(
      (element: HTMLImageElement, sourceRow) => {
        const rect = element.getBoundingClientRect();
        return rect.top + (sourceRow / element.naturalHeight) * rect.height;
      },
      row,
    );
    if (!lineBox) throw new Error('Horizon line has no bounding box');
    const x = lineBox.x + lineBox.width / 2;
    await this.page.mouse.move(x, lineBox.y + lineBox.height / 2);
    await this.page.mouse.down();
    await this.page.mouse.move(x, target, { steps: 8 });
    await this.page.mouse.up();
    // Committed: the server's horizon row now matches (within a CSS pixel's worth of rows).
    await expect
      .poll(async () => Math.abs(Number(await line.getAttribute('data-row')) - row))
      .toBeLessThanOrEqual(3);
  }

  async navigateCamera(direction: CameraDirection): Promise<void> {
    const before = await this.log().innerText();
    await this.page.getByTestId(`camera-${direction}`).click();
    await expect.poll(async () => (await this.log().innerText()) !== before).toBe(true);
  }

  async expectDepthModel(label: string): Promise<void> {
    const selected = this.page.getByTestId('depth-model').locator('option:checked');
    await expect(selected).toHaveText(label);
  }

  async expectInpaintingModel(label: string): Promise<void> {
    const selected = this.page.getByTestId('inpainting-model').locator('option:checked');
    await expect(selected).toHaveText(label);
  }

  async selectDepthModel(label: string): Promise<void> {
    await this.withModeTabVisible(async () => {
      await this.page.getByTestId('depth-model').selectOption({ label });
    });
  }

  async selectInpaintingModel(label: string): Promise<void> {
    await this.page.getByTestId('inpainting-model').selectOption({ label });
  }

  async setExternalServer(address: string): Promise<void> {
    const input = this.page.getByTestId('external-server-address');
    await input.fill(address);
    // Commits on blur (ConfigurationTab.svelte's `onchange`), same as
    // DashDriver's debounced field.
    await input.press('Tab');
  }

  async testExternalConnection(): Promise<void> {
    await this.page.getByTestId('external-test-connection').click();
  }

  private async expectStatus(
    locator: Locator,
    status: 'success' | 'failure' | 'none',
  ): Promise<void> {
    await expect(locator).toHaveAttribute('data-status', status);
  }

  async expectExternalConnectionStatus(status: 'success' | 'failure' | 'none'): Promise<void> {
    await this.expectStatus(this.page.getByTestId('external-server-address'), status);
  }

  async setApiKey(key: string): Promise<void> {
    const input = this.page.getByTestId('api-key');
    await input.fill(key);
    await input.press('Tab');
  }

  async validateApiKey(): Promise<void> {
    await this.page.getByTestId('validate-api-key').click();
  }

  async expectApiKeyStatus(status: 'success' | 'failure' | 'none'): Promise<void> {
    await this.expectStatus(this.page.getByTestId('api-key'), status);
  }

  // Project lifecycle

  async saveState(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.getByTestId('save-state'));
  }

  async restoreStateFromBytes(buffer: Buffer): Promise<void> {
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
  }

  // Export

  async exportGltf(): Promise<Download> {
    const downloadPromise = this.page.waitForEvent('download');
    await this.page.getByTestId('gltf-export').click();
    return downloadPromise;
  }

  async exportAnimation(): Promise<void> {
    await this.page.getByTestId('animation-export').click();
  }

  async setDofEnabled(enabled: boolean): Promise<void> {
    const checkbox = this.page.getByTestId('toggle-dof');
    if ((await checkbox.isChecked()) !== enabled) {
      await checkbox.click();
    }
    await expect(checkbox).toBeChecked({ checked: enabled });
  }

  async upscaleTextures(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.getByTestId('upscale-textures'));
  }

  async downloadSlice(index: number): Promise<Download> {
    const downloadPromise = this.page.waitForEvent('download');
    await this.page.getByTestId('slice-download').nth(index).click();
    return downloadPromise;
  }
}
