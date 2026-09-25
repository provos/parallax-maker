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

  /**
   * A real wheel gesture over the input-image box (utility.js's
   * `handleWheel`/JS-03, the only zoom mechanism Dash has -- no buttons, no
   * drag-to-pan, no reset; see `panBy`/`resetZoom` below and
   * `frontend/src/lib/state/viewport.svelte.ts`'s own doc comment).
   *
   * Confirmed empirically (see docs/svelte-migration/PARITY.md's Known
   * quirks): the wheel *listener* is only attached lazily, the first time
   * the mouse enters or presses down on `#canvas` (`setupMainCanvas`, called
   * from `canvas_draw`'s `mouseenter` case) -- and even once attached, a
   * wheel event only reaches it while `#canvas` is the topmost element under
   * the cursor, which is only true while the Inpainting tab is active
   * (`update_events`/CMP-01 swaps `#image`/`#canvas` z-index per tab). A
   * caller that wants this to actually zoom must `openTab('Inpainting')`
   * first; `hover()` here both moves the mouse (lazily registering the
   * listener if needed) and primes real hit-testing before the wheel tick.
   */
  async zoomIn(): Promise<void> {
    const box = await this.mainImage().boundingBox();
    if (!box) throw new Error('Main image has no bounding box');
    // A raw `page.mouse.move` (not `locator.hover`, which requires the
    // *image* itself to be the topmost element at that point): while the
    // Inpainting tab is active, `#canvas` is deliberately on top of
    // `#image` (see this method's own doc comment above), so hovering
    // "over the image" for a real user means hovering that same screen
    // position, whichever element real hit-testing currently resolves it
    // to - exactly what `page.mouse.move` does.
    await this.page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await this.page.mouse.wheel(0, -400); // negative deltaY zooms in, same sign Dash checks.
  }

  /**
   * A real left-button drag gesture over the input-image box. Dash has no
   * drag-to-pan handler at all for `#image`/`#canvas` (utility.js has no
   * `mousedown`-driven pan of any kind -- only `startDrawing`/`draw`, which
   * paint on `#canvas` and are gated on `currentSlice`/the Inpainting tab,
   * and the unrelated `NAV_*` 3D-camera-dolly buttons, see PARITY.md). This
   * is therefore a real, expected no-op on Dash: the gesture is performed
   * faithfully, but nothing in Dash responds to it.
   */
  async panBy(dx: number, dy: number): Promise<void> {
    const box = await this.mainImage().boundingBox();
    if (!box) throw new Error('Main image has no bounding box');
    const startX = box.x + box.width / 2;
    const startY = box.y + box.height / 2;
    await this.page.mouse.move(startX, startY);
    await this.page.mouse.down();
    await this.page.mouse.move(startX + dx, startY + dy, { steps: 10 });
    await this.page.mouse.up();
  }

  /**
   * Dash has no reset control for its CSS zoom at all (see `zoomIn`'s doc
   * comment); this is a documented best-effort substitute (repeated
   * zoom-out gestures towards JS-03's own 0.125x floor), not an exact reset
   * to 1x/no-pan the way SvelteDriver's Reset button is.
   */
  async resetZoom(): Promise<void> {
    const box = await this.mainImage().boundingBox();
    if (!box) throw new Error('Main image has no bounding box');
    await this.page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    for (let i = 0; i < 40; i += 1) {
      await this.page.mouse.wheel(0, 400);
    }
  }

  /**
   * Samples `#preview-canvas`'s own pixels for Dash's CLI-05
   * `visualize_point` dots (drawn directly onto the canvas, not as DOM
   * nodes -- there is nothing to query by position/class the way
   * SvelteDriver's `queued-point-marker` elements allow). This only checks
   * that a pixel of each expected color exists *somewhere* on the canvas,
   * not at an exact position -- replicating utility.js's own
   * zoom/pan-aware `translateCoordinates` inverse-transform purely to
   * locate a test pixel would duplicate real app logic in the test suite
   * itself; the position itself is already proven correct by the
   * depth-mode click assertions elsewhere in this scenario.
   */
  async expectQueuedPointMarkers(
    points: Array<{ x: number; y: number; negative: boolean }>,
  ): Promise<void> {
    const canvas = this.page.locator('#preview-canvas');
    const sample = async () =>
      canvas.evaluate((el: HTMLCanvasElement) => {
        const ctx = el.getContext('2d');
        if (!ctx || el.width === 0 || el.height === 0) return { green: false, red: false };
        const { data } = ctx.getImageData(0, 0, el.width, el.height);
        let green = false;
        let red = false;
        for (let i = 0; i < data.length; i += 4) {
          const [r, g, b, a] = [data[i], data[i + 1], data[i + 2], data[i + 3]];
          if (a < 200) continue;
          if (g > 200 && r < 60 && b < 60) green = true;
          else if (r > 200 && g < 60 && b < 60) red = true;
        }
        return { green, red };
      });
    const wantsPositive = points.some((p) => !p.negative);
    const wantsNegative = points.some((p) => p.negative);
    // CLI-05 `visualize_point` draws each dot from a *clientside* callback
    // reacting to `STORE_CLICKED_POINT.data`, itself only set after the
    // click's own server round trip lands - unlike SvelteDriver's DOM-count
    // assertion (which Playwright's own `toHaveCount` already retries),
    // there is no single locator to poll here, so this polls the sampled
    // canvas colors directly instead of a one-shot read.
    if (wantsPositive) {
      await expect.poll(async () => (await sample()).green, 'a green (positive) queued-point dot').toBe(
        true,
      );
    }
    if (wantsNegative) {
      await expect.poll(async () => (await sample()).red, 'a red (negative) queued-point dot').toBe(
        true,
      );
    }
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

  // Slice editing / mask tools

  /**
   * Clicks `locator` and waits until the log's text differs from its value
   * beforehand. Every slice-editing/mask-tool callback appends exactly one
   * log line on every path (success and every logged no-op alike), so this
   * generically proves the click was processed without baking in which
   * specific outcome (success vs. a particular no-op reason) occurred -
   * scenarios assert the resulting text themselves.
   */
  private async clickAndWaitForLogChange(locator: Locator): Promise<void> {
    const before = await this.log().innerText();
    await locator.click();
    await expect.poll(() => this.log().innerText()).not.toBe(before);
  }

  async createSlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.locator('#create-slice-button'));
  }

  async deleteSlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.locator('#delete-slice-button'));
  }

  async addMaskToSlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.locator('#add-to-slice-button'));
  }

  async removeMaskFromSlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.locator('#remove-from-slice-button'));
  }

  async copySlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.locator('#copy-button'));
  }

  async pasteSlice(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.locator('#paste-button'));
  }

  async balanceSlices(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.locator('#balance-slice-button'));
  }

  async setSliceDepth(index: number, depth: number): Promise<void> {
    const display = this.page.locator('div[id*=\'"type":"depth-display"\']').nth(index);
    await expect(display).toBeVisible();
    await display.click();
    const input = this.page.locator('input[id*=\'"type":"depth-input"\']').nth(index);
    await expect(input).not.toHaveClass(/hidden/);
    await input.fill(String(depth));
    await input.press('Enter');
    // The whole thumbnail strip is rebuilt (and may reorder) once the new depth
    // is committed; wait for a depth badge to show the committed value anywhere
    // in the (possibly reordered) strip rather than trusting `index` to still
    // point at the same slice.
    await expect(
      this.page.locator('div[id*=\'"type":"depth-display"\']').filter({ hasText: new RegExp(`^${depth}$`) }),
    ).not.toHaveCount(0);
  }

  async uploadSliceImage(
    index: number,
    file: { name: string; mimeType: string; buffer: Buffer },
  ): Promise<void> {
    const before = await this.log().innerText();
    const input = this.page
      .locator('div[id*=\'"type":"slice-upload"\'] input[type=file]')
      .nth(index);
    await input.setInputFiles(file);
    await expect.poll(() => this.log().innerText()).not.toBe(before);
  }

  async invertMask(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.locator('#invert-mask'));
  }

  async featherMask(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.locator('#feather-mask'));
  }

  async toggleCheckerboard(): Promise<void> {
    const button = this.page.locator('#toggle-checkerboard');
    const wasSelected = (await button.getAttribute('class'))?.includes('color-is-selected') ?? false;
    await button.click();
    await expect(button).toHaveClass(wasSelected ? /color-not-selected/ : /color-is-selected/);
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
    if (value < min || value > max) {
      throw new Error(`Value ${value} is out of range for ${id} (${min}..${max})`);
    }
    // Step directly from whatever the slider's current value is toward
    // `value`, one arrow key at a time, re-reading the committed value after
    // each press instead of pre-computing a press count from a possibly-wrong
    // `aria-valuestep` (rc-slider's real keyboard step - e.g. the
    // displacement slider's step=5, components.py's make_configuration_div -
    // is not reliably reflected by that attribute). Deliberately does *not*
    // press "Home" first: several camera sliders live under the same
    // `remember_camera_parameters` (WEB-30) callback, whose own
    // `Camera(camera_distance, focal_length, max_distance)` call passes
    // arguments in the *wrong* order for `Camera.__init__`'s actual
    // `(distance, max_distance, focal_length, ...)` signature - the live
    // max-distance slider value lands in the strictly-positive `focal_length`
    // constructor parameter. Resetting any camera slider to "Home" (0) while
    // the others still hold their real values would transiently 500 that
    // callback; stepping straight from the current value avoids ever
    // crossing 0 unless the caller's own target is 0.
    const initial = Number(await slider.getAttribute('aria-valuenow'));
    const key = value >= initial ? 'ArrowRight' : 'ArrowLeft';
    for (let guard = 0; guard < 1000; guard += 1) {
      const current = Number(await slider.getAttribute('aria-valuenow'));
      if (current === value) break;
      if ((key === 'ArrowRight' && current > value) || (key === 'ArrowLeft' && current < value)) {
        throw new Error(`Overshot ${id}: stepped past ${value} to ${current}`);
      }
      await slider.press(key);
    }
    await expect(slider).toHaveAttribute('aria-valuenow', String(value));
  }

  async expectDepthModel(label: string): Promise<void> {
    await expect(this.page.locator('#depth-model-dropdown')).toContainText(label);
  }

  async expectInpaintingModel(label: string): Promise<void> {
    await expect(this.page.locator('#inpainting-model-dropdown')).toContainText(label);
  }

  private async selectDropdownOption(dropdownId: string, label: string): Promise<void> {
    const dropdown = this.page.locator(`#${dropdownId}`);
    await dropdown.click();
    const option = this.page.getByText(label, { exact: true }).last();
    await expect(option).toBeVisible();
    await option.click();
  }

  async selectDepthModel(label: string): Promise<void> {
    await this.selectDropdownOption('depth-model-dropdown', label);
  }

  async selectInpaintingModel(label: string): Promise<void> {
    await this.selectDropdownOption('inpainting-model-dropdown', label);
  }

  async setExternalServer(address: string): Promise<void> {
    const input = this.page.locator('#external-server-address');
    await input.fill(address);
    // debounce=True: the value only commits (and reset_external_server_address
    // fires) on blur/Enter, not on every keystroke.
    await input.press('Tab');
  }

  async testExternalConnection(): Promise<void> {
    await this.page.locator('#external-test-connection-button').click();
  }

  private async expectHighlightStatus(
    locator: Locator,
    status: 'success' | 'failure' | 'none',
  ): Promise<void> {
    if (status === 'success') {
      await expect(locator).toHaveClass(/color-is-selected-light/);
    } else if (status === 'failure') {
      await expect(locator).toHaveClass(/failure-color/);
    } else {
      await expect(locator).not.toHaveClass(/color-is-selected-light|failure-color/);
    }
  }

  async expectExternalConnectionStatus(status: 'success' | 'failure' | 'none'): Promise<void> {
    await this.expectHighlightStatus(this.page.locator('#external-server-address'), status);
  }

  async setApiKey(key: string): Promise<void> {
    const input = this.page.locator('#api-key');
    await input.fill(key);
    // debounce=True: same commit-on-blur behavior as the server-address field.
    await input.press('Tab');
  }

  async validateApiKey(): Promise<void> {
    await this.page.locator('#validate-api-key').click();
  }

  async expectApiKeyStatus(status: 'success' | 'failure' | 'none'): Promise<void> {
    await this.expectHighlightStatus(this.page.locator('#api-key'), status);
  }

  // Project lifecycle

  async saveState(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.locator('#save-state'));
  }

  async restoreStateFromBytes(buffer: Buffer): Promise<void> {
    await this.page.locator('#upload-state input[type=file]').setInputFiles({
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
    await this.page.locator('#gltf-export').click();
    return downloadPromise;
  }

  async exportAnimation(): Promise<void> {
    await this.page.locator('#animation-export').click();
  }

  async setDofEnabled(enabled: boolean): Promise<void> {
    const checkbox = this.page.locator('#toggle-dof-support input[type=checkbox]');
    if ((await checkbox.isChecked()) !== enabled) {
      await checkbox.click();
    }
    await expect(checkbox).toBeChecked({ checked: enabled });
  }

  async upscaleTextures(): Promise<void> {
    await this.clickAndWaitForLogChange(this.page.locator('#upscale-textures'));
  }

  async downloadSlice(index: number): Promise<Download> {
    const downloadPromise = this.page.waitForEvent('download');
    await this.page.locator('button[id*=\'"type":"slice-info"\']').nth(index).click();
    return downloadPromise;
  }
}
