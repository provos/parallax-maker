/**
 * Manual parallax navigation (Dash's CMP-26 `navigate_image`): the camera
 * buttons move the preview camera over the slice cards and show the
 * server-rendered view; any selected slice is deselected.
 */
import { disableAnimations, expect, requireWorkflow, test } from './fixtures';
import { imageHash } from './helpers/image';
import { readE2EState } from './helpers/oracle';

/** The logged position, e.g. "[   0.    0. -125.]" (numpy formatting). */
function position(x: number, y: number, z: number): RegExp {
  return new RegExp(`Navigated to new camera position \\[\\s*${x}\\.\\s+${y}\\.\\s+${z}\\.\\]`);
}

test.beforeEach(async ({ ui, page }) => {
  await ui.goto();
  await disableAnimations(page);
});

test('camera buttons render the parallax view from a new position and back', async ({ page, ui }) => {
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');
  await ui.selectSlice(projectId, 1);

  // The fixture project's camera distance is 125.
  await ui.navigateCamera('reset');
  await expect(ui.log()).toContainText(position(0, 0, -125));
  await expect.poll(async () => (await readE2EState(page, projectId)).selected_slice).toBeNull();
  const centered = await imageHash(ui.mainImage());

  await ui.navigateCamera('left');
  await expect(ui.log()).toContainText(position(-1, 0, -125));
  await expect.poll(() => imageHash(ui.mainImage())).not.toBe(centered);

  // Moving back renders the identical view.
  await ui.navigateCamera('right');
  await expect.poll(() => imageHash(ui.mainImage())).toBe(centered);

  await ui.navigateCamera('in');
  await expect(ui.log()).toContainText(position(0, 0, -124));
});
