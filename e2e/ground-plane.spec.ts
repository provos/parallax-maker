/**
 * Ground plane: marking a slice as the ground, fitting the horizon and the
 * ground distance to the scene, dragging the horizon line, and adjusting
 * the ground distance by hand.
 */
import { disableAnimations, expect, requireWorkflow, test } from './fixtures';
import { readE2EState } from './helpers/oracle';

test.beforeEach(async ({ ui, page }) => {
  await ui.goto();
  await disableAnimations(page);
});

test('a slice becomes the ground plane, then the horizon and ground distance are set', async ({
  page,
  ui,
}) => {
  requireWorkflow(ui, 'segmentation');
  const projectId = await ui.restoreFixtureState();
  await ui.openTab('Segmentation');
  await ui.selectSlice(projectId, 1);

  await ui.toggleGroundPlane();
  await ui.expectGroundSlice(1, true);
  await ui.expectGroundSlice(0, false);
  await expect(ui.log()).toContainText('Slice 1 is now the ground plane');
  expect((await readE2EState(page, projectId)).ground_slice).toBe(1);

  // Fit: the horizon moves onto the ground mask's top edge (a pitch).
  await ui.fitGround();
  const fitted = (await readE2EState(page, projectId)).camera;
  expect(fitted.pitch).not.toBe(0);

  // Dragging the horizon line sets a new pitch.
  await ui.openTab('Segmentation');
  await ui.dragHorizonTo(60);
  await expect.poll(async () => (await readE2EState(page, projectId)).camera.pitch).not.toBe(fitted.pitch);

  // The ground distance is adjustable by hand.
  await ui.openTab('Export');
  await ui.setSlider('ground-distance', 30);
  await expect.poll(async () => (await readE2EState(page, projectId)).camera.ground_near).toBe(30);

  // The camera renders with the ground, and unmarking it works too.
  await ui.navigateCamera('up');
  await expect(ui.log()).toContainText('Navigated to new camera position');
  await ui.openTab('Segmentation');
  await ui.selectSlice(projectId, 1);
  await ui.toggleGroundPlane();
  await ui.expectGroundSlice(1, false);
  expect((await readE2EState(page, projectId)).ground_slice).toBeNull();
});
