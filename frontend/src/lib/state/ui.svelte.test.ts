import { beforeEach, describe, expect, it } from 'vitest';
import { uiStore } from './ui.svelte';

describe('uiStore', () => {
  beforeEach(() => {
    uiStore.reset();
  });

  it('defaults to the Image step, Input view, Pan tool and Object segmentation mode', () => {
    expect(uiStore.step).toBe('image');
    expect(uiStore.view).toBe('input');
    expect(uiStore.tool).toBe('pan');
    expect(uiStore.segmentationMode).toBe('segment');
  });

  describe('setStep applies each step\'s default view and tool', () => {
    it('image -> Input view, Pan tool', () => {
      uiStore.setStep('image');
      expect(uiStore.view).toBe('input');
      expect(uiStore.tool).toBe('pan');
      expect(uiStore.mainTab).toBe('Mode');
    });

    it('depth -> Depth view, Pan tool', () => {
      uiStore.setStep('depth');
      expect(uiStore.view).toBe('depth');
      expect(uiStore.tool).toBe('pan');
      expect(uiStore.mainTab).toBe('Mode');
    });

    it('slices -> Input view, Segment tool', () => {
      uiStore.setStep('slices');
      expect(uiStore.view).toBe('input');
      expect(uiStore.tool).toBe('segment');
      expect(uiStore.mainTab).toBe('Segmentation');
    });

    it('inpaint -> Slice view, Brush tool', () => {
      uiStore.setStep('inpaint');
      expect(uiStore.view).toBe('slice');
      expect(uiStore.tool).toBe('brush');
      expect(uiStore.mainTab).toBe('Inpainting');
    });

    it('ground -> Composite view, Horizon tool', () => {
      uiStore.setStep('ground');
      expect(uiStore.view).toBe('composite');
      expect(uiStore.tool).toBe('horizon');
      expect(uiStore.mainTab).toBe('Ground');
    });

    it('preview -> Parallax view, Pan tool, and marks the session as previewed', () => {
      expect(uiStore.previewed).toBe(false);
      uiStore.setStep('preview');
      expect(uiStore.view).toBe('parallax');
      expect(uiStore.tool).toBe('pan');
      expect(uiStore.mainTab).toBe('Preview');
      expect(uiStore.previewed).toBe(true);
    });

    it('export opens the Export dialog and leaves the step, panel, view and tool untouched', () => {
      uiStore.setStep('slices');
      expect(uiStore.view).toBe('input');
      expect(uiStore.tool).toBe('segment');

      uiStore.setStep('export');
      expect(uiStore.dialog).toBe('export');
      // step/mainTab/view/tool are untouched: Export has no panel or
      // STEP_DEFAULTS entry of its own, it just opens the dialog over
      // whatever step you were on.
      expect(uiStore.step).toBe('slices');
      expect(uiStore.mainTab).toBe('Segmentation');
      expect(uiStore.view).toBe('input');
      expect(uiStore.tool).toBe('segment');
    });
  });

  describe('setTool moves the step and switches to a compatible view', () => {
    it('segment -> Slices step; switches away from an incompatible view', () => {
      uiStore.setStep('depth'); // view: 'depth', not in segment's views (['input'])
      uiStore.setTool('segment');
      expect(uiStore.step).toBe('slices');
      expect(uiStore.view).toBe('input');
      expect(uiStore.tool).toBe('segment');
    });

    it('segment keeps the view when it is already compatible', () => {
      uiStore.setView('input');
      uiStore.setTool('segment');
      expect(uiStore.view).toBe('input');
    });

    it('brush -> Inpaint step; falls back to its own first compatible view', () => {
      uiStore.setStep('ground'); // view: 'composite', not in brush's views (['slice', 'input'])
      uiStore.setTool('brush');
      expect(uiStore.step).toBe('inpaint');
      expect(uiStore.view).toBe('slice');
    });

    it('brush keeps the view when it is already compatible (input)', () => {
      uiStore.setView('input');
      uiStore.setTool('brush');
      expect(uiStore.view).toBe('input');
    });

    it('horizon -> Ground step; falls back to its own first compatible view', () => {
      uiStore.setStep('inpaint'); // view: 'slice', not in horizon's views (['composite','input','depth'])
      uiStore.setTool('horizon');
      expect(uiStore.step).toBe('ground');
      expect(uiStore.view).toBe('composite');
    });

    it('pan has no owning step and works in every view', () => {
      uiStore.setStep('slices'); // step: 'slices', view: 'input', tool: 'segment'
      uiStore.setTool('pan');
      // Pan has no TOOL_STEPS entry, so the step is left as-is.
      expect(uiStore.step).toBe('slices');
      expect(uiStore.view).toBe('input');
      expect(uiStore.tool).toBe('pan');
    });
  });

  describe('setView falls back to Pan when the current tool does not work there', () => {
    it('switches to a view compatible with Segment (input only)', () => {
      uiStore.setTool('segment'); // view becomes 'input'
      uiStore.setView('depth'); // 'depth' not in segment's views
      expect(uiStore.tool).toBe('pan');
      expect(uiStore.view).toBe('depth');
    });

    it('keeps the tool when the new view is compatible', () => {
      uiStore.setTool('brush'); // view becomes 'slice'
      uiStore.setView('input'); // 'input' is in brush's views
      expect(uiStore.tool).toBe('brush');
      expect(uiStore.view).toBe('input');
    });

    it('Parallax and 3D move the step to Preview', () => {
      uiStore.setStep('slices');
      uiStore.setView('parallax');
      expect(uiStore.step).toBe('preview');
      expect(uiStore.tool).toBe('pan'); // segment doesn't work in parallax

      uiStore.setStep('slices');
      uiStore.setView('3d');
      expect(uiStore.step).toBe('preview');
    });
  });

  describe('dialogs', () => {
    it('openDialog/closeDialog set and clear the open dialog', () => {
      expect(uiStore.dialog).toBeNull();
      uiStore.openDialog('shortcuts');
      expect(uiStore.dialog).toBe('shortcuts');
      uiStore.closeDialog();
      expect(uiStore.dialog).toBeNull();
    });

    it('openSettings defaults to the current settingsSection and opens the settings dialog', () => {
      expect(uiStore.settingsSection).toBe('inpainting');
      uiStore.openSettings();
      expect(uiStore.dialog).toBe('settings');
      expect(uiStore.settingsSection).toBe('inpainting');
    });

    it('openSettings(section) jumps to that section', () => {
      uiStore.openSettings('appearance');
      expect(uiStore.dialog).toBe('settings');
      expect(uiStore.settingsSection).toBe('appearance');
    });

    it('setSettingsSection changes the section without touching the dialog', () => {
      uiStore.setSettingsSection('depth');
      expect(uiStore.settingsSection).toBe('depth');
      expect(uiStore.dialog).toBeNull();
    });

    it('setStep("export") opens the export dialog over whatever step is current', () => {
      uiStore.setStep('inpaint');
      uiStore.setStep('export');
      expect(uiStore.dialog).toBe('export');
      expect(uiStore.step).toBe('inpaint');
    });

    it('any other setStep closes an open dialog', () => {
      uiStore.openDialog('settings');
      uiStore.setStep('depth');
      expect(uiStore.dialog).toBeNull();
      expect(uiStore.step).toBe('depth');
    });
  });

  describe('reset', () => {
    it('restores every default, including progress flags', () => {
      uiStore.setStep('preview'); // marks previewed
      uiStore.markInpainted();
      uiStore.markExported();
      uiStore.setSegmentationMode('depth');
      uiStore.setTheme('light');
      uiStore.openDialog('settings');
      uiStore.setSettingsSection('appearance');

      uiStore.reset();

      expect(uiStore.step).toBe('image');
      expect(uiStore.view).toBe('input');
      expect(uiStore.tool).toBe('pan');
      expect(uiStore.segmentationMode).toBe('segment');
      expect(uiStore.previewed).toBe(false);
      expect(uiStore.inpainted).toBe(false);
      expect(uiStore.exported).toBe(false);
      expect(uiStore.renderedMainUrl).toBeNull();
      expect(uiStore.theme).toBe('dark');
      expect(uiStore.dialog).toBeNull();
      expect(uiStore.settingsSection).toBe('inpainting');
    });
  });

  describe('resetSession', () => {
    it('forgets progress and returns to the first step for a new project', () => {
      uiStore.setStep('inpaint');
      uiStore.markInpainted();
      uiStore.markPreviewed();
      uiStore.markExported();
      uiStore.setRenderedMainUrl('/api/v1/projects/old/assets/main?v=1');
      uiStore.openDialog('export');

      uiStore.resetSession();

      expect(uiStore.renderedMainUrl).toBeNull();

      expect(uiStore.step).toBe('image');
      expect(uiStore.view).toBe('input');
      expect(uiStore.tool).toBe('pan');
      expect(uiStore.inpainted).toBe(false);
      expect(uiStore.previewed).toBe(false);
      expect(uiStore.exported).toBe(false);
      expect(uiStore.dialog).toBeNull();
    });
  });
});
