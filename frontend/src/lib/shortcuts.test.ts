import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./workflow', () => ({
  undoSlice: vi.fn(),
  redoSlice: vi.fn(),
  addMaskToSlice: vi.fn(),
  removeMaskFromSlice: vi.fn(),
  createSlice: vi.fn(),
  selectSlice: vi.fn(),
}));

import * as workflow from './workflow';
import { handleShortcut, registerInpaintActions, SHORTCUTS } from './shortcuts';
import { uiStore } from './state/ui.svelte';
import { projectStore } from './state/project.svelte';
import { maskToolsStore } from './state/maskTools.svelte';
import { jobStore } from './state/jobs.svelte';
import type { ProjectView, SliceView } from './api/types';

function makeSlice(index: number, depth: number, overrides: Partial<SliceView> = {}): SliceView {
  return {
    index,
    depth,
    version: 1,
    canUndo: false,
    canRedo: false,
    positivePrompt: '',
    negativePrompt: '',
    image: { url: `/slice-${index}` },
    thumbnail: { url: `/thumb-${index}` },
    ...overrides,
  } as SliceView;
}

function makeView(overrides: Partial<ProjectView> = {}): ProjectView {
  return {
    id: 'appstate-test',
    revision: 1,
    image: { width: 320, height: 240 },
    assets: { input: { url: '/input' }, depth: { url: '/depth' } },
    depthModel: 'dinov2',
    numSlices: 3,
    thresholds: [],
    slices: [],
    selectedSlice: null,
    segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false },
    inpainting: {
      model: 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
      strength: 0.8,
      guidanceScale: 7.5,
      padding: 50,
      blur: 50,
      externalServer: 'localhost:7860',
      hasWorkflow: false,
      candidates: null,
      selectedCandidate: null,
    },
    busy: null,
    settings: {
      darkMode: true,
      camera: { distance: 100, focalLength: 100, maxDistance: 200 },
      meshDisplacement: 0,
      depthModel: 'dinov2',
    },
    exports: { gltf: null, upscaled: false },
    ...overrides,
  } as ProjectView;
}

/**
 * Dispatches a real `keydown` on `target` (so `event.target` is set the way
 * the browser would set it) and runs it through `handleShortcut`, returning
 * what that call returned.
 */
function fire(target: EventTarget, init: KeyboardEventInit): boolean {
  const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
  let handled = false;
  const listener = (e: Event) => {
    handled = handleShortcut(e as KeyboardEvent);
  };
  target.addEventListener('keydown', listener);
  target.dispatchEvent(event);
  target.removeEventListener('keydown', listener);
  return handled;
}

describe('shortcuts', () => {
  let textInput: HTMLInputElement;
  let textarea: HTMLTextAreaElement;
  let select: HTMLSelectElement;
  let checkbox: HTMLInputElement;
  let button: HTMLButtonElement;

  beforeEach(() => {
    uiStore.reset();
    projectStore.reset();
    jobStore.end();
    maskToolsStore.reset();
    vi.clearAllMocks();

    textInput = document.createElement('input');
    textInput.type = 'text';
    textarea = document.createElement('textarea');
    select = document.createElement('select');
    checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    button = document.createElement('button');
    document.body.append(textInput, textarea, select, checkbox, button);
  });

  afterEach(() => {
    textInput.remove();
    textarea.remove();
    select.remove();
    checkbox.remove();
    button.remove();
  });

  describe('text-field exclusion', () => {
    it('ignores a plain shortcut key typed into a text input', () => {
      projectStore.applyView(makeView());
      const handled = fire(textInput, { key: 's' });
      expect(handled).toBe(false);
      expect(uiStore.tool).toBe('pan');
    });

    it('ignores a plain shortcut key typed into a textarea', () => {
      projectStore.applyView(makeView());
      const handled = fire(textarea, { key: 's' });
      expect(handled).toBe(false);
      expect(uiStore.tool).toBe('pan');
    });

    it('ignores a plain shortcut key focused on a select', () => {
      projectStore.applyView(makeView());
      const handled = fire(select, { key: 's' });
      expect(handled).toBe(false);
    });

    it('does NOT treat a checkbox/radio/button-like input as a text field', () => {
      projectStore.applyView(makeView());
      const handled = fire(checkbox, { key: 's' });
      expect(handled).toBe(true);
      expect(uiStore.tool).toBe('segment');
    });

    it('blocks Ctrl+E while typing in a text field', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0)] }));
      const handled = fire(textInput, { key: 'e', ctrlKey: true });
      expect(handled).toBe(false);
      expect(uiStore.dialog).toBeNull();
    });
  });

  describe('Ctrl+Enter generates candidates', () => {
    it('works even from a text field, only on the inpaint step with actions registered', () => {
      projectStore.applyView(makeView());
      uiStore.setStep('inpaint');
      const generate = vi.fn();
      const unregister = registerInpaintActions({ generate, pick: vi.fn(), apply: vi.fn() });

      const handled = fire(textInput, { key: 'Enter', ctrlKey: true });

      expect(handled).toBe(true);
      expect(generate).toHaveBeenCalledOnce();
      unregister();
    });

    it('does nothing off the inpaint step', () => {
      projectStore.applyView(makeView());
      uiStore.setStep('slices');
      const generate = vi.fn();
      const unregister = registerInpaintActions({ generate, pick: vi.fn(), apply: vi.fn() });

      const handled = fire(document.body, { key: 'Enter', ctrlKey: true });

      expect(handled).toBe(false);
      expect(generate).not.toHaveBeenCalled();
      unregister();
    });

    it('does nothing without registered actions', () => {
      projectStore.applyView(makeView());
      uiStore.setStep('inpaint');
      const handled = fire(document.body, { key: 'Enter', ctrlKey: true });
      expect(handled).toBe(false);
    });

    it('does nothing while a dialog is open', () => {
      projectStore.applyView(makeView());
      uiStore.setStep('inpaint');
      const generate = vi.fn();
      const unregister = registerInpaintActions({ generate, pick: vi.fn(), apply: vi.fn() });
      uiStore.openDialog('settings');

      const handled = fire(document.body, { key: 'Enter', ctrlKey: true });

      expect(handled).toBe(false);
      expect(generate).not.toHaveBeenCalled();
      unregister();
    });

    it('a later unregister stops the actions from firing', () => {
      projectStore.applyView(makeView());
      uiStore.setStep('inpaint');
      const generate = vi.fn();
      const unregister = registerInpaintActions({ generate, pick: vi.fn(), apply: vi.fn() });
      unregister();

      const handled = fire(document.body, { key: 'Enter', ctrlKey: true });

      expect(handled).toBe(false);
      expect(generate).not.toHaveBeenCalled();
    });
  });

  describe('Ctrl+E export', () => {
    it('opens the export dialog only when there are slices', () => {
      projectStore.applyView(makeView({ slices: [] }));
      expect(fire(document.body, { key: 'e', ctrlKey: true })).toBe(false);
      expect(uiStore.dialog).toBeNull();

      projectStore.applyView(makeView({ revision: 2, slices: [makeSlice(0, 0)] }));
      expect(fire(document.body, { key: 'e', ctrlKey: true })).toBe(true);
      expect(uiStore.dialog).toBe('export');
    });

    it('works with metaKey too (Cmd+E)', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0)] }));
      expect(fire(document.body, { key: 'e', metaKey: true })).toBe(true);
      expect(uiStore.dialog).toBe('export');
    });

    it('is not blocked by an already-open dialog', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0)] }));
      uiStore.openDialog('shortcuts');
      expect(fire(document.body, { key: 'e', ctrlKey: true })).toBe(true);
      expect(uiStore.dialog).toBe('export');
    });

    it('Ctrl+Alt+E does nothing', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0)] }));
      expect(fire(document.body, { key: 'e', ctrlKey: true, altKey: true })).toBe(false);
      expect(uiStore.dialog).toBeNull();
    });
  });

  describe('Ctrl+, settings', () => {
    it('opens settings regardless of slices or an already-open dialog', () => {
      projectStore.applyView(makeView({ slices: [] }));
      uiStore.openDialog('shortcuts');
      expect(fire(document.body, { key: ',', ctrlKey: true })).toBe(true);
      expect(uiStore.dialog).toBe('settings');
    });
  });

  describe('Ctrl+Z / Ctrl+Shift+Z undo/redo', () => {
    it('undoes the selected slice', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0, { canUndo: true })], selectedSlice: 0 }));
      expect(fire(document.body, { key: 'z', ctrlKey: true })).toBe(true);
      expect(workflow.undoSlice).toHaveBeenCalledWith(0);
    });

    it('redoes the selected slice with Shift', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0, { canRedo: true })], selectedSlice: 0 }));
      expect(fire(document.body, { key: 'z', ctrlKey: true, shiftKey: true })).toBe(true);
      expect(workflow.redoSlice).toHaveBeenCalledWith(0);
    });

    it('does nothing (but is still handled) without a selected slice', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0)], selectedSlice: null }));
      expect(fire(document.body, { key: 'z', ctrlKey: true })).toBe(true);
      expect(workflow.undoSlice).not.toHaveBeenCalled();
    });

    it('does nothing while busy', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0, { canUndo: true })], selectedSlice: 0 }));
      jobStore.begin('slice-editing');
      expect(fire(document.body, { key: 'z', ctrlKey: true })).toBe(true);
      expect(workflow.undoSlice).not.toHaveBeenCalled();
    });

    it('does not fire when it cannot undo/redo', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0)], selectedSlice: 0 }));
      fire(document.body, { key: 'z', ctrlKey: true });
      expect(workflow.undoSlice).not.toHaveBeenCalled();
      fire(document.body, { key: 'z', ctrlKey: true, shiftKey: true });
      expect(workflow.redoSlice).not.toHaveBeenCalled();
    });

    it('is ignored while a dialog is open', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0, { canUndo: true })], selectedSlice: 0 }));
      uiStore.openDialog('settings');
      expect(fire(document.body, { key: 'z', ctrlKey: true })).toBe(false);
      expect(workflow.undoSlice).not.toHaveBeenCalled();
    });
  });

  describe('? shortcuts dialog', () => {
    it('opens the shortcuts dialog', () => {
      expect(fire(document.body, { key: '?' })).toBe(true);
      expect(uiStore.dialog).toBe('shortcuts');
    });

    it('works even while another dialog is open', () => {
      uiStore.openDialog('settings');
      expect(fire(document.body, { key: '?' })).toBe(true);
      expect(uiStore.dialog).toBe('shortcuts');
    });
  });

  describe('while a dialog is open, everything else is ignored', () => {
    it('` does not toggle the log', () => {
      uiStore.openDialog('settings');
      expect(fire(document.body, { key: '`' })).toBe(false);
      expect(uiStore.logOpen).toBe(false);
    });

    it('a tool key does not switch tools', () => {
      projectStore.applyView(makeView());
      uiStore.openDialog('settings');
      expect(fire(document.body, { key: 's' })).toBe(false);
      expect(uiStore.tool).toBe('pan');
    });
  });

  describe('` toggles the log', () => {
    it('toggles uiStore.logOpen', () => {
      expect(uiStore.logOpen).toBe(false);
      expect(fire(document.body, { key: '`' })).toBe(true);
      expect(uiStore.logOpen).toBe(true);
      fire(document.body, { key: '`' });
      expect(uiStore.logOpen).toBe(false);
    });
  });

  describe('Enter / Shift+Enter / Alt+Enter act on the mask', () => {
    it('plain Enter creates a slice when there is a mask', () => {
      projectStore.applyView(makeView({ segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true } }));
      expect(fire(document.body, { key: 'Enter' })).toBe(true);
      expect(workflow.createSlice).toHaveBeenCalledOnce();
    });

    it('Shift+Enter adds the selection to the selected slice', () => {
      projectStore.applyView(makeView({ segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true } }));
      expect(fire(document.body, { key: 'Enter', shiftKey: true })).toBe(true);
      expect(workflow.addMaskToSlice).toHaveBeenCalledOnce();
    });

    it('Alt+Enter removes the selection from the selected slice', () => {
      projectStore.applyView(makeView({ segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true } }));
      expect(fire(document.body, { key: 'Enter', altKey: true })).toBe(true);
      expect(workflow.removeMaskFromSlice).toHaveBeenCalledOnce();
    });

    it('does nothing without a mask', () => {
      projectStore.applyView(makeView({ segmentation: { multiPointMode: false, queuedPoints: [], hasMask: false } }));
      expect(fire(document.body, { key: 'Enter' })).toBe(false);
      expect(workflow.createSlice).not.toHaveBeenCalled();
    });

    it('does nothing while busy', () => {
      projectStore.applyView(makeView({ segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true } }));
      jobStore.begin('segmentation');
      expect(fire(document.body, { key: 'Enter' })).toBe(false);
      expect(workflow.createSlice).not.toHaveBeenCalled();
    });

    it('lets Enter on a focused button activate the button instead', () => {
      projectStore.applyView(makeView({ segmentation: { multiPointMode: false, queuedPoints: [], hasMask: true } }));
      expect(fire(button, { key: 'Enter' })).toBe(false);
      expect(workflow.createSlice).not.toHaveBeenCalled();
    });
  });

  describe('[ and ] step layers', () => {
    it('steps to the nearest-first neighbour and skips when Alt is held', () => {
      const rows = [makeSlice(0, 10), makeSlice(1, 200), makeSlice(2, 100)];
      // Nearest-first order (by depth desc): index 1 (200), index 2 (100), index 0 (10).
      projectStore.applyView(makeView({ slices: rows, selectedSlice: 1 }));

      expect(fire(document.body, { key: ']' })).toBe(true);
      expect(workflow.selectSlice).toHaveBeenCalledWith(2);

      expect(fire(document.body, { key: '[', altKey: true })).toBe(false);
      expect(workflow.selectSlice).toHaveBeenCalledTimes(1);
    });

    it('is a no-op (but handled) while busy', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0), makeSlice(1, 10)], selectedSlice: 0 }));
      jobStore.begin('slice-editing');
      expect(fire(document.body, { key: ']' })).toBe(true);
      expect(workflow.selectSlice).not.toHaveBeenCalled();
    });
  });

  describe('Shift+P selects the 3D view', () => {
    it('only when a 3D view is available (there are slices)', () => {
      projectStore.applyView(makeView({ slices: [] }));
      expect(fire(document.body, { key: 'P', shiftKey: true })).toBe(false);

      projectStore.applyView(makeView({ revision: 2, slices: [makeSlice(0, 0)] }));
      expect(fire(document.body, { key: 'P', shiftKey: true })).toBe(true);
      expect(uiStore.view).toBe('3d');
    });

    it('other Shift+ combinations are ignored', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0)] }));
      expect(fire(document.body, { key: 'S', shiftKey: true })).toBe(false);
    });
  });

  describe('S/B/G/H tool keys', () => {
    it('need an input image', () => {
      projectStore.applyView(makeView({ assets: { input: null, depth: null } }));
      expect(fire(document.body, { key: 's' })).toBe(false);
      expect(uiStore.tool).toBe('pan');

      projectStore.applyView(makeView({ revision: 2, assets: { input: { url: '/input' }, depth: null } }));
      expect(fire(document.body, { key: 'b' })).toBe(true);
      expect(uiStore.tool).toBe('brush');
    });

    it('maps g to horizon and h to pan', () => {
      projectStore.applyView(makeView());
      uiStore.setTool('brush');
      expect(fire(document.body, { key: 'g' })).toBe(true);
      expect(uiStore.tool).toBe('horizon');
      expect(fire(document.body, { key: 'h' })).toBe(true);
      expect(uiStore.tool).toBe('pan');
    });
  });

  describe('X toggles eraser', () => {
    it('only while the brush tool is active', () => {
      projectStore.applyView(makeView());
      uiStore.setTool('pan');
      expect(fire(document.body, { key: 'x' })).toBe(false);
      expect(maskToolsStore.erasing).toBe(false);

      uiStore.setTool('brush');
      expect(fire(document.body, { key: 'x' })).toBe(true);
      expect(maskToolsStore.erasing).toBe(true);
    });
  });

  describe('I/M/L/C/P view keys', () => {
    it('I needs the input asset', () => {
      projectStore.applyView(makeView({ assets: { input: null, depth: null } }));
      expect(fire(document.body, { key: 'i' })).toBe(false);

      projectStore.applyView(makeView({ revision: 2, assets: { input: { url: '/input' }, depth: null } }));
      expect(fire(document.body, { key: 'i' })).toBe(true);
      expect(uiStore.view).toBe('input');
    });

    it('M needs the depth asset', () => {
      projectStore.applyView(makeView({ assets: { input: { url: '/input' }, depth: null } }));
      expect(fire(document.body, { key: 'm' })).toBe(false);

      projectStore.applyView(makeView({ revision: 2, assets: { input: { url: '/input' }, depth: { url: '/depth' } } }));
      expect(fire(document.body, { key: 'm' })).toBe(true);
      expect(uiStore.view).toBe('depth');
    });

    it('L needs a selected slice', () => {
      projectStore.applyView(makeView({ slices: [makeSlice(0, 0)], selectedSlice: null }));
      expect(fire(document.body, { key: 'l' })).toBe(false);

      projectStore.applyView(makeView({ revision: 2, slices: [makeSlice(0, 0)], selectedSlice: 0 }));
      expect(fire(document.body, { key: 'l' })).toBe(true);
      expect(uiStore.view).toBe('slice');
    });

    it('C and P need at least one slice', () => {
      projectStore.applyView(makeView({ slices: [] }));
      expect(fire(document.body, { key: 'c' })).toBe(false);
      expect(fire(document.body, { key: 'p' })).toBe(false);

      projectStore.applyView(makeView({ revision: 2, slices: [makeSlice(0, 0)] }));
      expect(fire(document.body, { key: 'c' })).toBe(true);
      expect(uiStore.view).toBe('composite');
      expect(fire(document.body, { key: 'p' })).toBe(true);
      expect(uiStore.view).toBe('parallax');
    });
  });

  describe('1 2 3 A inpaint candidate actions', () => {
    it('only fire on the inpaint step with registered actions', () => {
      projectStore.applyView(makeView());
      const pick = vi.fn();
      const apply = vi.fn();
      const unregister = registerInpaintActions({ generate: vi.fn(), pick, apply });

      // Not on the inpaint step yet.
      expect(fire(document.body, { key: '1' })).toBe(false);
      expect(pick).not.toHaveBeenCalled();

      uiStore.setStep('inpaint');
      expect(fire(document.body, { key: '2' })).toBe(true);
      expect(pick).toHaveBeenCalledWith(1);

      expect(fire(document.body, { key: 'a' })).toBe(true);
      expect(apply).toHaveBeenCalledOnce();

      unregister();
      expect(fire(document.body, { key: '3' })).toBe(false);
    });
  });

  describe('composing / defaultPrevented', () => {
    it('ignores events already marked defaultPrevented', () => {
      projectStore.applyView(makeView());
      const event = new KeyboardEvent('keydown', { key: 's', cancelable: true });
      event.preventDefault();
      expect(handleShortcut(event)).toBe(false);
    });

    it('ignores IME composition events', () => {
      projectStore.applyView(makeView());
      const event = new KeyboardEvent('keydown', { key: 's', isComposing: true });
      expect(handleShortcut(event)).toBe(false);
    });
  });

  describe('SHORTCUTS list', () => {
    it('has a keys/action entry for every documented shortcut', () => {
      const actions = SHORTCUTS.map((s) => s.action);
      expect(actions).toContain('Export');
      expect(actions).toContain('Settings');
      expect(actions).toContain('Keyboard shortcuts');
      expect(SHORTCUTS.every(({ keys, action }) => keys.length > 0 && action.length > 0)).toBe(true);
    });
  });
});
