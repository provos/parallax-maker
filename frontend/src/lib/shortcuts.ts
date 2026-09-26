/**
 * Keyboard shortcuts (docs/redesign/HANDOFF.md §6), handled by the app
 * root's keydown listener (AppShell.svelte). Ignored while typing in a text
 * field, except Ctrl+Enter (generate) in the prompt fields. The Inpaint
 * panel registers generate / pick / apply, since it owns the prompt drafts.
 *
 * Delete stays with the focused layer list (LayerPanel.svelte), so a stray
 * key press elsewhere never deletes a slice.
 */
import { uiStore, type CanvasTool, type CanvasView } from './state/ui.svelte';
import { projectStore } from './state/project.svelte';
import { maskToolsStore } from './state/maskTools.svelte';
import { isBusy } from './state/busy.svelte';
import * as workflow from './workflow';

type InpaintActions = {
  generate: () => void;
  pick: (candidate: number) => void;
  apply: () => void;
};

let inpaintActions: InpaintActions | null = null;

/** Lets the Inpaint panel handle Ctrl+Enter, 1 2 3 and A; returns an unregister. */
export function registerInpaintActions(actions: InpaintActions): () => void {
  inpaintActions = actions;
  return () => {
    if (inpaintActions === actions) inpaintActions = null;
  };
}

/** Shortcuts as the Shortcuts dialog lists them. */
export const SHORTCUTS: { keys: string; action: string }[] = [
  { keys: 'S', action: 'Segment tool' },
  { keys: 'B', action: 'Inpaint brush' },
  { keys: 'X', action: 'Toggle brush / eraser' },
  { keys: 'G', action: 'Horizon tool' },
  { keys: 'H', action: 'Pan tool' },
  { keys: 'Enter', action: 'New slice from selection' },
  { keys: 'Shift+Enter', action: 'Add selection to selected slice' },
  { keys: 'Alt+Enter', action: 'Remove selection from selected slice' },
  { keys: '[  ]', action: 'Previous / next layer' },
  { keys: 'Ctrl+Enter', action: 'Generate candidates' },
  { keys: '1  2  3', action: 'Pick candidate' },
  { keys: 'A', action: 'Apply picked candidate' },
  { keys: 'I  M  L  C', action: 'Views: Input, Depth, Slice, Composite' },
  { keys: 'P  Shift+P', action: 'Views: Parallax 2D, 3D' },
  { keys: 'Ctrl+Z', action: 'Undo slice change' },
  { keys: 'Ctrl+Shift+Z', action: 'Redo slice change' },
  { keys: 'Delete', action: 'Delete selected slice (layer list)' },
  { keys: 'Ctrl+E', action: 'Export' },
  { keys: 'Ctrl+,', action: 'Settings' },
  { keys: '`', action: 'Log' },
  { keys: '?', action: 'Keyboard shortcuts' },
  { keys: 'Esc', action: 'Close dialog' },
];

const TOOL_KEYS: Record<string, CanvasTool> = { s: 'segment', b: 'brush', g: 'horizon', h: 'pan' };
const VIEW_KEYS: Record<string, CanvasView> = { i: 'input', m: 'depth', l: 'slice', c: 'composite', p: 'parallax' };

function isTextField(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) {
    return true;
  }
  if (target instanceof HTMLInputElement) {
    return !['range', 'checkbox', 'radio', 'button', 'submit', 'reset', 'file', 'color'].includes(target.type);
  }
  return false;
}

function viewAvailable(next: CanvasView): boolean {
  const view = projectStore.view;
  if (!view) return false;
  const hasSlices = view.slices.length > 0;
  switch (next) {
    case 'input':
      return !!view.assets.input;
    case 'depth':
      return !!view.assets.depth;
    case 'slice':
      return view.selectedSlice != null;
    default:
      return hasSlices;
  }
}

/** Selects the neighbouring layer in the layer list's order (nearest first). */
function stepLayer(direction: -1 | 1): void {
  const view = projectStore.view;
  if (!view || view.slices.length === 0) return;
  const rows = [...view.slices].sort((a, b) => b.depth - a.depth || b.index - a.index);
  const at = rows.findIndex((slice) => slice.index === view.selectedSlice);
  const next = at < 0 ? (direction > 0 ? 0 : rows.length - 1) : at + direction;
  if (next < 0 || next >= rows.length) return;
  void workflow.selectSlice(rows[next].index);
}

/** Handles one keydown; returns whether it was a shortcut. */
export function handleShortcut(event: KeyboardEvent): boolean {
  if (event.defaultPrevented || event.isComposing) return false;
  const key = event.key;
  const lower = key.toLowerCase();
  const mod = event.ctrlKey || event.metaKey;
  const view = projectStore.view;
  const selected = view?.slices.find((slice) => slice.index === view.selectedSlice) ?? null;

  // Ctrl+Enter generates, even from the prompt fields.
  if (mod && key === 'Enter') {
    if (uiStore.step !== 'inpaint' || !inpaintActions || uiStore.dialog) return false;
    inpaintActions.generate();
    return true;
  }

  if (isTextField(event.target)) return false;

  if (mod && !event.altKey) {
    if (lower === 'e') {
      if (!view?.slices.length) return false;
      uiStore.openDialog('export');
      return true;
    }
    if (key === ',') {
      uiStore.openSettings();
      return true;
    }
    if (lower === 'z' && !uiStore.dialog) {
      if (!selected || isBusy()) return true;
      if (event.shiftKey) {
        if (selected.canRedo) void workflow.redoSlice(selected.index);
      } else if (selected.canUndo) {
        void workflow.undoSlice(selected.index);
      }
      return true;
    }
    return false;
  }
  if (event.metaKey || event.ctrlKey) return false;

  if (key === '?') {
    uiStore.openDialog('shortcuts');
    return true;
  }
  // Everything else acts on the workspace behind any dialog.
  if (uiStore.dialog) return false;

  if (key === '`') {
    uiStore.toggleLog();
    return true;
  }

  if (key === 'Enter') {
    // Enter on a focused control activates that control instead.
    if (event.target instanceof HTMLElement && event.target.closest('button, a, [role="button"], [role="tab"]')) {
      return false;
    }
    if (!view?.segmentation.hasMask || isBusy()) return false;
    if (event.shiftKey) void workflow.addMaskToSlice();
    else if (event.altKey) void workflow.removeMaskFromSlice();
    else void workflow.createSlice();
    return true;
  }

  if (event.altKey) return false;

  if (key === '[' || key === ']') {
    if (isBusy()) return true;
    stepLayer(key === '[' ? -1 : 1);
    return true;
  }

  if (event.shiftKey) {
    if (lower === 'p' && viewAvailable('3d')) {
      uiStore.setView('3d');
      return true;
    }
    return false;
  }

  if (lower in TOOL_KEYS) {
    if (!view?.assets.input) return false;
    uiStore.setTool(TOOL_KEYS[lower]);
    return true;
  }
  if (lower === 'x') {
    if (uiStore.tool !== 'brush') return false;
    maskToolsStore.toggleErasing();
    return true;
  }
  if (lower in VIEW_KEYS) {
    const next = VIEW_KEYS[lower];
    if (!viewAvailable(next)) return false;
    uiStore.setView(next);
    return true;
  }

  if (uiStore.step === 'inpaint' && inpaintActions) {
    if (key === '1' || key === '2' || key === '3') {
      inpaintActions.pick(Number(key) - 1);
      return true;
    }
    if (lower === 'a') {
      inpaintActions.apply();
      return true;
    }
  }
  return false;
}
