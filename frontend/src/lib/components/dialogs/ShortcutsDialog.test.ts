import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import { flushSync } from 'svelte';
import ShortcutsDialog from './ShortcutsDialog.svelte';
import { uiStore } from '../../state/ui.svelte';
import { SHORTCUTS } from '../../shortcuts';

describe('ShortcutsDialog', () => {
  beforeEach(() => {
    uiStore.reset();
  });

  it('stays closed until uiStore.dialog is "shortcuts"', () => {
    render(ShortcutsDialog);
    expect(screen.getByTestId('shortcuts-dialog')).not.toHaveAttribute('open');
  });

  it('opens once uiStore.dialog is "shortcuts"', () => {
    render(ShortcutsDialog);
    uiStore.openDialog('shortcuts');
    flushSync();
    expect(screen.getByTestId('shortcuts-dialog')).toHaveAttribute('open');
  });

  it('lists every shortcut from shortcuts.ts, keys and action both', () => {
    uiStore.openDialog('shortcuts');
    render(ShortcutsDialog);

    expect(SHORTCUTS.length).toBeGreaterThan(0);
    // Some `keys` strings (e.g. "[  ]") use double spaces for alignment,
    // which testing-library's default whitespace normalizer collapses, so
    // compare against the dialog's own normalized text content instead of
    // an exact getByText match.
    const text = screen.getByTestId('shortcuts-dialog').textContent!.replace(/\s+/g, ' ');
    for (const { keys, action } of SHORTCUTS) {
      expect(text).toContain(action);
      expect(text).toContain(keys.replace(/\s+/g, ' '));
    }
  });
});
