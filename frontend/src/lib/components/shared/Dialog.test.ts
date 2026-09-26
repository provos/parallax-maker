import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import { createRawSnippet, flushSync } from 'svelte';
import Dialog from './Dialog.svelte';
import { uiStore } from '../../state/ui.svelte';

function childrenSnippet(text = 'dialog body') {
  return createRawSnippet(() => ({
    render: () => `<p data-testid="dialog-body">${text}</p>`,
  }));
}

describe('Dialog', () => {
  beforeEach(() => {
    uiStore.reset();
  });

  it('stays in the DOM (without the open attribute) while closed', () => {
    render(Dialog, {
      props: { name: 'export', title: 'Export', testId: 'export-dialog', children: childrenSnippet() },
    });
    expect(screen.getByTestId('export-dialog')).not.toHaveAttribute('open');
    // Content stays mounted even while closed.
    expect(screen.getByTestId('dialog-body')).toBeInTheDocument();
  });

  it('opens (gains the open attribute) once uiStore.dialog names it', () => {
    render(Dialog, {
      props: { name: 'export', title: 'Export', testId: 'export-dialog', children: childrenSnippet() },
    });
    uiStore.openDialog('export');
    flushSync();
    expect(screen.getByTestId('export-dialog')).toHaveAttribute('open');
  });

  it('does not open for a different dialog name', () => {
    render(Dialog, {
      props: { name: 'export', title: 'Export', testId: 'export-dialog', children: childrenSnippet() },
    });
    uiStore.openDialog('settings');
    flushSync();
    expect(screen.getByTestId('export-dialog')).not.toHaveAttribute('open');
  });

  it('closes (loses the open attribute) once uiStore.dialog is cleared', () => {
    uiStore.openDialog('export');
    render(Dialog, {
      props: { name: 'export', title: 'Export', testId: 'export-dialog', children: childrenSnippet() },
    });
    expect(screen.getByTestId('export-dialog')).toHaveAttribute('open');

    uiStore.closeDialog();
    flushSync();
    expect(screen.getByTestId('export-dialog')).not.toHaveAttribute('open');
  });

  it('renders the title and links it via aria-labelledby', () => {
    uiStore.openDialog('export');
    render(Dialog, {
      props: { name: 'export', title: 'Export', testId: 'export-dialog', children: childrenSnippet() },
    });
    const dialog = screen.getByTestId('export-dialog');
    const titleId = dialog.getAttribute('aria-labelledby');
    expect(titleId).toBe('export-dialog-title');
    expect(document.getElementById(titleId!)).toHaveTextContent('Export');
  });

  it('the close button calls uiStore.closeDialog()', async () => {
    uiStore.openDialog('export');
    render(Dialog, {
      props: { name: 'export', title: 'Export', testId: 'export-dialog', children: childrenSnippet() },
    });
    expect(uiStore.dialog).toBe('export');

    await fireEvent.click(screen.getByTestId('export-dialog-close'));
    expect(uiStore.dialog).toBeNull();
  });

  it('a click on the dialog element itself (the backdrop) calls uiStore.closeDialog()', async () => {
    uiStore.openDialog('export');
    render(Dialog, {
      props: { name: 'export', title: 'Export', testId: 'export-dialog', children: childrenSnippet() },
    });
    const dialog = screen.getByTestId('export-dialog');

    await fireEvent.click(dialog);
    expect(uiStore.dialog).toBeNull();
  });

  it('a click inside the dialog content does not close it', async () => {
    uiStore.openDialog('export');
    render(Dialog, {
      props: { name: 'export', title: 'Export', testId: 'export-dialog', children: childrenSnippet() },
    });

    await fireEvent.click(screen.getByTestId('dialog-body'));
    expect(uiStore.dialog).toBe('export');
  });

  it('the native close event calls uiStore.closeDialog() (e.g. Esc)', () => {
    uiStore.openDialog('export');
    render(Dialog, {
      props: { name: 'export', title: 'Export', testId: 'export-dialog', children: childrenSnippet() },
    });
    const dialog = screen.getByTestId('export-dialog');

    dialog.dispatchEvent(new Event('close'));
    expect(uiStore.dialog).toBeNull();
  });

  it('a stray close event for a dialog that is not the open one leaves the store alone', () => {
    uiStore.openDialog('settings');
    render(Dialog, {
      props: { name: 'export', title: 'Export', testId: 'export-dialog', children: childrenSnippet() },
    });
    const dialog = screen.getByTestId('export-dialog');

    dialog.dispatchEvent(new Event('close'));
    expect(uiStore.dialog).toBe('settings');
  });

  it('applies a numeric width/height as inline style', () => {
    uiStore.openDialog('export');
    render(Dialog, {
      props: { name: 'export', title: 'Export', testId: 'export-dialog', width: 500, height: 300, children: childrenSnippet() },
    });
    const dialog = screen.getByTestId('export-dialog');
    // jsdom normalizes calc() text, so just check the pieces that matter.
    expect(dialog.style.width).toContain('500px');
    expect(dialog.style.width).toContain('100vw');
    expect(dialog.style.height).toContain('300px');
    expect(dialog.style.height).toContain('100dvh');
  });
});
