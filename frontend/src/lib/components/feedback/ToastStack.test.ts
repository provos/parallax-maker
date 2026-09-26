import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import ToastStack from './ToastStack.svelte';
import { toastStore } from '../../state/toasts.svelte';

describe('ToastStack', () => {
  afterEach(() => {
    toastStore.reset();
  });

  it('renders nothing when there are no toasts', () => {
    render(ToastStack);
    expect(screen.queryAllByTestId('toast')).toHaveLength(0);
  });

  it('renders one toast per store entry with its kind as data-kind', () => {
    toastStore.success('Depth map ready');
    toastStore.error('Generation failed', 'boom');
    render(ToastStack);

    const toasts = screen.getAllByTestId('toast');
    expect(toasts).toHaveLength(2);
    expect(toasts[0]).toHaveAttribute('data-kind', 'success');
    expect(toasts[0]).toHaveTextContent('Depth map ready');
    expect(toasts[1]).toHaveAttribute('data-kind', 'error');
    expect(toasts[1]).toHaveTextContent('Generation failed');
    expect(toasts[1]).toHaveTextContent('boom');
  });

  it('clicking a toast-action dismisses the toast and runs the action', async () => {
    const run = vi.fn();
    toastStore.error('Generation failed', 'boom', [{ label: 'Retry', run }]);
    render(ToastStack);

    const actionButton = screen.getByTestId('toast-action');
    expect(actionButton).toHaveTextContent('Retry');
    await fireEvent.click(actionButton);

    expect(run).toHaveBeenCalledOnce();
    expect(screen.queryByTestId('toast')).not.toBeInTheDocument();
  });

  it('clicking toast-dismiss removes only that toast', async () => {
    toastStore.success('First');
    toastStore.success('Second');
    render(ToastStack);

    const dismissButtons = screen.getAllByTestId('toast-dismiss');
    await fireEvent.click(dismissButtons[0]);

    const remaining = screen.getAllByTestId('toast');
    expect(remaining).toHaveLength(1);
    expect(remaining[0]).toHaveTextContent('Second');
  });
});
