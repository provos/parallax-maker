import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { toastStore, TOAST_TIMEOUT_MS } from './toasts.svelte';

describe('toastStore', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    toastStore.reset();
    vi.useRealTimers();
  });

  it('success/info/error push a toast with the given kind, title and message', () => {
    toastStore.success('Depth map ready');
    toastStore.info('Cancelled', 'Generating inpainting candidates');
    toastStore.error('Generation failed', 'model exploded');

    expect(toastStore.toasts).toHaveLength(3);
    expect(toastStore.toasts[0]).toMatchObject({ kind: 'success', title: 'Depth map ready', message: undefined });
    expect(toastStore.toasts[1]).toMatchObject({ kind: 'info', title: 'Cancelled', message: 'Generating inpainting candidates' });
    expect(toastStore.toasts[2]).toMatchObject({ kind: 'error', title: 'Generation failed', message: 'model exploded' });
  });

  it('error toasts carry the given actions', () => {
    const run = vi.fn();
    toastStore.error('Generation failed', 'boom', [{ label: 'Retry', run }]);

    expect(toastStore.toasts[0].actions).toHaveLength(1);
    expect(toastStore.toasts[0].actions[0].label).toBe('Retry');
    toastStore.toasts[0].actions[0].run();
    expect(run).toHaveBeenCalledOnce();
  });

  it('success and info toasts auto-dismiss after TOAST_TIMEOUT_MS', () => {
    toastStore.success('Depth map ready');
    expect(toastStore.toasts).toHaveLength(1);

    vi.advanceTimersByTime(TOAST_TIMEOUT_MS - 1);
    expect(toastStore.toasts).toHaveLength(1);

    vi.advanceTimersByTime(1);
    expect(toastStore.toasts).toHaveLength(0);
  });

  it('error toasts persist past TOAST_TIMEOUT_MS', () => {
    toastStore.error('Generation failed', 'boom');
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS * 5);
    expect(toastStore.toasts).toHaveLength(1);
  });

  it('dismiss removes a toast immediately and clears its timer', () => {
    const id = toastStore.success('Depth map ready');
    toastStore.dismiss(id);
    expect(toastStore.toasts).toHaveLength(0);

    // The cleared timer must not throw or resurrect anything later.
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS * 2);
    expect(toastStore.toasts).toHaveLength(0);
  });

  it('dismissing an id that is not present is a no-op', () => {
    toastStore.success('Depth map ready');
    toastStore.dismiss(999);
    expect(toastStore.toasts).toHaveLength(1);
  });

  it('keeps at most 4 toasts, dropping the oldest and clearing its timer', () => {
    const ids = [1, 2, 3, 4, 5].map((n) => toastStore.success(`toast ${n}`));
    expect(toastStore.toasts).toHaveLength(4);
    expect(toastStore.toasts.map((t) => t.title)).toEqual(['toast 2', 'toast 3', 'toast 4', 'toast 5']);
    expect(ids).toHaveLength(5);

    // The dropped oldest toast's timer was cleared: advancing time doesn't
    // throw and doesn't remove anything it shouldn't.
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS);
    expect(toastStore.toasts).toHaveLength(0);
  });

  it('reset clears all toasts and pending timers', () => {
    toastStore.success('a');
    toastStore.error('b', 'persists');
    toastStore.reset();
    expect(toastStore.toasts).toHaveLength(0);

    vi.advanceTimersByTime(TOAST_TIMEOUT_MS * 2);
    expect(toastStore.toasts).toHaveLength(0);
  });
});
