import { describe, expect, it } from 'vitest';
import { canvasSaveStore } from './canvas.svelte';

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void; reject: (err: unknown) => void } {
  let resolve!: (value: T) => void;
  let reject!: (err: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('canvasSaveStore', () => {
  it('flush resolves immediately when nothing is pending', async () => {
    canvasSaveStore.reset();
    await expect(canvasSaveStore.flush()).resolves.toBeUndefined();
  });

  it('flush waits for a registered save to settle', async () => {
    canvasSaveStore.reset();
    const { promise, resolve } = deferred<void>();
    canvasSaveStore.register(promise);

    let flushed = false;
    const flushPromise = canvasSaveStore.flush().then(() => {
      flushed = true;
    });

    // Give any queued microtasks a chance to run; flush must still be pending.
    await Promise.resolve();
    await Promise.resolve();
    expect(flushed).toBe(false);

    resolve();
    await flushPromise;
    expect(flushed).toBe(true);
  });

  it('flush does not throw when the registered save rejects', async () => {
    canvasSaveStore.reset();
    const { promise, reject } = deferred<void>();
    canvasSaveStore.register(promise);
    reject(new Error('save failed'));

    await expect(canvasSaveStore.flush()).resolves.toBeUndefined();
  });

  it('a later registration replaces the pending save', async () => {
    canvasSaveStore.reset();
    const first = deferred<void>();
    const second = deferred<void>();
    canvasSaveStore.register(first.promise);
    canvasSaveStore.register(second.promise);

    let flushed = false;
    const flushPromise = canvasSaveStore.flush().then(() => {
      flushed = true;
    });

    first.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(flushed).toBe(false); // still waiting on `second`

    second.resolve();
    await flushPromise;
    expect(flushed).toBe(true);
  });
});
