import '@testing-library/jest-dom/vitest';

// jsdom has no ResizeObserver; Svelte's `bind:clientHeight` needs one.
if (!('ResizeObserver' in globalThis)) {
  globalThis.ResizeObserver = class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as unknown as typeof ResizeObserver;
}
