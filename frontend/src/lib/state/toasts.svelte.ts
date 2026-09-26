/**
 * Toasts (docs/redesign/HANDOFF.md §9), shown top-right of the canvas by
 * ToastStack.svelte. Success and info toasts dismiss themselves after 3 s;
 * errors stay until dismissed and can carry actions (Retry, Open settings,
 * View log).
 */

export type ToastKind = 'success' | 'info' | 'error';

export type ToastAction = { label: string; run: () => void };

export type Toast = {
  id: number;
  kind: ToastKind;
  title: string;
  message?: string;
  actions: ToastAction[];
};

/** How long success and info toasts stay up. */
export const TOAST_TIMEOUT_MS = 3000;
/** Older toasts give way beyond this many. */
const MAX_TOASTS = 4;

function createToastStore() {
  let toasts = $state<Toast[]>([]);
  let nextId = 1;
  const timers = new Map<number, ReturnType<typeof setTimeout>>();

  function dismiss(id: number): void {
    const timer = timers.get(id);
    if (timer !== undefined) clearTimeout(timer);
    timers.delete(id);
    toasts = toasts.filter((toast) => toast.id !== id);
  }

  function push(kind: ToastKind, title: string, message?: string, actions: ToastAction[] = []): number {
    const id = nextId++;
    const next = [...toasts, { id, kind, title, message, actions }];
    for (const old of next.slice(0, Math.max(0, next.length - MAX_TOASTS))) dismiss(old.id);
    toasts = next.slice(-MAX_TOASTS);
    if (kind !== 'error') timers.set(id, setTimeout(() => dismiss(id), TOAST_TIMEOUT_MS));
    return id;
  }

  return {
    get toasts(): Toast[] {
      return toasts;
    },
    success(title: string, message?: string): number {
      return push('success', title, message);
    },
    info(title: string, message?: string): number {
      return push('info', title, message);
    },
    error(title: string, message?: string, actions: ToastAction[] = []): number {
      return push('error', title, message, actions);
    },
    dismiss,
    /** Test-only: drops every toast and timer. */
    reset(): void {
      for (const timer of timers.values()) clearTimeout(timer);
      timers.clear();
      toasts = [];
      nextId = 1;
    },
  };
}

export const toastStore = createToastStore();
