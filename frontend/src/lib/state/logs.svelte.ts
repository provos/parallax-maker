import { getLogs } from '../api/client';

/**
 * The log pane's entries: server-side log entries (from
 * `GET /projects/{id}/logs?after=`, per the architecture doc) merged with
 * client-side entries (e.g. `ApiError` messages) in the order they occur.
 * Client entries use negative `seq` values so they never collide with the
 * server's monotonically increasing sequence numbers.
 */
export type DisplayLogEntry = {
  seq: number;
  level: string;
  message: string;
};

function createLogStore() {
  let entries = $state<DisplayLogEntry[]>([]);
  let after = 0;
  let nextClientSeq = -1;
  // Every mutation's `finally` block calls `refresh()` independently (see
  // workflow.ts), and some controls - the Export tab's camera/displacement
  // sliders, deliberately not busy-gated (see ExportTab.svelte's own
  // comment) - can commit several times in quick succession, so multiple
  // `refresh()` calls can genuinely overlap. Two overlapping calls that both
  // read the same `after` before either updates it would otherwise both
  // fetch (and append) the very same server entries, producing duplicate
  // `seq` values and breaking `LogDrawer.svelte`'s `{#each ... (entry.seq)}`
  // key (`each_key_duplicate`). Serialize instead: a call that arrives while
  // one is already in flight just flags a follow-up fetch, which the
  // in-flight call runs itself before returning - no overlapping requests,
  // and no entry is ever silently missed.
  let inFlight = false;
  let refreshAgain = false;

  function pushClient(message: string, level: string = 'error'): void {
    entries = [...entries, { seq: nextClientSeq, level, message }];
    nextClientSeq -= 1;
  }

  async function refresh(projectId: string): Promise<void> {
    if (inFlight) {
      refreshAgain = true;
      return;
    }
    inFlight = true;
    try {
      do {
        refreshAgain = false;
        const page = await getLogs(projectId, after);
        if (page.entries.length > 0) {
          entries = [...entries, ...page.entries];
          after = page.next;
        }
      } while (refreshAgain);
    } catch {
      // Best-effort: a failed log fetch should not itself surface as an
      // error (that would recurse), and should not block the caller.
    } finally {
      inFlight = false;
    }
  }

  function reset(): void {
    entries = [];
    after = 0;
    nextClientSeq = -1;
    inFlight = false;
    refreshAgain = false;
  }

  return {
    get entries(): DisplayLogEntry[] {
      return entries;
    },
    pushClient,
    refresh,
    reset,
  };
}

export const logStore = createLogStore();
export type LogStore = ReturnType<typeof createLogStore>;
