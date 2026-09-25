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

  function pushClient(message: string, level: string = 'error'): void {
    entries = [...entries, { seq: nextClientSeq, level, message }];
    nextClientSeq -= 1;
  }

  async function refresh(projectId: string): Promise<void> {
    try {
      const page = await getLogs(projectId, after);
      if (page.entries.length > 0) {
        entries = [...entries, ...page.entries];
        after = page.next;
      }
    } catch {
      // Best-effort: a failed log fetch should not itself surface as an
      // error (that would recurse), and should not block the caller.
    }
  }

  function reset(): void {
    entries = [];
    after = 0;
    nextClientSeq = -1;
  }

  return {
    get entries(): DisplayLogEntry[] {
      return entries;
    },
    get last3(): DisplayLogEntry[] {
      return entries.slice(-3);
    },
    pushClient,
    refresh,
    reset,
  };
}

export const logStore = createLogStore();
export type LogStore = ReturnType<typeof createLogStore>;
