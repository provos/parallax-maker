import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, JobFailedError, getJob, health, pollJob } from './client';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('ApiError parsing', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('resolves normally on a 2xx response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { ok: true, version: '0.1.0' }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await health();
    expect(res).toEqual({ ok: true, version: '0.1.0' });
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/health',
      expect.objectContaining({ method: 'GET' }),
    );
  });

  it('parses code and message from a well-formed {"error": {...}} body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(409, { error: { code: 'stale_revision', message: 'Revision is stale' } }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(health()).rejects.toMatchObject({
      name: 'ApiError',
      status: 409,
      code: 'stale_revision',
      message: 'Revision is stale',
    });
  });

  it('falls back to a generic code/message when the body is not the documented shape', async () => {
    const response = new Response('not json', { status: 500, statusText: 'Internal Server Error' });
    const fetchMock = vi.fn().mockResolvedValue(response);
    vi.stubGlobal('fetch', fetchMock);

    let caught: unknown;
    try {
      await health();
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(ApiError);
    const err = caught as ApiError;
    expect(err.status).toBe(500);
    expect(err.code).toBe('unknown');
    expect(err.message).toBe('Internal Server Error');
  });

  it('falls back to a generic code/message when the body is JSON but not the error shape', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(404, { detail: 'nope' }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(health()).rejects.toMatchObject({ status: 404, code: 'unknown' });
  });
});

describe('pollJob', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('polls until the job succeeds and resolves with the terminal job', async () => {
    const jobs = [
      { id: 'job-1', kind: 'depth', status: 'queued', progress: 0 },
      { id: 'job-1', kind: 'depth', status: 'running', progress: 0.5 },
      { id: 'job-1', kind: 'depth', status: 'succeeded', progress: 1 },
    ];
    let call = 0;
    const fetchMock = vi.fn().mockImplementation(() => {
      const job = jobs[Math.min(call, jobs.length - 1)];
      call += 1;
      return Promise.resolve(jsonResponse(200, job));
    });
    vi.stubGlobal('fetch', fetchMock);

    const progressUpdates: number[] = [];
    const result = await pollJob('job-1', {
      intervalMs: 0,
      onProgress: (job) => progressUpdates.push(job.progress),
    });

    expect(result.status).toBe('succeeded');
    expect(progressUpdates).toEqual([0, 0.5, 1]);
  });

  it('rejects with a JobFailedError when the job fails', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse(200, { id: 'job-2', kind: 'slices', status: 'failed', progress: 0.2, error: 'boom' }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const failure = pollJob('job-2', { intervalMs: 0 });
    await expect(failure).rejects.toBeInstanceOf(JobFailedError);
    await expect(failure).rejects.toMatchObject({
      message: 'boom',
      job: { id: 'job-2', status: 'failed' },
    });
  });

  it('getJob issues a GET to /api/v1/jobs/{jobId}', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(200, { id: 'job-3', kind: 'depth', status: 'running', progress: 0.1 }));
    vi.stubGlobal('fetch', fetchMock);

    await getJob('job-3');
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/jobs/job-3', expect.objectContaining({ method: 'GET' }));
  });
});
