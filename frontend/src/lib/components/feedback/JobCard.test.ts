import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import { flushSync } from 'svelte';
import JobCard from './JobCard.svelte';
import { jobStore } from '../../state/jobs.svelte';
import * as api from '../../api/client';

describe('JobCard', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    jobStore.end();
    jobStore.clearError();
  });

  it('is hidden (idle) when no job of its kinds is running and there is no matching error', () => {
    render(JobCard, { props: { kinds: ['depth'], testId: 'depth-progress' } });
    expect(screen.getByTestId('depth-progress')).toHaveClass('idle');
    expect(screen.queryByTestId('job-label')).not.toBeInTheDocument();
  });

  it('stays hidden for a running job whose kind is not in `kinds`', () => {
    jobStore.begin('inpainting');
    render(JobCard, { props: { kinds: ['depth'], testId: 'depth-progress' } });
    expect(screen.getByTestId('depth-progress')).toHaveClass('idle');
  });

  it('shows label, percent and progress bar for a running job in `kinds`', () => {
    jobStore.begin('depth');
    render(JobCard, { props: { kinds: ['depth'], testId: 'depth-progress' } });

    expect(screen.getByTestId('depth-progress')).not.toHaveClass('idle');
    expect(screen.getByTestId('job-label')).toHaveTextContent('Generating depth map');
    expect(screen.getByText('0%')).toBeInTheDocument();

    jobStore.setProgress(0.5);
    flushSync();
    expect(screen.getByText('50%')).toBeInTheDocument();
  });

  it('shows the detail line only when jobStore.detail is set', () => {
    jobStore.begin('depth');
    render(JobCard, { props: { kinds: ['depth'] } });
    expect(screen.queryByTestId('job-detail')).not.toBeInTheDocument();

    jobStore.track({ id: 'job-1', kind: 'depth', status: 'running', progress: 0.1, detail: 'Loading the depth model' });
    flushSync();
    expect(screen.getByTestId('job-detail')).toHaveTextContent('Loading the depth model');
  });

  it('shows Cancel only when the job is cancellable, and clicking it calls jobStore.cancel (a DELETE)', async () => {
    const cancelJobSpy = vi.spyOn(api, 'cancelJob').mockResolvedValue({
      id: 'job-1',
      kind: 'depth',
      status: 'cancelled',
      progress: 0.4,
    });
    jobStore.begin('depth');
    render(JobCard, { props: { kinds: ['depth'] } });
    expect(screen.queryByTestId('job-cancel')).not.toBeInTheDocument();

    jobStore.track({ id: 'job-1', kind: 'depth', status: 'running', progress: 0.4, cancellable: true });
    flushSync();
    const cancelButton = screen.getByTestId('job-cancel');
    expect(cancelButton).toHaveTextContent('Cancel');

    await fireEvent.click(cancelButton);
    expect(cancelJobSpy).toHaveBeenCalledWith('job-1');
  });

  it('shows "Cancelling…" and disables the button while a cancel is pending', async () => {
    vi.spyOn(api, 'cancelJob').mockReturnValue(new Promise(() => {}));
    jobStore.begin('depth');
    jobStore.track({ id: 'job-1', kind: 'depth', status: 'running', progress: 0.4, cancellable: true });
    render(JobCard, { props: { kinds: ['depth'] } });

    await fireEvent.click(screen.getByTestId('job-cancel'));

    const cancelButton = screen.getByTestId('job-cancel');
    expect(cancelButton).toHaveTextContent('Cancelling…');
    expect(cancelButton).toBeDisabled();
  });

  it('shows an error card for a matching kind when not running, with actions and dismiss', async () => {
    const run = vi.fn();
    jobStore.fail({ kind: 'depth', title: 'Depth map failed', message: 'model exploded', actions: [{ label: 'Retry', run }] });
    render(JobCard, { props: { kinds: ['depth'], testId: 'depth-progress' } });

    expect(screen.getByTestId('depth-progress')).toHaveClass('failed');
    expect(screen.getByTestId('job-error')).toHaveTextContent('Depth map failed');
    expect(screen.getByText('model exploded')).toBeInTheDocument();

    const actionButton = screen.getByTestId('job-error-action');
    expect(actionButton).toHaveTextContent('Retry');
    await fireEvent.click(actionButton);
    expect(run).toHaveBeenCalledOnce();

    await fireEvent.click(screen.getByTestId('job-error-dismiss'));
    expect(screen.queryByTestId('job-error')).not.toBeInTheDocument();
  });

  it('does not show an error card for a kind not in `kinds`', () => {
    jobStore.fail({ kind: 'inpainting', title: 'Generation failed', message: 'boom', actions: [] });
    render(JobCard, { props: { kinds: ['depth'], testId: 'depth-progress' } });
    expect(screen.getByTestId('depth-progress')).toHaveClass('idle');
    expect(screen.queryByTestId('job-error')).not.toBeInTheDocument();
  });

  it('prefers the running view over a stale error for the same kinds', () => {
    jobStore.begin('depth');
    // Not a state workflow.ts ever produces (begin() clears lastError), but
    // pins down JobCard's own `!running` guard on the error branch.
    jobStore.fail({ kind: 'depth', title: 'Depth map failed', message: 'boom', actions: [] });
    render(JobCard, { props: { kinds: ['depth'], testId: 'depth-progress' } });
    expect(screen.getByTestId('job-label')).toBeInTheDocument();
    expect(screen.queryByTestId('job-error')).not.toBeInTheDocument();
  });
});
