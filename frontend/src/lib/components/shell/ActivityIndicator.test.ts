import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import { flushSync } from 'svelte';
import ActivityIndicator, { SHOW_DELAY_MS } from './ActivityIndicator.svelte';
import { jobStore } from '../../state/jobs.svelte';
import { projectStore } from '../../state/project.svelte';

describe('ActivityIndicator', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    jobStore.end();
    projectStore.reset();
    vi.useRealTimers();
  });

  function advance(ms: number): void {
    vi.advanceTimersByTime(ms);
    flushSync();
  }

  it('shows the running job with an indeterminate bar, then its real progress', () => {
    render(ActivityIndicator);
    jobStore.begin('inpainting');
    flushSync();
    advance(SHOW_DELAY_MS);

    expect(screen.getByTestId('activity-label')).toHaveTextContent('Generating inpainting candidates…');
    expect(screen.getByTestId('activity-bar')).toHaveClass('indeterminate');

    jobStore.setProgress(0.42);
    flushSync();
    expect(screen.getByTestId('activity-label')).toHaveTextContent('Generating inpainting candidates 42%');
    expect(screen.getByTestId('activity-bar')).not.toHaveClass('indeterminate');
    expect(screen.getByTestId('activity-bar')).toHaveAttribute('aria-valuenow', '42');

    jobStore.end();
    flushSync();
    expect(screen.queryByTestId('activity-label')).not.toBeInTheDocument();
    expect(screen.queryByTestId('activity-bar')).not.toBeInTheDocument();
  });

  it('never flashes for an operation that finishes within the show delay', () => {
    render(ActivityIndicator);
    jobStore.begin('thresholds');
    flushSync();
    advance(SHOW_DELAY_MS - 50);
    jobStore.end();
    flushSync();
    advance(1000);
    expect(screen.queryByTestId('activity-label')).not.toBeInTheDocument();
  });
});
