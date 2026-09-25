import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import ViewerTabs from './ViewerTabs.svelte';
import { uiStore } from '../../state/ui.svelte';

// Model3DViewer lazy-loads @google/model-viewer, whose real WebGL/Three.js
// internals jsdom cannot run (see Model3DViewer.test.ts); mock it so
// switching to the 3D tab here doesn't attempt that.
vi.mock('@google/model-viewer', () => ({}));

describe('ViewerTabs', () => {
  beforeEach(() => {
    uiStore.reset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('shows the 2D input-image panel by default, with the 2D tab selected', () => {
    render(ViewerTabs);
    expect(screen.getByRole('tab', { name: '2D' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: '3D' })).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByTestId('viewer-2d')).not.toHaveClass('hidden');
    expect(screen.getByTestId('viewer-3d')).toHaveClass('hidden');
  });

  it('clicking the 3D tab switches the active viewer tab and its own panel visibility', async () => {
    render(ViewerTabs);
    await fireEvent.click(screen.getByRole('tab', { name: '3D' }));

    expect(screen.getByRole('tab', { name: '3D' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: '2D' })).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByTestId('viewer-3d')).not.toHaveClass('hidden');
    expect(screen.getByTestId('viewer-2d')).toHaveClass('hidden');
  });
});
