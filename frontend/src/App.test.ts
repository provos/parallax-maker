import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import App from './App.svelte';

describe('App', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders the "Parallax Maker" heading and workspace placeholders', () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true, version: 'test' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    render(App);

    expect(screen.getByTestId('app-title')).toHaveTextContent('Parallax Maker');
    expect(screen.getByTestId('main-image-area')).toBeInTheDocument();
    expect(screen.getByTestId('tabs-panel')).toBeInTheDocument();
    expect(screen.getByTestId('health-status')).toBeInTheDocument();
  });

  it('shows an error status when the health check fails', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { code: 'internal', message: 'down' } }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    render(App);

    const status = await screen.findByText(/Server error/);
    expect(status).toBeInTheDocument();
  });
});
