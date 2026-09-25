import { describe, expect, it } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import HelpTooltip from './HelpTooltip.svelte';

const TEXTS = ['First help text', 'Second help text'];

describe('HelpTooltip', () => {
  it('starts closed, with no tooltip panel in the DOM', () => {
    render(HelpTooltip, { label: 'Segmentation', texts: TEXTS });
    expect(screen.queryByTestId('help-panel')).toBeNull();
    expect(screen.getByTestId('help-button')).toHaveAttribute('aria-expanded', 'false');
  });

  it('opens the panel on click and links it via aria-describedby', async () => {
    render(HelpTooltip, { label: 'Segmentation', texts: TEXTS });
    const button = screen.getByTestId('help-button');

    await fireEvent.click(button);

    const panel = screen.getByTestId('help-panel');
    expect(panel).toBeInTheDocument();
    expect(button).toHaveAttribute('aria-expanded', 'true');
    expect(button.getAttribute('aria-describedby')).toBe(panel.id);
    expect(panel).toHaveTextContent('First help text');
    expect(panel).toHaveTextContent('Second help text');
  });

  it('has an accessible name that includes the context label', () => {
    render(HelpTooltip, { label: 'Inpainting', texts: TEXTS });
    expect(screen.getByRole('button', { name: 'Help: Inpainting' })).toBeInTheDocument();
  });

  it('closes again on a second click (toggle)', async () => {
    render(HelpTooltip, { label: 'Segmentation', texts: TEXTS });
    const button = screen.getByTestId('help-button');

    await fireEvent.click(button);
    expect(screen.getByTestId('help-panel')).toBeInTheDocument();

    await fireEvent.click(button);
    expect(screen.queryByTestId('help-panel')).toBeNull();
  });

  it('closes on Escape and returns focus to the button', async () => {
    render(HelpTooltip, { label: 'Segmentation', texts: TEXTS });
    const button = screen.getByTestId('help-button') as HTMLButtonElement;

    await fireEvent.click(button);
    expect(screen.getByTestId('help-panel')).toBeInTheDocument();

    await fireEvent.keyDown(button, { key: 'Escape' });

    expect(screen.queryByTestId('help-panel')).toBeNull();
    expect(document.activeElement).toBe(button);
  });

  it('closes when a click happens outside the tooltip', async () => {
    render(HelpTooltip, { label: 'Segmentation', texts: TEXTS });
    const button = screen.getByTestId('help-button');
    await fireEvent.click(button);
    expect(screen.getByTestId('help-panel')).toBeInTheDocument();

    await fireEvent.click(document.body);

    expect(screen.queryByTestId('help-panel')).toBeNull();
  });
});
