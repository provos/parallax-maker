import { describe, expect, it } from 'vitest';
import { rovingTabs } from './rovingTabindex';

function buildTablist(labels: string[]): { container: HTMLElement; tabs: HTMLButtonElement[] } {
  const container = document.createElement('div');
  container.setAttribute('role', 'tablist');
  const tabs = labels.map((label, index) => {
    const button = document.createElement('button');
    button.setAttribute('role', 'tab');
    button.tabIndex = index === 0 ? 0 : -1;
    button.textContent = label;
    container.appendChild(button);
    return button;
  });
  document.body.appendChild(container);
  return { container, tabs };
}

function press(target: HTMLElement, key: string): void {
  target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

describe('rovingTabs', () => {
  it('ArrowRight moves focus to the next tab and reports its index', () => {
    const { container, tabs } = buildTablist(['Mode', 'Segmentation', 'Inpainting']);
    let selected = -1;
    const action = rovingTabs(container, (index) => (selected = index));
    tabs[0].focus();

    press(tabs[0], 'ArrowRight');

    expect(document.activeElement).toBe(tabs[1]);
    expect(selected).toBe(1);
    action.destroy();
    container.remove();
  });

  it('ArrowRight wraps from the last tab back to the first', () => {
    const { container, tabs } = buildTablist(['Mode', 'Segmentation', 'Inpainting']);
    let selected = -1;
    const action = rovingTabs(container, (index) => (selected = index));
    tabs[2].focus();

    press(tabs[2], 'ArrowRight');

    expect(document.activeElement).toBe(tabs[0]);
    expect(selected).toBe(0);
    action.destroy();
    container.remove();
  });

  it('ArrowLeft moves focus to the previous tab, wrapping from the first to the last', () => {
    const { container, tabs } = buildTablist(['Mode', 'Segmentation', 'Inpainting']);
    let selected = -1;
    const action = rovingTabs(container, (index) => (selected = index));
    tabs[0].focus();

    press(tabs[0], 'ArrowLeft');

    expect(document.activeElement).toBe(tabs[2]);
    expect(selected).toBe(2);
    action.destroy();
    container.remove();
  });

  it('Home focuses the first tab, End focuses the last', () => {
    const { container, tabs } = buildTablist(['Mode', 'Segmentation', 'Inpainting', 'Export']);
    let selected = -1;
    const action = rovingTabs(container, (index) => (selected = index));
    tabs[1].focus();

    press(tabs[1], 'End');
    expect(document.activeElement).toBe(tabs[3]);
    expect(selected).toBe(3);

    press(tabs[3], 'Home');
    expect(document.activeElement).toBe(tabs[0]);
    expect(selected).toBe(0);

    action.destroy();
    container.remove();
  });

  it('ignores unrelated keys', () => {
    const { container, tabs } = buildTablist(['Mode', 'Segmentation']);
    let calls = 0;
    const action = rovingTabs(container, () => (calls += 1));
    tabs[0].focus();

    press(tabs[0], 'a');

    expect(document.activeElement).toBe(tabs[0]);
    expect(calls).toBe(0);
    action.destroy();
    container.remove();
  });

  it('stops responding after destroy', () => {
    const { container, tabs } = buildTablist(['Mode', 'Segmentation']);
    let calls = 0;
    const action = rovingTabs(container, () => (calls += 1));
    tabs[0].focus();
    action.destroy();

    press(tabs[0], 'ArrowRight');

    expect(calls).toBe(0);
    container.remove();
  });
});
