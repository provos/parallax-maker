/**
 * Roving-tabindex keyboard navigation for an ARIA `role="tab"` strip
 * (`[role="tablist"] > [role="tab"]`), per the WAI-ARIA Authoring Practices
 * "tabs" pattern: only the active tab is in the Tab order (`tabindex="0"`,
 * set by the component itself via `tabindex={tab === active ? 0 : -1}`);
 * ArrowLeft/ArrowRight (and Home/End) move focus *and* activate the tab
 * ("automatic activation"), matching how a plain click already behaves here
 * (MainTabs.svelte/ViewerTabs.svelte have no separate "focus vs. select"
 * state to preserve).
 *
 * Dash's own tab strips (components.py's `toggle_tab_container`, CMP-17/18)
 * are plain `<label>` click targets with no keyboard support and no
 * `role="tab"` semantics at all -- this is a deliberate accessibility
 * improvement over Dash, not a parity port (see PARITY.md).
 *
 * Used as a Svelte action: `<div role="tablist" use:rovingTabs={onSelect}>`.
 */
export function rovingTabs(node: HTMLElement, onSelect: (index: number) => void) {
  function tabs(): HTMLElement[] {
    return Array.from(node.querySelectorAll<HTMLElement>('[role="tab"]'));
  }

  function handleKeydown(event: KeyboardEvent): void {
    const items = tabs();
    if (items.length === 0) return;
    const current = items.indexOf(document.activeElement as HTMLElement);

    let nextIndex: number | null = null;
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        nextIndex = current < 0 ? 0 : (current + 1) % items.length;
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        nextIndex = current < 0 ? items.length - 1 : (current - 1 + items.length) % items.length;
        break;
      case 'Home':
        nextIndex = 0;
        break;
      case 'End':
        nextIndex = items.length - 1;
        break;
      default:
        return;
    }

    event.preventDefault();
    items[nextIndex].focus();
    onSelect(nextIndex);
  }

  node.addEventListener('keydown', handleKeydown);

  return {
    destroy(): void {
      node.removeEventListener('keydown', handleKeydown);
    },
  };
}
