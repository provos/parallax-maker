import type { Page } from '@playwright/test';

/**
 * Page-level layout probes. These read only the document itself (no
 * frontend selectors), so they work for any UI.
 */

/** Resizes the browser viewport. */
export async function setViewport(page: Page, size: { width: number; height: number }): Promise<void> {
  await page.setViewportSize(size);
}

/** How far the page itself can scroll, in CSS pixels (0 = it fits the viewport). */
export async function pageOverflow(page: Page): Promise<{ vertical: number; horizontal: number }> {
  return page.evaluate(() => {
    const root = document.scrollingElement ?? document.documentElement;
    return { vertical: root.scrollHeight - window.innerHeight, horizontal: root.scrollWidth - window.innerWidth };
  });
}
