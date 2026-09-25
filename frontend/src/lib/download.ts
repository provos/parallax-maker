/**
 * Triggers a real browser download of `url` via a transient `<a download>`
 * click, per the migration task's explicit guidance: not `window.open`,
 * which would just navigate/pop up a tab rather than reliably producing a
 * `download` event Playwright's `page.waitForEvent('download')` can observe.
 * Used for the glTF scene export and raw slice downloads - both server
 * routes already set `Content-Disposition: attachment`, so the browser
 * downloads the response body instead of navigating to it either way; the
 * `download` attribute is a fallback/hint (and sets the suggested filename)
 * for browsers that would otherwise navigate.
 */
export function triggerDownload(url: string, filename: string): void {
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = 'noopener';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}
