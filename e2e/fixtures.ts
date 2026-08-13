import { test as base, expect, type ConsoleMessage, type Page } from '@playwright/test';

type BrowserDiagnostics = {
  consoleErrors: string[];
  pageErrors: string[];
  failedRequests: string[];
  errorResponses: string[];
};

const ignoredExternalFailures = [
  /kit\.fontawesome\.com/,
  /cdn\.jsdelivr\.net/,
  /unpkg\.com/,
  // Chromium omits the URL from this console message. A same-origin DNS
  // failure is still caught by requestfailed below; the E2E app uses an IP URL.
  /Failed to load resource: net::ERR_NAME_NOT_RESOLVED/,
];

function isIgnoredExternalFailure(text: string): boolean {
  return ignoredExternalFailures.some((pattern) => pattern.test(text));
}

function consoleText(message: ConsoleMessage): string {
  return `${message.type()}: ${message.text()}`;
}

export const test = base.extend<{ diagnostics: BrowserDiagnostics }>({
  diagnostics: [async ({ page, baseURL }, use, testInfo) => {
    const diagnostics: BrowserDiagnostics = {
      consoleErrors: [],
      pageErrors: [],
      failedRequests: [],
      errorResponses: [],
    };
    const appOrigin = new URL(baseURL ?? 'http://127.0.0.1:8050').origin;

    page.on('console', (message) => {
      if (message.type() !== 'error') return;
      const text = consoleText(message);
      if (!isIgnoredExternalFailure(text)) diagnostics.consoleErrors.push(text);
    });
    page.on('pageerror', (error) => diagnostics.pageErrors.push(error.stack ?? error.message));
    page.on('response', (response) => {
      const url = response.url();
      if (new URL(url).origin === appOrigin && response.status() >= 400) {
        diagnostics.errorResponses.push(`${response.status()} ${response.request().method()} ${url}`);
      }
    });
    page.on('requestfailed', (request) => {
      const url = request.url();
      const errorText = request.failure()?.errorText ?? 'unknown failure';
      const supersededDashCallback =
        new URL(url).pathname === '/_dash-update-component' && errorText.includes('ERR_ABORTED');
      if (new URL(url).origin === appOrigin && !supersededDashCallback) {
        diagnostics.failedRequests.push(
          `${request.method()} ${url}: ${errorText}`,
        );
      }
    });

    await use(diagnostics);

    if (
      diagnostics.consoleErrors.length ||
      diagnostics.pageErrors.length ||
      diagnostics.failedRequests.length ||
      diagnostics.errorResponses.length
    ) {
      await testInfo.attach('browser-diagnostics', {
        body: JSON.stringify(diagnostics, null, 2),
        contentType: 'application/json',
      });
    }
    expect.soft(diagnostics.pageErrors, 'uncaught browser exceptions').toEqual([]);
    expect.soft(diagnostics.consoleErrors, 'unexpected browser console errors').toEqual([]);
    expect.soft(diagnostics.failedRequests, 'failed same-origin requests').toEqual([]);
    expect.soft(diagnostics.errorResponses, 'same-origin HTTP error responses').toEqual([]);
  }, { auto: true }],
});

export { expect };

export async function disableAnimations(page: Page): Promise<void> {
  await page.addStyleTag({
    content: `
      *, *::before, *::after {
        animation-duration: 0s !important;
        transition-duration: 0s !important;
        caret-color: transparent !important;
      }
    `,
  });
}
