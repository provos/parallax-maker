import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';

// The Dash Flask server (parallax_maker/server.py) serves this build under
// /next/ and proxies the API + e2e test-only endpoints during development.
// See docs/svelte-migration/ARCHITECTURE.md ("Frontend layout", HTTP contract).
export default defineConfig({
  base: '/next/',
  plugins: [svelte(), tailwindcss()],
  build: {
    outDir: '../parallax_maker/static/next',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8050',
        changeOrigin: true,
      },
      '/__e2e__': {
        target: 'http://127.0.0.1:8050',
        changeOrigin: true,
      },
    },
  },
  // Under Vitest, force Svelte's package exports to resolve their "browser"
  // condition (client runtime) instead of "node"/SSR; otherwise
  // @testing-library/svelte's `mount()` fails with
  // "mount(...) is not available on the server".
  resolve: process.env.VITEST ? { conditions: ['browser'] } : undefined,
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/setupTests.ts'],
    include: ['src/**/*.{test,spec}.{js,ts}'],
  },
});
