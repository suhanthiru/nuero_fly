import { defineConfig } from 'vite';

export default defineConfig({
  // Port comes from the environment so the harness can assign a free one.
  server: { port: Number(process.env.PORT) || 5173, host: '0.0.0.0' },
  // Scene binaries are already typed-array-ready; nothing should transform them.
  assetsInclude: ['**/*.bin'],
});
