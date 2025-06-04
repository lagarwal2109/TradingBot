import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  base: '/dashboard/',
  server: {
    host: '0.0.0.0',  // Allow access from any IP, not just localhost
    port: 3001,
    strictPort: true,
    allowedHosts: ['test.mayacode.io']      // Enable CORS for cross-domain requests
  },
});