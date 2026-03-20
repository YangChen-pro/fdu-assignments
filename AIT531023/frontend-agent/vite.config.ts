import { defineConfig, loadEnv } from 'vite';
import vue from '@vitejs/plugin-vue';
import path from 'node:path';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const proxyTarget = env.VITE_AGENT_PROXY_TARGET || 'http://127.0.0.1:8000';

  return {
    plugins: [vue()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: 4173,
      host: '0.0.0.0',
      proxy: {
        '/api-proxy': {
          target: proxyTarget,
          changeOrigin: true,
          rewrite: (pathStr) => pathStr.replace(/^\/api-proxy/, ''),
        },
      },
    },
  };
});
