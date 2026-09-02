import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 8765,
    strictPort: true,
    proxy: {
      "/api": "http://127.0.0.1:8766",
      "/labs": {
        target: "http://127.0.0.1:8766",
        changeOrigin: true,
        ws: true,
      },
    },
  },
  preview: {
    host: "127.0.0.1",
    port: 8767,
    strictPort: true,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: [],
  },
});
