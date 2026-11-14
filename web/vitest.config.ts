import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  // tsconfig says `"jsx": "preserve"` because Next compiles the JSX itself;
  // Vite would honour that and hand raw JSX to the bundler. Tests compile it.
  oxc: { jsx: { runtime: "automatic" } },
  resolve: {
    alias: { "@": fileURLToPath(new URL("./", import.meta.url)) },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    // One worker: this suite is small, and the development machine is shared.
    pool: "threads",
    maxWorkers: 1,
  },
});
