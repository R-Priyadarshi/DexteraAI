import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    // tsconfig maps "@/*" to "./src/*" and Next honours it, but vitest does
    // not read tsconfig paths, so a test importing "@/lib/x" failed to resolve
    // while the same import compiled fine in the app.
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  // Vite 8 transforms with oxc rather than esbuild, and ignores `esbuild`
  // options entirely if both are present.
  oxc: {
    // `tsconfig.json` sets "jsx": "preserve" because Next.js compiles JSX
    // itself. Vite honours that and then cannot parse the .tsx modules two of
    // these tests import. Overriding it here affects the test transform only —
    // `next build` still reads tsconfig as written.
    jsx: { runtime: "automatic" },
  },
  test: {
    // Nothing here touches the DOM; these are pure-logic tests over the
    // gesture engine, and jsdom would only make them slower.
    environment: "node",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
