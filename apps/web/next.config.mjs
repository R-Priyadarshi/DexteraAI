/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  // Turbopack is the default bundler as of Next 16, and it resolves browser
  // conditions itself rather than polyfilling Node built-ins, so the
  // `resolve.fallback` shim that used to sit here for onnxruntime-web's `fs`
  // and `path` references is no longer needed. Declaring the key explicitly —
  // even empty — is what tells Next this config has been migrated on purpose
  // rather than left half-converted.
  turbopack: {},
  // Applies to `next dev` and `next start` only. Under `output: "export"`
  // there is no server, so Next drops these — the static host has to send
  // them. `public/_headers` covers Netlify and Cloudflare Pages; see
  // docs/DEPLOYMENT-HEADERS.md for nginx, Vercel and GitHub Pages.
  headers: async () => [
    {
      source: "/(.*)",
      headers: [
        // Required for SharedArrayBuffer (ONNX Runtime Web)
        {
          key: "Cross-Origin-Opener-Policy",
          value: "same-origin",
        },
        {
          key: "Cross-Origin-Embedder-Policy",
          value: "require-corp",
        },
      ],
    },
  ],
};

export default nextConfig;
