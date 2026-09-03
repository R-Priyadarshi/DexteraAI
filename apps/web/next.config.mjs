/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  // Required for ONNX Runtime Web WASM files
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
    };
    return config;
  },
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
