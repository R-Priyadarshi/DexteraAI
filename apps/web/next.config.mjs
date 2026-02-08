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
