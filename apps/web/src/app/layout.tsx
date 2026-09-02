import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";

export const metadata: Metadata = {
  title: "DexteraAI — Industrial Gesture Intelligence",
  description:
    "Real-time, on-device, premium gesture intelligence platform.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="antialiased min-h-screen bg-[#050505]" suppressHydrationWarning>
        <Script src="/onnx/mediapipe/hands.js" strategy="beforeInteractive" />
        {children}
      </body>
    </html>
  );
}
