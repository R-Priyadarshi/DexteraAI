import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DexteraAI — Gesture Intelligence",
  description:
    "Real-time, on-device, privacy-preserving hand gesture recognition",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-black text-white antialiased">
        {children}
      </body>
    </html>
  );
}
