import type { Metadata, Viewport } from "next";
import { Archivo, JetBrains_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";
import { ServiceWorker } from "@/components/ServiceWorker";
import { asset } from "@/lib/base-path";

// Archivo: an industrial grotesk with real character at large sizes — it holds
// tight tracking without the anonymity of a default UI sans.
const archivo = Archivo({
  subsets: ["latin"],
  variable: "--font-archivo",
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

// Every number on this site is set in mono with tabular figures, so readouts
// don't reflow as values change.
const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  weight: ["400", "500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Dextera — Hand gesture recognition, on-device",
  description:
    "18 gestures at 98.1% on held-out test data. 2.1 MB model. Runs entirely in your browser — no frame ever leaves your machine.",
  manifest: "/manifest.webmanifest",
  applicationName: "Dextera",
  appleWebApp: {
    capable: true,
    title: "Dextera",
    // The console is a dark instrument panel; a translucent bar would let the
    // page's own background show through inconsistently across iOS versions.
    statusBarStyle: "black-translucent",
  },
  icons: {
    icon: "/icon.svg",
    apple: "/icon.svg",
  },
};

export const viewport: Viewport = {
  themeColor: "#0a0a0b",
  // The console fills the viewport and has its own scroll regions; letting the
  // whole page zoom breaks the alignment of the canvas overlay on the video.
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${archivo.variable} ${jetbrains.variable}`}
      suppressHydrationWarning
    >
      <body suppressHydrationWarning>
        <Script src={asset("/onnx/mediapipe/hands.js")} strategy="beforeInteractive" />
        <ServiceWorker />
        {children}
      </body>
    </html>
  );
}
