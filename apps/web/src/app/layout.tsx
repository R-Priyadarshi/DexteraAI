import type { Metadata } from "next";
import { Archivo, JetBrains_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";

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
        <Script src="/onnx/mediapipe/hands.js" strategy="beforeInteractive" />
        {children}
      </body>
    </html>
  );
}
