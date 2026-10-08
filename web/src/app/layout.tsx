import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";

export const metadata: Metadata = {
  title: "Arbiter — AI judgment for production ML",
  description:
    "Your monitoring tells you a number crossed a threshold. Arbiter tells you what to do about it.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // Geist is Vercel's own typeface; self-hosted via next/font so there is no
    // render-blocking request to a font CDN and no layout shift on load.
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body className="min-h-screen bg-bg font-sans text-fg antialiased">{children}</body>
    </html>
  );
}
