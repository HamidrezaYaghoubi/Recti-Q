import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Recti-Q | Robust 4-bit Perception",
  description:
    "Recti-Q repairs the out-of-distribution robustness of quantized vision models with a tiny feature-space adapter.",
  icons: {
    icon: "/favicon.png",
    shortcut: "/favicon.png",
  },
  openGraph: {
    title: "Recti-Q | Robust 4-bit Perception",
    description: "Robust 4-bit perception, repaired in feature space.",
    type: "website",
    images: [{ url: "/og.png", width: 1731, height: 909, alt: "Recti-Q paper website" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Recti-Q | Robust 4-bit Perception",
    description: "Robust 4-bit perception, repaired in feature space.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
