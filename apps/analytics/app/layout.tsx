import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Avatar Model Analytics",
  description: "Talking-head model comparison matrix and cost projections",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,560;0,9..40,650;1,9..40,400&family=Fraunces:opsz,wght@9..144,500;9..144,560&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
