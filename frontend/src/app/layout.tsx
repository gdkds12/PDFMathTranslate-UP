import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css"; // Import global CSS here

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "PDF 번역 서비스", // Update title
  description: "영문 PDF를 한국어로 번역합니다.", // Update description
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko"> {/* Set language to Korean */}
      <body className={inter.className}>{children}</body>
    </html>
  );
}
