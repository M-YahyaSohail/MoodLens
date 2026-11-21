import { Geist_Mono, Outfit, Roboto } from "next/font/google"

import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { cn } from "@/lib/utils";

const robotoHeading = Roboto({subsets:['latin'],variable:'--font-heading'});

const outfit = Outfit({subsets:['latin'],variable:'--font-sans'})

const fontMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
})

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn("antialiased", fontMono.variable, "font-sans", outfit.variable, robotoHeading.variable)}
    >
      <body>
        <ThemeProvider>
          <header className="sticky top-0 z-40 w-full border-b border-background/50 bg-background/80 backdrop-blur-sm supports-[backdrop-blur]:bg-background/60">
            <div className="flex max-w-7xl items-center justify-between gap-4 px-6 py-3 mx-auto">
              <div className="flex items-center gap-3">
                <span className="text-xl font-heading font-semibold tracking-tighter uppercase">MoodLens</span>
                <span className="text-xs font-medium tracking-widest text-muted-foreground uppercase">Multimodal emotion &amp; sentiment analysis</span>
              </div>
              <div className="hidden md:flex items-center gap-2 text-xs text-muted-foreground">
                <span
                  className="flex h-2 w-2 shrink-0 rounded-full bg-emerald-500 animate-pulse"
                  aria-label="API online"
                />
                <span>FastAPI · 127.0.0.1:8000</span>
              </div>
            </div>
          </header>
          <main>{children}</main>
        </ThemeProvider>
      </body>
    </html>
  )
}
