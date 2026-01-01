import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Navigation from '@/components/layout/Navigation'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'F1 AI Race Engineer',
  description: 'Real-time race strategy powered by LSTM & XGBoost',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900">
          <Navigation />
          <main className="container mx-auto">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}