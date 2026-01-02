'use client'

import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Home, Activity, GitCompare, Settings, BarChart3 } from 'lucide-react'

const Navigation = () => {
  const pathname = usePathname()
  
  const navItems = [
    { href: '/', label: 'Dashboard', icon: Home },
    { href: '/telemetry', label: 'Telemetry', icon: Activity },
    { href: '/compare', label: 'Compare', icon: GitCompare },
    { href: '/simulator', label: 'Simulator', icon: Settings },
    { href: '/models', label: 'Models', icon: BarChart3 },
  ]
  
  return (
    <nav className="bg-slate-900/80 backdrop-blur-lg border-b border-blue-500/30 sticky top-0 z-50">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-red-500 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">F1</span>
            </div>
            <span className="text-xl font-bold text-white">AI Race Engineer</span>
          </div>
          
          <div className="flex gap-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = pathname === item.href
              
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`
                    flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200
                    ${isActive 
                      ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50' 
                      : 'text-gray-400 hover:text-white hover:bg-slate-800/50'
                    }
                  `}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{item.label}</span>
                </Link>
              )
            })}
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navigation