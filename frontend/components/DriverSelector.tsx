'use client'

import React, { useState, useEffect } from 'react'
import { Users, ChevronDown } from 'lucide-react'
import { useRaceStore } from '@/lib/store/race-store'
import { apiClient } from '@/lib/api-client'

const COMMON_DRIVERS = [
  { code: 'VER', name: 'Max Verstappen', team: 'Red Bull Racing' },
  { code: 'PER', name: 'Sergio Perez', team: 'Red Bull Racing' },
  { code: 'HAM', name: 'Lewis Hamilton', team: 'Mercedes' },
  { code: 'RUS', name: 'George Russell', team: 'Mercedes' },
  { code: 'LEC', name: 'Charles Leclerc', team: 'Ferrari' },
  { code: 'SAI', name: 'Carlos Sainz', team: 'Ferrari' },
  { code: 'NOR', name: 'Lando Norris', team: 'McLaren' },
  { code: 'PIA', name: 'Oscar Piastri', team: 'McLaren' },
  { code: 'ALO', name: 'Fernando Alonso', team: 'Aston Martin' },
  { code: 'STR', name: 'Lance Stroll', team: 'Aston Martin' },
]

export default function DriverSelector() {
  const [isOpen, setIsOpen] = useState(false)
  const { raceData, setRaceData } = useRaceStore()

  const currentDriver = COMMON_DRIVERS.find(d => d.code === raceData?.driver) || COMMON_DRIVERS[0]

  const handleSelectDriver = async (driverCode: string) => {
    setRaceData({ driver: driverCode })
    setIsOpen(false)
    
    // Auto-fetch telemetry for new driver if session is loaded
    const sessionData = useRaceStore.getState().sessionData
    if (sessionData?.sessionKey) {
      console.log(`Fetching telemetry for ${driverCode}...`)
      
      try {
        const telemetry = await apiClient.getCurrentTelemetry(
          driverCode,
          sessionData.sessionKey
        )
        
        // Update race data with new driver's telemetry
        setRaceData({
          driver: driverCode,
          currentLap: telemetry.current_lap,
          totalLaps: telemetry.total_laps,
          position: telemetry.position,
          tireCompound: telemetry.tire_compound,
          tireAge: telemetry.tire_age,
          trackTemp: telemetry.track_temp,
          airTemp: telemetry.air_temp,
          degradationRate: telemetry.degradation_rate,
          gapToLeader: telemetry.gap_ahead || 0,
          gapToNext: telemetry.gap_behind || 0,
          weather: telemetry.weather,
        })
        
        // Fetch new strategy recommendation
        useRaceStore.getState().fetchStrategyRecommendation()
      } catch (error) {
        console.error('Failed to fetch telemetry:', error)
      }
    }
  }

  return (
    <div className="relative">
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 bg-slate-800 border-2 border-blue-500 rounded-lg font-semibold text-white hover:bg-slate-700 transition-all"
      >
        <Users className="w-4 h-4" />
        <span className="text-lg">{currentDriver.code}</span>
        <span className="text-sm text-gray-400">{currentDriver.name}</span>
        <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 z-40" 
            onClick={() => setIsOpen(false)}
          />
          
          {/* Menu */}
          <div className="absolute top-full mt-2 right-0 z-50 bg-slate-900 border-2 border-blue-500 rounded-lg shadow-2xl min-w-[300px] max-h-[400px] overflow-y-auto">
            <div className="p-2">
              <div className="px-3 py-2 text-xs font-semibold text-gray-400 uppercase">
                Select Driver
              </div>
              
              {COMMON_DRIVERS.map((driver) => (
                <button
                  key={driver.code}
                  onClick={() => handleSelectDriver(driver.code)}
                  className={`w-full px-3 py-2 rounded-lg text-left hover:bg-slate-800 transition-all ${
                    currentDriver.code === driver.code
                      ? 'bg-blue-500/20 border border-blue-500'
                      : ''
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-bold text-white">{driver.code}</div>
                      <div className="text-sm text-gray-400">{driver.name}</div>
                    </div>
                    <div className="text-xs text-gray-500">{driver.team}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}