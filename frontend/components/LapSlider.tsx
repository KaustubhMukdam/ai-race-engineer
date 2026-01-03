'use client'

import React, { useState, useEffect } from 'react'
import { FastForward, Rewind, SkipBack, SkipForward } from 'lucide-react'
import { useRaceStore } from '@/lib/store/race-store'
import { apiClient } from '@/lib/api-client'

export default function LapSlider() {
  const { raceData, sessionData, setRaceData, isLive } = useRaceStore()
  const [selectedLap, setSelectedLap] = useState(raceData?.currentLap || 1)
  const [isLoading, setIsLoading] = useState(false)

  // Sync slider with race data changes
  useEffect(() => {
    if (raceData?.currentLap) {
      setSelectedLap(raceData.currentLap)
    }
  }, [raceData?.currentLap])

  const fetchLapData = async (lapNumber: number) => {
    if (!raceData || !sessionData?.sessionKey) return

    setIsLoading(true)
    try {
      // Fetch telemetry history and extract specific lap
      const history = await apiClient.getTelemetryHistory(
        raceData.driver,
        sessionData.sessionKey,
        lapNumber,
        lapNumber
      )

      if (history.laps.length > 0) {
        const lapData = history.laps[0]
        
        setRaceData({
          currentLap: lapData.LapNumber,
          tireCompound: lapData.Compound,
          tireAge: lapData.TyreLife,
          trackTemp: lapData.TrackTemp,
          airTemp: lapData.AirTemp,
        })

        // Fetch new strategy for this lap
        useRaceStore.getState().fetchStrategyRecommendation()
      }
    } catch (error) {
      console.error('Failed to fetch lap data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const lap = parseInt(e.target.value)
    setSelectedLap(lap)
  }

  const handleSliderRelease = () => {
    fetchLapData(selectedLap)
  }

  const jumpToLap = (lap: number) => {
    if (!raceData) return
    const clampedLap = Math.max(1, Math.min(lap, raceData.totalLaps))
    setSelectedLap(clampedLap)
    fetchLapData(clampedLap)
  }

  if (!raceData || !sessionData) return null

  const progress = (selectedLap / raceData.totalLaps) * 100

  return (
    <div className="bg-slate-900/50 backdrop-blur-sm border border-blue-500/30 rounded-xl p-4 shadow-2xl">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-blue-300">Race Timeline</h3>
        {isLive && (
          <span className="text-xs text-green-400 animate-pulse">● LIVE MODE</span>
        )}
      </div>

      {/* Lap Counter */}
      <div className="text-center mb-3">
        <div className="text-3xl font-bold text-white">
          Lap {selectedLap} <span className="text-gray-500">/ {raceData.totalLaps}</span>
        </div>
        <div className="text-sm text-gray-400">{progress.toFixed(1)}% Complete</div>
      </div>

      {/* Slider */}
      <div className="relative mb-4">
        <input
          type="range"
          min={1}
          max={raceData.totalLaps}
          value={selectedLap}
          onChange={handleSliderChange}
          onMouseUp={handleSliderRelease}
          onTouchEnd={handleSliderRelease}
          disabled={isLive || isLoading}
          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            background: `linear-gradient(to right, #3b82f6 0%, #ef4444 ${progress}%, #334155 ${progress}%, #334155 100%)`,
          }}
        />
        
        {/* Lap markers */}
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>1</span>
          <span>{Math.floor(raceData.totalLaps / 4)}</span>
          <span>{Math.floor(raceData.totalLaps / 2)}</span>
          <span>{Math.floor((3 * raceData.totalLaps) / 4)}</span>
          <span>{raceData.totalLaps}</span>
        </div>
      </div>

      {/* Quick Jump Buttons */}
      <div className="grid grid-cols-4 gap-2">
        <button
          onClick={() => jumpToLap(1)}
          disabled={isLive || isLoading || selectedLap === 1}
          className="flex items-center justify-center gap-1 px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:cursor-not-allowed text-white rounded-lg transition-all text-sm font-semibold"
        >
          <SkipBack className="w-4 h-4" />
          Start
        </button>
        
        <button
          onClick={() => jumpToLap(selectedLap - 5)}
          disabled={isLive || isLoading || selectedLap <= 5}
          className="flex items-center justify-center gap-1 px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:cursor-not-allowed text-white rounded-lg transition-all text-sm font-semibold"
        >
          <Rewind className="w-4 h-4" />
          -5
        </button>
        
        <button
          onClick={() => jumpToLap(selectedLap + 5)}
          disabled={isLive || isLoading || selectedLap >= raceData.totalLaps - 5}
          className="flex items-center justify-center gap-1 px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:cursor-not-allowed text-white rounded-lg transition-all text-sm font-semibold"
        >
          +5
          <FastForward className="w-4 h-4" />
        </button>
        
        <button
          onClick={() => jumpToLap(raceData.totalLaps)}
          disabled={isLive || isLoading || selectedLap === raceData.totalLaps}
          className="flex items-center justify-center gap-1 px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:cursor-not-allowed text-white rounded-lg transition-all text-sm font-semibold"
        >
          End
          <SkipForward className="w-4 h-4" />
        </button>
      </div>

      {isLoading && (
        <div className="mt-3 text-center text-sm text-blue-400">
          Loading lap {selectedLap}...
        </div>
      )}

      {isLive && (
        <div className="mt-3 text-center text-xs text-gray-500">
          Lap slider disabled in live mode
        </div>
      )}
    </div>
  )
}