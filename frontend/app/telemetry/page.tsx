'use client'

import React, { useState, useEffect } from 'react'
import { Activity, TrendingUp, Thermometer, Zap, Loader2 } from 'lucide-react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import { useRaceStore } from '@/lib/store/race-store'
import { apiClient } from '@/lib/api-client'

const DRIVERS = ['VER', 'NOR', 'PER', 'LEC', 'HAM', 'RUS', 'SAI', 'PIA', 'ALO', 'STR']

export default function TelemetryPage() {
  const { sessionData } = useRaceStore()
  const [selectedDriver, setSelectedDriver] = useState('VER')
  const [isLoading, setIsLoading] = useState(false)
  const [lapTimeData, setLapTimeData] = useState<any[]>([])
  const [telemetryStats, setTelemetryStats] = useState<any>(null)

  // Fetch telemetry when driver or session changes
  useEffect(() => {
    if (sessionData?.sessionKey) {
      fetchTelemetryData()
    }
  }, [selectedDriver, sessionData])

  const fetchTelemetryData = async () => {
    if (!sessionData?.sessionKey) return

    setIsLoading(true)
    try {
      // Fetch full lap history for selected driver
      const history = await apiClient.getTelemetryHistory(
        selectedDriver,
        sessionData.sessionKey
      )

      if (history.laps && history.laps.length > 0) {
        // Transform data for charts
        const chartData = history.laps.map((lap: any, index: number) => ({
          lap: lap.LapNumber,
          lapTime: lap.LapTime_Seconds,
          compound: lap.Compound,
          tireAge: lap.TyreLife,
          trackTemp: lap.TrackTemp,
          airTemp: lap.AirTemp,
          stint: lap.Stint,
        }))

        setLapTimeData(chartData)

        // Calculate statistics
        const avgLapTime = chartData.reduce((sum: number, lap: any) => sum + lap.lapTime, 0) / chartData.length
        const bestLapTime = Math.min(...chartData.map((lap: any) => lap.lapTime))
        
        // Calculate stint averages
        const stintAverages: any = {}
        chartData.forEach((lap: any) => {
          if (!stintAverages[lap.stint]) {
            stintAverages[lap.stint] = { total: 0, count: 0, compound: lap.compound }
          }
          stintAverages[lap.stint].total += lap.lapTime
          stintAverages[lap.stint].count += 1
        })

        const stints = Object.keys(stintAverages).map((stint) => ({
          stint: parseInt(stint),
          avg: stintAverages[stint].total / stintAverages[stint].count,
          compound: stintAverages[stint].compound,
        }))

        setTelemetryStats({
          avgLapTime,
          bestLapTime,
          totalLaps: chartData.length,
          stints,
        })
      }
    } catch (error) {
      console.error('Failed to fetch telemetry:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const getCompoundColor = (compound: string) => {
    switch (compound) {
      case 'SOFT': return '#ef4444'
      case 'MEDIUM': return '#f59e0b'
      case 'HARD': return '#e5e7eb'
      case 'INTERMEDIATE': return '#10b981'
      case 'WET': return '#3b82f6'
      default: return '#6b7280'
    }
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Telemetry Analysis</h1>
          <p className="text-gray-400">Deep dive into performance metrics</p>
        </div>
        <div className="flex items-center gap-2">
          {sessionData?.loaded ? (
            <div className="px-4 py-2 bg-green-500/20 border border-green-500 rounded-lg flex items-center gap-2">
              <Activity className="w-4 h-4 text-green-400" />
              <span className="text-sm text-green-400 font-semibold">
                {sessionData.year} {sessionData.event}
              </span>
            </div>
          ) : (
            <div className="px-4 py-2 bg-red-500/20 border border-red-500 rounded-lg">
              <span className="text-sm text-red-400 font-semibold">No Session Loaded</span>
            </div>
          )}
        </div>
      </div>

      {/* Driver Selector */}
      <div className="bg-slate-900/50 backdrop-blur-sm border border-blue-500/30 rounded-xl p-4">
        <div className="flex gap-2">
          {DRIVERS.map((driver) => (
            <button
              key={driver}
              onClick={() => setSelectedDriver(driver)}
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                selectedDriver === driver
                  ? 'bg-blue-500 text-white'
                  : 'bg-slate-800 text-gray-400 hover:bg-slate-700'
              }`}
            >
              {driver}
            </button>
          ))}
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
          <span className="ml-3 text-gray-400">Loading telemetry data...</span>
        </div>
      )}

      {/* No Session Warning */}
      {!sessionData?.loaded && !isLoading && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-6 text-center">
          <p className="text-yellow-400 font-semibold">
            Please load a race session from the Dashboard first
          </p>
        </div>
      )}

      {/* Telemetry Data */}
      {!isLoading && sessionData?.loaded && lapTimeData.length > 0 && (
        <>
          {/* Stats Cards */}
          {telemetryStats && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
                <div className="text-sm text-gray-400 mb-1">Average Lap Time</div>
                <div className="text-3xl font-bold text-blue-400">
                  {telemetryStats.avgLapTime.toFixed(3)}s
                </div>
              </div>

              <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4">
                <div className="text-sm text-gray-400 mb-1">Best Lap Time</div>
                <div className="text-3xl font-bold text-green-400">
                  {telemetryStats.bestLapTime.toFixed(3)}s
                </div>
              </div>

              <div className="bg-purple-500/10 border border-purple-500/30 rounded-xl p-4">
                <div className="text-sm text-gray-400 mb-1">Total Laps</div>
                <div className="text-3xl font-bold text-purple-400">
                  {telemetryStats.totalLaps}
                </div>
              </div>

              <div className="bg-orange-500/10 border border-orange-500/30 rounded-xl p-4">
                <div className="text-sm text-gray-400 mb-1">Stints Completed</div>
                <div className="text-3xl font-bold text-orange-400">
                  {telemetryStats.stints.length}
                </div>
              </div>
            </div>
          )}

          {/* Lap Time Evolution */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-blue-500/30 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-blue-300">Lap Time Evolution</h2>
              <TrendingUp className="w-5 h-5 text-blue-400" />
            </div>
            
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={lapTimeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="lap" 
                  stroke="#9ca3af"
                  label={{ value: 'Lap Number', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
                />
                <YAxis 
                  stroke="#9ca3af"
                  label={{ value: 'Lap Time (s)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1e293b', 
                    border: '1px solid #3b82f6',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number) => value.toFixed(3) + 's'}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="lapTime" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  dot={false}
                  name={`${selectedDriver} Lap Time`}
                />
              </LineChart>
            </ResponsiveContainer>

            {/* Stint Averages */}
            {telemetryStats && telemetryStats.stints.length > 0 && (
              <div className="mt-4 grid grid-cols-3 gap-4">
                {telemetryStats.stints.map((stint: any) => (
                  <div key={stint.stint} className="p-3 bg-slate-800/50 rounded-lg">
                    <div className="text-xs text-gray-400 mb-1">
                      Stint {stint.stint} ({stint.compound})
                    </div>
                    <div className="text-lg font-bold text-white">
                      {stint.avg.toFixed(3)}s
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Tire Age vs Lap Time */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-orange-500/30 rounded-xl p-6">
            <h2 className="text-xl font-bold text-orange-300 mb-4">Tire Degradation Analysis</h2>
            
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lapTimeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="tireAge" 
                  stroke="#9ca3af"
                  label={{ value: 'Tire Age (laps)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis stroke="#9ca3af" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1e293b', 
                    border: '1px solid #f97316',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="lapTime" 
                  stroke="#f97316" 
                  strokeWidth={2}
                  dot={(props: any) => {
                    const { cx, cy, payload } = props
                    return (
                      <circle
                        cx={cx}
                        cy={cy}
                        r={3}
                        fill={getCompoundColor(payload.compound)}
                      />
                    )
                  }}
                  name="Lap Time"
                />
              </LineChart>
            </ResponsiveContainer>

            <div className="mt-4 flex gap-4 justify-center">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <span className="text-xs text-gray-400">Soft</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-orange-500" />
                <span className="text-xs text-gray-400">Medium</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-gray-300" />
                <span className="text-xs text-gray-400">Hard</span>
              </div>
            </div>
          </div>

          {/* Temperature Evolution */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-purple-500/30 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-purple-300">Track Conditions</h2>
              <Thermometer className="w-5 h-5 text-purple-400" />
            </div>
            
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={lapTimeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="lap" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1e293b', 
                    border: '1px solid #a855f7',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="trackTemp" 
                  stroke="#f97316" 
                  strokeWidth={2}
                  name="Track Temp (°C)"
                  dot={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="airTemp" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  name="Air Temp (°C)"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  )
}