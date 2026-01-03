'use client'

import React, { useState } from 'react'
import { Users, TrendingUp, Gauge, Zap, Loader2 } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts'
import { useRaceStore } from '@/lib/store/race-store'
import { apiClient } from '@/lib/api-client'

const DRIVERS = [
  'VER', 'PER', 'HAM', 'RUS', 'LEC', 'SAI', 'NOR', 'PIA', 'ALO', 'STR'
]

export default function ComparePage() {
  const { sessionData } = useRaceStore()
  const [driver1, setDriver1] = useState('VER')
  const [driver2, setDriver2] = useState('LEC')
  const [isComparing, setIsComparing] = useState(false)
  const [comparisonData, setComparisonData] = useState<any>(null)
  const [lapHistory1, setLapHistory1] = useState<any>(null)
  const [lapHistory2, setLapHistory2] = useState<any>(null)

  const runComparison = async () => {
    if (!sessionData?.sessionKey) {
      alert('Please load a race session first!')
      return
    }

    setIsComparing(true)
    try {
      // Fetch comparison metrics
      const comparison = await apiClient.compareDrivers(
        driver1,
        driver2,
        sessionData.sessionKey
      )

      // Fetch full lap history for both drivers
      const history1 = await apiClient.getTelemetryHistory(driver1, sessionData.sessionKey)
      const history2 = await apiClient.getTelemetryHistory(driver2, sessionData.sessionKey)

      setComparisonData(comparison)
      setLapHistory1(history1)
      setLapHistory2(history2)
    } catch (error: any) {
      console.error('Comparison failed:', error)
      alert(error.message || 'Comparison failed')
    } finally {
      setIsComparing(false)
    }
  }

  // Merge lap histories for chart
  const mergedLapData = React.useMemo(() => {
    if (!lapHistory1 || !lapHistory2) return []

    const data: any[] = []
    const maxLaps = Math.max(lapHistory1.laps.length, lapHistory2.laps.length)

    for (let i = 0; i < maxLaps; i++) {
      const lap1 = lapHistory1.laps[i]
      const lap2 = lapHistory2.laps[i]

      data.push({
        lap: (lap1?.LapNumber || lap2?.LapNumber) || i + 1,
        driver1_time: lap1?.LapTime_Seconds || null,
        driver2_time: lap2?.LapTime_Seconds || null,
        delta: lap1 && lap2 ? lap1.LapTime_Seconds - lap2.LapTime_Seconds : null,
      })
    }

    return data
  }, [lapHistory1, lapHistory2])

  // Radar chart data
  const radarData = React.useMemo(() => {
    if (!comparisonData) return []

    return [
      {
        metric: 'Avg Speed',
        [driver1]: 100 - (comparisonData.driver1.avg_lap_time / 100),
        [driver2]: 100 - (comparisonData.driver2.avg_lap_time / 100),
      },
      {
        metric: 'Best Lap',
        [driver1]: 100 - (comparisonData.driver1.best_lap_time / 100),
        [driver2]: 100 - (comparisonData.driver2.best_lap_time / 100),
      },
      {
        metric: 'Consistency',
        [driver1]: Math.random() * 100, // TODO: Calculate from std dev
        [driver2]: Math.random() * 100,
      },
      {
        metric: 'Tire Mgmt',
        [driver1]: Math.random() * 100, // TODO: Calculate from degradation
        [driver2]: Math.random() * 100,
      },
      {
        metric: 'Race Pace',
        [driver1]: comparisonData.driver1.total_laps / comparisonData.driver1.avg_lap_time * 10,
        [driver2]: comparisonData.driver2.total_laps / comparisonData.driver2.avg_lap_time * 10,
      },
    ]
  }, [comparisonData, driver1, driver2])

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Driver Comparison</h1>
          <p className="text-gray-400">Head-to-head performance analysis</p>
        </div>
        {sessionData?.loaded && (
          <div className="px-4 py-2 bg-green-500/20 border border-green-500 rounded-lg">
            <span className="text-green-400 font-semibold">
              {sessionData.year} {sessionData.event}
            </span>
          </div>
        )}
      </div>

      {/* Driver Selection */}
      <div className="bg-slate-900/50 backdrop-blur-sm border border-blue-500/30 rounded-xl p-6">
        <h2 className="text-xl font-bold text-blue-300 mb-4 flex items-center gap-2">
          <Users className="w-5 h-5" />
          Select Drivers to Compare
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          {/* Driver 1 */}
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              Driver 1
            </label>
            <select
              value={driver1}
              onChange={(e) => setDriver1(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white text-lg font-bold"
            >
              {DRIVERS.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          {/* VS */}
          <div className="text-center pb-2">
            <span className="text-3xl font-bold text-red-400">VS</span>
          </div>

          {/* Driver 2 */}
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              Driver 2
            </label>
            <select
              value={driver2}
              onChange={(e) => setDriver2(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white text-lg font-bold"
            >
              {DRIVERS.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
        </div>

        <button
          onClick={runComparison}
          disabled={isComparing || !sessionData?.loaded || driver1 === driver2}
          className="w-full mt-4 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-bold py-3 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isComparing ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Zap className="w-5 h-5" />
              Compare Drivers
            </>
          )}
        </button>
      </div>

      {/* Comparison Results */}
      {comparisonData && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Driver 1 Card */}
            <div className="bg-blue-500/10 border-2 border-blue-500 rounded-xl p-6">
              <h3 className="text-2xl font-bold text-blue-400 mb-4">{driver1}</h3>
              <div className="space-y-3">
                <div>
                  <div className="text-sm text-gray-400">Avg Lap Time</div>
                  <div className="text-2xl font-bold text-white">
                    {comparisonData.driver1.avg_lap_time.toFixed(3)}s
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Best Lap</div>
                  <div className="text-2xl font-bold text-green-400">
                    {comparisonData.driver1.best_lap_time.toFixed(3)}s
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Total Laps</div>
                  <div className="text-xl font-bold text-gray-300">
                    {comparisonData.driver1.total_laps}
                  </div>
                </div>
              </div>
            </div>

            {/* Delta Card */}
            <div className="bg-yellow-500/10 border-2 border-yellow-500 rounded-xl p-6 flex flex-col items-center justify-center">
              <TrendingUp className="w-12 h-12 text-yellow-400 mb-4" />
              <div className="text-sm text-gray-400 mb-2">Average Delta</div>
              <div className={`text-4xl font-bold ${
                comparisonData.delta.avg_lap_time_diff > 0 ? 'text-red-400' : 'text-green-400'
              }`}>
                {comparisonData.delta.avg_lap_time_diff > 0 ? '+' : ''}
                {comparisonData.delta.avg_lap_time_diff.toFixed(3)}s
              </div>
              <div className="text-xs text-gray-500 mt-2">
                {Math.abs(comparisonData.delta.avg_lap_time_diff) > 0 
                  ? `${driver1} is ${comparisonData.delta.avg_lap_time_diff > 0 ? 'slower' : 'faster'}`
                  : 'Equal pace'}
              </div>
            </div>

            {/* Driver 2 Card */}
            <div className="bg-red-500/10 border-2 border-red-500 rounded-xl p-6">
              <h3 className="text-2xl font-bold text-red-400 mb-4">{driver2}</h3>
              <div className="space-y-3">
                <div>
                  <div className="text-sm text-gray-400">Avg Lap Time</div>
                  <div className="text-2xl font-bold text-white">
                    {comparisonData.driver2.avg_lap_time.toFixed(3)}s
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Best Lap</div>
                  <div className="text-2xl font-bold text-green-400">
                    {comparisonData.driver2.best_lap_time.toFixed(3)}s
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Total Laps</div>
                  <div className="text-xl font-bold text-gray-300">
                    {comparisonData.driver2.total_laps}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Lap Time Evolution */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-green-500/30 rounded-xl p-6">
            <h3 className="text-xl font-bold text-green-300 mb-4">Lap Time Evolution</h3>
            
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={mergedLapData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="lap" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #10b981',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="driver1_time"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  name={driver1}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="driver2_time"
                  stroke="#ef4444"
                  strokeWidth={2}
                  name={driver2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Performance Radar */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-purple-500/30 rounded-xl p-6">
            <h3 className="text-xl font-bold text-purple-300 mb-4">Performance Radar</h3>
            
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="metric" stroke="#9ca3af" />
                <PolarRadiusAxis stroke="#9ca3af" />
                <Radar
                  name={driver1}
                  dataKey={driver1}
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.3}
                />
                <Radar
                  name={driver2}
                  dataKey={driver2}
                  stroke="#ef4444"
                  fill="#ef4444"
                  fillOpacity={0.3}
                />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  )
}