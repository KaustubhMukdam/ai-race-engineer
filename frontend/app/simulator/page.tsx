'use client'

import React, { useState } from 'react'
import { Zap, TrendingUp, AlertCircle, Target, Brain, Loader2 } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { useRaceStore } from '@/lib/store/race-store'
import { apiClient } from '@/lib/api-client'

export default function SimulatorPage() {
  const { sessionData } = useRaceStore()
  const [isSimulating, setIsSimulating] = useState(false)
  const [simulationResult, setSimulationResult] = useState<any>(null)
  const [startLap, setStartLap] = useState(1)
  const [endLap, setEndLap] = useState(30)
  const [tireCompound, setTireCompound] = useState('MEDIUM')
  const [trackTemp, setTrackTemp] = useState(35)
  const [airTemp, setAirTemp] = useState(25)

  const runSimulation = async () => {
    if (!sessionData?.sessionKey) {
      alert('Please load a race session first!')
      return
    }

    setIsSimulating(true)
    try {
      const result = await apiClient.simulateVerstappenStrategy({
        session_key: sessionData.sessionKey,
        starting_lap: startLap,
        ending_lap: endLap,
        tire_compound: tireCompound,
        track_temp: trackTemp,
        air_temp: airTemp,
      })

      setSimulationResult(result)
    } catch (error: any) {
      console.error('Simulation failed:', error)
      alert(error.message || 'Simulation failed')
    } finally {
      setIsSimulating(false)
    }
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Verstappen Simulator</h1>
          <p className="text-gray-400">Aggressive vs Baseline Strategy Comparison</p>
        </div>
        {sessionData?.loaded && (
          <div className="px-4 py-2 bg-green-500/20 border border-green-500 rounded-lg">
            <span className="text-green-400 font-semibold">
              {sessionData.year} {sessionData.event}
            </span>
          </div>
        )}
      </div>

      {/* Configuration Panel */}
      <div className="bg-slate-900/50 backdrop-blur-sm border border-blue-500/30 rounded-xl p-6">
        <h2 className="text-xl font-bold text-blue-300 mb-4 flex items-center gap-2">
          <Target className="w-5 h-5" />
          Simulation Parameters
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {/* Lap Range */}
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              Starting Lap
            </label>
            <input
              type="number"
              value={startLap}
              onChange={(e) => setStartLap(parseInt(e.target.value))}
              min={1}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              Ending Lap
            </label>
            <input
              type="number"
              value={endLap}
              onChange={(e) => setEndLap(parseInt(e.target.value))}
              min={startLap + 1}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            />
          </div>

          {/* Tire Compound */}
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              Tire Compound
            </label>
            <select
              value={tireCompound}
              onChange={(e) => setTireCompound(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            >
              <option value="SOFT">Soft</option>
              <option value="MEDIUM">Medium</option>
              <option value="HARD">Hard</option>
            </select>
          </div>

          {/* Track Temp */}
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              Track Temp (°C)
            </label>
            <input
              type="number"
              value={trackTemp}
              onChange={(e) => setTrackTemp(parseFloat(e.target.value))}
              step={0.5}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            />
          </div>

          {/* Air Temp */}
          <div>
            <label className="block text-sm font-semibold text-gray-300 mb-2">
              Air Temp (°C)
            </label>
            <input
              type="number"
              value={airTemp}
              onChange={(e) => setAirTemp(parseFloat(e.target.value))}
              step={0.5}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
            />
          </div>
        </div>

        {/* Run Simulation Button */}
        <button
          onClick={runSimulation}
          disabled={isSimulating || !sessionData?.loaded}
          className="w-full bg-gradient-to-r from-blue-500 to-red-500 hover:from-blue-600 hover:to-red-600 text-white font-bold py-3 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isSimulating ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Running Multi-Agent Simulation...
            </>
          ) : (
            <>
              <Zap className="w-5 h-5" />
              Run Verstappen Simulation
            </>
          )}
        </button>
      </div>

      {/* Simulation Results */}
      {simulationResult && (
        <>
          {/* Strategy Comparison Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Verstappen Strategy */}
            <div className="bg-red-500/10 border-2 border-red-500 rounded-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <Zap className="w-6 h-6 text-red-400" />
                <h3 className="text-xl font-bold text-red-300">Verstappen Aggressive</h3>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Avg Lap Time</span>
                  <span className="text-white font-bold">
                    {simulationResult.verstappen_strategy?.avg_lap_time?.toFixed(3)}s
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Time Loss</span>
                  <span className="text-red-400 font-bold">
                    {simulationResult.verstappen_strategy?.total_time_loss?.toFixed(2)}s
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Degradation Rate</span>
                  <span className="text-orange-400 font-bold">
                    {simulationResult.verstappen_strategy?.avg_degradation_per_lap?.toFixed(4)}s/lap
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Predicted Cliff</span>
                  <span className="text-yellow-400 font-bold">
                    Lap {simulationResult.verstappen_strategy?.predicted_cliff_lap || 'N/A'}
                  </span>
                </div>
              </div>
            </div>

            {/* Baseline Strategy */}
            <div className="bg-blue-500/10 border-2 border-blue-500 rounded-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-6 h-6 text-blue-400" />
                <h3 className="text-xl font-bold text-blue-300">Baseline Conservative</h3>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Avg Lap Time</span>
                  <span className="text-white font-bold">
                    {simulationResult.baseline_strategy?.avg_lap_time?.toFixed(3)}s
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Time Loss</span>
                  <span className="text-blue-400 font-bold">
                    {simulationResult.baseline_strategy?.total_time_loss?.toFixed(2)}s
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Degradation Rate</span>
                  <span className="text-green-400 font-bold">
                    {simulationResult.baseline_strategy?.avg_degradation_per_lap?.toFixed(4)}s/lap
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Predicted Cliff</span>
                  <span className="text-yellow-400 font-bold">
                    Lap {simulationResult.baseline_strategy?.predicted_cliff_lap || 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Multi-Agent Reasoning */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-purple-500/30 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="w-6 h-6 text-purple-400" />
              <h3 className="text-xl font-bold text-purple-300">Multi-Agent Analysis</h3>
            </div>
            
            <div className="bg-slate-800/50 rounded-lg p-4">
              <p className="text-gray-300 whitespace-pre-line">
                {simulationResult.reasoning || 'No reasoning available'}
              </p>
            </div>
          </div>

          {/* Comparison Chart */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-green-500/30 rounded-xl p-6">
            <h3 className="text-xl font-bold text-green-300 mb-4">Lap Time Comparison</h3>
            
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={simulationResult.comparison?.lap_by_lap || []}>
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
                  dataKey="verstappen_time"
                  stroke="#ef4444"
                  strokeWidth={2}
                  name="Verstappen"
                />
                <Line
                  type="monotone"
                  dataKey="baseline_time"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  name="Baseline"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Delta Chart */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-yellow-500/30 rounded-xl p-6">
            <h3 className="text-xl font-bold text-yellow-300 mb-4">Performance Delta</h3>
            
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={simulationResult.comparison?.lap_by_lap || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="lap" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #f59e0b',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="delta" fill="#f59e0b" name="Time Delta (s)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* Info Panel */}
      {!simulationResult && (
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6 flex items-start gap-4">
          <AlertCircle className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
          <div>
            <h3 className="text-lg font-bold text-blue-300 mb-2">About This Simulator</h3>
            <p className="text-gray-300 text-sm leading-relaxed">
              This simulator uses a multi-agent AI system (Strategy Agent + Telemetry Agent + LSTM tire model) 
              to compare Max Verstappen's aggressive driving style against a conservative baseline. 
              The aggressive strategy pushes tires harder for faster lap times but increases degradation, 
              while the baseline prioritizes tire preservation. Load a race session and configure parameters to see the trade-offs.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}