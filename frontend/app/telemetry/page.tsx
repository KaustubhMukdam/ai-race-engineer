'use client'

import React, { useState } from 'react'
import { Activity, TrendingUp, Thermometer, Zap } from 'lucide-react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, Cell
} from 'recharts'

// Mock data generators
const generateLapTimeData = () => {
  const compounds = ['SOFT', 'MEDIUM', 'HARD']
  return Array.from({ length: 50 }, (_, i) => {
    const stintChange = i === 20 || i === 40
    const compound = i < 20 ? 'MEDIUM' : i < 40 ? 'HARD' : 'MEDIUM'
    const baseTime = compound === 'SOFT' ? 78 : compound === 'MEDIUM' ? 80 : 82
    const degradation = (i % 20) * 0.05
    
    return {
      lap: i + 1,
      lapTime: baseTime + degradation + Math.random() * 0.3,
      compound,
      verstappen: baseTime - 0.5 + degradation + Math.random() * 0.2,
      norris: baseTime + 0.3 + degradation + Math.random() * 0.3
    }
  })
}

const generateSectorData = () => {
  return Array.from({ length: 10 }, (_, i) => ({
    lap: 40 + i,
    sector1: 22.5 + Math.random() * 0.5,
    sector2: 28.3 + Math.random() * 0.4,
    sector3: 29.1 + Math.random() * 0.6,
    verstappen_s1: 22.2 + Math.random() * 0.3,
    verstappen_s2: 28.0 + Math.random() * 0.3,
    verstappen_s3: 28.8 + Math.random() * 0.4
  }))
}

const generateTireHeatmap = () => {
  return Array.from({ length: 40 }, (_, i) => ({
    lap: i + 1,
    fl: 90 + (i * 0.8) + Math.random() * 5,
    fr: 92 + (i * 0.7) + Math.random() * 5,
    rl: 88 + (i * 0.9) + Math.random() * 5,
    rr: 89 + (i * 0.85) + Math.random() * 5
  }))
}

const generateSpeedTrap = () => {
  return Array.from({ length: 50 }, (_, i) => ({
    lap: i + 1,
    speed: 315 + Math.random() * 10 - (i * 0.1),
    avgSpeed: 318 - (i * 0.08)
  }))
}

const TelemetryPage = () => {
  const [selectedDriver, setSelectedDriver] = useState('VER')
  const lapTimeData = generateLapTimeData()
  const sectorData = generateSectorData()
  const tireHeatmap = generateTireHeatmap()
  const speedTrapData = generateSpeedTrap()

  const getCompoundColor = (compound: string) => {
    switch (compound) {
      case 'SOFT': return '#ef4444'
      case 'MEDIUM': return '#f59e0b'
      case 'HARD': return '#e5e7eb'
      default: return '#3b82f6'
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
          <Activity className="w-5 h-5 text-green-400 animate-pulse" />
          <span className="text-sm text-green-400 font-semibold">LIVE DATA</span>
        </div>
      </div>

      {/* Driver Selector */}
      <div className="bg-slate-900/50 backdrop-blur-sm border border-blue-500/30 rounded-xl p-4">
        <div className="flex gap-2">
          {['VER', 'NOR', 'PER', 'LEC'].map((driver) => (
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
              domain={['dataMin - 2', 'dataMax + 2']}
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
            <Line 
              type="monotone" 
              dataKey="verstappen" 
              stroke="#ef4444" 
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="VER (Comparison)"
            />
          </LineChart>
        </ResponsiveContainer>

        <div className="mt-4 grid grid-cols-3 gap-4">
          {lapTimeData.filter(d => d.lap === 20 || d.lap === 40).map((stint, idx) => (
            <div key={idx} className="p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xs text-gray-400 mb-1">Stint {idx + 1} Avg</div>
              <div className="text-lg font-bold text-white">{stint.lapTime.toFixed(3)}s</div>
              <div className="text-xs text-gray-500">{stint.compound}</div>
            </div>
          ))}
          <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
            <div className="text-xs text-gray-400 mb-1">Race Avg</div>
            <div className="text-lg font-bold text-purple-400">
              {(lapTimeData.reduce((sum, d) => sum + d.lapTime, 0) / lapTimeData.length).toFixed(3)}s
            </div>
            <div className="text-xs text-gray-500">All Laps</div>
          </div>
        </div>
      </div>

      {/* Sector Times Comparison */}
      <div className="bg-slate-900/50 backdrop-blur-sm border border-green-500/30 rounded-xl p-6">
        <h2 className="text-xl font-bold text-green-300 mb-4">Sector Times Comparison (Last 10 Laps)</h2>
        
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={sectorData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="lap" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1e293b', 
                border: '1px solid #10b981',
                borderRadius: '8px'
              }}
            />
            <Legend />
            <Bar dataKey="sector1" fill="#3b82f6" name="Sector 1" />
            <Bar dataKey="sector2" fill="#10b981" name="Sector 2" />
            <Bar dataKey="sector3" fill="#f59e0b" name="Sector 3" />
          </BarChart>
        </ResponsiveContainer>

        <div className="mt-4 p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
          <div className="text-sm text-green-300 font-semibold mb-2">Sector Analysis</div>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-gray-400">Best S1</div>
              <div className="text-white font-bold">22.234s</div>
            </div>
            <div>
              <div className="text-gray-400">Best S2</div>
              <div className="text-white font-bold">28.012s</div>
            </div>
            <div>
              <div className="text-gray-400">Best S3</div>
              <div className="text-white font-bold">28.901s</div>
            </div>
          </div>
        </div>
      </div>

      {/* Tire Temperature Heatmap */}
      <div className="bg-slate-900/50 backdrop-blur-sm border border-red-500/30 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-red-300">Tire Temperature Evolution</h2>
          <Thermometer className="w-5 h-5 text-red-400" />
        </div>
        
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={tireHeatmap}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="lap" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" domain={[80, 130]} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1e293b', 
                border: '1px solid #ef4444',
                borderRadius: '8px'
              }}
              formatter={(value: number) => value.toFixed(1) + '°C'}
            />
            <Legend />
            <Line type="monotone" dataKey="fl" stroke="#ef4444" name="Front Left" strokeWidth={2} />
            <Line type="monotone" dataKey="fr" stroke="#f59e0b" name="Front Right" strokeWidth={2} />
            <Line type="monotone" dataKey="rl" stroke="#3b82f6" name="Rear Left" strokeWidth={2} />
            <Line type="monotone" dataKey="rr" stroke="#10b981" name="Rear Right" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>

        <div className="mt-4 grid grid-cols-4 gap-3">
          {[
            { label: 'FL', value: 105.3, color: 'red' },
            { label: 'FR', value: 107.1, color: 'orange' },
            { label: 'RL', value: 102.8, color: 'blue' },
            { label: 'RR', value: 103.5, color: 'green' }
          ].map((tire) => (
            <div key={tire.label} className={`p-3 bg-${tire.color}-500/10 border border-${tire.color}-500/30 rounded-lg`}>
              <div className="text-xs text-gray-400 mb-1">{tire.label}</div>
              <div className={`text-2xl font-bold text-${tire.color}-400`}>{tire.value}°C</div>
            </div>
          ))}
        </div>
      </div>

      {/* Speed Trap */}
      <div className="bg-slate-900/50 backdrop-blur-sm border border-purple-500/30 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-purple-300">Speed Trap Analysis</h2>
          <Zap className="w-5 h-5 text-purple-400" />
        </div>
        
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={speedTrapData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="lap" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" domain={[300, 330]} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1e293b', 
                border: '1px solid #a855f7',
                borderRadius: '8px'
              }}
              formatter={(value: number) => value.toFixed(1) + ' km/h'}
            />
            <Legend />
            <Line type="monotone" dataKey="speed" stroke="#a855f7" name="Top Speed" strokeWidth={2} />
            <Line type="monotone" dataKey="avgSpeed" stroke="#6366f1" strokeDasharray="5 5" name="Avg Speed" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>

        <div className="mt-4 flex justify-between items-center p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg">
          <div>
            <div className="text-sm text-gray-400">Fastest Speed</div>
            <div className="text-2xl font-bold text-purple-400">324.7 km/h</div>
          </div>
          <div>
            <div className="text-sm text-gray-400">Average Speed</div>
            <div className="text-2xl font-bold text-purple-400">318.2 km/h</div>
          </div>
          <div>
            <div className="text-sm text-gray-400">vs VER</div>
            <div className="text-2xl font-bold text-green-400">-2.3 km/h</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TelemetryPage