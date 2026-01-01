import React, { useState, useEffect } from 'react';
import { Activity, Gauge, TrendingUp, AlertTriangle, Radio, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';

// Mock WebSocket connection (you'll replace with real WebSocket)
const useMockRaceData = () => {
  const [raceData, setRaceData] = useState({
    driver: 'VER',
    position: 3,
    currentLap: 25,
    totalLaps: 58,
    gapToLeader: 3.2,
    gapToNext: 1.4,
    tireCompound: 'MEDIUM',
    tireAge: 15,
    degradationRate: 0.0524,
    predictedCliffLap: 28,
    pitProbability: 0.32,
    recommendedPitLap: 28,
    trackTemp: 31.5,
    airTemp: 26.7,
    weather: 'Clear',
    lstmUsed: true,
    xgbUsed: true
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setRaceData(prev => ({
        ...prev,
        currentLap: prev.currentLap + 1,
        tireAge: prev.tireAge + 1,
        degradationRate: prev.degradationRate + 0.001,
        gapToLeader: prev.gapToLeader + (Math.random() - 0.5) * 0.2
      }));
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return raceData;
};

// Generate mock tire degradation data
const generateDegradationData = (tireAge, compound) => {
  const baseTime = compound === 'SOFT' ? 78 : compound === 'MEDIUM' ? 80 : 82;
  return Array.from({ length: Math.min(tireAge + 10, 30) }, (_, i) => ({
    lap: i + 1,
    lapTime: baseTime + (i * 0.05) + Math.random() * 0.2,
    predicted: baseTime + (i * 0.06)
  }));
};

const F1Dashboard = () => {
  const raceData = useMockRaceData();
  const [selectedView, setSelectedView] = useState('overview');
  const degradationData = generateDegradationData(raceData.tireAge, raceData.tireCompound);

  const raceProgress = (raceData.currentLap / raceData.totalLaps) * 100;
  const tireHealth = Math.max(0, 100 - (raceData.tireAge / 25 * 100));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900 text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-red-500">
              F1 AI Race Engineer
            </h1>
            <p className="text-blue-300 mt-1">Real-time Strategy Dashboard</p>
          </div>
          <div className="flex gap-2">
            <div className="flex items-center gap-2 px-4 py-2 bg-green-500/20 border border-green-500 rounded-lg">
              <Activity className="w-5 h-5 text-green-400 animate-pulse" />
              <span className="text-sm font-semibold text-green-400">LIVE</span>
            </div>
            {raceData.lstmUsed && (
              <div className="px-3 py-2 bg-purple-500/20 border border-purple-500 rounded-lg">
                <span className="text-xs text-purple-300">LSTM Active</span>
              </div>
            )}
            {raceData.xgbUsed && (
              <div className="px-3 py-2 bg-orange-500/20 border border-orange-500 rounded-lg">
                <span className="text-xs text-orange-300">XGBoost Active</span>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Live Race Status */}
        <div className="lg:col-span-1">
          <div className="bg-slate-900/50 backdrop-blur-sm border border-blue-500/30 rounded-xl p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-blue-300">Race Status</h2>
              <Radio className="w-5 h-5 text-blue-400" />
            </div>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-blue-500/10 rounded-lg">
                <span className="text-gray-400">Driver</span>
                <span className="text-2xl font-bold text-blue-400">{raceData.driver}</span>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-gradient-to-r from-yellow-500/20 to-red-500/20 rounded-lg">
                <span className="text-gray-400">Position</span>
                <span className="text-3xl font-bold text-yellow-400">P{raceData.position}</span>
              </div>
              
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="flex justify-between mb-2">
                  <span className="text-sm text-gray-400">Lap Progress</span>
                  <span className="text-sm font-semibold text-blue-300">
                    {raceData.currentLap} / {raceData.totalLaps}
                  </span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-blue-500 to-red-500 transition-all duration-500"
                    style={{ width: `${raceProgress}%` }}
                  />
                </div>
                <div className="mt-1 text-right text-xs text-gray-500">
                  {raceProgress.toFixed(1)}%
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                  <div className="text-xs text-gray-400 mb-1">Gap to Leader</div>
                  <div className="text-xl font-bold text-green-400">
                    {raceData.position === 1 ? 'Leading' : `+${raceData.gapToLeader.toFixed(1)}s`}
                  </div>
                </div>
                <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <div className="text-xs text-gray-400 mb-1">Gap to Next</div>
                  <div className="text-xl font-bold text-yellow-400">
                    +{raceData.gapToNext.toFixed(1)}s
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Weather & Track */}
          <div className="mt-6 bg-slate-900/50 backdrop-blur-sm border border-purple-500/30 rounded-xl p-6 shadow-2xl">
            <h3 className="text-lg font-bold text-purple-300 mb-4">Conditions</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Weather</span>
                <span className="text-purple-300 font-semibold">{raceData.weather}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Track Temp</span>
                <span className="text-orange-400 font-semibold">{raceData.trackTemp}°C</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Air Temp</span>
                <span className="text-blue-300 font-semibold">{raceData.airTemp}°C</span>
              </div>
            </div>
          </div>
        </div>

        {/* Tire Status & Degradation */}
        <div className="lg:col-span-2 space-y-6">
          {/* Tire Widget */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-red-500/30 rounded-xl p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-red-300">Tire Management</h2>
              <Gauge className="w-5 h-5 text-red-400" />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-lg">
                <div className="text-sm text-gray-400 mb-2">Current Compound</div>
                <div className="text-3xl font-bold text-red-400">{raceData.tireCompound}</div>
              </div>
              
              <div className="p-4 bg-orange-500/10 border border-orange-500/50 rounded-lg">
                <div className="text-sm text-gray-400 mb-2">Tire Age</div>
                <div className="text-3xl font-bold text-orange-400">{raceData.tireAge} laps</div>
              </div>
              
              <div className="p-4 bg-yellow-500/10 border border-yellow-500/50 rounded-lg">
                <div className="text-sm text-gray-400 mb-2">Health</div>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-bold text-yellow-400">{tireHealth.toFixed(0)}%</span>
                  {tireHealth < 30 && <AlertTriangle className="w-5 h-5 text-red-500 animate-pulse" />}
                </div>
              </div>
            </div>

            {/* Degradation Chart */}
            <div className="bg-slate-800/30 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">Predicted Degradation Curve</h3>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={degradationData}>
                  <defs>
                    <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="lap" 
                    stroke="#9ca3af" 
                    label={{ value: 'Lap', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
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
                  />
                  <Legend />
                  <Area 
                    type="monotone" 
                    dataKey="lapTime" 
                    stroke="#3b82f6" 
                    fillOpacity={1} 
                    fill="url(#colorActual)"
                    name="Actual"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="predicted" 
                    stroke="#ef4444" 
                    strokeDasharray="5 5"
                    fillOpacity={1} 
                    fill="url(#colorPredicted)"
                    name="LSTM Prediction"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 p-4 bg-gradient-to-r from-red-500/20 to-orange-500/20 border border-red-500/50 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-5 h-5 text-red-400" />
                <span className="font-semibold text-red-300">Cliff Warning</span>
              </div>
              <p className="text-sm text-gray-300">
                Predicted tire cliff at lap <span className="font-bold text-red-400">{raceData.predictedCliffLap}</span>
              </p>
              <p className="text-xs text-gray-400 mt-1">
                Degradation rate: {raceData.degradationRate.toFixed(4)}s/lap
              </p>
            </div>
          </div>

          {/* Pit Strategy Panel */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-green-500/30 rounded-xl p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-green-300">AI Pit Strategy</h2>
              <Zap className="w-5 h-5 text-green-400" />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-green-500/10 border border-green-500/50 rounded-lg">
                <div className="text-sm text-gray-400 mb-2">Pit Probability (XGBoost)</div>
                <div className="flex items-baseline gap-2">
                  <span className="text-4xl font-bold text-green-400">
                    {(raceData.pitProbability * 100).toFixed(0)}%
                  </span>
                  {raceData.pitProbability > 0.5 && (
                    <TrendingUp className="w-5 h-5 text-green-400" />
                  )}
                </div>
                <div className="mt-3 w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-green-500 to-red-500 transition-all duration-500"
                    style={{ width: `${raceData.pitProbability * 100}%` }}
                  />
                </div>
              </div>

              <div className="p-4 bg-blue-500/10 border border-blue-500/50 rounded-lg">
                <div className="text-sm text-gray-400 mb-2">Recommended Pit Lap</div>
                <div className="text-4xl font-bold text-blue-400">
                  Lap {raceData.recommendedPitLap}
                </div>
                <div className="mt-2 text-sm text-gray-400">
                  In {raceData.recommendedPitLap - raceData.currentLap} laps
                </div>
              </div>
            </div>

            <div className="mt-4 p-4 bg-gradient-to-r from-blue-500/10 to-green-500/10 border border-blue-500/30 rounded-lg">
              <h3 className="font-semibold text-blue-300 mb-2">Strategy Options</h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center p-2 bg-slate-800/50 rounded">
                  <span className="text-sm text-gray-300">2-Stop (MEDIUM → HARD)</span>
                  <span className="text-xs text-green-400 font-semibold">Recommended</span>
                </div>
                <div className="flex justify-between items-center p-2 bg-slate-800/30 rounded">
                  <span className="text-sm text-gray-400">1-Stop (MEDIUM only)</span>
                  <span className="text-xs text-yellow-400">Risky</span>
                </div>
                <div className="flex justify-between items-center p-2 bg-slate-800/30 rounded">
                  <span className="text-sm text-gray-400">3-Stop (SOFT → MEDIUM → HARD)</span>
                  <span className="text-xs text-gray-400">Too slow</span>
                </div>
              </div>
            </div>

            <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/50 rounded-lg flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm text-yellow-300 font-semibold">Multi-Agent Analysis</p>
                <p className="text-xs text-gray-300 mt-1">
                  LSTM tire model predicts cliff at lap {raceData.predictedCliffLap}. 
                  XGBoost classifier suggests {raceData.pitProbability > 0.5 ? 'pitting now' : 'extending stint'}. 
                  Strategy agent recommends lap {raceData.recommendedPitLap} for optimal undercut opportunity.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-8 text-center">
        <p className="text-sm text-gray-500">
          Powered by LSTM (R² = 0.9909) + XGBoost (ROC-AUC = 0.9584) + LangGraph Multi-Agent System
        </p>
      </div>
    </div>
  );
};

export default F1Dashboard;