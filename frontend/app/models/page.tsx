'use client'

import React from 'react'
import { Brain, TrendingUp, Target, Award, Activity, CheckCircle } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts'

// Mock training history data
const lstmTrainingHistory = Array.from({ length: 50 }, (_, i) => ({
  epoch: i + 1,
  train_loss: 0.25 * Math.exp(-i / 15) + 0.01,
  val_loss: 0.28 * Math.exp(-i / 15) + 0.015,
}))

const featureImportance = [
  { feature: 'Tire Age', importance: 0.285 },
  { feature: 'Compound', importance: 0.218 },
  { feature: 'Track Temp', importance: 0.165 },
  { feature: 'Lap Number', importance: 0.142 },
  { feature: 'Degradation Rate', importance: 0.098 },
  { feature: 'Position', importance: 0.055 },
  { feature: 'Air Temp', importance: 0.037 },
]

const confusionMatrix = [
  { actual: 'Stay Out', predicted_stay: 1847, predicted_pit: 152 },
  { actual: 'Pit Now', predicted_stay: 98, predicted_pit: 623 },
]

export default function ModelsPage() {
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Model Performance</h1>
        <p className="text-gray-400">LSTM & XGBoost Training Metrics</p>
      </div>

      {/* Model Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* LSTM Model Card */}
        <div className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 border-2 border-purple-500 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Brain className="w-8 h-8 text-purple-400" />
            <div>
              <h2 className="text-2xl font-bold text-purple-300">LSTM Tire Model</h2>
              <p className="text-sm text-gray-400">Lap-by-Lap Degradation Prediction</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-900/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-4 h-4 text-green-400" />
                <span className="text-sm text-gray-400">R² Score</span>
              </div>
              <div className="text-3xl font-bold text-green-400">0.9909</div>
              <div className="text-xs text-gray-500 mt-1">Excellent fit</div>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-gray-400">MAE</span>
              </div>
              <div className="text-3xl font-bold text-blue-400">0.64s</div>
              <div className="text-xs text-gray-500 mt-1">Mean error</div>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-gray-400">RMSE</span>
              </div>
              <div className="text-3xl font-bold text-yellow-400">0.89s</div>
              <div className="text-xs text-gray-500 mt-1">Root mean sq</div>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-4 h-4 text-purple-400" />
                <span className="text-sm text-gray-400">Epochs</span>
              </div>
              <div className="text-3xl font-bold text-purple-400">50</div>
              <div className="text-xs text-gray-500 mt-1">Training iters</div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-purple-500/10 rounded-lg">
            <div className="text-sm text-purple-300 font-semibold mb-1">Architecture</div>
            <div className="text-xs text-gray-400">
              2-layer LSTM (128 hidden units) + Dense layers
            </div>
            <div className="text-xs text-gray-400 mt-1">
              Input: 10-lap sequence × 8 features
            </div>
          </div>
        </div>

        {/* XGBoost Model Card */}
        <div className="bg-gradient-to-br from-orange-500/10 to-red-500/10 border-2 border-orange-500 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Award className="w-8 h-8 text-orange-400" />
            <div>
              <h2 className="text-2xl font-bold text-orange-300">XGBoost Classifier</h2>
              <p className="text-sm text-gray-400">Optimal Pit Window Prediction</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-900/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-4 h-4 text-green-400" />
                <span className="text-sm text-gray-400">Accuracy</span>
              </div>
              <div className="text-3xl font-bold text-green-400">90.76%</div>
              <div className="text-xs text-gray-500 mt-1">Correct preds</div>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-gray-400">ROC-AUC</span>
              </div>
              <div className="text-3xl font-bold text-blue-400">0.9584</div>
              <div className="text-xs text-gray-500 mt-1">Class separation</div>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-gray-400">F1 Score</span>
              </div>
              <div className="text-3xl font-bold text-yellow-400">0.8845</div>
              <div className="text-xs text-gray-500 mt-1">Balanced metric</div>
            </div>

            <div className="bg-slate-900/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-4 h-4 text-orange-400" />
                <span className="text-sm text-gray-400">Precision</span>
              </div>
              <div className="text-3xl font-bold text-orange-400">86.53%</div>
              <div className="text-xs text-gray-500 mt-1">Pit accuracy</div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-orange-500/10 rounded-lg">
            <div className="text-sm text-orange-300 font-semibold mb-1">Configuration</div>
            <div className="text-xs text-gray-400">
              200 trees, max depth 6, learning rate 0.1
            </div>
            <div className="text-xs text-gray-400 mt-1">
              Features: tire age, compound, degradation, position
            </div>
          </div>
        </div>
      </div>

      {/* LSTM Training History */}
      <div className="bg-slate-900/50 backdrop-blur-sm border border-purple-500/30 rounded-xl p-6">
        <h3 className="text-xl font-bold text-purple-300 mb-4">LSTM Training History</h3>
        
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={lstmTrainingHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="epoch" stroke="#9ca3af" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
            <YAxis stroke="#9ca3af" label={{ value: 'Loss (MSE)', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #a855f7',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" strokeWidth={2} name="Training Loss" />
            <Line type="monotone" dataKey="val_loss" stroke="#ef4444" strokeWidth={2} name="Validation Loss" />
          </LineChart>
        </ResponsiveContainer>

        <div className="mt-4 grid grid-cols-3 gap-4">
          <div className="p-3 bg-blue-500/10 rounded-lg">
            <div className="text-sm text-gray-400">Final Train Loss</div>
            <div className="text-2xl font-bold text-blue-400">0.0124</div>
          </div>
          <div className="p-3 bg-red-500/10 rounded-lg">
            <div className="text-sm text-gray-400">Final Val Loss</div>
            <div className="text-2xl font-bold text-red-400">0.0168</div>
          </div>
          <div className="p-3 bg-green-500/10 rounded-lg">
            <div className="text-sm text-gray-400">Convergence</div>
            <div className="text-2xl font-bold text-green-400">Epoch 43</div>
          </div>
        </div>
      </div>

      {/* XGBoost Feature Importance */}
      <div className="bg-slate-900/50 backdrop-blur-sm border border-orange-500/30 rounded-xl p-6">
        <h3 className="text-xl font-bold text-orange-300 mb-4">XGBoost Feature Importance</h3>
        
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={featureImportance} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis type="number" stroke="#9ca3af" />
            <YAxis dataKey="feature" type="category" stroke="#9ca3af" width={120} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #f97316',
                borderRadius: '8px',
              }}
            />
            <Bar dataKey="importance" fill="#f97316" name="Importance" />
          </BarChart>
        </ResponsiveContainer>

        <div className="mt-4 p-4 bg-orange-500/10 rounded-lg">
          <div className="text-sm text-orange-300 font-semibold mb-2">Key Insights</div>
          <ul className="text-xs text-gray-400 space-y-1">
            <li>• <span className="text-white">Tire Age</span> is the most important feature (28.5%)</li>
            <li>• <span className="text-white">Compound type</span> significantly impacts pit decisions (21.8%)</li>
            <li>• <span className="text-white">Track temperature</span> influences degradation rates (16.5%)</li>
          </ul>
        </div>
      </div>

      {/* Confusion Matrix */}
      <div className="bg-slate-900/50 backdrop-blur-sm border border-green-500/30 rounded-xl p-6">
        <h3 className="text-xl font-bold text-green-300 mb-4">XGBoost Confusion Matrix</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-center">
            <thead>
              <tr>
                <th className="p-4"></th>
                <th className="p-4 text-blue-400 font-bold">Predicted: Stay Out</th>
                <th className="p-4 text-red-400 font-bold">Predicted: Pit Now</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-slate-700">
                <td className="p-4 text-blue-400 font-bold">Actual: Stay Out</td>
                <td className="p-4 bg-green-500/20 text-green-400 text-2xl font-bold">1847</td>
                <td className="p-4 bg-red-500/20 text-red-400 text-xl">152</td>
              </tr>
              <tr className="border-t border-slate-700">
                <td className="p-4 text-red-400 font-bold">Actual: Pit Now</td>
                <td className="p-4 bg-red-500/20 text-red-400 text-xl">98</td>
                <td className="p-4 bg-green-500/20 text-green-400 text-2xl font-bold">623</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4">
          <div className="p-3 bg-green-500/10 rounded-lg">
            <div className="text-sm text-gray-400">True Positives (Correct Pit)</div>
            <div className="text-3xl font-bold text-green-400">623</div>
          </div>
          <div className="p-3 bg-green-500/10 rounded-lg">
            <div className="text-sm text-gray-400">True Negatives (Correct Stay)</div>
            <div className="text-3xl font-bold text-green-400">1847</div>
          </div>
        </div>
      </div>

      {/* Model Deployment Info */}
      <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/30 rounded-xl p-6">
        <h3 className="text-xl font-bold text-blue-300 mb-4">Deployment Information</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <div className="text-sm font-semibold text-gray-300 mb-2">Training Data</div>
            <div className="text-xs text-gray-400 space-y-1">
              <div>• Multi-season F1 telemetry (2023-2024)</div>
              <div>• 15,000+ race laps processed</div>
              <div>• Per-circuit baseline learning</div>
              <div>• IQR outlier filtering applied</div>
            </div>
          </div>

          <div>
            <div className="text-sm font-semibold text-gray-300 mb-2">Production Environment</div>
            <div className="text-xs text-gray-400 space-y-1">
              <div>• PyTorch 2.0 (LSTM)</div>
              <div>• XGBoost 2.0 (Classifier)</div>
              <div>• FastAPI backend serving</div>
              <div>• ~50ms inference latency</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}