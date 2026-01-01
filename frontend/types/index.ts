// Race Data Types
export interface RaceData {
  driver: string
  position: number
  currentLap: number
  totalLaps: number
  gapToLeader: number
  gapToNext: number
  tireCompound: 'SOFT' | 'MEDIUM' | 'HARD'
  tireAge: number
  degradationRate: number
  predictedCliffLap: number
  pitProbability: number
  recommendedPitLap: number
  trackTemp: number
  airTemp: number
  weather: string
  lstmUsed: boolean
  xgbUsed: boolean
}

// Telemetry Data
export interface TelemetryData {
  lap: number
  position: number
  tire_age: number
  lap_time: number
  compound: 'SOFT' | 'MEDIUM' | 'HARD'
  sector1: number
  sector2: number
  sector3: number
  speed_trap: number
  tire_temp_fl: number
  tire_temp_fr: number
  tire_temp_rl: number
  tire_temp_rr: number
}

// Lap Data
export interface LapData {
  lap_number: number
  lap_time: number
  sector1: number
  sector2: number
  sector3: number
  compound: 'SOFT' | 'MEDIUM' | 'HARD'
  tire_age: number
  position: number
  driver: string
}

// Stint Data
export interface StintData {
  stint_number: number
  compound: 'SOFT' | 'MEDIUM' | 'HARD'
  start_lap: number
  end_lap: number
  total_laps: number
  avg_lap_time: number
  degradation_rate: number
}

// Strategy Comparison
export interface StrategyComparison {
  driver: string
  strategy_name: string
  pit_laps: number[]
  compounds: string[]
  total_race_time: number
  advantage_vs_baseline: number
}

// Model Metrics
export interface ModelMetrics {
  model_name: string
  version: string
  metrics: {
    r2_score?: number
    mae?: number
    rmse?: number
    accuracy?: number
    roc_auc?: number
    f1_score?: number
  }
  training_date: string
  experiment_id: string
}

// WebSocket Message
export interface WebSocketMessage {
  type: 'telemetry' | 'strategy' | 'status'
  data: TelemetryData | RaceData | any
  timestamp: string
}

// Competitor Data
export interface CompetitorData {
  driver: string
  position: number
  tire_compound: 'SOFT' | 'MEDIUM' | 'HARD'
  tire_age: number
  gap_to_us: number
  last_pit_lap: number | null
  predicted_pit_lap: number | null
}

// Multi-Agent Analysis
export interface MultiAgentAnalysis {
  final_decision: {
    decision: 'PIT_NOW' | 'STAY_OUT' | 'PIT_IN_X_LAPS'
    laps_to_wait: number | null
    target_tire: 'SOFT' | 'MEDIUM' | 'HARD'
    confidence: number
    reasoning: string
    key_factors: string[]
    risks: string
    alternative_strategy: string
  }
  strategy_recommendation: any
  telemetry_analysis: any
  competitor_analysis: any
  race_control_analysis: any
}

// Simulator Input
export interface SimulatorInput {
  track_name: string
  starting_compound: 'SOFT' | 'MEDIUM' | 'HARD'
  starting_lap: number
  stint_length: number
  track_temp: number
  air_temp: number
  driver: string
}

// Simulator Output
export interface SimulatorOutput {
  status: 'success' | 'error'
  predicted_laps: Array<{
    lap_number: number
    tire_age: number
    predicted_time: number
  }>
  avg_degradation_per_lap: number
  total_time_loss: number
  baseline_lap_time: number
}