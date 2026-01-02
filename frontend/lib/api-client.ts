/**
 * API Client for F1 Race Engineer Backend
 * Handles all communication with FastAPI backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ==================== TYPE DEFINITIONS ====================

export interface RaceContext {
  driver: string;
  current_lap: number;
  total_laps: number;
  position: number;
  gap_ahead?: number;
  gap_behind?: number;
  track_name: string;
  weather: string;
}

export interface StrategyRequest {
  driver: string;
  current_lap: number;
  total_laps: number;
  current_compound: string;
  tire_age: number;
  track_temp: number;
  air_temp: number;
  race_context?: string;
  race_name?: string;
  position?: number;
}

export interface StrategyResponse {
  status: string;
  driver: string;
  current_lap: number;
  recommendation: string;
  lstm_used: boolean;
  xgb_used: boolean;
  llm_model: string;
}

export interface SessionInfo {
  year: number;
  event: string;
  session: string;
  force_reload?: boolean;
}

export interface SessionLoadResponse {
  status: string;
  session_key: string;
  cached: boolean;
  metadata: any;
  message: string;
}

export interface VerstappenSimulationRequest {
  session_key: string;
  starting_lap: number;
  ending_lap: number;
  tire_compound: string;
  track_temp: number;
  air_temp: number;
}

export interface VerstappenSimulationResponse {
  status: string;
  verstappen_strategy: any;
  baseline_strategy: any;
  comparison: any;
  reasoning: string;
}

// ==================== API CLIENT ====================

class RaceEngineerAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  // ==================== HEALTH CHECK ====================
  
  async healthCheck() {
    const response = await fetch(`${this.baseURL}/health`);
    if (!response.ok) throw new Error('API health check failed');
    return response.json();
  }

  // ==================== STRATEGY ENDPOINTS ====================

  async getStrategyRecommendation(request: StrategyRequest): Promise<StrategyResponse> {
    const response = await fetch(`${this.baseURL}/strategy/recommend-pit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Strategy recommendation failed');
    }

    return response.json();
  }

  async explainDegradation(driver: string, stint: number) {
    const response = await fetch(`${this.baseURL}/strategy/explain-degradation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ driver, stint }),
    });

    if (!response.ok) throw new Error('Degradation explanation failed');
    return response.json();
  }

  async analyzeUndercut(params: {
    driver: string;
    driver_ahead: string;
    gap_seconds: number;
    your_tire_age: number;
    their_tire_age: number;
  }) {
    const response = await fetch(`${this.baseURL}/strategy/analyze-undercut`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });

    if (!response.ok) throw new Error('Undercut analysis failed');
    return response.json();
  }

  // ==================== SESSION MANAGEMENT ====================

  async getAvailableSeasons(): Promise<number[]> {
    const response = await fetch(`${this.baseURL}/sessions/seasons`);
    if (!response.ok) throw new Error('Failed to fetch seasons');
    return response.json();
  }

  async getSeasonSchedule(year: number) {
    const response = await fetch(`${this.baseURL}/sessions/schedule/${year}`);
    if (!response.ok) throw new Error('Failed to fetch schedule');
    return response.json();
  }

  async loadSession(sessionInfo: SessionInfo): Promise<SessionLoadResponse> {
    const { year, event, session, force_reload } = sessionInfo;
    const params = new URLSearchParams({
      year: year.toString(),
      event,
      session,
      force_reload: force_reload ? 'true' : 'false',
    });

    const response = await fetch(`${this.baseURL}/sessions/load?${params}`, {
      method: 'POST',
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Session load failed');
    }

    return response.json();
  }

  async getCachedSessions() {
    const response = await fetch(`${this.baseURL}/sessions/cached`);
    if (!response.ok) throw new Error('Failed to fetch cached sessions');
    return response.json();
  }

  // ==================== VERSTAPPEN SIMULATOR ====================

  async simulateVerstappenStrategy(
    request: VerstappenSimulationRequest
  ): Promise<VerstappenSimulationResponse> {
    const response = await fetch(`${this.baseURL}/verstappen/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Verstappen simulation failed');
    }

    return response.json();
  }

  async getVerstappenComparison(session_key: string) {
    const response = await fetch(
      `${this.baseURL}/verstappen/compare?session_key=${session_key}`
    );
    if (!response.ok) throw new Error('Verstappen comparison failed');
    return response.json();
  }

  // ==================== WEBSOCKET FOR LIVE STREAMING ====================

  createTelemetryStream(driver: string): WebSocket {
    const wsURL = this.baseURL.replace('http', 'ws');
    const ws = new WebSocket(`${wsURL}/api/telemetry/stream/${driver}`);
    
    ws.onopen = () => console.log(`WebSocket connected for driver: ${driver}`);
    ws.onerror = (error) => console.error('WebSocket error:', error);
    ws.onclose = () => console.log('WebSocket closed');
    
    return ws;
  }
}

// ==================== SINGLETON INSTANCE ====================

export const apiClient = new RaceEngineerAPI();

// ==================== HELPER HOOKS ====================

/**
 * React hook for real-time telemetry streaming
 */
export function useTelemetryStream(driver: string, enabled: boolean = true) {
  const [data, setData] = React.useState<any>(null);
  const [isConnected, setIsConnected] = React.useState(false);
  const wsRef = React.useRef<WebSocket | null>(null);

  React.useEffect(() => {
    if (!enabled) return;

    const ws = apiClient.createTelemetryStream(driver);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const telemetryData = JSON.parse(event.data);
      setData(telemetryData);
    };
    ws.onclose = () => setIsConnected(false);

    return () => {
      ws.close();
    };
  }, [driver, enabled]);

  return { data, isConnected };
}

// Export for use in React components
import React from 'react';