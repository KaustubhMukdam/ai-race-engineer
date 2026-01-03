/**
 * Global State Management for F1 Race Engineer
 * Uses Zustand for lightweight, TypeScript-friendly state management
 */

import { create } from 'zustand';
import { apiClient, StrategyResponse } from '../api-client';

// ==================== TYPE DEFINITIONS ====================

export interface RaceData {
  driver: string;
  position: number;
  currentLap: number;
  totalLaps: number;
  gapToLeader: number;
  gapToNext: number;
  tireCompound: string;
  tireAge: number;
  degradationRate: number;
  predictedCliffLap: number;
  pitProbability: number;
  recommendedPitLap: number;
  trackTemp: number;
  airTemp: number;
  weather: string;
  lstmUsed: boolean;
  xgbUsed: boolean;
  sessionKey?: string;
  raceName?: string;
}

export interface SessionData {
  year: number;
  event: string;
  session: string;
  sessionKey: string;
  cached: boolean;
  loaded: boolean;
}

interface RaceStore {
  // State
  raceData: RaceData | null;
  sessionData: SessionData | null;
  strategyRecommendation: StrategyResponse | null;
  isLive: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  setRaceData: (data: Partial<RaceData>) => void;
  loadSession: (year: number, event: string, session: string) => Promise<void>;
  fetchStrategyRecommendation: () => Promise<void>;
  startLiveMode: () => void;
  stopLiveMode: () => void;
  reset: () => void;
}

// ==================== INITIAL STATE ====================

const initialRaceData: RaceData = {
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
  xgbUsed: true,
};

// ==================== ZUSTAND STORE ====================

export const useRaceStore = create<RaceStore>((set, get) => ({
  // Initial state
  raceData: initialRaceData,
  sessionData: null,
  strategyRecommendation: null,
  isLive: false,
  isLoading: false,
  error: null,

  // ==================== ACTIONS ====================

  setRaceData: (data) => {
    set((state) => ({
      raceData: state.raceData ? { ...state.raceData, ...data } : null,
    }));
  },

  loadSession: async (year, event, session) => {
    set({ isLoading: true, error: null });

    try {
      const response = await apiClient.loadSession({
        year,
        event,
        session,
        force_reload: false,
      });

      const sessionKey = response.session_key;

      set({
        sessionData: {
          year,
          event,
          session,
          sessionKey,
          cached: response.cached,
          loaded: true,
        },
        isLoading: false,
      });

      // Fetch real telemetry for current driver
      const { raceData } = get();
      if (raceData) {
        try {
          const telemetry = await apiClient.getCurrentTelemetry(
            raceData.driver,
            sessionKey
          );

          // Update race data with real telemetry
          set({
            raceData: {
              ...raceData,
              sessionKey,
              raceName: sessionKey,
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
            },
          });

          // Fetch strategy recommendation with real data
          await get().fetchStrategyRecommendation();
        } catch (error) {
          console.error('Failed to fetch telemetry:', error);
        }
      }
    } catch (error: any) {
      set({
        error: error.message || 'Failed to load session',
        isLoading: false,
      });
    }
  },

  fetchStrategyRecommendation: async () => {
    const { raceData } = get();
    if (!raceData) return;

    set({ isLoading: true, error: null });

    try {
      const response = await apiClient.getStrategyRecommendation({
        driver: raceData.driver,
        current_lap: raceData.currentLap,
        total_laps: raceData.totalLaps,
        current_compound: raceData.tireCompound,
        tire_age: raceData.tireAge,
        track_temp: raceData.trackTemp,
        air_temp: raceData.airTemp,
        race_context: `P${raceData.position}, gap to leader ${raceData.gapToLeader}s`,
        race_name: raceData.raceName,
        position: raceData.position,
      });

      set({
        strategyRecommendation: response,
        isLoading: false,
      });

      // Update race data with ML model usage
      set((state) => ({
        raceData: state.raceData
          ? {
              ...state.raceData,
              lstmUsed: response.lstm_used,
              xgbUsed: response.xgb_used,
            }
          : null,
      }));
    } catch (error: any) {
      set({
        error: error.message || 'Failed to fetch strategy',
        isLoading: false,
      });
    }
  },

  startLiveMode: () => {
    const { raceData, sessionData } = get();
    if (!raceData || !sessionData) return;

    set({ isLive: true });

    // Start interval to advance laps
    const interval = setInterval(async () => {
      const state = get();
      if (!state.isLive || !state.raceData) {
        clearInterval(interval);
        return;
      }

      const currentLap = state.raceData.currentLap;
      const totalLaps = state.raceData.totalLaps;

      // Stop at end of race
      if (currentLap >= totalLaps) {
        clearInterval(interval);
        set({ isLive: false });
        return;
      }

      const nextLap = currentLap + 1;

      try {
        // Fetch real telemetry for next lap
        const history = await apiClient.getTelemetryHistory(
          state.raceData.driver,
          sessionData.sessionKey!,
          nextLap,
          nextLap
        );

        if (history.laps.length > 0) {
          const lapData = history.laps[0];

          // Calculate degradation rate from recent laps
          const recentHistory = await apiClient.getTelemetryHistory(
            state.raceData.driver,
            sessionData.sessionKey!,
            Math.max(1, nextLap - 5),
            nextLap
          );

          let degradationRate = 0;
          if (recentHistory.laps.length >= 2) {
            const firstLap = recentHistory.laps[0].LapTime_Seconds;
            const lastLap = recentHistory.laps[recentHistory.laps.length - 1].LapTime_Seconds;
            degradationRate = (lastLap - firstLap) / recentHistory.laps.length;
          }

          // Get current telemetry for position and gaps
          const currentTelemetry = await apiClient.getCurrentTelemetry(
            state.raceData.driver,
            sessionData.sessionKey!
          );

          // Calculate predicted cliff lap (simple heuristic)
          const tireAge = lapData.TyreLife;
          let predictedCliffLap = nextLap;
          if (lapData.Compound === 'SOFT') {
            predictedCliffLap = nextLap + Math.max(0, 18 - tireAge);
          } else if (lapData.Compound === 'MEDIUM') {
            predictedCliffLap = nextLap + Math.max(0, 25 - tireAge);
          } else if (lapData.Compound === 'HARD') {
            predictedCliffLap = nextLap + Math.max(0, 35 - tireAge);
          }

          // Calculate pit probability (simple heuristic based on tire age)
          let pitProbability = 0;
          if (lapData.Compound === 'SOFT' && tireAge > 15) {
            pitProbability = Math.min(0.9, (tireAge - 15) / 10);
          } else if (lapData.Compound === 'MEDIUM' && tireAge > 20) {
            pitProbability = Math.min(0.9, (tireAge - 20) / 15);
          } else if (lapData.Compound === 'HARD' && tireAge > 30) {
            pitProbability = Math.min(0.9, (tireAge - 30) / 20);
          }

          // Recommended pit lap
          const recommendedPitLap = predictedCliffLap - 2; // Pit 2 laps before cliff

          set({
            raceData: {
              ...state.raceData,
              currentLap: lapData.LapNumber,
              tireAge: lapData.TyreLife,
              tireCompound: lapData.Compound,
              trackTemp: lapData.TrackTemp,
              airTemp: lapData.AirTemp,
              degradationRate: degradationRate,
              position: currentTelemetry.position,
              gapToLeader: currentTelemetry.gap_ahead || 0,
              gapToNext: currentTelemetry.gap_behind || 0,
              predictedCliffLap: predictedCliffLap,
              pitProbability: pitProbability,
              recommendedPitLap: recommendedPitLap,
            },
          });

          // Fetch strategy every 5 laps
          if (nextLap % 5 === 0) {
            get().fetchStrategyRecommendation();
          }
        }
      } catch (error) {
        console.error('Live mode error:', error);
      }
    }, 2000); // Advance every 2 seconds

    // Store interval ID for cleanup
    (get as any).liveInterval = interval;
  },

  stopLiveMode: () => {
    set({ isLive: false });
    
    // Clear interval
    const interval = (get as any).liveInterval;
    if (interval) {
      clearInterval(interval);
      (get as any).liveInterval = null;
    }
  },

  reset: () => {
    set({
      raceData: initialRaceData,
      sessionData: null,
      strategyRecommendation: null,
      isLive: false,
      isLoading: false,
      error: null,
    });
  },
}));

// ==================== SELECTOR HOOKS ====================

export const useRaceData = () => useRaceStore((state) => state.raceData);
export const useSessionData = () => useRaceStore((state) => state.sessionData);
export const useStrategyRecommendation = () =>
  useRaceStore((state) => state.strategyRecommendation);
export const useIsLive = () => useRaceStore((state) => state.isLive);
export const useIsLoading = () => useRaceStore((state) => state.isLoading);
export const useError = () => useRaceStore((state) => state.error);