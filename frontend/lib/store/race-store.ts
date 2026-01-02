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

      set({
        sessionData: {
          year,
          event,
          session,
          sessionKey: response.session_key,
          cached: response.cached,
          loaded: true,
        },
        isLoading: false,
      });

      // Update race data with session info
      const { raceData } = get();
      if (raceData) {
        set({
          raceData: {
            ...raceData,
            sessionKey: response.session_key,
            raceName: `${year}_${event.replace(/\s+/g, '_')}_${session}`,
          },
        });
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
    set({ isLive: true });
    
    // Simulate live updates every 3 seconds
    const interval = setInterval(() => {
      const { raceData, isLive } = get();
      if (!isLive || !raceData) {
        clearInterval(interval);
        return;
      }

      // Increment lap and update tire age
      set({
        raceData: {
          ...raceData,
          currentLap: raceData.currentLap + 1,
          tireAge: raceData.tireAge + 1,
          degradationRate: raceData.degradationRate + 0.001,
          gapToLeader: raceData.gapToLeader + (Math.random() - 0.5) * 0.2,
        },
      });

      // Fetch new strategy recommendation every 5 laps
      if (raceData.currentLap % 5 === 0) {
        get().fetchStrategyRecommendation();
      }
    }, 3000);
  },

  stopLiveMode: () => {
    set({ isLive: false });
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