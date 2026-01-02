'use client'

import React, { useState, useEffect } from 'react'
import { Database, Calendar, Flag, Loader2, CheckCircle } from 'lucide-react'
import { apiClient } from '@/lib/api-client'
import { useRaceStore } from '@/lib/store/race-store'

interface Event {
  RoundNumber: number
  EventName: string
  EventDate: string
  EventFormat: string
  Country: string
  Location: string
}

export default function SessionSelector() {
  const [isOpen, setIsOpen] = useState(false)
  const [seasons, setSeasons] = useState<number[]>([])
  const [selectedYear, setSelectedYear] = useState<number | null>(null)
  const [events, setEvents] = useState<Event[]>([])
  const [selectedEvent, setSelectedEvent] = useState<string>('')
  const [sessionType, setSessionType] = useState<string>('Race')
  const [loading, setLoading] = useState(false)

  const { sessionData, loadSession, isLoading } = useRaceStore()

  // Load available seasons on mount
  useEffect(() => {
    const fetchSeasons = async () => {
      try {
        const availableSeasons = await apiClient.getAvailableSeasons()
        setSeasons(availableSeasons.sort((a, b) => b - a)) // Newest first
        setSelectedYear(availableSeasons[availableSeasons.length - 1]) // Default to latest
      } catch (error) {
        console.error('Failed to load seasons:', error)
      }
    }
    fetchSeasons()
  }, [])

  // Load events when year changes
  useEffect(() => {
    if (!selectedYear) return

    const fetchEvents = async () => {
      setLoading(true)
      try {
        const schedule = await apiClient.getSeasonSchedule(selectedYear)
        setEvents(schedule.events)
      } catch (error) {
        console.error('Failed to load events:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchEvents()
  }, [selectedYear])

  const handleLoadSession = async () => {
    if (!selectedYear || !selectedEvent) return

    try {
      await loadSession(selectedYear, selectedEvent, sessionType)
      setIsOpen(false)
    } catch (error) {
      console.error('Failed to load session:', error)
    }
  }

  return (
    <div className="relative">
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold transition-all ${
          sessionData?.loaded
            ? 'bg-green-500/20 border-2 border-green-500 text-green-400'
            : 'bg-blue-500/20 border-2 border-blue-500 text-blue-400 hover:bg-blue-500/30'
        }`}
      >
        <Database className="w-4 h-4" />
        {sessionData?.loaded ? (
          <>
            <CheckCircle className="w-4 h-4" />
            <span>
              {sessionData.year} {sessionData.event}
            </span>
          </>
        ) : (
          <span>Load Race Data</span>
        )}
      </button>

      {/* Modal */}
      {isOpen && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border-2 border-blue-500 rounded-2xl p-6 max-w-2xl w-full shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <Database className="w-6 h-6 text-blue-400" />
                Load Race Session
              </h2>
              <button
                onClick={() => setIsOpen(false)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Season Selector */}
            <div className="mb-4">
              <label className="block text-sm font-semibold text-gray-300 mb-2 flex items-center gap-2">
                <Calendar className="w-4 h-4" />
                Season
              </label>
              <select
                value={selectedYear || ''}
                onChange={(e) => setSelectedYear(Number(e.target.value))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="">Select Year...</option>
                {seasons.map((year) => (
                  <option key={year} value={year}>
                    {year} Season
                  </option>
                ))}
              </select>
            </div>

            {/* Event Selector */}
            <div className="mb-4">
              <label className="block text-sm font-semibold text-gray-300 mb-2 flex items-center gap-2">
                <Flag className="w-4 h-4" />
                Grand Prix
              </label>
              <select
                value={selectedEvent}
                onChange={(e) => setSelectedEvent(e.target.value)}
                disabled={!selectedYear || loading}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white focus:border-blue-500 focus:outline-none disabled:opacity-50"
              >
                <option value="">Select Event...</option>
                {events.map((event) => (
                  <option key={event.RoundNumber} value={event.EventName}>
                    Round {event.RoundNumber}: {event.EventName} ({event.Location})
                  </option>
                ))}
              </select>
            </div>

            {/* Session Type Selector */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-300 mb-2">
                Session Type
              </label>
              <div className="flex gap-2">
                {['Race', 'Qualifying', 'Practice'].map((type) => (
                  <button
                    key={type}
                    onClick={() => setSessionType(type)}
                    className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                      sessionType === type
                        ? 'bg-blue-500 text-white'
                        : 'bg-slate-800 text-gray-400 hover:bg-slate-700'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* Load Button */}
            <button
              onClick={handleLoadSession}
              disabled={!selectedYear || !selectedEvent || isLoading}
              className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-bold py-3 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Loading Session...
                </>
              ) : (
                <>
                  <Database className="w-5 h-5" />
                  Load Session
                </>
              )}
            </button>

            {/* Info Text */}
            <p className="mt-4 text-xs text-gray-500 text-center">
              This will load historical telemetry data and enable AI strategy analysis
            </p>
          </div>
        </div>
      )}
    </div>
  )
}