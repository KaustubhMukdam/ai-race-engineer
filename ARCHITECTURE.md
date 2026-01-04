# F1 AI Race Engineer – Architecture

This document explains how the system is wired end‑to‑end: data flow, services, and ML components.

---

## 1. System Overview

At a high level, the project has three layers:

1. **Data & Models**
   - FastF1 telemetry ingestion
   - Preprocessing pipeline
   - LSTM + XGBoost models

2. **Backend API (FastAPI)**
   - Session management
   - Telemetry endpoints
   - Strategy endpoints
   - Verstappen simulator

3. **Frontend Dashboard (Next.js)**
   - Real‑time strategy view
   - Telemetry/compare views
   - Simulator UI

---

## 2. Data & ML Layer

### 2.1 Telemetry Sources

- Raw data is pulled using **FastF1** and cached in `data/raw/fastf1_cache/`.
- For each race, the preprocessing pipeline generates:
  - `processed_laps.csv` – cleaned lap‑by‑lap telemetry
  - `tire_degradation_analysis.csv` – stint‑level summaries
  - `optimal_pit_windows.json` – pre‑computed pit windows

### 2.2 LSTM Tire Degradation Model

- Input features (per lap):
  - Lap time, tire compound, tire age
  - Track and air temperature
  - Stint index, race identifier
- Objective:
  - Predict degradation rate and future lap times over a stint.
- Output:
  - Average degradation per lap
  - Predicted lap‑by‑lap times
  - Estimated “cliff lap” where degradation accelerates

Models are trained offline and stored (via Git LFS) in `ml/saved_models/`.

### 2.3 XGBoost Pit Window Classifier

- Features:
  - Current lap, race progress
  - Tire age and compound
  - Degradation rate estimate (from LSTM or recent laps)
  - Track and air temperature
  - Driver position and last lap time
- Output:
  - `should_pit` (boolean)
  - `pit_probability`
  - `confidence` and optional “reason” string

This model supports both the strategy dashboard and dedicated pit‑probability API endpoints.

---

## 3. Backend API

The backend is a **FastAPI** application exposing several routers:

- `sessions`:
  - Load and preprocess a session (season, event, session type)
  - List and delete cached sessions
- `telemetry`:
  - Get current telemetry snapshot for a driver
  - Get historical lap‑by‑lap data
  - Get pit probability for a given driver and lap
- `strategy`:
  - Generate pit strategy recommendation
  - Explain tire degradation for a stint
  - Analyze undercut opportunities vs the car ahead
  - Expose LSTM availability status
- `verstappen`:
  - Run an aggressive vs baseline stint simulation using the LSTM model

On startup the backend:

1. Ensures the session manager is initialized.
2. Loads preprocessed race data into the strategy agent.
3. Initializes the LSTM and XGBoost inference components if models are available.

---

## 4. Frontend Dashboard

The frontend is built with **Next.js (App Router)** and **Tailwind CSS**.

Key UI components:

- **Navigation** – Dashboard, Telemetry, Compare, Simulator, Models.
- **SessionSelector** – choose season/event/session.
- **DriverSelector** – choose driver (e.g., VER, NOR, LEC).
- **Race Status Panel** – driver, position, lap progress, gaps.
- **Tire Management Panel** – compound, age, health, predicted degradation chart.
- **Strategy Toggles** – enable/disable LSTM and XGBoost.
- **Simulator View** – runs Verstappen aggressive vs baseline simulations.

State is managed via a race store that coordinates:

- Selected session and driver
- Loaded telemetry and strategy results
- Loading/error states and user selections

All API calls are routed through a typed `api-client` module, so the UI stays thin and declarative.

---

## 5. Deployment & Environments

### 5.1 Backend (Render)

- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app -k uvicorn.workers.UvicornWorker`
- Environment variables:
  - `APP_ENV=production`
  - Optional: keys for external model storage if used

### 5.2 Frontend (Vercel)

- Build: `npm run build`
- Key env var:
  - `NEXT_PUBLIC_API_URL=https://ai-race-engineer.onrender.com`

Both services are configured to auto‑deploy on push to `main`.

---

## 6. Extending the System

Here are some ideas for future extensions:

- **More tracks & seasons**  
  Automate telemetry collection for full seasons and multiple years.

- **Live race integration**  
  Stream live timing (where terms allow) and update the dashboard in real time.

- **LLM “Race Engineer”**  
  Wrap strategy outputs in a conversational agent that can answer questions like  
  “If we pit now, where do we rejoin relative to Norris?”

- **User profiles & saved strategies**  
  Persist user scenarios and simulations in a database for later review.

---

If you’re reading this from the repo and want to dive deeper, start with:

- `backend/routes/strategy.py` – strategy API
- `backend/agents/strategy_agent.py` – orchestration of LSTM + XGBoost + prompts
- `frontend/app/page.tsx` – main dashboard layout
