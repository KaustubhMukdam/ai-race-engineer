# F1 AI Race Engineer 🏎️

An end-to-end **AI Race Engineer** that analyzes real-time F1 telemetry, predicts tire degradation with an LSTM model, evaluates pit windows with XGBoost, and delivers strategy recommendations through a modern web dashboard.

> "What if you had your own F1 race engineer in the browser?"

---

## 🚀 Live Demo

- **Frontend (Vercel)**: https://ai-race-engineer-8bk1hdaim-kaustubhs-projects-8cdc8a98.vercel.app/models
- **Backend API (Render)**: https://ai-race-engineer.onrender.com

---

## ✨ Core Features

- **Real-time strategy dashboard**
  - Live race status: driver, position, lap progress, gaps
  - Tire state: compound, age, health, predicted degradation curve
  - Strategy toggles: LSTM on/off, XGBoost pit suggestions

- **ML-powered decision engine**
  - **LSTM tire degradation model** to forecast lap-by-lap pace drop
  - **XGBoost pit window classifier** for "pit now vs stay out"
  - Multi-race training on historical telemetry to generalize across tracks

- **Telemetry & session management**
  - Processed F1 telemetry (FastF1) for multiple seasons and tracks
  - Session loading & caching (e.g., *2024 Abu Dhabi GP – Race*)
  - Driver comparison view for lap times and degradation

- **Developer-friendly architecture**
  - FastAPI backend with clean modular routers
  - Next.js + Tailwind CSS frontend
  - Typed API client and state management for predictable UI

---

## 🧱 Tech Stack

**Frontend**
- Next.js (App Router)
- TypeScript
- Tailwind CSS
- React hooks + Zustand (race store)

**Backend**
- FastAPI
- Pydantic
- Uvicorn / Gunicorn

**ML & Data**
- PyTorch (LSTM)
- XGBoost
- Pandas / NumPy
- FastF1 for telemetry ingestion

**Infrastructure**
- Vercel (frontend)
- Render (backend)
- Git LFS for large models & datasets

---

## 🧠 How It Works (High Level)

1. **Data ingestion & preprocessing**
   - Raw F1 telemetry is pulled via FastF1 and cached locally
   - Preprocessing scripts generate:
     - `processed_laps.csv`
     - `tire_degradation_analysis.csv`
     - `optimal_pit_windows.json`

2. **Model training**
   - LSTM model is trained on multi-race lap time and tire data to predict:
     - Degradation rate per lap
     - Tire "cliff" behavior over a stint
   - XGBoost classifier is trained on race features to output:
     - Pit probability
     - Recommended pit decision
     - Confidence score

3. **API layer**
   - Strategy routes expose:
     - Pit recommendations
     - Degradation explanations
     - Undercut analysis
   - Telemetry routes expose:
     - Live snapshot for a driver
     - Historical lap-by-lap data
     - XGBoost-based pit probability for a given lap

4. **Frontend dashboard**
   - Fetches telemetry + strategy data via the API
   - Renders a real-time dashboard with charts, cards, toggles, and comparisons
   - Lets you "drive" the race like a race engineer

---

## 📁 Project Structure

```
ai-race-engineer/
├── agents/                    # Multi-agent strategy system
├── backend/
│   ├── app.py                 # FastAPI app entrypoint
│   ├── requirements.txt
│   ├── routes/
│   │   ├── sessions.py        # Session load/list/delete
│   │   ├── strategy.py        # Strategy recommendations, LSTM status
│   │   ├── telemetry.py       # Live + historical telemetry
│   │   └── verstappen.py      # Verstappen simulator (aggressive vs baseline)
│   └── schemas/               # Request/response schemas
├── config/                    # App and path configuration
├── data/
│   ├── processed/             # Processed telemetry by session
│   ├── raw/                   # FastF1 cache files
│   └── scripts/               # Data fetching and preprocessing
├── frontend/
│   ├── app/                   # Next.js app router pages
│   ├── components/            # Dashboard UI components
│   ├── lib/                   # API client & race store
│   └── styles/                # Global styles
├── ml/
│   ├── datasets/              # Training datasets
│   ├── models/                # Model definitions
│   ├── model_registry/        # Experiment tracking
│   ├── saved_models/          # Production models
│   └── training/              # Training scripts and pipeline
├── tests/                     # Unit and integration tests
├── utils/                     # Logging and utilities
└── docs/                      # Architecture & phase documentation
```

---

## ⚙️ Local Development

### 1️⃣ Backend

```bash
# 1. Clone the repo
git clone https://github.com/KaustubhMukdam/ai-race-engineer.git
cd ai-race-engineer

# 2. Create virtualenv and install dependencies
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) Fetch and preprocess telemetry
python data/scripts/fetch_fastf1_data.py
python data/scripts/preprocess_telemetry.py

# 4. (Optional) Train models (LSTM + XGBoost)
python ml/training/train_multi_race_lstm.py
python ml/training/train_pit_window_classifier.py

# 5. Run backend
uvicorn backend.app:app --reload --port 8000
```

**Backend will be live at:** http://localhost:8000

### 2️⃣ Frontend

```bash
cd frontend

# 1. Install dependencies
npm install

# 2. Set environment variables
# Create .env.local file with:
# NEXT_PUBLIC_API_URL=http://localhost:8000

# 3. Run dev server
npm run dev
```

**Frontend will be live at:** http://localhost:3000

---

## 🌐 Production Deployment

### Backend: Deploy FastAPI to Render
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn backend.app:app -k uvicorn.workers.UvicornWorker`

### Frontend: Deploy Next.js to Vercel
- **Build command:** `npm run build`
- **Environment variable:** `NEXT_PUBLIC_API_URL=https://ai-race-engineer.onrender.com`

Both platforms are configured to auto-deploy on push to the main branch.

---

## 📹 Demo Video

A short walkthrough of the dashboard, ML models, and race strategy flow:

👉 **Video:** https://drive.google.com/file/d/14snUOBNsJj-IEERMaaaIROyNOLsm9DQ4/view

---

## 🔮 Future Improvements

- Live streaming of telemetry during actual races
- AI commentary mode ("race engineer radio")
- Fine-tuned LLM for natural language strategy conversations
- Support for multiple drivers and teams in a single session
- Real-time weather integration and dynamic strategy adjustments

---

## 🙌 Credits

- F1 telemetry via [FastF1](https://github.com/theOehrly/Fast-F1)
- Inspired by F1 race engineering and data-driven motorsport strategy

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---