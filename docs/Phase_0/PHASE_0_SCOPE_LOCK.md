# PHASE 0: SCOPE LOCK
**Status:** FROZEN  
**Date:** December 27, 2025  
**Prepared by:** Technical Lead (Phase 0)

---

## 1. SCOPE DEFINITION

### 1.1 Project Goal (Revised)

**Original Vision:**
> "AI agent that behaves like a real F1 race engineer. Feed it live telemetry + race context, and it suggests strategies, explains tradeoffs, predicts tire degradation."

**Phase 1 Goal (Revised & Locked):**
> Build a **historical strategy simulator** that predicts optimal pit windows within **±1 lap accuracy** by analyzing past race data with tire degradation models extracted from free practice sessions. All outputs must include explicit uncertainty bounds and documented limitations.

**Rationale for Revision:**
- Live telemetry unavailable in FastF1 (post-session only)
- Real-time reasoning architecture not feasible with public data
- Historical simulation is falsifiable, reproducible, and a necessary foundation
- Phase 1 success enables Phase 2 (live agent with infrastructure)

---

## 2. FROZEN DECISIONS FOR PHASE 1

### 2.1 Dataset Selection

**Season:** 2023 (Modern regulations, clean data, recent enough for relevance)

**Rationale:**
- F1 2022-2024 regulations stable (different from 2014-2021 V6 era)
- 2023 data complete in FastF1 (no missing sessions)
- Tires: Pirelli C1-C6 compounds consistent across 24 races
- Car performance delta manageable (within-season progression vs. regulation change shock)

**Not Selected:**
- 2024 Season: Too recent; analysis would conflict with live racing
- 2022 Season: Regulation change year; confounds (new cars, new tire compounds)
- 2021 and Earlier: Old regulations; tire behavior fundamentally different (C0-C5, different Pirelli generation)
- Pre-2018: Telemetry sparse, incomplete in FastF1

---

### 2.2 Driver Selection

**Primary Driver (Phase 1 Pilot):** Lewis Hamilton

**Rationale:**
- Consistent driving style across season
- Mercedes data typically complete in FastF1
- Mid-grid + occasional podiums in 2023 (realistic strategy variation)
- No dominance bias (unlike RB/VER in 2022-2024)
- Sufficient sample: 22 races, 1500+ laps

**Secondary Drivers (Validation):** George Russell, Carlos Sainz, Yuki Tsunoda

**Rationale:**
- Validate model transfers across drivers
- Mix of driving styles (aggressive, conservative, medium)
- Different teams (Mercedes, Ferrari, AlphaTauri) = different strategies
- Used only AFTER pilot model is validated

---

### 2.3 Race Selection for Backtesting

**Diversity Requirement:** 5-10 races spanning track types and strategies

**Races Committed (Locked Selection):**

| Race | Reason | Track Type | Primary Strategy |
|------|--------|-----------|------------------|
| Abu Dhabi (Round 22) | Season finale; two-stop locked strategy | Street | 2-stop stable |
| Silverstone (Round 9) | High-speed, tire degradation pronounced | Road course | 2-stop or 1-stop |
| Singapore (Round 16) | Night race, slow, long (62 laps) | Street | 2-stop extended |
| Monza (Round 15) | Fastest lap, pit window tight, weather risk | Road course | 2-stop or undercut |
| Bahrain (Round 1) | Season opener, weather neutral | Road course | Baseline data |

**These 5 races provide:**
- Different pit window sizes (Monza tight, Singapore extended)
- Different weather patterns (potential rain: Singapore)
- Different tire compounds (all C-compounds used across season)
- Independent strategy validation points

---

### 2.4 Data Sessions Locked

**Session Types Required:**

| Session | Purpose | Status |
|---------|---------|--------|
| **FP3** (Free Practice 3) | Degradation curve extraction | PRIMARY (all 5 races) |
| **Race** | Actual pit windows + comparison | PRIMARY (all 5 races) |
| **FP2** | Backup degradation if FP3 incomplete | FALLBACK |

**Why These?**
- FP3: Latest practice before race (tires freshest, setup most race-relevant)
- Race: Ground truth for pit window validation
- FP2: If FP3 data missing, fallback to FP2 (less fresh but acceptable)
- Not using Qualifying: Qualifying laps are single-lap, non-representative

---

## 3. PHASE 1 WILL ATTEMPT

### 3.1 Deliverables (In Scope)

**Working Simulator:**
- [x] Load race lap times (FastF1 API)
- [x] Load FP3 telemetry for same track
- [x] Extract tire degradation curves (linear/polynomial/exponential fit)
- [x] Implement fuel consumption correction (constant burn rate model)
- [x] Optimize pit windows (deterministic; best lap time = minimum time strategy)
- [x] Output: "Pit window is lap N ± M" with confidence bounds

**Validation Evidence:**
- [x] Backtest pit timing vs. actual pit stop (measure error in laps)
- [x] Sensitivity analysis (how much do assumptions affect output?)
- [x] Cross-driver validation (train on HAM, test on RUS/SAI/TSU)
- [x] Uncertainty quantification (R² values, error bars, confidence intervals)

**Documentation:**
- [x] Explicit assumptions locked (fuel burn, degradation form, driver filtering)
- [x] Uncertainty bounds on all inputs
- [x] Failure mode analysis (where does model break? why?)
- [x] Limitations clearly stated
- [x] Code README with reproducibility steps

---

### 3.2 Model Scope (In Scope)

**What Will Be Modeled:**

1. **Tire Degradation (Primary Feature)**
   - Form: Linear or polynomial (exponential if R² > 0.75)
   - Data source: FP3 long runs, fuel-corrected lap times
   - Fit quality: R² > 0.70 target
   - Applies to: Medium (C3/C4) and Hard (C1/C2) compounds

2. **Fuel Consumption Impact**
   - Assumption: Constant burn rate (1.4-1.8 kg/lap range tested)
   - Method: Reverse-engineered from regulations (capacity 110 kg)
   - Lap time impact: ~0.05 seconds per kg (empirical estimate)
   - Validation: Compare vs. literature fuel consumption rates

3. **Pit Window Optimization (Deterministic)**
   - Objective: Minimize race time via pit stop timing
   - Constraint: One pit stop only (constraint for Phase 1)
   - Variables: Pit lap (1-lap granularity)
   - Output: "Pit at lap N for optimal final time"

4. **Traffic & Dirty Air (Simple Filter)**
   - Heuristic: Exclude laps where gap to car ahead < 1 second
   - Rationale: Remove dirty air confound from degradation curve fitting
   - Method: Filter practice laps before fitting

---

## 4. PHASE 1 WILL NOT ATTEMPT

### 4.1 Explicitly Out of Scope

**Live / Real-Time Components:**
- ❌ Live telemetry ingestion (FastF1 is post-session only)
- ❌ Live pit decision reasoning (no real-time state available)
- ❌ In-race strategy adjustments (weather, safety cars, traffic dynamics)
- ❌ Real-time agent framework or decision trees

**Advanced Modeling:**
- ❌ Multi-stop strategy optimization (2+ stops; complexity explosion)
- ❌ Undercut/overcut reasoning (requires modeling competitor pit logic)
- ❌ Weather impact modeling (no per-lap weather data)
- ❌ Brake balance or suspension setup optimization (no sensor data)
- ❌ Driver coaching / onboard video analysis

**AI/LLM Components:**
- ❌ Large language models for reasoning
- ❌ Multi-agent debate or Constitutional AI
- ❌ Neural networks for prediction (use classical models only)
- ❌ Reinforcement learning or game-theoretic strategies

**Data-Intensive Modeling:**
- ❌ Tire pressure impact (not measured in FastF1)
- ❌ Tire temperature dynamics (not measured in FastF1)
- ❌ Brake temperature management (only binary on/off available)
- ❌ Suspension load analysis (not measured)
- ❌ Aerodynamic drag impact (not measured)

**Cross-Domain Features:**
- ❌ Generalization across years (2022, 2024 have different regulations)
- ❌ Generalization across teams (different fuel consumption profiles)
- ❌ Compound-specific degradation (C5 vs. C6 indistinguishable)
- ❌ Driver-specific aggression modeling (confounded with car performance)

**Explainability Features:**
- ❌ Natural language explanations (will provide numerical outputs only)
- ❌ Visualization beyond plots
- ❌ Interactive what-if UI (command-line only)

---

### 4.2 Rationale for Scope Exclusions

**Why No Live Reasoning?**
- FastF1 is post-session only; no live API available
- F1 proprietary live timing not public
- Real-time infrastructure requires separate engineering (Phase 2 goal)

**Why No Multi-Stop?**
- Combinatorial explosion (pit lap × fuel load × compound choices)
- Requires modeling competitor strategies (out of scope)
- Single-stop is 80%+ of race strategies (sufficient baseline)

**Why No AI/LLM?**
- Cannot justify AI before classical model is validated
- Risk: AI reasoning masks poor underlying model
- Phase 2 goal: Add reasoning layer after foundation proven

**Why No Proprietary Sensors?**
- Tire pressure, temperature, suspension loads not in public data
- FastF1 fundamentally limited by broadcast telemetry
- Either accept public data or shift to proprietary data source

**Why No Cross-Year Generalization?**
- 2022 regulations changed tires, fuel load, power units
- 2024 regulations changed again
- Same model cannot work across regulation boundaries
- 2023 is fixed scope; cross-year is separate effort

---

## 5. PHASE 1 ENTRY CRITERIA (Go/No-Go)

**Phase 0 must validate these before Phase 1 proceeds:**

### 5.1 Data Quality Signals

- [x] FP3 telemetry available for all 5 races (checked)
- [x] Hamilton lap time data complete for all 5 races (checked)
- [x] Pit stop records accessible and unambiguous (checked)
- [x] No major data gaps in fuel capacity or tire compound info (checked)

### 5.2 Model Feasibility Signals

- [x] Degradation signal visible in FP3 (qualitative inspection Week 1)
- [x] Fuel correction reduces lap time variance (Week 2 validation)
- [x] At least one degradation model (linear/poly/exp) achieves R² > 0.70 (Week 2)
- [x] Pit optimizer runs without error on 1 test race (Week 3)

### 5.3 Assumption Validation Signals

- [x] Fuel burn rate sensible (1.4-1.8 kg/lap in tests)
- [x] Lap time improvement rate defensible (~0.05s per kg)
- [x] Pit window precision within ±2 laps on first backtest (Week 3)
- [x] All assumptions documented with sources (Week 3)

### 5.4 Documentation Completeness

- [x] This scope lock document (frozen)
- [x] Data availability matrix (frozen)
- [x] Assumptions & risks document (frozen)
- [x] Phase 0 summary with go/no-go (frozen)

---

## 6. DEFINITION OF DONE FOR PHASE 0

Phase 0 is complete when ALL of the following are true:

1. **Scope is frozen in writing** (this document)
2. **Data signals are validated** (Week 1-3 checks passed)
3. **Go/No-Go decision is documented** (formal recommendation)
4. **Phase 1 entry criteria are locked** (measurable, defensible)
5. **All documents committed to git**

---

## 7. PHASE 1 ENTRY DECISION

**FORMAL RECOMMENDATION: Proceed to Phase 1**

**Justification:** Phase 0 validation completed; all entry criteria met.

**Conditions:** Engineering lead must review go/no-go summary before Phase 1 sprint starts.

**Next Step:** Phase 1 Planning Sprint (1 week) → Phase 1 Execution (4-8 weeks)

---

**Document Status: LOCKED FOR PHASE 1**  
**Signed Off By:** Technical Lead  
**Date:** December 27, 2025
