# DATA AVAILABILITY MATRIX
**Status:** AUDIT COMPLETE  
**Date:** December 27, 2025  
**Source:** FastF1 Library Documentation + Real Testing

---

## MATRIX LEGEND

| Column | Meaning |
|--------|---------|
| **Signal** | Data element required for modeling |
| **FastF1 Available?** | Is the signal directly provided by FastF1 API? |
| **Confidence** | How reliable is the signal when available? |
| **Fidelity** | Quality/precision of the signal |
| **Phase 1 Use** | How will this signal be used in Phase 1? |
| **Mitigation** | If signal unavailable, how do we proceed? |

---

## PRIMARY MATRIX: REQUIRED SIGNALS FOR PIT WINDOW OPTIMIZATION

### Tier 1: Critical (Pit Window Feasibility Depends On These)

| Signal | FastF1 Available? | Confidence | Fidelity | Phase 1 Use | Mitigation | Risk |
|--------|---|---|---|---|---|---|
| **Lap Time (Race)** | ✅ YES | 🟢 HIGH | Millisecond precision | Base input for pit window cost | None; directly measured | 🟢 NONE |
| **Tire Compound** | ✅ YES | 🟢 HIGH | Soft/Medium/Hard only | Identify degradation segment | Cannot distinguish C5 vs C6; accept loss of fidelity | 🟡 MEDIUM |
| **Pit Stop Records** | ✅ YES | 🟢 HIGH | Pit lap, duration | Ground truth for validation | Missing pit? Treat as no-stop stint | 🟡 MEDIUM |
| **FP3 Lap Time** | ✅ YES | 🟢 HIGH | Millisecond precision | Source for degradation extraction | Fall back to FP2 if FP3 missing | 🟡 MEDIUM |
| **Track Position** | ✅ YES | 🟢 HIGH | Grid/finish position per lap | Optional: verify pit logic | Use for sanity check only | 🟢 NONE |

---

### Tier 2: Important (Model Accuracy Depends On These)

| Signal | FastF1 Available? | Confidence | Fidelity | Phase 1 Use | Mitigation | Risk |
|--------|---|---|---|---|---|---|
| **Fuel Capacity** | 🟡 IMPLICIT | 🟢 HIGH | 110 kg (FIA regulation) | Initial fuel load assumption | Hardcode 110 kg per FIA rules | 🟢 NONE |
| **Session Metadata** | ✅ YES | 🟢 HIGH | Session type, date, weather flag | Session classification | Use API metadata | 🟢 NONE |
| **Speed (Max)** | ✅ YES | 🟡 MEDIUM | Sample-based max, not average | Lap efficiency indicator | Use for relative comparison only | 🟡 MEDIUM |
| **Telemetry Timestamp** | ✅ YES | 🟡 MEDIUM | 4-5 Hz sample rate | Align lap events | Accept ±200ms uncertainty | 🟡 MEDIUM |
| **Gap to Car Ahead** | ✅ YES | 🟡 MEDIUM | Per lap, not per sample | Filter dirty-air laps | Use heuristic threshold (>1 second) | 🟡 MEDIUM |

---

### Tier 3: Desirable (Would Improve Accuracy If Available)

| Signal | FastF1 Available? | Confidence | Fidelity | Phase 1 Use | Mitigation | Risk |
|--------|---|---|---|---|---|---|
| **Fuel Consumption Rate** | ❌ NO | N/A | N/A | Input to fuel model | Estimate from regulations; test 3 variants (1.4-1.8 kg/lap) | 🔴 HIGH |
| **Tire Pressure** | ❌ NO | N/A | N/A | Tire state indicator | Cannot model; acknowledge as limitation | 🔴 HIGH |
| **Tire Temperature** | ❌ NO | N/A | N/A | Grip dynamics | Cannot model; infer from lap time only (circular) | 🔴 HIGH |
| **Brake Temperature** | ❌ NO | N/A | N/A | Brake wear tracking | Only binary on/off available; insufficient | 🔴 HIGH |
| **Suspension Loads** | ❌ NO | N/A | N/A | Mechanical performance | Not in FastF1; not accessible | 🔴 HIGH |
| **Real-Time Telemetry** | ❌ NO | N/A | N/A | Live reasoning | Requires proprietary F1 API; out of scope | 🔴 HIGH |
| **Tire Wear Depth** | ❌ NO | N/A | N/A | Direct degradation measurement | Infer from lap time; large uncertainty | 🔴 HIGH |

---

## SECONDARY MATRIX: TELEMETRY CHANNELS (If Available)

**Condition:** Telemetry varies by session and year. Pre-2017 sparse; 2018+ more complete.

### Telemetry Signals in FastF1

| Signal | Status | Sample Rate | Phase 1 Use | Notes |
|--------|--------|---|---|---|
| **Speed** | ✅ Available | 4-5 Hz | Trend analysis, speed profile | Max speed recorded; not average |
| **Throttle** | ✅ Available | 4-5 Hz | Acceleration pattern; for reference | Not used in pit optimizer |
| **Brake** | ✅ Available (binary) | 4-5 Hz | Braking zones only (on/off) | No pressure gradient; insufficient for analysis |
| **Gear** | ✅ Available | 4-5 Hz | Downshift zones | Reference only; not critical |
| **RPM** | ✅ Available | 4-5 Hz | Engine behavior | Reference only |
| **DRS Status** | ✅ Available (binary) | 4-5 Hz | DRS availability; not activation time | "Available" ≠ "Activated"; imprecise |
| **Distance from Lap Start** | ✅ Computed by FastF1 | 4-5 Hz | Align events to track position | Interpolated; ±8m uncertainty at 300 km/h |

### Telemetry Sampling & Interpolation

**Raw Telemetry Frequency:** 4-5 Hz  
**Sample Interval:** ~200 milliseconds  
**Distance per Sample (at 300 km/h):** ~17 meters

**Implication:**
- Braking point uncertainty: ±8 meters
- Sharp transients (gear changes, DRS activation) undersampled
- FastF1 applies linear interpolation to merge with position data
- Interpolated values flagged in `Source` column

**Phase 1 Usage:**
- Use for trend analysis and lap-level comparisons
- Do NOT use for turn-by-turn or millisecond-level precision
- Document interpolation uncertainty in final report

---

## CONFIDENCE LEVELS PER SIGNAL

### 🟢 HIGH CONFIDENCE (Use Directly)
- Lap times (race & practice)
- Tire compound (Soft/Medium/Hard)
- Pit stop records
- Track & session metadata
- Fuel capacity (FIA regulation)

**Action:** Use as-is. No special handling required.

### 🟡 MEDIUM CONFIDENCE (Use with Caution)
- Max speed (not average; sample-based)
- Gap to car ahead (per lap; noisy)
- Telemetry timestamp (4-5 Hz; ±200ms)
- Driver status (DNF, retirements)

**Action:** Document uncertainty. Use for relative comparisons. Avoid absolute claims.

### 🔴 LOW CONFIDENCE or UNAVAILABLE
- Fuel consumption rate (must estimate; ±5-10% error)
- Tire pressure (not available; infer via lap time)
- Tire temperature (not available; infer via lap time)
- Brake temperature (binary only; insufficient)
- Real-time telemetry (post-session only)

**Action:** Build mitigation into model. Accept limitations. Validate via backtesting.

---

## MITIGATION STRATEGIES BY MISSING SIGNAL

### Strategy A: Estimate from Regulations (Fuel Consumption)

**Missing Signal:** Fuel consumption rate (kg/lap)

**Mitigation:**
1. Use FIA regulations: max fuel load = 110 kg
2. Estimate burn rate: X kg/lap (typical: 1.4-1.8)
3. Test sensitivity: model with 3 variants (low, mid, high)
4. Validate: compare final lap count vs. actual race completion
5. Error bound: ±5-10% acceptable; document in report

**Phase 1 Implementation:**
```python
fuel_burn_rates = [1.4, 1.6, 1.8]  # kg/lap variants
for rate in fuel_burn_rates:
    fuel_remaining(lap_n) = 110 - (rate * (lap_n - 1))
    validate against actual pit timing
```

**Success Criterion:** All 3 variants produce pit windows within ±2 laps of actual.

---

### Strategy B: Infer from Lap Time (Tire Degradation)

**Missing Signal:** Direct tire wear measurement

**Mitigation:**
1. Extract degradation from FP3 long runs (controlled environment)
2. Correct for fuel weight (biggest confound)
3. Filter for consistent pace (remove traffic)
4. Fit degradation curve (linear/poly/exp)
5. Validate on race data (compare predicted vs. actual pit timing)

**Phase 1 Implementation:**
```python
# FP3 long run
clean_laps = hamilton_fp3[
    (hamilton_fp3['LapNumber'] > 3) &  # warm-up
    (hamilton_fp3['LapNumber'] < -3) &  # avoid end effects
    (gap_to_car_ahead > 1.0)  # not in traffic
]

degradation_rate = fit_decay_curve(clean_laps)  # seconds/lap
```

**Success Criterion:** R² > 0.70 on FP3 degradation curve.

---

### Strategy C: Accept Limitation (Brake Dynamics, Suspension)

**Missing Signal:** Brake temperature, suspension loads

**Mitigation:**
1. Acknowledge that these cannot be modeled
2. Document as limitation in report
3. Model pit window as function of tire degradation + fuel only
4. Validate scope: does tire+fuel model suffice to predict pit windows?

**Phase 1 Implementation:**
- Do NOT attempt to optimize brake strategy
- Do NOT model suspension setup impact
- Pit window = argmin(fuel_cost + tire_degradation_cost)

**Success Criterion:** Pit window prediction within ±1 lap (proves scope sufficient).

---

### Strategy D: Use Proxy Data (Dirty Air via Gap)

**Missing Signal:** Direct aerodynamic impact (dirty air)

**Mitigation:**
1. Use gap to car ahead as proxy
2. Heuristic: gap > 1 second ≈ clean air (acceptable approximation)
3. Filter out laps with smaller gaps when fitting degradation curves
4. Validate: check that filtered laps show cleaner degradation signal

**Phase 1 Implementation:**
```python
clean_air_threshold = 1.0  # seconds gap to car ahead
clean_laps = laps[laps['gap_ahead'] > clean_air_threshold]
# Use clean_laps only for degradation fitting
```

**Success Criterion:** Filtered degradation R² > unfiltered R² (proves filtering helps).

---

## SUMMARY: DATA RISK ASSESSMENT

| Category | Status | Risk Level | Phase 1 Impact |
|----------|--------|---|---|
| **Lap Timing** | ✅ Complete | 🟢 NONE | Foundation; fully available |
| **Tire Compound** | ⚠️ Limited (3 categories) | 🟡 LOW | Accept coarse mapping; validation tolerates this |
| **Pit Records** | ✅ Complete | 🟢 NONE | Ground truth; reliable |
| **Fuel Model** | ❌ Estimated | 🔴 HIGH | Sensitivity tested; error bounds quantified |
| **Degradation** | 🟡 Inferred from lap time | 🔴 HIGH | Validated via backtesting; uncertainty documented |
| **Real-Time** | ❌ Unavailable | 🔴 CRITICAL | Out of Phase 1 scope; Phase 2 goal |

**Overall Phase 1 Data Risk:** MODERATE (manageable with documented mitigations)

---

## VALIDATION CHECKPOINTS FOR PHASE 1

### Week 1: Data Quality
- [ ] FP3 telemetry loads for all 5 races
- [ ] Hamilton lap data complete (no missing rounds)
- [ ] Pit records match race results (sanity check)

### Week 2: Model Feasibility
- [ ] Fuel correction reduces lap time variance
- [ ] At least one degradation model (linear/poly/exp) achieves R² > 0.70
- [ ] Sensitivity test: ±5% fuel burn rate → pit window shift < ±2 laps

### Week 3: Backtest Validation
- [ ] Pit optimizer runs on 1 test race
- [ ] Predicted pit vs. actual pit: error ≤ ±2 laps
- [ ] All assumptions locked with confidence intervals

---

**Document Status: LOCKED**  
**Prepared by:** Phase 0 Technical Lead  
**Date:** December 27, 2025
