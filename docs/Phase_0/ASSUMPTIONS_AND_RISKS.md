# ASSUMPTIONS AND RISKS
**Status:** LOCKED AFTER PHASE 0 VALIDATION  
**Date:** December 27, 2025  
**Prepared by:** Phase 0 Technical Lead

---

## 1. EXPLICIT ASSUMPTIONS (FROZEN FOR PHASE 1)

### 1.1 Fuel Consumption Model

**Assumption:**
> Fuel consumption is constant at rate **R kg/lap**, with initial fuel load **F₀ = 110 kg**.

**Mathematical Form:**
```
Fuel(lap_n) = F₀ - R × (n - 1)
Lap_Time_Impact = -α × Fuel(lap_n)  where α ≈ 0.05 seconds/kg
```

**Values Tested (Sensitivity Range):**
- R = 1.4 kg/lap (conservative; ~15.7% margin to FIA max)
- R = 1.6 kg/lap (nominal; literature baseline)
- R = 1.8 kg/lap (aggressive; accounts for higher fuel flow in race)

**Source of Confidence:**
- FIA Regulation Article 10.2: Fuel tank capacity = 110 kg maximum
- Empirical observation: ~77 laps average race length → avg burn ~1.4-1.6 kg/lap
- Literature: Cowell et al. (2010) estimate 0.05s lap time per kg fuel weight

**Known Biases:**
- Burn rate varies with driving style (aggressive = higher consumption)
- Burn rate varies by track (high-speed Monza ≠ low-speed Singapore)
- Mid-race fuel consumption may be higher than practice (race pace vs. controlled practice)
- Early-race fuel burn may be suppressed due to initial setup/warm-up

**Validation Method (Phase 1):**
1. Test all 3 variants (1.4, 1.6, 1.8) on same race
2. Compare pit window predictions across variants
3. If pit shifts >2 laps between variants → high uncertainty; document
4. If pit shift <1 lap → model robust; accept

**Fallback (If Fuel Model Breaks):**
- Accept ±5% error in pit window accuracy
- Document fuel model uncertainty in final report
- Use mid-range (1.6 kg/lap) as baseline for Phase 1

**Risk Level:** 🟡 **MEDIUM** (Fuel directly affects pit window; no direct measurement available; must estimate)

---

### 1.2 Tire Degradation Model Form

**Assumption:**
> Tire degradation follows **linear or polynomial decay** of lap time with lap count.

**Mathematical Forms (To Be Tested):**

**Linear:**
```
Lap_Time(n) = A + B × n
where B < 0 (time increases as tires degrade)
```

**Polynomial (Quadratic):**
```
Lap_Time(n) = A + B×n + C×n²
where B < 0 (primary term)
```

**Exponential (If R² > 0.75):**
```
Lap_Time(n) = A + D × exp(E × n)
```

**Rationale:**
- Linear: Simplest; assumes constant degradation rate
- Quadratic: Accounts for accelerating wear (degeneracy effect)
- Exponential: Only if data strongly suggests; rare in F1 (usually linear/quad)

**Application Scope:**
- Applies only to **Medium (C3/C4) and Hard (C1/C2) compounds**
- Soft compounds excluded from Phase 1 (short stints; confounded by in-lap/out-lap dynamics)
- Fit only on FP3 long runs (minimum 8 consecutive laps on same compound, same fuel level)
- Filter: exclude laps 1-3 (warm-up) and final 2 laps (end effects, traffic)

**Validation Metrics:**
- R² > 0.70: Acceptable fit
- R² 0.60-0.70: Marginal; document high uncertainty
- R² < 0.60: Reject model; retry with filtered data or different form

**Known Biases:**
- Track evolution: Early laps dirty, later laps rubbered-in (opposes degradation trend)
- Fuel weight: Lap times drop as fuel burns (confounds degradation signal)
- Driver strategy: In practice, drivers may ease off near end (not degradation)
- Traffic: Dirty air from other cars masks true degradation rate

**Controlled Conditions (FP3):**
- Long run with minimal traffic (filter: gap to car ahead > 1 second)
- Fresh tires start (no cold-tire outliers)
- Stable fuel load (don't mix in-lap/out-lap)
- Same driver (exclude multi-driver sessions)

**Fallback (If Degradation Model Fails):**
- Use median degradation rate from literature: 0.08 seconds/lap for Medium compound
- Document as "assumed constant degradation; could not extract from data"
- Accept higher uncertainty in pit window (±2 laps instead of ±1)

**Risk Level:** 🔴 **HIGH** (Signal is noisy; many confounds; R² < 0.70 possible even with clean data)

---

### 1.3 Pit Stop Loss (Time to Stop)

**Assumption:**
> Pit stop duration is constant: **τ_pit ≈ 22-24 seconds** (including in-lap, stop duration, out-lap).

**Source:**
- F1 average pit stop duration 2023: 2.3 seconds (mechanics)
- In-lap loss: ~1.5 seconds (reduced pace approaching pit)
- Out-lap loss: ~4-5 seconds (reduced pace leaving pit, tire warm-up)
- Total: ~22-24 seconds of elapsed race time per pit stop

**Validation Method:**
- Extract actual pit stop duration from race data
- Compare predicted vs. actual
- If variance > ±3 seconds: note in Phase 1 report

**Known Biases:**
- Safety car periods reduce pit cost (relative; not in this model)
- Weather (rain) extends pit duration (not modeled)
- Tire compound change adds complexity (not modeled; assume same compound in Phase 1)
- Driver traffic near pit entry/exit affects in/out-lap times

**Risk Level:** 🟡 **LOW-MEDIUM** (Pit loss is less variable than tire degradation; ±3 seconds acceptable)

---

### 1.4 Dirty Air & Traffic Threshold

**Assumption:**
> Laps where gap to car ahead < **1.0 second** are excluded from degradation fitting (assumed to be in dirty air).

**Rationale:**
- Dirty air reduces downforce ~5-15% (depends on following distance)
- Lap time impact of dirty air: ~0.3-0.5 seconds per lap
- Gap of 1.0 second = ~300 meters at F1 speeds = outside main dirty air cone

**Implementation:**
```python
clean_air_laps = laps[laps['gap_to_car_ahead'] > 1.0]
# Use clean_air_laps for degradation model fitting
```

**Validation:**
- Compare R² (degradation fit) with vs. without dirty air filtering
- If filtering improves R² by >0.10: confirm filter is necessary
- If filtering changes R² by <0.05: filter optional; use without for more data

**Known Biases:**
- Gap is measured per lap (not per sample); noisy
- Early laps in FP3 may have higher gaps (not necessarily clean air; could be pacing)
- Cannot distinguish "behind slower car" from "intentionally pacing"

**Fallback (If Threshold Ambiguous):**
- Test both: gap > 1.0 and gap > 1.5 seconds
- Use whichever produces higher R²
- Document choice in final report

**Risk Level:** 🟡 **MEDIUM** (Threshold somewhat arbitrary; sensitivity analysis will quantify impact)

---

### 1.5 Practice-to-Race Degradation Transfer

**Assumption:**
> Tire degradation curve extracted from **FP3 is applicable to race** (same tire compound, same track, reasonable track temperature).

**Rationale:**
- Same track = same surface/temperature characteristics
- Same tire compound = same wear rate expected
- FP3 is last practice before race; tires, setup, and conditions closest to race

**Known Biases:**
- FP3 may have different fuel loads (unknown; assume lighter than race)
- FP3 drivers may not be pushing at limit (FP3 is setup work; race is all-out)
- Race-day track temperature may differ from FP3 temperature (5-10°C variation)
- Tire condition unknown (fresh FP3; unknown race usage)

**Validation Method:**
1. Extract degradation from FP3
2. Apply to race; predict lap times for stint 1 and stint 2
3. Compare predicted vs. actual race lap times
4. If RMSE < 0.5 seconds: model acceptable
5. If RMSE > 0.5 seconds: document bias; consider separate race-based extraction

**Fallback (If Transfer Fails):**
- Extract degradation separately from race data (noisier but unbiased)
- Accept ±0.10 seconds/lap uncertainty (higher than practice-based)
- Use race extraction as Phase 1 baseline

**Risk Level:** 🔴 **HIGH** (Practice ≠ race; unknowns in setup, pace, tire condition)

---

## 2. SOURCES OF BIAS

### 2.1 Fuel Bias (Largest Confound)

**Description:**
Lap time improves as fuel is consumed, opposite to degradation (which increases lap time).

**Magnitude:**
- Fresh fuel (lap 1): ~90 kg on board
- Late race (lap 60): ~20 kg on board
- Time delta: ~3.5 seconds over 60 laps
- Rate: ~0.05-0.06 seconds/lap improvement due to fuel only

**Interaction with Degradation:**
```
Observed_Lap_Time = Baseline + Fuel_Effect − Degradation_Effect + Noise
                           ↑ negative (improves)    ↑ positive (slows)

If Fuel_Effect > Degradation_Effect:
  → Observed lap times improve despite tire degradation
  → Cannot extract degradation signal from raw lap times

Mitigation: Correct all lap times by fuel weight before fitting degradation
```

**Validation:**
- Fit degradation with fuel correction: should show positive trend (slower times)
- Fit degradation without fuel correction: may show flat or negative trend (wrong)
- If with-correction R² is substantially higher: fuel correction is necessary

**Phase 1 Implementation:**
```python
# Fuel-corrected lap time
corrected_time = raw_time - (0.05 * fuel_remaining)
# Fit degradation on corrected_time
```

---

### 2.2 Track Evolution Bias

**Description:**
Track surface improves during a session (rubber laid down from other sessions, temperature rise), reducing lap times independent of tire degradation.

**Magnitude:**
- Lap 1-3 (cold, dirty track): ~0.3-0.5 seconds slower than lap 10+
- Lap 10-20 (hot, rubbered): Baseline
- Lap 21+ (potential cooling): May regress slightly

**Interaction with Degradation:**
```
If early laps are slow due to cold track:
  → Fitted degradation curve suggests improvement (opposed to real degradation)
  → Underestimate actual tire wear rate

Mitigation: Exclude first 3 laps from degradation fitting
```

**Validation:**
- Compare degradation fits with/without first-lap filtering
- If excluding laps 1-3 increases slope (more degradation): track evolution was masking signal
- If excluding changes fit by <0.01 s/lap: track evolution minimal for that session

---

### 2.3 Driver Consistency Bias (Practice vs. Race)

**Description:**
FP3 is not race pace. Drivers may conserve fuel, test setups, or coast to save tires. Race is all-out.

**Magnitude:**
- FP3 "pilot lap" strategies: fuel-light, lower pace
- Race strategies: some pushing hard, some managing
- Expected pace delta: 0.2-0.5 seconds per lap

**Interaction with Degradation:**
```
If FP3 pace is lighter than race pace:
  → Tire degradation is underestimated
  → Model will predict earlier pit windows than optimal

Mitigation: Accept this as limitation; quantify uncertainty in Phase 1
```

**Validation:**
- Compare FP3-based degradation rate with race-derived degradation rate
- If race shows 20%+ higher degradation: note discrepancy; use race data
- Document: "FP3 degradation may underestimate race tire wear"

---

### 2.4 Driving Style Bias (Driver-Specific Degradation)

**Description:**
Different drivers have different degradation profiles.
- Aggressive drivers (sharp braking, high lateral G): higher wear
- Smooth drivers (progressive inputs, low G): lower wear

**Magnitude:**
- Aggressive vs. smooth: ~0.02-0.05 seconds/lap difference in degradation rate
- Compounded over 20-lap stint: ~0.4-1.0 second cumulative difference

**Interaction:**
```
Phase 1 Model: Extract Hamilton's degradation from his FP3 data
Validation: Apply Hamilton's model to Russell's race
  → If Russell's actual pace is <0.3s off prediction: model robust
  → If off by >0.5s: Hamilton's driving style influences model

Mitigation: Phase 1 uses Hamilton only; Phase 2 generalizes to other drivers
```

**Phase 1 Approach:**
- Explicitly model driver-specific degradation
- In validation, test cross-driver transfer
- Document: "Model extracted for Hamilton; unknown generalization to other drivers"

**Risk Level:** 🟡 **MEDIUM** (Known issue; Phase 1 scoped to single driver; Phase 2 problem)

---

### 2.5 Tire Compound Mapping Bias

**Description:**
FastF1 maps 6 actual compounds (C1-C6) to 3 categories (Soft/Medium/Hard).

**Mapping:**
```
Hard:    C1, C2
Medium:  C3, C4
Soft:    C5, C6
```

**Lost Information:**
- C5 vs. C6: Same category but different degradation rates
  - C6 is softer; degrades faster (~0.05 s/lap more)
  - C5 is harder; degrades slower
- Example: Monaco (C5) vs. Australian GP (C6) both map to "Soft"

**Interaction:**
```
Phase 1 Model: Extract "Soft" compound degradation across multiple races
  → C5 races and C6 races mixed in sample
  → Average degradation rate; neither exactly right
  → Model degradation ≈ 0.08 s/lap (true range: 0.06-0.10)

Mitigation: Acknowledge limitation; would require manual compound identification
```

**Phase 1 Approach:**
- Use only Medium and Hard compounds (less variation within category)
- Exclude Soft compound from Phase 1 (too much within-category variation)
- Document: "Cannot distinguish C5 from C6; Phase 2 would require manual mapping"

**Risk Level:** 🟡 **MEDIUM** (Known limitation; workaround is to avoid Soft)

---

## 3. FAILURE MODES AND DETECTION

### Failure Mode 1: Degradation Signal Too Noisy (R² < 0.60)

**Symptoms:**
- FP3 degradation fitting yields R² < 0.60 across all models (linear/poly/exp)
- Even after fuel correction and traffic filtering, scatter is high

**Root Causes:**
- Track evolution dominates signal (cold track effect stronger than expected)
- Tire warm-up period extends beyond lap 3 (laps 4-6 still show trend)
- Driver is not maintaining consistent pace (easing off, testing setups)
- Weather variation during session (wind, track temp changes)

**Detection (Phase 1 Week 1-2):**
- Plot FP3 lap times vs. lap number for each race
- Visually inspect: Is there clear downward trend?
- If trend is not visually obvious: R² likely < 0.60

**Mitigation (In Order of Preference):**
1. Filter more aggressively: exclude laps 1-5 instead of 1-3
2. Filter by pace consistency: exclude outlier laps (>0.5s off median)
3. Use polynomial fit (may capture track evolution + degradation together)
4. Fall back to literature value (0.08 s/lap for Medium compound)
5. Extract degradation from race instead of FP3 (noisier but unbiased)

**Go/No-Go Condition:**
- If Mode 1 resolved by Week 2: Proceed to Phase 1
- If Mode 1 unresolved by Week 2: Escalate; consider revising scope (single-track focus only)

---

### Failure Mode 2: Fuel Model Systematically Off (>10% Error)

**Symptoms:**
- Pit window prediction off by >2 laps consistently
- Sensitivity test: fuel burn variants (1.4, 1.6, 1.8) all predict pit too early or too late
- Cross-check: estimated fuel at pit stop doesn't match actual fuel burn

**Root Causes:**
- Actual fuel consumption rate very different from assumed rate
- Driver aggressive/conservative fuel strategy unknown
- Fuel consumption varies significantly by track (Monza high-speed ≠ Monaco tight)

**Detection (Phase 1 Week 2-3):**
- After implementing fuel model, backtest on 1 race
- Extract actual fuel consumed: Fuel_Initial − Fuel_at_Pit ÷ Laps
- Compare with assumed burn rate: If diff > 10%, model is off

**Mitigation:**
1. Calibrate fuel burn rate to actual race data (reverse-engineer from pit timing)
2. Use median burn rate across all backtested races
3. Increase sensitivity bounds: test 1.2-2.0 kg/lap instead of 1.4-1.8
4. Separate track-specific fuel models (but adds complexity)

**Go/No-Go Condition:**
- If fuel error <10% after calibration: Acceptable; proceed
- If fuel error >10% even after calibration: Document high uncertainty; reduce accuracy target from ±1 lap to ±2 laps

---

### Failure Mode 3: Practice-to-Race Degradation Transfer Fails

**Symptoms:**
- FP3 degradation model looks good (R² > 0.75)
- But when applied to actual race pit timing, predictions are off by >2 laps
- Actual race lap times don't match FP3 model predictions

**Root Causes:**
- FP3 pace is lighter than race pace (drivers conserving)
- Race-day track temperature significantly different from FP3 (temperature affects tire degradation)
- Tire condition different (fresh FP3; degraded in race)
- Downforce package or fuel load different between sessions

**Detection (Phase 1 Week 3):**
- Fit degradation from FP3
- Apply to race; predict lap times for first 10 laps of first stint
- Compare predicted vs. actual: If RMSE > 0.5s, transfer failed

**Mitigation:**
1. Use slower FP3 pace as baseline; add offset correction (+0.1-0.2 s/lap)
2. Extract degradation separately from race data (accept higher noise)
3. Use blend: 60% FP3 degradation + 40% race degradation
4. Identify specific race conditions that broke model (temperature, weather)

**Go/No-Go Condition:**
- If Mode 3 resolved with offset correction: Proceed
- If Mode 3 unresolved: Consider Phase 1b (iterative refinement) or Phase 1 reduced scope

---

### Failure Mode 4: Pit Window Optimizer Breaks Down at Integration

**Symptoms:**
- Individual components work (fuel model, degradation, pit cost)
- But when integrated into optimizer, pit window is >2 laps off for all test races
- Systematic bias (always too early or always too late)

**Root Causes:**
- Bug in pit window search algorithm
- Interaction between fuel model and degradation model (double-counting effect)
- Pit stop loss assumption (22-24 sec) is significantly wrong
- Race pit stop rule not accounted for (min pit window, safety car effects)

**Detection (Phase 1 Week 3):**
- Manual sanity check: Given fuel = 50kg at pit, degradation = 0.08 s/lap, when should pit happen?
- Compare manual calculation with optimizer output
- If they differ by >1 lap: bug in optimizer

**Mitigation:**
1. Debug pit optimizer line-by-line; validate against manual calculation
2. Use simpler greedy algorithm (test all pit laps 1-70; pick best) instead of optimization
3. Verify pit stop loss assumption: extract from actual pit stops in race data
4. Add safety bounds: pit window must be within ±3 laps of actual pit (sanity check)

**Go/No-Go Condition:**
- If Mode 4 resolved by Week 3 end: Proceed to Phase 1
- If Mode 4 unresolved: Defer optimizer; pivot to simpler heuristic (e.g., "pit when pace delta >0.5s/lap")

---

## 4. RISK SUMMARY TABLE

| Assumption | Confidence | Risk | Impact if Wrong | Detection Time | Mitigation Available? |
|---|---|---|---|---|---|
| Fuel burn rate constant | Medium | HIGH | ±2 lap pit window error | Week 2 | ✅ Yes (calibrate) |
| Degradation linear/poly | Medium | HIGH | R²<0.60; no signal | Week 1-2 | ✅ Yes (filter, fallback) |
| FP3→Race transfer | Low | HIGH | Model off by >1 lap | Week 3 | ✅ Yes (offset, re-fit) |
| Pit stop loss 22-24 sec | Medium | LOW | ±0.5 lap error | Week 3 | ✅ Yes (extract from data) |
| Dirty air threshold 1.0s | Medium | MEDIUM | R² improves slightly | Week 2 | ✅ Yes (sensitivity test) |
| Driver consistency | Low | MEDIUM | Only applies to HAM | Phase 2 | ⚠️ Partial (document) |
| Tire compound mapping | High | MEDIUM | Exclude Soft; use Med/Hard | Week 1 | ✅ Yes (filter) |

---
# 5. Assumption → Validation Metric → Go/No-Go Threshold

This section defines the objective validation criteria used during Phase 1 to verify whether Phase 0 assumptions hold. These metrics are evaluated at Week 1–3 validation gates and determine continuation, mitigation, or escalation.

## 5.1 Validation Mapping Table

| Assumption | Validation Metric | Evaluation Method | Acceptable Threshold | Action if Failed |
|------------|-------------------|-------------------|---------------------|------------------|
| Fuel burn rate constant | Pit window sensitivity across burn variants | Compare pit lap predictions using 1.4 / 1.6 / 1.8 kg/lap | Shift ≤ ±2 laps | Document high uncertainty; reduce accuracy target to ±2 laps |
| Fuel correction improves signal | Variance reduction after correction | Std-dev of lap times before vs after correction | ≥10% variance reduction | Revisit α coefficient or fuel model |
| Degradation model form valid | R² of fitted degradation curve | Linear / polynomial fit on FP3 long runs | R² ≥ 0.70 | Apply stronger filtering or fallback to literature value |
| FP3 → Race transfer valid | RMSE of predicted vs actual race lap times | First 10 race laps after applying FP3 model | RMSE ≤ 0.5 s | Apply offset correction or switch to race-based extraction |
| Pit stop loss estimate valid | Pit loss prediction error | Predicted vs actual pit stop delta | Error ≤ ±3 s | Extract pit loss directly from race data |
| Dirty air filter effective | Change in degradation fit quality | R² with vs without gap filtering | ΔR² ≥ +0.05 | Remove filter; accept higher noise |
| Overall pit window accuracy | Pit lap prediction error | Predicted vs actual pit lap | ≤ ±2 laps (Week 3), target ±1 lap | Escalate if >±2 laps without diagnosed cause |

## 5.2 Phase-1 Escalation Rules

* **Single failure** → Apply documented mitigation and proceed
* **Two related failures** (e.g., fuel + degradation) → Reduce scope or accuracy target
* **Core failure** (degradation R² < 0.60 across all attempts) → Escalate for scope revision
* **Unexplained pit error > ±2 laps** → Immediate engineering review required

## 5.3 Governance Note

This section serves as the execution contract between Phase 0 and Phase 1. No new assumptions may be added in Phase 1 without updating this table and receiving engineering lead approval.

---
**Document Status: LOCKED FOR PHASE 1**  
**Prepared by:** Phase 0 Technical Lead  
**Date:** December 27, 2025
