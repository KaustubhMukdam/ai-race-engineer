# PHASE 0 SUMMARY
## Feasibility Analysis & Go/No-Go Decision

**Status:** PHASE 0 COMPLETE  
**Date:** December 27, 2025  
**Prepared by:** Phase 0 Technical Lead (FIA Perspective)

---

## EXECUTIVE SUMMARY

**Project:** AI Race Engineer Simulator (Formula 1)  
**Original Vision:** Live real-time AI reasoning agent for pit strategy  
**Revised Phase 1 Goal:** Historical strategy simulator (pit window prediction ±1 lap)

**Verdict:** ✅ **RECOMMENDATION: GO (pending approval)** (with revised scope and explicit constraints documented)

---

## SECTION 1: WHAT CHANGED FROM ORIGINAL VISION

### Original Vision (Infeasible)
```
Live AI Agent that:
  → Ingests real-time telemetry during race
  → Reasons about tire degradation in real-time
  → Suggests pit windows as race unfolds
  → Explains strategic tradeoffs

Status: ❌ NOT FEASIBLE with FastF1
Reason: FastF1 is post-session only (no live data)
```

### Revised Phase 1 Scope (Feasible)
```
Historical Strategy Simulator that:
  → Analyzes past race data (lap times, pit stops)
  → Extracts tire degradation from practice (FP3)
  → Optimizes pit windows via deterministic model
  → Validates against actual pit decisions
  → Quantifies uncertainty explicitly

Status: ✅ FEASIBLE and VALIDATED
Why: All data is known, stable, and falsifiable
```

### Why This Revision Matters
1. **Validates foundation before adding complexity** — Prove the model works on known races before attempting live reasoning
2. **Reproducible and falsifiable** — Compare predictions to ground truth; easy to debug failures
3. **Honest about constraints** — No live telemetry, no real-time inference; document limitations clearly
4. **Natural path to Phase 2** — Once Phase 1 works, Phase 2 adds live telemetry + real-time reasoning

---

## SECTION 2: KEY FINDINGS FROM PHASE 0

### Finding 1: FastF1 Data Is Usable But Limited

**What's Available ✅**
- Lap timing (millisecond precision) — EXCELLENT
- Tire compound (Soft/Medium/Hard) — GOOD
- Pit stop records — EXCELLENT
- Free practice telemetry — GOOD (4-5 Hz sampling)

**What's Missing ❌**
- Tire wear state (direct measurement) — NO DATA
- Tire pressure/temperature — NO DATA
- Brake temperatures — ONLY ON/OFF BINARY
- Real-time telemetry — POST-SESSION ONLY
- Fuel consumption — MUST ESTIMATE

**Verdict:** Sufficient for pit window prediction. Not sufficient for real-time physics modeling.

---

### Finding 2: Tire Degradation Is Extractable But Noisy

**From Free Practice (FP3):**
- Clean signal possible: R² > 0.70 achievable
- Best conditions: long runs (10+ laps), minimal traffic, same fuel level
- Degradation range: 0.04–0.12 seconds/lap (varies by compound, track, driver)

**From Race Data:**
- Highly corrupted by fuel bias, track evolution, traffic, driver pacing
- R² typically < 0.60 even with fuel correction
- **Conclusion:** Avoid race data for degradation fitting; use FP3 only

**Validation Plan (Phase 1):**
- Extract degradation from FP3
- Apply to race pit timing prediction
- If pit window within ±1 lap of actual: model validated
- If off by >1 lap: debug fuel model or degradation curve

---

### Finding 3: Fuel Consumption Must Be Estimated

**Issue:**
FastF1 doesn't provide fuel consumption. Must reverse-engineer from regulations.

**Mitigation:**
- Use FIA max (110 kg) as baseline
- Test 3 variants: 1.4, 1.6, 1.8 kg/lap
- Pit window sensitivity: If shift <1 lap between variants → model robust
- If shift >2 laps: High uncertainty; document and use mid-range

**Expected Accuracy:**
- ±5-10% error on fuel model
- ~0.05-0.10 seconds/lap impact on pit window

---

### Finding 4: Real-Time Reasoning Is Phase 2, Not Phase 1

**Why Not Phase 1?**
- FastF1 is post-session (historical only)
- F1 live timing API is proprietary (not public)
- No infrastructure for live decision-making
- Must build custom telemetry client (large effort)

**Phase 1 Approach:** Historical simulation (known data, offline)  
**Phase 2 Approach:** Live reasoning (with infrastructure)  
**Phase 3 Approach:** Multi-agent + Constitutional AI (reasoning layer)

---

## SECTION 3: LOCKED DECISIONS FOR PHASE 1

### Scope Lock
| Decision | Value | Rationale |
|----------|-------|-----------|
| **Season** | 2023 | Modern regs; complete data; recent enough |
| **Primary Driver** | Lewis Hamilton | Consistent style; complete data; realistic strategy |
| **Secondary Drivers** | Russell, Sainz, Tsunoda | Cross-driver validation |
| **Races** | Abu Dhabi, Silverstone, Singapore, Monza, Bahrain | Diverse track types & strategies |
| **Data Sessions** | FP3 (primary), FP2 (fallback), Race | Degradation extraction + validation |
| **Tire Compounds** | Medium, Hard only | Soft too variable; exclude |
| **Pit Stops** | Single-stop only | Multi-stop is complexity for Phase 2 |

### Model Scope
| What to Build | Status | Why |
|---|---|---|
| Tire degradation extraction | ✅ IN | Core feature; feasible with FP3 data |
| Fuel consumption correction | ✅ IN | Essential confound correction |
| Pit window optimization | ✅ IN | Deterministic; no AI needed |
| Backtest validation | ✅ IN | Falsifiable via race comparison |
| **Live telemetry** | ❌ OUT | FastF1 doesn't support |
| **Real-time reasoning** | ❌ OUT | Requires infrastructure |
| **Multi-stop strategies** | ❌ OUT | Combinatorial; Phase 2 goal |
| **AI/LLM reasoning** | ❌ OUT | Premature; validate model first |
| **Weather impact** | ❌ OUT | No per-lap weather data |

---

## SECTION 4: CRITICAL ASSUMPTIONS (FROZEN)

### Assumption 1: Fuel Consumption
**Value:** 1.4–1.8 kg/lap (test all 3 variants)  
**Source:** FIA regulations (110 kg max) + literature (Cowell et al.)  
**Uncertainty:** ±5-10%  
**Impact on Pit Window:** ±1-2 laps max  
**Risk Level:** 🟡 MEDIUM (No direct data; must estimate)

### Assumption 2: Degradation Form
**Value:** Linear or polynomial (test both)  
**Fit Target:** R² > 0.70  
**Source:** FP3 long-run data  
**Fallback:** If R² < 0.60, use literature value (0.08 s/lap)  
**Risk Level:** 🔴 HIGH (Signal is noisy; many confounds)

### Assumption 3: FP3→Race Transfer
**Value:** FP3 degradation applies to race (with offset correction if needed)  
**Validation:** RMSE < 0.5s on first 10 race laps  
**Fallback:** Extract degradation separately from race (noisier)  
**Risk Level:** 🔴 HIGH (Practice ≠ race; unknowns)

### Assumption 4: Pit Stop Loss
**Value:** 22-24 seconds total (in-lap + stop + out-lap)  
**Source:** Average pit stop duration + pace loss  
**Validation:** Extract from actual race pit stops  
**Risk Level:** 🟡 LOW-MEDIUM (More stable than tire degradation)

### Assumption 5: Dirty Air Threshold
**Value:** Gap > 1.0 second ≈ clean air  
**Implementation:** Filter FP3 degradation fitting by gap  
**Validation:** Does filtering improve R²? If yes, keep filter  
**Risk Level:** 🟡 MEDIUM (Threshold somewhat arbitrary)

---

## SECTION 5: VALIDATION CHECKPOINTS (GO/NO-GO)

### Week 1: Data Quality
- [ ] FP3 telemetry loads for all 5 locked races
- [ ] Hamilton lap data complete (no missing rounds)
- [ ] Pit records match race results
- [ ] Degradation signal visible qualitatively (plot lap times vs. lap number)

**Go Criteria:** All 4 pass → Proceed to Week 2  
**No-Go Criteria:** Any fail → Debug and retry

---

### Week 2: Model Feasibility
- [ ] Fuel correction reduces lap time variance (corrected times show cleaner trend)
- [ ] At least one degradation model achieves R² > 0.70 (linear OR polynomial)
- [ ] Fuel burn variant test: pit window shift < ±2 laps across 1.4–1.8 kg/lap range
- [ ] Dirty air filtering improves degradation R² (optional but recommended)

**Go Criteria:** At least 3 of 4 pass → Proceed to Week 3  
**No-Go Criteria:** Fuel model off by >10%, or degradation R² < 0.60 unrecoverable → Escalate

---

### Week 3: Integration & Backtest
- [ ] Pit optimizer runs on 1 test race without error
- [ ] Predicted pit lap vs. actual pit lap: error ≤ ±2 laps
- [ ] Uncertainty quantified (R² values, error bounds documented)
- [ ] All assumptions locked with sources and fallbacks

**Go Criteria:** All 4 pass → **RECOMMENDATION: GO (pending approval)**  
**No-Go Criteria:** Pit window off by >2 laps with no diagnosed cause → Decision meeting required

---

## SECTION 6: GO/NO-GO RECOMMENDATION

### FORMAL RECOMMENDATION: ✅ **GO TO PHASE 1**

**Justification:**
1. ✅ Data is available and usable (FastF1 provides core signals)
2. ✅ Scope revised to be achievable (historical simulation, not live)
3. ✅ Assumptions are explicit and testable (frozen for Phase 1)
4. ✅ Risk mitigations are documented (fallbacks for each assumption)
5. ✅ Validation approach is falsifiable (backtest against known race outcomes)
6. ✅ Entry criteria are measurable (Week 1-3 checkpoints)

**Conditions:**
- Engineering lead reviews this Phase 0 summary
- Phase 0 artifacts (this document + 3 locked documents) are merged to main branch
- Phase 1 begins with Week 1 checklist from PHASE_0_SCOPE_LOCK.md

**Success Probability:**
- 🟢 **HIGH (70-80%)** that Phase 1 produces a working simulator
- 🟡 **MEDIUM (50-60%)** that ±1 lap accuracy is achieved
- 🔴 **LOW (20-30%)** that live reasoning Phase 2 is immediately feasible (still depends on infrastructure)

**Risk Mitigation for Phase 1:**
- Week 1-3 validation gates catch major issues early
- Multiple fallback strategies documented for each failure mode
- Scope is conservative (single driver, single season, medium/hard compounds only)

---

## SECTION 7: WHAT PHASE 1 WILL DELIVER

### Deliverable 1: Working Simulator
```
Input:  Historical race data (2023) + FP3 telemetry
Output: Optimal pit window (lap N ± M) with uncertainty bounds

Example:
  Race: Abu Dhabi 2023, Driver: Hamilton
  Degradation: 0.085 ± 0.012 s/lap (from FP3)
  Fuel model: 1.6 kg/lap, ±0.08 s/lap impact
  Prediction: Pit at lap 15 ± 1 (confidence: 85%)
  Actual pit: Lap 14
  Error: 1 lap (within target ±1 lap)
```

### Deliverable 2: Validation Evidence
- Backtest results on 5-10 races
- Pit window error distribution (histogram)
- Sensitivity analysis (fuel, degradation, pit cost)
- R² values and confidence intervals
- Cross-driver validation (HAM model tested on RUS, SAI, TSU)

### Deliverable 3: Documentation
- Technical report (10-15 pages)
  - Data pipeline (FastF1 API usage)
  - Model equations (fuel, degradation, optimizer)
  - Validation methodology
  - Failure modes identified & addressed
  - Uncertainty quantification
  - Limitations clearly stated

- Code README with reproducibility steps
- Figure: pit window prediction error distribution
- Table: degradation parameters per compound/track/driver

---

## SECTION 8: WHAT PHASE 1 WILL NOT DELIVER

❌ **Out of Scope:**
- Live real-time reasoning (Phase 2 goal)
- Live telemetry infrastructure (Phase 2 goal)
- Multi-stop strategy optimization (Phase 2 goal)
- AI/LLM reasoning layer (Phase 3 goal)
- Multi-agent debate (Phase 3 goal)
- Weather impact modeling (requires per-lap data not available)
- Generalization to 2024 or 2022 seasons (regulation boundaries)

**These are intentional scoping decisions, not bugs.**

---

## SECTION 9: RESOURCE ESTIMATE FOR PHASE 1

| Activity | Effort | Timeline |
|----------|--------|----------|
| Data download & exploration | 1 week | Week 1 |
| Fuel model implementation | 1 week | Week 2 |
| Degradation fitting | 1 week | Week 2 |
| Pit optimizer build | 1.5 weeks | Week 3-4 |
| Backtest on 5-10 races | 1.5 weeks | Week 4-5 |
| Documentation & reporting | 1 week | Week 6 |
| **Total** | **6-7 weeks** | **6-7 weeks** |

**Buffer:** +1 week for debugging/iteration  
**Realistic Timeline:** 8 weeks (accounting for unknowns)

---

## SECTION 10: DECISION AUTHORITY & SIGN-OFF

**Phase 0 Completed By:** Technical Lead (Phase 0)  
**Reviewed By:** Engineering Lead (required before Phase 1 start)  
**Approved By:** Project Manager (funding/timeline)  

**Approval Checklist:**
- [ ] Phase 0 summary reviewed and understood
- [ ] Revised scope (historical simulation) is acceptable
- [ ] Locked decisions (2023 season, Hamilton, 5 races) are acceptable
- [ ] 8-week Phase 1 timeline is feasible
- [ ] Risk level (moderate) is acceptable
- [ ] Phase 1 entry criteria make sense
- [ ] Proceed to Phase 1 planning sprint

---

## APPENDIX: PHASE 1 ENTRY CRITERIA (DETAILED)

**These must all pass before Phase 1 is considered "ready to go":**

### Data Quality (Week 1)
```
✅ FP3 telemetry available for Abu Dhabi, Silverstone, Singapore, Monza, Bahrain
✅ Hamilton lap times complete for all 5 races (no missing rounds)
✅ Pit stop records unambiguous (pit lap, duration, tires)
✅ Degradation signal visible in at least 3 of 5 races (plot shows trend)
```

### Model Feasibility (Week 2)
```
✅ Fuel-corrected lap times show cleaner degradation trend than raw times
✅ Linear OR polynomial degradation fit achieves R² > 0.70 on at least 1 race
✅ Pit window prediction stable across fuel burn rates (shift < ±2 laps)
✅ Pit optimizer runs without error on test race
```

### Integration & Validation (Week 3)
```
✅ Predicted pit lap within ±2 laps of actual pit lap (first backtest)
✅ All assumptions documented with sources and fallbacks
✅ Uncertainty bounds quantified (R² values, error intervals)
✅ Degradation extraction code tested and reproducible
```

**If all criteria pass:** Phase 1 is approved.  
**If any criterion fails:** Debug and retry; escalate if unresolved.

---

## FINAL STATEMENT

Phase 0 has validated that the project is **feasible in scope and achievable in timeline**. The revised Phase 1 goal (historical strategy simulator) is honest, falsifiable, and defensible. 

**All critical assumptions are frozen and documented.** Risk levels are understood. Fallback strategies exist for major failure modes. Validation checkpoints are in place.

**Phase 1 should proceed as planned.**

---

**Document Status: LOCKED — READY FOR PHASE 1**  
**Prepared by:** Phase 0 Technical Lead  
**Date:** December 27, 2025  
**Next Step:** Engineering lead approval → Phase 1 planning sprint (1 week) → Phase 1 execution (6-8 weeks)
