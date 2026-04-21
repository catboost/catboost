# Sprint 25 — DEC-026 Cascade-Robust GAIN Research

**Branch**: TBD (cut from Sprint 24 tip `784f82a891`)
**Campaign**: Post-Verstappen research — R8 recovery investigation
**Gate config**: 50k/RMSE/d6/128b (unchanged from S19–S24)
**Authority**: `DECISIONS.md DEC-026`
**R8 entry position**: 1.01× (honest post-S24)

---

## §1 Background

Sprint 24 D0 resolved DEC-023 (features 1-3 atomic-float race in T2-accum) by rewriting T2-accum
to use T1-style SIMD-shuffle accumulation for all four features (commit `784f82a891`). All four
acceptance criteria passed. DEC-023 is closed.

The fix collapsed R8 from 1.90× to 1.01×. T2's structural histogram speedup (0.317× hist_ms
ratio, driving the 1.90× cumulative) derived from feature-0's bin-range scan over sorting-ordered
docs — the same mechanism that produced a different FP reduction topology from T1. Matching the
topology for correctness eliminates the speed. The 1.90× record is superseded.

Sprint 25 investigates whether a cascade-robust GAIN comparison mechanism can allow T2's
sort-based accumulation to produce deterministically consistent tree structures with T1, so that
T2's structural speedup can be recovered.

---

## §2 Problem Structure

### Why config #8 is the sole failure case

T2's feature-0 bin-range scan produces Value B (0.48231912); T1's SIMD fold produces Value A
(0.48231599). The per-bin histogram difference is 1-2 ULP. At 17 of 18 DEC-008 configs, this
difference does not reach a GAIN near-tie and the final loss values converge to within DEC-008
tolerance (RMSE ulp ≤ 4). Config #8 has a near-tie GAIN comparison at some iteration k where
the 1-2 ULP bin difference changes the winner — selecting a different split node. All subsequent
iterations diverge, cascading to 105 ULP at iters=50.

### The cascade-robust GAIN hypothesis

A deterministic tiebreak rule — applied when `|GAIN_A - GAIN_B| < ε` for a calibrated ε —
could prevent the near-tie flip from selecting a different node. The tiebreak uses a
lexicographic ordering (featureIdx, binIdx) as a secondary key, so both T1 and T2 (despite
different GAIN values) select the same winning split candidate when they are within ε of each
other.

If the tiebreak succeeds:

1. T2 (sort-based accumulation, producing Value B inputs) and T1 (SIMD accumulation, producing
   Value A inputs) produce the same tree structure at every near-tie iteration.
2. End-to-end loss values for T2 converge to within DEC-008 tolerance of T1 across all 18
   DEC-008 configs.
3. T2 Path 5 (T2-sort + int-atomic fixed-point for features 1-3 + tiebreak) is shippable at its
   pre-S24 hist_ms ratio of ~0.317×.

**This is a research hypothesis, not an established fact.**

---

## §3 Research Design: Falsification-First

Sprint 25 is a falsification-first research sprint. Each gate is a falsification checkpoint.
If any kill-switch fires, the research stops and DEC-026 is declared FALSIFIED.

### Gate G1: Epsilon calibration study (highest-risk step)

**Owner**: @research-scientist
**What to measure**:

1. Run T1 and T2 (Path 5) at config #8 across the full 50 iterations. At each iteration k,
   record the best GAIN value for T1 and for T2. Compute `|GAIN_T1 - GAIN_T2|` at each k.

2. Identify the iteration(s) where T1 and T2 select different splits (the near-tie flip
   iterations). Record `|GAIN_T1 - GAIN_T2|` at those iterations.

3. For each of the other 17 DEC-008 configs, record `min(|GAIN_T1 - GAIN_T2|)` across all
   iterations (the closest any legitimate GAIN gap comes to being overridden by a tiebreak).

4. Determine if a viable ε range exists: ε must be large enough to catch the near-tie flip(s)
   at config #8, but small enough not to override legitimate GAIN gaps at any of the other 17
   configs.

**Kill-switch**: If no ε threads this needle (near-tie gap at config #8 > minimum legitimate
gap at another config), cascade-robust GAIN is not achievable without false-positive tiebreaks
that alter model behavior at non-#8 configs. Declare DEC-026 FALSIFIED. Do not proceed to G2.

**Output**: `docs/sprint25/g1_epsilon_calibration.md` with GAIN gap distribution tables and
ε recommendation or FALSIFIED verdict.

### Gate G2: Tiebreak implementation in scoring kernel

**Owner**: @ml-engineer
**Blocked on**: G1 PASS
**What to implement**:

Modify the scoring kernel (`kScoreSplitsLookupSource`) to apply a lexicographic secondary
comparison when `|GAIN_A - GAIN_B| < ε`. Secondary key: (featureIdx, binIdx), ascending.

The tiebreak must be deterministic regardless of Metal thread scheduling — it uses only the
feature and bin indices, which are static per dispatch. No atomics.

**Kill-switch**: If the tiebreak fires at ≥2 of the 17 non-#8 configs at any of their 5 parity
runs, ε is miscalibrated. Halt, revisit G1.

### Gate G3: T2 Path 5 rebuild

**Owner**: @ml-engineer
**Blocked on**: G2 PASS
**What to implement**:

Rebuild T2 with the Path 5 design:
- T2-sort kernel: deterministic prefix-sum scatter (from S24 D0 Path 5 work).
- T2-accum: feature-0 bin-range scan over `sortedDocs` + int-atomic fixed-point for features 1-3.
- Scoring kernel: tiebreak active (from G2).

Measure `hist_ms(T2) / hist_ms(T1)` at gate config. Target: ratio ≤ 0.45× (Path 5 historically
measured ~0.317×).

**Kill-switch**: If ratio > 0.45×, T2 Path 5 misses the performance target and the speedup
recovery goal is not achievable. Halt.

### Gate G4: 18-config parity sweep + determinism

**Owner**: @qa-engineer
**Blocked on**: G3 PASS
**Protocol**: ≥5 runs per non-gate config; 100 runs at gate config; config #8 10/10 deterministic.
DEC-008 tolerance: RMSE ulp ≤ 4, Logloss ulp ≤ 4, MultiClass ulp ≤ 8.

**Kill-switch**: Any config fails ULP tolerance, or bimodality detected at config #8 across ≥5
runs. Halt.

### Gate G5: Model-quality validation

**Owner**: @qa-engineer
**Blocked on**: G4 PASS
**What to measure**: AUC/RMSE at each of the 18 DEC-008 configs for T2 Path 5 vs T1 baseline.
Threshold: ≤ 0.5% degradation on any metric at any config.

The tiebreak may select a different tree structure from both T1 (naturally) and T2 (without
tiebreak). A different tree is acceptable only if the quality impact is within this threshold.

**Kill-switch**: Quality drop > 0.5% at any DEC-008 config. Halt.

---

## §4 Kill-Switch Summary

| Trigger | Action |
|---------|--------|
| G1: No viable ε range | DEC-026 FALSIFIED. R8 stays at 1.01×. |
| G2: Tiebreak fires on ≥2 non-#8 configs | ε miscalibrated. Return to G1 or FALSIFY. |
| G3: ratio > 0.45× at gate config | T2 Path 5 misses perf target. FALSIFY DEC-026. |
| G4: Any config fails ULP tolerance or bimodal | Parity failure. FALSIFY DEC-026. |
| G5: Quality drop > 0.5% at any config | Quality failure. FALSIFY DEC-026. |

If all five gates pass, T2 Path 5 ships and R8 is re-measured at gate config.

---

## §5 R8 Target If Research Succeeds

T2 Path 5's pre-S24 measured hist_ms ratio was ~0.317× at gate config. If the Path 5 rebuild
(G3) achieves a similar ratio:

| Metric | T2 v5 (current) | T2 Path 5 target |
|--------|:---------------:|:----------------:|
| hist_ms (gate config) | ~20.75 ms (0.959× T1) | ~6.9 ms (~0.317× T1) |
| iter_total_ms warm mean | ~33–35 ms | ~17–18 ms |
| e2e speedup vs S16 baseline | ~1.01× | ~1.85–1.90× |

This is the ceiling estimate, conditional on all 5 gates passing. Actual ratio may differ from
0.317× if the tiebreak adds non-trivial scoring overhead.

If any kill-switch fires at any gate, R8 stays at 1.01× honest position.

---

## §6 Budget

Research sprint. 1-2 weeks with kill-switches at each gate.

- G1 (epsilon calibration) is the highest-risk step. If it fails — which is the most likely
  failure mode — the sprint completes in 1-2 days with a FALSIFIED verdict.
- If G1 passes, G2–G3 are 2-4 days engineering. G4–G5 are 1-2 days QA sweep.
- No guaranteed delivery. Success requires the near-tie GAIN gap at config #8 to be strictly
  smaller than all legitimate GAIN gaps at the other 17 configs, AND the selected tiebreak
  winner to produce a tree of acceptable quality. Both conditions are empirical questions.

---

## §7 D-Document Placeholders

| Doc | Status | Description |
|-----|--------|-------------|
| `g1_epsilon_calibration.md` | PENDING | GAIN gap distribution tables; ε range or FALSIFIED verdict |
| `g3_t2_path5_rebuild.md` | PENDING | Path 5 rebuild implementation + ratio measurement |
| `g4_parity_sweep.md` | PENDING | 18-config parity sweep results |
| `g5_quality_validation.md` | PENDING | Model-quality AUC/RMSE comparison table |
