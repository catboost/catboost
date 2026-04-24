# Sprint 30 Close — S30-COSINE-KAHAN

**Sprint:** 30  |  **Status:** PARTIALLY CLOSED — precision fix class exhausted, structural divergence surfaced
**Branch:** `mlx/sprint-30-cosine-kahan`  |  **Base:** master `4d855d47db`  |  **Tip:** `24a0e829b8`
**Date:** 2026-04-24  |  **DEC authority:** DEC-035 PARTIALLY CLOSED; DEC-036 OPEN

---

## Executive Summary

Sprint 30 executed the full phased T1→T4 Kahan fix plan prescribed by DEC-035, driven by
DEC-034's moderate-confidence verdict that ST+Cosine and LG+Cosine shared a float32
joint-denominator compounding mechanism. The T1 and T2 phases passed cleanly: the cosDen
accumulator was correctly fingered (residual 4.07e-3), and K4 fp64 widening reduced the
gain residual 12.5× at iter=1. T3 failed all three primary envelope gates at 50 iterations
(G3a 53.30% ST, G3b 27–31% LG-Mid, G3c 44–45% LG-Stress), triggering K2 and blocking
guard removal. A full verification battery followed — D1 (CPU audit), D2 and D2-redux (stack
instrumentation), D3 (LG mechanism discriminator), D4 (joint-denominator amplification probe),
V1 (N-scaling), V2 (D2 methodology audit), V5 (DW at scale), Fix 2 (fp64 gain widening),
and V6 (N=500 cheap falsifier). The battery's final result was unambiguous: V6 measured 50.72%
drift at N=500, nearly identical to 53.30% at N=50000, yielding a scaling exponent b ≈ 0.0
across a 100× N range. Flat N-scaling falsifies the entire precision-accumulation hypothesis
class. K4 and Fix 2 ship as logically correct precision fixes — they remove floors that will
matter once the structural mechanism is resolved — but they did not close the guards. DEC-036
opens a structural divergence investigation for S31, whose entry point is an iter=1
split-selection comparison between CPU CatBoost and MLX.

---

## Phase Outcome Table

| Phase | Task | Gate | Result | Takeaway |
|-------|------|------|--------|----------|
| T1 | #90 INSTRUMENT | G1 mechanism fingered | **PASS** — cosDen residual 4.067e-3 (seed 43, depth 5); K1 NOT fired | cosDen and cosNum are correct Kahan targets; gSum attenuation confirmed non-mechanism |
| T2 | #91 KAHAN | G2 ≥10× residual reduction | **PASS** (12.5×); K4 fired → fp64 widening | Neumaier float32 achieved only 3.1–3.4× (per-term floor dominates); K4 pre-authorized |
| T3 | #92 MEASURE | G3a < 2.0% ST; G3b/G3c [0.98, 1.02] | **FAIL** — G3a 53.30%, G3b 1.274–1.314, G3c 1.440–1.452; K2 FIRED at G3c | Iteration-compound cascade unchanged; gain residual reduction at iter=1 insufficient |
| D1 | #100 CPU AUDIT | CPU precision baseline | CPU fp64 throughout L1–L5 (`__m128d`, static_assert); MLX fp32 at L1, L3, L4 post-K4 | Reveals three remaining narrow points after K4 |
| D2 | #101 FULL-STACK INSTRUMENT | L3/L4 flip rate | Initially 0/18 flips; V2 later invalidated the methodology | Original D2 measured cast-ULP floor (not pre-K4 fp32 path); conclusion biased |
| D2-redux | #106 METHODOLOGY FIX | Honest fp32 shadow | L3/L4 RULED OUT — 5.03e-5 residual, 0/18 flips (genuine) | D2 conclusion survives with corrected methodology; no flips even on true fp32 path |
| D3 | #102 LG OUTCOME A/B | Discriminate LG mechanism | Outcome B confirmed — flip rate 0.11 → 0.81 across max_leaves {8,16,31,64}; 0% BFS-sequence match at every max_leaves | LG cannot be closed by any Cosine precision fix; priority-queue ordering is the driver |
| D4 | #107 JOINT-DENOM 64× | V5 amplification hypothesis | FALSIFIED at accumulator level — measured amplification 2.42×, not 64× (per-term floor dominates) | K4 closed the joint-denominator mechanism; 53% residual is L1 + structural |
| V1 | #103 N-SCALING | L0 precision-class predictor | FLAT — exponent b = 0.0017 across N ∈ {1k, 5k, 10k, 25k, 50k} | Eliminates fp32 histogram accumulation error as dominant mechanism |
| V2 | #104 D2 METHODOLOGY AUDIT | Verify D2 measurement is real | D2 L3/L4 measurement is biased — `gain_f32` derived from same fp64 as `gain_f64` | Audits must verify their own measurement; D2-redux required |
| V5 | #105 DW @ 50k | Isolate ST-specific mechanism | MIXED — DW 6.33%, ST 53.30%; 8.4× gap; L0 real for DW but cannot account for ST gap alone | ST joint-denominator structure (across 64 partitions) vs DW per-leaf (1 partition) explains part of gap |
| Fix 2 | #108 FP64 GAIN | L3/L4 widening (fp64 gain/argmax) | SHIPPED; prediction FAILED — ST 53.30% → 53.30% bit-identical | L3/L4 not the binding constraint; L1 fp32 histogram inputs are 10× larger than the gain boundary |
| V6 | #109 N=500 CONFIRMER | Cheap L1 falsification | **L1 FALSIFIED** — drift 50.72% at N=500 vs 53.30% at N=50k; b ≈ 0.0 across 100× N range; reduction 1.1× vs 100× predicted | Entire precision fix class exhausted; divergence is structural and N-independent |

---

## Chain of Evidence

### Step 1: DEC-034 outcome A confirmed — enter T1

Sprint 30 opened on moderate-confidence DEC-034 (2026-04-23): LG+Cosine and ST+Cosine share
the float32 joint-Cosine denominator compounding mechanism. T1 confirmed the target cleanly.
cosDen residuals grew monotonically with tree depth (4.07e-3 at depth 5), the link to gain
error was direct (gain residual 4.75e-5), and approxUpdate residual was exactly 0. gSum
absolute residuals (~6e-3) were large but Newton-step attenuation suppressed their effect on
leaf values to <4e-8 — non-mechanism. K1 did not fire. T2 proceeded.

### Step 2: K4 fires — Neumaier insufficient, fp64 fallback applied

T2 applied Neumaier float32 compensated summation and inspected the ARM64 assembly to confirm
the compensation pattern survived the compiler. It did, but only achieved 3.1–3.4× gain
residual reduction. The dominant error was per-term float32 computation floor (~1e-3 even at
depth=0 with a single partition, no accumulation). Neumaier corrects summation rounding; it
cannot address per-term rounding. K4 was pre-authorized in DEC-035 and fired: all four call
sites (FindBestSplit one-hot/ordinal, FindBestSplitPerPartition one-hot/ordinal) converted to
double accumulators, with the final gain scalar still cast to float. G2 passed at 12.5× gain
reduction.

### Step 3: T3 — all primary gates fail, K2 fires

T3 measured the full gate matrix at 50 iterations. G3a came in at 53.30% ST drift (threshold
<2.0%). The iter-1 drift had improved marginally (0.74% vs 0.77% pre-K4), but iteration-compound
amplification was unchanged: each mis-split propagates to wrong gradients in the next tree, and
50 trees of compound misdirection reach ~53%. G3b (LG-Mid at max_leaves=31) showed 27–31%
drift; G3c (LG-Stress at max_leaves=64) showed 44–45% drift, triggering K2. G4 parity was
clean (28/28). G5 showed +7.1% performance overhead from K4 (threshold <5%), triggering K3.

### Step 4: Diagnostic battery D1–D3

Three diagnostic tasks ran in parallel to understand why T3 failed:

D1 (CPU audit) found that CPU CatBoost's Cosine path is fp64 end-to-end (L1–L5) with a
`static_assert` on the SIMD type (`__m128d`). MLX, after K4, is still fp32 at L1 (histogram
inputs from the Metal kernel), L3 (gain cast back to float after fp64 accumulation), and L4
(argmax uses float bestGain). This crystallised the three remaining narrow points.

D2 (full-stack instrumentation) ran a 3-seed × 6-depth flip count. It reported 0/18 argmax
flips and 3.81e-6 gain residual, suggesting L3/L4 were not dominant. However, V2's methodology
audit later revealed D2 was measuring the wrong quantity: both the "fp32 gain" and "fp64 gain"
columns were derived from the same fp64 accumulators — the residual was literally `|float(x) - x|`
(the cast ULP), not the fp32 accumulation path divergence. D2-redux corrected this using a true
pre-K4 fp32 shadow path and confirmed 0/18 flips even with genuine gain residuals up to 5.03e-5
at depth 5. L3 and L4 are genuinely ruled out. The V2 episode is a reminder that measurement
methodology must be audited before conclusions are drawn.

D3 (LG mechanism discriminator) measured the LG priority-queue flip rate across max_leaves ∈
{8, 16, 31, 64}. The flip rate grew monotonically from 0.11 to 0.81 — all five rows of the
outcome A/B discrimination table landed in column B. The S29 spike's bit-identical BFS at
depth=3 / max_leaves=8 was a degenerate cell: when max_leaves ≥ 2^depth, the queue makes no
contested choices. D3 confirmed that LG+Cosine drift is driven by priority-queue ordering, not
fp32 accumulator precision. No precision fix within the current plan can address this.

### Step 5: V1 and V5 reshape the picture — L0 hypothesis emerges

V1 measured ST drift vs N across {1k, 5k, 10k, 25k, 50k}. The drift was flat: b = 0.0017.
All five N-values were within 0.89% of the 53.11% grand mean. This was the first signal that
the precision-class hypothesis was structurally wrong: fp32 accumulation errors are proportional
to N; N-independent drift implies a different mechanism.

V5 measured DW+Cosine at N=50k: 6.33% drift (vs 53.30% for ST at the same N). This 8.4× gap
is explained partly by ST's joint-denominator structure (accumulating cosDen across all 2^d =
64 partitions before taking sqrt, vs DW's per-leaf accumulation), which causes gain magnitudes
~4× larger in ST, amplifying the fp32 cast ULP. But neither V1's flat scaling nor the 8.4× gap
was consistent with a pure precision mechanism.

### Step 6: D4 falsifies the 64× amplification hypothesis

V5's verdict suggested the ST joint-denominator aggregation might explain the 8.4× gap via
64× fp32 rounding amplification. D4 directly measured the depth-scaling of the pre-K4 cosDen
residuals from T1 data. The amplification from depth=0 (1 partition) to depth=5 (32 partitions)
was 2.42×, not 64×. K4 had already closed the fp32 accumulator mechanism; the remaining 53%
drift was at L1 (float32 histogram inputs) and L3/L4 (gain cast and argmax), both of which
D4 identified but neither of which had been directly measured as causing split flips.

### Step 7: Fix 2 shipped and failed — prediction failed cleanly

D4 recommended Fix 2 as the low-risk correct next step: widen totalGain, bestGain,
TBestSplitProperties::Gain, perturbedGain, and TLeafCandidate::Gain to double. 15 sites
changed across FindBestSplit and FindBestSplitPerPartition. The prediction was that L3/L4
fp32 cast noise (~3.81e-6) was the binding argmax constraint. The fix shipped, the G3a re-run
showed 53.30% → 53.30% bit-identical. Fix 2 failed cleanly: the gain boundary at issue was
~0.006 within the top feature's bin cluster (from D4 §5), but the per-bin histogram input
error from L1 was ~4.7e-5 — already 10× larger than the argmax margin. L1 is the binding
constraint, not L3/L4.

### Step 8: V6 falsifies the entire precision class

V6 ran G3a at N=500. If L1 fp32 histogram accumulation (error ∝ N × eps_f32 / bins) were
dominant, 100× smaller N should reduce drift ~100×, from 53% to ~0.5%. The measured drift
was 50.72%. Combined with V1's five-point N-scaling curve, the exponent b ≈ 0.0 across the
full N=500 to N=50000 range. The precision class is exhausted. Four mechanisms were measured
in sequence — cosDen accumulation (K4), L3/L4 gain cast (Fix 2), joint-denominator 64×
amplification (D4, falsified), L1 histogram N-scaling (V1/V6, falsified) — each was measurably
real at the measurement layer but none moved the trajectory-layer outcome. The 53% drift is
structural: MLX and CPU make systematically different split decisions, independent of dataset
size. DEC-036 opens the iter=1 structural audit for S31.

---

## Ships / Does Not Ship

### Ships

| Item | Commit(s) | Description |
|------|-----------|-------------|
| K4 — fp64 cosNum/cosDen accumulators | `108c7a59d2`-family (T1/T2 commits) | Logically correct precision fix on all four cosDen/cosNum call sites. Removes the ~4e-3 float32 accumulation floor that will otherwise re-surface after the structural mechanism is resolved. |
| Fix 2 — fp64 gain/argmax | `90a0cb4475`, `364d4ee962` | Logically correct fp64 widening across 15 sites in FindBestSplit and FindBestSplitPerPartition. Removes the 3.81e-6 cast ULP floor from L3/L4. Prediction failed; fix is correct and kept. |
| 13 verdict docs under `docs/sprint30/` | Various verdict commits | Full chain of evidence for precision-class exhaustion. Each verdict doc is the authoritative record for its task. |
| `COSINE_RESIDUAL_INSTRUMENT` instrumentation in `csv_train.cpp` | `108c7a59d2`-family | Compile-gate retained for S31 audit reuse. Release builds unaffected. |
| DEC-035 PARTIALLY CLOSED | `24a0e829b8` (state-files commit) | Atomicity clause and Kahan rationale preserved for audit trail. |
| DEC-036 OPEN | `24a0e829b8` (state-files commit) | Structural divergence investigation; S31 T1 is iter=1 split-selection audit. |
| DEC-034 PARTIALLY FALSIFIED | `24a0e829b8` (state-files commit) | Outcome A remains valid for ST accumulator path (K4 targeted the right thing); outcome B confirmed dominant for LG (D3); V6 rules out outcome A as sufficient explanation for ST 53% drift. |

### Does Not Ship

| Item | Reason |
|------|--------|
| T4a (#93 ST-REMOVE) — Python/C++/CLI guard removal for ST+Cosine | Mechanism not fixed; G3a 53.30% fails <2.0% threshold. Guards at Python `_validate_params`, `train_api.cpp:TrainConfigToInternal`, and `csv_train.cpp:ParseArgs` remain intact. |
| T4b (#94 LG-REMOVE) — guard removal for LG+Cosine | D3 confirmed priority-queue ordering as the dominant LG mechanism; no precision fix addresses this. K2 fired. |
| T5 (#95 CLI exit wrap) | Not bundled into close; carried to S31-T-CLEANUP or S30 close commit. |
| T6 (#96 S29 CR residuals N-1/N-2/N-3/SF-3) | Same carry-forward. |

---

## Kill-Switches

| Kill-Switch | Criterion | Status |
|-------------|-----------|--------|
| K1 (T1 mechanism miss) | cosDen/cosNum not confirmed as dominant accumulator | NOT TRIGGERED — T1 confirmed cosDen at 4.07e-3, 407× above threshold |
| K2 (LG-Stress fail) | G3c RMSE ratio outside [0.98, 1.02] | **FIRED** at T3 G3c (1.440–1.452); T4b deferred; LG re-scoped as S31 priority-queue ordering task |
| K3 (performance regression) | G5 overhead >5% vs baseline | **TRIGGERED** at T3 G5 (+7.1%; measurement noisy at 8.5% CV); @performance-engineer consultation required; guards remain; K4 overhead is borderline and may clear on quieter hardware |
| K4 (Metal auto-reassociation) | Neumaier compensation defeated; fp32 insufficient | **FIRED** at T2 — per-term float32 computation floor (~1e-3 at depth=0) not addressable by compensated summation; fp64 fallback applied at all four call sites |

---

## DEC Transitions

| DEC | Previous Status | New Status | Notes |
|-----|-----------------|------------|-------|
| DEC-034 | RESOLVED — outcome A (moderate confidence) | PARTIALLY FALSIFIED for ST; outcome B confirmed dominant for LG | V6 flat N-scaling rules out pure precision mechanism for ST. S29's moderate confidence framing was appropriate: it left room to be partially wrong, and it was. D3 provides the LG outcome B evidence. |
| DEC-035 | ACTIVE | PARTIALLY CLOSED | Precision fix class exhausted. K4 + Fix 2 shipped; guard removal blocked. Atomicity clause and phased-plan rationale preserved as audit trail. |
| DEC-032 | PARTIALLY CLOSED | PARTIALLY CLOSED (unchanged) | Both ST+Cosine and LG+Cosine guards still in place at all three entry points (Python, nanobind, CLI). SA-H1 remains closed (S29). |
| DEC-036 | (did not exist) | OPEN | Structural divergence investigation. S31-T1-ITER1-AUDIT is the entry point. See DECISIONS.md. |

---

## Lessons Learned

**1. Precision-layer hypothesis pattern.** Four distinct precision mechanisms were sequentially
hypothesized and measured in S30: cosDen accumulation (K4), L3/L4 gain cast (Fix 2),
joint-denominator 64× amplification (D4), and L1 histogram N-scaling (V1/V6). Each was
measurably real at the measurement layer — the residuals were genuine — but none moved the
trajectory-layer outcome. The correct falsification oracle is N-scaling: a precision-
compounding mechanism produces b ≈ 1.0; b ≈ 0.0 rules out the whole class at once. Run the
cheap N-scaling check before committing to a multi-task precision sprint.

**2. Measurement-layer gates can mask trajectory-layer failures.** G2 passed at 12.5× gain
residual reduction at iter=1, but G3a did not move at iter=50. The gate spec tested the lever's
measurable effect (residual reduction), not the lever's actual mechanism against the target
outcome (compound drift reduction). This is the DEC-028 "kernel-ULP=0 ≠ full-path parity"
trap in a different costume. Gate criteria must test the mechanism, not a measurement-layer
proxy.

**3. Verification audits must verify their own methodology.** V2 discovered that D2's L3/L4
ruling was biased: the "fp32 gain" column in D2's CSV was actually derived from the same fp64
accumulators as the "fp64 gain" column — not from a true pre-K4 fp32 path. A stale struct-field
comment at `csv_train.cpp:135` was the tell. D2-redux corrected the methodology in ~30 lines
and confirmed D2's conclusion held, but only after a full re-run. Audits that are not
independently verifiable should be re-verified before load-bearing conclusions are drawn from
them.

**4. Two falsified predictions in a row means stop guessing precision and measure structure
directly.** When D4 falsified the 64× amplification hypothesis and Fix 2's prediction failed
cleanly at 0.00% improvement, the correct response was not to hypothesize a fifth precision
mechanism. It was to run V6 (cheap, 9-second sweep) to falsify or confirm the precision class
as a whole, and then open DEC-036 for the structural audit. The two-falsified-predictions
heuristic would have saved several tasks if applied after D4.

---

## S31 Handoff

S31 entry point: **S31-T1-ITER1-AUDIT** per DEC-036.

Build an iter=1 split-selection comparison harness. For each of the 6 tree layers on the S28
anchor cell (ST+Cosine, N=50k, depth=6, 128 bins, seeds 42–46), dump winning (feature_idx,
bin_idx, gain) from both CPU CatBoost and MLX. Identify the first diverging layer. The layer
identity names the mechanism class:

| First diverging layer | Mechanism class |
|----------------------|-----------------|
| Depth 0 (root split) | Cosine formula algebraic difference — formula port work |
| Depth 1–5 | Partition assignment error — tree-structure alignment |
| No divergence at iter=1 | Expand to iter=2 leaf-value and approx-update audit (K1 pre-authorized) |
| Divergence is a missing MLX feature | Feature-port sprint, not precision sprint (K2 pre-authorized) |

Deliverable: `docs/sprint31/t1-audit/verdict.md` naming the mechanism class with file:line
pointers on both sides.

Kill-switches K1 and K2 are pre-authorized per DEC-036 — no user checkpoint required.

The existing `COSINE_RESIDUAL_INSTRUMENT` instrumentation in `csv_train.cpp` is available for
reuse. The `csv_train_t3` binary (K4 + Fix 2 active, COSINE_T3_MEASURE guard bypass) is the
correct MLX reference binary.

---

## Appendix: File-of-Record Index

### Verdict documents

| File | Task | Verdict |
|------|------|---------|
| `docs/sprint30/t1-instrument/verdict.md` | T1 instrumentation | G1 PASS — cosDen named target |
| `docs/sprint30/t2-kahan/verdict.md` | T2 Kahan/K4 | G2 PASS (12.5×); K4 FIRED |
| `docs/sprint30/t3-measure/verdict.md` | T3 measurement | ALL PRIMARY GATES FAIL; K2/K3 FIRED |
| `docs/sprint30/d1-cpu-audit/verdict.md` | D1 CPU audit | CPU fp64 throughout L1–L5 |
| `docs/sprint30/d2-stack-instrument/verdict.md` | D2 stack instrument | 0/18 flips (biased — see V2) |
| `docs/sprint30/d2-redux/verdict.md` | D2-redux methodology fix | L3/L4 RULED OUT (genuine) |
| `docs/sprint30/d3-lg-outcome-ab/verdict.md` | D3 LG discriminator | Outcome B confirmed for LG |
| `docs/sprint30/d4-joint-denom/verdict.md` | D4 joint-denominator | 64× amplification FALSIFIED (2.42×) |
| `docs/sprint30/v1-drift-vs-n/verdict.md` | V1 N-scaling | L0 FALSIFIED (b = 0.0017) |
| `docs/sprint30/v2-d2-audit/verdict.md` | V2 D2 methodology audit | D2 L3/L4 measurement biased |
| `docs/sprint30/v5-dw-at-scale/verdict.md` | V5 DW at scale | DW 6.33% vs ST 53.30%; 8.4× gap |
| `docs/sprint30/v6-n500-confirmer/verdict.md` | V6 N=500 confirmer | L1 FALSIFIED; b ≈ 0.0 |
| `docs/sprint30/fix2-fp64-gain/verdict.md` | Fix 2 fp64 gain | Prediction FAILED (53.30% → 53.30%) |

### Close and decision documents

| File | Contents |
|------|----------|
| `docs/sprint30/sprint-close.md` | This document — authoritative close record |
| `docs/sprint30/sprint-close/pr-body.md` | PR #28 body draft |
| `.claude/state/DECISIONS.md` DEC-035 | Partially closed; closure addendum appended |
| `.claude/state/DECISIONS.md` DEC-036 | Structural divergence investigation; S31 T1 spec |
| `.claude/state/HANDOFF.md` | S30 CLOSING + S31 KICKOFF sections |
| `.claude/state/TODOS.md` | S30 phase-by-phase outcome table; S31 task skeleton |
| `.claude/state/CHANGELOG-DEV.md` | 2026-04-24 S30 CLOSING entry |
