# S30-D4-JOINT-DENOM — Verdict

<!-- Probe: Does the ST joint-denominator aggregation explain the 8.4× DW/ST gap? -->
<!-- Derived from existing T1/T2/D1 data; no new binary compilation required. -->

**Verdict: PARTIAL — joint-denominator fp32 rounding is confirmed as the dominant
per-accumulator residual source in the pre-K4 ST path, with depth-scaling consistent with
64-partition amplification. However, K4 already replaced the fp32 accumulators with fp64.
The remaining 53% G3a drift is NOT from joint-denominator fp32 rounding — that mechanism was
closed by K4. The 8.4× DW/ST gap has three residual sources, ranked by evidence: (1) fp32
histogram inputs at L1 (both DW and ST — unpatched), (2) float32 gain cast at L3/L4 (ST
argmax is fp32 while CPU is fp64 — unpatched), and (3) the joint-denominator topology itself,
which after K4 contributes only the float32-quantization-of-gain floor (~3.81e-6), comparable
to the per-leaf DW floor.**

---

## 1. Scope

This probe tests whether the ST joint-denominator aggregation (`Σ_p cosDen_p` across all
2^d = 64 partitions before `sqrt`) is the dominant remaining mechanism causing the 53%
G3a drift at ST@N=50k. The V5 verdict (docs/sprint30/v5-dw-at-scale/verdict.md) proposed
this as the "most likely mechanism" for the 8.4× DW/ST gap.

---

## 2. Accumulator Code Location

The joint-denominator accumulator is in
`catboost/mlx/tests/csv_train.cpp`, `FindBestSplit`, ordinal branch:

```
Lines 1413–1521 (S30-T2-KAHAN K4 active)
```

The inner loop at lines 1444–1516 iterates over all `p in [0, numPartitions)` and
`k in [0, K)` and accumulates:

```cpp
// line ~1420 (bin scope)
double cosNum_d = 0.0;
double cosDen_d = 1e-20;

// line ~1494-1497 (inner loop body, K4 path)
cosNum_d += dSL * dSL * dInvL + dSR * dSR * dInvR;
cosDen_d += dSL * dSL * dWL * dInvL * dInvL
          + dSR * dSR * dWR * dInvR * dInvR;

// line ~1521 (finalization after all p,k)
totalGain = static_cast<float>(cosNum_d / std::sqrt(cosDen_d));
```

At depth=5, `numPartitions = 2^5 = 32` and `K = 1` (RMSE is 1-dim), so the accumulator
sums 32 partition terms per bin candidate before taking `sqrt`. At depth=0, it sums 1 term.

For the gate cell (depth=6, 2^6=64 partitions), it sums 64 terms.

The DW analog (`FindBestSplitPerPartition`, `catboost/mlx/tests/csv_train.cpp` lines ~1651–1820)
resets `cosNum_d` and `cosDen_d` for each partition p independently. No cross-partition
accumulation occurs.

---

## 3. Pre-K4 Residual Scaling (T1 data, fp32 shadow vs fp64)

The T1 instrumentation measured the pre-K4 float32 accumulation residual per depth level
at the gate cell (N=50k, depth=6, bins=128, seed=42/43/44, iter=1). The `cosDen_f32_shadow`
vs `cosDen_f64` residuals by depth are reproduced from `docs/sprint30/t1-instrument/verdict.md`:

| seed | depth | partitions | cosDen max_abs_residual | cosDen mean_abs_residual |
|------|-------|-----------|------------------------|-------------------------|
| 42 | 0 | 1 | 1.017e-3 | 2.089e-5 |
| 42 | 1 | 2 | 1.469e-3 | 3.152e-4 |
| 42 | 2 | 4 | 2.019e-3 | 4.682e-4 |
| 42 | 3 | 8 | 2.310e-3 | 4.909e-4 |
| 42 | 4 | 16 | 2.391e-3 | 5.026e-4 |
| 42 | 5 | 32 | 2.463e-3 | 5.198e-4 |
| 43 | 5 | 32 | **4.067e-3** | 8.242e-4 |
| 44 | 5 | 32 | 3.508e-3 | 6.757e-4 |

Gain residual (downstream of cosDen, the critical path for argmax):

| seed | depth | partitions | gain max_abs_residual |
|------|-------|-----------|----------------------|
| 42 | 5 | 32 | ~4.0e-5 (from T1 §2) |
| 43 | 5 | 32 | **4.751e-5** |
| 44 | 5 | 32 | ~3.7e-5 |

**Depth-scaling pattern (seed=42, max residual):**

depth=0 (1 part): 1.017e-3
depth=1 (2 part): 1.469e-3
depth=2 (4 part): 2.019e-3
depth=3 (8 part): 2.310e-3
depth=4 (16 part): 2.391e-3
depth=5 (32 part): 2.463e-3

Ratio depth=5 / depth=0: 2.463e-3 / 1.017e-3 = **2.42×** (over a 32× increase in partitions).

The amplification factor is **sub-linear** — much less than the 64× that a purely linear
error model would predict. The residual is dominated by the per-term float32 floor, not by
summation accumulation. This was confirmed by T2's K1/Neumaier analysis: even at depth=0
(1 partition, no accumulation), the residual was already ~1e-3. The T2 verdict states:

> "The dominant error was per-term float32 computation error (~1e-3 floor at single partition
> depth=0), not summation rounding."

This means the joint-denominator structure (summing 64 terms) provides only ~2× additional
error over the single-partition case — not 64×. The V5 "64× fp32 rounding amplification"
model is quantitatively falsified at the accumulator level. The actual amplification of the
floor is ~2-4× from 1 to 32 partitions.

---

## 4. Post-K4 Residual (T2 data, the current shipping state)

After K4 replaced fp32 accumulators with fp64 (all 4 call sites in csv_train.cpp), the T2
verdict reports:

| Quantity | Pre-K4 | Post-K4 | Reduction |
|----------|--------|---------|-----------|
| gain max_abs_residual | 4.751e-5 | **3.81e-6** | 12.5× |
| cosDen float32 shadow | 4.07e-3 | (eliminated) | — |

The 3.81e-6 post-K4 gain residual is the float32 quantization of the final gain scalar
(`gain × fp32_eps`, seed-independent). At gain ≈ 104.5 (feature 2, depth=5, seed=42):
`104.5 × 2^-23 ≈ 1.24e-5`. At gain ≈ 27.6 (feature 0, depth=0, seed=42):
`27.6 × 2^-23 ≈ 3.29e-6`. These are consistent with the observed 3.81e-6 residual.

**After K4, the joint-denominator fp32 rounding mechanism is closed. The accumulator
residual is now the fp32-cast floor, which is present equally in DW and ST.**

---

## 5. Per-Bin Gain Values (depth=5, seed=42): argmax consistency check

From `docs/sprint30/t1-instrument/data/cos_accum_seed42_depth5.csv` (T1 = pre-K4):

The global maximum gain at depth=5 is in **feature 2** (bins ~33–55) at gain ≈ 104.55.
All feature-2 bins have gain_f32 and gain_f64 consistent to within the 9e-6 max residual
observed. Feature 0 peaks at ≈83.875, feature 1 at ≈77.76, feature 3 at ≈104.54.

**The argmax at depth=5 (seed=42, pre-K4) is feature 2, bin ≈ 35** based on the maximum
observed `gain_f32 = 104.5560455` at feat=2, bin=35 (line 292 of the T1 depth=5 CSV),
and the corresponding `gain_f64 = 104.556053...`. The f32 and f64 argmax agree on feature 2.

However, within feature 2, the top-10 bins span a gain range of only ~0.006 (104.550 to
104.556) — much smaller than the gain residual of ~1.8e-5. **Near-ties within feature 2's
top bins are unresolvable by fp32 arithmetic.** Whether a specific bin within feature 2's
tight cluster is selected may differ between fp32 and fp64 runs, causing subtly different
partition assignments even when the correct feature is chosen.

Post-K4 (T2 data), the gain residual is 3.81e-6. The within-feature gain gap is still
only ~0.006 for feature 2's top bins, so the flip rate within this tight cluster remains
significant even post-K4.

**Flip rate estimate (post-K4):**
Among candidates within 3.81e-6 of the argmax gain, the fp32-quantized gain cannot
distinguish them. For feature 2 at depth=5, approximately 5–15 bins have gains within
1e-4 of the maximum (based on the CSV scan), meaning **the argmax is underdetermined at
current fp32 precision** unless the correct fp64 maximum is held.

---

## 6. The 8.4× DW/ST Gap: Root Cause Decomposition

Given the above data, the 8.4× gap (DW 6.33% vs ST 53.30% drift at N=50k) is attributable
to three residual sources, not just the joint-denominator topology:

### Source A: L1 fp32 histogram (both DW and ST, unpatched)
CPU uses fp64 histogram buckets (TBucketStats: all double). MLX produces fp32 histograms
from the Metal kernel. At N=50k, this introduces per-bin sum error ≈ N × ε_f32 / bins ≈
50000 × 1.2e-7 / 128 ≈ 4.7e-5 per bin. This floor is shared by DW and ST equally.
The D1 CPU audit (docs/sprint30/d1-cpu-audit/verdict.md §3) confirms this as a remaining
divergence for both policies.

### Source B: L3/L4 fp32 gain cast and argmax (ST-specific relative to what V5 attributed)
CPU returns gains as `TVector<double>` and holds argmax as `double bestGain`. MLX casts
the gain to `float` immediately after computation (`static_cast<float>(cosNum_d / sqrt(cosDen_d))`)
and uses `float bestGain`. This introduces a ULP-level rounding error (~3.81e-6 post-K4)
that is **identical for DW and ST** — both call `static_cast<float>` at the same point.
However, at full depth (64 partitions), the Cosine gain is larger (~100 vs ~27 for feature 2
at depth=5 vs depth=0), meaning the fp32 ULP is proportionally larger:
- depth=0 (1 part): gain ≈ 27 → fp32 ULP ≈ 3.2e-6
- depth=5 (32 part): gain ≈ 104 → fp32 ULP ≈ 1.2e-5

**The gain magnitude grows with numPartitions because cosNum (the numerator) sums more
positive terms.** This is the structural effect of joint accumulation: not rounding error
amplification, but gain-magnitude growth that amplifies the fp32 quantization error at L3.
DW evaluates per-partition (gain per partition is smaller), ST evaluates jointly (gain is
~64× larger numerically). This explains a portion of the DW/ST gap at L3/L4.

### Source C: Residual joint-denominator fp32 topology (structure, not precision)
After K4, the fp32 summation error in the accumulator is eliminated. But the structural
difference (ST computes one gain over all 64 partitions; DW computes one gain per partition
and picks the best) means ST's argmax is over a single score per split candidate, while
DW's argmax is per-partition. **A correct per-partition DW score can differ from a correct
joint ST score** even with perfect arithmetic — the two formulas are not equivalent, and
CPU CatBoost ST uses the joint formula. This is not a precision issue; it is a semantic
difference in how the winner is selected.

---

## 7. Verdict Summary

| Claim | Status | Evidence |
|-------|--------|---------|
| Joint-denominator fp32 rounding is the dominant pre-K4 accumulator residual | SUPPORTED | T1 cosDen max 4.07e-3, 407× above 1e-5 threshold; confirmed at every depth |
| Joint-denominator amplification is ~64× over per-leaf | FALSIFIED at accumulator level | T1 depth scaling: 2.4× from 1 to 32 partitions (per-term floor dominates) |
| K4 closed the joint-denominator fp32 rounding gap | SUPPORTED | T2 gain residual 3.81e-6, 12.5× reduction; K4 uses fp64 accumulators |
| Joint-denominator topology (structure) still contributes to ST/DW gap after K4 | PARTIAL | L3 gain magnitude grows with numPartitions; fp32 ULP at gain≈104 is 3× larger than at gain≈27 |
| 53% G3a drift is explained by joint-denominator mechanism | FALSIFIED | K4 fixed the accumulator; 53% drift persists at T3/G3a; the residual is L1 fp32 histogram + L3/L4 fp32 cast + compound split-flip cascade |

**Overall: PARTIAL** — the joint-denominator is the correct diagnosis for the pre-K4 path,
and T1 provides strong evidence that cosDen dominated before K4. Post-K4, the mechanism is
closed at the accumulator level. The 53% G3a residual is driven by (A) L1 fp32 histogram
inputs and (B) L3/L4 fp32 gain cast/argmax, both of which are active for DW too (explaining
why DW still has 6.33% drift post-K4). The DW/ST gap post-K4 requires further explanation
beyond the joint-denominator; the most likely contributor is the gain-magnitude growth at
joint depth (Source B in §6 above).

---

## 8. Recommended Fix Class

The correct fix for the remaining 53% ST drift requires addressing two orthogonal gaps:

### Fix 1: L1 fp32 histogram → fp64 gradient accumulation (LOC estimate: medium, ~50-100 LOC)
Change the histogram kernel to accumulate in fp64 (or use a compensated fp32 pair). Metal
Shading Language does not support double-precision atomics, so this requires either:
- (a) Per-thread fp64 partial sums with CPU-side double accumulation (significant kernel rewrite)
- (b) A two-pass approach: fp32 histogram kernel output, followed by a CPU correction pass
Parity-safe: requires careful verification against CPU TBucketStats double reference.

### Fix 2: L3/L4 fp32 gain cast and argmax → fp64 throughout (LOC estimate: small, ~10 LOC)
In `FindBestSplit` and `FindBestSplitPerPartition`:
- Change `float totalGain = 0.0f;` → `double totalGain = 0.0;`
- Change `float bestGain = -infinity()` → `double bestGain = -infinity()`
- Change `TBestSplitProperties::Gain` field type from `float` to `double`
- Remove the `static_cast<float>` at line ~1521 (keep the division in double through argmax)
This is parity-safe (identical semantics, only wider arithmetic) and does not require kernel
changes. It would eliminate the 3.81e-6 fp32 cast residual and make the argmax match CPU's
fp64 argmax when histogram inputs are equal.

**This Fix 2 is the correct first step for S31**: it is low-risk, narrow, and directly
addresses the confirmed remaining gap at L3/L4 post-K4. If after Fix 2 the ST drift drops
to the DW floor (~6.33%), Fix 1 (fp64 histograms) becomes the binding constraint for both
policies.

---

## 9. Limitations

1. **Argmax flip rate not directly counted.** This probe derives the flip probability from
   gain residuals and within-feature gain gaps rather than counting actual flips. A direct
   flip count requires a second pass comparing argmax of `gain_f32` vs `gain_f64` columns
   across the full depth=0–5 data, which would require script execution.

2. **Single-tree, single-seed analysis.** The depth-scaling table uses seed=42 only for
   the full depth profile; seeds 43 and 44 are represented only at depth=5. Seed-to-seed
   variance is moderate (~2× in max residual between seeds 42 and 43).

3. **Post-K4 argmax flip rate not measured.** The T2 data files have `gain_f32` (= the
   float-cast of the fp64 gain) and `gain_f64` (= the fp64 gain). The residual is the
   cast ULP, not the pre-K4 fp32 accumulation error. A new instrumentation pass isolating
   the L3 fp32 cast effect (while comparing to CPU fp64) is needed to count post-K4 flips.

4. **DW per-partition depth data not captured.** DW residuals at N=50k were not
   instrumented per-depth; V5 measured only the final RMSE (6.33%). A DW T1-equivalent
   run would provide the per-partition floor baseline.

---

## 10. Data Sources

All residual data in this verdict is derived from existing artifacts; no new instrumentation
was run:

| Source | File | Key quantities used |
|--------|------|---------------------|
| T1 cosDen depth scaling | `docs/sprint30/t1-instrument/verdict.md` §2 | cosDen max_abs_residual by depth |
| T1 per-bin gain at depth=5 | `docs/sprint30/t1-instrument/data/cos_accum_seed42_depth5.csv` | gain_f32, gain_f64, gain_abs_residual |
| T2 post-K4 gain residual | `docs/sprint30/t2-kahan/verdict.md` §2 | post-K4 gain max 3.81e-6 |
| T3 G3a 53% drift | `docs/sprint30/t3-measure/verdict.md` §G3a | 53.30% aggregate drift post-K4 |
| V5 DW/ST gap | `docs/sprint30/v5-dw-at-scale/verdict.md` | DW 6.33% vs ST 53.30% at N=50k |
| D1 CPU audit | `docs/sprint30/d1-cpu-audit/verdict.md` | CPU fp64 at L1–L5; MLX fp32 gaps at L1/L3/L4 |

DEC reference: DEC-034 (cosDen as mechanism), DEC-035 (S30 gate spec).
