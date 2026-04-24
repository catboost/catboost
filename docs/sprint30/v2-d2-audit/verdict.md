# S30-V2-D2-AUDIT — Methodology Audit of D2 (S30-D2-INSTRUMENT)

**Verdict: BIASED (L3 layer) / PARTIAL overall.**

D2's **L3 "gain residual" (3.81e-6) and L4 "0/18 argmax flips" measurements are
NOT measurements of fp32-path-vs-fp64-path divergence**.  They are measurements
of the final `static_cast<float>(gain_f64)` cast alone.  Both the "fp32 gain"
and "fp64 gain" columns in `gain_scalar_seedN_depthD.csv` are derived from the
**same fp64 accumulators** (`cosNum_d`, `cosDen_d`) introduced by T2/K4.  The
only difference between them is the final scalar cast.

**L5 (leaf value) is measured correctly** (fp64 re-scatter from fp32
gradients is a legitimate shadow).  **L1/L2 (cosNum/cosDen) are measured
correctly** via the `cosNum_f32_shadow` / `cosDen_f32_shadow` parallel
accumulators — but D2 inherits these from T2 and does not re-analyse them at
L3.

S31 MUST NOT proceed on D2's "L3 is not dominant" conclusion until L3 is
re-measured with a true pre-K4 fp32 gain path.

---

## Per-Layer Audit

### L1 / L2 — cosNum / cosDen accumulators (T2 scope)

**Measured correctly.**  After commit `afdfbc0af8` (T2/K4) the primary code
path accumulates in double (`cosNum_d`, `cosDen_d`), but a **parallel fp32
shadow** (`cosNum_f32_shadow`, `cosDen_f32_shadow`) is also accumulated
specifically for instrumentation.  The CSV residual
`|cosNum_f32_shadow - cosNum_d|` reflects the real pre-K4 fp32 accumulation
error (T1 reported ~4e-3, which is a meaningful divergence, not a cast
artifact).

Evidence: `catboost/mlx/tests/csv_train.cpp:1426–1427, 1501–1508, 1540–1543`.

### L3 — Gain scalar after cast

**BIASED.**  Post-K4, the gain is computed entirely in double and cast to
float only at the end:

```
// csv_train.cpp:1518–1522
if (scoreFunction == EScoreFunction::Cosine) {
    // S30-T2-KAHAN K4: compute gain in double, cast only the final gain to float.
    totalGain = static_cast<float>(cosNum_d / std::sqrt(cosDen_d));
}
```

The instrumentation then records:

```
// csv_train.cpp:1530–1546
if (g_cosInstr.dumpCosDen && scoreFunction == EScoreFunction::Cosine) {
    double gain_f64_ref = cosNum_d / std::sqrt(cosDen_d);
    ...
    rec.gain_f32   = totalGain;           // = float(cosNum_d / sqrt(cosDen_d))
    rec.gain_f64   = gain_f64_ref;        // = cosNum_d / sqrt(cosDen_d)
    ...
}
```

Therefore `gain_f32 - gain_f64` in the CSV is literally
`float(x) - x` where `x = cosNum_d / sqrt(cosDen_d)`.  That is the fp32
cast ULP of the gain, not the difference between an fp32 accumulation path
and an fp64 accumulation path.

Numerical check: at gain magnitude ≈ 64–80, fp32 ULP ≈ gain × 2^-24
≈ 80 × 5.96e-8 ≈ 4.8e-6.  D2's reported 3.81e-6 sits right at this floor and
is **seed-independent across all 18 depth-seed pairs** (D2 verdict table).
A true divergence measurement would be seed-dependent because it would reflect
different cancellation patterns per seed.  The uniform 3.81e-6 is the fingerprint
of measurement floor, not signal.

Note that `gain_f32`'s own struct-field comment ("ComputeCosineGainKDim result
using f32 inputs", csv_train.cpp:135) is **stale**; T2/K4 changed the assigned
value but not the comment.  This is likely how the methodology bug was missed
in self-review.

What a correct L3 measurement would look like: maintain a parallel pre-K4
fp32 gain

```
float gain_f32_path = static_cast<float>(
    cosNum_f32_shadow / sqrt(cosDen_f32_shadow));   // pre-K4 arithmetic
double gain_f64_ref = cosNum_d / sqrt(cosDen_d);    // true fp64 path
residual = |gain_f32_path - gain_f64_ref|;
```

Given T1 reported cosNum/cosDen residuals of ~4e-3 and gain ~= ratio-of-sums,
the true L3 residual is very likely **3–5 orders of magnitude larger** than
D2's reported 3.81e-6 — i.e. potentially in the 1e-3 to 1e-4 range.

### L4 — Argmax flip rate

**BIASED (downstream of L3 bias).**  `run_d2_instrument.py` independently
argmaxes the `gain_f32` and `gain_f64` columns of the same CSV
(commit `a7bdc6a88e` cover text: "independently re-argmax gain_f32 and
gain_f64 columns per depth").  Since those two columns differ only by the
final cast, the argmaxes almost never disagree: the gap between the top two
candidates at each depth is generally much larger than 3.81e-6.

A **true** flip test would compare:
  * MLX's actual split choice (produced by full fp32 path, pre-K4)
  * An fp64-from-scratch path's split choice

Because the current "fp32" column is already fp64-derived, D2's "0 / 18
flips" is the trivial observation that `argmax(x) == argmax(float(x))` when
all gaps exceed one float ULP — it is **not** a measurement of split-stability
under true fp32 accumulation.

### L5 — Leaf value (scatter_add + Newton step)

**Measured correctly.**  At `csv_train.cpp:4347–4369` the fp64 reference is
constructed by **re-running scatter_add in fp64** on the per-document fp32
gradient/hessian arrays, then applying the Newton step in fp64:

```
// csv_train.cpp:4353–4361
std::vector<double> gSums_d(numLeaves, 0.0);
std::vector<double> hSums_d(numLeaves, 0.0);
for (ui32 i = 0; i < trainDocs; ++i) {
    ui32 leaf = pPtr[i];
    if (leaf < numLeaves) {
        gSums_d[leaf] += static_cast<double>(gPtr[i]);
        hSums_d[leaf] += static_cast<double>(hPtr[i]);
    }
}
```

This is a legitimate shadow accumulation.  One caveat: the input gradients
`gPtr` / `hPtr` are themselves fp32 (the upstream gradient computation is
fp32).  So this measures "fp32 scatter_add vs fp64 scatter_add of the SAME
fp32 inputs".  That matches what the MLX kernel actually does relative to a
true CPU fp64 reference over fp32 inputs — acceptable for this sprint's
scope.  The reported 4.79e-8 leafVal max residual is trustworthy **as a
measurement of scatter + Newton step rounding**, not of full-precision
cascade divergence.

### L4b / approxUpdate — cursor update

**BIASED (minor).**  `csv_train.cpp:4591–4611` explicitly upcasts the fp32
leaf values as the "reference":

```
// csv_train.cpp:4606–4611
// Double reference: upcast float32 leaf value for this doc's partition
uint32_t leaf = pPtr[i];
double dlvRef = (leaf < numLeaves)
    ? static_cast<double>(lvF[leaf])       // <-- promote-from-fp32
    : 0.0;
double res = std::fabs(static_cast<double>(dlvF[i]) - dlvRef);
```

Inline comment acknowledges this: "this measures the cursor-update
float32 → float64 step, not the full gSums residual" (line 4591–4593).
The reported 0.0 cursor residual is literally `|float(x) - double(float(x))|`
after an integer-indexed lookup — trivially zero.  This does NOT contradict
D2 because D2 doesn't use this layer to draw a load-bearing conclusion, but
it is a known-biased measurement and should not be cited as evidence of
anything.

---

## Summary Table

| Layer | D2 measurement | Methodology | fp64 reference source | Conclusion |
|-------|----------------|-------------|-----------------------|------------|
| L1    | cosNum residual (from T2) | Parallel fp32 shadow accumulator | fp32_shadow vs fp64_primary | **CLEAN** |
| L2    | cosDen residual (from T2) | Parallel fp32 shadow accumulator | fp32_shadow vs fp64_primary | **CLEAN** |
| L3    | Gain scalar residual 3.81e-6 | `gain_f32 = float(cosNum_d/sqrt(cosDen_d))` | **Same fp64 inputs, only final cast differs** | **BIASED — measures cast ULP, not path divergence** |
| L4    | Argmax flip rate 0/18 | Post-processes L3 CSV columns | Inherits L3 bias | **BIASED — derived from biased L3** |
| L5    | leafVal residual 4.79e-8 | True fp64 scatter_add + Newton in fp64 | fp64 re-scatter from fp32 grads | **CLEAN** (scoped) |
| approxUpdate | 0.0 | Upcast fp32 leaf value | **Promote-from-fp32** | **BIASED — measures nothing** |

---

## Impact Assessment (if D2 re-measured correctly)

If L3 were re-measured against a true pre-K4 fp32 gain path:

1. The L3 residual is plausibly **1e-3 to 1e-4** (derived from T1's
   ~4e-3 cosNum/cosDen residuals and typical cosNum/cosDen magnitudes).  This
   would be large enough to produce argmax flips when the top-vs-second-best
   gain margin is comparable.

2. With corrected L3, the "argmax flip" test (L4) should be re-run in two
   variants: (a) flip-count between the true pre-K4 fp32 path and fp64 path
   on the same histograms; (b) flip-count between post-K4 fp32-cast path and
   fp64 path.  The (a) variant is what actually diagnoses whether K4 closed
   the drift; the (b) variant is what D2 actually ran and is ~guaranteed 0.

3. **A non-zero flip rate in (a) would reopen the hypothesis that L3 IS the
   dominant source of drift and K4 fully closes it — the opposite of D2's
   current conclusion.**  In that case S31's "L0 histogram kernel
   quantization" direction is either unnecessary or significantly
   lower-priority.

4. If flip rate in (a) is still 0, D2's broader conclusion (L3 is not the
   dominant source) survives and S31 can proceed to L0 — but the evidence
   base would be stronger and defensible.

---

## Recommendation

**S31 plan should be PAUSED pending a D2 re-measurement with corrected L3/L4
methodology.**  Concretely:

1. Extend the instrumentation so `rec.gain_f32` records
   `float(cosNum_f32_shadow / sqrt(cosDen_f32_shadow))`, i.e. the true pre-K4
   fp32 gain path — not the post-K4 cast-from-fp64 path.
2. Re-run the 3-seed × 6-depth sweep.
3. Re-run the Python argmax post-processor on the corrected CSVs.
4. Either confirm D2's conclusion (flip rate 0 on the corrected data → S31
   goes to L0) or invalidate it (flip rate > 0 → K4 actually WAS load-bearing
   and the drift investigation needs to shift back to the pre-K4
   accumulation path or K4's trajectory effects).

The re-measurement is a ~30-line code change and a half-hour of compute.  The
cost of NOT re-measuring is a potential multi-sprint investment in the wrong
layer.

---

## Code Locations (for the re-measurement)

- L3 fp32-path fix site: `catboost/mlx/tests/csv_train.cpp:1530–1546`
  (ordinal branch) plus the one-hot branch at the same file.  Change
  `rec.gain_f32 = totalGain;` to
  `rec.gain_f32 = static_cast<float>(cosNum_f32_shadow / std::sqrt(cosDen_f32_shadow));`
- Stale comment to fix: `catboost/mlx/tests/csv_train.cpp:135` ("using f32
  inputs" — this was accurate at T1 and became false at T2/K4).
- L4 recomputation: runs automatically from the corrected gain_scalar CSV via
  `docs/sprint30/d2-stack-instrument/run_d2_instrument.py` (no change needed
  to the post-processor).
