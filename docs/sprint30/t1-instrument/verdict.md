# S30-T1-INSTRUMENT — Triage Verdict

**G1: PASS — dominant accumulator fingered: cosDen (Cosine joint-denominator), residual max 4.07e-3 (seed 43, depth 5); cosDen is the correct T2 Kahan target per DEC-034. K1 does NOT fire.**

---

## 1. Cell Used

| Parameter | Value | Source |
|-----------|-------|--------|
| N | 50,000 | Standard gate config (DEC-035) |
| Features | 20 (canonical S26: f0×0.5 + f1×0.3 + noise×0.1) | `make_data(N, seed)` in `run_instrument.py` |
| Depth | 6 | ST anchor cell from `docs/sprint28/fu-obliv-dispatch/t7-gate-report.md` G6a |
| max_leaves | 64 (2^6, SymmetricTree) | Derived from depth |
| Bins (max_bin) | 128 | Standard gate config (DEC-035) |
| Learning rate | 0.03 | Matches S28 T7 gate config |
| L2 reg | 3.0 | Default |
| Loss | RMSE | ST anchor cell |
| score_function | Cosine | ST anchor cell |
| grow_policy | SymmetricTree | ST anchor cell |
| iterations | 1 (iter=0 internally, "iter=1" in 1-indexed terms) | T1 spec: instrument at iter=1 before compounding |
| Seeds | 42, 43, 44 | 3 seeds per DEC-035 T1 minimum |
| Binaries | `csv_train_instrument` (-DCOSINE_RESIDUAL_INSTRUMENT) | Guard bypassed for instrumentation only |

**Cell source:** The exact T7 cell (`N=50000, depth=6, bins=128, seeds=42..46, iters=1`) is documented in
`docs/sprint28/fu-obliv-dispatch/t7-gate-report.md` §G6a, where the 0.77% drift was first measured.
Seeds 42–44 (3 of the original 5) are used here. No proxy cell was needed.

---

## 2. Per-Accumulator Residual Summary Table

Residual = |float32_value − float64_reference|, measured at iter=0 (1-indexed iter=1).

### cosDen — Cosine joint-denominator (FindBestSplit, ordinal branch)

Accumulates `Σ_{p,k} sumLeft² × weightLeft / (weightLeft + λ)² + sumRight² × weightRight / (weightRight + λ)²`
across all numPartitions × approxDim combinations per bin candidate.

| seed | depth | n bins | max residual | mean residual | p99 residual |
|------|-------|--------|-------------|---------------|-------------|
| 42 | 0 | 2540 | 1.017e-3 | 2.089e-5 | 5.077e-4 |
| 42 | 1 | 2540 | 1.469e-3 | 3.152e-4 | 1.039e-3 |
| 42 | 2 | 2540 | 2.019e-3 | 4.682e-4 | 1.478e-3 |
| 42 | 3 | 2540 | 2.310e-3 | 4.909e-4 | 1.596e-3 |
| 42 | 4 | 2540 | 2.391e-3 | 5.026e-4 | 1.688e-3 |
| 42 | 5 | 2540 | 2.463e-3 | 5.198e-4 | 1.775e-3 |
| 43 | 5 | 2540 | **4.067e-3** | 8.242e-4 | 2.870e-3 |
| 44 | 5 | 2540 | 3.508e-3 | 6.757e-4 | 2.289e-3 |

**Across all seeds, all depths: max cosDen residual = 4.067e-3 (seed=43, depth=5).**

### cosNum — Cosine numerator (same topology)

Accumulates `Σ_{p,k} sumLeft²/(weightLeft + λ) + sumRight²/(weightRight + λ)`.

| seed | depth | max residual | mean residual | p99 residual |
|------|-------|-------------|---------------|-------------|
| 42 | 5 | 2.222e-3 | 4.891e-4 | 1.701e-3 |
| 43 | 5 | **4.067e-3** | 8.009e-4 | 2.986e-3 |
| 44 | 4 | 3.722e-3 | 6.655e-4 | 2.265e-3 |

**Across all seeds, all depths: max cosNum residual = 4.067e-3 (seed=43, depth=5).**

### gain — ComputeCosineGainKDim result (derived from cosDen/cosNum)

`gain = cosNum / sqrt(cosDen)`; residual is the downstream gain error.

| seed | depth | max residual | mean residual | p99 residual |
|------|-------|-------------|---------------|-------------|
| 42 | 3 | 3.002e-5 | 5.688e-6 | 1.935e-5 |
| 43 | 5 | **4.751e-5** | 9.128e-6 | 3.311e-5 |
| 44 | 4 | 3.649e-5 | 7.774e-6 | 2.649e-5 |

**Across all seeds, all depths: max gain residual = 4.751e-5 (seed=43, depth=5).**

### gSum — per-leaf gradient scatter_add (leaf value Newton step)

Accumulates `Σ_{i: partition[i]==leaf} grad[i]` via `mx::scatter_add_axis`.
The RMSE hessian is 1 for all docs, so hSum residual is exactly 0 (integer arithmetic).

| seed | n leaves | max gSum residual | mean gSum residual | p99 gSum residual |
|------|----------|-------------------|---------------------|------------------|
| 42 | 64 | 1.906e-3 | 1.470e-4 | 1.556e-3 |
| 43 | 64 | **6.257e-3** | 2.390e-4 | 2.683e-3 |
| 44 | 64 | 5.223e-3 | 2.246e-4 | 2.499e-3 |

### leafVal — Newton leaf value (derived from gSum)

`leafVal = -lr × gSum / (hSum + l2)`. For RMSE: hSum ≈ N/numLeaves ≈ 780, so
residual propagation factor is `lr / (hSum + l2) ≈ 0.03 / 783 ≈ 3.8e-5`.

| seed | n leaves | max leafVal residual | mean leafVal residual | p99 leafVal residual |
|------|----------|----------------------|-----------------------|----------------------|
| 42 | 64 | 1.593e-8 | 2.732e-9 | 1.484e-8 |
| 43 | 64 | 3.963e-8 | 3.524e-9 | 3.285e-8 |
| 44 | 64 | 3.186e-8 | 2.587e-9 | 1.346e-8 |

### approxUpdate — per-doc cursor increment (mx::take + mx::add)

`docLeafValue = leafValues[partitions[i]]`, cursor += docLeafValue.
Double reference: upcast float32 leafValues to double before lookup.

| seed | n docs (sampled) | max inc residual | verdict |
|------|-----------------|-----------------|---------|
| 42 | 10000 | 0.0 | exact |
| 43 | 10000 | 0.0 | exact |
| 44 | 10000 | 0.0 | exact |

---

## 3. Verdict

**Named target: `cosDen` — the float32 joint-Cosine denominator accumulator in `FindBestSplit` (SymmetricTree ordinal branch).**

**Max observed residual: 4.067e-3 (cosDen, seed=43, depth=5 of 6), well above the 10⁻⁵ threshold.**
**Ratio cosDen : gain : gSum (at seed=43, depth=5): 4.07e-3 : 4.75e-5 : 6.26e-3**

### Analysis

Four accumulators were measured. The pattern is unambiguous:

1. **cosDen and cosNum** both exceed 10⁻⁵ by ~3 orders of magnitude (max ~4e-3). They grow
   monotonically with tree depth as more partitions are accumulated into the joint sum, matching the
   expected float32 catastrophic-cancellation signature for running sums of small positive terms.

2. **gain** (downstream of cosDen/cosNum) exceeds 10⁻⁵ (max ~4.8e-5). This is the per-bin gain
   error that can flip split selection, providing the link between accumulator precision loss and
   the 0.77% aggregate RMSE drift measured in T7.

3. **gSum** exceeds 10⁻⁵ in absolute units (max ~6.3e-3) but its downstream leafVal residual
   is ~4e-8 (negligible). The propagation factor `lr / (hSum + l2) ≈ 3.8e-5` suppresses gSum
   error to below 10⁻⁷ in the leaf values themselves. gSum precision is **not** the dominant
   source of the observed RMSE drift.

4. **leafVal** residual: max ~4e-8 (< 10⁻⁷). Not a meaningful error source.

5. **approxUpdate** residual: exactly 0. MLX `mx::take` is bit-exact; cursor update introduces no
   additional error.

### Mechanistic interpretation

The observed 0.77% RMSE drift originates in `cosDen`/`cosNum` accumulator precision loss
→ gain residual ~4.8e-5 per bin candidate → incorrect split selection at some depth levels
→ subtly wrong partition assignment for the full tree → downstream RMSE divergence.

The gSum residual is larger in absolute terms but contributes ~3 orders of magnitude less to the
final prediction error because of the Newton step's attenuation factor. DEC-034's diagnosis
("float32 joint-Cosine denominator compounding") is confirmed: `cosDen` (and `cosNum`, same
accumulation topology) are the correct T2 targets.

### Dominant accumulator selection

`cosDen` is named as the **primary T2 target** over `cosNum` for the following reasons:
- Both have equivalent max residuals (~4e-3). They share the same accumulation topology and will
  be patched in the same Kahan/Neumaier pass.
- `cosDen` is the operand of `std::sqrt(...)` in `ComputeCosineGainKDim`, so its residual
  directly drives the denominator error in the gain comparison. A relative error in the denominator
  of 4e-3/denominator_value is the critical path.
- The T2 Kahan patch will necessarily cover both `cosNum` and `cosDen` in the same loop body,
  since they are adjacent `+=` statements. Naming `cosDen` as the target implicitly patches both.

---

## 4. K1 Trigger Assessment

**K1 does NOT fire.**

DEC-034's named target (`cosDen`, float32 joint-Cosine denominator) is confirmed as the
dominant accumulator source for residuals exceeding 10⁻⁵. The actual dominant accumulator
is `cosDen`, consistent with DEC-034.

T2 (`S30-T2-KAHAN`) proceeds as planned: apply Kahan/Neumaier compensated summation to the
`cosNum += ...` and `cosDen += ...` statements in:

- `FindBestSplit` (SymmetricTree path) — ordinal branch inner loop
- `FindBestSplit` (SymmetricTree path) — one-hot branch inner loop
- `FindBestSplitPerPartition` (Depthwise/Lossguide paths) — ordinal and one-hot branches

The Kahan/Neumaier approach (T2's planned remedy) is appropriate: it directly addresses the
float32 running-sum rounding error that accumulates over numPartitions × approxDim loop
iterations in the `cosDen`/`cosNum` accumulators.

**gSum is not the T2 target.** Although gSum exceeds 10⁻⁵ in absolute units, its contribution
to leaf values is below 10⁻⁷ and does not explain the observed RMSE drift. DW+Cosine ships
in-envelope at ~1.6% despite having the same gSum accumulation path, confirming that gSum
precision is not the bottleneck.

---

## 5. Limitations

1. **3 seeds only (42, 43, 44).** The minimum specified by DEC-035 T1. Seed variance in
   residuals is moderate (depth-5 max cosDen: 2.46e-3 to 4.07e-3 across seeds). Additional
   seeds would narrow the confidence interval on the max-residual estimate.

2. **Leaf-value gSum residual measured but not confirmed as non-mechanism.** The analysis above
   argues that gSum's large absolute residual is suppressed by the Newton step attenuation
   factor. This is a mathematical argument, not an empirical one. To confirm, T2 could be run
   with Kahan on cosDen/cosNum only (not gSum) and G2 (≥10× drift reduction) checked. If
   G2 passes, gSum is confirmed non-mechanism. If G2 fails, gSum would need instrumentation.

3. **Only SymmetricTree (ST) instrumented.** LG+Cosine and DW+Cosine were not instrumented.
   The DEC-034 verdict (moderate confidence) holds that LG shares the same cosDen accumulator;
   this T1 result is consistent with that claim but does not independently verify it for LG.

4. **Depth range limited to 0–5 (one tree).** A single tree at depth=6 grows through 6 depth
   levels; residuals were captured per-level. No multi-iteration data (this is iter=1 only,
   by design — multi-iteration compounding is T3's scope, not T1's).

5. **approxUpdate residual limited to first 10,000 docs.** Full 50,000-doc sampling was not
   done to keep CSV size tractable. The 10,000-doc sample is sufficient to establish that
   `mx::take` + `mx::add` introduce zero additional error (all 0.0 across 3 seeds).

---

## Data Artifacts

All CSV files are under `docs/sprint30/t1-instrument/data/`:

| File pattern | Contents | Rows per file |
|---|---|---|
| `cos_accum_seedN_depthD.csv` | cosNum/cosDen/gain f32 vs f64 per bin candidate | 2540 |
| `leaf_sum_seedN.csv` | gSum/hSum/leafVal f32 vs f64 per leaf | 64 |
| `approx_update_seedN.csv` | cursor increment f32 vs f64 per doc (first 10k) | 10000 |

CSV columns: `featIdx, bin, {accum}_f32, {accum}_f64, {accum}_abs_residual`

Runner script: `docs/sprint30/t1-instrument/run_instrument.py`
Build command: see §5 of T7 gate report or run_instrument.py docstring.
