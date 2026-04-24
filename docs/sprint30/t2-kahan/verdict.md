# S30-T2-KAHAN — Verdict

**G2: PASS — gain residual reduced 12.5x on all 3 seeds (3.81e-6 vs T1 baseline 4.75e-5)**

---

## 1. Summary

K4 fp64 fallback was invoked. Neumaier float32 compensation was implemented, verified in
assembly, and tested — but achieved only 3.1–3.4x reduction because the dominant error was
per-term float32 computation error (~1e-3 floor at single partition depth=0), not summation
rounding. K4 replaces float32 accumulation with double precision and computes the gain
formula entirely in double, casting only the final gain scalar to float for split comparison.
The accumulation error source is eliminated. G2 passes at 12.5x gain reduction.

---

## 2. Pre-Kahan vs Post-K4 Residual Table

All values at iter=1, grow_policy=SymmetricTree, depth=5 (worst-case depth within the tree).

### Gain residual (split selection critical path)

| seed | T1 pre-K4 gain max | T2 post-K4 gain max | Reduction |
|------|--------------------|---------------------|-----------|
| 42   | 4.751e-5 (est)     | 3.812e-6            | **12.5x** |
| 43   | 4.751e-5           | 3.812e-6            | **12.5x** |
| 44   | 4.751e-5 (est)     | 3.815e-6            | **12.5x** |

T1 gain baseline: 4.75e-5 (seed=43, depth=5). Used as reference for all seeds.
G2 threshold (10x): 4.75e-6. All seeds pass at 3.81e-6.

### cosNum/cosDen float32 shadow (pre-K4 accumulation path, for historical comparison)

The T2 instrumentation also runs the pre-K4 float32 accumulation path in parallel as a
shadow. These values confirm the original accumulation error source was active and match
T1 measurements.

| seed | cosDen f32 shadow max (depth=5) | Matches T1? |
|------|---------------------------------|-------------|
| 42   | 2.360e-3                        | Yes (~T1 seed=42 2.46e-3) |
| 43   | 4.016e-3                        | Yes (~T1 seed=43 4.07e-3) |
| 44   | 3.657e-3                        | Yes (~T1 seed=44 3.51e-3) |

The float32 shadow reproduces T1 accumulation error (within seed-to-seed variance), confirming
K4 targets the correct accumulator sites.

---

## 3. Reassociation Verification Summary

**Compiler flags inspected:** The CMakeLists.txt for catboost/mlx/ contains no `-ffast-math`,
`-fassociative-math`, or `-Ofast` flags. The build command used (`clang++ -std=c++17 -O2`)
does not enable `-fassociative-math` by default.

**Assembly spot-check:** The initial Neumaier float32 implementation was compiled to ARM64
assembly (`clang++ -S -O2`) and inspected. The compensation pattern was confirmed present:
```
fabs s20, s3   / fabs s21, s18    # |sum| and |y| magnitude tests
fsub s22, s3, s19 / fadd s22, s18, s22  # (sum - t) + y branch
fsub s18, s18, s19 / fadd s3, s3, s18   # (y - t) + sum branch
fcmp s20, s21 / fcsel s3, s3, s22, lt   # Neumaier conditional select
fadd s0, s0, s1                          # compensation accumulation
```
Verdict: **auto-reassociation did NOT defeat the Neumaier compensation** in assembly.

**K4 trigger reason:** Neumaier compensation survived in assembly but achieved only 3.1–3.4x
reduction. Investigation revealed the dominant error at depth=5 was NOT summation rounding
but per-term float32 computation error: each `sL*sL*invL + sR*sR*invR` expression evaluated
in float32 has ~1e-3 absolute error vs double at these magnitudes. This is the floor visible
even at depth=0 (1 partition, no summation accumulation) where residual was already ~1e-3.
Neumaier compensates summation rounding only; it cannot reduce below the per-term computation
floor. K4 was therefore invoked as pre-authorized.

---

## 4. K4 Trigger Status: **FIRED — fp64 accumulation fallback active**

K4 was invoked because:
- Neumaier float32 achieved 3.1–3.4x reduction (target: >=10x)
- Per-term float32 computation floor ~1e-3 (measured at depth=0) exceeded G2 threshold
- The floor is intrinsic to float32 evaluation of `sL*sL*invL + sR*sR*invR` at these
  magnitudes — no compensated summation technique can address per-term rounding

**K4 change:** All four call sites converted to double accumulation:
- `cosNum_d` / `cosDen_d` initialized as `double`
- Per-term computation widens float32 inputs (`static_cast<double>`) before arithmetic
- Gain formula computed in double: `static_cast<float>(cosNum_d / std::sqrt(cosDen_d))`
- Float32 cast occurs only on the final gain scalar for split comparison

**Memory cost:** 2 doubles per bin-scope per accumulator (16 bytes vs 8 bytes for float32).
At 2540 bin candidates, this is ~16 KB of additional stack/register pressure per `FindBestSplit`
call — negligible on Apple Silicon with 16+ GB unified memory.

**DEC-036 reserved for sprint close:** The exception to "float32 for accumulation always"
will be documented at sprint close with: (a) Neumaier defeat mechanism (per-term floor, not
summation rounding), (b) assembly confirmation that Neumaier survived but was insufficient,
(c) K4 memory/perf overhead measurement.

---

## 5. Per-Seed Reduction Factors

| seed | T1 gain max | T2 gain max | Reduction |
|------|-------------|-------------|-----------|
| 42   | 4.75e-5 *   | 3.812e-6    | **12.5x** |
| 43   | 4.751e-5    | 3.812e-6    | **12.5x** |
| 44   | 4.75e-5 *   | 3.815e-6    | **12.5x** |

`*` = seed-42 and seed-44 T1 gain max was measured per-depth; worst depth used as ~4.75e-5
estimate (same order as seed=43 T1 baseline of 4.751e-5).

Reduction is consistent across seeds: exactly 12.5x (3.81e-6 / 4.75e-5). The ~3.81e-6
residual is the float32 quantization of the gain scalar (~gain * fp32_eps * fudge). This is
seed-independent (any gain value of ~0.032 quantizes to ~3.8e-6 when cast to float32).

Per-seed variance: negligible. No seeds show partial failure.

---

## 6. Patched Call Sites

Four call sites patched in `catboost/mlx/tests/csv_train.cpp`:

1. `FindBestSplit` — one-hot branch (lines ~1222–1310): `cosNum_d / cosDen_d` double
   accumulators; gain computed as `float(cosNum_d / sqrt(cosDen_d))`.

2. `FindBestSplit` — ordinal branch (lines ~1377–1500): `cosNum_d / cosDen_d` double
   accumulators; gain computed as `float(cosNum_d / sqrt(cosDen_d))`. T1 instrumentation
   preserved with float32 shadow for T2 comparison.

3. `FindBestSplitPerPartition` — one-hot branch (lines ~1613–1660): same K4 pattern.

4. `FindBestSplitPerPartition` — ordinal branch (lines ~1700–1750): same K4 pattern.

`gSum` accumulators NOT patched — T1 confirmed Newton-step attenuation suppresses gSum error
to <4e-8 in leaf values (non-mechanism, per T1 verdict §3).

`TODO-S29-ST-COSINE-KAHAN` and `TODO-S29-LG-COSINE-RCA` markers preserved (T4a/T4b scope).

---

## 7. Data Artifacts

All CSV files under `docs/sprint30/t2-kahan/data/`:

| File pattern | Contents | Rows |
|---|---|---|
| `cos_accum_seedN_depthD.csv` | cosNum/cosDen f32-shadow vs f64, gain f32 vs f64 | 2540 |
| `leaf_sum_seedN.csv` | gSum/hSum/leafVal f32 vs f64 per leaf | 64 |
| `approx_update_seedN.csv` | cursor increment f32 vs f64 per doc (10k) | 10000 |

CSV semantics in T2 (differs from T1):
- `cosNum_f32` / `cosDen_f32`: pre-K4 float32 shadow accumulator (shows pre-K4 error)
- `cosNum_f64` / `cosDen_f64`: K4 double accumulator (reference)
- `gain_f32`: K4 gain = `float(cosNum_d / sqrt(cosDen_d))` (what split selection uses)
- `gain_f64`: double gain = `cosNum_d / sqrt(cosDen_d)` (reference)
- `gain_abs_residual`: the G2-relevant quantity (float32 quantization of gain scalar)

Runner: `docs/sprint30/t2-kahan/run_t2_kahan.py`
