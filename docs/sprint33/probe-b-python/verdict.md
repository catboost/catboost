# PROBE-B Verdict — Python (nanobind) Production Path

**Date**: 2026-04-24
**Branch**: `mlx/sprint-33-iter2-scaffold`
**Tip at run**: `9dfd62ccc3eef0dc8fb022ff7058b912a3df2afe`
**Kernel md5**: `9edaef45b99b9db3e2717da93800e76f` (`catboost/mlx/kernels/kernel_sources.h`, byte-identical to S33 baseline; unchanged)
**Anchor**: N=50_000, F=20, seed=42, depth=6, bins=127, lr=0.03,
SymmetricTree+Cosine, RMSE, bootstrap=No, random_strength=0.0, has_time=true.
Dataset generator matches `docs/sprint33/l4-fix/run_phase1.py` exactly:
`X ~ N(0,1); y = 0.5·X[:,0] + 0.3·X[:,1] + 0.1·N(0,1)`.

---

## Class call: **PRODUCTION-BROKEN**

The nanobind Python path uses **the exact same** `csv_train.cpp::QuantizeFeatures`
(static 127-border grid for every feature) as the csv_train CLI harness, and
exhibits **the same 52.6% iter=50 ST+Cosine drift** as the L4-fix verdict.

The L4 verdict's Option 3 — _"the production path is the nanobind Python API
which uses CatBoost's own `Pool` + `QuantizedPool` for data preparation, which
handles quantization correctly"_ — is **structurally false**. There is no such
separate quantization route in the codebase.

---

## Data Flow Trace

The MLX Python entry point is `catboost_mlx.CatBoostMLXRegressor`
(`python/catboost_mlx/__init__.py:21`). There is **no** `task_type='MLX'` flag
on the stock `catboost.CatBoostRegressor` — the MLX backend is a separate
package with its own class hierarchy, not a backend plugin.

When a user calls `.fit(X, y)`:

1. **Python — `core.py:1090 _fit_nanobind`**
   - Encodes string categoricals to integer indices (lines 1142-1160).
   - Sorts by `group_ids` if present (BUG-007 contract, lines 1172-1178).
   - Calls `_nb_core.train(features=X_f32, targets=y_f32, ..., config=cfg)`
     at line 1215 — passing **raw float32** feature values, not quantized bins.

2. **nanobind shim — `python/catboost_mlx/_core/bindings.cpp:79-138`**
   - Zero-copy wraps the numpy float32 buffer.
   - Releases the GIL (`nb::gil_scoped_release`, line 128).
   - Calls `TrainFromArrays(feat_ptr, tgt_ptr, ...)` at line 129.

3. **Public API shim — `catboost/mlx/train_api.cpp:14`**
   - Includes `csv_train.cpp` directly via `#include "catboost/mlx/tests/csv_train.cpp"`
     (with `CATBOOST_MLX_NO_MAIN` to suppress its `main`).
   - `TrainFromArrays` (lines 217-492) does:
     - `BuildDatasetFromArrays` (line 236) — copies row-major floats into
       `TDataset::Features` column-major **without** any quantization
       (lines 130-137: just stores the raw float, sets `HasNaN`).
     - `auto quant = QuantizeFeatures(ds, config.MaxBins);` at **line 268**.

4. **The quantization call — `csv_train.cpp:1177 QuantizeFeatures`**
   - Builds a static GreedyLogSum border grid for **every** numeric feature
     (line 1235: `GreedyLogSumBestSplit(allVals, maxBordersCount)`),
     regardless of target correlation.
   - This is **the same function** the csv_train CLI calls at line 5592
     (`auto quant = QuantizeFeatures(ds, config.MaxBins);`).

There is no `Pool::Quantize`, no `QuantizedPool` materialisation, and no call
into CatBoost's CPU `NCB::TQuantizedFeaturesInfo` from the Python path. The
phrase "CatBoost's own quantization" does not correspond to anything that
actually runs when `_fit_nanobind` is invoked.

**Code excerpts**:

```cpp
// catboost/mlx/train_api.cpp:13-16
#define CATBOOST_MLX_NO_MAIN
#include "catboost/mlx/tests/csv_train.cpp"

#include "catboost/mlx/train_api.h"
```

```cpp
// catboost/mlx/train_api.cpp:235-269
// Build dataset from flat arrays
TDataset ds = BuildDatasetFromArrays(
    features, targets, featureNames, isCategorical,
    weights, groupIds, catHashMaps, numDocs, numFeatures);
...
// Quantize and pack features
auto quant = QuantizeFeatures(ds, config.MaxBins);
auto packed = PackFeatures(quant, ds);
```

```cpp
// catboost/mlx/tests/csv_train.cpp:1177-1240 (excerpt)
TQuantization QuantizeFeatures(const TDataset& ds, ui32 maxBins) {
    ...
    // Numeric: GreedyLogSum border selection (CatBoost default EBorderSelectionType).
    ...
    std::vector<float> borders = GreedyLogSumBestSplit(allVals, maxBordersCount);
```

This is identical to the static-grid quantizer that L4 identified as the
mechanism behind the 52.6% drift.

---

## Build Procedure

The Python ST+Cosine guard at `core.py:638-647` and the C++ ST+Cosine guard
at `train_api.cpp:41-54` both raise `ValueError` for the gate config, so the
production binary cannot exercise this path. To **measure** the underlying
drift (i.e., what would happen if #93 ST-REMOVE landed today), both guards
were bypassed:

1. **Stale build cache cleared** (per `project_nanobind_build_cache.md`):
   ```
   rm -rf python/build/temp.macosx-11.1-arm64-cpython-313/catboost_mlx._core/
   ```

2. **C++ guard wrapped in a probe-only macro** by editing `catboost/mlx/train_api.cpp`
   to surround both the LG-GUARD and ST-GUARD `if`/`throw` blocks with
   `#ifndef CATBOOST_MLX_PROBE_BYPASS` ... `#endif`. The edit was **reverted
   via `mv catboost/mlx/train_api.cpp.bak catboost/mlx/train_api.cpp` before
   the verdict commit.** `git diff catboost/mlx/train_api.cpp` shows no
   change at commit time.

3. **Built with the bypass macro**:
   ```
   cd python && CXXFLAGS="-DCATBOOST_MLX_PROBE_BYPASS" \
       python3 -m pip install --no-deps --no-build-isolation -e .
   ```
   Bypass `_core.so` md5 (during measurement run): `597a6966cdc581c79faa79042adb068e`,
   mtime 2026-04-24 22:10:46.

4. **Python guard monkey-patched** in the driver (no source edit):
   `_validate_params` is wrapped to temporarily flip `score_function` to `'L2'`
   for the validation step only.

5. **After measurement**, the C++ source was reverted and `_core.so` rebuilt
   clean. Final tip-of-tree `_core.so` md5: `bb3c970a8b3abaf368199ccaa902d659`.
   Smoke test confirms ST+Cosine raises `ValueError` as expected.

Kernel sources `.metal` and `kernel_sources.h` were never touched (md5
`9edaef45b99b9db3e2717da93800e76f` preserved).

---

## Drift Measurement

Driver: `docs/sprint33/probe-b-python/run_probe_b.py`. Raw output:
`docs/sprint33/probe-b-python/data/run_log.txt`. Structured results:
`docs/sprint33/probe-b-python/data/probe_b_results.json`.

| iters | CPU RMSE   | MLX (Python path) RMSE | Ratio  | Drift  | L4 csv_train ratio |
|------:|-----------:|-----------------------:|-------:|-------:|-------------------:|
|    50 | 0.193679   | 0.295626               | 1.5264 | +52.64% | 1.527 (+52.6%) |
|   100 | 0.114801   | 0.188376               | 1.6409 | +64.09% | 1.641 |

The Python-path drift matches the L4-fix csv_train CLI drift to four
significant figures at iter=50 and iter=100, consistent with both paths
calling the same `QuantizeFeatures` and the same `RunTraining` over the same
seed. (Tiny differences below the fourth digit are expected from CSV float
round-trip vs nanobind zero-copy float32, and from deterministic-but-
slightly-different reduction orders inside MLX evaluation between the two
binary builds.)

CPU baseline values match the L4 verdict's reference table exactly:
- L4 verdict §Phase 2: CPU 0.1937 / 0.1148 vs probe-B: 0.193679 / 0.114801.
- L4 verdict §Phase 2: csv_train 0.2956 / 0.1884 vs probe-B Python path:
  0.295626 / 0.188376.

---

## Implication for L4 Closure

**Option 3 is invalid.** The L4 recommendation paragraph appeals to a
"production path" (the nanobind Python API) that allegedly uses a separate,
correct quantization. No such separate path exists. The Python API and the
csv_train CLI both compile around the same `QuantizeFeatures` static-border
grid in `csv_train.cpp`, and both produce the same 52.6% iter=50 drift on the
gate config.

Concretely, this means:

- **#93 (S31-T4a, ST-REMOVE) cannot proceed** on the basis that "the Python
  path is fine." Removing the Python+C++ guards today would expose users to
  the same 52.6% drift csv_train shows.
- **#94 (S31-T4b, LG-REMOVE) is also blocked** for the same reason: LG+Cosine
  shares the joint-denominator compounding mechanism (S29 DEC-034 outcome A)
  AND the static-grid quantization mechanism (DEC-041); the Python path
  cannot rescue it.
- **DEC-036 cannot be PARTIAL-CLOSED on the Option 3 rationale.** Either
  Option 1 (port dynamic border accumulation) or Option 2 (target-correlation
  feature filter) must be implemented before the guards can be lifted, or a
  scope-narrowed gate (e.g. dataset shape with no pure-noise features) must
  be defined as the qualified-deployment envelope.

The mechanism explanation in L4 verdict §Phase 2 (static vs dynamic borders,
noise-feature waste) remains correct. Only the recommendation paragraph
(§ "DEC-041 Opened" → Option 3 framing) needs revision.

---

## Caveats

- The probe ran a single seed (42) at the gate anchor. The drift ratio is
  expected to be insensitive to seed at this N (consistent with S30 V6 N=500
  confirmer showing b≈0 across 100× N range), but a multi-seed sweep would
  formally verify the result.
- The Python predict path (`_predict_utils`) was used end-to-end and
  reproduces the L4 csv_train RMSE table to four significant figures. This
  confirms predict-path equivalence on the same dataset; predict-path drift
  (if any) is at most O(1e-5) and not the source of the 52% number.
- The test dataset has 18/20 pure-noise features. Datasets with fewer noise
  features (or with target-correlated features only) may exhibit smaller
  drift; the probe does not characterise the drift envelope across dataset
  shapes — only confirms that the gate config behaves identically to L4.

---

## Artifacts

- `data/run_log.txt` — full stdout of the probe run.
- `data/probe_b_results.json` — structured CPU/MLX RMSE table + so md5.
- `run_probe_b.py` — driver script (reproducible).
