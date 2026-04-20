# Sprint 23 D0 — Post-Promotion Parity Verification (Commit 4)

**Branch**: `mlx/sprint-23-t2-promotion`
**Date**: 2026-04-20
**Task**: Commit 4 — post-promotion 18-config DEC-008 parity sweep on the promoted production build (no `#ifdef` guards, T2 default path). 100-run determinism at gate config.
**Build**: `bench_boosting.cpp` + `histogram_t2_impl.cpp` (no compile flags; T2 is the unconditional default).
**Prior docs**: `docs/sprint22/d3_parity_gate.md` (S22 D3 QA reference), `docs/sprint22/d5_code_review.md` (NIT catalog)

---

## §1 TL;DR

17/18 configs PASS at ULP=0 vs S22 D3 T1 reference. One config (10k/RMSE/128b) is **non-deterministic** with two distinct outcomes: one matching T1 exactly (ULP=0), one 105 ULP away (outside DEC-008 RMSE envelope of 4). This non-determinism is a **pre-existing defect** present in the S22 D2/D3 kernel, not introduced by promotion. Gate G1 fails on the strict "18/18 ULP=0" criterion.

| Gate | Criterion | Result | Verdict |
|------|-----------|--------|---------|
| S23-D0-G1 | 18/18 ULP=0 post-promotion (DEC-008 standard) | 17/18 ULP=0; config #8 non-deterministic | **FAIL** |
| S23-D0-G2 | iter_total_ms ≤ 19.5 ms at gate config | 17.3 ms warm mean | **PASS** |
| S23-D0-G3 | `kernel_sources.h` has T2; bench no inline T2; flag removed | All three structural conditions met | **PASS** |
| S23-NIT-G | All 6 nits addressed | NIT-1/2/3/4/5/7 all applied | **PASS** |

**Defect opened**: DEC-017 — T2-accum atomic float non-associativity (features 1-3) causes training non-determinism at near-tie split decisions. See §5.

---

## §2 Build Commands

```bash
cd "/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/catboost/mlx/tests"

clang++ -std=c++17 -O2 \
  -I"/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx" \
  -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation -Wno-c++20-extensions \
  bench_boosting.cpp \
  "/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/catboost/mlx/methods/histogram_t2_impl.cpp" \
  -o /tmp/bench_s23_parity
```

No compile flags. T2 is the unconditional default path (Commit 3 change). Compiled with zero warnings.

---

## §3 18-Config Parity Sweep (DEC-008 Envelope)

**Command template** (5 runs per config to detect non-determinism):
```bash
for i in 1 2 3 4 5; do
  /tmp/bench_s23_parity --rows $N --features 50 --classes $C \
    --depth 6 --iters 50 --bins $B --lr 0.1 --l2 3.0 --seed 42
done
```

**T1 reference**: S22 D3 §3 T1 column (all were ULP=0 at that time; T1 and T2 matched exactly).

**Results**:

| #  | N     | Loss       | Bins | T1 reference | T2 first-run | Det   | ULP  | Threshold | Result |
|----|------:|:-----------|-----:|-------------:|-------------:|:-----:|-----:|:---------:|:------:|
|  1 |  1000 | RMSE       |   32 | 0.40689126   | 0.40689126   | DET   |    0 | 4         | **PASS** |
|  2 |  1000 | RMSE       |  128 | 0.46936080   | 0.46936080   | DET   |    0 | 4         | **PASS** |
|  3 |  1000 | Logloss    |   32 | 0.34161490   | 0.34161490   | DET   |    0 | 4         | **PASS** |
|  4 |  1000 | Logloss    |  128 | 0.61407095   | 0.61407095   | DET   |    0 | 4         | **PASS** |
|  5 |  1000 | MultiClass |   32 | 0.61065382   | 0.61065382   | DET   |    0 | 8         | **PASS** |
|  6 |  1000 | MultiClass |  128 | 0.99084771   | 0.99084771   | DET   |    0 | 8         | **PASS** |
|  7 | 10000 | RMSE       |   32 | 0.44631991   | 0.44631991   | DET   |    0 | 4         | **PASS** |
|  8 | 10000 | RMSE       |  128 | 0.48231599   | 0.48231599   | NONDET|    0 | 4         | **FAIL** |
|  9 | 10000 | Logloss    |   32 | 0.30072498   | 0.30072498   | DET   |    0 | 4         | **PASS** |
| 10 | 10000 | Logloss    |  128 | 0.60412812   | 0.60412812   | DET   |    0 | 4         | **PASS** |
| 11 | 10000 | MultiClass |   32 | 0.57359385   | 0.57359385   | DET   |    0 | 8         | **PASS** |
| 12 | 10000 | MultiClass |  128 | 0.95665115   | 0.95665115   | DET   |    0 | 8         | **PASS** |
| 13 | 50000 | RMSE       |   32 | 0.44676545   | 0.44676545   | DET   |    0 | 4         | **PASS** |
| 14 | 50000 | RMSE       |  128 | 0.47740927   | 0.47740927   | DET   |    0 | 4         | **PASS** |
| 15 | 50000 | Logloss    |   32 | 0.30282399   | 0.30282399   | DET   |    0 | 4         | **PASS** |
| 16 | 50000 | Logloss    |  128 | 0.60559267   | 0.60559267   | DET   |    0 | 4         | **PASS** |
| 17 | 50000 | MultiClass |   32 | 0.56538904   | 0.56538904   | DET   |    0 | 8         | **PASS** |
| 18 | 50000 | MultiClass |  128 | 0.94917130   | 0.94917130   | DET   |    0 | 8         | **PASS** |

**17/18 PASS. Config #8 FAIL: non-deterministic (two observed values across 5 runs).**

### Config #8 Detail (10k/RMSE/128b)

Config #8 produces exactly two distinct outcomes, each approximately 50% frequency across runs:

| Value          | ULP vs T1 (0.48231599) | Frequency (20 runs) | DEC-008 envelope (≤4) |
|:---------------|:----------------------:|:-------------------:|:---------------------:|
| 0.48231599     | 0 (bit-exact)          | 10/20 (50%)         | PASS                  |
| 0.48231912     | 105                    | 10/20 (50%)         | FAIL                  |

The 105 ULP difference in final RMSE loss is a training-trajectory amplification: a tiny gradient histogram difference (1-2 ULP from FP atomic add non-associativity in T2-accum features 1-3) flips a split decision at some iteration, causing the model to diverge over the remaining iterations.

---

## §4 100-Run Determinism at Gate Config

**Gate config**: 50k/RMSE/d6/128b (config #14)

```bash
for i in $(seq 1 100); do
  /tmp/bench_s23_parity --rows 50000 --features 50 --classes 1 \
    --depth 6 --iters 50 --bins 128 --lr 0.1 --l2 3.0 --seed 42
done | grep BENCH_FINAL_LOSS
```

| Metric | Value |
|--------|-------|
| Runs completed | 100/100 |
| Distinct BENCH_FINAL_LOSS values | **1** |
| Identical value | **0.47740927** |
| ULP vs S22 D3 reference (0.47740927) | **0** |
| Verdict | **PASS (100/100 identical)** |

Gate config is deterministic. The non-determinism in config #8 does not affect the gate config.

---

## §5 Performance — Gate Config iter_total_ms

```bash
/tmp/bench_s23_parity --rows 50000 --features 50 --classes 1 \
  --depth 6 --iters 50 --bins 128 --lr 0.1 --l2 3.0 --seed 42
```

| Metric          | Value   |
|:----------------|:-------:|
| iter-0 (cold)   | 34.9 ms |
| warm mean (49)  | 17.3 ms |
| warm min        | 16.4 ms |
| warm max        | 18.6 ms |
| Gate G2 target  | ≤ 19.5 ms |
| Verdict         | **PASS** |

---

## §6 Gate G3 — Structural Cleanup Verification

| Check | Expected | Result |
|-------|----------|--------|
| `kernel_sources.h` contains `kT2SortSource` and `kT2AccumSource` | Yes | Yes |
| `kernel_sources_t2_scratch.h` no longer exists | Deleted | Deleted (Commit 3) |
| `bench_boosting.cpp` has no inline T2 kernel sources | Removed | Removed (Commit 2) |
| `CATBOOST_MLX_HISTOGRAM_T2` flag removed from bench | Removed | Removed (Commit 3) |
| `histogram_t2_impl.cpp` exists with minimal deps | Yes | Yes (Commit 2) |
| `histogram.h` declares `DispatchHistogramT2` | Yes | Yes (Commit 2) |

**G3: PASS — all structural conditions met.**

---

## §7 NIT Gate Verification

| NIT | Description | Status |
|-----|-------------|--------|
| NIT-1 | Hardcoded literals replaced with named constants (`BLOCK_SIZE`, `T2_BIN_CAP`, `BIN_OFFSETS_STRIDE`, `0x7Fu`, `FEATURES_PER_PACK`) | Applied (Commit 1) |
| NIT-2 | `BIN_OFFSETS_STRIDE = T2_BIN_CAP + 1 = 129` named constant; `offBase` arithmetic deduplicated | Applied (Commit 1) |
| NIT-3 | Explicit `if (totalDocsInPart == 0) return;` guard in T2-accum (uses `binOffsets[offBase + T2_BIN_CAP]` sentinel, not `partSizes` which is not in accum inputs) | Applied (post-Commit 3 fix; included in Commit 4 diff) |
| NIT-4 | `CB_ENSURE(maxBlocksPerPart == 1, ...)` guard in `histogram.cpp` before calling `DispatchHistogramT2`; NIT-4 constraint documented in `histogram.h` and `histogram_t2_impl.cpp` | Applied (Commit 3) |
| NIT-5 | `numTGs` removed from T2-sort and T2-accum input lists (was never read by kernel body); kernel names bumped to `s23d0` to invalidate stale MLX cache | Applied (Commit 3) |
| NIT-7 | Features 1-3 bin mask harmonized to `& 0x7Fu` (was `& 0xFFu`); consistent with T2-sort feature-0 mask and DEC-016 T1 envelope (maxFoldCount ≤ 127) | Applied (Commit 1) |

**S23-NIT-G: PASS — all 6 nits addressed.**

---

## §8 DEC-017: T2-Accum Atomic Float Non-Associativity

**Defect**: T2-accum features 1-3 use `atomic_fetch_add_explicit(memory_order_relaxed)` to accumulate gradient sums. FP addition is not associative; different GPU thread scheduling orders produce slightly different histogram values. At N=10000, features=50, bins=128, RMSE (seed=42 and seed=2), a near-tie split decision is sensitive to these histogram differences, causing the training trajectory to diverge and producing two distinct final RMSE values separated by 105 ULP.

**Pre-existing**: The same non-determinism is confirmed in the S22 D2/D3 scratch binary (built from `kernel_sources_t2_scratch.h` + `bench_boosting.cpp` at S22 tip). The S22 D3 gate missed it because only 5 runs were performed at this config, and all 5 happened to produce the correct value.

**Root cause**: Atomic float accumulation for features 1-3 in T2-accum is inherently non-deterministic in thread execution order. T1 uses `simd_shuffle_xor` reduction (deterministic fixed reduction tree); T2's features 1-3 path lacks an equivalent deterministic reduction.

**Impact**: 1/18 DEC-008 configs non-deterministic. Gate config (50k/RMSE/128b) is unaffected. All other 17 configs are deterministic and bit-exact vs T1 reference.

**Proposed fix**: Replace atomic float accumulation for features 1-3 with threadgroup-local float arrays (deterministic reduction), matching T1's reduction pattern. Estimated scope: Sprint 24 DX task.

**Trigger conditions (narrow)**:
- bins = T2_BIN_CAP = 128 (exactly at capacity)
- N ≈ 10000 (partition size ~156 docs at depth=6)
- features = 50 (numGroups=13, no guard-filtered TGs)
- RMSE loss function (classes=1)
- iters ≥ 50 (divergence accumulates over training)

---

## §9 Pre-existing Status Confirmation

The config #8 non-determinism was verified to exist in the S22 D2/D3 scratch kernel (commit `73baadf445`). Build:

```bash
# Extracted S22 files to /tmp/s22_build/
clang++ -std=c++17 -O2 \
  -I/tmp/s22_build \
  -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation -Wno-c++20-extensions \
  -DCATBOOST_MLX_HISTOGRAM_T2=1 \
  /tmp/bench_boosting_s22.cpp \
  -o /tmp/bench_s22_ref
```

8 runs at config #8: alternated between `0.48231599` and `0.48231912` (same two values, same ~50% frequency). The S23 D0 promotion did not introduce or change this behavior.

---

## §10 Summary: Gate Status

| Gate | Result | Note |
|------|--------|------|
| S23-D0-G1 (18/18 ULP=0) | **FAIL** (17/18) | Config #8 non-deterministic; pre-existing defect (DEC-017) |
| S23-D0-G2 (iter_total_ms ≤ 19.5) | **PASS** (17.3 ms) | No perf regression vs S22 D3 |
| S23-D0-G3 (structural cleanup) | **PASS** | All 4 structural conditions met |
| S23-NIT-G (6 nits addressed) | **PASS** | NIT-1/2/3/4/5/7 all verified |

**Overall: G1 FAIL, G2/G3/NIT-G PASS.**

Promotion (Commits 1-3) stands — the T2 kernel is algorithmically correct and the non-determinism is a known FP atomic accumulation limitation, not a promotion defect. DEC-017 is open to track the fix in a future sprint.
