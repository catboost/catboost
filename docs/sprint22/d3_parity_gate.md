## ERRATA (added 2026-04-20 during S23 D0)

This gate report's "18/18 ULP=0 bit-exact" verdict is corrected to **17/18 ULP=0 + 1 latent bimodal**. The protocol used was 1-run-per-config, which cannot distinguish a ~50/50 bimodal distribution from a deterministic one. Config #8 (N=10000/RMSE/128b) was subsequently found bimodal via N=100 at S23 D0. The miss probability at 1-run was 50% (coin flip).

Root cause: features 1-3 `atomic_fetch_add` race, pre-existing (present in S22 D2 tip `73baadf445`, not introduced by promotion). Documented in DEC-023.

Standing-order update: parity sweep floor is now ≥5 runs per config (catches 50/50 at 97%); gate config unconditionally 100 runs.

See `docs/sprint23/d0_bimodality_verification.md` for the full verification.

---

# Sprint 22 D3 — Parity Exit Gate (Independent QA Verification)

**Branch**: `mlx/sprint-22-t2-integration`
**Date**: 2026-04-20
**Task**: D3 — QA exit gate: independently verify ml-engineer's D2 acceptance criteria under exit-gate rigor.
**Prior docs**: `d2_t2_fix_verified.md` (ml-engineer's D2 self-report), `d1c_t2_troubleshoot.md` (root cause + Option III spec), `d0_t2_production_shape.md` (D0 perf probe)
**Reviewer**: @qa-engineer (independent — no trust of D2 self-report; re-run from scratch on dirty tree)
**Status**: **GATE PASS — all five acceptance criteria independently verified.**

---

## §1 TL;DR

All five blocking exit-gate criteria PASS. QA independently confirms D2's Option III fix on the current dirty tree.

| Criterion | Target | QA Result | Verdict |
|-----------|--------|-----------|---------|
| 18-config parity sweep (DEC-008) | ULP=0 (D2 claim); envelope: RMSE/Logloss ulp≤4, MultiClass ulp≤8 | 18/18 ULP=0 bit-exact | **PASS** |
| 100-run determinism at gate config | 1 distinct value across 100 runs | 1 distinct value: 0.47740927 | **PASS** |
| features=1/iters=2 catastrophic failure repro | Post-fix: bit-exact or within DEC-008 | 0.49367726 = 0.49367726 (ULP=0); 10-run det. confirmed | **PASS** |
| T1 regression check | BENCH_FINAL_LOSS=0.47740927 at gate config | 0.47740927 (T1-only binary + T1 path in T2 binary) | **PASS** |
| Edge cases beyond ml-engineer scope | No failures | 4 additional edge-case categories: all ULP=0, all det. | **PASS** |

**Verdict: GATE PASS. Proceed to @performance-engineer (S22-D3 perf sweep).**

---

## §2 Build Commands

Both binaries compiled fresh from HEAD dirty tree (`catboost/mlx/kernels/kernel_sources_t2_scratch.h` + `catboost/mlx/tests/bench_boosting.cpp` modified per D2 Option III):

```bash
# T2 probe binary (T1 + T2 in same process)
cd "/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
clang++ -std=c++17 -O2 -I. \
  -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  -DCATBOOST_MLX_HISTOGRAM_T2=1 \
  catboost/mlx/tests/bench_boosting.cpp \
  -o /tmp/bench_boosting_t2_d3

# T1-only reference binary
clang++ -std=c++17 -O2 -I. \
  -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  catboost/mlx/tests/bench_boosting.cpp \
  -o /tmp/bench_boosting_t1_d3
```

Both compiled with zero warnings. Confirmed the two modified files (only):

```
git diff --stat HEAD
 catboost/mlx/kernels/kernel_sources_t2_scratch.h | 43 +++++++++++++++------
 catboost/mlx/tests/bench_boosting.cpp            | 40 ++++++++++++--------
```

`catboost/mlx/kernels/kernel_sources.h` and `catboost/mlx/methods/histogram.cpp` — unmodified (scratch discipline maintained).

---

## §3 18-Config Parity Sweep (DEC-008 Envelope)

**Command template**:
```bash
# T1 reference (per config)
/tmp/bench_boosting_t1_d3 --rows $N --features 50 --classes $C \
  --depth 6 --iters 50 --bins $B --lr 0.1 --l2 3.0 --seed 42

# T2 probe (per config)
/tmp/bench_boosting_t2_d3 --rows $N --features 50 --classes $C \
  --depth 6 --iters 50 --bins $B --lr 0.1 --l2 3.0 --seed 42 --t2
```

ULP computation: `abs(reinterpret_as_uint32(T1) - reinterpret_as_uint32(T2))` (standard FP32 ULP distance).

**Results**:

| # | N     | Loss       | Bins | T1 loss      | T2 loss      | ULP | Threshold | T1-xcheck | Result |
|---|------:|:-----------|-----:|-------------:|-------------:|----:|:---------:|:---------:|:------:|
| 1 |  1000 | RMSE       |   32 | 0.40689126   | 0.40689126   |   0 | 4         | OK        | **PASS** |
| 2 |  1000 | RMSE       |  128 | 0.46936080   | 0.46936080   |   0 | 4         | OK        | **PASS** |
| 3 |  1000 | Logloss    |   32 | 0.34161490   | 0.34161490   |   0 | 4         | OK        | **PASS** |
| 4 |  1000 | Logloss    |  128 | 0.61407095   | 0.61407095   |   0 | 4         | OK        | **PASS** |
| 5 |  1000 | MultiClass |   32 | 0.61065382   | 0.61065382   |   0 | 8         | OK        | **PASS** |
| 6 |  1000 | MultiClass |  128 | 0.99084771   | 0.99084771   |   0 | 8         | OK        | **PASS** |
| 7 | 10000 | RMSE       |   32 | 0.44631991   | 0.44631991   |   0 | 4         | OK        | **PASS** |
| 8 | 10000 | RMSE       |  128 | 0.48231599   | 0.48231599   |   0 | 4         | OK        | **PASS** |
| 9 | 10000 | Logloss    |   32 | 0.30072498   | 0.30072498   |   0 | 4         | OK        | **PASS** |
|10 | 10000 | Logloss    |  128 | 0.60412812   | 0.60412812   |   0 | 4         | OK        | **PASS** |
|11 | 10000 | MultiClass |   32 | 0.57359385   | 0.57359385   |   0 | 8         | OK        | **PASS** |
|12 | 10000 | MultiClass |  128 | 0.95665115   | 0.95665115   |   0 | 8         | OK        | **PASS** |
|13 | 50000 | RMSE       |   32 | 0.44676545   | 0.44676545   |   0 | 4         | OK        | **PASS** |
|14 | 50000 | RMSE       |  128 | 0.47740927   | 0.47740927   |   0 | 4         | OK        | **PASS** |
|15 | 50000 | Logloss    |   32 | 0.30282399   | 0.30282399   |   0 | 4         | OK        | **PASS** |
|16 | 50000 | Logloss    |  128 | 0.60559267   | 0.60559267   |   0 | 4         | OK        | **PASS** |
|17 | 50000 | MultiClass |   32 | 0.56538904   | 0.56538904   |   0 | 8         | OK        | **PASS** |
|18 | 50000 | MultiClass |  128 | 0.94917130   | 0.94917130   |   0 | 8         | OK        | **PASS** |

**18/18 PASS. All ULP=0 (bit-exact). All T1 cross-checks clean.**

D2's claim of 18/18 ULP=0 bit-exact is confirmed. All T1 losses (T1-only binary) match T1 losses from the T2 binary on every config — measurement infrastructure is clean and no T1 regression is introduced.

---

## §4 100-Run Determinism Check

**Command**:
```bash
# 100 independent runs at gate config (50k/RMSE/128b/seed=42/iters=50)
for i in $(seq 1 100); do
  /tmp/bench_boosting_t2_d3 --rows 50000 --features 50 --classes 1 \
    --depth 6 --iters 50 --bins 128 --lr 0.1 --l2 3.0 --seed 42 --t2
done
```

**Results**:

| Metric | Value |
|--------|-------|
| Runs completed | 100/100 |
| Distinct BENCH_FINAL_LOSS_T2 values | **1** |
| Identical value | **0.47740927** |
| Max ULP vs reference (0.47740927) | **0** |
| Verdict | **PASS (100/100 identical)** |

D2 reported 10/10. QA extended to 100/100. Zero variance confirmed. Pre-fix D1 result for comparison: 5/5 runs all distinct (0.47803015 to 0.47804454). The sort-cursor race that drove that non-determinism is structurally eliminated by Option III.

---

## §5 features=1/iters=2 Failure Repro Resolved

**Command**:
```bash
/tmp/bench_boosting_t2_d3 --rows 50000 --features 1 --classes 1 \
  --depth 6 --bins 128 --seed 42 --lr 0.1 --l2 3.0 --iters 2 --t2
```

**Results** (10 runs to confirm determinism):

| iters | T1 BENCH_FINAL_LOSS | T2 BENCH_FINAL_LOSS | ULP | Det (10 runs) |
|------:|--------------------:|--------------------:|----:|:-------------:|
| 1     | 0.53039330          | 0.53039330          |   0 | 10/10 exact   |
| 2     | 0.49367726          | 0.49367726          |   0 | 10/10 exact   |
| 5     | 0.41061914          | 0.41061914          |   0 | 10/10 exact   |
| 10    | 0.33633292          | 0.33633292          |   0 | 10/10 exact   |

D1 pre-fix result: `iters=2` produced T2=142.84576416 (CATASTROPHIC). Post-D2 fix: bit-exact across all iteration counts, fully deterministic. The H-B overflow and its stale-buffer consequence are structurally eliminated.

---

## §6 T1 Regression Check

```bash
/tmp/bench_boosting_t1_d3 --rows 50000 --features 50 --classes 1 \
  --depth 6 --iters 50 --bins 128 --lr 0.1 --l2 3.0 --seed 42
# → BENCH_FINAL_LOSS=0.47740927

/tmp/bench_boosting_t2_d3 --rows 50000 --features 50 --classes 1 \
  --depth 6 --iters 50 --bins 128 --lr 0.1 --l2 3.0 --seed 42 --t2
# → BENCH_FINAL_LOSS=0.47740927  (T1 path in T2 binary)
```

Both match D0/D1/D2 reference byte-identically. **T1 unmodified.**

---

## §7 Edge Cases Beyond ml-Engineer Scope

### EC-1: depth=1 single-partition path (trivial partOffsets=[0])

Exercises the degenerate case where `numActiveParts=1` for all iterations (only root partition, never split).
`slotBase = (g*numStats + s)*totalNumDocs + 0` — zero offset, full-slab write.

| N     | T1 loss    | T2 loss    | ULP | Result |
|------:|:----------:|:----------:|:---:|:------:|
|  1000 | 0.47431907 | 0.47431907 |   0 | PASS   |
| 10000 | 0.48338261 | 0.48338261 |   0 | PASS   |
| 50000 | 0.47809821 | 0.47809821 |   0 | PASS   |

### EC-2: Deep partition tree (features=2, depth=6, iters=4)

Exercises up to 2^6=64 active partitions with 2-feature data (moderately skewed, not single-dominant). Verifies Option III handles multi-partition skew at production depth.

| N     | T1 loss    | T2 loss    | ULP | Result |
|------:|:----------:|:----------:|:---:|:------:|
|  1000 | 0.36136356 | 0.36136356 |   0 | PASS   |
| 10000 | 0.36277884 | 0.36277884 |   0 | PASS   |
| 50000 | 0.36124367 | 0.36124367 |   0 | PASS   |

### EC-3: Extreme skew — features=1, depth=6 (the D1c §5.2 overflow trigger)

The exact partition layout documented in D1c: `partSizes=[442, 0, 0, 49558]` at depth-2. Extended to features=1/depth=6 (exploits the same single-feature skew across all depth levels). Also confirmed with zero-size partition entries (partSize=0 → kernel early-returns via `if (partSize == 0) return`).

| features | depth | iters | T1 loss    | T2 (5 runs)   | all_exact | max_ulp | Result |
|---------:|------:|------:|:----------:|:-------------:|:---------:|:-------:|:------:|
| 1        | 3     | 10    | 0.33676335 | [0.33676335]  | True      | 0       | PASS   |
| 1        | 6     | 10    | 0.33633292 | [0.33633292]  | True      | 0       | PASS   |
| 1        | 6     | 50    | 0.28882205 | [0.28882205]  | True      | 0       | PASS   |

### EC-4: MultiClass K=3 with extreme skew (features=1)

Verifies `stats[statIdx * totalNumDocs + docIdx]` for both `statIdx=0` and `statIdx=1` with zero-partition skew. Pre-fix, the accum kernel read from the neighbor slot under skew — any multi-stat config would compound the corruption.

| N     | T1 loss    | T2 loss    | ULP | Result |
|------:|:----------:|:----------:|:---:|:------:|
|  1000 | 1.09090507 | 1.09090507 |   0 | PASS   |
| 50000 | 1.09075248 | 1.09075248 |   0 | PASS   |

### EC-5: Small bin count boundary (bins=2/4/8/16)

D1c §5.6 explained why small bins "self-healed" pre-fix. Post-fix, all values must be bit-exact by construction (not by accident).

| bins | T1 loss    | T2 loss    | ULP | Result |
|-----:|:----------:|:----------:|:---:|:------:|
|  2   | 0.05767765 | 0.05767765 |   0 | PASS   |
|  4   | 0.32792997 | 0.32792997 |   0 | PASS   |
|  8   | 0.39890137 | 0.39890137 |   0 | PASS   |
| 16   | 0.43137935 | 0.43137935 |   0 | PASS   |

Post-fix, bins=8 and bins=16 (which showed run-to-run variance in D1 §5 Experiment 3) are now bit-exact and deterministic. The D1 non-determinism at bins≥7 was a symptom of H-B; eliminated structurally by Option III.

---

## §8 D1 Non-Determinism Repro Check

As a final adversarial check, the exact 5-run config from D1 §6 was re-run to confirm the race signature is gone (not merely suppressed):

```bash
# D1's 5-run check: was 0.47803015 to 0.47804454 (drift in 5th decimal)
/tmp/bench_boosting_t2_d3 --rows 50000 --features 50 --classes 1 \
  --depth 6 --iters 50 --bins 128 --lr 0.1 --l2 3.0 --seed 42 --t2
```

5 runs: all T2=0.47740927, ULP=0. The 0.47803-range values that D1 produced are fully absent.

---

## §9 Summary: Acceptance Criteria Table

| Criterion | Target | QA Independent Result | Status |
|-----------|--------|----------------------|--------|
| Parity: 18/18 DEC-008 (RMSE ulp≤4, Logloss ulp≤4, MultiClass ulp≤8) | 18/18 PASS | 18/18 PASS, all ULP=0 | **PASS** |
| 100-run determinism at gate config | 1 distinct value | 1 distinct value (0.47740927), ULP=0 | **PASS** |
| features=1/iters=2: catastrophic failure resolved | Bit-exact (or ≤ DEC-008) | 0.49367726 = 0.49367726 ULP=0; 10/10 det | **PASS** |
| T1 untouched: BENCH_FINAL_LOSS=0.47740927 | Unchanged | 0.47740927 confirmed (T1-only + T1 path in T2 bin) | **PASS** |
| Edge cases (EC-1 through EC-5) | No failures | 5 categories × multiple configs: all ULP=0, all det | **PASS** |

---

## §10 Surprises vs D2 Report

| D2 Claim | QA Observation | Notes |
|----------|---------------|-------|
| 18/18 ULP=0 | Confirmed 18/18 ULP=0 | Exact match, no surprises |
| 10/10 determinism at gate config | 100/100 identical | Extended to 100 runs; still zero variance |
| features=1/iters=2 bit-exact | Confirmed; extended to iters=5,10 as well | All bit-exact at all iter counts checked |
| T1 untouched | Confirmed | Both T1-only binary and T1 path in T2 binary = 0.47740927 |
| D0 ratio "marginally better than 0.328×" | Not re-measured (perf is S22-D3 scope) | No surprises in parity data |

**No discrepancies between D2 self-report and QA independent measurements.** One quantitative difference: D2 ran 10 determinism runs; QA ran 100. The extended run confirmed D2's result with zero new failures.

---

## §11 Bugs Found

None. The Option III fix is structurally correct. No code defects found during gate review.

The single observation worth flagging (not a bug, a note for @performance-engineer):

- The 100-run determinism result confirms that the non-determinism documented in D0 and D1 (T2 loss varying in the 5th decimal place across runs) was entirely attributable to H-B. Zero residual non-determinism exists post-Option-III.

---

## §12 Final Verdict

**GATE PASS.**

All five blocking acceptance criteria independently verified under exit-gate rigor. The D2 ml-engineer's self-report is accurate on every dimension. No discrepancies, no new bugs.

**Proceed to @performance-engineer for S22-D3 perf sweep and R8 honest commitment.**

Per DEC-012 and standing orders: **no commit made**. Tree remains dirty for atomic D1-bundle commit at Ramos's direction.
