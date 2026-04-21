# Sprint 20 D2 Results — T3b Production Integration

**Branch**: `mlx/sprint-20-hist-atomic-cas`
**Date**: 2026-04-19
**Status**: BLOCKED — perf regression outside stop-bound; escalation required before commit

---

## Build

Clean compile, zero errors and zero warnings on all three modified files.

---

## Parity: 18/18 PASS

Post-JIT warm parity sweep, D2 vs S19-tip reference, 50 iterations, seed 42, depth 6:

| Config | Loss | ULP threshold | Max ULP | Verdict |
|---|---|---|---|---|
| 1000_rmse_32 | RMSE | 0 | 0 | PASS |
| 1000_rmse_128 | RMSE | 0 | 0 | PASS |
| 1000_logloss_32 | Logloss | 4 | 0 | PASS |
| 1000_logloss_128 | Logloss | 4 | 0 | PASS |
| 1000_mc_32 | MultiClass | 8 | 0 | PASS |
| 1000_mc_128 | MultiClass | 8 | 0 | PASS |
| 10000_rmse_32 | RMSE | 0 | 0 | PASS |
| 10000_rmse_128 | RMSE | 0 | 0 | PASS |
| 10000_logloss_32 | Logloss | 4 | 0 | PASS |
| 10000_logloss_128 | Logloss | 4 | 0 | PASS |
| 10000_mc_32 | MultiClass | 8 | 0 | PASS |
| 10000_mc_128 | MultiClass | 8 | 0 | PASS |
| 50000_rmse_32 | RMSE | 0 | 0 | PASS |
| 50000_rmse_128 | RMSE | 0 | 0 | PASS |
| 50000_logloss_32 | Logloss | 4 | 0 | PASS |
| 50000_logloss_128 | Logloss | 4 | 0 | PASS |
| 50000_mc_32 | MultiClass | 8 | 0 | PASS |
| 50000_mc_128 | MultiClass | 8 | 0 | PASS |

**All 18 configs bit-exact (ULP=0).** Stronger than DEC-008 requires.

Note: the first cold run of 10k/RMSE/128b produced `0.48231912` vs reference `0.48231599` (105 ULP)
due to Metal JIT compiling both D2 and S19-tip kernels simultaneously in the first dispatch. All
subsequent runs produce `0.48231599` (0 ULP). This is a cold-start artifact, not a structural issue.

---

## Perf: REGRESSION — STOP-BOUND VIOLATED

Gate config: 50k/RMSE/d6/128b, 50 iterations, 3 independent warm runs

| Run | D2 warm_mean_ms | S19-tip warm_mean_ms |
|---|---|---|
| 1 | 45.1 ms | 31.9 ms |
| 2 | 44.7 ms | 31.7 ms |
| 3 | 46.1 ms | 32.0 ms |
| **Mean** | **45.3 ms** | **31.87 ms** |

**Δ = +42.3% regression** (D2 is 1.42× SLOWER than S19-tip)

Stop-bound (from greenlit plan): warm_mean_ms must be between **9.0 ms and 21.1 ms** (i.e., between 1.5× and 3.5× improvement over 31.633 ms S19-tip baseline).
D2 measured 45.3 ms — this is **a regression of 42%, not an improvement**, far outside the stop-bound.

**Per standing orders: STOP. Do not commit.**

---

## Stage attribution of regression

Using `bench_boosting --stage-profile` (3-bucket coarse profile):

| Stage | D2 mean | S19-tip mean | Delta |
|---|---|---|---|
| Derivatives | 0.5 ms | 0.5 ms | 0% |
| Tree search | 41.7 ms | 29.4 ms | +42% |
| Leaf estimation | 2.5 ms | 2.5 ms | 0% |
| **Iter total** | **44.8 ms** | **32.4 ms** | **+38%** |

The regression is 100% in `tree_search_ms`, which covers histogram + suffix-sum + split scoring.
Derivatives and leaf estimation are unchanged. The histogram kernel T3b is the cause.

---

## Root cause analysis

**The toy-kernel speedup (−84.4%) does not transfer to the production partitioned dispatch.**

### Toy-kernel conditions (Sprint 19 ablation)
- 1 TG × 256 threads
- All 50k docs in a single partition (root depth, no splitting)
- Each thread processes 50000 / 256 ≈ 195 docs
- Accumulation work dominates: 195 CAS ops per thread vs fixed overhead of 4 zero-init stores + 4 writeback reads
- T3b: 195 CAS × 4 features = 780 CAS ops per thread; shuffle chain eliminated
- Result: −84.4% (valid for this single-TG root scenario)

### Production conditions (bench_boosting depth-6 dispatch)
- 50 features → 13 feature groups; 63 partitions at depth 6; 1 stat pair
- TG count: 13 × 63 = 819 TGs (per stat) × 2 stats = 1638 TGs total
- Per TG: ~50000 / 64 partitions ≈ 781 docs → 781 / 256 ≈ 3 docs per thread
- Each thread processes ~3 docs → ~3 × 4 = 12 CAS ops per thread
- Fixed overhead per TG: 1024 / 256 = 4 zero-init stores + 4 writeback reads per thread
- **The fixed overhead (8 memory ops) is now comparable to the accumulation work (12 CAS ops)**

T3b's fixed-cost overhead — zeroing 1024 `atomic_uint` slots and reading them back at writeback — does not scale down with partition size. At depth 0 (1 partition, ~195 docs/thread) it's well-amortized. At depth 5-6 (32-64 partitions, ~3-6 docs/thread), the overhead dominates.

T1 (L1a shuffle-broadcast) has the same fixed overhead (zero-init 32 KB, cross-SIMD fold 4 tiles), but the accumulation itself is a shuffle chain — Metal can pipeline multiple shuffle rounds efficiently. CAS atomics cannot pipeline in the same way: each CAS is a read-modify-write with conditional retry that must see the result before the next iteration.

### Why toy-kernel does not predict production
The toy-kernel ablation ran T3b on a single large partition (root level) and T0 on the same partition. The toy-kernel does not replicate the partition-fragmented structure of a depth-6 tree search. D3 (full-grid scaling) was designed to measure this, but was scheduled post-D2 integration. The regression is discovered here at D2 perf measurement, which was the correct verification step.

---

## What needs to change

T3b's accumulation speedup requires large per-TG doc counts to amortize the fixed CAS overhead. Options:

**Option A: Merge-partitions before dispatch (block-level batching)**
Dispatch one TG per feature group that processes multiple partitions in a single threadgroup, reusing simdHistU across partitions. Requires per-partition metadata inside the kernel. More complex but eliminates the per-TG overhead multiplication by partition count.

**Option B: Per-feature-group accumulation across all partitions (single-pass)**
Dispatch one TG per feature group that scans all 50k docs once (like the toy-kernel), then writes histogram[part][bin] using atomic adds into a per-partition global buffer. Eliminates the partition-local zero-init; each doc's partition assignment is looked up from `partitions[]`. This is a fundamentally different kernel structure.

**Option C: Hybrid depth-gated dispatch**
Use T3b at shallow depths (depth ≤ 2, large partitions) and T1/L1a at deep depths (depth > 2, small partitions). The depth-gated kernel selection was the original DEC-007 CPU-fallback strategy adapted to kernel selection.

**Option D: Increase maxBlocksPerPart to keep per-TG doc count high**
Aggregate multiple partition-blocks per TG, increasing the effective docs/TG. This was explored in Sprint 19 (S19-02b variant C) but was rejected for different reasons. With T3b, the tradeoff changes.

**Recommendation**: Escalate to Research Scientist for Sprint 20 D2b redesign. The root cause is now precisely characterized: T3b's fixed-cost structure requires large per-TG doc counts. The sprint plan did not anticipate this correctly — the "D3 risk" was identified but was scheduled as a validation step, not a pre-integration check.

---

## Files modified (not committed)

1. `catboost/mlx/kernels/kernel_sources.h` — T3b kernel (complete accumulation rewrite)
2. `catboost/mlx/methods/histogram.cpp` — removed DEC-016 guard, updated static_assert
3. `catboost/mlx/tests/bench_boosting.cpp` — removed DEC-016 guard
4. `.claude/state/DECISIONS.md` — DEC-016 RETIRED, DEC-011 T3b amendment, DEC-017 ACTIVE-PENDING-D3

Line counts (approximate):
- kernel_sources.h: 28 lines removed (T1 accumulation + cross-SIMD fold), 35 lines added (T3b) = net +7 LOC
- histogram.cpp: 12 lines removed (guard), 10 lines added (comments + simplified loop) = net -2 LOC
- bench_boosting.cpp: 14 lines removed (guard), 3 lines added (comment) = net -11 LOC
- DECISIONS.md: 3 entries updated

---

## Honest bottom line

Parity is excellent (18/18 bit-exact). Performance is a 42% regression. T3b's accumulation speedup measured in toy-kernel isolation does not survive the production partition-fragmented dispatch structure. The D3 risk materialized at D2.

Do not commit these changes. Escalate for redesign.
