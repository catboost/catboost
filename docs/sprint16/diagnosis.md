# Sprint 16 Performance Diagnosis Report

**Status:** In progress — @performance-engineer, @research-scientist  
**Branch:** `mlx/sprint-16-perf-diagnosis`  
**Profiling output:** `.cache/profiling/sprint16/`

This document is a skeleton. Each section has instructions for what to fill in. When complete, this is the primary input for the Sprint 17 fix plan.

---

## 1. Per-Stage Wall-Clock Breakdown

**Instructions for @performance-engineer (S16-01, S16-02):**

Run `bench_boosting --stage-profile` and `bench_mlx_vs_cpu.py --stage-profile` to produce per-stage JSON. Collapse results into the tables below. Run each configuration three times; report median. Use 100 iterations, depth 6 unless noted.

Stages:
1. `ComputeDerivatives` — gradient and hessian computation (target function)
2. `InitPartitions` — reset leaf assignments
3. `ComputePartitionLayout` — GPU bucket sort
4. `ComputeHistograms` — Metal histogram kernel dispatch (all groups)
5. `SuffixSum` — Metal suffix-sum transform
6. `ScoreSplits` — Metal scoring + per-block argmax
7. `BestSplitReadback` — CPU reduction over block candidates (EvalNow)
8. `UpdatePartitions` — bitwise leaf assignment update (EvalNow)
9. `LeafEstimation` — leaf sums + Newton step + tree apply

### Table 1 — Depthwise, RMSE, 32 bins (ms per iteration, median of 3 runs)

| Stage | N=10k d=6 | N=10k d=10 | N=100k d=6 | N=100k d=10 | N=1M d=6 |
|-------|-----------|------------|------------|-------------|----------|
| ComputeDerivatives | | | | | |
| InitPartitions | | | | | |
| ComputePartitionLayout | | | | | |
| ComputeHistograms | | | | | |
| SuffixSum | | | | | |
| ScoreSplits | | | | | |
| BestSplitReadback | | | | | |
| UpdatePartitions | | | | | |
| LeafEstimation | | | | | |
| **Total** | | | | | |

### Table 2 — Depthwise, RMSE, 128 bins (ms per iteration, median of 3 runs)

| Stage | N=10k d=6 | N=10k d=10 | N=100k d=6 | N=100k d=10 | N=1M d=6 |
|-------|-----------|------------|------------|-------------|----------|
| ComputeDerivatives | | | | | |
| InitPartitions | | | | | |
| ComputePartitionLayout | | | | | |
| ComputeHistograms | | | | | |
| SuffixSum | | | | | |
| ScoreSplits | | | | | |
| BestSplitReadback | | | | | |
| UpdatePartitions | | | | | |
| LeafEstimation | | | | | |
| **Total** | | | | | |

### Table 3 — Multiclass (K=10), Lossguide max_leaves=31, 128 bins (ms per iteration)

| Stage | N=10k | N=100k | N=1M |
|-------|-------|--------|------|
| ComputeDerivatives | | | |
| InitPartitions | | | |
| ComputePartitionLayout | | | |
| ComputeHistograms | | | |
| SuffixSum | | | |
| ScoreSplits | | | |
| BestSplitReadback | | | |
| UpdatePartitions | | | |
| LeafEstimation | | | |
| **Total** | | | |

---

## 2. Metal System Trace Analysis

**Instructions for @research-scientist (S16-03):**

Capture a Metal System Trace from `xcode instruments -t "Metal System Trace"` or `xcrun xctrace` while running 50 warm iterations of the 100k RMSE depth-6 benchmark. Save the `.trace` file to `.cache/profiling/sprint16/baseline_100k_rmse_d6.trace`.

Fill in below:

### GPU timeline observations

<!-- What fraction of GPU time is active vs idle? Are there visible gaps between command buffer submissions? -->

_To be filled._

### Kernel occupancy

<!-- Report per-kernel theoretical occupancy from the trace. Compare against the 70% target. -->

| Kernel | Observed occupancy | Target | Gap |
|--------|-------------------|--------|-----|
| `histogram_one_byte_features` | | ≥ 70% | |
| `suffix_sum_histogram` | | ≥ 70% | |
| `score_splits_lookup` | | ≥ 70% | |
| `tree_apply` | | ≥ 70% | |

### CPU-GPU sync gaps

<!-- How many command buffer commits appear per iteration? How long are the CPU stall gaps? -->

_To be filled._

### Unexpected hotspots

<!-- Any surprising entries in the encoder timeline — e.g., unexpectedly large blit encoders, excessive pipeline state creation? -->

_To be filled._

---

## 3. Sync-Point Inventory

**Instructions for @performance-engineer (S16-04):**

See [`docs/sprint16/sync_inventory.md`](sync_inventory.md) when complete (to be created as part of S16-04). Summarize the findings here once available.

### Sync counts per iteration (to be filled from S16-04)

| Location | EvalNow calls / iteration | Estimated cost (ms, N=100k) |
|----------|--------------------------|----------------------------|
| `pointwise_target.h` (pre-fix) | 18 | |
| `score_calcer.cpp` best-split readback | 1–2 per depth level | |
| `structure_searcher.cpp` partition update | 1 per depth level | |
| `structure_searcher.cpp` depthwise data readback (line 252, 550) | up to 2 per depth level | |
| `structure_searcher.cpp` lossguide doc walker (line 645) | 1 per leaf expansion | |
| `tree_applier.cpp` lossguide inference path (line 251) | 1 per inference call | |

_Per-iteration totals and costs: to be filled._

---

## 4. Top-3 Bottleneck Ranking

**Instructions for @performance-engineer:**

After completing Tables 1–3 and the sync inventory, rank the top three bottlenecks by their contribution to total wall-clock time at N=100k, 128 bins, depth 6. Use the per-stage data to attribute cost.

See [`docs/sprint16/bottlenecks.md`](bottlenecks.md) for the full candidate list (six pre-identified bottlenecks). The ranking here should be evidence-based, not assumed.

### Ranked bottlenecks (to be filled)

| Rank | Bottleneck | Stage | Estimated % of iteration time | Source |
|------|-----------|-------|------------------------------|--------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

_Rationale: to be filled._

---

## 5. Sprint 17 Recommendation

**Instructions for @performance-engineer and @research-scientist:**

Based on the ranked bottleneck list and MST observations, recommend the single highest-impact fix for Sprint 17. The recommendation should:
- Name the specific code change (file, function, proposed approach)
- Estimate the expected speedup with a rationale (not a guess)
- Flag any correctness or numerical risks
- Identify what measurements from this sprint inform the estimate

_To be filled after diagnosis data is complete._

---

## Appendix: Raw profiling data

Raw JSON from `--stage-profile` runs is at:
```
.cache/profiling/sprint16/
  baseline_10k_rmse_d6_32bins.json
  baseline_10k_rmse_d6_128bins.json
  baseline_100k_rmse_d6_32bins.json
  baseline_100k_rmse_d6_128bins.json
  baseline_1m_rmse_d6_32bins.json
  baseline_1m_rmse_d6_128bins.json
  baseline_100k_multiclass_k10_lossguide_128bins.json
  baseline_100k_rmse_d6.trace          (Metal System Trace)
```

Files will be present after S16-02 and S16-03 complete.
