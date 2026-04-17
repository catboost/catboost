# Sprint 16 Baseline Profile Results

Captured 2026-04-17 with `csv_train_profiled` (built with `-DCATBOOST_MLX_STAGE_PROFILE`). Raw JSONs live under `.cache/profiling/sprint16/baseline_<N>_<loss>_d6_<bins>bins.json` (gitignored) and are summarized in `baseline_summary.json` (also gitignored).

All runs: depth 6, lr 0.1, 50 iterations, 50 features. Timings are mean ms per iteration across all recorded iterations.

## Attribution table (all 18 combos)

|     N | loss       | bins | total_ms | hist_ms | hist% | acct% |
|------:|:-----------|-----:|---------:|--------:|------:|------:|
|  1000 | logloss    |   32 |   289.74 |  284.75 | 98.3% | 99.4% |
|  1000 | logloss    |  128 |   284.94 |  278.46 | 97.7% | 99.4% |
| 10000 | logloss    |   32 |   307.12 |  302.10 | 98.4% | 99.5% |
| 10000 | logloss    |  128 |   310.51 |  303.65 | 97.8% | 99.4% |
| 50000 | logloss    |   32 |   463.06 |  457.31 | 98.8% | 99.6% |
| 50000 | logloss    |  128 |   474.80 |  467.71 | 98.5% | 99.6% |
|  1000 | multiclass |   32 |   553.46 |  547.75 | 99.0% | 99.7% |
|  1000 | multiclass |  128 |   547.96 |  540.04 | 98.6% | 99.7% |
| 10000 | multiclass |   32 |   594.88 |  588.53 | 98.9% | 99.7% |
| 10000 | multiclass |  128 |   596.39 |  588.12 | 98.6% | 99.7% |
| 50000 | multiclass |   32 |   905.42 |  898.63 | 99.2% | 99.8% |
| 50000 | multiclass |  128 |   923.62 |  913.84 | 98.9% | 99.8% |
|  1000 | rmse       |   32 |   295.70 |  290.23 | 98.2% | 99.4% |
|  1000 | rmse       |  128 |   287.78 |  280.68 | 97.5% | 99.4% |
| 10000 | rmse       |   32 |   313.71 |  308.73 | 98.4% | 99.5% |
| 10000 | rmse       |  128 |   314.65 |  308.20 | 97.9% | 99.5% |
| 50000 | rmse       |   32 |   469.25 |  463.98 | 98.9% | 99.6% |
| 50000 | rmse       |  128 |   481.21 |  473.66 | 98.4% | 99.6% |

## Headline findings

1. **Histogram phase is 97.5–99.2% of iter time in every configuration.** Nothing else comes close.
2. **Bin count barely matters.** 32 vs 128 bins differ by <2% at fixed (N, loss). Whatever the kernel is doing, binning isn't the constraint.
3. **Scale scales sub-linearly.** 1k → 50k (50×) adds only 1.6× iter time (290 → 470 ms for RMSE). Compute per doc is cheap; per-call fixed cost dominates.
4. **MultiClass ~2× heavier than binary.** approxDim=3 triggers three `DispatchHistogram()` calls per depth instead of one. Matches the `97.7% of iter = histogram` finding: tripling histogram calls roughly doubles iter time (the base loop cost doesn't triple).

## Other stages — attribution (RMSE 10k 128 bins, representative)

| stage              | mean ms |    % |
|:-------------------|--------:|-----:|
| derivatives_ms     |    0.29 | 0.1% |
| init_partitions_ms |    0.00 | 0.0% |
| partition_layout_ms|    1.71 | 0.5% |
| histogram_ms       |  310.99 |97.7% |
| suffix_scoring_ms  |    2.29 | 0.7% |
| leaf_sums_ms       |    0.26 | 0.1% |
| leaf_values_ms     |    0.24 | 0.1% |
| tree_apply_ms      |    0.24 | 0.1% |
| loss_eval_ms       |    0.26 | 0.1% |
| cpu_readback_ms    |    0.13 | 0.0% |
| **sum_of_stages**  |  316.40 |99.4% |
| iter_total_ms      |  318.34 |100.0%|

## Hypotheses falsified by this data

- **B5 (per-depth CPU readback syncs)** — `cpu_readback_ms` is 0.13 ms (0.04%). Not a bottleneck.
- **B2 (`maxBlocksPerPart = 1`)** — the hardcoded value documented in `bottlenecks.md` is no longer present in the production path. `csv_train.cpp:891-894` computes `maxBlocksPerPart = clamp(ceil(avgDocsPerPart/4096), 1, 8)`. The underlying kernel is still dominant, so Sprint 17 must revisit the kernel itself — not the dispatch parameter.

## Per-depth shape (RMSE 10k 128 bins, iter 0)

| depth | partitions | layout_ms | hist_ms | split_ms | readback_ms |
|------:|-----------:|----------:|--------:|---------:|------------:|
|     0 |          1 |      0.58 |   45.48 |     0.08 |        0.04 |
|     1 |          2 |      0.27 |   23.55 |     0.11 |        0.02 |
|     2 |          4 |      0.22 |   23.64 |     0.12 |        0.03 |
|     3 |          8 |      0.23 |   38.55 |     0.19 |        0.08 |
|     4 |         16 |      0.17 |   74.16 |     0.39 |        0.11 |
|     5 |         32 |      0.37 |  114.42 |     0.77 |        0.29 |
| **total** |       |      1.84 |  319.80 |     1.66 |        0.57 |

The histogram phase grows ~2.5× from depth 0 (1 partition) to depth 5 (32 partitions), and the cost is neither cpu-readback nor split scoring — it is the kernel dispatch + eval chain inside `DispatchHistogram()` (`csv_train.cpp:869-955`). Stage 4 spans Phase 1 (per-dim `DispatchHistogram` call with its own `mx::eval(histogram)` at csv_train.cpp:953) and Phase 2 (`mx::eval(toEval)` for grad/hess scatters). Subtracting the Phase 2 eval (~1.65 ms across all depths) leaves ~309 ms in Phase 1.

## Sprint 17 implication

The originally planned target (B2, `maxBlocksPerPart=1`) was already fixed. The next target is the histogram Metal kernel itself: reduce the per-call cost and/or the per-depth call count. Candidate levers (pre-capture, needs MST evidence to rank):

- **Per-feature-group dispatch fusion.** `DispatchHistogram` already calls the kernel once per invocation with all groups embedded (`numGroups` parameter) — confirmed in `csv_train.cpp:934-949`. Per-group loop described in `bottlenecks.md` B4 applies to the *library path* (`histogram.cpp:112-155`), which is dead code for csv_train. Lever re-prioritized to only the library path.
- **Per-dim dispatch fusion for multiclass.** Phase 1 loop at `csv_train.cpp:3185-3204` still serializes `approxDim` histograms. This is B3 and is real — MultiClass iter time ≈ 2× binary confirms the cost.
- **Kernel-level work reduction.** Each `mx::fast::metal_kernel()` call in `DispatchHistogram` may be incurring per-call kernel-prep overhead (cache lookup, arg encoding). Needs xctrace Metal System Trace to confirm.
- **`mx::eval(histogram)` at `csv_train.cpp:953`.** Removing this would defer the sync to Phase 2's `mx::eval(toEval)`. Would not change the kernel's own runtime but might allow better GPU scheduling across depths.
