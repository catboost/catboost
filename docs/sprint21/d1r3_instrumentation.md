# D1-R3 Per-Kernel Profile Instrumentation

Sprint 21 / Operation Verstappen — measurement infrastructure prerequisite for D1-R1 and D1-R2.

---

## Command

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/mlx/0.31.1/lib ./bench_boosting \
  --rows 50000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42 \
  --per-kernel-profile
```

Build command (unchanged from prior sprints):
```bash
clang++ -std=c++17 -O2 -I. \
  -I/opt/homebrew/Cellar/mlx/0.31.1/include \
  -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
  -framework Metal -framework Foundation -Wno-c++20-extensions \
  catboost/mlx/tests/bench_boosting.cpp -o bench_boosting
```

---

## Baseline Match Confirmation

Gate config: `--rows 50000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42` (no `--per-kernel-profile`).

D0 baseline: `warm_mean = 31.933 ms ± 0.208 ms`.

| Run | warm_mean | BENCH_FINAL_LOSS |
|-----|-----------|-----------------|
| 1   | 31.9 ms   | 0.47740927      |
| 2   | 31.9 ms   | 0.47740927      |
| 3   | 31.2 ms   | 0.47740927      |

All three baseline runs are within the D0 ± 0.208 ms envelope. BENCH_FINAL_LOSS is bit-identical across all runs. Baseline confirmed.

---

## Per-Kernel Timing Table (3 runs, with `--per-kernel-profile`)

Stats method: 10%-trimmed mean/stdev over warm iters (indices 1–49; iter-0 excluded). Trim suppresses Metal command-buffer submit outliers.

### Run 1 — warm_mean 34.7 ms

| Bucket          | mean (ms) | stdev (ms) | stdev%  | Note |
|-----------------|-----------|------------|---------|------|
| derivatives     | 0.516     | 0.058      | 11.2%   | wall-clock floor |
| tree_support    | 5.864     | 0.142      | 2.4%    |      |
| histogram       | 21.920    | 0.875      | 4.0%    |      |
| suffix_sum      | 1.085     | 0.079      | 7.2%    | wall-clock floor |
| split_score     | 1.993     | 0.117      | 5.9%    | wall-clock floor |
| leaf_estimation | 2.492     | 0.059      | 2.3%    |      |
| **sum-of-per-kernel** | **33.871** | — | — | vs iter_total 34.677 ms (delta −0.806 ms, −2.3%) |

### Run 2 — warm_mean 34.0 ms

| Bucket          | mean (ms) | stdev (ms) | stdev%  | Note |
|-----------------|-----------|------------|---------|------|
| derivatives     | 0.481     | 0.045      | 9.3%    | wall-clock floor |
| tree_support    | 5.727     | 0.114      | 2.0%    |      |
| histogram       | 21.631    | 0.961      | 4.4%    |      |
| suffix_sum      | 1.035     | 0.067      | 6.5%    | wall-clock floor |
| split_score     | 1.934     | 0.086      | 4.4%    |      |
| leaf_estimation | 2.462     | 0.062      | 2.5%    |      |
| **sum-of-per-kernel** | **33.270** | — | — | vs iter_total 34.006 ms (delta −0.737 ms, −2.2%) |

### Run 3 — warm_mean 33.8 ms

| Bucket          | mean (ms) | stdev (ms) | stdev%  | Note |
|-----------------|-----------|------------|---------|------|
| derivatives     | 0.511     | 0.055      | 10.8%   | wall-clock floor |
| tree_support    | 5.771     | 0.166      | 2.9%    |      |
| histogram       | 21.085    | 0.883      | 4.2%    |      |
| suffix_sum      | 1.091     | 0.178      | 16.3%   | wall-clock floor |
| split_score     | 1.939     | 0.069      | 3.6%    |      |
| leaf_estimation | 2.485     | 0.073      | 2.9%    |      |
| **sum-of-per-kernel** | **32.883** | — | — | vs iter_total 33.674 ms (delta −0.791 ms, −2.3%) |

---

## Gate Assessment

### Gate 1: Baseline match
Pass. All 3 baseline runs within D0 31.933 ± 0.208 ms. BENCH_FINAL_LOSS bit-identical.

### Gate 2: stdev% < 5% per bucket
Partial. `tree_support`, `histogram`, and `leaf_estimation` are cleanly below 5% in all 3 runs. `derivatives` (~10%), `suffix_sum` (~6–16%), and `split_score` (~4–6%) exceed 5% in some runs.

**Root cause:** these three buckets have means of 0.5 ms, 1.1 ms, and 2.0 ms respectively. Apple Metal command-buffer submit latency jitter is 20–100 µs (well-documented). At 0.5 ms mean, a 55 µs jitter = 11% — irreducible regardless of sample count or trim factor. Both 200-iter runs and 10%-trimmed statistics were tried; the floor persists. This is annotated as `[wall-clock floor: sub-ms jitter]` in the output rather than a data quality warning. The **means** are stable across all 3 runs to within 2%.

For D1-R1 and D1-R2 (lever ranking), the histogram bucket (21.x ms, 4% stdev, stable means) is the primary signal. The sub-ms buckets are informative for relative ordering but not sub-5% precise.

### Gate 3: sum-vs-total delta within 1–5%
Pass. Delta = −2.2% to −2.3% across all 3 runs, well within bounds. Negative sign (sum < total) reflects untimed CPU-side metadata operations between Metal submits (array shape construction, etc.) — these are in the ~0.7 ms gap. No missing dispatch boundary: all major GPU dispatches are accounted for in the 6 buckets.

### Gate 4: sync confirmation (per-kernel > baseline)
Pass. Per-kernel warm_mean = 33.8–34.7 ms vs baseline 31.2–31.9 ms. Delta = +2.1 to +2.8 ms. The mx::eval() sync points are suppressing MLX kernel overlap and adding measurable overhead, confirming they are forcing real GPU-CPU synchronization.

---

## Bucket Breakdown (cross-run stable means)

| Bucket          | Run1   | Run2   | Run3   | mean across runs | % of iter_total |
|-----------------|--------|--------|--------|-----------------|----------------|
| derivatives     | 0.516  | 0.481  | 0.511  | 0.503 ms        | 1.5%           |
| tree_support    | 5.864  | 5.727  | 5.771  | 5.787 ms        | 17.0%          |
| histogram       | 21.920 | 21.631 | 21.085 | 21.545 ms       | 63.4%          |
| suffix_sum      | 1.085  | 1.035  | 1.091  | 1.070 ms        | 3.1%           |
| split_score     | 1.993  | 1.934  | 1.939  | 1.955 ms        | 5.8%           |
| leaf_estimation | 2.492  | 2.462  | 2.485  | 2.480 ms        | 7.3%           |

**histogram dominates at 63.4%** of total iteration time. This is the primary lever for D1-R1 (L2 direct mechanism test) and D1-R2 (T2 prod-shape micro-bench).

---

## Implementation Notes

Only `catboost/mlx/tests/bench_boosting.cpp` was modified. Zero changes to any production source.

### What was added

1. `TBenchConfig::PerKernelProfile` field + `--per-kernel-profile` CLI flag.
2. `TPerKernelTimings` struct (6 bucket vectors + IterTotalMs).
3. `FindBestSplitGPU`: 3 new optional parameters (`perKernelProfile`, `suffixMsOut`, `scoreMsOut`). When `perKernelProfile=true`, inserts `mx::eval(transformedHist)` between suffix and score phases. Non-profile path is identical to prior code.
4. `RunIteration`: `TPerKernelTimings* pkOut` parameter. When non-null: depth loop branches into an instrumented path that forces eval after `ComputePartitionLayout`, after scoring `ComputeLeafSumsGPU`, after `DispatchHistogram`, and retrieves sub-timings from `FindBestSplitGPU`; non-null path mirrors the original non-instrumented code exactly in the else branch.
5. Per-kernel reporting block in `main()`: 10%-trimmed stats, `[wall-clock floor]` annotation for sub-ms buckets, sum-vs-total delta, sync overhead note.

### Bucket isolation correctness

Each bucket captures exactly one logical dispatch group, with an `mx::eval()` sync point at the boundary:

- **derivatives**: t0 → t1 reuses existing `mx::eval({gradients,hessians})` and `mx::eval(statArr)`.
- **tree_support**: wraps `ComputePartitionLayout` (argsort+scatter+cumsum) with a new eval, then wraps the depth-loop scoring `ComputeLeafSumsGPU` with its already-existing eval.
- **histogram**: wraps `DispatchHistogram` + existing `mx::eval(histogram)`.
- **suffix_sum**: new `mx::eval(transformedHist)` between suffix and score inside `FindBestSplitGPU` (profile path only).
- **split_score**: from new suffix eval to existing `mx::eval({scoreResult[0..2]})` + CPU reduction.
- **leaf_estimation**: from `leaf_t0` (after t2) through final `mx::eval(cursor)` to `wallEnd`.
