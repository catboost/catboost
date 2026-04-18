# Sprint 18 Results — 18-Config Before/After Delta (S18-05b)

**Verdict: PASS.** All three performance gates clear. `histogram_ms` reduced **56.6–85.5%** across all 18 configs. Gate config (N=10k, RMSE, d6, 128b) reduced from 28.75 ms to 9.56 ms — **-66.8%**, beating the S18-G1 target of ≤18.7 ms by 9.1 ms margin.

Captured 2026-04-17. "Before" is the Sprint 17 after-profile (seeded as Sprint 18 baseline at Day 0, `.cache/profiling/sprint18/`). "After" is the Sprint 18 L1a fixed kernel (`BUG-S18-001` corrected, commit `19fa5ce6cc`, branch `mlx/sprint-18-hist-privhist-tile`). Binary: `csv_train_s18_profiled` compiled from `catboost/mlx/tests/csv_train.cpp` with `-DCATBOOST_MLX_STAGE_PROFILE`. All runs: depth 6, lr 0.10, l2 3.0, 50 iterations, 50 features. Timings are mean ms per iteration (all 50 iters, including iter 0 per Sprint 16 methodology).

Raw JSONs: `.cache/profiling/sprint18/` (baseline) and `.cache/profiling/sprint18/after/` (gitignored).

Parity: S18-04b parity re-run (commit `7ab4e8e804`) shows 108/108 checkpoints bit-exact and 100/100 determinism bit-exact on the fixed kernel.

---

## Headline

**Gate config (N=10k, RMSE, depth=6, 128 bins):**

| metric | S18 baseline (= S17 after) | S18 after | delta |
|:-------|---------------------------:|----------:|------:|
| `histogram_ms` | 28.75 ms | **9.56 ms** | **-66.8%** |
| `iter_total_ms` | 34.94 ms | 15.03 ms | -57.0% |

- S18-G1 (≥35% histogram_ms reduction, target ≤18.7 ms): **PASS** — 9.56 ms is 9.1 ms under target.
- S18-G4 (no non-histogram stage regresses >10% on gate config): **PASS** — all secondary stages improved 3–21%.

---

## 18-config delta table

|     N | loss       | bins | baseline_hist_ms | s18_hist_ms | delta_hist% | baseline_iter_ms | s18_iter_ms | delta_iter% | gate |
|------:|:-----------|-----:|-----------------:|------------:|------------:|-----------------:|------------:|------------:|:-----|
|  1000 | rmse       |   32 |            20.62 |        4.54 |      -78.0% |            25.26 |        8.60 |      -66.0% | OK |
|  1000 | rmse       |  128 |            20.31 |        4.51 |      -77.8% |            26.22 |        9.81 |      -62.6% | OK |
|  1000 | logloss    |   32 |            21.41 |        4.35 |      -79.7% |            26.53 |        8.24 |      -69.0% | OK |
|  1000 | logloss    |  128 |            29.42 |        4.26 |      -85.5% |            44.47 |        9.42 |      -78.8% | OK |
|  1000 | multiclass |   32 |            39.58 |        6.86 |      -82.7% |            46.52 |       11.71 |      -74.8% | OK |
|  1000 | multiclass |  128 |            37.92 |        7.02 |      -81.5% |            46.39 |       14.07 |      -69.7% | OK |
| 10000 | rmse       |   32 |            23.54 |        7.00 |      -70.3% |            28.84 |       11.16 |      -61.3% | OK |
| **10000** | **rmse** | **128** | **28.75** | **9.56** | **-66.8%** | **34.94** | **15.03** | **-57.0%** | **S18-G1 gate** |
| 10000 | logloss    |   32 |            30.90 |        6.93 |      -77.6% |            45.21 |       11.15 |      -75.3% | OK |
| 10000 | logloss    |  128 |            23.81 |        6.94 |      -70.8% |            31.13 |       12.43 |      -60.1% | OK |
| 10000 | multiclass |   32 |            43.38 |       12.10 |      -72.1% |            51.51 |       17.12 |      -66.8% | OK |
| 10000 | multiclass |  128 |            56.38 |       12.26 |      -78.2% |            75.40 |       19.63 |      -74.0% | OK |
| 50000 | rmse       |   32 |            36.44 |       15.21 |      -58.3% |            43.82 |       19.64 |      -55.2% | OK |
| 50000 | rmse       |  128 |            45.30 |       15.46 |      -65.9% |            59.65 |       21.21 |      -64.4% | OK |
| 50000 | logloss    |   32 |            34.47 |       14.82 |      -57.0% |            40.33 |       19.33 |      -52.1% | OK |
| 50000 | logloss    |  128 |            35.42 |       15.06 |      -57.5% |            41.63 |       20.75 |      -50.1% | OK |
| 50000 | multiclass |   32 |            62.74 |       27.24 |      -56.6% |            68.77 |       32.76 |      -52.4% | OK |
| 50000 | multiclass |  128 |            67.44 |       27.44 |      -59.3% |            75.69 |       35.12 |      -53.6% | OK |

**Gate summary:**
- S18-G1 (≥35% histogram_ms reduction on gate config): **PASS** (66.8%).
- S18-G2 (no config regresses >5% on histogram_ms): **PASS** — range is -56.6% to -85.5% — all improved.
- S18-G4 (no non-histogram stage regresses >10%): **PASS** — worst across all 18 configs is +4.5% (cpu_readback_ms, 50k/Logloss/128b — within noise).

---

## Non-histogram stage check (S18-G4, gate config)

| stage | baseline_ms | s18_ms | delta% |
|:---|---:|---:|---:|
| derivatives_ms | 0.336 | 0.306 | -8.8% |
| partition_layout_ms | 1.490 | 1.349 | -9.5% |
| suffix_scoring_ms | 1.672 | 1.625 | -2.8% |
| leaf_sums_ms | 0.225 | 0.195 | -13.6% |
| leaf_values_ms | 0.248 | 0.198 | -19.9% |
| tree_apply_ms | 0.227 | 0.190 | -16.2% |
| loss_eval_ms | 0.238 | 0.204 | -14.4% |
| cpu_readback_ms | 0.096 | 0.076 | -21.1% |

All secondary stages improve on the gate config (pipeline backpressure reduction, same mechanism as Sprint 17). Worst non-histogram delta across all 18 configs: +4.5% (cpu_readback_ms, 50k/Logloss/128b). All 18 configs well under the 10% threshold.

---

## Analysis

**1. Gate config exceeds target by 9.1 ms.** S18-G1 required ≤18.7 ms. The fixed L1a kernel delivers 9.56 ms — 9.1 ms headroom. This is consistent with the S18-02 ablation's 35–60% predicted range; actual 66.8% lands in the upper half, driven by the elimination of per-thread register spill writes (privHist[1024] → simdHist in threadgroup memory removes ~4 KB/thread × 256 threads = 1 MB of device-memory traffic per threadgroup per iter).

**2. Improvement is N-dependent as expected.** Larger N shows smaller reduction percentage (56–66% at 50k vs 78–86% at 1k). This matches the attribution model: accumulation cost scales with N while zero-init and writeback are N-independent; at higher N the N-proportional term grows, making the per-SIMD zero-init saving a smaller fraction of the total.

**3. BUG-S18-001 fix vindicated.** The earlier broken-kernel sanity run reported -50.9% reduction. The fixed kernel delivers -66.8% — 16 percentage points higher. The broken kernel's 1/32 doc-inclusion rate meant ~97% of accumulation work was silently dropped; correctness was fatal and the timing was misleadingly favorable (less work done). The fixed result is the true measurement.

**4. Sprint 19 lever.** The 50k configs show a floor emerging at ~15 ms (rmse/32b). The writeback (global-atomic) phase, estimated at 5 ms in S18-01, is now the next dominant term. Batched-atomic writeback or shared-memory prefix-scan reduction of the per-SIMD histograms before global writeback is the likely S19 L1.

---

## Benchmark methodology

- Tool: `csv_train_s18_profiled` (compiled 2026-04-17 from fixed source, `-DCATBOOST_MLX_STAGE_PROFILE`)
- Data: synthetic 50-feature CSVs in `/tmp/bench_{N}_50f{suffix}.csv`
- Configs: depth=6, lr=0.10, l2=3.0, 50 iterations, target-col=50
- Iterations: 50 per run; mean includes iter 0 (JIT cold-start) per Sprint 16 methodology
- Hardware: Apple Silicon (arm64), MLX 0.31.1, Metal
- Reproduce:
  ```bash
  export DYLD_LIBRARY_PATH=/opt/homebrew/opt/mlx/lib
  CATBOOST_MLX_PROFILE_PATH=.cache/profiling/sprint18/after/10000_rmse_d6_128bins.json \
    ./csv_train_s18_profiled /tmp/bench_10000_50f.csv \
    --target-col 50 --loss rmse --depth 6 --bins 128 --iterations 50
  ```
