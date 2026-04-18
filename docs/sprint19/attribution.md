# Sprint 19 — S19-01: Writeback Attribution on 50k/RMSE/d6/128b

**Config:** N=50k, RMSE, depth=6, 128 bins  
**Source data:** `.cache/profiling/sprint19/baseline/50000_rmse_d6_128bins.json` — 50 iterations, iters 0–49 (all-iters methodology per S16 convention)  
**Date captured:** 2026-04-17  
**Baseline:** S18 after-JSON (L1a kernel, `simdHist[8][1024]` on-chip, commit `dccb7ec0a2`)  
**Reported by:** @performance-engineer (S19-01)

---

## Method

Instrumentation plan (PROPOSE → CRITIQUE):

**Proposed:** Metal System Trace dynamic capture via `xcrun xctrace` to obtain per-shader phase timings (accumulation, fold, writeback) directly from AGX hardware counters.

**Critique / fallback:** MST dynamic capture was blocked in S18-09 (see `docs/sprint18/mst_findings.md`) due to sandbox permission limits — that condition is unchanged here. The S18-09 fallback — two-component linear regression on per-depth `histogram_ms` breakdown — is the implemented method, applied now to the 50k gate config JSON.

**Regression model:**
```
hist_ms(depth_d) = K_fixed × TG_count(d) + K_accum
```
Where `TG_count(d) = numFeatureGroups × 2^d × numStats × maxBlocksPerPart`.

From `catboost/mlx/methods/histogram.cpp`:
- `maxBlocksPerPart = 1` (hardcoded, line 124)
- `numStats = 1` (RMSE, no hessian)
- `numFeatureGroups = ceil(100 features / 4) = 25`
- `TG_count(d) = 25 × 2^d`
- Total TGs per iter = `25 × (1+2+4+8+16+32) = 1575`

---

## Baseline statistics

All 50 iterations, iters 0–49 (all-iters methodology, consistent with S16/S17/S18).

| Metric | All-iters | Steady-state (iters 1–49) |
|---|---:|---:|
| `histogram_ms` mean | **15.462 ms** | **15.425 ms** |
| `histogram_ms` stdev | 1.420 ms | 1.411 ms |
| `histogram_ms` SEM | 0.201 ms | 0.202 ms |
| `iter_total_ms` mean | **21.211 ms** | **21.032 ms** |
| `histogram_ms` / `iter_total_ms` | 72.9% | 73.3% |
| iter 0 `histogram_ms` | 17.251 ms | — |

**Observation — JIT gap at N=50k:** iter 0 excess above SS mean = 1.83 ms. This is much smaller than the 105× spike seen at N=10k (S18). At N=50k the accumulation phase is so dominant that JIT compilation adds only ~12% overhead to iter 0, not 10–100×. No secondary depth-3 spike is visible in the data (same two dispatch geometries, but both JITs are absorbed into a 5x larger denominator).

---

## Per-depth histogram_ms (steady-state iters 1–49)

| Depth | Partitions | TGs per group | Mean (ms) | Stdev (ms) |
|---|---:|---:|---:|---:|
| 0 | 1 | 25 | 2.321 | 0.081 |
| 1 | 2 | 50 | 2.138 | 0.117 |
| 2 | 4 | 100 | 2.318 | 0.252 |
| 3 | 8 | 200 | 2.876 | 0.356 |
| 4 | 16 | 400 | 3.065 | 0.599 |
| 5 | 32 | 800 | 2.706 | 0.403 |
| **SUM** | | **1575 total** | **15.425** | |

**Critical observation:** The per-depth profile is nearly flat (2.14–3.07 ms across depths). At N=10k in S18 the profile rose steeply with depth (depth 0 ≈1.5 ms, depth 5 ≈5+ ms), giving R²=0.97 for the `K_fixed × 2^d` term. Here R²=0.34 — the depth-proportional signal is buried. This single fact tells us the structure of the bottleneck has changed.

**What a flat depth profile means:** `cost(depth d) = 2^d × cost(N/2^d) + K_fixed × 2^d`. If cost is linear in docs, `2^d × cost(N/2^d) = N × c` — constant across depths. The flat profile confirms the **accumulation term is linear in N and dominates at N=50k**, while the TG-proportional term (zero-init + fold + writeback) is negligible relative to it.

---

## Linear regression fit

Fitted `K_fixed` and `K_accum` on SS per-depth means:

| Parameter | N=10k (S18-01) | N=50k (S19-01) | Ratio |
|---|---:|---:|---:|
| `K_fixed` | 9.36 µs/TG | 0.714 µs/TG | 13× smaller |
| `K_accum` | 1.073 ms/depth | 2.384 ms/depth | 2.2× larger |
| R² | 0.97 | 0.34 | degraded (flat profile) |

The 13× drop in `K_fixed` is explained by the L1a kernel change: the S18 `K_fixed` was dominated by **device-memory traffic** — `privHist[1024]` zero-init (4 KB DRAM write per TG) and the D1c reduction reading `privHist` back from device memory. L1a eliminated both by moving them on-chip to `simdHist[8][1024]`. What remains in `K_fixed` is purely on-chip work plus the writeback to device memory.

---

## Phase decomposition

Using the measured `K_accum = 2.384 ms/depth` and residual `(SS_mean − 6×K_accum) = 15.43 − 14.30 = 1.13 ms` for all TG-proportional phases combined.

| Phase | Lines (`kernel_sources.h`) | ms estimate | ±err | % of SS 15.43 ms |
|---|---|---:|---:|---:|
| Accumulation (32-doc scatter, N-proportional) | 165–209 | **14.30** | ±1.5 | **93%** |
| Zero-init (on-chip simdHist, TG-proportional) | 160–163 | ~0.16 | ±0.10 | ~1% |
| D1c cross-SIMD fold (on-chip, TG-proportional) | 224–238 | ~0.16 | ±0.10 | ~1% |
| Writeback (global atomic, TG-proportional) | 251–264 | **~0.79** | ±0.30 | **~5%** |
| Launch/dispatch/other | — | ~0.02 | ±0.20 | ~0% |
| **TOTAL** | | **15.43** | | 100% |

**Writeback derivation:** 1575 TGs × 0.50 µs/TG. The 0.50 µs/TG estimate comes from the 1.13 ms residual minus estimated on-chip phases (zero-init ~0.10 µs/TG + fold ~0.10 µs/TG = 0.20 µs/TG), leaving 0.50 µs/TG for writeback. Per-TG atomic count: 508 writes (4 features × up to 127 active bins each at 128-bin config). At 1 ns per non-contended atomic write after pipelining, 508 writes ~ 0.5 µs/TG is plausible. Error bar of ±0.30 ms reflects the R²=0.34 regression quality.

---

## Atomic contention measurement

**Instruments / MST dynamic capture:** blocked (see `docs/sprint18/mst_findings.md §Status`). Same condition applies in S19-01; no `.trace` bundle was produced. Stall cycle counts are not available without the capture.

**Static analysis — contention proof:**

From `catboost/mlx/methods/histogram.cpp` line 124: `maxBlocksPerPart = 1` (hardcoded, unconditional). Per the dispatch geometry:

```
grid = (256 * maxBlocksPerPart,  numPartitions, numStats)
     = (256 * 1,                  2^d,            1)
threadgroup = (256, 1, 1)
=> threadgroups = 1 per (partition, numStats, feature-group dispatch)
```

Each `(partition, bin, feature)` slot in the output histogram is written by **exactly one threadgroup**. The `atomic_fetch_add_explicit` at `kernel_sources.h:262` was written to handle the `maxBlocksPerPart > 1` case (multiple TGs per partition, each contributing partial counts). With `maxBlocksPerPart = 1` always, every atomic is a **non-contended single-writer atomic** — there is zero hardware-level collision on any histogram slot.

This is the key structural fact for S19: the current writeback cost is not from atomic contention but from the **atomic pipeline overhead** itself (the hardware cost of issuing an atomic instruction versus a plain store, even when uncontended). On Apple Silicon GPU, non-contended `atomic_fetch_add` to device memory costs approximately 2–5× a plain device store due to the atomic instruction pipeline.

Per-threadgroup atomic count: 508 writes (4 features × ~127 active bins × 1 non-zero check).

---

## Root cause classification

**Classification: (B) Hardware-bound scheduling artifact — but with a software-addressable component.**

More precisely: this is a **false positive for classification (A)**. The writeback is not slow because of atomic contention (there is zero contention, proven by `maxBlocksPerPart = 1`). It is slow because of the **atomic instruction overhead itself** — a cost that can be eliminated by substituting plain device stores, which is exactly what the two-phase reduction achieves.

However, the correct classification for the R8 gate question is different:

**The root cause of the 15.43 ms `histogram_ms` floor at N=50k is accumulation, not writeback.**

Writeback = 0.79 ms (5% of SS). Accumulation = 14.30 ms (93% of SS). The writeback elimination addresses 5% of the histogram cost. Writeback is **not the plurality** at the new gate config.

**Why does the S18 attribution give a different picture?**

The S18-01 attribution (N=10k) measured writeback = 5.0 ms = 21% of SS (the second-largest phase). The task brief projected this to ~15 ms at N=50k, but that scaling is wrong. The S18 writeback cost of 5 ms reflects the `K_fixed` term at N=10k where `K_fixed = 9.36 µs/TG`. At N=50k the effective `K_fixed = 0.71 µs/TG` because L1a eliminated the dominant device-memory operations from the TG-proportional path. Writeback survives but is now 1/13th the prior `K_fixed`, not a scaled-up version of it.

The S18-01 S19 planning signal (`docs/sprint18/mst_findings.md §e, point 3`) correctly noted "atomic writeback at the tail still does one `atomic_fetch_add` per non-zero bin per partition... fine for accuracy, but at small N (1k config), the writeback may dominate launch overhead." That note was cautiously correct. The extrapolation to ~15 ms on 50k in the Sprint 19 README was not grounded in the data.

---

## R8 gate verdict

**R8 FAILS. Projection must revise DOWN before S19-03 implementation begins.**

| Metric | Baseline (SS) | Writeback → 0 (optimistic) | Writeback store substitution (conservative, 50% saving) |
|---|---:|---:|---:|
| `histogram_ms` | 15.43 ms | ~14.64 ms | ~14.83 ms |
| `iter_total_ms` | 21.03 ms | ~20.24 ms | ~20.64 ms |
| `histogram_ms` reduction | — | ~5.1% | ~3.9% |
| e2e speedup | 1.00× | **1.04×** | **1.02×** |

**R8 target was 1.5× e2e on 50k/RMSE/128b.** Maximum achievable via writeback alone: **1.04× (optimistic).** The gate requires a 7.01 ms saving on a 21.03 ms baseline; writeback can contribute at most 0.79 ms (11% of required).

**What blocks 1.5×:** Accumulation at 14.30 ms (93% of `histogram_ms`). No amount of writeback optimization closes this gap.

---

## Two-option pivot for @ml-product-owner

**Option 1 — Proceed with writeback two-phase reduction, revised gate.**

Deliver S19-03 as planned (two-phase + prefix-scan, DEC-013). Project realistic gains:

- `histogram_ms` reduction: 3–5%
- e2e speedup: 1.02–1.04×
- Value: eliminates the false-atomic overhead permanently; clean code simplification (stores are simpler than atomics); establishes foundation for Sprint 20 accumulation work.
- Gate: revise R8 to ≥1.03× e2e (25th percentile of writeback saving range).

**Option 2 — Pivot Sprint 19 to accumulation redesign.**

The 14.30 ms accumulation is the real target at N=50k. The 32-doc cooperative scatter batch in `kernel_sources.h:175–209` does `32 × SIMD_SIZE = 32 × 32 = 1024` `simd_shuffle` calls per batch, with a branch per feature per shuffle. Reducing shuffle depth (e.g., coalescing FEATURES_PER_PACK=4 into one packed compare) or pre-computing bin-owner masks could cut accumulation 30–50%.

- Projected `histogram_ms` reduction: 30–50% (4.3–7.1 ms saving)
- e2e speedup: 1.25–1.50×
- R8 gate: achievable at upper end of range.
- Risk: higher implementation complexity, parity risk requires careful DEC-008 re-verification.

**Recommendation:** Option 2 has 7–10× the lever of Option 1 at the 50k gate config. If R8 (1.5× e2e at 50k) is a committed target, Option 1 cannot deliver it regardless of implementation quality. However, if writeback elimination is desirable as a correctness simplification (removing unnecessary atomics), it can be included as a sub-task of Option 2 at negligible extra cost.

---

## Lineage and traceability

- `docs/sprint18/attribution.md §Anomaly 2` — writeback at 5 ms on N=10k flagged as second-largest; note: "If L1 reduces kernel latency and raises concurrent threadgroup count, writeback contention could increase — track in S18-05." The increase did not materialize; L1a also reduced K_fixed dramatically.
- `docs/sprint18/mst_findings.md §e, point 3` — correct identification of atomic writeback as a future bottleneck at small N. The note to "track in S18-05" applied to small-N configs, not 50k.
- `docs/sprint19/README.md §Performance projection` — the ~15 ms writeback floor claim is not supported by S19-01 data. This finding supersedes it.
- Benchmark methodology: iter0 included per S16 convention (`feedback_iter0_included.md`). 50-iter mean, all-iters column used for gate comparison. Source JSON at `.cache/profiling/sprint19/baseline/50000_rmse_d6_128bins.json`.
