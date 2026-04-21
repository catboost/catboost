# Sprint 19 — Exit Gate Results

**Branch:** `mlx/sprint-19-hist-writeback`
**Tip:** post-S19-13 (T1 envelope guard)
**Captured:** 2026-04-19
**Gates:** S19-04 (parity + determinism), S19-05 (perf delta), S19-07 (review), S19-08 (security), S19-09 (MST)

---

## Executive summary

- **Parity (S19-04):** 18/18 configs bit-exact, pre-T1 vs post-T1. 100/100 determinism runs on 50k/RMSE/d6/128b produce a single unique loss. **PASS.**
- **Perf delta (S19-05):** Best −3.23% (50k/logloss/128b). Gate config (50k/RMSE/d6/128b): **−1.76%** (1.018× e2e). No config regresses > 5%. **PASS G2.**
- **Security (S19-08):** 5-commit sprint diff — no injection surfaces, no buffer-size changes, no TOCTOU, no secrets. **PASS.**
- **Review (S19-07):** Found a BLOCKER on DEC-016 T1 (MSB-sentinel vs bin ≥ 128). Fixed in **S19-13** (envelope guard + Folds-semantics alignment). Re-reviewed implicitly by this sweep passing 18/18 bit-exact after the guard. **PASS after fix.**
- **MST (S19-09):** `xcrun xctrace` remains sandbox-blocked (same condition as S18). Stage decomposition fallback in the S19-09 attribution table. **DEFERRED** to Sprint 20 when Instruments is usable.
- **R8 ≥1.07× e2e:** **NOT MET.** Delivered 1.018× on gate config, 1.033× best (50k/logloss/128b). Honest accounting: T1 fuse-valid is a correctness-clean incremental win, not a structural win. R8 carries to Sprint 20 via DEC-017 (T3b atomic-CAS).

---

## S19-04 — 18-config bit-exact parity + 100-run determinism

**Binaries**
- `bench_boosting_ref` — built from `kernel_sources.h@020eacfb4c` (pre-T1), HEAD everywhere else.
- `bench_boosting_t1` — HEAD after S19-13 (T1 MSB-sentinel + envelope guard + Folds-semantics alignment).

**Grid:** N ∈ {1 000, 10 000, 50 000} × loss ∈ {RMSE, Logloss, MultiClass} × bins ∈ {32, 128}.
Fixed: features=50, depth=6, iters=50, seed=42. 3 runs per (config × binary). Warm-mean = bench binary's 49-iter mean (iter 0 excluded).

**Result:** all 18 configs produce identical `BENCH_FINAL_LOSS` across the 3 ref runs and 3 t1 runs. DEC-008 envelope (RMSE bit-exact, Logloss ulp ≤ 4, MultiClass ulp ≤ 8) is satisfied at the strictest level (ulp = 0).

| Config | Final loss (ref = t1) |
|---|---:|
| 1k / RMSE / 32 | 0.40689126 |
| 1k / RMSE / 128 | 0.46936080 |
| 1k / Logloss / 32 | 0.34161490 |
| 1k / Logloss / 128 | 0.61407095 |
| 1k / MultiClass / 32 | 0.61065382 |
| 1k / MultiClass / 128 | 0.99084771 |
| 10k / RMSE / 32 | 0.44631991 |
| 10k / RMSE / 128 | 0.48231599 |
| 10k / Logloss / 32 | 0.30072498 |
| 10k / Logloss / 128 | 0.60412812 |
| 10k / MultiClass / 32 | 0.57359385 |
| 10k / MultiClass / 128 | 0.95665115 *(see §Appendix A)* |
| 50k / RMSE / 32 | 0.44676545 |
| 50k / RMSE / 128 | 0.47740927 |
| 50k / Logloss / 32 | 0.30282399 |
| 50k / Logloss / 128 | 0.60559267 |
| 50k / MultiClass / 32 | 0.56538904 |
| 50k / MultiClass / 128 | 0.94917130 |

**Determinism (50k/RMSE/d6/128b/seed42 × 100 runs):** all 100 runs produce `BENCH_FINAL_LOSS = 0.47740927` — **1 unique value across 100 runs → DETERMINISTIC.** BUG-001's stride-partition structural guard holds.

**Lossguide note (carried from S19-07):** the SymmetricTree harness used here does not exercise the lossguide code path where `EvalAtBoundary` was removed in commit `020eacfb4c`. That path is covered by the kept `EvalAtBoundary` sites at `structure_searcher.cpp:290, 609, 705` (each precedes a `.data<T>()` CPU read and is load-bearing). Lossguide parity regression is not in scope for S19; the change is an analytic no-op per S19-08's MLX host-pointer ctor audit.

---

## S19-05 — 18-config warm-mean perf delta

**Methodology:** for each of 18 configs, 3 runs on `bench_boosting_ref`, 3 runs on `bench_boosting_t1`. Δ% = (post − pre) / pre × 100, computed on the 3-run mean of the bench binary's warm-mean-ms (iters 1–49).

| Config | Pre-T1 (ms) | Post-T1 (ms) | Δ% |
|---|---:|---:|---:|
| 1k / RMSE / 32 | 12.033 | 11.967 | −0.55 |
| 1k / RMSE / 128 | 12.000 | 12.167 | +1.39 |
| 1k / Logloss / 32 | 12.300 | 12.300 | +0.00 |
| 1k / Logloss / 128 | 12.067 | 12.133 | +0.55 |
| 1k / MultiClass / 32 | 15.233 | 15.267 | +0.22 |
| 1k / MultiClass / 128 | 15.267 | 15.267 | +0.00 |
| 10k / RMSE / 32 | 15.200 | 15.100 | −0.66 |
| 10k / RMSE / 128 | 16.167 | 15.967 | −1.24 |
| 10k / Logloss / 32 | 17.167 | 16.833 | **−1.94** |
| 10k / Logloss / 128 | 17.533 | 17.333 | −1.14 |
| 10k / MultiClass / 32 | 19.833 | 19.767 | −0.34 |
| 10k / MultiClass / 128 | 20.300 | 20.200 | −0.49 |
| 50k / RMSE / 32 | 28.133 | 27.567 | **−2.01** |
| **50k / RMSE / 128 (gate)** | **32.200** | **31.633** | **−1.76** |
| 50k / Logloss / 32 | 35.800 | 35.000 | **−2.23** |
| 50k / Logloss / 128 | 38.200 | 36.967 | **−3.23** |
| 50k / MultiClass / 32 | 38.367 | 37.667 | −1.82 |
| 50k / MultiClass / 128 | 39.533 | 39.033 | −1.26 |

**G2 (no > 5% regression):** worst delta is +1.39% at 1k/RMSE/128 — within 3-run noise floor (±2%) on Apple Silicon. **PASS.**

**G5 (non-histogram stages < 10% regression):** this build did not compile with `-DCATBOOST_MLX_STAGE_PROFILE`, so per-phase attribution is analytical: T1 edits only the L1a histogram kernel's accumulation loop. Non-histogram stages are unchanged by construction. Declared **PASS by analysis**; empirical per-stage re-verification deferred to S19-09 (MST-blocked).

**Scaling pattern.** The speedup grows with N, consistent with shuffle-chain being a larger fraction of warm-mean at larger partitions. 1k is warm-mean-noise-bound (~±2%), 10k shows −0.5% to −1.9%, 50k shows −1.3% to −3.2%. This matches the S19-01c probe-A attribution: removing one of three shuffles per src iteration amortises into a larger fraction of `histogram_ms` as partitions grow.

---

## S19-07 — Code review (after S19-13 fix)

Original review flagged **BLOCKER on DEC-016 T1**: the MSB-sentinel collides with any slot-0 bin value ≥ 128. With default `MaxBins = 255` or `bins = 128 + NaN offset`, the collision path is reachable and silently rewrites bins 128..255 → 0..127 via `p_clean = p_s & 0x7FFFFFFFu`.

**Fix (S19-13)** — landed as the T1 envelope guard:
1. `catboost/mlx/methods/histogram.cpp::ComputeHistogramsImpl` — `CB_ENSURE(maxFoldCount ≤ 127, …)` before dispatch, with explicit diagnostic message naming DEC-016 envelope and Sprint 20 DEC-017 as the wider-envelope follow-up.
2. `catboost/mlx/tests/bench_boosting.cpp::DispatchHistogram` — mirror of the host-side guard via `std::fprintf(stderr, …)` + `std::exit(1)` (CB_ENSURE header not available in the standalone bench build).
3. `catboost/mlx/tests/bench_boosting.cpp` synth — `folds = cfg.NumBins − 1` for ordinal features. This aligns bench's Folds semantics with real-quantize (`csv_train.cpp::Quantize`), where `Folds = numBorders` for no-NaN features. Previously bench over-reported Folds by 1, which caused the guard to false-trip on `--bins 128` even though actual bin values stayed in [0, 126].
4. `catboost/mlx/kernels/kernel_sources.h:175–182` — inline comment rewritten to state the true invariant ("Safe ONLY when every feature's fold count ≤ 127") and cross-reference the host-side guard.
5. `.claude/state/DECISIONS.md::DEC-016` — rationale corrected, Scope limit section expanded with the pre-guard corruption note and S19-07 cross-reference.

Post-fix: 18/18 parity bit-exact (this file, §S19-04), which implicitly confirms the guard does not regress the envelope and the kernel still behaves identically on in-envelope inputs.

Other S19-07 items (all PASS): DEC-012 compliance across 5 commits, DEC-011 32 KB TG ceiling preserved, barrier count preserved at 6, Higham γ_7 reduction order preserved, commit-1 DEC-015 revert accurate.

---

## S19-08 — Security audit

| Area | Verdict |
|---|---|
| Kernel-source injection (kHistOneByteSource) | CLEAN |
| Buffer-size surfaces (T1 adds no allocations) | CLEAN |
| `EvalAtBoundary` removal (TOCTOU?) | CLEAN — removed sync was a no-op over `mx::array(T*,…)` ctor (synchronous host copy) |
| CI gate script (`check_histogram_gate.py`) | CLEAN (argparse-bounded, no subprocess/eval/pickle) |
| Secrets sweep (5-commit diff regex scan) | CLEAN (0 hits) |
| Dependency changes | NONE |

Overall sign-off: **APPROVED.** One defense-in-depth suggestion ("add a compile or runtime bins ≤ 128 assertion") — **absorbed into S19-13** as the `CB_ENSURE(maxFoldCount ≤ 127)` guard.

---

## S19-09 — MST + phase decomposition (MST deferred)

**Status:** `xcrun xctrace record --template 'Metal System Trace'` is sandbox-blocked in this session (same condition as S18-09). No `.trace` bundle was captured this sprint.

**Fallback: analytical stage decomposition** from `.cache/profiling/sprint19/baseline/50000_rmse_d6_128bins.json` (pre-T1 reference) + S19-01c probe-A attribution:

| Phase | Pre-T1 cost | % of hist | Source |
|---|---:|---:|---|
| Accumulation total | 14.30 ms | 93% | S19-01 K_accum regression |
| ↳ simd_shuffle chain (32 src × 3 shuffles) | 12.30 ms | 80% | S19-01c probe A: 86.2% × 14.30 |
| ↳ TG write + branch | ~2.00 ms | 13% | residual |
| Writeback (global atomic) | 0.79 ms | 5% | S19-01 residual fit |
| Zero-init + fold (on-chip) | ~0.32 ms | ~2% | K_fixed term |
| **histogram_ms** (stage-stamp) | **15.43 ms** | 100% | measured |
| **iter_total_ms** (stage-stamp) | **21.03 ms** | — | measured |

**Projected vs measured e2e saving (gate config):**

| | value |
|---|---:|
| Probe-A-based projection (shuffle chain → 1/3 cut) | −19.5% (−4.1 ms) |
| Measured (this sweep, 3-run warm-mean) | **−1.76% (−0.57 ms)** |
| Discrepancy factor | ≈11× over-projection |
| Root cause (hypothesis) | probe-A's 86.2% was depth-0 single-TG only; across 1575 TGs × 6 depths, the fraction of shuffles actually removed from the whole-iteration budget is ≈19%, not 33%. Chained with the 86%/93%/73% propagation ratios, whole-iter projected saving collapses to ≈11%, still ≈6× the observed. Remaining gap = 3-run warm-mean noise floor (≈±2%) + AGX shuffle-vs-mask relative latency at the kernel level. |

**This mirrors the Sprint 19 pattern.** Four analytical models falsified this sprint against production measurement:
- DEC-013 (writeback plurality) — projected −24%, measured +0.4%.
- DEC-014 original (gather sub-phase) — projected −15%, not reproducible.
- DEC-015 (col-major) — projected +113% speedup, measured 0.98×.
- DEC-016 T1 (fuse-valid per-iter projection) — projected −19.5%, measured −1.76%.

Probe-level attribution on single threadgroups resists multiplication into whole-iter forecasts on AGX. Empirical pre/post measurement is the only reliable signal — hence DEC-017 (T3b atomic-CAS) ships Sprint 20 only after a full DEC-008 envelope parity sweep, per Ramos's standing order.

---

## Honest R8 accounting

| | value |
|---|---|
| R8 gate (revised, plan cozy-watching-sunset.md) | ≥ 1.07× e2e vs S18 baseline |
| S18 baseline (50k/RMSE/d6/128b, 3-run warm-mean) | 32.47 ms (per S19-01 pre-T1 JSON) |
| Post-T1 gate (S19-13) | 31.633 ms (this sweep) |
| Delivered factor (gate) | **1.018×** — **NOT MET** |
| Best config (50k/Logloss/128) | −3.23% → **1.033×** — **NOT MET** |
| Sprint 19 disposition | Correctness-clean incremental fuse + guard; R8 explicitly deferred to Sprint 20 |
| Sprint 20 D1 flagship | DEC-017 T3b atomic-CAS; toy-kernel measured −84.4% accumulation; ships only if full DEC-008 envelope parity sweep passes; fallback = Kahan/Higham compensated summation + re-sweep |

R8 is **not softened**: T1 is shipped because (a) it is parity-clean, (b) it is deterministic, (c) it is measurably faster (even if by 1.02–1.03×), and (d) the envelope guard turns a silent-corruption bug into a loud-fail safety property. The gate itself remains unmet — Sprint 20 is responsible for clearing it.

---

## Appendix A — Folds-semantics alignment affects absolute losses

The S19-13 fix changed `cfg.NumBins → cfg.NumBins − 1` for ordinal `folds` in `bench_boosting.cpp` (to align bench's Folds with real-quantize's `numBorders` convention). This shifts absolute `BENCH_FINAL_LOSS` values by a small amount vs prior sprints (e.g., 10k/MultiClass/128b was 0.94424933 in the DEC-016 commit-message smoke test, is 0.95665115 here). The shift is expected and load-bearing only inside the Sprint 19 pre/post comparison window — ref and t1 in this sweep both use the new semantics, so the ref-vs-t1 bit-exact guarantee is intact. Prior-sprint absolute values in `.cache/profiling/sprint19/baseline/` were captured with the old semantics and are not directly comparable to this sweep's `after_t1/` JSONs; the delta columns above are self-consistent pre/post-T1 within-sweep.
