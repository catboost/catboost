# Sprint 19 — S19-10: Algorithmic Ablation (Shuffle-Reduction Variants)

**Config:** N=50k, RMSE, depth=6, 128 bins (gate config)  
**Date:** 2026-04-19  
**Branch tip:** `b0c853a6f6` (S19-01c micro-benchmark re-attribution)  
**Owner:** @research-scientist (S19-10 docs sweep)  
**Sources:** `docs/sprint19/reattribution.md` (S19-01c), `docs/sprint19/scratch/algorithmic/`

---

## 1. Context — Why Algorithmic Ablation Was Needed

Sprint 19 entered with three sequentially falsified analytical models:

**DEC-013 falsified (Day 2).** The sprint opened targeting writeback as the plurality bottleneck (~13–15 ms projected). S19-01 ground-truth attribution measured writeback at 0.79 ms / 5% of `histogram_ms`. DEC-013 (two-phase writeback reduction) was superseded before implementation.

**DEC-014 original hypothesis falsified (Day 2–3).** S19-02b pivoted to accumulation redesign. The dominant cost was identified as the 14.30 ms accumulation phase (93% of `histogram_ms`). The initial sub-phase model attributed this to `compressedIndex` gather latency — 25 cache lines per 32-doc batch, 4 L2 stall rounds per batch, projected 12.78 ms. Variant (A1) wider-batch and DEC-015 col-major layout were chosen based on this model.

**DEC-015 falsified (Day 3).** Direct measurement of col-major layout produced 0.98× e2e vs the projected 2.13×. The layout change that should have been most sensitive to gather latency produced no improvement.

S19-01c (reattribution) was commissioned to determine the actual dominant sub-phase. Probes A/B/C/D on a 1-TG × 256-thread harness at N=50k, depth-0 equivalent established that `simd_shuffle` is the plurality bottleneck at 86.2% of single-TG accumulation time. Probe D (global loads stripped) ran 2% SLOWER than production, proving AGX hides gather latency entirely behind the shuffle inner loop.

With gather ruled out, three algorithmic alternatives to the shuffle-based broadcast were measured via toy kernel isolation: fuse-valid (T1), bin-major sort-by-bin (T2), and threadgroup atomic-CAS (T3b). This document records the harness, results, and disposition of each.

---

## 2. Method

**Harness pattern:** `docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp`. Single threadgroup × 256 threads, 1 TG × 256 threads, processing all N=50,000 docs in a single partition (depth-0 equivalent, 196 outer-batch iterations at BATCH_DOCS=32). Timed via `std::chrono::steady_clock` + `mx::eval()` blocking. 5 warm runs + 5 timed runs per variant. Same methodology as `microbench_gather.cpp` (S19-01c).

**T0 baseline:** Production L1a kernel at branch tip `108c7a59d2` (col-major address expression from DEC-015 WIP). Probe D established that address expression is irrelevant — T0 result is representative of both row-major and col-major layouts.

**Correctness harness:** `docs/sprint19/scratch/algorithmic/verify_correctness.cpp` compares T0 and T3b output for feature-0 histogram (bins 0–127) against a FP64 CPU reference. T3b verified bit-exact on this check. T2 correctness is established by construction (per-bin serial sum with no shuffle). T1 parity is identical to T0 by inspection (MSB-sentinel fusion does not alter the accumulation path for valid docs).

**T3b contention sweep:** `docs/sprint19/scratch/algorithmic/microbench_contention.cpp` runs T0 and T3b at bins ∈ {128, 64, 32, 16} to measure how atomic contention scales with bin density.

---

## 3. Results

### 3.1 Ablation table

| Variant | Toy mean (ms) | Δ vs T0 (2.485) | Shuffle chain | New cost | Parity | Integration | Verdict |
|---|---:|---:|---|---|---|---|---|
| T0 production | 2.485 | — | 32×3 shuffles | — | γ_7, bit-exact | 0 | baseline |
| T1 fuse-valid | 2.396 | −3.6% | 32×2 shuffles | MSB-sentinel (safe ≤128 bins) | γ_7, bit-exact | ~1 day | SHIP S19 |
| T2 bin-major | 0.483 | −80.6% accum | eliminated | Pre-pass bucket-sort + partition mismatch | γ_N within-bucket | 5–8 days | DROP |
| T3b atomic-CAS | 0.387 | −84.4% | eliminated | TG atomic contention (low) | γ_N up to 2.3e-5, DEC-008 γ_8 ≈ 4.77e-7 | 2–3 days + parity sweep | S20 FLAGSHIP |

### 3.2 Contention sweep (T3b robustness)

| bins | T0 ms | T3b ms | T3b/T0 | docs/bin |
|---:|---:|---:|---:|---:|
| 128 | 2.469 | 0.436 | 0.176 | 390.6 |
| 64 | 2.515 | 0.389 | 0.155 | 781.2 |
| 32 | 2.448 | 0.404 | 0.165 | 1562.5 |
| 16 | 2.430 | 0.530 | 0.218 | 3125.0 |

T3b holds its speedup across all bin counts including 16-bin (3125 docs/bin average). The 16-bin case degrades to 0.218 T3b/T0 ratio from 0.176 at 128-bin — still 4.6× faster in absolute terms. Contention risk at the gate config (128 bins, 390 docs/bin) is low.

T3b functional correctness verified bit-exact on feature-0 histogram at 50k/128b vs FP64 CPU reference (`verify_correctness.cpp`). Parity risk is purely FP32 reduction-order drift across the full DEC-008 envelope — not structural incorrectness.

---

## 4. Per-Variant Detail

### 4.1 T1 — Fuse-valid (MSB-sentinel)

**Mechanism.** The production inner loop issues three `simd_shuffle` calls per src iteration: `simd_shuffle(packed, src)`, `simd_shuffle(stat, src)`, `simd_shuffle(valid, src)`. T1 fuses the valid flag into bit 31 of `packed`. At ≤128-bin configs, bits 6–0 of each byte hold the bin value; bit 31 of the packed uint32 is the high bit of the feature-0 slot and is never set by any valid bin index (bins 0–127 fit in 7 bits, not 8). Invalid lanes write `packed = 0` (no sentinel bit). The inner loop then checks `(p_s >> 31)` instead of `simd_shuffle(valid, src)`, reducing 3 shuffles per src to 2.

**Measurement.** −3.6% single-TG accumulation. Scales to ~0.68 ms saving on the full 14.30 ms accumulation budget (1/3 of shuffle cost × 86% shuffle share × 14.30 ms).

**Parity.** Identical to T0 for valid docs. The packed value for invalid lanes changes from 0 to 0 (no sentinel set), and the bin-extraction `(0 >> 24) & 0xFF = 0` still fires the bin-check which `foldCounts[f]+1` gates correctly. γ_7 unchanged. Bit-exact.

**Integration cost.** ~1 day. One-line change to the inner shuffle loop in `kernel_sources.h` + one-line change to the packed-value assignment. No structural changes. DEC-011 ceiling unchanged (0 new TG memory). No new kernel, no new dispatch.

**Constraint.** Safe only at ≤128-bin configs. At 256 bins, bin values can reach 255 = 0xFF; the high bit of the feature-0 byte would be bit 31 of packed for features packed in the top byte, colliding with the sentinel. Sprint 19 gate config is 128 bins — constraint satisfied. Wider bin configs require a different sentinel strategy (e.g. pack valid into a separate feature's unused bit) and must re-validate.

### 4.2 T2 — Bin-major accumulation (sort-by-bin pre-pass)

**Mechanism.** Eliminate shuffle entirely by pre-sorting docs into per-feature-per-bin buckets before dispatch. Each thread owns one bin per feature; it iterates over its bucket and sums stats sequentially. No cross-lane communication. The toy kernel measures the accumulation-only cost assuming the bucket sort is free.

**Measurement.** 0.483 ms — 80.6% accumulation reduction in isolation. This is the best-case floor for a shuffle-free formulation.

**Why dropped.** The pre-pass cost is not measured in the toy kernel but is substantial. For each of 25 feature groups at each of 6 depths, the pre-pass must bin-count, prefix-scan, and scatter N docs across `4 × 128 = 512` buckets per group per partition. At depth 5, there are 32 partitions each requiring an independent pre-pass. The pre-pass is a full-N scatter operation structurally similar to the histogram itself — it cannot be faster than the accumulation it replaces.

Additionally, the bucket-sort order does not align with CatBoost's partition layout. The production kernel partitions docs by tree-leaf membership; T2 would require per-partition bucket sorts that are recomputed at every depth. The data layout change (bucket-sorted vs partition-sorted doc indices) has no clear integration path into the existing `structure_searcher.cpp` pipeline without a major architectural change.

**Integration cost.** 5–8 days minimum. Not viable within a single sprint.

### 4.3 T3b — Threadgroup atomic-CAS no-shuffle accumulator

**Mechanism.** Replace the 32-iteration simd_shuffle broadcast loop with direct per-doc threadgroup atomic adds. Each thread processes its own doc at stride `BLOCK_SIZE` (256), atomically adding to `simdHistU[f * BINS_PER_BYTE + bin]` via a uint CAS-float loop. No shuffle, no bin ownership predicate — every doc contributes once to its correct bin.

TG memory changes from `float simdHist[8][1024]` (32 KB) to `atomic_uint simdHistU[1024]` (4 KB). This relaxes DEC-011's 32 KB ceiling to 4 KB — a 8× reduction in TG memory pressure, which in principle allows higher concurrent occupancy on AGX.

**Measurement.** 0.387 ms single-TG, 84.4% accumulation reduction vs T0 at 128 bins. Contention sweep confirms speedup holds across all bin counts tested (best ratio 0.155 at 64 bins, worst 0.218 at 16 bins — all at least 4.5× faster than T0).

**Parity analysis.** T3b correctness is established at the toy-kernel level (feature-0 bit-exact vs FP64 CPU reference). The parity risk for integration is FP32 reduction-order drift. T3b accumulates in arbitrary per-thread-dispatch order rather than the fixed stride-partition ownership order of T0. The Higham γ_N for the CAS accumulation is bounded by the number of concurrent writers per bin: at gate config, 256 threads × 196 outer-batch steps / 128 bins ≈ 390 effective updates per bin. This gives γ_N ≈ 2.3e-5 in the worst case.

DEC-008's RMSE/Logloss bound is γ_8 ≈ 4.77e-7 — about 50× tighter than the worst-case T3b bound. This does not mean T3b fails parity; FP32 accumulation of same-sign values with magnitudes bounded to [0, 1] typically realizes errors far below the Higham worst-case. The verify_correctness harness showed bit-exact results on the 50k/128b synthetic data, but that data has a specific uniformly-distributed structure. The full DEC-008 18-config grid with real gradient/hessian distributions must be swept before declaring parity safe.

Sprint 20 D1 is the full DEC-008 parity sweep against T3b toy-kernel-equivalent across all 18 configs. If parity holds — integrate (D2). If parity fails — apply Kahan compensated summation (+1 uint32 per bin as running compensation term) and re-sweep.

**Integration cost.** 2–3 days for the kernel rewrite + the DEC-008 parity sweep (additional 1–2 days). The kernel change is ground-up: `simdHist[8][1024]` → `atomic_uint simdHistU[1024]`, the accumulation loop structure changes entirely, cross-SIMD fold is eliminated (T3b produces the full per-bin sum directly without a fold phase). DEC-011 must be amended from 32 KB → 4 KB for this variant.

---

## 5. Recommendation

**Ship T1 in Sprint 19.** T1 is a one-line change, bit-exact, zero architectural risk, ~1.03× e2e improvement. Combined with DEC-014 (A1) BATCH_DOCS=64 (~1.04× e2e), the stacked Sprint 19 improvement is ~1.07× e2e — honest and shippable.

**Drop T2.** The pre-pass integration cost is prohibitive and the bucket-sort architecture does not map onto CatBoost's partition-first dispatch model.

**Promote T3b to Sprint 20 flagship.** The 84.4% single-TG accumulation reduction at gate config is the highest measured lever in Sprint 19. The integration path is clear. The single remaining gate is the DEC-008 parity sweep, which is Sprint 20 D1 — cheap to run (no kernel changes required for the sweep; use the toy-kernel-equivalent at all 18 configs). If parity holds, the full integration is D2. Projected e2e improvement from T3b: ~2.0–2.2× (84.4% accumulation reduction × 93% accumulation share of histogram_ms × 73% histogram share of iter_total).

---

## 6. Risk Disclosure

**Third-iteration failure mode.** Sprint 19 has falsified two major analytical models (S19-01b gather model, DEC-015) and one premise (DEC-013 writeback plurality). The pattern of falsification by direct measurement is a signal that analytical reasoning about AGX cache hierarchy is unreliable for this kernel. T3b's toy-kernel speedup is empirically robust; the integration risk is:

1. **Full-grid scaling.** The toy kernel measures 1 TG × 256 threads. Production dispatches 1575 TGs concurrently at depth 5–6. Whether T3b's TG-atomic pattern holds its speedup under 1575-TG concurrent dispatch is unmeasured (S19-01c note §4: "Scaled to full iteration... the per-TG probe is representative... but the dominant cost is at depth 0"). Sprint 20 D3 validates full-grid scaling before R8 is reset.

2. **MultiClass approxDim=3 parity drift.** MultiClass loss composes three independent reductions with per-dim gradients. The reduction-order non-determinism of T3b's CAS loop may compound per dim. Sprint 20 D1 must include all MultiClass configs in the parity sweep.

3. **Kahan overhead.** If the parity sweep fails and Kahan compensation is needed, the compensation term adds 1 atomic_uint per bin (4 KB at 128 bins, 4× TG memory but still only 16 KB vs the 32 KB DEC-011 ceiling). The Kahan-corrected T3b still fits DEC-011. However, the compensation loop adds 2 additional atomic operations per doc per bin — potentially compressing the speedup. If the compressed speedup falls below 2× e2e, Sprint 20 should consider whether integration is still worth the complexity.

---

## 7. Lineage

- S19-01c (`docs/sprint19/reattribution.md`) — probe evidence establishing `simd_shuffle` as 86% of accumulation; Probe D falsifying gather as bottleneck.
- S19-01 (`docs/sprint19/attribution.md`) — writeback=5%, accumulation=93% decomposition.
- S19-02b (`docs/sprint19/ablation_accumulation.md`) — DEC-014 (A1) analysis; initial DEC-015 col-major hypothesis.
- Toy kernel sources: `docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp`, `verify_correctness.cpp`, `microbench_contention.cpp`.
- Decisions: DEC-008 (parity envelope) · DEC-011 (32 KB TG ceiling) · DEC-014 (A1 BATCH_DOCS=64) · DEC-015 (col-major, REJECTED) · DEC-016 (T1 fuse-valid, Sprint 19) · DEC-017 (T3b atomic-CAS, Sprint 20 draft).
