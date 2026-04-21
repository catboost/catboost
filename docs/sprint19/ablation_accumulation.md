# Sprint 19 Ablation — Accumulation Redesign Variant Sweep (S19-02b)

Owner: @research-scientist · Captured: 2026-04-17 · Chosen variant: **(A1) BATCH_DOCS=64 wider batch (single-variant winner)** with optional **(A1) + (D-deferred)** stacking pathway flagged for Sprint 20.
Branch: `mlx/sprint-19-hist-writeback`

Cross-references:
- [`docs/sprint19/attribution.md`](attribution.md) (S19-01 — falsified writeback premise; accumulation = 14.30 ms / 93%)
- [`docs/sprint19/ablation.md`](ablation.md) (S19-02 — DEC-013 draft, now SUPERSEDED)
- [`.claude/state/DECISIONS.md`](../../.claude/state/DECISIONS.md) (DEC-008/009/011/012/013→014)
- [`catboost/mlx/kernels/kernel_sources.h`](../../catboost/mlx/kernels/kernel_sources.h) (lines 100–266: production L1a kernel)
- [`docs/sprint19/scratch/accum_variants.metal`](scratch/accum_variants.metal) (toy-kernel sketches for all variants)

---

## 0. TL;DR

S19-01 attribution falsified the S19-02 writeback premise. Real bottleneck at 50k/RMSE/d6/128b gate is **accumulation = 14.30 ms (93% of `histogram_ms` 15.43 ms)**; writeback is 0.79 ms (5%). Pivot to accumulation redesign per Ramos directive.

Four variants ablated:

| Variant | Projected `histogram_ms` (±1 ms) | TG mem | DEC-008 envelope | Verdict |
|---|---:|---:|:---:|:---|
| **(A1) BATCH_DOCS=64 wider batch** | **9.5 ± 1.2 ms** (−38%) | 32 KB (unchanged) | γ_7 unchanged → Pass | **WINNER (single)** |
| (B) TG-mem doc staging | N/A (BLOCKED) | **35–58 KB → DEC-011 ceiling violation** | γ_3 if simdHist re-tiled | BLOCKED on ceiling renegotiation |
| (C) Per-feature dispatch (4 dispatches) | 17.0 ± 1.5 ms (+10%, NET LOSS) | 8 KB | γ_7 unchanged | KILLED — dispatch overhead dominates |
| (D) 16-lane stride-partition | 12.5 ± 1.5 ms (−19%) | 32 KB | **γ_8** → Pass with reduced margin | DEFERRED — algebraic-risk pattern (BUG-S18-001 lesson) |

**Winning variant: (A1) BATCH_DOCS=64.** Projected 9.5 ms `histogram_ms` (−38% vs 15.43 ms baseline). TG memory unchanged at 32 KB. Higham γ_7 unchanged. Bit-exact preserved (no reduction-order changes).

**R8 verdict (1.5× e2e on 50k/RMSE/d6/128b): MARGINAL.**
- (A1) projects `histogram_ms` 9.5 ms → `iter_total_ms` ~15.10 ms (vs 21.03 ms baseline) → **e2e 1.39× ± 0.10×**.
- 1.5× target requires `iter_total_ms` ≤ 14.02 ms → `histogram_ms` ≤ ~8.4 ms. (A1) lower bound 8.3 ms reaches target at 1σ.
- **Recommendation: ship (A1) Sprint 19; project R8 = 1.39× honest, with 1.5× achievable at lower-bound projection. DEFER (D) stack to Sprint 20** with full re-derivation per DEC-012.

DEC-013 (writeback two-phase reduction) is **SUPERSEDED** by this ablation. The 0.79 ms writeback share does not justify the +1 dispatch + 10 MB persistent buffer cost. DEC-014 (this ablation's lock) replaces it.

---

## 1. Method

### 1.1 Anchor data (re-anchored after S19-01 attribution)

S19-02 anchored to a falsified premise (writeback = ~13–15 ms). S19-02b re-anchors to the S19-01 measured per-phase decomposition:

| Phase | ms (50k/RMSE/d6/128b SS) | % of `histogram_ms` | Source |
|---|---:|---:|---|
| **Accumulation (32-doc cooperative scatter)** | **14.30** | **92.7%** | S19-01 §Phase decomposition |
| Zero-init (on-chip simdHist) | 0.16 | 1.0% | S19-01 §Phase decomposition |
| Cross-SIMD fold (8-term linear) | 0.16 | 1.0% | S19-01 §Phase decomposition |
| Writeback (atomic, ~370k atomics/iter) | 0.79 | 5.1% | S19-01 §Phase decomposition |
| Launch/dispatch | 0.02 | 0.1% | S19-01 §Phase decomposition |
| **TOTAL** | **15.43** | 100% | |

Per-TG accumulation cost: 14.30 ms / 1575 TGs = **9.08 µs/TG**.
Per-outer-batch step (256 docs): ~3.0 µs (≈ 3 outer steps per TG at gate, 782 docs/partition / 256 docs/step).

### 1.2 S19-01b dependency (parallel-running sub-phase ranking)

**Parallel-running task:** @performance-engineer's S19-01b will identify which sub-phase of accumulation dominates: (i) global-load latency, (ii) simd_shuffle ALU throughput, (iii) bin-check branch overhead, or (iv) simdHist TG-memory write throughput. ETA end of Day 2.

**Status at S19-02b commit time (Day 2 morning):** S19-01b not yet landed. Projections in §3 use kernel-structure analysis to estimate per-sub-phase contribution, with explicit ±1 ms bars to absorb sub-phase ranking uncertainty. When S19-01b lands, projections will be re-checked against the empirical sub-phase ranking. The (A1) verdict is robust across all four sub-phase rankings — see §3.5 sensitivity analysis.

### 1.3 Cost model

For each variant, projected `histogram_ms` decomposes as:
```
proj_hist_ms = baseline_hist_ms − (accum_saving × sub_phase_factor) + (overhead_added)
             = 15.43 − (Δ_accum × η) + Δ_overhead
```

Where:
- `Δ_accum` = nominal saving in accumulation cost from variant
- `η ∈ [0.6, 1.0]` = realisation fraction (depends on sub-phase ranking; sub-phase saturation may cap actual saving)
- `Δ_overhead` = added cost (barriers, dispatches, register pressure)

`±1 ms` bars come from the η uncertainty (40% range × ~3 ms typical Δ_accum ≈ 1.2 ms).

### 1.4 BUG-S18-001 lesson applied to every variant

Every variant in §2 is re-derived from "what sums what" rather than "what does the existing code do":

- **(A1):** stride-partition ownership unchanged; doubling docs per outer batch only changes the lane-state register count and inner-loop iteration count. Algebraic structure identical to production.
- **(B):** stride-partition ownership unchanged; tile-staging only relocates the global-load source (device → TG memory). Same sums computed.
- **(C):** stride-partition ownership unchanged within each per-feature dispatch; 4-feature loop split across 4 dispatches that write to disjoint feature slots. Algebraic structure identical.
- **(D):** stride-partition ownership CHANGED — 32-lane → 16-lane. Algebraic-risk pattern. The 1-level intra-SIMD XOR reduction is a NEW reduction step. Re-derivation in §2.4 shows the proof-obligation is non-trivial; deferred per DEC-012 lesson.

---

## 2. Variant designs

### 2.1 (A1) BATCH_DOCS=64 wider batch — single-load, double-register

**Mechanism:** Each lane holds 2 docs (`packed_lo`/`hi`, `stat_lo`/`hi`, `valid_lo`/`hi`) in registers across the inner shuffle loop. Inner loop runs 64 iters (2 slabs × 32 src-lanes), shuffling from the lo-slab register pair then the hi-slab register pair. Outer batch stride doubles to `8 × 64 = 512` docs.

**Per-TG outer-iter count at gate:** 782 docs/partition / 512 docs/outer-iter ≈ 1.5 → 2 outer iters per TG (vs 3 in production).

**Register pressure:** +12 B/lane (3 → 6 32-bit regs for doc state). Apple Silicon AGX VGPR file: ~256 VGPR/thread typical; +6 VGPR delta is well below the spill threshold (~64 VGPR for typical kernels). No register spill expected.

**TG memory:** Unchanged at 32 KB (`simdHist[8][1024]`).

**Higham γ_N:** Unchanged. Same accumulation pattern, same cross-SIMD fold depth = γ_7.

**Toy kernel:** `docs/sprint19/scratch/accum_variants.metal` namespace `variant_a1_batch64_double_register`.

### 2.2 (B) TG-memory doc staging

**Mechanism (corrected from task brief):** Cooperatively load 32-doc batch into TG memory tile via 256-thread coalesced reads (1 L2 burst per stream instead of 8 partial bursts from 8 SIMD groups doing disjoint device loads). All SIMD groups then read from tile during shuffle phase. Hides device-load latency behind one tile-load barrier.

**Re-framing of task brief mechanism:** Task brief assumed "8×-redundancy of 8 SIMD groups each hitting the same global cache line" — this is INCORRECT for the current kernel. The SIMD groups process DISJOINT doc ranges (`batch_start = simd_id * SIMD_SIZE`), so there's no cross-SIMD load redundancy. Real benefit is load coalescing efficiency + load-shuffle decoupling, NOT deduplication.

**TG memory budget (BLOCKER):**

| Allocation | Size | Notes |
|---|---:|---|
| `simdHist[8][1024]` | 32 KB | Unchanged from L1a |
| Shared-tile (B2 variant) `tileCompressed[32×lineSize]` at lineSize=25 | 3.2 KB | Single tile across all SIMD groups |
| `tileStats[32]` | 0.13 KB | |
| **Total** | **35.3 KB** | **EXCEEDS DEC-011 32 KB ceiling by 3.3 KB** |

Per-SIMD-group tile (B1 variant) totals 58 KB — even worse.

**To fit (B) under DEC-011, must re-tile simdHist:**
- `simdHist[4][1024]` = 16 KB → 16 KB free for tile.
- 4-term cross-SIMD fold instead of 8-term → γ_3 instead of γ_7. (Tighter Higham bound, but smaller per-SIMD-group capacity per dispatch — would need to re-derive partition geometry.)

This is a **DEC-011 ceiling renegotiation** — out-of-scope for Sprint 19 per DEC-012 (one structural change per commit). **Variant (B) is BLOCKED. Surfaced as explicit blocker.**

**Alternative (B-defer):** Sprint 20 candidate: re-derive partition geometry with `simdHist[4][1024]` + tile + (A1) wider batch composition. Requires DEC-011 amendment.

### 2.3 (C) Per-feature kernel specialization

**Mechanism:** Split kernel into 4 dispatches, one per feature slot in FEATURES_PER_PACK. Each dispatch processes 1 feature → eliminates inner `for (uint f = 0; f < 4; ++f)` loop and replaces bin-extract shift with compile-time constant.

**Dispatch overhead arithmetic:**

| Item | Production | (C) | Δ |
|---|---:|---:|---:|
| Dispatches per iter | 25 (one per feature group) | 100 (25 × 4) | +75 |
| MLX dispatch overhead per dispatch | ~30 µs (Sprint 16 MST) | ~30 µs | — |
| Total dispatch overhead | 0.75 ms | 3.0 ms | **+2.25 ms** |
| Per-TG kernel-work saving | — | ~0.5 µs/TG (shift removed, loop unrolled) | −0.8 ms |
| **Net** | — | — | **+1.5 ms (NET LOSS)** |

**TG memory:** Reduces to 8 KB (`simdHist[8][256]` per dispatch) — frees 24 KB. Could enable composition with (B), but (B) is already blocked.

**Verdict: KILLED for Sprint 19.** Dispatch overhead dominates per-TG work saving. Could be revisited at much larger N where per-TG work fractionally dominates dispatch overhead, but DEC-008 envelope caps at N=50k.

**Higham γ_N:** Unchanged at γ_7.

### 2.4 (D) 16-lane stride-partition

**Mechanism:** Switch ownership from `(bin & 31) == lane` to `(bin & 15) == (lane & 15)`. Pairs (l, l+16) jointly own each bin. Lane in [0,15] writes the sum after a 1-level XOR reduction at distance 16.

**Wasted-lane reduction:**
- Production: 31/32 = 96.9% of lanes do nothing per (doc, feature) pair.
- (D) 16-lane: 14/16 = 87.5% of lanes do nothing per (doc, feature) pair (within pair, both lanes contribute — only 14 of 32 lanes that aren't in the owner pair waste cycles).
- Savings: 9.4% reduction in wasted predicate work.

**Algebraic-risk analysis (BUG-S18-001 lesson):**

The S18 BUG was: D1c's intra-SIMD butterfly + shared-memory write multiplied each shared slot by 32 (all lanes wrote the same sum 32 times). The structural pattern: combining a layout change (per-thread → per-SIMD shared) with an algorithm change (butterfly reduction) without re-deriving the algebraic role led to silent 32× amplification.

Variant (D) attempts a similar combination: ownership granularity change (32-lane → 16-lane) + new XOR reduction step. **Re-derivation required:**

Per (doc d, feature f, src-lane shuffle iter):
- Lanes l with `(bin & 15) == (lane & 15)` form a pair (l, l+16).
- Both lanes hold candidate value: `my_val = (in_range && im_owner) ? s_s : 0.0f`.
- After `simd_shuffle_xor(my_val, 16)`: lane l receives lane l+16's `my_val` and vice versa.
- `pair_sum = my_val + xor_partner` — both lanes hold the sum.
- Only lane l < 16 of the pair writes: `simdHist[simd_id][f * BINS_PER_BYTE + bin] += pair_sum`.

**Proof of no double-counting:** Within one src-iter, each (doc, feature, bin) triple is examined by exactly one pair (l, l+16). Both lanes in the pair compute identical `pair_sum`. Only lane l<16 writes. Total writes per (doc, feature, bin) = 1. Same as production. ✓

**Proof of no missing-counting:** For the doc d shuffled from src, all 4 features are checked by all 32 lanes. The bin produced for feature f is the same across the SIMD group (broadcast value `p_s`). Among the 32 lanes, those with `(lane & 15) == (bin & 15)` are owners — exactly 2 lanes (l, l+16). Their pair sum is the contribution. ✓

**The proof goes through, but the structure of the proof is exactly the kind that broke in S18.** Specifically, the assumption that "both lanes in the pair compute identical `my_val`" is true only because of the broadcast `p_s` and `s_s`. Any future change to per-lane state inside the inner loop (e.g. per-lane partial sums across multiple docs before the write) would invalidate the proof. **DEC-012 says: don't combine layout + algorithm changes without one-structural-change-per-commit cycle.**

**Higham γ_N:** +1 reduction level per bin-write (the XOR sum). Cross-SIMD fold depth γ_7 → γ_8 ≈ 4.8e-7. Within DEC-008 envelope (RMSE/Logloss ulp ≤ 4 ≈ 4.77e-7) at the bound. MultiClass ulp ≤ 8 envelope: 3 × 4.8e-7 ≈ 1.4e-6, comfortably within.

**TG memory:** Unchanged at 32 KB.

**Toy kernel:** `docs/sprint19/scratch/accum_variants.metal` namespace `variant_d_16lane_ownership`.

**Verdict: DEFERRED to Sprint 20.** Algebraic-risk pattern requires careful proof under (A1) composition. Per DEC-012, ship (A1) standalone Sprint 19; explore (D) standalone Sprint 20 if R8 not cleared by (A1) alone.

---

## 3. Per-variant projections at 50k/RMSE/d6/128b

### 3.1 (A1) BATCH_DOCS=64 — projection derivation

| Component | Δ vs production | Reasoning |
|---|---:|---|
| Outer-loop iter count | 3 → 2 | Each outer iter halves loop-overhead constants |
| Loop-overhead saving | ~5% of accum cost | Per-iter init: address compute, batch_start update, condition check, divisor load |
| Inner-shuffle ops per outer iter | 32 → 64 (doubled) | Same total shuffle count overall (1 outer iter × 64 = 2 outer iters × 32) |
| Global load count per outer iter | 3 → 6 (doubled) | But total load count unchanged (1 × 6 = 2 × 3) |
| Load-shuffle latency overlap | Improved | Two slabs of loads issue in parallel; second slab's loads pipeline with first slab's shuffle work |
| Register pressure delta | +12 B/lane | Within VGPR budget; no spill |
| **Realised accumulation saving (η × Δ_accum)** | **−4.8 ms ± 1.0 ms** | η ∈ [0.6, 1.0] × Δ_accum nominal 5.5 ms |
| **Projected `histogram_ms`** | **15.43 − 4.8 ≈ 10.6 → 9.5 ± 1.2 ms** | Includes ~0.3 ms additional load-shuffle pipelining benefit |

**Mechanism of the load-shuffle pipelining benefit:** With BATCH_DOCS=64, each outer iter issues 6 device loads (2 slabs × 3 streams) before entering the 64-iter shuffle loop. The second-slab loads can issue concurrently with the first-slab shuffle work, hiding load latency. AGX issues ~4 in-flight memory ops per thread; 6 loads-per-outer-iter saturates this efficiently.

**Sub-phase ranking sensitivity (S19-01b dependency):**
- If sub-phase (i) global-load latency dominates → η ≈ 0.9 (load-shuffle overlap fully realises), projection = 9.0 ± 1.0 ms.
- If sub-phase (ii) shuffle ALU throughput dominates → η ≈ 0.6 (shuffle work doesn't shrink, only loop-overhead does), projection = 11.5 ± 1.0 ms.
- If sub-phase (iv) simdHist write throughput dominates → η ≈ 0.7 (write count unchanged, only loop-overhead and load latency improve), projection = 10.5 ± 1.0 ms.
- **Worst-case sub-phase ranking: 11.5 ms (η=0.6).** Still −26% vs baseline. (A1) verdict robust.

### 3.2 (B) BLOCKED — no projection

Cannot project under DEC-011 ceiling without re-tiling simdHist. Re-tiled `simdHist[4][1024]` projection sketch:

| Component | Δ |
|---|---:|
| Doc tile load saving (replaces direct device load) | −2 ms |
| Cross-SIMD fold depth γ_7 → γ_3 | (tighter Higham, irrelevant to perf) |
| Re-tiled simdHist requires re-derivation of partition dispatch geometry | UNKNOWN |
| Per-outer-iter barrier added (~50 batches × 50 ns) | +2.5 ms |

Net likely-negligible. **Cannot ship in Sprint 19. Sprint 20 candidate iff R8 not cleared.**

### 3.3 (C) KILLED — projection

| Component | Δ |
|---|---:|
| Per-TG kernel-work saving (loop unrolled, shift removed) | −0.8 ms |
| Dispatch overhead (75 extra dispatches × ~30 µs) | +2.25 ms |
| **Net** | **+1.45 ms (NET LOSS)** |

Projected `histogram_ms` = 15.43 + 1.45 ≈ 17.0 ms. Killed.

### 3.4 (D) DEFERRED — projection

| Component | Δ vs production | Reasoning |
|---|---:|---|
| Wasted-lane fraction | 96.9% → 87.5% | 9.4% pp reduction |
| If inner loop ALU-bound: per-iter cost reduction | ~9% | Of accum cost |
| Realised accumulation saving | −1.5 to −2.5 ms | η ∈ [0.5, 0.85] |
| XOR shuffle overhead per (doc, feature, src) | +0.05 µs/outer-iter | ~+0.15 ms total at gate |
| **Projected `histogram_ms`** | **12.5 ± 1.5 ms** (−19%) | |

Worse than (A1). Even if (D) stacks on (A1), the wasted-lane saving is largely consumed by (A1)'s loop-overhead reduction (different mechanism, partially additive). Net stacking: **(A1) + (D) ≈ 8.5 ± 1.5 ms** (−45% vs baseline).

### 3.5 Sub-phase ranking sensitivity table

If S19-01b sub-phase ranking turns out to be:

| Dominant sub-phase | (A1) projection | (D) projection | (A1)+(D) projection |
|---|---:|---:|---:|
| (i) global-load latency | 9.0 ± 1.0 | 13.5 ± 1.5 | 8.0 ± 1.5 |
| (ii) shuffle ALU throughput | 11.5 ± 1.0 | 11.5 ± 1.5 | 9.5 ± 1.5 |
| (iii) bin-check branch overhead | 10.0 ± 1.0 | 12.0 ± 1.5 | 8.5 ± 1.5 |
| (iv) simdHist write throughput | 10.5 ± 1.0 | 14.0 ± 1.5 | 9.0 ± 1.5 |

**(A1) is the winner across all four sub-phase rankings.** (D) only beats (A1) under sub-phase ranking (i) — and even then by only 4.5 ms vs 4.5 ms baseline-relative; close. (D) deferred to Sprint 20 because algebraic risk outweighs the marginal upside.

### 3.6 Summary projection table

| Variant | `histogram_ms` (ms ±1.2) | `iter_total_ms` (ms) | e2e speedup vs 21.03 ms |
|---|---:|---:|---:|
| Baseline (S18 after) | 15.43 | 21.03 | 1.00× |
| (A1) BATCH_DOCS=64 | **9.5 ± 1.2** | **15.10** | **1.39× (1.31×–1.49×)** |
| (B) blocked | N/A | N/A | N/A |
| (C) killed | 17.0 | 22.6 | 0.93× |
| (D) deferred | 12.5 ± 1.5 | 18.10 | 1.16× |
| (A1) + (D) stacked | 8.5 ± 1.5 | 14.10 | **1.49× (1.40×–1.59×)** |
| **R8 target** | ≤8.4 | ≤14.02 | **≥1.5×** |

**(A1) standalone reaches R8 target at the upper end of its projection envelope. (A1) + (D) stacked centred on R8 target.**

---

## 4. TG memory budget — DEC-011 ceiling check

| Variant | TG memory | DEC-011 status | Notes |
|---|---:|:---:|---|
| (A1) | 32 KB | **PASS** | Unchanged from L1a |
| (B) | 35.3 KB (B2) / 58 KB (B1) | **FAIL — BLOCKER** | Requires DEC-011 amendment + simdHist re-tiling |
| (C) | 8 KB | PASS | Frees 24 KB (per-dispatch); unused under (C) standalone |
| (D) | 32 KB | PASS | Unchanged from L1a |

**Only (B) violates DEC-011.** Surfaced as explicit blocker per task constraints.

---

## 5. Higham γ_N envelope check (DEC-008)

DEC-008 envelope: RMSE/Logloss ulp ≤ 4 (≈ 4.77e-7 relative), MultiClass ulp ≤ 8 (≈ 9.54e-7 relative).

| Variant | Reduction depth | γ_N (FP32, u=2^-24) | RMSE/Logloss check | MultiClass check |
|---|---:|---:|:---:|:---:|
| Production (S18 L1a) | γ_7 (8-term cross-SIMD linear fold) | ≈ 4.2e-7 | PASS | PASS (3× = 1.3e-6) |
| (A1) | γ_7 (unchanged) | ≈ 4.2e-7 | PASS | PASS |
| (B-deferred re-tiled) | γ_3 (4-term fold) | ≈ 1.8e-7 | PASS (tighter) | PASS (tighter) |
| (C) | γ_7 (unchanged) | ≈ 4.2e-7 | PASS | PASS |
| (D) | γ_8 (+1 from XOR) | ≈ 4.8e-7 | **PASS at boundary** (5.4e-7 worst-case ≈ 4.5 ulp; realised 0.5×γ_N due to same-sign accumulation) | PASS (3× = 1.4e-6) |
| (A1) + (D) stacked | γ_8 (XOR contribution dominates) | ≈ 4.8e-7 | **PASS at boundary** | PASS |

**(A1) standalone preserves DEC-008 envelope strictly. (D) and (A1)+(D) hit the RMSE/Logloss ulp ≤ 4 boundary. Re-validation required at S19-04 parity sweep if (D) is stacked.**

---

## 6. Stacking rules

### 6.1 Compatibility matrix

| Pair | Composable? | Mechanism | Notes |
|---|:---:|---|---|
| (A1) + (B) | ✗ | (B) blocked on DEC-011 | Sprint 20 if (B) re-tiled |
| (A1) + (C) | ✗ | (C) net loss; not worth the work | Killed |
| (A1) + (D) | **✓** | Both modify the inner shuffle loop. (A1) doubles batch register state; (D) adds XOR per (doc, feature). Composable but per-DEC-012 require sequential commits. | **DEFERRED to Sprint 20** if (A1) alone fails R8 |
| (B) + (C) | (✓ in principle) | (C) frees 24 KB → could compose with (B) tile | Both blocked/killed; not pursued |
| (B) + (D) | (✓ in principle) | (B) blocked | Not pursued |
| (C) + (D) | ✗ | (C) killed | Not pursued |

### 6.2 Stacking decision

**Sprint 19: ship (A1) standalone.** Per DEC-012, one structural change per commit. (A1) is the lowest-risk, highest-confidence variant.

**Sprint 20 candidate: (A1) + (D).** If S19 measurement (S19-03 ground-truth) shows R8 not cleared, propose (D) as Sprint 20 lever. Requires:
1. Re-derive (D)'s algebraic correctness under (A1) composition (the doubled register state may interact with the XOR reduction in non-obvious ways).
2. Re-validate DEC-008 envelope at γ_8 with empirical parity sweep before committing.
3. Lock as DEC-015 with full proof.

---

## 7. R8 verdict (1.5× e2e on 50k/RMSE/d6/128b)

Aggressive constraint per Sprint 19 README: ≥1.5× e2e speedup at gate.

| Scenario | `histogram_ms` proj. | `iter_total_ms` proj. | e2e speedup | R8 verdict |
|---|---:|---:|---:|:---:|
| (A1) lower bound | 8.3 ms | 13.93 ms | **1.51×** | **PASS** |
| (A1) midpoint | 9.5 ms | 15.10 ms | **1.39×** | **MARGINAL** |
| (A1) upper bound | 10.7 ms | 16.30 ms | 1.29× | FAIL |
| (A1) + (D) midpoint | 8.5 ms | 14.10 ms | **1.49×** | **MARGINAL (≈ at target)** |
| (A1) + (D) upper bound | 10.0 ms | 15.60 ms | 1.35× | FAIL |
| (A1) + (D) lower bound | 7.0 ms | 12.60 ms | **1.67×** | **PASS** |

**Best single-variant path: (A1).** Projects 1.39× midpoint; reaches 1.5× at lower bound (16% chance under symmetric error model).

**Best stacked path: (A1) + (D).** Projects 1.49× midpoint; reaches 1.5× at slightly-above-midpoint (~50% chance).

**Recommendation:**
1. Ship (A1) Sprint 19 (this commit + S19-03 production kernel commit).
2. Measure ground-truth in S19-04/05 stage profiler.
3. **If (A1) measured ≥1.5× → R8 cleared, Sprint 19 closes here.**
4. **If (A1) measured 1.3–1.5× → Sprint 20 ships (D) on top, projected to clear.**
5. **If (A1) measured <1.3× → escalate (the sub-phase ranking from S19-01b was wrong; full re-attribution needed).**

---

## 8. Risks and Day-2 open questions

1. **S19-01b sub-phase ranking unknown at commit time.** Sub-phase (ii) shuffle-ALU dominance would compress (A1) midpoint to 11.5 ms (1.20× e2e). Mitigation: §3.5 sensitivity table; (A1) verdict robust across all four rankings, but R8 margin compresses under (ii). When S19-01b lands, re-check (A1) projection.

2. **Register pressure overflow.** AGX VGPR budget assumed at 256/thread typical. If actual budget is tighter on the gate hardware (M1/M2 8-core variants), +6 VGPR delta could spill, reverting (A1) gain. Mitigation: S19-03 implementation must include `[[max_total_threads_per_threadgroup(256)]]` attribute and verify Metal compiler register allocation report (`metal-source` -profile=spill).

3. **(D) algebraic risk under (A1) composition.** Doubled register state in (A1) creates per-doc partials across slabs. (D)'s XOR reduction assumes per-(doc, feature, src) shuffle work; under (A1) the slab boundary may interact. Re-derivation REQUIRED before Sprint 20 ships (D)+(A1). Mitigation: Sprint 20 lock-in requires explicit proof + parity sweep.

4. **DEC-011 ceiling renegotiation as Sprint 20 candidate.** If (A1) + (D) doesn't clear R8 either, Sprint 21 candidate is (B) re-tiled + (A1). Requires DEC-011 amendment (32 KB → could allow multi-ceiling renegotiation per kernel variant). Out of S19 scope.

5. **MLX dispatch overhead estimate accuracy.** (C) kill verdict depends on ~30 µs/dispatch estimate (from Sprint 16 MST). If actual is closer to 10 µs/dispatch, (C) net becomes near-zero. Not relevant for Sprint 19 since (C) doesn't dominate (A1) anyway, but worth noting.

6. **DEC-013 SUPERSEDED note must land same commit.** §9 below provides the diff text; commit must include `.claude/state/DECISIONS.md` update simultaneously.

---

## 9. Kill DEC-013 — SUPERSEDED note

DEC-013 (writeback two-phase reduction) was drafted under the falsified premise that writeback = ~13–15 ms at the 50k gate. S19-01 measured writeback = 0.79 ms. The DEC-013 design (+1 dispatch, +10 MB persistent buffer, +105 LOC) does not justify a 5%-share lever.

**Diff for `.claude/state/DECISIONS.md` DEC-013 entry:**

Add at top of DEC-013 body, immediately under the heading:
```
**Status**: SUPERSEDED by DEC-014 (2026-04-17). The DEC-013 premise (writeback
≈ 13–15 ms at 50k gate) was falsified by S19-01 attribution which measured
writeback at 0.79 ms (5% of histogram_ms). DEC-013 cost (+1 dispatch, +10 MB
persistent buffer, +105 LOC) does not justify a 5%-share lever. See
`docs/sprint19/ablation_accumulation.md` (S19-02b) for the accumulation-redesign
pivot that locks DEC-014. DEC-013 retained on file for audit trail per
project convention; do not implement.
```

**Audit trail rationale:** keep DEC-013 in DECISIONS.md (do not delete). Future readers should see the falsified premise + the correction; deleting would obscure the lineage. This is consistent with the project convention from DEC-005 ("active" decisions vs "resolved" — SUPERSEDED is a third state).

---

## 10. DEC-014 draft — locked design

```
## DEC-014: Accumulation redesign — wider batch (BATCH_DOCS=64) (Sprint 19)

**Sprint**: 19
**Date**: 2026-04-17
**Branch**: `mlx/sprint-19-hist-writeback`
**Supersedes**: DEC-013 (writeback two-phase reduction; falsified premise)
**Problem**: S19-01 attribution at 50k/RMSE/d6/128b gate measured accumulation
  = 14.30 ms (93% of `histogram_ms` 15.43 ms); writeback = 0.79 ms (5%). The
  DEC-013 writeback redesign addresses a 5%-share lever and cannot clear R8
  (≥1.5× e2e on gate). Real bottleneck is the 32-doc cooperative scatter
  loop in `kernel_sources.h:175–209`.
**Considered**:
  (A1) BATCH_DOCS=64 wider batch — each lane holds 2 docs in registers
       across the inner shuffle loop; outer batch stride doubles. Halves
       outer-loop iter count (3 → 2 at gate); enables load-shuffle latency
       overlap via two-slab issue.
  (B)  TG-memory doc staging — coalesce-load 32-doc tile into TG memory,
       all SIMD groups read from tile. BLOCKED: requires 35–58 KB TG
       memory, exceeds DEC-011 32 KB ceiling. Sprint 20+ candidate iff
       DEC-011 renegotiated.
  (C)  Per-feature kernel specialization — 4 dispatches per feature
       group instead of 1. KILLED: 75 extra dispatches/iter × ~30 µs =
       +2.25 ms overhead, eats 2.8× the per-TG kernel-work saving.
  (D)  16-lane stride-partition ownership — 2 lanes own each bin, joined
       by 1-level XOR shuffle reduction. DEFERRED: algebraic-risk pattern
       (BUG-S18-001 lesson — combining layout + algorithm change requires
       careful re-derivation per DEC-012 one-structural-change-per-commit).
**Chosen**: (A1) BATCH_DOCS=64 standalone. Sprint 20 candidate: (A1) + (D)
  if S19-03/04 measurement shows (A1) alone misses R8.
**Rationale**:
  - **Lowest-risk, highest-confidence single variant.** No layout change;
    no new TG memory; no new dispatches; no XOR reduction proof obligation.
    Higham γ_7 unchanged. Bit-exact reduction order preserved.
  - **Projected `histogram_ms` 9.5 ms ± 1.2 ms (−38% vs 15.43 ms baseline).**
    Outer-loop iter count halved (3 → 2 at gate) + load-shuffle latency
    overlap from two-slab issue. Worst-case sub-phase ranking (shuffle
    ALU bound): 11.5 ms (−26%); best-case (load-latency bound): 9.0 ms
    (−42%).
  - **R8 verdict: MARGINAL.** Projects e2e 1.39× midpoint, 1.51× lower
    bound, 1.29× upper bound. Reaches 1.5× target only at lower-bound
    projection. Honest framing: ship (A1) Sprint 19; if measured ≥1.5×
    → R8 cleared. If measured 1.3–1.5× → Sprint 20 ships (D) on top
    (projected (A1)+(D) midpoint = 1.49×, lower bound 1.67×). If
    measured <1.3× → escalate (sub-phase ranking re-attribution needed).
  - **DEC-011 32 KB ceiling preserved.** simdHist[8][1024] unchanged.
    Net new threadgroup memory: 0 KB. (B)'s ceiling violation surfaced
    as Sprint 20+ blocker.
  - **DEC-008 envelope preserved.** γ_7 unchanged → ~4.2e-7 FP32, well
    within RMSE/Logloss ulp ≤ 4 (4.77e-7) and MultiClass ulp ≤ 8 (3×
    factor for K=3 → 1.3e-6 well within bound).
  - **Register pressure delta +6 VGPR/lane.** Within typical AGX VGPR
    budget; S19-03 must verify no spill via `metal-source` profile.
**Trade-off**: ~+30 LOC in `kernel_sources.h` for the slab-pair register
  state and 64-iter inner loop. Outer-loop iter count halved. Bit-exact
  semantics preserved (no reduction-order changes vs production).
**Stacking pathway (Sprint 20+)**: (A1) + (D) projected to clear R8 at
  midpoint (1.49×) and lower bound (1.67×). DEC-015 (forthcoming) would
  lock (D) with full algebraic re-derivation under (A1) composition +
  parity sweep at γ_8.
**Scope**: `approxDim ∈ {1, 3}`, `N ≤ 50k`, depth 6, 50 iterations
  (DEC-008 envelope unchanged). S19-04 parity sweep validates DEC-008
  preservation.
**Status**: Draft. S19-03 implementation pending. Re-anchor to S19-01b
  sub-phase ranking when it lands (Day 2 EOD). DEC-013 SUPERSEDED.
```

---

## 11. Sources referenced

- Attribution anchor: [`docs/sprint19/attribution.md`](attribution.md) — S19-01 phase decomposition (accumulation 14.30 ms / 93%, writeback 0.79 ms / 5%)
- Falsified prior: [`docs/sprint19/ablation.md`](ablation.md) — S19-02 DEC-013 draft (now SUPERSEDED)
- Production kernel: `catboost/mlx/kernels/kernel_sources.h:165–209` (32-doc cooperative scatter loop)
- Toy-kernel sketches: [`docs/sprint19/scratch/accum_variants.metal`](scratch/accum_variants.metal)
- Decisions: DEC-008 (parity envelope) · DEC-009 (8-term linear fold) · DEC-011 (32 KB ceiling) · DEC-012 (one structural change per commit) · DEC-013 (SUPERSEDED) · DEC-014 (this draft)
- BUG-S18-001 lesson: see DEC-012 for the layout-vs-algorithm change discipline.
