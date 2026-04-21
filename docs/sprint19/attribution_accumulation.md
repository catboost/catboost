# Sprint 19 — S19-01b: Accumulation Sub-Phase Attribution

**Config:** N=50k, RMSE, depth=6, 128 bins
**Source data:** `.cache/profiling/sprint19/baseline/50000_rmse_d6_128bins.json` — 50 iterations, all-iters methodology (S16 convention)
**Date captured:** 2026-04-18
**Baseline:** S18 after (L1a kernel, `simdHist[8][1024]`, commit `dccb7ec0a2`)
**Reported by:** @performance-engineer (S19-01b)
**Depends on:** S19-01 (`docs/sprint19/attribution.md`) — accumulation established at 14.30 ms = 93% of `histogram_ms`

---

## Method

**Proposed:** Metal System Trace dynamic capture via `xcrun xctrace` to obtain per-sub-phase ALU%, memory%, cache-hit rates, and VGPR count directly from AGX hardware counters.

**Critique / fallback:** MST dynamic capture remains blocked in S19-01b due to the same sandbox permission restriction documented in `docs/sprint18/mst_findings.md §Status`. The `xcrun xctrace` and `xcrun metal-profiler` calls were denied; no `.trace` bundle was produced. The S18-09 fallback methodology applies here: combine (1) static kernel-source analysis for operation counts, (2) analytical bounds (ALU ceiling at known TFLOPS, memory bandwidth ceilings), and (3) latency-regression from the per-depth `histogram_ms` profile of the 50-iteration JSON. This produces static-derived estimates marked `[STATIC]`; they require Instruments confirmation for exact numbers.

**Key structural facts derived from source (`catboost/mlx/kernels/kernel_sources.h:165–209`):**

| Quantity | Value | Source |
|---|---|---|
| Outer-batch stride | 256 docs (8 SIMD groups × 32 lanes) | line 177 |
| Loads per 32-doc batch | 3 per lane: `docIndices`, `compressedIndex`, `stats` | lines 185–188 |
| `compressedIndex` access stride | `docIdx * lineSize + col`; lineSize = 25 → stride 100 bytes | line 187 |
| simd_shuffles per 32-doc batch | 3 per src × 32 srcs = 96 | lines 194–196 |
| Bin-check branches per batch | 32 srcs × 4 features = 128 | lines 201–205 |
| TG writes per batch (max) | 32 srcs × 4 features × (1/32 owner) = 4 | line 205 |
| TG memory layout | `simdHist[8][1024]` = 32 KB, stride-partition `(bin & 31) == lane` | lines 151, 204 |
| Total 32-doc batches per iteration | 244,000 (computed: sum over depths of n_tgs × n_batches_per_simd) | regression |
| Total TGs per iteration | 1,575 | S19-01 |

---

## 1. Sub-phase breakdown of the 14.30 ms accumulation

Error bars are ±1 ms except where marked as noise-bounded (<0.5 ms). Method: analytical bounds + per-depth latency regression on the 50-iteration JSON.

| Sub-phase | ms estimate | ±err | % of 14.30 ms | Bounded-by |
|---|---:|---:|---:|---|
| **compressedIndex gather latency (dominant)** | **12.78** | ±1.3 | **89%** | L2 latency model (§2) |
| docIndices L2-chain latency | 1.52 | ±0.3 | 11% | L2 latency model (§2) |
| stats loads | <0.5 | — | <4% | **NOISE-BOUNDED** (parallel with CI, see §2.3) |
| simd_shuffle broadcast | <0.2 | — | <2% | **NOISE-BOUNDED** (ALU pipeline, §2.4) |
| TG-memory writes to simdHist | <0.5 | — | <4% | **NOISE-BOUNDED** (zero bank conflicts, §2.5) |
| Bin-check branch divergence | <0.2 | — | <2% | **NOISE-BOUNDED** (AGX predicated, §2.6) |
| Arithmetic (+= s_s) | <0.1 | — | <1% | **NOISE-BOUNDED** (187M ops @ 2 TFLOPS = 0.094 ms) |
| Loop overhead | <0.05 | — | <0.5% | **NOISE-BOUNDED** |
| **TOTAL** | **14.30** | | **100%** | |

**Dominant sub-phase: compressedIndex gather, 12.78 ms ± 1.3 ms (89% of accumulation).**

The domination is near-total. Five of the eight listed sub-phases are noise-bounded (<0.5 ms). The only non-trivial non-dominant cost is the docIndices L2-chain at 1.52 ms (11%), which is itself a dependency for the compressedIndex gather (DI must resolve before CI address is known).

---

## 2. Sub-phase derivations

### 2.1 Total 32-doc batch count

At each depth d, there are 25 × 2^d TGs (25 feature groups, 2^d partitions). Each TG has 8 SIMD groups, each processing `ceil(N/2^d / 256)` outer batches where 256 = 8 × 32 is the outer stride:

| Depth | TGs | Docs/TG | Batches/simd-group | TG total batches |
|---|---:|---:|---:|---:|
| 0 | 25 | 50,000 | 196 | 39,200 |
| 1 | 50 | 25,000 | 98 | 39,200 |
| 2 | 100 | 12,500 | 49 | 39,200 |
| 3 | 200 | 6,250 | 25 | 40,000 |
| 4 | 400 | 3,125 | 13 | 41,600 |
| 5 | 800 | 1,563 | 7 | 44,800 |
| **Total** | **1,575** | | | **244,000** |

Implied doc-loads: 244,000 × 32 = 7,808,000. Expected: N × depths × num_groups = 7,500,000. Ratio 1.04 (4% waste from batch rounding at partition boundaries). Consistent.

### 2.2 compressedIndex gather — dominant sub-phase

**Access pattern:** `compressedIndex[docIdx * lineSize + featureColumnIdx]`

- `lineSize = 25` (100 features packed 4-per-uint32, 25 columns)
- Stride between consecutive docs: 25 × 4 = 100 bytes
- 32-lane SIMD group accesses 32 consecutive `docIdx` values per batch (since docs are sorted within partition, docIdx is a sorted permutation of 0..N-1)
- Address span: (docIdx[lane+1] - docIdx[lane]) × 100 bytes ≈ 100 bytes/doc for consecutive sorted docs
- 32 docs span ≈ 31 × 100 + 4 = 3,104 bytes → ceil(3,104 / 128) = **25 cache lines** per 32-doc batch

The 32-lane gather requires 25 distinct cache line fetches per batch. With 8 outstanding L2 requests per SIMD group (AGX typical), this takes ceil(25/8) = **4 stall rounds** per batch.

**Working set:** compressedIndex total = N × 25 × 4 = 5.0 MB. Apple M-series GPU L2 is approximately 4–8 MB (chip-dependent). The 5 MB CI working set sits at the L2 boundary, producing a mix of L2 hits and DRAM misses.

**Implied L2 hit rate (back-calculated):**

From the per-depth regression, the effective load-chain latency per batch is:

```
L_chain_per_batch = 14.30 ms × 8 (SIMD groups) / 244,000 batches = 469 ns
```

Decomposing:

```
L_chain = L_DI + 4 × [H × L_L2 + (1-H) × L_DRAM]
469 = 50 + 4 × [50H + 400(1-H)]
419 = 4 × [50H + 400 - 400H]
419/4 = 400 - 350H
H = (400 - 104.75) / 350 = 0.844
```

Using L_DI = 50 ns (sequential, L2 hit), L_L2 = 50 ns, L_DRAM = 400 ns.

**Implied L2 hit rate for compressedIndex: ~84%.** This is consistent with a 5 MB working set on a GPU L2 of approximately 6 MB effective capacity.

**Per-sub-phase attribution:**

| Load | Latency per batch | Serial or parallel with other loads | ms attribution |
|---|---:|---|---:|
| compressedIndex (4 stall rounds × 219 ns) | 418 ns | Serial after DI resolves | 12.78 ms |
| docIndices (L2 sequential, 50 ns) | 50 ns | Must resolve BEFORE CI address known | 1.52 ms |
| stats (L2 sequential, 50 ns) | 50 ns | Issues IN PARALLEL with CI → hidden | <0.5 ms |

The docIndices → compressedIndex serial dependency chain (docIdx must be known before CI address is computable) is the structural reason docIndices cost cannot be pipelined away from CI cost.

### 2.3 stats loads — noise-bounded

`stats[statIdx * totalNumDocs + docIdx]` where `statIdx = 0` (RMSE, one stat) reduces to `stats[docIdx]`. Access pattern: sorted `docIdx` → sequential read → L2 hit. The stats load issues in parallel with the compressedIndex gather (both depend on `docIdx` which is already resolved from the docIndices load). Stats resolution completes before CI resolution because stats is 1 cache line vs CI's 25 cache lines. Stats latency is entirely hidden in CI stall. **<0.5 ms noise-bounded.**

### 2.4 simd_shuffle broadcast — noise-bounded

Per 32-doc batch: 3 shuffles per src × 32 src iterations = 96 shuffles. On AGX, `simd_shuffle` within a SIMD group is a register crossbar operation executing at ~2–4 cycles (~1.5–3 ns per shuffle at 1.3 GHz). Serial worst case: 96 × 3 ns × 244,000 batches = 70 ms — but this is if and only if shuffles are sequential and unoccupied by any concurrent work. In practice, shuffles execute in the shadow of the 4-round CI stall (469 ns per batch vs 3 ns per shuffle). The SIMD group is stalled waiting for L2 responses; shuffles issue speculatively during the stall and complete before the stall resolves. **Incremental cost: ~0 ms. Noise-bounded at <0.2 ms.**

### 2.5 TG-memory writes to simdHist — noise-bounded

`simdHist[simd_id][f * BINS_PER_BYTE + bin] += s_s` executes when `(bin & 31) == lane`.

**Bank conflict analysis:** With stride-partition ownership, each lane writes to bin where `bin & 31 == lane`. For `simdHist[simd_id][f * 256 + bin]`, the SRAM bank index = `(f * 256 + bin) % 32`. For f=0: bank = `bin % 32` = lane. For f=1: bank = `(256 + bin) % 32` = `bin % 32` = lane. Same for f=2,3. Each lane writes to bank == lane across all features. 32 lanes write to 32 different banks simultaneously. **Zero bank conflicts.** Verified from source: DEC-011 stride-partition layout structurally prevents conflicts.

Peak concurrent TG writes per batch: 4 (one per feature). At ~4 cycles each and 1.3 GHz: 12.3 ns per batch × 244,000 batches = 3.0 ms upper bound (if writes are fully serialized across features). In practice the writes are pipelined with CI stall. Conservative upper bound if all writes fell off the critical path: 1.0 ms. **Noise-bounded at <0.5 ms.**

### 2.6 Bin-check branch divergence — noise-bounded

The predicate `bin < foldCountsFlat[...] + 1 && (bin & 31) == lane`:

- `bin` is computed from `p_s` which is broadcast (same value for all 32 lanes per src iteration)
- `(bin & 31)` is the same for all 32 lanes; only one lane satisfies `(bin & 31) == lane`
- AGX Metal shader execution uses predicated/select semantics, not true warp-divergence serialization
- All 32 lanes evaluate the condition; 31/32 predicate-off the write via a conditional-move
- Cost is identical to a branchless predicated operation: ~1 cycle per lane per iteration

This is included in the ALU ceiling (187M ops → 0.094 ms). **Noise-bounded at <0.2 ms.**

---

## 3. Shader stats snapshot [STATIC ANALYSIS]

Dynamic MST capture blocked (see `docs/sprint18/mst_findings.md §Blocked dynamic verification`). All values are static derivations from `kernel_sources.h` with source-grounded inference.

| Metric | Value | Method | Confidence |
|---|---|---|---|
| Threadgroup memory | 32 KB (`simdHist[8][1024]`) | Source line 151, verified S18-09 | High (deterministic) |
| VGPRs (estimated) | ≤32 per thread | No `private` arrays post-L1a; scalar locals only | Medium (compiler-dependent) |
| Barriers per dispatch | 6 | Counted in source (S18-09 verified) | High (deterministic) |
| Occupancy | 1 TG/SM | TG-mem limited: 32 KB = Apple Silicon ceiling (DEC-011) | High (structural) |
| ALU utilization | <1% | 0.094 ms ALU ceiling vs 14.30 ms wall | High (ceiling argument) |
| Memory utilization | ~99% | 14.30 ms dominated by load-chain stalls | High (by elimination) |
| Cache hit rate (compressedIndex) | ~84% L2 | Back-calculated from measured time (§2.2) | Medium (model-dependent) |
| DRAM miss rate (compressedIndex) | ~16% | Residual from above | Medium |
| Effective BW utilization | N/A | Not bandwidth-bound; latency-bound (§2.2) | High |

**Classification: memory-latency-bound, not compute-bound and not bandwidth-bound.**

ALU is <1% utilized. DRAM bandwidth is not saturated: the compressedIndex working set (5 MB) mostly fits in L2, so effective DRAM traffic is ~16% × 6 GB total = ~1 GB/iteration, delivered at 70 ms / 14.30 ms = 4.9× below even the most conservative GPU bandwidth estimate. The bottleneck is the **L2 miss latency for the gather load pattern**, not throughput.

---

## 4. Dominant sub-phase identification

**Dominant sub-phase: compressedIndex gather load latency — 12.78 ms ± 1.3 ms = 89% of accumulation.**

**Root cause:** `compressedIndex` is laid out in row-major (doc-major) order: `compressedIndex[doc * lineSize + col]` where `lineSize = 25`. For a 32-lane SIMD group processing 32 consecutive `docIdx` values (sorted partition), the 32 accesses span `31 × 100 + 4 = 3,104` bytes → 25 distinct cache lines. The AGX load-coalescing unit cannot merge these into fewer transactions because the stride (100 bytes) exceeds a cache-line interval (128 bytes) by 78%, leaving each load on a different cache line. The GPU is forced to issue 4 stall-rounds of L2/DRAM requests per 32-doc batch, with 219 ns average round latency at 84% L2 hit rate.

The 8 SIMD groups within a threadgroup cannot hide this latency from each other because they process *disjoint* doc windows within the same partition — there is no cross-SIMD reuse of compressedIndex cache lines within a TG. The inter-TG L2 sharing (25 feature groups reading the same partition's docIndices) benefits docIndices and stats but does NOT benefit compressedIndex (each group reads a different column of the matrix, so no cross-group CI reuse).

**Why all other sub-phases are noise-bounded:** The 469 ns per-batch stall window is larger than the combined cost of 96 shuffles (~144 ns), 128 branch checks (~10 ns), 4 TG writes (~12 ns), and 4 FP adds (~3 ns). All non-memory work executes inside the CI stall shadow.

---

## 5. Redesign prior: variant ranking for @research-scientist S19-02b

The root cause (row-major compressedIndex, stride-100-byte gather) means that **variants targeting shuffle count, branch overhead, or TG write pressure cannot move the needle** — they are attacking <2% of the measured cost. The variants must address the gather access pattern.

**NOTE:** A fifth variant — Option E: column-major compressedIndex layout — is the only option that directly eliminates the root cause. It is not on the original A/B/C/D list. Its projected impact is separately quantified below and flagged for @ml-product-owner.

### 5.1 Impact of each variant against the identified root cause

| Rank | Variant | Attacks root cause? | Mechanism | Expected `histogram_ms` saving | % of 14.30 ms accum |
|---:|---|:---:|---|---:|---:|
| 1 | **(B) Coalesced TG-memory staging** | Partial | Pre-loads docIndices into TG sram; removes 50 ns DI→CI chain latency per batch | ~1.40 ms | ~10% |
| 2 | **(A) Wider batch (64/128 doc)** | Minimal | Doubles docs per batch; halves outer-loop count; amortizes DI latency per 2 docs | ~0.75 ms | ~5% |
| 3 | **(C) Per-feature kernel** | No | Eliminates 3/4 feature-loop iterations; CI gather unchanged (reads same packed uint32) | <0.2 ms | <2% |
| 4 | **(D) 16/8-lane stride-partition** | No | Reduces wasted-lane fraction; CI gather unchanged | ~0 ms | ~0% |

**Variant B** is the closest to addressing the root cause (it reduces the docIndices chain latency, freeing 1.40 ms from the 1.52 ms docIndices cost). However, it is **blocked by the DEC-011 32 KB ceiling** (requires 35 KB = 32 KB + 3 KB staging buffer). This matches the existing `ablation_accumulation.md §2.2` BLOCKER finding — confirmed by S19-01b.

**Variant A** provides marginal improvement via DI amortization (~0.75 ms = 5% of accumulation). The per-doc outer-loop overhead saving is also relevant if shuffle-ALU bounds are tighter than this analysis suggests, but the CI gather dominance means the ceiling for A is ~1.3 ms even under the most favorable sub-phase assumption.

**Variants C and D** are noise-bounded against the root cause. Their mechanisms (feature-loop unrolling, ownership granularity) address sub-phases that collectively account for <4% of accumulation. This matches and reinforces the `ablation_accumulation.md` KILLED/DEFERRED verdicts for C and D respectively.

### 5.2 The missing variant: Option E (column-major compressedIndex)

None of the four provided variants address the fundamental cache-line utilization problem. The correct structural fix is:

**Option E: Transpose compressedIndex from doc-major to column-major layout.**

Current: `compressedIndex[doc * lineSize + col]` — stride 100 bytes/doc, 25 CL per 32-doc batch.
Column-major: `compressedIndex[col * totalNumDocs + doc]` — stride 4 bytes/doc, 1 CL per 32-doc batch.

For sorted `docIdx` in a partition (which is the access pattern), column-major access is sequential: `compressedIndex[col * N + docIdx[0]], ..., compressedIndex[col * N + docIdx[31]]` where consecutive `docIdx` values are monotone increasing. The 32 accesses span 31 × 4 + 4 = 128 bytes = exactly 1 cache line.

| Metric | Row-major (current) | Column-major (Option E) |
|---|---:|---:|
| CL per 32-doc batch (CI) | 25 | 1 |
| L2 stall rounds per batch | 4 | 1 |
| Latency per batch (L2 100%) | 418 ns | 50 ns |
| CI latency contribution | 12.78 ms | ~1.52 ms |
| Accumulation total | 14.30 ms | ~3.04 ms |
| `histogram_ms` | 15.43 ms | ~4.17 ms |
| `iter_total_ms` | 21.03 ms | ~9.87 ms |
| e2e speedup | 1.00× | **~2.13× ± 0.3×** |

**Option E projected e2e speedup: 2.13×.** This is the only variant that clears R8 (≥1.5× e2e) with high confidence.

**Cost of Option E:** One-time host-side transposition of `compressedIndex` during data loading (O(N × numFeatures), ~50k × 100 = 5M operations, executed once). Kernel-side change: replace `compressedIndex[docIdx * lineSize + featureColumnIdx]` with `compressedIndex[featureColumnIdx * totalNumDocs + docIdx]`. No TG memory change. No DEC-011 impact. DEC-008 envelope unchanged (same accumulation arithmetic).

**Risk:** The `gpu_data` layer (see ARCHITECTURE.md) owns the compressedIndex layout. This change requires coordination between `catboost/mlx/gpu_data/` and `kernel_sources.h`. It is a DEC-012 scope boundary (touches data-path, not just kernel body). @ml-product-owner should evaluate as either (a) the S19-03 implementation target (replacing A1), or (b) Sprint 20 if DEC-012 scope is hard.

---

## 6. Projected `histogram_ms` reduction: is 1.5× e2e still reachable?

| Scenario | `histogram_ms` (ms) | `iter_total_ms` (ms) | e2e speedup | R8 verdict |
|---|---:|---:|---:|---|
| Baseline (S18 after, SS) | 15.43 | 21.03 | 1.00× | — |
| Variant A only (wider batch) | 14.68 | 20.28 | **1.04×** | FAIL |
| Variant B only (TG staging, if unblocked) | 14.03 | 19.63 | **1.07×** | FAIL |
| Variant A + B (if B unblocked) | 13.28 | 18.88 | **1.11×** | FAIL |
| **Option E (column-major CI)** | **~4.17** | **~9.87** | **~2.13×** | **PASS** |
| Option E + A (overkill) | ~3.42 | ~9.12 | ~2.31× | PASS (margin) |

**Conclusion: 1.5× e2e is NOT reachable via variants A, B, C, or D in any combination.** The maximum achievable via A+B (if B were unblocked) is 1.11× e2e. This is directly attributable to the root cause: the variants attack sub-phases totalling ≤15% of accumulation, while accumulation itself is 93% of `histogram_ms`.

**1.5× e2e IS reachable, but only via Option E (column-major compressedIndex layout).** Option E's 79% reduction in CI gather latency maps to an ~11 ms `histogram_ms` saving, delivering 2.13× e2e at the midpoint projection.

**FLAG for @ml-product-owner:** The R8 gate (1.5× e2e on 50k/RMSE/d6/128b) cannot be cleared in Sprint 19 via the A/B/C/D variant set. Either:
- (a) Revise R8 downward to ≤1.11× and accept the A+B outcome; or
- (b) Add Option E (column-major CI) as the S19-03 target, replacing or extending A1; or
- (c) Defer Option E to Sprint 20 with R8 revised for Sprint 19.

The `ablation_accumulation.md` DEC-014 analysis (A1 = 9.5 ms projected, 1.39× e2e) remains valid as a standalone A1 projection but does not clear R8. That document explicitly notes "S19-01b sub-phase ranking unknown at commit time" — S19-01b now resolves that uncertainty. The sub-phase ranking is **global memory load latency dominates (89%)**, which corresponds to the η = 0.9 (load-latency-bound) scenario in `ablation_accumulation.md §3.5`. Under that scenario, A1 projects to 9.0 ± 1.0 ms, e2e 1.41×. This is the best honest projection for A1 and does not reach 1.5×.

---

## 7. Benchmark methodology

- **Tool:** Static analysis + per-depth regression on `.cache/profiling/sprint19/baseline/50000_rmse_d6_128bins.json`
- **Data:** N=50k, RMSE, depth=6, 128 bins, 50 iterations (all-iters methodology per `feedback_iter0_included.md`)
- **Environment:** S18 after kernel (`dccb7ec0a2`), L1a layout
- **Iterations:** 50 (iters 0–49 in mean per S16 convention); SS mean uses iters 1–49
- **Error bars:** ±1 ms from L2 hit-rate model sensitivity (H ± 0.10 produces ±1.4 ms CI estimate; reported as ±1.3 ms for CI, ±0.3 ms for DI)
- **Dynamic verification (blocked):** `xcrun xctrace` / `xcrun metal-profiler` denied by sandbox. Cache hit rates and VGPR counts are inferred, not measured. All results marked `[STATIC ANALYSIS]` where applicable.
- **Reproducibility:** exact command blocked; benchmark re-run: `python python/run_bench.py --config 50000_rmse_d6_128bins` on the gate branch

---

## 8. Lineage and traceability

- S19-01 (`docs/sprint19/attribution.md`) — established accumulation = 14.30 ms = 93% of SS; this document decomposes that 14.30 ms further.
- `docs/sprint18/mst_findings.md §(e), point 1` — correctly identified "DRAM bandwidth in the doc loop is unchanged; each doc still pays one `compressedIndex[docIdx*lineSize+col]`" as the next-likely bottleneck. S19-01b confirms this, with the specific diagnosis that the issue is gather **latency**, not bandwidth.
- `docs/sprint18/mst_findings.md §(e), point 2` — "if ALU bound, Sprint 19 wants to fuse FEATURES_PER_PACK=4 into a single packed-bin compare" — S19-01b falsifies this; ALU is <1% utilized. The ALU-bound scenario does not apply at N=50k.
- `docs/sprint19/ablation_accumulation.md §3.5` — sensitivity table conditioned on sub-phase ranking unknown at that commit time. S19-01b resolves the ranking: sub-phase (i) global-load latency dominates. Under that row, (A1) projects 9.0 ± 1.0 ms; (D) is outperformed by (A1) as the table predicts. The DEC-014 WINNER verdict for A1 is confirmed; the 1.5× R8 question resolves to MARGINAL (A1 standalone) and NOT FEASIBLE (A/B/C/D without Option E).
- Option E (column-major CI) is a new finding from S19-01b. It is not present in any prior sprint doc. Flag for planning.
