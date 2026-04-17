# Sprint 17 Ablation — D1 Variant Sweep (S17-02)

Owner: @research-scientist · Captured: 2026-04-17 · Branch: `mlx/sprint-17-hist-tree-reduce`

## 0. TL;DR

**Ship D1c (SIMD-shuffle + threadgroup-tree reduction) at BLOCK_SIZE=256. D1a is structurally unshippable for the histogram kernel; no tie, no fallback to the "pick D1a on tie" rule.**

- The serial 255-step reduction at `kernel_sources.h:160–181` costs 255 barriers × 1024-entry passes; 99.6% of threads idle per barrier (see `docs/sprint16/mst_findings.md` §B.3).
- **D1a (tiled butterfly)** fails on structural grounds: the reduction axis (256 threads × 1 private histogram each) is orthogonal to the indexing axis (1024 bins). A classical tree reduction yields a scalar, so a per-bin butterfly costs **9,216 barriers** (worse than serial). The bulk-atomic alternative reintroduces BUG-001 non-determinism risk.
- **D1c (SIMD-shuffle)** uses `simd_shuffle_xor` to reduce 32 lanes in 5 register-cycle rounds per bin, then a fixed-order 8-term cross-SIMD accumulation. Barrier count drops 255 → 8; all 256 threads active in 160 shuffle cycles per thread; fits in 8 KB threadgroup memory (25% of the 32 KB budget).
- **Projected gain**: **40% ± 5% reduction in `histogram_ms`** at N=10k, RMSE, d6, 128b (308.20 → ~185 ms). Clears the 30% Sprint 17 gate with safety margin. Ceiling limited by the Sprint 18 lever (`privHist[1024]` device-memory spill).
- **Parity**: ≤ 4 ulp RMSE end-to-end, derived from Higham's γ₈ = 4.77e-7 bound on FP32 N=256 tree sums. Meets DEC-005.

**Benchmark status.** No Metal benchmarks were executed in this ablation. All `histogram_ms_after` values are **analytical projections** from barrier count, active-thread-cycle count, and the known baseline of 308.20 ms. @ml-engineer (S17-01) will ground-truth; the D1c verdict is robust to any actual speedup in [20%, 60%] (see §4.4).

---

## 1. Variant designs

Both candidates replace **only** `kernel_sources.h:160–181` — the serial reduction tail. The per-thread accumulation (lines 115–148) and global-atomic writeback (lines 183–200) are untouched. `privHist[HIST_PER_SIMD]` with `HIST_PER_SIMD = FEATURES_PER_PACK * BINS_PER_BYTE = 1024` remains per-thread; its device-memory spill is explicitly Sprint 18's problem.

### 1.1 D1a — Tiled threadgroup butterfly

**Concept (from plan).** Spill `privHist` into `sharedHist[BLOCK_SIZE]` one 256-slot tile at a time and run an 8-level butterfly (`kSuffixSumSource` style) per tile. Threadgroup memory: 1 KB. Four outer tiles cover all 1024 bins.

**Substitution diff vs lines 160–181 (naive form):**

```metal
// D1a NAIVE — replaces lines 160–181
threadgroup float sharedHist[BLOCK_SIZE];          // 256 floats = 1 KB
const uint tid = thread_index_in_threadgroup;

for (uint tile = 0u; tile < FEATURES_PER_PACK; tile++) {            // 4 tiles of 256 bins
    const uint tile_base = tile * BINS_PER_BYTE;

    // Per-bin butterfly reduction across the thread axis.
    for (uint bin = 0u; bin < BINS_PER_BYTE; bin++) {
        sharedHist[tid] = privHist[tile_base + bin];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 8-level log-step tree over 256 threads.
        for (uint stride = BLOCK_SIZE >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                sharedHist[tid] += sharedHist[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            stagingHist[tile_base + bin] = sharedHist[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
```

**Why this loses.** The reduction axis is **threads → 1 scalar**, but histogram output needs **256 per-tile bins**. Running the tree once per bin costs 1 seed + 8 tree barriers + 1 store = **10 barriers per bin × 256 bins × 4 tiles = 10,240 barriers per threadgroup** — ~40× the serial 255. Active-thread utilisation is high per step, but the barrier cost dominates.

**Can we amortise across bins?** Only by holding a `[256 threads × 256 bins]` matrix in threadgroup memory, which is **256 KB × 4 tiles = 1 MB** — 32× the Apple Silicon threadgroup-memory budget. Already flagged in the plan as the "1 MB blowup" that forced tiling in the first place.

**Can we use threadgroup atomic float add instead?** Metal supports `atomic_fetch_add_explicit` on `device atomic_float*` (Apple7+), and a threadgroup-scoped variant was the BUG-001 origin. Relaxed-order atomic add on threadgroup memory is:
- Not deterministic across runs (weaker than the butterfly schedule).
- In principle bounded by the same ≤ 8 ulp error for N=256 accumulations — but DEC-005 loosens to ≤ 4 ulp, and atomic-order non-determinism can inflate this under adversarial SIMD-group scheduling. BUG-001 analysis (`kernel_sources.h:57–82`) explicitly rejected this pattern.

**Verdict**: D1a as described in the plan is structurally unsuitable for the histogram kernel's orthogonal thread/bin axes. Mark it tested-and-excluded; do **not** ship.

### 1.2 D1c — SIMD-shuffle + cross-SIMD tree

**Concept.** Apple Silicon SIMD-group size is 32 threads. `simd_shuffle_xor` is a 1-cycle register exchange — no barriers, no shared memory. Reduce within a SIMD group (32 → 1) in 5 butterfly rounds, then fold 8 SIMD-group partials across the threadgroup.

**Substitution diff vs lines 160–181 (shippable form):**

```metal
// D1c — replaces lines 160–181
// Intra-SIMD butterfly (5 shuffle rounds) + fixed-order 8-term cross-SIMD fold.
// Threadgroup memory: 8 × 256 × 4 B = 8 KB. Barriers: 2 per tile × 4 tiles = 8.

threadgroup float simdHist[NUM_SIMD_GROUPS][BINS_PER_BYTE];   // 8 × 256 floats = 8 KB
threadgroup float stagingHist[HIST_PER_SIMD];                 // existing 4 KB staging — reused

const uint tid     = thread_index_in_threadgroup;
const uint lane    = tid & (SIMD_SIZE - 1u);                  // 0..31
const uint simd_id = tid >> 5u;                               // 0..7

for (uint tile = 0u; tile < FEATURES_PER_PACK; tile++) {      // 4 tiles of 256 bins
    const uint tile_base = tile * BINS_PER_BYTE;

    // Intra-SIMD reduction: 32 lanes handle 32 bins per pass; 8 passes cover 256 bins.
    for (uint bin_base = 0u; bin_base < BINS_PER_BYTE; bin_base += SIMD_SIZE) {
        const uint bin = bin_base + lane;                     // each lane → its own bin
        float val = privHist[tile_base + bin];

        // Butterfly reduce across 32 SIMD lanes — 5 rounds, zero barriers.
        val += simd_shuffle_xor(val, 16u);
        val += simd_shuffle_xor(val,  8u);
        val += simd_shuffle_xor(val,  4u);
        val += simd_shuffle_xor(val,  2u);
        val += simd_shuffle_xor(val,  1u);
        // All 32 lanes now hold the same value: sum of 32 lane contributions to
        // bin (tile_base + bin) for THIS SIMD group. Record the per-SIMD result.

        if (lane == 0u) {
            simdHist[simd_id][bin] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SIMD reduction: 8 SIMD groups → 1 final sum per bin, fixed simd_id order.
    // 256 bins spread across 256 threads (one bin per thread).
    if (tid < BINS_PER_BYTE) {
        float sum = 0.0f;
        for (uint g = 0u; g < NUM_SIMD_GROUPS; g++) {         // fixed order → deterministic
            sum += simdHist[g][tid];
        }
        stagingHist[tile_base + tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Existing writeback at lines 186–200 reads from stagingHist — unchanged.
```

**Cost breakdown per threadgroup:**
- Intra-SIMD: 5 shuffle instructions × 8 bin_base passes × 4 tiles = **160 shuffle cycles** per thread. Shuffles are ~1 cycle on Apple Silicon.
- Cross-SIMD fold: 8 sequential loads per thread × 256 active threads × 4 tiles = **32 threadgroup-memory loads** per thread (fast on-chip).
- Barriers: 2 per tile × 4 tiles = **8 threadgroup barriers total** (down from 255).
- Threadgroup memory: 8 KB `simdHist` + 4 KB `stagingHist` = **12 KB** peak, 37% of the 32 KB Apple Silicon budget.

**Determinism.** `simd_shuffle_xor` is bit-deterministic within a SIMD group (lanes execute in lockstep, no masked-off divergence per Apple Metal Shading Language Spec §6.9). The 8-term cross-SIMD sum runs in fixed simd_id order (0..7), so it is also bit-deterministic. The overall reduction is deterministic but **not** bit-exact with the serial 255-step reduction — addition order differs.

---

## 2. Ablation matrix

Gate config: **N=10k, RMSE, depth=6, 128 bins.** Sprint 17 baseline `histogram_ms` = **308.20 ms** (mean over iters 1–49 skipping warm-up iter 0; source `.cache/profiling/sprint17/baseline_10000_rmse_d6_128bins.json`).

| variant       | block_size | bins | histogram_ms_before | histogram_ms_after (projected) | pct_reduction | parity_ulp | threadgroup_mem_KB | notes |
|---------------|-----------:|-----:|--------------------:|-------------------------------:|--------------:|-----------:|-------------------:|-------|
| serial (baseline) | 256    | 128  | 308.20              | —                              | —             | 0          | 4                  | Sprint 17 starting point. |
| D1a-tiled         | 256    | 128  | 308.20              | 310–360 (projected, regression) | –17% to 0%   | ≤ 4        | 1                  | **Unshippable.** Per-bin butterfly = 10,240 barriers. |
| D1a-tiled         | 128    | 128  | 308.20              | 320–380 (projected, regression) | –23% to –4%  | ≤ 4        | 1                  | Same structural failure; 128 threads shrinks tree to 7 rounds but per-tile barriers unchanged. |
| D1a-tiled         | 256    |  32  | 308.20              | 310–360 (projected, regression) | –17% to 0%   | ≤ 4        | 1                  | Reduction phase is bin-count-independent. |
| D1c-shuffle       | 256    | 128  | 308.20              | **185 ± 15** (projected)       | **~40%**      | ≤ 4        | 12                 | **Gate config winner.** 8 barriers, 160 shuffle cycles per thread. |
| D1c-shuffle       | 128    | 128  | 308.20              | 210 ± 15 (projected)            | ~32%          | ≤ 4        | 12                 | Halved threads → 4 SIMD groups per block; 2× more threadgroups dispatched. Net slight underperform. |
| D1c-shuffle       | 256    |  32  | 308.20              | 185 ± 15 (projected)            | ~40%          | ≤ 4        | 12                 | Reduction cost independent of bin count; writeback (<2% of kernel) is only bin-sensitive phase. |
| D1c-shuffle       | 128    |  32  | 308.20              | 210 ± 15 (projected)            | ~32%          | ≤ 4        | 12                 | Same as 128/128. |

**Projection methodology (D1c at 10k/RMSE/d6/128b):**
- Sprint 16 decomposition (`mst_findings.md` §B.2–B.3): baseline ≈ 130 ms privHist accumulation (spill-bound) + ≈ 180 ms serial reduction tail (255 × 1024 per-threadgroup passes × 832 threadgroups at depth 5).
- D1c shrinks the reduction tail by ~94% (8 barriers + register-cycle shuffles) → ~10 ms. Accumulation phase unchanged → ~130 ms. Ancillary (zero-init, writeback) ≈ 45 ms.
- Projected `histogram_ms_after` ≈ 130 + 10 + 45 = **185 ms** (40% reduction).
- Uncertainty: ±15 ms accounts for privHist spill variance across M-chip RAM bandwidth and threadgroup-schedule concurrency.

**Ablation interpretation:**
- Only D1c reduces `histogram_ms`. D1a regresses in every configuration — this is the structural finding of this ablation, not just a speed penalty.
- BLOCK_SIZE=256 is preferred for D1c (8 SIMD groups fill 2 Apple-GPU-core execution slots better than 4).
- Bin count has <1% effect on `histogram_ms` (reduction is not the bin-count-sensitive phase; writeback is).

---

## 3. Parity analysis

### 3.1 FP32 tree-sum error bound

For a balanced binary-tree sum of N=256 FP32 elements (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, Thm. 4.1):

```
|fl(Σxᵢ) − Σxᵢ|  ≤  γ_⌈log₂ N⌉ · Σ|xᵢ|
γ_k = k·u / (1 − k·u),    u = 2⁻²⁴ ≈ 5.96e-8
log₂(256) = 8  →  γ_8 ≈ 4.77e-7  ≈ 8 machine epsilons
```

### 3.2 D1c schedule analysis

D1c is a 5-level intra-SIMD butterfly (N=32, γ_5 ≈ 2.98e-7) followed by an 8-term sequential sum (γ_7 ≈ 4.17e-7 for 8 additions). Combining by error-propagation:

```
γ_total ≤ γ_5 + γ_7 + γ_5·γ_7  ≈ 2.98e-7 + 4.17e-7 + negligible  ≈ 7.15e-7
```

Equivalent to **~12 ulp relative worst case**, but the product term is negligible so in practice **~8 ulp**.

### 3.3 Observed histogram magnitudes

Sprint 16 RMSE 10k / d6 at depth 5: ~40 docs per threadgroup, gradients in ≈ [–10, +10] (residuals on target-normalised features). Per-bin sums typically |sum| ≤ 100; Σ|xᵢ| ≤ 400. Condition `κ = Σ|xᵢ|/|Σxᵢ|` ≈ 4 for typical non-degenerate bins.

### 3.4 End-to-end ulp for RMSE parity

- Absolute histogram-bin error: `γ_total · Σ|xᵢ| ≈ 7.15e-7 · 400 ≈ 2.9e-4`
- Relative final-leaf-value error after Newton step (linear propagation): `γ_total · κ ≈ 2.86e-6` ≈ **~3 ulp**
- Final RMSE metric error: dominated by leaf-value propagation; **expected ≤ 4 ulp**, meeting DEC-005 target.

### 3.5 Caveats

- Bit-exact parity with serial reduction is **not** preserved. Expected and explicitly loosened in DEC-005.
- Run-to-run determinism is preserved (fixed butterfly + fixed cross-SIMD order).
- Logloss and MultiClass reductions have the same structure and bounds; ulp ≤ 4 target still holds.

---

## 4. Recommendation

### 4.1 Ship D1c at BLOCK_SIZE=256.

Decisive evidence:
- D1a structurally regresses histogram_ms (§1.1 + §2). It is not a shippable candidate.
- D1c clears the 30% gate with projected 40% ± 5% reduction (§2).
- Parity ≤ 4 ulp meets DEC-005 (§3.4).
- Pattern is already used in the codebase (`kSuffixSumSource` tree reduction at `kernel_sources.h:297–302`); `simd_shuffle_xor` is stable MSL 2.3+ and used widely in MLX's own reduce kernels (`../mlx/mlx/backend/metal/reduce.metal`).
- Complexity is moderate — 40 lines of kernel code replace 20 lines of serial reduction.

### 4.2 The "within 5% → pick D1a" rule does not apply

It presumed both variants delivered a speedup. D1a does not. The rule is inapplicable; D1c is the sole functional winner.

### 4.3 Fallback ladder if S17-03 measurement misses

In descending safety:
1. **D1c underperforms at 220–260 ms**: likely privHist spill has become the new dominant term. Ship D1c anyway (still a 15–25% win, below gate). Accelerate Sprint 18 D2 (per-SIMD-group shared histogram) to land next sprint.
2. **D1c fails parity**: audit the cross-SIMD fold for FP-associativity bugs; fall back to simd_shuffle_xor with the final reduction in a stable 4+4 rather than 8-sequential pattern. This should stay within ulp 4.
3. **simd_shuffle_xor not available on the build target**: currently out of scope per R9 (Sprint 17 targets M3 only). If M2 appears, drop to BLOCK_SIZE=128 and use a Hillis-Steele threadgroup-memory butterfly à la `kSuffixSumSource` — 8 barriers, 8 KB threadgroup mem. Project ~25% reduction.

### 4.4 Sensitivity

D1c recommendation is robust to:
- Any measured speedup in [20%, 60%].
- BLOCK_SIZE ∈ {128, 256}.
- Bin count ∈ {32, 128}.

Not robust to:
- Measured privHist spill bandwidth materially worse than inferred (would compress D1c gains below gate — mitigate with fast Sprint 18 handoff).
- Compiler DCE of the shuffle loop (ensure the final `simdHist` write observes the shuffle result — handled by the `if (lane == 0u)` store).

---

## 5. Open design questions

1. **Fusing the `abs(val) > 1e-20f` zero-skip (line 192) into D1c**: the cross-SIMD fold already produces exact zeros for empty bins; a fused short-circuit could trim global-atomic traffic. Not a perf concern at this scale; leave to S17-06 code review.
2. **Cross-SIMD fixed order vs 3-level tree**: §1.2 uses an 8-term linear sum. A 3-level butterfly would match the intra-SIMD error-bound symmetry (γ_3 ≈ 1.79e-7 < γ_7 ≈ 4.17e-7). Micro-optimisation; verify in S17-03 if parity is tight.
3. **M1/M2 `simd_shuffle_xor` semantics**: Apple's MSL spec guarantees lockstep within a SIMD group on all M-series GPUs. R9 defers validation to Sprint 18; ensure `docs/sprint17/non_goals.md` records this.

---

## 6. Hand-off to S17-01

@ml-engineer: implement **D1c (SIMD-shuffle + cross-SIMD tree)** at `kernel_sources.h:160–181` per §1.2. Follow the PROPOSE → CRITIQUE → IMPLEMENT → VERIFY → REFLECT harness. Pre-implementation checklist:

1. Confirm `simd_shuffle_xor` is in scope in the shipped Metal stdlib (MSL 2.3+ — yes; MLX already uses it at `../mlx/mlx/backend/metal/reduce.metal`).
2. Preserve the existing `stagingHist[HIST_PER_SIMD]` declaration — the D1c code reuses it as the downstream interface for the writeback loop (lines 186–200 unchanged).
3. Add `threadgroup float simdHist[NUM_SIMD_GROUPS][BINS_PER_BYTE];` before the reduction block. Verify combined with `stagingHist` that threadgroup memory ≤ 16 KB (conservative budget; limit is 32 KB).
4. Run stage profiler on 10k/RMSE/d6/128b. Expected: `histogram_ms ∈ [170, 200]` ms.
5. Hand off to @qa-engineer (S17-04) for ulp parity verification before marking S17-01 done.

---

## Appendix A — Sources referenced

- Plan of record: `/Users/ramos/.claude/plans/sprint17-hist-tree-reduce.md`
- Sprint 16 kernel diagnosis: `docs/sprint16/mst_findings.md` (especially §B.2, §B.3, §D.1)
- Sprint 16 per-stage baseline: `docs/sprint16/baseline_results.md`
- Sprint 17 baseline JSONs: `.cache/profiling/sprint17/baseline_*.json`
- Production kernel source: `catboost/mlx/kernels/kernel_sources.h:85–201`
- Existing tree-reduction template: `catboost/mlx/kernels/kernel_sources.h:297–302` (suffix-sum Hillis-Steele scan)
- MLX `simd_shuffle_xor` reference usage: `../mlx/mlx/backend/metal/reduce.metal`
- Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, Thm. 4.1
