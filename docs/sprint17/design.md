# Sprint 17 Design — D1: Log-Step Tree Reduction

## What is being replaced

**File:line:** `catboost/mlx/kernels/kernel_sources.h:160–181`  
**Kernel:** `kHistOneByteSource` (production path — not `hist.metal`, which is dead code)

The current reduction is a serial fold. Thread 0 seeds a `threadgroup float stagingHist[HIST_PER_SIMD]` (4 KB on-chip), then threads 1–255 each add their `privHist[]` into it one at a time:

```metal
// kernel_sources.h:161-181 — BEING REPLACED
threadgroup float stagingHist[HIST_PER_SIMD];  // 4 KB

if (thread_index_in_threadgroup == 0u) {
    for (uint i = 0u; i < HIST_PER_SIMD; i++) stagingHist[i] = privHist[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint t = 1u; t < BLOCK_SIZE; t++) {          // t = 1..255
    if (thread_index_in_threadgroup == t) {
        for (uint i = 0u; i < HIST_PER_SIMD; i++) stagingHist[i] += privHist[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // 255 barriers
}
```

Per-threadgroup cost: 255 barriers + 255 × 1024 = 261,120 float adds, with 99.6% of threads idle at each barrier. If `privHist` spills to device memory (4 KB per thread × 256 threads = 1 MB per threadgroup — almost certain on M-series; see `mst_findings.md` §B.2), each add pulls a float from off-chip. That is ~1 MB of device-memory traffic per threadgroup in the reduction alone.

**Why it was written this way:** The header comment at `kernel_sources.h:57–82` (BUG-001 FIX) records that the original CAS-atomic design was non-deterministic across dispatches. The serial fold was the quickest correctness fix. It is now the dominant cost.

---

## D1 tree reduction algorithm

Replace lines 160–181 with a balanced binary-tree reduction over threadgroup memory.

**Algorithm (8 levels for BLOCK_SIZE=256):**

1. Each thread copies its `privHist` slice into a shared buffer.
2. At level `s` (s = 0..7), threads with `tid < (BLOCK_SIZE >> (s+1))` add `sharedHist[tid + (BLOCK_SIZE >> (s+1))]` into `sharedHist[tid]`.
3. One `threadgroup_barrier(mem_flags::mem_threadgroup)` per level.
4. After 8 levels, `sharedHist[0]` holds the reduced histogram.
5. Write-back to global atomics (`kernel_sources.h:186–199`) is unchanged.

Barrier count drops from **255 to 8**. All 256 threads are active at level 0, 128 at level 1, ... 1 at level 7 — total active work is the same, latency is log-order.

**Determinism preserved:** The stride schedule is fixed at compile time, so each (input, launch geometry) pair produces byte-identical output across runs. Run-to-run determinism is not regressed. The reduction order does change vs. the serial fold — see Numerical Stability below.

**Precedent in the same file:** `kSuffixSumSource` (`kernel_sources.h:264–311`) uses this exact pattern for a 256-element scan. The tree-reduction pattern is already proven correct in this codebase.

---

## Variants under consideration

The naïve 2D shared buffer `sharedHist[BLOCK_SIZE][HIST_PER_SIMD]` = 256 × 1024 × 4 B = **1 MB per threadgroup**, which far exceeds Apple Silicon's 32 KB threadgroup memory limit. Three variants address this.

### D1a — Tiled shared-memory reduction (recommended starting point)

Tile the reduction over `HIST_PER_SIMD` slices. The shared buffer holds one 1024-float slice at a time; an outer loop iterates over slices, and the inner loop is the 8-step tree.

- **Shared memory:** 256 × 1 × 4 B = **1 KB** (well within 32 KB)
- **Outer loop iterations:** `HIST_PER_SIMD / BLOCK_SIZE` = 4
- **Implementation complexity:** Low — textbook tiled reduction
- **Risk:** The outer loop adds 4× the reduction passes; measured gain depends on whether the serial-loop latency or the raw float-add count is the dominant cost

### D1b — Per-feature reduction

Reduce per feature pack (FEATURES_PER_PACK=4 outer iterations), shared buffer `[BLOCK_SIZE][BINS_PER_BYTE]` = 256 × 256 × 4 B = **256 KB**. Still exceeds the 32 KB limit unless further tiled. Listed for completeness; not recommended without additional tiling analysis.

### D1c — SIMD-shuffle + 3-step threadgroup tree

Collapse each 32-thread SIMD group to 8 SIMD-leader values per bin using `simd_shuffle_xor`, then run a 3-step threadgroup tree on the 8 leaders.

- **Shared memory:** 8 × 1024 × 4 B = **32 KB** (at the limit; tiling may still be required)
- **Barrier count:** 3 threadgroup + intra-SIMD cost
- **Implementation complexity:** Higher — SIMD intrinsics, requires careful BUG-001 parity validation
- **Upside:** Lower threadgroup-memory pressure than D1a, potentially better SIMD utilisation

### Variant selection — pending ablation

**D1a is the default implementation target** (lowest risk, proven pattern). D1c is the performance upside.

@research-scientist's S17-02 ablation will sweep {D1a, D1c} × {BLOCK_SIZE 128, 256} × {bins 32, 128} on N=10k RMSE depth=6 and select the variant to ship. Implementation (S17-01) does not lock in until the ablation completes.

See [`ablation.md`](ablation.md) for results and variant selection.

---

## Storage trade-off summary

| Variant | Shared mem | Outer loop | Complexity | Within 32 KB limit? |
|---------|-----------|------------|------------|---------------------|
| Naïve 2D | 1 MB | 1 | Low | No |
| D1a tiled | 1 KB | 4 | Low | Yes |
| D1b per-feature | 256 KB | 4 | Low | No (needs tiling) |
| D1c SIMD+tree | ~32 KB | 1–2 | High | Marginal |

---

## Numerical stability

The tree reduction changes the addition order of the 256 `privHist` values relative to the current serial fold. For FP32 with 256 elements, a balanced binary tree accumulates at most log2(256) = 8 levels of rounding error vs. the serial fold's 255-step chain. The tree is numerically superior to the serial fold in the general case.

In practice, GBDT gradient/hessian values are small floats (typically in [-10, 10]), so catastrophic cancellation is not a concern. Expected behaviour: RMSE and Logloss results will differ from Sprint 16 baseline by up to a few ULP, but are not bit-exact due to the changed summation order.

**Parity gate (S17-G3, hard merge gate):**
- RMSE: ulp ≤ 4 (loosened from bit-exact; justified as DEC-005 in `DECISIONS.md`)
- Logloss: ulp ≤ 4
- MultiClass: ulp ≤ 8

Measured across N ∈ {1k, 10k, 50k} by @qa-engineer (S17-04). This is a hard gate — no override.

**Determinism retained:** The butterfly schedule is fixed by `tid`, so every run with the same input and launch geometry produces identical output. Run-to-run bit-exactness is preserved; only the comparison to the Sprint 16 serial-fold output relaxes.
