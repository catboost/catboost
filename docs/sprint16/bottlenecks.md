# Sprint 16 Bottleneck Inventory

## Post-measurement revision (2026-04-17)

After the full 18-config per-stage baseline (see
[`baseline_results.md`](baseline_results.md)) and the MST scout
([`mst_findings.md`](mst_findings.md)), the pre-measurement rankings below
need to be updated. Summary of how each original item holds up under
measurement:

- **B1 (sync storm in `pointwise_target.h`)** — **FIXED in Sprint 16.**
  `cpu_readback_ms` is 0.13 ms / 0.04% of iter time. Done.
- **B2 (`maxBlocksPerPart = 1`)** — **FALSIFIED for the production path.**
  `csv_train.cpp:891-894` now computes `maxBlocksPerPart = clamp(ceil(avg
  docs per part / 4096), 1, 8)`. The original `histogram.cpp:105` still
  hardcodes `maxBlocksPerPart = 1` but that path is dead code for csv_train
  (see "Divergence" section in `mst_findings.md`). Sprint 17 MUST NOT target
  this; it will do nothing.
- **B3 (multiclass per-dim dispatch loop)** — **CONFIRMED but RE-RANKED.**
  Multiclass is 2× binary in baseline (matches prediction). But the root
  cause of multiclass-ness-being-slow turns out to be the per-dim loop
  calling the same slow histogram kernel three times. Fixing the kernel
  itself (new lever D1 in `mst_findings.md`) is higher-leverage than
  fusing the per-dim calls. B3 deferred to Sprint 18.
- **B4 (per-feature-group serial dispatch)** — **FALSIFIED for the production
  path.** `csv_train.cpp:883-944` batches all feature groups into a single
  kernel call via the `numGroups` parameter (extracted from grid.x). Still
  present in `histogram.cpp:112-155` but that's dead code.
- **B5 (per-depth CPU readback syncs in `structure_searcher.cpp`)** —
  **FALSIFIED for the production path.** `cpu_readback_ms` is 0.13 ms
  (0.04%). Even if all six depth readbacks were removed the gain would be
  0.78 ms out of 318 ms. Not worth a sprint. Dead code in the library path
  only.
- **B6 (Lossguide CPU doc walker)** — **UNMEASURED, deferred.** Baseline ran
  only SymmetricTree / Depthwise. Lossguide performance remains uninstrumented
  — deserves its own diagnosis sprint before we optimise it.

**New top bottleneck (not in the original inventory): the histogram
kernel's serial 255-step threadgroup reduction.** See `mst_findings.md`
section B.3 and lever D1. It dominates the kernel's wall-clock cost and
was not visible to static code analysis because it lives in an embedded
string constant (`kernel_sources.h:161-181`) rather than in a .metal
file. Sprint 17 headline target.

---

Six bottlenecks identified by static code analysis before profiling began. Ranking by actual wall-clock impact will be determined from Sprint 16 diagnosis data and filled into [`docs/sprint16/diagnosis.md`](diagnosis.md) section 4.

Each entry includes: description, evidence (file and line), estimated impact, and the fix planned for Sprint 17+. Impact estimates are pre-measurement heuristics and will be revised once profiling data arrives.

---

## B1 — Sync storm in `pointwise_target.h` (Sprint 16 fix)

**Description.**  
Every loss function's `ComputeDerivatives` and `ComputeLoss` calls `TMLXDevice::EvalNow` on the gradient, hessian, and loss arrays before returning. This forces a CPU-GPU round-trip 18 times per iteration — after computing gradients for RMSE, after computing loss for RMSE, after computing gradients for Logloss, etc. MLX's lazy evaluation model makes these syncs completely unnecessary: the downstream histogram kernel would materialize the gradients as part of its own command buffer.

**Evidence.**  
`catboost/mlx/targets/pointwise_target.h`: 9 loss classes × ~2 `EvalNow` calls per class = 18 `EvalNow` calls visible by grep. Specifically:

- `TRMSETarget::ComputeDerivatives` (line 35), `ComputeLoss` (line 48)
- `TLoglossTarget::ComputeDerivatives` (line 84), `ComputeLoss` (line 105)
- `TMultiClassTarget::ComputeDerivatives` (line 167), `ComputeLoss` (line 212)
- `TMAETarget::ComputeDerivatives` (line 261), `ComputeLoss` (line 271)
- `TQuantileTarget::ComputeDerivatives` (line 300), `ComputeLoss` (line 314)
- `THuberTarget::ComputeDerivatives` (line 347), `ComputeLoss` (line 364)
- `TPoissonTarget::ComputeDerivatives` (line 399), `ComputeLoss` (line 413)
- `TTweedieTarget::ComputeDerivatives` (line 456), `ComputeLoss` (line 476)
- `TMAPETarget::ComputeDerivatives` (line 513), `ComputeLoss` (line 526)

**Estimated impact.**  
Each `EvalNow` is a CPU-GPU round-trip with ~0.5–2 ms overhead on Apple Silicon depending on pending work. At 18 calls per iteration, 100 iterations: 900 avoidable syncs. Pre-measurement estimate: 10–30% of total training time on small-to-medium N where compute per iteration is short. For N=100k at depth 6, this is likely the dominant term. The acceptance criterion for S16-07 is ≥10% e2e improvement on N=100k RMSE depth=6.

**Fix (Sprint 16 — S16-07).**  
Remove all `EvalNow` calls from `pointwise_target.h`. Add a single `EvalAtBoundary` at the top of the boosting iteration in `mlx_boosting.cpp::RunBoosting`, after `ComputeDerivatives` returns and before `InitPartitions`, to flush the accumulated lazy graph once per iteration. The `EvalAtBoundary` name signals that this sync is a deliberate iteration boundary, not a stale workaround.

**Risk.**  
`ComputeLoss` is called for validation evaluation, which happens outside the main training loop. Removing `EvalNow` from `ComputeLoss` is safe — the caller of `ComputeLoss` owns the sync responsibility. Confirm validation evaluation still works correctly in S16-08 numerical parity check.

---

## B2 — Histogram occupancy disaster (`maxBlocksPerPart = 1`)

**Description.**  
`ComputeHistogramsImpl` in `histogram.cpp` hardcodes `maxBlocksPerPart = 1` at line 105. This means the histogram kernel launches exactly one threadgroup (256 threads) per partition per feature group, regardless of how large the partition is. For a partition with 50k documents, a single threadgroup processes all 50k documents serially in one thread's strided loop. The Metal GPU has hundreds of SIMD groups available; this configuration occupies one.

**Evidence.**  
`catboost/mlx/methods/histogram.cpp`, line 105:
```cpp
const ui32 maxBlocksPerPart = 1;
```
This feeds into `DispatchHistogramGroup`'s grid X dimension: `256 * maxBlocksPerPart * numGroups`. At 1 block per partition, the GPU is massively underutilized for any N > 256.

**Estimated impact.**  
This is likely the largest single bottleneck at N ≥ 10k. The histogram stage dominates per-iteration cost (see ARCHITECTURE.md "Histogram bottleneck"). Making `maxBlocksPerPart` proportional to partition size (e.g., `ceil(partitionSize / 256)`, capped at a hardware-appropriate maximum) could deliver 4–8x speedup on the histogram stage alone for N ≥ 100k. Impact estimate: 40–60% of total training time at N=100k.

**Fix (Sprint 17).**  
Tune `maxBlocksPerPart` as a function of `numDocs / numPartitions`. The atomic `fetch_add` in the global write pass is already correct for multiple blocks writing to the same partition slot (uses `memory_order_relaxed` — see ARCHITECTURE.md "kHistOneByteSource"). No correctness change needed, only the grid dimension.

**Risk.**  
Higher block counts increase atomic contention on the global histogram buffer. Need to find the knee: beyond a certain block count, atomic contention overtakes parallelism gains. Profile at several values of `maxBlocksPerPart` (2, 4, 8, 16) to find the sweet spot for M-series hardware.

---

## B3 — Multiclass per-dim histogram dispatch loop

**Description.**  
For multiclass with `approxDim = K`, `structure_searcher.cpp` contains a loop that iterates over each dimension `k = 0..K-1` and dispatches a separate histogram for each dimension. This serializes K histogram dispatches that could be fused into a single dispatch with a Z-dimension covering all K dimensions simultaneously.

**Evidence.**  
`catboost/mlx/methods/structure_searcher.cpp`: the per-dim slicing loop wrapping histogram dispatch. With K=10, this produces 10 serialized kernel launches where 1 would suffice if the stats array layout is adjusted. The histogram kernel already has a Z-dimension for `numStats` (grad + hess); extending this to also cover `approxDim` is a natural generalization.

**Estimated impact.**  
Pre-measurement estimate: 2–4x multiclass speedup. At K=10 and N=10k, MLX is 15.6x slower than CPU — this is the worst-performing case in the baseline data. K-fold serialization is the most likely explanation for multiclass being disproportionately slower than binary.

**Fix (Sprint 18).**  
Fuse the per-dim loop into a single histogram dispatch. Requires updating the stats layout from `[numStats, numDocs]` to `[approxDim * numStats, numDocs]` (or equivalent), adjusting the Z grid dimension, and updating the histogram kernel's indexing arithmetic. Must verify correctness against the CPU reference for all K.

**Risk.**  
The stats array reshape touches the CPU-GPU data layout contract. Both code paths (`csv_train.cpp` and the library path) must be updated consistently. Regression risk is high — numerical parity testing is mandatory before and after this change.

---

## B4 — Serialized per-feature-group dispatch loop

**Description.**  
`ComputeHistogramsImpl` dispatches one Metal kernel per feature group (4 features per group) in a sequential loop. For 200 features (50 groups), this produces 50 sequential kernel dispatches. While MLX batches these into the same command buffer, the encoding overhead accumulates, and there is no opportunity for the GPU scheduler to overlap execution across groups.

**Evidence.**  
`catboost/mlx/methods/histogram.cpp`, lines 112–155: the `for (groupIdx = 0; groupIdx < numFeatureGroups; ++groupIdx)` loop. Each iteration calls `DispatchHistogramGroup` and accumulates results with `mx::add`. For 50 feature groups, this is 50 `mx::fast::metal_kernel()` dispatch calls per histogram per depth level.

**Estimated impact.**  
Pre-measurement: moderate. At feature counts ≥ 100, kernel encoding overhead is non-trivial. Batching all feature groups into a single dispatch by adding a group index dimension to the grid could reduce dispatch overhead significantly. Impact likely secondary to B2 (occupancy) — actual cost only visible after B2 is fixed.

**Fix (Sprint 17 or 18).**  
Add a feature-group dimension to the histogram kernel's grid (X dimension becomes `256 * maxBlocksPerPart * numGroups`). The kernel selects its group from the threadgroup's X index. This collapses the entire `for groupIdx` loop into a single dispatch. The `mx::add` accumulation step also simplifies to a direct output slot per group.

**Risk.**  
The kernel's private histogram arrays are sized for one feature group (4 features × 256 bins = 1024 floats). Expanding to multi-group would require either per-group private arrays (register pressure increase) or sequential group processing within the kernel (partial loop collapse). This is a kernel rewrite — needs careful register pressure analysis on M-series.

---

## B5 — Per-depth CPU readback syncs in `structure_searcher.cpp`

**Description.**  
Two `EvalNow` calls in `structure_searcher.cpp` materialize the full gradient/hessian partition sums to CPU memory before a per-partition loop that computes the complement stats for each partition. These readbacks gate the entire depth level on a CPU-GPU round-trip before any scoring begins.

**Evidence.**  
- `structure_searcher.cpp` line 252: `TMLXDevice::EvalNow({allGradSums, allHessSums})` — in the SymmetricTree path, before the per-partition complement computation loop.
- `structure_searcher.cpp` line 550: same pattern in the Depthwise path.

The per-partition complement computation (subtracting child sums from parent sum) is a pure arithmetic operation that could run on-GPU. The CPU-side loop exists because the complement is computed as `totalSum - childSum`, which requires knowing the total — itself a readback.

**Estimated impact.**  
Pre-measurement: 2 syncs × depth_levels × ms_per_sync = significant at depth 10. For depth=6 at N=100k, this adds 12 blocking round-trips per iteration on top of the 2 unavoidable ones.

**Fix (Sprint 19).**  
Implement complement computation as a Metal kernel or MLX element-wise op that runs on-GPU. The total sum is already available as a GPU-resident array. No CPU readback is needed if the complement is computed in a single vectorized subtract pass on the GPU before passing to the scorer.

**Risk.**  
The complement computation touches the per-partition stats layout. Any change here must be verified against the split scoring kernel, which reads the same arrays. Numerical identity with the CPU-path reference is required.

---

## B6 — Lossguide CPU-side doc walker

**Description.**  
`SearchLossguideTreeStructure` in `structure_searcher.cpp` (lines 645–660) materializes `compressedData` to CPU for a per-document BFS traversal that assigns leaf indices during the lossguide tree expansion loop. This is the hot path during Lossguide tree construction: every time a leaf is split, the affected documents must be re-assigned. At N=100k with 31 max_leaves and 30 splits, this materializes `compressedData` up to 30 times per tree.

Additionally, `structure_searcher.cpp` line 676 calls `EvalNow(result.LeafDocIds)` after each leaf assignment update.

**Evidence.**  
- `structure_searcher.cpp` line 645: `TMLXDevice::EvalNow(compressedData)` + CPU BFS loop through lines 646–660.
- `structure_searcher.cpp` line 676: `TMLXDevice::EvalNow(result.LeafDocIds)` after each assignment.
- The same `EvalNow(flatData)` pattern appears in `tree_applier.cpp` line 251 for the inference path.

**Estimated impact.**  
Lossguide is disproportionately expensive: the baseline shows Lossguide is substantially slower than SymmetricTree at equal leaf counts (see ARCHITECTURE.md). Pre-measurement: 30 compressedData materializations × ~5 ms per sync at N=100k = 150 ms overhead just in syncs, before any GPU compute. This explains why Lossguide falls furthest from the CPU reference.

**Fix (Sprint 17 or later).**  
Implement leaf assignment as a Metal kernel: for each document, evaluate the split condition for the relevant split and update `LeafDocIds` in-place on the GPU. Only the affected documents (those currently assigned to the splitting leaf) need to be processed. This eliminates the CPU BFS traversal entirely and keeps `compressedData` device-resident.

**Risk.**  
The Lossguide path is structurally more complex than SymmetricTree — the GPU kernel needs access to the BFS split map and the current leaf-to-BFS-ID mapping. These are small structures (at most 30 entries for max_leaves=31) and can be uploaded as constant buffers. Correctness verification requires testing against the SymmetricTree-equivalent case (max_leaves = 2^depth).
