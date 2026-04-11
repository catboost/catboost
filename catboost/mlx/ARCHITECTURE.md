# CatBoost-MLX Architecture

A deep-dive for contributors and anyone who wants to understand how the Metal GPU backend works.

---

## Table of Contents

1. [Overview](#overview)
2. [Training Pipeline](#training-pipeline)
3. [Metal Kernels](#metal-kernels)
4. [Data Layout](#data-layout)
5. [CPU-GPU Synchronization](#cpu-gpu-synchronization)
6. [Two Code Paths](#two-code-paths)
7. [Performance Characteristics](#performance-characteristics)

---

## Overview

CatBoost-MLX is a Metal GPU backend for CatBoost's gradient boosting decision tree (GBDT) algorithm, built on Apple's MLX framework. It runs on Apple Silicon exclusively and targets the same training accuracy as the CPU reference implementation.

**What CatBoost builds: oblivious trees.**
An oblivious tree applies the same split condition at every node of a given depth level. At depth `d`, there are exactly `2^d` leaves, and every leaf is reachable by a unique binary string of "go left / go right" decisions — one per level. This regularity makes the algorithm highly parallelizable: all documents traverse the same set of split conditions, and their leaf assignment is a bitwise OR of `d` independent binary decisions.

**Why MLX instead of raw Metal.**
MLX handles Metal device management, command buffer encoding, memory allocation, pipeline state caching, and JIT kernel compilation. Writing those from scratch for GBDT would replicate most of MLX's source. Instead, custom GBDT kernels are registered through `mx::fast::metal_kernel()`, which compiles and caches them the first time they run. MLX's lazy evaluation model lets multiple kernel dispatches accumulate into a single command buffer, reducing CPU-GPU round-trip costs. (See ADR-001 in `docs/decisions.md`.)

**Why not autograd.**
CatBoost computes gradients analytically from the loss function (first and second derivatives of the loss with respect to predictions). There is no backpropagation graph. MLX autograd is not used and must not be added — it would impose memory and computation overhead with no benefit.

---

## Training Pipeline

The library path entry point is `RunBoosting()` in `catboost/mlx/methods/mlx_boosting.cpp`. Each boosting iteration follows this sequence:

```
ComputeDerivatives          (loss gradients and hessians — MLX ops on GPU)
        |
        v
InitPartitions              (reset all doc leaf assignments to 0)
        |
        v
  for each depth level d (0 .. maxDepth-1):
        |
        v
  ComputePartitionLayout    (GPU bucket sort → DocIndices, PartOffsets, PartSizes)
        |
        v
  ComputeHistograms          (Metal kernel: kHistOneByteSource — one dispatch per feature group)
        |
        v
  SuffixSum transform        (Metal kernel: kSuffixSumSource — one threadgroup per feature)
        |
        v
  ScoreSplits                (Metal kernel: kScoreSplitsLookupSource — block-level argmax)
        |
        v
  CPU final reduction        (EvalNow — readback per-block best-split candidates, pick global best)
        |
        v
  UpdatePartitions           (MLX bitwise ops — EvalNow — update leaf assignments)
        |
        v
ComputeLeafSumsGPU          (Metal kernel: kLeafAccumSource — single threadgroup accumulation)
        |
        v
ComputeLeafValues            (MLX ops: single vectorized Newton step over [approxDim * numLeaves] — lazy, no EvalNow)
        |
        v
  [multiclass only: reshape [approxDim * numLeaves] → transpose → [numLeaves, approxDim] on GPU]
        |
        v
ApplyObliviousTree           (Metal kernel: kTreeApplySource — one thread per document)
                             (dual output: cursorOut + partitionsOut — single dispatch)
        |
        v
UpdateCursor + UpdatePartitions  (both outputs written back from kernel results — no recompute)
```

### Key per-iteration state in `TMLXDataSet`

| Field | Shape | Dtype | Role |
|---|---|---|---|
| `CompressedIndex` | `[numDocs, numColumns]` | `uint32` | Packed quantized features |
| `Targets` | `[numDocs]` | `float32` | Training labels |
| `Weights` | `[numDocs]` | `float32` | Per-document weights |
| `Cursor` | `[approxDim, numDocs]` | `float32` | Accumulated model predictions |
| `Gradients` | `[approxDim, numDocs]` | `float32` | First-order loss derivatives |
| `Hessians` | `[approxDim, numDocs]` | `float32` | Second-order loss derivatives |
| `Partitions` | `[numDocs]` | `uint32` | Current leaf index per document |

---

## Metal Kernels

All kernel source strings live in `catboost/mlx/kernels/kernel_sources.h`. They are plain C++ string literals passed to `mx::fast::metal_kernel()`, which compiles them on first use and caches the resulting `MTLComputePipelineState`. There are four production kernels.

---

### `kHistOneByteSource` — Histogram accumulation

**What it computes.**
For each document, for each quantized feature, increments a gradient-sum (or hessian-sum) accumulator at the bin index that the document lands in. The output is a per-partition histogram: `histogram[partitionIdx][statIdx][binFeatureIdx]`.

**Parallelization strategy.**
One threadgroup per feature group. Features are packed four per `uint32` column, so the kernel processes exactly four logical features per threadgroup dispatch. A batched dispatch covers all feature groups simultaneously:

- Grid X dimension: `maxBlocksPerPart * numGroups` (each threadgroup encodes which group and which within-partition block it handles)
- Grid Y dimension: `numPartitions` (one partition per Y slice)
- Grid Z dimension: `numStats` (grad-only = 1, grad+hess = 2)
- Threadgroup size: `(256, 1, 1)`

Each thread processes a strided subset of the partition's documents (`thread_index_in_threadgroup`, `thread_index + 256`, ...).

**Memory access pattern.**
Private per-thread histogram arrays (`float privHist[1024]`) accumulate without any shared memory contention during the document loop. After accumulation, a fixed-order sequential reduction (threads 1 through 255 add into a `threadgroup float stagingHist[1024]` one at a time, separated by `threadgroup_barrier`) produces a fully deterministic result. The final write to the global output buffer uses `atomic_fetch_add_explicit` with `memory_order_relaxed` because multiple blocks may write to the same partition slot.

**Threadgroup size rationale.**
256 threads = 8 SIMD groups (32 threads each) on Apple Silicon. This is the canonical occupancy-maximizing size for a compute kernel with modest register pressure. It also matches `BLOCK_SIZE = 256` in `kHistHeader`, which sets the private histogram accumulation stride.

**Key correctness invariant.**
The addition order of per-thread private histograms into the staging buffer is fixed (thread 1, then thread 2, ..., thread 255). This is the BUG-001 fix: the original CAS-based shared-memory approach produced non-deterministic results because Apple Silicon does not guarantee CAS arbitration order across SIMD groups. Per-thread private accumulation eliminates the race entirely; the sequential reduction makes the final float sum bit-for-bit identical across all dispatches.

---

### `kSuffixSumSource` — Histogram suffix-sum transform

**What it computes.**
Converts raw per-bin gradient sums into suffix sums so that split scoring is O(1) per bin. For an ordinal feature with bins `h[0..F-1]`, the output is `h'[b] = sum(h[b..F-2])`. The rightmost bin (`b = F-1`) is left as zero (its write is skipped), which prevents the scorer from selecting an all-right split with no left-side documents.

One-hot features are skipped entirely: their histogram entries are independent category lookups, not ordinal ranges.

**Parallelization strategy.**
One threadgroup per `(feature, partition*approxDim, stat)` triple:

- Grid X: `numFeatures`
- Grid Y: `approxDim * numPartitions`
- Grid Z: `numStats`
- Threadgroup size: `(256, 1, 1)` — one thread per possible bin slot (max 255 bins + 1 guard lane)

**Memory access pattern.**
Each threadgroup loads bin values in reversed order into `threadgroup float scanBuf[256]`. A Hillis-Steele inclusive prefix scan runs for 8 rounds (strides 1, 2, 4, ..., 128), separated by `threadgroup_barrier` after each round. The result is a right-to-left suffix sum.

**Why 256 threads (not 32).**
The `scanBuf[256]` array covers all 256 possible bin slots. With 32 threads, positions 32–255 would remain uninitialized (Metal threadgroup memory is not zeroed between dispatches), producing garbage suffix sums for features with more than 32 bins. 256 threads fill all slots deterministically.

**Key correctness invariant.**
The Hillis-Steele algorithm's addition order is determined by the stride pattern, not by hardware scheduling. This is the BUG-001 fix for the suffix-sum kernel: the previous `simd_prefix_inclusive_sum + simd_broadcast` path produced alternating values because `simd_broadcast` reads from an architecturally-undefined lane state when the active-lane mask is non-convergent across separate Metal command-buffer submissions.

---

### `kScoreSplitsLookupSource` — Split scoring with argmax reduction

**What it computes.**
Each thread evaluates one `(feature, bin)` candidate split, computing the gain from partitioning documents left vs right at that bin across all partitions and all `approxDim` dimensions:

```
gain = sum_over_partitions_and_dims(
    sumLeft^2 / (weightLeft + lambda) +
    sumRight^2 / (weightRight + lambda) -
    totalSum^2 / (totalWeight + lambda)
)
```

After the suffix-sum transform, `sumRight = histogram[histBase + firstFold + binInFeature]`, which is an O(1) lookup.

**Parallelization strategy.**
One thread per bin-feature candidate:

- Grid X: `256 * numBlocks` where `numBlocks = ceil(totalBinFeatures / 256)`
- Threadgroup size: `(256, 1, 1)`

A tree-reduction argmax over `sharedGain[256]` / `sharedFeat[256]` / `sharedBin[256]` in shared threadgroup memory produces one best-split candidate per block. Thread 0 of each block writes the result. The CPU then performs a final reduction over `numBlocks` candidates (an `EvalNow` readback).

**Lookup table optimization (OPT-2).**
The `kScoreSplitsSource` variant finds which feature a global bin index belongs to via a linear scan over features. `kScoreSplitsLookupSource` replaces this with a precomputed `binToFeature[]` array, making feature identification O(1). The lookup is built on the CPU before dispatch and uploaded as a Metal buffer. This optimization is used on the fast path (`FindBestSplitGPU` with GPU partition stats).

**Key correctness invariant.**
The scorer reads from the suffix-sum-transformed histogram. Calling this kernel before `kSuffixSumSource` completes would produce incorrect split gains. The suffix-sum kernel and score kernel run in two separate `mx::fast::metal_kernel()` dispatches, and MLX serializes GPU dispatches within the same stream — no explicit barrier is needed between them.

---

### `kTreeApplySource` — Tree application

**What it computes.**
For each document, applies all split levels of a trained oblivious tree to compute the leaf index, then adds the corresponding leaf value to the prediction cursor. The kernel produces two outputs in a single dispatch:

```
leafIdx = 0
for level d in [0, depth):
    featureVal = (compressedData[doc * lineSize + splitColIdx[d]] >> splitShift[d]) & splitMask[d]
    goRight    = (isOneHot[d]) ? (featureVal == threshold[d]) : (featureVal > threshold[d])
    leafIdx   |= goRight << d

cursorOut[k * numDocs + doc] = cursorIn[k * numDocs + doc] + leafValues[leafIdx * approxDim + k]
partitionsOut[doc] = leafIdx
```

The second output, `partitionsOut`, captures the per-document leaf assignment directly from the already-computed `leafIdx` inside the kernel. This replaces the O(depth) MLX bitwise-op recompute loop that previously recalculated `leafIdx` from the split arrays on the CPU side after the kernel returned (Sprint 7, TODO-020, −28 lines in `tree_applier.cpp`).

**Parallelization strategy.**
One thread per document:

- Grid X: `ceil(numDocs / 256) * 256`
- Threadgroup size: `(256, 1, 1)`

**Memory access pattern.**
No shared memory. No atomics. Each thread owns its own document's input and output slots. The `depth` loop (at most 6 iterations in practice) accesses five small arrays (`splitColIdx`, `splitShift`, `splitMask`, `splitThreshold`, `splitIsOneHot`) that are typically small enough to be held in the constant address space. Reads from `compressedData` are coalesced: consecutive thread indices correspond to consecutive document rows.

**Key correctness invariant.**
The learning rate is baked into `leafValues` by `ComputeLeafValues()` before the kernel call. The kernel performs no scalar multiply — it does a single load-add-store per output slot. This means `leafValues` must not be reused across multiple calls to `ApplyObliviousTree` without recomputing them.

---

## Data Layout

### Compressed feature index

Features are quantized at load time into integer bin indices. The in-memory layout is a 2D array `[numDocs, numColumns]` of `uint32`, where each `uint32` column holds up to four one-byte feature values packed left-to-right:

```
uint32 column:  [feat A (bits 24-31)] [feat B (bits 16-23)] [feat C (bits 8-15)] [feat D (bits 0-7)]
```

To extract feature `f` from a packed column:

```c++
uint32_t featureVal = (packedColumn >> feat.Shift) & feat.Mask;
```

### `TCFeature` descriptor

Each logical feature is described by a `TCFeature` struct (`catboost/mlx/gpu_data/gpu_structures.h`):

| Field | Type | Meaning |
|---|---|---|
| `Offset` | `uint64_t` | Column index in the compressed data (i.e., `compressedData[doc * lineSize + Offset]`) |
| `Mask` | `uint32_t` | Bitmask applied after the right-shift to extract this feature's bits |
| `Shift` | `uint32_t` | Right-shift count to align this feature to bit 0 |
| `FirstFoldIndex` | `uint32_t` | Starting bin index in the global histogram buffer for this feature |
| `Folds` | `uint32_t` | Number of bins (quantization folds) for this feature |
| `OneHotFeature` | `bool` | If true, split comparison uses equality; otherwise uses greater-than |

Features are packed in groups of four into one uint32 column. `Shift` values within a group are 24, 16, 8, and 0 for features packed left-to-right. `Mask` is always `0xFF` for one-byte features.

### Histogram buffer layout

The histogram buffer is flat 1D `float32` with logical shape:

```
[numPartitions * numStats * totalBinFeatures]
```

Where `totalBinFeatures = sum(feature.Folds for all features)`. To index bin `b` of feature `f` in partition `p` with stat `s`:

```
index = p * numStats * totalBinFeatures + s * totalBinFeatures + feature.FirstFoldIndex + b
```

### Partition assignments

`TMLXDataSet::Partitions_` is `[numDocs] uint32`. Each element holds the leaf index for that document, accumulated bit by bit across depth levels:

```
partitions[doc] |= (goRight_at_level_d << d)
```

At depth `d`, valid leaf indices are in the range `[0, 2^d)`. After all `maxDepth` levels, leaf indices are in `[0, 2^maxDepth)`.

---

## CPU-GPU Synchronization

`TMLXDevice::EvalNow()` forces MLX to flush the pending computation graph and synchronize (blocking CPU until all queued Metal commands have completed). Every call is a CPU-GPU round-trip and adds latency. As of Sprint 7, two `EvalNow` calls per depth level remain unavoidable (best-split readback and partition update); all other unnecessary syncs have been removed.

### Unavoidable syncs

**Best-split readback** (`score_calcer.cpp`):
The GPU produces `numBlocks` per-block best-split candidates. The CPU must read these to select the global winner, because the selection depends on a floating-point max across variable-length data — implementing this as another kernel dispatch for such a small number of blocks (typically 1–4 for realistic feature counts) would cost more in kernel launch overhead than the reduction saves.

```
EvalNow({bestScoresArr, bestFeatArr, bestBinArr})
```

**Partition update** (`structure_searcher.cpp`):
After each depth level, the CPU needs the `partitions` array in a consistent state to correctly initialize the next depth level's partition layout computation. The MLX `argsort` in `ComputePartitionLayout` consumes `partitions` as input; that consumption must see the post-update value.

```
TMLXDevice::EvalNow(partitions);
```

**Histogram result** (`histogram.cpp`):
After all feature groups are dispatched and accumulated, `EvalNow` is called once per histogram to materialize it before passing to the suffix-sum kernel. This could be eliminated if MLX supported chaining all group dispatches and the suffix-sum into a single command buffer — a future optimization.

### Removed syncs (Sprint optimizations)

- **OPT-1 (Sprint 3):** Eliminated `approxDim` CPU-GPU round trips per depth level by computing all-dimension partition statistics in a single GPU dispatch (`ComputeLeafSumsGPU` with `approxDim` handled internally).
- **BUG-001 fix (Sprint 5):** Removed the CPU-side verification loop that read back and compared histogram values to check for non-determinism. Determinism is now guaranteed by construction.
- **TODO-019 (Sprint 7):** Eliminated K `EvalNow` calls per iteration that the old per-dimension multiclass leaf loop incurred. `ComputeLeafValues` now returns a lazy MLX array; the Newton step executes over the full `[approxDim * numLeaves]` array in a single element-wise dispatch. For K=10 multiclass, this removes 10 CPU-GPU round trips per boosting iteration.
- **TODO-020 (Sprint 7):** Eliminated the O(depth) MLX bitwise-op recompute loop for partition assignments. `kTreeApplySource` now produces `partitionsOut` directly as a second kernel output — the leaf index computed inside the kernel is written out without any post-kernel recomputation on the CPU side.

---

## Two Code Paths

There are two independent implementations of the GBDT training algorithm in this repository. **This is a known architectural tension, not an accident.**

### Path 1 — Library path

```
python/catboost_mlx/      (Python bindings)
    └─> subprocess: csv_train binary
        └─> csv_train.cpp (standalone, no CatBoost headers)
            └─> all kernels inline
```

Wait — the Python layer actually calls `csv_train` as a subprocess. The `csv_train.cpp` binary is a **self-contained reimplementation** of the full GBDT pipeline that duplicates the kernel dispatch logic from the library path. It mirrors `gpu_structures.h` types (`TCFeature`, `TBestSplitProperties`, etc.) locally and does not link against any CatBoost library code.

### Path 2 — Library path (via CatBoost training framework)

```
catboost/mlx/train_lib/train.cpp   (TMLXModelTrainer — registered as GPU backend)
    └─> methods/mlx_boosting.cpp   (RunBoosting)
        └─> methods/structure_searcher.cpp  (SearchTreeStructure, ComputePartitionLayout)
        └─> methods/histogram.cpp           (ComputeHistograms)
        └─> methods/score_calcer.cpp        (FindBestSplitGPU, suffix-sum)
        └─> methods/leaves/leaf_estimator.cpp (ComputeLeafSumsGPU, ComputeLeafValues)
        └─> methods/tree_applier.cpp        (ApplyObliviousTree)
```

This path uses the full CatBoost library, integrates with `TTrainingDataProviders`, and produces a `TFullModel` compatible with CatBoost's native format.

### Why this matters

BUG-001 (non-deterministic histogram accumulation) was discovered in the `csv_train` path during QA testing. Because `csv_train.cpp` duplicates kernel dispatch logic independently, fixes must be applied to **both paths separately** — the fix in `csv_train.cpp` does not propagate to `histogram.cpp`, `score_calcer.cpp`, or `leaf_estimator.cpp`, and vice versa.

Before touching any kernel source or dispatch pattern, check whether the change applies to both paths. If `csv_train.cpp` and the library path diverge in kernel behavior, the Python layer will produce different results from what `bench_boosting` measures, and integration tests will fail in non-obvious ways.

The authoritative kernel source strings are in `catboost/mlx/kernels/kernel_sources.h`. Both paths include this header. However, the dispatch wrappers (grid dimensions, input arrays, threadgroup sizes) are separate in each path.

---

## Performance Characteristics

### Reference numbers from `bench_boosting`

`bench_boosting` exercises the library path (Path 2) on synthetic in-memory data. Representative timings on M-series hardware (100k rows, 50 features, depth 6, 100 iterations):

| Metric | Typical value |
|---|---|
| Iteration 0 (cold start — Metal kernel compile) | ~344 ms (pre-cache) → ~109 ms (post TODO-008 cache) |
| Warm iteration average (iters 1–99) | ~5–15 ms |
| Dominant cost per warm iteration | Histogram build (atomic writes in global output pass) |

### Cold-start cost

MLX JIT-compiles Metal kernels the first time they run. This is a one-time cost per process lifetime; subsequent iterations reuse the compiled pipeline state. TODO-008 introduced a kernel compile cache that reduces cold-start from ~344 ms to ~109 ms by reusing cached `.metallib` artefacts across process restarts.

### Histogram bottleneck

The histogram kernel's final write pass uses `atomic_fetch_add_explicit` on the global output buffer (multiple blocks and groups write to the same partition slots). On Apple Silicon, atomic writes to device memory are serialized at the memory subsystem level — this is the primary throughput bottleneck for large feature counts or many partitions. The private accumulation phase (no atomics) is fast; the global atomic write is the limiting factor.

### Memory constraints

Apple Silicon's unified memory architecture means GPU and CPU share physical memory. Large histogram buffers (`numPartitions * numStats * totalBinFeatures * 4 bytes`) and the compressed feature matrix (`numDocs * numColumns * 4 bytes`) both live in this shared space. For 16M rows (the current `float32` accumulator limit, see DEC-003 in `.claude/state/DECISIONS.md` and ADR-005 in `docs/decisions.md`), the compressed feature matrix alone can exceed 1 GB. Monitor memory allocation with Metal's GPU counters when benchmarking large datasets.

### Relationship between ADR-005 and DEC-003

ADR-005 chose `float32` as the primary compute precision. DEC-003 documents the consequence: `ComputePartitionLayout` uses `float32` accumulators for counting documents per partition, which limits exact integer representation to `numDocs < 2^24 = 16,777,216`. These two decisions are coupled — switching to `int32` accumulators (DEC-003 future work) would remove the 16M row ceiling without changing the `float32` gradient accumulation policy (ADR-005).
