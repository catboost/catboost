# CatBoost-MLX Architecture

A deep-dive for contributors and anyone who wants to understand how the Metal GPU backend works.

---

## Table of Contents

1. [Overview](#overview)
2. [Training Pipeline](#training-pipeline)
3. [Metal Kernels](#metal-kernels)
4. [Depthwise Grow Policy](#depthwise-grow-policy)
5. [Lossguide Grow Policy](#lossguide-grow-policy)
6. [Multi-Pass Leaf Accumulation (depth > 6)](#multi-pass-leaf-accumulation-depth--6)
7. [Data Layout](#data-layout)
8. [CPU-GPU Synchronization](#cpu-gpu-synchronization)
9. [Loss Functions (Target Functions)](#loss-functions-target-functions)
10. [Ranking Losses](#ranking-losses)
11. [Model Format and Versioning](#model-format-and-versioning)
12. [Two Code Paths](#two-code-paths)
13. [Nanobind Binding Architecture](#nanobind-binding-architecture)
14. [Performance Characteristics](#performance-characteristics)

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

### `kTreeApplyDepthwiseSource` — Depthwise tree application

**What it computes.**
Applies a non-symmetric (depthwise) tree where each internal node has its own split condition. The tree is stored in BFS order: node 0 is the root, children of node n are at indices 2n+1 (left) and 2n+2 (right). After traversing `depth` levels, the leaf index is `nodeIdx − numNodes` where `numNodes = 2^depth − 1`.

```
nodeIdx = 0
for level d in [0, depth):
    featureVal = (compressedData[doc * lineSize + nodeColIdx[nodeIdx]] >> nodeShift[nodeIdx]) & nodeMask[nodeIdx]
    goRight    = (nodeIsOneHot[nodeIdx]) ? (featureVal == threshold) : (featureVal > threshold)
    nodeIdx    = 2 * nodeIdx + 1 + goRight

leafIdx       = nodeIdx - numNodes
cursorOut[k * numDocs + doc] = cursorIn[k * numDocs + doc] + leafValues[leafIdx * approxDim + k]
partitionsOut[doc] = leafIdx
```

Like `kTreeApplySource`, the kernel produces two outputs in a single dispatch: updated predictions and per-document leaf assignments.

**Parallelization strategy.**
Identical to `kTreeApplySource`: one thread per document, grid and threadgroup size `(256, 1, 1)`.

**BFS node array.**
The `nodeColIdx`, `nodeShift`, `nodeMask`, `nodeThreshold`, `nodeIsOneHot` arrays each have `numNodes = 2^depth − 1` entries. At depth 0 a 1-element placeholder is allocated so the Metal kernel always receives a non-empty buffer (the traversal loop does not execute when `depth == 0`).

**Key correctness invariant.**
The BFS index arithmetic `2n+1` / `2n+2` is exact for `depth ≤ 10` (max nodeIdx = 2047). The leaf index range `[0, 2^depth)` matches the output of `kLeafAccumSource` and `kLeafAccumChunkedSource`, so leaf values produced by `ComputeLeafValues` are compatible between the symmetric and depthwise paths.

---

## Depthwise Grow Policy

The default "SymmetricTree" (oblivious tree) applies one split condition shared across all nodes at the same depth level. "Depthwise" gives each node its own best split, producing more expressive trees at the same depth — this is equivalent to XGBoost's `grow_policy=depthwise`.

### `EGrowPolicy` enum (`mlx_boosting.h`)

| Value | Meaning |
|---|---|
| `SymmetricTree` | Default. One `FindBestSplitGPU` call per depth level, shared split. |
| `Depthwise` | 2^d `FindBestSplitGPU` calls at depth level d; each node gets the best split for its document subset. |

### `SearchDepthwiseTreeStructure` (`structure_searcher.cpp`)

The depthwise search loop proceeds depth-first, level by level:

```
for d in [0, maxDepth):
    for each live node n in [0, 2^d):
        FindBestSplitGPU(documents in node n)
        NodeSplits[n] = best split
    UpdatePartitions()    (EvalNow — partition update is unavoidable)
```

The node-to-document mapping is derived from the running `Partitions_` array. Documents with `partitions[doc] == n` belong to node n at depth level d. This reuses `ComputePartitionLayout` and the existing split-search pipeline without structural change.

### Tree structure storage

`TDepthwiseTreeStructure` holds:
- `NodeSplits`: `TVector<TObliviousSplitLevel>` in BFS order, length `2^depth − 1`
- `Depth`: the actual tree depth

Leaf estimation and tree application read the same `Partitions_` array that SymmetricTree uses, so `ComputeLeafSumsGPU` and `ComputeLeafValues` require no changes.

### Python and CLI interface

Select grow policy via:
- **Python:** `CatBoostMLX(grow_policy="Depthwise")` — "SymmetricTree" is the default
- **CLI:** `csv_train --grow-policy Depthwise`

See DEC-004 in `.claude/state/DECISIONS.md` for the full design rationale.

---

## Lossguide Grow Policy

Lossguide (added Sprint 10) is the third grow policy alongside SymmetricTree and Depthwise. It grows the tree one leaf at a time, always splitting the leaf with the highest loss-reduction gain — equivalent to LightGBM's `num_leaves`-controlled best-first expansion. The resulting tree is unbalanced: different branches may reach different depths. Complexity is controlled by `max_leaves` (default 31) rather than `max_depth`.

### `TLossguideTreeStructure` (`structure_searcher.h`)

| Field | Type | Description |
|---|---|---|
| `NodeSplitMap` | `std::unordered_map<ui32, TObliviousSplitLevel>` | BFS node index → split descriptor, internal nodes only |
| `LeafBfsIds` | `TVector<ui32>` | Dense leaf index k → BFS node index (length = `NumLeaves`) |
| `NumLeaves` | `ui32` | Current leaf count; starts at 1, increments on each split |
| `LeafDocIds` | `mx::array [numDocs] uint32` | Dense per-document leaf assignment; updated incrementally during search |

The sparse `unordered_map` avoids the O(2^depth) allocation that a full BFS array would require for unbalanced trees. At `max_leaves=31`, there are at most 30 internal nodes regardless of individual branch depth.

### `SearchLossguideTreeStructure` (`structure_searcher.cpp`)

```
Start: root = leaf 0, LeafDocIds[:] = 0

Initialize pq: for each initial leaf, compute histograms + FindBestSplitGPU
               push (gain, leafId, bestSplit) onto priority_queue

while NumLeaves < maxLeaves and pq not empty:
    (gain, leafId, split) = pq.pop()          // highest-gain leaf
    if gain <= 0: break                        // no further improvement possible

    Add split to NodeSplitMap[bfsIdx]
    Assign two new dense leaf indices (leftLeaf, rightLeaf)
    Update LeafDocIds: docs in leafId → leftLeaf or rightLeaf based on split
    Update LeafBfsIds

    Compute histograms for leftLeaf and rightLeaf
    FindBestSplitGPU for each; push valid results onto pq
```

The priority queue ensures the globally best split is always chosen next — this matches LightGBM's behaviour and produces the lowest training loss for a given `max_leaves` budget.

`LeafDocIds` is updated in-place after each split, so at the end of search it is immediately ready for `ComputeLeafSumsGPU` without a separate partition scan.

### `ApplyLossguideTree` (`tree_applier.cpp`)

**Training path:** `LeafDocIds` from the search is used directly as the partition array. No BFS re-traversal is needed.

**Inference/validation path:** `ComputeLeafIndicesLossguide` performs a CPU-side BFS traversal of `NodeSplitMap` for each document. Starting from BFS node 0 (root), at each node it looks up the split in `NodeSplitMap` and follows left or right until it reaches a node absent from the map (a leaf). The dense leaf index is then looked up from `LeafBfsIds`.

Both paths produce the same `LeafDocIds` format consumed by `ComputeLeafSumsGPU` and `ComputeLeafValues`, which require no changes.

### Python and CLI interface

Select lossguide policy via:
- **Python:** `CatBoostMLX(grow_policy="Lossguide", max_leaves=31)`
- **CLI:** `csv_train --grow-policy Lossguide --max-leaves 31`

`max_leaves` (minimum 2) controls tree complexity; `max_depth` (optional, 0 = no limit) prevents any single branch from growing arbitrarily deep.

See DEC-006 in `.claude/state/DECISIONS.md` for the full design rationale including the rejected alternatives.

---

## Multi-Pass Leaf Accumulation (depth > 6)

### Background

`kLeafAccumSource` pre-allocates a per-thread private array of `LEAF_PRIV_SIZE = MAX_APPROX_DIM * MAX_LEAVES * 2 = 1280 floats = 5 KB`. This is the maximum that fits in GPU registers without spilling to threadgroup memory on Apple Silicon. With `MAX_LEAVES = 64`, the kernel is limited to depth ≤ 6 (2^6 = 64 leaves). Sprint 9 removes this ceiling.

### `kLeafAccumChunkedSource` kernel

The chunked kernel is identical to `kLeafAccumSource` in algorithm (private accumulation + fixed-order sequential reduction) but processes only the leaf slice `[chunkBase, chunkBase + chunkSize)` per dispatch:

- `chunkBase` and `chunkSize` are passed as scalar inputs.
- `LEAF_PRIV_SIZE` remains 1280 floats — `chunkSize ≤ LEAF_CHUNK_SIZE = 64`, so the private array size is constant regardless of total tree depth.
- A thread accumulates a document only if `partitions[doc] ∈ [chunkBase, chunkBase + chunkSize)`.
- Output shape per dispatch: `[approxDim * chunkSize]` for gradients and hessians separately.

### `ComputeLeafSumsGPUMultiPass` (`leaf_estimator.cpp`)

```
numPasses = ceil(numLeaves / 64)
for chunkBase in [0, numLeaves) step 64:
    chunkSize = min(64, numLeaves - chunkBase)
    dispatch kLeafAccumChunkedSource(chunkBase, chunkSize)
    copy chunk output into gradBuf[k * numLeaves + chunkBase .. + chunkBase + chunkSize)
```

The caller (`ComputeLeafSumsGPU`) selects single-pass or multi-pass automatically:

```
if numLeaves <= 64:
    ComputeLeafSumsGPUSinglePass(...)   // kLeafAccumSource — depth 1-6
else:
    ComputeLeafSumsGPUMultiPass(...)    // kLeafAccumChunkedSource — depth 7-10
```

### Depth range

`CB_ENSURE(numLeaves >= 2 && numLeaves <= 1024)` enforces depth 1–10. The old `CB_ENSURE(numLeaves <= 64)` guard is removed. At depth 10 (1024 leaves), 16 passes are required — negligible relative to histogram build time, which dominates per-iteration cost.

See DEC-005 in `.claude/state/DECISIONS.md` for the full design rationale.

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

`TMLXDevice::EvalNow()` forces MLX to flush the pending computation graph and synchronize (blocking CPU until all queued Metal commands have completed). Every call is a CPU-GPU round-trip and adds latency. As of Sprint 9, two `EvalNow` calls per depth level remain unavoidable (best-split readback and partition update); all other unnecessary syncs have been removed.

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
After all feature groups are dispatched and accumulated, the histogram is returned as a lazy MLX expression — no `EvalNow` is issued. The suffix-sum kernel in `FindBestSplitGPU` consumes the histogram as a kernel input, so MLX materialises the full group-accumulation graph in the same command buffer as the suffix-sum dispatch. This eliminates one CPU-GPU sync point per histogram per depth level (removed in Sprint 9).

### Removed syncs (Sprint optimizations)

- **OPT-1 (Sprint 3):** Eliminated `approxDim` CPU-GPU round trips per depth level by computing all-dimension partition statistics in a single GPU dispatch (`ComputeLeafSumsGPU` with `approxDim` handled internally).
- **BUG-001 fix (Sprint 5):** Removed the CPU-side verification loop that read back and compared histogram values to check for non-determinism. Determinism is now guaranteed by construction.
- **TODO-019 (Sprint 7):** Eliminated K `EvalNow` calls per iteration that the old per-dimension multiclass leaf loop incurred. `ComputeLeafValues` now returns a lazy MLX array; the Newton step executes over the full `[approxDim * numLeaves]` array in a single element-wise dispatch. For K=10 multiclass, this removes 10 CPU-GPU round trips per boosting iteration.
- **TODO-020 (Sprint 7):** Eliminated the O(depth) MLX bitwise-op recompute loop for partition assignments. `kTreeApplySource` now produces `partitionsOut` directly as a second kernel output — the leaf index computed inside the kernel is written out without any post-kernel recomputation on the CPU side.
- **Item-G (Sprint 9):** Eliminated `EvalNow` after histogram group accumulation in `ComputeHistogramsImpl`. The suffix-sum Metal kernel in `FindBestSplitGPU` consumes the histogram as a lazy input, so MLX folds the group-dispatch graph into the same command buffer. Also removed the unnecessary `EvalNow` in `CreateZeroHistogram`. This saves one CPU-GPU sync per histogram per depth level (i.e. `approxDim` syncs per iteration at each depth).

---

## Loss Functions (Target Functions)

Target functions live in `catboost/mlx/targets/pointwise_target.h`. Each class inherits from `IMLXTargetFunc` and implements three methods: `ComputeDerivatives` (gradients and hessians), `ComputeLoss`, and `GetApproxDimension`.

### Supported losses

| Loss | Class | Task | Notes |
|------|-------|------|-------|
| RMSE | `TRMSETarget` | Regression | L2, cursor ≈ prediction |
| Logloss | `TLoglossTarget` | Binary classification | Sigmoid link |
| CrossEntropy | `TLoglossTarget` | Binary classification | Alias for Logloss |
| MultiClass | `TMultiClassTarget` | Multi-class | Softmax, `approxDim = K` |
| MAE | `TMAETarget` | Regression | L1, sign-based gradient |
| Quantile | `TQuantileTarget(alpha)` | Regression | Asymmetric L1; default alpha=0.5 |
| Huber | `THuberTarget(delta)` | Regression | Smooth L1; default delta=1.0 |
| Poisson | `TPoissonTarget` | Count regression | Log-link; cursor is log-space |
| Tweedie | `TTweedieTarget(p)` | Zero-inflated regression | Log-link, variance power p∈(1,2); default p=1.5 |
| MAPE | `TMAPETarget` | Regression | Relative error; epsilon-clamped denominator |
| PairLogit | CPU pairwise (csv_train.cpp) | Ranking | Pairwise logistic; pairs generated from group_ids |
| YetiRank | CPU pairwise (csv_train.cpp) | Ranking | Stochastic pairs with position-dependent weights |

All 12 losses are supported in `csv_train.cpp` and are exposed through the nanobind `TrainFromArrays` path. The pointwise losses (`TRMSETarget` through `TMAPETarget`) are implemented as `IMLXTargetFunc` subclasses and compute gradients on-GPU via MLX array ops. The ranking losses (PairLogit and YetiRank) use CPU-side pair generation followed by gradient scatter onto document arrays; see the [Ranking Losses](#ranking-losses) section for details. The library path (`train.cpp`) supports the pointwise losses only.

### Gradient and hessian computation

`ComputeDerivatives` operates on MLX arrays (`cursor`, `targets`, `weights`) and returns lazy MLX expressions. The training loop calls `EvalNow` once per iteration to materialize them — gradient evaluation is not a separate sync point; it is folded into the first downstream kernel that needs the values.

---

## Ranking Losses

PairLogit and YetiRank are the two ranking losses. Both are CPU-side implementations in `csv_train.cpp` and are accessible from Python through the nanobind `TrainFromArrays` path. They require `group_ids` — documents must be pre-sorted by group.

### Pair representation

Both losses operate on document pairs `(winner, loser, weight)`. The `TPair` struct holds these three fields. After pair generation, gradients are scattered to per-document arrays via `ScatterPairwiseGradients`, which computes the sigmoid of the score difference and assigns opposing gradient signs to winner and loser.

```
p = sigmoid(pred[winner] - pred[loser])
grad[winner] += weight * (p - 1)
grad[loser]  += weight * (1 - p)
hess[winner] += weight * p * (1 - p)
hess[loser]  += weight * p * (1 - p)
```

Once document-level gradients and hessians are populated, the remaining boosting steps (histogram build, split search, leaf estimation) are identical to pointwise losses.

### PairLogit

PairLogit generates all ordered pairs `(i, j)` within each group where `target[i] > target[j]`. Pair generation happens once before training (`GeneratePairLogitPairs`) and the same pair set is reused for every iteration. All pairs have weight 1.0.

Loss function:
```
sum over pairs: weight * log(1 + exp(-(pred[winner] - pred[loser])))
```

### YetiRank

YetiRank generates a fresh pair set each iteration (`GenerateYetiRankPairs`). Within each group, documents are randomly permuted; adjacent pairs in the permutation are formed, weighted by relevance difference divided by a logarithmic position discount:

```
weight = |target[i] - target[j]| / log2(2 + position)
```

The position discount penalizes errors at higher ranks more heavily, approximating NDCG optimization. Pairs with near-zero relevance difference (`|diff| < 1e-8`) are skipped.

### group_ids requirement

Both ranking losses require document-level group IDs passed as `group_ids` to `TrainFromArrays` (or via `--group-col` on the CLI). Documents must be **pre-sorted by group** before being passed to the API — the group offset computation in `BuildDatasetFromArrays` assumes contiguous group membership. Training will error if `group_ids` is absent when a ranking loss is selected.

### Metric: NDCG

The evaluation metric for both ranking losses is NDCG (Normalized Discounted Cumulative Gain), computed per-group and averaged. The `ComputeNDCG` function in `csv_train.cpp` handles this; the loss column in training output labeled as "NDCG" is `1 - mean_NDCG` so that lower is better (consistent with other losses).

---

## Model Format and Versioning

Models are saved as JSON via `save_model` and loaded via `load_model` / `CatBoostMLX.load`. As of Sprint 10, saved files carry a `format_version` field at the top level.

### Version history

| `format_version` | Introduced | Changes |
|---|---|---|
| 1 (implicit) | Sprint 1 | Initial JSON format. No version field — `format_version` absent means 1. |
| 2 | Sprint 10 | `format_version` key explicitly written. Backward-compatible: all v1 fields unchanged. |

### Compatibility rules

- **Older files (no `format_version` key):** `load_model` treats them as version 1. No error.
- **Version 2 files on version 2 code:** Normal load path.
- **Version > 2 files on version 2 code:** `load_model` raises `ValueError` with a message instructing the user to upgrade `catboost-mlx`. This ensures future format additions do not silently produce incorrect models.

### JSON structure (`format_version=2`)

```json
{
  "format_version": 2,
  "model_info": { "loss": "...", "num_trees": N, ... },
  "trees": [ ... ],
  "features": { ... }
}
```

The `format_version` key is consumed and removed during `load_model` before the rest of the payload is processed. It does not appear in `_model_data` after loading.

---

## Two Code Paths

There are two independent implementations of the GBDT training algorithm in this repository. **This is a known architectural tension, not an accident.**

### Path 1 — csv_train path (nanobind primary, subprocess fallback)

```
python/catboost_mlx/core.py
    └─> nanobind _core.train()              (primary — compiled extension)
        └─> train_api.cpp (TrainFromArrays)
            └─> #define CATBOOST_MLX_NO_MAIN
                #include "csv_train.cpp"    (all internal functions included)

    └─> subprocess: csv_train binary        (fallback — when _core not compiled)
        └─> csv_train.cpp (standalone, no CatBoost headers)
```

`csv_train.cpp` is a **self-contained reimplementation** of the full GBDT pipeline. It mirrors `gpu_structures.h` types (`TCFeature`, `TBestSplitProperties`, etc.) locally and does not link against any CatBoost library code. When compiled for the nanobind extension, `train_api.cpp` includes `csv_train.cpp` with `CATBOOST_MLX_NO_MAIN` defined, which suppresses `main()` and exposes all internal functions and types for use by `TrainFromArrays`. When the compiled extension is unavailable, `core.py` falls back to spawning the `csv_train` binary as a subprocess.

This path supports all 12 losses (including PairLogit and YetiRank), all three grow policies, and the full hyperparameter surface exposed by `TTrainConfig`.

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

This path uses the full CatBoost library, integrates with `TTrainingDataProviders`, and produces a `TFullModel` compatible with CatBoost's native format. It currently supports the 10 pointwise losses only.

### Why this matters

BUG-001 (non-deterministic histogram accumulation) was discovered in the `csv_train` path during QA testing. Because `csv_train.cpp` duplicates kernel dispatch logic independently, fixes must be applied to **both paths separately** — the fix in `csv_train.cpp` does not propagate to `histogram.cpp`, `score_calcer.cpp`, or `leaf_estimator.cpp`, and vice versa.

Before touching any kernel source or dispatch pattern, check whether the change applies to both paths. If `csv_train.cpp` and the library path diverge in kernel behavior, integration tests will fail in non-obvious ways.

The authoritative kernel source strings are in `catboost/mlx/kernels/kernel_sources.h`. Both paths include this header. However, the dispatch wrappers (grid dimensions, input arrays, threadgroup sizes) are separate in each path.

---

## Nanobind Binding Architecture

The compiled Python extension is the primary interface between Python and the Metal GPU engine. It lives in `python/catboost_mlx/_core/`.

### Key files

| File | Role |
|---|---|
| `catboost/mlx/train_api.h` | Public API types: `TTrainConfig`, `TTrainResultAPI`, `TrainFromArrays()` |
| `catboost/mlx/train_api.cpp` | `TrainFromArrays()` implementation; includes `csv_train.cpp` via the include trick |
| `python/catboost_mlx/_core/bindings.cpp` | Nanobind module `_core`; exposes `TrainConfig`, `TrainResult`, and `train()` to Python |

### The include trick

`train_api.cpp` includes the entire `csv_train.cpp` translation unit with a preprocessor guard:

```cpp
#define CATBOOST_MLX_NO_MAIN
#include "catboost/mlx/tests/csv_train.cpp"
```

When `CATBOOST_MLX_NO_MAIN` is defined, `csv_train.cpp` omits its `main()` function but exposes all internal functions and types (`TConfig`, `TDataset`, `RunTraining`, etc.) into `train_api.cpp`'s scope. `TrainFromArrays` then converts the public `TTrainConfig` to the internal `TConfig`, builds a `TDataset` from the flat numpy arrays, and calls `RunTraining` directly. The CLI binary is compiled separately from `csv_train.cpp` without the define, producing a standalone executable with `main()`.

### `TTrainConfig` and `TConfig`

`TTrainConfig` (in `train_api.h`) is the public API struct. It mirrors `TConfig` exactly except for five CLI-specific fields that have no meaning in the in-process path: `CsvPath`, `TargetCol`, `OutputModelPath`, `CVFolds`, and `EvalFile`. The `TrainConfigToInternal` function in `train_api.cpp` converts between them; all defaults must be kept in sync.

### GIL release

The nanobind `train()` wrapper releases the GIL for the entire duration of Metal GPU training:

```cpp
{
    nb::gil_scoped_release release;
    result = TrainFromArrays(...);
}
```

All Metal and MLX calls are thread-safe. Python objects are not accessed inside `TrainFromArrays`. Releasing the GIL allows other Python threads to run during training (relevant for multi-threaded experiment harnesses).

### Zero-copy numpy array access

Numpy arrays are passed as `nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>`. Nanobind checks that the array is C-contiguous and on CPU, then provides a raw pointer via `.data()` with no copy. The pointer is valid only for the duration of the GIL-release block; `TrainFromArrays` must not retain it after return.

### Subprocess fallback

`core.py` tries `from . import _core as _nb_core` at import time. If the extension is absent (e.g., an editable install without a build step), `_nb_core` is set to `None` and `core.py` routes all training calls through the subprocess path instead. The subprocess path writes data to disk, invokes the `csv_train` binary, and parses JSON output — functionally equivalent but higher latency.

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

Apple Silicon's unified memory architecture means GPU and CPU share physical memory. Large histogram buffers (`numPartitions * numStats * totalBinFeatures * 4 bytes`) and the compressed feature matrix (`numDocs * numColumns * 4 bytes`) both live in this shared space. At very large row counts the compressed feature matrix can exceed 1 GB — monitor memory allocation with Metal's GPU counters when benchmarking large datasets.

### DEC-003 resolved (Sprint 9)

As of Sprint 9, `ComputePartitionLayout` uses int32 accumulators in `scatter_add_axis`, which are exact for values up to 2^31 (~2.1B docs). The previous float32 accumulator limited exact integer representation to 2^24 = 16,777,216 rows; that ceiling has been removed. See DEC-003 in `.claude/state/DECISIONS.md` for the full history.

ADR-005's choice of float32 for gradient accumulation is unaffected — that policy applies to gradient and hessian sums, not to partition document counts.
