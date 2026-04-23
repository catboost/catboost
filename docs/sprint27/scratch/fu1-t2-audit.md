# S27-FU-1-T2: CPU-Source Audit and Fix Specification

**Status**: Complete — DEC-030 drafted
**Date**: 2026-04-22
**Author**: @ml-engineer
**Branch**: `mlx/sprint-27-correctness-closeout`
**Feeds**: T3 (implementation), DEC-030

---

## Part 1 — CPU-Source Audit

### 1.1 The canonical CatBoost CPU non-symmetric traversal

`catboost/libs/model/cpu/evaluator_impl.cpp` lines 462–492
(`CalcIndexesNonSymmetric`):

```cpp
std::fill(indexesVec + firstDocId, indexesVec + docCountInBlock,
          trees.GetModelTreeData()->GetTreeStartOffsets()[treeId]);
while (countStopped != docCountInBlock - firstDocId) {
    countStopped = 0;
    for (size_t docId = firstDocId; docId < docCountInBlock; ++docId) {
        const TNonSymmetricTreeStepNode* stepNode = treeStepNodes + indexesVec[docId];
        const TRepackedBin split = treeSplitsPtr[indexesVec[docId]];
        ui8 featureValue = binFeatures[split.FeatureIndex * docCountInBlock + docId];
        const auto diff = (featureValue >= split.SplitIdx)
                            ? stepNode->RightSubtreeDiff
                            : stepNode->LeftSubtreeDiff;
        countStopped += (diff == 0);
        indexesVec[docId] += diff;
    }
}
```

This is the CPU canonical traversal. Key observations:

1. **Step-node indexed array, not formula**: the CPU stores `TNonSymmetricTreeStepNode` for every
   BFS position with non-zero `LeftSubtreeDiff` / `RightSubtreeDiff`. Leaf resolution uses
   `NonSymmetricNodeIdToLeafId[nodeIndex]` — an explicit per-node map built at model-save time.
   There is no `nodeIdx − numNodes` arithmetic anywhere.

2. **BFS node → leaf value is an explicit map**: `model.cpp:606–608`:
   ```cpp
   const auto& node = GetModelTreeData()->GetNonSymmetricStepNodes()[nodeIndex];
   if (node.LeftSubtreeDiff == 0 || node.RightSubtreeDiff == 0) {
       const ui32 leafValueIndex = GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[nodeIndex];
   }
   ```
   CatBoost never assumes `leafValueIndex = nodeIdx − numNodes`. The leaf-value array is indexed
   through `NonSymmetricNodeIdToLeafId`, which is built in leaf-training-order, not BFS-position order.

3. **Implication for the MLX port**: The MLX project chose a different representation — bit-packed
   partition order — for its leaf-value array (established in DEC-029 and the training path at
   `csv_train.cpp:3660–3683`). The canonical CPU authority for this specific representation is not
   `evaluator_impl.cpp` (which uses a different format), but the MLX training path's own partition
   accumulation loop. **The CPU authority confirms that `nodeIdx − numNodes` is never used in
   CatBoost CPU to resolve a leaf value.**

### 1.2 MLX canonical encoding — the authoritative reference for this port

The MLX port's canonical leaf-index encoding is defined at two complementary sites:

**Site A — partition accumulation at `csv_train.cpp:3660–3683`**:

```cpp
auto bits = mx::left_shift(updateBits,
    mx::array(static_cast<uint32_t>(depth), mx::uint32));
partitions = mx::bitwise_or(partitions, bits);
```

After `D` depth levels: `partitions[d] = goRight[0] | (goRight[1] << 1) | ... | (goRight[D-1] << D-1)`
This is the bit-packed leaf index — bit `k` = go-right at depth `k`.

The code comment at `csv_train.cpp:3991` makes this explicit:
```cpp
// For depthwise trees, partitions is already the correct leaf index (same bit-encoding
// as oblivious trees — the partition update loop sets bit `depth` based on per-node splits).
```

**Site B — BFS-to-partition mapping at `csv_train.cpp:3644–3654`** (DEC-029):

```cpp
// DEC-029: compute BFS node index for partition p at depth `depth`.
// Partition p is a bit-packed right-turn vector: bit k = direction at depth k.
// BFS node is derived by traversing the tree following bits 0..depth-1 of p.
ui32 bfsNode = 0u;
for (ui32 lvl = 0; lvl < depth; ++lvl) {
    ui32 goRight = (p >> lvl) & 1u;
    bfsNode = 2u * bfsNode + 1u + goRight;
}
splitBfsNodeIds.push_back(bfsNode);
```

This loop is the authoritative forward map: partition `p` → BFS node. The fix requires its
inverse: BFS node → partition `p`.

**Canonical citation**: `csv_train.cpp:3660–3683` and `csv_train.cpp:3644–3654`.

### 1.3 Why `nodeIdx − numNodes` is wrong

For a balanced complete binary tree of depth `D`, the BFS node indices of the `2^D` leaves are
`numNodes..2*numNodes−1` where `numNodes = 2^D − 1`. `nodeIdx − numNodes` gives the BFS leaf's
0-based rank among leaves in left-to-right BFS order.

BFS leaf order is **not** the same as bit-packed partition order. The BFS traversal enumerates
depth-`D` nodes as:
- BFS index `numNodes` = all-left path → bit-packed `0b000...0`  (match)
- BFS index `numNodes+1` = RL...L path → bit-packed `0b000...1`  (mismatch for D >= 2)

The two orderings agree only when `D == 1` (one split, two leaves: BFS-offset 0 = left, 1 = right;
bit-packed 0 = left, 1 = right). For all `D >= 2`, they diverge.

---

## Part 2 — Fix Approach Specification (for T3)

### 2.1 Root cause summary

Two independent bugs in `ComputeLeafIndicesDepthwise`:

- **Bug A**: `leafVec[d] = nodeIdx - numNodes` uses BFS leaf offset, not bit-packed partition.
- **Bug B**: `nodeSplits[nodeIdx]` indexes the flat `splits` vector by BFS index, but `splits`
  is built in partition order (bit-packed order per depth level). At depth >= 3, positions
  3..6 of `splits` hold splits for BFS nodes 3, 5, 4, 6 — not 3, 4, 5, 6.

### 2.2 Data structures for the fix

The `splitBfsNodeIds` side-array is already available at the call site
(`csv_train.cpp:4040`): it is built in the same loop that builds `splits`
(`csv_train.cpp:3638` + `3654`) and passed along with `splits`.

T3 must thread `splitBfsNodeIds` into `ComputeLeafIndicesDepthwise`:

```
function ComputeLeafIndicesDepthwise(
    compressedData,
    splits,              // flat array, partition order
    splitBfsNodeIds,     // parallel array: splitBfsNodeIds[i] = BFS node for splits[i]
    numDocs,
    depth
) -> array[numDocs]
```

### 2.3 Pre-traversal setup: BFS-node-to-split-index map

Build once per call, before the doc loop:

```pseudo
// Build nodeSplitMap: BFS node index → TObliviousSplitLevel
std::unordered_map<ui32, TObliviousSplitLevel> nodeSplitMap;
nodeSplitMap.reserve(splits.size());
for i in 0..splits.size()-1:
    bfsIdx = splitBfsNodeIds[i]
    nodeSplitMap[bfsIdx] = splits[i]
```

This replaces `nodeSplits[nodeIdx]` (flat-array BFS-indexed access, which is Bug B) with
`nodeSplitMap.at(nodeIdx)` (BFS-keyed lookup, correct). This mirrors `ComputeLeafIndicesLossguide`
which already uses `const std::unordered_map<ui32, TObliviousSplitLevel>& nodeSplitMap`.

### 2.4 Traversal loop: accumulate bit-packed partition

For each doc, traverse from root, accumulating goRight bits:

```pseudo
for d in 0..numDocs-1:
    nodeIdx = 0
    partBits = 0
    for lvl in 0..depth-1:
        ns = nodeSplitMap.at(nodeIdx)     // BFS-keyed, fixes Bug B
        fv = extract_feature(dataPtr, d, ns)
        goRight = (ns.IsOneHot) ? (fv == ns.BinThreshold ? 1 : 0)
                                : (fv >  ns.BinThreshold ? 1 : 0)
        partBits |= (goRight << lvl)      // accumulate bit k at depth k, fixes Bug A
        nodeIdx = 2 * nodeIdx + 1 + goRight
    leafVec[d] = partBits
```

`partBits` after `depth` iterations equals the bit-packed partition, matching `partitions[d]`
from the training path. This directly replaces `nodeIdx − numNodes`.

### 2.5 Signature change and call-site threading

Current signature:
```cpp
mx::array ComputeLeafIndicesDepthwise(
    const mx::array& compressedData,
    const std::vector<TObliviousSplitLevel>& nodeSplits,
    ui32 numDocs,
    ui32 depth
);
```

Required signature:
```cpp
mx::array ComputeLeafIndicesDepthwise(
    const mx::array& compressedData,
    const std::vector<TObliviousSplitLevel>& splits,
    const std::vector<ui32>& splitBfsNodeIds,   // new: parallel BFS-index array
    ui32 numDocs,
    ui32 depth
);
```

Call site at `csv_train.cpp:4040` already has `splitBfsNodeIds` in scope (built at line 3654).
T3 must update the call to pass it.

### 2.6 Edge cases

| Case | Behavior | Notes |
|------|----------|-------|
| `depth == 0` | Return `mx::zeros` immediately (existing early return) | No change. Correct — single leaf, index always 0. |
| `depth == 1` | Both bugs are invisible (BFS offset = bit-packed for 1 level) | Fix is still correct and harmless. |
| `valDocs == 0` | Function never called — gated at `csv_train.cpp:4040` | No change needed. |
| No-op splits (Mask == 0) | `nodeSplitMap` contains the no-op entry; `goRight` is always 0; `partBits` bit k stays 0. Equivalent to existing training-path behavior. | Correct. |
| BFS node not in `nodeSplitMap` | Should not occur for a well-formed depth-`D` complete tree. T3 may add a `CB_ENSURE(nodeSplitMap.count(nodeIdx))` assertion for debug builds. | Defense in depth. |

### 2.7 Mirror to `ComputeLeafIndicesLossguide`

The Lossguide version already holds the correct pattern:
- Takes `nodeSplitMap` (BFS-keyed) instead of a flat `nodeSplits` vector (addresses Bug B analogue).
- Resolves the terminal node via `bfsToLeafId[nodeIdx]` (an explicit inverse map) instead of arithmetic (addresses Bug A analogue).

For Depthwise, the final resolution is simpler than Lossguide: because the tree is a complete
binary tree of fixed depth, the bit-packed accumulation already IS the leaf index — no explicit
`bfsToLeafId` map is needed. The fix accumulates `partBits` inline during traversal.

---

## Part 3 — Implementation Checklist for T3

1. Add `splitBfsNodeIds` parameter to `ComputeLeafIndicesDepthwise`.
2. Build `nodeSplitMap` (unordered_map, BFS index → TObliviousSplitLevel) from `splits` + `splitBfsNodeIds`.
3. Replace `nodeSplits[nodeIdx]` with `nodeSplitMap.at(nodeIdx)`.
4. Replace `partitions` accumulation: add `partBits |= (goRight << lvl)` inside the depth loop.
5. Replace `leafVec[d] = nodeIdx − numNodes` with `leafVec[d] = partBits`.
6. Thread `splitBfsNodeIds` into the call at `csv_train.cpp:4040`.
7. Remove `const ui32 numNodes = (1u << depth) - 1u;` (no longer used).
8. Optional: `CB_ENSURE` that every `nodeIdx` visited is in `nodeSplitMap` (debug assertion).
