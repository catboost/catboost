# S26-D0-8 Root Cause: Depthwise / Lossguide Prediction Failure

**Date**: 2026-04-22  
**Branch**: `mlx/sprint-26-python-parity`  
**DEC**: DEC-029  
**Status**: Fix implemented, pending build + verification

---

## 1. Root Cause

**Primary bug**: `TTreeRecord.SplitProps` was never populated for Depthwise or
Lossguide trees. As a result, `WriteModelJSON` emitted `"splits": []` for every
non-oblivious tree in the model JSON.

**Secondary bug (same root)**: `_predict_utils.py:compute_leaf_indices` assumed
every tree was oblivious (SymmetricTree) and used bit-packed leaf indexing, which
only works when `splits` has exactly `depth` entries and all nodes at a given
depth level share the same split. For a Depthwise tree, `splits` was always empty,
so every sample was assigned to leaf 0.

**Bug class**: Model export / prediction dispatch — not a training correctness
issue. The training `cursor` (internal accumulated predictions) was correct;
the bug was in the path `model JSON → Python predict()`.

---

## 2. Evidence

### Failure mechanism (pre-fix)

After fitting a Depthwise model:
1. `TTreeRecord.SplitProps = splitProps` (line 3819 of csv_train.cpp).
2. `splitProps` was populated only in the SymmetricTree branch (`splitProps.push_back(bestSplit)`,
   line 3619). Depthwise and Lossguide never called this line.
3. `splitProps` remained empty for the entire tree build.
4. `WriteModelJSON` loops `for si in range(len(tree.SplitProps)):` — 0 iterations.
5. JSON: `"splits": []` for every Depthwise/Lossguide tree.
6. `compute_leaf_indices`: `for level, split in enumerate(tree["splits"])` — 0 iterations.
7. `leaf_idx = all zeros`. Every doc assigned to leaf 0.
8. `predict(X) = base_prediction + sum_over_trees(leaf_values[0])` = constant for all docs.
9. RMSE vs. a model with real predictions: 560–598% (constant prediction ≈ mean of y ≈ 0,
   while residuals are O(0.2) — the constant is wrong, not just imprecise).

### Confirmation by code reading

Files examined:
- `csv_train.cpp:3619` — `splitProps.push_back(bestSplit)` exists ONLY in the
  `else` (SymmetricTree) branch of `if (isDepthwise) { ... } else { ... }`.
- `csv_train.cpp:3318` — Lossguide path: `splits.push_back(ns)` but no
  `splitProps.push_back(...)`.
- `_predict_utils.py:compute_leaf_indices` — no `grow_policy` dispatch;
  always calls the oblivious path.

---

## 3. Fix

### C++ side (`csv_train.cpp`)

**3a. `TTreeRecord` struct** — add `SplitBfsNodeIds` field:
```cpp
std::vector<ui32> SplitBfsNodeIds;  // BFS node index per split (Depthwise+Lossguide)
```

**3b. Depthwise path** — in the partition loop, alongside `splits.push_back(nodeSplit)`:
```cpp
splitProps.push_back(perPartSplits[p]);  // populates SplitProps for model JSON
// Compute BFS node index for partition p at depth `depth`
ui32 bfsNode = 0u;
for (ui32 lvl = 0; lvl < depth; ++lvl) {
    uint32_t goRight = (p >> lvl) & 1u;
    bfsNode = 2u * bfsNode + 1u + goRight;
}
splitBfsNodeIds.push_back(bfsNode);
```

**3c. Lossguide path** — alongside `splits.push_back(ns)`:
```cpp
splitProps.push_back(bestSp);
splitBfsNodeIds.push_back(bfsNode);  // bfsNode already known in this scope
```

**3d. `WriteModelJSON`** — add `grow_policy` field per tree, and emit
`"bfs_node_index"` on each split entry for non-oblivious trees:
```json
{
  "grow_policy": "Depthwise",
  "splits": [
    {"feature_idx": 3, "bin_threshold": 5, "is_one_hot": false, "bfs_node_index": 0},
    {"feature_idx": 7, "bin_threshold": 2, "is_one_hot": false, "bfs_node_index": 1},
    {"feature_idx": 1, "bin_threshold": 9, "is_one_hot": false, "bfs_node_index": 2}
  ],
  "leaf_values": [...],
  "leaf_bfs_ids": [3, 4, 5, 6]   // Lossguide only
}
```

### Python side (`_predict_utils.py`)

**3e. `compute_leaf_indices`** — dispatch on `tree.get("grow_policy", "SymmetricTree")`:
- `SymmetricTree` → `_compute_leaf_indices_oblivious` (existing bit-packing, unchanged)
- `Depthwise` → `_compute_leaf_indices_depthwise` (new: BFS traversal, bit-packed result)
- `Lossguide` → `_compute_leaf_indices_lossguide` (new: BFS traversal, dense leaf id result)

**3f. Key insight for Depthwise**: the leaf_values array is indexed by the
training-time **bit-packed partition value** (bit `d` = direction at depth `d`).
BFS traversal must accumulate the same bit-packed value: at BFS node `n` of depth
`d = floor(log2(n+1))`, a right-turn sets bit `d`. This is NOT the same as
`BFS_node_index - num_nodes` (which gives left-to-right BFS leaf order, a different
permutation). Using `bfs_node_index` emitted by C++ avoids any index arithmetic confusion.

**3g. `_build_bfs_node_map`** — extract `{bfs_node_index: split}` from the splits
list, skipping no-op entries where `feature_idx == 0xFFFFFFFF`.

---

## 4. Depthwise vs Lossguide — Same Bug?

Yes. Mechanically identical: `SplitProps` never populated → empty `splits` in
JSON → all-zeros leaf assignment.

The RMSE deltas differ (560% vs 598%) because:
- Depthwise produces a depth-6 symmetric-shaped tree (always 64 leaves at depth 6)
  with `leaf_values[0]` typically near the global mean.
- Lossguide produces an unbalanced tree with up to `max_leaves=31` leaves;
  `leaf_values[0]` may differ slightly, explaining the 38-point difference in delta.

The fix is structurally the same for both. DEC-029 covers both.

---

## 5. Verification (expected post-fix)

**localize.py targets** (50 iterations, depth=6, LR=0.03, rs=0):

| policy        | CPU RMSE  | MLX RMSE  | delta %     |
|---------------|-----------|-----------|-------------|
| SymmetricTree | ~0.2010   | ~0.1948   | ~3.1% (DEC-028, unchanged)  |
| Depthwise     | ~0.1950   | expected ≤ 5% of CPU |
| Lossguide     | ~0.1970   | expected ≤ 5% of CPU |

**one_tree_depthwise.py targets** (1 iteration, LR=1, rs=0):

| policy × depth | std_ratio (MLX/CPU) | expected |
|----------------|---------------------|----------|
| SymmetricTree d=2 | ≈ 1.0  | control |
| Depthwise d=2     | ≈ 1.0  | main test |
| Lossguide d=2     | ≈ 1.0  | main test |
| Depthwise d=6     | ≈ 1.0  | higher depth |
| Lossguide d=6     | ≈ 1.0  | higher depth |

If std_ratio is 0 (as before the fix), all docs are in leaf 0.

---

## 6. Known Remaining Issue

`ComputeLeafIndicesDepthwise` in csv_train.cpp (C++ validation path) returns
`nodeIdx - numNodes` (BFS leaf order), but `leafValues` is indexed by the
bit-packed partition. For validation data during training with Depthwise, this
produces wrong validation RMSE tracking. This is a pre-existing issue (does not
affect training correctness or Python prediction after this fix) and will be
addressed in a follow-up sprint.

---

## 7. Files Changed

- `catboost/mlx/tests/csv_train.cpp`:
  - `TTreeRecord`: add `SplitBfsNodeIds`
  - `RunTraining`: declare `splitBfsNodeIds`, populate it in Depthwise and Lossguide
    paths alongside existing `splits`/`splitProps` population
  - `WriteModelJSON`: add `grow_policy` field; emit `bfs_node_index` for non-oblivious splits;
    emit `leaf_bfs_ids` for Lossguide
- `python/catboost_mlx/_predict_utils.py`:
  - `compute_leaf_indices`: dispatch on `grow_policy`
  - `_compute_leaf_indices_oblivious`: extracted from existing `compute_leaf_indices`
  - `_build_bfs_node_map`: extracts `{bfs_node_idx: split}` from splits list
  - `_bfs_traverse_bitpacked`: BFS traversal accumulating bit-packed partition value (Depthwise)
  - `_compute_leaf_indices_depthwise`: delegates to above
  - `_compute_leaf_indices_lossguide`: BFS traversal returning dense leaf id via `leaf_bfs_ids`
- `benchmarks/sprint26/d0/one_tree_depthwise.py`: new 1-iter diagnostic
- `docs/sprint26/d0/depthwise-lossguide-root-cause.md`: this file
