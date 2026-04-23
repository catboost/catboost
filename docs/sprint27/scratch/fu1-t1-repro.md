# S27-FU-1-T1: ComputeLeafIndicesDepthwise — Repro and Code Audit

**Status**: BUG CONFIRMED — 103/200 mismatches at depth=3, 112/200 at depth=2.
**Date**: 2026-04-22
**Author**: @qa-engineer
**File under**: DEC-029 Risks, Sprint 27 Track A

---

## 1. Code Audit

### 1a. `ComputeLeafIndicesDepthwise` — current implementation

File: `catboost/mlx/tests/csv_train.cpp:1751–1796`

```cpp
mx::array ComputeLeafIndicesDepthwise(
    const mx::array& compressedData,
    const std::vector<TObliviousSplitLevel>& nodeSplits,
    ui32 numDocs,
    ui32 depth
) {
    if (depth == 0) {
        return mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
    }
    const ui32 numNodes = (1u << depth) - 1u;
    mx::eval(compressedData);
    auto flatData = mx::reshape(compressedData, {static_cast<int>(numDocs), -1});
    mx::eval(flatData);
    const uint32_t* dataPtr = flatData.data<uint32_t>();
    const ui32 lineSize = static_cast<ui32>(flatData.shape(1));

    std::vector<uint32_t> leafVec(numDocs);
    for (ui32 d = 0; d < numDocs; ++d) {
        ui32 nodeIdx = 0u;
        for (ui32 lvl = 0; lvl < depth; ++lvl) {
            const auto& ns = nodeSplits[nodeIdx];   // BUG B: flat pos used as BFS idx
            uint32_t packed = dataPtr[d * lineSize + ns.FeatureColumnIdx];
            uint32_t fv = (packed >> ns.Shift) & ns.Mask;
            uint32_t goRight = ns.IsOneHot ? (fv == ns.BinThreshold ? 1u : 0u)
                                           : (fv > ns.BinThreshold ? 1u : 0u);
            nodeIdx = 2u * nodeIdx + 1u + goRight;
        }
        leafVec[d] = nodeIdx - numNodes;             // BUG A: BFS-offset, not bit-packed
    }
    return mx::array(reinterpret_cast<const int32_t*>(leafVec.data()),
        {static_cast<int>(numDocs)}, mx::uint32);
}
```

**Two bugs are marked above.** Both must be fixed together — they are not independent.

### 1b. `ComputeLeafIndicesLossguide` for comparison

File: `catboost/mlx/tests/csv_train.cpp:1802–1842`

```cpp
mx::array ComputeLeafIndicesLossguide(
    const mx::array& compressedData,
    const std::unordered_map<ui32, TObliviousSplitLevel>& nodeSplitMap,  // keyed by BFS index
    const std::vector<ui32>& leafBfsIds,
    ui32 numDocs,
    ui32 numLeaves
) {
    // Build inverse map: BFS node index → dense leaf id.
    std::unordered_map<ui32, ui32> bfsToLeafId;
    for (ui32 k = 0; k < static_cast<ui32>(leafBfsIds.size()); ++k)
        bfsToLeafId[leafBfsIds[k]] = k;

    // ... traversal using nodeSplitMap.at(nodeIdx) (BFS-keyed lookup) ...
    // Final: bfsToLeafId[nodeIdx] — uses an explicit BFS→dense mapping.
}
```

The Lossguide version avoids both bugs: it uses a BFS-keyed map for lookup (not a flat array indexed by BFS position), and it maps the final BFS node to a dense leaf via an explicit inverse map, not arithmetic subtraction.

### 1c. Training-path encoding (the canonical reference)

File: `catboost/mlx/tests/csv_train.cpp:3660–3683` (Depthwise partition-update loop):

```cpp
// Update partitions for all partitions in one vectorised pass.
auto bits = mx::left_shift(updateBits,
    mx::array(static_cast<uint32_t>(depth), mx::uint32));
partitions = mx::bitwise_or(partitions, bits);
```

After depth levels, `partitions[d] = goRight[0] | (goRight[1] << 1) | ... | (goRight[D-1] << D-1)`.
This is the bit-packed leaf index. It is the key used for `scatter_add_axis` at line 3819 to build
`leafValues`, and for `mx::take(leafValues, partitions)` at line 3996. The validation path must
return this same encoding.

The code comment at line 3991 is explicit:
```cpp
// For depthwise trees, partitions is already the correct leaf index (same bit-encoding
// as oblivious trees — the partition update loop sets bit `depth` based on per-node splits).
```

---

## 2. Encoding Reference — CPU-side canonical encoding

The MLX project does not use CatBoost's `TNonSymmetricTreeStepNode` model format for inference
(it uses its own JSON-based predictor). The canonical encoding for this project is defined by
the training-path partition accumulation above. The CPU-side CatBoost reference that most directly
corresponds is `catboost/libs/model/cpu/evaluator_impl.cpp` around line 480–493, which uses
`TNonSymmetricTreeStepNode.LeftSubtreeDiff` / `RightSubtreeDiff` for non-symmetric traversal —
a different representation. The MLX project chose bit-packed partition as its leaf-index encoding.
The validation path must match.

---

## 3. Two-Bug Analysis

There are **two orthogonal bugs** in `ComputeLeafIndicesDepthwise`:

### Bug A — Encoding mismatch (depth >= 2)

`nodeIdx - numNodes` returns the **BFS-array-leaf-offset**: among all `2^depth` leaf nodes
(BFS indices `numNodes..2*numNodes`), it is the 0-based position counting from the first leaf.

But `leafValues` is indexed by the **bit-packed partition**: `bit k = goRight at depth k`.

For a depth-2 tree, the mapping is:

| Path | BFS nodeIdx | BFS-offset (buggy) | Bit-packed (correct) | Match? |
|------|-------------|-------------------|----------------------|--------|
| LL   | 3           | 0                  | 0b00 = 0             | YES    |
| LR   | 4           | 1                  | 0b10 = 2             | NO     |
| RL   | 5           | 2                  | 0b01 = 1             | NO     |
| RR   | 6           | 3                  | 0b11 = 3             | YES    |

Samples on mixed-direction paths (LR or RL) are routed to the wrong `leafValues` entry.
At depth=3, the same permutation applies within each subtree.

### Bug B — Split-lookup mismatch (depth >= 3)

`nodeSplits[nodeIdx]` indexes into the `splits` vector using the BFS node index `nodeIdx`.
But `splits` is built by appending partitions at each depth level in **bit-packed partition order**
(p = 0, 1, ..., 2^d - 1 for each depth d). The correspondence between partition index and BFS
node index is:

At depth d, partition `p` maps to BFS node:
```
bfs = 0; for lvl in 0..d-1: bfs = 2*bfs + 1 + ((p >> lvl) & 1)
```

For depth=3 (the first depth where Bug B fires), depth-level 2 holds positions [3..6] in `splits`:

| Flat pos | Partition p | BFS node |
|----------|-------------|----------|
| 3        | p=0 (0b00)  | 3        |
| 4        | p=1 (0b01)  | 5        |  ← flat[4] has BFS=5, not BFS=4
| 5        | p=2 (0b10)  | 4        |  ← flat[5] has BFS=4, not BFS=5
| 6        | p=3 (0b11)  | 6        |

`nodeSplits[4]` returns the split for BFS node 5, and `nodeSplits[5]` returns BFS node 4.
Any traversal that reaches BFS nodes 4 or 5 at depth level 2 uses the wrong split descriptor
(wrong feature, wrong threshold), producing a wrong go-right decision. Combined with Bug A,
this compounds the error.

---

## 4. Evidence Table

Output from `docs/sprint27/scratch/fu1_repro.py` (N=200, depth=3, iters=5, seed=7, bins=8):

### Structural diagnosis

Every tree at depth=3 has `flat_ids != bfs_ids`:
```
Tree | Flat positions           | BFS indices              | flat==bfs
   0 | [0, 1, 2, 3, 4, 5, 6]   | [0, 1, 2, 3, 5, 4, 6]   | False
   1 | [0, 1, 2, 3, 4, 5, 6]   | [0, 1, 2, 3, 5, 4, 6]   | False
   2 | [0, 1, 2, 3, 4, 5, 6]   | [0, 1, 2, 3, 5, 4, 6]   | False
   3 | [0, 1, 2, 3, 4, 5, 6]   | [0, 1, 2, 3, 5, 4, 6]   | False
   4 | [0, 1, 2, 3, 4, 5, 6]   | [0, 1, 2, 3, 5, 4, 6]   | False
```

### Per-sample evidence table (tree 0, first 15 rows)

```
sample | expected_bfs_leaf | actual_from_code | match | correct_lv |   buggy_lv
     0 |                 5 |                4 |    NO |   0.101525 |   0.000000
     1 |                 2 |                2 |   YES |  -0.123936 |  -0.123936
     2 |                 2 |                2 |   YES |  -0.123936 |  -0.123936
     3 |                 1 |                4 |    NO |  -0.041390 |   0.000000
     4 |                 0 |                0 |   YES |  -0.263360 |  -0.263360
     5 |                 0 |                0 |   YES |  -0.263360 |  -0.263360
     6 |                 5 |                4 |    NO |   0.101525 |   0.000000
     7 |                 5 |                4 |    NO |   0.101525 |   0.000000
     8 |                 2 |                2 |   YES |  -0.123936 |  -0.123936
     9 |                 5 |                5 |   YES |   0.101525 |   0.101525
    10 |                 5 |                4 |    NO |   0.101525 |   0.000000
    11 |                 3 |                6 |    NO |   0.139368 |   0.011305
    12 |                 6 |                3 |    NO |   0.011305 |   0.139368
    13 |                 5 |                5 |   YES |   0.101525 |   0.101525
    14 |                 5 |                5 |   YES |   0.101525 |   0.101525
```

**Mismatch count (tree 0, N=200): 103/200 (51.5%)**

Note the non-symmetric error: sample 11 gets leaf 3 instead of 6 (value +0.139 instead of +0.011),
while sample 12 gets leaf 6 instead of 3 (value +0.011 instead of +0.139). The swap is symmetric
within the pair but the wrong samples are mapped to wrong leaf values.

---

## 5. Characterization

### Accumulated prediction error

```
Correct encoding RMSE:   1.444203
Buggy encoding RMSE:     1.618473
C++ val_loss (buggy):    1.618473
Ratio buggy/correct:     1.1207
```

The C++ `eval_loss_history` matches the buggy encoding RMSE exactly — confirming the harness
accurately simulates the C++ validation path. The bug inflates validation RMSE by ~12% for
this config (N=200, depth=3, 5 trees). The ratio will vary with tree structure but will always
be >= 1.0 (validation RMSE can only be overstated, not understated, since wrong leaf values
are generally less accurate than correct ones; however, there exist configurations where a
lucky swap reduces the reported RMSE).

### Depth-conditional fire condition

```
depth=1: struct_mismatch=False  sample_mismatches=0/200    (no bug)
depth=2: struct_mismatch=False  sample_mismatches=112/200  (Bug A only)
depth=3: struct_mismatch=True   sample_mismatches=74-103/200 (Bug A + Bug B)
depth=4: struct_mismatch=True   sample_mismatches=124/200  (Bug A + Bug B, more positions)
```

- **depth=1**: No bug. Only path is L or R; BFS-offset = bit-packed = 0 or 1 (same for 1 level).
- **depth=2**: Bug A fires. 56% of samples affected — exactly those that take mixed-direction
  paths (LR or RL). Bug B does not fire because no internal nodes at depth-2 are ever visited
  during traversal (traversal reaches depth-2 nodes as leaves, not as `nodeIdx` values used
  to index `nodeSplits`).
- **depth=3+**: Both bugs fire. The mismatch rate is not simple to predict because Bug B's
  wrong split descriptor causes wrong go-right decisions, changing which leaf is reached and
  potentially moving a sample from a mismatched leaf to a matched one or vice versa.

### Scope: does this affect training RMSE?

No. The training path uses `partitions` (bit-packed) directly for `scatter_add_axis` and
`mx::take`. `ComputeLeafIndicesDepthwise` is only called in the validation path (line 4040),
where it feeds `valCursor`. Training correctness is unaffected.

### Does the bug affect `use_best_model=True` selection?

Yes — when an eval set is provided, the per-iteration `val_loss` values (pushed to
`result.EvalLossHistory`) are computed from the buggy `valCursor`. If the bug inflates
validation RMSE, the best-iteration selection will find a different optimum than the correct
implementation. In the N=200 example above, the buggy RMSE is 12% higher; over 50+ iterations
the best-iteration estimate shifts accordingly.

### Is there a configuration where the bug is invisible?

Yes:
- **depth=1**: No bug fires (single split level, encoding is trivially identical).
- **Samples on LL or RR paths**: Both bugs produce correct results because both encodings agree
  at these paths (matched rows in the evidence table, e.g. samples 4,5,8).
- **Trees where no-op nodes dominate**: If many partitions at depth 2 are no-op splits
  (Mask=0), many samples land on leaf_value=0 both before and after the bug fires — RMSE
  difference is suppressed near iteration 0 when trees are shallow and gains are small.

---

## 6. Encoding Reference for T2

T2 (@ml-engineer) should audit `catboost/libs/model/cpu/evaluator_impl.cpp:462–493`
(`CalcNonSymmetricTreesSingle` / `CalcNonSymmetricTrees`) to understand the CPU-canonical
traversal using `TNonSymmetricTreeStepNode.LeftSubtreeDiff`/`RightSubtreeDiff`. The relevant
section of the MLX codebase is the training-path partition accumulation at
`csv_train.cpp:3660–3683` and the DEC-029 comment block at `csv_train.cpp:3644–3654` which
describes the intended BFS-node-to-partition mapping. The fix for T3 should mirror
`ComputeLeafIndicesLossguide`: use a BFS-keyed `nodeSplitMap` and resolve the final node to a
dense leaf via an explicit inverse map that converts BFS node index to bit-packed partition,
rather than using `nodeIdx - numNodes`.

---

## 7. Repro Script

`docs/sprint27/scratch/fu1_repro.py` — self-contained Python harness.
Run with: `python docs/sprint27/scratch/fu1_repro.py` from the project root.
Requires `python` directory on `sys.path` (or `pip install -e python/`).
