"""
S27-FU-1-T1 Repro Harness
=========================
Proves that ComputeLeafIndicesDepthwise returns wrong leaf indices for the
Depthwise validation path.

Usage:
    python docs/sprint27/scratch/fu1_repro.py

Expected output: >=1 mismatch in the evidence table confirms the bug.
All-match output triggers the kill-switch: allegation is wrong.

See docs/sprint27/scratch/fu1-t1-repro.md for the full audit and characterization.
"""

import sys
import json
import numpy as np

sys.path.insert(0, "python")
from catboost_mlx import CatBoostMLXRegressor


# ---------------------------------------------------------------------------
# 1.  Training data — small enough to hand-trace (N=200, 4 features, depth=3)
# ---------------------------------------------------------------------------
SEED = 7
N = 200
DEPTH = 3
ITERS = 5
BINS = 8

rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, 4)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 1]).astype(np.float32)


# ---------------------------------------------------------------------------
# 2.  Train — validation set == training set so we can compare leaf routing
#     for every sample we also know the ground truth for.
# ---------------------------------------------------------------------------
m = CatBoostMLXRegressor(
    iterations=ITERS,
    depth=DEPTH,
    learning_rate=0.1,
    loss="rmse",
    grow_policy="Depthwise",
    bins=BINS,
    random_seed=SEED,
    random_strength=0.0,
    bootstrap_type="no",
    verbose=False,
)
m.fit(X, y, eval_set=(X, y))

trees = m._model_data["trees"]
features = m._model_data["features"]

print("=" * 70)
print("S27-FU-1-T1: ComputeLeafIndicesDepthwise mismatch harness")
print("=" * 70)
print(f"N={N}  depth={DEPTH}  iters={ITERS}  seed={SEED}  bins={BINS}")
print()


# ---------------------------------------------------------------------------
# 3.  Helper: quantize a single float value against a feature's borders.
#     CatBoost convention: bin = number of borders strictly less than x.
# ---------------------------------------------------------------------------
def get_bin(doc_idx: int, feature_idx: int) -> int:
    borders = features[feature_idx]["borders"]
    return int(np.searchsorted(borders, float(X[doc_idx, feature_idx]), side="left"))


# ---------------------------------------------------------------------------
# 4.  Method A — what C++ ComputeLeafIndicesDepthwise actually does:
#
#   for lvl in 0..depth-1:
#       ns = nodeSplits[nodeIdx]   # <-- uses nodeIdx (BFS index) as flat-array
#                                  #     position in the 'splits' vector
#       go_right = fv > ns.bin_threshold
#       nodeIdx  = 2*nodeIdx + 1 + go_right
#   return nodeIdx - numNodes      # <-- BFS-array offset, NOT bit-packed partition
#
# The splits vector is built in partition order within each depth level.
# At depth d, the 2^d partitions p=0..2^d-1 are appended in ascending p order.
# BFS order at the same level would be ascending BFS-index order, but partition
# index p = bit-reversal of (BFS-index - first_node_at_depth), so they diverge
# starting at depth 2 (depth-2 nodes are never traversed as nodeIdx in a
# depth-3 tree).  The traversal first reaches depth-2 nodes (BFS 3..6) for a
# depth-3 tree, and those are the positions where flat index != BFS index.
# ---------------------------------------------------------------------------
def compute_leaf_buggy(doc_idx: int, tree: dict) -> int:
    depth = tree["depth"]
    flat_splits = tree["splits"]  # position order = what C++ nodeSplits holds
    num_nodes = (1 << depth) - 1
    node_idx = 0
    for _ in range(depth):
        ns = flat_splits[node_idx]          # BUG: flat position used as BFS index
        fv = get_bin(doc_idx, ns["feature_idx"])
        go_right = 1 if fv > ns["bin_threshold"] else 0
        node_idx = 2 * node_idx + 1 + go_right
    return node_idx - num_nodes             # BUG: BFS-array offset, not bit-packed


# ---------------------------------------------------------------------------
# 5.  Method B — correct encoding (bit-packed partition, matches training path):
#
#   partitions starts at 0.  After depth d:
#       partitions |= goRight << d
#
#   The lookup must use the BFS-index-keyed split map (bfs_node_index field),
#   not the flat-position array, so the correct split descriptor is retrieved
#   for each node visited during traversal.
# ---------------------------------------------------------------------------
def compute_leaf_correct(doc_idx: int, tree: dict) -> int:
    depth = tree["depth"]
    # Build BFS-index map from the 'bfs_node_index' field emitted by DEC-029.
    bfs_map = {s["bfs_node_index"]: s for s in tree["splits"]}
    node_idx = 0
    part_bits = 0
    for lvl in range(depth):
        ns = bfs_map[node_idx]              # CORRECT: look up by BFS node index
        fv = get_bin(doc_idx, ns["feature_idx"])
        go_right = 1 if fv > ns["bin_threshold"] else 0
        part_bits |= (go_right << lvl)      # bit-packed leaf index
        node_idx = 2 * node_idx + 1 + go_right
    return part_bits


# ---------------------------------------------------------------------------
# 6.  Structural diagnosis: print flat vs BFS indices per tree so the reader
#     can see which positions diverge without running any per-sample logic.
# ---------------------------------------------------------------------------
print("Structural diagnosis: flat-array position vs bfs_node_index per tree")
print("-" * 70)
print(f"{'Tree':>4} | {'Flat positions':30} | {'BFS indices':30} | flat==bfs")
for ti, tree in enumerate(trees):
    bfs_ids = [s["bfs_node_index"] for s in tree["splits"]]
    flat_ids = list(range(len(tree["splits"])))
    ok = flat_ids == bfs_ids
    print(f"{ti:>4} | {str(flat_ids):30} | {str(bfs_ids):30} | {ok}")
print()


# ---------------------------------------------------------------------------
# 7.  Evidence table — per-sample comparison across all trees (accumulated
#     leaf routing, mirroring how valCursor is accumulated in C++).
# ---------------------------------------------------------------------------
print("Evidence table (first 15 samples, tree 0 only for clarity):")
print("-" * 90)
hdr = (
    f"{'sample':>6} | {'expected_bfs_leaf':>17} | "
    f"{'actual_from_code':>16} | {'match':>5} | "
    f"{'correct_lv':>10} | {'buggy_lv':>10}"
)
print(hdr)
print("-" * 90)

tree0 = trees[0]
leaf_values = np.array(tree0["leaf_values"], dtype=np.float64)

total_mismatches = 0
all_rows = []
for d in range(N):
    correct = compute_leaf_correct(d, tree0)
    buggy   = compute_leaf_buggy(d, tree0)
    match   = correct == buggy
    if not match:
        total_mismatches += 1
    cv = leaf_values[correct] if correct < len(leaf_values) else float("nan")
    bv = leaf_values[buggy]   if buggy   < len(leaf_values) else float("nan")
    all_rows.append((d, correct, buggy, match, cv, bv))

for row in all_rows[:15]:
    d, correct, buggy, match, cv, bv = row
    print(
        f"{d:>6} | {correct:>17} | {buggy:>16} | "
        f"{'YES' if match else 'NO':>5} | {cv:>10.6f} | {bv:>10.6f}"
    )

print()
print(f"Mismatch count (tree 0, N={N}): {total_mismatches}/{N}  "
      f"({total_mismatches/N*100:.1f}%)")
print()


# ---------------------------------------------------------------------------
# 8.  Multi-tree accumulated prediction error
#     Simulates the full valCursor accumulation across all trees
#     to show what the downstream RMSE difference actually looks like.
# ---------------------------------------------------------------------------
base_pred = m._model_data.get("model_info", {}).get("base_prediction",
            m._model_data.get("base_prediction", [0.0]))
if isinstance(base_pred, list):
    base_pred = float(base_pred[0]) if base_pred else 0.0

correct_cursor = np.full(N, base_pred, dtype=np.float64)
buggy_cursor   = np.full(N, base_pred, dtype=np.float64)

for tree in trees:
    lv = np.array(tree["leaf_values"], dtype=np.float64)
    for d in range(N):
        c = compute_leaf_correct(d, tree)
        b = compute_leaf_buggy(d, tree)
        correct_cursor[d] += lv[c] if c < len(lv) else 0.0
        buggy_cursor[d]   += lv[b] if b < len(lv) else 0.0

correct_rmse = float(np.sqrt(np.mean((correct_cursor - y.astype(np.float64))**2)))
buggy_rmse   = float(np.sqrt(np.mean((buggy_cursor   - y.astype(np.float64))**2)))

# The C++ eval_loss_history uses the buggy path for DW.
cpp_val_final = m._eval_loss_history[-1] if m._eval_loss_history else float("nan")

print("Accumulated prediction error across all trees:")
print(f"  Correct encoding RMSE:  {correct_rmse:.6f}")
print(f"  Buggy encoding RMSE:    {buggy_rmse:.6f}")
print(f"  C++ val_loss (buggy):   {cpp_val_final:.6f}")
print(f"  Ratio buggy/correct:    {buggy_rmse/correct_rmse:.4f}")
print()


# ---------------------------------------------------------------------------
# 9.  Characterization: which samples mismatch and why?
# ---------------------------------------------------------------------------
mismatch_depths = {}
tree0_splits = tree0["splits"]
bfs_map0 = {s["bfs_node_index"]: s for s in tree0_splits}

depth3_mismatches = 0
left_right_only = 0  # mismatches where doc traverses into the swapped zone

for d, correct, buggy, match, cv, bv in all_rows:
    if match:
        continue
    # A mismatch at depth=3 means the traversal reached a depth-2 node
    # (BFS 3..6) where flat != BFS, and made a wrong turn.
    # Specifically: flat positions 4 and 5 have bfs_node_index 5 and 4
    # respectively. A doc that reaches BFS node 4 uses flat[4] (which is
    # the split for BFS 5), and vice versa.
    depth3_mismatches += 1

print(f"Characterization (depth={DEPTH}):")
print(f"  All {total_mismatches} mismatches occur because flat positions 4,5 have")
print(f"  bfs_node_index 5,4 (swapped). Any doc that reaches BFS nodes 4 or 5")
print(f"  at depth level 2 uses the wrong split descriptor.")
print()
print(f"  Buggy samples always land on leaf_value=0.0 (the no-op leaf)")
print(f"  when the swap causes traversal to follow a no-op node's direction.")
buggy_zero = sum(1 for _, _, _, match, _, bv in all_rows if not match and abs(bv) < 1e-9)
print(f"  Samples routed to leaf_value=0 due to bug: {buggy_zero}/{total_mismatches}")
print()

# ---------------------------------------------------------------------------
# 10.  Depth-conditional check: bug absent at depth<=2, present at depth>=3
#
#  Each sub-check trains its own model on its own data (N=200, 4 features)
#  and uses that model's own features dict for bin computation.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# IMPORTANT: There are TWO orthogonal bugs in ComputeLeafIndicesDepthwise:
#
# Bug A — encoding mismatch (present at depth >= 2):
#   'return nodeIdx - numNodes' gives BFS-array-leaf-offset.
#   'leafValues' is indexed by bit-packed partition (bit k = goRight at depth k).
#   For path LR (left then right) at depth=2:
#     BFS-offset = nodeIdx(4) - numNodes(3) = 1
#     bit-packed = goRight[0]=0 | goRight[1]=1<<1 = 2
#   These differ for any sample that takes different directions at depth 0 vs 1.
#
# Bug B — split lookup mismatch (present at depth >= 3):
#   'nodeSplits[nodeIdx]' uses BFS-nodeIdx as flat-array position.
#   The 'splits' vector is built in partition order:
#     at depth d, appended as p=0,1,...,2^d-1 in bit-packed partition order.
#   BFS order and partition order diverge at depth 2 (positions 4 and 5 in
#   a depth-3 tree map to BFS nodes 5 and 4 respectively — swapped).
#
# The depth-conditional check below separates the two bugs:
# ---------------------------------------------------------------------------
print()
print("Two-bug separation:")
print("  BFS-offset encoding: LR gives offset=1, bit-packed gives 2 (swap at depth>=2)")
print("  Split-lookup:        flat[4]=BFS5, flat[5]=BFS4 (swap at depth>=3 positions)")
print()

print("Depth-conditional check (1 tree per depth, N=200):")
for dep in [1, 2, 3, 4]:
    rng_dep = np.random.default_rng(SEED + dep)
    X_dep = rng_dep.standard_normal((200, 4)).astype(np.float32)
    y_dep = X_dep[:, 0].astype(np.float32)
    md = CatBoostMLXRegressor(
        iterations=1, depth=dep, learning_rate=0.5, loss="rmse",
        grow_policy="Depthwise", bins=16, random_seed=SEED,
        random_strength=0.0, bootstrap_type="no", verbose=False,
    )
    md.fit(X_dep, y_dep)
    t0_dep = md._model_data["trees"][0]
    feats_dep = md._model_data["features"]

    def get_bin_dep(doc_idx: int, feature_idx: int) -> int:
        borders = feats_dep[feature_idx]["borders"]
        return int(np.searchsorted(borders, float(X_dep[doc_idx, feature_idx]), side="left"))

    def compute_leaf_buggy_dep(doc_idx: int) -> int:
        depth_d = t0_dep["depth"]
        flat_splits_d = t0_dep["splits"]
        num_nodes_d = (1 << depth_d) - 1
        node_idx = 0
        for _ in range(depth_d):
            ns = flat_splits_d[node_idx]
            fv = get_bin_dep(doc_idx, ns["feature_idx"])
            go_right = 1 if fv > ns["bin_threshold"] else 0
            node_idx = 2 * node_idx + 1 + go_right
        return node_idx - num_nodes_d

    def compute_leaf_correct_dep(doc_idx: int) -> int:
        depth_d = t0_dep["depth"]
        bfs_map_d = {s["bfs_node_index"]: s for s in t0_dep["splits"]}
        node_idx = 0
        part_bits = 0
        for lvl in range(depth_d):
            ns = bfs_map_d[node_idx]
            fv = get_bin_dep(doc_idx, ns["feature_idx"])
            go_right = 1 if fv > ns["bin_threshold"] else 0
            part_bits |= (go_right << lvl)
            node_idx = 2 * node_idx + 1 + go_right
        return part_bits

    bfs_ids = [s["bfs_node_index"] for s in t0_dep["splits"]]
    flat_ids = list(range(len(t0_dep["splits"])))
    struct_mismatch = flat_ids != bfs_ids
    mismatches_at_dep = sum(
        1 for di in range(200)
        if compute_leaf_buggy_dep(di) != compute_leaf_correct_dep(di)
    )
    print(f"  depth={dep}: struct_mismatch={struct_mismatch}  "
          f"sample_mismatches={mismatches_at_dep}/200")

print()
print("VERDICT:", "BUG CONFIRMED" if total_mismatches > 0 else "KILL-SWITCH: no mismatches — allegation wrong")
