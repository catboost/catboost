"""Shared tree unfolding logic for CoreML and ONNX export."""

from typing import Dict, List, Any


def unfold_oblivious_tree(tree: dict, features: List[dict], approx_dim: int = 1
                          ) -> List[dict]:
    """Convert an oblivious (symmetric) tree to standard binary tree nodes.

    An oblivious tree of depth D has D splits (one per level) and 2^D leaves.
    Each sample's leaf index is a D-bit integer where bit i = 1 means
    "went right at level i".

    Parameters
    ----------
    tree : dict
        Tree dict with keys: depth, splits, leaf_values.
    features : list of dict
        Feature metadata with borders and has_nan.
    approx_dim : int
        Number of output dimensions per leaf (1 for regression/binary, K for multiclass).

    Returns
    -------
    list of dict
        Nodes in BFS order. Each is either:
        - branch: {"type": "branch", "node_id": int, "feature_idx": int,
                    "threshold": float, "is_one_hot": bool,
                    "true_child": int, "false_child": int}
        - leaf: {"type": "leaf", "node_id": int, "values": list[float]}
    """
    depth = tree["depth"]
    splits = tree["splits"]
    leaf_values = tree["leaf_values"]

    if depth == 0:
        vals = leaf_values[:approx_dim]
        return [{"type": "leaf", "node_id": 0, "values": list(vals)}]

    # We build the tree top-down via BFS. At each internal node, the split
    # level determines which oblivious split to use.  The "path" to a node
    # is encoded as a partial leaf index (bits for levels already decided).

    nodes: List[dict] = []
    # Queue items: (node_id, level, partial_leaf_index)
    node_id_counter = [0]

    def _alloc_id() -> int:
        nid = node_id_counter[0]
        node_id_counter[0] += 1
        return nid

    # BFS
    queue = [(_alloc_id(), 0, 0)]  # root: id=0, level=0, partial=0
    while queue:
        nid, level, partial = queue.pop(0)
        if level == depth:
            # Leaf node
            start = partial * approx_dim
            vals = leaf_values[start:start + approx_dim]
            nodes.append({"type": "leaf", "node_id": nid, "values": list(vals)})
        else:
            split = splits[level]
            feat_idx = split["feature_idx"]
            bin_threshold = split["bin_threshold"]
            is_one_hot = split.get("is_one_hot", False)

            threshold = _bin_to_threshold(feat_idx, bin_threshold, is_one_hot, features)

            left_id = _alloc_id()   # go left (bit=0, condition NOT met)
            right_id = _alloc_id()  # go right (bit=1, condition met)

            nodes.append({
                "type": "branch",
                "node_id": nid,
                "feature_idx": feat_idx,
                "threshold": threshold,
                "is_one_hot": is_one_hot,
                "true_child": right_id,   # condition met → go right
                "false_child": left_id,   # condition not met → go left
            })

            # Left child: bit at this level = 0, partial unchanged
            queue.append((left_id, level + 1, partial))
            # Right child: bit at this level = 1
            queue.append((right_id, level + 1, partial | (1 << level)))

    # Sort by node_id for deterministic output
    nodes.sort(key=lambda n: n["node_id"])
    return nodes


def _bin_to_threshold(feat_idx: int, bin_threshold: int, is_one_hot: bool,
                      features: List[dict]) -> float:
    """Convert bin threshold index to real-valued threshold.

    For numeric features: border_idx = bin_threshold - nan_offset,
    threshold = features[feat_idx]["borders"][border_idx].

    For categorical (one-hot): threshold is the bin value directly
    (equality check).
    """
    if is_one_hot:
        return float(bin_threshold)

    feat = features[feat_idx]
    nan_offset = 1 if feat.get("has_nan", False) else 0
    border_idx = bin_threshold - nan_offset
    borders = feat.get("borders", [])

    if border_idx < 0 or border_idx >= len(borders):
        # Fallback: if out of range, return bin_threshold as float
        return float(bin_threshold)

    return float(borders[border_idx])
