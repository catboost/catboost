"""
_predict_utils.py -- Pure-Python tree evaluation engine (no C++ binary needed).

What this file does:
    Normally, predictions call a compiled C++ program on the GPU. But sometimes
    you want predictions at every step of the training process (like watching
    a student improve answer by answer). This file re-implements the tree
    evaluation in Python/NumPy so you can accumulate predictions tree-by-tree
    without launching a subprocess each time.

How it fits into the project:
    Imported by core.py for staged_predict(), staged_predict_proba(), and
    apply(). This is a private module (underscore prefix) -- users never
    import it directly.

Key concepts:
    - Quantization: Converting raw feature values into bin numbers (like rounding
      temperatures to the nearest 5 degrees) so the tree can compare quickly.
    - Leaf index: Each tree assigns every data point to a "leaf" (the bottom
      node of the tree). The leaf index tells you which leaf it landed in.
    - Link function: The final math transformation on raw predictions (e.g.
      sigmoid converts log-odds to a 0-1 probability).
    - Oblivious tree bit-indexing: In a tree of depth D, each leaf is identified
      by a D-bit number. Bit i = 1 means "went right at level i."
"""

from typing import List, Optional

import numpy as np


def quantize_features(X: np.ndarray, features: List[dict],
                      cat_features: Optional[List[int]] = None) -> np.ndarray:
    """Convert raw feature values to bin indices, mirroring C++ quantization.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Raw feature values (numeric or string for categoricals).
    features : list of dict
        Feature metadata from model JSON (borders, has_nan, cat_hash_map, etc.)
    cat_features : list of int, optional
        Indices of categorical features.

    Returns
    -------
    ndarray of shape (n_samples, n_features), dtype uint8
        Binned feature values.
    """
    n_samples, n_features = X.shape
    binned = np.zeros((n_samples, n_features), dtype=np.uint8)
    cat_set = set(cat_features) if cat_features else set()

    # Quantize each feature column independently
    for f in range(n_features):
        feat = features[f]
        if feat.get("is_categorical", False) or f in cat_set:
            # Categorical: look up the hash map for each unique value
            cat_map = feat.get("cat_hash_map", {})
            for d in range(n_samples):
                val = str(X[d, f])
                if val in cat_map:
                    binned[d, f] = cat_map[val]
                else:
                    binned[d, f] = 0  # unknown category -> bin 0
        else:
            # Numeric: binary-search on quantization borders
            borders = feat.get("borders", [])
            has_nan = feat.get("has_nan", False)
            # NaN offset: if this feature has NaN values, bin 0 is reserved
            # for NaN, so all border-based bins shift up by 1
            bin_offset = 1 if has_nan else 0

            col = X[:, f].astype(float)
            nan_mask = np.isnan(col)

            if len(borders) > 0:
                border_arr = np.array(borders, dtype=float)
                # searchsorted with side='right' gives the count of borders
                # strictly less than the value, matching C++ quantization
                bins = np.searchsorted(border_arr, col, side="right") + bin_offset
            else:
                bins = np.full(n_samples, bin_offset, dtype=np.uint8)

            bins[nan_mask] = 0  # NaN → bin 0
            binned[:, f] = np.clip(bins, 0, 255).astype(np.uint8)

    return binned


def compute_leaf_indices(binned_X: np.ndarray, tree: dict) -> np.ndarray:
    """Compute leaf index for each sample in an oblivious tree.

    Parameters
    ----------
    binned_X : ndarray of shape (n_samples, n_features), dtype uint8
    tree : dict with keys 'depth', 'splits'

    Returns
    -------
    ndarray of shape (n_samples,), dtype uint32
    """
    n_samples = binned_X.shape[0]
    leaf_idx = np.zeros(n_samples, dtype=np.uint32)

    # Build the leaf index bit by bit. For an oblivious tree of depth D,
    # the leaf index is a D-bit integer. Bit i = 1 means "went right at level i."
    for level, split in enumerate(tree["splits"]):
        feat_idx = split["feature_idx"]
        bin_threshold = split["bin_threshold"]
        is_one_hot = split.get("is_one_hot", False)

        bval = binned_X[:, feat_idx].astype(np.uint32)
        if is_one_hot:
            go_right = (bval == bin_threshold)
        else:
            go_right = (bval > bin_threshold)

        leaf_idx |= go_right.astype(np.uint32) << level

    return leaf_idx


def evaluate_trees(binned_X: np.ndarray, trees: List[dict],
                   approx_dim: int, n_trees: Optional[int] = None,
                   base_prediction: Optional[List[float]] = None) -> np.ndarray:
    """Accumulate leaf values across trees.

    Parameters
    ----------
    binned_X : ndarray of shape (n_samples, n_features), dtype uint8
    trees : list of tree dicts
    approx_dim : int
        Output dimensions per leaf (1 for regression/binary, K-1 for K-class).
    n_trees : int, optional
        Only use the first n_trees. Defaults to all.
    base_prediction : list of float, optional
        Starting constant per dimension (from model_info). Added before tree sums.

    Returns
    -------
    ndarray
        shape (n_samples,) if approx_dim == 1, else (n_samples, approx_dim)
    """
    n_samples = binned_X.shape[0]
    if n_trees is None:
        n_trees = len(trees)

    if approx_dim == 1:
        bp = base_prediction[0] if base_prediction else 0.0
        cursor = np.full(n_samples, bp, dtype=float)
        for t in range(n_trees):
            tree = trees[t]
            leaf_idx = compute_leaf_indices(binned_X, tree)
            leaf_vals = np.array(tree["leaf_values"], dtype=float)
            cursor += leaf_vals[leaf_idx]
    else:
        cursor = np.zeros((n_samples, approx_dim), dtype=float)
        if base_prediction and len(base_prediction) >= approx_dim:
            cursor += np.array(base_prediction[:approx_dim], dtype=float)
        for t in range(n_trees):
            tree = trees[t]
            leaf_idx = compute_leaf_indices(binned_X, tree)
            leaf_vals = np.array(tree["leaf_values"], dtype=float).reshape(-1, approx_dim)
            cursor += leaf_vals[leaf_idx]

    return cursor


def apply_link(cursor: np.ndarray, loss_type: str,
               num_classes: int = 0) -> dict:
    """Apply the link function to raw cursor values.

    Parameters
    ----------
    cursor : ndarray
        Raw accumulated predictions.
    loss_type : str
        Loss type from model_info (rmse, logloss, multiclass, etc.)
    num_classes : int
        Number of classes (for multiclass).

    Returns
    -------
    dict with prediction arrays matching csv_predict output format.
    """
    if loss_type == "logloss":
        prob = 1.0 / (1.0 + np.exp(-cursor))
        predicted_class = (prob > 0.5).astype(int)
        return {"prediction": cursor, "probability": prob,
                "predicted_class": predicted_class}
    elif loss_type == "multiclass":
        # Multiclass softmax with an implicit last class.
        # The model outputs K-1 raw values; the K-th class has implicit value 0.
        # We use the max-subtraction trick (max_c) for numerical stability:
        # subtracting the max from exponents prevents overflow.
        max_c = np.maximum(cursor.max(axis=1), 0.0)
        exp_c = np.exp(cursor - max_c[:, None])
        exp_implicit = np.exp(-max_c)
        sum_exp = exp_c.sum(axis=1) + exp_implicit
        probs = exp_c / sum_exp[:, None]
        prob_last = exp_implicit / sum_exp
        all_probs = np.column_stack([probs, prob_last[:, None]])
        predicted_class = all_probs.argmax(axis=1)
        result = {"predicted_class": predicted_class}
        for k in range(all_probs.shape[1]):
            result[f"prob_class_{k}"] = all_probs[:, k]
        return result
    elif loss_type in ("poisson", "tweedie"):
        pred = np.exp(cursor)
        return {"prediction": pred}
    else:
        return {"prediction": cursor}
