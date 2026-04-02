"""Python-side tree evaluation for staged_predict and apply."""

import numpy as np
from typing import List, Optional


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

    for f in range(n_features):
        feat = features[f]
        if feat.get("is_categorical", False) or f in cat_set:
            # Categorical: look up cat_hash_map
            cat_map = feat.get("cat_hash_map", {})
            for d in range(n_samples):
                val = str(X[d, f])
                if val in cat_map:
                    binned[d, f] = cat_map[val]
                else:
                    binned[d, f] = 0  # unknown → bin 0
        else:
            # Numeric: upper_bound on borders + nan offset
            borders = feat.get("borders", [])
            has_nan = feat.get("has_nan", False)
            bin_offset = 1 if has_nan else 0

            col = X[:, f].astype(float)
            nan_mask = np.isnan(col)

            if len(borders) > 0:
                border_arr = np.array(borders, dtype=float)
                bins = np.searchsorted(border_arr, col, side="right") + bin_offset
            else:
                bins = np.full(n_samples, bin_offset, dtype=np.uint8)

            bins[nan_mask] = 0  # NaN → bin 0
            binned[:, f] = bins.astype(np.uint8)

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
                   approx_dim: int, n_trees: Optional[int] = None) -> np.ndarray:
    """Accumulate leaf values across trees.

    Parameters
    ----------
    binned_X : ndarray of shape (n_samples, n_features), dtype uint8
    trees : list of tree dicts
    approx_dim : int
        Output dimensions per leaf (1 for regression/binary, K-1 for K-class).
    n_trees : int, optional
        Only use the first n_trees. Defaults to all.

    Returns
    -------
    ndarray
        shape (n_samples,) if approx_dim == 1, else (n_samples, approx_dim)
    """
    n_samples = binned_X.shape[0]
    if n_trees is None:
        n_trees = len(trees)

    if approx_dim == 1:
        cursor = np.zeros(n_samples, dtype=float)
        for t in range(n_trees):
            tree = trees[t]
            leaf_idx = compute_leaf_indices(binned_X, tree)
            leaf_vals = np.array(tree["leaf_values"], dtype=float)
            cursor += leaf_vals[leaf_idx]
    else:
        cursor = np.zeros((n_samples, approx_dim), dtype=float)
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
        # cursor shape: (n_samples, K-1) where K = num_classes
        # Implicit last class: log-prob = 0 (i.e. exp=1)
        n_samples = cursor.shape[0]
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
