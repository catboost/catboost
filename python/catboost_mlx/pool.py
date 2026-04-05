"""
pool.py -- Unified data container that bundles training data with metadata.

What this file does:
    When you train a model, you need your data plus extra info -- like which
    columns are categories (e.g. "red", "blue") vs numbers, or which rows
    belong to the same search query. Pool is like a labeled box: you put
    everything in once, and the model knows exactly what each piece is.
    It also automatically figures out column names and category columns
    if you pass a pandas DataFrame.

How it fits into the project:
    Imported by core.py (which unpacks Pool objects inside fit()) and by
    __init__.py (re-exported to users). Has no dependencies on other
    catboost_mlx modules.

Key concepts:
    - Categorical features: Columns with text labels (like colors or countries)
      instead of numbers. These are treated differently during training.
    - Feature names: Human-readable labels for each column (e.g. "age", "income").
    - Sample weights: Numbers that tell the model some rows matter more than others.
    - DataFrame auto-detection: If you pass a pandas DataFrame, Pool reads its
      column names and dtype info automatically.
"""

from typing import List, Optional, Union

import numpy as np

from ._utils import _to_numpy


def _is_dataframe(X) -> bool:
    """Check if X is a pandas DataFrame without importing pandas.

    Uses duck-typing (check the class name + columns attribute) to avoid
    the cost of importing pandas when it's not needed.
    """
    return type(X).__name__ == "DataFrame" and hasattr(X, "columns")


def _resolve_cat_features(cat_features, feature_names):
    """Resolve string cat_features to integer indices using feature_names."""
    if cat_features is None:
        return None
    resolved = []
    for cf in cat_features:
        if isinstance(cf, str):
            if feature_names is None:
                raise ValueError(
                    f"Cannot resolve categorical feature name '{cf}' "
                    "without feature_names."
                )
            try:
                idx = feature_names.index(cf)
            except ValueError:
                raise ValueError(
                    f"Categorical feature '{cf}' not found in feature_names: "
                    f"{feature_names}"
                )
            resolved.append(idx)
        else:
            resolved.append(int(cf))
    if len(resolved) != len(set(resolved)):
        dupes = sorted({i for i in resolved if resolved.count(i) > 1})
        raise ValueError(f"Duplicate cat_features indices: {dupes}")
    return resolved


class Pool:
    """Unified data container for CatBoost-MLX training and prediction.

    Bundles features, labels, categorical feature indices, feature names,
    group IDs, and sample weights into a single object.

    When X is a pandas DataFrame:
    - Feature names are auto-extracted from column names
    - Categorical features can be specified by column name (strings)
    - Object/category dtype columns are auto-detected as categorical

    Parameters
    ----------
    X : array-like or pandas DataFrame
        Feature matrix of shape (n_samples, n_features).
    y : array-like, optional
        Target values of shape (n_samples,).
    cat_features : list of int or str, optional
        Indices or names of categorical features. If None and X is a
        DataFrame, object/category columns are auto-detected.
    feature_names : list of str, optional
        Names for each feature. Auto-extracted from DataFrame columns if
        not provided.
    group_id : array-like, optional
        Group/query IDs for ranking tasks.
    sample_weight : array-like, optional
        Per-sample weights.

    Notes
    -----
    Data is stored as C-contiguous numpy arrays. If the input is already a
    C-contiguous ndarray, no copy is made (zero-copy). Non-contiguous or
    non-numpy inputs are converted automatically.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"color": ["red", "blue", "red"], "size": [1, 2, 3]})
    >>> pool = Pool(df, y=[0, 1, 0])
    >>> pool.feature_names
    ['color', 'size']
    >>> pool.cat_features
    [0]
    """

    def __init__(
        self,
        X,
        y=None,
        cat_features: Optional[List[Union[int, str]]] = None,
        feature_names: Optional[List[str]] = None,
        group_id=None,
        sample_weight=None,
    ):
        # When X is a DataFrame, extract column names and detect categoricals
        # BEFORE converting to numpy, because numpy loses dtype information.
        if _is_dataframe(X):
            if feature_names is None:
                feature_names = list(X.columns)
            if cat_features is None:
                # Auto-detect object/category/string dtype columns
                cat_cols = X.select_dtypes(
                    include=["object", "category", "string"]
                ).columns
                if len(cat_cols) > 0:
                    cat_features = sorted(
                        feature_names.index(c) for c in cat_cols
                    )

        # Convert to numpy (ascontiguousarray is a no-op for C-contiguous ndarrays)
        self.X = np.ascontiguousarray(_to_numpy(X))
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)

        self.y = np.ascontiguousarray(_to_numpy(y)) if y is not None else None

        # Resolve string cat_features to indices
        self.cat_features = _resolve_cat_features(cat_features, feature_names)

        # Validate cat_features bounds
        if self.cat_features:
            for idx in self.cat_features:
                if idx < 0 or idx >= self.X.shape[1]:
                    raise ValueError(
                        f"cat_features index {idx} is out of bounds "
                        f"for data with {self.X.shape[1]} features"
                    )

        self.feature_names = feature_names
        self.group_id = (
            np.asarray(group_id) if group_id is not None else None
        )
        self.sample_weight = (
            np.asarray(sample_weight, dtype=float)
            if sample_weight is not None
            else None
        )

        # Validate shapes
        if self.y is not None and self.y.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"X has {self.X.shape[0]} samples but y has {self.y.shape[0]}"
            )
        if self.group_id is not None and len(self.group_id) != self.X.shape[0]:
            raise ValueError(
                f"group_id length ({len(self.group_id)}) != "
                f"number of samples ({self.X.shape[0]})"
            )
        if self.sample_weight is not None and len(self.sample_weight) != self.X.shape[0]:
            raise ValueError(
                f"sample_weight length ({len(self.sample_weight)}) != "
                f"number of samples ({self.X.shape[0]})"
            )
        if self.feature_names is not None and len(self.feature_names) != self.X.shape[1]:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries but "
                f"X has {self.X.shape[1]} features"
            )
        if self.feature_names is not None and len(self.feature_names) != len(set(self.feature_names)):
            dupes = sorted({n for n in self.feature_names if self.feature_names.count(n) > 1})
            raise ValueError(f"Duplicate feature names: {dupes}")

    @property
    def num_samples(self) -> int:
        return self.X.shape[0]

    @property
    def num_features(self) -> int:
        return self.X.shape[1]

    @property
    def shape(self):
        return self.X.shape

    def __repr__(self) -> str:
        n_cat = len(self.cat_features) if self.cat_features else 0
        has_y = "with labels" if self.y is not None else "no labels"
        return (
            f"Pool({self.num_samples} samples, {self.num_features} features, "
            f"{n_cat} categorical, {has_y})"
        )

    def __len__(self) -> int:
        return self.num_samples
