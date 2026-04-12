"""
core.py -- The heart of CatBoost-MLX: model classes that train, predict, and export.

What this file does:
    Imagine you want to teach a computer to predict things (like house prices
    or whether an email is spam). This file has the "brain" classes that do that.
    They don't do the heavy math themselves -- instead, they write your data to
    CSV files, call fast C++ programs (csv_train and csv_predict) that run on
    your Mac's GPU, then read back the results. Think of it like a translator
    between Python and the GPU engine.

How it fits into the project:
    This is the main module. Imported by __init__.py and is the primary module
    users interact with. It imports:
    - pool.py (data container)
    - _predict_utils.py (Python-side tree evaluation for staged predictions)
    - _tree_utils.py (tree structure conversion for get_trees)
    - export_coreml.py and export_onnx.py (model export)

Key concepts:
    - Gradient Boosted Decision Trees (GBDT): Build many small decision trees,
      each one fixing the mistakes of the previous ones, until you have a
      strong predictor.
    - Oblivious trees: A special kind of decision tree where every node at the
      same level uses the same split rule. Fast on GPUs.
    - Loss function: Measures how wrong predictions are. Different tasks use
      different loss functions (RMSE for regression, Logloss for classification).
    - sklearn compatibility: Mimics scikit-learn's interface (fit/predict/score,
      get_params/set_params) so it plugs into sklearn pipelines and cross-val.
    - PTY (pseudo-terminal): A trick to capture real-time output from the C++
      binary, which otherwise buffers everything when piped to a subprocess.

Public API:
    - CatBoostMLX: Base class with all functionality (27 hyperparameters).
    - CatBoostMLXRegressor: Regression specialization (default loss='rmse').
    - CatBoostMLXClassifier: Classification specialization (default loss='auto').
"""

import csv
import json
import logging
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ._utils import _to_numpy

logger = logging.getLogger(__name__)

# ── sklearn optional dependency ──────────────────────────────────────────────
# When sklearn IS installed, we use its real BaseEstimator/RegressorMixin/
# ClassifierMixin. When it's NOT installed, we provide minimal fallbacks below
# so the core classes still work without sklearn as a dependency.
try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

    class BaseEstimator:
        """Minimal fallback when sklearn is not installed."""

        @classmethod
        def _get_param_names(cls):
            """Get parameter names from __init__ signature.

            Walks the MRO (method resolution order) to find the first __init__
            that doesn't use **kwargs, so we get the real parameter list from
            the base class rather than a subclass that just passes through.
            """
            import inspect
            for klass in cls.__mro__:
                sig = inspect.signature(klass.__init__)
                if "kwargs" not in sig.parameters and klass is not object:
                    break
            else:
                sig = inspect.signature(cls.__init__)
            return sorted([
                p.name for p in sig.parameters.values()
                if p.name != "self"
            ])

        def get_params(self, deep=True):
            params = {}
            for name in self._get_param_names():
                params[name] = getattr(self, name, None)
            return params

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    class RegressorMixin:
        """Minimal fallback -- provides score() returning R-squared (R²).

        R² = 1 means perfect predictions; R² = 0 means no better than
        predicting the mean; R² < 0 means worse than the mean.
        """
        _estimator_type = "regressor"

        def score(self, X, y, sample_weight=None):
            """Return R² score on test data."""
            y_pred = self.predict(X)
            y_true = np.asarray(y, dtype=float)
            ss_res = np.sum((y_true - y_pred) ** 2)  # residual sum of squares
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # total sum of squares
            if ss_tot == 0:
                return 0.0
            return float(1.0 - ss_res / ss_tot)

    class ClassifierMixin:
        """Minimal fallback -- provides score() returning accuracy.

        Accuracy = fraction of predictions that match the true labels.
        Supports optional sample_weight for weighted accuracy.
        """
        _estimator_type = "classifier"

        def score(self, X, y, sample_weight=None):
            """Return accuracy score on test data."""
            y_pred = self.predict(X)
            y_true = np.asarray(y)
            if sample_weight is not None:
                w = np.asarray(sample_weight, dtype=float)
                return float(np.sum((y_pred == y_true) * w) / np.sum(w))
            return float(np.mean(y_pred == y_true))


def _normalize_loss_str(loss: str) -> str:
    """Normalize a loss string for the csv_train binary.

    Handles:
    - Case: 'MAE' -> 'mae', 'Quantile:alpha=0.7' -> 'quantile:0.7'
    - Named params: 'quantile:alpha=0.7' -> 'quantile:0.7',
                    'huber:delta=1.0'     -> 'huber:1.0'
    - Preserves positional: 'quantile:0.7' -> 'quantile:0.7'
    """
    if ":" in loss:
        base, suffix = loss.split(":", 1)
        base = base.lower()
        for prefix in ("alpha=", "delta=", "variance_power="):
            if suffix.lower().startswith(prefix):
                suffix = suffix[len(prefix):]
                break
        return f"{base}:{suffix}"
    return loss.lower()


def _find_binary(name: str, binary_path: Optional[str] = None) -> str:
    """Locate a compiled binary (csv_train or csv_predict).

    Search order (first match wins):
      1. User-specified binary_path (explicit override)
      2. System PATH (e.g. /usr/local/bin)
      3. Bundled in package: catboost_mlx/bin/ (built by build_binaries.py)
      4. Current working directory
      5. Package directory and parent directories
    """
    def _check_executable(candidate: Path) -> str:
        if candidate.is_file():
            if not os.access(str(candidate), os.X_OK):
                raise PermissionError(
                    f"Binary '{candidate}' exists but is not executable. "
                    f"Try: chmod +x {candidate}"
                )
            return str(candidate)
        return ""

    # Priority 1: Explicit path from user
    if binary_path:
        p = Path(binary_path)
        if p.is_dir():
            candidate = p / name
        else:
            candidate = p
        result = _check_executable(candidate)
        if result:
            return result
        raise FileNotFoundError(f"Binary not found at {candidate}")

    # Priority 2: Check system PATH
    found = shutil.which(name)
    if found:
        return found

    # Priority 3: Check bundled binaries in package (catboost_mlx/bin/)
    bin_dir = Path(__file__).parent / "bin"
    result = _check_executable(bin_dir / name)
    if result:
        return result

    # Priority 4: Check current working directory
    result = _check_executable(Path(name).resolve())
    if result:
        return result

    # Priority 5: Check package directory and ancestors (for dev installs)
    pkg_dir = Path(__file__).parent
    for search_dir in [pkg_dir, pkg_dir.parent, pkg_dir.parent.parent]:
        result = _check_executable(search_dir / name)
        if result:
            return result

    raise FileNotFoundError(
        f"Cannot find '{name}' binary. Either:\n"
        "  1. Add it to your PATH\n"
        "  2. Place it in the current directory\n"
        f"  3. Pass binary_path='<directory containing {name}>'"
    )


def _format_numeric_col(col: np.ndarray) -> List[str]:
    """Format a numeric column to strings with 10 significant digits, empty for NaN."""
    mask = np.isnan(col)
    out = [f"{v:.10g}" for v in col]
    for i in np.flatnonzero(mask):
        out[i] = ""
    return out


def _format_cat_col(col: np.ndarray) -> List[str]:
    """Format a categorical column to strings, empty for NaN."""
    out = []
    for val in col:
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            out.append("")
        else:
            out.append(str(val))
    return out


def _array_to_csv(path: str, X: np.ndarray, y: Optional[np.ndarray] = None,
                   feature_names: Optional[List[str]] = None,
                   cat_features: Optional[List[int]] = None,
                   group_id: Optional[np.ndarray] = None,
                   sample_weight: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    """Write numpy arrays to CSV for the C++ binaries.

    The C++ binaries (csv_train / csv_predict) read CSV files as input.
    This function converts numpy arrays into that CSV format.

    Column layout (when training): [group_id?] [weight?] features... target
    Column layout (when predicting): features...

    Returns (target_col_index, group_col_index, weight_col_index).
    Index is -1 if the column is absent.
    """
    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    else:
        for fname in feature_names:
            if "," in fname or "\n" in fname or "\x00" in fname:
                raise ValueError(
                    f"Feature name {fname!r} contains invalid characters "
                    "(comma, newline, or null byte) that would corrupt the CSV."
                )

    cat_set = frozenset(cat_features) if cat_features else frozenset()
    group_col_idx = -1
    weight_col_idx = -1

    # Pre-format all feature columns (vectorized per-column instead of per-cell)
    formatted_cols: List[List[str]] = []
    for j in range(n_features):
        col = X[:, j]
        if j in cat_set:
            formatted_cols.append(_format_cat_col(col))
        else:
            formatted_cols.append(_format_numeric_col(col.astype(float)))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if y is not None:
            header = []
            col_offset = 0
            if group_id is not None:
                header.append("group_id")
                group_col_idx = col_offset
                col_offset += 1
            if sample_weight is not None:
                header.append("weight")
                weight_col_idx = col_offset
                col_offset += 1
            header.extend(feature_names)
            header.append("target")
            writer.writerow(header)
            target_col_idx = len(header) - 1

            # Pre-format prefix and target columns
            gid_strs = [str(g) for g in group_id] if group_id is not None else None
            sw_strs = [f"{w:.10g}" for w in sample_weight] if sample_weight is not None else None
            y_strs = [f"{v:.10g}" for v in y]

            for i in range(n_samples):
                row = []
                if gid_strs is not None:
                    row.append(gid_strs[i])
                if sw_strs is not None:
                    row.append(sw_strs[i])
                for j in range(n_features):
                    row.append(formatted_cols[j][i])
                row.append(y_strs[i])
                writer.writerow(row)
            return target_col_idx, group_col_idx, weight_col_idx
        else:
            writer.writerow(feature_names)
            for i in range(n_samples):
                row = []
                for j in range(n_features):
                    row.append(formatted_cols[j][i])
                writer.writerow(row)
            return -1, -1, -1


def _array_to_binary(path: str, X: np.ndarray, y: Optional[np.ndarray] = None,
                     group_id: Optional[np.ndarray] = None,
                     sample_weight: Optional[np.ndarray] = None) -> None:
    """Write arrays in CBMX binary format for fast C++ loading.

    Column-major layout: each feature column is contiguous in the file.
    ~1000x faster than CSV for large numeric datasets.
    """
    n_samples, n_features = X.shape
    flags = 0
    if y is not None:
        flags |= 1
    if group_id is not None:
        flags |= 2
    if sample_weight is not None:
        flags |= 4

    with open(path, "wb") as fp:
        fp.write(b"CBMX")
        fp.write(struct.pack("<IIII", 1, n_samples, n_features, flags))
        # Features column-major: transpose then write in C order
        np.ascontiguousarray(X.T, dtype=np.float32).tofile(fp)
        if y is not None:
            np.asarray(y, dtype=np.float32).tofile(fp)
        if sample_weight is not None:
            np.asarray(sample_weight, dtype=np.float32).tofile(fp)
        if group_id is not None:
            np.asarray(group_id, dtype=np.uint32).tofile(fp)


class CatBoostMLX(BaseEstimator):
    """CatBoost-style gradient boosted decision trees using Apple Silicon Metal GPU.

    Wraps the compiled csv_train/csv_predict binaries via subprocess.

    Parameters
    ----------
    iterations : int
        Number of boosting iterations (trees).
    depth : int
        Maximum tree depth.
    learning_rate : float
        Step size shrinkage.
    l2_reg_lambda : float
        L2 regularization coefficient.
    loss : str
        Loss function. One of: auto, rmse, mae, quantile, quantile:<alpha>,
        huber, huber:<delta>, poisson, tweedie, tweedie:<p>, mape, logloss, multiclass.
    bins : int
        Maximum number of quantization bins per feature.
    cat_features : list of int, optional
        Indices of categorical feature columns.
    eval_fraction : float
        Fraction of data reserved for validation (0 = no validation split).
    early_stopping_rounds : int
        Stop after this many iterations without validation improvement.
    subsample : float
        Row subsampling fraction per iteration.
    colsample_bytree : float
        Feature subsampling fraction per tree.
    random_seed : int
        Random seed.
    nan_mode : str
        NaN handling mode: "min" or "forbidden".
    ctr : bool
        Enable CTR target encoding for high-cardinality categoricals.
    ctr_prior : float
        CTR Bayesian prior.
    max_onehot_size : int
        Max categories for OneHot; above uses CTR.
    min_data_in_leaf : int
        Minimum number of documents per leaf (default: 1 = no restriction).
    monotone_constraints : list of int, optional
        Per-feature monotone constraints: 0=none, 1=increasing, -1=decreasing.
    grow_policy : str
        Tree grow policy. "SymmetricTree" (default) grows oblivious trees where
        all leaves at the same depth share one split rule. "Depthwise" grows
        non-symmetric trees where each leaf at a given depth gets its own best
        split (XGBoost-style). "Lossguide" grows best-first leaf-wise trees
        (LightGBM-style): at each step the leaf with the highest gain is split,
        producing unbalanced trees controlled by ``max_leaves`` instead of
        ``depth``.
    max_leaves : int
        Maximum number of terminal leaves for the "Lossguide" grow policy
        (default: 31). Ignored for other grow policies. Must be >= 2.
    mlflow_logging : bool
        If True, log hyperparameters, per-iteration loss, and final metrics to
        MLflow after training. Requires ``mlflow`` to be installed
        (``pip install mlflow``). Uses the active run if one is already open,
        otherwise starts and ends a new run automatically.
    mlflow_run_name : str, optional
        Custom run name for the MLflow run. Only used when ``mlflow_logging``
        starts a new run (ignored if a run is already active).
    verbose : bool
        Print per-iteration training loss.
    binary_path : str, optional
        Path to directory containing csv_train/csv_predict, or path to csv_train directly.
    train_timeout : float or None
        Maximum seconds for a training subprocess (default: None = no limit).
    predict_timeout : float or None
        Maximum seconds for a prediction subprocess (default: None = no limit).
    """

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        l2_reg_lambda: float = 3.0,
        loss: str = "auto",
        bins: int = 255,
        cat_features: Optional[List[int]] = None,
        eval_fraction: float = 0.0,
        early_stopping_rounds: int = 0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_seed: int = 42,
        nan_mode: str = "min",
        ctr: bool = False,
        ctr_prior: float = 0.5,
        max_onehot_size: int = 10,
        bootstrap_type: str = "no",
        bagging_temperature: float = 1.0,
        mvs_reg: float = 0.0,
        group_col: int = -1,
        min_data_in_leaf: int = 1,
        random_strength: float = 1.0,
        monotone_constraints: Optional[List[int]] = None,
        snapshot_path: Optional[str] = None,
        snapshot_interval: int = 1,
        auto_class_weights: Optional[str] = None,
        grow_policy: str = "SymmetricTree",
        max_leaves: int = 31,
        verbose: bool = False,
        binary_path: Optional[str] = None,
        train_timeout: Optional[float] = None,
        predict_timeout: Optional[float] = None,
        mlflow_logging: bool = False,
        mlflow_run_name: Optional[str] = None,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda
        self.loss = loss
        self.bins = bins
        self.cat_features = cat_features
        self.eval_fraction = eval_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_seed = random_seed
        self.nan_mode = nan_mode
        self.ctr = ctr
        self.ctr_prior = ctr_prior
        self.max_onehot_size = max_onehot_size
        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = bagging_temperature
        self.mvs_reg = mvs_reg
        self.group_col = group_col
        self.min_data_in_leaf = min_data_in_leaf
        self.random_strength = random_strength
        self.monotone_constraints = monotone_constraints
        self.snapshot_path = snapshot_path
        self.snapshot_interval = snapshot_interval
        self.auto_class_weights = auto_class_weights
        self.grow_policy = grow_policy
        self.max_leaves = max_leaves
        self.verbose = verbose
        self.binary_path = binary_path
        self.train_timeout = train_timeout
        self.predict_timeout = predict_timeout
        self.mlflow_logging = mlflow_logging
        self.mlflow_run_name = mlflow_run_name

        # Set after fit()
        self._model_path: Optional[str] = None
        self._model_data: Optional[dict] = None
        self._feature_importance: Optional[Dict[str, float]] = None
        self._train_loss_history: List[float] = []
        self._eval_loss_history: List[float] = []
        self._is_fitted = False
        self._model_json_cache: Optional[str] = None

    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status for sklearn compatibility (sklearn 1.3+)."""
        return self._is_fitted

    def _get_loss_type(self) -> str:
        """Extract loss type from model data."""
        if self._model_data:
            info = self._model_data.get("model_info", {})
            lt = info.get("loss_type", "")
            if lt:
                return lt.split(":")[0].lower()
        return self.loss.split(":")[0].lower()

    _KNOWN_LOSSES = frozenset({
        "auto", "rmse", "mae", "quantile", "huber", "poisson",
        "tweedie", "mape", "logloss", "multiclass", "pairlogit", "yetirank",
    })

    def _validate_params(self) -> None:
        """Validate hyperparameters before training."""
        # Guard against bool values -- bool is a subclass of int in Python,
        # so isinstance(True, int) is True. Check bool first for all numeric params.
        _bool_params = [
            ("iterations", self.iterations), ("depth", self.depth),
            ("learning_rate", self.learning_rate), ("l2_reg_lambda", self.l2_reg_lambda),
            ("bins", self.bins), ("eval_fraction", self.eval_fraction),
            ("early_stopping_rounds", self.early_stopping_rounds),
            ("subsample", self.subsample), ("colsample_bytree", self.colsample_bytree),
            ("bagging_temperature", self.bagging_temperature), ("mvs_reg", self.mvs_reg),
            ("max_onehot_size", self.max_onehot_size), ("ctr_prior", self.ctr_prior),
            ("random_strength", self.random_strength),
            ("min_data_in_leaf", self.min_data_in_leaf),
            ("snapshot_interval", self.snapshot_interval),
        ]
        for name, val in _bool_params:
            if isinstance(val, bool):
                raise ValueError(f"{name} must be a number, got {val!r} (bool)")
        if not isinstance(self.iterations, int) or not (1 <= self.iterations <= 100000):
            raise ValueError(f"iterations must be an integer in [1, 100000], got {self.iterations!r}")
        if not isinstance(self.depth, int) or not (1 <= self.depth <= 16):
            raise ValueError(f"depth must be an integer in [1, 16], got {self.depth!r}")
        if not isinstance(self.learning_rate, (int, float)) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be a finite number > 0, got {self.learning_rate!r}")
        if not isinstance(self.l2_reg_lambda, (int, float)) or not math.isfinite(self.l2_reg_lambda) or self.l2_reg_lambda < 0:
            raise ValueError(f"l2_reg_lambda must be a finite number >= 0, got {self.l2_reg_lambda!r}")

        loss_base = self.loss.split(":")[0].lower()
        if loss_base not in self._KNOWN_LOSSES:
            raise ValueError(
                f"Unknown loss '{self.loss}'. Known losses: {sorted(self._KNOWN_LOSSES)}"
            )
        if ":" in self.loss:
            suffix = self.loss.split(":", 1)[1]
            # Strip named-parameter prefix if present (e.g. 'alpha=0.7' -> '0.7')
            for prefix in ("alpha=", "delta=", "variance_power="):
                if suffix.lower().startswith(prefix):
                    suffix = suffix[len(prefix):]
                    break
            try:
                float(suffix)
            except ValueError:
                raise ValueError(
                    f"Loss parameter must be numeric, got '{self.loss}'. "
                    "Use positional syntax, e.g. 'quantile:0.7' or 'huber:1.0'."
                )

        if not isinstance(self.bins, int) or not (2 <= self.bins <= 255):
            raise ValueError(f"bins must be an integer in [2, 255], got {self.bins!r}")
        if not isinstance(self.eval_fraction, (int, float)) or not (0 <= self.eval_fraction < 1):
            raise ValueError(f"eval_fraction must be in [0, 1), got {self.eval_fraction!r}")
        if not isinstance(self.early_stopping_rounds, int) or self.early_stopping_rounds < 0:
            raise ValueError(f"early_stopping_rounds must be >= 0, got {self.early_stopping_rounds!r}")
        if not isinstance(self.subsample, (int, float)) or not (0 < self.subsample <= 1):
            raise ValueError(f"subsample must be in (0, 1], got {self.subsample!r}")
        if not isinstance(self.colsample_bytree, (int, float)) or not (0 < self.colsample_bytree <= 1):
            raise ValueError(f"colsample_bytree must be in (0, 1], got {self.colsample_bytree!r}")
        if self.nan_mode not in ("min", "forbidden"):
            raise ValueError(f"nan_mode must be 'min' or 'forbidden', got {self.nan_mode!r}")
        if self.grow_policy not in (None, "", "SymmetricTree", "Depthwise", "Lossguide"):
            raise ValueError(
                f"grow_policy must be 'SymmetricTree', 'Depthwise', or 'Lossguide', "
                f"got {self.grow_policy!r}"
            )
        if not isinstance(self.max_leaves, int) or self.max_leaves < 2:
            raise ValueError(
                f"max_leaves must be an integer >= 2, got {self.max_leaves!r}"
            )
        if self.bootstrap_type not in ("no", "bayesian", "bernoulli", "mvs"):
            raise ValueError(
                f"bootstrap_type must be one of 'no','bayesian','bernoulli','mvs', got {self.bootstrap_type!r}"
            )
        if not isinstance(self.bagging_temperature, (int, float)) or not math.isfinite(self.bagging_temperature) or self.bagging_temperature < 0:
            raise ValueError(f"bagging_temperature must be a finite number >= 0, got {self.bagging_temperature!r}")
        if not isinstance(self.mvs_reg, (int, float)) or not math.isfinite(self.mvs_reg) or self.mvs_reg < 0:
            raise ValueError(f"mvs_reg must be a finite number >= 0, got {self.mvs_reg!r}")
        if not isinstance(self.max_onehot_size, int) or self.max_onehot_size < 0:
            raise ValueError(f"max_onehot_size must be a non-negative integer, got {self.max_onehot_size!r}")
        if not isinstance(self.ctr_prior, (int, float)) or not math.isfinite(self.ctr_prior) or self.ctr_prior <= 0:
            raise ValueError(f"ctr_prior must be a finite number > 0, got {self.ctr_prior!r}")
        if not isinstance(self.random_strength, (int, float)) or not math.isfinite(self.random_strength) or self.random_strength < 0:
            raise ValueError(f"random_strength must be a finite number >= 0, got {self.random_strength!r}")
        if not isinstance(self.min_data_in_leaf, int) or self.min_data_in_leaf < 1:
            raise ValueError(f"min_data_in_leaf must be >= 1, got {self.min_data_in_leaf!r}")
        if self.monotone_constraints is not None:
            for c in self.monotone_constraints:
                if c not in (-1, 0, 1):
                    raise ValueError(
                        f"monotone_constraints values must be -1, 0, or 1, got {c!r}"
                    )
        if not isinstance(self.snapshot_interval, int) or self.snapshot_interval < 1:
            raise ValueError(f"snapshot_interval must be >= 1, got {self.snapshot_interval!r}")
        # Sanitize paths that get passed to subprocess to prevent injection
        if self.snapshot_path is not None:
            if not isinstance(self.snapshot_path, str) or "\x00" in self.snapshot_path:
                raise ValueError(f"snapshot_path must be a valid path string, got {self.snapshot_path!r}")
        if self.binary_path is not None:
            if not isinstance(self.binary_path, str) or "\x00" in self.binary_path:
                raise ValueError(f"binary_path must be a valid path string, got {self.binary_path!r}")
        if self.train_timeout is not None:
            if not isinstance(self.train_timeout, (int, float)) or self.train_timeout <= 0:
                raise ValueError(
                    f"train_timeout must be a positive number or None, got {self.train_timeout!r}"
                )
        if self.predict_timeout is not None:
            if not isinstance(self.predict_timeout, (int, float)) or self.predict_timeout <= 0:
                raise ValueError(
                    f"predict_timeout must be a positive number or None, got {self.predict_timeout!r}"
                )
        if self.auto_class_weights is not None:
            if self.auto_class_weights.lower() not in ("balanced", "sqrtbalanced"):
                raise ValueError(
                    "auto_class_weights must be 'Balanced' or 'SqrtBalanced', "
                    f"got {self.auto_class_weights!r}"
                )

    def _validate_fit_inputs(self, X, y, sample_weight=None, group_id=None,
                             feature_names=None) -> None:
        """Validate fit() input arrays."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if not np.issubdtype(y.dtype, np.number) and not np.issubdtype(y.dtype, np.bool_):
            raise ValueError(
                f"y must contain numeric values, got dtype '{y.dtype}'. "
                "For classification, use numeric labels (0, 1, 2, ...)."
            )
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]} samples"
            )
        if sample_weight is not None and len(sample_weight) != X.shape[0]:
            raise ValueError(
                f"sample_weight has {len(sample_weight)} elements but X has {X.shape[0]} samples"
            )
        if group_id is not None and len(group_id) != X.shape[0]:
            raise ValueError(
                f"group_id has {len(group_id)} elements but X has {X.shape[0]} samples"
            )
        if feature_names is not None and len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names has {len(feature_names)} names but X has {X.shape[1]} features"
            )
        if feature_names is not None:
            for i, fname in enumerate(feature_names):
                if not isinstance(fname, str):
                    raise ValueError(
                        f"feature_names[{i}] must be a string, got "
                        f"{type(fname).__name__}: {fname!r}"
                    )
                if "," in fname or "\n" in fname or "\r" in fname or "\x00" in fname:
                    raise ValueError(
                        "feature_names contain invalid characters (comma, "
                        f"newline, carriage return, or null byte): {fname!r}"
                    )
        if self.cat_features:
            for idx in self.cat_features:
                if idx < 0 or idx >= X.shape[1]:
                    raise ValueError(
                        f"cat_features index {idx} is out of bounds "
                        f"for X with {X.shape[1]} features"
                    )
        if self.monotone_constraints is not None:
            if len(self.monotone_constraints) != X.shape[1]:
                raise ValueError(
                    f"monotone_constraints has {len(self.monotone_constraints)} values "
                    f"but X has {X.shape[1]} features"
                )
        # Check for all-constant numeric features (would fail in C++ quantization)
        cat_set = set(self.cat_features) if self.cat_features else set()
        numeric_cols = [j for j in range(X.shape[1]) if j not in cat_set]
        if numeric_cols:
            try:
                num_data = X[:, numeric_cols].astype(float)
            except (ValueError, TypeError):
                pass  # can't convert to float (e.g. mixed categorical), skip check
            else:
                variances = np.var(num_data, axis=0)
                if np.all(variances == 0):
                    raise ValueError(
                        "All numeric features are constant (zero variance). "
                        "The model cannot learn from constant features."
                    )

    def _build_train_args(self, csv_path: str, model_path: str, target_col: int,
                          eval_file: Optional[str] = None,
                          cv_folds: Optional[int] = None,
                          cat_col_offset: int = 0) -> List[str]:
        """Build the command-line argument list for the csv_train binary.

        Always includes --verbose and --feature-importance so we can parse
        training progress and importance from stdout. Only includes non-default
        values for optional parameters to keep the command short.

        When cv_folds is set, adds --cv N instead of --output model_path.
        """
        binary = _find_binary("csv_train", self.binary_path)
        args = [
            binary, csv_path,
            "--iterations", str(self.iterations),
            "--depth", str(self.depth),
            "--lr", str(self.learning_rate),
            "--l2", str(self.l2_reg_lambda),
            "--loss", _normalize_loss_str(self.loss),
            "--bins", str(self.bins),
            "--target-col", str(target_col),
            "--seed", str(self.random_seed),
            "--nan-mode", self.nan_mode,
        ]
        if cv_folds is not None:
            args.extend(["--cv", str(cv_folds)])
        else:
            args.extend(["--output", model_path])
        args.extend(["--feature-importance", "--verbose"])
        # Categorical features — offset indices to account for prepended
        # group/weight columns so they map to CSV column positions
        if self.cat_features:
            csv_cat_cols = [c + cat_col_offset for c in self.cat_features]
            args.extend(["--cat-features", ",".join(str(c) for c in csv_cat_cols)])
        # Validation data: eval_file (external) takes priority over eval_fraction (auto-split)
        if eval_file is not None:
            args.extend(["--eval-file", eval_file])
        elif self.eval_fraction > 0:
            args.extend(["--eval-fraction", str(self.eval_fraction)])
        if self.early_stopping_rounds > 0:
            args.extend(["--early-stopping", str(self.early_stopping_rounds)])
        if self.subsample < 1.0:
            args.extend(["--subsample", str(self.subsample)])
        if self.colsample_bytree < 1.0:
            args.extend(["--colsample-bytree", str(self.colsample_bytree)])
        # CTR (target encoding) configuration
        if self.ctr:
            args.append("--ctr")
            args.extend(["--ctr-prior", str(self.ctr_prior)])
            args.extend(["--max-onehot-size", str(self.max_onehot_size)])
        # Bootstrap configuration (only sent if non-default)
        if self.bootstrap_type != "no":
            args.extend(["--bootstrap-type", self.bootstrap_type])
            if self.bootstrap_type == "bayesian":
                args.extend(["--bagging-temperature", str(self.bagging_temperature)])
            if self.bootstrap_type == "mvs":
                args.extend(["--mvs-reg", str(self.mvs_reg)])
        if self.group_col >= 0:
            args.extend(["--group-col", str(self.group_col)])
        if self.min_data_in_leaf > 1:
            args.extend(["--min-data-in-leaf", str(self.min_data_in_leaf)])
        if self.monotone_constraints:
            args.extend(["--monotone-constraints", ",".join(str(c) for c in self.monotone_constraints)])
        if self.snapshot_path:
            args.extend(["--snapshot-path", self.snapshot_path])
            if self.snapshot_interval > 1:
                args.extend(["--snapshot-interval", str(self.snapshot_interval)])
        if self.random_strength != 1.0:
            args.extend(["--random-strength", str(self.random_strength)])
        if self.grow_policy and self.grow_policy != "SymmetricTree":
            args.extend(["--grow-policy", self.grow_policy])
        if self.grow_policy == "Lossguide":
            args.extend(["--max-leaves", str(self.max_leaves)])
        return args

    def _run_train_subprocess(self, args: List[str]) -> str:
        """Run csv_train with real-time stdout streaming when verbose=True.

        Why PTY? The C++ binary uses printf(), which is fully-buffered when
        its stdout is a pipe (i.e. subprocess). This means you'd see nothing
        until training finishes. A PTY (pseudo-terminal) tricks the binary
        into thinking it's writing to a real terminal, forcing line-buffered
        output so we can print progress in real time.
        """
        if self.verbose:
            try:
                import pty  # noqa: F401
                import select  # noqa: F401
                return self._run_with_pty(args)
            except (ImportError, OSError):
                pass

        # Non-verbose or pty fallback: standard subprocess.run
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=self.train_timeout
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"csv_train timed out after {self.train_timeout}s. "
                "Increase train_timeout or reduce dataset size/iterations."
            ) from e
        if result.returncode != 0:
            raise RuntimeError(
                f"csv_train failed (exit code {result.returncode}):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        return result.stdout

    def _run_with_pty(self, args: List[str]) -> str:
        """Run subprocess with a PTY for real-time line output."""
        import pty
        import select

        deadline = (time.monotonic() + self.train_timeout) if self.train_timeout else None

        # Create a pseudo-terminal pair: child writes to slave, we read from master
        master_fd, slave_fd = pty.openpty()
        # Launch the C++ binary with its stdout connected to the PTY slave
        proc = subprocess.Popen(args, stdout=slave_fd, stderr=subprocess.PIPE)
        os.close(slave_fd)  # parent doesn't need the slave side

        stdout_lines = []
        buf = b""
        try:
            while True:
                if deadline and time.monotonic() > deadline:
                    proc.kill()
                    proc.wait()
                    raise RuntimeError(
                        f"csv_train timed out after {self.train_timeout}s. "
                        "Increase train_timeout or reduce dataset size/iterations."
                    )
                # Poll the master fd with 100ms timeout (avoids busy-waiting)
                r, _, _ = select.select([master_fd], [], [], 0.1)
                if r:
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    # Accumulate bytes and split on newlines for clean output
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        text = line.decode("utf-8", errors="replace").rstrip("\r")
                        stdout_lines.append(text)
                        logger.debug(text)
                        print(text, flush=True)
                elif proc.poll() is not None:
                    # Process exited -- drain any remaining buffered output
                    if buf:
                        text = buf.decode("utf-8", errors="replace").rstrip("\r\n")
                        if text:
                            stdout_lines.append(text)
                            logger.debug(text)
                            print(text, flush=True)
                    break
        finally:
            os.close(master_fd)

        # Read stderr BEFORE wait to prevent deadlock if buffer fills
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"csv_train failed (exit code {proc.returncode}):\n"
                f"stdout: {chr(10).join(stdout_lines)}\n"
                f"stderr: {stderr}"
            )
        return "\n".join(stdout_lines)

    def _parse_train_output(self, stdout: str) -> None:
        """Parse csv_train stdout for loss history and feature importance.

        The C++ binary prints structured lines we can regex-match:
        - Iteration lines: 'iter=0  trees=1  depth=6  loss=0.123 [val_loss=0.456]'
        - Feature importance: '  1     feature_name        45.23   52.3%'
        """
        self._train_loss_history = []
        self._eval_loss_history = []
        self._feature_importance = {}

        for line in stdout.split("\n"):
            # Parse iteration lines: iter=0  trees=1  depth=6  loss=0.123  ...
            m = re.match(r"iter=\d+\s+trees=\d+\s+depth=\d+\s+loss=([\-\d.]+)", line)
            if m:
                self._train_loss_history.append(float(m.group(1)))
                # Check for validation loss
                vm = re.search(r"val_loss=([\-\d.]+)", line)
                if vm:
                    self._eval_loss_history.append(float(vm.group(1)))
                continue

            # Parse feature importance: "  1     feature_name        45.23   52.3%"
            m = re.match(r"\s*\d+\s+(\S+)\s+([\d.]+)\s+([\d.]+)%", line)
            if m:
                self._feature_importance[m.group(1)] = float(m.group(2))

    def _log_to_mlflow(self) -> None:
        """Log hyperparameters, per-iteration loss, and final metrics to MLflow.

        Called at the end of fit() when mlflow_logging=True. Imports mlflow
        lazily so it remains an optional dependency.

        Run scoping rules:
        - If a run is already active (user started one externally), log into it
          and leave it open when we exit -- we don't own it.
        - If no run is active, start a new one with mlflow_run_name (or a
          default name), log everything, then end it before returning.
        """
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "mlflow is required for mlflow_logging=True. "
                "Install it with: pip install mlflow"
            )

        # Detect whether a run is already active so we know who owns it.
        active_run = mlflow.active_run()
        we_started_run = active_run is None

        if we_started_run:
            run_name = self.mlflow_run_name or (
                f"catboost_mlx_{self.__class__.__name__.lower()}"
            )
            active_run = mlflow.start_run(run_name=run_name)

        try:
            # ── Hyperparameters ───────────────────────────────────────────────
            params = {
                "iterations": self.iterations,
                "depth": self.depth,
                "learning_rate": self.learning_rate,
                "l2_reg_lambda": self.l2_reg_lambda,
                "loss": self.loss,
                "bins": self.bins,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "eval_fraction": self.eval_fraction,
                "early_stopping_rounds": self.early_stopping_rounds,
                "random_seed": self.random_seed,
                "nan_mode": self.nan_mode,
                "min_data_in_leaf": self.min_data_in_leaf,
                "bootstrap_type": self.bootstrap_type,
                "random_strength": self.random_strength,
            }
            mlflow.log_params(params)

            # ── Per-iteration train loss ──────────────────────────────────────
            for step, loss_val in enumerate(self._train_loss_history):
                mlflow.log_metric("train_loss", loss_val, step=step)

            # ── Per-iteration eval loss (when validation data was used) ───────
            for step, loss_val in enumerate(self._eval_loss_history):
                mlflow.log_metric("eval_loss", loss_val, step=step)

            # ── Final summary metrics ─────────────────────────────────────────
            final_metrics: Dict[str, float] = {}
            if self._train_loss_history:
                final_metrics["final_train_loss"] = self._train_loss_history[-1]
            if self._eval_loss_history:
                final_metrics["final_eval_loss"] = self._eval_loss_history[-1]
            # Trees actually built (may be fewer than iterations with early stopping)
            if self._model_data:
                info = self._model_data.get("model_info", {})
                trees_built = len(self._model_data.get("trees", []))
                final_metrics["trees_built"] = float(trees_built)
                if "approx_dimension" in info:
                    final_metrics["approx_dimension"] = float(
                        info["approx_dimension"]
                    )
            if final_metrics:
                mlflow.log_metrics(final_metrics)

        finally:
            # Only end the run if we were the one who opened it.
            if we_started_run:
                mlflow.end_run()

    def fit(self, X_or_pool, y=None, eval_set=None, feature_names: Optional[List[str]] = None,
            group_id=None, sample_weight=None) -> "CatBoostMLX":
        """Train a model on the given data.

        Parameters
        ----------
        X_or_pool : array-like, DataFrame, or Pool
            Feature matrix of shape (n_samples, n_features), or a Pool object.
        y : array-like of shape (n_samples,), optional
            Target values. Required unless X_or_pool is a Pool with labels.
        eval_set : tuple of (X_eval, y_eval) or Pool, optional
            External validation data. Mutually exclusive with eval_fraction.
        feature_names : list of str, optional
            Names for each feature column.
        group_id : array-like of shape (n_samples,), optional
            Group/query IDs for ranking losses. All docs in the same group
            must have the same group_id.
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights for training.

        Returns
        -------
        self
        """
        from .pool import Pool, _is_dataframe, _resolve_cat_features

        # ── Phase 1: Unpack input (Pool object or raw arrays) ──
        if isinstance(X_or_pool, Pool):
            pool = X_or_pool
            X = pool.X
            y = pool.y if y is None else _to_numpy(y)
            feature_names = feature_names or pool.feature_names
            group_id = group_id if group_id is not None else pool.group_id
            sample_weight = sample_weight if sample_weight is not None else pool.sample_weight
            if self.cat_features is None and pool.cat_features:
                self.cat_features = pool.cat_features
        else:
            # Auto-extract feature names from DataFrame before conversion
            if _is_dataframe(X_or_pool) and feature_names is None:
                feature_names = list(X_or_pool.columns)
            X = _to_numpy(X_or_pool)
            y = _to_numpy(y)

        if y is None or (isinstance(y, np.ndarray) and y.ndim == 0):
            raise ValueError("y is required when X is not a Pool with labels.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # ── Phase 2: Resolve categorical features ──
        # Resolve string cat_features to indices
        if self.cat_features and any(isinstance(c, str) for c in self.cat_features):
            self.cat_features = _resolve_cat_features(self.cat_features, feature_names)

        # Unpack eval_set Pool
        if isinstance(eval_set, Pool):
            eval_set = (eval_set.X, eval_set.y)

        # Convert group_id and sample_weight to numpy if provided
        gid = np.asarray(group_id) if group_id is not None else None
        sw = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None

        # ── Phase 3: Validation ──
        self._validate_params()
        self._validate_fit_inputs(X, y, sw, gid, feature_names)

        # ── Phase 4: Auto class weights ──
        # Compute balanced sample weights from class distribution if requested
        if self.auto_class_weights and sw is None:
            classes, counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            n_classes = len(classes)
            class_weight = {}
            for cls, cnt in zip(classes, counts):
                if self.auto_class_weights.lower() == "balanced":
                    class_weight[cls] = n_samples / (n_classes * cnt)
                else:  # sqrtbalanced
                    class_weight[cls] = np.sqrt(n_samples / (n_classes * cnt))
            sw = np.array([class_weight[yi] for yi in y], dtype=float)

        # Warn on constant target (GBDT converges slowly)
        if np.std(y) == 0:
            import warnings
            warnings.warn(
                "Target has zero variance (constant). Predictions may not converge "
                f"with only {self.iterations} iterations at lr={self.learning_rate}.",
                UserWarning, stacklevel=2,
            )

        # ── Phase 5: Set sklearn-required attributes and write training data ──
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1
        names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.feature_names_in_ = np.array(names, dtype=object)

        # Use binary format for numeric-only data (1000x faster than CSV)
        use_binary = not self.cat_features

        tmpdir = tempfile.mkdtemp(prefix="catboost_mlx_")
        try:
            model_path = os.path.join(tmpdir, "model.json")

            if use_binary:
                data_path = os.path.join(tmpdir, "train.cbmx")
                _array_to_binary(data_path, X, y, group_id=gid,
                                 sample_weight=sw)
                target_col = -1  # not used for binary format
                group_col_idx = -1
                weight_col_idx = -1
                csv_path = data_path
            else:
                csv_path = os.path.join(tmpdir, "train.csv")
                target_col, group_col_idx, weight_col_idx = _array_to_csv(
                    csv_path, X, y, feature_names, self.cat_features,
                    group_id=gid, sample_weight=sw
                )

            # ── Phase 6: Handle eval_set (external validation data) ──
            eval_file_path = None
            if eval_set is not None:
                if not (isinstance(eval_set, (tuple, list)) and len(eval_set) == 2):
                    raise ValueError("eval_set must be a tuple of (X_val, y_val)")
                if self.eval_fraction > 0:
                    raise ValueError(
                        "eval_set and eval_fraction are mutually exclusive"
                    )
                # Warn if eval_set DataFrame columns don't match training names
                if feature_names and hasattr(eval_set[0], "columns"):
                    eval_cols = list(eval_set[0].columns)
                    if eval_cols != feature_names:
                        import warnings
                        warnings.warn(
                            f"eval_set feature names {eval_cols} differ from "
                            f"training feature names {feature_names}. "
                            "Column order will follow training names.",
                            UserWarning,
                            stacklevel=2,
                        )
                X_val = _to_numpy(eval_set[0])
                y_val = _to_numpy(eval_set[1])
                if X_val.ndim == 1:
                    X_val = X_val.reshape(-1, 1)
                if X_val.shape[1] != X.shape[1]:
                    raise ValueError(
                        f"eval_set X has {X_val.shape[1]} features, "
                        f"training X has {X.shape[1]} features"
                    )
                if use_binary:
                    eval_file_path = os.path.join(tmpdir, "eval.cbmx")
                    eval_gid = np.zeros(len(y_val), dtype=int) if gid is not None else None
                    eval_sw = np.ones(len(y_val), dtype=float) if sw is not None else None
                    _array_to_binary(eval_file_path, X_val, y_val,
                                     group_id=eval_gid, sample_weight=eval_sw)
                else:
                    eval_file_path = os.path.join(tmpdir, "eval.csv")
                    # Write eval CSV with same column layout as training CSV
                    # (include dummy group_id/weight columns so target_col aligns)
                    eval_gid = np.zeros(len(y_val), dtype=int) if gid is not None else None
                    eval_sw = np.ones(len(y_val), dtype=float) if sw is not None else None
                    _array_to_csv(eval_file_path, X_val, y_val, feature_names,
                                  self.cat_features, group_id=eval_gid,
                              sample_weight=eval_sw)

            # ── Phase 7: Build CLI command and run the C++ training binary ──
            # Temporarily override group_col and weight_col if provided
            # Compute column offset for cat_features: group and weight columns
            # are prepended before feature columns in the CSV
            cat_col_offset = (1 if gid is not None else 0) + (1 if sw is not None else 0)
            orig_group_col = self.group_col
            try:
                if gid is not None and self.group_col < 0:
                    self.group_col = group_col_idx
                args = self._build_train_args(csv_path, model_path, target_col,
                                              eval_file=eval_file_path,
                                              cat_col_offset=cat_col_offset)
            finally:
                self.group_col = orig_group_col

            # Add weight-col if sample_weight was provided
            if weight_col_idx >= 0:
                args.extend(["--weight-col", str(weight_col_idx)])

            # ── Phase 8: Parse output and load the trained model ──
            stdout_text = self._run_train_subprocess(args)
            self._parse_train_output(stdout_text)

            # Load model JSON before cleanup — tmpdir is about to be deleted
            with open(model_path, "r") as f:
                self._model_data = json.load(f)
            self._model_path = None
            self._is_fitted = True
            self._model_json_cache = None  # invalidate cached serialization
            # Inject real feature names into model data (C++ binary uses
            # generic f0/f1/... for binary format)
            if hasattr(self, "feature_names_in_"):
                for i, feat in enumerate(self._model_data.get("features", [])):
                    if i < len(self.feature_names_in_):
                        feat["name"] = str(self.feature_names_in_[i])
            # Persist cat_features in model_info so save/load roundtrips
            # correctly (CTR models don't set is_categorical on features)
            info = self._model_data.get("model_info", {})
            info["cat_features"] = self.cat_features
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # ── MLflow logging (optional) ──────────────────────────────────────────
        if self.mlflow_logging:
            self._log_to_mlflow()

        return self

    def _validate_feature_names(self, X) -> None:
        """Warn if DataFrame column names don't match training names (sklearn 1.2+)."""
        if not hasattr(self, "feature_names_in_"):
            return
        try:
            import pandas as pd
            if not isinstance(X, pd.DataFrame):
                return
        except ImportError:
            return
        import warnings
        train_names = list(self.feature_names_in_)
        input_names = list(X.columns)
        if train_names != input_names:
            warnings.warn(
                "X has feature names that differ from the feature names seen "
                f"in `fit`. Feature names seen in `fit`: {train_names}. "
                f"Feature names in input: {input_names}.",
                UserWarning,
                stacklevel=3,
            )

    def _unpack_predict_input(
        self, X, feature_names=None
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Unpack predict input, handling Pool objects transparently."""
        from .pool import Pool
        if isinstance(X, Pool):
            feature_names = feature_names or X.feature_names
            X = X.X
        return X, feature_names

    def predict(self, X, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Predict raw values / class labels.

        For regression: returns predicted values.
        For binary classification: returns predicted class (0 or 1).
        For multiclass: returns predicted class index.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X, feature_names = self._unpack_predict_input(X, feature_names)
        self._validate_feature_names(X)
        output = self._run_predict(X, feature_names)
        loss_type = self._get_loss_type()

        if loss_type in ("logloss", "multiclass"):
            return output["predicted_class"].astype(int)
        return output["prediction"]

    def predict_proba(self, X, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Predict class probabilities (classification only).

        For binary classification: returns array of shape (n_samples, 2).
        For multiclass: returns array of shape (n_samples, n_classes).
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X, feature_names = self._unpack_predict_input(X, feature_names)
        self._validate_feature_names(X)

        output = self._run_predict(X, feature_names)
        loss_type = self._get_loss_type()

        if loss_type == "logloss":
            prob = output["probability"]
            return np.column_stack([1.0 - prob, prob])
        elif loss_type == "multiclass":
            # Collect prob_class_N columns
            prob_cols = sorted(
                [k for k in output if k.startswith("prob_class_")],
                key=lambda k: int(k.split("_")[-1])
            )
            return np.column_stack([output[c] for c in prob_cols])
        else:
            raise ValueError(f"predict_proba is not supported for loss '{loss_type}'")

    def staged_predict(self, X, feature_names: Optional[List[str]] = None,
                       eval_period: int = 1):
        """Yield predictions at every eval_period boosting iterations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature_names : list of str, optional
        eval_period : int
            Yield every eval_period trees (default 1).

        Yields
        ------
        np.ndarray
            Predictions at each checkpoint.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if eval_period < 1:
            raise ValueError(f"eval_period must be >= 1, got {eval_period}")

        X, feature_names = self._unpack_predict_input(X, feature_names)

        from ._predict_utils import apply_link, compute_leaf_indices, quantize_features

        X = _to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was trained on "
                f"{self.n_features_in_} features."
            )

        features = self._model_data["features"]
        trees = self._model_data["trees"]
        info = self._model_data.get("model_info", {})
        approx_dim = info.get("approx_dimension", 1)
        loss_type = self._get_loss_type()
        num_classes = info.get("num_classes", 0)
        n_trees = len(trees)
        n_samples = X.shape[0]

        binned = quantize_features(X, features, self.cat_features)

        # Initialize cursor with base prediction (boost from average)
        base_pred = info.get("base_prediction", [])
        if approx_dim == 1:
            bp = base_pred[0] if base_pred else 0.0
            cursor = np.full(n_samples, bp, dtype=float)
        else:
            cursor = np.zeros((n_samples, approx_dim), dtype=float)
            if base_pred and len(base_pred) >= approx_dim:
                cursor += np.array(base_pred[:approx_dim], dtype=float)

        for t in range(n_trees):
            tree = trees[t]
            leaf_idx = compute_leaf_indices(binned, tree)
            leaf_vals = np.array(tree["leaf_values"], dtype=float)
            if approx_dim == 1:
                cursor += leaf_vals[leaf_idx]
            else:
                cursor += leaf_vals.reshape(-1, approx_dim)[leaf_idx]

            # Yield predictions at regular intervals (eval_period=10 means
            # yield after trees 10, 20, 30, ... plus always the final tree)
            if (t + 1) % eval_period == 0 or t == n_trees - 1:
                result = apply_link(cursor.copy(), loss_type, num_classes)
                if loss_type in ("logloss", "multiclass"):
                    yield result["predicted_class"]
                else:
                    yield result["prediction"]

    def staged_predict_proba(self, X, feature_names: Optional[List[str]] = None,
                             eval_period: int = 1):
        """Yield class probabilities at every eval_period boosting iterations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature_names : list of str, optional
        eval_period : int
            Yield every eval_period trees (default 1).

        Yields
        ------
        np.ndarray
            Probability array at each checkpoint.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if eval_period < 1:
            raise ValueError(f"eval_period must be >= 1, got {eval_period}")

        X, feature_names = self._unpack_predict_input(X, feature_names)

        loss_type = self._get_loss_type()
        if loss_type not in ("logloss", "multiclass"):
            raise ValueError(f"staged_predict_proba not supported for loss '{loss_type}'")

        from ._predict_utils import apply_link, compute_leaf_indices, quantize_features

        X = _to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was trained on "
                f"{self.n_features_in_} features."
            )

        features = self._model_data["features"]
        trees = self._model_data["trees"]
        info = self._model_data.get("model_info", {})
        approx_dim = info.get("approx_dimension", 1)
        num_classes = info.get("num_classes", 0)
        n_trees = len(trees)
        n_samples = X.shape[0]

        binned = quantize_features(X, features, self.cat_features)

        base_pred = info.get("base_prediction", [])
        if approx_dim == 1:
            bp = base_pred[0] if base_pred else 0.0
            cursor = np.full(n_samples, bp, dtype=float)
        else:
            cursor = np.zeros((n_samples, approx_dim), dtype=float)
            if base_pred and len(base_pred) >= approx_dim:
                cursor += np.array(base_pred[:approx_dim], dtype=float)

        for t in range(n_trees):
            tree = trees[t]
            leaf_idx = compute_leaf_indices(binned, tree)
            leaf_vals = np.array(tree["leaf_values"], dtype=float)
            if approx_dim == 1:
                cursor += leaf_vals[leaf_idx]
            else:
                cursor += leaf_vals.reshape(-1, approx_dim)[leaf_idx]

            if (t + 1) % eval_period == 0 or t == n_trees - 1:
                result = apply_link(cursor.copy(), loss_type, num_classes)
                if loss_type == "logloss":
                    prob = result["probability"]
                    yield np.column_stack([1.0 - prob, prob])
                else:
                    prob_cols = sorted(
                        [k for k in result if k.startswith("prob_class_")],
                        key=lambda k: int(k.split("_")[-1])
                    )
                    yield np.column_stack([result[c] for c in prob_cols])

    def apply(self, X, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Return leaf indices for each sample in each tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature_names : list of str, optional

        Returns
        -------
        np.ndarray of shape (n_samples, n_trees), dtype int32
            Leaf index for each sample in each tree. Values in [0, 2^depth).
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X, feature_names = self._unpack_predict_input(X, feature_names)

        from ._predict_utils import compute_leaf_indices, quantize_features

        X = _to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was trained on "
                f"{self.n_features_in_} features."
            )

        features = self._model_data["features"]
        trees = self._model_data["trees"]

        binned = quantize_features(X, features, self.cat_features)

        result = np.empty((X.shape[0], len(trees)), dtype=np.int32)
        for t, tree in enumerate(trees):
            result[:, t] = compute_leaf_indices(binned, tree)

        return result

    def _run_predict(self, X, feature_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Predict using in-process tree evaluation or C++ subprocess.

        For numeric-only models, evaluates trees directly in Python/NumPy
        (30x faster than subprocess). Falls back to the C++ subprocess for
        models with categorical features (which need CTR encoding).
        """
        X = _to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was trained on "
                f"{self.n_features_in_} features."
            )

        # Early return for empty input
        if X.shape[0] == 0:
            return {"prediction": np.array([], dtype=float)}

        # In-process prediction for numeric-only models (no CTR encoding needed)
        if not self.cat_features:
            return self._predict_inprocess(X)

        # Fallback: C++ subprocess for categorical features
        return self._predict_subprocess(X, feature_names)

    def _predict_inprocess(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate trees in Python/NumPy without subprocess overhead."""
        from ._predict_utils import apply_link, evaluate_trees, quantize_features

        features = self._model_data["features"]
        trees = self._model_data["trees"]
        info = self._model_data.get("model_info", {})
        approx_dim = info.get("approx_dimension", 1)
        loss_type = self._get_loss_type()
        num_classes = info.get("num_classes", 0)
        base_pred = info.get("base_prediction")

        binned = quantize_features(X, features)
        cursor = evaluate_trees(binned, trees, approx_dim,
                                base_prediction=base_pred)
        return apply_link(cursor, loss_type, num_classes)

    def _predict_subprocess(self, X: np.ndarray,
                            feature_names: Optional[List[str]] = None
                            ) -> Dict[str, np.ndarray]:
        """Run csv_predict via subprocess (needed for categorical features)."""
        tmpdir = tempfile.mkdtemp(prefix="catboost_mlx_pred_")
        try:
            out_path = os.path.join(tmpdir, "predictions.csv")
            model_path = os.path.join(tmpdir, "model.json")

            # Write model JSON (regenerate to avoid stale cache)
            model_json = json.dumps(self._model_data)
            self._model_json_cache = model_json
            with open(model_path, "w") as f:
                f.write(model_json)

            data_path = os.path.join(tmpdir, "data.csv")
            _array_to_csv(data_path, X, feature_names=feature_names,
                          cat_features=self.cat_features)

            binary = _find_binary("csv_predict", self.binary_path)
            args = [binary, model_path, data_path, "--output", out_path]

            try:
                result = subprocess.run(
                    args, capture_output=True, text=True, timeout=self.predict_timeout
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"csv_predict timed out after {self.predict_timeout}s. "
                    "Increase predict_timeout or reduce dataset size."
                ) from e
            if result.returncode != 0:
                raise RuntimeError(
                    f"csv_predict failed (exit code {result.returncode}):\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            # Parse output CSV
            columns: Dict[str, List[float]] = {}
            with open(out_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, val in row.items():
                        if key not in columns:
                            columns[key] = []
                        columns[key].append(float(val))

            return {k: np.array(v) for k, v in columns.items()}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def save_model(self, path: str) -> None:
        """Save the trained model to a JSON file."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        payload = {"format_version": 2, **self._model_data}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def export_coreml(self, path: str) -> None:
        """Export model to CoreML format (.mlmodel).

        Requires coremltools: pip install coremltools>=7.0
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        from .export_coreml import export_coreml
        export_coreml(self._model_data, path)

    def export_onnx(self, path: str) -> None:
        """Export model to ONNX format (.onnx).

        Requires onnx: pip install onnx>=1.14
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        from .export_onnx import export_onnx
        export_onnx(self._model_data, path)

    def load_model(self, path: str) -> "CatBoostMLX":
        """Load a model from a JSON file."""
        with open(path, "r") as f:
            self._model_data = json.load(f)
        # Format version check — default to 1 for files saved before versioning.
        fmt_version = self._model_data.pop("format_version", 1)
        if fmt_version > 2:
            raise ValueError(
                f"Model was saved with a newer version of catboost-mlx "
                f"(format_version={fmt_version}). Upgrade catboost-mlx to load it."
            )
        required = {"model_info", "trees", "features"}
        missing = required - set(self._model_data.keys())
        if missing:
            raise ValueError(
                f"Invalid model JSON: missing required keys {missing}"
            )
        self._model_path = path
        self._is_fitted = True
        self._model_json_cache = None  # invalidate cached serialization
        # Restore sklearn-compatible attributes from model data
        features = self._model_data.get("features", [])
        self.n_features_in_ = len(features)
        self.n_outputs_ = 1
        names = [f.get("name", f"f{f.get('index', i)}") for i, f in enumerate(features)]
        self.feature_names_in_ = np.array(names, dtype=object)
        # Restore cat_features: prefer explicit list from model_info (set by
        # fit()), fall back to is_categorical flags for older model files.
        info = self._model_data.get("model_info", {})
        saved_cat = info.get("cat_features")
        if saved_cat is not None:
            self.cat_features = saved_cat if saved_cat else None
        else:
            cat_indices = [i for i, f in enumerate(features)
                           if f.get("is_categorical", False)]
            self.cat_features = cat_indices if cat_indices else None
        # Sync loss parameter so the instance reflects the trained model's loss
        model_loss = self._model_data.get("model_info", {}).get("loss_type")
        if model_loss:
            self.loss = model_loss.split(":")[0].lower()
        return self

    @classmethod
    def load(cls, path: str, binary_path: Optional[str] = None) -> "CatBoostMLX":
        """Load a model from a JSON file, returning a new instance.

        Parameters
        ----------
        path : str
            Path to the model JSON file.
        binary_path : str, optional
            Path to directory containing csv_train/csv_predict binaries.

        Returns
        -------
        CatBoostMLX
            A new fitted model instance.
        """
        instance = cls(binary_path=binary_path)
        instance.load_model(path)
        return instance

    def get_feature_importance(self) -> Dict[str, float]:
        """Return gain-based feature importance as {name: gain_value}."""
        if self._feature_importance:
            return dict(self._feature_importance)
        if self._model_data and "feature_importance" in self._model_data:
            return {
                fi["name"]: fi["gain"]
                for fi in self._model_data["feature_importance"]
            }
        return {}

    @property
    def tree_count_(self) -> int:
        """Number of trees in the fitted model."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return len(self._model_data.get("trees", []))

    @property
    def feature_names_(self) -> List[str]:
        """Feature names used during training."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if hasattr(self, "feature_names_in_"):
            return list(self.feature_names_in_)
        # Fallback: extract from model data
        features = self._model_data.get("features", [])
        return [f.get("name", f"f{f['index']}") for f in features]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Normalized feature importance array (sklearn-compatible).

        Returns array of shape (n_features,) summing to 1.0.
        Features never split on get importance 0.0.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        fi = self.get_feature_importance()
        n = self.n_features_in_
        names = (list(self.feature_names_in_) if hasattr(self, "feature_names_in_")
                 else [f"f{i}" for i in range(n)])
        arr = np.zeros(n, dtype=float)
        for i, name in enumerate(names):
            arr[i] = fi.get(name, 0.0)
        total = arr.sum()
        if total > 0:
            arr /= total
        return arr

    def get_trees(self) -> List[dict]:
        """Return list of tree dicts with real-valued thresholds.

        Each tree dict contains:
        - depth: int
        - nodes: list of branch/leaf node dicts (from unfold_oblivious_tree)
        - leaf_values: flat list of leaf values
        - split_gains: list of gains per split level
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        from ._tree_utils import unfold_oblivious_tree
        features = self._model_data["features"]
        approx_dim = self._model_data["model_info"].get("approx_dimension", 1)
        return [
            {
                "depth": t["depth"],
                "nodes": unfold_oblivious_tree(t, features, approx_dim),
                "leaf_values": t["leaf_values"],
                "split_gains": t.get("split_gains", []),
            }
            for t in self._model_data["trees"]
        ]

    def get_model_info(self) -> dict:
        """Return model metadata (loss, dimensions, feature count, tree count)."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        info = dict(self._model_data.get("model_info", {}))
        info["num_features"] = len(self._model_data.get("features", []))
        return info

    def plot_feature_importance(self, max_features: int = 20) -> None:
        """Print text bar chart of feature importance to terminal."""
        fi = self.get_feature_importance()
        if not fi:
            print("No feature importance available.")
            return
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:max_features]
        max_gain = sorted_fi[0][1] if sorted_fi else 1.0
        max_name_len = max(len(name) for name, _ in sorted_fi)
        for name, gain in sorted_fi:
            bar_len = int(40 * gain / max_gain) if max_gain > 0 else 0
            print(f"  {name:<{max_name_len}}  {'#' * bar_len}  {gain:.4f}")

    def get_shap_values(self, X, feature_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Compute TreeSHAP values for each prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature_names : list of str, optional

        Returns
        -------
        dict with keys:
            shap_values: array of shape (n_samples, n_features)
            expected_value: float (base score)
            feature_names: list of str
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X, feature_names = self._unpack_predict_input(X, feature_names)

        X = _to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was trained on "
                f"{self.n_features_in_} features."
            )

        tmpdir = tempfile.mkdtemp(prefix="catboost_mlx_shap_")
        try:
            out_path = os.path.join(tmpdir, "predictions.csv")
            model_path = os.path.join(tmpdir, "model.json")

            with open(model_path, "w") as f:
                self._model_json_cache = json.dumps(self._model_data)
                f.write(self._model_json_cache)

            if not self.cat_features:
                csv_path = os.path.join(tmpdir, "data.cbmx")
                _array_to_binary(csv_path, X)
            else:
                csv_path = os.path.join(tmpdir, "data.csv")
                _array_to_csv(csv_path, X, feature_names=feature_names,
                              cat_features=self.cat_features)

            binary = _find_binary("csv_predict", self.binary_path)
            args = [binary, model_path, csv_path, "--output", out_path, "--shap"]

            try:
                result = subprocess.run(
                    args, capture_output=True, text=True, timeout=self.predict_timeout
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"csv_predict --shap timed out after {self.predict_timeout}s. "
                    "Increase predict_timeout or reduce dataset size."
                ) from e
            if result.returncode != 0:
                raise RuntimeError(
                    f"csv_predict --shap failed (exit code {result.returncode}):\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            # Parse the SHAP output CSV (_shap.csv)
            shap_path = out_path.rsplit(".csv", 1)[0] + "_shap.csv"
            if not os.path.exists(shap_path):
                raise RuntimeError(
                    f"SHAP output file not found at {shap_path}. "
                    "The C++ binary may not support SHAP for this model configuration."
                )
            columns: Dict[str, List[float]] = {}
            with open(shap_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, val in row.items():
                        if key not in columns:
                            columns[key] = []
                        columns[key].append(float(val))

            # Separate SHAP columns from metadata
            # Check for multiclass format: feature_shap_classN
            multiclass_cols = [k for k in columns if re.search(r'_shap_class\d+$', k)]

            if multiclass_cols:
                # Multiclass: columns like feat_shap_class0, feat_shap_class1, ...
                # Group by feature name and approx dimension
                feat_dim_map: Dict[str, Dict[int, str]] = {}
                for col in multiclass_cols:
                    m = re.match(r'(.+)_shap_class(\d+)$', col)
                    if m:
                        feat_name = m.group(1)
                        dim_idx = int(m.group(2))
                        if feat_name not in feat_dim_map:
                            feat_dim_map[feat_name] = {}
                        feat_dim_map[feat_name][dim_idx] = col

                feat_names = list(feat_dim_map.keys())
                n_dims = max(max(dims.keys()) for dims in feat_dim_map.values()) + 1
                n_samples = len(next(iter(columns.values())))

                shap_3d = np.zeros((n_samples, len(feat_names), n_dims))
                for fi, fname in enumerate(feat_names):
                    for k in range(n_dims):
                        col_key = feat_dim_map[fname].get(k)
                        if col_key and col_key in columns:
                            shap_3d[:, fi, k] = np.array(columns[col_key])

                # Expected value is per-dimension
                ev = np.array([columns.get(f"expected_value_class{k}", [0.0])[0] for k in range(n_dims)])

                return {
                    "shap_values": shap_3d,
                    "expected_value": ev,
                    "feature_names": feat_names,
                }
            else:
                # Single-output: columns like feat_shap
                shap_cols = [k for k in columns if k.endswith("_shap")]
                expected_value = columns.get("expected_value", [0.0])[0]
                feat_names = [k.replace("_shap", "") for k in shap_cols]
                shap_matrix = np.column_stack([np.array(columns[c]) for c in shap_cols])

                return {
                    "shap_values": shap_matrix,
                    "expected_value": expected_value,
                    "feature_names": feat_names,
                }
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def cross_validate(
        self,
        X,
        y,
        n_folds: int = 5,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Union[List[float], float]]:
        """Run N-fold cross-validation using the C++ binary's built-in CV mode.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target values.
        n_folds : int
            Number of cross-validation folds (default: 5).
        feature_names : list of str, optional
            Names for each feature column.

        Returns
        -------
        dict
            fold_metrics : list of float
                Per-fold loss values in fold order. The metric type matches
                the model's loss function (e.g. RMSE for regression).
            mean : float
                Arithmetic mean of fold_metrics.
            std : float
                Standard deviation of fold_metrics.
        """
        self._validate_params()
        if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
            raise ValueError(
                f"n_folds must be an integer >= 2, got {n_folds!r}"
            )
        X = _to_numpy(X)
        y = _to_numpy(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if n_folds > X.shape[0]:
            raise ValueError(
                f"n_folds ({n_folds}) cannot exceed number of samples ({X.shape[0]})"
            )
        self._validate_fit_inputs(X, y)

        tmpdir = tempfile.mkdtemp(prefix="catboost_mlx_cv_")
        try:
            if not self.cat_features:
                csv_path = os.path.join(tmpdir, "data.cbmx")
                _array_to_binary(csv_path, X, y)
                target_col = -1
            else:
                csv_path = os.path.join(tmpdir, "data.csv")
                target_col, _, _ = _array_to_csv(csv_path, X, y, feature_names,
                                                 self.cat_features)

            args = self._build_train_args(csv_path, "", target_col,
                                          cv_folds=n_folds)

            cv_timeout = (
                self.train_timeout * n_folds if self.train_timeout is not None else None
            )
            try:
                result = subprocess.run(
                    args, capture_output=True, text=True, timeout=cv_timeout
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"csv_train CV timed out after {cv_timeout}s. "
                    "Increase train_timeout or reduce dataset size/iterations/folds."
                ) from e
            if result.returncode != 0:
                raise RuntimeError(
                    f"csv_train CV failed (exit code {result.returncode}):\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            if self.verbose:
                logger.info(result.stdout)

            # Parse CV output
            fold_metrics = []
            mean_val = None
            std_val = None
            for line in result.stdout.split("\n"):
                # "Fold 1: train_loss=0.251  test_loss=0.200  trees=10"
                m = re.search(r"Fold\s+\d+.*?test_loss=([\-\d.]+)", line)
                if m:
                    fold_metrics.append(float(m.group(1)))
                # "Test  loss: 0.240926 +/- 0.028575"
                m = re.search(r"(?:Test\s+loss|CV result).*?(?::\s*|=\s*)([\-\d.]+)\s*\+/-\s*([\-\d.]+)", line)
                if m:
                    mean_val = float(m.group(1))
                    std_val = float(m.group(2))

            if mean_val is None and fold_metrics:
                mean_val = np.mean(fold_metrics)
                std_val = np.std(fold_metrics)

            return {
                "fold_metrics": fold_metrics,
                "mean": mean_val if mean_val is not None else 0.0,
                "std": std_val if std_val is not None else 0.0,
            }
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @property
    def train_loss_history(self) -> List[float]:
        """Training loss at each iteration."""
        return list(self._train_loss_history)

    @property
    def eval_loss_history(self) -> List[float]:
        """Validation loss at each iteration (empty if no validation split)."""
        return list(self._eval_loss_history)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"CatBoostMLX(loss='{self.loss}', iterations={self.iterations}, "
            f"depth={self.depth}, lr={self.learning_rate}, [{status}])"
        )

    def __getstate__(self):
        """Return state for pickling. Excludes _model_path and _model_json_cache."""
        state = self.__dict__.copy()
        state.pop("_model_path", None)
        state.pop("_model_json_cache", None)
        return state

    def __setstate__(self, state):
        """Restore state from pickle."""
        self.__dict__.update(state)
        self._model_path = None
        self._model_json_cache = None


# ── Subclasses ───────────────────────────────────────────────────────────────
# These override the default loss and add sklearn mixin behavior. They delegate
# all parameter handling to CatBoostMLX to avoid parameter duplication issues
# with sklearn's clone() and get_params().

class CatBoostMLXRegressor(RegressorMixin, CatBoostMLX):
    """CatBoostMLX with default loss='rmse' for regression tasks."""

    _estimator_type = "regressor"

    def __init__(self, loss: str = "rmse", **kwargs):
        super().__init__(loss=loss, **kwargs)

    @classmethod
    def _get_param_names(cls):
        """Return parameter names from CatBoostMLX.__init__ (not subclass).

        Without this override, sklearn would inspect CatBoostMLXRegressor.__init__
        and only see 'loss' (since the rest are in **kwargs). By delegating to
        CatBoostMLX, sklearn sees all 27 parameters for clone/get_params/set_params.
        """
        return CatBoostMLX._get_param_names()

    def get_params(self, deep=True):
        return CatBoostMLX.get_params(self, deep=deep)


class CatBoostMLXClassifier(ClassifierMixin, CatBoostMLX):
    """CatBoostMLX with default loss='auto' for classification tasks.

    Auto-detection selects logloss for binary and multiclass for multi-class targets.
    """

    _estimator_type = "classifier"

    def __init__(self, loss: str = "auto", **kwargs):
        super().__init__(loss=loss, **kwargs)

    def fit(self, X, y=None, **kwargs):
        from .pool import Pool
        # Extract y from Pool before super().fit() consumes it, so classes_
        # is computed from the actual labels (not None when Pool is passed).
        if isinstance(X, Pool) and y is None:
            y_for_classes = X.y
        else:
            y_for_classes = y
        result = super().fit(X, y, **kwargs)
        if y_for_classes is not None:
            self.classes_ = np.unique(np.asarray(y_for_classes))
        return result

    def load_model(self, path: str) -> "CatBoostMLXClassifier":
        """Load model and restore classifier-specific state."""
        super().load_model(path)
        info = self._model_data.get("model_info", {})
        num_classes = info.get("num_classes", 0)
        approx_dim = info.get("approx_dimension", 1)
        if num_classes > 0:
            self.classes_ = np.arange(num_classes, dtype=int)
        elif approx_dim > 1:
            # Multiclass: approx_dimension = number of output dimensions
            self.classes_ = np.arange(approx_dim + 1, dtype=int)
        elif self.loss and self.loss.startswith("logloss"):
            # Binary: self.loss is already lowercased by base load_model
            self.classes_ = np.array([0, 1], dtype=int)
        return self

    @classmethod
    def _get_param_names(cls):
        """Return parameter names from CatBoostMLX.__init__ (not subclass)."""
        return CatBoostMLX._get_param_names()

    def get_params(self, deep=True):
        return CatBoostMLX.get_params(self, deep=deep)
