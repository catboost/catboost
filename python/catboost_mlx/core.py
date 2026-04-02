"""Core CatBoostMLX classes wrapping the csv_train/csv_predict CLI binaries."""

import csv
import json
import os
import shutil
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


def _find_binary(name: str, binary_path: Optional[str] = None) -> str:
    """Locate a compiled binary (csv_train or csv_predict)."""
    if binary_path:
        p = Path(binary_path)
        if p.is_dir():
            candidate = p / name
        else:
            candidate = p
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"Binary not found at {candidate}")

    # Check PATH
    found = shutil.which(name)
    if found:
        return found

    # Check current directory
    local = Path(name)
    if local.is_file():
        return str(local.resolve())

    # Check package directory (if binaries are co-located)
    pkg_dir = Path(__file__).parent
    for search_dir in [pkg_dir, pkg_dir.parent, pkg_dir.parent.parent]:
        candidate = search_dir / name
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"Cannot find '{name}' binary. Either:\n"
        f"  1. Add it to your PATH\n"
        f"  2. Place it in the current directory\n"
        f"  3. Pass binary_path='<directory containing {name}>'"
    )


def _array_to_csv(path: str, X: np.ndarray, y: Optional[np.ndarray] = None,
                   feature_names: Optional[List[str]] = None,
                   cat_features: Optional[List[int]] = None,
                   group_id: Optional[np.ndarray] = None,
                   sample_weight: Optional[np.ndarray] = None) -> tuple:
    """Write numpy arrays to CSV. Returns (target_col_index, group_col_index, weight_col_index).
    target_col is -1 if no y, group_col is -1 if no group_id, weight_col is -1 if no sample_weight."""
    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    group_col_idx = -1
    weight_col_idx = -1

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
            for i in range(n_samples):
                row = []
                if group_id is not None:
                    row.append(str(group_id[i]))
                if sample_weight is not None:
                    row.append(f"{sample_weight[i]:.10g}")
                for j in range(n_features):
                    val = X[i, j]
                    if cat_features and j in cat_features:
                        row.append(str(val) if not (isinstance(val, float) and np.isnan(val)) else "")
                    else:
                        row.append(f"{val:.10g}" if not np.isnan(val) else "")
                row.append(f"{y[i]:.10g}")
                writer.writerow(row)
            return target_col_idx, group_col_idx, weight_col_idx
        else:
            writer.writerow(feature_names)
            for i in range(n_samples):
                row = []
                for j in range(n_features):
                    val = X[i, j]
                    if cat_features and j in cat_features:
                        row.append(str(val) if not (isinstance(val, float) and np.isnan(val)) else "")
                    else:
                        row.append(f"{val:.10g}" if not np.isnan(val) else "")
                writer.writerow(row)
            return -1, -1, -1


def _to_numpy(data) -> np.ndarray:
    """Convert input data to numpy array, handling pandas DataFrames/Series."""
    if isinstance(data, np.ndarray):
        return data
    try:
        import pandas as pd
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
    except ImportError:
        pass
    return np.asarray(data)


class CatBoostMLX:
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
    verbose : bool
        Print per-iteration training loss.
    binary_path : str, optional
        Path to directory containing csv_train/csv_predict, or path to csv_train directly.
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
        monotone_constraints: Optional[List[int]] = None,
        snapshot_path: Optional[str] = None,
        snapshot_interval: int = 1,
        verbose: bool = False,
        binary_path: Optional[str] = None,
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
        self.monotone_constraints = monotone_constraints
        self.snapshot_path = snapshot_path
        self.snapshot_interval = snapshot_interval
        self.verbose = verbose
        self.binary_path = binary_path

        # Set after fit()
        self._model_path: Optional[str] = None
        self._model_data: Optional[dict] = None
        self._feature_importance: Optional[Dict[str, float]] = None
        self._train_loss_history: List[float] = []
        self._eval_loss_history: List[float] = []
        self._is_fitted = False

    def _get_loss_type(self) -> str:
        """Extract loss type from model data."""
        if self._model_data:
            info = self._model_data.get("model_info", {})
            lt = info.get("loss_type", "")
            if lt:
                return lt.split(":")[0].lower()
        return self.loss.split(":")[0].lower()

    def _build_train_args(self, csv_path: str, model_path: str, target_col: int) -> List[str]:
        """Build CLI arguments for csv_train."""
        binary = _find_binary("csv_train", self.binary_path)
        args = [
            binary, csv_path,
            "--iterations", str(self.iterations),
            "--depth", str(self.depth),
            "--lr", str(self.learning_rate),
            "--l2", str(self.l2_reg_lambda),
            "--loss", self.loss,
            "--bins", str(self.bins),
            "--target-col", str(target_col),
            "--seed", str(self.random_seed),
            "--nan-mode", self.nan_mode,
            "--output", model_path,
            "--feature-importance",
            "--verbose",
        ]
        if self.cat_features:
            args.extend(["--cat-features", ",".join(str(c) for c in self.cat_features)])
        if self.eval_fraction > 0:
            args.extend(["--eval-fraction", str(self.eval_fraction)])
        if self.early_stopping_rounds > 0:
            args.extend(["--early-stopping", str(self.early_stopping_rounds)])
        if self.subsample < 1.0:
            args.extend(["--subsample", str(self.subsample)])
        if self.colsample_bytree < 1.0:
            args.extend(["--colsample-bytree", str(self.colsample_bytree)])
        if self.ctr:
            args.append("--ctr")
            args.extend(["--ctr-prior", str(self.ctr_prior)])
            args.extend(["--max-onehot-size", str(self.max_onehot_size)])
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
        return args

    def _parse_train_output(self, stdout: str) -> None:
        """Parse csv_train stdout for loss history and feature importance."""
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

    def fit(self, X, y, eval_set=None, feature_names: Optional[List[str]] = None,
            group_id=None, sample_weight=None) -> "CatBoostMLX":
        """Train a model on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        eval_set : tuple of (X_eval, y_eval), optional
            Ignored (use eval_fraction instead for now).
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
        X = _to_numpy(X)
        y = _to_numpy(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        tmpdir = tempfile.mkdtemp(prefix="catboost_mlx_")
        csv_path = os.path.join(tmpdir, "train.csv")
        model_path = os.path.join(tmpdir, "model.json")

        # Convert group_id to numpy if provided
        gid = None
        if group_id is not None:
            gid = np.asarray(group_id)

        # Convert sample_weight to numpy if provided
        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)

        target_col, group_col_idx, weight_col_idx = _array_to_csv(
            csv_path, X, y, feature_names, self.cat_features, group_id=gid,
            sample_weight=sw
        )

        # Temporarily override group_col and weight_col if provided
        orig_group_col = self.group_col
        if gid is not None and self.group_col < 0:
            self.group_col = group_col_idx
        args = self._build_train_args(csv_path, model_path, target_col)
        self.group_col = orig_group_col

        # Add weight-col if sample_weight was provided
        if weight_col_idx >= 0:
            args.extend(["--weight-col", str(weight_col_idx)])

        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"csv_train failed (exit code {result.returncode}):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        self._parse_train_output(result.stdout)

        if self.verbose:
            print(result.stdout)

        # Load model JSON
        with open(model_path, "r") as f:
            self._model_data = json.load(f)
        self._model_path = model_path
        self._is_fitted = True

        return self

    def predict(self, X, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Predict raw values / class labels.

        For regression: returns predicted values.
        For binary classification: returns predicted class (0 or 1).
        For multiclass: returns predicted class index.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        output = self._run_predict(X, feature_names)
        loss_type = self._get_loss_type()

        if loss_type == "logloss":
            return output["predicted_class"].astype(int)
        elif loss_type == "multiclass":
            return output["predicted_class"].astype(int)
        else:
            return output["prediction"]

    def predict_proba(self, X, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Predict class probabilities (classification only).

        For binary classification: returns array of shape (n_samples, 2).
        For multiclass: returns array of shape (n_samples, n_classes).
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

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

    def _run_predict(self, X, feature_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Run csv_predict and parse its output CSV."""
        X = _to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        tmpdir = tempfile.mkdtemp(prefix="catboost_mlx_pred_")
        csv_path = os.path.join(tmpdir, "data.csv")
        out_path = os.path.join(tmpdir, "predictions.csv")
        model_path = os.path.join(tmpdir, "model.json")

        # Write model JSON
        with open(model_path, "w") as f:
            json.dump(self._model_data, f)

        _array_to_csv(csv_path, X, feature_names=feature_names, cat_features=self.cat_features)

        binary = _find_binary("csv_predict", self.binary_path)
        args = [binary, model_path, csv_path, "--output", out_path]

        result = subprocess.run(args, capture_output=True, text=True)
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

    def save_model(self, path: str) -> None:
        """Save the trained model to a JSON file."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        with open(path, "w") as f:
            json.dump(self._model_data, f, indent=2)

    def load_model(self, path: str) -> "CatBoostMLX":
        """Load a model from a JSON file."""
        with open(path, "r") as f:
            self._model_data = json.load(f)
        self._model_path = path
        self._is_fitted = True
        return self

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

        X = _to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        tmpdir = tempfile.mkdtemp(prefix="catboost_mlx_shap_")
        csv_path = os.path.join(tmpdir, "data.csv")
        out_path = os.path.join(tmpdir, "predictions.csv")
        model_path = os.path.join(tmpdir, "model.json")

        with open(model_path, "w") as f:
            json.dump(self._model_data, f)

        _array_to_csv(csv_path, X, feature_names=feature_names, cat_features=self.cat_features)

        binary = _find_binary("csv_predict", self.binary_path)
        args = [binary, model_path, csv_path, "--output", out_path, "--shap"]

        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"csv_predict --shap failed (exit code {result.returncode}):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Parse the SHAP output CSV (_shap.csv)
        shap_path = out_path.replace(".csv", "_shap.csv")
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

    def cross_validate(
        self,
        X,
        y,
        n_folds: int = 5,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Union[List[float], float]]:
        """Run N-fold cross-validation.

        Returns
        -------
        dict with keys:
            fold_metrics: list of per-fold metric values
            mean: mean metric across folds
            std: standard deviation across folds
        """
        X = _to_numpy(X)
        y = _to_numpy(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        tmpdir = tempfile.mkdtemp(prefix="catboost_mlx_cv_")
        csv_path = os.path.join(tmpdir, "data.csv")
        target_col, _, _ = _array_to_csv(csv_path, X, y, feature_names, self.cat_features)

        binary = _find_binary("csv_train", self.binary_path)
        args = [
            binary, csv_path,
            "--iterations", str(self.iterations),
            "--depth", str(self.depth),
            "--lr", str(self.learning_rate),
            "--l2", str(self.l2_reg_lambda),
            "--loss", self.loss,
            "--bins", str(self.bins),
            "--target-col", str(target_col),
            "--seed", str(self.random_seed),
            "--nan-mode", self.nan_mode,
            "--cv", str(n_folds),
        ]
        if self.cat_features:
            args.extend(["--cat-features", ",".join(str(c) for c in self.cat_features)])
        if self.subsample < 1.0:
            args.extend(["--subsample", str(self.subsample)])
        if self.colsample_bytree < 1.0:
            args.extend(["--colsample-bytree", str(self.colsample_bytree)])
        if self.ctr:
            args.append("--ctr")
            args.extend(["--ctr-prior", str(self.ctr_prior)])
            args.extend(["--max-onehot-size", str(self.max_onehot_size)])

        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"csv_train CV failed (exit code {result.returncode}):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        if self.verbose:
            print(result.stdout)

        # Parse CV output
        fold_metrics = []
        mean_val = None
        std_val = None
        for line in result.stdout.split("\n"):
            # "Fold 1/5: RMSE=1.234"
            m = re.search(r"Fold\s+\d+/\d+:\s+\w+=([\-\d.]+)", line)
            if m:
                fold_metrics.append(float(m.group(1)))
            # "CV result: RMSE = 1.234 +/- 0.123"
            m = re.search(r"CV result:.*?=\s*([\-\d.]+)\s*\+/-\s*([\-\d.]+)", line)
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


class CatBoostMLXRegressor(CatBoostMLX):
    """CatBoostMLX with default loss='rmse' for regression tasks."""

    def __init__(self, loss: str = "rmse", **kwargs):
        super().__init__(loss=loss, **kwargs)


class CatBoostMLXClassifier(CatBoostMLX):
    """CatBoostMLX with default loss='auto' for classification tasks.

    Auto-detection selects logloss for binary and multiclass for multi-class targets.
    """

    def __init__(self, loss: str = "auto", **kwargs):
        super().__init__(loss=loss, **kwargs)
