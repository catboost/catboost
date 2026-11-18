"""Vector objective branches for the existing Metal regressor/classifier."""
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from ._data import unpack_pool, prepare_features, sample_weights, numeric_array


OBJECTIVES = ("MultiRMSE", "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy")


def _targets(y, rows, objective, dimensions=None):
    targets = numeric_array(y, "y", 1 if objective == "RMSEWithUncertainty" else 2)
    if targets.shape[0] != rows:
        raise ValueError("X must have exactly one target row per feature row.")
    actual = 2 if objective == "RMSEWithUncertainty" else targets.shape[1]
    if not 2 <= actual <= 64 or dimensions is not None and actual != dimensions:
        raise ValueError("Vector targets require 2 to 64 outputs matching the training dimensions.")
    if objective in ("MultiLogloss", "MultiCrossEntropy"):
        if (targets < 0).any() or (targets > 1).any():
            raise ValueError("Multilabel targets must be in [0,1].")
        if objective == "MultiLogloss" and ((targets != 0) & (targets != 1)).any():
            raise ValueError("MultiLogloss targets must be binary.")
    return targets, actual


def fit_multioutput(estimator, X, y=None, sample_weight=None, *, eval_set=None, cat_features=None,
                    early_stopping_rounds=None, use_best_model=None, save_snapshot=False,
                    snapshot_file=None, snapshot_interval=600., resume=True, callback=None):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    from .regressor import _integer, _model_json, _training_metadata
    from . import _multioutput

    started = time.perf_counter()
    estimator._model = None
    objective = estimator._objective
    if objective not in OBJECTIVES:
        raise ValueError("Unsupported multioutput objective.")
    if estimator.boosting_type != "Plain" or estimator.grow_policy != "SymmetricTree":
        raise ValueError("Vector Metal objectives currently require Plain symmetric trees.")
    if estimator.score_function not in ("L2", "Cosine", "SolarL2", "LOOL2", "SatL2"):
        raise ValueError("Vector Metal structure search supports L2, Cosine, SolarL2, LOOL2 or SatL2 scoring.")
    if estimator.class_weights is not None:
        raise ValueError("class_weights is not supported for these vector objectives; use row sample_weight.")
    cats = estimator.cat_features if cat_features is None else cat_features
    raw, y, sample_weight, cats, names = unpack_pool(X, y, sample_weight, cats)
    if cats:
        raise ValueError("The standalone vector objective frontend currently supports numeric features.")
    targets, dimensions = _targets(y, raw.shape[0], objective)
    weights = sample_weights(sample_weight, len(targets))
    layout, bins, cf, cb, ct = prepare_features(raw, estimator.border_count,
        estimator.nan_mode, [], names, estimator.one_hot_max_size, targets=targets,
        objective=objective, random_seed=estimator.random_seed)
    bias = np.zeros(dimensions, np.float32)
    if estimator.boost_from_average:
        if objective != "MultiRMSE":
            raise ValueError(f"CatBoost does not support boost_from_average for {objective}.")
        w = np.ones(len(targets), np.float64) if weights is None else weights.astype(np.float64)
        bias = np.asarray((targets.astype(np.float64) * w[:, None]).sum(axis=0) / w.sum(), np.float32)
        if not np.isfinite(bias).all():
            raise ValueError("Vector target mean must be finite in float32.")
    eval_bins = eval_targets = eval_weight = None
    if eval_set is not None:
        if isinstance(eval_set, Pool):
            eval_raw, eval_y, eval_weight, _, _ = unpack_pool(eval_set)
        elif isinstance(eval_set, tuple) and len(eval_set) in (2, 3):
            eval_raw, eval_y = eval_set[:2]
            eval_weight = eval_set[2] if len(eval_set) == 3 else None
        else:
            raise ValueError("eval_set must be one Pool or an (X, y[, weight]) tuple.")
        eval_bins = layout.transform(eval_raw)
        if not eval_bins.shape[1]:
            raise ValueError("eval_set must contain rows.")
        eval_targets, _ = _targets(eval_y, eval_bins.shape[1], objective, dimensions)
        eval_weight = sample_weights(eval_weight, len(eval_targets), "eval_weight")
    if early_stopping_rounds is not None:
        _integer(early_stopping_rounds, "early_stopping_rounds", 1, 100000)
        if eval_set is None:
            raise ValueError("early_stopping_rounds requires eval_set.")
    if use_best_model is not None and not isinstance(use_best_model, bool):
        raise ValueError("use_best_model must be boolean.")
    if use_best_model and eval_set is None:
        raise ValueError("use_best_model requires eval_set.")
    options = dict(iterations=estimator.iterations, depth=estimator.depth,
        learning_rate=estimator.learning_rate, l2_leaf_reg=estimator.l2_leaf_reg,
        bias=bias, score_function=estimator.score_function, objective=objective,
        classes=dimensions, sample_weight=weights, candidate_types=ct,
        leaf_estimation_iterations=estimator.leaf_estimation_iterations,
        leaf_estimation_method=estimator.leaf_estimation_method,
        leaf_estimation_backtracking=estimator.leaf_estimation_backtracking,
        bootstrap_type=estimator.bootstrap_type, bagging_temperature=estimator.bagging_temperature,
        subsample=estimator.subsample, random_seed=estimator.random_seed,
        random_strength=estimator.random_strength, mvs_reg=estimator.mvs_reg)
    train_started = time.perf_counter()
    if (eval_set is not None or save_snapshot or snapshot_file is not None or callback is not None
            or estimator.eval_metric is not None):
        from ._training import run_training
        result = run_training(bins, targets, cf, cb, **options,
            eval_bins=eval_bins, eval_targets=eval_targets, eval_weight=eval_weight,
            early_stopping_rounds=early_stopping_rounds, use_best_model=use_best_model,
            save_snapshot=save_snapshot, snapshot_file=snapshot_file,
            snapshot_interval=snapshot_interval, resume=resume, callback=callback,
            **({"eval_metric": estimator.eval_metric} if estimator.eval_metric is not None else {}),
            metadata={"borders": [border.tolist() for border in layout.borders],
                "has_nans": layout.has_nans, "nan_mode": layout.nan_mode,
                "feature_names": names, "output_dimensions": dimensions, "objective": objective})
    else:
        result = _multioutput.train(bins, targets, cf, cb, **options)
    result.stats["native_wall_seconds"] = time.perf_counter() - train_started
    result.stats["preprocessing_seconds"] = train_started - started
    result.stats["ctr_gpu_seconds"] = 0.
    result.stats["ctr_kernel_dispatches"] = 0
    model_data = _model_json(layout.borders, result, bias, estimator.score_function, layout)
    bootstrap = {"type": estimator.bootstrap_type}
    if estimator.bootstrap_type == "Bayesian":
        bootstrap["bagging_temperature"] = estimator.bagging_temperature
    elif estimator.bootstrap_type != "No":
        bootstrap["subsample"] = estimator.subsample
    model_data["model_info"].update(_training_metadata(
        iterations=estimator.iterations, depth=estimator.depth, learning_rate=estimator.learning_rate,
        l2_leaf_reg=estimator.l2_leaf_reg, border_count=estimator.border_count,
        score_function=estimator.score_function, objective=objective,
        boost_from_average=estimator.boost_from_average,
        leaf_estimation_iterations=estimator.leaf_estimation_iterations,
        leaf_estimation_method=estimator.leaf_estimation_method,
        leaf_estimation_backtracking=estimator.leaf_estimation_backtracking,
        nan_mode=estimator.nan_mode, use_best_model=eval_set is not None if use_best_model is None else use_best_model,
        eval_metric=estimator.eval_metric, loss_description=estimator.loss_function,
        bootstrap=bootstrap, random_strength=estimator.random_strength,
        random_seed=estimator.random_seed, permutation_count=1))
    with tempfile.TemporaryDirectory(prefix="catbooster-metal-vector-") as directory:
        path = Path(directory) / "model.json"
        path.write_text(json.dumps(model_data, allow_nan=False))
        model_type = CatBoostClassifier if estimator._classifier else CatBoostRegressor
        model = model_type().load_model(str(path), format="json")
    estimator._model, estimator._result, estimator._layout = model, result, layout
    estimator.borders_, estimator.bias_ = layout.borders, bias
    estimator.n_features_in_, estimator.feature_names_ = raw.shape[1], names
    estimator.n_outputs_ = dimensions
    if estimator._classifier:
        estimator.classes_ = model.classes_.copy()
    estimator.tree_count_, estimator.tree_depths_ = len(result.depths), result.depths.copy()
    estimator.training_predictions_ = result.predictions.copy()
    estimator.loss_history_, estimator.training_stats_ = result.loss.tolist(), result.stats.copy()
    estimator.training_stats_["fit_wall_seconds"] = time.perf_counter() - started
    estimator.best_iteration_ = getattr(result, "best_iteration", -1)
    estimator.best_score_ = getattr(result, "best_score", {"learn": {estimator.loss_function: min(estimator.loss_history_[1:])}})
    estimator.evals_result_ = getattr(result, "evals_result", {"learn": {estimator.loss_function: estimator.loss_history_[1:]}})
    return estimator


def predict_multioutput(estimator, X, prediction_type=None, *, task_type="CPU", ntree_start=0, ntree_end=0):
    estimator._require_fitted()
    objective = estimator._objective
    default = "Class" if estimator._classifier else "RMSEWithUncertainty" if objective == "RMSEWithUncertainty" else "RawFormulaVal"
    prediction_type = prediction_type or default
    if task_type not in ("CPU", "METAL", "GPU"):
        raise ValueError("Prediction task_type must be CPU, METAL, or GPU.")
    allowed = {"RawFormulaVal", "Exponent", "Probability", "LogProbability"}
    if estimator._classifier:
        allowed.add("Class")
    if objective == "RMSEWithUncertainty":
        allowed.add("RMSEWithUncertainty")
    if prediction_type not in allowed:
        raise ValueError("Unsupported vector prediction_type.")
    bins = estimator._layout.transform(X)
    if task_type == "CPU" and bins.shape[1]:
        return estimator._model.predict(X, prediction_type=prediction_type,
                                       ntree_start=ntree_start, ntree_end=ntree_end)
    result = estimator._result
    if estimator.grow_policy != "SymmetricTree":
        from ._greedy_inference import predict_bins
        raw = predict_bins(bins, result.trees, estimator.bias_, tree_start=ntree_start,
                           tree_end=ntree_end or None)
    else:
        from ._inference import predict_bins
        raw = np.column_stack([predict_bins(bins, result.depths, result.split_features, result.split_bins,
            result.leaf_values[:, :, dimension], split_types=result.split_types,
            bias=estimator.bias_[dimension], tree_start=ntree_start, tree_end=ntree_end or None)
            for dimension in range(estimator.n_outputs_)])
    if prediction_type == "RawFormulaVal":
        return raw
    if prediction_type == "RMSEWithUncertainty":
        raw[:, 1] = np.exp(2 * raw[:, 1])
        return raw
    if prediction_type == "Exponent":
        # Standard CatBoost's Exponent prediction transforms the first output.
        return np.exp(raw[:, 0]).reshape(-1, 1)
    if prediction_type == "Class":
        return (raw > 0).astype(np.int64)
    if objective in ("MultiLogloss", "MultiCrossEntropy"):
        log_probability = -np.logaddexp(0, -raw)
    else:
        shifted = raw - raw.max(axis=1, keepdims=True)
        log_probability = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return log_probability if prediction_type == "LogProbability" else np.exp(log_probability)
