"""CatBoost model/data bridge for the translated Metal training backend."""

import json
import hashlib
import numbers
from pathlib import Path
import tempfile
import time

import numpy as np

from . import _native
from ._options import parse_loss
from ._data import (numeric_array as _numeric_array, sample_weights, quantize_features,
                    unpack_pool, prepare_features)

_MULTIOUTPUT = {"MultiRMSE", "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy"}


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
    return int(value)


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number.")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.float32(value)
    if not np.isfinite(converted) or converted < 0 or (positive and converted == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'} in float32.")
    return float(converted)


def _model_json(borders, result, bias, score_function, layout=None):
    """Export numeric, one-hot, then OnlineCtr binary split indices."""
    categorical = layout.categorical if layout else {}
    ctrs = layout.ctrs if layout else {}
    raw_count = len(layout.names) if layout else len(borders)
    numeric_indices = {f: i for i, f in enumerate(f for f in range(raw_count) if f not in categorical)}
    cat_indices = {f: i for i, f in enumerate(sorted(categorical))}
    offsets, offset = {}, 0
    for feature in numeric_indices:
        offsets[feature] = offset
        offset += len(borders[feature])
    cat_offsets = {}
    for feature in sorted(categorical):
        cat_offsets[feature] = offset
        offset += len(categorical[feature].candidate_bins)
    ctr_offsets = {}
    for feature in ctrs:
        ctr_offsets[feature] = offset
        offset += len(borders[feature])
    trees = []
    for tree, actual_depth in enumerate(result.depths):
        splits = []
        for level in range(int(actual_depth)):
            feature = int(result.split_features[tree, level])
            border = int(result.split_bins[tree, level])
            if feature in ctrs:
                from ._ctr_model import online_ctr_split_json
                split = online_ctr_split_json(float(borders[feature][border]), ctr_offsets[feature] + border)
            elif feature in categorical:
                from ._categorical import one_hot_split_json
                split = one_hot_split_json(cat_indices[feature], border, categorical[feature],
                                           cat_offsets[feature] + border)
            else:
                split = {"split_type": "FloatFeature", "float_feature_index": numeric_indices[feature],
                         "border": float(borders[feature][border]), "split_index": int(offsets[feature] + border)}
            splits.append(split)
        count = 1 << int(actual_depth)
        trees.append({"splits": splits, "leaf_values": result.leaf_values[tree, :count].reshape(-1).tolist(),
                      "leaf_weights": result.leaf_weights[tree, :count].tolist()})
    float_features = []
    for feature, index in numeric_indices.items():
        has_nans = bool(layout.has_nans[feature]) if layout else False
        treatment = ("AsTrue" if layout.nan_mode == "Max" else "AsFalse") if has_nans else "AsIs"
        float_features.append({"feature_index": index, "flat_feature_index": feature,
                               "feature_id": layout.names[feature] if layout else str(feature),
                               "has_nans": has_nans, "nan_value_treatment": treatment,
                               "borders": borders[feature].tolist()})
    info = {"float_features": float_features}
    if categorical:
        from ._categorical import categorical_feature_json
        info["categorical_features"] = [categorical_feature_json(
            cat_indices[f], f, categorical[f], layout.names[f]) for f in sorted(categorical)]
    ctr_data = {}
    if ctrs:
        from ._ctr_model import ctr_feature_json, ctr_table_json
        info["ctrs"] = []
        for feature, ctr in ctrs.items():
            stats = ctr.result
            descriptor = ctr_feature_json(cat_indices[ctr.source_feature], stats.ctr_type, borders[feature],
                                          prior_numerator=stats.prior_numerator,
                                          prior_denominator=stats.prior_denominator)
            info["ctrs"].append(descriptor)
            ctr_data[descriptor["identifier"]] = ctr_table_json(
                stats.hashes, stats.counts, ctr_type=stats.ctr_type,
                **({} if stats.ctr_type == "FeatureFreq" else {"sums": stats.sums}))
    model = {"features_info": info, "oblivious_trees": trees,
             "scale_and_bias": [1.0, np.asarray(bias, dtype=np.float64).reshape(-1).tolist()],
             "model_info": {"metal_port": "CUDA algorithms translated to Apple Metal",
                            "metal_score_function": score_function, "metal_device": result.stats["device"]}}
    if ctr_data:
        model["ctr_data"] = ctr_data
        count = len(getattr(layout, "permutation_bins", (None,)))
        model["model_info"]["metal_ctr_scope"] = (
            f"Single categorical projections, {count} permutation(s), learn-only final tables")
    return model


def _training_metadata(*, iterations, depth, learning_rate, l2_leaf_reg, border_count,
                       score_function, objective="RMSE", boost_from_average=True,
                       leaf_estimation_iterations=1, nan_mode="Forbidden", use_best_model=False,
                       classes=None, class_weights=None, categorical=False, one_hot_max_size=255,
                       eval_metric=None, loss_parameters=None, leaf_estimation_method="Newton",
                       loss_description=None, bootstrap=None, random_strength=0, random_seed=0,
                       leaf_estimation_backtracking="No", permutation_count=1, model_size_reg=None,
                       boosting_type="Plain", fold_len_multiplier=2.0, min_fold_size=100,
                       fold_size_loss_normalization=False, grow_policy="SymmetricTree",
                       max_leaves=None, min_data_in_leaf=1, fold_permutation_block=64):
    boosting = {"iterations": iterations, "learning_rate": learning_rate, "boosting_type": boosting_type,
                "data_partition": "FeatureParallel" if boosting_type == "Ordered" else "DocParallel",
                "boost_from_average": boost_from_average,
                "permutation_count": permutation_count}
    if boosting_type == "Ordered":
        boosting.update(fold_len_multiplier=fold_len_multiplier, min_fold_size=min_fold_size,
                        fold_permutation_block=fold_permutation_block)
    tree = {"depth": depth, "l2_leaf_reg": l2_leaf_reg, "score_function": score_function,
            "grow_policy": grow_policy, "random_strength": random_strength, "leaf_estimation_method": leaf_estimation_method,
            "leaf_estimation_iterations": leaf_estimation_iterations,
            "leaf_estimation_backtracking": leaf_estimation_backtracking,
            "fold_size_loss_normalization": fold_size_loss_normalization,
            "add_ridge_penalty_to_loss_function": False}
    if grow_policy != "SymmetricTree":
        tree["min_data_in_leaf"] = min_data_in_leaf
        if max_leaves is not None and grow_policy != "Region":
            tree["max_leaves"] = max_leaves
    if model_size_reg is not None:
        tree["model_size_reg"] = model_size_reg
    quantization = {"border_count": border_count, "border_type": "GreedyLogSum", "nan_mode": nan_mode}
    output = {"use_best_model": bool(use_best_model)}
    bootstrap = bootstrap or {"type": "No"}
    flat = {"task_type": "GPU", "loss_function": loss_description or objective,
            **boosting, **tree, **output, "bootstrap_type": bootstrap["type"],
            "border_count": border_count, "feature_border_type": "GreedyLogSum", "nan_mode": nan_mode}
    processing = {"float_features_binarization": quantization}
    if classes is not None:
        processing["class_names"] = classes.tolist()
        flat["class_names"] = classes.tolist()
    if class_weights is not None:
        processing["class_weights"] = list(class_weights)
        flat["class_weights"] = list(class_weights)
    params = {"task_type": "GPU", "loss_function": {"type": objective, "params": {
                  key: str(value) for key, value in (loss_parameters or {}).items()}},
              "boosting_options": {**boosting, "od_config": {"type": "None"}},
              "tree_learner_options": {**tree, "bootstrap": bootstrap},
              "data_processing_options": processing, "flat_params": flat}
    flat.update({key: value for key, value in bootstrap.items() if key != "type"})
    if random_seed or random_strength or bootstrap["type"] != "No":
        flat["random_seed"] = random_seed
        params["random_seed"] = random_seed
    if eval_metric is not None:
        name, _, arguments = eval_metric.partition(":")
        params["metrics"] = {"eval_metric": {"type": name, "params": dict(
            argument.split("=", 1) for argument in arguments.split(";") if argument)}}
        flat["eval_metric"] = eval_metric
    if categorical:
        params["cat_feature_params"] = {"one_hot_max_size": one_hot_max_size}
        flat["one_hot_max_size"] = one_hot_max_size
    metadata = {"params": json.dumps(params, allow_nan=False),
                "output_options": json.dumps(output), "metal_backend": "METAL"}
    if classes is not None:
        kind = {"b": "Boolean", "i": "Integer", "u": "Integer", "f": "Float"}.get(classes.dtype.kind, "String")
        metadata["class_params"] = json.dumps({"class_label_type": kind, "class_to_label": list(range(len(classes))),
                                                "class_names": classes.tolist(), "classes_count": 0})
    return metadata


class CatBoostMetalRegressor:
    """Metal training with standard CatBoost model serialization and prediction."""

    _classifier = False

    def __init__(self, *, iterations=100, depth=4, learning_rate=0.1, l2_leaf_reg=3.0,
                 border_count=32, score_function=None, loss_function=None,
                 leaf_estimation_iterations=None, leaf_estimation_backtracking="No", leaf_estimation_method=None,
                 boost_from_average=None, nan_mode="Forbidden", cat_features=None,
                 one_hot_max_size=255, class_weights=None, ctr_type="Borders", ctr_prior=0.5,
                 ctr_target_border=None, ctr_border_count=15, ctr_history_unit="Sample", random_seed=0, eval_metric=None,
                 permutation_count=None, model_size_reg=0.5,
                 boosting_type="Plain", data_partition=None, fold_len_multiplier=2.0,
                 min_fold_size=100, fold_size_loss_normalization=False, fold_permutation_block=64,
                 grow_policy="SymmetricTree", max_leaves=None, min_data_in_leaf=1,
                 bootstrap_type="No", bagging_temperature=None, subsample=None, mvs_reg=None,
                 random_strength=0, **options):
        supported = {"task_type": "METAL"}
        for name, value in options.items():
            if name not in supported:
                raise TypeError(f"Unsupported Metal training option: {name}.")
            if value != supported[name]:
                raise ValueError(f"Metal currently requires {name}={supported[name]!r}.")
        if boosting_type not in ("Plain", "Ordered"):
            raise ValueError("boosting_type must be Plain or Ordered.")
        self.boosting_type = boosting_type
        if grow_policy not in ("SymmetricTree", "Depthwise", "Lossguide", "Region"):
            raise ValueError("grow_policy must be SymmetricTree, Depthwise, Lossguide, or Region.")
        if grow_policy != "SymmetricTree" and boosting_type != "Plain":
            raise ValueError("Non-symmetric trees require Plain boosting.")
        self.grow_policy = grow_policy
        self.data_partition = "FeatureParallel" if boosting_type == "Ordered" else "DocParallel"
        if data_partition is not None and data_partition != self.data_partition:
            raise ValueError(f"{boosting_type} requires data_partition={self.data_partition!r}.")
        self.fold_len_multiplier = _number(fold_len_multiplier, "fold_len_multiplier", positive=True)
        if self.fold_len_multiplier <= 1:
            raise ValueError("fold_len_multiplier must exceed 1.")
        self.min_fold_size = _integer(min_fold_size, "min_fold_size", 1, 2**32 - 1)
        self.fold_permutation_block = _integer(fold_permutation_block, "fold_permutation_block", 0, 2**32 - 1)
        if boosting_type != "Ordered" and self.fold_permutation_block != 64:
            raise ValueError("Custom fold_permutation_block is currently connected only for Ordered boosting.")
        if not isinstance(fold_size_loss_normalization, bool):
            raise ValueError("fold_size_loss_normalization must be boolean.")
        if boosting_type == "Plain" and fold_size_loss_normalization:
            raise ValueError("Plain fold_size_loss_normalization is not yet connected.")
        self.fold_size_loss_normalization = fold_size_loss_normalization
        self._default_classification_loss = self._classifier and loss_function is None
        self._default_leaf_iterations = leaf_estimation_iterations is None
        self.loss_function, self._objective, self._loss_parameters, self._objective_param = parse_loss(
            loss_function, classifier=self._classifier)
        _number(self._objective_param, "loss parameter")
        self.iterations = _integer(iterations, "iterations", 1, 10000)
        depth_limit = 2**32 - 1 if grow_policy == "Lossguide" else 65535 if grow_policy == "Region" else 16
        self.depth = _integer(depth, "depth", 0, depth_limit)
        default_leaves = 31 if grow_policy == "Lossguide" else self.depth + 1 if grow_policy == "Region" else 1 << self.depth
        self.max_leaves = _integer(
            default_leaves if max_leaves is None else max_leaves,
            "max_leaves", 1, 65536)
        if grow_policy == "Region":
            legacy_value = 1 << self.depth if self.depth < 32 else None
            if self.max_leaves not in (default_leaves, legacy_value):
                raise ValueError("Region leaf capacity is depth + 1; max_leaves is a Lossguide option.")
            self.max_leaves = default_leaves
        elif grow_policy != "Lossguide" and self.max_leaves != 1 << self.depth:
            raise ValueError("max_leaves can be changed only with Lossguide.")
        self.min_data_in_leaf = _integer(min_data_in_leaf, "min_data_in_leaf", 1, 16777216)
        if grow_policy == "SymmetricTree" and self.min_data_in_leaf != 1:
            raise ValueError("min_data_in_leaf requires a non-symmetric grow_policy.")
        self.border_count = _integer(border_count, "border_count", 1, 255)
        self.learning_rate = _number(learning_rate, "learning_rate", positive=True)
        if self.learning_rate > 1:
            raise ValueError("The Metal port requires learning_rate in (0, 1].")
        self.l2_leaf_reg = _number(l2_leaf_reg, "l2_leaf_reg")
        self._default_score_function = score_function is None
        if score_function is None:
            score_function = ("L2" if self._objective in ("MultiClass", "MultiClassOneVsAll", "RMSEWithUncertainty")
                              else "NewtonL2") if grow_policy == "Lossguide" else "Cosine"
        if score_function not in ("L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2", "SatL2"):
            raise ValueError("score_function must be L2, Cosine, NewtonL2, NewtonCosine, SolarL2, LOOL2, or SatL2.")
        self.score_function = score_function
        if boosting_type == "Ordered" and score_function not in ("Cosine", "NewtonCosine"):
            raise ValueError("Ordered supports Cosine and NewtonCosine scoring.")
        exact_losses = ("MAE", "Quantile", "MAPE")
        gradient_only = self._objective in (*exact_losses, "LogLinQuantile") or (
            self._objective == "Lq" and self._objective_param < 2)
        if leaf_estimation_method is None:
            leaf_estimation_method = "Exact" if self._objective in exact_losses and boosting_type == "Plain" else (
                "Gradient" if gradient_only else "Newton")
        if leaf_estimation_method not in ("Newton", "Gradient", "Exact"):
            raise ValueError("leaf_estimation_method must be Newton, Gradient, or Exact.")
        if leaf_estimation_method == "Exact" and self._objective not in exact_losses:
            raise ValueError("Exact leaf estimation requires MAE, Quantile, or MAPE.")
        if leaf_estimation_method == "Exact" and boosting_type == "Ordered":
            raise ValueError("CUDA does not support Exact leaf estimation with Ordered boosting.")
        if leaf_estimation_method == "Newton" and gradient_only:
            raise ValueError(f"Newton leaf estimation is not supported for {self.loss_function}.")
        self.leaf_estimation_method = leaf_estimation_method
        if leaf_estimation_iterations is None:
            defaults = {"Poisson": (10, 1), "Huber": (1, 1), "Expectile": (5, 10), "Tweedie": (20, 20),
                        "MultiClass": (1, 10), "MultiClassOneVsAll": (1, 10),
                        "MultiLogloss": (10, 40), "MultiCrossEntropy": (10, 40)}
            leaf_estimation_iterations = defaults.get(self._objective, (1, 1))[leaf_estimation_method == "Gradient"]
        self.leaf_estimation_iterations = _integer(leaf_estimation_iterations, "leaf_estimation_iterations", 1, 100)
        if leaf_estimation_backtracking not in ("No", "AnyImprovement", "Armijo"):
            raise ValueError("leaf_estimation_backtracking must be No, AnyImprovement, or Armijo.")
        self.leaf_estimation_backtracking = leaf_estimation_backtracking
        if boost_from_average is not None and not isinstance(boost_from_average, bool):
            raise ValueError("boost_from_average must be boolean.")
        self.boost_from_average = self._objective in ("RMSE", "MultiRMSE", *exact_losses) if boost_from_average is None else boost_from_average
        if self.boost_from_average and self._objective not in ("RMSE", "MultiRMSE", "Logloss", "CrossEntropy", *exact_losses):
            raise ValueError(f"CatBoost does not support boost_from_average for {self._objective}.")
        if nan_mode not in ("Forbidden", "Min", "Max"):
            raise ValueError("nan_mode must be Forbidden, Min, or Max.")
        self.nan_mode = nan_mode
        self.cat_features = cat_features
        self.one_hot_max_size = _integer(one_hot_max_size, "one_hot_max_size", 1, 255)
        if ctr_type not in ("Borders", "FeatureFreq"):
            raise ValueError("The training adapter currently supports Borders and FeatureFreq CTRs.")
        if ctr_history_unit not in ("Sample", "Group"):
            raise ValueError("ctr_history_unit must be Sample or Group.")
        if ctr_history_unit == "Group" and boosting_type != "Ordered":
            raise ValueError("Standalone Group CTR histories currently require Ordered boosting.")
        self.ctr_history_unit = ctr_history_unit
        self.ctr_type = ctr_type
        self.ctr_prior = _number(ctr_prior, "ctr_prior")
        if ctr_target_border is not None and (isinstance(ctr_target_border, bool)
                or not isinstance(ctr_target_border, numbers.Real) or not np.isfinite(ctr_target_border)):
            raise ValueError("ctr_target_border must be a finite number.")
        self.ctr_target_border = float(ctr_target_border) if ctr_target_border is not None else None
        self.ctr_border_count = _integer(ctr_border_count, "ctr_border_count", 1, 255)
        self.random_seed = _integer(random_seed, "random_seed", 0, 2**64 - 1)
        self.permutation_count = (None if permutation_count is None else
                                  _integer(permutation_count, "permutation_count", 1, 64))
        self.model_size_reg = _number(model_size_reg, "model_size_reg")
        if bootstrap_type not in ("No", "Bayesian", "Bernoulli", "Poisson", "MVS"):
            raise ValueError("Unknown bootstrap_type.")
        if bootstrap_type == "No" and (bagging_temperature is not None or subsample is not None or mvs_reg is not None):
            raise ValueError("Bootstrap parameters require an enabled bootstrap_type.")
        if bootstrap_type == "Bayesian" and subsample is not None:
            raise ValueError("Bayesian bootstrap does not support subsample.")
        if bagging_temperature is not None and bootstrap_type != "Bayesian":
            raise ValueError("bagging_temperature requires Bayesian bootstrap.")
        if mvs_reg is not None and bootstrap_type != "MVS":
            raise ValueError("mvs_reg requires MVS bootstrap.")
        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = _number(1.0 if bagging_temperature is None else bagging_temperature, "bagging_temperature")
        self.subsample = _number(0.66 if subsample is None else subsample, "subsample", positive=True)
        if self.subsample > 1 or (bootstrap_type == "Poisson" and self.subsample >= 1):
            raise ValueError("subsample must be in (0, 1], and strictly below 1 for Poisson.")
        self.mvs_reg = None if mvs_reg is None else _number(mvs_reg, "mvs_reg")
        self.random_strength = _number(random_strength, "random_strength")
        if grow_policy != "SymmetricTree":
            if self._objective not in ("RMSE", "Logloss", "CrossEntropy", "Poisson", "Huber",
                    "Expectile", "Tweedie", "LogLinQuantile", "Quantile", "MAE", "MAPE",
                    "MultiClass", "MultiClassOneVsAll", "RMSEWithUncertainty"):
                raise ValueError("This objective is not registered for CUDA non-symmetric trees.")
            if bootstrap_type == "MVS":
                raise ValueError("CUDA non-symmetric trees do not support MVS bootstrap.")
        if eval_metric is not None and (not isinstance(eval_metric, str) or not eval_metric):
            raise ValueError("eval_metric must be a nonempty CatBoost metric description.")
        self.eval_metric = eval_metric
        if eval_metric is not None:
            from ._training import _metric_direction
            _metric_direction(eval_metric)
        if class_weights is not None and self._objective not in ("Logloss", "MultiClass", "MultiClassOneVsAll"):
            raise ValueError("class_weights requires Logloss or multiclass classification.")
        self.class_weights = class_weights
        self._model = None

    def _targets(self, y, *, fitting):
        if not self._classifier:
            return _numeric_array(y, "y", 1)
        if self.loss_function == "CrossEntropy":
            targets = _numeric_array(y, "y", 1)
            if (targets < 0).any() or (targets > 1).any():
                raise ValueError("CrossEntropy targets must be in [0, 1].")
            if fitting:
                self.classes_ = np.array([0, 1])
            return targets
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.dtype.kind not in "biufUSO":
            raise ValueError("Classification labels must be a vector of numbers or strings.")
        if labels.dtype.kind in "f" and not np.isfinite(labels).all():
            raise ValueError("Classification labels must be finite.")
        if labels.dtype.kind == "O":
            if all(isinstance(v, str) for v in labels):
                labels = labels.astype(str)
            elif all(isinstance(v, numbers.Real) and not isinstance(v, complex) for v in labels):
                labels = labels.astype(np.float64)
                if not np.isfinite(labels).all():
                    raise ValueError("Classification labels must be finite.")
            else:
                raise ValueError("Classification labels must consistently be numeric or strings.")
        if fitting:
            self.classes_ = np.unique(labels)
            if self._default_classification_loss:
                self.loss_function = self._objective = "MultiClass" if len(self.classes_) > 2 else "Logloss"
                if self._default_score_function and self.grow_policy == "Lossguide":
                    self.score_function = "L2" if self._objective == "MultiClass" else "NewtonL2"
                if self._default_leaf_iterations:
                    self.leaf_estimation_iterations = (10 if self._objective == "MultiClass"
                        and self.leaf_estimation_method == "Gradient" else 1)
            multiclass = self._objective in ("MultiClass", "MultiClassOneVsAll")
            if multiclass and len(self.classes_) < 2:
                raise ValueError("Multiclass training requires at least two classes.")
            if not multiclass and len(self.classes_) != 2:
                raise ValueError("Binary classification requires exactly two classes.")
        if not np.isin(labels, self.classes_).all():
            raise ValueError("Evaluation labels contain a class absent from training.")
        multiclass = self._objective in ("MultiClass", "MultiClassOneVsAll")
        return np.ascontiguousarray(np.searchsorted(self.classes_, labels) if multiclass else
                                    labels == self.classes_[1], np.float32)

    def _weights(self, value, targets):
        weights = sample_weights(value, len(targets))
        self._effective_class_weights = None
        if self.class_weights is not None:
            if isinstance(self.class_weights, dict):
                if set(self.class_weights) != set(self.classes_.tolist()):
                    raise ValueError("class_weights must specify every training label.")
                multipliers = [self.class_weights[label] for label in self.classes_.tolist()]
            else:
                multipliers = self.class_weights
            multipliers = sample_weights(multipliers, len(self.classes_), "class_weights")
            weights = (np.ones(len(targets), np.float32) if weights is None else weights) * multipliers[targets.astype(np.int32)]
            weights = sample_weights(weights, len(targets))
            self._effective_class_weights = multipliers.tolist()
        return weights

    def fit(self, X, y=None, sample_weight=None, *, eval_set=None, cat_features=None,
            early_stopping_rounds=None, use_best_model=None, save_snapshot=False,
            snapshot_file=None, snapshot_interval=600.0, resume=True, callback=None,
            group_id=None, group_weight=None, eval_group_id=None, eval_group_weight=None):
        self._model = None
        if self.boosting_type != "Ordered" and any(value is not None for value in (
                group_id, group_weight, eval_group_id, eval_group_weight)):
            raise ValueError("Grouped scalar fitting currently requires Ordered boosting.")
        if self._default_classification_loss:
            from catboost import Pool
            labels = X.get_label() if y is None and isinstance(X, Pool) else y
            labels = np.asarray(labels)
            if labels.ndim == 2 and labels.shape[1] > 1:
                try:
                    numeric = labels.astype(np.float64)
                except (ValueError, TypeError) as error:
                    raise ValueError("Multilabel targets must be numeric values in [0, 1].") from error
                if not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric > 1).any():
                    raise ValueError("Multilabel targets must be finite values in [0, 1].")
                self.loss_function = self._objective = (
                    "MultiLogloss" if np.all((numeric == 0) | (numeric == 1)) else "MultiCrossEntropy")
                if self._default_leaf_iterations:
                    self.leaf_estimation_iterations = 40 if self.leaf_estimation_method == "Gradient" else 10
            elif self._objective in _MULTIOUTPUT:
                self.loss_function = self._objective = "Logloss"
        if self._objective in _MULTIOUTPUT and not (
                self._objective == "RMSEWithUncertainty" and self.grow_policy != "SymmetricTree"):
            from ._multioutput_frontend import fit_multioutput
            return fit_multioutput(self, X, y, sample_weight, eval_set=eval_set, cat_features=cat_features,
                early_stopping_rounds=early_stopping_rounds, use_best_model=use_best_model,
                save_snapshot=save_snapshot, snapshot_file=snapshot_file, snapshot_interval=snapshot_interval,
                resume=resume, callback=callback)
        from catboost import CatBoostRegressor, CatBoostClassifier
        fit_started = time.perf_counter()
        # A failed refit must not leave an old model paired with newly encoded
        # labels or feature mappings.
        self._model = None
        selected_cats = self.cat_features if cat_features is None else cat_features
        ordered_group_sizes = None
        if self.boosting_type == "Ordered":
            from ._ordered_data import unpack_ordered_data
            raw, y, sample_weight, cats, names, ordered_group_sizes, eval_set = unpack_ordered_data(
                X, y, sample_weight, selected_cats, eval_set, group_id, group_weight, eval_group_id, eval_group_weight)
        else:
            raw, y, sample_weight, cats, names = unpack_pool(X, y, sample_weight, selected_cats)
        targets = self._targets(y, fitting=True)
        multiclass = self._objective in ("MultiClass", "MultiClassOneVsAll")
        ordered = self.boosting_type == "Ordered"
        greedy = self.grow_policy != "SymmetricTree"
        if ordered and multiclass:
            raise ValueError("Ordered training currently supports scalar objectives.")
        if targets.shape != (raw.shape[0],):
            raise ValueError("X must have exactly one target per row.")
        weights = self._weights(sample_weight, targets)
        effective_class_weights = self._effective_class_weights
        if raw.size > np.iinfo(np.uint32).max:
            raise ValueError("The Metal port requires fewer than 2^32 feature values.")
        greedy_evaluation = None
        greedy_category_metadata = {}
        greedy_cardinalities = None
        greedy_learn_hashes = {}
        if (greedy or ordered) and cats:
            from ._categorical import cat_feature_hashes
            if eval_set is not None:
                from catboost import Pool
                if isinstance(eval_set, Pool):
                    greedy_evaluation = unpack_pool(eval_set)
                elif isinstance(eval_set, tuple) and len(eval_set) in (2, 3):
                    greedy_evaluation = unpack_pool(eval_set[0], eval_set[1],
                        eval_set[2] if len(eval_set) == 3 else None, cats)
                    if hasattr(eval_set[0], "columns") and greedy_evaluation[4] != names:
                        raise ValueError("Validation feature names/order must match training data.")
                else:
                    raise ValueError("eval_set must be one Pool or an (X, y[, weight]) tuple.")
                if greedy_evaluation[3] != cats or greedy_evaluation[0].shape[1] != raw.shape[1]:
                    raise ValueError("Validation features and categorical indices must match training data.")
            eval_hashes = {}
            greedy_cardinalities = {}
            for feature in cats:
                hashes = cat_feature_hashes(raw[:, feature])
                greedy_learn_hashes[str(feature)] = hashlib.sha256(hashes.tobytes()).hexdigest()
                if greedy_evaluation is not None:
                    heldout = cat_feature_hashes(greedy_evaluation[0][:, feature])
                    eval_hashes[str(feature)] = hashlib.sha256(heldout.tobytes()).hexdigest()
                    hashes = np.concatenate((hashes, heldout))
                greedy_cardinalities[feature] = int(np.unique(hashes).size)
            if eval_hashes:
                greedy_category_metadata["eval_categorical_hashes"] = eval_hashes
            if ordered:
                greedy_category_metadata["learn_categorical_hashes"] = greedy_learn_hashes
        ordered_histories = ordered_ctr_groups = None
        if ordered and greedy_cardinalities and any(count > self.one_hot_max_size for count in greedy_cardinalities.values()):
            from ._ordered_rng import cuda_ordered_history_order, cuda_ordered_group_history_order
            ordered_histories = np.stack([cuda_ordered_history_order(len(targets), p, self.fold_permutation_block)
                if ordered_group_sizes is None else cuda_ordered_group_history_order(ordered_group_sizes, p, self.fold_permutation_block)
                for p in range(self.permutation_count or 4)])
            if self.ctr_history_unit == "Group" and ordered_group_sizes is not None:
                ordered_ctr_groups = np.repeat(np.arange(len(ordered_group_sizes), dtype=np.uint32), ordered_group_sizes)
            greedy_category_metadata["ctr_history_unit"] = self.ctr_history_unit
        layout, bins, candidate_features, candidate_bins, candidate_types = prepare_features(
            raw, self.border_count, self.nan_mode, cats, names, self.one_hot_max_size,
            targets=targets, objective=self._objective, ctr_type=self.ctr_type,
            ctr_prior=self.ctr_prior, ctr_target_border=self.ctr_target_border, random_seed=self.random_seed,
            ctr_border_count=self.ctr_border_count,
            permutation_count=4 if self.permutation_count is None else self.permutation_count,
            one_hot_cardinalities=greedy_cardinalities, history_orders=ordered_histories, ctr_group_ids=ordered_ctr_groups)
        if greedy and layout.ctrs:
            greedy_category_metadata["learn_categorical_hashes"] = greedy_learn_hashes
        uncertainty = self._objective == "RMSEWithUncertainty"
        dimensions = len(self.classes_) if multiclass else 2 if uncertainty else None
        bias = np.zeros(dimensions, np.float32) if dimensions else 0.0
        if self.boost_from_average:
            from ._initialization import initialize_bias
            bias = initialize_bias(targets, weights, self._objective, self._loss_parameters)
        eval_bins = eval_targets = eval_weight = None
        if eval_set is not None:
            from catboost import Pool
            if greedy_evaluation is not None:
                eval_raw, eval_y, eval_weight, _, _ = greedy_evaluation
            elif isinstance(eval_set, Pool):
                eval_raw, eval_y, eval_weight, _, _ = unpack_pool(eval_set)
            elif isinstance(eval_set, tuple) and len(eval_set) in (2, 3):
                eval_raw, eval_y = eval_set[:2]
                eval_weight = eval_set[2] if len(eval_set) == 3 else None
            else:
                raise ValueError("eval_set must be one Pool or an (X, y[, weight]) tuple.")
            eval_bins = layout.transform(eval_raw)
            eval_targets = self._targets(eval_y, fitting=False)
            if eval_targets.shape != (eval_bins.shape[1],) or not eval_targets.size:
                raise ValueError("eval_set must have rows and one target per row.")
            eval_weight = self._weights(eval_weight, eval_targets)
        if early_stopping_rounds is not None:
            _integer(early_stopping_rounds, "early_stopping_rounds", 1, 100000)
            if eval_set is None:
                raise ValueError("early_stopping_rounds requires eval_set.")
        if use_best_model is not None and not isinstance(use_best_model, bool):
            raise ValueError("use_best_model must be boolean.")
        if use_best_model and eval_set is None:
            raise ValueError("use_best_model requires eval_set.")
        start = time.perf_counter()
        native_options = dict(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
                              l2_leaf_reg=self.l2_leaf_reg, bias=bias, score_function=self.score_function,
                              objective=self._objective, sample_weight=weights,
                              leaf_estimation_iterations=self.leaf_estimation_iterations,
                              leaf_estimation_backtracking=self.leaf_estimation_backtracking,
                              candidate_types=candidate_types, random_seed=self.random_seed)
        if dimensions:
            native_options.update(classes=dimensions, leaf_estimation_method=self.leaf_estimation_method)
        elif self._objective not in ("RMSE", "Logloss", "CrossEntropy") or self.leaf_estimation_method != "Newton":
            native_options.update(objective_param=self._objective_param,
                                  leaf_estimation_method=self.leaf_estimation_method)
        if self.bootstrap_type != "No" or self.random_strength:
            native_options.update(bootstrap_type=self.bootstrap_type, random_seed=self.random_seed,
                                  bagging_temperature=self.bagging_temperature, subsample=self.subsample,
                                  mvs_reg=self.mvs_reg)
        if self.random_strength:
            native_options["random_strength"] = self.random_strength
        if ordered:
            native_options.update(boosting_type="Ordered", permutation_count=self.permutation_count or 4,
                                  fold_len_multiplier=self.fold_len_multiplier, min_fold_size=self.min_fold_size,
                                  fold_permutation_block=self.fold_permutation_block,
                                  fold_size_loss_normalization=self.fold_size_loss_normalization,
                                  leaf_estimation_method=self.leaf_estimation_method)
        if ordered_group_sizes is not None:
            native_options["group_sizes"] = ordered_group_sizes
        if greedy:
            native_options.pop("mvs_reg", None)
            native_options.update(grow_policy=self.grow_policy, max_leaves=self.max_leaves,
                                  min_data_in_leaf=self.min_data_in_leaf,
                                  leaf_estimation_method=self.leaf_estimation_method)
        if layout.ctrs and not greedy:
            native_options.update(ctr_unique_values=layout.ctr_unique_values(
                eval_raw if eval_set is not None else None), model_size_reg=self.model_size_reg)
        permutation_count = (self.permutation_count or 4) if ordered else len(layout.permutation_bins)
        if (greedy or ordered or layout.ctrs or permutation_count > 1 or eval_set is not None or save_snapshot or snapshot_file is not None
                or callback is not None or self.eval_metric is not None):
            if greedy:
                from ._greedy_training import run_training
            else:
                from ._training import run_training
            result = run_training(
                bins, targets, candidate_features, candidate_bins, **native_options,
                eval_bins=eval_bins, eval_targets=eval_targets, eval_weight=eval_weight,
                early_stopping_rounds=early_stopping_rounds, use_best_model=use_best_model,
                save_snapshot=save_snapshot, snapshot_file=snapshot_file,
                snapshot_interval=snapshot_interval, resume=resume, callback=callback,
                **({"eval_metric": self.eval_metric} if self.eval_metric is not None else {}),
                **({"permutation_bins": layout.permutation_bins} if len(layout.permutation_bins) > 1 else {}),
                metadata={"borders": [b.tolist() for b in layout.borders], "has_nans": layout.has_nans,
                          "nan_mode": layout.nan_mode, "feature_names": names,
                          "categorical_hashes": {str(f): list(e.hashes) for f, e in layout.categorical.items()},
                          "ctrs": layout.ctr_metadata(),
                          "classes": self.classes_.tolist() if self._classifier else None,
                          **greedy_category_metadata})
        else:
            if multiclass:
                from . import _multiclass
                result = _multiclass.train(bins, targets, candidate_features, candidate_bins, **native_options)
            else:
                result = _native.train(bins, targets, candidate_features, candidate_bins, **native_options)
        result.stats["native_wall_seconds"] = time.perf_counter() - start
        result.stats["preprocessing_seconds"] = start - fit_started
        result.stats["ctr_gpu_seconds"] = layout.ctr_gpu_stats["gpu_seconds"]
        result.stats["ctr_kernel_dispatches"] = layout.ctr_gpu_stats["kernel_dispatches"]
        if greedy:
            from ._greedy_model import model_json, dumps_model_json
            model_data = model_json(result, layout.borders, bias=bias, objective=self._objective,
                                    feature_names=names, grow_policy=self.grow_policy, layout=layout,
                                    objective_param=self._objective_param, loss_parameters=self._loss_parameters)
        else:
            model_data = _model_json(layout.borders, result, bias, self.score_function, layout)
        bootstrap_metadata = {"type": self.bootstrap_type}
        if self.bootstrap_type == "Bayesian":
            bootstrap_metadata["bagging_temperature"] = self.bagging_temperature
        elif self.bootstrap_type != "No":
            bootstrap_metadata["subsample"] = self.subsample
            if self.bootstrap_type == "MVS" and self.mvs_reg is not None:
                bootstrap_metadata["mvs_reg"] = self.mvs_reg
        model_data["model_info"].update(_training_metadata(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg, border_count=self.border_count, score_function=self.score_function,
            objective=self._objective, boost_from_average=self.boost_from_average,
            leaf_estimation_iterations=self.leaf_estimation_iterations, nan_mode=self.nan_mode,
            use_best_model=(eval_set is not None if use_best_model is None else use_best_model),
            classes=self.classes_ if self._classifier else None, class_weights=effective_class_weights,
            categorical=bool(cats), one_hot_max_size=self.one_hot_max_size, eval_metric=self.eval_metric,
            loss_parameters=self._loss_parameters, loss_description=self.loss_function,
            leaf_estimation_method=self.leaf_estimation_method, bootstrap=bootstrap_metadata,
            random_strength=self.random_strength, random_seed=self.random_seed,
            leaf_estimation_backtracking=self.leaf_estimation_backtracking, permutation_count=permutation_count,
            model_size_reg=self.model_size_reg if layout.ctrs else None, boosting_type=self.boosting_type,
            fold_len_multiplier=self.fold_len_multiplier, min_fold_size=self.min_fold_size,
            fold_size_loss_normalization=self.fold_size_loss_normalization,
            grow_policy=self.grow_policy, max_leaves=self.max_leaves, min_data_in_leaf=self.min_data_in_leaf,
            fold_permutation_block=self.fold_permutation_block))
        with tempfile.TemporaryDirectory(prefix="catbooster-metal-model-") as temporary:
            path = Path(temporary) / "model.json"
            path.write_text(dumps_model_json(model_data) if greedy else json.dumps(model_data, allow_nan=False))
            model_type = CatBoostClassifier if self._classifier else CatBoostRegressor
            model = model_type().load_model(str(path), format="json")
        self._model, self._result, self._layout = model, result, layout
        self.borders_, self.bias_ = layout.borders, bias
        self.n_features_in_, self.feature_names_ = raw.shape[1], names
        if dimensions:
            self.n_outputs_ = dimensions
        if greedy:
            self.tree_count_ = len(result.trees)
            def graph_depth(nodes):
                stack, maximum = [(0, 0)], 0
                while stack:
                    index, depth = stack.pop()
                    maximum = max(maximum, depth)
                    if nodes[index, 5] == np.iinfo(np.uint32).max:
                        stack.extend(((int(nodes[index, 3]), depth + 1), (int(nodes[index, 4]), depth + 1)))
                return maximum
            self.tree_depths_ = np.array([graph_depth(tree.nodes) for tree in result.trees], np.uint32)
        else:
            self.tree_count_, self.tree_depths_ = len(result.depths), result.depths.copy()
        self.training_predictions_ = result.predictions.copy()
        self.loss_history_, self.training_stats_ = (result.loss if greedy else result.rmse).tolist(), result.stats.copy()
        self.training_stats_["fit_wall_seconds"] = time.perf_counter() - fit_started
        self.best_iteration_ = getattr(result, "best_iteration", -1)
        self.best_score_ = getattr(result, "best_score", {"learn": {self.loss_function: min(self.loss_history_[1:])}})
        self.evals_result_ = getattr(result, "evals_result", {"learn": {self.loss_function: self.loss_history_[1:]}})
        return self

    def _require_fitted(self):
        if self._model is None:
            raise RuntimeError("Call fit before using this model.")

    def predict(self, X, prediction_type=None, *, task_type="CPU", ntree_start=0, ntree_end=0):
        self._require_fitted()
        if self._objective in _MULTIOUTPUT:
            from ._multioutput_frontend import predict_multioutput
            return predict_multioutput(self, X, prediction_type=prediction_type, task_type=task_type,
                                       ntree_start=ntree_start, ntree_end=ntree_end)
        prediction_type = prediction_type or ("Class" if self._classifier else
                                              "Exponent" if self._objective in ("Poisson", "Tweedie") else "RawFormulaVal")
        bins = self._layout.transform(X)
        if task_type not in ("CPU", "METAL", "GPU"):
            raise ValueError("Prediction task_type must be CPU, METAL, or GPU.")
        if task_type == "CPU":
            if bins.shape[1] == 0:
                if self._objective in ("MultiClass", "MultiClassOneVsAll"):
                    return np.empty((0, 1) if prediction_type in ("Class", "Exponent") else (0, len(self.classes_)))
                return np.empty((0, 2), np.float64) if prediction_type in ("Probability", "LogProbability") else np.empty(0)
            return self._model.predict(X, prediction_type=prediction_type,
                                       ntree_start=ntree_start, ntree_end=ntree_end)
        from ._inference import predict_bins
        result = self._result
        multiclass = self._objective in ("MultiClass", "MultiClassOneVsAll")
        if self.grow_policy != "SymmetricTree":
            from ._greedy_inference import predict_bins as predict_greedy_bins
            raw = predict_greedy_bins(bins, result.trees, self.bias_,
                                     tree_start=ntree_start, tree_end=ntree_end or None)
        elif multiclass:
            raw = np.column_stack([predict_bins(
                bins, result.depths, result.split_features, result.split_bins,
                result.leaf_values[:, :, index], split_types=getattr(result, "split_types", None),
                bias=self.bias_[index], tree_start=ntree_start, tree_end=ntree_end or None)
                for index in range(len(self.classes_))])
        else:
            raw = predict_bins(bins, result.depths, result.split_features, result.split_bins,
                               result.leaf_values, split_types=getattr(result, "split_types", None),
                               bias=self.bias_, tree_start=ntree_start, tree_end=ntree_end or None)
        if prediction_type == "RawFormulaVal":
            return raw
        if prediction_type == "Exponent":
            # CatBoost applies Exponent to the first output of vector models.
            return np.exp(raw[:, 0]).reshape(-1, 1) if multiclass else np.exp(raw)
        if not self._classifier:
            raise ValueError("Metal regression prediction currently supports RawFormulaVal.")
        if multiclass:
            if prediction_type == "Class":
                return self.classes_[np.argmax(raw, axis=1)].reshape(-1, 1)
            if self._objective == "MultiClassOneVsAll":
                log_probability = -np.logaddexp(0, -raw)
            else:
                shifted = raw - raw.max(axis=1, keepdims=True)
                log_probability = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            if prediction_type == "LogProbability":
                return log_probability
            if prediction_type == "Probability":
                return np.exp(log_probability)
            raise ValueError("Unknown prediction_type.")
        if prediction_type == "Class":
            return self.classes_[(raw > 0).astype(np.int32)]
        if prediction_type == "LogProbability":
            return np.column_stack((-np.logaddexp(0, raw), -np.logaddexp(0, -raw)))
        if prediction_type == "Probability":
            positive = np.exp(-np.logaddexp(0, -raw))
            return np.column_stack((1 - positive, positive))
        raise ValueError("Unknown prediction_type.")

    def save_model(self, filename, format="cbm"):
        self._require_fitted()
        if format not in ("cbm", "json"):
            raise ValueError("The Metal bridge supports saving cbm or json models.")
        self._model.save_model(str(filename), format=format)

    def to_catboost(self):
        self._require_fitted()
        return self._model.copy()

    def get_best_iteration(self):
        self._require_fitted()
        return self.best_iteration_

    def get_best_score(self):
        import copy
        self._require_fitted()
        return copy.deepcopy(self.best_score_)

    def get_evals_result(self):
        import copy
        self._require_fitted()
        return copy.deepcopy(self.evals_result_)


class CatBoostMetalClassifier(CatBoostMetalRegressor):
    """CatBoost classification algorithms translated from CUDA to Metal."""

    _classifier = True

    def predict_proba(self, X, *, task_type="CPU", ntree_start=0, ntree_end=0):
        return self.predict(X, "Probability", task_type=task_type,
                            ntree_start=ntree_start, ntree_end=ntree_end)
