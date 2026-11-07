"""Standalone access to supported native Metal training options.

The native adapter owns Pool preparation, dynamic CTR registration, permutation
cursors, snapshots and final CTR tables. Reusing that adapter keeps one scheduler
and one snapshot implementation for these newly exposed modes.
"""
from copy import deepcopy
import math
from pathlib import Path
import platform
import time

import numpy as np


def _pool(data, label=None, *, cats=None, weight=None, groups=None, group_weight=None,
          subgroups=None, pairs=None, pairs_weight=None):
    from catboost import Pool
    if isinstance(data, Pool):
        supplied = (label, weight, groups, group_weight, subgroups, pairs, pairs_weight)
        if any(value is not None for value in supplied):
            raise ValueError("Supply labels, weights, groups and pairs inside the Pool, or use array inputs.")
        if cats is not None:
            names = data.get_feature_names()
            indices = [names.index(feature) if isinstance(feature, str) else feature for feature in cats]
            if sorted(indices) != sorted(data.get_cat_feature_indices()):
                raise ValueError("cat_features disagrees with the Pool's categorical features.")
        return data
    if weight is not None and group_weight is not None:
        from ._query_data import prepare_groups
        prepare_groups(groups, len(data), weight, group_weight)
        # Pool's constructor excludes simultaneous weights, but the public
        # setter preserves both. Query metrics need the original group mass
        # as well as the effective object mass used by target derivatives.
        result = Pool(data, label=label, cat_features=cats, weight=weight, group_id=groups,
                      subgroup_id=subgroups, pairs=pairs, pairs_weight=pairs_weight)
        result.set_group_weight(np.asarray(group_weight, dtype=np.float32))
        return result
    return Pool(data, label=label, cat_features=cats, weight=weight, group_id=groups,
                group_weight=group_weight, subgroup_id=subgroups, pairs=pairs, pairs_weight=pairs_weight)


def _evaluation_pool(eval_set, *, ranker, cats, groups, weight, group_weight, subgroups, pairs, pairs_weight):
    from catboost import Pool
    extras = [groups, weight, group_weight, subgroups]
    if eval_set is None:
        if any(value is not None for value in (*extras, pairs, pairs_weight)):
            raise ValueError("Evaluation groups, weights and pairs require eval_set.")
        return None
    if isinstance(eval_set, Pool):
        return _pool(eval_set, cats=cats, groups=groups, weight=weight, group_weight=group_weight,
                     subgroups=subgroups, pairs=pairs, pairs_weight=pairs_weight)
    maximum = 6 if ranker else 3
    if not isinstance(eval_set, tuple) or not 2 <= len(eval_set) <= maximum:
        raise ValueError("eval_set must be a Pool or an (X, y[, group_id, weight, group_weight, subgroup_id]) tuple."
                         if ranker else "eval_set must be a Pool or an (X, y[, weight]) tuple.")
    for index, value in enumerate(eval_set[2:]):
        destination = index if ranker else 1
        if extras[destination] is not None:
            raise ValueError("Supply evaluation groups and weights in the tuple or keywords, not both.")
        extras[destination] = value
    return _pool(eval_set[0], eval_set[1], cats=cats, groups=extras[0], weight=extras[1],
                 group_weight=extras[2], subgroups=extras[3], pairs=pairs, pairs_weight=pairs_weight)


def _ctr_descriptions(value, fallback, name):
    if value is None:
        return [fallback]
    result = [value] if isinstance(value, str) else value
    if not isinstance(result, (list, tuple)) or not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must be a nonempty CTR description or sequence of descriptions.")
    return list(result)


def _feature_names(pool):
    names = pool.get_feature_names()
    occupied = {name for name in names if name}
    for index, name in enumerate(names):
        if not name:
            candidate = str(index)
            while candidate in occupied:
                candidate = "feature_" + candidate
            names[index] = candidate
            occupied.add(candidate)
    return names


def native_parameters(estimator):
    """Translate every accepted standalone setting into its native option."""
    if estimator.ctr_target_border is not None:
        raise ValueError("Explicit ctr_target_border is unavailable in the native Metal adapter; use native CTR target binarization options.")
    if getattr(estimator, "yeti_legacy_prefix_centering", False):
        raise ValueError("The native Metal adapter does not support the standalone legacy YetiRank centering option.")
    ctr = (f"{estimator.ctr_type}:CtrBorderType=Uniform:CtrBorderCount={estimator.ctr_border_count}"
           f":Prior={estimator.ctr_prior}")
    result = dict(task_type="GPU", data_partition=estimator.data_partition, boosting_type=estimator.boosting_type,
        grow_policy=estimator.grow_policy, loss_function=estimator.loss_function, iterations=estimator.iterations,
        depth=estimator.depth, learning_rate=estimator.learning_rate, l2_leaf_reg=estimator.l2_leaf_reg,
        border_count=estimator.border_count, nan_mode=estimator.nan_mode, score_function=estimator.score_function,
        leaf_estimation_method=estimator.leaf_estimation_method,
        leaf_estimation_iterations=estimator.leaf_estimation_iterations,
        leaf_estimation_backtracking=estimator.leaf_estimation_backtracking,
        boost_from_average=estimator.boost_from_average, random_seed=estimator.random_seed,
        random_strength=estimator.random_strength, bootstrap_type=estimator.bootstrap_type,
        permutation_count=estimator.permutation_count or 4, has_time=estimator.permutation_count == 1,
        fold_len_multiplier=estimator.fold_len_multiplier, min_fold_size=estimator.min_fold_size,
        fold_permutation_block=estimator.fold_permutation_block,
        fold_size_loss_normalization=estimator.fold_size_loss_normalization,
        add_ridge_penalty_to_loss_function=estimator.add_ridge_penalty_to_loss_function,
        meta_l2_exponent=estimator.meta_l2_exponent, meta_l2_frequency=estimator.meta_l2_frequency,
        langevin=estimator.langevin, diffusion_temperature=estimator.diffusion_temperature, rsm=estimator.rsm,
        one_hot_max_size=estimator.one_hot_max_size, max_ctr_complexity=estimator.max_ctr_complexity,
        simple_ctr=_ctr_descriptions(estimator.simple_ctr, ctr, "simple_ctr"),
        combinations_ctr=_ctr_descriptions(estimator.combinations_ctr, ctr, "combinations_ctr"),
        ctr_target_border_count=estimator.ctr_target_border_count,
        ctr_history_unit=estimator.ctr_history_unit, counter_calc_method=estimator.counter_calc_method,
        model_size_reg=estimator.model_size_reg, metric_period=1, verbose=False, allow_writing_files=False)
    if estimator.eval_metric is not None:
        result["eval_metric"] = estimator.eval_metric
    if estimator.fixed_binary_splits is not None:
        result["fixed_binary_splits"] = list(estimator.fixed_binary_splits)
    if estimator.grow_policy != "SymmetricTree":
        result["min_data_in_leaf"] = estimator.min_data_in_leaf
        if estimator.grow_policy == "Lossguide":
            result["max_leaves"] = estimator.max_leaves
    if hasattr(estimator, "sampling_unit"):
        result["sampling_unit"] = estimator.sampling_unit
    if estimator._objective in ("PairLogitPairwise", "QueryCrossEntropy", "YetiRankPairwise"):
        result["bayesian_matrix_reg"] = estimator.bayesian_matrix_reg
    if estimator._objective in ("YetiRank", "YetiRankPairwise"):
        # Preserve the standalone learn history despite the shared metric's
        # usual skip_train default. Hints do not alter metric arithmetic.
        if result.get("eval_metric") == "PFound":
            result["eval_metric"] = "PFound:hints=skip_train~false"
        result["custom_metric"] = ["PFound:use_weights=true;hints=skip_train~false"]
    if estimator.class_weights is not None:
        result["class_weights"] = estimator.class_weights
    if estimator.bootstrap_type == "Bayesian":
        result["bagging_temperature"] = estimator.bagging_temperature
    elif estimator.bootstrap_type != "No":
        result["subsample"] = estimator.subsample
    if estimator.mvs_reg is not None:
        result["mvs_reg"] = estimator.mvs_reg
    return result


class _Callback:
    def __init__(self, callback):
        self.callback, self.stopped, self.iterations = callback, False, []

    def after_iteration(self, info):
        self.iterations.append(info.iteration)
        if self.callback is not None and self.callback(deepcopy(info)) is False:
            self.stopped = True
            return False
        return True


def fit_feature_parallel(estimator, X, y=None, sample_weight=None, *, ranker=False, eval_set=None,
        cat_features=None, group_id=None, group_weight=None, subgroup_id=None, pairs=None, pairs_weight=None,
        eval_group_id=None, eval_sample_weight=None, eval_group_weight=None, eval_subgroup_id=None,
        eval_pairs=None, eval_pairs_weight=None, early_stopping_rounds=None, use_best_model=None,
        save_snapshot=False, snapshot_file=None, snapshot_interval=600., resume=True, callback=None, init_model=None):
    from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor
    from .regressor import _integer

    started = time.perf_counter()
    estimator._model, estimator._native_bridge_fitted = None, False
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Native Metal training requires Apple Silicon macOS.")
    model_type = CatBoostRanker if ranker else CatBoostClassifier if estimator._classifier else CatBoostRegressor
    model = model_type().set_params(**native_parameters(estimator))
    if not all(hasattr(model._object, name) for name in ("_get_metal_training_cursor", "_get_metal_training_info")):
        raise RuntimeError("Native Metal training requires this repository's rebuilt CatBoost Metal extension with online-cursor support.")
    if callback is not None and not callable(callback):
        raise ValueError("callback must be callable.")
    for name, value in (("save_snapshot", save_snapshot), ("resume", resume)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean.")
    if use_best_model is not None and not isinstance(use_best_model, bool):
        raise ValueError("use_best_model must be boolean.")
    if ranker and estimator._objective not in ("PairLogit", "PairLogitPairwise") and any(
            value is not None for value in (pairs, pairs_weight, eval_pairs, eval_pairs_weight)):
        raise ValueError("Explicit pairs require PairLogit or PairLogitPairwise.")
    if isinstance(snapshot_interval, bool) or not isinstance(snapshot_interval, (int, float)) or not math.isfinite(snapshot_interval) or snapshot_interval < 0:
        raise ValueError("snapshot_interval must be finite and nonnegative.")
    cats = estimator.cat_features if cat_features is None else cat_features
    train = _pool(X, y, cats=cats, weight=sample_weight, groups=group_id, group_weight=group_weight,
                  subgroups=subgroup_id, pairs=pairs, pairs_weight=pairs_weight)
    if estimator._default_boost_from_average and (init_model is not None or train.get_baseline().size):
        model.set_params(boost_from_average=False)
    names = _feature_names(train)
    if estimator.feature_weights is not None:
        model.set_params(feature_weights=estimator._feature_weights(names).tolist())
    if estimator._classifier and estimator._objective not in ("MultiLogloss", "MultiCrossEntropy"):
        estimator._targets(train.get_label(), fitting=True)
        if estimator.data_partition == "FeatureParallel" and estimator._objective not in ("Logloss", "CrossEntropy"):
            raise ValueError("Native FeatureParallel classification currently supports binary scalar objectives.")
        model.set_params(loss_function=estimator.loss_function,
                         leaf_estimation_iterations=estimator.leaf_estimation_iterations,
                         score_function=estimator.score_function)
        if estimator._objective != "CrossEntropy":
            model.set_params(class_names=estimator.classes_.tolist())
    validation = _evaluation_pool(eval_set, ranker=ranker, cats=train.get_cat_feature_indices(),
        groups=eval_group_id, weight=eval_sample_weight, group_weight=eval_group_weight, subgroups=eval_subgroup_id,
        pairs=eval_pairs, pairs_weight=eval_pairs_weight)
    if ranker:
        for pool in (train, validation):
            if pool is None:
                continue
            if pool.get_group_id_hash() is None:
                raise ValueError("group_id is required for query training and evaluation.")
            if estimator._objective in ("PairLogit", "PairLogitPairwise") and not pool.num_pairs():
                raise ValueError("Explicit pairs are required; automatic pair generation is not supported.")
            if estimator._objective not in ("PairLogit", "PairLogitPairwise") and pool.num_pairs():
                raise ValueError("Explicit pairs require PairLogit or PairLogitPairwise.")
    if validation is not None:
        if validation.num_col() != train.num_col() or validation.get_cat_feature_indices() != train.get_cat_feature_indices():
            raise ValueError("Validation features and categorical indices must match training data.")
        if hasattr(eval_set[0] if isinstance(eval_set, tuple) else eval_set, "columns") and validation.get_feature_names() != train.get_feature_names():
            raise ValueError("Validation feature names/order must match training data.")
    if early_stopping_rounds is not None:
        _integer(early_stopping_rounds, "early_stopping_rounds", 1, 100000)
        if validation is None:
            raise ValueError("early_stopping_rounds requires eval_set.")
    if use_best_model and validation is None:
        raise ValueError("use_best_model requires eval_set.")
    if init_model is not None and hasattr(init_model, "to_catboost"):
        init_model = init_model.to_catboost()
    snapshot = Path(snapshot_file).expanduser().resolve() if snapshot_file is not None else None
    if save_snapshot and snapshot is None:
        raise ValueError("save_snapshot requires snapshot_file.")
    if snapshot is not None:
        if float(snapshot_interval) != int(snapshot_interval):
            raise ValueError("Native snapshots require snapshot_interval to be a whole number of seconds.")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            snapshot.unlink(missing_ok=True)
        model.set_params(save_snapshot=True, snapshot_file=str(snapshot), snapshot_interval=int(snapshot_interval),
                         allow_writing_files=True, train_dir=str(snapshot.parent))
    callback_adapter = _Callback(callback)
    model.fit(train, eval_set=validation, use_best_model=validation is not None if use_best_model is None else use_best_model,
              early_stopping_rounds=early_stopping_rounds, callbacks=[callback_adapter], init_model=init_model)
    metadata = model.get_metadata()
    if metadata.get("metal_backend") != "METAL":
        raise RuntimeError("The native training result was not produced by the Metal backend.")
    cursor = np.asarray(model._object._get_metal_training_cursor(), np.float32)
    if cursor.ndim not in (1, 2) or cursor.shape[0] != train.num_row() or not np.isfinite(cursor).all():
        raise RuntimeError("Native Metal returned an invalid online training cursor.")
    info = model._object._get_metal_training_info()
    initial_loss = float(info["initial_loss"])
    if not math.isfinite(initial_loss):
        raise RuntimeError("Native Metal did not return its initial objective loss.")
    history = model.get_evals_result()
    objective_key = info["objective_metric"]
    learn = history.get("learn", {})
    if objective_key and objective_key not in learn:
        weighted_key = objective_key + (";" if ":" in objective_key else ":") + "use_weights=true"
        if weighted_key in learn:
            objective_key = weighted_key
    if not objective_key or objective_key not in learn:
        raise RuntimeError("Native Metal objective history is missing.")
    estimator._model, estimator._result, estimator._layout = model, None, None
    estimator._native_bridge_fitted = True
    estimator.n_features_in_, estimator.feature_names_ = train.num_col(), names
    estimator.n_outputs_ = cursor.shape[1] if cursor.ndim == 2 else 1
    if estimator._classifier:
        estimator.classes_ = model.classes_.copy()
    borders = model.get_borders()
    estimator.borders_ = [np.asarray(borders.get(feature, []), np.float32) for feature in range(train.num_col())]
    estimator.bias_ = model.get_scale_and_bias()[1]
    estimator.tree_count_ = model.tree_count_
    estimator.tree_depths_ = _tree_depths(model, estimator.grow_policy)
    estimator.training_predictions_ = cursor.copy()
    estimator.loss_history_ = [initial_loss, *learn[objective_key]]
    estimator.evals_result_, estimator.best_score_ = deepcopy(history), model.get_best_score()
    estimator.best_iteration_ = model.get_best_iteration()
    estimator.training_stats_ = dict(backend="METAL", device=metadata.get("metal_device"),
        device_name=metadata.get("metal_device"), kernel_dispatches=int(info["kernel_dispatches"]),
        gpu_seconds=float(info["gpu_seconds"]),
        boosting_type=estimator.boosting_type, data_partition=estimator.data_partition,
        grow_policy=estimator.grow_policy,
        permutation_count=int(metadata["metal_permutations"]),
        tree_ctr_features=int(metadata.get("metal_tree_ctr_features", "0")),
        resumed_iterations=int(info["resumed_iterations"]), completed_iterations=len(learn[objective_key]),
        stop_reason="callback" if callback_adapter.stopped else "early_stopping" if len(learn[objective_key]) < estimator.iterations else "iterations",
        fit_wall_seconds=time.perf_counter() - started)
    if ranker:
        groups = np.asarray(train.get_group_id_hash())
        estimator.group_offsets_ = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)].astype(np.uint32)
        if estimator._objective in ("PairLogit", "PairLogitPairwise"):
            estimator.pairs_ = np.asarray(train.get_pairs(), np.uint32).reshape(-1, 2)
            estimator.pairs_weight_ = np.asarray(train.get_pairs_weight(), np.float32)
            estimator.training_stats_["supplied_pairs"] = len(estimator.pairs_)
    return estimator


def _tree_depths(model, grow_policy):
    if grow_policy == "SymmetricTree":
        return np.log2(model.get_tree_leaf_counts()).astype(np.uint32)
    depths = []
    for tree in range(model.tree_count_):
        steps = model._get_tree_step_nodes(tree)
        pending, maximum = [(0, 0)], 0
        while pending:
            node, depth = pending.pop()
            left, right = steps[node]
            if not left and not right:
                maximum = max(maximum, depth)
                continue
            maximum = max(maximum, depth + 1)
            for child in (left, right):
                if child:
                    pending.append((node + child, depth + 1))
        depths.append(maximum)
    return np.asarray(depths, np.uint32)


def predict_feature_parallel(estimator, X, *, prediction_type=None, task_type="CPU", ntree_start=0, ntree_end=0):
    if task_type not in ("CPU", "METAL", "GPU"):
        raise ValueError("Prediction task_type must be CPU, METAL, or GPU.")
    prediction_type = prediction_type or ("Class" if estimator._classifier else
        "Exponent" if estimator._objective in ("Poisson", "Tweedie") else
        "RMSEWithUncertainty" if estimator._objective == "RMSEWithUncertainty" else "RawFormulaVal")
    options = dict(task_type="GPU" if task_type == "METAL" else task_type, ntree_start=ntree_start, ntree_end=ntree_end)
    if estimator._objective not in ("QueryRMSE", "QuerySoftMax", "PairLogit", "YetiRank",
                                    "PairLogitPairwise", "QueryCrossEntropy", "YetiRankPairwise"):
        options["prediction_type"] = prediction_type
    return estimator._model.predict(X, **options)
