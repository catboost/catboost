"""Standalone query-objective training on the Apple GPU."""

import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from ._data import numeric_array, prepare_features
from ._query_data import QUERY_OBJECTIVES, PAIR_OBJECTIVES, query_metric, prepare_pairs, unpack_pairs, unpack_query_pool, unpack_subgroups
from ._training import run_training, _RANKING_METRICS
from .regressor import CatBoostMetalRegressor, _model_json, _training_metadata, _number


def _query_loss(description):
    if not isinstance(description, str):
        raise ValueError("loss_function must be a query loss description string.")
    base, separator, arguments = description.partition(":")
    if base == "QueryCrossEntropy":
        from ._query_cross_entropy import parse_description
        return base, parse_description(description)
    if base not in (*QUERY_OBJECTIVES, *PAIR_OBJECTIVES, "YetiRank", "YetiRankPairwise"):
        raise ValueError("The standalone ranker currently supports QueryRMSE, QuerySoftMax, supplied-pair PairLogit/PairLogitPairwise, QueryCrossEntropy, and classic YetiRank.")
    parameters = {}
    for item in arguments.split(";") if separator else []:
        key, equals, value = item.partition("=")
        allowed = {"beta", "lambda"} if base == "QuerySoftMax" else ({"permutations", "decay", "mode"} if base in ("YetiRank", "YetiRankPairwise") else set())
        if not equals or key in parameters or key not in allowed:
            raise ValueError("Invalid or duplicate query loss parameter.")
        if key == "mode":
            if value != "Classic":
                raise ValueError("The CUDA-derived YetiRank target currently supports mode=Classic.")
            parameters[key] = value
            continue
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError("Query parameters must be finite numbers.") from exc
        if not np.isfinite(number) or abs(number) > np.finfo(np.float32).max:
            raise ValueError("Query parameters must be finite float32 numbers.")
        if key == "permutations":
            if number != int(number) or not 1 <= number <= 10000:
                raise ValueError("YetiRank permutations must be an integer in [1, 10000].")
            number = int(number)
        if key == "decay" and not 0 <= number <= 1:
            raise ValueError("YetiRank decay must be in [0, 1].")
        parameters[key] = number
    return base, parameters


class CatBoostMetalRanker(CatBoostMetalRegressor):
    """Metal ranker with explicit contiguous query groups.

    QueryRMSE, QuerySoftMax, PairLogit and classic YetiRank support Ordered
    symmetric trees and Plain symmetric, Depthwise, Lossguide or Region trees.
    ``fit`` accepts arrays plus ``group_id``, optional object/group weights and
    ``subgroup_id`` for PFound's duplicate-subgroup handling,
    or a numeric Pool from a CatBoost build exposing ``get_group_weight``.
    Arrays/DataFrames accept one-hot categories; categorical/prequantized Pools
    use the native ``CatBoostRanker(task_type="GPU")`` entry point.
    Validation can be a Pool or ``(X, y[, group_id[, weight[, group_weight[, subgroup_id]]]])``.
    PairLogit requires ``pairs`` and optional ``pairs_weight``; their literal
    weights determine training mass independently of object and group weights.
    """

    def __init__(self, *, loss_function="QueryRMSE", leaf_estimation_method=None,
                 leaf_estimation_iterations=None, boost_from_average=False, yeti_legacy_prefix_centering=False,
                 bayesian_matrix_reg=None, sampling_unit="Object", **options):
        objective, parameters = _query_loss(loss_function)
        coupled = objective == "PairLogitPairwise"
        yeti_pair = objective == "YetiRankPairwise"
        qce = objective == "QueryCrossEntropy"
        if sampling_unit not in ("Object", "Group") or (sampling_unit == "Group" and not yeti_pair):
            raise ValueError("Group sampling requires YetiRankPairwise.")
        self.sampling_unit = sampling_unit
        if bayesian_matrix_reg is not None and not (coupled or qce or yeti_pair):
            raise ValueError("bayesian_matrix_reg requires PairLogitPairwise or QueryCrossEntropy.")
        self.bayesian_matrix_reg = _number(.1 if bayesian_matrix_reg is None else bayesian_matrix_reg, "bayesian_matrix_reg")
        if coupled or qce or yeti_pair:
            options.setdefault("l2_leaf_reg", 1. if qce else 0. if yeti_pair else 5.)
            options.setdefault("border_count", 32)
            options.setdefault("random_strength", 0.)
            options.setdefault("leaf_estimation_backtracking", "No" if yeti_pair else "AnyImprovement")
            options.setdefault("bootstrap_type", "Bayesian" if "bagging_temperature" in options else "Bernoulli")
            if options['bootstrap_type'] in ('Bernoulli', 'Poisson'): options.setdefault("subsample", .5)
            if options['bootstrap_type'] == 'MVS': raise ValueError("PairLogitPairwise does not support MVS.")
            if yeti_pair and options['bootstrap_type'] == 'Poisson': raise ValueError("YetiRankPairwise does not support Poisson bootstrap.")
            if yeti_pair and options['leaf_estimation_backtracking'] != 'No': raise ValueError("YetiRankPairwise does not support backtracking.")
            if qce and options['bootstrap_type'] not in ('No', 'Bernoulli'):
                raise ValueError("QueryCrossEntropy supports No or Bernoulli query bootstrap only.")
            if qce and options.get('score_function') == 'L2': raise ValueError("QueryCrossEntropy does not support L2 structure score like CUDA.")
            if options.get("eval_metric") is None: options["eval_metric"] = loss_function if qce else "PFound" if yeti_pair else "PairLogit"
        if not isinstance(yeti_legacy_prefix_centering, bool):
            raise ValueError("yeti_legacy_prefix_centering must be boolean.")
        if yeti_legacy_prefix_centering and objective != "YetiRank":
            raise ValueError("Legacy YetiRank centering requires YetiRank.")
        self.yeti_legacy_prefix_centering = yeti_legacy_prefix_centering
        if objective == "YetiRank":
            options.setdefault("l2_leaf_reg", 0.)
            if options.get("eval_metric") is None:
                options["eval_metric"] = "PFound"
        if boost_from_average:
            raise ValueError("Query objectives initialize raw predictions at zero.")
        if options.get("boosting_type", "Plain") == "Ordered" and objective not in (
                "QueryRMSE", "QuerySoftMax", "PairLogit", "YetiRank"):
            raise ValueError("Ordered ranking supports QueryRMSE, QuerySoftMax, PairLogit and classic YetiRank.")
        if options.get("grow_policy", "SymmetricTree") != "SymmetricTree" and objective not in (
                "QueryRMSE", "QuerySoftMax", "PairLogit", "YetiRank"):
            raise ValueError("Greedy ranking supports QueryRMSE, QuerySoftMax, PairLogit and classic YetiRank.")
        method = leaf_estimation_method or ("Simple" if yeti_pair else "Gradient" if objective == "QuerySoftMax" else "Newton")
        if objective == "YetiRank" and (method != "Newton" or options.get("leaf_estimation_backtracking", "No") != "No"):
            raise ValueError("YetiRank requires Newton leaves and no leaf backtracking like CUDA.")
        if qce and method not in ("Newton", "Simple"): raise ValueError("QueryCrossEntropy requires Newton or Simple leaves like CUDA.")
        if method not in ("Newton", "Gradient", "Simple"):
            raise ValueError("Query objectives support Newton, Gradient and Simple leaves.")
        if method == "Simple" and leaf_estimation_iterations not in (None, 1):
            raise ValueError("Simple leaves require one estimation iteration.")
        if leaf_estimation_iterations is None:
            if method == "Simple":
                leaf_estimation_iterations = 1
            elif objective == "QuerySoftMax":
                leaf_estimation_iterations = 100 if method == "Gradient" else 10
            elif qce:
                leaf_estimation_iterations = 10
            elif yeti_pair:
                leaf_estimation_iterations = 1
            elif coupled:
                leaf_estimation_iterations = 5 if method == "Gradient" else 1
            elif objective in PAIR_OBJECTIVES:
                leaf_estimation_iterations = 40 if method == "Gradient" else 10
            else:
                leaf_estimation_iterations = 1
        # Reuse option validation and prediction/export methods. No training is
        # performed by this constructor; native fit receives the query objective.
        # Full-matrix Simple already has a standalone runtime. Select the native
        # adapter only when an independent option needs that route.
        super().__init__(loss_function="RMSE", leaf_estimation_method="Newton" if method == "Simple" and (coupled or qce or yeti_pair) else method,
                         leaf_estimation_iterations=leaf_estimation_iterations,
                         boost_from_average=False, **options)
        if (coupled or qce or yeti_pair) and self.depth > 8: raise ValueError("Full-matrix query objectives support depth <= 8 like CUDA.")
        if method == "Simple" and (coupled or qce or yeti_pair) and self.depth == 0: raise ValueError("Full-matrix Simple leaves require depth 1..8.")
        if self._native_feature_parallel and objective not in ("QueryRMSE", "QuerySoftMax", "PairLogit", "YetiRank"):
            raise ValueError("FeatureParallel ranking supports QueryRMSE, QuerySoftMax, PairLogit and classic YetiRank.")
        self.leaf_estimation_method = method
        self.loss_function, self._objective, self._loss_parameters = loss_function, objective, parameters
        self.query_beta = float(np.float32(parameters.get("beta", 1)))
        self.query_lambda = float(np.float32(parameters.get("lambda", .01)))

    def fit(self, X, y=None, sample_weight=None, *, group_id=None, group_weight=None, subgroup_id=None,
            eval_set=None, eval_group_id=None, eval_group_weight=None, eval_sample_weight=None, eval_subgroup_id=None,
            pairs=None, pairs_weight=None, eval_pairs=None, eval_pairs_weight=None,
            cat_features=None, early_stopping_rounds=None, use_best_model=None,
            save_snapshot=False, snapshot_file=None, snapshot_interval=600.0, resume=True, callback=None, init_model=None):
        if init_model is self:
            init_model = self.to_catboost()
        self._model = None
        self._native_bridge_fitted = False
        if self._native_adapter:
            from ._feature_parallel_frontend import fit_feature_parallel
            return fit_feature_parallel(self, X, y, sample_weight, group_id=group_id, group_weight=group_weight,
                subgroup_id=subgroup_id, pairs=pairs, pairs_weight=pairs_weight,
                eval_set=eval_set, eval_group_id=eval_group_id, eval_group_weight=eval_group_weight,
                eval_sample_weight=eval_sample_weight, eval_subgroup_id=eval_subgroup_id,
                eval_pairs=eval_pairs, eval_pairs_weight=eval_pairs_weight, cat_features=cat_features,
                early_stopping_rounds=early_stopping_rounds, use_best_model=use_best_model,
                save_snapshot=save_snapshot, snapshot_file=snapshot_file, snapshot_interval=snapshot_interval,
                resume=resume, callback=callback, init_model=init_model, ranker=True)
        if init_model is not None:
            raise ValueError("init_model requires native FeatureParallel training.")
        from catboost import CatBoostRanker, Pool

        started = time.perf_counter()
        self._model = None
        paired = self._objective in PAIR_OBJECTIVES
        coupled = self._objective == "PairLogitPairwise"
        yeti_pair = self._objective == "YetiRankPairwise"
        qce = self._objective == "QueryCrossEntropy"
        if not paired and any(value is not None for value in (pairs, pairs_weight, eval_pairs, eval_pairs_weight)):
            raise ValueError("Explicit pairs require PairLogit or PairLogitPairwise.")
        if paired:
            pairs, pairs_weight = unpack_pairs(X, pairs, pairs_weight)
        raw, labels, offsets, weights, cats, names = unpack_query_pool(
            X, y, sample_weight, group_id, group_weight,
            self.cat_features if cat_features is None else cat_features, combine_weights=not paired or coupled)
        subgroup_hashes = unpack_subgroups(X, subgroup_id, len(raw))
        ranking_metric = self.eval_metric is not None and self.eval_metric.partition(":")[0] in _RANKING_METRICS
        if ranking_metric and eval_set is None and labels is None:
            raise ValueError("Ranking evaluation requires relevance labels on the evaluated data.")
        targets = (np.zeros(len(raw), np.float32) if labels is None and paired else numeric_array(labels, "y", 1))
        if targets.shape != (len(raw),):
            raise ValueError("X must have exactly one query target per row.")
        train_pairs = prepare_pairs(pairs, pairs_weight, len(raw), offsets) if paired else None
        if qce:
            from ._query_cross_entropy import select_scales
            select_scales(self._loss_parameters.get("raw_values_scale", ""), targets, offsets)
        if not paired and self._objective not in ("YetiRank", "YetiRankPairwise") and not qce:
            query_metric(np.zeros(len(targets)), targets, weights, offsets, self._objective,
                         self.query_beta, self.query_lambda)
        eval_bins = eval_targets = eval_weights = eval_offsets = eval_subgroup_hashes = None
        validation_pairs = None
        eval_raw = None
        if eval_set is not None:
            if isinstance(eval_set, Pool):
                if paired:
                    eval_pairs, eval_pairs_weight = unpack_pairs(eval_set, eval_pairs, eval_pairs_weight)
                evaluation = unpack_query_pool(eval_set, sample_weight=eval_sample_weight,
                    group_id=eval_group_id, group_weight=eval_group_weight, combine_weights=not paired or coupled)
                eval_subgroup_hashes = unpack_subgroups(eval_set, eval_subgroup_id, len(evaluation[0]))
            elif isinstance(eval_set, tuple) and 2 <= len(eval_set) <= 6:
                extras = [eval_group_id, eval_sample_weight, eval_group_weight, eval_subgroup_id]
                for index, value in enumerate(eval_set[2:]):
                    if extras[index] is not None:
                        raise ValueError("Supply evaluation group IDs, subgroup IDs and weights in the tuple or keywords, not both.")
                    extras[index] = value
                evaluation = unpack_query_pool(eval_set[0], eval_set[1], extras[1], extras[0], extras[2], cats,
                                                combine_weights=not paired or coupled)
                if hasattr(eval_set[0], "columns") and evaluation[5] != names:
                    raise ValueError("Validation feature names/order must match training data.")
                eval_subgroup_hashes = unpack_subgroups(eval_set[0], extras[3], len(evaluation[0]))
            else:
                raise ValueError("eval_set must be one grouped Pool or an (X, y[, group_id[, weight[, group_weight[, subgroup_id]]]]) tuple.")
            eval_raw, eval_y, eval_offsets, eval_weights, eval_cats, _ = evaluation
            if ranking_metric and eval_y is None:
                raise ValueError("Ranking evaluation requires relevance labels on the evaluated data.")
            if eval_cats != cats:
                raise ValueError("Validation categorical features must match training data.")
            if eval_raw.shape[1] != raw.shape[1]:
                raise ValueError("Validation feature count must match training data.")
            eval_targets = (np.zeros(len(eval_raw), np.float32) if eval_y is None and paired
                            else numeric_array(eval_y, "eval_y", 1))
            if paired:
                validation_pairs = prepare_pairs(eval_pairs, eval_pairs_weight, len(eval_raw), eval_offsets)
            elif qce:
                select_scales(self._loss_parameters.get("raw_values_scale", ""), eval_targets, eval_offsets)
            elif self._objective not in ("YetiRank", "YetiRankPairwise"):
                query_metric(np.zeros(len(eval_targets)), eval_targets, eval_weights, eval_offsets,
                             self._objective, self.query_beta, self.query_lambda)
        elif any(value is not None for value in (eval_group_id, eval_group_weight, eval_sample_weight, eval_subgroup_id,
                                                 eval_pairs, eval_pairs_weight)):
            raise ValueError("Evaluation group IDs and weights require eval_set.")
        categorical_metadata = {}
        if cats:
            from ._categorical import cat_feature_hashes
            eval_hashes = {}
            for feature in cats:
                hashes = cat_feature_hashes(raw[:, feature])
                if eval_raw is not None:
                    heldout = cat_feature_hashes(eval_raw[:, feature])
                    eval_hashes[str(feature)] = hashlib.sha256(heldout.tobytes()).hexdigest()
                    hashes = np.concatenate((hashes, heldout))
                if np.unique(hashes).size > self.one_hot_max_size:
                    raise ValueError("Standalone ranking supports one-hot categorical features only; "
                                     "learn plus validation categories exceed one_hot_max_size. "
                                     "Grouped CTR histories are not yet connected.")
            if eval_hashes:
                categorical_metadata["eval_categorical_hashes"] = eval_hashes
        layout, bins, candidate_features, candidate_bins, candidate_types = prepare_features(
            raw, self.border_count, self.nan_mode, cats, names, self.one_hot_max_size)
        if cats:
            categorical_metadata["categorical_hashes"] = {
                str(feature): list(encoding.hashes) for feature, encoding in layout.categorical.items()}
        if eval_raw is not None:
            eval_bins = layout.transform(eval_raw)
        prepared = time.perf_counter()
        native = dict(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
                      l2_leaf_reg=self.l2_leaf_reg, bias=0., score_function=self.score_function,
                      objective=self._objective, sample_weight=weights, group_offsets=offsets,
                      query_beta=self.query_beta, query_lambda=self.query_lambda,
                      candidate_types=candidate_types, leaf_estimation_method=self.leaf_estimation_method,
                      leaf_estimation_iterations=self.leaf_estimation_iterations,
                      leaf_estimation_backtracking=self.leaf_estimation_backtracking,
                      random_seed=self.random_seed, random_strength=self.random_strength,
                      bootstrap_type=self.bootstrap_type, bagging_temperature=self.bagging_temperature,
                      subsample=self.subsample, mvs_reg=self.mvs_reg)
        if self.feature_weights is not None:
            native["feature_weights"] = self._feature_weights(names, layout)
        if coupled or qce or yeti_pair:
            native["non_diagonal_regularization"] = self.bayesian_matrix_reg
        if qce:
            native["query_loss_description"] = self.loss_function
        if self._objective == "YetiRank":
            native.pop("query_beta")
            native.pop("query_lambda")
            native.update(permutations=self._loss_parameters.get("permutations", 10),
                          decay=self._loss_parameters.get("decay", .85),
                          legacy_prefix_centering=self.yeti_legacy_prefix_centering)
        if self.boosting_type == "Ordered":
            native.update(boosting_type="Ordered", permutation_count=self.permutation_count or 4,
                          fold_len_multiplier=self.fold_len_multiplier, min_fold_size=self.min_fold_size,
                          fold_permutation_block=self.fold_permutation_block,
                          fold_size_loss_normalization=self.fold_size_loss_normalization)
            if self._objective == "YetiRank":
                native["yeti_permutations"] = native.pop("permutations")
        if yeti_pair:
            native.pop("query_beta")
            native.pop("query_lambda")
            native.update(permutations=self._loss_parameters.get("permutations", 10),
                          decay=self._loss_parameters.get("decay", .85),sampling_unit=self.sampling_unit)
        if paired:
            native.update(pair_winners=train_pairs[0], pair_losers=train_pairs[1], pair_weights=train_pairs[2])
        evaluation_options = {} if validation_pairs is None else dict(
            eval_pair_winners=validation_pairs[0], eval_pair_losers=validation_pairs[1],
            eval_pair_weights=validation_pairs[2])
        greedy = self.grow_policy != "SymmetricTree"
        train = run_training
        if greedy:
            from ._greedy_training import run_training as train
            native.pop("mvs_reg")
            if paired:
                native.pop("query_beta"); native.pop("query_lambda")
            native.update(grow_policy=self.grow_policy, max_leaves=self.max_leaves, min_data_in_leaf=self.min_data_in_leaf)
        result = train(bins, targets, candidate_features, candidate_bins, **native,
            eval_bins=eval_bins, eval_targets=eval_targets, eval_weight=eval_weights, eval_group_offsets=eval_offsets,
            subgroup_hashes=subgroup_hashes, eval_subgroup_hashes=eval_subgroup_hashes,
            eval_metric=self.eval_metric, early_stopping_rounds=early_stopping_rounds, use_best_model=use_best_model,
            save_snapshot=save_snapshot, snapshot_file=snapshot_file, snapshot_interval=snapshot_interval,
            resume=resume, callback=callback, **evaluation_options,
            metadata={"borders": [value.tolist() for value in layout.borders], "has_nans": layout.has_nans,
                      "nan_mode": layout.nan_mode, "feature_names": names,
                      "query_weight_semantics": ("original_edges_and_documents" if coupled else
                          "supplied_literal_incident_mass" if paired else "object_times_group")} | categorical_metadata)
        result.stats.update(native_wall_seconds=time.perf_counter() - prepared,
                            preprocessing_seconds=prepared - started, groups=len(offsets) - 1)
        if greedy:
            from ._greedy_model import model_json, dumps_model_json
            model_data = model_json(result, layout.borders, objective=self._objective, feature_names=names,
                grow_policy=self.grow_policy, layout=layout, loss_parameters=self._loss_parameters,
                leaf_estimation_method=self.leaf_estimation_method)
        else:
            model_data = _model_json(layout.borders, result, 0., self.score_function, layout)
        bootstrap = {"type": self.bootstrap_type}
        if yeti_pair: bootstrap["sampling_unit"] = self.sampling_unit
        if self.bootstrap_type == "Bayesian":
            bootstrap["bagging_temperature"] = self.bagging_temperature
        elif self.bootstrap_type != "No":
            bootstrap["subsample"] = self.subsample
            if self.bootstrap_type == "MVS" and self.mvs_reg is not None:
                bootstrap["mvs_reg"] = self.mvs_reg
        model_data["model_info"].update(_training_metadata(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg, border_count=self.border_count, score_function=self.score_function,
            objective=self._objective, loss_description=self.loss_function, loss_parameters=self._loss_parameters,
            boost_from_average=False, leaf_estimation_method=self.leaf_estimation_method,
            leaf_estimation_iterations=self.leaf_estimation_iterations,
            leaf_estimation_backtracking=self.leaf_estimation_backtracking, nan_mode=self.nan_mode,
            categorical=bool(cats), one_hot_max_size=self.one_hot_max_size,
            eval_metric=self.eval_metric, use_best_model=eval_set is not None if use_best_model is None else use_best_model,
            bootstrap=bootstrap, random_seed=self.random_seed, random_strength=self.random_strength,
            grow_policy=self.grow_policy, max_leaves=self.max_leaves, min_data_in_leaf=self.min_data_in_leaf,
            boosting_type=self.boosting_type,
            permutation_count=(self.permutation_count or 4) if self.boosting_type == "Ordered" else 1,
            fold_len_multiplier=self.fold_len_multiplier, min_fold_size=self.min_fold_size,
            fold_permutation_block=self.fold_permutation_block,
            fold_size_loss_normalization=self.fold_size_loss_normalization, feature_weights=self._feature_weights(names)))
        if coupled or qce or yeti_pair:
            parameters = json.loads(model_data["model_info"]["params"])
            parameters["tree_learner_options"]["bayesian_matrix_reg"] = self.bayesian_matrix_reg
            parameters["flat_params"]["bayesian_matrix_reg"] = self.bayesian_matrix_reg
            model_data["model_info"]["params"] = json.dumps(parameters, allow_nan=False)
            model_data["model_info"]["metal_pair_bootstrap"] = "item_iteration_domains_v1" if yeti_pair else "query_index_absolute_iteration" if qce else "edge_index_absolute_iteration"
        with tempfile.TemporaryDirectory(prefix="catbooster-query-model-") as temporary:
            path = Path(temporary) / "model.json"
            path.write_text(dumps_model_json(model_data) if greedy else json.dumps(model_data, allow_nan=False))
            model = CatBoostRanker().load_model(str(path), format="json")
        self._model, self._result, self._layout = model, result, layout
        self.borders_, self.bias_ = layout.borders, 0.
        self.n_features_in_, self.feature_names_ = raw.shape[1], names
        if greedy:
            from ._greedy_inference import tree_depth
            self.tree_count_ = len(result.trees)
            self.tree_depths_ = np.array([tree_depth(tree, bins.shape[0], self.depth) for tree in result.trees], np.uint32)
        else:
            self.tree_count_, self.tree_depths_ = len(result.depths), result.depths.copy()
        self.training_predictions_ = result.predictions.copy()
        self.loss_history_, self.training_stats_ = (result.loss if greedy else result.rmse).tolist(), result.stats.copy()
        self.training_stats_["fit_wall_seconds"] = time.perf_counter() - started
        self.group_offsets_ = offsets.copy()
        if paired:
            self.pairs_ = np.column_stack(train_pairs[:2])
            self.pairs_weight_ = train_pairs[2].copy()
            self.training_stats_["supplied_pairs"] = len(self.pairs_)
        self.best_iteration_, self.best_score_, self.evals_result_ = result.best_iteration, result.best_score, result.evals_result
        return self

    def predict(self, X, prediction_type="RawFormulaVal", *, task_type="CPU", ntree_start=0, ntree_end=0):
        if prediction_type not in (None, "RawFormulaVal"):
            raise ValueError("A ranker predicts raw relevance scores.")
        if self._native_bridge_fitted:
            return super().predict(X, prediction_type="RawFormulaVal", task_type=task_type,
                                   ntree_start=ntree_start, ntree_end=ntree_end)
        if task_type == "CPU":
            self._require_fitted()
            bins = self._layout.transform(X)
            if not bins.shape[1]:
                return np.empty(0, np.float64)
            return self._model.predict(X, ntree_start=ntree_start, ntree_end=ntree_end)
        return super().predict(X, prediction_type="RawFormulaVal", task_type=task_type,
                               ntree_start=ntree_start, ntree_end=ntree_end)
