"""Greedy training lifecycle with GPU validation and numeric snapshot arrays."""
import copy
from contextlib import ExitStack
from dataclasses import replace
import json
import numbers
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import zipfile

import numpy as np

from . import _greedy
from ._greedy_inference import EvaluationCursor, predict_bins, tree_depth
from ._data import cuda_search_permutation
from ._training import (_fingerprint, _json, _vector, _initial_cursor, metric,
                        _metric_direction, _shared_metric, _METRIC_SEMANTICS_VERSION)


_VERSION = 1
_VECTOR_OBJECTIVES = ("MultiClass", "MultiClassOneVsAll", "RMSEWithUncertainty")


def _pack(result):
    nodes = [tree.nodes for tree in result.trees]
    values = [tree.leaf_values for tree in result.trees]
    return dict(node_offsets=np.r_[0, np.cumsum([len(v) for v in nodes])].astype(np.uint64),
        leaf_offsets=np.r_[0, np.cumsum([len(v) for v in values])].astype(np.uint64),
        nodes=np.concatenate(nodes).astype(np.uint32, copy=False),
        leaf_values=np.concatenate(values).astype(np.float32, copy=False),
        leaf_weights=np.concatenate([tree.leaf_weights for tree in result.trees]).astype(np.float32, copy=False),
        predictions=np.asarray(result.predictions, np.float32), loss=np.asarray(result.loss, np.float32))


def _write_snapshot(path, fingerprint, result, history, best_iteration, best_value, eval_raw, permutation_predictions=None,
                    optimization_predictions=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _pack(result)
    if permutation_predictions is not None:
        arrays["permutation_predictions"] = np.asarray(permutation_predictions, np.float32)
    if optimization_predictions is not None:
        arrays["optimization_predictions"] = np.asarray(optimization_predictions, np.float32)
    if eval_raw is not None:
        arrays["eval_predictions"] = np.asarray(eval_raw, np.float32)
    header = dict(version=_VERSION, backend="MetalGreedy", fingerprint=fingerprint,
        completed_iterations=len(result.trees), history=history, best_iteration=best_iteration,
        best_value=best_value if np.isfinite(best_value) else None, stats=result.stats)
    header["checksum"] = _fingerprint(arrays, header)
    arrays["metadata"] = np.asarray(_json(header))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            np.savez(output, **arrays); output.flush(); os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_snapshot(path, fingerprint, *, rows, features, depth, max_leaves, iterations,
                   objective, eval_rows=None, selection_metric=None, maximize=False, permutation_count=1, classes=None):
    selection_metric = selection_metric or objective
    try:
        with zipfile.ZipFile(path) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > 1 << 30:
                raise ValueError("Greedy snapshot exceeds the 1 GiB expanded size limit.")
        with np.load(path, allow_pickle=False) as archive:
            metadata = archive["metadata"]
            if metadata.shape != () or metadata.dtype.kind != "U":
                raise ValueError("Invalid greedy snapshot metadata.")
            header = json.loads(metadata.item())
            if not isinstance(header, dict):
                raise ValueError("Invalid greedy snapshot metadata object.")
            if header.get("version") != _VERSION or header.get("backend") != "MetalGreedy":
                raise ValueError("Unsupported greedy snapshot version or backend.")
            if header.get("fingerprint") != fingerprint:
                raise ValueError("Snapshot does not match training data, borders, or parameters.")
            count = header["completed_iterations"]
            if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= iterations:
                raise ValueError("Invalid snapshot tree count or requested iteration count.")
            names = {"node_offsets", "leaf_offsets", "nodes", "leaf_values", "leaf_weights", "predictions", "loss"}
            if permutation_count > 1 or classes: names.add("permutation_predictions")
            if classes: names.add("optimization_predictions")
            if eval_rows is not None: names.add("eval_predictions")
            if set(archive.files) != names | {"metadata"}:
                raise ValueError("Invalid greedy snapshot array names.")
            arrays = {name: archive[name].copy() for name in names}
        checksum = header.pop("checksum", None)
        if checksum != _fingerprint(arrays, header):
            raise ValueError("Greedy snapshot checksum mismatch.")
        for name in ("node_offsets", "leaf_offsets"):
            value = arrays[name]
            if (value.dtype != np.dtype("uint64") or value.shape != (count + 1,) or
                    value[0] != 0 or (value[1:] <= value[:-1]).any()):
                raise ValueError("Invalid greedy snapshot offsets.")
        node_count, leaf_count = int(arrays["node_offsets"][-1]), int(arrays["leaf_offsets"][-1])
        specs = {"nodes": ((node_count, 6), "uint32"), "leaf_values": (((leaf_count, classes) if classes else (leaf_count,)), "float32"),
                 "leaf_weights": ((leaf_count,), "float32"), "predictions": (((rows, classes) if classes else (rows,)), "float32"),
                 "loss": ((count + 1,), "float32")}
        if eval_rows is not None:
            specs["eval_predictions"] = (((eval_rows, classes) if classes else (eval_rows,)), "float32")
        if permutation_count > 1 or classes:
            specs["permutation_predictions"] = (((permutation_count, rows, classes) if classes
                                                  else (permutation_count, rows)), "float32")
        if classes:
            active = classes - int(objective == "MultiClass")
            specs["optimization_predictions"] = ((permutation_count, active, rows), "float32")
        for name, (shape, dtype) in specs.items():
            value = arrays[name]
            if value.dtype != np.dtype(dtype) or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid greedy snapshot array: {name}.")
        if permutation_count > 1 or classes:
            if not np.array_equal(arrays["predictions"], arrays["permutation_predictions"][-1]):
                raise ValueError("Greedy snapshot export cursor differs from its final dataset.")
            header["permutation_predictions"] = arrays["permutation_predictions"]
        if classes:
            header["optimization_predictions"] = arrays["optimization_predictions"]
        if ((arrays["leaf_weights"] < 0).any() or
                (objective.partition(":")[0] not in ("Poisson", "RMSEWithUncertainty") and (arrays["loss"] < 0).any())):
            raise ValueError("Invalid greedy snapshot weights or loss.")
        stats = header["stats"]
        if not isinstance(stats, dict) or not isinstance(stats.get("device", ""), str):
            raise ValueError("Invalid greedy snapshot runtime statistics.")
        for name in ("gpu_seconds", "kernel_dispatches"):
            value = stats.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value < 0:
                raise ValueError("Invalid greedy snapshot runtime statistics.")
        trees = []
        for index in range(count):
            a, b = map(int, arrays["node_offsets"][index:index + 2])
            c, d = map(int, arrays["leaf_offsets"][index:index + 2])
            if d - c > max_leaves:
                raise ValueError("Snapshot leaf count exceeds max_leaves.")
            tree = _greedy.StepResult(index + 1, index + 1 == count, arrays["nodes"][a:b],
                arrays["leaf_values"][c:d], arrays["leaf_weights"][c:d], float(arrays["loss"][index + 1]), stats.copy())
            tree_depth(tree, features, depth)
            trees.append(tree)
        history = header["history"]
        expected = {"learn", "validation"} if eval_rows is not None else {"learn"}
        if not isinstance(history, dict) or set(history) != expected:
            raise ValueError("Invalid snapshot metric datasets.")
        for dataset in expected:
            metrics = {objective, selection_metric} if dataset == "validation" or eval_rows is None else {objective}
            if set(history[dataset]) != metrics:
                raise ValueError("Invalid snapshot metric names.")
            for name in metrics:
                values = history[dataset][name]
                if not isinstance(values, list) or len(values) != count or not np.isfinite(values).all():
                    raise ValueError("Invalid snapshot metric history.")
        if not np.array_equal(np.asarray(history["learn"][objective], np.float32), arrays["loss"][1:]):
            raise ValueError("Snapshot learn history disagrees with its loss cursor.")
        best = header["best_iteration"]
        if isinstance(best, bool) or not isinstance(best, int) or not -1 <= best < count:
            raise ValueError("Invalid snapshot best iteration.")
        if eval_rows is not None:
            selected = history["validation"][selection_metric]
            actual_best = int(np.argmax(selected) if maximize else np.argmin(selected))
            if best != actual_best or header["best_value"] != selected[best]:
                raise ValueError("Snapshot best score disagrees with metric history.")
        elif best != -1 or header["best_value"] is not None:
            raise ValueError("Invalid snapshot best score without validation.")
        result = _greedy.TrainResult(tuple(trees), arrays["predictions"], arrays["loss"], stats.copy())
        return result, header, arrays.get("eval_predictions")
    except (OSError, KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise ValueError(f"Could not read greedy snapshot {path}: {error}") from error


def _merge(prior, segment):
    if prior is None: return segment
    count = len(prior.trees)
    result = _greedy.TrainResult(prior.trees + tuple(replace(tree, completed_iterations=count + i + 1)
        for i, tree in enumerate(segment.trees)), segment.predictions.copy(),
        np.r_[prior.loss, segment.loss[1:]].astype(np.float32), segment.stats.copy())
    for name in ("kernel_dispatches", "gpu_seconds"):
        result.stats[name] = prior.stats.get(name, 0) + segment.stats.get(name, 0)
    return result


def run_training(bins, targets, candidate_features, candidate_bins, *, iterations,
                 depth, learning_rate, l2_leaf_reg, bias, score_function,
                 grow_policy="Lossguide", max_leaves=None, min_data_in_leaf=1,
                 objective="RMSE", sample_weight=None, eval_bins=None, eval_targets=None,
                 eval_weight=None, early_stopping_rounds=None, use_best_model=None,
                 eval_metric=None, save_snapshot=False, snapshot_file=None, snapshot_interval=600.,
                 resume=True, metadata=None, callback=None, permutation_bins=None,
                 ctr_unique_values=None, model_size_reg=.5, feature_weights=None,
                 boosting_type="Plain", group_offsets=None, eval_group_offsets=None,
                 query_beta=1., query_lambda=.01, subgroup_hashes=None, eval_subgroup_hashes=None,
                 pair_winners=None, pair_losers=None, pair_weights=None,
                 eval_pair_winners=None, eval_pair_losers=None, eval_pair_weights=None, **native_options):
    """Same fit lifecycle contract as _training, with variable-node trees.

    Snapshots keep the full untrimmed cursor, numeric flat arrays and offsets.
    Retained-model predictions are rebuilt on the GPU after best-model trimming.
    No pickle, CPU fitting, or CPU row traversal is used. Metrics use the shared
    CatBoost utilities. Supports scalar objectives and CUDA-registered vector
    objectives, Plain, No/Bayesian/Bernoulli/Poisson bootstrap, and shared-grid CTR datasets.
    Leaf updates support No, AnyImprovement, and Armijo backtracking.
    """
    if objective not in (*_greedy.OBJECTIVES, *_VECTOR_OBJECTIVES) or boosting_type != "Plain":
        raise ValueError("Greedy lifecycle requires a CUDA-registered scalar or vector objective with Plain boosting.")
    grouped = objective in ("QueryRMSE", "QuerySoftMax")
    paired = objective == "PairLogit"
    if not (grouped or paired) and any(value is not None for value in (group_offsets, eval_group_offsets, subgroup_hashes, eval_subgroup_hashes)):
        raise ValueError("Greedy query grouping requires QueryRMSE, QuerySoftMax or PairLogit.")
    if not grouped and (query_beta != 1 or query_lambda != .01):
        raise ValueError("Query parameters require QueryRMSE or QuerySoftMax.")
    if paired and sample_weight is not None:
        raise ValueError("PairLogit training uses literal incident pair mass; sample_weight must be None.")
    if not paired and any(v is not None for v in (pair_winners,pair_losers,pair_weights,eval_pair_winners,eval_pair_losers,eval_pair_weights)):
        raise ValueError("Supplied pair arrays require PairLogit.")
    classes, session_type = None, _greedy.TrainingSession
    if objective in _VECTOR_OBJECTIVES:
        from ._multiclass import Session
        session_type = Session
        classes = _greedy._integer("classes", native_options.get("classes", 2 if objective == "RMSEWithUncertainty" else None), 2, 64)
        if objective == "RMSEWithUncertainty" and classes != 2:
            raise ValueError("RMSEWithUncertainty requires exactly two outputs.")
        if native_options.get("objective_param") is not None:
            raise ValueError("Vector objectives do not accept objective_param.")
        native_options = {**native_options, "classes": classes}
    else:
        parameter = _greedy.objective_parameter(objective, native_options.get("objective_param"))
    objective_metric = objective
    if objective in _greedy.OBJECTIVE_PARAMETERS:
        display_parameter = native_options.get("objective_param")
        if display_parameter is None:
            display_parameter = parameter
        objective_metric = f"{objective}:{_greedy.OBJECTIVE_PARAMETERS[objective]}={float(display_parameter)!r}"
    if grouped:
        query_beta = float(_greedy._finite_array("query_beta", query_beta, ()))
        query_lambda = float(_greedy._finite_array("query_lambda", query_lambda, ()))
        if objective == "QuerySoftMax" and (query_beta != 1 or np.float32(query_lambda) != np.float32(.01)):
            objective_metric = f"QuerySoftMax:beta={query_beta!r};lambda={query_lambda!r}"
    if ctr_unique_values is not None or feature_weights is not None:
        raise ValueError("Greedy lifecycle does not support feature penalty overrides.")
    iterations = _greedy._integer("iterations", iterations, 1, 100000)
    if grow_policy not in ("Depthwise", "Lossguide", "Region"):
        raise ValueError("Invalid greedy grow_policy.")
    depth = _greedy._integer("depth", depth, 0, {"Depthwise": 16, "Region": 65535, "Lossguide": 2**32 - 1}[grow_policy])
    capacity = (depth + 1 if grow_policy == "Region" else
                (31 if grow_policy == "Lossguide" else 1 << depth)) if max_leaves is None else max_leaves
    capacity = _greedy._integer("max_leaves", capacity, 1, 65536)
    if classes:
        capacity = (depth + 1 if grow_policy == "Region" else 1 << depth if grow_policy == "Depthwise"
                    else min(capacity, 1 << min(depth, 16)))
    effective_depth = min(depth, capacity - 1)
    if early_stopping_rounds is not None:
        early_stopping_rounds = _greedy._integer("early_stopping_rounds", early_stopping_rounds, 1, 100000)
    for name, value in (("save_snapshot", save_snapshot), ("resume", resume)):
        if not isinstance(value, bool): raise ValueError(f"{name} must be boolean.")
    if use_best_model is not None and not isinstance(use_best_model, bool):
        raise ValueError("use_best_model must be boolean or None.")
    if (isinstance(snapshot_interval, bool) or not isinstance(snapshot_interval, numbers.Real) or
            not np.isfinite(snapshot_interval) or snapshot_interval < 0):
        raise ValueError("snapshot_interval must be finite and nonnegative.")
    if callback is not None and not callable(callback): raise TypeError("callback must be callable.")
    has_eval = (eval_bins is not None or eval_targets is not None or eval_weight is not None or eval_group_offsets is not None
        or any(v is not None for v in (eval_pair_winners,eval_pair_losers,eval_pair_weights)))
    if has_eval and (eval_bins is None or eval_targets is None):
        raise ValueError("Validation bins and targets must both be supplied.")
    if not has_eval and (early_stopping_rounds is not None or use_best_model is True):
        raise ValueError("Early stopping and use_best_model require validation data.")
    keep_best = has_eval if use_best_model is None else use_best_model
    bins = np.asarray(bins)
    if (bins.ndim != 2 or bins.dtype.kind not in "iu" or not all(bins.shape) or
            (bins < 0).any() or (bins > 255).any()):
        raise ValueError("bins must be a nonempty feature-major byte-valued integer matrix.")
    bins = np.ascontiguousarray(bins, np.uint8)
    features, rows = bins.shape
    permutation_count = 1
    if permutation_bins is not None:
        matrices = np.asarray(permutation_bins)
        if (matrices.ndim != 3 or not 1 <= len(matrices) <= 64 or matrices.shape[1:] != bins.shape
                or matrices.dtype.kind not in "iu" or (matrices < 0).any() or (matrices > 255).any()
                or not np.array_equal(matrices[0], bins)):
            raise ValueError("Permutation bins must share the original geometry and dataset zero.")
        permutation_count = len(matrices)
        permutation_bins = np.ascontiguousarray(matrices, np.uint8) if permutation_count > 1 else None
    targets = _vector(targets, rows, "targets", objective=objective, classes=classes)
    if sample_weight is not None: sample_weight = _vector(sample_weight, rows, "sample_weight", weight=True)
    if has_eval:
        eval_bins = np.asarray(eval_bins)
        if (eval_bins.ndim != 2 or eval_bins.shape[0] != features or not eval_bins.shape[1] or
                eval_bins.dtype.kind not in "iu" or (eval_bins < 0).any() or (eval_bins > 255).any()):
            raise ValueError("Validation bins must have matching features and byte-valued nonempty rows.")
        eval_bins = np.ascontiguousarray(eval_bins, np.uint8)
        eval_targets = _vector(eval_targets, eval_bins.shape[1], "eval_targets", objective=objective, classes=classes)
        if eval_weight is not None: eval_weight = _vector(eval_weight, len(eval_targets), "eval_weight", weight=True)
    if grouped:
        from ._query_data import validate_offsets, validate_subgroup_hashes, query_metric
        group_offsets = validate_offsets(group_offsets, rows)
        subgroup_hashes = validate_subgroup_hashes(subgroup_hashes, rows)
        query_metric(np.zeros(rows), targets, sample_weight, group_offsets, objective, query_beta, query_lambda)
        if has_eval:
            eval_group_offsets = validate_offsets(eval_group_offsets, len(eval_targets), "eval_group_offsets")
            eval_subgroup_hashes = validate_subgroup_hashes(eval_subgroup_hashes, len(eval_targets), "eval_subgroup_hashes")
            query_metric(np.zeros(len(eval_targets)), eval_targets, eval_weight, eval_group_offsets, objective, query_beta, query_lambda)
        elif eval_subgroup_hashes is not None:
            raise ValueError("Validation subgroup hashes require validation data.")
    train_pairs = eval_pairs = None
    if paired:
        from ._query_data import validate_offsets, validate_subgroup_hashes, prepare_pair_arrays
        if group_offsets is not None: group_offsets = validate_offsets(group_offsets, rows)
        subgroup_hashes = validate_subgroup_hashes(subgroup_hashes, rows)
        train_pairs = prepare_pair_arrays(pair_winners,pair_losers,pair_weights,rows,group_offsets)
        if has_eval:
            if eval_group_offsets is not None: eval_group_offsets = validate_offsets(eval_group_offsets,len(eval_targets),"eval_group_offsets")
            eval_subgroup_hashes = validate_subgroup_hashes(eval_subgroup_hashes,len(eval_targets),"eval_subgroup_hashes")
            eval_pairs = prepare_pair_arrays(eval_pair_winners,eval_pair_losers,eval_pair_weights,len(eval_targets),eval_group_offsets)
        elif eval_subgroup_hashes is not None:
            raise ValueError("Validation subgroup hashes require validation data.")
    selection = objective_metric if eval_metric is None else eval_metric
    maximize = False if eval_metric is None else _metric_direction(selection)
    if eval_metric is not None:
        selected_y, selected_w = (eval_targets, eval_weight) if has_eval else (targets, sample_weight)
        _shared_metric(selection, _initial_cursor(len(selected_y), bias, classes), selected_y, selected_w,
            eval_group_offsets if has_eval else group_offsets, eval_pairs if has_eval else train_pairs,
            subgroup_hashes=eval_subgroup_hashes if has_eval else subgroup_hashes)
    supplied_options = copy.deepcopy(native_options)
    random_seed = native_options.pop("random_seed", 0)
    _greedy._integer("random_seed", random_seed, 0, 2**64 - 1)
    offset = native_options.pop("iteration_offset", 0)
    _greedy._integer("iteration_offset", offset, 0, 2**32 - 1 - iterations)
    params = dict(iterations=iterations, depth=depth, max_leaves=capacity, min_data_in_leaf=min_data_in_leaf,
        learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg, bias=bias, score_function=score_function,
        objective=objective, sample_weight=sample_weight, grow_policy=grow_policy, boosting_type=boosting_type,
        random_seed=random_seed, iteration_offset=offset,
        **native_options)
    if grouped:
        params.update(group_offsets=group_offsets, query_beta=query_beta, query_lambda=query_lambda)
    if paired:
        params.update(group_offsets=group_offsets,pair_winners=train_pairs[0],pair_losers=train_pairs[1],pair_weights=train_pairs[2])
    if classes:
        params.pop("boosting_type")
    path = Path(snapshot_file or "catboost_metal_greedy.snapshot.npz") if save_snapshot else None
    fingerprint = None
    if path is not None:
        settings = dict(depth=depth, max_leaves=capacity, min_data_in_leaf=min_data_in_leaf,
            grow_policy=grow_policy, learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg, bias=bias,
            score_function=score_function, objective=objective, early_stopping_rounds=early_stopping_rounds,
            use_best_model=keep_best, metadata=metadata, native_options=supplied_options,
            selection_metric=selection, metric_semantics_version=_METRIC_SEMANTICS_VERSION)
        fingerprint_arrays = dict(bins=bins, targets=targets, candidate_features=candidate_features,
            candidate_bins=candidate_bins, sample_weight=sample_weight, eval_bins=eval_bins,
            eval_targets=eval_targets, eval_weight=eval_weight)
        if grouped:
            settings.update(query_beta=query_beta, query_lambda=query_lambda)
            fingerprint_arrays.update(group_offsets=group_offsets, eval_group_offsets=eval_group_offsets,
                subgroup_hashes=subgroup_hashes, eval_subgroup_hashes=eval_subgroup_hashes)
        if paired:
            settings["pair_weight_semantics"] = "supplied_literal_incident_mass"
            fingerprint_arrays.update(group_offsets=group_offsets,eval_group_offsets=eval_group_offsets,
                subgroup_hashes=subgroup_hashes,eval_subgroup_hashes=eval_subgroup_hashes,
                pair_winners=train_pairs[0],pair_losers=train_pairs[1],pair_weights=train_pairs[2],
                eval_pair_winners=None if eval_pairs is None else eval_pairs[0],
                eval_pair_losers=None if eval_pairs is None else eval_pairs[1],
                eval_pair_weights=None if eval_pairs is None else eval_pairs[2])
        if permutation_count > 1:
            fingerprint_arrays["permutation_bins"] = permutation_bins
        fingerprint = _fingerprint(fingerprint_arrays, settings)
    history = {"learn": {objective_metric: []}}
    if has_eval: history["validation"] = {objective_metric: []}
    if selection != objective_metric: history["validation" if has_eval else "learn"][selection] = []
    prior, eval_raw = None, _initial_cursor(len(eval_targets), bias, classes) if has_eval else None
    permutation_predictions = optimization_predictions = None
    best, best_value = -1, float("-inf") if maximize else float("inf")
    if path is not None and resume and path.exists():
        prior, state, eval_raw = _read_snapshot(path, fingerprint, rows=rows, features=features, depth=depth,
            max_leaves=capacity, iterations=iterations, objective=objective_metric,
            eval_rows=len(eval_targets) if has_eval else None, selection_metric=selection, maximize=maximize,
            permutation_count=permutation_count, classes=classes)
        permutation_predictions = state.get("permutation_predictions")
        optimization_predictions = state.get("optimization_predictions")
        history, best = state["history"], state["best_iteration"]
        if has_eval: best_value = state["best_value"]
    resumed = len(prior.trees) if prior is not None else 0
    completed, reason, result = resumed, "iterations", prior
    already_stopped = bool(early_stopping_rounds is not None and resumed and
                           resumed - 1 - best >= early_stopping_rounds)
    if already_stopped: reason = "early_stopping"
    if completed < iterations and not already_stopped:
        params["iterations"] = iterations - resumed
        params["iteration_offset"] = offset + resumed
        if prior is not None:
            params["initial_predictions"] = prior.predictions
            if classes:
                params["initial_optimization_predictions"] = optimization_predictions[-1]
        last_snapshot = time.monotonic()
        with ExitStack() as resources:
            session = resources.enter_context(session_type(bins, targets, candidate_features, candidate_bins, **params))
            if permutation_count > 1:
                session.configure_permutations(permutation_bins, permutation_predictions,
                    **({"optimization_predictions": optimization_predictions} if classes else {}))
            cursor = resources.enter_context(EvaluationCursor(eval_bins, max_depth=effective_depth,
                initial_predictions=eval_raw)) if has_eval else None

            def current_result():
                merged = _merge(prior, session.result())
                if cursor:
                    statistics = cursor.stats
                    previous = {} if prior is None else prior.stats.get("validation", {})
                    for name in ("kernel_dispatches", "gpu_seconds", "dataset_uploads", "bins_upload_bytes", "tree_upload_bytes"):
                        statistics[name] += previous.get(name, 0)
                    merged.stats["validation"] = statistics
                return merged

            while completed < iterations:
                if permutation_count > 1:
                    session.select_permutation(cuda_search_permutation(random_seed, offset + completed, permutation_count))
                step = session.step(); completed += 1
                history["learn"][objective_metric].append(float(step.loss))
                if has_eval:
                    eval_raw = cursor.add_tree(step)
                    value = metric(eval_raw, eval_targets, eval_weight, objective_metric, group_offsets=eval_group_offsets,
                        **({} if eval_pairs is None else dict(pair_winners=eval_pairs[0],pair_losers=eval_pairs[1],pair_weights=eval_pairs[2])))
                    history["validation"][objective_metric].append(value)
                    if eval_metric is not None:
                        value = _shared_metric(selection, eval_raw, eval_targets, eval_weight, eval_group_offsets, eval_pairs, subgroup_hashes=eval_subgroup_hashes)
                        if selection == objective_metric: history["validation"][selection][-1] = value
                        else: history["validation"][selection].append(value)
                    if value > best_value if maximize else value < best_value:
                        best, best_value = completed - 1, value
                elif selection != objective_metric:
                    history["learn"][selection].append(_shared_metric(selection, session.predictions(), targets, sample_weight, group_offsets, train_pairs, subgroup_hashes=subgroup_hashes))
                if callback is not None and callback(SimpleNamespace(iteration=completed, metrics=copy.deepcopy(history))) is False:
                    reason = "callback"
                if early_stopping_rounds is not None and completed - 1 - best >= early_stopping_rounds:
                    reason = "early_stopping"
                if path is not None and time.monotonic() - last_snapshot >= snapshot_interval:
                    if permutation_count > 1 or classes:
                        state = session.permutation_state
                        permutation_predictions = state["predictions"]
                        optimization_predictions = state.get("optimization_predictions")
                    _write_snapshot(path, fingerprint, current_result(), history, best, best_value, eval_raw,
                                    permutation_predictions, optimization_predictions)
                    last_snapshot = time.monotonic()
                if reason != "iterations": break
            result = current_result()
            if path is not None and (permutation_count > 1 or classes):
                state = session.permutation_state
                permutation_predictions = state["predictions"]
                optimization_predictions = state.get("optimization_predictions")
    if path is not None:
        _write_snapshot(path, fingerprint, result, history, best, best_value, eval_raw,
                        permutation_predictions, optimization_predictions)
    if keep_best and best + 1 < len(result.trees):
        result.trees = result.trees[:best + 1]
        # A caller-provided training baseline is external to the trees. Restore
        # it exactly for trimming rather than replacing it with a scalar bias.
        initial = native_options.get("initial_predictions")
        estimation_bins = bins if permutation_count == 1 else permutation_bins[-1]
        with EvaluationCursor(estimation_bins, bias=bias, max_depth=effective_depth, initial_predictions=initial, dimensions=classes) as cursor:
            for tree in result.trees: cursor.add_tree(tree)
            result.predictions = cursor.predictions()
        eval_raw = predict_bins(eval_bins, result.trees, bias)
    result.best_iteration, result.evals_result = best, copy.deepcopy(history)
    result.best_score = {dataset: {name: (max(values) if name == selection and maximize else min(values))
        for name, values in metrics.items()} for dataset, metrics in history.items()}
    result.stopped_iteration = completed - 1
    result.eval_predictions = None if eval_raw is None else eval_raw.copy()
    result.trained_iterations, result.resumed_iterations = completed, resumed
    result.selected_metric, result.snapshot_file = selection, None if path is None else str(path)
    result.stats.update(iterations_trained=completed, iterations_retained=len(result.trees),
        resumed_iterations=resumed, stop_reason=reason, best_iteration=best,
        best_validation_score=best_value if has_eval else None, selection_metric=selection, metric_maximized=maximize)
    return result
