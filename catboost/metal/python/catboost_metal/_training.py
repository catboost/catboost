"""Iteration control for the Metal trainer, with portable, validated snapshots.

Tree construction stays in the persistent Metal session. This module handles
validation metrics, stopping, and serialization of completed training state.
Snapshots contain JSON and numeric arrays only; they never deserialize pickle.
"""

import copy
from contextlib import ExitStack
import hashlib
import json
import numbers
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from . import _native


_SNAPSHOT_VERSION = 1
_METRIC_SEMANTICS_VERSION = 2
_TREE_FIELDS = ("depths", "split_features", "split_bins", "split_types",
                "leaf_values", "leaf_weights")
_MULTICLASS_OBJECTIVES = ("MultiClass", "MultiClassOneVsAll")
_MULTIOUTPUT_OBJECTIVES = ("MultiRMSE", "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy")
_QUERY_OBJECTIVES = ("QueryRMSE", "QuerySoftMax")
_PAIR_OBJECTIVES = ("PairLogit", "PairLogitPairwise")
_RANKING_METRICS = ("NDCG", "MAP", "PFound")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Snapshot metadata cannot serialize {type(value).__name__}.")


def _json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=_json_value)


def _fingerprint(arrays, parameters):
    digest = hashlib.sha256(_json(parameters).encode("utf-8"))
    for name, array in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        if array is None:
            digest.update(b"null")
            continue
        array = np.ascontiguousarray(array)
        digest.update(_json([array.dtype.str, array.shape]).encode("utf-8"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _vector(value, rows, name, *, weight=False, objective="RMSE", classes=None):
    raw = np.asarray(value)
    matrix_target = not weight and objective in _MULTIOUTPUT_OBJECTIVES and objective != "RMSEWithUncertainty"
    shape = (rows, classes) if matrix_target else (rows,)
    if raw.shape != shape or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be numeric with shape {shape}.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(raw, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite float32 values.")
    if weight and ((array < 0).any() or not (array > 0).any()):
        raise ValueError(f"{name} must be nonnegative with positive total weight.")
    if not weight and objective in ("Logloss", "CrossEntropy", "MultiLogloss", "MultiCrossEntropy"):
        if (array < 0).any() or (array > 1).any():
            raise ValueError(f"{name} must be in [0, 1] for {objective}.")
        if objective in ("Logloss", "MultiLogloss") and not np.isin(array, (0, 1)).all():
            raise ValueError(f"{name} must be binary for {objective}.")
    if not weight and objective in ("Poisson", "Tweedie", "QuerySoftMax") and (array < 0).any():
        raise ValueError(f"{name} must be nonnegative for {objective}.")
    if not weight and objective in _MULTICLASS_OBJECTIVES:
        if (array != np.floor(array)).any() or (array < 0).any() or (array >= classes).any():
            raise ValueError(f"{name} must contain class indices in [0, {classes}).")
    return array


def _initial_cursor(rows, bias, classes=None):
    with np.errstate(over="ignore", invalid="ignore"):
        bias = np.asarray(bias, dtype=np.float32)
    if not np.isfinite(bias).all() or bias.shape not in ((), (classes,) if classes else ()):
        raise ValueError("bias must be finite and scalar, or one value per output dimension.")
    return np.broadcast_to(bias, (rows, classes) if classes else (rows,)).copy()


def metric(raw_predictions, targets, sample_weight=None, objective="RMSE", *, group_offsets=None,
           pair_winners=None, pair_losers=None, pair_weights=None, query_scales=None, subgroup_hashes=None, qce_session=None):
    """Evaluate weighted scalar and vector objectives with CUDA loss semantics."""
    raw = np.asarray(raw_predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if objective.partition(":")[0] == "QueryCrossEntropy":
        from . import _query_cross_entropy as qce
        parameters = qce.parse_description(objective, metric=True)
        if qce_session is not None:
            return qce_session.evaluate(raw,parameters.get("alpha",.95))
        scales = query_scales if query_scales is not None else qce.select_scales(parameters.get("raw_values_scale", ""), target, group_offsets)
        return qce.metric(raw, target, sample_weight, group_offsets, alpha=parameters.get("alpha", .95), query_scales=scales)
    if objective in _RANKING_METRICS:
        return _shared_metric(objective, raw, target, sample_weight, group_offsets, subgroup_hashes=subgroup_hashes)
    if objective in _PAIR_OBJECTIVES:
        from ._query_data import pair_metric
        return pair_metric(raw, pair_winners, pair_losers, pair_weights, "PairLogit")
    if objective.partition(":")[0] in _QUERY_OBJECTIVES:
        from ._query_data import query_metric
        base, _, arguments = objective.partition(":")
        parameters = dict(item.split("=", 1) for item in arguments.split(";") if "=" in item)
        return query_metric(raw, target, sample_weight, group_offsets, base,
                            float(parameters.get("beta", 1)), float(parameters.get("lambda", .01)))
    if objective == "RMSE":
        losses = np.square(raw - target)
    elif objective in ("Logloss", "CrossEntropy"):
        losses = np.logaddexp(0.0, raw) - target * raw
    elif objective in _MULTICLASS_OBJECTIVES:
        if raw.ndim != 2 or target.shape != (len(raw),):
            raise ValueError("Multiclass metrics need [rows, classes] predictions and one label per row.")
        labels = target.astype(np.int64)
        if (labels != target).any() or (labels < 0).any() or (labels >= raw.shape[1]).any():
            raise ValueError("Multiclass metric labels must be valid class indices.")
        selected = raw[np.arange(len(raw)), labels]
        if objective == "MultiClass":
            maximum = raw.max(axis=1)
            losses = (maximum - selected) + np.log(np.exp(raw - maximum[:, None]).sum(axis=1))
        else:
            binary_losses = np.logaddexp(0, raw)
            binary_losses[np.arange(len(raw)), labels] = np.logaddexp(0, -selected)
            losses = binary_losses.mean(axis=1)
    elif objective in _MULTIOUTPUT_OBJECTIVES:
        expected = (len(raw),) if objective == "RMSEWithUncertainty" else raw.shape
        if raw.ndim != 2 or target.shape != expected:
            raise ValueError("Multioutput metrics need matching row-major prediction and target dimensions.")
        if objective == "RMSEWithUncertainty":
            if raw.shape[1] != 2:
                raise ValueError("RMSEWithUncertainty needs two prediction dimensions.")
            with np.errstate(over="ignore", invalid="ignore"):
                # CUDA multilogit.cu clamps the inverse variance exponent.
                losses = (.5 * np.log(2 * np.pi) + raw[:, 1]
                          + .5 * np.square(target - raw[:, 0]) * np.exp(np.minimum(-2 * raw[:, 1], 70)))
        elif objective == "MultiRMSE":
            losses = np.square(raw - target).sum(axis=1)
        else:
            losses = (np.logaddexp(0.0, raw) - target * raw).mean(axis=1)
    elif objective.partition(":")[0] in ("Poisson", "Huber", "Expectile", "Lq", "Tweedie",
                                         "LogLinQuantile", "MAE", "Quantile", "MAPE"):
        return _shared_metric(objective, raw, target, sample_weight)
    else:
        raise ValueError(f"Unsupported lifecycle objective: {objective}.")
    value = float(np.average(losses, weights=sample_weight))
    if objective in ("RMSE", "MultiRMSE"):
        value = float(np.sqrt(value))
    if not np.isfinite(value):
        raise RuntimeError(f"Metal produced a non-finite {objective} metric.")
    return value


def _metric_direction(name):
    """Use the same registry as CatBoost's C++ metric selection logic."""
    from catboost import _catboost

    if not isinstance(name, str) or not name.strip():
        raise ValueError("eval_metric must be a nonempty CatBoost metric string.")
    try:
        maximize = bool(_catboost.is_maximizable_metric(name))
        minimize = bool(_catboost.is_minimizable_metric(name))
    except Exception as exc:
        raise ValueError(f"Invalid eval_metric {name!r}: {exc}") from exc
    if maximize == minimize:
        raise ValueError(f"eval_metric {name!r} must have a scalar minimum or maximum optimum.")
    return maximize


def _shared_metric(name, predictions, targets, weights, group_offsets=None, pairs=None, *, pair_query_unit_weights=True, qce_scales=None, subgroup_hashes=None, qce_session=None):
    from catboost import metrics
    from catboost.utils import eval_metric

    base, _, arguments = name.partition(":")
    parameters = dict(item.split("=", 1) for item in arguments.split(";") if "=" in item)
    if base in ("PairLogit", "PairAccuracy") and pairs is None:
        raise ValueError("Pair metrics require explicit supplied pairs; automatic pair generation is not yet supported.")
    # EvalMetricsForUtils unconditionally sets UseWeights=true when a weight
    # vector is supplied (helpers.cpp). Resolve the training metric's choice
    # first, including registry defaults such as AUC/PRAUC's false default.
    if "use_weights" in parameters:
        requested = parameters["use_weights"].lower()
        if requested not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            raise ValueError("Metric use_weights must be a valid boolean.")
        weighted = requested in ("true", "1", "yes", "on")
    else:
        implementation = getattr(metrics, base, None)
        defaults = implementation.params_with_defaults() if implementation is not None else {}
        weighted = defaults.get("use_weights", {}).get("default_value", True)
    effective_weights = weights if weighted else None
    if base == "QueryCrossEntropy":
        if pairs is not None: raise ValueError("QueryCrossEntropy metrics require a QueryCrossEntropy target.")
        # CUDA TTargetFallbackMetric retains the target's original weights and
        # scale table; only metric alpha is overridden. Other standalone uses
        # without a target scale array honor their own weight/scale parameters.
        return metric(predictions, targets, weights if qce_scales is not None else effective_weights,
                      name, group_offsets=group_offsets, query_scales=qce_scales, qce_session=qce_session)
    group_options = {}
    if pairs is not None:
        if base not in ("PairLogit", "PairAccuracy", *_RANKING_METRICS):
            raise ValueError("Supplied-pair training supports PairLogit, PairAccuracy, NDCG, MAP, and PFound eval metrics.")
        if base in ("PairLogit", "PairAccuracy"):
            group_options["pairs"] = np.column_stack(pairs[:2])
        if group_offsets is None:
            group_options["group_id"] = np.zeros(len(predictions), np.uint32)
    if group_offsets is not None:
        group_options["group_id"] = np.repeat(np.arange(len(group_offsets) - 1), np.diff(group_offsets))
        from catboost import _catboost
        if _catboost.is_groupwise_metric(name) and base not in (*_QUERY_OBJECTIVES, "PairLogit", "PairAccuracy", *_RANKING_METRICS):
            raise ValueError(f"Grouped eval_metric {base!r} is not yet supported by the standalone query controller.")
    if base in _RANKING_METRICS:
        if group_offsets is None:
            raise ValueError("Ranking evaluation requires explicit group boundaries.")
        from ._query_data import validate_offsets
        offsets = validate_offsets(group_offsets, len(predictions))
        # CUDA boosting_metric_calcer.h caches the first effective row weight
        # as TQueryInfo::Weight. Pointwise PairLogit uses unit query weights,
        # independently of incident mass; coupled PairLogitPairwise retains
        # original effective document weights, selected explicitly by its caller.
        # TMAPKMetric itself ignores weights in the pinned shared source; keep
        # that CUDA convention instead of silently introducing weighted MAP.
        query_weights = (np.ones(len(offsets) - 1, np.float32) if effective_weights is None or (pairs is not None and pair_query_unit_weights)
                         else np.asarray(effective_weights, np.float32)[offsets[:-1]])
        group_options["group_weight"] = np.repeat(query_weights, np.diff(offsets))
        effective_weights = None
    try:
        validation_predictions = (np.zeros_like(predictions)
                                  if base in (*_QUERY_OBJECTIVES, *_MULTIOUTPUT_OBJECTIVES,
                                              "PairLogit", "PairAccuracy") else predictions)
        if base == "PFound" and subgroup_hashes is not None:
            from catboost import _catboost
            from ._query_data import validate_subgroup_hashes
            stored_subgroups = validate_subgroup_hashes(subgroup_hashes, len(predictions))
            # Pool IDs have already been hashed. Rehashing their decimal form
            # can merge unrelated subgroups through a second uint32 collision.
            values = _catboost._eval_metric_util([targets], [validation_predictions], name,
                effective_weights, group_options.get("group_id"), group_options.get("group_weight"),
                stored_subgroups, None, 1, subgroup_id_is_hashed=True)
        else:
            values = eval_metric(targets, validation_predictions, name, weight=effective_weights,
                                 thread_count=1, **group_options)
    except Exception as exc:
        raise ValueError(f"Cannot evaluate {name!r} on these targets and weights: {exc}") from exc
    if len(values) != 1 or not np.isfinite(values[0]):
        raise ValueError(f"eval_metric {name!r} must return one finite value.")
    if pairs is not None and base in ("PairLogit", "PairAccuracy"):
        from ._query_data import pair_metric
        return pair_metric(predictions, pairs[0], pairs[1], pairs[2] if weighted else None, base)
    if base in _QUERY_OBJECTIVES:
        return metric(predictions, targets, effective_weights, name, group_offsets=group_offsets)
    if base in _MULTIOUTPUT_OBJECTIVES:
        return metric(predictions, targets, effective_weights, base)
    if base in ("MAE", "Quantile"):
        # CUDA gpu_metrics.cpp calls TQuantileTarget::Score with no delta
        # deadzone. MAE's public metric doubles its alpha=.5 training loss.
        # The shared helper above validates parameters, then CUDA arithmetic
        # supplies the metric, including when selected explicitly by the user.
        residual = np.asarray(targets, np.float64) - np.asarray(predictions, np.float64)
        if base == "MAE":
            losses = np.abs(residual)
        else:
            alpha = float(parameters.get("alpha", 0.5))
            losses = np.where(residual > 0, alpha * residual, -(1 - alpha) * residual)
        return float(np.average(losses, weights=effective_weights))
    return float(values[0])


def _write_snapshot(path, fingerprint, result, history, best_iteration, best_value, eval_raw,
                    permutation_state=None, feature_penalty_state=None, optimization_predictions=None,
                    ordered_state=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "version": _SNAPSHOT_VERSION, "fingerprint": fingerprint,
        "completed_iterations": len(result.depths),
        "history": history, "best_iteration": best_iteration,
        "best_value": best_value if np.isfinite(best_value) else None,
        "stats": result.stats,
    }
    if "yeti_rng" in result.stats:
        header["yeti_rng_checksum"] = _fingerprint({}, result.stats["yeti_rng"])
    arrays = {name: getattr(result, name) for name in _TREE_FIELDS}
    arrays.update(predictions=result.predictions, rmse=result.rmse)
    if eval_raw is not None:
        arrays["eval_predictions"] = np.asarray(eval_raw, dtype=np.float32)
    if permutation_state is not None:
        header["permutation_count"] = len(permutation_state["predictions"])
        for name, value in permutation_state.items():
            arrays["permutation_" + name] = value
    if feature_penalty_state is not None:
        arrays["feature_penalty_used_features"] = feature_penalty_state["used_features"]
    if optimization_predictions is not None:
        arrays["optimization_predictions"] = optimization_predictions
    if ordered_state is not None:
        ordered_header = {name: ordered_state[name] for name in ("version", "fingerprint", "iteration_offset")}
        if "mvs_lambda" in ordered_state:
            ordered_header["mvs_lambda"] = ordered_state["mvs_lambda"]
        ordered_arrays = {name: ordered_state[name] for name in ("descriptors", "cursors")}
        if "selection_rng" in ordered_state:
            rng = ordered_state["selection_rng"]
            ordered_header["selection_rng"] = {name: value for name, value in rng.items() if name != "words"}
            ordered_arrays["selection_rng_words"] = np.asarray(rng["words"], dtype=np.uint64)
            header["ordered_selection_rng_required"] = True
        header["ordered_state"] = ordered_header
        header["ordered_state_checksum"] = _fingerprint(ordered_arrays, ordered_header)
        arrays.update({"ordered_" + name: value for name, value in ordered_arrays.items()})
    arrays["metadata"] = np.asarray(_json(header))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            np.savez(output, **arrays)
            output.flush()
            import os
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_snapshot(path, fingerprint, *, rows, features, depth, iterations,
                   objective, eval_rows, selection_metric=None, maximize=False,
                   initial_iteration_offset=0, classes=None, permutation_count=1,
                   ctr_unique_values=None, ordered_options=None, yeti_options=None):
    has_eval = eval_rows is not None
    selection_metric = selection_metric or objective
    try:
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(str(archive["metadata"].item()))
            if header.get("version") != _SNAPSHOT_VERSION:
                raise ValueError("Unsupported Metal snapshot version.")
            if ctr_unique_values is not None and "feature_penalty_used_features" not in archive:
                raise ValueError("Snapshot lacks feature-penalty continuation state and is incompatible; "
                                 "restart training with resume=False.")
            if classes:
                optimization_name = ("permutation_optimization_predictions" if permutation_count > 1
                                     else "optimization_predictions")
                if optimization_name not in archive:
                    raise ValueError("Snapshot lacks exact vector optimization continuation state and is "
                                     "incompatible; restart training with resume=False.")
            if ordered_options is not None and (
                    "ordered_state" not in header or "ordered_descriptors" not in archive
                    or "ordered_cursors" not in archive):
                raise ValueError("Snapshot lacks Ordered fold continuation state and is incompatible; "
                                 "restart training with resume=False.")
            if header.get("fingerprint") != fingerprint:
                raise ValueError("Snapshot does not match training data, borders, or parameters.")
            count = header["completed_iterations"]
            if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= iterations:
                raise ValueError("Snapshot tree count exceeds the requested iterations or is invalid.")
            specifications = {
                "depths": ((count,), np.dtype("uint32")),
                "split_features": ((count, depth), np.dtype("uint32")),
                "split_bins": ((count, depth), np.dtype("uint32")),
                "split_types": ((count, depth), np.dtype("uint8")),
                "leaf_values": (((count, 1 << depth, classes) if classes else (count, 1 << depth)), np.dtype("float32")),
                "leaf_weights": ((count, 1 << depth), np.dtype("float32")),
                "predictions": (((rows, classes) if classes else (rows,)), np.dtype("float32")),
                "rmse": ((count + 1,), np.dtype("float32")),
            }
            if has_eval:
                specifications["eval_predictions"] = (((eval_rows, classes) if classes else (eval_rows,)), np.dtype("float32"))
            if header.get("permutation_count", 1) != permutation_count:
                raise ValueError("Snapshot permutation count does not match the training datasets.")
            if permutation_count > 1:
                specifications.update(
                    permutation_predictions=(((permutation_count, rows, classes) if classes
                                               else (permutation_count, rows)), np.dtype("float32")),
                    permutation_mvs_lambdas=((permutation_count,), np.dtype("float32")),
                    permutation_mvs_valid=((permutation_count,), np.dtype("uint8")))
            if ctr_unique_values is not None:
                specifications["feature_penalty_used_features"] = ((features,), np.dtype("uint8"))
            if classes:
                dimensions = classes - 1 if objective == "MultiClass" else classes
                optimization_shape = ((permutation_count, dimensions, rows) if permutation_count > 1
                                      else (dimensions, rows))
                specifications[optimization_name] = (optimization_shape, np.dtype("float32"))
            arrays = {}
            for name, (shape, dtype) in specifications.items():
                array = archive[name]
                if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
                    raise ValueError(f"Invalid snapshot array: {name}.")
                arrays[name] = array.copy()
            if ordered_options is not None:
                descriptors, cursors = archive["ordered_descriptors"], archive["ordered_cursors"]
                if (descriptors.ndim != 2 or descriptors.shape[1] != 4 or not len(descriptors)
                        or descriptors.dtype != np.dtype("uint32") or cursors.ndim != 1
                        or cursors.dtype != np.dtype("float32") or not len(cursors)
                        or not np.isfinite(cursors).all()):
                    raise ValueError("Invalid Ordered snapshot descriptor or cursor array.")
                arrays.update(ordered_descriptors=descriptors.copy(), ordered_cursors=cursors.copy())
                if "selection_rng" in header.get("ordered_state", {}):
                    if "ordered_selection_rng_words" not in archive:
                        raise ValueError("Ordered snapshot lacks selection RNG words; restart training with resume=False.")
                    words = archive["ordered_selection_rng_words"]
                    if words.shape != (312,) or words.dtype != np.dtype("uint64"):
                        raise ValueError("Invalid Ordered snapshot selection RNG word array.")
                    arrays["ordered_selection_rng_words"] = words.copy()
        if (arrays["depths"] > depth).any() or (arrays["leaf_weights"] < 0).any():
            raise ValueError("Invalid snapshot tree depths or weights.")
        for tree, tree_depth in enumerate(arrays["depths"]):
            if ((arrays["split_features"][tree, :tree_depth] >= features).any()
                    or (arrays["split_bins"][tree, :tree_depth] > 255).any()
                    or (arrays["split_types"][tree, :tree_depth] > 1).any()):
                raise ValueError("Invalid snapshot split indices.")
        if ctr_unique_values is not None:
            used = arrays["feature_penalty_used_features"]
            if (used > 1).any() or used[np.asarray(ctr_unique_values) == 0].any():
                raise ValueError("Invalid snapshot used-feature penalty state.")
            for tree, tree_depth in enumerate(arrays["depths"]):
                selected = arrays["split_features"][tree, :tree_depth]
                if (used[selected[np.asarray(ctr_unique_values)[selected] > 0]] == 0).any():
                    raise ValueError("Snapshot used-feature penalty state disagrees with its CTR splits.")
            header["feature_penalty_state"] = {"used_features": used}
        history = header["history"]
        expected = {"learn", "validation"} if has_eval else {"learn"}
        if set(history) != expected:
            raise ValueError("Invalid snapshot metric datasets.")
        for dataset in expected:
            expected_metrics = ({objective, selection_metric}
                                if dataset == "validation" or not has_eval else {objective})
            if set(history[dataset]) != expected_metrics:
                raise ValueError("Invalid snapshot metric names.")
            for name in expected_metrics:
                values = history[dataset][name]
                if len(values) != count or not np.isfinite(values).all():
                    raise ValueError("Invalid snapshot metric history.")
        best = header["best_iteration"]
        if isinstance(best, bool) or not isinstance(best, int) or not -1 <= best < count:
            raise ValueError("Invalid snapshot best iteration.")
        if has_eval:
            selected_values = history["validation"][selection_metric]
            actual_best = int(np.argmax(selected_values) if maximize else np.argmin(selected_values))
            if best != actual_best or header["best_value"] != selected_values[best]:
                raise ValueError("Snapshot best metric disagrees with its history.")
        elif best != -1 or header["best_value"] is not None:
            raise ValueError("Invalid snapshot best metric without validation data.")
        if not isinstance(header["stats"], dict):
            raise ValueError("Invalid snapshot runtime statistics.")
        for name in ("kernel_dispatches", "gpu_seconds"):
            value = header["stats"].get(name, 0)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value < 0:
                raise ValueError("Invalid snapshot runtime statistics.")
        if yeti_options is not None:
            from ._yeti_rng import YetiRankRng
            rng = header["stats"].get("yeti_rng")
            if rng is None or header.get("yeti_rng_checksum") != _fingerprint({}, rng):
                raise ValueError("YetiRank snapshot RNG is missing or its checksum is invalid.")
            YetiRankRng(yeti_options.get("random_seed", 0), yeti_options.get("bootstrap_type", "No"),
                        yeti_options.get("leaf_estimation_iterations", 1), initial_state=rng,
                        iteration_offset=initial_iteration_offset + count)
        if "bootstrap_state" in header["stats"]:
            state = header["stats"]["bootstrap_state"]
            if (not isinstance(state, dict)
                    or isinstance(state.get("iteration_offset"), bool)
                    or not isinstance(state.get("iteration_offset"), int)
                    or state.get("iteration_offset") != initial_iteration_offset + count):
                raise ValueError("Snapshot bootstrap iteration state is invalid.")
            value = state.get("mvs_lambda")
            if value is not None and (isinstance(value, bool) or not isinstance(value, numbers.Real)
                                      or not np.isfinite(value) or value < 0):
                raise ValueError("Snapshot MVS regularization state is invalid.")
        if permutation_count > 1:
            cursor = arrays["permutation_predictions"]
            lambdas, valid = arrays["permutation_mvs_lambdas"], arrays["permutation_mvs_valid"]
            if not np.array_equal(cursor[-1], arrays["predictions"]):
                raise ValueError("Snapshot estimation permutation disagrees with the exported training cursor.")
            if (lambdas < 0).any() or (valid > 1).any():
                raise ValueError("Snapshot permutation MVS state is invalid.")
            exported_lambda = header["stats"].get("bootstrap_state", {}).get("mvs_lambda")
            if exported_lambda != (float(lambdas[-1]) if valid[-1] else None):
                raise ValueError("Snapshot estimation permutation disagrees with its MVS state.")
            header["permutation_state"] = {
                "predictions": cursor, "mvs_lambdas": lambdas, "mvs_valid": valid}
            if classes:
                header["permutation_state"]["optimization_predictions"] = arrays[optimization_name]
        elif classes:
            header["optimization_predictions"] = arrays[optimization_name]
        if ordered_options is not None:
            state = header["ordered_state"]
            required_keys = {"version", "fingerprint", "iteration_offset"}
            if (not isinstance(state, dict) or not required_keys <= set(state)
                    or set(state) - required_keys - {"mvs_lambda", "selection_rng"}
                    or type(state["version"]) is not int or state["version"] != 1
                    or type(state["iteration_offset"]) is not int
                    or state["iteration_offset"] != initial_iteration_offset + count
                    or not isinstance(state["fingerprint"], str) or len(state["fingerprint"]) != 64
                    or any(value not in "0123456789abcdef" for value in state["fingerprint"])):
                raise ValueError("Invalid Ordered snapshot continuation metadata.")
            ordered_lambda = state.get("mvs_lambda")
            if (ordered_lambda is not None and (
                    isinstance(ordered_lambda, bool) or not isinstance(ordered_lambda, numbers.Real)
                    or not np.isfinite(ordered_lambda) or not 0 <= ordered_lambda <= np.finfo(np.float32).max)):
                raise ValueError("Invalid Ordered snapshot MVS regularization state.")
            if ordered_lambda != header["stats"].get("bootstrap_state", {}).get("mvs_lambda"):
                raise ValueError("Ordered snapshot MVS state disagrees with its bootstrap state.")
            descriptors, cursors = arrays["ordered_descriptors"], arrays["ordered_cursors"]
            checked_arrays = dict(descriptors=descriptors, cursors=cursors)
            restored_rng = None
            if "selection_rng" in state:
                rng = state["selection_rng"]
                if not isinstance(rng, dict) or set(rng) != {
                        "version", "index", "bootstrap_initialized", "completed_iterations"}:
                    raise ValueError("Invalid Ordered snapshot selection RNG metadata.")
                words = arrays["ordered_selection_rng_words"]
                checked_arrays["selection_rng_words"] = words
                restored_rng = dict(rng, words=words.tolist())
                from ._ordered_rng import OrderedSelectionRng
                OrderedSelectionRng(ordered_options.get("random_seed", 0),
                    ordered_options.get("permutation_count", 1), restored_rng, state["iteration_offset"])
                if not rng["bootstrap_initialized"]:
                    raise ValueError("Completed Ordered snapshot has uninitialized selection RNG bootstrap state.")
            elif header.get("ordered_selection_rng_required"):
                raise ValueError("Ordered snapshot lacks selection RNG continuation state; restart with resume=False.")
            if header.get("ordered_state_checksum") != _fingerprint(checked_arrays, state):
                raise ValueError("Ordered snapshot continuation state checksum does not match.")
            ordered_count = ordered_options.get("permutation_count", 1)
            offsets = np.r_[np.uint64(0), np.cumsum(descriptors[:-1, 1], dtype=np.uint64)]
            if (not np.array_equal(descriptors[:, 2], offsets)
                    or np.sum(descriptors[:, 1], dtype=np.uint64) != len(cursors)
                    or (descriptors[:, 0] == 0).any() or (descriptors[:, 0] > descriptors[:, 1]).any()
                    or (descriptors[:, 1] > rows).any() or (descriptors[:, 3] >= ordered_count).any()
                    or not np.array_equal(descriptors[-1, [0, 1, 3]], [rows, rows, ordered_count - 1])):
                raise ValueError("Invalid Ordered snapshot fold descriptors.")
            maps = ordered_options.get("permutations")
            if maps is None:
                if restored_rng is None:
                    from ._data import cuda_history_order
                    final_order = cuda_history_order(rows, ordered_count - 1)
                elif ordered_options.get("group_sizes") is not None:
                    from ._ordered_rng import cuda_ordered_group_history_order
                    final_order = cuda_ordered_group_history_order(ordered_options["group_sizes"],
                        ordered_count - 1, ordered_options.get("fold_permutation_block", 64))
                else:
                    from ._ordered_rng import cuda_ordered_history_order
                    final_order = cuda_ordered_history_order(
                        rows, ordered_count - 1, ordered_options.get("fold_permutation_block", 64))
            else:
                final_order = np.asarray(maps)[-1]
            if not np.array_equal(cursors[int(descriptors[-1, 2]):], arrays["predictions"][final_order]):
                raise ValueError("Ordered snapshot full-estimation cursor disagrees with exported predictions.")
            header["ordered_state"] = dict(state, descriptors=descriptors, cursors=cursors)
            if restored_rng is not None:
                header["ordered_state"]["selection_rng"] = restored_rng
        result = _native.TrainResult(
            **{name: arrays[name] for name in _TREE_FIELDS if name != "split_types"},
            predictions=arrays["predictions"], rmse=arrays["rmse"], stats=header["stats"])
        result.split_types = arrays["split_types"]
        return result, header, arrays.get("eval_predictions")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Metal snapshot {path}: {exc}") from exc


def _merge(prior, segment):
    if prior is None:
        return segment
    values = {name: np.concatenate((getattr(prior, name), getattr(segment, name)))
              for name in _TREE_FIELDS}
    result = _native.TrainResult(
        **{name: values[name] for name in _TREE_FIELDS if name != "split_types"},
        predictions=segment.predictions.copy(),
        rmse=np.concatenate((prior.rmse, segment.rmse[1:])), stats=segment.stats.copy())
    result.split_types = values["split_types"]
    for name in ("kernel_dispatches", "gpu_seconds"):
        result.stats[name] = prior.stats.get(name, 0) + segment.stats.get(name, 0)
    if "search_permutations" in segment.stats:
        result.stats["search_permutations"] = (prior.stats.get("search_permutations", [])
                                               + segment.stats["search_permutations"])
    return result


def _with_validation_stats(result, prior, stats):
    if stats is not None:
        merged = stats.copy()
        previous = {} if prior is None else prior.stats.get("validation", {})
        for name in ("kernel_dispatches", "gpu_seconds", "dataset_uploads", "bins_upload_bytes", "tree_upload_bytes"):
            merged[name] = previous.get(name, 0) + stats.get(name, 0)
        result.stats["validation"] = merged
    return result


def _predict(bins, result, bias, *, start=0, end=None):
    from ._inference import predict_bins

    selection = slice(start, end)
    if result.leaf_values.ndim == 3:
        classes = result.leaf_values.shape[2]
        biases = np.broadcast_to(np.asarray(bias), (classes,))
        return np.column_stack([predict_bins(
            bins, result.depths[selection], result.split_features[selection],
            result.split_bins[selection], result.leaf_values[selection, :, column], bias=biases[column],
            split_types=result.split_types[selection]) for column in range(classes)])
    return predict_bins(
        bins, result.depths[selection], result.split_features[selection],
        result.split_bins[selection], result.leaf_values[selection], bias=bias,
        split_types=result.split_types[selection])


def _prepare_permutations(permutation_bins, bins):
    if permutation_bins is None:
        return None
    if (not isinstance(permutation_bins, (tuple, list, np.ndarray))
            or isinstance(permutation_bins, np.ndarray) and permutation_bins.ndim == 0
            or not 1 <= len(permutation_bins) <= 64):
        raise ValueError("permutation_bins must contain between 1 and 64 bin matrices.")
    original = np.asarray(bins)
    matrices = []
    for value in permutation_bins:
        matrix = np.asarray(value)
        if (matrix.ndim != 2 or matrix.shape != original.shape or not all(matrix.shape)
                or matrix.dtype.kind not in "iu" or (matrix < 0).any() or (matrix > 255).any()):
            raise ValueError("Permutation bins must be nonempty byte-valued matrices matching the training bins.")
        matrices.append(np.ascontiguousarray(matrix, dtype=np.uint8))
    if not np.array_equal(original, matrices[0]):
        raise ValueError("Training bins must match permutation zero.")
    return tuple(matrices)


def _with_permutation_stats(result, count):
    if count > 1:
        result.stats.update(permutation_count=count, estimation_permutation=count - 1,
                            permutation_schedule="CatBoostMT19937_64")
    return result


def _prepare_feature_penalties(bins, ctr_unique_values, model_size_reg, feature_weights):
    if ctr_unique_values is None and feature_weights is None:
        return None, feature_weights
    shape = np.shape(bins)
    if len(shape) != 2 or not all(shape):
        raise ValueError("Feature penalties require a nonempty features-by-rows bin matrix.")
    features = shape[0]
    counts = np.zeros(features, np.uint32) if ctr_unique_values is None else np.asarray(ctr_unique_values)
    if (counts.shape != (features,) or counts.dtype.kind not in "iu" or (counts < 0).any()
            or (counts > np.iinfo(np.uint32).max).any()):
        raise ValueError("ctr_unique_values must be a uint32-compatible feature vector.")
    if (isinstance(model_size_reg, bool) or not isinstance(model_size_reg, numbers.Real)
            or not np.isfinite(model_size_reg) or not 0 <= model_size_reg <= np.finfo(np.float32).max):
        raise ValueError("model_size_reg must be finite, nonnegative, and float32-compatible.")
    if feature_weights is not None:
        weights = np.asarray(feature_weights)
        if weights.shape != (features,) or weights.dtype.kind not in "biuf":
            raise ValueError("feature_weights must be a numeric feature vector.")
        with np.errstate(over="ignore", invalid="ignore"):
            feature_weights = np.ascontiguousarray(weights, np.float32)
        if not np.isfinite(feature_weights).all() or (feature_weights < 0).any():
            raise ValueError("feature_weights must be finite and nonnegative.")
    return np.ascontiguousarray(counts, np.uint32), feature_weights


def run_training(bins, targets, candidate_features, candidate_bins, *, iterations,
                 depth, learning_rate, l2_leaf_reg, bias, score_function,
                 objective="RMSE", sample_weight=None, eval_bins=None,
                 eval_targets=None, eval_weight=None, early_stopping_rounds=None,
                 use_best_model=None, eval_metric=None, save_snapshot=False, snapshot_file=None,
                 snapshot_interval=600.0, resume=True, metadata=None, callback=None,
                 permutation_bins=None, ctr_unique_values=None, model_size_reg=0.5,
                 feature_weights=None, boosting_type="Plain",
                 group_offsets=None, eval_group_offsets=None, query_beta=1.0, query_lambda=0.01, query_loss_description=None,
                 subgroup_hashes=None, eval_subgroup_hashes=None,
                 pair_winners=None, pair_losers=None, pair_weights=None,
                 eval_pair_winners=None, eval_pair_losers=None, eval_pair_weights=None,
                 **native_options):
    """Train with a persistent GPU session and return a native-compatible result.

    ``iterations`` is the total desired tree count, including resumed trees.
    ``metadata`` binds snapshots to preprocessing (borders and feature mappings).
    ``eval_metric`` uses CatBoost's shared metric implementation and optimization
    direction. The full parameterized string names its history. It controls
    validation stopping, while the training objective is always recorded.
    The callback receives ``iteration`` (one based) and ``metrics``; returning
    False stops after that completed tree. Histories describe all trained trees,
    including trees discarded by ``use_best_model``. Predictions match retained
    trees. Snapshots always preserve the complete, untrimmed training cursor.
    ``permutation_bins`` supplies CUDA Plain DocParallel history datasets in
    provider row order. CUDA's seeded chooser selects the structure dataset;
    every dataset updates its own cursor and the last supplies exported leaves.
    CTR unique counts and feature weights configure CUDA's split penalties;
    snapshots preserve the global forest's used-CTR flags as numeric arrays.
    Ordered boosting delegates its row permutations and prefix folds to the
    Ordered backend, preserving every fold cursor in snapshots.
    """
    if objective not in ("RMSE", "Logloss", "CrossEntropy", "Poisson", "Huber", "Expectile",
                         "Lq", "Tweedie", "LogLinQuantile", "MAE", "Quantile", "MAPE", "YetiRank", "YetiRankPairwise", "QueryCrossEntropy",
                         *_MULTICLASS_OBJECTIVES, *_MULTIOUTPUT_OBJECTIVES, *_QUERY_OBJECTIVES, *_PAIR_OBJECTIVES):
        raise ValueError(f"Unsupported lifecycle objective: {objective}.")
    if boosting_type not in ("Plain", "Ordered"):
        raise ValueError("boosting_type must be Plain or Ordered.")
    ordered = boosting_type == "Ordered"
    yeti = objective == "YetiRank"
    yeti_pair = objective == "YetiRankPairwise"
    coupled = objective == "PairLogitPairwise"
    qce = objective == "QueryCrossEntropy"
    qce_scales = eval_qce_scales = None
    if qce:
        from . import _query_cross_entropy
        query_loss_description = query_loss_description or "QueryCrossEntropy"
        qce_parameters = _query_cross_entropy.parse_description(query_loss_description)
        native_options = {**native_options, "alpha": qce_parameters.get("alpha", .95)}
    elif query_loss_description is not None:
        raise ValueError("query_loss_description requires QueryCrossEntropy.")
    grouped = objective in _QUERY_OBJECTIVES or yeti or yeti_pair or qce
    paired = objective in _PAIR_OBJECTIVES
    if grouped and not yeti and not yeti_pair and not qce:
        for name, value in (("query_beta", query_beta), ("query_lambda", query_lambda)):
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not np.isfinite(value) or abs(value) > np.finfo(np.float32).max):
                raise ValueError(f"{name} must be a finite float32 number.")
    if ordered and (grouped or paired):
        raise ValueError("Query objectives currently require Plain boosting.")
    if not (grouped or paired) and (group_offsets is not None or eval_group_offsets is not None):
        raise ValueError("Query boundaries require a supported query objective.")
    if subgroup_hashes is not None and group_offsets is None:
        raise ValueError("Subgroup hashes require explicit query boundaries.")
    if eval_subgroup_hashes is not None and eval_group_offsets is None:
        raise ValueError("Evaluation subgroup hashes require explicit query boundaries.")
    if not paired and any(value is not None for value in (
            pair_winners, pair_losers, pair_weights, eval_pair_winners, eval_pair_losers, eval_pair_weights)):
        raise ValueError("Supplied pair arrays require the PairLogit objective.")
    if paired and not coupled and (sample_weight is not None or eval_weight is not None):
        raise ValueError("PairLogit uses incident pair mass; sample_weight and eval_weight must be None.")
    backend, classes = _native, None
    if objective in _MULTICLASS_OBJECTIVES:
        from . import _multiclass
        backend = _multiclass
        classes = native_options.get("classes")
        if isinstance(classes, bool) or not isinstance(classes, numbers.Integral) or not 2 <= classes <= 64:
            raise ValueError("Multiclass training requires classes in [2, 64].")
    elif objective in _MULTIOUTPUT_OBJECTIVES:
        from . import _multioutput
        backend = _multioutput
        classes = _multioutput._dimensions(
            targets, objective, native_options.get("dimensions"), native_options.get("classes"))
        if isinstance(classes, bool) or not isinstance(classes, numbers.Integral) or not 2 <= classes <= 64:
            raise ValueError("Multioutput training requires output dimensions in [2, 64].")
        if objective == "RMSEWithUncertainty" and classes != 2:
            raise ValueError("RMSEWithUncertainty requires two output dimensions.")
        native_options = {**native_options, "classes": int(classes)}
    if coupled:
        from . import _pair_matrix
        backend = SimpleNamespace(Session=_pair_matrix.TrainingSession)
    if yeti_pair:
        from . import _yeti_pair
        backend = SimpleNamespace(Session=_yeti_pair.TrainingSession)
    if qce:
        backend = SimpleNamespace(Session=_query_cross_entropy.TrainingSession)
    if yeti:
        from . import _yeti
        backend = SimpleNamespace(Session=_yeti.TrainingSession)
    if ordered:
        if classes:
            raise ValueError("Ordered boosting does not yet support vector objectives.")
        from . import _ordered
        backend = _ordered
    permutation_bins = _prepare_permutations(permutation_bins, bins)
    permutation_count = len(permutation_bins) if permutation_bins is not None else 1
    ctr_unique_values, feature_weights = _prepare_feature_penalties(
        bins, ctr_unique_values, model_size_reg, feature_weights)
    has_feature_penalties = ctr_unique_values is not None
    ordered_banks = ordered_counts = ordered_weights = None
    if ordered:
        expected = native_options.get("permutation_count", 1)
        if permutation_count not in (1, expected):
            raise ValueError("Ordered CTR banks must match the prefix permutation count.")
        ordered_banks = permutation_bins if permutation_count > 1 else None
        ordered_counts, ordered_weights = ctr_unique_values, feature_weights
        # Ordered's session owns every prefix and the estimation bank. The
        # DocParallel cursor/used-feature controller below remains inactive.
        permutation_count = 1
        ctr_unique_values = None
        has_feature_penalties = False
    if yeti and (permutation_count > 1 or has_feature_penalties):
        raise ValueError("YetiRank currently requires numeric P1 data without CTR feature penalties.")
    if (coupled or qce or yeti_pair) and (permutation_count > 1 or has_feature_penalties):
        raise ValueError("Full-matrix targets currently require numeric P1 data without CTR feature penalties.")
    objective_metric = "PFound" if yeti or yeti_pair else "PairLogit" if coupled else query_loss_description if qce else objective
    if objective == "QuerySoftMax" and (np.float32(query_beta) != 1 or np.float32(query_lambda) != np.float32(.01)):
        objective_metric = f"QuerySoftMax:beta={float(query_beta)!r};lambda={float(query_lambda)!r}"
    objective_parameters = {"Huber": ("delta", 1.0), "Expectile": ("alpha", 0.5),
                            "Lq": ("q", 2.0), "Tweedie": ("variance_power", 1.5),
                            "LogLinQuantile": ("alpha", 0.5), "Quantile": ("alpha", 0.5)}
    if objective in objective_parameters:
        key, default_parameter = objective_parameters[objective]
        parameter = native_options.get("objective_param", default_parameter)
        objective_metric = f"{objective}:{key}={float(parameter)!r}"
    selection_metric = objective_metric if eval_metric is None else eval_metric
    maximize = _metric_direction(selection_metric) if yeti or yeti_pair or eval_metric is not None else False
    if isinstance(iterations, bool) or not isinstance(iterations, numbers.Integral) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    if early_stopping_rounds is not None and (
            isinstance(early_stopping_rounds, bool)
            or not isinstance(early_stopping_rounds, numbers.Integral)
            or early_stopping_rounds < 1):
        raise ValueError("early_stopping_rounds must be a positive integer.")
    for name, value in (("save_snapshot", save_snapshot), ("resume", resume)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean.")
    if use_best_model is not None and not isinstance(use_best_model, bool):
        raise ValueError("use_best_model must be boolean or None.")
    if (isinstance(snapshot_interval, bool) or not isinstance(snapshot_interval, numbers.Real)
            or not np.isfinite(snapshot_interval) or snapshot_interval < 0):
        raise ValueError("snapshot_interval must be a finite nonnegative number of seconds.")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable.")
    has_eval = (eval_bins is not None or eval_targets is not None or eval_weight is not None
                or eval_group_offsets is not None or eval_pair_winners is not None
                or eval_pair_losers is not None or eval_pair_weights is not None or eval_subgroup_hashes is not None)
    if has_eval and (eval_bins is None or eval_targets is None):
        raise ValueError("Validation bins and targets must both be supplied.")
    if not has_eval and (early_stopping_rounds is not None or use_best_model is True):
        raise ValueError("Early stopping and use_best_model require validation data.")
    keep_best = has_eval if use_best_model is None else use_best_model
    params = dict(iterations=iterations, depth=depth, learning_rate=learning_rate,
                  l2_leaf_reg=l2_leaf_reg, bias=bias, score_function=score_function,
                  objective=objective, sample_weight=sample_weight, **native_options)
    if ordered:
        params.update(permutation_bins=ordered_banks, ctr_unique_values=ordered_counts,
                      model_size_reg=model_size_reg, feature_weights=ordered_weights)
    if grouped or paired:
        from ._query_data import validate_offsets, prepare_pair_arrays
        training_shape = np.shape(bins)
        if len(training_shape) != 2 or not all(training_shape):
            raise ValueError("Query bins must be a nonempty features-by-rows matrix.")
        if grouped or group_offsets is not None:
            group_offsets = validate_offsets(group_offsets, training_shape[1])
        params["group_offsets"] = group_offsets
        if grouped:
            if qce:
                qce_scales = _query_cross_entropy.select_scales(qce_parameters.get("raw_values_scale", ""), targets, group_offsets)
                params["query_scales"] = qce_scales
            elif not yeti and not yeti_pair:
                params.update(query_beta=query_beta, query_lambda=query_lambda)
        else:
            pair_winners, pair_losers, pair_weights = prepare_pair_arrays(
                pair_winners, pair_losers, pair_weights, training_shape[1], group_offsets)
            params.update(pair_winners=pair_winners, pair_losers=pair_losers, pair_weights=pair_weights)
    from ._query_data import validate_subgroup_hashes
    subgroup_hashes = validate_subgroup_hashes(subgroup_hashes, len(targets))
    if not (has_eval or save_snapshot or callback is not None or eval_metric is not None
            or permutation_count > 1 or has_feature_penalties or yeti or yeti_pair or coupled or qce):
        result = backend.train(bins, targets, candidate_features, candidate_bins, **params)
        history = {"learn": {objective_metric: result.rmse[1:].astype(float).tolist()}}
        return _finish(result, history, -1, None, "iterations", len(result.depths), 0)

    bins = np.asarray(bins)
    if bins.ndim != 2 or not all(bins.shape):
        raise ValueError("bins must be a nonempty features-by-rows matrix.")
    features, rows = bins.shape
    targets = _vector(targets, rows, "targets", objective=objective, classes=classes)
    if sample_weight is not None:
        sample_weight = _vector(sample_weight, rows, "sample_weight", weight=True)
        params["sample_weight"] = sample_weight
    if has_eval:
        eval_bins = np.asarray(eval_bins)
        if (eval_bins.ndim != 2 or eval_bins.shape[0] != features or not eval_bins.shape[1]
                or eval_bins.dtype.kind not in "iu" or (eval_bins < 0).any()
                or (eval_bins > 255).any()):
            raise ValueError("Validation bins must be a nonempty byte-valued matrix with matching features.")
        eval_bins = np.ascontiguousarray(eval_bins, dtype=np.uint8)
        eval_targets = _vector(eval_targets, eval_bins.shape[1], "eval_targets", objective=objective, classes=classes)
        if eval_weight is not None:
            eval_weight = _vector(eval_weight, len(eval_targets), "eval_weight", weight=True)
        if grouped:
            eval_group_offsets = validate_offsets(eval_group_offsets, len(eval_targets), "eval_group_offsets")
            if qce:
                eval_qce_scales = _query_cross_entropy.select_scales(qce_parameters.get("raw_values_scale", ""), eval_targets, eval_group_offsets)
            metric(_initial_cursor(len(eval_targets), bias), eval_targets, eval_weight,
                   objective_metric, group_offsets=eval_group_offsets, query_scales=eval_qce_scales)
        elif paired:
            if eval_group_offsets is not None:
                eval_group_offsets = validate_offsets(eval_group_offsets, len(eval_targets), "eval_group_offsets")
            eval_pair_winners, eval_pair_losers, eval_pair_weights = prepare_pair_arrays(
                eval_pair_winners, eval_pair_losers, eval_pair_weights, len(eval_targets), eval_group_offsets)
    elif eval_group_offsets is not None:
        raise ValueError("eval_group_offsets requires validation data.")
    eval_subgroup_hashes = validate_subgroup_hashes(eval_subgroup_hashes, len(eval_targets) if has_eval else 0, "eval_subgroup_hashes")
    if yeti or yeti_pair:
        params["subgroup_hashes"] = subgroup_hashes
    train_pairs = (pair_winners, pair_losers, pair_weights) if paired else None
    eval_pairs = (eval_pair_winners, eval_pair_losers, eval_pair_weights) if paired and has_eval else None
    if eval_metric is not None:
        # Reject invalid parameters, incompatible labels, missing group data,
        # and metrics returning multiple values before constructing a GPU session.
        selected_targets = eval_targets if has_eval else targets
        selected_weights = eval_weight if has_eval else sample_weight
        _shared_metric(selection_metric, _initial_cursor(len(selected_targets), bias, classes),
                       selected_targets, selected_weights, eval_group_offsets if has_eval else group_offsets,
                       eval_pairs if has_eval else train_pairs, pair_query_unit_weights=not coupled, qce_scales=eval_qce_scales if has_eval else qce_scales,
                       subgroup_hashes=eval_subgroup_hashes if has_eval else subgroup_hashes)

    path = Path(snapshot_file or "catboost_metal.snapshot.npz") if save_snapshot else None
    fingerprint_parameters = dict(
        depth=depth, learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
        bias=bias, score_function=score_function, objective=objective,
        early_stopping_rounds=early_stopping_rounds, use_best_model=keep_best,
        metadata=metadata, native_options=native_options)
    if eval_metric is not None:
        fingerprint_parameters["eval_metric"] = eval_metric
        fingerprint_parameters["metric_semantics_version"] = _METRIC_SEMANTICS_VERSION
    if ordered:
        fingerprint_parameters["boosting_type"] = "Ordered"
    fingerprint_arrays = dict(
        bins=bins, targets=targets, candidate_features=candidate_features,
        candidate_bins=candidate_bins, sample_weight=sample_weight,
        eval_bins=eval_bins, eval_targets=eval_targets, eval_weight=eval_weight)
    if ordered_banks is not None:
        fingerprint_arrays.update({f"ordered_feature_bank_{i}": bank for i, bank in enumerate(ordered_banks)})
    if ordered_counts is not None:
        fingerprint_arrays.update(ordered_ctr_counts=ordered_counts, ordered_feature_weights=ordered_weights)
        fingerprint_parameters.update(ordered_ctr_model_size_reg=float(model_size_reg), ordered_ctr_penalty_policy="static_feature_parallel_v1")
    if subgroup_hashes is not None or eval_subgroup_hashes is not None:
        fingerprint_arrays.update(subgroup_hashes=subgroup_hashes, eval_subgroup_hashes=eval_subgroup_hashes)
    if qce:
        fingerprint_parameters["query_loss_description"] = query_loss_description
        fingerprint_arrays.update(query_scales=qce_scales, eval_query_scales=eval_qce_scales)
    if grouped:
        fingerprint_parameters.update(query_beta=float(query_beta), query_lambda=float(query_lambda))
        fingerprint_arrays.update(group_offsets=group_offsets, eval_group_offsets=eval_group_offsets)
    elif paired:
        fingerprint_parameters["pair_weight_semantics"] = ("original_edges_and_documents" if coupled else "supplied_literal_incident_mass")
        fingerprint_arrays.update(group_offsets=group_offsets, eval_group_offsets=eval_group_offsets,
            pair_winners=pair_winners, pair_losers=pair_losers, pair_weights=pair_weights,
            eval_pair_winners=eval_pair_winners, eval_pair_losers=eval_pair_losers, eval_pair_weights=eval_pair_weights)
    if permutation_count > 1:
        fingerprint_parameters.update(permutation_count=permutation_count,
                                      permutation_schedule="CatBoostMT19937_64")
        fingerprint_arrays.update({f"permutation_bins_{i}": matrix
                                   for i, matrix in enumerate(permutation_bins)})
    if has_feature_penalties:
        fingerprint_parameters.update(model_size_reg=float(model_size_reg), feature_penalty_semantics_version=1)
        fingerprint_arrays.update(ctr_unique_values=ctr_unique_values, feature_weights=feature_weights)
    fingerprint = _fingerprint(fingerprint_arrays, fingerprint_parameters) if save_snapshot else None
    prior = None
    permutation_state = None
    feature_penalty_state = None
    optimization_predictions = None
    ordered_state = None
    eval_raw = None
    history = {"learn": {objective_metric: []}}
    if has_eval:
        history["validation"] = {objective_metric: []}
    if selection_metric != objective_metric:
        history["validation" if has_eval else "learn"][selection_metric] = []
    best_iteration, best_value = -1, float("-inf") if maximize else float("inf")
    if path is not None and resume and path.exists():
        prior, state, eval_raw = _read_snapshot(
            path, fingerprint, rows=rows, features=features, depth=depth,
            iterations=iterations, objective=objective_metric,
            eval_rows=len(eval_targets) if has_eval else None,
            selection_metric=selection_metric, maximize=maximize,
            initial_iteration_offset=native_options.get("iteration_offset", 0), classes=classes,
            permutation_count=permutation_count, ctr_unique_values=ctr_unique_values,
            ordered_options=native_options if ordered else None, yeti_options=native_options if yeti else None)
        permutation_state = state.get("permutation_state")
        feature_penalty_state = state.get("feature_penalty_state")
        optimization_predictions = state.get("optimization_predictions")
        ordered_state = state.get("ordered_state")
        if yeti:
            params["initial_rng_state"] = prior.stats["yeti_rng"]
        bootstrap_state = prior.stats.get("bootstrap_state")
        if native_options.get("bootstrap_type", "No") != "No" and bootstrap_state is None:
            raise ValueError("Snapshot lacks bootstrap continuation state.")
        if (native_options.get("bootstrap_type") == "MVS" and native_options.get("mvs_reg") is None
                and bootstrap_state.get("mvs_lambda") is None):
            raise ValueError("Snapshot lacks adaptive MVS regularization state.")
        if (permutation_count > 1 and native_options.get("bootstrap_type") == "MVS"
                and native_options.get("mvs_reg") is None
                and not permutation_state["mvs_valid"].all()):
            raise ValueError("Snapshot lacks adaptive MVS regularization state for every permutation.")
        history = state["history"]
        best_iteration = state["best_iteration"]
        best_value = state["best_value"] if has_eval else float("inf")
    resumed = len(prior.depths) if prior is not None else 0
    completed = resumed
    stop_reason = "iterations"
    result = prior
    if has_eval and eval_raw is None:
        eval_raw = _initial_cursor(len(eval_targets), bias, classes)
    already_stopped = (early_stopping_rounds is not None and resumed
                       and resumed - 1 - best_iteration >= early_stopping_rounds)
    if already_stopped:
        stop_reason = "early_stopping"
    if completed < iterations and not already_stopped:
        params["iterations"] = iterations - resumed
        if prior is not None:
            if ordered:
                params["initial_state"] = ordered_state
            else:
                params["initial_predictions"] = prior.predictions
            if classes and permutation_count == 1:
                params["initial_optimization_predictions"] = optimization_predictions
            # Older no-bootstrap snapshots have no state object. They still
            # need the absolute offset when resumed by the newer native API.
            params["iteration_offset"] = native_options.get("iteration_offset", 0) + resumed
            if "bootstrap_state" in prior.stats:
                bootstrap_state = prior.stats["bootstrap_state"]
                mvs_lambda = bootstrap_state.get("mvs_lambda")
                if mvs_lambda is not None:
                    params["initial_mvs_lambda"] = mvs_lambda
        last_snapshot = time.monotonic()
        with ExitStack() as resources:
            session = resources.enter_context(backend.Session(
                bins, targets, candidate_features, candidate_bins, **params))
            learn_qce_metric = eval_qce_metric = None
            if qce:
                def retain_qce_metric(y,w,offsets,scales):
                    # Retained metric buffers have a separate 256 MiB cap;
                    # larger datasets keep bounded one-shot evaluation.
                    groups=len(offsets)-1;rows=len(y)
                    required=28*rows+33*groups+16+16*min((rows+255)//256,4096)
                    if required>1<<28:return None
                    return resources.enter_context(_query_cross_entropy.MetricSession(y,w,offsets,scales,budget=1<<28))
                if has_eval:
                    eval_qce_metric=retain_qce_metric(eval_targets,eval_weight,eval_group_offsets,eval_qce_scales)
                elif selection_metric!=objective_metric and selection_metric.partition(":")[0]=="QueryCrossEntropy":
                    learn_qce_metric=retain_qce_metric(targets,sample_weight,group_offsets,qce_scales)
            if has_feature_penalties:
                session.configure_feature_penalties(
                    ctr_unique_values, model_size_reg=model_size_reg, feature_weights=feature_weights,
                    used_features=None if feature_penalty_state is None else feature_penalty_state["used_features"])
            if permutation_count > 1:
                from ._data import cuda_search_permutation
                session.configure_permutations(
                    permutation_bins,
                    initial_predictions=None if permutation_state is None else permutation_state["predictions"],
                    mvs_lambdas=None if permutation_state is None else permutation_state["mvs_lambdas"],
                    mvs_valid=None if permutation_state is None else permutation_state["mvs_valid"],
                    **({"optimization_predictions": permutation_state["optimization_predictions"]}
                       if classes and permutation_state is not None else {}))
            eval_cursor = None
            if has_eval:
                from ._evaluation import EvaluationCursor
                eval_cursor = resources.enter_context(EvaluationCursor(
                    eval_bins, max_depth=depth, classes=classes or 1, initial_predictions=eval_raw))
            while completed < iterations:
                if permutation_count > 1:
                    session.select_permutation(cuda_search_permutation(
                        native_options.get("random_seed", 0),
                        native_options.get("iteration_offset", 0) + completed, permutation_count))
                step = session.step()
                completed += 1
                if not np.isfinite(step.loss):
                    raise RuntimeError(f"Metal produced a non-finite training {objective} metric.")
                history["learn"][objective_metric].append(float(step.loss))
                if not has_eval and selection_metric != objective_metric:
                    selected_value = _shared_metric(
                        selection_metric, session.predictions(), targets, sample_weight, group_offsets, train_pairs, pair_query_unit_weights=not coupled, qce_scales=qce_scales,
                        subgroup_hashes=subgroup_hashes,qce_session=learn_qce_metric)
                    history["learn"][selection_metric].append(selected_value)
                if has_eval:
                    eval_raw = eval_cursor.add_tree(
                        step.depth, step.split_features[:step.depth], step.split_bins[:step.depth],
                        step.leaf_values[:1 << step.depth], split_types=step.split_types[:step.depth])
                    value = metric(eval_raw, eval_targets, eval_weight, objective_metric,
                                   group_offsets=eval_group_offsets, pair_winners=eval_pair_winners,
                                   pair_losers=eval_pair_losers, pair_weights=eval_pair_weights, query_scales=eval_qce_scales,
                                   subgroup_hashes=eval_subgroup_hashes,qce_session=eval_qce_metric)
                    history["validation"][objective_metric].append(value)
                    if eval_metric is not None:
                        if not (qce and selection_metric == objective_metric):
                            value = _shared_metric(selection_metric, eval_raw, eval_targets, eval_weight,
                                                   eval_group_offsets, eval_pairs, pair_query_unit_weights=not coupled, qce_scales=eval_qce_scales,
                                                   subgroup_hashes=eval_subgroup_hashes,qce_session=eval_qce_metric)
                        if selection_metric == objective_metric:
                            history["validation"][selection_metric][-1] = value
                        else:
                            history["validation"][selection_metric].append(value)
                    improved = value > best_value if maximize else value < best_value
                    if improved:
                        best_iteration, best_value = completed - 1, value
                if callback is not None:
                    keep_training = callback(SimpleNamespace(
                        iteration=completed, metrics=copy.deepcopy(history)))
                    if keep_training is False:
                        stop_reason = "callback"
                if (early_stopping_rounds is not None
                        and completed - 1 - best_iteration >= early_stopping_rounds):
                    stop_reason = "early_stopping"
                if path is not None and time.monotonic() - last_snapshot >= snapshot_interval:
                    result = _with_validation_stats(_merge(prior, session.result()), prior,
                                                    eval_cursor.stats if eval_cursor else None)
                    _with_permutation_stats(result, permutation_count)
                    if permutation_count > 1:
                        permutation_state = session.permutation_state
                    if has_feature_penalties:
                        feature_penalty_state = session.feature_penalty_state
                    if classes and permutation_count == 1:
                        optimization_predictions = session.optimization_predictions()
                    if ordered:
                        ordered_state = session.state()
                    _write_snapshot(path, fingerprint, result, history, best_iteration, best_value, eval_raw,
                                    permutation_state, feature_penalty_state, optimization_predictions, ordered_state)
                    last_snapshot = time.monotonic()
                if stop_reason != "iterations":
                    break
            result = _with_validation_stats(_merge(prior, session.result()), prior,
                                            eval_cursor.stats if eval_cursor else None)
            if path is not None and permutation_count > 1:
                permutation_state = session.permutation_state
            if path is not None and has_feature_penalties:
                feature_penalty_state = session.feature_penalty_state
            if path is not None and classes and permutation_count == 1:
                optimization_predictions = session.optimization_predictions()
            if path is not None and ordered:
                ordered_state = session.state()
    _with_permutation_stats(result, permutation_count)
    if path is not None:
        _write_snapshot(path, fingerprint, result, history, best_iteration, best_value, eval_raw,
                        permutation_state, feature_penalty_state, optimization_predictions, ordered_state)
    if keep_best and best_iteration + 1 < len(result.depths):
        retained = best_iteration + 1
        for name in _TREE_FIELDS:
            setattr(result, name, getattr(result, name)[:retained].copy())
        estimation_bins = ordered_banks[-1] if ordered_banks is not None else permutation_bins[-1] if permutation_count > 1 else bins
        result.predictions = np.asarray(_predict(estimation_bins, result, bias), dtype=np.float32)
    return _finish(result, history, best_iteration, best_value if has_eval else None,
                   stop_reason, completed, resumed, selection_metric, maximize)


def _finish(result, history, best_iteration, best_value, stop_reason, completed, resumed,
            selection_metric=None, maximize=False):
    result.best_iteration = best_iteration
    result.best_score = {dataset: {name: (max(values) if (name == selection_metric and maximize) or name.partition(":")[0] in _RANKING_METRICS else min(values))
                                  for name, values in metrics.items()}
                         for dataset, metrics in history.items()}
    result.evals_result = copy.deepcopy(history)
    result.stopped_iteration = completed - 1
    result.stats.update(iterations_trained=completed, iterations_retained=len(result.depths),
                        resumed_iterations=resumed, stop_reason=stop_reason,
                        best_iteration=best_iteration, best_validation_score=best_value)
    if selection_metric is not None:
        result.stats.update(selection_metric=selection_metric, metric_maximized=maximize)
    return result
