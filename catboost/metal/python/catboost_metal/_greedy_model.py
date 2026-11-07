"""Export greedy Metal trees through CatBoost's standard JSON model schema.

The recursive ``trees`` format is read by GetNonSymmetricModelTrees in
libs/model/model_export/json_model_helpers.cpp. No model fitting is involved.
"""
import copy
import json
import numbers
from types import SimpleNamespace

import numpy as np


_INTERNAL = np.iinfo(np.uint32).max
_MAX_LEAVES = 65536
_MAX_CHILD_OFFSET = np.iinfo(np.uint16).max


def _vector(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite numeric vector.")
    return value


def tree_json(tree, borders, *, split_descriptors=None, leaf_estimation_method=None):
    """Convert one flat tree using numeric feature borders in training order.

    Internal nodes are [feature, border_bin, type, left, right, UINT32_MAX].
    Leaves refer to the independently ordered leaf_values/leaf_weights arrays.
    Graph validation prevents silently exporting a different routing topology.
    """
    nodes = np.asarray(tree.nodes)
    values = np.asarray(tree.leaf_values)
    if (values.ndim not in (1, 2) or values.dtype.kind not in "iuf" or not np.isfinite(values).all()
            or values.ndim == 2 and not 2 <= values.shape[1] <= 64):
        raise ValueError("leaf_values must be finite scalar values or a [leaves, 2..64 outputs] matrix.")
    weights = _vector(tree.leaf_weights, "leaf_weights")
    if (weights.shape != (len(values),) or not len(values)
            or leaf_estimation_method != "Simple" and (weights < 0).any()):
        raise ValueError("Leaf values and valid weights must have matching nonempty shapes; signed weights require Simple leaves.")
    if len(values) > _MAX_LEAVES:
        raise ValueError("Greedy model trees support at most 65536 leaves.")
    if (nodes.ndim != 2 or nodes.shape[1] != 6 or nodes.dtype.kind not in "iu"
            or (nodes < 0).any() or (nodes > _INTERNAL).any()
            or len(nodes) != 2 * len(values) - 1):
        raise ValueError("nodes must be a full binary tree as uint32[2*leaves-1,6].")
    offsets = np.cumsum([0, *[len(feature) for feature in borders]], dtype=np.int64)
    visited, leaves, preorder = set(), set(), []
    exported = [None] * len(nodes)
    pending = [0]
    while pending:
        index = pending.pop()
        if index >= len(nodes) or index in visited:
            raise ValueError("Tree graph has an invalid child, cycle, or shared node.")
        visited.add(index)
        preorder.append(index)
        feature, border, kind, left, right, leaf = map(int, nodes[index])
        if leaf != _INTERNAL:
            if leaf >= len(values) or leaf in leaves:
                raise ValueError("Tree leaves must reference each value exactly once.")
            leaves.add(leaf)
            exported[index] = {"value": float(values[leaf]) if values.ndim == 1 else values[leaf].tolist(),
                               "weight": float(weights[leaf])}
            continue
        if split_descriptors is not None:
            split = split_descriptors.get((feature, kind, border))
            if split is None:
                raise ValueError("A tree split has a feature, bin or type inconsistent with its feature layout.")
        elif kind != 0:
            raise ValueError("Greedy model JSON export currently supports numeric features only.")
        elif feature >= len(borders) or border >= len(borders[feature]):
            raise ValueError("A tree split references a missing numeric feature or border.")
        else:
            split = {"split_type": "FloatFeature", "float_feature_index": feature,
                     "border": float(borders[feature][border]),
                     "split_index": int(offsets[feature] + border)}
        exported[index] = {"split": dict(split)}
        # The upstream JSON reader writes left-first preorder. Validate that
        # order, independently of the session's flat node and leaf ID order.
        pending.extend((right, left))
    if len(visited) != len(nodes) or len(leaves) != len(values):
        raise ValueError("Tree contains unreachable nodes or unreferenced leaf values.")
    positions = {node: position for position, node in enumerate(preorder)}
    for index in reversed(preorder):
        if int(nodes[index, 5]) == _INTERNAL:
            left, right = map(int, nodes[index, 3:5])
            # TNonSymmetricTreeStepNode stores uint16 child offsets. Without
            # this check, upstream's JSON importer silently truncates a large
            # left subtree's right-child offset and corrupts model routing.
            if any(positions[child] - positions[index] > _MAX_CHILD_OFFSET for child in (left, right)):
                raise ValueError("CatBoost JSON preorder child offsets must fit uint16; this tree needs a native model builder.")
            exported[index]["left"] = exported[left]
            exported[index]["right"] = exported[right]
    return exported[0]


def iter_model_json(document):
    """Serialize model JSON iteratively, including trees deeper than Python's stack.

    Produces ordinary UTF-8-compatible JSON text chunks without changing the
    process recursion limit. Circular containers and nonfinite numbers fail
    explicitly, as they do with ``json.dumps(..., allow_nan=False)``.
    """
    active, pending = set(), [("value", document)]
    while pending:
        operation, value = pending.pop()
        if operation == "text":
            yield value
        elif operation == "end":
            identifier, closing = value
            active.remove(identifier)
            yield closing
        elif isinstance(value, (dict, list, tuple)):
            identifier = id(value)
            if identifier in active:
                raise ValueError("Circular reference in model JSON.")
            active.add(identifier)
            mapping = isinstance(value, dict)
            yield "{" if mapping else "["
            pending.append(("end", (identifier, "}" if mapping else "]")))
            entries = list(value.items()) if mapping else value
            for index in range(len(entries) - 1, -1, -1):
                if mapping:
                    key, item = entries[index]
                    if not isinstance(key, str):
                        raise TypeError("Model JSON object keys must be strings.")
                    pending.extend((("value", item), ("text", ":"), ("value", key)))
                else:
                    pending.append(("value", entries[index]))
                if index:
                    pending.append(("text", ","))
        else:
            yield json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))


def dumps_model_json(document):
    """Return model JSON text without recursive Python container traversal."""
    return "".join(iter_model_json(document))


def _layout_model(grid, layout, feature_names, bias, stats):
    """Share the established feature metadata, with explicit greedy split IDs."""
    from .regressor import _model_json
    from ._categorical import one_hot_split_json
    from ._ctr_model import ctr_feature_json, ctr_table_json, online_ctr_split_json

    raw_count = len(layout.names)
    if (not 0 < raw_count <= len(grid) or len(layout.has_nans) != len(grid)
            or len(layout.borders) != len(grid)
            or any(not np.array_equal(before, after) for before, after in zip(layout.borders, grid))):
        raise ValueError("Feature layout names, NaN flags and borders must match the supplied feature grid.")
    if layout.nan_mode not in ("Forbidden", "Min", "Max"):
        raise ValueError("Feature layout nan_mode must be Forbidden, Min or Max.")
    if layout.nan_mode == "Forbidden" and any(layout.has_nans):
        raise ValueError("A Forbidden NaN layout cannot mark features as containing NaNs.")
    categorical, ctrs = layout.categorical, layout.ctrs
    for name, mapping, low, high in (("categorical", categorical, 0, raw_count),
                                    ("CTR", ctrs, raw_count, len(grid))):
        if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, numbers.Integral)
               or not low <= index < high for index in mapping):
            raise ValueError(f"Invalid {name} feature index in feature layout.")
    if set(ctrs) != set(range(raw_count, len(grid))):
        raise ValueError("Every generated feature must have a CTR layout entry.")
    if any(len(grid[index]) or layout.has_nans[index] for index in categorical):
        raise ValueError("Categorical feature slots must have empty numeric borders and no NaNs.")
    if any(layout.has_nans[index] or ctr.source_feature not in categorical for index, ctr in ctrs.items()):
        raise ValueError("Each CTR must reference a categorical source and cannot contain NaNs.")
    names = list(layout.names if feature_names is None else feature_names)
    if len(names) != raw_count or any(not isinstance(name, str) for name in names):
        raise ValueError("feature_names must contain one string per original feature.")
    prepared = copy.copy(layout)
    prepared.names = names
    prepared.borders = grid
    # The common builder is only asked for feature metadata and full CTR tables;
    # it sees no symmetric trees and invokes no quantization or fitting code.
    empty = SimpleNamespace(depths=(), stats={"device": stats.get("device", "")})
    model = _model_json(grid, empty, bias, "Cosine", prepared)
    del model["oblivious_trees"]
    # Score choice belongs to the caller's training metadata, not this exporter.
    model["model_info"].pop("metal_score_function", None)
    descriptors, offset = {}, 0
    float_indices = {feature: index for index, feature in enumerate(
        feature for feature in range(raw_count) if feature not in categorical)}
    cat_indices = {feature: index for index, feature in enumerate(sorted(categorical))}
    for feature, index in float_indices.items():
        for border, threshold in enumerate(grid[feature]):
            descriptors[feature, 0, border] = {"split_type": "FloatFeature",
                "float_feature_index": index, "border": float(threshold), "split_index": offset + border}
        offset += len(grid[feature])
    for feature in sorted(categorical):
        encoding = categorical[feature]
        for border in encoding.candidate_bins:
            descriptors[feature, 1, int(border)] = one_hot_split_json(
                cat_indices[feature], int(border), encoding, offset + int(border))
        offset += len(encoding.candidate_bins)
    ctr_descriptors, ctr_tables = [], {}
    for feature, ctr in ctrs.items():
        result = ctr.result
        target_index = getattr(result, "target_border_idx", 0)
        # CtrResult stores a single positive-count vector, so only the binary
        # target histories it can reconstruct are representable here.
        if ((result.ctr_type == "Borders" and target_index != 0)
                or (result.ctr_type == "Buckets" and target_index not in (0, 1))):
            raise ValueError("This CTR layout stores binary target counts; its target_border_idx is unsupported.")
        descriptor = ctr_feature_json(cat_indices[ctr.source_feature], result.ctr_type, grid[feature],
            target_border_idx=target_index, prior_numerator=result.prior_numerator,
            prior_denominator=result.prior_denominator)
        extra = {} if result.ctr_type == "FeatureFreq" else {"sums": result.sums}
        if result.ctr_type == "Buckets" and target_index == 0:
            extra = {"class_counts": np.column_stack((result.sums, result.counts - result.sums)).astype(np.int64)}
        table = ctr_table_json(result.hashes, result.counts, ctr_type=result.ctr_type, **extra)
        identifier = descriptor["identifier"]
        if identifier in ctr_tables and ctr_tables[identifier] != table:
            raise ValueError("CTR configurations sharing an identifier must have the same full statistics table.")
        ctr_tables[identifier] = table
        ctr_descriptors.append(descriptor)
        for border, threshold in enumerate(grid[feature]):
            descriptors[feature, 0, border] = online_ctr_split_json(float(threshold), offset + border, target_index)
        offset += len(grid[feature])
    if ctrs:
        model["features_info"]["ctrs"] = ctr_descriptors
        model["ctr_data"] = ctr_tables
    return model, descriptors


def model_json(result, borders, *, bias=0., objective="RMSE", feature_names=None,
               grow_policy="Lossguide", layout=None, objective_param=None, loss_parameters=None,
               leaf_estimation_method=None):
    """Return a standard CatBoost JSON dict for greedy training output.

    Leaf values already contain the learning rate. ``bias`` is the common
    initial prediction; per-row external baselines are not part of the model.
    A FeatureLayout preserves original feature IDs, NaN routing, one-hot hashes
    and final inference CTR tables. Without a layout, only numeric features are
    supported. Training permutation CTR values are never exported as full tables.
    Simple leaf weights preserve the sampled statistics used during search,
    including signed query curvature.
    """
    from ._greedy import LEAF_METHODS, OBJECTIVES, OBJECTIVE_PARAMETERS, objective_parameter
    from ._options import parse_loss
    vector = objective in ("MultiClass", "MultiClassOneVsAll", "RMSEWithUncertainty")
    if leaf_estimation_method is None:
        leaf_estimation_method = result.stats.get("leaf_estimation_method")
    if leaf_estimation_method is not None and leaf_estimation_method not in LEAF_METHODS:
        raise ValueError("Invalid greedy leaf_estimation_method for model export.")
    if objective not in OBJECTIVES and not vector:
        raise ValueError("objective must be a supported scalar or vector greedy objective.")
    parameters = {} if loss_parameters is None else dict(loss_parameters)
    parameter_name = OBJECTIVE_PARAMETERS.get(objective)
    if parameter_name:
        supplied = parameters.get(parameter_name, objective_param)
        validated = objective_parameter(objective, supplied)
        if objective_param is not None and validated != objective_parameter(objective, objective_param):
            raise ValueError("objective_param and loss_parameters disagree.")
        parameters.setdefault(parameter_name, float(supplied) if supplied is not None else validated)
    elif not vector:
        objective_parameter(objective, objective_param)
    elif objective_param is not None and objective_param != 1:
        raise ValueError("Vector greedy objectives do not have a scalar objective parameter.")
    description = objective + (":" + ";".join(f"{key}={value}" for key, value in parameters.items()) if parameters else "")
    if objective in ("QueryRMSE", "QuerySoftMax", "PairLogit", "YetiRank"):
        from .ranker import _query_loss
        _, parameters = _query_loss(description)
    else:
        _, _, parameters, _ = parse_loss(description, classifier=objective in ("Logloss", "CrossEntropy", "MultiClass", "MultiClassOneVsAll"))
    if grow_policy not in ("Depthwise", "Lossguide", "Region"):
        raise ValueError("grow_policy must be Depthwise, Lossguide, or Region.")
    if not result.trees:
        raise ValueError("At least one completed tree is required for model export.")
    first = np.asarray(result.trees[0].leaf_values)
    dimensions = first.shape[1] if first.ndim == 2 else 1
    if (vector and not 2 <= dimensions <= 64 or not vector and dimensions != 1
            or objective == "RMSEWithUncertainty" and dimensions != 2):
        raise ValueError("Greedy leaf dimensions do not match the objective.")
    for tree in result.trees:
        values = np.asarray(tree.leaf_values)
        if (values.ndim != (2 if vector else 1) or vector and values.shape[1] != dimensions):
            raise ValueError("Every greedy tree must have the same output dimensions.")
    bias = np.asarray(bias)
    if (bias.shape not in ((), (dimensions,) if vector else ()) or bias.dtype.kind not in "iuf"
            or not np.isfinite(bias).all()):
        raise ValueError("bias must be finite and scalar, or contain one value per vector output.")
    bias = np.broadcast_to(bias, (dimensions,)).astype(np.float64) if vector else float(bias)
    grid = []
    for feature in borders:
        feature = _vector(feature, "borders")
        with np.errstate(over="ignore", invalid="ignore"):
            feature = feature.astype(np.float32)
        if len(feature) > 255 or not np.isfinite(feature).all() or (np.diff(feature) <= 0).any():
            raise ValueError("Each feature needs at most 255 strictly increasing finite float32 borders.")
        grid.append(feature)
    if not grid:
        raise ValueError("At least one feature border vector is required.")
    descriptors = None
    if layout is None:
        names = [str(index) for index in range(len(grid))] if feature_names is None else list(feature_names)
        if len(names) != len(grid) or any(not isinstance(name, str) for name in names):
            raise ValueError("feature_names must contain one string per feature.")
        features = [{"feature_index": index, "flat_feature_index": index,
                     "feature_id": names[index], "has_nans": False,
                     "nan_value_treatment": "AsIs", "borders": feature.tolist()}
                    for index, feature in enumerate(grid)]
        model = {"features_info": {"float_features": features},
                 "scale_and_bias": [1.0, np.asarray(bias).reshape(-1).tolist()], "model_info": {}}
    else:
        model, descriptors = _layout_model(grid, layout, feature_names, bias, result.stats)
    trees = [tree_json(tree, grid, split_descriptors=descriptors,
                       leaf_estimation_method=leaf_estimation_method) for tree in result.trees]
    if not trees:
        raise ValueError("At least one completed tree is required for model export.")
    model["trees"] = trees
    model["model_info"].update({"metal_backend": "METAL", "metal_grow_policy": grow_policy,
                               "metal_device": result.stats.get("device", ""),
                               "params": {"loss_function": {"type": objective, "params": {
                                              key: str(value) for key, value in parameters.items()}},
                                          "tree_learner_options": {"grow_policy": grow_policy}}})
    if leaf_estimation_method == "Simple":
        model["model_info"]["params"]["tree_learner_options"].update(
            leaf_estimation_method="Simple", leaf_estimation_iterations=1)
        model["model_info"]["metal_leaf_weight_semantics"] = "bootstrapped weak score weights"
    if objective == "Lq":
        model["model_info"]["metal_objective_scope"] = (
            "Lq uses CUDA pointwise derivative equations; upstream CUDA does not register Lq for greedy grow policies.")
    return model
