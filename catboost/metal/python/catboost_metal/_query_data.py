"""Query boundaries, effective weights, and CUDA query objective metrics.

Group and object weights remain separate until preparation, where CUDA's
target-provider product is formed exactly once. No training occurs here.
"""

import numbers

import numpy as np

from ._data import unpack_pool, numeric_array


QUERY_OBJECTIVES = ("QueryRMSE", "QuerySoftMax")
PAIR_OBJECTIVES = ("PairLogit", "PairLogitPairwise")


def validate_offsets(offsets, rows, name="group_offsets"):
    values = np.asarray(offsets)
    if (values.ndim != 1 or len(values) < 2 or values.dtype.kind not in "iu"
            or values[0] != 0 or values[-1] != rows
            or (values < 0).any() or (values > np.iinfo(np.uint32).max).any()
            or (values[1:] <= values[:-1]).any()):
        raise ValueError(f"{name} must be increasing uint32-compatible boundaries from zero to the row count.")
    return np.ascontiguousarray(values, np.uint32)


def _weights(values, rows, name):
    if values is None:
        return np.ones(rows, np.float32)
    result = numeric_array(values, name, 1)
    if result.shape != (rows,) or (result < 0).any():
        raise ValueError(f"{name} must contain one finite nonnegative value per row.")
    return result


def prepare_groups(group_id, rows, sample_weight=None, group_weight=None, *, combine_weights=True):
    """Return original-row query offsets and float32 object-times-group weights."""
    if group_id is None:
        raise ValueError("group_id is required for query training and evaluation.")
    identifiers = np.asarray(group_id, dtype=object)
    if identifiers.shape != (rows,) or rows < 1:
        raise ValueError("group_id must contain one identifier per row.")
    starts, seen, previous = [0], set(), None
    for index, value in enumerate(identifiers):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (str, numbers.Integral))):
            raise ValueError("Group identifiers must be strings or integers.")
        key = ("string", value) if isinstance(value, str) else ("integer", int(value))
        if key != previous:
            if key in seen:
                raise ValueError("Each group must occupy one contiguous block of rows.")
            if index:
                starts.append(index)
            seen.add(key)
            previous = key
    offsets = np.asarray(starts + [rows], np.uint32)
    objects, groups = _weights(sample_weight, rows, "sample_weight"), _weights(group_weight, rows, "group_weight")
    for begin, end in zip(offsets[:-1], offsets[1:]):
        if not np.all(groups[begin:end] == groups[begin]):
            raise ValueError("group_weight must be constant within every group.")
    if not combine_weights:
        # Supplied-pair training obtains row mass from incident pair weights.
        # Original object/group weights do not participate in that objective.
        return offsets, None
    with np.errstate(over="ignore", invalid="ignore"):
        effective = np.ascontiguousarray(objects * groups, np.float32)
    total = effective.sum(dtype=np.float64)
    if not np.isfinite(effective).all() or not 0 < total <= np.finfo(np.float32).max:
        raise ValueError("Effective object-times-group weights must be finite with positive float32 total.")
    return offsets, effective


def unpack_query_pool(X, y=None, sample_weight=None, group_id=None, group_weight=None, cat_features=None,
                      *, combine_weights=True):
    """Read raw Pool data without losing group weights hidden by stock wheels."""
    from catboost import Pool

    pool_names = None
    if isinstance(X, Pool):
        if (not combine_weights and X.num_pairs()
                and (not hasattr(X, "get_pairs") or not hasattr(X, "get_pairs_weight"))):
            raise ValueError("This Pool cannot expose its stored pairs and pair weights. "
                             "Pass the original feature arrays with explicit pairs and pairs_weight.")
        if y is not None or sample_weight is not None:
            raise ValueError("Labels and object weights must come from the supplied Pool.")
        if group_id is not None:
            raise ValueError("Group identifiers must come from the supplied Pool.")
        group_id = X.get_group_id_hash()
        getter = getattr(X, "get_group_weight", None)
        if getter is not None:
            if group_weight is not None:
                raise ValueError("Group weights must come from the supplied Pool.")
            group_weight = getter()
        elif combine_weights:
            raise ValueError("This CatBoost Pool build cannot expose group weights. Use the rebuilt Metal fork, "
                             "or pass the original arrays with explicit group_id and group_weight.")
        elif group_weight is not None:
            raise ValueError("Group weights must come from the supplied Pool.")
        if X.get_baseline().size:
            raise ValueError("Query Pool baselines are not yet supported by the standalone ranker.")
        if (X.get_cat_feature_indices() or X.get_text_feature_indices()
                or X.get_embedding_feature_indices() or cat_features):
            raise ValueError("The standalone query ranker requires numeric Pool features.")
        if X.is_quantized():
            raise ValueError("Prequantized Pool input requires the native CatBoost Metal trainer.")
        y, sample_weight = X.get_label(), X.get_weight()
        pool_names = X.get_feature_names()
        occupied = {name for name in pool_names if name}
        for index, name in enumerate(pool_names):
            if not name:
                candidate = str(index)
                while candidate in occupied:
                    candidate = "feature_" + candidate
                pool_names[index] = candidate
                occupied.add(candidate)
        X = X.get_features()
    raw, targets, sample_weight, cats, names = unpack_pool(X, y, sample_weight, cat_features)
    if pool_names is not None:
        if len(set(pool_names)) != len(pool_names):
            raise ValueError("Pool feature names must be unique after naming anonymous columns.")
        names = pool_names
    offsets, weights = prepare_groups(group_id, len(raw), sample_weight, group_weight,
                                      combine_weights=combine_weights)
    return raw, targets, offsets, weights, cats, names


def validate_subgroup_hashes(values, rows, name="subgroup_hashes"):
    if values is None:
        return None
    values = np.asarray(values)
    if (values.shape != (rows,) or values.dtype.kind not in "iu"
            or (values < 0).any() or (values > np.iinfo(np.uint32).max).any()):
        raise ValueError(f"{name} must contain one uint32-compatible hash per row.")
    return np.array(values, dtype=np.uint32, order="C", copy=True)


def unpack_subgroups(X, subgroup_id, rows):
    """Keep Pool hashes intact; hash array tokens using CatBoost's own routine."""
    from catboost import Pool, _catboost

    if isinstance(X, Pool):
        if subgroup_id is not None:
            raise ValueError("Subgroup identifiers must come from the supplied Pool.")
        getter = getattr(X, "get_subgroup_id_hash", None)
        if getter is None:
            raise ValueError("This Pool cannot expose subgroup metadata. Use the rebuilt Metal fork "
                             "or the original arrays with explicit subgroup_id.")
        return validate_subgroup_hashes(getter(), rows)
    if subgroup_id is None:
        return None
    values = np.asarray(subgroup_id, dtype=object)
    if values.shape != (rows,):
        raise ValueError("subgroup_id must contain one identifier per row.")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, numbers.Integral)) for value in values):
        raise ValueError("Subgroup identifiers must be strings or integers.")
    if not hasattr(_catboost, "_hash_subgroup_ids"):
        raise ValueError("Subgroup identifiers require the rebuilt Metal CatBoost extension.")
    return _catboost._hash_subgroup_ids(values)


def unpack_pairs(X, pairs=None, weights=None):
    """Read stored Pool edges only when their original weights are available."""
    from catboost import Pool

    if isinstance(X, Pool) and X.num_pairs():
        if pairs is not None or weights is not None:
            raise ValueError("Pairs and pair weights must come from the supplied Pool.")
        getter, weight_getter = getattr(X, "get_pairs", None), getattr(X, "get_pairs_weight", None)
        if getter is None or weight_getter is None:
            raise ValueError("This Pool cannot expose its stored pairs and pair weights. "
                             "Pass the original feature arrays with explicit pairs and pairs_weight.")
        pairs, weights = getter(), weight_getter()
        if len(pairs) != X.num_pairs() or len(weights) != X.num_pairs():
            raise ValueError("Pool pair metadata does not match its stored pair count.")
    return pairs, weights


def prepare_pair_arrays(winners, losers, weights, rows, offsets=None):
    """Validate literal supplied edges, retaining duplicates and original order."""
    winners, losers = np.asarray(winners), np.asarray(losers)
    if (winners.ndim != 1 or winners.dtype.kind not in "iu"
            or losers.shape != winners.shape or losers.dtype.kind not in "iu"
            or not 0 < len(winners) <= np.iinfo(np.uint32).max // 2
            or (winners < 0).any() or (winners >= rows).any()
            or (losers < 0).any() or (losers >= rows).any()
            or (winners == losers).any()):
        raise ValueError("PairLogit requires nonempty valid winner/loser arrays without self-pairs; "
                         "automatic pair generation is not yet supported.")
    winners, losers = (np.ascontiguousarray(value, np.uint32) for value in (winners, losers))
    if offsets is not None:
        offsets = validate_offsets(offsets, rows)
        if np.any(np.searchsorted(offsets, winners, side="right") != np.searchsorted(offsets, losers, side="right")):
            raise ValueError("Every pair's winner and loser must belong to the same query group.")
    weights = _weights(weights, len(winners), "pairs_weight")
    if not 0 < 2 * weights.sum(dtype=np.float64) <= np.finfo(np.float32).max:
        raise ValueError("Pair weights must have positive finite float32 incident mass.")
    return winners, losers, weights


def prepare_pairs(pairs, weights, rows, offsets):
    """Convert the public [winner, loser] table to validated native arrays."""
    values = np.asarray(pairs)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Explicit pairs must be a nonempty [pair_count, 2] winner/loser matrix; "
                         "automatic pair generation is not yet supported.")
    return prepare_pair_arrays(values[:, 0], values[:, 1], weights, rows, offsets)


def pair_metric(raw, winners, losers, weights=None, objective="PairLogit"):
    """CUDA supplied-pair loss, or strict weighted pair-order accuracy."""
    predictions = np.asarray(raw, np.float64)
    if predictions.ndim != 1 or not np.isfinite(predictions).all():
        raise ValueError("Pair metrics require a finite raw prediction vector.")
    winners, losers, weights = prepare_pair_arrays(winners, losers, weights, len(predictions))
    difference = predictions[winners] - predictions[losers]
    if objective == "PairLogit":
        losses = np.logaddexp(0., -difference)
    elif objective == "PairAccuracy":
        losses = difference > 0
    else:
        raise ValueError(f"Unsupported supplied-pair metric: {objective}.")
    return float(np.average(losses, weights=weights))


def query_metric(raw, targets, weights, offsets, objective, query_beta=1.0, query_lambda=0.01):
    """Positive CUDA objective, normalized across queries by its native denominator."""
    predictions, labels = np.asarray(raw, np.float64), np.asarray(targets, np.float64)
    if predictions.ndim != 1 or labels.shape != predictions.shape:
        raise ValueError("Query metrics need one raw prediction and target per row.")
    offsets = validate_offsets(offsets, len(predictions))
    effective = np.ones(len(labels), np.float64) if weights is None else np.asarray(weights, np.float64)
    if (effective.shape != labels.shape or not np.isfinite(predictions).all()
            or not np.isfinite(labels).all() or not np.isfinite(effective).all() or (effective < 0).any()):
        raise ValueError("Query metrics require finite data and nonnegative weights.")
    if objective not in QUERY_OBJECTIVES:
        raise ValueError(f"Unsupported query objective: {objective}.")
    if not np.isfinite(query_beta) or not np.isfinite(query_lambda):
        raise ValueError("Query parameters must be finite.")
    if objective == "QuerySoftMax" and (labels < 0).any():
        raise ValueError("QuerySoftMax targets must be nonnegative.")
    numerator = denominator = 0.0
    for begin, end in zip(offsets[:-1], offsets[1:]):
        a, y, w = predictions[begin:end], labels[begin:end], effective[begin:end]
        active = w > 0
        if not active.any():
            continue
        a, y, w = a[active], y[active], w[active]
        if objective == "QueryRMSE":
            residual = y - a
            total = w.sum()
            centered = residual - np.dot(w, residual) / total
            numerator += np.dot(w, centered * centered)
            denominator += total
        else:
            weighted_targets = w * y
            target_total = weighted_targets.sum()
            if target_total == 0:
                continue
            reference = np.max(a) if query_beta >= 0 else np.min(a)
            logits = query_beta * (a - reference) + np.log(w)
            maximum = np.max(logits)
            log_probabilities = (logits - maximum) - np.log(np.exp(logits - maximum).sum())
            numerator -= np.dot(weighted_targets, log_probabilities)
            denominator += target_total
    if denominator <= 0:
        raise ValueError("Query metrics require positive effective weight/target total.")
    value = numerator / denominator
    if objective == "QueryRMSE":
        value = np.sqrt(value)
    if not np.isfinite(value):
        raise ValueError("Query metric became non-finite.")
    return float(value)
