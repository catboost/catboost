"""Independent rounded-model prefix replay for the uncertainty CLI workflow.

CUDA doc_parallel_boosting.h scales each model before appending it. Model-based
evaluation appends those saved float leaves to reconstruct its starting point.
The accepted Metal vector training kernel instead evaluates base + rate * raw
with a possible fused multiply-add. Near a tied CTR split, their small cursor
difference can change later trees, so continuous training is not that oracle.
"""

import json
from pathlib import Path
import struct
import subprocess

import numpy as np


def _saved_leaves(snapshot, trees):
    payload = snapshot.read_bytes()
    matches = []
    at = payload.find(struct.pack("<I", 0x4D4D4231))
    while at >= 0:
        if at + 24 <= len(payload):
            _, permutations, capacity, dimension, count = struct.unpack_from("<IIIIQ", payload, at)
            if (permutations == 4 and dimension == 2 and 1 <= capacity <= 65536
                    and count == trees * permutations * capacity * dimension
                    and at + 24 + count * 4 == len(payload)):
                matches.append(np.frombuffer(payload, "<f4", count=count, offset=at + 24)
                               .reshape(trees, permutations, capacity, dimension).copy())
        at = payload.find(struct.pack("<I", 0x4D4D4231), at + 1)
    assert len(matches) == 1 and np.isfinite(matches[0]).all()
    return matches[0]


def _pool_order(directory, rows, seed):
    source = Path(__file__).with_name("model_based_pool_shuffle_probe.cpp")
    root = source.parents[3]
    executable = directory / "pool_shuffle_probe"
    result = subprocess.run(["clang++", "-std=c++20", "-DNDEBUG", "-nostdinc++", "-I", str(root),
        "-I", str(root / "contrib/libs/cxxsupp/libcxx/include"), str(source), "-o", str(executable)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(executable), str(seed), str(rows)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    order = np.asarray(result.stdout.split(), np.uint32)
    np.testing.assert_array_equal(np.sort(order), np.arange(rows))
    return order


def _prepared_banks(data, model, directory):
    from catboost.utils import calculate_quantization_grid
    from catboost_metal._data import cuda_history_order

    params = model["model_info"]["params"]
    if isinstance(params, str):
        params = json.loads(params)
    train = np.asarray([line.split("\t") for line in data.train.read_text().splitlines()], object)
    test = np.asarray([line.split("\t") for line in data.test.read_text().splitlines()], object)
    order = _pool_order(directory, len(train), params["random_seed"])
    train = train[order]
    targets, weights = train[:, 0].astype(np.float32), train[:, 1].astype(np.float32)
    x, tx = train[:, 2:], test[:, 2:]
    assert x.shape[1] == 4
    binarization = params["data_processing_options"]["float_features_binarization"]
    grids, columns, test_columns, numeric = [], [], [], {}

    def grid(values, options):
        return np.asarray(calculate_quantization_grid(np.asarray(values, np.float32),
            options["border_count"], border_type=options["border_type"]), np.float32)

    def append(values, test_values, borders):
        feature = len(grids)
        grids.append(borders)
        columns.append(np.searchsorted(borders, values, side="left").astype(np.uint8))
        test_columns.append(np.searchsorted(borders, test_values, side="left").astype(np.uint8))
        return feature

    for flat in range(1, 4):
        values, test_values = x[:, flat].astype(np.float32), tx[:, flat].astype(np.float32)
        numeric[flat - 1] = append(values, test_values, grid(values, binarization))
    target_options = params["cat_feature_params"]["target_binarization"]
    target_borders = grid(targets, target_options)
    assert target_borders.size == 1
    target_bins = (targets > target_borders[0]).astype(np.uint32)
    descriptors = {}
    ctr_values = []
    categories = x[:, 0]
    totals = {category: (int(target_bins[categories == category].sum()), int((categories == category).sum()))
              for category in np.unique(categories)}
    for description in params["cat_feature_params"]["simple_ctrs"]:
        kind = description["ctr_type"]
        assert kind in ("Borders", "FeatureFreq") and description["prior_estimation"] == "No"
        for numerator, denominator in description["priors"]:
            banks = []
            for permutation in range(4):
                seen = {}
                values = np.empty(len(x), np.float32)
                for row in cuda_history_order(len(x), permutation):
                    category = categories[row]
                    total, count = seen.get(category, (0, 0))
                    values[row] = np.float32((total + numerator) / (count + denominator))
                    seen[category] = (total + int(target_bins[row]), count + 1)
                if kind == "FeatureFreq":
                    values = np.asarray([(totals[category][1] + numerator) / (len(x) + denominator)
                                         for category in categories], np.float32)
                banks.append(values)
            test_values = np.asarray([
                ((totals.get(category, (0, 0))[0] + numerator) /
                 (totals.get(category, (0, 0))[1] + denominator)) if kind == "Borders" else
                (totals.get(category, (0, 0))[1] + numerator) / (len(x) + denominator)
                for category in tx[:, 0]], np.float32)
            feature = append(banks[0], test_values, grid(banks[0], description["ctr_binarization"]))
            descriptors[kind, float(numerator), float(denominator), 0] = feature
            ctr_values.append((feature, banks))
    bins = np.stack(columns)
    banks = np.stack([bins.copy() for _ in range(4)])
    for feature, values in ctr_values:
        for permutation in range(4):
            banks[permutation, feature] = np.searchsorted(grids[feature], values[permutation], side="left")
    assert any(not np.array_equal(banks[0], bank) for bank in banks[1:])
    features = np.asarray([feature for feature, borders in enumerate(grids) for _ in borders], np.uint32)
    thresholds = np.asarray([border for borders in grids for border in range(len(borders))], np.uint32)
    # JSON split_index addresses the exported feature grid, which may omit
    # unused borders. Map semantic CTR descriptors into the independently built
    # complete search grid rather than assuming native dense feature IDs.
    model_splits = {}
    index = 0
    for feature in model["features_info"]["float_features"]:
        for border in feature["borders"]:
            dense = numeric[feature["feature_index"]]
            model_splits[index] = dense, int(np.flatnonzero(grids[dense] == np.float32(border))[0])
            index += 1
    for ctr in model["features_info"]["ctrs"]:
        dense = descriptors[ctr["ctr_type"], float(ctr["prior_numerator"]),
                            float(ctr["prior_denomerator"]), ctr["target_border_idx"]]
        for border in ctr["borders"]:
            matches = np.flatnonzero(grids[dense] == np.float32(border))
            assert matches.size == 1, "Independent CTR grid must contain every saved baseline border"
            model_splits[index] = dense, int(matches[0])
            index += 1
    return params, targets, weights, banks, np.stack(test_columns), features, thresholds, model_splits


def rounded_uncertainty_prefix_history(data, model, snapshot, directory):
    from catboost_metal import _multiclass
    from catboost_metal._data import cuda_search_permutation
    from test_vector_permutation_leaves import LeafBanks

    params, target, weight, banks, test_bins, features, thresholds, splits = _prepared_banks(data, model, directory)
    trees = model["oblivious_trees"]
    leaves = _saved_leaves(snapshot, len(trees))
    assert leaves.shape == (8, 4, 4, 2)

    def model_ids(tree, bins):
        ids = np.zeros(bins.shape[1], np.uint32)
        for level, split in enumerate(tree["splits"]):
            feature, border = splits[split["split_index"]]
            ids |= (bins[feature] > border).astype(np.uint32) << level
        return ids

    def step_ids(tree, bins):
        ids = np.zeros(bins.shape[1], np.uint32)
        for level, (feature, border) in enumerate(zip(tree.split_features, tree.split_bins)):
            ids |= (bins[feature] > border).astype(np.uint32) << level
        return ids

    args = dict(bins=banks[0], targets=target, candidate_features=features, candidate_bins=thresholds,
        classes=2, objective="RMSEWithUncertainty", sample_weight=weight, iterations=8, depth=2,
        learning_rate=.2, l2_leaf_reg=2, leaf_estimation_method="Newton", leaf_estimation_iterations=1,
        leaf_estimation_backtracking="No", score_function="Cosine", bootstrap_type="No",
        random_strength=0, random_seed=params["random_seed"])
    rounded = np.zeros((4, len(target), 2), np.float32)
    evaluation = np.zeros((len(data.test_targets), 2), np.float32)
    prefixes, test_prefixes = [rounded.copy()], [evaluation.copy()]
    budget = np.zeros_like(rounded, dtype=np.float64)
    has_rounding_difference = False
    # First validate all banks against ordinary training, independently of the
    # new analysis command. Accumulate explicit per-operation float ULP bounds
    # for FMA training versus replay of already rounded, scaled leaves.
    with _multiclass.Session(**args) as ordinary:
        ordinary.configure_permutations(banks)
        exported = LeafBanks(ordinary, 4, 4)
        for iteration, tree in enumerate(trees):
            previous = ordinary.permutation_state["predictions"]
            ordinary.select_permutation(cuda_search_permutation(params["random_seed"], iteration, 4))
            step = ordinary.step()
            np.testing.assert_array_equal(exported.read(), leaves[iteration])
            delta = np.empty_like(rounded)
            for p in range(4):
                ids = model_ids(tree, banks[p])
                np.testing.assert_array_equal(step_ids(step, banks[p]), ids)
                delta[p] = leaves[iteration, p, ids]
            rounded = np.float32(rounded + delta)
            actual = ordinary.permutation_state["predictions"]
            budget += (np.spacing(np.abs(previous)).astype(float) + np.spacing(np.abs(delta)).astype(float)
                       + np.spacing(np.abs(rounded)).astype(float))
            assert np.all(np.abs(actual.astype(float) - rounded.astype(float)) <= budget)
            has_rounding_difference |= not np.array_equal(actual, rounded)
            evaluation = np.float32(evaluation + leaves[iteration, -1, model_ids(tree, test_bins)])
            prefixes.append(rounded.copy())
            test_prefixes.append(evaluation.copy())
    assert has_rounding_difference, "The explicit FMA versus rounded-leaf boundary must be exercised"
    assert any(not np.array_equal(prefixes[3][0], bank) for bank in prefixes[3][1:])

    def metric(raw, labels, weights):
        raw = raw.astype(float)
        loss = .5 * np.log(2 * np.pi) + raw[:, 1] + .5 * np.exp(-2 * raw[:, 1]) * (labels - raw[:, 0]) ** 2
        return np.average(loss, weights=weights)

    histories = {}
    for fold, start in enumerate((3, 5)):
        cursor = test_prefixes[start].copy()
        learn_values, test_values = [], []
        with _multiclass.Session(**(args | {"iterations": 2, "initial_predictions": prefixes[start][-1]})) as trial:
            trial.configure_permutations(banks, initial_predictions=prefixes[start])
            np.testing.assert_array_equal(trial.permutation_state["predictions"], prefixes[start])
            for local in range(2):
                trial.select_permutation(cuda_search_permutation(params["random_seed"], start + local, 4))
                step = trial.step()
                cursor = np.float32(cursor + step.leaf_values[step_ids(step, test_bins)])
                learn_values.append(metric(trial.predictions(), target, weight))
                test_values.append(metric(cursor, data.test_targets, data.test_weights))
        histories[fold, "learn"] = np.asarray(learn_values)[:, None]
        histories[fold, "test"] = np.asarray(test_values)[:, None]
    return histories
