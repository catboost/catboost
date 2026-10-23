"""Explain the historical Cloudness L2 difference using an earlier CTR tie.

This is a diagnostic counterfactual, not a production tie-policy override.
It records normal Metal training, verifies identical train/test predicates for
an alternate CTR prior, then resumes from six unchanged trees with the alternate
used-CTR state. Neither branch invokes CPU CatBoost training. Quantization and
category hashing use the existing host helpers.

Run with PYTHONPATH=catboost/metal/python and a Python environment containing
numpy and catboost. Outputs retain the actual errors against historical CUDA
fixtures; no tolerance is changed and no live NVIDIA run is implied.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from catboost.utils import calculate_quantization_grid
from catboost_metal import _multiclass
from catboost_metal._categorical import cat_feature_hashes


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "catboost/pytest/data/cloudness_small"


def prepare_candidates():
    learn = np.genfromtxt(DATA / "train_small", delimiter="\t", dtype=str)
    test = np.genfromtxt(DATA / "test_small", delimiter="\t", dtype=str)
    targets, test_targets = learn[:, 0].astype(np.uint32), test[:, 0].astype(np.uint32)
    learn, test = learn[:, 1:], test[:, 1:]
    cats = [int(line.split()[0]) - 1 for line in (DATA / "train.cd").read_text().splitlines()
            if line.split()[1] == "Categ"]
    nums = [i for i in range(learn.shape[1]) if i not in cats]
    target_border = calculate_quantization_grid(targets.astype(np.float32), 1, border_type="MinEntropy")[0]
    ctr_target = (targets > target_border).astype(np.float32)
    columns, counts, candidates, train_right, test_right = [], [], [], [], []

    def append(values, validation, borders, descriptor, unique_count=0, onehot=False):
        feature = len(columns)
        columns.append(np.searchsorted(borders, values, side="left").astype(np.uint8))
        counts.append(unique_count)
        for border, threshold in enumerate(borders):
            candidates.append((feature, border, int(onehot), (*descriptor, float(threshold))))
            train_right.append(values == threshold if onehot else values > threshold)
            test_right.append(validation == threshold if onehot else validation > threshold)

    for index, column in enumerate(nums):
        values, validation = learn[:, column].astype(np.float32), test[:, column].astype(np.float32)
        borders = calculate_quantization_grid(values, 128, border_type="GreedyLogSum")
        if len(borders):
            append(values, validation, borders, ("float", index))
    for index, column in enumerate(cats):
        values, validation = cat_feature_hashes(learn[:, column]), cat_feature_hashes(test[:, column])
        unique_count = len(np.unique(np.r_[values, validation]))
        if unique_count <= 2:
            borders = np.unique(values)
            if unique_count > 1 and len(borders) > 1:
                append(values, validation, borders, ("onehot", index), onehot=True)
            continue
        seen, total = {}, {}
        prefix, prefix_count = np.empty(len(targets), np.float32), np.empty(len(targets), np.float32)
        for row, category in enumerate(values):
            prefix[row], prefix_count[row] = total.get(category, 0), seen.get(category, 0)
            seen[category] = seen.get(category, 0) + 1
            total[category] = total.get(category, 0) + ctr_target[row]
        for prior in (0.0, 0.5, 1.0):
            ctr = (prefix + np.float32(prior)) / (prefix_count + np.float32(1))
            ev = np.array([(total.get(c, 0) + prior) / (seen.get(c, 0) + 1)
                           for c in validation], np.float32)
            borders = calculate_quantization_grid(ctr, 15, border_type="Uniform")
            append(ctr, ev, borders, ("ctr", index, "Borders", prior), unique_count)
        frequency = np.array([seen[c] / (len(targets) + 1) for c in values], np.float32)
        ev = np.array([seen.get(c, 0) / (len(targets) + 1) for c in validation], np.float32)
        borders = calculate_quantization_grid(frequency, 15, border_type="MinEntropy")
        append(frequency, ev, borders, ("ctr", index, "FeatureFreq", 0.0), unique_count)
    return (np.array(columns), targets, test_targets, np.array(counts, np.uint32), candidates,
            np.array(train_right), np.array(test_right))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "catboost/metal/.build/multiclass-ctr-tie.json")
    args = parser.parse_args()
    bins, targets, test_targets, counts, candidates, train_right, test_right = prepare_candidates()
    features = np.array([c[0] for c in candidates], np.uint32)
    borders = np.array([c[1] for c in candidates], np.uint32)
    kinds = np.array([c[2] for c in candidates], np.uint8)
    lookup = {(f, b): i for i, (f, b, _, _) in enumerate(candidates)}
    options = dict(classes=3, depth=6, learning_rate=0.5, l2_leaf_reg=3,
                   score_function="L2", leaf_estimation_backtracking="AnyImprovement")
    validation = np.zeros((len(test_targets), 3), np.float64)
    baseline, sequences = [], []

    def consume(step):
        sequence = [lookup[int(f), int(b)] for f, b in zip(step.split_features, step.split_bins)]
        ids = np.zeros(len(test_targets), np.uint32)
        for depth, candidate in enumerate(sequence):
            ids |= test_right[candidate].astype(np.uint32) << depth
        validation[:] += step.leaf_values[ids]
        loss = np.mean(np.logaddexp.reduce(validation, axis=1)
                       - validation[np.arange(len(test_targets)), test_targets])
        return sequence, [step.loss, float(loss)]

    with _multiclass.Session(bins, targets, features, borders, candidate_types=kinds,
                            iterations=20, **options) as session:
        session.configure_feature_penalties(counts, 0.5)
        for tree in range(20):
            sequence, losses = consume(session.step())
            sequences.append(sequence)
            baseline.append(losses)
            if tree == 5:
                active = session.optimization_predictions()
                published = session.predictions()
                used = session.feature_penalty_state["used_features"]
                prior_validation = validation.copy()

    original = next(i for sequence in sequences[:6] for i in sequence if candidates[i][3][0] == "ctr")
    descriptor = candidates[original][3]
    aliases = [i for i, candidate in enumerate(candidates)
               if candidate[3][:3] == descriptor[:3] and candidate[3][3] != descriptor[3]
               and np.array_equal(train_right[i], train_right[original])
               and np.array_equal(test_right[i], test_right[original])]
    if len(aliases) != 1:
        raise RuntimeError("The historical fixture's unique alternate CTR alias was not found")
    alias = aliases[0]
    original_feature, alternate_feature = features[original], features[alias]
    # Every earlier use must admit an identical predicate from the alternate
    # prior, so this alternate state describes unchanged prior tree outputs.
    for sequence in sequences[:6]:
        for index in sequence:
            if features[index] == original_feature and not any(
                features[j] == alternate_feature and np.array_equal(train_right[index], train_right[j])
                and np.array_equal(test_right[index], test_right[j]) for j in range(len(candidates))
            ):
                raise RuntimeError("An earlier CTR split has no equivalent alternate-prior predicate")
    alternate_used = used.copy()
    alternate_used[original_feature], alternate_used[alternate_feature] = 0, 1
    validation[:] = prior_validation
    alternate = baseline[:6].copy()
    with _multiclass.Session(bins, targets, features, borders, candidate_types=kinds,
                            iterations=14, iteration_offset=6, initial_predictions=published,
                            initial_optimization_predictions=active, **options) as session:
        session.configure_feature_penalties(counts, 0.5, used_features=alternate_used)
        for _ in range(14):
            _, losses = consume(session.step())
            alternate.append(losses)
    fixture = ROOT / "catboost/pytest/cuda_tests/canondata/test_gpu.test_grow_policies_MultiClass-L2-SymmetricTree-Plain_"
    expected = np.array([np.loadtxt(fixture / name, skiprows=1)[:, 1]
                         for name in ("learn_error.tsv", "test_error.tsv")]).T
    result = dict(description="Diagnostic alternate state from a tied CTR alias; production policy unchanged",
                  original_candidate=candidates[original][3], alternate_candidate=candidates[alias][3],
                  rows=len(targets), features=bins.shape[0], candidates=len(candidates),
                  cuda_fixture=expected.tolist(), metal=baseline, alternate_state=alternate,
                  metal_max_absolute_difference=np.max(abs(expected - baseline), axis=0).tolist(),
                  alternate_max_absolute_difference=np.max(abs(expected - alternate), axis=0).tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("original_candidate", "alternate_candidate",
                      "metal_max_absolute_difference", "alternate_max_absolute_difference")}))


if __name__ == "__main__":
    main()
