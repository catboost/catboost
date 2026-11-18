"""CUDA symmetric multiclass gain/noise semantics through the actual session."""

import numpy as np
import pytest

from catboost_metal import _multiclass
from test_multiclass import no_cpu_training
from test_bootstrap import probe as bootstrap_probe
from test_multiclass_bootstrap_kernels import reference_statistics


def arguments(objective, sampler="No", **kwargs):
    rng = np.random.default_rng(5921)
    rows, features, classes = 1031, 12, 4
    bins = rng.integers(0, 4, (features, rows), dtype=np.uint8)
    labels = bins[3].astype(np.uint32)
    labels[::5] = rng.integers(0, classes, len(labels[::5]))
    weights = rng.integers(1, 5, rows).astype(np.float32)
    weights[::19] = 0
    result = dict(bins=bins, targets=labels, classes=classes, objective=objective,
                  candidate_features=np.repeat(np.arange(features, dtype=np.uint32), 3),
                  candidate_bins=np.tile(np.arange(3, dtype=np.uint32), features),
                  sample_weight=weights, iterations=1, depth=1, learning_rate=.13,
                  leaf_estimation_iterations=1, leaf_estimation_method="Gradient",
                  bootstrap_type=sampler, random_seed=1972, subsample=.61,
                  bagging_temperature=.7, l2_leaf_reg=3., score_function="Cosine")
    result.update(kwargs)
    return result


def initial_sampled_gradients(args, bootstrap_probe):
    labels, weights, classes = args["targets"], args["sample_weight"], args["classes"]
    multi_logit = args["objective"] == "MultiClass"
    dimensions = classes - int(multi_logit)
    if args.get("initial_predictions") is None:
        probability = np.float32(1 / classes if multi_logit else .5)
    else:
        logits = np.asarray(args["initial_predictions"], np.float32).T.astype(np.float64)
        if multi_logit:
            exponentials = np.exp(logits - logits.max(axis=0))
            probability = (exponentials / exponentials.sum(axis=0))[:dimensions].astype(np.float32)
        else:
            probability = (1 / (1 + np.exp(-logits))).astype(np.float32)
    gradients = ((np.arange(dimensions)[:, None] == labels) - probability) * weights
    kind = ("No", "Bayesian", "Bernoulli", "Poisson").index(args["bootstrap_type"])
    draws = bootstrap_probe(kind, rows=len(labels), seed=args["random_seed"],
        iteration=args.get("iteration_offset", 0), subsample=args["subsample"],
        temperature=args["bagging_temperature"])["values"]
    return (gradients * draws).astype(np.float32), (weights * draws).astype(np.float32)


def initial_cosine_scores(args, gradients, weights):
    """Independent float32 CUDA leaf score equations after double summation."""
    scores = []
    for feature, border in zip(args["candidate_features"], args["candidate_bins"]):
        selected = args["bins"][feature] <= border
        side_g = np.stack([gradients[:, selected].sum(axis=1, dtype=np.float64),
                           gradients[:, ~selected].sum(axis=1, dtype=np.float64)]).astype(np.float32)
        side_w = np.array([weights[selected].sum(dtype=np.float64),
                           weights[~selected].sum(dtype=np.float64)], np.float32)
        if args["objective"] == "MultiClass":
            side_g = np.concatenate((side_g, -side_g.sum(axis=1, keepdims=True)), axis=1)
        numerator, denominator = np.float32(0), np.float32(1e-10)
        for k in range(side_g.shape[1]):
            for side in range(2):
                gradient, weight = side_g[side, k], side_w[side]
                value = np.float32(gradient / np.float32(weight + args["l2_leaf_reg"])) if weight else np.float32(0)
                numerator = np.float32(numerator + np.float32(gradient * value))
                denominator = np.float32(denominator + np.float32(np.float32(weight * value) * value))
        scores.append(np.float32(-numerator / np.sqrt(denominator)))
    return np.array(scores)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("sampler", ["Bayesian", "Bernoulli", "Poisson"])
def test_default_noise_preserves_separated_symmetric_gain_winner(objective, sampler):
    args = arguments(objective, sampler)
    plain = _multiclass.train(**args, random_strength=0)
    noisy = _multiclass.train(**args, random_strength=1)
    # Source candidate-minus-baseline noise cancels; this fixture has a large
    # score margin, so the remaining float32 rounding cannot change selection.
    np.testing.assert_array_equal(noisy.split_features, plain.split_features)
    np.testing.assert_array_equal(noisy.split_bins, plain.split_bins)
    np.testing.assert_array_equal(noisy.leaf_values, plain.leaf_values)
    np.testing.assert_array_equal(noisy.predictions, plain.predictions)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("seed,expected_depth", [(1972, 1), (0, 0)])
def test_huge_noise_retains_cuda_gain_selection_and_raw_score_stop(objective, seed, expected_depth, bootstrap_probe):
    args = arguments(objective, "Bernoulli", random_seed=seed)
    gradients, weights = initial_sampled_gradients(args, bootstrap_probe)
    statistic = reference_statistics(gradients, weights, objective == "MultiClass")
    rows = len(weights)
    scale = np.float32(1e20 * (rows / (rows + 1)) * np.sqrt(statistic[0] / statistic[1]))
    noise = bootstrap_probe(9, rows=args["bins"].shape[0], seed=args["random_seed"],
                            stream=1, noise_scale=scale)["values"]
    raw = initial_cosine_scores(args, gradients, weights)
    shared_noise = noise[args["candidate_features"]]
    noisy_scores = (raw + shared_noise).astype(np.float32)
    gains = noisy_scores - shared_noise
    np.testing.assert_array_equal(gains, 0)
    # Computing gain as raw score directly would lose CUDA's rounding here;
    # comparing noisy score directly would pick the most-negative noise.
    assert raw.argmin() != 0
    assert noisy_scores.argmin() != 0
    # CUDA first selects by Gain (candidate zero wins the exact tie), then
    # stops growth if that selected candidate's noisy Score is nonnegative.
    # A negative-score alternative must not replace a stopped gain winner.
    assert bool(noisy_scores[0] < 0) == bool(expected_depth)
    assert noisy_scores.min() < 0
    result = _multiclass.train(**args, random_strength=1e20)
    assert result.depths[0] == expected_depth
    if expected_depth:
        assert result.split_features[0, 0] == 0
        assert result.split_bins[0, 0] == 0
    else:
        np.testing.assert_array_equal(
            result.predictions,
            np.broadcast_to(result.leaf_values[0, 0], result.predictions.shape),
        )


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("sampler", ["Bayesian", "Bernoulli", "Poisson"])
def test_sampled_vector_statistic_controls_rounded_winner(objective, sampler, bootstrap_probe):
    # Each selected noisy score stays negative. Both variance formulas thus
    # permit growth, while producing different rounded-gain winners.
    fixtures = {("MultiClass", "Bayesian"): (1974, 22085742),
                ("MultiClass", "Bernoulli"): (1986, 290706880),
                ("MultiClass", "Poisson"): (1973, 40168320),
                ("MultiClassOneVsAll", "Bayesian"): (1977, 42499684),
                ("MultiClassOneVsAll", "Bernoulli"): (1974, 19125236),
                ("MultiClassOneVsAll", "Poisson"): (1979, 9345034)}
    seed, strength = fixtures[objective, sampler]
    args = arguments(objective, sampler, iteration_offset=9, random_seed=seed)
    args["initial_predictions"] = np.random.default_rng(331).normal(0, 2, (1031, 4)).astype(np.float32)
    gradients, weights = initial_sampled_gradients(args, bootstrap_probe)
    statistic = reference_statistics(gradients, weights, objective == "MultiClass")
    rows = len(weights)
    # These fixtures sit well inside different rounded-gain regions for the
    # post-bootstrap CUDA statistic and the incorrect pre-bootstrap statistic.
    strength = np.float32(strength)
    multiplier = 1 / (1 + np.exp(-(np.log(rows) - 9 * np.float32(args["learning_rate"]))))
    scale = np.float32(strength * multiplier * np.sqrt(statistic[0] / statistic[1]))
    noise = bootstrap_probe(9, rows=args["bins"].shape[0], seed=args["random_seed"],
                            iteration=9, stream=1, noise_scale=scale)["values"]
    raw = initial_cosine_scores(args, gradients, weights)
    shared_noise = noise[args["candidate_features"]]
    noisy_scores = (raw + shared_noise).astype(np.float32)
    gains = noisy_scores - shared_noise
    candidate = int(gains.argmin())
    assert noisy_scores[candidate] < 0
    original_g, original_w = initial_sampled_gradients(dict(args, bootstrap_type="No"), bootstrap_probe)
    wrong_stat = reference_statistics(original_g, original_w, objective == "MultiClass")
    wrong_scale = np.float32(strength * multiplier * np.sqrt(wrong_stat[0] / wrong_stat[1]))
    wrong_noise = bootstrap_probe(9, rows=args["bins"].shape[0], seed=args["random_seed"],
                                  iteration=9, stream=1, noise_scale=wrong_scale)["values"][args["candidate_features"]]
    wrong_scores = (raw + wrong_noise).astype(np.float32)
    wrong_candidate = int((wrong_scores - wrong_noise).argmin())
    assert wrong_scores[wrong_candidate] < 0
    assert wrong_candidate != candidate
    result = _multiclass.train(**args, random_strength=float(strength))
    assert result.depths[0] == 1
    assert result.split_features[0, 0] == args["candidate_features"][candidate]
    assert result.split_bins[0, 0] == args["candidate_bins"][candidate]


def test_missing_class_energy_changes_session_gain_rounding(bootstrap_probe):
    args = arguments("MultiClass", "Bayesian", iteration_offset=9, random_seed=1973)
    args["initial_predictions"] = np.random.default_rng(331).normal(0, 2, (1031, 4)).astype(np.float32)
    gradients, weights = initial_sampled_gradients(args, bootstrap_probe)
    raw = initial_cosine_scores(args, gradients, weights)
    strength = np.float32(348707968)
    multiplier = 1 / (1 + np.exp(-(np.log(len(weights)) - 9 * np.float32(args["learning_rate"])) ))
    winners = []
    for reconstruct in [True, False]:
        statistic = reference_statistics(gradients, weights, reconstruct)
        scale = np.float32(strength * multiplier * np.sqrt(statistic[0] / statistic[1]))
        noise = bootstrap_probe(9, rows=args["bins"].shape[0], seed=args["random_seed"],
                                iteration=9, stream=1, noise_scale=scale)["values"][args["candidate_features"]]
        noisy_scores = (raw + noise).astype(np.float32)
        winners.append(int((noisy_scores - noise).argmin()))
        assert noisy_scores[winners[-1]] < 0
    assert winners[0] != winners[1]
    result = _multiclass.train(**args, random_strength=float(strength))
    assert result.depths[0] == 1
    assert result.split_features[0, 0] == args["candidate_features"][winners[0]]
    assert result.split_bins[0, 0] == args["candidate_bins"][winners[0]]


def test_mvs_rejection_matches_cuda_multiclass_gate():
    with pytest.raises(ValueError, match="CUDA rejects multiclass MVS"):
        _multiclass.train(**arguments("MultiClass", "MVS"))
