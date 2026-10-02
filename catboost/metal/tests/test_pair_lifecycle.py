"""Public supplied-pair ranking, persistence, and export on the real Metal GPU."""

import json
import platform

import numpy as np
import pytest

from catboost_metal import _native
from catboost_metal._training import run_training
from catboost_metal.ranker import CatBoostMetalRanker
from test_pairwise_kernels import reference
from test_pairwise_training import leaf_reference


@pytest.fixture(autouse=True)
def forbid_cpu_fit(monkeypatch):
    from catboost import CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor

    def forbidden(*args, **kwargs):
        pytest.fail("PairLogit lifecycle tests must not fit a CPU CatBoost model")

    for cls in (CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)
    monkeypatch.setattr(CatBoost, "_fit", forbidden)


@pytest.fixture
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("PairLogit training requires an Apple Silicon Metal GPU")
    assert _native.device_info()["backend"] == "Metal"


def problem():
    rng = np.random.default_rng(29401)
    offsets = np.array([0, 17, 41, 67], np.uint32)
    bins = rng.integers(0, 4, (2, offsets[-1]), dtype=np.uint8)
    signal = 2 * (bins[0] > 1) - .4 * bins[1]
    winners, losers = [], []
    for begin, end in zip(offsets[:-1], offsets[1:]):
        # The last row of each group is isolated, permitting a valid alternate
        # group boundary for the snapshot-fingerprint test.
        count = end - begin - 1
        left = rng.integers(begin, end - 1, 71)
        right = begin + (left - begin + rng.integers(1, count, 71)) % count
        first_wins = signal[left] >= signal[right]
        winners.extend(np.where(first_wins, left, right))
        losers.extend(np.where(first_wins, right, left))
    pairs = np.column_stack((winners, losers)).astype(np.uint32)
    weights = rng.uniform(.1, 3, len(pairs)).astype(np.float32)
    pairs = np.concatenate((pairs, pairs[:1], pairs[:1, ::-1]))
    weights = np.r_[weights, np.float32(2), np.float32(.7)]
    weights[::17] = 0
    targets = np.zeros(bins.shape[1], np.float32)
    features = np.repeat(np.arange(2, dtype=np.uint32), 3)
    borders = np.tile(np.arange(3, dtype=np.uint32), 2)
    groups = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    return bins, targets, features, borders, offsets, groups, pairs, weights


def native_options(**extra):
    return dict(iterations=5, depth=2, learning_rate=.23, l2_leaf_reg=2,
                bias=0., score_function="Cosine", objective="PairLogit",
                leaf_estimation_method="Newton", leaf_estimation_iterations=3,
                leaf_estimation_backtracking="No", random_seed=1327) | extra


def public_options(**extra):
    return dict(loss_function="PairLogit", iterations=5, depth=2, learning_rate=.23,
                l2_leaf_reg=2, border_count=3, score_function="Cosine",
                leaf_estimation_method="Newton", leaf_estimation_iterations=3,
                leaf_estimation_backtracking="No", random_seed=1327) | extra


def pair_options(pairs, weights, offsets, *, evaluation=False):
    result = dict(pair_winners=pairs[:, 0].copy(), pair_losers=pairs[:, 1].copy(),
                  pair_weights=weights.copy(), group_offsets=offsets.copy())
    return {"eval_" + name: value for name, value in result.items()} if evaluation else result


def assert_same_model(actual, expected):
    for name in ("depths", "split_features", "split_bins", "split_types", "leaf_values",
                 "leaf_weights", "predictions", "rmse"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)


def forbid_gpu(monkeypatch):
    from catboost_metal import _evaluation

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid PairLogit lifecycle input reached the GPU")

    monkeypatch.setattr(_native, "Session", forbidden)
    monkeypatch.setattr(_native, "train", forbidden)
    monkeypatch.setattr(_evaluation, "EvaluationCursor", forbidden)


@pytest.mark.parametrize("method,steps", [(None, 10), ("Newton", 10), ("Gradient", 40)])
def test_public_pair_leaf_defaults_without_gpu(monkeypatch, method, steps):
    forbid_gpu(monkeypatch)
    model = CatBoostMetalRanker(loss_function="PairLogit", leaf_estimation_method=method)
    assert model.leaf_estimation_method == (method or "Newton")
    assert model.leaf_estimation_iterations == steps
    explicit = CatBoostMetalRanker(loss_function="PairLogit", leaf_estimation_method=method,
                                  leaf_estimation_iterations=3)
    assert explicit.leaf_estimation_iterations == 3


@pytest.mark.parametrize("method,steps", [("Newton", 10), ("Gradient", 40)])
def test_default_leaf_iterations_reach_gpu_and_export_metadata(metal_device, tmp_path, method, steps):
    values = np.array([[0], [1], [0], [1], [0], [1]], np.float32)
    pairs = np.array([[1, 0], [3, 2], [1, 2], [0, 1]], np.uint32)
    weights = np.array([1, 3, .25, .7], np.float32)
    model = CatBoostMetalRanker(loss_function="PairLogit", iterations=1, depth=1,
        learning_rate=.3, l2_leaf_reg=2, border_count=1, score_function="L2",
        leaf_estimation_method=method).fit(values, group_id=np.zeros(6, np.int32),
                                         pairs=pairs, pairs_weight=weights)
    result = model._result
    assert result.depths[0] == 1
    ids = values[:, 0].astype(np.uint32)
    args = native_options(learning_rate=.3, leaf_estimation_method=method,
                          leaf_estimation_iterations=steps, pair_winners=pairs[:, 0],
                          pair_losers=pairs[:, 1], pair_weights=weights)
    leaves, masses, _ = leaf_reference(args, np.zeros(6, np.float32), ids, 2)
    np.testing.assert_allclose(result.leaf_values[0, :2], leaves, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(result.leaf_weights[0, :2], masses, rtol=1e-6)
    path = tmp_path / "defaults.json"
    model.save_model(path, format="json")
    params = json.loads(path.read_text())["model_info"]["params"]
    if isinstance(params, str):
        params = json.loads(params)
    assert params["loss_function"]["type"] == "PairLogit"
    assert params["tree_learner_options"]["leaf_estimation_method"] == method
    assert params["tree_learner_options"]["leaf_estimation_iterations"] == steps


def test_public_pair_metadata_preserves_literal_edges_and_ignores_labels(metal_device):
    bins, targets, _, _, offsets, groups, pairs, weights = problem()
    x = bins.T.astype(np.float32)
    first = CatBoostMetalRanker(**public_options(iterations=3)).fit(
        x, group_id=groups, pairs=pairs, pairs_weight=weights)
    changed = CatBoostMetalRanker(**public_options(iterations=3)).fit(
        x, np.linspace(500, -500, len(targets)), group_id=groups,
        sample_weight=np.linspace(.25, 7, len(targets)),
        group_weight=np.repeat([2., .3, 11.], np.diff(offsets)),
        pairs=pairs, pairs_weight=weights)
    assert_same_model(changed._result, first._result)
    terms = reference(first.training_predictions_, pairs[:, 0], pairs[:, 1], weights)
    assert first.loss_history_[-1] == pytest.approx(terms["objective"][0] / terms["objective"][1], rel=3e-6)
    assert terms["incident_weights"][offsets[1:] - 1].sum() == 0
    np.testing.assert_allclose(first._result.leaf_weights.sum(axis=1),
                               2 * weights.sum(dtype=np.float64), rtol=1e-6)


@pytest.mark.parametrize("options", [dict(leaf_estimation_method="Exact"),
    dict(loss_function="YetiRankPairwise:mode=NDCG"), dict(loss_function="PairLogit:max_pairs=10"),
    dict(loss_function="PairLogitPairwise", boosting_type="Ordered"),
    dict(loss_function="PairLogitPairwise", grow_policy="Depthwise")])
def test_unsupported_public_pair_training_modes_fail_before_gpu(monkeypatch, options):
    forbid_gpu(monkeypatch)
    with pytest.raises(ValueError):
        CatBoostMetalRanker(**(dict(loss_function="PairLogit") | options))


@pytest.mark.parametrize("change", ["missing_pairs", "missing_groups", "noncontiguous_groups",
    "cross_group", "fractional_pairs", "negative_pair_weight", "object_weight_shape",
    "nonconstant_group_weight", "missing_eval_pairs", "orphan_eval_pairs"])
def test_public_pair_metadata_rejected_before_gpu(monkeypatch, change):
    forbid_gpu(monkeypatch)
    bins, targets, _, _, _, groups, pairs, weights = problem()
    fit = dict(group_id=groups, pairs=pairs, pairs_weight=weights)
    if change == "missing_pairs":
        fit.pop("pairs")
        fit.pop("pairs_weight")
    elif change == "missing_groups":
        fit.pop("group_id")
    elif change == "noncontiguous_groups":
        fit["group_id"] = np.arange(len(groups)) % 2
    elif change == "cross_group":
        fit["pairs"] = pairs.copy()
        fit["pairs"][0] = [0, 25]
    elif change == "fractional_pairs":
        fit["pairs"] = pairs.astype(float) + .1
    elif change == "negative_pair_weight":
        fit["pairs_weight"] = -weights
    elif change == "object_weight_shape":
        fit["sample_weight"] = [1]
    elif change == "nonconstant_group_weight":
        fit["group_weight"] = np.arange(len(groups)) + 1
    elif change == "missing_eval_pairs":
        fit["eval_set"] = (bins.T, targets, groups)
    else:
        fit["eval_pairs"] = pairs
    with pytest.raises(ValueError):
        CatBoostMetalRanker(**public_options()).fit(bins.T, targets, **fit)


@pytest.mark.parametrize("change", [dict(sample_weight=np.ones(67)),
    dict(leaf_estimation_method="Exact"), dict(pair_winners=None),
    dict(eval_metric="RMSE"), dict(eval_metric="QueryRMSE")])
def test_low_level_pair_lifecycle_restrictions_precede_gpu(monkeypatch, change):
    # The controller validates pair metadata and metrics; leaf-method validation
    # is shared with _native._prepare, which runs before loading the GPU library.
    monkeypatch.setattr(_native, "build_library", lambda: pytest.fail("Invalid input reached Metal library loading"))
    bins, targets, features, borders, offsets, _, pairs, weights = problem()
    options = native_options() | pair_options(pairs, weights, offsets) | change
    with pytest.raises(ValueError):
        run_training(bins, targets, features, borders, **options)


@pytest.mark.parametrize("permutations", [1, 4])
@pytest.mark.parametrize("bootstrap", ["No", "Bayesian", "Bernoulli", "Poisson", "MVS"])
def test_pair_snapshot_resume_restores_p1_and_p4_exact_state(metal_device, tmp_path, permutations, bootstrap):
    bins, targets, features, borders, offsets, _, pairs, weights = problem()
    rng = np.random.default_rng(814)
    matrices = (bins,) + tuple(bins[:, rng.permutation(len(targets))].copy()
                              for _ in range(permutations - 1))
    sampling = {"subsample": .7} if bootstrap in ("Bernoulli", "Poisson", "MVS") else {}
    options = native_options(bootstrap_type=bootstrap, random_strength=.35, **sampling)
    common = pair_options(pairs, weights, offsets) | pair_options(pairs, weights, offsets, evaluation=True)
    common.update(permutation_bins=matrices, eval_bins=bins, eval_targets=targets,
                  use_best_model=False, snapshot_interval=0)
    path, full_path = tmp_path / "pair.snapshot", tmp_path / "full.snapshot"
    stopped = run_training(bins, targets, features, borders, **options, **common,
        save_snapshot=True, snapshot_file=path, callback=lambda step: step.iteration < 2)
    assert stopped.stats["stop_reason"] == "callback"
    assert len(stopped.depths) == 2
    resumed = run_training(bins, targets, features, borders, **options, **common,
                           save_snapshot=True, snapshot_file=path)
    full = run_training(bins, targets, features, borders, **options, **common,
                        save_snapshot=True, snapshot_file=full_path)
    assert_same_model(resumed, full)
    assert resumed.evals_result == full.evals_result
    assert resumed.stats["resumed_iterations"] == 2
    assert resumed.stats.get("bootstrap_state") == full.stats.get("bootstrap_state")
    with np.load(path, allow_pickle=False) as actual, np.load(full_path, allow_pickle=False) as expected:
        assert all(actual[name].dtype.kind != "O" for name in actual.files)
        assert set(actual.files) == set(expected.files)
        for name in actual.files:
            if name != "metadata":
                np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        if permutations > 1:
            assert actual["permutation_predictions"].shape == (4, len(targets))
            np.testing.assert_array_equal(actual["permutation_predictions"][-1], resumed.predictions)
            if bootstrap == "MVS":
                np.testing.assert_array_equal(actual["permutation_mvs_valid"], np.ones(4, np.uint8))
                assert np.isfinite(actual["permutation_mvs_lambdas"]).all()


@pytest.mark.parametrize("field", ["pair_winners", "pair_weights", "pair_order", "group_offsets",
                                    "eval_pair_losers", "eval_pair_weights", "eval_group_offsets"])
def test_pair_and_group_fingerprints_reject_changed_input_before_gpu(metal_device, tmp_path, monkeypatch, field):
    bins, targets, features, borders, offsets, _, pairs, weights = problem()
    options = native_options(iterations=2)
    common = pair_options(pairs, weights, offsets) | pair_options(pairs, weights, offsets, evaluation=True)
    common.update(eval_bins=bins, eval_targets=targets, use_best_model=False,
                  save_snapshot=True, snapshot_file=tmp_path / "bound.snapshot")
    run_training(bins, targets, features, borders, **options, **common)
    changed = common.copy()
    if field == "pair_order":
        for name in ("pair_winners", "pair_losers", "pair_weights"):
            changed[name] = common[name][::-1].copy()
    elif field.endswith("group_offsets"):
        changed[field] = common[field].copy()
        changed[field][1] -= 1  # Moves only an isolated row; pairs remain valid.
    elif field.endswith("weights"):
        changed[field] = common[field].copy()
        changed[field][1] *= 2
    else:
        changed[field] = common[field].copy()
        other = common["pair_losers" if field == "pair_winners" else "eval_pair_winners"][0]
        changed[field][0] = (other + 1) % 16
        if changed[field][0] == common[field][0]:
            changed[field][0] = (other + 2) % 16
    forbid_gpu(monkeypatch)
    with pytest.raises(ValueError, match="does not match"):
        run_training(bins, targets, features, borders, **(options | {"iterations": 5}), **changed)


@pytest.mark.parametrize("selection", ["PairLogit:use_weights=false", "PairAccuracy", "PairAccuracy:use_weights=false"])
def test_public_pair_selection_metrics_use_supplied_validation_edges(metal_device, selection):
    bins, targets, _, _, _, groups, pairs, weights = problem()
    validation_pairs = pairs[:, ::-1].copy()
    model = CatBoostMetalRanker(**public_options(iterations=3, eval_metric=selection)).fit(
        bins.T, targets, group_id=groups, pairs=pairs, pairs_weight=weights,
        eval_set=(bins.T, targets + 500, groups), eval_pairs=validation_pairs,
        eval_pairs_weight=weights, use_best_model=False)
    raw = model.predict(bins.T, task_type="METAL")
    metric_weights = np.ones(len(pairs)) if "use_weights=false" in selection else weights.astype(np.float64)
    difference = raw[validation_pairs[:, 0]] - raw[validation_pairs[:, 1]]
    if selection.startswith("PairAccuracy"):
        expected = np.average(difference > 0, weights=metric_weights)
    else:
        expected = np.average(np.logaddexp(0, -difference), weights=metric_weights)
    assert model.get_evals_result()["validation"][selection][-1] == pytest.approx(expected, abs=2e-6)
    assert model.training_stats_["metric_maximized"] == selection.startswith("PairAccuracy")


def test_public_pair_snapshot_exports_cbm_json_and_matches_gpu_inference(metal_device, tmp_path):
    from catboost import CatBoostRanker

    bins, targets, _, _, _, groups, pairs, weights = problem()
    x = bins.T.astype(np.float32)
    fit = dict(group_id=groups, pairs=pairs, pairs_weight=weights,
               eval_set=(x, targets, groups), eval_pairs=pairs, eval_pairs_weight=weights,
               use_best_model=False)
    path = tmp_path / "ranker.snapshot"
    CatBoostMetalRanker(**public_options(iterations=2)).fit(
        x, targets, save_snapshot=True, snapshot_file=path, **fit)
    resumed = CatBoostMetalRanker(**public_options()).fit(
        x, targets, save_snapshot=True, snapshot_file=path, **fit)
    full = CatBoostMetalRanker(**public_options()).fit(x, targets, **fit)
    assert_same_model(resumed._result, full._result)
    assert resumed.get_evals_result() == full.get_evals_result()
    assert resumed.training_stats_["resumed_iterations"] == 2
    raw = resumed.predict(x, task_type="METAL")
    np.testing.assert_allclose(raw, resumed.training_predictions_, rtol=1e-6, atol=2e-6)
    for format in ("cbm", "json"):
        exported = tmp_path / f"ranker.{format}"
        resumed.save_model(exported, format=format)
        restored = CatBoostRanker().load_model(str(exported), format=format)
        np.testing.assert_allclose(restored.predict(x), raw, rtol=1e-6, atol=2e-6)
        assert restored.get_all_params()["loss_function"] == "PairLogit"
        np.testing.assert_allclose(restored.predict(x, ntree_start=1, ntree_end=4),
            resumed.predict(x, task_type="METAL", ntree_start=1, ntree_end=4), rtol=1e-6, atol=2e-6)


def test_public_reversed_validation_edges_select_best_tree_and_keep_snapshot_state(metal_device, tmp_path):
    x = np.array([[0], [1], [0], [1], [0], [1]], np.float32)
    groups = np.repeat([0, 1, 2], 2)
    pairs = np.array([[1, 0], [3, 2], [5, 4]], np.uint32)
    options = public_options(iterations=10, depth=1, learning_rate=.4, border_count=1)
    path = tmp_path / "early.snapshot"
    model = CatBoostMetalRanker(**options).fit(x, group_id=groups, pairs=pairs,
        eval_set=(x, np.zeros(len(x)), groups), eval_pairs=pairs[:, ::-1].copy(),
        early_stopping_rounds=2, use_best_model=True, save_snapshot=True, snapshot_file=path)
    assert model.get_best_iteration() == 0
    assert model.tree_count_ == 1
    assert model.training_stats_["iterations_trained"] == 3
    assert model.training_stats_["stop_reason"] == "early_stopping"
    first = CatBoostMetalRanker(**(options | {"iterations": 1})).fit(x, group_id=groups, pairs=pairs)
    np.testing.assert_array_equal(model.training_predictions_, first.training_predictions_)
    with np.load(path, allow_pickle=False) as snapshot:
        assert len(snapshot["depths"]) == 3
        assert len(json.loads(str(snapshot["metadata"].item()))["history"]["validation"]["PairLogit"]) == 3


def test_pair_pool_without_hidden_pairs_accepts_explicit_metadata(metal_device):
    from catboost import Pool

    bins, targets, _, _, offsets, groups, pairs, weights = problem()
    pool = Pool(bins.T, targets, group_id=groups,
                group_weight=np.repeat([2., .5, 4.], np.diff(offsets)))
    pooled = CatBoostMetalRanker(**public_options(iterations=2)).fit(pool, pairs=pairs, pairs_weight=weights)
    arrays = CatBoostMetalRanker(**public_options(iterations=2)).fit(
        bins.T, targets, group_id=groups, pairs=pairs, pairs_weight=weights)
    assert_same_model(pooled._result, arrays._result)


def test_pair_pool_with_inaccessible_stored_edges_is_rejected_before_gpu(monkeypatch):
    from catboost import Pool

    forbid_gpu(monkeypatch)
    bins, targets, _, _, _, groups, pairs, weights = problem()
    pool = Pool(bins.T, targets, group_id=groups, pairs=pairs, pairs_weight=weights)
    with pytest.raises(ValueError, match="(?i)(stored|access|pair)"):
        CatBoostMetalRanker(**public_options()).fit(pool, pairs=pairs, pairs_weight=weights)
