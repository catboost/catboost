"""Query-group semantics and actual Metal ranking lifecycle checks."""

import json
import platform

import numpy as np
import pytest

from catboost_metal import _native
from catboost_metal._query_data import prepare_groups, query_metric, unpack_query_pool
from catboost_metal._training import metric, run_training
from catboost_metal.ranker import CatBoostMetalRanker


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier, CatBoostRanker
    def forbidden(*args, **kwargs):
        pytest.fail("Query tests must not train CPU CatBoost")
    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier, CatBoostRanker):
        monkeypatch.setattr(cls, "fit", forbidden)
    monkeypatch.setattr(CatBoost, "_fit", forbidden)


@pytest.fixture
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Query training needs Apple Silicon")


def problem(objective="QueryRMSE"):
    rng = np.random.default_rng(88)
    sizes = np.asarray([1, 3, 7, 17, 33])
    groups = np.repeat(np.arange(len(sizes)), sizes)
    offsets = np.r_[0, sizes.cumsum()].astype(np.uint32)
    bins = rng.integers(0, 4, (2, len(groups)), dtype=np.uint8)
    targets = ((bins[0] > 1) * 2 + rng.uniform(0, .2, len(groups))).astype(np.float32)
    if objective == "QueryRMSE":
        targets += groups * 10
    weights = rng.uniform(.2, 2, len(groups)).astype(np.float32)
    weights[::13] = 0
    return bins, targets, np.repeat(np.arange(2, dtype=np.uint32), 3), np.tile(np.arange(3, dtype=np.uint32), 2), offsets, weights


def options(objective="QueryRMSE", **extra):
    result = dict(iterations=7, depth=2, learning_rate=.2, l2_leaf_reg=2, bias=0,
                  objective=objective, score_function="Cosine", leaf_estimation_method="Gradient",
                  leaf_estimation_iterations=2, random_seed=13)
    result.update(extra)
    return result


def test_object_and_group_weights_multiply_once():
    offsets, weights = prepare_groups(["a", "a", "b", "b"], 4, [1, 3, 2, 0], [2, 2, 4, 4])
    np.testing.assert_array_equal(offsets, [0, 2, 4])
    np.testing.assert_array_equal(weights, [2, 6, 8, 0])


@pytest.mark.parametrize("ids,weights,error", [
    (["a", "b", "a"], [1, 1, 1], "contiguous"),
    (["a", "a", "b"], [1, 2, 1], "constant"),
    (["a", None, "b"], [1, 1, 1], "identifiers"),
    ([1., 1., 2.], [1, 1, 1], "identifiers"),
    ([1, 1, 2], [0, 0, 0], "positive"),
])
def test_invalid_group_inputs_fail_before_training(ids, weights, error):
    with pytest.raises(ValueError, match=error):
        prepare_groups(ids, 3, group_weight=weights)


def test_query_metrics_match_weighted_cuda_equations_and_ignore_softmax_lambda():
    offsets, weights = [0, 2, 3, 5], [1, 2, 3, 0, 0]
    assert query_metric([0] * 5, [0, 1, 2, 0, 1], weights, offsets, "QueryRMSE") == pytest.approx(1 / 3)
    expected = -2 * np.log(2 / 3) / 8
    assert query_metric([0] * 5, [0, 1, 2, 0, 1], weights, offsets, "QuerySoftMax") == pytest.approx(expected)
    assert query_metric([0] * 5, [0, 1, 2, 0, 1], weights, offsets, "QuerySoftMax", -2, -.8) == pytest.approx(expected)
    assert query_metric([1000, -1000], [1, 0], [1, 1], [0, 2], "QuerySoftMax", -1) == 2000
    assert metric([0, 0], [0, 1], [1, 2], "QuerySoftMax:beta=-1;lambda=-.2", group_offsets=[0, 2]) == pytest.approx(-np.log(2 / 3))


def test_pool_group_weights_are_extracted_or_explicitly_rejected_without_getter():
    from catboost import Pool
    pool = Pool([[0.], [1.], [2.]], [0., 1., 2.], group_id=["a", "a", "b"], group_weight=[2., 2., 3.])
    if not hasattr(pool, "get_group_weight"):
        with pytest.raises(ValueError, match="cannot expose group weights"):
            unpack_query_pool(pool)
        return
    _, _, offsets, weights, _, _ = unpack_query_pool(pool)
    np.testing.assert_array_equal(offsets, [0, 2, 3])
    np.testing.assert_array_equal(weights, [2, 2, 3])
    assert not pool.is_quantized()


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax"])
@pytest.mark.parametrize("bootstrap", ["No", "Bayesian", "Bernoulli", "Poisson", "MVS"])
def test_grouped_gpu_snapshot_matches_native_and_uninterrupted(metal_device, tmp_path, objective, bootstrap):
    bins, targets, features, borders, offsets, weights = problem(objective)
    params = options(objective, bootstrap_type=bootstrap,
                     **({"subsample": .6} if bootstrap in ("Bernoulli", "Poisson", "MVS") else {}))
    expected = _native.train(bins, targets, features, borders, group_offsets=offsets, sample_weight=weights, **params)
    common = dict(group_offsets=offsets, sample_weight=weights, eval_bins=bins, eval_targets=targets,
                  eval_weight=weights, eval_group_offsets=offsets, use_best_model=False)
    path = tmp_path / "query.snapshot"
    run_training(bins, targets, features, borders, save_snapshot=True, snapshot_file=path,
                 **{**params, "iterations": 2}, **common)
    resumed = run_training(bins, targets, features, borders, save_snapshot=True, snapshot_file=path, **params, **common)
    full = run_training(bins, targets, features, borders, **params, **common)
    for field in ("depths", "split_features", "split_bins", "leaf_values", "leaf_weights", "predictions", "rmse"):
        np.testing.assert_array_equal(getattr(resumed, field), getattr(expected, field))
        np.testing.assert_array_equal(getattr(resumed, field), getattr(full, field))
    assert resumed.evals_result == full.evals_result
    assert resumed.evals_result["validation"][objective][-1] == pytest.approx(
        query_metric(resumed.predictions, targets, weights, offsets, objective), abs=2e-6)


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax"])
def test_query_boundaries_bind_snapshots_before_gpu(metal_device, tmp_path, monkeypatch, objective):
    bins, targets, features, borders, offsets, weights = problem(objective)
    path = tmp_path / "query.snapshot"
    run_training(bins, targets, features, borders, **options(objective, iterations=2), group_offsets=offsets,
                 sample_weight=weights, save_snapshot=True, snapshot_file=path)
    altered = offsets.copy()
    altered[2] += 1
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Changed queries reached GPU"))
    with pytest.raises(ValueError, match="does not match"):
        run_training(bins, targets, features, borders, **options(objective), group_offsets=altered,
                     sample_weight=weights, save_snapshot=True, snapshot_file=path)


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax"])
def test_public_ranker_group_weights_evaluation_snapshot_and_export(metal_device, tmp_path, objective):
    from catboost import CatBoostRanker, Pool
    bins, targets, _, _, offsets, objects = problem(objective)
    features = bins.T.astype(np.float32)
    groups = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    group_weights = np.repeat(np.asarray([1., 2., .5, 3., 1.]), np.diff(offsets))
    params = dict(loss_function=objective, depth=2, learning_rate=.2, leaf_estimation_iterations=2,
                  leaf_estimation_method="Gradient", random_seed=13)
    fit = dict(group_id=groups, sample_weight=objects, group_weight=group_weights,
               eval_set=(features, targets, groups, objects, group_weights), use_best_model=False)
    path = tmp_path / "ranker.snapshot"
    CatBoostMetalRanker(iterations=2, **params).fit(features, targets, save_snapshot=True, snapshot_file=path, **fit)
    resumed = CatBoostMetalRanker(iterations=7, **params).fit(features, targets, save_snapshot=True, snapshot_file=path, **fit)
    full = CatBoostMetalRanker(iterations=7, **params).fit(features, targets, **fit)
    np.testing.assert_array_equal(resumed.training_predictions_, full.training_predictions_)
    assert resumed.get_evals_result() == full.get_evals_result()
    np.testing.assert_allclose(resumed.predict(features, task_type="METAL"), resumed.predict(features), atol=3e-6)
    np.testing.assert_allclose(resumed.predict(features, task_type="METAL"), resumed.training_predictions_, atol=3e-6)
    exported = tmp_path / "ranker.json"
    resumed.save_model(exported, format="json")
    metadata = json.loads(exported.read_text())["model_info"]["params"]
    assert metadata["loss_function"]["type"] == objective
    restored = CatBoostRanker().load_model(str(exported), format="json")
    np.testing.assert_allclose(restored.predict(features), resumed.predict(features), atol=1e-7)
    if hasattr(Pool, "get_group_weight"):
        pool = Pool(features, targets, group_id=groups, group_weight=group_weights)
        pooled = CatBoostMetalRanker(iterations=3, **params).fit(pool, eval_set=pool, use_best_model=False)
        arrays = CatBoostMetalRanker(iterations=3, **params).fit(
            features, targets, group_id=groups, group_weight=group_weights,
            eval_set=(features, targets, groups, None, group_weights), use_best_model=False)
        np.testing.assert_array_equal(pooled.training_predictions_, arrays.training_predictions_)
        assert pooled.get_evals_result() == arrays.get_evals_result()


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax"])
def test_public_query_early_stopping_keeps_best_exported_tree(metal_device, objective):
    bins, targets, _, _, offsets, weights = problem(objective)
    groups = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    wrong = -targets if objective == "QueryRMSE" else 3 - targets
    params = dict(loss_function=objective, depth=2, learning_rate=.2,
                  leaf_estimation_method="Gradient", leaf_estimation_iterations=2)
    fitted = CatBoostMetalRanker(iterations=12, **params).fit(
        bins.T, targets, group_id=groups, sample_weight=weights,
        eval_set=(bins.T, wrong, groups, weights), early_stopping_rounds=2, use_best_model=True)
    assert fitted.get_best_iteration() == 0
    assert fitted.tree_count_ == 1 and fitted.training_stats_["iterations_trained"] == 3
    first = CatBoostMetalRanker(iterations=1, **params).fit(bins.T, targets, group_id=groups, sample_weight=weights)
    np.testing.assert_allclose(fitted.training_predictions_, first.training_predictions_, atol=2e-6)


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax"])
@pytest.mark.parametrize("bootstrap", ["No", "MVS"])
def test_grouped_four_permutation_callback_snapshot_restores_every_cursor(metal_device, tmp_path, objective, bootstrap):
    bins, targets, features, borders, offsets, weights = problem(objective)
    rng = np.random.default_rng(338)
    matrices = (bins,) + tuple(bins[:, rng.permutation(bins.shape[1])] for _ in range(3))
    params = options(objective, bootstrap_type=bootstrap, random_strength=.4,
                     **({"subsample": .7} if bootstrap == "MVS" else {}))
    common = dict(group_offsets=offsets, sample_weight=weights, permutation_bins=matrices)
    path = tmp_path / "query.permutations.snapshot"
    stopped = run_training(bins, targets, features, borders, **params, **common,
        save_snapshot=True, snapshot_file=path, callback=lambda step: step.iteration < 2)
    assert stopped.stats["stop_reason"] == "callback" and len(stopped.depths) == 2
    resumed = run_training(bins, targets, features, borders, **params, **common,
                            save_snapshot=True, snapshot_file=path)
    full = run_training(bins, targets, features, borders, **params, **common)
    for field in ("depths", "split_features", "split_bins", "leaf_values", "predictions", "rmse"):
        np.testing.assert_array_equal(getattr(resumed, field), getattr(full, field))
    with np.load(path, allow_pickle=False) as snapshot:
        assert snapshot["permutation_predictions"].shape == (4, len(targets))
        np.testing.assert_array_equal(snapshot["permutation_predictions"][-1], resumed.predictions)


@pytest.mark.parametrize("beta,regularization", [(-1., -.02), (.5, .1)])
def test_parameterized_query_softmax_metric_and_snapshot(metal_device, tmp_path, beta, regularization):
    bins, targets, _, _, offsets, weights = problem("QuerySoftMax")
    groups = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    loss = f"QuerySoftMax:beta={beta};lambda={regularization}"
    model = CatBoostMetalRanker(iterations=3, depth=2, loss_function=loss,
        leaf_estimation_method="Gradient", leaf_estimation_iterations=2, eval_metric=loss).fit(
            bins.T, targets, group_id=groups, sample_weight=weights,
            eval_set=(bins.T, targets, groups, weights), use_best_model=False,
            save_snapshot=True, snapshot_file=tmp_path / "parameterized.query.snapshot")
    assert model.get_evals_result()["validation"][loss][-1] == pytest.approx(
        query_metric(model.training_predictions_, targets, weights, offsets, "QuerySoftMax", beta, regularization), abs=2e-6)


def test_query_evaluation_requires_its_own_group_boundaries_before_gpu(monkeypatch):
    bins, targets, features, borders, offsets, weights = problem()
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Invalid evaluation groups reached GPU"))
    with pytest.raises(ValueError, match="eval_group_offsets"):
        run_training(bins, targets, features, borders, **options(), group_offsets=offsets,
                     sample_weight=weights, eval_bins=bins, eval_targets=targets)


@pytest.mark.parametrize("evaluation_metric", ["PairLogit", "PairAccuracy"])
def test_query_metric_does_not_implicitly_generate_pairs(monkeypatch, evaluation_metric):
    from catboost import utils

    bins, targets, features, borders, offsets, weights = problem()
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Missing pairs reached GPU"))
    monkeypatch.setattr(utils, "eval_metric", lambda *a, **k: pytest.fail("Missing pairs reached pair generation"))
    with pytest.raises(ValueError, match="explicit supplied pairs"):
        run_training(bins, targets, features, borders, **options(), group_offsets=offsets,
                     sample_weight=weights, eval_metric=evaluation_metric)
