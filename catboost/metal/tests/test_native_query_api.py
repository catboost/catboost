"""Standard CatBoostRanker acceptance for the native grouped-objective adapter."""

import os

import numpy as np
import pytest
from catboost import CatBoostError, CatBoostRanker, Pool


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_QUERY_TESTS") != "1",
    reason="requires the rebuilt native Metal query adapter",
)


def dataset():
    rng = np.random.default_rng(412)
    features = rng.normal(size=(120, 4)).astype(np.float32)
    labels = (features[:, 0] * 1.4 - features[:, 1] * 0.4).astype(np.float32)
    groups = np.repeat(np.arange(20), 6)
    weights = np.linspace(0.4, 1.4, len(labels), dtype=np.float32)
    group_weights = np.repeat(np.linspace(0.5, 2, 20, dtype=np.float32), 6)
    return features, labels, groups, weights, group_weights


def params(**extra):
    result = dict(task_type="GPU", loss_function="QueryRMSE", iterations=12,
                  depth=3, learning_rate=0.15, random_seed=47, border_count=20,
                  bootstrap_type="No", random_strength=0, score_function="Cosine",
                  leaf_estimation_backtracking="No", verbose=False,
                  allow_writing_files=False)
    result.update(extra)
    return result


def pool_for(loss, *, separate=False):
    x, y, groups, weights, group_weights = dataset()
    if loss.startswith("QuerySoftMax"):
        y = np.exp(y / 3).astype(np.float32)
    if separate:
        pool = Pool(x, y, group_id=groups, group_weight=group_weights)
        pool.set_weight(weights)
    else:
        pool = Pool(x, y, group_id=groups, weight=weights * group_weights)
    return pool, x, y, groups, weights * group_weights


def query_rmse(raw, target, groups, weights):
    residual = target.astype(np.float64) - raw
    for group in np.unique(groups):
        mask = groups == group
        residual[mask] -= np.average(residual[mask], weights=weights[mask])
    return np.sqrt(np.average(residual ** 2, weights=weights))


@pytest.mark.parametrize("loss", ["QueryRMSE", "QuerySoftMax:beta=0.7;lambda=0.03"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_native_query_weighted_objective_and_standard_model(tmp_path, loss, method):
    pool, x, y, groups, weights = pool_for(loss)
    model = CatBoostRanker(**params(loss_function=loss, leaf_estimation_method=method,
                                   leaf_estimation_iterations=3)).fit(pool, eval_set=pool,
                                                                     use_best_model=False)
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert model.get_all_params()["boosting_type"] == "Plain"
    history = next(iter(model.get_evals_result()["learn"].values()))
    assert np.isfinite(history).all()
    assert history[-1] < history[0]
    raw = model.predict(x)
    np.testing.assert_allclose(model.predict(x, task_type="GPU"), raw, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(model.get_test_eval(), raw, atol=3e-6, rtol=3e-6)
    if loss == "QueryRMSE":
        assert history[-1] == pytest.approx(query_rmse(raw, y, groups, weights), rel=3e-6)
    for format_ in ("cbm", "json"):
        path = tmp_path / ("query." + format_)
        model.save_model(path, format=format_)
        restored = CatBoostRanker().load_model(path, format=format_)
        if format_ == "cbm":
            np.testing.assert_array_equal(restored.predict(x), raw)
        else:
            np.testing.assert_array_max_ulp(restored.predict(x), raw, maxulp=2)


@pytest.mark.parametrize("loss,method,iterations", [
    ("QueryRMSE", "Newton", 1), ("QuerySoftMax", "Gradient", 100),
])
def test_native_query_public_leaf_defaults(loss, method, iterations):
    pool, *_ = pool_for(loss)
    # Shared CatBoost data-dependent tuning sets one leaf step for very small
    # fits with fewer than20 features. Keep that separate from loss defaults.
    pool = Pool(np.tile(pool.get_features(), (1, 5)), pool.get_label(),
                group_id=dataset()[2], weight=pool.get_weight())
    model = CatBoostRanker(**params(loss_function=loss, iterations=2)).fit(pool)
    options = model.get_all_params()
    assert options["leaf_estimation_method"] == method
    assert options["leaf_estimation_iterations"] == iterations


@pytest.mark.parametrize("loss", ["QueryRMSE", "QuerySoftMax"])
def test_native_query_group_and_object_weights_are_multiplied_once(loss):
    separate, x, *_ = pool_for(loss, separate=True)
    merged, *_ = pool_for(loss)
    options = params(loss_function=loss, leaf_estimation_iterations=2, iterations=6)
    first = CatBoostRanker(**options).fit(separate)
    second = CatBoostRanker(**options).fit(merged)
    np.testing.assert_array_equal(first.get_leaf_values(), second.get_leaf_values())
    np.testing.assert_array_equal(first.get_leaf_weights(), second.get_leaf_weights())
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    assert first.get_evals_result() == second.get_evals_result()


@pytest.mark.parametrize("score", ["L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2"])
def test_native_query_scalar_score_functions(score):
    pool, *_ = pool_for("QueryRMSE")
    model = CatBoostRanker(**params(score_function=score, iterations=5)).fit(pool)
    history = model.get_evals_result()["learn"]["QueryRMSE"]
    assert history[-1] < history[0]


class StopAfter:
    def after_iteration(self, info):
        return info.iteration < 3


@pytest.mark.parametrize("bootstrap", ["No", "Bayesian", "Bernoulli", "Poisson", "MVS"])
def test_native_query_snapshot_restores_weighted_group_training(tmp_path, bootstrap):
    pool, x, *_ = pool_for("QueryRMSE", separate=True)
    sampling = dict(bootstrap_type=bootstrap)
    if bootstrap == "Bayesian":
        sampling["bagging_temperature"] = 0.6
    elif bootstrap != "No":
        sampling["subsample"] = 0.8
    common = params(**sampling, iterations=7, random_strength=0.6,
                    leaf_estimation_iterations=3, leaf_estimation_backtracking="AnyImprovement")
    snapshot_options = dict(common, allow_writing_files=True, train_dir=str(tmp_path),
                            save_snapshot=True, snapshot_interval=0, snapshot_file="query.snapshot")
    first = CatBoostRanker(**snapshot_options).fit(pool, callbacks=[StopAfter()])
    assert first.tree_count_ == 3
    resumed = CatBoostRanker(**snapshot_options).fit(pool)
    direct = CatBoostRanker(**common).fit(pool)
    np.testing.assert_array_equal(resumed.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(resumed.get_leaf_weights(), direct.get_leaf_weights())
    np.testing.assert_array_equal(resumed.predict(x), direct.predict(x))
    assert resumed.get_evals_result() == direct.get_evals_result()


def test_native_query_prequantized_pool_and_initial_model():
    pool, x, *_ = pool_for("QueryRMSE")
    pool.quantize(border_count=20)
    initial = CatBoostRanker(**params(iterations=3)).fit(pool)
    continued = CatBoostRanker(**params(iterations=3)).fit(pool, init_model=initial)
    assert continued.tree_count_ == 6
    assert np.isfinite(continued.predict(x)).all()


def test_native_query_rejects_ordered_boosting():
    pool, *_ = pool_for("QueryRMSE")
    with pytest.raises(CatBoostError, match="(?i)query.*plain|plain.*query|ordered"):
        CatBoostRanker(**params(boosting_type="Ordered")).fit(pool)
