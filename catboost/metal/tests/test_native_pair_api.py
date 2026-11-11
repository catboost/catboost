"""Native PairLogit uses CatBoost Pool's prepared literal pair weights."""

import os

import numpy as np
import pytest
from catboost import CatBoostError, CatBoostRanker, Pool


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_QUERY_TESTS") != "1",
    reason="requires the rebuilt native Metal grouped adapter",
)


def data():
    rng = np.random.default_rng(934)
    x = rng.normal(size=(48, 3)).astype(np.float32)
    y = x[:, 0] + 0.4 * x[:, 1]
    groups = np.repeat(np.arange(8), 6)
    edges = []
    for start in range(0, 48, 6):
        order = np.argsort(y[start:start + 6]) + start
        edges.extend((int(order[i + 1]), int(order[i])) for i in range(5))
    weights = np.linspace(0.4, 2.0, len(edges), dtype=np.float32)
    return x, y, groups, np.asarray(edges, dtype=np.uint32), weights


def params(**extra):
    result = dict(task_type="GPU", loss_function="PairLogit", iterations=8,
                  depth=3, learning_rate=0.2, border_count=16, random_seed=41,
                  bootstrap_type="No", random_strength=0, score_function="Cosine",
                  leaf_estimation_backtracking="No", verbose=False, allow_writing_files=False)
    result.update(extra)
    return result


@pytest.mark.parametrize("score", ["L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_native_pair_loss_scores_leaf_weights_and_export(tmp_path, score, method):
    x, y, groups, edges, edge_weights = data()
    pool = Pool(x, y, group_id=groups, pairs=edges, pairs_weight=edge_weights)
    model = CatBoostRanker(**params(score_function=score, leaf_estimation_method=method,
                                   leaf_estimation_iterations=3)).fit(pool, eval_set=pool,
                                                                     use_best_model=False)
    raw = model.predict(x)
    expected = np.average(np.logaddexp(0, raw[edges[:, 1]] - raw[edges[:, 0]]), weights=edge_weights)
    history = model.get_evals_result()["learn"]["PairLogit"]
    assert history[-1] < history[0]
    assert history[-1] == pytest.approx(expected, rel=3e-6)
    cursor = 0
    for count in model.get_tree_leaf_counts():
        count = int(count)
        assert model.get_leaf_weights()[cursor:cursor + count].sum() == pytest.approx(2 * edge_weights.sum(), rel=3e-6)
        assert model.get_leaf_values()[cursor:cursor + count].sum() == pytest.approx(0, abs=3e-6)
        cursor += count
    np.testing.assert_allclose(model.predict(x, task_type="GPU"), raw, rtol=3e-6, atol=3e-6)
    path = tmp_path / "pairs.cbm"
    model.save_model(path)
    restored = CatBoostRanker().load_model(path)
    np.testing.assert_array_equal(restored.predict(x), raw)


def test_native_pair_supplied_weights_ignore_extra_object_and_group_weights():
    x, y, groups, edges, edge_weights = data()
    plain = Pool(x, y, group_id=groups, pairs=edges, pairs_weight=edge_weights)
    weighted = Pool(x, y, group_id=groups, pairs=edges, pairs_weight=edge_weights,
                    group_weight=np.repeat(np.linspace(0.5, 3, 8), 6))
    weighted.set_weight(np.linspace(0.7, 2.3, len(x)))
    config = params(leaf_estimation_iterations=2)
    first = CatBoostRanker(**config).fit(plain)
    second = CatBoostRanker(**config).fit(weighted)
    np.testing.assert_array_equal(first.get_leaf_values(), second.get_leaf_values())
    np.testing.assert_array_equal(first.get_leaf_weights(), second.get_leaf_weights())
    np.testing.assert_array_equal(first.predict(x), second.predict(x))


def test_native_pair_pool_does_not_require_labels_for_supplied_edges():
    x, _, groups, edges, edge_weights = data()
    x = np.tile(x, (1, 7))  # Avoid shared small-fit leaf-step tuning below20 features.
    pool = Pool(x, group_id=groups, pairs=edges, pairs_weight=edge_weights)
    model = CatBoostRanker(**params(iterations=3)).fit(pool)
    assert model.get_all_params()["leaf_estimation_iterations"] == 10
    assert np.isfinite(model.predict(x)).all()


def test_native_pair_reuses_shared_generated_pairs():
    x, y, groups, _, _ = data()
    pool = Pool(x, y, group_id=groups)
    model = CatBoostRanker(**params(iterations=4)).fit(pool)
    history = model.get_evals_result()["learn"]["PairLogit"]
    assert history[-1] < history[0]


class StopAfter:
    def after_iteration(self, info):
        return info.iteration < 3


@pytest.mark.parametrize("bootstrap", ["No", "Bayesian", "Bernoulli", "Poisson", "MVS"])
def test_native_pair_snapshot_preserves_edges_and_optimizer_state(tmp_path, bootstrap):
    x, y, groups, edges, edge_weights = data()
    pool = Pool(x, y, group_id=groups, pairs=edges, pairs_weight=edge_weights)
    sampling = dict(bootstrap_type=bootstrap)
    if bootstrap == "Bayesian":
        sampling["bagging_temperature"] = 0.6
    elif bootstrap != "No":
        sampling["subsample"] = 0.8
    common = params(**sampling, iterations=7, random_strength=0.6,
                    leaf_estimation_iterations=3, leaf_estimation_backtracking="AnyImprovement")
    snapshot = dict(common, allow_writing_files=True, train_dir=str(tmp_path),
                    save_snapshot=True, snapshot_interval=0, snapshot_file="pairs.snapshot")
    assert CatBoostRanker(**snapshot).fit(pool, callbacks=[StopAfter()]).tree_count_ == 3
    resumed = CatBoostRanker(**snapshot).fit(pool)
    direct = CatBoostRanker(**common).fit(pool)
    np.testing.assert_array_equal(resumed.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(resumed.get_leaf_weights(), direct.get_leaf_weights())
    np.testing.assert_array_equal(resumed.predict(x), direct.predict(x))
    assert resumed.get_evals_result() == direct.get_evals_result()
    changed = edge_weights.copy()
    changed[0] *= 2
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ"):
        CatBoostRanker(**snapshot).fit(Pool(x, y, group_id=groups, pairs=edges, pairs_weight=changed))
