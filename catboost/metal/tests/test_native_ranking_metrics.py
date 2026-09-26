"""Native Metal ranking metrics share the CUDA target query-weight contract."""
import os
import numpy as np
import pytest
from catboost import CatBoost, CatBoostRanker, Pool
from test_ranking_metrics import METRICS, problem, expected_metric

pytestmark = pytest.mark.skipif(os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
                              reason="requires rebuilt native ranking metric weights")


@pytest.fixture(autouse=True)
def only_gpu_training(monkeypatch):
    original = CatBoost._fit
    def fit(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CatBoost, "_fit", fit)


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax", "PairLogit"])
@pytest.mark.parametrize("name", METRICS)
@pytest.mark.parametrize("weighted", [True, False])
def test_native_query_metric_weights_and_selection(objective, name, weighted):
    x, y, offsets, groups, weights, pairs, pair_weights = problem()
    name += ";use_weights=" + str(weighted).lower()
    group_weights = np.repeat(np.array([.5, 2, 1, .7, 3, .8], np.float32), np.diff(offsets))
    paired = objective == "PairLogit"
    pool = Pool(x, y, group_id=groups, group_weight=group_weights,
                **(dict(pairs=pairs, pairs_weight=pair_weights) if paired else {}))
    pool.set_weight(weights)
    model = CatBoostRanker(task_type="GPU", loss_function=objective, eval_metric=name,
             iterations=8, depth=3, learning_rate=.12, random_strength=0, bootstrap_type="No",
             leaf_estimation_iterations=2, leaf_estimation_backtracking="No", verbose=False,
             allow_writing_files=False, metric_period=1).fit(pool, eval_set=pool, use_best_model=False)
    raw = np.asarray(model.get_test_eval())
    expected = expected_metric(raw, y, offsets, weights * group_weights, name, paired)
    # Native metric descriptions canonicalize parameter order (NDCG places
    # use_weights first). Verify the same options before selecting its history.
    key = next(key for key in model.evals_result_["validation"] if key.partition(":")[0] == name.partition(":")[0])
    parameters = lambda value: dict(item.split("=", 1) for item in value.partition(":")[2].split(";"))
    assert parameters(key) == parameters(name)
    history = model.evals_result_["validation"][key]
    assert history[-1] == pytest.approx(expected, abs=2e-8)
    assert model.best_iteration_ == int(np.argmax(history))
