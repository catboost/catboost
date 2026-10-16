"""Native vector Langevin models, exact snapshots, and source no-noise cases.

All fits use the rebuilt GPU adapter. Controlled callback equations live in
test_vector_langevin.py; these tests exercise the public model and snapshot path.
"""

import json
import os
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool

from test_native_langevin import check_same, snapshot_options, tail


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1"
    or platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="requires the rebuilt native vector Langevin Metal adapter",
)

OBJECTIVES = (
    "MultiClass", "MultiClassOneVsAll", "MultiRMSE", "RMSEWithUncertainty",
    "MultiLogloss", "MultiCrossEntropy",
)
SCALAR_TARGET_OBJECTIVES = ("MultiClass", "MultiClassOneVsAll", "RMSEWithUncertainty")
GREEDY_CASES = (("Depthwise", 1, "Newton"), ("Lossguide", 4, "Gradient"),
                ("Region", 4, "Newton"))


@pytest.fixture(autouse=True)
def prohibit_cpu_fitting(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def options(objective, histories=1, **extra):
    return dict(
        task_type="GPU", loss_function=objective, boosting_type="Plain",
        data_partition="DocParallel", grow_policy="SymmetricTree",
        iterations=4, depth=2, learning_rate=.08, l2_leaf_reg=8.,
        random_seed=85117, random_strength=0., border_count=8,
        bootstrap_type="No", score_function="L2", boost_from_average=False,
        leaf_estimation_method="Newton", leaf_estimation_iterations=2,
        leaf_estimation_backtracking="No", permutation_count=histories,
        has_time=histories == 1, max_ctr_complexity=1, one_hot_max_size=1,
        langevin=True, diffusion_temperature=.02, metric_period=1,
        verbose=False, allow_writing_files=False, use_best_model=False,
    ) | extra


def problem(objective, categorical=False):
    rng = np.random.default_rng(782351)
    rows = 96
    x = rng.normal(size=(rows, 4)).astype(np.float32)
    signal = np.column_stack((1.2 * x[:, 0] - .4 * x[:, 1],
                              .9 * x[:, 1] + .3 * x[:, 2],
                              -.8 * x[:, 0] - .5 * x[:, 2])).astype(np.float32)
    if objective in ("MultiClass", "MultiClassOneVsAll"):
        target = np.argmax(signal, axis=1)
    elif objective == "MultiRMSE":
        target = signal + np.array([.7, -.3, .2], np.float32)
    elif objective == "RMSEWithUncertainty":
        target = signal[:, 0] + .15 * x[:, 3]
    elif objective == "MultiLogloss":
        target = (signal > np.array([.1, -.2, .15])).astype(np.float32)
    else:
        target = (.08 + .84 / (1 + np.exp(-signal))).astype(np.float32)
    weights = (.4 + np.arange(rows) % 9 / 6).astype(np.float32)
    pool_options = dict(weight=weights)
    evaluation = x.copy()
    if categorical:
        # Only scalar-target vector objectives currently have native CTR
        # histories. MultiRMSE/multilabel use numeric features below.
        assert objective in SCALAR_TARGET_OBJECTIVES
        category = rng.permutation(np.tile(np.arange(12), rows // 12))
        x = np.array([[f"kind-{value}"] for value in category], object)
        target = category % 3
        if objective == "RMSEWithUncertainty":
            target = ((category - 5.5) * .2 + .04 * rng.normal(size=rows)).astype(np.float32)
        evaluation = x.copy()
        evaluation[::19, 0] = "unseen-vector-langevin"
        pool_options["cat_features"] = [0]
    return Pool(x, target, **pool_options), Pool(evaluation, target, **pool_options)


def fit(config, learn, evaluation):
    model_type = (CatBoostRegressor if config["loss_function"] in
                  ("MultiRMSE", "RMSEWithUncertainty") else CatBoostClassifier)
    return model_type().set_params(**config).fit(learn, eval_set=evaluation)


def check_model(model, config, learn, evaluation, effective_histories):
    assert model.tree_count_ == config["iterations"]
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert model.get_metadata()["metal_permutations"] == str(effective_histories)
    assert model.get_metadata()["metal_langevin_weak_rng"] == "no_weak_noise"
    assert int(model.get_metadata()["metal_langevin_host_draw_count"]) > 1
    assert model.get_all_params()["langevin"] is True
    assert model.get_all_params()["diffusion_temperature"] == pytest.approx(config["diffusion_temperature"])
    assert np.isfinite(model.get_leaf_values()).all()
    assert np.isfinite(model._object._get_metal_training_cursor()).all()
    dimensions = 2 if config["loss_function"] == "RMSEWithUncertainty" else 3
    for pool in (learn, evaluation):
        raw = model.predict(pool, prediction_type="RawFormulaVal", task_type="GPU")
        assert raw.shape == (pool.num_row(), dimensions)
        assert np.isfinite(raw).all() and np.any(raw != 0)


def check_exact_resume(config, learn, evaluation, effective_histories, directory):
    direct, repeat = fit(config, learn, evaluation), fit(config, learn, evaluation)
    check_model(direct, config, learn, evaluation, effective_histories)
    check_same(direct, repeat, evaluation)
    assert direct.get_evals_result() == repeat.get_evals_result()

    partial = fit(snapshot_options(config, directory, 2), learn, evaluation)
    assert partial.tree_count_ == 2
    path = directory / "langevin.snapshot"
    policy = config["grow_policy"]
    capacity = (config["depth"] + 1 if policy == "Region" else
                min(config["max_leaves"], 1 << min(config["depth"], 16)) if policy == "Lossguide" else
                1 << config["depth"])
    history_shape = dict(permutations=effective_histories, leaf_capacity=capacity,
                         dimension=2 if config["loss_function"] == "RMSEWithUncertainty" else 3)
    partial_bytes = path.read_bytes()
    offset, saved = tail(partial_bytes, trees=2, **history_shape)
    assert tail(partial_bytes[:offset + 17], trees=2, **history_shape) == (offset, saved)
    cache = int(config["bootstrap_type"] != "No")
    assert saved == (int(partial.get_metadata()["metal_langevin_host_draw_count"]), 2, cache)
    resumed = fit(snapshot_options(config, directory, config["iterations"]), learn, evaluation)
    assert resumed._object._get_metal_training_info()["resumed_iterations"] == 2
    check_same(direct, resumed, evaluation)
    np.testing.assert_array_equal(direct.get_test_evals(), resumed.get_test_evals())
    assert direct.get_evals_result() == resumed.get_evals_result()
    _, restored_state = tail(path.read_bytes(), trees=config["iterations"], **history_shape)
    assert restored_state == (int(direct.get_metadata()["metal_langevin_host_draw_count"]),
                              config["iterations"], cache)

    complete_bytes = path.read_bytes()
    completed = fit(snapshot_options(config, directory, config["iterations"]), learn, evaluation)
    assert completed._object._get_metal_training_info()["resumed_iterations"] == config["iterations"]
    check_same(direct, completed, evaluation)
    assert path.read_bytes() == complete_bytes

    model_path = directory / "vector-langevin.cbm"
    direct.save_model(str(model_path))
    reloaded = CatBoost().load_model(str(model_path))
    np.testing.assert_array_equal(reloaded.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(
        reloaded.predict(evaluation, prediction_type="RawFormulaVal", task_type="GPU"),
        direct.predict(evaluation, prediction_type="RawFormulaVal", task_type="GPU"))
    return direct


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("histories", (1, 4))
def test_symmetric_vector_langevin_repeat_and_snapshot(objective, histories, tmp_path):
    categorical = objective in SCALAR_TARGET_OBJECTIVES
    learn, evaluation = problem(objective, categorical)
    config = options(objective, histories,
                     leaf_estimation_method="Newton" if histories == 1 else "Gradient")
    if categorical:
        ctr_type = "FloatTargetMeanValue" if objective == "RMSEWithUncertainty" else "Borders"
        config["simple_ctr"] = [ctr_type + ":CtrBorderCount=7:Prior=0.5"]
    if histories == 4:
        config.update(bootstrap_type="Bernoulli", subsample=.8)
    # Requested P4 has one effective history when no dependent features exist.
    effective = histories if categorical else 1
    direct = check_exact_resume(config, learn, evaluation, effective, tmp_path)
    if categorical:
        model_path = tmp_path / "categorical.json"
        direct.save_model(str(model_path), format="json")
        trees = json.loads(model_path.read_text())["oblivious_trees"]
        assert any(split["split_type"] == "OnlineCtr"
                   for tree in trees for split in tree.get("splits") or [])


@pytest.mark.parametrize("objective", SCALAR_TARGET_OBJECTIVES)
@pytest.mark.parametrize("policy,histories,method", GREEDY_CASES)
def test_greedy_vector_langevin_repeat_and_snapshot(objective, policy, histories, method, tmp_path):
    learn, evaluation = problem(objective, categorical=True)
    ctr_type = "FloatTargetMeanValue" if objective == "RMSEWithUncertainty" else "Borders"
    config = options(objective, histories, grow_policy=policy,
                     leaf_estimation_method=method,
                     simple_ctr=[ctr_type + ":CtrBorderCount=7:Prior=0.5"])
    if policy == "Lossguide":
        config["max_leaves"] = 4
    if histories == 4:
        config.update(bootstrap_type="Bernoulli", subsample=.8)
    check_exact_resume(config, learn, evaluation, histories, tmp_path)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_zero_temperature_preserves_leaf_draws_and_positive_temperature_changes_leaves(objective):
    learn, evaluation = problem(objective)
    # Root-only trees avoid different split choices and isolate leaf noise.
    config = options(objective, depth=0, iterations=3, diffusion_temperature=0.)
    zero = fit(config, learn, evaluation)
    disabled = fit(config | {"langevin": False}, learn, evaluation)
    hot = fit(config | {"diffusion_temperature": .02}, learn, evaluation)
    np.testing.assert_allclose(zero.get_leaf_values(), disabled.get_leaf_values(), rtol=8e-5, atol=8e-6)
    np.testing.assert_allclose(
        zero.predict(evaluation, prediction_type="RawFormulaVal", task_type="GPU"),
        disabled.predict(evaluation, prediction_type="RawFormulaVal", task_type="GPU"),
        rtol=8e-5, atol=8e-6)
    assert "metal_langevin_host_rng" not in disabled.get_metadata()
    # DocParallel constructor: one draw. Each root still owns a search draw;
    # leaf walker calls initial G/H then trial/accepted G for each iteration.
    expected = 1 + config["iterations"] * (1 + 2 + 2 * config["leaf_estimation_iterations"])
    assert int(zero.get_metadata()["metal_langevin_host_draw_count"]) == expected
    assert int(hot.get_metadata()["metal_langevin_host_draw_count"]) == expected
    assert np.isfinite(hot.get_leaf_values()).all()
    assert np.max(np.abs(hot.get_leaf_values() - zero.get_leaf_values())) > 1e-6


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_simple_root_has_search_draws_without_weak_or_leaf_noise(objective):
    learn, evaluation = problem(objective)
    config = options(objective, depth=0, iterations=3,
                     leaf_estimation_method="Simple", leaf_estimation_iterations=1,
                     diffusion_temperature=1000.)
    enabled = fit(config, learn, evaluation)
    disabled = fit(config | {"langevin": False}, learn, evaluation)
    np.testing.assert_array_equal(enabled.get_tree_leaf_counts(), np.ones(config["iterations"], dtype=np.uint32))
    np.testing.assert_array_equal(enabled.get_leaf_values(), disabled.get_leaf_values())
    np.testing.assert_array_equal(
        enabled.predict(evaluation, prediction_type="RawFormulaVal", task_type="GPU"),
        disabled.predict(evaluation, prediction_type="RawFormulaVal", task_type="GPU"))
    np.testing.assert_array_equal(enabled._object._get_metal_training_cursor(),
                                  disabled._object._get_metal_training_cursor())
    assert enabled.get_metadata()["metal_langevin_weak_rng"] == "no_weak_noise"
    assert int(enabled.get_metadata()["metal_langevin_host_draw_count"]) == 1 + config["iterations"]
    assert "metal_langevin_host_rng" not in disabled.get_metadata()
