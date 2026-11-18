"""Native Simple keeps repeated negative vector winners through requested depth.

A unique category per row retains P4 CTR histories while every training CTR
equals its prior. The only usable split is one numeric binary candidate.
"""

import json
import os

import numpy as np
import pytest
from catboost import CatBoost, Pool

from test_native_greedy_api import StopAfter
from test_native_vector_simple import gradients, leaf_equations


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native repeated-split vector Simple adapter",
)


@pytest.fixture(autouse=True)
def only_gpu_fits(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def problem(tmp_path):
    row = np.arange(128)
    binary = (row % 2).astype(np.float32)
    target = (row % 2).astype(np.int32)
    target[row % 11 == 0] = 2
    weights = (.4 + row % 9 / 5).astype(np.float32)
    weights[::17] = 0
    x = np.column_stack((binary.astype(object), [f"unique-{value}" for value in row]))
    pool = Pool(x, target, weight=weights, cat_features=[1])
    borders = tmp_path / "one-binary-border.tsv"
    borders.write_text("0\t0.5\n")
    pool.quantize(input_borders=str(borders))
    saved = tmp_path / "checked-borders.tsv"
    pool.save_quantization_borders(str(saved))
    assert saved.read_text().splitlines() == ["0\t0.5"]
    return pool, x, binary, target, weights


def options(objective, histories, **extra):
    return dict(task_type="GPU", loss_function=objective, boosting_type="Plain",
        data_partition="DocParallel", grow_policy="SymmetricTree", iterations=4,
        depth=3, learning_rate=.17, l2_leaf_reg=2.3, random_seed=761,
        random_strength=0, bootstrap_type="No", boost_from_average=False,
        score_function="L2" if objective == "MultiClass" else "Cosine",
        leaf_estimation_method="Simple", leaf_estimation_iterations=1,
        leaf_estimation_backtracking="No", permutation_count=histories,
        has_time=histories == 1, one_hot_max_size=1, max_ctr_complexity=1,
        simple_ctr=["Borders:CtrBorderCount=1:Prior=0.5"], ctr_target_border_count=1,
        rsm=1, verbose=False, metric_period=1, allow_writing_files=False) | extra


def document(model, path):
    model.save_model(str(path), format="json")
    return json.loads(path.read_text())


def check_repeated_simple_forest(model, payload, binary, target, weights, config):
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), [8] * model.tree_count_)
    cursor = np.zeros((len(binary), 3), np.float32)
    ids = binary.astype(np.uint32) * 7
    all_values, all_weights = [], []
    for tree in payload["oblivious_trees"]:
        splits = tree["splits"]
        assert len(splits) == 3
        assert all(split["split_type"] == "FloatFeature"
                   and split["float_feature_index"] == 0 and split["border"] == .5
                   for split in splits)
        assert splits[0] == splits[1] == splits[2]
        # No bootstrap means the weak source mass is the original object
        # weight. Repetition routes every row to leaf 0 or leaf 7 only.
        expected, masses = leaf_equations(config["loss_function"],
            gradients(config["loss_function"], target, cursor, weights), weights, ids, 8,
            config["l2_leaf_reg"], config["learning_rate"])
        actual = np.asarray(tree["leaf_values"]).reshape(8, 3)
        np.testing.assert_allclose(actual, expected, rtol=8e-5, atol=6e-6)
        np.testing.assert_allclose(tree["leaf_weights"], masses, rtol=5e-6, atol=5e-6)
        np.testing.assert_array_equal(actual[1:7], 0)
        np.testing.assert_array_equal(np.asarray(tree["leaf_weights"])[1:7], 0)
        assert np.all(masses[[0, 7]] > 0)
        assert np.max(np.abs(actual[[0, 7]])) > 1e-5
        cursor = np.float32(cursor + expected[ids])
        all_values.extend(expected.ravel())
        all_weights.extend(masses)
    np.testing.assert_allclose(model.get_leaf_values(), all_values, rtol=8e-5, atol=6e-6)
    np.testing.assert_allclose(model.get_leaf_weights(), all_weights, rtol=5e-6, atol=5e-6)
    np.testing.assert_allclose(np.asarray(model.get_test_evals()[0]).T, cursor, rtol=8e-5, atol=8e-6)


def exact_forest(actual, expected, x):
    assert actual.tree_count_ == expected.tree_count_
    for method in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_evals"):
        np.testing.assert_array_equal(getattr(actual, method)(), getattr(expected, method)())
    assert actual.get_evals_result() == expected.get_evals_result()
    np.testing.assert_array_equal(
        actual.predict(x, prediction_type="RawFormulaVal", task_type="GPU"),
        expected.predict(x, prediction_type="RawFormulaVal", task_type="GPU"))


@pytest.mark.parametrize("objective", ("MultiClass", "MultiClassOneVsAll"))
@pytest.mark.parametrize("histories", (1, 4))
def test_repeated_simple_binary_splits_and_inactive_leaves_resume_exactly(tmp_path, objective, histories):
    pool, x, binary, target, weights = problem(tmp_path)
    config = options(objective, histories)

    def trained(parameters, **extra):
        return CatBoost(parameters).fit(pool, eval_set=pool, use_best_model=False, **extra)

    direct = trained(config)
    assert direct.get_metadata()["metal_backend"] == "METAL"
    assert direct.get_metadata()["metal_permutations"] == str(histories)
    assert direct.get_all_params()["leaf_estimation_method"] == "Simple"
    check_repeated_simple_forest(direct, document(direct, tmp_path / "direct.json"),
                                 binary, target, weights, config)
    saved = config | dict(save_snapshot=True, snapshot_interval=0,
        snapshot_file="repeated.snapshot", allow_writing_files=True, train_dir=str(tmp_path))
    callback = StopAfter(2)
    partial = trained(saved, callbacks=[callback])
    assert partial.tree_count_ == 2 and callback.iterations == [1, 2]
    assert (tmp_path / "repeated.snapshot").is_file()
    resumed = trained(saved)
    exact_forest(resumed, direct, x)
    exact_forest(trained(saved), direct, x)
    assert document(resumed, tmp_path / "resumed.json")["oblivious_trees"] == document(
        direct, tmp_path / "compared.json")["oblivious_trees"]
    extended = trained(saved | dict(iterations=6))
    exact_forest(extended, trained(config | dict(iterations=6)), x)
    check_repeated_simple_forest(extended, document(extended, tmp_path / "extended.json"),
                                 binary, target, weights, config)


@pytest.mark.parametrize("objective", ("MultiClass", "MultiClassOneVsAll"))
def test_existing_default_leaf_method_retains_published_duplicate_stop(tmp_path, objective):
    pool, _, _, _, _ = problem(tmp_path)
    config = options(objective, 1, iterations=1)
    del config["leaf_estimation_method"]
    model = CatBoost(config).fit(pool)
    assert model.get_all_params()["leaf_estimation_method"] == "Newton"
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), [2])
    split, = document(model, tmp_path / "default.json")["oblivious_trees"][0]["splits"]
    assert split["split_type"] == "FloatFeature" and split["float_feature_index"] == 0
