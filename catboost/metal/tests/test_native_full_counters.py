"""Phase-specific CUDA Full frequencies, checked without CPU model fitting."""
from collections import Counter
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool

from test_native_compound_ctrs import (
    categorical_problem, check_final_tables, exported, independent_prediction,
    options as compound_options, snapshot_options,
)
from test_native_greedy_api import StopAfter


pytestmark = pytest.mark.skipif(os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal Full counter adapter")


@pytest.fixture(autouse=True)
def only_gpu_fits(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def problem():
    learn = np.repeat(np.array(["a", "b", "c", "d"], object), [12, 24, 36, 48])[:, None]
    learn = learn[np.random.default_rng(831).permutation(len(learn))]
    target = (learn[:, 0] == "a").astype(np.float32)
    weights = (.4 + np.arange(len(learn)) % 7 / 5).astype(np.float32)
    weights[::13] = 0
    evaluation = np.repeat(np.array(["a", "b", "c", "d", "eval-only"], object), [60, 2, 2, 2, 18])[:, None]
    eval_target = (evaluation[:, 0] == "a").astype(np.float32)
    return learn, target, weights, evaluation, eval_target


def options(path="PlainDP", count=1, **extra):
    config = compound_options("FeatureFreq", "Ordered" if path == "OrderedFP" else "Plain", count,
        complexity=1, iterations=1, depth=1, learning_rate=1, l2_leaf_reg=0,
        leaf_estimation_iterations=1, counter_calc_method="Full")
    if path == "PlainDP":
        config["data_partition"] = "DocParallel"
    return config | extra


def pools():
    x, y, w, test, test_y = problem()
    return Pool(x, y, cat_features=[0], weight=w), Pool(test, test_y, cat_features=[0])


@pytest.mark.parametrize("path", ["PlainDP", "PlainFP", "OrderedFP"])
@pytest.mark.parametrize("count", [1, 4])
def test_full_training_and_eval_counts_but_learn_only_export(path, count, tmp_path):
    x, y, weight, evaluation, eval_y = problem()
    learn, test = pools()
    model = CatBoost(options(path, count)).fit(learn, eval_set=test, use_best_model=False)
    document = exported(model, tmp_path / "full.json")
    ctr, = document["features_info"]["ctrs"]
    assert ctr["ctr_type"] == "FeatureFreq"
    split, = document["oblivious_trees"][0]["splits"]
    assert split["split_type"] == "OnlineCtr"
    counts = Counter(np.concatenate((x[:, 0], evaluation[:, 0])))
    # Counts ignore original sample weights, including zero-weight rows.
    encoded = np.array([(counts[value] + ctr["prior_numerator"]) /
                        (len(x) + len(evaluation) + ctr["prior_denomerator"]) for value in x[:, 0]], np.float32)
    encoded = (encoded + np.float32(ctr["shift"])) * np.float32(ctr["scale"])
    leaf = encoded > np.float32(split["border"])
    assert leaf.any() and not leaf.all()
    expected_weights = [weight[leaf == index].sum(dtype=np.float64) for index in (0, 1)]
    expected_leaves = [(weight[leaf == index] * y[leaf == index]).sum(dtype=np.float64) / expected_weights[index]
                       for index in (0, 1)]
    np.testing.assert_allclose(model.get_leaf_weights(), expected_weights, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(model.get_leaf_values(), expected_leaves, rtol=3e-6, atol=3e-6)
    full_eval = independent_prediction(document, np.concatenate((x, evaluation)), np.r_[y, eval_y], evaluation)
    exported_eval = independent_prediction(document, x, y, evaluation)
    np.testing.assert_allclose(model.get_test_eval(), full_eval, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(model.predict(test, task_type="GPU"), exported_eval, rtol=3e-6, atol=3e-6)
    assert np.max(np.abs(full_eval - exported_eval)) > .1
    check_final_tables(document, x, y)
    for fmt in ("cbm", "json"):
        file = tmp_path / ("restored." + fmt)
        model.save_model(file, format=fmt)
        restored = CatBoost().load_model(file, format=fmt)
        np.testing.assert_allclose(restored.predict(test, task_type="GPU"), exported_eval, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("path", ["PlainDP", "PlainFP", "OrderedFP"])
def test_full_without_eval_is_identical_to_skiptest(path):
    learn, _ = pools()
    full = CatBoost(options(path, 4, iterations=3)).fit(learn)
    skip = CatBoost(options(path, 4, iterations=3, counter_calc_method="SkipTest")).fit(learn)
    for method in ("get_leaf_values", "get_leaf_weights", "get_tree_leaf_counts"):
        np.testing.assert_array_equal(getattr(full, method)(), getattr(skip, method)())
    np.testing.assert_array_equal(full.predict(learn, task_type="GPU"), skip.predict(learn, task_type="GPU"))


@pytest.mark.parametrize("kind", ["Borders", "Buckets", "FloatTargetMeanValue"])
def test_full_does_not_change_target_dependent_histories(kind):
    learn, test = pools()
    config = options("PlainFP", 4, iterations=3, simple_ctr=[f"{kind}:CtrBorderCount=15:Prior=.5"])
    full = CatBoost(config).fit(learn, eval_set=test, use_best_model=False)
    skip = CatBoost(config | dict(counter_calc_method="SkipTest")).fit(learn, eval_set=test, use_best_model=False)
    for method in ("get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(full, method)(), getattr(skip, method)())
    assert full.get_evals_result() == skip.get_evals_result()


@pytest.mark.parametrize("path", ["PlainDP", "PlainFP", "OrderedFP"])
def test_full_snapshot_restores_eval_cursors_and_rejects_changed_first_eval(path, tmp_path):
    learn, test = pools()
    config = options(path, 4, iterations=4, learning_rate=.2)
    direct = CatBoost(config).fit(learn, eval_set=test, use_best_model=False)
    saved = snapshot_options(config, tmp_path)
    partial = CatBoost(saved).fit(learn, eval_set=test, use_best_model=False, callbacks=[StopAfter(2)])
    assert partial.tree_count_ == 2
    resumed = CatBoost(saved).fit(learn, eval_set=test, use_best_model=False)
    for method in ("get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(resumed, method)(), getattr(direct, method)())
    assert resumed.get_evals_result() == direct.get_evals_result()
    snapshot = tmp_path / "compound.snapshot"
    accepted = snapshot.read_bytes()
    x, y, _, evaluation, eval_y = problem()
    evaluation[0, 0] = "changed-eval-category"
    changed = Pool(evaluation, eval_y, cat_features=[0])
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot"):
        CatBoost(saved).fit(learn, eval_set=changed, use_best_model=False)
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot|parameters.*differ"):
        CatBoost(saved | dict(counter_calc_method="SkipTest")).fit(learn, eval_set=test, use_best_model=False)
    assert snapshot.read_bytes() == accepted


@pytest.mark.parametrize("boosting", ["Plain", "Ordered"])
def test_dynamic_frequency_helpers_ignore_full_like_cuda(boosting, tmp_path):
    x, y, future, pool_options = categorical_problem()
    learn = Pool(x, y, **pool_options)
    test = Pool(future, np.arange(len(future)) % 2, cat_features=[0, 1])
    config = compound_options("Borders", boosting, 4, iterations=4,
        combinations_ctr=["FeatureFreq:CtrBorderType=Uniform:CtrBorderCount=15:Prior=.5"])
    full = CatBoost(config | dict(counter_calc_method="Full")).fit(learn, eval_set=test, use_best_model=False)
    skip = CatBoost(config).fit(learn, eval_set=test, use_best_model=False)
    # The simple target CTRs ignore Full, and all dynamically generated
    # frequency tensors also keep learn-only statistics, including eval apply.
    for method in ("get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(full, method)(), getattr(skip, method)())
    assert int(full.get_metadata()["metal_tree_ctr_features"]) > 0
    assert full.get_evals_result() == skip.get_evals_result()
    check_final_tables(exported(full, tmp_path / "dynamic-full.json"), x, y)
