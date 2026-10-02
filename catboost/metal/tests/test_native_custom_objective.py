"""Native Metal scalar custom objectives and their public training lifecycle.

The Python callback supplies an MSL function body once; per-row derivatives and
objective values must execute on Metal. Built-in RMSE fits also use Metal.
The quartic checks derive the depth-zero update directly from original rows,
and ordinary CPU prediction is used only to check exported model readers.
"""

import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, CatBoostRegressor, Pool

from test_native_compound_ctrs import categorical_problem, options as compound_options


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal custom-objective adapter",
)

PATHS = ("plain-doc", "plain-feature", "ordered-feature")
RMSE_BODY = """
const float residual = target - approx;
return float3(-weight * residual * residual, weight * residual, weight);
"""
QUARTIC_BODY = """
const float residual = target - approx;
const float square = residual * residual;
return float3(-weight * (0.25f * square * square + 0.5f * square),
              weight * residual * (square + 1.0f),
              weight * (3.0f * square + 1.0f));
"""


class MetalObjective:
    def __init__(self, source=RMSE_BODY):
        self.source = source

    def calc_ders_range_metal(self):
        return self.source

    def calc_ders_range(self, *args):
        raise AssertionError("Metal training must not evaluate a CPU objective callback")

    def calc_ders_range_gpu(self, *args):
        raise AssertionError("Metal training must not evaluate a CUDA objective callback")


class StopAfter:
    def __init__(self, stop=None):
        self.stop = stop
        self.iterations = []

    def after_iteration(self, info):
        self.iterations.append(info.iteration)
        return self.stop is None or info.iteration < self.stop


@pytest.fixture(autouse=True)
def require_gpu_training(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def options(path="plain-doc", **extra):
    result = dict(
        task_type="GPU", grow_policy="SymmetricTree",
        boosting_type="Ordered" if path == "ordered-feature" else "Plain",
        data_partition="DocParallel" if path == "plain-doc" else "FeatureParallel",
        loss_function=MetalObjective(), eval_metric="RMSE", custom_metric=["MAE"],
        iterations=6, depth=3, learning_rate=.15, l2_leaf_reg=2,
        random_seed=413, bootstrap_type="No", random_strength=0,
        score_function="Cosine", leaf_estimation_method="Newton",
        leaf_estimation_iterations=2, leaf_estimation_backtracking="No",
        boost_from_average=False, permutation_count=1, has_time=True,
        one_hot_max_size=2, max_ctr_complexity=1, model_size_reg=0,
        counter_calc_method="SkipTest", border_count=16,
        verbose=False, allow_writing_files=False, metric_period=1,
    )
    if path == "ordered-feature":
        result.update(min_fold_size=16, fold_len_multiplier=1.7, fold_permutation_block=3)
    return result | extra


def numeric_problem():
    rng = np.random.default_rng(836)
    x = rng.normal(size=(128, 4)).astype(np.float32)
    y = (.6 * x[:, 0] - .3 * x[:, 1] + .15 * x[:, 2] ** 2).astype(np.float32)
    weights = np.linspace(.3, 1.8, len(y), dtype=np.float32)
    weights[::19] = 0
    return x, y, weights


def problem(kind):
    if kind == "compound":
        x, y, future, pool_options = categorical_problem()
        return Pool(x, y, **pool_options), future
    x, y, weights = numeric_problem()
    if kind == "numeric":
        return Pool(x, y, weight=weights), x
    categories = np.where(x[:, 0] > 0, "left", "right")
    if kind == "simple":
        categories = np.asarray([f"category-{row % 7}" for row in range(len(x))])
        y = y + np.asarray([(row % 7) / 4 for row in range(len(x))], dtype=np.float32)
    mixed = np.column_stack((categories, x[:, 1:])).astype(object)
    return Pool(mixed, y, weight=weights, cat_features=[0]), mixed


def categorical_options(path, kind, count=1, **extra):
    if kind == "compound":
        result = compound_options(boosting="Ordered" if path == "ordered-feature" else "Plain",
                                  count=count)
        result.update(loss_function=MetalObjective(), eval_metric="RMSE", custom_metric=["MAE"])
    else:
        ctr = "Borders:CtrBorderType=Uniform:CtrBorderCount=15:Prior=0.5"
        result = options(path, permutation_count=count, has_time=count == 1,
                         simple_ctr=[ctr], ctr_target_border_count=1)
    return result | extra


def fit(config, pool, **kwargs):
    return CatBoostRegressor().set_params(**config).fit(pool, **kwargs)


def raw(model, data, task="GPU"):
    return model.predict(data, prediction_type="RawFormulaVal", task_type=task)


def assert_rmse_equivalent(actual, expected, future):
    assert actual.get_metadata()["metal_backend"] == "METAL"
    assert expected.get_metadata()["metal_backend"] == "METAL"
    np.testing.assert_array_equal(actual.get_tree_leaf_counts(), expected.get_tree_leaf_counts())
    np.testing.assert_allclose(actual.get_leaf_values(), expected.get_leaf_values(), atol=4e-6, rtol=5e-6)
    np.testing.assert_allclose(actual.get_leaf_weights(), expected.get_leaf_weights(), atol=4e-6, rtol=5e-6)
    np.testing.assert_allclose(raw(actual, future), raw(expected, future), atol=5e-6, rtol=5e-6)
    for dataset, history in actual.get_evals_result().items():
        for metric in ("RMSE", "MAE:use_weights=true", "MAE:use_weights=false"):
            np.testing.assert_allclose(history[metric], expected.get_evals_result()[dataset][metric],
                                       atol=4e-6, rtol=5e-6)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
@pytest.mark.parametrize("backtracking", ("No", "AnyImprovement", "Armijo"))
def test_custom_rmse_matches_builtin_metal_with_weighted_metrics(path, method, backtracking):
    x, y, weights = numeric_problem()
    pool = Pool(x, y, weight=weights)
    config = options(path, leaf_estimation_method=method, leaf_estimation_iterations=3,
                     leaf_estimation_backtracking=backtracking)
    custom = fit(config, pool, eval_set=pool, use_best_model=False)
    builtin = fit(config | dict(loss_function="RMSE"), pool, eval_set=pool, use_best_model=False)
    assert_rmse_equivalent(custom, builtin, x)
    prediction = raw(custom, x)
    metrics = custom.get_evals_result()["validation"]
    assert metrics["RMSE"][-1] == pytest.approx(
        np.sqrt(np.average((prediction - y) ** 2, weights=weights)), rel=5e-6, abs=3e-6)
    assert metrics["MAE:use_weights=true"][-1] == pytest.approx(
        np.average(np.abs(prediction - y), weights=weights), rel=5e-6, abs=3e-6)
    assert metrics["MAE:use_weights=false"][-1] == pytest.approx(
        np.mean(np.abs(prediction - y)), rel=5e-6, abs=3e-6)
    np.testing.assert_allclose(custom.get_test_eval(), prediction, atol=4e-6, rtol=5e-6)


CATEGORICAL_CASES = [(path, kind) for path in PATHS for kind in ("onehot", "simple", "compound")
                     if not (path == "plain-doc" and kind == "compound")]


@pytest.mark.parametrize("path,kind", CATEGORICAL_CASES)
@pytest.mark.parametrize("count", (1, 4))
def test_custom_rmse_categorical_permutations_and_standard_readers(tmp_path, path, kind, count):
    pool, future = problem(kind)
    config = categorical_options(path, kind, count)
    custom = fit(config, pool, eval_set=pool, use_best_model=False)
    builtin = fit(config | dict(loss_function="RMSE"), pool, eval_set=pool, use_best_model=False)
    assert_rmse_equivalent(custom, builtin, future)
    if kind == "compound":
        assert int(custom.get_metadata()["metal_tree_ctr_features"]) > 0
    expected = raw(custom, future)
    for fmt in ("cbm", "json"):
        path_out = tmp_path / ("custom." + fmt)
        custom.save_model(path_out, format=fmt)
        restored = CatBoostRegressor().load_model(path_out, format=fmt)
        for reader in (custom, restored):
            for task in ("CPU", "GPU"):
                np.testing.assert_allclose(raw(reader, future, task), expected, atol=5e-6, rtol=5e-6)
        if kind == "compound" and fmt == "json":
            document = json.loads(path_out.read_text())
            assert any(len(ctr["elements"]) > 1 for ctr in document["features_info"]["ctrs"])


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
def test_quartic_depth_zero_matches_independent_weighted_updates(path, method):
    # Nonconstant positive curvature and a nonquadratic gradient distinguish
    # this from treating every custom objective as a built-in squared error.
    y = np.asarray([-.7, -.3, .15, .4, .65, .9, 100.], dtype=np.float32)
    weights = np.asarray([.4, 1.3, 2.1, .7, 1.8, .5, 0.], dtype=np.float32)
    x = np.column_stack((np.arange(len(y)), np.arange(len(y)) ** 2)).astype(np.float32)
    config = options(path, loss_function=MetalObjective(QUARTIC_BODY), iterations=4,
                     depth=0, l2_leaf_reg=0, learning_rate=.2,
                     leaf_estimation_method=method, leaf_estimation_iterations=1,
                     min_fold_size=2)
    model = fit(config, Pool(x, y, weight=weights))
    cursor = 0.
    leaves = []
    for _ in range(config["iterations"]):
        residual = y.astype(np.float64) - cursor
        gradient = np.sum(weights * residual * (1 + residual ** 2))
        denominator = (np.sum(weights * (1 + 3 * residual ** 2))
                       if method == "Newton" else np.sum(weights, dtype=np.float64))
        update = config["learning_rate"] * gradient / denominator
        leaves.append(update)
        cursor += update
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), np.ones(len(leaves), dtype=np.uint32))
    np.testing.assert_allclose(model.get_leaf_values(), leaves, atol=3e-6, rtol=5e-6)
    np.testing.assert_allclose(raw(model, x), cursor, atol=3e-6, rtol=5e-6)
    np.testing.assert_allclose(model.get_leaf_weights(), weights.sum(dtype=np.float64), atol=3e-6)


LIFECYCLE_CASES = [(path, kind) for path in PATHS for kind in ("numeric", "compound")
                   if not (path == "plain-doc" and kind == "compound")]


@pytest.mark.parametrize("path,kind", LIFECYCLE_CASES)
def test_custom_objective_prequantized_pool_matches_builtin_rmse(tmp_path, path, kind):
    pool, future = problem(kind)
    pool.quantize(border_count=16)
    saved = tmp_path / "custom.quantized"
    pool.save(saved)
    restored = Pool("quantized://" + str(saved))
    config = categorical_options(path, kind)
    custom = fit(config, restored, eval_set=restored, use_best_model=False)
    builtin = fit(config | dict(loss_function="RMSE"), restored,
                  eval_set=restored, use_best_model=False)
    assert_rmse_equivalent(custom, builtin, future)
    np.testing.assert_allclose(raw(custom, restored), raw(custom, pool), atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("path,kind", LIFECYCLE_CASES)
def test_custom_objective_exact_snapshot_recovery_uses_saved_cursors(tmp_path, path, kind):
    pool, future = problem(kind)
    config = categorical_options(path, kind, count=4, iterations=7,
                                 leaf_estimation_backtracking="Armijo")
    snapshot = config | dict(save_snapshot=True, snapshot_interval=0, snapshot_file="custom.snapshot",
                             train_dir=str(tmp_path), allow_writing_files=True)
    stopped = fit(snapshot, pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter(3)])
    assert stopped.tree_count_ == 3
    assert (tmp_path / "custom.snapshot").is_file()
    recorder = StopAfter()
    # A new Python object with identical source must recover the saved state.
    resumed = fit(snapshot | dict(loss_function=MetalObjective()), pool, eval_set=pool,
                  use_best_model=False, callbacks=[recorder])
    direct = fit(config, pool, eval_set=pool, use_best_model=False)
    assert recorder.iterations == [4, 5, 6, 7]
    for method in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(resumed, method)(), getattr(direct, method)())
    np.testing.assert_array_equal(raw(resumed, future), raw(direct, future))
    assert resumed.get_evals_result() == direct.get_evals_result()
    completed = fit(snapshot, pool, eval_set=pool, use_best_model=False)
    np.testing.assert_array_equal(raw(completed, future), raw(direct, future))


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("completed", (False, True))
def test_custom_snapshot_rejects_changed_source_even_without_more_iterations(tmp_path, path, completed):
    pool, _ = problem("numeric")
    config = options(path, iterations=4, save_snapshot=True, snapshot_interval=0,
                     snapshot_file="source.snapshot", train_dir=str(tmp_path), allow_writing_files=True)
    fit(config, pool, callbacks=[] if completed else [StopAfter(2)])
    saved = (tmp_path / "source.snapshot").read_bytes()
    changed = MetalObjective(QUARTIC_BODY)
    with pytest.raises(CatBoostError, match="(?i)source|objective|snapshot|incompat|match"):
        fit(config | dict(loss_function=changed), pool)
    assert (tmp_path / "source.snapshot").read_bytes() == saved


@pytest.mark.parametrize("path,kind", LIFECYCLE_CASES)
def test_custom_objective_initial_model_and_baseline_match_builtin_rmse(path, kind):
    pool, future = problem(kind)
    config = categorical_options(path, kind, iterations=3)
    first = fit(config | dict(loss_function="RMSE"), pool)
    custom = fit(config, pool, init_model=first)
    builtin = fit(config | dict(loss_function="RMSE"), pool, init_model=first)
    assert custom.tree_count_ == builtin.tree_count_ == 6
    assert_rmse_equivalent(custom, builtin, future)
    baseline = raw(first, pool)
    pool.set_baseline(baseline)
    from_baseline = fit(config, pool, eval_set=pool, use_best_model=False)
    # Predictions on a Pool include its baseline; feature-only predictions do
    # not. The saved first forest supplies exactly that baseline on future rows.
    np.testing.assert_allclose(raw(from_baseline, future) + raw(first, future),
                               raw(custom, future), atol=5e-6, rtol=5e-6)
    np.testing.assert_allclose(from_baseline.get_test_eval(), raw(custom, pool) - baseline,
                               atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("path", PATHS)
def test_custom_metric_drives_early_stopping_and_best_model(path):
    x, y, weights = numeric_problem()
    model = fit(options(path, iterations=30, early_stopping_rounds=3, learning_rate=.3,
                        use_best_model=True, best_model_min_trees=2),
                Pool(x, y, weight=weights), eval_set=Pool(x, -y, weight=weights))
    assert 2 <= model.tree_count_ < 30
    assert model.get_best_iteration() >= 0
    assert np.isfinite(raw(model, x)).all()
    assert model.get_evals_result()["validation"]["RMSE"]


class CpuOnlyObjective:
    def calc_ders_range(self, *args):
        raise AssertionError("CPU objective callback must never be invoked by Metal")


class CudaOnlyObjective:
    def calc_ders_range_gpu(self, *args):
        raise AssertionError("CUDA objective callback must never be invoked by Metal")


@pytest.mark.parametrize("objective", (CpuOnlyObjective(), CudaOnlyObjective()))
def test_custom_objective_requires_explicit_metal_body(objective):
    pool, _ = problem("numeric")
    with pytest.raises(CatBoostError, match="(?i)metal|objective"):
        fit(options(loss_function=objective, iterations=1), pool)


@pytest.mark.parametrize("source", (None, b"return float3(0.0f);", 42, ""))
def test_custom_objective_requires_nonempty_source_string(source):
    pool, _ = problem("numeric")
    with pytest.raises((CatBoostError, TypeError, ValueError), match="(?i)source|body|string|metal|objective"):
        fit(options(loss_function=MetalObjective(source), iterations=1), pool)


def test_custom_objective_reports_msl_compiler_failure():
    pool, _ = problem("numeric")
    objective = MetalObjective("return this_symbol_does_not_exist(approx, target, weight);")
    with pytest.raises(CatBoostError, match="(?i)metal|compile|source|symbol|library"):
        fit(options(loss_function=objective, iterations=1), pool)


@pytest.mark.parametrize("component", (0, 1, 2))
@pytest.mark.parametrize("bits", ("0x7fc00000u", "0x7f800000u"))
@pytest.mark.parametrize("path", ("plain-doc", "ordered-feature"))
def test_custom_objective_rejects_nonfinite_values_derivatives_and_curvature(path, component, bits):
    values = ["-weight * (target - approx) * (target - approx)",
              "weight * (target - approx)", "weight"]
    values[component] = f"as_type<float>({bits})"
    source = "return float3(" + ", ".join(values) + ");"
    pool, _ = problem("numeric")
    with pytest.raises(CatBoostError, match="(?i)finite|invalid|objective|curvature|derivative"):
        fit(options(path, loss_function=MetalObjective(source), iterations=1, depth=0), pool)


@pytest.mark.parametrize("path", ("plain-doc", "ordered-feature"))
def test_custom_objective_rejects_negative_curvature(path):
    pool, _ = problem("numeric")
    source = "return float3(-1.0f, weight * (target - approx), -weight);"
    with pytest.raises(CatBoostError, match="(?i)finite|invalid|curvature|hessian|nonnegative|objective"):
        fit(options(path, loss_function=MetalObjective(source), iterations=1), pool)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
def test_negative_row_curvature_cannot_cancel_positive_rows_in_one_leaf(path, method):
    x = np.arange(48, dtype=np.float32).reshape(16, 3)
    y = np.linspace(.2, 1.2, len(x), dtype=np.float32)
    y[0] = -.25
    weights = np.linspace(.5, 1.5, len(x), dtype=np.float32)
    curvature = weights * np.where(y < 0, -1., 3.)
    assert curvature[0] < 0 and curvature.sum() > 0
    source = """
const float residual = target - approx;
return float3(-weight * residual * residual, weight * residual,
              weight * (target < 0.0f ? -1.0f : 3.0f));
"""
    # With no split, the negative row is hidden inside a positive aggregate
    # Hessian. Validation must inspect rows even for Gradient leaf estimation.
    with pytest.raises(CatBoostError, match="(?i)finite|invalid|curvature|hessian|nonnegative|objective"):
        fit(options(path, loss_function=MetalObjective(source), iterations=1, depth=0,
                    leaf_estimation_method=method), Pool(x, y, weight=weights))


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("backtracking", ("No", "AnyImprovement", "Armijo"))
def test_positive_objective_offset_accepts_signed_loss_without_changing_updates(path, backtracking):
    x, y, weights = numeric_problem()
    y = .5 + .2 * y
    source = """
const float residual = target - approx;
return float3(weight * (4.0f - residual * residual), weight * residual, weight);
"""
    pool = Pool(x, y, weight=weights)
    config = options(path, loss_function=MetalObjective(source), iterations=4, depth=0,
                     leaf_estimation_iterations=1, leaf_estimation_backtracking=backtracking)
    custom = fit(config, pool, eval_set=pool, use_best_model=False)
    builtin = fit(config | dict(loss_function="RMSE"), pool, eval_set=pool, use_best_model=False)
    assert_rmse_equivalent(custom, builtin, x)
    # The scalar runtime reports the negative weighted mean objective. Both
    # its initial and final losses are negative here; finite signed losses must
    # reach training/backtracking without a nonnegative-loss restriction.
    assert np.average(y.astype(np.float64) ** 2, weights=weights) - 4 < 0
    signed_loss = np.average((raw(custom, x) - y) ** 2, weights=weights) - 4
    assert signed_loss < 0
    assert custom.tree_count_ == config["iterations"]


@pytest.mark.parametrize("path", PATHS)
def test_custom_simple_leaf_uses_partition_specific_statistics(path):
    y = np.asarray([-.7, -.3, .15, .4, .65, .9, 4.], dtype=np.float32)
    weights = np.asarray([.4, 1.3, 2.1, .7, 1.8, .5, 0.], dtype=np.float32)
    x = np.column_stack((np.arange(len(y)), np.arange(len(y)) ** 2)).astype(np.float32)
    config = options(path, loss_function=MetalObjective(QUARTIC_BODY), iterations=1,
                     depth=0, l2_leaf_reg=.75, learning_rate=.2, bootstrap_type="No",
                     score_function="NewtonCosine", leaf_estimation_method="Simple",
                     leaf_estimation_iterations=1, min_fold_size=2)
    model = fit(config, Pool(x, y, weight=weights))
    target = y.astype(np.float64)
    mass = weights.astype(np.float64)
    gradient = np.sum(mass * target * (target ** 2 + 1))
    curvature = np.sum(mass * (3 * target ** 2 + 1))
    # DocParallel Simple reuses the weak-target statistics selected by its
    # Newton score. FeatureParallel Simple instead performs one Gradient leaf
    # update; Ordered's full-estimation task uses the original row mass too.
    denominator = curvature if path == "plain-doc" else mass.sum()
    expected = config["learning_rate"] * gradient / (denominator + config["l2_leaf_reg"])
    assert abs(curvature - mass.sum()) > 1
    assert model.get_metadata()["metal_backend"] == "METAL"
    assert model.get_all_params()["leaf_estimation_method"] == "Simple"
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), [1])
    np.testing.assert_allclose(model.get_leaf_values(), [expected], atol=3e-6, rtol=5e-6)
    np.testing.assert_allclose(raw(model, x), expected, atol=3e-6, rtol=5e-6)
    np.testing.assert_allclose(model.get_leaf_weights(), [denominator], atol=3e-6, rtol=5e-6)
