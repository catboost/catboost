"""Native Langevin source schedules, GPU-only models and exact snapshot recovery."""
import os
import platform
import struct

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool

from test_native_training_modes import numeric_problem, compound_problem
from test_native_compound_ctrs import options as compound_options

pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1"
    or platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="requires the rebuilt native Langevin Metal adapter",
)

MODES = ("PlainDP", "PlainFP", "OrderedFP")
TAG = 0x4D4C4731
COMBINATION = "Combination:loss0=YetiRank:permutations=5;weight0=0.02;loss1=RMSE;weight1=2"


@pytest.fixture(autouse=True)
def prohibit_cpu_fitting(monkeypatch):
    original = CatBoost._fit
    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CatBoost, "_fit", checked)


def options(mode="PlainDP", **extra):
    return dict(
        task_type="GPU", loss_function="RMSE", boosting_type="Ordered" if mode == "OrderedFP" else "Plain",
        data_partition="DocParallel" if mode == "PlainDP" else "FeatureParallel", grow_policy="SymmetricTree",
        iterations=4, depth=2, learning_rate=.1, l2_leaf_reg=2, random_seed=713,
        border_count=12, bootstrap_type="No", random_strength=0, score_function="Cosine",
        leaf_estimation_method="Newton", leaf_estimation_iterations=1,
        leaf_estimation_backtracking="No", boost_from_average=False,
        permutation_count=1, has_time=True, min_fold_size=4, fold_len_multiplier=1.7,
        max_ctr_complexity=1, langevin=True, diffusion_temperature=1000.,
        verbose=False, allow_writing_files=False, use_best_model=False,
    ) | extra


def numeric(loss="RMSE"):
    x, y, po = numeric_problem("PairLogit" if loss == "PairLogit" else "QuerySoftMax")
    if loss == "RMSE":
        # Trivial grouping exercises ordinary numeric Ordered fold geometry.
        po.pop("group_id")
        y = (x[:, 0] - .3 * x[:, 1] + .2 * x[:, 2] ** 2).astype(np.float32)
    return Pool(x, y, **po)


def fit(config, pool):
    return CatBoost(config).fit(pool)


def check_same(first, second, pool):
    np.testing.assert_array_equal(first.get_tree_leaf_counts(), second.get_tree_leaf_counts())
    np.testing.assert_array_equal(first.get_leaf_values(), second.get_leaf_values())
    np.testing.assert_array_equal(first.get_leaf_weights(), second.get_leaf_weights())
    np.testing.assert_array_equal(first.predict(pool, prediction_type="RawFormulaVal", task_type="GPU"),
                                  second.predict(pool, prediction_type="RawFormulaVal", task_type="GPU"))
    np.testing.assert_array_equal(first._object._get_metal_training_cursor(),
                                  second._object._get_metal_training_cursor())
    assert first.get_metadata()["metal_langevin_host_draw_count"] == second.get_metadata()["metal_langevin_host_draw_count"]


def tail(raw):
    at = raw.rfind(struct.pack("<I", TAG))
    assert at >= 0 and len(raw) == at + 17
    return at, struct.unpack_from("<QIB", raw, at + 4)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("leaf_iterations", (1, 3))
def test_numeric_exact_source_host_draw_count(mode, count, leaf_iterations):
    pool = numeric()
    config = options(mode, permutation_count=count, has_time=count == 1,
                     leaf_estimation_iterations=leaf_iterations)
    first, second = fit(config, pool), fit(config, pool)
    check_same(first, second, pool)
    assert first.get_metadata()["metal_backend"] == "METAL"
    depths = np.log2(first.get_tree_leaf_counts()).astype(int)
    assert np.all(2 ** depths == first.get_tree_leaf_counts())
    # No target RNG or dependent feature packs. Symmetric CUDA gets the one
    # shared 65,537-draw GPU seed cache, even for bootstrap=No and T=0.
    chooser = config["iterations"] if mode != "PlainDP" and count > 2 else 0
    search = np.minimum(depths + 1, config["depth"]).sum()
    leaves = config["iterations"] * (2 if leaf_iterations == 1 else 2 + 2 * leaf_iterations)
    expected = (mode == "PlainDP") + 65537 + chooser + search + leaves
    assert int(first.get_metadata()["metal_langevin_host_draw_count"]) == expected
    assert first.get_metadata()["metal_langevin_weak_rng"] == (
        "no_weak_noise" if mode == "PlainFP" else "metal_item_iteration_domains_v1")


@pytest.mark.parametrize("mode", MODES)
def test_positive_temperature_does_not_enable_gpu_langevin(mode):
    pool = numeric()
    default = options(mode)
    default.pop("langevin"); default.pop("diffusion_temperature")
    first, second = fit(default, pool), fit(default | {"diffusion_temperature": 1000.}, pool)
    np.testing.assert_array_equal(first.get_leaf_values(), second.get_leaf_values())
    assert "metal_langevin_host_rng" not in first.get_metadata()
    assert "metal_langevin_host_rng" not in second.get_metadata()
    assert not second.get_all_params().get("langevin", False)


@pytest.mark.parametrize("mode", MODES)
def test_zero_temperature_keeps_source_callback_draws(mode):
    pool = numeric()
    zero = fit(options(mode, diffusion_temperature=0), pool)
    implicit = options(mode); implicit.pop("diffusion_temperature")
    default = fit(implicit, pool)
    check_same(zero, default, pool)
    assert int(zero.get_metadata()["metal_langevin_host_draw_count"]) >= 65537 + 8
    disabled = fit(options(mode, langevin=False), pool)
    np.testing.assert_allclose(zero.predict(pool, task_type="GPU"), disabled.predict(pool, task_type="GPU"),
                               rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("loss", ("QueryRMSE", "QuerySoftMax", "PairLogit", "YetiRank:permutations=5", COMBINATION))
def test_registered_query_oracles_share_langevin_stream(mode, loss, tmp_path):
    pool = numeric("PairLogit" if loss == "PairLogit" else "QuerySoftMax")
    config = options(mode, loss_function=loss, permutation_count=4, has_time=False,
                     leaf_estimation_iterations=3, iterations=3, diffusion_temperature=100000.)
    first, second = fit(config, pool), fit(config, pool)
    check_same(first, second, pool)
    assert np.isfinite(first.get_leaf_values()).all()
    raw = first.predict(pool, prediction_type="RawFormulaVal", task_type="GPU")
    assert np.isfinite(raw).all() and np.any(raw != 0)
    path = tmp_path / "model.cbm"
    first.save_model(str(path))
    restored = CatBoost(); restored.load_model(str(path))
    np.testing.assert_array_equal(restored.predict(pool, prediction_type="RawFormulaVal", task_type="GPU"), raw)


@pytest.mark.parametrize("loss", ("PairLogitPairwise", "QueryCrossEntropy", "YetiRankPairwise"))
@pytest.mark.parametrize("temperature", (0., 1000.))
def test_cuda_full_matrix_oracles_reject_langevin(loss, temperature):
    pool = numeric("PairLogit" if loss == "PairLogitPairwise" else "QuerySoftMax")
    with pytest.raises(CatBoostError, match="Langevin is not supported"):
        fit(options(loss_function=loss, diffusion_temperature=temperature, depth=1, score_function="NewtonCosine"), pool)


@pytest.mark.parametrize("mode", ("PlainDP", "PlainFP"))
def test_exact_estimator_has_no_leaf_noise_callbacks(mode):
    pool = numeric()
    config = options(mode, loss_function="Quantile", leaf_estimation_method="Exact",
                     leaf_estimation_iterations=1, depth=0)
    first, second = fit(config, pool), fit(config | {"diffusion_temperature": 2.}, pool)
    check_same(first, second, pool)
    assert int(first.get_metadata()["metal_langevin_host_draw_count"]) == 65537 + (mode == "PlainDP")


@pytest.mark.parametrize("mode", MODES)
def test_simple_uses_source_weak_or_leaf_noise_path(mode):
    pool = numeric()
    config = options(mode, leaf_estimation_method="Simple", depth=0)
    first, second = fit(config, pool), fit(config | {"diffusion_temperature": 2.}, pool)
    assert not np.array_equal(first.get_leaf_values(), second.get_leaf_values())
    expected = 65537 + (mode == "PlainDP") + (0 if mode == "PlainDP" else 2 * config["iterations"])
    assert int(first.get_metadata()["metal_langevin_host_draw_count"]) == expected


def snapshot_options(config, directory, iterations):
    return config | dict(iterations=iterations, save_snapshot=True, snapshot_interval=0,
                         snapshot_file="langevin.snapshot", train_dir=str(directory), allow_writing_files=True)


@pytest.mark.parametrize("mode", ("PlainFP", "OrderedFP"))
@pytest.mark.parametrize("loss", ("RMSE", COMBINATION))
def test_p4_dynamic_ctr_meta_langevin_snapshot_resume(mode, loss, tmp_path):
    x, y, po, _, _, _ = compound_problem("QuerySoftMax")
    pool = Pool(x, y, **po)
    config = compound_options(boosting="Ordered" if mode == "OrderedFP" else "Plain", count=4)
    config.update(options(mode, loss_function=loss, permutation_count=4, has_time=False,
                          max_ctr_complexity=2, depth=3, leaf_estimation_iterations=3,
                          leaf_estimation_backtracking="AnyImprovement", diffusion_temperature=1000000.,
                          score_function="Cosine" if mode == "OrderedFP" else "NewtonL2",
                          meta_l2_exponent=.6, meta_l2_frequency=.47))
    direct = fit(config | {"iterations": 6}, pool)
    partial = fit(snapshot_options(config, tmp_path, 2), pool)
    path = tmp_path / "langevin.snapshot"
    _, saved = tail(path.read_bytes())
    assert saved[1:] == (2, 1)
    assert saved[0] == int(partial.get_metadata()["metal_langevin_host_draw_count"])
    resumed = fit(snapshot_options(config, tmp_path, 6), pool)
    assert resumed._object._get_metal_training_info()["resumed_iterations"] == 2
    check_same(direct, resumed, pool)
    assert int(resumed.get_metadata()["metal_tree_ctr_features"]) > 0
    complete_bytes = path.read_bytes()
    complete = fit(snapshot_options(config, tmp_path, 6), pool)
    assert complete._object._get_metal_training_info()["resumed_iterations"] == 6
    check_same(direct, complete, pool)
    assert path.read_bytes() == complete_bytes


def test_rejected_trials_are_saved_as_actual_host_draws(tmp_path):
    # At the exact optimum every nonzero noisy direction initially worsens
    # RMSE. AnyImprovement must halve repeatedly before float32 loss equality.
    pool = Pool(np.linspace(-1, 1, 12, dtype=np.float32)[:, None], np.zeros(12, np.float32))
    config = options(depth=0, allow_const_label=True, leaf_estimation_iterations=3,
                     leaf_estimation_backtracking="AnyImprovement", l2_leaf_reg=10,
                     diffusion_temperature=1000., iterations=2)
    direct = fit(config, pool)
    partial = fit(snapshot_options(config, tmp_path, 1), pool)
    assert int(partial.get_metadata()["metal_langevin_host_draw_count"]) > 65538 + 2 + 2 * 3
    resumed = fit(snapshot_options(config, tmp_path, 2), pool)
    check_same(direct, resumed, pool)
    _, state = tail((tmp_path / "langevin.snapshot").read_bytes())
    assert state[0] == int(direct.get_metadata()["metal_langevin_host_draw_count"])


@pytest.mark.parametrize("field", ("missing", "draw_count", "completed", "cache"))
def test_snapshot_random_corruption_rejected_without_replacing_file(field, tmp_path):
    pool = numeric(); config = snapshot_options(options(depth=0), tmp_path, 2)
    fit(config, pool)
    path = tmp_path / "langevin.snapshot"
    raw = bytearray(path.read_bytes()); offset, _ = tail(raw)
    if field == "missing": del raw[offset:]
    elif field == "draw_count": struct.pack_into("<Q", raw, offset + 4, 2 ** 64 - 1)
    elif field == "completed": struct.pack_into("<I", raw, offset + 12, 3)
    else: raw[offset + 16] = 0
    path.write_bytes(raw)
    with pytest.raises(CatBoostError, match="Langevin"):
        fit(config, pool)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("changed", ({"diffusion_temperature": 2000.}, {"langevin": False},
                                     {"leaf_estimation_backtracking": "AnyImprovement"}))
def test_snapshot_changed_noise_options_are_rejected(changed, tmp_path):
    pool = numeric(); config = snapshot_options(options(depth=0, leaf_estimation_iterations=3), tmp_path, 2)
    fit(config, pool)
    path = tmp_path / "langevin.snapshot"; raw = path.read_bytes()
    with pytest.raises(CatBoostError, match="Langevin|parameters differ"):
        fit(config | changed, pool)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
@pytest.mark.parametrize("loss", ("RMSE", "YetiRank:permutations=5"))
def test_greedy_p4_langevin_snapshot_keeps_shared_stream(policy, loss, tmp_path):
    x, y, po, _, _, _ = compound_problem("QuerySoftMax")
    pool = Pool(x, y, **po)
    config = compound_options(count=4, complexity=1)
    config.update(options(loss_function=loss, permutation_count=4, has_time=False,
                          grow_policy=policy, depth=2, leaf_estimation_iterations=3,
                          diffusion_temperature=100000., iterations=5))
    if policy == "Lossguide": config["max_leaves"] = 4
    direct = fit(config, pool)
    assert int(direct.get_metadata()["metal_permutations"]) == 4
    partial = fit(snapshot_options(config, tmp_path, 2), pool)
    _, saved = tail((tmp_path / "langevin.snapshot").read_bytes())
    assert saved[1:] == (2, 0)  # Greedy has no weak noise; No bootstrap has no GPU seed cache.
    resumed = fit(snapshot_options(config, tmp_path, 5), pool)
    assert resumed._object._get_metal_training_info()["resumed_iterations"] == 2
    check_same(direct, resumed, pool)
    completed = fit(snapshot_options(config, tmp_path, 5), pool)
    check_same(direct, completed, pool)
    assert completed.get_metadata()["metal_langevin_weak_rng"] == "no_weak_noise"


@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
@pytest.mark.parametrize("method,loss", (("Simple", "RMSE"), ("Exact", "Quantile")))
def test_greedy_simple_and_exact_have_no_langevin_noise_calls(policy, method, loss):
    pool = numeric()
    config = options(grow_policy=policy, loss_function=loss, leaf_estimation_method=method,
                     leaf_estimation_iterations=1, depth=2)
    if policy == "Lossguide": config["max_leaves"] = 4
    first, second = fit(config, pool), fit(config | {"diffusion_temperature": 2.}, pool)
    check_same(first, second, pool)
    assert int(first.get_metadata()["metal_langevin_host_draw_count"]) < 65537


@pytest.mark.parametrize("loss", ("YetiRank:permutations=5", COMBINATION))
def test_changed_langevin_option_is_checked_before_stochastic_payload_type(loss, tmp_path):
    pool = numeric("QuerySoftMax")
    config = snapshot_options(options(loss_function=loss, depth=0, iterations=2), tmp_path, 2)
    fit(config, pool)
    path = tmp_path / "langevin.snapshot"; raw = path.read_bytes()
    with pytest.raises(CatBoostError, match="parameters differ"):
        fit(config | {"langevin": False}, pool)
    assert path.read_bytes() == raw
