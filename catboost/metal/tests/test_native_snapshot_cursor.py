"""Native online-cursor snapshot regressions; every fit executes Metal."""

import os
import platform
import struct

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, CatBoostRegressor, Pool


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1"
    or platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="requires the rebuilt native Metal snapshot cursor adapter",
)

BEST_V1 = 0x4D424C31
BEST_V2 = 0x4D424C32


@pytest.fixture(autouse=True)
def require_gpu_training(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU", "Cursor acceptance cannot fit CPU models"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def options(tmp_path, boosting, **extra):
    return dict(
        task_type="GPU", loss_function="RMSE", boosting_type=boosting,
        data_partition="FeatureParallel", permutation_count=4, iterations=7,
        depth=2, learning_rate=.15, l2_leaf_reg=2, random_seed=1947,
        bootstrap_type="No", random_strength=0, score_function="Cosine",
        border_count=12, leaf_estimation_iterations=1, boost_from_average=False,
        max_ctr_complexity=1, min_fold_size=4, fold_len_multiplier=1.7,
        has_time=False, verbose=False, use_best_model=False,
        save_snapshot=True, snapshot_interval=0, snapshot_file="cursor.snapshot",
        allow_writing_files=True, train_dir=str(tmp_path),
    ) | extra


def estimator(config):
    # Native-only options such as permutation_count use set_params rather
    # than the narrower public regressor constructor signature.
    return CatBoostRegressor().set_params(**config)


def cursor(model):
    return model._object._get_metal_training_cursor()


def info(model):
    return model._object._get_metal_training_info()


def current_cursor_pool(order, quantized):
    rng = np.random.default_rng(319)
    numeric = rng.normal(size=(80, 3)).astype(np.float32)
    x = np.column_stack(([f"category-{i % 5}" for i in range(80)], numeric)).astype(object)
    y = (.6 * numeric[:, 0] - .25 * numeric[:, 1] + .1 * numeric[:, 2] ** 2).astype(np.float32)
    weights = (.5 + np.arange(80) % 7 / 5).astype(np.float32)
    kwargs = {"timestamp": rng.permutation(80).astype(np.uint64)} if order == "timestamp" else {}
    result = Pool(x, y, cat_features=[0], weight=weights, **kwargs)
    if order == "slice":
        # Include duplicates and a nonmonotonic subset of a larger Pool. The
        # public cursor must use positions in this Pool, not storage indices.
        rows = np.r_[rng.permutation(80)[:47], [11, 3, 11, 27, 3]]
        result = result.slice(rows)
    if quantized:
        result.quantize(border_count=12)
    return result


@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
@pytest.mark.parametrize("order", ("shuffle", "timestamp", "slice"))
@pytest.mark.parametrize("quantized", (False, True))
def test_completed_snapshot_restores_current_cursor_in_original_pool_order(tmp_path, boosting, order, quantized):
    pool = current_cursor_pool(order, quantized)
    config = options(tmp_path, boosting, one_hot_max_size=8, has_time=order == "timestamp")
    first = estimator(config).fit(pool)
    expected = first.predict(pool, prediction_type="RawFormulaVal", task_type="GPU")
    np.testing.assert_allclose(cursor(first), expected, rtol=6e-6, atol=3e-6)
    assert info(first)["resumed_iterations"] == 0
    original = (tmp_path / "cursor.snapshot").read_bytes()

    restored = estimator(config).fit(pool)
    assert info(restored)["resumed_iterations"] == config["iterations"]
    np.testing.assert_array_equal(restored.get_leaf_values(), first.get_leaf_values())
    np.testing.assert_array_equal(cursor(restored), cursor(first))
    np.testing.assert_allclose(cursor(restored), expected, rtol=6e-6, atol=3e-6)
    assert (tmp_path / "cursor.snapshot").read_bytes() == original


def retained_best_fit(tmp_path, boosting):
    x = np.linspace(-2, 2, 80, dtype=np.float32)[:, None]
    y = np.where(x[:, 0] > 0, 1., -1.).astype(np.float32)
    learn, validation = Pool(x, y), Pool(x, -y)
    # One border gives a repeated signed split, so every positive training
    # update strictly worsens validation and the retained best is tree one.
    config = options(tmp_path, boosting, depth=1, border_count=1,
                     use_best_model=True, best_model_min_trees=1)
    first = estimator(config).fit(learn, eval_set=validation)
    assert first.get_best_iteration() == 0 and first.tree_count_ == 1
    np.testing.assert_allclose(cursor(first), first.predict(learn, task_type="GPU"), rtol=6e-6, atol=3e-6)
    return first, learn, validation, config


def indexed_tail(raw, rows, iterations):
    offset = raw.rfind(struct.pack("<I", BEST_V2))
    assert offset >= 0
    iteration, count = struct.unpack_from("<iI", raw, offset + 4)
    assert 0 <= iteration < iterations and count == rows
    assert offset + 12 + count * 4 == len(raw)
    return offset, iteration


@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
def test_changed_minimum_retention_cannot_reuse_another_iterations_best_cursor(tmp_path, boosting):
    first, learn, validation, config = retained_best_fit(tmp_path, boosting)
    original = (tmp_path / "cursor.snapshot").read_bytes()
    _, iteration = indexed_tail(original, learn.num_row(), config["iterations"])
    assert iteration == 0
    same = estimator(config).fit(learn, eval_set=validation)
    assert info(same)["resumed_iterations"] == config["iterations"]
    np.testing.assert_array_equal(cursor(same), cursor(first))

    changed = estimator(config | {"best_model_min_trees": 3}).fit(learn, eval_set=validation)
    assert changed.tree_count_ == 3 and changed.get_best_iteration() == 0
    assert info(changed)["resumed_iterations"] == config["iterations"]
    with pytest.raises(CatBoostError, match="online training cursor is unavailable"):
        cursor(changed)
    direct = estimator(config | dict(
        best_model_min_trees=3, save_snapshot=False, train_dir=str(tmp_path / "direct"),
    )).fit(learn, eval_set=validation)
    np.testing.assert_array_equal(changed.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_allclose(changed.predict(learn, task_type="GPU"), cursor(direct), rtol=6e-6, atol=3e-6)
    assert (tmp_path / "cursor.snapshot").read_bytes() == original


@pytest.mark.parametrize("boosting", ("Plain", "Ordered"))
@pytest.mark.parametrize("legacy_tail", ("absent", "unindexed_v1"))
def test_legacy_v6_best_model_recovery_does_not_fabricate_online_cursor(tmp_path, boosting, legacy_tail):
    first, learn, validation, config = retained_best_fit(tmp_path, boosting)
    path = tmp_path / "cursor.snapshot"
    raw = path.read_bytes()
    offset, _ = indexed_tail(raw, learn.num_row(), config["iterations"])
    old = raw[:offset]
    if legacy_tail == "unindexed_v1":
        old += struct.pack("<I", BEST_V1) + raw[offset + 8:]
    path.write_bytes(old)

    restored = estimator(config).fit(learn, eval_set=validation)
    assert info(restored)["resumed_iterations"] == config["iterations"]
    np.testing.assert_array_equal(restored.get_leaf_values(), first.get_leaf_values())
    np.testing.assert_array_equal(restored.predict(learn, task_type="GPU"), first.predict(learn, task_type="GPU"))
    with pytest.raises(CatBoostError, match="online training cursor is unavailable"):
        cursor(restored)
    assert path.read_bytes() == old


@pytest.mark.parametrize("corruption,match", (
    ("tag", "Unknown Metal best-learn snapshot payload"),
    ("negative_iteration", "invalid iteration"),
    ("past_iteration", "invalid iteration"),
    ("dimension", "inconsistent dimensions"),
    ("nonfinite", "nonfinite"),
))
def test_optional_best_cursor_corruption_is_rejected_without_replacing_snapshot(tmp_path, corruption, match):
    _, learn, validation, config = retained_best_fit(tmp_path, "Plain")
    path = tmp_path / "cursor.snapshot"
    broken = bytearray(path.read_bytes())
    offset, _ = indexed_tail(broken, learn.num_row(), config["iterations"])
    if corruption == "tag":
        struct.pack_into("<I", broken, offset, 0x4D424C7F)
    elif corruption in ("negative_iteration", "past_iteration"):
        struct.pack_into("<i", broken, offset + 4, -1 if corruption == "negative_iteration" else config["iterations"])
    elif corruption == "dimension":
        struct.pack_into("<I", broken, offset + 8, learn.num_row() - 1)
        del broken[-4:]
    else:
        struct.pack_into("<f", broken, offset + 12, float("nan"))
    path.write_bytes(broken)
    with pytest.raises(CatBoostError, match=match):
        estimator(config).fit(learn, eval_set=validation)
    assert path.read_bytes() == broken
