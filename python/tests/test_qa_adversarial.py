"""
test_qa_adversarial.py -- Adversarial QA tests to break catboost-mlx.

Written by QA to find bugs, edge cases, and surprising behavior.
Tests are organized by attack surface area.
"""

import copy
import json
import os
import pickle
import tempfile
import threading
import warnings

import numpy as np
import pytest

from catboost_mlx import CatBoostMLX, CatBoostMLXClassifier, CatBoostMLXRegressor, Pool
from catboost_mlx._predict_utils import (
    apply_link,
    compute_leaf_indices,
    evaluate_trees,
    quantize_features,
)
from catboost_mlx._tree_utils import _bin_to_threshold, unfold_oblivious_tree
from catboost_mlx._utils import _to_numpy
from catboost_mlx.core import _array_to_csv, _find_binary, _format_cat_col, _format_numeric_col

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT


def _has_binaries():
    csv_train = os.path.join(BINARY_PATH, "csv_train")
    csv_predict = os.path.join(BINARY_PATH, "csv_predict")
    return os.path.isfile(csv_train) and os.path.isfile(csv_predict)


def _skip_no_binaries():
    if not _has_binaries():
        pytest.skip("Compiled binaries not found")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: Pool -- the data container
# ════════════════════════════════════════════════════════════════════════════


class TestPoolEdgeCases:
    """Try to break Pool with weird inputs."""

    def test_single_sample(self):
        """Pool should handle a single row."""
        pool = Pool(np.array([[1.0, 2.0]]), y=np.array([1.0]))
        assert pool.num_samples == 1
        assert pool.num_features == 2

    def test_single_feature(self):
        """Pool should handle a single column."""
        pool = Pool(np.array([[1.0], [2.0], [3.0]]), y=np.array([1.0, 2.0, 3.0]))
        assert pool.num_features == 1

    def test_1d_input_auto_reshape(self):
        """1D array should be reshaped to (n, 1)."""
        pool = Pool(np.array([1.0, 2.0, 3.0]))
        assert pool.X.shape == (3, 1)

    def test_very_large_values(self):
        """Pool should accept extreme float values without crashing."""
        X = np.array([[1e308, -1e308], [1e-308, -1e-308]])
        pool = Pool(X, y=np.array([0.0, 1.0]))
        assert pool.num_samples == 2

    def test_inf_values(self):
        """Pool should accept inf values -- does it crash later?"""
        X = np.array([[np.inf, -np.inf], [0.0, 1.0]])
        pool = Pool(X, y=np.array([0.0, 1.0]))
        assert pool.num_samples == 2

    def test_all_nan_features(self):
        """Pool should accept all-NaN features."""
        X = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        pool = Pool(X, y=np.array([0.0, 1.0]))
        assert pool.num_samples == 2

    def test_empty_feature_names_list(self):
        """Empty feature_names list with features should raise."""
        with pytest.raises(ValueError, match="feature_names"):
            Pool(np.array([[1.0, 2.0]]), feature_names=[])

    def test_duplicate_feature_names(self):
        """Duplicate feature names should be rejected."""
        with pytest.raises(ValueError, match="Duplicate feature names"):
            Pool(np.array([[1.0, 2.0]]), feature_names=["a", "a"])

    def test_cat_features_negative_index(self):
        """Negative cat_features index should be rejected."""
        with pytest.raises(ValueError, match="out of bounds"):
            Pool(np.array([[1.0, 2.0]]), cat_features=[-1])

    def test_cat_features_float_index(self):
        """Float cat_features index should be truncated to int."""
        pool = Pool(np.array([[1.0, 2.0]]), cat_features=[0.9])
        # _resolve_cat_features does int(cf), so 0.9 -> 0. Silently truncates.
        assert pool.cat_features == [0]

    def test_cat_feature_string_no_feature_names(self):
        """String cat_feature without feature_names should raise."""
        with pytest.raises(ValueError, match="Cannot resolve"):
            Pool(np.array([[1.0, 2.0]]), cat_features=["age"])

    def test_cat_feature_nonexistent_name(self):
        """String cat_feature not in feature_names should raise."""
        with pytest.raises(ValueError, match="not found"):
            Pool(np.array([[1.0, 2.0]]), cat_features=["missing"],
                 feature_names=["a", "b"])

    def test_x_y_shape_mismatch(self):
        """X and y with different sample counts should raise."""
        with pytest.raises(ValueError, match="samples"):
            Pool(np.array([[1.0], [2.0]]), y=np.array([1.0]))

    def test_sample_weight_shape_mismatch(self):
        with pytest.raises(ValueError, match="sample_weight"):
            Pool(np.array([[1.0], [2.0]]), y=np.array([1.0, 2.0]),
                 sample_weight=np.array([1.0]))

    def test_group_id_shape_mismatch(self):
        with pytest.raises(ValueError, match="group_id"):
            Pool(np.array([[1.0], [2.0]]), y=np.array([1.0, 2.0]),
                 group_id=np.array([1]))

    def test_object_dtype_array(self):
        """Object dtype array (mixed types) -- does Pool handle it?"""
        X = np.array([["hello", 1.0], ["world", 2.0]], dtype=object)
        pool = Pool(X, y=np.array([0.0, 1.0]))
        # Accepted, but X is now an object array. This could cause issues
        # downstream when the code tries to do float math on it.
        assert pool.X.dtype == object

    def test_integer_labels(self):
        """Integer y should be accepted."""
        pool = Pool(np.array([[1.0]]), y=np.array([1]))
        assert pool.y.dtype in (np.int64, np.int32, int)

    def test_boolean_labels(self):
        """Boolean y should work."""
        pool = Pool(np.array([[1.0], [2.0]]), y=np.array([True, False]))
        assert pool.num_samples == 2

    def test_repr(self):
        """__repr__ should not crash."""
        pool = Pool(np.array([[1.0, 2.0]]), y=np.array([1.0]))
        repr_str = repr(pool)
        assert "Pool" in repr_str
        assert "1 samples" in repr_str

    def test_len(self):
        pool = Pool(np.array([[1.0], [2.0], [3.0]]))
        assert len(pool) == 3

    def test_zero_columns(self):
        """Zero-column feature matrix -- should this be accepted?"""
        X = np.zeros((5, 0))
        # Pool should either reject this or handle it gracefully
        pool = Pool(X)
        assert pool.num_features == 0


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: Parameter validation edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestParamValidationEdges:
    """Push validation boundaries."""

    def test_iterations_boundary_low(self):
        model = CatBoostMLX(iterations=1)
        model._validate_params()  # should pass

    def test_iterations_boundary_high(self):
        model = CatBoostMLX(iterations=100000)
        model._validate_params()  # should pass

    def test_iterations_zero(self):
        with pytest.raises(ValueError):
            CatBoostMLX(iterations=0)._validate_params()

    def test_iterations_negative(self):
        with pytest.raises(ValueError):
            CatBoostMLX(iterations=-1)._validate_params()

    def test_iterations_float(self):
        """Float iterations -- should be rejected (type check)."""
        with pytest.raises(ValueError):
            CatBoostMLX(iterations=10.5)._validate_params()

    def test_iterations_100001(self):
        with pytest.raises(ValueError):
            CatBoostMLX(iterations=100001)._validate_params()

    def test_depth_boundary_low(self):
        CatBoostMLX(depth=1)._validate_params()

    def test_depth_boundary_high(self):
        CatBoostMLX(depth=16)._validate_params()

    def test_depth_zero(self):
        with pytest.raises(ValueError):
            CatBoostMLX(depth=0)._validate_params()

    def test_depth_17(self):
        with pytest.raises(ValueError):
            CatBoostMLX(depth=17)._validate_params()

    def test_learning_rate_zero(self):
        with pytest.raises(ValueError):
            CatBoostMLX(learning_rate=0.0)._validate_params()

    def test_learning_rate_negative(self):
        with pytest.raises(ValueError):
            CatBoostMLX(learning_rate=-0.1)._validate_params()

    def test_learning_rate_very_small(self):
        """Extremely small lr should be accepted."""
        CatBoostMLX(learning_rate=1e-15)._validate_params()

    def test_learning_rate_very_large(self):
        """Very large lr should be accepted (validation only checks > 0)."""
        CatBoostMLX(learning_rate=1e10)._validate_params()

    def test_l2_reg_lambda_zero(self):
        """l2_reg_lambda=0 should be valid (no regularization)."""
        CatBoostMLX(l2_reg_lambda=0.0)._validate_params()

    def test_l2_reg_lambda_negative(self):
        with pytest.raises(ValueError):
            CatBoostMLX(l2_reg_lambda=-1.0)._validate_params()

    def test_bins_boundary_low(self):
        CatBoostMLX(bins=2)._validate_params()

    def test_bins_boundary_high(self):
        CatBoostMLX(bins=255)._validate_params()

    def test_bins_1(self):
        with pytest.raises(ValueError):
            CatBoostMLX(bins=1)._validate_params()

    def test_bins_256(self):
        with pytest.raises(ValueError):
            CatBoostMLX(bins=256)._validate_params()

    def test_eval_fraction_zero(self):
        CatBoostMLX(eval_fraction=0.0)._validate_params()

    def test_eval_fraction_just_under_1(self):
        CatBoostMLX(eval_fraction=0.999)._validate_params()

    def test_eval_fraction_1(self):
        """eval_fraction=1.0 means ALL data for eval, none for training."""
        with pytest.raises(ValueError):
            CatBoostMLX(eval_fraction=1.0)._validate_params()

    def test_eval_fraction_negative(self):
        with pytest.raises(ValueError):
            CatBoostMLX(eval_fraction=-0.1)._validate_params()

    def test_subsample_zero(self):
        with pytest.raises(ValueError):
            CatBoostMLX(subsample=0.0)._validate_params()

    def test_subsample_over_1(self):
        with pytest.raises(ValueError):
            CatBoostMLX(subsample=1.1)._validate_params()

    def test_colsample_zero(self):
        with pytest.raises(ValueError):
            CatBoostMLX(colsample_bytree=0.0)._validate_params()

    def test_unknown_loss(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            CatBoostMLX(loss="xgboost_style")._validate_params()

    def test_loss_with_non_numeric_param(self):
        with pytest.raises(ValueError, match="numeric"):
            CatBoostMLX(loss="quantile:abc")._validate_params()

    def test_nan_mode_invalid(self):
        with pytest.raises(ValueError, match="nan_mode"):
            CatBoostMLX(nan_mode="drop")._validate_params()

    def test_bootstrap_type_invalid(self):
        with pytest.raises(ValueError, match="bootstrap_type"):
            CatBoostMLX(bootstrap_type="xgboost")._validate_params()

    def test_min_data_in_leaf_zero(self):
        with pytest.raises(ValueError):
            CatBoostMLX(min_data_in_leaf=0)._validate_params()

    def test_monotone_constraints_invalid_value(self):
        with pytest.raises(ValueError, match="monotone"):
            CatBoostMLX(monotone_constraints=[0, 2, 0])._validate_params()

    def test_snapshot_path_null_byte(self):
        with pytest.raises(ValueError, match="snapshot_path"):
            CatBoostMLX(snapshot_path="/tmp/model\x00.json")._validate_params()

    def test_binary_path_null_byte(self):
        with pytest.raises(ValueError, match="binary_path"):
            CatBoostMLX(binary_path="/tmp/bin\x00ary")._validate_params()

    def test_auto_class_weights_invalid(self):
        with pytest.raises(ValueError, match="auto_class_weights"):
            CatBoostMLX(auto_class_weights="invalid")._validate_params()

    def test_auto_class_weights_valid(self):
        CatBoostMLX(auto_class_weights="Balanced")._validate_params()
        CatBoostMLX(auto_class_weights="SqrtBalanced")._validate_params()

    def test_learning_rate_inf(self):
        """Inf learning rate should be rejected."""
        with pytest.raises(ValueError, match="finite"):
            CatBoostMLX(learning_rate=float("inf"))._validate_params()

    def test_learning_rate_nan(self):
        """NaN learning rate should be rejected (IEEE 754 NaN comparison trap)."""
        with pytest.raises(ValueError, match="finite"):
            CatBoostMLX(learning_rate=float("nan"))._validate_params()

    def test_l2_reg_inf(self):
        """Inf l2 regularization should be rejected."""
        with pytest.raises(ValueError, match="finite"):
            CatBoostMLX(l2_reg_lambda=float("inf"))._validate_params()

    def test_l2_reg_nan(self):
        """NaN l2_reg_lambda should be rejected."""
        with pytest.raises(ValueError, match="finite"):
            CatBoostMLX(l2_reg_lambda=float("nan"))._validate_params()

    def test_string_iterations(self):
        """String iterations should fail isinstance check."""
        with pytest.raises(ValueError):
            CatBoostMLX(iterations="100")._validate_params()

    def test_none_iterations(self):
        """None iterations should fail isinstance check."""
        with pytest.raises(ValueError):
            CatBoostMLX(iterations=None)._validate_params()

    def test_bool_iterations(self):
        """bool is subclass of int in Python. True=1 should pass, False=0 should fail."""
        CatBoostMLX(iterations=True)._validate_params()  # True == 1
        with pytest.raises(ValueError):
            CatBoostMLX(iterations=False)._validate_params()  # False == 0


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: CSV serialization edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestCSVSerialization:
    """Attack _array_to_csv and formatting functions."""

    def test_format_numeric_all_nan(self):
        col = np.array([np.nan, np.nan, np.nan])
        result = _format_numeric_col(col)
        assert result == ["", "", ""]

    def test_format_numeric_inf(self):
        col = np.array([np.inf, -np.inf, 0.0])
        result = _format_numeric_col(col)
        assert result[0] == "inf"
        assert result[1] == "-inf"

    def test_format_numeric_very_small(self):
        col = np.array([1e-300])
        result = _format_numeric_col(col)
        assert float(result[0]) == pytest.approx(1e-300)

    def test_format_cat_with_nan(self):
        col = np.array([np.nan, "hello", np.nan], dtype=object)
        result = _format_cat_col(col)
        assert result[0] == ""
        assert result[1] == "hello"
        assert result[2] == ""

    def test_format_cat_with_comma(self):
        """Categorical values with commas -- does CSV quoting handle it?"""
        col = np.array(["hello,world", "foo"], dtype=object)
        result = _format_cat_col(col)
        assert result[0] == "hello,world"
        # The CSV writer should handle quoting, but the formatter doesn't

    def test_format_cat_with_newline(self):
        """Categorical values with newlines."""
        col = np.array(["hello\nworld", "foo"], dtype=object)
        result = _format_cat_col(col)
        assert result[0] == "hello\nworld"

    def test_format_cat_with_quotes(self):
        """Categorical values with double quotes."""
        col = np.array(['he said "hi"', "foo"], dtype=object)
        result = _format_cat_col(col)
        assert 'he said "hi"' in result[0]

    def test_array_to_csv_feature_name_with_comma(self):
        """Feature names with commas should be rejected."""
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="invalid characters"):
                _array_to_csv(path, X, y, feature_names=["a,b", "c"])
        finally:
            os.unlink(path)

    def test_array_to_csv_feature_name_with_newline(self):
        """Feature names with newlines should be rejected."""
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="invalid characters"):
                _array_to_csv(path, X, y, feature_names=["a\nb", "c"])
        finally:
            os.unlink(path)

    def test_array_to_csv_feature_name_with_null(self):
        """Feature names with null bytes should be rejected."""
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="invalid characters"):
                _array_to_csv(path, X, y, feature_names=["a\x00b", "c"])
        finally:
            os.unlink(path)

    def test_array_to_csv_predict_mode(self):
        """Predict mode (no y) should work."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            target_col, group_col, weight_col = _array_to_csv(path, X)
            assert target_col == -1
            assert group_col == -1
            assert weight_col == -1
        finally:
            os.unlink(path)

    def test_array_to_csv_with_group_and_weight(self):
        """CSV with group_id and sample_weight prepended columns."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0.0, 1.0])
        gid = np.array([0, 0])
        sw = np.array([1.0, 2.0])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            target_col, group_col, weight_col = _array_to_csv(
                path, X, y, group_id=gid, sample_weight=sw
            )
            assert group_col == 0
            assert weight_col == 1
            assert target_col == 4  # group, weight, f0, f1, target
        finally:
            os.unlink(path)

    def test_array_to_csv_empty_matrix(self):
        """Empty feature matrix -- does it produce a valid CSV?"""
        X = np.zeros((0, 3))
        y = np.zeros(0)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            _array_to_csv(path, X, y)
            with open(path) as f:
                lines = f.readlines()
            # Should have at least a header line
            assert len(lines) >= 1
        finally:
            os.unlink(path)

    def test_array_to_csv_unicode_feature_names(self):
        """Unicode feature names."""
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            _array_to_csv(path, X, y, feature_names=["价格", "尺寸"])
            with open(path) as f:
                header = f.readline()
            assert "价格" in header
        finally:
            os.unlink(path)

    def test_array_to_csv_very_long_feature_name(self):
        """Very long feature names."""
        X = np.array([[1.0]])
        y = np.array([1.0])
        long_name = "x" * 10000
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            _array_to_csv(path, X, y, feature_names=[long_name])
            # Should not crash
        finally:
            os.unlink(path)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: _predict_utils edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestPredictUtils:
    """Break the pure-Python prediction engine."""

    def test_quantize_empty_borders(self):
        """Feature with no borders -- everything goes to one bin."""
        X = np.array([[1.0], [2.0], [3.0]])
        features = [{"borders": [], "has_nan": False}]
        result = quantize_features(X, features)
        assert result.shape == (3, 1)
        # With no borders and no NaN offset, all values should be bin 0
        np.testing.assert_array_equal(result[:, 0], [0, 0, 0])

    def test_quantize_all_nan(self):
        """All-NaN column should map to bin 0."""
        X = np.array([[np.nan], [np.nan]])
        features = [{"borders": [0.5, 1.5], "has_nan": True}]
        result = quantize_features(X, features)
        np.testing.assert_array_equal(result[:, 0], [0, 0])

    def test_quantize_nan_no_has_nan_flag(self):
        """NaN values when has_nan=False -- bin 0 regardless."""
        X = np.array([[np.nan], [1.0]])
        features = [{"borders": [0.5], "has_nan": False}]
        result = quantize_features(X, features)
        assert result[0, 0] == 0  # NaN always maps to bin 0

    def test_quantize_categorical_unknown(self):
        """Unknown categorical value should map to bin 0."""
        X = np.array([["unseen"]], dtype=object)
        features = [{"is_categorical": True, "cat_hash_map": {"hello": 1, "world": 2}}]
        result = quantize_features(X, features, cat_features=[0])
        assert result[0, 0] == 0

    def test_quantize_bin_overflow(self):
        """Values that would produce bin > 255 should be clipped."""
        borders = list(range(260))  # 260 borders -> up to bin 261
        X = np.array([[300.0]])
        features = [{"borders": borders, "has_nan": False}]
        result = quantize_features(X, features)
        assert result[0, 0] <= 255

    def test_compute_leaf_indices_depth_0(self):
        """Depth-0 tree (stump) has only one leaf."""
        binned = np.array([[0], [1], [2]], dtype=np.uint8)
        tree = {"depth": 0, "splits": []}
        result = compute_leaf_indices(binned, tree)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_compute_leaf_indices_one_hot(self):
        """One-hot split: go right when bval == threshold."""
        binned = np.array([[1], [2], [1]], dtype=np.uint8)
        tree = {
            "depth": 1,
            "splits": [{"feature_idx": 0, "bin_threshold": 1, "is_one_hot": True}],
        }
        result = compute_leaf_indices(binned, tree)
        # bit 0 = (bval == 1): [True, False, True] = [1, 0, 1]
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_evaluate_trees_empty(self):
        """Zero trees should return zeros."""
        binned = np.array([[0, 0]], dtype=np.uint8)
        result = evaluate_trees(binned, [], approx_dim=1)
        np.testing.assert_array_equal(result, [0.0])

    def test_evaluate_trees_n_trees_limit(self):
        """n_trees parameter should limit number of trees used."""
        binned = np.array([[0]], dtype=np.uint8)
        trees = [
            {"depth": 0, "splits": [], "leaf_values": [1.0]},
            {"depth": 0, "splits": [], "leaf_values": [2.0]},
            {"depth": 0, "splits": [], "leaf_values": [3.0]},
        ]
        result = evaluate_trees(binned, trees, approx_dim=1, n_trees=2)
        # Only first 2 trees: 1.0 + 2.0 = 3.0
        assert result[0] == pytest.approx(3.0)

    def test_apply_link_rmse(self):
        """RMSE: identity link -- output == input."""
        cursor = np.array([1.0, 2.0, 3.0])
        result = apply_link(cursor, "rmse")
        np.testing.assert_array_equal(result["prediction"], cursor)

    def test_apply_link_logloss(self):
        """Logloss: sigmoid link."""
        cursor = np.array([0.0])
        result = apply_link(cursor, "logloss")
        assert result["probability"][0] == pytest.approx(0.5)
        assert result["predicted_class"][0] == 0

    def test_apply_link_logloss_large_positive(self):
        """Large positive cursor should give probability ~1."""
        cursor = np.array([100.0])
        result = apply_link(cursor, "logloss")
        assert result["probability"][0] == pytest.approx(1.0, abs=1e-10)
        assert result["predicted_class"][0] == 1

    def test_apply_link_logloss_large_negative(self):
        """Large negative cursor should give probability ~0."""
        cursor = np.array([-100.0])
        result = apply_link(cursor, "logloss")
        assert result["probability"][0] == pytest.approx(0.0, abs=1e-10)
        assert result["predicted_class"][0] == 0

    def test_apply_link_multiclass_stability(self):
        """Multiclass softmax with extreme values should not overflow."""
        cursor = np.array([[1000.0, -1000.0]])
        result = apply_link(cursor, "multiclass", num_classes=3)
        # Should not have NaN or inf
        for k in range(3):
            assert not np.isnan(result[f"prob_class_{k}"][0])
            assert not np.isinf(result[f"prob_class_{k}"][0])
        # Probabilities should sum to 1
        total = sum(result[f"prob_class_{k}"][0] for k in range(3))
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_apply_link_multiclass_all_zeros(self):
        """All-zero cursor: equal probabilities."""
        cursor = np.array([[0.0, 0.0]])
        result = apply_link(cursor, "multiclass", num_classes=3)
        # 3 classes, implicit last class also 0 -> all equal
        for k in range(3):
            assert result[f"prob_class_{k}"][0] == pytest.approx(1.0 / 3, abs=1e-6)

    def test_apply_link_poisson(self):
        """Poisson: exp link."""
        cursor = np.array([0.0, 1.0])
        result = apply_link(cursor, "poisson")
        np.testing.assert_allclose(result["prediction"], [1.0, np.e], rtol=1e-6)

    def test_apply_link_poisson_overflow(self):
        """Poisson with very large cursor could overflow."""
        cursor = np.array([1000.0])
        result = apply_link(cursor, "poisson")
        # exp(1000) is inf in float64
        assert np.isinf(result["prediction"][0])
        # BUG?: No protection against overflow in Poisson/Tweedie link

    def test_apply_link_unknown_loss(self):
        """Unknown loss type should default to identity."""
        cursor = np.array([42.0])
        result = apply_link(cursor, "unknown_loss_xyz")
        assert result["prediction"][0] == 42.0


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: _tree_utils edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestTreeUtils:
    """Break tree unfolding and threshold conversion."""

    def test_unfold_depth_0(self):
        """Depth-0 tree should produce a single leaf."""
        tree = {"depth": 0, "splits": [], "leaf_values": [1.0]}
        features = []
        result = unfold_oblivious_tree(tree, features, approx_dim=1)
        assert len(result) == 1
        assert result[0]["type"] == "leaf"
        assert result[0]["values"] == [1.0]

    def test_unfold_depth_1(self):
        """Depth-1 tree: 1 branch + 2 leaves."""
        tree = {
            "depth": 1,
            "splits": [{"feature_idx": 0, "bin_threshold": 1}],
            "leaf_values": [10.0, 20.0],
        }
        features = [{"borders": [0.5, 1.5], "has_nan": False}]
        result = unfold_oblivious_tree(tree, features, approx_dim=1)
        assert len(result) == 3  # 1 branch + 2 leaves
        branches = [n for n in result if n["type"] == "branch"]
        leaves = [n for n in result if n["type"] == "leaf"]
        assert len(branches) == 1
        assert len(leaves) == 2

    def test_unfold_multiclass(self):
        """Multiclass tree with approx_dim > 1."""
        tree = {
            "depth": 1,
            "splits": [{"feature_idx": 0, "bin_threshold": 1}],
            "leaf_values": [1.0, 2.0, 3.0, 4.0],  # 2 leaves x 2 dims
        }
        features = [{"borders": [0.5], "has_nan": False}]
        result = unfold_oblivious_tree(tree, features, approx_dim=2)
        leaves = [n for n in result if n["type"] == "leaf"]
        assert len(leaves) == 2
        assert len(leaves[0]["values"]) == 2
        assert len(leaves[1]["values"]) == 2

    def test_bin_to_threshold_out_of_range(self):
        """Border index out of range should return float(bin_threshold)."""
        features = [{"borders": [0.5], "has_nan": False}]
        # bin_threshold=10 -> border_idx=10, but only 1 border
        result = _bin_to_threshold(0, 10, False, features)
        assert result == 10.0

    def test_bin_to_threshold_with_nan_offset(self):
        """NaN offset shifts border index."""
        features = [{"borders": [0.5, 1.5], "has_nan": True}]
        # bin_threshold=1, nan_offset=1 -> border_idx=0 -> borders[0]=0.5
        result = _bin_to_threshold(0, 1, False, features)
        assert result == 0.5

    def test_bin_to_threshold_one_hot(self):
        """One-hot: returns bin_threshold directly as float."""
        features = [{"borders": [0.5]}]
        result = _bin_to_threshold(0, 5, True, features)
        assert result == 5.0

    def test_bin_to_threshold_empty_borders(self):
        """Empty borders with bin_threshold > 0 -> fallback."""
        features = [{"borders": []}]
        result = _bin_to_threshold(0, 1, False, features)
        assert result == 1.0


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: _find_binary edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestFindBinary:
    """Try to break binary discovery."""

    def test_nonexistent_explicit_path(self):
        with pytest.raises(FileNotFoundError):
            _find_binary("csv_train", binary_path="/nonexistent/path")

    def test_explicit_directory_without_binary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                _find_binary("csv_train", binary_path=tmpdir)

    def test_non_executable_file(self):
        """Binary exists but is not executable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            binary = os.path.join(tmpdir, "csv_train")
            with open(binary, "w") as f:
                f.write("not a real binary")
            os.chmod(binary, 0o644)  # remove execute permission
            with pytest.raises(PermissionError, match="not executable"):
                _find_binary("csv_train", binary_path=tmpdir)

    def test_find_nonexistent_binary_no_path(self):
        """Looking for a binary that doesn't exist anywhere."""
        with pytest.raises(FileNotFoundError):
            _find_binary("totally_nonexistent_binary_xyz123")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: Model state & lifecycle
# ════════════════════════════════════════════════════════════════════════════


class TestModelLifecycle:
    """Break the model by abusing state transitions."""

    def test_predict_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.zeros((5, 2)))

    def test_predict_proba_before_fit(self):
        model = CatBoostMLXClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.zeros((5, 2)))

    def test_save_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save_model("/tmp/nope.json")

    def test_tree_count_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.tree_count_

    def test_feature_names_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.feature_names_

    def test_feature_importances_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.feature_importances_

    def test_get_trees_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_trees()

    def test_get_model_info_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_model_info()

    def test_apply_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.apply(np.zeros((5, 2)))

    def test_staged_predict_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            list(model.staged_predict(np.zeros((5, 2))))

    def test_export_coreml_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.export_coreml("/tmp/nope.mlmodel")

    def test_export_onnx_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.export_onnx("/tmp/nope.onnx")

    def test_get_shap_values_before_fit(self):
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_shap_values(np.zeros((5, 2)))

    def test_load_model_missing_keys(self):
        """Model JSON with missing required keys."""
        model = CatBoostMLX()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"trees": [], "features": []}, f)  # missing model_info
            path = f.name
        try:
            with pytest.raises(ValueError, match="missing required keys"):
                model.load_model(path)
        finally:
            os.unlink(path)

    def test_load_model_empty_json(self):
        """Empty JSON object."""
        model = CatBoostMLX()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({}, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="missing required keys"):
                model.load_model(path)
        finally:
            os.unlink(path)

    def test_repr_unfitted(self):
        model = CatBoostMLX()
        r = repr(model)
        assert "not fitted" in r

    def test_repr_fitted_no_crash(self):
        """repr should work on a mock-fitted model."""
        model = CatBoostMLX()
        model._is_fitted = True
        model._model_data = {"trees": [], "features": [], "model_info": {}}
        r = repr(model)
        assert "fitted" in r

    def test_get_feature_importance_no_data(self):
        """No feature importance data should return empty dict."""
        model = CatBoostMLX()
        assert model.get_feature_importance() == {}

    def test_plot_feature_importance_no_data(self, capsys):
        """plot_feature_importance with no data should print message."""
        model = CatBoostMLX()
        model.plot_feature_importance()
        captured = capsys.readouterr()
        assert "No feature importance" in captured.out


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8: Pickle/serialization edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestSerializationEdges:
    """Break pickle roundtrips."""

    def test_pickle_unfitted_roundtrip(self):
        model = CatBoostMLXRegressor(iterations=42, depth=7)
        data = pickle.dumps(model)
        model2 = pickle.loads(data)
        assert model2.iterations == 42
        assert model2.depth == 7
        assert not model2._is_fitted

    def test_pickle_preserves_params(self):
        model = CatBoostMLX(
            iterations=50, depth=8, learning_rate=0.05,
            loss="mae", bins=128, cat_features=[1, 3],
            subsample=0.8, random_seed=123
        )
        data = pickle.dumps(model)
        model2 = pickle.loads(data)
        assert model2.iterations == 50
        assert model2.depth == 8
        assert model2.learning_rate == 0.05
        assert model2.loss == "mae"
        assert model2.bins == 128
        assert model2.cat_features == [1, 3]
        assert model2.subsample == 0.8
        assert model2.random_seed == 123

    def test_pickle_model_path_excluded(self):
        """_model_path should not survive pickling."""
        model = CatBoostMLX()
        model._model_path = "/some/path"
        data = pickle.dumps(model)
        model2 = pickle.loads(data)
        assert model2._model_path is None

    def test_pickle_json_cache_excluded(self):
        """_model_json_cache should not survive pickling."""
        model = CatBoostMLX()
        model._model_json_cache = '{"big": "json"}'
        data = pickle.dumps(model)
        model2 = pickle.loads(data)
        assert model2._model_json_cache is None

    def test_deepcopy_unfitted(self):
        model = CatBoostMLXRegressor(iterations=10, depth=3)
        model2 = copy.deepcopy(model)
        model2.iterations = 99
        assert model.iterations == 10  # original unchanged

    def test_deepcopy_mock_fitted(self):
        """Deep copy of a fitted model should produce independent copy."""
        model = CatBoostMLX()
        model._is_fitted = True
        model._model_data = {"trees": [{"leaf": 1}], "features": [], "model_info": {}}
        model2 = copy.deepcopy(model)
        model2._model_data["trees"][0]["leaf"] = 999
        # Original should be unchanged
        assert model._model_data["trees"][0]["leaf"] == 1


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9: sklearn compatibility edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestSklearnCompat:
    """Test sklearn protocol adherence."""

    def test_get_params_returns_all(self):
        model = CatBoostMLX()
        params = model.get_params()
        # Should include all __init__ params
        assert "iterations" in params
        assert "depth" in params
        assert "learning_rate" in params
        assert "loss" in params
        assert "bins" in params
        assert "binary_path" in params

    def test_set_params(self):
        model = CatBoostMLX()
        model.set_params(iterations=999, depth=12)
        assert model.iterations == 999
        assert model.depth == 12

    def test_set_params_returns_self(self):
        model = CatBoostMLX()
        result = model.set_params(iterations=5)
        assert result is model

    def test_regressor_get_params_all(self):
        """Regressor should expose all parent params, not just 'loss'."""
        model = CatBoostMLXRegressor()
        params = model.get_params()
        assert "iterations" in params
        assert "depth" in params
        assert "learning_rate" in params

    def test_classifier_get_params_all(self):
        """Classifier should expose all parent params."""
        model = CatBoostMLXClassifier()
        params = model.get_params()
        assert "iterations" in params
        assert "depth" in params

    def test_regressor_default_loss(self):
        model = CatBoostMLXRegressor()
        assert model.loss == "rmse"

    def test_classifier_default_loss(self):
        model = CatBoostMLXClassifier()
        assert model.loss == "auto"

    def test_sklearn_is_fitted_false(self):
        model = CatBoostMLX()
        assert model.__sklearn_is_fitted__() is False

    def test_sklearn_is_fitted_true(self):
        model = CatBoostMLX()
        model._is_fitted = True
        assert model.__sklearn_is_fitted__() is True

    def test_estimator_type_regressor(self):
        model = CatBoostMLXRegressor()
        assert model._estimator_type == "regressor"

    def test_estimator_type_classifier(self):
        model = CatBoostMLXClassifier()
        assert model._estimator_type == "classifier"

    def test_clone_regressor(self):
        """sklearn clone should preserve all hyperparameters."""
        try:
            from sklearn.base import clone
        except ImportError:
            pytest.skip("sklearn not installed")
        model = CatBoostMLXRegressor(
            iterations=50, depth=8, learning_rate=0.05, binary_path="/tmp"
        )
        cloned = clone(model)
        assert cloned.iterations == 50
        assert cloned.depth == 8
        assert cloned.learning_rate == 0.05
        assert cloned.binary_path == "/tmp"
        assert not cloned._is_fitted

    def test_clone_classifier(self):
        try:
            from sklearn.base import clone
        except ImportError:
            pytest.skip("sklearn not installed")
        model = CatBoostMLXClassifier(
            iterations=30, depth=4, loss="logloss"
        )
        cloned = clone(model)
        assert cloned.iterations == 30
        assert cloned.loss == "logloss"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10: fit() input validation edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestFitInputValidation:
    """Break fit() with bad inputs (doesn't need binaries for validation checks)."""

    def test_fit_3d_input(self):
        """3D array should be rejected."""
        model = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.zeros((5, 3, 2))
        y = np.zeros(5)
        with pytest.raises(ValueError, match="2-dimensional"):
            model.fit(X, y)

    def test_fit_y_wrong_length(self):
        model = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.zeros((10, 3))
        y = np.zeros(5)
        with pytest.raises(ValueError, match="samples"):
            model.fit(X, y)

    def test_fit_y_none(self):
        model = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="y is required"):
            model.fit(np.zeros((10, 3)))

    def test_fit_pool_without_y(self):
        """Pool without y and no separate y should raise."""
        model = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        pool = Pool(np.zeros((10, 3)))
        with pytest.raises(ValueError, match="y is required"):
            model.fit(pool)

    def test_fit_sample_weight_wrong_length(self):
        model = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.zeros((10, 3))
        y = np.zeros(10)
        sw = np.ones(5)
        with pytest.raises(ValueError, match="sample_weight"):
            model.fit(X, y, sample_weight=sw)

    def test_fit_group_id_wrong_length(self):
        model = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.zeros((10, 3))
        y = np.zeros(10)
        gid = np.zeros(5)
        with pytest.raises(ValueError, match="group_id"):
            model.fit(X, y, group_id=gid)

    def test_fit_feature_names_wrong_length(self):
        model = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.zeros((10, 3))
        y = np.zeros(10)
        with pytest.raises(ValueError, match="feature_names"):
            model.fit(X, y, feature_names=["a", "b"])

    def test_fit_cat_features_out_of_bounds(self):
        model = CatBoostMLXRegressor(cat_features=[99], binary_path=BINARY_PATH)
        X = np.zeros((10, 3))
        y = np.zeros(10)
        with pytest.raises(ValueError, match="out of bounds"):
            model.fit(X, y)

    def test_fit_monotone_wrong_length(self):
        model = CatBoostMLXRegressor(
            monotone_constraints=[0, 1], binary_path=BINARY_PATH
        )
        X = np.zeros((10, 3))
        y = np.zeros(10)
        with pytest.raises(ValueError, match="monotone_constraints"):
            model.fit(X, y)

    def test_fit_eval_set_wrong_format(self):
        """eval_set must be tuple of (X, y)."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        with pytest.raises(ValueError, match="eval_set"):
            model.fit(X, y, eval_set="invalid")

    def test_fit_eval_set_wrong_features(self):
        """eval_set with different feature count should raise."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        X_val = np.random.rand(10, 5)  # wrong feature count
        y_val = np.random.rand(10)
        with pytest.raises(ValueError, match="features"):
            model.fit(X, y, eval_set=(X_val, y_val))

    def test_fit_eval_set_and_eval_fraction_conflict(self):
        """eval_set and eval_fraction together should raise."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(
            iterations=5, eval_fraction=0.2, binary_path=BINARY_PATH
        )
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        X_val = np.random.rand(5, 3)
        y_val = np.random.rand(5)
        with pytest.raises(ValueError, match="mutually exclusive"):
            model.fit(X, y, eval_set=(X_val, y_val))

    def test_fit_1d_x_auto_reshape(self):
        """1D X should be auto-reshaped to (n, 1)."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        X = np.random.rand(20)
        y = np.random.rand(20)
        model.fit(X, y)
        assert model._is_fitted


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11: Numerical edge cases with actual training (needs binaries)
# ════════════════════════════════════════════════════════════════════════════


class TestNumericalEdgeCases:
    """Push numerical limits during training and prediction."""

    def test_constant_target(self):
        """All-constant y: should emit a warning about slow convergence."""
        _skip_no_binaries()
        X = np.random.rand(30, 3)
        y = np.full(30, 5.0)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X, y)
            assert any("zero variance" in str(warning.message) for warning in w)

    def test_constant_features(self):
        """All-constant features should be caught before calling C++ binary."""
        X = np.ones((30, 3))
        y = np.random.rand(30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="constant"):
            model.fit(X, y)

    def test_single_sample_training(self):
        """Training on a single sample."""
        _skip_no_binaries()
        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([1.0])
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        # This might fail or produce a degenerate model
        try:
            model.fit(X, y)
            preds = model.predict(X)
            assert preds.shape == (1,)
        except (RuntimeError, ValueError):
            pass  # Acceptable to fail

    def test_two_samples_training(self):
        """Training on two samples."""
        _skip_no_binaries()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0.0, 1.0])
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (2,)

    def test_many_features_few_samples(self):
        """More features than samples (p >> n)."""
        _skip_no_binaries()
        X = np.random.rand(10, 100)
        y = np.random.rand(10)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (10,)

    def test_binary_classification_all_same_class(self):
        """All labels are the same class."""
        _skip_no_binaries()
        X = np.random.rand(30, 3)
        y = np.zeros(30)  # all class 0
        model = CatBoostMLXClassifier(iterations=10, depth=3, binary_path=BINARY_PATH)
        try:
            model.fit(X, y)
            preds = model.predict(X)
            # Should predict all zeros
            np.testing.assert_array_equal(preds, 0)
        except (RuntimeError, ValueError):
            pass  # Some implementations reject single-class training

    def test_negative_targets_regression(self):
        """Negative target values."""
        _skip_no_binaries()
        X = np.random.rand(30, 3)
        y = np.random.rand(30) * -10
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(preds < 0) or np.all(np.isfinite(preds))

    def test_very_large_targets(self):
        """Very large target values."""
        _skip_no_binaries()
        X = np.random.rand(30, 3)
        y = np.random.rand(30) * 1e10
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))

    def test_nan_in_training_features(self):
        """NaN in features with nan_mode=min."""
        _skip_no_binaries()
        X = np.random.rand(30, 3)
        X[0, 0] = np.nan
        X[5, 1] = np.nan
        y = np.random.rand(30)
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, nan_mode="min", binary_path=BINARY_PATH
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))

    def test_all_nan_column(self):
        """Entire column is NaN."""
        _skip_no_binaries()
        X = np.random.rand(30, 3)
        X[:, 1] = np.nan
        y = np.random.rand(30)
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, nan_mode="min", binary_path=BINARY_PATH
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,)

    def test_predict_different_shape_than_train(self):
        """Predicting with wrong number of features should fail."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X_train = rng.rand(30, 3)
        y_train = rng.rand(30)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(X_train, y_train)
        X_test = rng.rand(10, 5)  # wrong feature count
        # Should fail at CSV writing or C++ binary level
        with pytest.raises((RuntimeError, ValueError)):
            model.predict(X_test)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 12: Classifier-specific edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestClassifierEdges:
    """Break the classifier subclass."""

    def test_predict_proba_on_regressor(self):
        """predict_proba on a regression model should fail."""
        _skip_no_binaries()
        X = np.random.rand(30, 3)
        y = np.random.rand(30)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(X, y)
        with pytest.raises(ValueError, match="not supported"):
            model.predict_proba(X)

    def test_staged_predict_proba_on_regressor(self):
        """staged_predict_proba on regression loss should fail."""
        _skip_no_binaries()
        X = np.random.rand(30, 3)
        y = np.random.rand(30)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(X, y)
        with pytest.raises(ValueError, match="not supported"):
            list(model.staged_predict_proba(X))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 13: Model mutation after fit
# ════════════════════════════════════════════════════════════════════════════


class TestModelMutation:
    """What happens when you mutate model internals after fitting?"""

    def test_mutate_model_data_after_fit(self):
        """Directly mutating _model_data should affect predictions."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        preds_before = model.predict(X).copy()

        # Corrupt the model by zeroing all leaf values
        for tree in model._model_data["trees"]:
            tree["leaf_values"] = [0.0] * len(tree["leaf_values"])
        model._model_json_cache = None  # force re-serialization

        preds_after = model.predict(X)
        # Predictions should change (all zeros now)
        assert not np.allclose(preds_before, preds_after)

    def test_change_params_after_fit(self):
        """Changing hyperparameters after fit shouldn't affect predict."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        preds_before = model.predict(X).copy()

        model.iterations = 999  # doesn't affect already-trained model
        model.depth = 1
        preds_after = model.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 14: Cross-validate edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestCrossValidateEdges:
    """Break cross_validate."""

    def test_cv_1_fold(self):
        """1-fold CV doesn't make sense but might not be validated."""
        _skip_no_binaries()
        X = np.random.rand(30, 3)
        y = np.random.rand(30)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        # 1 fold = use all data for both train and test? Or error?
        try:
            result = model.cross_validate(X, y, n_folds=1)
        except (RuntimeError, ValueError):
            pass  # Acceptable to reject

    def test_cv_more_folds_than_samples(self):
        """More folds than samples."""
        _skip_no_binaries()
        X = np.random.rand(3, 2)
        y = np.random.rand(3)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        try:
            result = model.cross_validate(X, y, n_folds=10)
        except (RuntimeError, ValueError):
            pass  # Acceptable to reject

    def test_cv_0_folds(self):
        """0-fold CV."""
        _skip_no_binaries()
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        try:
            result = model.cross_validate(X, y, n_folds=0)
            # Should probably fail
        except (RuntimeError, ValueError):
            pass

    def test_cv_validates_params(self):
        """cross_validate should validate params."""
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model = CatBoostMLXRegressor(iterations=5, depth=0, binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="depth"):
            model.cross_validate(X, y, n_folds=3)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 15: _to_numpy edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestToNumpy:
    """Break the _to_numpy utility."""

    def test_numpy_passthrough(self):
        arr = np.array([1, 2, 3])
        result = _to_numpy(arr)
        assert result is arr

    def test_list_input(self):
        result = _to_numpy([1, 2, 3])
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_nested_list(self):
        result = _to_numpy([[1, 2], [3, 4]])
        assert result.shape == (2, 2)

    def test_tuple_input(self):
        result = _to_numpy((1, 2, 3))
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_scalar_input(self):
        result = _to_numpy(42)
        assert result == 42

    def test_empty_list(self):
        result = _to_numpy([])
        assert len(result) == 0

    def test_mixed_types_list(self):
        """Mixed int and float."""
        result = _to_numpy([1, 2.5, 3])
        assert result.dtype == float


# ════════════════════════════════════════════════════════════════════════════
# SECTION 16: Integration -- full train/predict cycles with edge data
# ════════════════════════════════════════════════════════════════════════════


class TestIntegrationEdgeCases:
    """Full train-predict cycles with tricky data."""

    def test_train_predict_with_pool(self):
        """Train with Pool, predict with Pool."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 40)
        pool_train = Pool(X, y=y)
        pool_test = Pool(X[:10])
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(pool_train)
        preds = model.predict(pool_test)
        assert preds.shape == (10,)

    def test_train_with_eval_set(self):
        """Training with external eval_set."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 40)
        X_val = rng.rand(10, 3)
        y_val = X_val[:, 0] * 2 + rng.normal(0, 0.1, 10)
        model = CatBoostMLXRegressor(
            iterations=20, depth=3, early_stopping_rounds=5,
            binary_path=BINARY_PATH
        )
        model.fit(X, y, eval_set=(X_val, y_val))
        assert model._is_fitted

    def test_train_with_eval_set_pool(self):
        """eval_set as Pool object."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 40)
        eval_pool = Pool(rng.rand(10, 3), y=rng.rand(10))
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y, eval_set=eval_pool)
        assert model._is_fitted

    def test_save_load_predict_consistency(self):
        """Save -> Load -> Predict should give identical results."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        preds_original = model.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            model2 = CatBoostMLXRegressor(binary_path=BINARY_PATH)
            model2.load_model(path)
            preds_loaded = model2.predict(X)
            np.testing.assert_allclose(preds_original, preds_loaded, atol=1e-10)
        finally:
            os.unlink(path)

    def test_staged_predict_final_matches_predict(self):
        """Final staged_predict output should match predict."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        preds = model.predict(X)
        staged = list(model.staged_predict(X))
        np.testing.assert_allclose(staged[-1], preds, atol=1e-4)

    def test_apply_output_shape(self):
        """apply() should return (n_samples, n_trees) int32 array."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3)
        y = rng.rand(20)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        leaves = model.apply(X)
        assert leaves.shape == (20, 10)
        assert leaves.dtype == np.int32
        # Leaf indices should be in [0, 2^depth)
        assert np.all(leaves >= 0)
        assert np.all(leaves < 2 ** 3)

    def test_feature_importance_sum(self):
        """feature_importances_ should sum to 1."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, 50)
        model = CatBoostMLXRegressor(iterations=20, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        fi = model.feature_importances_
        if fi.sum() > 0:
            assert fi.sum() == pytest.approx(1.0, abs=1e-6)

    def test_train_loss_history_length(self):
        """train_loss_history should have one entry per iteration."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)
        model = CatBoostMLXRegressor(
            iterations=15, depth=3, verbose=False, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        history = model.train_loss_history
        # Should have entries (might be empty if verbose output wasn't captured)
        # But if present, should have <= iterations entries
        assert len(history) <= 15

    def test_model_info_keys(self):
        """get_model_info() should return expected keys."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)
        model = CatBoostMLXRegressor(iterations=5, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        info = model.get_model_info()
        assert "num_features" in info

    def test_refit_overwrites_model(self):
        """Fitting twice should replace the first model."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X1 = rng.rand(30, 3)
        y1 = rng.rand(30) * 100

        X2 = rng.rand(30, 3)
        y2 = rng.rand(30) * -100

        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X1, y1)
        preds1 = model.predict(X1).copy()

        model.fit(X2, y2)
        preds2 = model.predict(X2)

        # The two sets of predictions should be very different
        assert not np.allclose(preds1, preds2, atol=1.0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 17: Concurrency / thread safety
# ════════════════════════════════════════════════════════════════════════════


class TestConcurrency:
    """Check if concurrent predict calls are safe."""

    def test_concurrent_predict(self):
        """Multiple threads calling predict simultaneously."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)

        results = {}
        errors = []

        def predict_in_thread(tid):
            try:
                preds = model.predict(X)
                results[tid] = preds
            except Exception as e:
                errors.append((tid, e))

        threads = [threading.Thread(target=predict_in_thread, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in threads: {errors}"
        # All threads should get the same result
        for tid in results:
            np.testing.assert_allclose(results[tid], results[0], atol=1e-10)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 18: Auto class weights
# ════════════════════════════════════════════════════════════════════════════


class TestAutoClassWeights:
    """Test automatic class weight computation."""

    def test_balanced_weights_binary(self):
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3)
        y = np.array([0.0] * 30 + [1.0] * 10)  # imbalanced
        model = CatBoostMLXClassifier(
            iterations=10, depth=3, auto_class_weights="Balanced",
            binary_path=BINARY_PATH
        )
        model.fit(X, y)
        assert model._is_fitted

    def test_sqrtbalanced_weights(self):
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3)
        y = np.array([0.0] * 30 + [1.0] * 10)
        model = CatBoostMLXClassifier(
            iterations=10, depth=3, auto_class_weights="SqrtBalanced",
            binary_path=BINARY_PATH
        )
        model.fit(X, y)
        assert model._is_fitted


# ════════════════════════════════════════════════════════════════════════════
# SECTION 19: Misc bugs and gotchas
# ════════════════════════════════════════════════════════════════════════════


class TestMiscBugsAndGotchas:
    """Miscellaneous potential issues."""

    def test_loss_type_case_sensitivity(self):
        """Loss type 'RMSE' vs 'rmse' -- should be case-insensitive."""
        # _validate_params lowercases: loss_base = self.loss.split(":")[0].lower()
        # But the actual binary might be case-sensitive
        model = CatBoostMLX(loss="RMSE")
        model._validate_params()  # Should pass due to .lower()

    def test_loss_with_multiple_colons(self):
        """Loss like 'quantile:0.5:extra' -- how is this handled?"""
        # split(":", 1)[1] = "0.5:extra", float() will fail
        with pytest.raises(ValueError, match="numeric"):
            CatBoostMLX(loss="quantile:0.5:extra")._validate_params()

    def test_predict_proba_binary_sums_to_1(self):
        """predict_proba columns should sum to 1 for binary classification."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3)
        y = (X[:, 0] > 0.5).astype(float)
        model = CatBoostMLXClassifier(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        proba = model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_eval_fraction_high(self):
        """eval_fraction=0.999 means almost no training data."""
        _skip_no_binaries()
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        model = CatBoostMLXRegressor(
            iterations=5, depth=2, eval_fraction=0.99,
            binary_path=BINARY_PATH
        )
        # Only ~1 sample for training. Should either work or raise a clear error.
        try:
            model.fit(X, y)
        except (RuntimeError, ValueError):
            pass  # Acceptable

    def test_subsample_very_small(self):
        """subsample=0.01 means very few samples per tree."""
        _skip_no_binaries()
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, subsample=0.01,
            binary_path=BINARY_PATH
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_max_depth_tree(self):
        """Maximum depth=16 tree."""
        _skip_no_binaries()
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        model = CatBoostMLXRegressor(
            iterations=5, depth=16, binary_path=BINARY_PATH
        )
        # depth=16 means 2^16 = 65536 leaves per tree. Memory-heavy.
        try:
            model.fit(X, y)
        except (RuntimeError, MemoryError):
            pass  # Acceptable to fail with memory error

    def test_predict_on_zero_rows(self):
        """Predicting on zero rows."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X_train = rng.rand(30, 3)
        y_train = rng.rand(30)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(X_train, y_train)
        X_empty = np.zeros((0, 3))
        # Should either return empty array or raise a clear error
        try:
            preds = model.predict(X_empty)
            assert preds.shape == (0,)
        except (RuntimeError, ValueError):
            pass  # Also acceptable

    def test_get_loss_type_auto(self):
        """_get_loss_type with auto loss and no model data."""
        model = CatBoostMLX(loss="auto")
        assert model._get_loss_type() == "auto"

    def test_get_loss_type_from_model_data(self):
        """_get_loss_type should prefer model_data over self.loss."""
        model = CatBoostMLX(loss="rmse")
        model._model_data = {"model_info": {"loss_type": "MAE"}}
        assert model._get_loss_type() == "mae"

    def test_cat_features_from_pool_to_model(self):
        """cat_features from Pool should propagate to model when model has none."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = np.column_stack([
            rng.choice(["a", "b", "c"], 30),
            rng.rand(30)
        ])
        y = rng.rand(30)
        pool = Pool(X, y=y, cat_features=[0])
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        assert model.cat_features is None
        model.fit(pool)
        assert model.cat_features == [0]
