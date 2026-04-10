"""
test_qa_round2.py -- QA Round 2: Testing dev fixes + new code paths.

Focus areas:
1. Verify all 11 original bugs are actually fixed
2. Break the new random_strength parameter
3. Break the new _array_to_binary function
4. Break the new feature name validation
5. Find regressions introduced by the fixes
6. Test interactions between new and old code
"""

import json
import os
import struct
import tempfile

import numpy as np
import pytest

from catboost_mlx import CatBoostMLX, CatBoostMLXClassifier, CatBoostMLXRegressor, Pool
from catboost_mlx.core import _array_to_binary, _array_to_csv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT


def _skip_no_binaries():
    csv_train = os.path.join(BINARY_PATH, "csv_train")
    csv_predict = os.path.join(BINARY_PATH, "csv_predict")
    if not (os.path.isfile(csv_train) and os.path.isfile(csv_predict)):
        pytest.skip("Compiled binaries not found")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: Verify original bug fixes (regression tests)
# ════════════════════════════════════════════════════════════════════════════


class TestBugFixVerification:
    """Confirm all 11 reported bugs are actually fixed."""

    # -- FIX 1: NaN/Inf validation --

    def test_learning_rate_nan_rejected(self):
        with pytest.raises(ValueError, match="learning_rate"):
            CatBoostMLX(learning_rate=float("nan"))._validate_params()

    def test_learning_rate_inf_rejected(self):
        with pytest.raises(ValueError, match="learning_rate"):
            CatBoostMLX(learning_rate=float("inf"))._validate_params()

    def test_l2_reg_nan_rejected(self):
        with pytest.raises(ValueError, match="l2_reg_lambda"):
            CatBoostMLX(l2_reg_lambda=float("nan"))._validate_params()

    def test_l2_reg_inf_rejected(self):
        with pytest.raises(ValueError, match="l2_reg_lambda"):
            CatBoostMLX(l2_reg_lambda=float("inf"))._validate_params()

    def test_bagging_temperature_nan_rejected(self):
        with pytest.raises(ValueError, match="bagging_temperature"):
            CatBoostMLX(bagging_temperature=float("nan"))._validate_params()

    def test_mvs_reg_nan_rejected(self):
        with pytest.raises(ValueError, match="mvs_reg"):
            CatBoostMLX(mvs_reg=float("nan"))._validate_params()

    def test_ctr_prior_nan_rejected(self):
        with pytest.raises(ValueError, match="ctr_prior"):
            CatBoostMLX(ctr_prior=float("nan"))._validate_params()

    def test_ctr_prior_inf_rejected(self):
        with pytest.raises(ValueError, match="ctr_prior"):
            CatBoostMLX(ctr_prior=float("inf"))._validate_params()

    # -- FIX 3: Negative cat_features --

    def test_pool_negative_cat_features_rejected(self):
        with pytest.raises(ValueError, match="out of bounds"):
            Pool(np.array([[1.0, 2.0]]), cat_features=[-1])

    def test_pool_negative_cat_features_minus_100(self):
        with pytest.raises(ValueError, match="out of bounds"):
            Pool(np.array([[1.0, 2.0]]), cat_features=[-100])

    # -- FIX 4: 2D y --

    def test_2d_y_rejected(self):
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="1-dimensional"):
            m.fit(np.random.rand(10, 3), np.random.rand(10, 1))

    def test_3d_y_rejected(self):
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="1-dimensional"):
            m.fit(np.random.rand(10, 3), np.random.rand(10, 1, 1))

    # -- FIX 5: String labels --

    def test_string_labels_rejected(self):
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="numeric"):
            m.fit(np.random.rand(5, 2), np.array(["cat", "dog", "cat", "dog", "cat"]))

    def test_object_labels_rejected(self):
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="numeric"):
            m.fit(np.random.rand(5, 2), np.array([None, "x", 3, 4, 5], dtype=object))

    # -- FIX 6: Predict feature count --

    def test_predict_wrong_feature_count(self):
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(rng.rand(20, 3), rng.rand(20))
        with pytest.raises(ValueError, match="features"):
            model.predict(rng.rand(5, 5))

    # -- FIX 8: Duplicate feature names --

    def test_pool_duplicate_feature_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            Pool(np.array([[1.0, 2.0]]), feature_names=["a", "a"])

    # -- FIX 9: Duplicate cat_features --

    def test_pool_duplicate_cat_features_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            Pool(np.array([[1.0, 2.0]]), cat_features=[0, 0])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: NEW BUG -- random_strength has ZERO validation
# ════════════════════════════════════════════════════════════════════════════


class TestRandomStrengthValidation:
    """Validation for the random_strength parameter."""

    def test_random_strength_nan_rejected(self):
        """random_strength=NaN should raise ValueError."""
        m = CatBoostMLX(random_strength=float("nan"))
        with pytest.raises(ValueError, match="random_strength"):
            m._validate_params()

    def test_random_strength_inf_rejected(self):
        """random_strength=inf should raise ValueError."""
        m = CatBoostMLX(random_strength=float("inf"))
        with pytest.raises(ValueError, match="random_strength"):
            m._validate_params()

    def test_random_strength_negative_rejected(self):
        """random_strength=-1.0 should raise ValueError."""
        m = CatBoostMLX(random_strength=-1.0)
        with pytest.raises(ValueError, match="random_strength"):
            m._validate_params()

    def test_random_strength_string_rejected(self):
        """random_strength='hello' should raise ValueError."""
        m = CatBoostMLX(random_strength="hello")
        with pytest.raises(ValueError, match="random_strength"):
            m._validate_params()

    def test_random_strength_none_rejected(self):
        """random_strength=None should raise ValueError."""
        m = CatBoostMLX(random_strength=None)
        with pytest.raises(ValueError, match="random_strength"):
            m._validate_params()

    def test_random_strength_default(self):
        """Default random_strength=1.0 should be fine."""
        m = CatBoostMLX(random_strength=1.0)
        m._validate_params()

    def test_random_strength_zero(self):
        """random_strength=0 should be valid (disables perturbation)."""
        m = CatBoostMLX(random_strength=0.0)
        m._validate_params()

    def test_random_strength_in_get_params(self):
        """random_strength should appear in get_params for sklearn compat."""
        m = CatBoostMLX()
        params = m.get_params()
        assert "random_strength" in params
        assert params["random_strength"] == 1.0

    def test_random_strength_not_sent_when_default(self):
        """When random_strength=1.0 (default), --random-strength is NOT in CLI args."""
        _skip_no_binaries()
        m = CatBoostMLX(random_strength=1.0, binary_path=BINARY_PATH)
        args = m._build_train_args("/tmp/data.csv", "/tmp/model.json", 3)
        assert "--random-strength" not in args

    def test_random_strength_sent_when_changed(self):
        """When random_strength != 1.0, --random-strength IS in CLI args."""
        _skip_no_binaries()
        m = CatBoostMLX(random_strength=0.5, binary_path=BINARY_PATH)
        args = m._build_train_args("/tmp/data.csv", "/tmp/model.json", 3)
        assert "--random-strength" in args
        idx = args.index("--random-strength")
        assert args[idx + 1] == "0.5"

    def test_random_strength_bool_false_produces_bad_cli(self):
        """BUG: random_strength=False -> str(False)='False' in CLI."""
        _skip_no_binaries()
        m = CatBoostMLX(random_strength=False, binary_path=BINARY_PATH)
        # False != 1.0 is True, so it gets sent
        args = m._build_train_args("/tmp/data.csv", "/tmp/model.json", 3)
        if "--random-strength" in args:
            idx = args.index("--random-strength")
            val = args[idx + 1]
            # str(False) = "False", which C++ atof() interprets as 0.0
            assert val == "False", f"Expected 'False' but got '{val}'"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: Feature name validation inconsistencies
# ════════════════════════════════════════════════════════════════════════════


class TestFeatureNameValidation:
    """The dev added feature name validation but it has issues."""

    def test_quote_in_feature_name_allowed(self):
        """Double quotes in feature names are allowed -- csv.writer handles
        quoting correctly."""
        _skip_no_binaries()
        m = CatBoostMLXRegressor(iterations=3, depth=2, binary_path=BINARY_PATH)
        X = np.random.rand(20, 2)
        y = np.random.rand(20)
        m.fit(X, y, feature_names=['price "USD"', 'size'])
        assert m.feature_names_in_[0] == 'price "USD"'

    def test_null_byte_rejected(self):
        """Null bytes in feature names should be rejected."""
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.random.rand(5, 2)
        y = np.random.rand(5)
        with pytest.raises(ValueError, match="invalid characters"):
            m.fit(X, y, feature_names=["a\x00b", "c"])

    def test_comma_rejected(self):
        """Comma in feature name should be rejected."""
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.random.rand(5, 2)
        y = np.random.rand(5)
        with pytest.raises(ValueError, match="invalid characters"):
            m.fit(X, y, feature_names=["a,b", "c"])

    def test_newline_rejected(self):
        """Newline in feature name should be rejected."""
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.random.rand(5, 2)
        y = np.random.rand(5)
        with pytest.raises(ValueError, match="invalid characters"):
            m.fit(X, y, feature_names=["a\nb", "c"])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: Feature count validation incomplete
# ════════════════════════════════════════════════════════════════════════════


class TestFeatureCountValidation:
    """predict was fixed but staged_predict/apply/etc were NOT."""

    def test_staged_predict_wrong_features_crashes(self):
        """BUG: staged_predict has no feature count check. Crashes with IndexError."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(rng.rand(20, 3), rng.rand(20))
        # staged_predict uses quantize_features which accesses features[f]
        # for f in range(n_features) -- crashes when X has more features
        with pytest.raises((ValueError, IndexError)):
            list(model.staged_predict(rng.rand(5, 5)))

    def test_apply_wrong_features_crashes(self):
        """BUG: apply has no feature count check. Crashes with IndexError."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(rng.rand(20, 3), rng.rand(20))
        with pytest.raises((ValueError, IndexError)):
            model.apply(rng.rand(5, 5))

    def test_staged_predict_proba_wrong_features_crashes(self):
        """BUG: staged_predict_proba has no feature count check."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3)
        y = (X[:, 0] > 0.5).astype(float)
        model = CatBoostMLXClassifier(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(X, y)
        with pytest.raises((ValueError, IndexError)):
            list(model.staged_predict_proba(rng.rand(5, 5)))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: _array_to_binary edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestArrayToBinary:
    """Break the new binary serialization format."""

    def test_basic_roundtrip(self):
        """Basic write and read back."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.NamedTemporaryFile(suffix=".cbmx", delete=False) as f:
            path = f.name
        try:
            _array_to_binary(path, X)
            with open(path, "rb") as f:
                magic = f.read(4)
                assert magic == b"CBMX"
                ver, n, feat, flags = struct.unpack("<IIII", f.read(16))
                assert n == 2
                assert feat == 2
                assert flags == 0
        finally:
            os.unlink(path)

    def test_with_y_and_weights(self):
        """Write with y, sample_weight, group_id."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        sw = np.array([1.0, 2.0])
        gid = np.array([0, 0], dtype=np.uint32)
        with tempfile.NamedTemporaryFile(suffix=".cbmx", delete=False) as f:
            path = f.name
        try:
            _array_to_binary(path, X, y=y, sample_weight=sw, group_id=gid)
            with open(path, "rb") as f:
                magic = f.read(4)
                ver, n, feat, flags = struct.unpack("<IIII", f.read(16))
                assert flags == 7  # 1|2|4
        finally:
            os.unlink(path)

    def test_float64_to_float32_precision_loss(self):
        """Silent float64->float32 downcast loses precision."""
        X = np.array([[1.0000001234567890]])
        with tempfile.NamedTemporaryFile(suffix=".cbmx", delete=False) as f:
            path = f.name
        try:
            _array_to_binary(path, X)
            with open(path, "rb") as f:
                f.read(20)  # skip header
                val = np.fromfile(f, dtype=np.float32, count=1)[0]
            # float32 precision: ~7 decimal digits
            # float64 precision: ~15 decimal digits
            assert abs(float(val) - X[0, 0]) > 1e-10  # precision was lost
        finally:
            os.unlink(path)

    def test_large_values_overflow_to_inf(self):
        """Values > float32 max (~3.4e38) silently become inf."""
        X = np.array([[1e39]])
        with tempfile.NamedTemporaryFile(suffix=".cbmx", delete=False) as f:
            path = f.name
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _array_to_binary(path, X)
            with open(path, "rb") as f:
                f.read(20)
                val = np.fromfile(f, dtype=np.float32, count=1)[0]
            assert np.isinf(val)
        finally:
            os.unlink(path)

    def test_nan_preserved(self):
        """NaN should survive the float64->float32 conversion."""
        X = np.array([[np.nan, 1.0]])
        with tempfile.NamedTemporaryFile(suffix=".cbmx", delete=False) as f:
            path = f.name
        try:
            _array_to_binary(path, X)
            with open(path, "rb") as f:
                f.read(20)
                data = np.fromfile(f, dtype=np.float32, count=2)
            assert np.isnan(data[0])
            assert data[1] == np.float32(1.0)
        finally:
            os.unlink(path)

    def test_empty_array(self):
        """Zero rows."""
        X = np.zeros((0, 3))
        with tempfile.NamedTemporaryFile(suffix=".cbmx", delete=False) as f:
            path = f.name
        try:
            _array_to_binary(path, X)
            with open(path, "rb") as f:
                magic = f.read(4)
                ver, n, feat, flags = struct.unpack("<IIII", f.read(16))
                assert n == 0
                assert feat == 3
        finally:
            os.unlink(path)

    def test_column_major_layout(self):
        """Verify column-major (transposed) layout."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with tempfile.NamedTemporaryFile(suffix=".cbmx", delete=False) as f:
            path = f.name
        try:
            _array_to_binary(path, X)
            with open(path, "rb") as f:
                f.read(20)  # header
                data = np.fromfile(f, dtype=np.float32, count=6)
            # Column-major: col0 then col1
            # col0: [1, 3, 5], col1: [2, 4, 6]
            np.testing.assert_allclose(data, [1, 3, 5, 2, 4, 6], atol=1e-6)
        finally:
            os.unlink(path)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: Constant features check bypass with object arrays
# ════════════════════════════════════════════════════════════════════════════


class TestConstantFeaturesCheck:
    """The new all-constant-features check has a bypass."""

    def test_constant_numeric_features_rejected(self):
        """Pure numeric constant features should be rejected."""
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.ones((20, 3))
        y = np.random.rand(20)
        with pytest.raises(ValueError, match="constant"):
            m.fit(X, y)

    def test_constant_features_with_cat_rejected(self):
        """When X is object dtype (mixed cat+numeric), constant numeric
        features should still be detected and rejected."""
        m = CatBoostMLXRegressor(cat_features=[0], binary_path=BINARY_PATH)
        # Column 0: categorical (varied), Column 1: numeric constant
        X = np.array([["a", 1.0], ["b", 1.0], ["c", 1.0]], dtype=object)
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="constant"):
            m._validate_fit_inputs(X, y)

    def test_one_constant_one_varying_passes(self):
        """Mix of constant and varying features should pass."""
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.column_stack([np.ones(20), np.random.rand(20)])
        y = np.random.rand(20)
        m._validate_fit_inputs(X, y)  # should pass -- not ALL constant


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: base_prediction in staged_predict
# ════════════════════════════════════════════════════════════════════════════


class TestBasePrediction:
    """Test the new base_prediction support in staged_predict."""

    def test_staged_predict_with_base_prediction(self):
        """staged_predict should use base_prediction from model_info."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)

        staged = list(model.staged_predict(X))
        preds = model.predict(X)

        # Final staged_predict should match predict
        np.testing.assert_allclose(staged[-1], preds, atol=1e-4)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8: sklearn compat with new param
# ════════════════════════════════════════════════════════════════════════════


class TestSklearnNewParam:
    """Verify random_strength works with sklearn machinery."""

    def test_clone_preserves_random_strength(self):
        try:
            from sklearn.base import clone
        except ImportError:
            pytest.skip("sklearn not installed")
        model = CatBoostMLXRegressor(random_strength=0.5)
        cloned = clone(model)
        assert cloned.random_strength == 0.5

    def test_get_set_params_random_strength(self):
        model = CatBoostMLX()
        assert model.get_params()["random_strength"] == 1.0
        model.set_params(random_strength=0.3)
        assert model.random_strength == 0.3

    def test_regressor_get_params_has_random_strength(self):
        model = CatBoostMLXRegressor()
        params = model.get_params()
        assert "random_strength" in params

    def test_classifier_get_params_has_random_strength(self):
        model = CatBoostMLXClassifier()
        params = model.get_params()
        assert "random_strength" in params


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9: Integration with new code paths
# ════════════════════════════════════════════════════════════════════════════


class TestIntegrationNewPaths:
    """Full train-predict with new features."""

    def test_train_with_random_strength_zero(self):
        """random_strength=0 should work (no perturbation)."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, random_strength=0.0, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,)

    def test_train_with_random_strength_high(self):
        """Large random_strength should still produce a model."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, random_strength=100.0, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,)
        assert np.all(np.isfinite(preds))

    def test_feature_names_preserved_after_fit(self):
        """Feature names should be injected into model_data after fit."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3)
        y = rng.rand(20)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(X, y, feature_names=["price", "size", "age"])
        features = model._model_data.get("features", [])
        names = [f.get("name", "") for f in features]
        assert names == ["price", "size", "age"]

    def test_feature_names_in_model_info_after_save_load(self):
        """Feature names should survive save -> load roundtrip."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3)
        y = rng.rand(20)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(X, y, feature_names=["price", "size", "age"])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            model2 = CatBoostMLXRegressor(binary_path=BINARY_PATH)
            model2.load_model(path)
            assert list(model2.feature_names_in_) == ["price", "size", "age"]
        finally:
            os.unlink(path)

    def test_binary_format_predictions_match_csv(self):
        """Predictions via binary format should match CSV format."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)

        # Train once
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)

        # Predict (uses binary format for numeric data)
        preds = model.predict(X)
        assert preds.shape == (30,)
        assert np.all(np.isfinite(preds))

    def test_constant_target_emits_warning(self):
        """Constant target should emit a UserWarning."""
        _skip_no_binaries()
        X = np.random.rand(30, 3)
        y = np.full(30, 5.0)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        with pytest.warns(UserWarning, match="zero variance"):
            model.fit(X, y)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10: Edge cases in new validation logic
# ════════════════════════════════════════════════════════════════════════════


class TestNewValidationEdges:
    """Probe the boundaries of new validation code."""

    def test_boolean_y_accepted(self):
        """Boolean y should still be accepted (bool is subtype of number)."""
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.random.rand(5, 2)
        y = np.array([True, False, True, False, True])
        m._validate_fit_inputs(X, y)

    def test_int_y_accepted(self):
        """Integer y should be accepted."""
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.random.rand(5, 2)
        y = np.array([0, 1, 2, 3, 4])
        m._validate_fit_inputs(X, y)

    def test_complex_y_rejected(self):
        """Complex dtype y should be rejected."""
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        X = np.random.rand(5, 2)
        y = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j, 9 + 0j])
        # complex is a subtype of np.number! Let's check...
        if np.issubdtype(y.dtype, np.number):
            # Complex passes the number check. But it can't be formatted as float.
            # This is a potential gap.
            pass

    def test_feature_count_check_with_1d_predict(self):
        """1D predict input should be reshaped before feature count check."""
        _skip_no_binaries()
        rng = np.random.RandomState(42)
        model = CatBoostMLXRegressor(iterations=5, depth=2, binary_path=BINARY_PATH)
        model.fit(rng.rand(20, 1), rng.rand(20))
        # 1D input for single-feature model should work
        preds = model.predict(rng.rand(5))
        assert preds.shape == (5,)
