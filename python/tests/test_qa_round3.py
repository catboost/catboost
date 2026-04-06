"""
test_qa_round3.py -- QA Round 3: Testing dev fixes for round 2 issues + new bugs.

Focus areas:
1. Verify all 5 round 2 issues (NEW-1 through NEW-5) are fixed
2. Break get_shap_values (missing feature count check, stale cache)
3. Boolean bypass in parameter validation (systemic, 21 params affected)
4. Non-string feature names crash with TypeError
5. Carriage return bypass in feature name validation
6. Constant features check edge cases (NaN bypass, single-row false positive)
"""

import json
import math
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
# SECTION 1: Verify round 2 fixes (NEW-1 through NEW-5)
# ════════════════════════════════════════════════════════════════════════════


class TestRound2FixVerification:
    """Confirm all 5 round 2 issues are fixed."""

    # -- NEW-1 fix: random_strength validation --

    def test_random_strength_nan_rejected(self):
        with pytest.raises(ValueError, match="random_strength"):
            CatBoostMLX(random_strength=float("nan"))._validate_params()

    def test_random_strength_inf_rejected(self):
        with pytest.raises(ValueError, match="random_strength"):
            CatBoostMLX(random_strength=float("inf"))._validate_params()

    def test_random_strength_negative_rejected(self):
        with pytest.raises(ValueError, match="random_strength"):
            CatBoostMLX(random_strength=-1.0)._validate_params()

    def test_random_strength_string_rejected(self):
        with pytest.raises(ValueError, match="random_strength"):
            CatBoostMLX(random_strength="hello")._validate_params()

    def test_random_strength_none_rejected(self):
        with pytest.raises(ValueError, match="random_strength"):
            CatBoostMLX(random_strength=None)._validate_params()

    # -- NEW-2 fix: null bytes in feature names --

    def test_null_byte_in_feature_name_rejected(self):
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        with pytest.raises(ValueError, match="invalid characters"):
            model._validate_fit_inputs(X, y, feature_names=["a\x00b", "c"])

    # -- NEW-3 fix: feature count check in staged_predict/apply --

    def test_staged_predict_wrong_features_gives_valueerror(self):
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        model.fit(X, y)
        X_wrong = np.random.randn(5, 7)
        with pytest.raises(ValueError, match="features"):
            list(model.staged_predict(X_wrong))

    def test_apply_wrong_features_gives_valueerror(self):
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        model.fit(X, y)
        X_wrong = np.random.randn(5, 7)
        with pytest.raises(ValueError, match="features"):
            model.apply(X_wrong)

    # -- NEW-4 fix: constant features with object dtype --

    def test_constant_features_with_cat_detected(self):
        model = CatBoostMLXRegressor(iterations=5, cat_features=[0])
        X = np.array([["a", 1.0], ["b", 1.0], ["c", 1.0]], dtype=object)
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="constant"):
            model._validate_fit_inputs(X, y)

    # -- NEW-5 fix: quotes allowed in feature names --

    def test_quotes_in_feature_name_accepted(self):
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        # Should NOT raise -- quotes are valid
        model._validate_fit_inputs(X, y, feature_names=['price "USD"', 'size'])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: get_shap_values missing feature count validation (NEW-1)
# ════════════════════════════════════════════════════════════════════════════


class TestShapFeatureCount:
    """get_shap_values has its own prediction path -- NOT via _run_predict.
    The feature count check was added to _run_predict, staged_predict, apply,
    staged_predict_proba, but NOT to get_shap_values.
    """

    def test_shap_wrong_feature_count_gives_valueerror(self):
        """get_shap_values should raise ValueError for wrong feature count,
        not RuntimeError from C++ binary.

        BUG: get_shap_values (line ~1560) does _to_numpy + reshape but has
        no feature count check. Wrong feature count hits C++ binary which
        returns RuntimeError('csv_predict --shap failed...').
        """
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        model.fit(X, y)

        X_wrong = np.random.randn(5, 7)
        # BUG: This crashes with RuntimeError from C++ binary instead of
        # a clear Python-side ValueError like the other predict methods
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            model.get_shap_values(X_wrong)

        # Document the bug: should be ValueError, currently RuntimeError
        if isinstance(exc_info.value, RuntimeError):
            pytest.xfail(
                "BUG NEW-1: get_shap_values raises RuntimeError from C++ "
                "instead of ValueError. Missing feature count check at "
                "core.py ~line 1562."
            )

    def test_shap_correct_feature_count_succeeds(self):
        """Smoke test: SHAP with correct features should work."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        model.fit(X, y)

        result = model.get_shap_values(X[:5])
        assert "shap_values" in result
        assert result["shap_values"].shape == (5, 3)

    def test_shap_1d_input_wrong_features(self):
        """1D input gets reshaped to (n, 1) -- feature count check should
        still fire if model has >1 feature."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        model.fit(X, y)

        X_1d = np.array([1.0, 2.0, 3.0])
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            model.get_shap_values(X_1d)

        if isinstance(exc_info.value, RuntimeError):
            pytest.xfail(
                "BUG: get_shap_values 1D input gives RuntimeError, "
                "not ValueError for wrong feature count."
            )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: Boolean bypass in parameter validation (NEW-2, SYSTEMIC)
# ════════════════════════════════════════════════════════════════════════════


class TestBooleanParamBypass:
    """Python bool is a subclass of int: isinstance(True, int) == True.
    This means boolean values bypass all isinstance(x, (int, float)) checks.
    str(True) = "True" and str(False) = "False", which C++ atof/atoi parse
    as 0.0/0, leading to silently wrong behavior.

    21 out of 30 boolean param combinations bypass validation.
    """

    # -- Critical params: boolean causes training failure or silent corruption --

    def test_iterations_true_rejected(self):
        """iterations=True passes as int 1, but str(True)='True' -> C++ atoi=0.
        BUG: Causes FileNotFoundError (empty model) during training.
        """
        try:
            CatBoostMLX(iterations=True)._validate_params()
            pytest.xfail(
                "BUG NEW-2: iterations=True passes validation. "
                "str(True)='True' -> C++ atoi('True')=0 -> 0 iterations -> "
                "empty model -> FileNotFoundError"
            )
        except (ValueError, TypeError):
            pass  # Fixed

    def test_iterations_false_rejected(self):
        """iterations=False: int(False)=0, which is < 1, so it's rejected by
        the range check. This one works correctly by accident."""
        with pytest.raises(ValueError, match="iterations"):
            CatBoostMLX(iterations=False)._validate_params()

    def test_depth_true_rejected(self):
        """depth=True -> str='True' -> C++ depth=0."""
        try:
            CatBoostMLX(depth=True)._validate_params()
            pytest.xfail(
                "BUG NEW-2: depth=True passes validation. "
                "str(True)='True' -> C++ depth=0."
            )
        except (ValueError, TypeError):
            pass

    def test_learning_rate_true_rejected(self):
        """learning_rate=True -> str='True' -> C++ lr=0.0 -> no learning."""
        try:
            CatBoostMLX(learning_rate=True)._validate_params()
            pytest.xfail(
                "BUG NEW-2: learning_rate=True passes validation. "
                "str(True)='True' -> C++ atof('True')=0.0 -> no learning."
            )
        except (ValueError, TypeError):
            pass

    def test_random_strength_true_rejected(self):
        """random_strength=True -> str='True' -> C++ = 0.0."""
        try:
            CatBoostMLX(random_strength=True)._validate_params()
            pytest.xfail(
                "BUG NEW-2: random_strength=True passes validation. "
                "str(True)='True' -> C++ atof('True')=0.0."
            )
        except (ValueError, TypeError):
            pass

    def test_random_strength_false_rejected(self):
        """random_strength=False -> str='False' -> C++ = 0.0."""
        try:
            CatBoostMLX(random_strength=False)._validate_params()
            pytest.xfail(
                "BUG NEW-2: random_strength=False passes validation. "
                "str(False)='False' -> C++ atof('False')=0.0."
            )
        except (ValueError, TypeError):
            pass

    def test_subsample_true_rejected(self):
        """subsample=True -> str='True' -> C++ = 0.0."""
        try:
            CatBoostMLX(subsample=True)._validate_params()
            pytest.xfail(
                "BUG NEW-2: subsample=True passes validation. "
                "str(True)='True' -> C++ atof('True')=0.0."
            )
        except (ValueError, TypeError):
            pass

    def test_min_data_in_leaf_true_rejected(self):
        """min_data_in_leaf=True -> str='True' -> C++ = 0 (no minimum)."""
        try:
            CatBoostMLX(min_data_in_leaf=True)._validate_params()
            pytest.xfail(
                "BUG NEW-2: min_data_in_leaf=True passes validation. "
                "str(True)='True' -> C++ atoi('True')=0 -> no leaf minimum."
            )
        except (ValueError, TypeError):
            pass

    # -- Params that happen to give correct C++ result by accident --

    def test_l2_reg_lambda_false_accepted_but_wrong_type(self):
        """l2_reg_lambda=False passes validation. str='False' -> C++ atof=0.0.
        By coincidence this is a valid value (no regularization), but the user
        likely didn't intend to pass a boolean.
        """
        try:
            CatBoostMLX(l2_reg_lambda=False)._validate_params()
            pytest.xfail(
                "BUG NEW-2: l2_reg_lambda=False passes validation. "
                "Works by accident (C++ gets 0.0 = valid), but wrong type."
            )
        except (ValueError, TypeError):
            pass

    # -- Integration test: boolean actually causes training failure --

    def test_boolean_iterations_causes_training_crash(self):
        """End-to-end: iterations=True causes C++ to train 0 trees -> no model."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=True, binary_path=BINARY_PATH)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        with pytest.raises(Exception):
            model.fit(X, y)

    # -- Count total bypasses --

    def test_boolean_bypass_count(self):
        """At least 15 boolean param combinations should be rejected."""
        bypasses = 0
        for param, val in [
            ('iterations', True), ('depth', True), ('learning_rate', True),
            ('l2_reg_lambda', True), ('l2_reg_lambda', False),
            ('eval_fraction', False), ('early_stopping_rounds', True),
            ('early_stopping_rounds', False), ('subsample', True),
            ('colsample_bytree', True), ('bagging_temperature', True),
            ('bagging_temperature', False), ('mvs_reg', True),
            ('mvs_reg', False), ('max_onehot_size', True),
            ('max_onehot_size', False), ('ctr_prior', True),
            ('random_strength', True), ('random_strength', False),
            ('min_data_in_leaf', True), ('snapshot_interval', True),
        ]:
            try:
                CatBoostMLX(**{param: val})._validate_params()
                bypasses += 1
            except (ValueError, TypeError):
                pass

        if bypasses > 0:
            pytest.xfail(
                f"BUG NEW-2: {bypasses} boolean param combinations bypass "
                "validation. isinstance(True, int) is True in Python."
            )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: Non-string feature names crash with TypeError (NEW-3)
# ════════════════════════════════════════════════════════════════════════════


class TestFeatureNameTypeValidation:
    """Feature name validation uses `',' in fname` which raises TypeError
    when fname is not a string. Should be a clear ValueError.
    """

    def test_integer_feature_names_give_valueerror(self):
        """feature_names=[1, 2, 3] should raise ValueError, not TypeError.

        BUG: `',' in 1` raises TypeError: argument of type 'int' is not iterable
        """
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        try:
            model._validate_fit_inputs(X, y, feature_names=[1, 2, 3])
            pytest.fail("Should have raised an error for integer feature names")
        except ValueError:
            pass  # Correct behavior
        except TypeError:
            pytest.xfail(
                "BUG NEW-3: Integer feature names raise TypeError "
                "('argument of type int is not iterable') instead of ValueError."
            )

    def test_none_feature_name_gives_valueerror(self):
        """feature_names=[None, 'b'] should raise ValueError, not TypeError."""
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        try:
            model._validate_fit_inputs(X, y, feature_names=[None, "b"])
            pytest.fail("Should have raised an error for None feature name")
        except ValueError:
            pass
        except TypeError:
            pytest.xfail(
                "BUG NEW-3: None feature name raises TypeError "
                "instead of ValueError."
            )

    def test_mixed_type_feature_names_give_valueerror(self):
        """feature_names=['a', 42] should raise ValueError."""
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        try:
            model._validate_fit_inputs(X, y, feature_names=["a", 42])
            pytest.fail("Should have raised an error for non-string feature name")
        except ValueError:
            pass
        except TypeError:
            pytest.xfail(
                "BUG NEW-3: Mixed-type feature names raise TypeError."
            )

    def test_boolean_feature_names_give_valueerror(self):
        """feature_names=[True, False] should raise ValueError."""
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        try:
            model._validate_fit_inputs(X, y, feature_names=[True, False])
            pytest.fail("Should have raised an error for boolean feature name")
        except ValueError:
            pass
        except TypeError:
            pytest.xfail(
                "BUG NEW-3: Boolean feature names raise TypeError. "
                "',' in True -> TypeError."
            )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: Carriage return bypass in feature names (NEW-4)
# ════════════════════════════════════════════════════════════════════════════


class TestFeatureNameCRBypass:
    r"""Feature name validation checks \n and \x00 but not \r.
    Carriage return can corrupt CSV parsing and cause issues in text
    serialization formats.
    """

    def test_carriage_return_not_rejected(self):
        r"""Feature names with \r pass validation but could corrupt CSV.

        BUG: Validation checks for \n and \x00 but not \r. On Windows and
        in some CSV parsers, \r is a line terminator.
        """
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        try:
            model._validate_fit_inputs(X, y, feature_names=["a\rb", "c"])
            pytest.xfail(
                r"BUG NEW-4: Feature names with \r pass validation. "
                r"Validation checks \n and \x00 but not \r."
            )
        except ValueError:
            pass  # Fixed

    def test_crlf_not_rejected(self):
        r"""Feature name with \r\n -- the \n part is caught, so this works.
        But \r alone is not caught."""
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        # \r\n: the \n part triggers the check
        with pytest.raises(ValueError, match="invalid characters"):
            model._validate_fit_inputs(X, y, feature_names=["a\r\nb", "c"])

    def test_cr_in_csv_path_with_cat_features(self):
        r"""End-to-end: \r in feature name with categorical features (CSV path)."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=5, cat_features=[0],
                                     binary_path=BINARY_PATH)
        X = np.array([["a", 1.0], ["b", 2.0], ["c", 3.0],
                       ["a", 4.0], ["b", 5.0]], dtype=object)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        try:
            model.fit(X, y, feature_names=["col\rwith_cr", "normal"])
            pytest.xfail(
                r"BUG NEW-4: Training succeeded with \r in feature name. "
                "Could corrupt CSV header parsing."
            )
        except ValueError:
            pass  # Fixed


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: get_shap_values stale JSON cache (NEW-5)
# ════════════════════════════════════════════════════════════════════════════


class TestShapStaleCache:
    """get_shap_values uses `if cache is None` pattern (line 1570) instead
    of always regenerating like _run_predict (line 1326). This is the same
    root cause as original BUG-10 but in a different method.
    """

    def test_shap_does_not_regenerate_cache(self):
        """After predict sets cache, get_shap_values should regenerate,
        not reuse stale cache.

        BUG: _run_predict always regenerates (line 1326: json.dumps every time).
        get_shap_values uses `if self._model_json_cache is None` (line 1570),
        reusing whatever was cached by the last _run_predict call.
        """
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        model.fit(X, y)

        # Set cache via predict
        model.predict(X[:5])
        cache_after_predict = model._model_json_cache

        # Mutate model_data (simulates any internal modification)
        model._model_data["model_info"]["_test_marker"] = "stale_test"

        # get_shap_values should regenerate, but doesn't
        try:
            model.get_shap_values(X[:5])
        except Exception:
            pass  # SHAP may fail for other reasons

        cache_after_shap = model._model_json_cache
        has_marker = "_test_marker" in cache_after_shap

        if not has_marker:
            pytest.xfail(
                "BUG NEW-5: get_shap_values uses stale JSON cache. "
                "After model_data mutation, SHAP still uses old cache "
                "(inconsistent with _run_predict which always regenerates)."
            )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: Constant features edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestConstantFeaturesEdgeCases:
    """Edge cases in the round 3 constant features check
    (try/astype(float)/except pattern).
    """

    def test_nan_column_bypasses_constant_check(self):
        """When one column is all-NaN and the other is constant,
        np.var returns [NaN, 0.0]. np.all([NaN, 0.0] == 0) is False
        because NaN != 0. So the constant features check passes.

        With nan_mode='min', NaN maps to minimum bin, making it effectively
        constant too.
        """
        model = CatBoostMLXRegressor(iterations=5, nan_mode="min")
        X = np.array([[np.nan, 1.0],
                      [np.nan, 1.0],
                      [np.nan, 1.0],
                      [np.nan, 1.0],
                      [np.nan, 1.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # This SHOULD raise ValueError (all features are effectively constant)
        # but NaN variance prevents detection
        try:
            model._validate_fit_inputs(X, y)
            pytest.xfail(
                "OBSERVATION: NaN column bypasses constant features check. "
                "np.var(NaN_column) = NaN, and NaN != 0, so np.all(variances == 0) "
                "is False even when all non-NaN features are constant."
            )
        except ValueError:
            pass  # Fixed

    def test_single_row_false_positive(self):
        """Single-row input always has zero variance, triggering the
        constant features check even though features aren't truly constant.
        """
        model = CatBoostMLXRegressor(iterations=5)
        X = np.array([[1.0, 2.0, 3.0]])  # 1 row, 3 different features
        y = np.array([5.0])

        # np.var of 1-element columns is always 0.0
        # The check triggers even though the features are distinct values
        try:
            model._validate_fit_inputs(X, y)
        except ValueError as e:
            if "constant" in str(e).lower():
                pytest.xfail(
                    "OBSERVATION: Single-row input triggers false positive "
                    "constant features error. np.var of 1-element array is 0. "
                    "Error message says 'constant features' but they're not — "
                    "there's just 1 sample."
                )
            raise


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8: Round 2 fix verification — integration tests
# ════════════════════════════════════════════════════════════════════════════


class TestRound2FixIntegration:
    """Integration tests confirming round 2 fixes work end-to-end."""

    def test_random_strength_zero_trains_successfully(self):
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=10, random_strength=0.0,
                                     binary_path=BINARY_PATH)
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)
        assert np.all(np.isfinite(preds))

    def test_random_strength_high_trains_successfully(self):
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=10, random_strength=100.0,
                                     binary_path=BINARY_PATH)
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)
        assert np.all(np.isfinite(preds))

    def test_feature_count_check_consistent_across_methods(self):
        """All predict-family methods should give ValueError for wrong features."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        model.fit(X, y)

        X_wrong = np.random.randn(5, 7)

        # predict (via _run_predict)
        with pytest.raises(ValueError, match="features"):
            model.predict(X_wrong)

        # staged_predict
        with pytest.raises(ValueError, match="features"):
            list(model.staged_predict(X_wrong))

        # apply
        with pytest.raises(ValueError, match="features"):
            model.apply(X_wrong)

    def test_quote_in_feature_name_survives_roundtrip(self):
        """Quotes in feature names should be stored and retrievable."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        model.fit(X, y, feature_names=['price "USD"', "size"])
        assert model.feature_names_in_[0] == 'price "USD"'
        assert model.feature_names_in_[1] == "size"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9: Miscellaneous edge cases from round 3 changes
# ════════════════════════════════════════════════════════════════════════════


class TestMiscRound3Edges:
    """Additional edge cases discovered during round 3 testing."""

    def test_empty_string_feature_name_accepted(self):
        """Empty string feature names pass validation. Not necessarily a bug
        but worth documenting."""
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        # Empty string has no comma/newline/null, so it passes
        model._validate_fit_inputs(X, y, feature_names=["", "b"])

    def test_whitespace_only_feature_name_accepted(self):
        """Whitespace-only feature names pass validation."""
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        model._validate_fit_inputs(X, y, feature_names=["   ", "b"])

    def test_tab_in_feature_name_accepted(self):
        r"""Tab character in feature names passes validation.
        Could cause issues in tab-delimited formats."""
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        model._validate_fit_inputs(X, y, feature_names=["a\tb", "c"])

    def test_constant_features_with_mixed_varying(self):
        """Mix of constant and varying features should pass
        (only errors when ALL are constant)."""
        model = CatBoostMLXRegressor(iterations=5)
        X = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])  # col 1 constant
        y = np.array([1.0, 2.0, 3.0])
        # Should pass — col 0 varies
        model._validate_fit_inputs(X, y)

    def test_sklearn_clone_preserves_random_strength(self):
        """Verify sklearn clone still works with random_strength."""
        from sklearn.base import clone
        m = CatBoostMLXRegressor(random_strength=0.5)
        m2 = clone(m)
        assert m2.random_strength == 0.5

    def test_classifier_boolean_bypass(self):
        """Boolean bypass also affects classifier subclass."""
        try:
            CatBoostMLXClassifier(learning_rate=True)._validate_params()
            pytest.xfail(
                "BUG NEW-2: Boolean bypass affects CatBoostMLXClassifier too."
            )
        except (ValueError, TypeError):
            pass
